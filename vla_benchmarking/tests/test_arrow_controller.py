from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1]))

from arrow_controller import (  # noqa: E402
    ArrowCommand2D,
    ArrowEncoding,
    ArrowObservation,
    COLOR_ENDPOINT_ARROW_ENCODING,
    BowlWaypointConfig,
    build_bowl_waypoints,
    compute_endpoint_change_evidence,
    decode_arrow,
    decode_arrow_diagnostics,
    deproject_endpoint,
    derive_rgbd_region_grasp_candidates,
    estimate_endpoint_depth,
    normalized_osc_action,
    refine_rgbd_endpoint,
)


FIXTURE_DIR = Path(__file__).parents[1] / "visual_arrow_previews_highres"


def _synthetic_arrow(size=(128, 128), start=(25, 100), end=(100, 25), width=3, head=14):
    clean = np.zeros((*size, 3), dtype=np.uint8)
    image = Image.fromarray(clean.copy())
    draw = ImageDraw.Draw(image)
    draw.line([start, end], fill=(0, 166, 107), width=width)
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = np.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    wing_a = (round(end[0] - head * ux + head * 0.5 * px), round(end[1] - head * uy + head * 0.5 * py))
    wing_b = (round(end[0] - head * ux - head * 0.5 * px), round(end[1] - head * uy - head * 0.5 * py))
    draw.polygon([end, wing_a, wing_b], fill=(0, 166, 107))
    return clean, np.asarray(image).copy()


def test_decode_current_1024_preview_pair_points_to_bowl_destination():
    clean = np.asarray(Image.open(FIXTURE_DIR / "task_00_1024_raw.png"))
    overlay = np.asarray(Image.open(FIXTURE_DIR / "task_00_1024_bowl_to_target.png"))
    command = decode_arrow(clean, overlay)
    assert command.image_shape == (1024, 1024)
    # The rendered preview's actual line runs from approximately (767, 411)
    # to (789, 283); tolerate antialiasing and endpoint pixel conventions.
    assert np.allclose(command.source_xy, (767, 411), atol=4)
    assert np.allclose(command.target_xy, (789, 283), atol=4)
    assert command.confidence > 0.7


@pytest.mark.parametrize("start,end", [((20, 100), (105, 30)), ((100, 25), (22, 105)), ((15, 60), (110, 65))])
def test_decode_varied_arrow_directions(start, end):
    clean, overlay = _synthetic_arrow(start=start, end=end)
    command = decode_arrow(clean, overlay)
    assert np.linalg.norm(np.asarray(command.target_xy) - end) < 5
    assert np.linalg.norm(np.asarray(command.source_xy) - start) < 5


def test_decode_rejects_missing_multiple_and_unaligned_arrows():
    clean, overlay = _synthetic_arrow()
    with pytest.raises(ValueError, match="no arrow"):
        decode_arrow(clean, clean)
    _, second = _synthetic_arrow(start=(105, 105), end=(25, 25))
    with pytest.raises(ValueError, match="exactly one|ambiguous"):
        decode_arrow(clean, np.maximum(overlay, second))
    with pytest.raises(ValueError, match="identical shapes"):
        decode_arrow(clean, overlay[:64])


def test_decode_ignores_disconnected_changed_speck_for_endpoint_geometry():
    clean, overlay = _synthetic_arrow()
    baseline = decode_arrow(clean, overlay)
    noisy = overlay.copy()
    # This is deliberately beyond the shaft and too small to qualify as an
    # arrow component on its own.  It must not enter endpoint PCA.
    noisy[2, 2] = (255, 255, 255)
    decoded = decode_arrow(clean, noisy)
    assert decoded.source_xy == pytest.approx(baseline.source_xy)
    assert decoded.target_xy == pytest.approx(baseline.target_xy)
    assert decoded.confidence == pytest.approx(baseline.confidence)


def test_decode_rejects_two_separate_valid_arrow_components():
    clean = np.zeros((180, 180, 3), dtype=np.uint8)
    _, first = _synthetic_arrow(size=(180, 180), start=(15, 35), end=(75, 35))
    _, second = _synthetic_arrow(size=(180, 180), start=(105, 145), end=(165, 145))
    overlay = np.maximum(first, second)
    with pytest.raises(ValueError, match="exactly one"):
        decode_arrow(clean, overlay)


def test_versioned_color_endpoint_encoding_and_diagnostics():
    clean, overlay = _synthetic_arrow()
    marked = overlay.copy()
    marked[22:29, 97:104] = COLOR_ENDPOINT_ARROW_ENCODING.endpoint_color_rgb
    command = decode_arrow(clean, marked, encoding="color_endpoint")
    assert np.allclose(command.target_xy, (100, 25), atol=5)
    diagnostics = decode_arrow_diagnostics(clean, marked, encoding=COLOR_ENDPOINT_ARROW_ENCODING)
    assert diagnostics.ok
    assert diagnostics.command == command
    assert diagnostics.encoding_version == "color_endpoint"
    failed = decode_arrow_diagnostics(clean, clean)
    assert not failed.ok
    assert failed.command is None
    assert failed.reason and "no arrow" in failed.reason
    with pytest.raises(ValueError, match="unsupported"):
        ArrowEncoding(version="unknown")


def test_decode_rejects_unpointed_and_ambiguous_overlay():
    clean = np.zeros((80, 80, 3), dtype=np.uint8)
    overlay = clean.copy()
    overlay[39:42, 10:71] = (255, 0, 0)
    with pytest.raises(ValueError, match="ambiguous|pointy"):
        decode_arrow(clean, overlay)


def test_arrow_observation_validates_pair_and_optional_geometry():
    clean, overlay = _synthetic_arrow()
    observation = ArrowObservation(clean, overlay, np.full((128, 128), 0.5), np.diag([100, 100, 1.0]), np.eye(4))
    assert observation.clean_rgb.shape == (128, 128, 3)
    with pytest.raises(ValueError):
        ArrowObservation(clean, overlay[:64])


def test_depth_estimate_rejects_invalid_and_reduces_outlier():
    depth = np.full((10, 10), 0.8, dtype=np.float64)
    depth[4, 4] = 4.0
    assert estimate_endpoint_depth(depth, (4, 4), radius=1, min_valid=3) == pytest.approx(0.8)
    depth[:] = 0
    with pytest.raises(ValueError, match="insufficient"):
        estimate_endpoint_depth(depth, (4, 4), radius=1, min_valid=3)


def test_deprojection_metric_roundtrip_with_world_transform():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = (1.0, 2.0, 3.0)
    world = deproject_endpoint((60, 50), 2.0, K, camera_to_world)
    assert np.allclose(world, (1.2, 2.2, 5.0))
    command = ArrowCommand2D((10, 10), (60, 50), 0.9, component_area=20, image_shape=(80, 100))
    assert np.allclose(deproject_endpoint(command, 2.0, K), (0.2, 0.2, 2.0))
    outside = ArrowCommand2D((10, 10), (200, 50), 0.9, component_area=20, image_shape=(80, 100))
    with pytest.raises(ValueError, match="outside"):
        deproject_endpoint(outside, 2.0, K)


def test_rgbd_refinement_keeps_pixel_and_depth_provenance_aligned():
    clean, overlay = _synthetic_arrow()
    depth = np.full((128, 128), 2.0, dtype=np.float64)
    K = np.array([[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]])
    command = decode_arrow(clean, overlay)
    refined = refine_rgbd_endpoint(
        command, depth, K, depth_method="trimmed_median",
    )
    assert refined.depth_m == pytest.approx(2.0)
    expected_pixel = command.target_xy
    assert refined.pixel_provenance.endswith(f"pixel_xy=({expected_pixel[0]:.3f},{expected_pixel[1]:.3f})")
    assert refined.depth_provenance.endswith(f"pixel_xy=({expected_pixel[0]:.3f},{expected_pixel[1]:.3f})")
    assert refined.method.endswith("trimmed_median")
    assert refined.command == command
    with pytest.raises(ValueError, match="resolution mismatch"):
        refine_rgbd_endpoint(command, np.full((64, 64), 2.0), K)


def test_endpoint_change_evidence_is_array_only_and_reports_proprioception():
    evidence = compute_endpoint_change_evidence(
        np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.8, 3.0]),
        before_proprioception=np.array([0.0, 0.2]),
        after_proprioception=np.array([0.1, 0.4]),
    )
    assert np.allclose(evidence.endpoint_delta, (0.1, -0.2, 0.0))
    assert np.allclose(evidence.delta, evidence.endpoint_delta)
    assert evidence.endpoint_distance == pytest.approx(np.sqrt(0.05))
    assert evidence.proprioception_distance == pytest.approx(np.sqrt(0.05))
    with pytest.raises(ValueError, match="matching shapes"):
        compute_endpoint_change_evidence(np.zeros(3), np.ones(3), before_proprioception=np.zeros(2), after_proprioception=np.zeros(3))


def test_signed_image_aligned_intrinsics_roundtrip_libero_vertical_flip():
    """LIBERO's top-left ``v`` projection is represented by a signed ``fy``."""
    # Conventional LIBERO projection: v = H - (fy*y/z + cy), H=100, cy=40.
    # Image-aligned K therefore has fy_image=-fy and cy_image=H-cy.
    K = np.array([[100.0, 0, 50.0], [0, -100.0, 60.0], [0, 0, 1.0]])
    camera_point = np.array((0.2, -0.1, 2.0))
    pixel = (100.0 * camera_point[0] / camera_point[2] + 50.0,
             -100.0 * camera_point[1] / camera_point[2] + 60.0)
    assert pixel == pytest.approx((60.0, 65.0))
    assert np.allclose(deproject_endpoint(pixel, camera_point[2], K), camera_point)


def test_waypoints_depend_on_both_endpoints_and_preserve_z_clearance():
    source = np.array([0.1, -0.2, 0.3])
    target = np.array([0.4, 0.5, 0.1])
    waypoints = build_bowl_waypoints(source, target, np.eye(3), BowlWaypointConfig(lift_height_m=0.12))
    assert waypoints.shape == (6, 3)
    assert np.allclose(waypoints[[1, 4]], (source, target))
    assert np.allclose(waypoints[0], source + (0, 0, 0.03))
    assert np.allclose(waypoints[2, 2], waypoints[3, 2])
    assert waypoints[2, 2] == pytest.approx(max(source[2], target[2]) + 0.12)
    assert np.allclose(waypoints[5], target + (0, 0, 0.03))
    assert not np.allclose(waypoints[3], build_bowl_waypoints(source, target + (0.2, 0, 0), np.eye(3))[3])


def test_normalized_osc_action_is_7d_finite_and_bounded():
    action = normalized_osc_action(
        (0, 0, 0), np.eye(3), (1, -1, 0.1), np.eye(3), -1,
        (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
    )
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0)
    with pytest.raises(ValueError):
        normalized_osc_action((0, 0, 0), np.eye(3), (0, 0, 0), np.eye(3), 0, (0, 0, 1, 1, 1, 1))


def test_arrow_world_xy_basis_is_derived_from_deprojected_endpoints():
    from arrow_controller import arrow_world_xy_basis

    forward, lateral = arrow_world_xy_basis((1.0, 2.0, 0.4), (1.0, 3.0, 9.0))
    assert forward == pytest.approx((0.0, 1.0, 0.0))
    assert lateral == pytest.approx((-1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="degenerate"):
        arrow_world_xy_basis((0, 0, 0), (0, 0, 1))


def test_rgbd_region_candidates_follow_observed_geometry_not_rgb_texture():
    height = width = 100
    yy, xx = np.mgrid[:height, :width]
    region = (xx - 50) ** 2 + (yy - 50) ** 2 <= 12 ** 2
    depth = np.full((height, width), 2.0, dtype=np.float64)
    depth[region] = 1.0 + (yy[region] - 50) * 0.0005
    # Deliberately high-frequency texture: RGB cannot be a hard region gate.
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (xx * 17 + yy * 31) % 256
    rgb[..., 1] = (xx * 43 + yy * 7) % 256
    rgb[..., 2] = (xx * 3 + yy * 59) % 256
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    targets, audit = derive_rgbd_region_grasp_candidates(
        rgb, depth, K, np.eye(4), (50, 50), (0.10, 0.0, 1.2),
        region_radius_m=0.15,
        profile_quantiles=(0.8, 0.6, 0.4),
    )
    assert targets.shape == (3, 3)
    assert np.all(np.diff(targets[:, 0]) < 0.0)
    assert np.all(np.abs(targets[:, 1]) <= 0.01)
    assert np.allclose(targets[:, 2], np.quantile(depth[region], 0.70))
    assert audit["region_area_px"] == int(region.sum())
    assert audit["method"] == "arrow_seeded_metric_depth_component_v1"
    assert audit["targets_world_m"] == targets.tolist()


def test_rgbd_region_candidates_accept_clipped_object_and_reject_leakage():
    height = width = 80
    yy, xx = np.mgrid[:height, :width]
    clipped = (xx - 1) ** 2 + (yy - 40) ** 2 <= 14 ** 2
    depth = np.full((height, width), np.nan, dtype=np.float64)
    depth[clipped] = 0.8
    rgb = np.full((height, width, 3), 127, dtype=np.uint8)
    K = np.array([[100.0, 0, 40.0], [0, -100.0, 40.0], [0, 0, 1.0]])
    targets, audit = derive_rgbd_region_grasp_candidates(
        rgb, depth, K, np.eye(4), (1, 40), (0.0, 0.0, 0.8),
        region_radius_m=0.12,
    )
    assert targets.shape[0] >= 1
    assert audit["touches_image_border"] is True
    with pytest.raises(ValueError, match="exceeds max_region_fraction"):
        derive_rgbd_region_grasp_candidates(
            rgb, np.full_like(depth, 0.8), K, np.eye(4), (40, 40), (0, 0, 0.8),
            region_radius_m=0.15, max_region_fraction=0.01,
        )
