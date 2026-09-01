from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1]))

from arrow_controller import (  # noqa: E402
    ArrowCommand2D,
    ArrowObservation,
    BowlWaypointConfig,
    build_bowl_waypoints,
    decode_arrow,
    deproject_endpoint,
    estimate_endpoint_depth,
    normalized_osc_action,
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
