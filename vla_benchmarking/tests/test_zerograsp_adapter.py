from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from zerograsp_adapter import (  # noqa: E402
    build_arrow_seeded_masks,
    dynamic_T_G_E,
    compose_T_W_E_from_grasp,
    make_letterbox_affine,
    make_pregrasp,
    parse_grasp_array,
    parse_inference_result,
    preprocess_rgbd,
    rank_grasps,
    reconstruction_assisted_placement,
    transform_T_W_E,
    select_grasp_result,
    ZeroGraspSelectionError,
    ZeroGraspProcessAdapter,
)
from zerograsp_contracts import (  # noqa: E402
    GRASPNET_CAMERA_FRAME,
    GRASPNET_TRANSLATION_REFERENCE,
    GraspCandidate,
    ReconstructionSummary,
    ZeroGraspInferenceResult,
    ZeroGraspConfig,
    ZeroGraspObservation,
    calibration_sha256,
)
from zerograsp_worker import camera_yaml_payload  # noqa: E402
import zerograsp_adapter as zg_adapter  # noqa: E402


_CANDIDATE_TAGS = {
    "translation_reference": GRASPNET_TRANSLATION_REFERENCE,
    "translation_frame": GRASPNET_CAMERA_FRAME,
    "rotation_frame": GRASPNET_CAMERA_FRAME,
}
_PAYLOAD_TAGS = {name: [value] for name, value in _CANDIDATE_TAGS.items()}


def _observation(fy: float = -100.0) -> ZeroGraspObservation:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[20:70, 20:70] = [100, 100, 100]
    rgb[180:230, 180:230] = [120, 120, 120]
    depth = np.full((256, 256), np.nan, dtype=np.float32)
    depth[20:70, 20:70] = 1.0
    depth[180:230, 180:230] = 1.1
    K = np.array([[100.0, 0, 128.0], [0, fy, 128.0], [0, 0, 1.0]])
    return ZeroGraspObservation(rgb, rgb.copy(), depth, K, np.eye(4), (45.0, 45.0), (205.0, 205.0))


def test_negative_fy_flip_and_letterbox_preserve_rays_and_pixels():
    obs = _observation(-100.0)
    prepared = preprocess_rgbd(obs, ZeroGraspConfig())
    assert prepared.vertically_flipped
    assert prepared.source_px_model == pytest.approx((308.0, 840.0))
    assert prepared.destination_px_model == pytest.approx((948.0, 200.0))
    X = np.array([0.0, 0.1, 1.0, 1.0])
    original = obs.K @ X[:3]
    original /= original[2]
    mapped = prepared.K_model @ X[:3]
    mapped /= mapped[2]
    assert mapped[:2] == pytest.approx(tuple(np.asarray(make_letterbox_affine((256, 256)))[:2, :2] @ np.array([original[0], 255 - original[1]]) + [128, 0]))


def test_masks_are_disjoint_and_seed_containment_survives_preprocessing():
    obs = _observation()
    masks = build_arrow_seeded_masks(obs)
    assert masks.diagnostics.ok
    assert not np.any(masks.source_mask & masks.destination_mask)
    prepared = __import__("zerograsp_adapter", fromlist=["prepare_zero_grasp_input"]).prepare_zero_grasp_input(obs)
    su, sv = map(round, __import__("zerograsp_adapter", fromlist=["map_pixel"]).map_pixel(obs.source_px, prepared.pixel_affine))
    du, dv = map(round, __import__("zerograsp_adapter", fromlist=["map_pixel"]).map_pixel(obs.destination_px, prepared.pixel_affine))
    assert prepared.source_mask[sv, su]
    assert prepared.destination_mask[dv, du]
    assert not np.any(prepared.source_mask & prepared.destination_mask)


def test_overlap_partition_keeps_each_pixel_with_nearest_seed(monkeypatch):
    obs = _observation(100.0)
    shared = np.zeros(obs.depth_m.shape, dtype=bool)
    shared[128, 128] = True
    def fake_region(_rgb, _depth, seed, _config):
        # Return the same ambiguous pixel for both roles; endpoint distances
        # decide ownership and must not be swapped.
        return shared.copy(), 1.0, 1.0
    monkeypatch.setattr(zg_adapter, "_seeded_region", fake_region)
    masks = zg_adapter.build_arrow_seeded_masks(obs)
    assert not (masks.source_mask[128, 128] and masks.destination_mask[128, 128])
    # The midpoint pixel is closer to the destination seed in this fixture.
    assert masks.destination_mask[128, 128]


def test_strict_grasp_parser_and_pregrasp_plus_x_convention():
    matrix = np.eye(4)[None]
    matrices, scores = parse_grasp_array({"grasps": matrix, "scores": [0.9]})
    assert matrices.shape == (1, 4, 4) and scores.tolist() == [0.9]
    pose = make_pregrasp(np.eye(4), 0.1, (1, 0, 0))
    assert pose[:3, 3].tolist() == [-0.1, 0.0, 0.0]
    with pytest.raises(ValueError):
        parse_grasp_array(np.zeros((1, 3, 4)))


def test_rank_filter_and_world_eef_transform_are_deterministic():
    cfg = ZeroGraspConfig(grasp_width_range_m=(0.005, 0.05))
    grasp_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    grasp_pose = np.eye(4); grasp_pose[:3, :3] = grasp_rotation
    candidates = [GraspCandidate(grasp_pose, score=0.2, width_m=0.04, depth_m=0.02, collision_free=True, source_index=1, **_CANDIDATE_TAGS), GraspCandidate(grasp_pose, score=0.9, width_m=0.04, depth_m=0.02, collision_free=True, source_index=0, **_CANDIDATE_TAGS)]
    ranked = rank_grasps(candidates, np.eye(4), cfg)
    assert ranked[0][0].score == 0.9
    assert np.allclose(transform_T_W_E(np.eye(4), np.eye(4)), np.eye(4))


def test_source_association_rejects_projected_candidate_outside_mask():
    obs = _observation(100.0)
    mask = np.zeros(obs.depth_m.shape, dtype=bool)
    mask[20:70, 20:70] = True
    good = np.eye(4); good[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float); good[:3, 3] = [-0.83, -0.83, 1.0]  # projects near source seed (45,45)
    bad = good.copy(); bad[0, 3] = 0.5
    cfg = ZeroGraspConfig()
    candidates = [GraspCandidate(good, score=0.5, width_m=0.04, depth_m=0.02, collision_free=True, **_CANDIDATE_TAGS), GraspCandidate(bad, score=1.0, width_m=0.04, depth_m=0.02, collision_free=True, **_CANDIDATE_TAGS)]
    ranked = rank_grasps(candidates, np.eye(4), cfg, observation=obs, source_mask=mask)
    assert len(ranked) == 1 and ranked[0][0].score == 0.5


def test_dynamic_eef_transform_is_depth_sensitive_and_selection_is_eef_explicit():
    obs = _observation(100.0)
    grasp_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    grasp = np.eye(4); grasp[:3, :3] = grasp_rotation; grasp[:3, 3] = [-0.83, -0.83, 1.0]
    candidate = GraspCandidate(grasp, score=1.0, width_m=0.04, depth_m=0.04, collision_free=True, **_CANDIDATE_TAGS)
    cfg_base = ZeroGraspConfig(R_grasp_eef=np.eye(3), R_H_E=np.eye(3), calibration_source="test-probe", tip_to_eef_residual_m=(0.01, 0, 0))
    cfg = replace(cfg_base, eef_calibration_verified=True, calibration_sha256=calibration_sha256(cfg_base), probe_sha256="a" * 64)
    dynamic = dynamic_T_G_E(candidate, cfg)
    assert dynamic[:3, 3].tolist() == pytest.approx([0.05, 0, 0])
    result = ZeroGraspInferenceResult((candidate,), None, "req", "out")
    selected = select_grasp_result(result, obs, cfg)
    assert selected["eef_pose_ready"] and "T_W_E" in selected and "T_W_E_pregrasp" in selected
    assert "T_W_pregrasp_E" not in selected
    assert "T_W_G" not in selected


def test_reconstruction_assisted_placement_uses_destination_surface():
    obs = _observation(100.0)
    mask = np.zeros(obs.depth_m.shape, dtype=bool)
    mask[180:230, 180:230] = True
    placement = reconstruction_assisted_placement(obs, ReconstructionSummary((0.08, 0.06, 0.04), 1.0, centroid_camera_m=(0.0, 0.0, 1.0), bounds_camera_m=(-0.04, -0.03, 0.98, 0.04, 0.03, 1.02)), selected_T_world_eef=np.eye(4), destination_mask=mask)
    assert placement.T_world_eef[2, 3] == pytest.approx(0.124, abs=0.01)
    assert placement.footprint_m[0] > 0.08


def test_reconstruction_placement_transforms_bounds_with_rotated_camera():
    obs = _observation(100.0)
    rotated = np.eye(4)
    rotated[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    obs = ZeroGraspObservation(obs.clean_rgb, obs.arrow_rgb, obs.depth_m, obs.K, rotated, obs.source_px, obs.destination_px)
    mask = np.zeros(obs.depth_m.shape, dtype=bool); mask[180:230, 180:230] = True
    reconstruction = ReconstructionSummary((0.08, 0.06, 0.04), 1.0, centroid_camera_m=(0.0, 0.0, 1.0), bounds_camera_m=(-0.04, -0.03, 0.98, 0.04, 0.03, 1.02))
    placement = reconstruction_assisted_placement(obs, reconstruction, selected_T_world_eef=np.eye(4), destination_mask=mask)
    assert placement.T_world_eef[2, 3] == pytest.approx(0.124, abs=0.01)


def test_reconstruction_non_camera_frame_fails_closed():
    obs = _observation(100.0)
    mask = np.zeros(obs.depth_m.shape, dtype=bool); mask[180:230, 180:230] = True
    reconstruction = ReconstructionSummary((0.08, 0.06, 0.04), 1.0, source_frame="world", centroid_camera_m=(0.0, 0.0, 1.0), bounds_camera_m=(-0.04, -0.03, 0.98, 0.04, 0.03, 1.02))
    with pytest.raises(ValueError, match="source_frame"):
        reconstruction_assisted_placement(obs, reconstruction, selected_T_world_eef=np.eye(4), destination_mask=mask)


def test_parse_result_checks_request_hash():
    with pytest.raises(ValueError, match="request hash"):
        parse_inference_result({"request_hash": "other", "grasps": [], "scores": []}, "expected")


def test_missing_collision_metadata_fails_closed():
    matrix = np.eye(4)[None].tolist()
    required_tags = {
        "translation_reference": [GRASPNET_TRANSLATION_REFERENCE],
        "translation_frame": [GRASPNET_CAMERA_FRAME],
        "rotation_frame": [GRASPNET_CAMERA_FRAME],
    }
    parsed = parse_inference_result({"request_hash": "expected", "grasps": matrix, "scores": [1.0], **required_tags}, "expected")
    assert parsed.candidates[0].collision_free is False
    with pytest.raises(ValueError, match="translation_reference"):
        parse_inference_result({"request_hash": "expected", "grasps": matrix, "scores": [1.0], "width_m": [0.04], "depth_m": [0.02], "collision_free": [True], "translation_reference": ["palm_center"], "translation_frame": [GRASPNET_CAMERA_FRAME], "rotation_frame": [GRASPNET_CAMERA_FRAME]}, "expected")
    with pytest.raises(ValueError, match="booleans"):
        parse_inference_result({"request_hash": "expected", "grasps": matrix, "scores": [1.0], "width_m": [0.04], "depth_m": [0.02], "collision_free": ["false"], **required_tags}, "expected")


@pytest.mark.parametrize("missing", ("translation_reference", "translation_frame", "rotation_frame"))
def test_parse_result_rejects_missing_frame_tags_even_when_collision_is_true(missing):
    matrix = np.eye(4)[None].tolist()
    payload = {"request_hash": "expected", "grasps": matrix, "scores": [1.0], "width_m": [0.04], "depth_m": [0.02], "collision_free": [True], **_PAYLOAD_TAGS}
    payload.pop(missing)
    with pytest.raises(ValueError, match=missing):
        parse_inference_result(payload, "expected")


@pytest.mark.parametrize("field,bad_value", (("translation_reference", None), ("translation_frame", None), ("rotation_frame", "camera")))
def test_parse_result_rejects_null_or_wrong_camera_frame_tags(field, bad_value):
    matrix = np.eye(4)[None].tolist()
    payload = {"request_hash": "expected", "grasps": matrix, "scores": [1.0], "width_m": [0.04], "depth_m": [0.02], "collision_free": [True], **_PAYLOAD_TAGS}
    payload[field] = [bad_value]
    with pytest.raises(ValueError, match=field):
        parse_inference_result(payload, "expected")


@pytest.mark.parametrize("source_frame", (None, "world"))
def test_parse_result_requires_explicit_camera_reconstruction_frame(source_frame):
    matrix = np.eye(4, dtype=float)[None].tolist()
    reconstruction = {
        "dimensions_m": [0.08, 0.06, 0.04],
        "confidence": 1.0,
        "centroid_camera_m": [0.0, 0.0, 1.0],
        "bounds_camera_m": [-0.04, -0.03, 0.98, 0.04, 0.03, 1.02],
    }
    if source_frame is not None:
        reconstruction["source_frame"] = source_frame
    payload = {"request_hash": "expected", "grasps": matrix, "scores": [1.0], **_PAYLOAD_TAGS, "reconstruction": reconstruction}
    with pytest.raises(ValueError, match="source_frame"):
        parse_inference_result(payload, "expected")


def test_process_adapter_calibration_gate_precedes_worker_start_or_prepare():
    adapter = ZeroGraspProcessAdapter(["definitely-not-a-worker"], ZeroGraspConfig())
    with pytest.raises(RuntimeError, match="verified EEF calibration"):
        adapter.infer(_observation(100.0))


def test_process_adapter_retains_filter_audit_when_all_candidates_reject():
    obs = _observation(100.0)
    candidate = GraspCandidate(np.eye(4), score=1.0, width_m=0.04, depth_m=0.02, collision_free=False, **_CANDIDATE_TAGS)
    result = ZeroGraspInferenceResult((candidate,), None, "request", "output", {"backend": "fixture"})
    adapter = ZeroGraspProcessAdapter(["fixture-worker"], ZeroGraspConfig())
    adapter.last_audit = {"candidate_count": 1, "worker_diagnostics": result.diagnostics}
    with pytest.raises(ZeroGraspSelectionError, match="no valid ZeroGrasp candidate"):
        adapter.select(result, obs)
    assert adapter.last_audit["status"] == "failed"
    assert adapter.last_audit["selection_audit"]["candidate_rejections"] == [{"index": 0, "reason": "collision"}]


def test_swept_path_uses_calibrated_eef_endpoint_and_unverified_fails_closed(monkeypatch):
    obs = _observation(100.0)
    mask = np.zeros(obs.depth_m.shape, dtype=bool); mask[20:70, 20:70] = True
    grasp_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    grasp = np.eye(4); grasp[:3, :3] = grasp_rotation; grasp[:3, 3] = [-0.83, -0.83, 1.0]
    candidate = GraspCandidate(grasp, score=1.0, width_m=0.04, depth_m=0.03, collision_free=True, **_CANDIDATE_TAGS)
    cfg_base = ZeroGraspConfig(R_grasp_eef=np.eye(3), R_H_E=np.eye(3), calibration_source="test-probe")
    cfg = replace(cfg_base, eef_calibration_verified=True, calibration_sha256=calibration_sha256(cfg_base), probe_sha256="b" * 64)
    seen = []
    monkeypatch.setattr(zg_adapter, "observed_depth_swept_path_clear", lambda _obs, _start, end, _clearance, **_kwargs: (seen.append(np.asarray(end)) or (True, 1.0)))
    ranked = rank_grasps([candidate], np.eye(4), cfg, eef_position_world_m=np.zeros(3), observation=obs, source_mask=mask)
    expected_pregrasp = compose_T_W_E_from_grasp(
        make_pregrasp(grasp, cfg.pregrasp_distance_m, cfg.approach_axis_local),
        dynamic_T_G_E(candidate, cfg),
    )
    assert ranked and np.allclose(seen[0], expected_pregrasp[:3, 3])
    unverified = rank_grasps([candidate], np.eye(4), ZeroGraspConfig(), eef_position_world_m=np.zeros(3), observation=obs, source_mask=mask)
    assert not unverified


def test_official_fetch_data_camera_payload_is_yaml_projection_shape():
    payload = camera_yaml_payload(np.eye(3))
    assert list(payload) == ["left_p"]
    assert len(payload["left_p"]) == 12
