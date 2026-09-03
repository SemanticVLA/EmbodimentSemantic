"""Deterministic frame and filtering tests for Molmo/SAM3 grasp geometry."""

from __future__ import annotations

import numpy as np
import pytest

from vla_benchmarking.molmo_sam3.grasp_candidates import (
    CameraCalibration,
    CandidatePolicy,
    RobotGraspCalibration,
    generate_grasp_candidates,
)


def _scene(*, focal: float = 400.0, radius: int = 10, depth_value: float = 1.0):
    height = width = 64
    yy, xx = np.indices((height, width))
    mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= radius**2
    depth = np.full((height, width), np.nan, dtype=np.float64)
    depth[mask] = depth_value
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    calibration = CameraCalibration(
        width=width,
        height=height,
        intrinsic=((focal, 0.0, 32.0), (0.0, -focal, 32.0), (0.0, 0.0, 1.0)),
        world_from_camera=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    robot = RobotGraspCalibration(
        max_aperture_m=0.08,
        finger_clearance_m=0.002,
        pregrasp_distance_m=0.05,
    )
    policy = CandidatePolicy(
        molmo_snap_radius_px=5,
        local_seed_radius_px=5,
        rim_support_radius_px=5,
        min_rim_support_pixels=2,
        min_depth_support_pixels=2,
    )
    return rgb, depth, mask, calibration, robot, policy


def test_known_pixel_deprojects_using_signed_fy_and_returns_original_uv():
    rgb, depth, mask, calibration, robot, policy = _scene()
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0)],
        calibration=calibration,
        robot_calibration=robot,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert result.candidates
    candidate = result.candidates[0]
    # K has negative fy: v=22 at z=1 maps to +0.025 world y.
    assert candidate.source_pixel_uv == (32.0, 22.0)
    assert np.allclose(candidate.contact_world_m[:2], (0.0, 0.025), atol=1e-9)
    assert candidate.audit["no_legacy_source_offset"] is True


def test_yaw_grid_changes_jaw_rotation_but_keeps_upright_approach():
    rgb, depth, mask, calibration, robot, policy = _scene()
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0)],
        calibration=calibration,
        robot_calibration=robot,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert {candidate.yaw_deg for candidate in result.candidates} == {-15.0, 0.0, 15.0}
    for candidate in result.candidates:
        assert np.allclose(candidate.rotation_world_grasp[:, 0], (0.0, 0.0, -1.0))
        assert np.allclose(candidate.rotation_world_grasp.T @ candidate.rotation_world_grasp, np.eye(3), atol=1e-7)
        assert np.linalg.det(candidate.rotation_world_grasp) == pytest.approx(1.0)
        assert np.linalg.norm(candidate.quaternion_world_grip_site_xyzw) == pytest.approx(1.0)


def test_dense_policy_generates_many_candidates_and_caps_at_128():
    rgb, depth, mask, calibration, robot, policy = _scene()
    dense = CandidatePolicy(**{**policy.__dict__, "name": "molmo_dense", "max_seeds": 16})
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0)],
        calibration=calibration,
        robot_calibration=robot,
        policy=dense,
    )
    assert len(result.seeds_uv) == 16
    assert len(result.candidates) == 128
    assert result.audit["candidate_grid_size"] == 144
    assert all(candidate.required_aperture_m <= robot.max_aperture_m for candidate in result.candidates)


def test_workspace_and_aperture_filters_are_audited():
    rgb, depth, mask, calibration, robot, policy = _scene()
    tiny_workspace = RobotGraspCalibration(workspace_min_m=(-0.01, -0.01, 0.0), workspace_max_m=(0.01, 0.01, 1.8))
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0)],
        calibration=calibration,
        robot_calibration=tiny_workspace,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert not result.candidates
    assert any(item.reason == "workspace" for item in result.rejected)

    wide = RobotGraspCalibration(max_aperture_m=0.01, finger_clearance_m=0.004)
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0)],
        calibration=calibration,
        robot_calibration=wide,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert not result.candidates
    assert any(item.reason == "aperture_exceeded" for item in result.rejected)


def test_bad_depth_and_outside_molmo_point_fail_without_fallback_in_local_mode():
    rgb, depth, mask, calibration, robot, policy = _scene()
    depth[22, 32] = np.nan
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(0.0, 0.0)],
        calibration=calibration,
        robot_calibration=robot,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert not result.candidates
    assert any(item["reason"] == "outside_mask_or_snap_radius" for item in result.audit["seed_audit"])


def test_duplicate_molmo_points_are_deduplicated_before_grid_expansion():
    rgb, depth, mask, calibration, robot, policy = _scene()
    result = generate_grasp_candidates(
        rgb=rgb,
        metric_depth_m=depth,
        sam_mask=mask,
        molmo_points=[(32.0, 22.0), (32.0, 22.0), {"x": 32.0, "y": 22.0}],
        calibration=calibration,
        robot_calibration=robot,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    assert len(result.seeds_uv) == 1
    assert len(result.candidates) == 9
