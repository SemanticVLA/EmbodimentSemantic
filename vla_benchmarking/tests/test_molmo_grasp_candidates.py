"""Deterministic frame and filtering tests for Molmo/SAM3 grasp geometry."""

from __future__ import annotations

import numpy as np
import pytest

from vla_benchmarking.molmo_sam3.grasp_candidates import (
    CameraCalibration,
    CandidatePolicy,
    RobotGraspCalibration,
    _hand_volume_obstruction,
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


def _single_world_point_scene(world_point):
    rgb, depth, mask, calibration, robot, policy = _scene()
    x, y, z = np.asarray(world_point, dtype=float)
    u = int(round(calibration.intrinsic[0][0] * x / z + calibration.intrinsic[0][2]))
    v = int(round(calibration.intrinsic[1][1] * y / z + calibration.intrinsic[1][2]))
    mask[:] = False
    depth[:] = np.nan
    depth[v, u] = z
    return rgb, depth, mask, calibration, robot, (u, v)


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


def test_live_pose_ranks_by_position_and_rotation_and_records_audit():
    rgb, depth, mask, calibration, robot, policy = _scene()
    baseline = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=robot,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    target = next(item for item in baseline.candidates if item.yaw_deg == 0.0 and item.insertion_depth_m == 0.0)
    live = RobotGraspCalibration(
        max_aperture_m=robot.max_aperture_m,
        finger_clearance_m=robot.finger_clearance_m,
        pregrasp_distance_m=robot.pregrasp_distance_m,
        current_grip_site_world_m=target.pregrasp_world_m,
        current_rotation_world_grip_site=target.rotation_world_grip_site,
    )
    result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=live,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    first = result.candidates[0]
    assert first.audit["current_pose_available"] is True
    assert first.audit["position_distance_m"] == pytest.approx(0.0)
    assert first.audit["rotation_distance_rad"] == pytest.approx(0.0)
    assert first.audit["motion_score"] == pytest.approx(1.0)
    assert result.audit["current_pose_available"] is True


def test_live_pose_selects_180_degree_jaw_symmetry_and_recomputes_asymmetric_offset():
    rgb, depth, mask, calibration, robot, policy = _scene()
    asymmetric = RobotGraspCalibration(
        contact_to_grip_site_m=(0.02, 0.01, 0.0),
        max_aperture_m=robot.max_aperture_m,
        finger_clearance_m=robot.finger_clearance_m,
        pregrasp_distance_m=robot.pregrasp_distance_m,
    )
    baseline = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=asymmetric,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    target = next(item for item in baseline.candidates if item.yaw_deg == 0.0 and item.insertion_depth_m == 0.0)
    flipped_grasp = target.rotation_world_grasp @ np.diag((1.0, -1.0, -1.0))
    flipped_grip = flipped_grasp @ np.asarray(asymmetric.grasp_to_grip_site)
    flipped_contact = target.contact_world_m
    flipped_grip_site = flipped_contact + flipped_grasp @ np.asarray(asymmetric.contact_to_grip_site_m)
    baseline_live = RobotGraspCalibration(
        grasp_to_grip_site=asymmetric.grasp_to_grip_site,
        contact_to_grip_site_m=asymmetric.contact_to_grip_site_m,
        max_aperture_m=asymmetric.max_aperture_m,
        finger_clearance_m=asymmetric.finger_clearance_m,
        pregrasp_distance_m=asymmetric.pregrasp_distance_m,
        current_grip_site_world_m=target.pregrasp_world_m,
        current_rotation_world_grip_site=target.rotation_world_grip_site,
    )
    baseline_result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=baseline_live,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    live = RobotGraspCalibration(
        grasp_to_grip_site=asymmetric.grasp_to_grip_site,
        contact_to_grip_site_m=asymmetric.contact_to_grip_site_m,
        max_aperture_m=asymmetric.max_aperture_m,
        finger_clearance_m=asymmetric.finger_clearance_m,
        pregrasp_distance_m=asymmetric.pregrasp_distance_m,
        current_grip_site_world_m=flipped_grip_site - np.asarray(asymmetric.approach_axis_world) * asymmetric.pregrasp_distance_m,
        current_rotation_world_grip_site=flipped_grip,
    )
    result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=live,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local"}),
    )
    selected = next(item for item in result.candidates if item.yaw_deg == 0.0 and item.insertion_depth_m == 0.0)
    baseline_selected = next(item for item in baseline_result.candidates if item.yaw_deg == 0.0 and item.insertion_depth_m == 0.0)
    assert selected.audit["jaw_flip"] is True
    assert baseline_selected.candidate_id == selected.candidate_id
    assert np.allclose(selected.rotation_world_grasp, flipped_grasp)
    assert np.allclose(selected.grip_site_world_m, flipped_grip_site)
    assert selected.audit["rotation_distance_rad"] == pytest.approx(0.0)


def test_hand_volume_reports_off_centerline_obstruction_with_pixel_and_primitive_details():
    rgb, depth, mask, calibration, robot, policy = _scene()
    # An observed obstacle away from the centerline, but inside a contact-
    # relative hand sphere.  Its pixel is intentionally outside the target mask.
    depth[22, 44] = 1.0
    live = RobotGraspCalibration(
        max_aperture_m=robot.max_aperture_m,
        finger_clearance_m=robot.finger_clearance_m,
        pregrasp_distance_m=robot.pregrasp_distance_m,
        hand_collision_spheres_grasp=(((0.0, 0.0, -0.03), 0.012),),
        current_grip_site_world_m=(0.0, 0.0, 1.2),
        current_rotation_world_grip_site=np.eye(3),
    )
    result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=live,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local", "obstruction_clearance_m": 0.006}),
    )
    obstruction = [item for item in result.rejected if item.reason == "approach_obstruction"]
    assert obstruction
    detail = next(item.details for item in obstruction if item.details.get("closest_pixel_uv") == [44, 22])
    assert detail["primitive_index"] == 0
    assert detail["segment"] == "pregrasp_to_contact"
    assert detail["aperture_m"] > 0


def test_obstruction_policy_rejects_fabricated_empty_observation():
    rgb, depth, mask, calibration, robot, policy = _scene()
    depth[~mask] = np.nan
    live = RobotGraspCalibration(
        max_aperture_m=robot.max_aperture_m,
        finger_clearance_m=robot.finger_clearance_m,
        pregrasp_distance_m=robot.pregrasp_distance_m,
        hand_collision_spheres_grasp=(((0.0, 0.0, 0.0), 0.01),),
        current_grip_site_world_m=(0.0, 0.0, 1.2),
        current_rotation_world_grip_site=np.eye(3),
    )
    result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=live,
        policy=CandidatePolicy(**{**policy.__dict__, "name": "molmo_local", "obstruction_clearance_m": 0.006}),
    )
    assert not result.candidates
    assert any(item.reason == "approach_obstruction" for item in result.rejected) or any(item.reason == "no_observed_scene" for item in result.rejected)


def test_terminal_target_contact_allowance_does_not_lower_successful_clearance():
    rgb, depth, mask, calibration, robot, policy = _scene()
    live = RobotGraspCalibration(
        max_aperture_m=robot.max_aperture_m,
        finger_clearance_m=robot.finger_clearance_m,
        pregrasp_distance_m=robot.pregrasp_distance_m,
        hand_collision_spheres_grasp=(((0.0, 0.0, 0.0), 0.001),),
        current_grip_site_world_m=(0.0, 0.0, 1.2),
        current_rotation_world_grip_site=np.eye(3),
    )
    result = generate_grasp_candidates(
        rgb=rgb, metric_depth_m=depth, sam_mask=mask, molmo_points=[(32.0, 22.0)],
        calibration=calibration, robot_calibration=live,
        policy=CandidatePolicy(**{
            **policy.__dict__, "name": "molmo_local", "obstruction_clearance_m": 0.006,
            "terminal_contact_allowance_m": 0.2,
        }),
    )
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.audit["obstruction"]["status"] == "ok"
        assert candidate.clearance_m >= 0.0


def test_oriented_box_accepts_point_inside_legacy_large_sphere_but_outside_obb():
    point = np.array((0.03, 0.0, 1.0))
    rgb, depth, mask, calibration, _, pixel = _single_world_point_scene(point)
    assert np.linalg.norm(point - np.array((0.0, 0.0, 1.0))) < 0.05
    details = _hand_volume_obstruction(
        depth=depth, mask=mask, K=np.asarray(calibration.intrinsic), T=np.asarray(calibration.world_from_camera),
        contact=np.array((0.0, 0.0, 1.0)), grip_site=np.array((0.0, 0.0, 1.0)),
        pregrasp=np.array((0.0, 0.0, 1.05)), rotation_world_grasp=np.eye(3), spheres=(),
        boxes=({
            "center_grasp_m": np.zeros(3), "rotation_grasp_box": np.eye(3),
            "half_extents_m": np.full(3, 0.002), "geom_id": 7, "geom_name": "finger", "source": "test",
        },),
        clearance=0.006, terminal_allowance_m=0.012, required_aperture_m=0.04,
    )
    assert details["status"] == "ok"
    assert details["observed_point_count"] == 1
    assert pixel == (44, 32)


def test_oriented_box_rejects_point_inside_box_with_full_primitive_audit():
    point = np.array((0.03, 0.0, 1.0))
    rgb, depth, mask, calibration, _, pixel = _single_world_point_scene(point)
    details = _hand_volume_obstruction(
        depth=depth, mask=mask, K=np.asarray(calibration.intrinsic), T=np.asarray(calibration.world_from_camera),
        contact=np.array((0.0, 0.0, 1.0)), grip_site=np.array((0.0, 0.0, 1.0)),
        pregrasp=np.array((0.0, 0.0, 1.05)), rotation_world_grasp=np.eye(3), spheres=(),
        boxes=({
            "center_grasp_m": np.array((0.03, 0.0, 0.0)), "rotation_grasp_box": np.eye(3),
            "half_extents_m": np.array((0.01, 0.01, 0.01)), "geom_id": 8, "geom_name": "palm", "source": "test",
        },),
        clearance=0.006, terminal_allowance_m=0.012, required_aperture_m=0.04,
    )
    assert details["status"] == "collision"
    assert details["closest_pixel_uv"] == list(pixel)
    assert details["primitive_type"] == "box"
    assert details["geom_id"] == 8
    assert details["primitive_half_extents_m"] == [0.01, 0.01, 0.01]
    assert details["primitive_rotation_world_box"] == np.eye(3).tolist()


def test_rotated_nonzero_center_box_and_current_robot_filter_are_applied():
    angle = np.deg2rad(45.0)
    box_rotation = np.array(((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0)))
    point = np.array((0.025, 0.02, 1.0))
    rgb, depth, mask, calibration, _, pixel = _single_world_point_scene(point)
    box = {
        "center_grasp_m": np.array((0.025, 0.02, 0.0)), "rotation_grasp_box": box_rotation,
        "half_extents_m": np.array((0.01, 0.004, 0.01)), "geom_id": 9, "geom_name": "rotated", "source": "test",
    }
    kwargs = dict(
        depth=depth, mask=mask, K=np.asarray(calibration.intrinsic), T=np.asarray(calibration.world_from_camera),
        contact=np.array((0.0, 0.0, 1.0)), grip_site=np.array((0.0, 0.0, 1.0)),
        pregrasp=np.array((0.0, 0.0, 1.05)), rotation_world_grasp=np.eye(3), spheres=(), boxes=(box,),
        clearance=0.006, terminal_allowance_m=0.012, required_aperture_m=0.04,
    )
    collision = _hand_volume_obstruction(**kwargs)
    assert collision["status"] == "collision"
    assert collision["closest_pixel_uv"] == list(pixel)
    current_box = {**box, "center_world_m": point, "rotation_world_box": box_rotation}
    filtered = _hand_volume_obstruction(**kwargs, ignored_robot_boxes=(current_box,))
    assert filtered["status"] == "no_observed_scene_after_robot_exclusion"
