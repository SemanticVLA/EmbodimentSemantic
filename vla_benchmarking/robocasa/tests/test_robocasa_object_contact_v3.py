"""Focused tests for v3 upper-surface promotion and target-aware sweep."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import object_contact
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.grasp_candidates import (
    RobotGraspCalibration,
    generate_grasp_candidates,
)
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.object_contact_collision import (
    _hand_sweep,
    _support_surface_exemptions,
    filter_target_aware_candidates,
)
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.policy import canonical_candidate_policy
from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt

from .test_robocasa_independent_validation import _synthetic_contact_request
from .test_robocasa_object_contact import _Molmo


def test_v3_promotes_low_seed_on_raised_solid(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    request.source_capture.metric_depth[40, 44] = 1.50
    request.source_capture.metric_depth[41, 40] = 1.01
    request = replace(request, source_uv=(40.0, 41.0))
    result = object_contact.propose_object_contact_v3(
        molmo=_Molmo((SimpleNamespace(x=40.0, y=41.0),)), request=request,
        robot_calibration=RobotGraspCalibration(hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)]),
        prompt=object_contact_prompt("cube"),
    )
    local = [item for item in result["diagnostics"]["seed_diagnostics"] if item.get("status") == "local_patch"]
    assert local and local[0]["promotion"] == "promoted_upper_surface"
    assert local[0]["spatial_distance_m"] <= 0.015
    assert local[0]["component_pixels"] > 0
    assert local[0]["mask_sha256"] == result["diagnostics"]["geometry_audits"][0].get("mask_sha256", local[0]["mask_sha256"])


def test_v3_rejects_unbounded_coplanar_counter_component(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    request.source_capture.metric_depth[...] = 1.50
    with pytest.raises(ValueError, match="coplanar support"):
        object_contact._promote_upper_surface(
            request.source_capture, seed_uv=(40.0, 40.0),
            seed_world=object_contact._deproject(request.source_capture, (40.0, 40.0)),
            component_radius_m=0.0195,
        )


def test_v3_bowl_rim_component_is_valid(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="rim")
    result = object_contact.propose_object_contact_v3(
        molmo=_Molmo((SimpleNamespace(x=40.0, y=40.0),)), request=request,
        robot_calibration=RobotGraspCalibration(hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)]),
        prompt=object_contact_prompt("bowl"),
    )
    assert result["diagnostics"]["profile"] == "object_contact_v3"
    assert any(item.get("promotion") == "promoted_upper_surface" for item in result["diagnostics"]["seed_diagnostics"])


def test_v3_preserves_thin_diagonal_rim_support(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    request.source_capture.metric_depth[...] = 1.5
    for index in range(36, 45):
        request.source_capture.metric_depth[index, index] = 1.0
    promoted, _, component, audit = object_contact._promote_upper_surface(
        request.source_capture, seed_uv=(40.0, 40.0),
        seed_world=object_contact._deproject(request.source_capture, (40.0, 40.0)),
        component_radius_m=0.0195,
    )
    assert int(component.sum()) == 9
    assert audit["connectivity"] == 8
    raw = generate_grasp_candidates(
        rgb=request.source_capture.rgb,
        metric_depth_m=request.source_capture.metric_depth,
        sam_mask=component,
        molmo_points=(promoted,),
        calibration=object_contact._capture_calibration(request.source_capture),
        robot_calibration=RobotGraspCalibration(),
        policy=replace(canonical_candidate_policy(), name="molmo_local", max_seeds=1, obstruction_clearance_m=None),
    )
    assert raw.candidates


def test_v3_external_wall_remains_obstruction(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    # Same-height observed wall immediately outside the target component.
    request.source_capture.metric_depth[39:42, 46] = 1.0
    seed_world = object_contact._deproject(request.source_capture, (40.0, 40.0))
    promoted, _, target_mask, promotion_audit = object_contact._promote_upper_surface(
        request.source_capture, seed_uv=(40.0, 40.0), seed_world=seed_world,
        component_radius_m=0.0195,
    )
    # The RGB-D component is only a proposal; even if a same-height wall is
    # connected to it, target-aware collision must keep that point in the
    # observed sweep unless the terminal pad relation certifies contact.
    assert bool(target_mask[40, 46])
    assert promotion_audit["connectivity"] == 8
    robot = RobotGraspCalibration(hand_collision_spheres_grasp=[([0.0, 0.0, 0.0], 0.010)])
    policy = replace(canonical_candidate_policy(), name="molmo_local", max_seeds=1, obstruction_clearance_m=None)
    raw = generate_grasp_candidates(
        rgb=request.source_capture.rgb, metric_depth_m=request.source_capture.metric_depth,
        sam_mask=target_mask, molmo_points=(promoted,),
        calibration=object_contact._capture_calibration(request.source_capture),
        robot_calibration=robot, policy=policy,
    )
    filtered = filter_target_aware_candidates(
        raw, capture=request.source_capture, target_mask=target_mask,
        robot_calibration=robot, clearance_m=0.006,
    )
    assert raw.candidates
    assert not filtered.candidates
    assert any(item.reason == "target_aware_approach_obstruction" for item in filtered.rejected)
    details = next(item.details for item in filtered.rejected if item.reason == "target_aware_approach_obstruction")
    assert details["offender_in_target_component"] is True
    assert details["certified_target_contact"] is False
    assert details["target_component_excluded"] is False
    assert details["target_contact_relation"] == "late_finger_or_pad_only"


def test_v3_measured_pad_prism_allows_compact_target_at_terminal_contact(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    seed_world = object_contact._deproject(request.source_capture, (40.0, 40.0))
    promoted, _, target_mask, _ = object_contact._promote_upper_surface(
        request.source_capture, seed_uv=(40.0, 40.0), seed_world=seed_world,
        component_radius_m=0.0195,
    )
    policy = replace(canonical_candidate_policy(), name="molmo_local", max_seeds=1, obstruction_clearance_m=None)
    raw = generate_grasp_candidates(
        rgb=request.source_capture.rgb, metric_depth_m=request.source_capture.metric_depth,
        sam_mask=target_mask, molmo_points=(promoted,),
        calibration=object_contact._capture_calibration(request.source_capture),
        robot_calibration=RobotGraspCalibration(), policy=policy,
    )
    assert raw.candidates
    robot = RobotGraspCalibration(hand_collision_boxes_grasp=[
        {"center_grasp_m": (0.0, -0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger1_pad_collision"},
        {"center_grasp_m": (0.0, 0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger2_pad_collision"},
    ])
    filtered = filter_target_aware_candidates(
        raw, capture=request.source_capture, target_mask=target_mask,
        robot_calibration=robot, clearance_m=0.006,
    )
    assert filtered.candidates
    audit = filtered.candidates[0].audit["target_aware_collision"]
    assert audit["contact_prism"]["status"] == "certified"
    assert audit["contact_prism"]["certified_contact_points"] > 0


def test_v3_robot_occluded_arrow_tail_cannot_fallback(tmp_path, monkeypatch):
    request = _synthetic_contact_request(tmp_path)
    robot = RobotGraspCalibration(
        current_grip_site_world_m=(0.0, 0.0, 0.5),
        current_rotation_world_grip_site=np.eye(3),
        hand_collision_boxes_grasp=[{
            "center_grasp_m": (0.0, 0.0, 0.0),
            "rotation_grasp_box": np.eye(3),
            "half_extents_m": (0.20, 0.20, 0.20),
            "geom_name": "gripper_hand_collision",
        }],
    )
    calls = []
    monkeypatch.setattr(object_contact, "generate_grasp_candidates", lambda **kwargs: calls.append(kwargs))
    result = object_contact.propose_object_contact_v3(
        molmo=_Molmo(()), request=request, robot_calibration=robot,
        prompt=object_contact_prompt("cube"),
    )
    assert result["diagnostics"]["anchor_status"] == "robot_occluded"
    assert not calls
    assert any(item.get("reason") == "robot_occluded" for item in result["diagnostics"]["seed_diagnostics"])


def test_v3_profile_identity_is_distinct():
    assert object_contact.object_contact_v3_profile()["name"] == "object_contact_v3"
    assert object_contact.object_contact_v3_profile_sha256() != object_contact.object_contact_v2_profile_sha256()


def test_v4_split_keeps_ambiguous_outer_support_out_of_contact_mask(tmp_path):
    """Sparse geometry may help the frozen engine without certifying a counter."""
    request = _synthetic_contact_request(tmp_path, patch="solid")
    depth = request.source_capture.metric_depth
    depth[...] = 1.50
    depth[39:42, 20:61] = 1.0
    seed_world = object_contact._deproject(request.source_capture, (40.0, 40.0))
    promoted, promoted_world, contact_mask, promotion_audit = object_contact._promote_upper_surface(
        request.source_capture,
        seed_uv=(40.0, 40.0), seed_world=seed_world,
        component_radius_m=0.040, reject_outer_annulus=False,
        select_generator_anchor=True,
    )
    assert promotion_audit["outer_annulus_ambiguous"] is True
    from vla_benchmarking.robocasa.arrow_grasp_controller.controller.object_contact import _geometry_support_mask_for_promotion
    geometry_mask, geometry_audit = _geometry_support_mask_for_promotion(
        request.source_capture, contact_mask=contact_mask,
        promoted_world=promoted_world, promotion_audit=promotion_audit,
    )
    assert geometry_audit["status"] == "outer_annulus_geometry_only"
    assert geometry_audit["geometry_support_outer_pixels"] > 0
    assert geometry_mask.sum() > contact_mask.sum()
    assert np.all(geometry_mask[contact_mask])
    assert geometry_audit["geometry_support_contact_exempted"] is False
    assert promoted.label == "promoted_upper_surface"


def test_v4_profile_identity_is_distinct_and_valid():
    prompt = object_contact_prompt("cube")
    profile = object_contact._validate_object_contact_v4_profile(prompt)
    assert profile["name"] == "object_contact_v4"
    assert object_contact.object_contact_v4_profile_sha256() != object_contact.object_contact_v3_profile_sha256()


def test_v4_runtime_records_split_mask_provenance(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    result = object_contact.propose_object_contact_v4(
        molmo=_Molmo((SimpleNamespace(x=40.0, y=40.0),)), request=request,
        robot_calibration=RobotGraspCalibration(
            hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)]
        ),
        prompt=object_contact_prompt("cube"),
    )
    assert result["diagnostics"]["profile"] == "object_contact_v4"
    local = [item for item in result["diagnostics"]["seed_diagnostics"] if item.get("status") == "local_patch"]
    assert local
    assert local[0]["contact_mask_sha256"] == local[0]["mask_sha256"]
    assert len(local[0]["generator_mask_sha256"]) == 64


def test_v4_blocker_away_axis_uses_only_non_target_same_frame_pixel(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    request.source_capture.metric_depth[40, 44] = 1.50
    contact = object_contact._deproject(request.source_capture, (40.0, 40.0))
    result = object_contact._blocker_away_approach(
        request.source_capture,
        contact_world=contact,
        rejection_details=({
            "reason": "target_aware_approach_obstruction",
            "offender_in_target_component": False,
            "primitive_name": "gripper0_right_finger1_pad",
            "segment_fraction": 0.75,
            "closest_offending_pixel_uv": [44, 40],
        },),
    )
    assert result is not None
    axis, audit = result
    assert np.isfinite(axis).all()
    # Blocker is to +X; the approach points contact->blocker so the generated
    # pregrasp moves to -X, away from that blocker.
    assert axis[0] > 0.0
    assert audit["approach_label"] == "blocker_away"
    assert audit["basis"] == "same_frame_rgbd_non_target_rejection"
    assert audit["approach_tilt_deg"] <= 55.0


def test_v4_blocker_away_rejects_target_or_malformed_offender(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    contact = object_contact._deproject(request.source_capture, (40.0, 40.0))
    assert object_contact._blocker_away_approach(
        request.source_capture, contact_world=contact,
        rejection_details=({
            "reason": "target_aware_approach_obstruction",
            "offender_in_target_component": True,
            "primitive_name": "finger1_pad_collision",
            "segment_fraction": 0.0,
            "closest_offending_pixel_uv": [44, 40],
        },),
    ) is None


def test_v4_blocker_away_requires_both_primary_hypotheses():
    blocked = (
        {"approach_label": "world_down", "executed": True, "survivor_count": 0, "non_target_obstruction": True},
        {"approach_label": "camera_ray", "executed": True, "survivor_count": 0, "non_target_obstruction": True},
    )
    assert object_contact._blocker_away_recovery_allowed(blocked)
    assert not object_contact._blocker_away_recovery_allowed(blocked[:1])
    assert not object_contact._blocker_away_recovery_allowed((
        blocked[0], {**blocked[1], "non_target_obstruction": False},
    ))
    assert not object_contact._blocker_away_recovery_allowed((
        blocked[0], {**blocked[1], "executed": False},
    ))


def test_v5_wider_admission_gate_is_profile_scoped(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    result = object_contact.propose_object_contact_v5(
        molmo=_Molmo((SimpleNamespace(x=40.0, y=40.0),)), request=request,
        robot_calibration=RobotGraspCalibration(),
        prompt=object_contact_prompt("cube"),
    )
    assert result["diagnostics"]["profile"] == "object_contact_v5"
    gate = result["diagnostics"]["admission_gate"]
    calibration = RobotGraspCalibration()
    assert gate["threshold_m"] == pytest.approx(
        0.5 * calibration.max_aperture_m + calibration.finger_clearance_m
    )


def test_v5_reflex_ladder_is_low_tilt_and_deterministic(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    request.source_capture.metric_depth[40, 44] = 1.50
    contact = object_contact._deproject(request.source_capture, (40.0, 40.0))
    hypotheses = object_contact._blocker_away_ladder(
        request.source_capture, contact_world=contact,
        rejection_details=({
            "reason": "target_aware_approach_obstruction",
            "offender_in_target_component": False,
            "primitive_name": "finger1_pad_collision",
            "segment_fraction": 0.0,
            "closest_offending_pixel_uv": [44, 40],
        },),
    )
    assert [audit["ladder_tilt_deg"] for _, audit in hypotheses] == [15.0, 30.0, 45.0]
    assert all(audit["approach_label"] == "blocker_away_ladder" for _, audit in hypotheses)
    assert all(audit["approach_tilt_deg"] <= 45.0 for _, audit in hypotheses)


def test_v3_admission_metadata_is_profile_fixed_for_decoded_points(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    result = object_contact.propose_object_contact_v3(
        molmo=_Molmo((SimpleNamespace(x=40.0, y=40.0),)), request=request,
        robot_calibration=RobotGraspCalibration(),
        prompt=object_contact_prompt("cube"),
    )
    gate = result["diagnostics"]["admission_gate"]
    assert gate["distance_frame"] == "same_frame_rgbd_image_metric_ellipse"
    assert gate["threshold_m"] == pytest.approx(0.5 * RobotGraspCalibration().max_aperture_m)


def test_v3_admission_metadata_is_profile_fixed_for_empty_fallback(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    result = object_contact.propose_object_contact_v3(
        molmo=_Molmo(()), request=request,
        robot_calibration=RobotGraspCalibration(),
        prompt=object_contact_prompt("cube"),
    )
    gate = result["diagnostics"]["admission_gate"]
    assert gate["distance_frame"] == "same_frame_rgbd_image_metric_ellipse"
    assert gate["threshold_m"] == pytest.approx(0.5 * RobotGraspCalibration().max_aperture_m)


def test_support_exemption_requires_below_surface_and_local_footprint():
    world_down = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)))
    candidate = SimpleNamespace(
        contact_world_m=np.asarray([0.0, 0.0, 0.50]),
        rotation_world_grasp=world_down,
    )
    contact_audit = {
        "status": "certified",
        "capture_extent_x_m": [-0.020, 0.020],
        "capture_extent_y_m": [-0.025, 0.025],
        "capture_extent_z_m": [-0.020, 0.020],
    }
    worlds = np.asarray([
        [0.0, 0.0, 0.50],   # validated target surface
        [0.010, -0.020, 0.49],  # local support below the target
        [0.050, 0.0, 0.49],  # outside pad footprint
        [0.010, 0.0, 0.495],  # below surface by less than 6 mm
        [0.010, -0.020, 0.0],  # far below the bounded support band
    ])
    flags, audit = _support_surface_exemptions(
        worlds, target_flags=np.asarray([True, False, False, False, False]),
        candidate=candidate, contact_audit=contact_audit,
        robot_calibration=RobotGraspCalibration(),
        clearance_m=0.006,
    )
    assert flags.tolist() == [False, True, False, False, False]
    assert audit["reason"] == "terminal_pad_footprint_support_below_promoted_contact"
    assert audit["support_exempted_points"] == 1


def test_support_exemption_never_certifies_missing_or_misaligned_target():
    candidate = SimpleNamespace(contact_world_m=np.asarray([0.0, 0.0, 0.50]))
    worlds = np.asarray([[0.0, 0.0, 0.0]])
    flags, audit = _support_surface_exemptions(
        worlds, target_flags=np.asarray([False]), candidate=candidate,
        contact_audit={},
        robot_calibration=RobotGraspCalibration(), clearance_m=0.006,
    )
    assert not flags.any()
    assert audit["reason"] == "no_target_component"


def test_terminal_support_penetration_remains_a_sweep_collision():
    """Support handling may waive a safety band, never a geometric penetration."""
    contact = np.asarray([0.0, 0.0, 0.50])
    candidate = SimpleNamespace(
        contact_world_m=contact,
        grip_site_world_m=contact,
        pregrasp_world_m=contact + np.asarray([0.0, 0.0, 0.08]),
        rotation_world_grasp=np.eye(3),
    )
    # The left pad's inward face is y=-0.025.  This observed point is 1 mm
    # inside that pad at the terminal pose, so it must remain collision-active.
    worlds = np.asarray([
        contact,
        [0.0, -0.026, 0.49],
    ])
    flags = np.asarray([True, False])
    boxes = [
        {"center_grasp_m": (0.0, -0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger1_pad_collision"},
        {"center_grasp_m": (0.0, 0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger2_pad_collision"},
    ]
    valid, audit = _hand_sweep(
        candidate,
        worlds=worlds,
        pixels_vu=np.asarray([[40, 40], [40, 41]]),
        target_flags=flags,
        robot_calibration=RobotGraspCalibration(hand_collision_boxes_grasp=boxes),
        calibration=SimpleNamespace(),
        clearance_m=0.006,
    )
    assert not valid
    assert audit["reason"] == "target_aware_approach_obstruction"
    assert audit["support_surface"]["temporal_gate"] == "terminal_only"


def test_support_footprint_is_invariant_to_equivalent_jaw_flip():
    """A physical world-down grasp and its jaw flip must classify identically."""
    contact = np.asarray([0.0, 0.0, 0.50])
    world_down = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)))
    jaw_flip = world_down @ np.diag([1.0, -1.0, -1.0])
    boxes = [
        {"center_grasp_m": (0.0, -0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger1_pad_collision"},
        {"center_grasp_m": (0.0, 0.030, 0.0), "rotation_grasp_box": np.eye(3),
         "half_extents_m": (0.020, 0.005, 0.020), "geom_name": "finger2_pad_collision"},
    ]
    worlds = np.asarray([contact, [-0.021, 0.0, 0.49]])
    flags = np.asarray([True, False])
    common = {
        "grip_site_world_m": contact,
        "pregrasp_world_m": contact + np.asarray([0.0, 0.0, 0.08]),
    }
    outcomes = []
    for rotation in (world_down, jaw_flip):
        candidate = SimpleNamespace(contact_world_m=contact, rotation_world_grasp=rotation, **common)
        outcomes.append(_hand_sweep(
            candidate,
            worlds=worlds,
            pixels_vu=np.asarray([[40, 40], [40, 41]]),
            target_flags=flags,
            robot_calibration=RobotGraspCalibration(hand_collision_boxes_grasp=boxes),
            calibration=SimpleNamespace(),
            clearance_m=0.006,
        ))
    assert [valid for valid, _ in outcomes] == [True, True]
    assert all(audit["support_surface"]["support_exempted_points"] == 1 for _, audit in outcomes)
