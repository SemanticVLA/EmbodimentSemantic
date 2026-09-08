"""Focused tests for the exploratory v2 approach hypotheses."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import object_contact
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.grasp_candidates import (
    CameraCalibration,
    CandidatePolicy,
    RobotGraspCalibration,
    generate_grasp_candidates,
)
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.policy import canonical_candidate_policy
from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt

from .test_robocasa_independent_validation import _synthetic_contact_request
from .test_robocasa_object_contact import _Molmo


def _tiny_frame() -> tuple[np.ndarray, np.ndarray, CameraCalibration]:
    depth = np.ones((9, 9), dtype=np.float64)
    mask = np.ones((9, 9), dtype=bool)
    calibration = CameraCalibration(
        width=9, height=9,
        intrinsic=((100.0, 0.0, 4.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0)),
        world_from_camera=np.eye(4).tolist(),
    )
    return np.zeros((9, 9, 3), dtype=np.uint8), mask, calibration


def test_canonical_rejects_tilted_approach_as_before() -> None:
    rgb, mask, calibration = _tiny_frame()
    tilted = RobotGraspCalibration(approach_axis_world=(0.5, 0.0, -np.sqrt(0.75)))
    with pytest.raises(ValueError, match="upright grasp approach_axis_world"):
        generate_grasp_candidates(
            rgb=rgb, metric_depth_m=np.ones(mask.shape), sam_mask=mask,
            molmo_points=((4.0, 4.0),), calibration=calibration,
            robot_calibration=tilted, policy=CandidatePolicy(name="molmo_local"),
        )


def test_v2_accepts_bounded_camera_ray_axis_and_labels_audit(tmp_path) -> None:
    request = _synthetic_contact_request(tmp_path)
    capture = request.source_capture
    yy, xx = np.indices(capture.metric_depth.shape)
    center = capture.metric_depth.shape[0] // 2
    mask = (np.abs(xx - center) <= 5) & (np.abs(yy - center) <= 2)
    ray = (np.sin(np.deg2rad(40.0)), 0.0, -np.cos(np.deg2rad(40.0)))
    result = object_contact._generate_for_approach(
        capture=capture, robot_calibration=RobotGraspCalibration(),
        policy=replace(canonical_candidate_policy(), name="molmo_local", obstruction_clearance_m=None),
        point=object_contact.MolmoPoint(float(center), float(center)),
        mask=mask, approach_axis_world=np.asarray(ray), approach_label="camera_ray",
    )
    assert result.audit["approach_label"] == "camera_ray"
    assert np.asarray(result.audit["approach_axis_world"]) == pytest.approx(ray)
    assert result.candidates
    assert all("approach_camera_ray" in candidate.candidate_id for candidate in result.candidates)
    assert all(candidate.audit["approach_label"] == "camera_ray" for candidate in result.candidates)


def test_over_tilted_hypothesis_fails_closed() -> None:
    _, _, calibration = _tiny_frame()
    capture = SimpleNamespace(
        metric_depth=np.ones((9, 9), dtype=np.float64),
        calibration=calibration,
    )
    camera_origin = np.asarray(calibration.world_from_camera, dtype=np.float64)[:3, 3]
    over_tilted = camera_origin + np.asarray((np.sin(np.deg2rad(60.0)), 0.0, -0.5), dtype=np.float64)
    with pytest.raises(ValueError, match="camera_ray_over_tilted_or_upward"):
        object_contact._camera_ray_approach(capture, over_tilted)


def test_v2_rechecks_original_workspace_after_virtual_rotation(tmp_path) -> None:
    request = _synthetic_contact_request(tmp_path)
    # The virtual enclosing AABB is intentionally wider than this original
    # workspace for a tilted camera ray.  Candidates outside B0 must still be
    # rejected after the rigid inverse transform.
    robot = RobotGraspCalibration(
        workspace_min_m=(-0.02, -0.02, 0.45),
        workspace_max_m=(0.02, 0.02, 0.60),
    )
    ray = np.asarray((np.sin(np.deg2rad(40.0)), 0.0, -np.cos(np.deg2rad(40.0))))
    result = object_contact._generate_for_approach(
        capture=request.source_capture,
        robot_calibration=robot,
        policy=replace(canonical_candidate_policy(), name="molmo_local", obstruction_clearance_m=None),
        point=object_contact.MolmoPoint(40.0, 40.0),
        mask=np.ones_like(request.source_capture.metric_depth, dtype=bool),
        approach_axis_world=ray,
        approach_label="camera_ray",
    )
    assert not result.candidates
    assert any(item.reason == "workspace_original_frame" for item in result.rejected)


def test_v2_fallback_audits_all_model_points_rejected(tmp_path, monkeypatch) -> None:
    request = _synthetic_contact_request(tmp_path)
    direct_calls = []
    virtual_calls = []

    def fake_generate(**kwargs):
        direct_calls.append(kwargs)
        return object_contact.GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    def fake_virtual(**kwargs):
        virtual_calls.append(kwargs)
        return object_contact.GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    monkeypatch.setattr(object_contact, "generate_grasp_candidates", fake_generate)
    monkeypatch.setattr(object_contact, "_generate_for_approach", fake_virtual)
    result = object_contact.propose_object_contact_v2(
        molmo=_Molmo((SimpleNamespace(x=0.0, y=0.0),)), request=request,
        robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("cube"),
    )
    fallback = [item for item in result["diagnostics"]["seed_diagnostics"] if item.get("reason") == "all_model_points_rejected"]
    assert len(fallback) == 1
    assert len(direct_calls) == 1
    assert [call["approach_label"] for call in virtual_calls] == ["camera_ray"]
    assert result["diagnostics"]["seed_budget"]["fallback_mode"] == "empty_or_all_model_points_rejected"


def test_v2_solid_and_rim_candidates_are_finite_and_deterministic(tmp_path) -> None:
    for patch in ("solid", "rim"):
        request = _synthetic_contact_request(tmp_path / patch, patch=patch)
        robot = RobotGraspCalibration(hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)])
        molmo = _Molmo((SimpleNamespace(x=40.0, y=40.0),))
        first = object_contact.propose_object_contact_v2(
            molmo=molmo, request=request, robot_calibration=robot,
            prompt=object_contact_prompt("cube"),
        )
        second = object_contact.propose_object_contact_v2(
            molmo=_Molmo((SimpleNamespace(x=40.0, y=40.0),)), request=request,
            robot_calibration=robot, prompt=object_contact_prompt("cube"),
        )
        assert [item.candidate_id for item in first["candidates"]] == [item.candidate_id for item in second["candidates"]]
        for candidate in first["candidates"]:
            assert np.isfinite(candidate.contact_world_m).all()
            assert np.isfinite(candidate.rotation_world_grasp).all()
            assert candidate.audit["approach_label"] in {"world_down", "camera_ray"}
