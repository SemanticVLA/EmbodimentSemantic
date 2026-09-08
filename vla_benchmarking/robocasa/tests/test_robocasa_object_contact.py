from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import object_contact
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.grasp_candidates import (
    GraspCandidateResult,
    RobotGraspCalibration,
)
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner import ModelPerceptionWorker, PerceptionRequest, VARIANTS
from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt


class _Capture:
    def __init__(self, size: int = 7):
        self.rgb = np.zeros((size, size, 3), dtype=np.uint8)
        self.metric_depth = np.ones((size, size), dtype=np.float64)
        self.calibration = SimpleNamespace(
            width=size,
            height=size,
            intrinsic=((80.0, 0.0, 3.0), (0.0, 80.0, 3.0), (0.0, 0.0, 1.0)),
            world_from_camera=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            camera_name="agentview",
        )


class _Molmo:
    def __init__(self, points=(), *, error: Exception | None = None):
        self.points = tuple(points)
        self.error = error
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(points=self.points, provenance={"model": "fixture"})


def _request(tmp_path, capture, arrow=None):
    return PerceptionRequest(
        VARIANTS["canonical"], capture, capture, (3.0, 3.0), None, (), tmp_path,
        arrow_rgb=np.zeros_like(capture.rgb) if arrow is None else arrow,
    )


def test_prompt_is_green_arrow_contact_prompt_without_rim_wording():
    prompt = object_contact_prompt("small solid")
    assert prompt == (
        "Point to visible grasp contact locations on the small solid at the tail, or "
        "start, of the green arrow. Ignore the arrowhead and destination. Choose "
        "locations where a parallel-jaw gripper can descend from above, "
        "straddle the object, and close without touching nearby objects or support "
        "surfaces. Point only on the small solid."
    )
    assert "rim" not in prompt.lower()


def test_contact_queries_exact_arrow_without_mask_and_uses_point_local_patch(tmp_path, monkeypatch):
    capture = _Capture()
    arrow = np.full_like(capture.rgb, (0, 166, 107))
    molmo = _Molmo((SimpleNamespace(x=4.0, y=3.0), SimpleNamespace(x=0.0, y=0.0)))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    monkeypatch.setattr(object_contact, "generate_grasp_candidates", fake_generate)
    result = object_contact.propose_object_contact(
        molmo=molmo, request=_request(tmp_path, capture, arrow),
        robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("cube"),
    )

    assert len(molmo.requests) == 1
    assert molmo.requests[0].mask is None
    assert np.array_equal(molmo.requests[0].rgb, arrow)
    assert len(calls) == 1
    assert calls[0]["policy"].name == "molmo_local"
    assert calls[0]["policy"].max_seeds == 1
    # A 0.015 m point-local patch on this fixture is much smaller than the
    # whole frame, so nearby broad support cannot become the target mask.
    assert int(calls[0]["sam_mask"].sum()) < capture.metric_depth.size
    assert any(item.get("reason") == "outside_arrow_anchor_gate" for item in result["diagnostics"]["seed_diagnostics"])
    assert result["diagnostics"]["admission_gate"]["distance_frame"] == "same_frame_rgbd_world"
    assert result["diagnostics"]["input_image"]["status"] == "saved"


def test_empty_decoded_points_use_exactly_one_arrow_tail_anchor(tmp_path, monkeypatch):
    capture = _Capture()
    molmo = _Molmo(())
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    monkeypatch.setattr(object_contact, "generate_grasp_candidates", fake_generate)
    result = object_contact.propose_object_contact(
        molmo=molmo, request=_request(tmp_path, capture),
        robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("bowl"),
    )

    assert len(calls) == 1
    assert calls[0]["molmo_points"][0].label == "arrow_tail_anchor"
    assert sum(item.get("status") == "local_patch" for item in result["diagnostics"]["seed_diagnostics"]) == 1
    assert result["diagnostics"]["molmopoint_count"] == 0


def test_distinct_capture_inputs_are_content_addressed_and_preserved(tmp_path, monkeypatch):
    capture = _Capture()
    molmo = _Molmo(())

    monkeypatch.setattr(
        object_contact,
        "generate_grasp_candidates",
        lambda **kwargs: GraspCandidateResult((), (), (), kwargs["policy"].name, {}),
    )
    first = object_contact.propose_object_contact(
        molmo=molmo, request=_request(tmp_path, capture, np.zeros_like(capture.rgb)),
        robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("bowl"),
    )["diagnostics"]["input_image"]
    second_arrow = np.full_like(capture.rgb, 7)
    second = object_contact.propose_object_contact(
        molmo=molmo, request=_request(tmp_path, capture, second_arrow),
        robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("bowl"),
    )["diagnostics"]["input_image"]

    assert first["status"] == second["status"] == "saved"
    assert first["path"] != second["path"]
    assert first["sha256"] != second["sha256"]
    assert first["path"].endswith(f"{first['sha256']}.png")
    assert second["path"].endswith(f"{second['sha256']}.png")
    assert all(Path(item["path"]).is_file() for item in (first, second))


@pytest.mark.parametrize("mutation", [
    lambda profile: profile["local_geometry"].update(rim_local_radius_m=0.02),
    lambda profile: profile.update(prompt="tampered prompt"),
])
def test_tampered_profile_is_rejected_before_molmo_call(tmp_path, monkeypatch, mutation):
    profile = object_contact.object_contact_profile()
    mutation(profile)
    profile_path = tmp_path / "object_contact_v1.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(object_contact, "_PROFILE_PATH", profile_path)
    capture = _Capture()
    molmo = _Molmo(())
    with pytest.raises(ValueError, match="profile"):
        object_contact.propose_object_contact(
            molmo=molmo, request=_request(tmp_path, capture),
            robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("bowl"),
        )
    assert molmo.requests == []


def test_arbitrary_nonempty_runtime_prompt_is_rejected_before_molmo_call(tmp_path):
    capture = _Capture()
    molmo = _Molmo(())
    with pytest.raises(ValueError, match="instantiate"):
        object_contact.propose_object_contact(
            molmo=molmo, request=_request(tmp_path, capture),
            robot_calibration=RobotGraspCalibration(), prompt="Point anywhere on the bowl.",
        )
    assert molmo.requests == []


def test_nonempty_model_error_does_not_fallback_to_arrow_anchor(tmp_path):
    capture = _Capture()
    molmo = _Molmo(error=RuntimeError("decoder failed"))
    with pytest.raises(RuntimeError, match="decoder failed"):
        object_contact.propose_object_contact(
            molmo=molmo, request=_request(tmp_path, capture),
            robot_calibration=RobotGraspCalibration(), prompt=object_contact_prompt("bowl"),
        )


def test_worker_profile_is_explicit_and_canonical_default_remains(tmp_path):
    class _NoCall:
        def predict(self, request):
            raise AssertionError("canonical fixture should not enter object-contact branch")

    worker = ModelPerceptionWorker(_NoCall(), RobotGraspCalibration())
    assert worker.grasp_profile == "canonical_rim"
    worker_contact = ModelPerceptionWorker(
        _NoCall(), RobotGraspCalibration(), grasp_profile="object_contact_v1",
        effective_prompt=object_contact_prompt("bowl"),
    )
    assert worker_contact.grasp_profile == "object_contact_v1"
