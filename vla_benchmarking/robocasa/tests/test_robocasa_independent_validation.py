"""Adversarial checks for the RoboCasa calibration and action boundaries.

These tests model the compiled robosuite wrapper observed by the live API probe:
the public object has a raw ``._model`` but no named accessors, while the legacy
name lookup methods exist and raise ``ValueError`` for every alias.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.calibration import (
    probe_panda_grip_site_frame as probe,
)
from vla_benchmarking.robocasa.arrow_grasp_controller.controller import runner
from vla_benchmarking.robocasa.arrow_grasp_controller.controller import object_contact
from vla_benchmarking.robocasa.arrow_grasp_controller.controller.grasp_candidates import RobotGraspCalibration
from vla_benchmarking.robocasa.environment.runtime import compose_panda_omron_action
from vla_benchmarking.robocasa.evaluation import live
from vla_benchmarking.robocasa.evaluation import runner as evaluation_runner
from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
from vla_benchmarking.robocasa.tools import probe_runtime

from .test_robocasa_dynamic_base_frame import _MovingRaw, _quat_z
from .test_robocasa_live_calibration import _compiled_fixture
from .test_robocasa_runtime_probe import _RawProbeEnv, _calibration_ok


def _synthetic_contact_request(tmp_path: Path, *, patch: str = "solid"):
    """Compact raised RGB-D target over a farther support plane."""
    size = 81
    center = size // 2
    yy, xx = np.indices((size, size))
    metric_depth = np.full((size, size), 1.50, dtype=np.float64)
    if patch == "solid":
        target = (np.abs(xx - center) <= 5) & (np.abs(yy - center) <= 2)
    elif patch == "rim":
        radius = np.hypot(xx - center, yy - center)
        target = (radius >= 1.0) & (radius <= 2.0)
    else:
        raise AssertionError(patch)
    metric_depth[target] = 1.0
    if patch == "rim":
        # Keep the arrow-tail anchor on the raised top plane while the
        # annulus supplies a distinct, visible local rim tangent.
        metric_depth[center, center] = 1.0
    calibration = SimpleNamespace(
        width=size,
        height=size,
        # 500 px focal length keeps the 11x5 pixel target compact in metric
        # space.  The proper 180-degree x/y camera flip makes the nearer
        # target plane (depth=1.0) physically above the support (depth=1.5).
        intrinsic=((500.0, 0.0, float(center)), (0.0, 500.0, float(center)), (0.0, 0.0, 1.0)),
        world_from_camera=np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0),
             (0.0, 0.0, -1.0, 1.5), (0.0, 0.0, 0.0, 1.0)),
        ),
        camera_name="synthetic_agentview",
    )
    capture = SimpleNamespace(
        rgb=np.zeros((size, size, 3), dtype=np.uint8),
        metric_depth=metric_depth,
        calibration=calibration,
    )
    request = runner.PerceptionRequest(
        runner.VARIANTS["canonical"], capture, capture, (float(center), float(center)),
        None, (), tmp_path, arrow_rgb=capture.rgb.copy(),
    )
    return request


def _install_native_name_api(monkeypatch, raw_model) -> None:
    names = {
        "site": {3: "gripper0_right_grip_site"},
        "body": {
            0: "gripper0_right_right_gripper",
            1: "robot0_right_hand",
            2: "gripper0_right_leftfinger",
            3: "gripper0_right_rightfinger",
        },
        "geom": {
            0: "gripper0_right_finger1_pad_collision",
            1: "gripper0_right_finger2_pad_collision",
            2: "gripper0_right_hand_collision",
            3: "gripper0_right_finger1_collision",
            4: "gripper0_right_finger2_collision",
        },
    }

    class MjObj:
        mjOBJ_SITE = "site"
        mjOBJ_BODY = "body"
        mjOBJ_GEOM = "geom"

    def id2name(source, object_type, index):
        # A real binding_utils.MjModel is a wrapper.  Native MuJoCo requires
        # its underlying MjModel, so accepting the wrapper here would hide the
        # exact integration bug this test is intended to catch.
        assert source is raw_model
        return names[object_type].get(int(index))

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(mjtObj=MjObj, mj_id2name=id2name),
    )


def _legacy_wrapper(env):
    compiled_model = env.sim.model

    class WrappedModel:
        _model = compiled_model
        nsite = compiled_model.nsite
        nbody = compiled_model.nbody
        ngeom = compiled_model.ngeom

        def __getattr__(self, name):
            if name.endswith("_name2id") or name.endswith("_id2name"):
                raise AttributeError(name)
            if name in {"site", "body", "geom", "site_names", "body_names", "geom_names", "names"}:
                raise AttributeError(name)
            return getattr(compiled_model, name)

        def site_name2id(self, _name):
            raise ValueError("legacy lookup unavailable")

        def body_name2id(self, _name):
            raise ValueError("legacy lookup unavailable")

        def geom_name2id(self, _name):
            raise ValueError("legacy lookup unavailable")

    env.sim.model = WrappedModel()
    return env


def test_controller_native_name_fallback_unwraps_raw_model(monkeypatch) -> None:
    env = _legacy_wrapper(_compiled_fixture())
    _install_native_name_api(monkeypatch, env.sim.model._model)

    calibration, transform, record = runner.probe_robot_calibration(env)

    assert record["resolved_site_name"] == "gripper0_right_grip_site"
    assert record["resolved_body_name"] == "robot0_right_hand"
    assert calibration.grasp_to_grip_site.shape == (3, 3)
    assert np.allclose(transform, probe.RZ_MINUS_90)


def test_standalone_native_name_fallback_unwraps_raw_model(monkeypatch) -> None:
    env = _legacy_wrapper(_compiled_fixture())
    _install_native_name_api(monkeypatch, env.sim.model._model)
    data = SimpleNamespace(
        site_xmat=env.sim.data.site_xmat,
        body_xmat=env.sim.data.xmat,
        site_xpos=env.sim.data.site_xpos,
        body_xpos=env.sim.data.xpos,
    )

    record = probe.probe_grip_site_frame(SimpleNamespace(model=env.sim.model, data=data))

    assert record["passed"] is True
    assert record["site_id"] == 3
    assert record["resolved_site_name"] == "gripper0_right_grip_site"
    assert record["resolved_body_name"] == "robot0_right_hand"


def test_panda_omron_packing_preserves_canonical_arm_and_parked_channels() -> None:
    canonical = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0)
    packed = compose_panda_omron_action(canonical)

    assert packed[:7] == canonical
    assert packed[7:11] == (0.0, 0.0, 0.0, 0.0)
    assert len(packed) == 12


def test_runtime_probe_close_exception_cannot_be_reported_as_success(monkeypatch, tmp_path) -> None:
    raw = _RawProbeEnv(no_op=False)

    def close_with_error():
        raise RuntimeError("close failed")

    raw.close = close_with_error
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)

    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove",
        seed=1000,
        output=tmp_path / "probe.json",
        execute_motion=True,
    )

    assert evidence["exceptions"][-1]["stage"] == "close"
    assert evidence["success"] is False
    assert evidence["exit_code"] != 0


def test_runtime_probe_write_failure_cannot_claim_motion_success(monkeypatch, tmp_path) -> None:
    raw = _RawProbeEnv(no_op=False)
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)

    def fail_write(*_args, **_kwargs):
        raise OSError("evidence path unavailable")

    monkeypatch.setattr(probe_runtime, "_write_diagnostic_json", fail_write)
    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove",
        seed=1000,
        output=tmp_path / "probe.json",
        execute_motion=True,
    )

    assert evidence["success"] is False
    assert evidence["exit_code"] != 0


def test_runtime_probe_rejects_eef_initial_position_inconsistent_with_compiled_site(
    monkeypatch, tmp_path
) -> None:
    raw = _RawProbeEnv(no_op=False)
    # The observation reports z=0.8, while the calibration record says the
    # authoritative compiled EEF site is at z=1.8 in the same reset B0 frame.
    # A finite B0 matrix alone is insufficient provenance for a valid motion
    # canary.
    raw.sim.data.site_xpos[0] = np.asarray([0.0, 0.0, 1.8])

    def calibration_with_mismatched_site(_env):
        return object(), np.eye(3), {
            "passed": True,
            "site_id": 0,
            "resolved_site_name": "gripper0_right_grip_site",
            "gripper_geometry": {"current_grip_site_world_m": [0.0, 0.0, 1.8]},
        }

    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", calibration_with_mismatched_site)

    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove",
        seed=1000,
        output=tmp_path / "probe.json",
        execute_motion=True,
    )

    assert evidence["success"] is False
    assert evidence["motion_gate_passed"] is False


def test_dynamic_base_archives_raw_inputs_and_clips_rotated_normalized_action() -> None:
    class Yaw45Raw(_MovingRaw):
        def _observation(self):
            if self.reset_count == 1 and self.step_count == 0:
                base = np.asarray([0.0, 0.0, 0.0])
                quat = _quat_z(0.0)
            else:
                base = np.asarray([0.0, 0.0, 0.0])
                quat = _quat_z(np.pi / 4.0)
            return {
                "robot0_base_pos": base,
                "robot0_base_quat": quat,
                "robot0_base_to_eef_pos": np.asarray([1.0, 0.0, 0.0]),
                "robot0_base_to_eef_quat": _quat_z(0.0),
                "robot0_eef_pos": base + np.asarray([1.0, 0.0, 0.0]),
                "robot0_eef_quat": quat,
            }

    raw = Yaw45Raw()
    env = live.RoboCasaControllerEnv(raw)
    env.reset()
    initial = live.begin_motion_diagnostics(env)

    # A 45-degree current-base yaw maps [1, 1] to a component above one;
    # the adapter must preserve its normalized action contract after rotation.
    env.step(np.zeros(7))
    env.step(np.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0]))
    transformed = np.asarray(env._action_history[-1]["controller_action_current_base"])
    assert np.all(np.abs(transformed[:6]) <= 1.0)
    assert initial["initial_raw_base_pose"]["robot0_base_pos"] == [0.0, 0.0, 0.0]
    assert initial["initial_raw_proprioception"]["robot0_base_to_eef_pos"] == [1.0, 0.0, 0.0]
    assert env._action_history[-1]["action_frame_provenance"]["rotation_current_base_from_B0"]

    finalized = live.finalize_motion_diagnostics(env, initial, outcome="test")
    assert finalized["raw_base_pose_initial"]["robot0_base_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert finalized["raw_base_pose_final"]["robot0_base_quat"] == list(_quat_z(np.pi / 4.0))
    assert finalized["raw_proprioception_initial"]["robot0_base_to_eef_pos"] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("patch", ["solid", "rim"])
def test_object_contact_real_generator_returns_finite_straddle_candidates(tmp_path, patch):
    request = _synthetic_contact_request(tmp_path, patch=patch)

    class _SyntheticMolmo:
        def predict(self, model_request):
            return SimpleNamespace(
                points=(SimpleNamespace(x=40.0, y=40.0),),
                provenance={"fixture": "synthetic_rgbd"},
            )

    # A measured hand sphere activates the same local hand-volume collision
    # path used by the live calibration; it is parked away from the target.
    robot = RobotGraspCalibration(
        hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)],
    )
    result = object_contact.propose_object_contact(
        molmo=_SyntheticMolmo(), request=request, robot_calibration=robot,
        prompt=object_contact_prompt("cube"),
    )

    assert result["candidates"], result["diagnostics"]
    candidate = result["candidates"][0]
    assert np.all(np.isfinite(candidate.contact_world_m))
    assert np.all(np.isfinite(candidate.rotation_world_grasp))
    assert np.allclose(candidate.contact_world_m, [0.0, 0.0, 0.5])
    assert candidate.required_aperture_m <= robot.max_aperture_m
    if patch == "solid":
        # The 5-pixel short side is 8 mm at this focal length; the jaw
        # aperture must still include both finger clearances.
        assert candidate.required_aperture_m >= 0.008 + 2.0 * robot.finger_clearance_m
    assert abs(float(candidate.jaw_axis_world[1])) > abs(float(candidate.jaw_axis_world[0]))
    assert candidate.audit["source_frame"] == "original_image_uv"
    assert candidate.audit["obstruction"]["status"] == "ok"


def test_object_contact_v2_blocked_path_rejects_all_hypotheses(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    # The added observed point is just outside the target mask and lies in
    # the hand-volume path.  A large measured sphere makes the obstruction
    # deterministic while preserving the RGB-D-only collision check.
    request.source_capture.metric_depth[40, 45] = 1.0
    robot = RobotGraspCalibration(
        hand_collision_spheres_grasp=[([0.0, 0.0, 0.0], 0.05)],
    )

    result = object_contact.propose_object_contact_v2(
        molmo=SimpleNamespace(
            predict=lambda _request: SimpleNamespace(
                points=(SimpleNamespace(x=40.0, y=40.0),),
                provenance={"fixture": "blocked_path"},
            )
        ),
        request=request,
        robot_calibration=robot,
        prompt=object_contact_prompt("cube"),
    )

    assert not result["candidates"]
    assert result["diagnostics"]["rejection_count"] == 18
    assert {item["reason"] for item in result["diagnostics"]["rejections"]} == {
        "approach_obstruction"
    }


def test_object_contact_v2_candidate_output_respects_canonical_cap(tmp_path):
    request = _synthetic_contact_request(tmp_path, patch="solid")
    robot = RobotGraspCalibration(
        hand_collision_spheres_grasp=[([0.3, 0.3, 0.3], 0.001)],
    )
    result = object_contact.propose_object_contact_v2(
        molmo=SimpleNamespace(
            predict=lambda _request: SimpleNamespace(
                points=(SimpleNamespace(x=40.0, y=40.0),),
                provenance={"fixture": "cap"},
            )
        ),
        request=request,
        robot_calibration=robot,
        prompt=object_contact_prompt("cube"),
    )

    assert len(result["candidates"]) <= 128
    assert result["diagnostics"]["aggregate"]["returned_count"] <= 128
    assert all(
        np.isfinite(candidate.contact_world_m).all()
        and np.isfinite(candidate.rotation_world_grasp).all()
        for candidate in result["candidates"]
    )
    assert any(
        candidate.audit.get("approach_label") == "camera_ray"
        for candidate in result["candidates"]
    )


def test_object_contact_rejects_arrow_image_with_different_rgbd_shape(tmp_path):
    request = _synthetic_contact_request(tmp_path)
    request = replace(request, arrow_rgb=np.zeros((80, 80, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="same-frame"):
        object_contact.propose_object_contact(
            molmo=SimpleNamespace(predict=lambda _request: SimpleNamespace(points=())),
            request=request,
            robot_calibration=RobotGraspCalibration(),
            prompt=object_contact_prompt("cube"),
        )


def test_object_contact_zero_depth_fails_closed_without_model_fallback(tmp_path):
    request = _synthetic_contact_request(tmp_path)
    request.source_capture.metric_depth[...] = 0.0

    with pytest.raises(ValueError, match="finite metric depth"):
        object_contact.propose_object_contact(
            molmo=SimpleNamespace(predict=lambda _request: SimpleNamespace(points=())),
            request=request,
            robot_calibration=RobotGraspCalibration(),
            prompt=object_contact_prompt("cube"),
        )


def test_object_contact_invalid_model_points_are_rejected(tmp_path, monkeypatch):
    request = _synthetic_contact_request(tmp_path)
    calls = []

    def no_candidates(**kwargs):
        calls.append(kwargs)
        return object_contact.GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    monkeypatch.setattr(object_contact, "generate_grasp_candidates", no_candidates)
    result = object_contact.propose_object_contact(
        molmo=SimpleNamespace(
            predict=lambda _request: SimpleNamespace(
                points=(SimpleNamespace(x=float("nan"), y=40.0),),
                provenance={"fixture": "invalid"},
            )
        ),
        request=request,
        robot_calibration=RobotGraspCalibration(),
        prompt=object_contact_prompt("cube"),
    )

    assert not calls
    assert not result["candidates"]
    assert result["diagnostics"]["seed_diagnostics"][0]["status"] == "rejected"


def test_object_contact_seed_budget_caps_admitted_points(tmp_path, monkeypatch):
    request = _synthetic_contact_request(tmp_path)
    calls = []

    def no_candidates(**kwargs):
        calls.append(kwargs)
        return object_contact.GraspCandidateResult((), (), (), kwargs["policy"].name, {})

    monkeypatch.setattr(object_contact, "generate_grasp_candidates", no_candidates)
    points = tuple(SimpleNamespace(x=40.0, y=40.0) for _ in range(20))
    result = object_contact.propose_object_contact(
        molmo=SimpleNamespace(predict=lambda _request: SimpleNamespace(points=points)),
        request=request,
        robot_calibration=RobotGraspCalibration(),
        prompt=object_contact_prompt("cube"),
    )
    budget = result["diagnostics"]["seed_budget"]
    assert budget["accepted_before_cap"] == 20
    assert budget["accepted_after_cap"] == 16
    assert budget["dropped_count"] == 4
    assert len(calls) == 16


def test_evaluation_cli_and_live_cell_forward_contact_profile(monkeypatch, tmp_path):
    cli_call = {}
    monkeypatch.setattr(
        evaluation_runner,
        "run",
        lambda **kwargs: (cli_call.update(kwargs) or 0),
    )
    assert evaluation_runner.main([
        "--mode", "smoke", "--output-dir", str(tmp_path),
        "--tasks", "PickPlaceCounterToStove",
        "--grasp-profile", "object_contact_v1",
    ]) == 0
    assert cli_call["grasp_profile"] == "object_contact_v1"

    live_call = {}
    monkeypatch.setattr(evaluation_runner.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        live,
        "run_live_cell",
        lambda **kwargs: (live_call.update(kwargs) or {
            "status": "controller_failure", "error": "fixture", "terminal_reason": "RuntimeError",
        }),
    )
    row = evaluation_runner._run_live_cell(
        task=evaluation_runner.PICK_PLACE_TASKS[0], episode_index=0, seed=1000,
        output_dir=tmp_path / "cell", mode="smoke", experiment_identity="fixture",
        execute_motion=True, grasp_profile="object_contact_v1",
    )
    assert live_call["grasp_profile"] == "object_contact_v1"
    assert row.metadata["controller_identity"] == {}


def test_worker_contact_profile_reaches_object_contact_proposer(monkeypatch, tmp_path):
    request = _synthetic_contact_request(tmp_path)
    calls = {}
    sentinel = {"candidates": (), "diagnostics": {"fixture": True}}

    def fake_propose(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(object_contact, "propose_object_contact", fake_propose)
    worker = runner.ModelPerceptionWorker(
        SimpleNamespace(), RobotGraspCalibration(),
        effective_prompt=object_contact_prompt("cube"),
        grasp_profile="object_contact_v1",
    )

    assert worker.propose(request) is sentinel
    assert calls["request"] is request
    assert calls["prompt"] == object_contact_prompt("cube")


def test_grasp_profile_identity_is_distinct_from_canonical():
    canonical = evaluation_runner._grasp_profile_identity("canonical_rim")
    contact = evaluation_runner._grasp_profile_identity("object_contact_v1")
    assert canonical["name"] == "canonical_rim"
    assert contact["name"] == "object_contact_v1"
    assert "{noun}" in contact["config"]["prompt"]
    assert canonical != contact
