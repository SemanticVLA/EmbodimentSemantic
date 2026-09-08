from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import runner
from vla_benchmarking.robocasa.evaluation import capture
from vla_benchmarking.robocasa.evaluation import live
from vla_benchmarking.robocasa.tools import probe_runtime


class _ProbeModel:
    nsite = 2
    nbody = 1
    ngeom = 0
    njoint = 0
    nactuator = 0
    nsensor = 0
    site_names = ["mobilebase0_center", "gripper0_right_grip_site"]

    def site(self, index):
        return SimpleNamespace(id=int(index), name=self.site_names[int(index)])


class _ProbeData:
    def __init__(self):
        self.site_xpos = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]], dtype=float)
        self.site_xmat = np.asarray([np.eye(3), np.eye(3)], dtype=float).reshape(2, 9)
        self.body_xmat = np.asarray([np.eye(3)], dtype=float).reshape(1, 9)


class _RawProbeEnv:
    action_dim = 12
    action_space = None
    horizon = 100

    def __init__(self, *, no_op: bool):
        self.no_op = no_op
        controller = SimpleNamespace(
            name="OSC_POSE", input_ref_frame="base", input_type="delta", control_dim=6,
            output_max=np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
            output_min=np.asarray([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
        )
        robot_model = SimpleNamespace(
            eef_name="right_hand",
            base=SimpleNamespace(correct_naming=lambda name: f"mobilebase0_{name}"),
        )
        robot = SimpleNamespace(
            eef_site_id={"right": 1}, robot_model=robot_model,
            part_controllers={"right": controller}, gripper={"right": SimpleNamespace(important_sites={"grip_site": 1})},
        )
        self.sim = SimpleNamespace(model=_ProbeModel(), data=_ProbeData())
        self.robots = [robot]
        self.steps = 0
        self.actions = []

    def _observation(self):
        z = 0.8 if self.no_op else 0.8 + 0.001 * self.steps
        return {
            "robot0_base_pos": np.asarray([0.0, 0.0, 0.0]),
            "robot0_base_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_base_to_eef_pos": np.asarray([0.0, 0.0, z]),
            "robot0_base_to_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    def reset(self):
        self.steps = 0
        return self._observation()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float))
        self.steps += 1
        return self._observation(), 0.0, False, {}

    def close(self):
        return None


def _calibration_ok(_env):
    return object(), np.eye(3), {"passed": True, "site_id": 1, "resolved_site_name": "gripper0_right_grip_site"}


def test_noop_runtime_probe_fails_motion_gate_and_emits_json(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=True)
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    output = tmp_path / "probe.json"

    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000, output=output, execute_motion=True
    )

    assert evidence["success"] is False
    assert evidence["motion_gate_passed"] is False
    assert evidence["action_count"] == 10
    assert output.exists()
    assert json.loads(output.read_text())["success"] is False


def test_positive_runtime_probe_passes_with_exact_canonical_actions(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    output = tmp_path / "probe.json"

    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000, output=output, execute_motion=True
    )

    assert evidence["success"] is True
    assert evidence["z_delta_m"] >= 0.005
    assert evidence["base_torso_zero"] is True
    assert all(np.allclose(action[:7], probe_runtime.CANONICAL_ACTION) for action in raw.actions)


def test_calibration_error_is_persisted_without_motion(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)

    def calibration_error(_env):
        raise RuntimeError("calibration unavailable")

    monkeypatch.setattr(runner, "probe_robot_calibration", calibration_error)
    output = tmp_path / "probe.json"
    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000, output=output, execute_motion=True
    )

    assert evidence["success"] is False
    assert evidence["action_count"] == 0
    assert any(item["stage"] == "calibration" for item in evidence["pre_motion_diagnostics"]["errors"])
    assert output.exists()


def test_official_pre_motion_diagnostics_send_zero_actions(monkeypatch):
    raw = _RawProbeEnv(no_op=False)
    wrapped = probe_runtime.RoboCasaControllerEnv(raw)
    wrapped.reset()
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)

    diagnostics = probe_runtime.collect_pre_motion_diagnostics(wrapped)

    assert diagnostics["calibration_passed"] is True
    assert raw.steps == 0
    assert wrapped.steps == 0


def test_controller_contract_mismatch_is_gated_before_motion(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    raw.robots[0].part_controllers["right"].output_max[0] = 0.1
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)

    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000,
        output=tmp_path / "probe.json", execute_motion=True,
    )

    assert evidence["controller_contract_matches"] is False
    assert evidence["action_count"] == 0
    assert evidence["success"] is False


def test_failed_step_cannot_satisfy_ten_step_gate(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    original_step = raw.step

    def fail_step(action):
        if raw.steps == 4:
            raw.actions.append(np.asarray(action, dtype=float))
            raise RuntimeError("step failed")
        return original_step(action)

    raw.step = fail_step
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000,
        output=tmp_path / "probe.json", execute_motion=True,
    )

    assert evidence["action_count"] == 4
    assert evidence["all_steps_sent"] is False
    assert evidence["success"] is False


def test_close_exception_invalidates_probe_success(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    raw.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000,
        output=tmp_path / "probe.json", execute_motion=True,
    )

    assert evidence["motion_gate_passed"] is True
    assert evidence["success"] is False
    assert any(item["stage"] == "close" for item in evidence["exceptions"])


def test_probe_write_failure_is_nonzero(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    monkeypatch.setattr(probe_runtime, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    monkeypatch.setattr(probe_runtime, "_write_diagnostic_json", lambda *_a, **_k: False)
    evidence = probe_runtime.run_runtime_probe(
        task_name="PickPlaceCounterToStove", seed=1000,
        output=tmp_path / "probe.json", execute_motion=True,
    )

    assert evidence["success"] is False
    assert evidence["exit_code"] == 2


def test_official_cell_close_error_preserves_controller_outcome(monkeypatch, tmp_path):
    raw = _RawProbeEnv(no_op=False)
    raw.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    frame = capture.CapturedRGBD(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        normalized_depth=np.ones((4, 4), dtype=np.float32),
        metric_depth=np.ones((4, 4), dtype=np.float32),
        calibration=capture.CameraCalibration(
            camera_name="agentview", width=4, height=4,
            intrinsic=[[1, 0, 2], [0, 1, 2], [0, 0, 1]],
            world_from_camera=np.eye(4).tolist(),
        ),
    )
    monkeypatch.setattr(live, "create_robocasa_env", lambda *_a, **_k: raw)
    monkeypatch.setattr(live, "_capture", lambda *_a, **_k: frame)
    monkeypatch.setattr(live, "project_task_bboxes", lambda *_a, **_k: ({"src": (0, 0, 1, 1), "dst": (2, 2, 3, 3)}, "src"))
    monkeypatch.setattr(live, "render_bbox_center_arrow", lambda image, *_a, **_k: (image, {"relation_count": 1}))
    monkeypatch.setattr(live, "_source_label", lambda *_a, **_k: "source")
    monkeypatch.setattr(runner, "probe_robot_calibration", _calibration_ok)
    monkeypatch.setattr(runner, "run_episode", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("controller failed")))

    result = live.run_live_cell(
        task_name="PickPlaceCounterToStove", seed=1000,
        output_dir=tmp_path, resolution=4, execute_motion=True,
    )

    assert result["status"] == "controller_failure"
    assert result["terminal_reason"] == "RuntimeError"
    assert result["diagnostic_errors"][-1]["stage"] == "close"
    assert result["motion_diagnostics"]["close_error"]["type"] == "RuntimeError"
    persisted = json.loads((tmp_path / "motion_diagnostics.json").read_text())
    assert persisted["close_error"]["stage"] == "close"
