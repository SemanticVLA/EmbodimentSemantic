"""Independent contracts for the RoboCasa-local canonical controller copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import episode_contract
from vla_benchmarking.robocasa.arrow_grasp_controller.controller import runner
from vla_benchmarking.robocasa.arrow_grasp_controller.legacy_engine.arrow_controller import (
    deproject_endpoint,
)
from vla_benchmarking.robocasa.evaluation.adapter import adapt_capture_to_base
from vla_benchmarking.robocasa.evaluation.capture import CameraCalibration, CapturedRGBD
from vla_benchmarking.robocasa.evaluation.live import RoboCasaControllerEnv


def test_rotated_translated_base_preserves_camera_deprojection() -> None:
    calibration = CameraCalibration(
        camera_name="agentview", width=4, height=4,
        intrinsic=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
        world_frame="world",
        extrinsic_direction="world_from_camera",
    )
    capture = CapturedRGBD(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        normalized_depth=np.ones((4, 4), dtype=np.float32),
        metric_depth=np.ones((4, 4), dtype=np.float32),
        calibration=calibration,
    )
    # B0 is translated and yawed 90 degrees in the world.  The camera-origin
    # world point (0, 0, 1) therefore appears at (-2, 1, 1) in B0.
    theta = np.pi / 2.0
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = [[np.cos(theta), -np.sin(theta), 0.0],
                               [np.sin(theta), np.cos(theta), 0.0],
                               [0.0, 0.0, 1.0]]
    world_from_base[:3, 3] = [1.0, 2.0, 0.0]
    adapted = adapt_capture_to_base(capture, np.linalg.inv(world_from_base))
    point = deproject_endpoint(
        (1.0, 1.0), 1.0,
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        adapted.calibration.world_from_camera,
    )
    assert np.allclose(point, [-2.0, 1.0, 1.0], atol=1e-8)
    assert adapted.calibration.world_frame == "robocasa_pandaomron_base_B0"


def test_base_relative_proprioception_drives_same_osc_direction() -> None:
    class Raw:
        action_space = None

        def reset(self):
            return {
                "robot0_base_pos": np.array([4.0, 2.0, 0.0]),
                "robot0_base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "robot0_base_to_eef_pos": np.array([0.0, 0.0, 0.8]),
                "robot0_base_to_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "robot0_eef_pos": np.array([4.0, 2.0, 0.8]),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }

    env = RoboCasaControllerEnv(Raw())
    env.reset()
    assert np.allclose(env._last_observation["robot0_eef_pos"], [0.0, 0.0, 0.8])
    action = episode_contract.normalized_action_for_waypoint(
        episode_contract._proprioception(env._last_observation),
        {"position": np.array([0.05, 0.0, 0.8])},
        gripper=0.0,
    )
    assert np.allclose(action[:3], [1.0, 0.0, 0.0])


def test_render_depth_is_flipped_once_and_declared() -> None:
    class Sim:
        def __init__(self):
            self.calls = []

        def render(self, **kwargs):
            self.calls.append(kwargs)
            return np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8), np.array([[0.1], [0.2]], dtype=np.float32)

    class Raw:
        sim = Sim()
        action_space = None

    env = RoboCasaControllerEnv(Raw())
    rgb, depth = env.render(camera_name="agentview", width=1, height=2, depth=True)
    assert Raw.sim.calls[-1]["depth"] is True
    assert tuple(rgb[0, 0]) == (4, 5, 6)
    assert np.isclose(depth[0, 0], 0.2)
    assert env._arrow_depth_encoding == "normalized"


def test_render_agentview_uses_selected_physical_fallback_camera() -> None:
    class Sim:
        def __init__(self):
            self.calls = []

        def render(self, **kwargs):
            self.calls.append(kwargs)
            return np.zeros((1, 1, 3), dtype=np.uint8), np.ones((1, 1), dtype=np.float32)

    class Raw:
        sim = Sim()
        action_space = None

    env = RoboCasaControllerEnv(Raw())
    env._arrow_physical_camera_name = "robot0_agentview_right"
    env.render(camera_name="agentview", width=1, height=1, depth=True)
    assert Raw.sim.calls[-1]["camera_name"] == "robot0_agentview_right"


def test_frozen_algorithm_modules_match_libero_exactly() -> None:
    root = Path(__file__).resolve().parents[3]
    pairs = (
        ("libero/arrow_grasp_controller/legacy_engine/arrow_controller.py", "robocasa/arrow_grasp_controller/legacy_engine/arrow_controller.py"),
        ("libero/arrow_grasp_controller/controller/grasp_candidates.py", "robocasa/arrow_grasp_controller/controller/grasp_candidates.py"),
        ("libero/arrow_grasp_controller/controller/molmopoint.py", "robocasa/arrow_grasp_controller/controller/molmopoint.py"),
    )
    for left, right in pairs:
        assert (root / "vla_benchmarking" / left).read_bytes() == (root / "vla_benchmarking" / right).read_bytes()


def test_robocasa_runtime_contains_no_libero_imports() -> None:
    root = Path(__file__).resolve().parents[1] / "arrow_grasp_controller"
    for path in root.rglob("*.py"):
        assert "vla_benchmarking.libero" not in path.read_text(encoding="utf-8"), path


def test_public_runner_uses_high_level_canary_path(monkeypatch, tmp_path) -> None:
    class Env:
        _last_observation = {
            "robot0_base_to_eef_pos": np.array([0.0, 0.0, 0.8]),
            "robot0_base_to_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    calibration = CameraCalibration(
        camera_name="agentview", width=4, height=4,
        intrinsic=[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    capture = CapturedRGBD(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        normalized_depth=np.ones((4, 4), dtype=np.float32),
        metric_depth=np.ones((4, 4), dtype=np.float32),
        calibration=calibration,
    )
    calls = {}

    class Worker:
        def __init__(self, *args):
            self.robot_calibration = args[1]

    monkeypatch.setattr(runner, "build_local_molmo_runtime", lambda: object())
    monkeypatch.setattr(runner, "ModelPerceptionWorker", Worker)
    monkeypatch.setattr(runner, "probe_robot_calibration", lambda _env: (object(), np.eye(3), {"passed": True}))
    monkeypatch.setattr(episode_contract, "decode_arrow_pixels", lambda *_args: (np.array([1.0, 1.0]), np.array([2.0, 2.0])))

    def canary(**kwargs):
        calls.update(kwargs)
        return {"status": "selected"}

    monkeypatch.setattr(runner, "run_canary_episode", canary)
    result = runner.run_episode(
        env=Env(), capture=capture, arrow_rgb=np.zeros_like(capture.rgb),
        bboxes={"src": (0, 0, 1, 1), "dst": (2, 2, 3, 3)},
        output_dir=tmp_path, seed=0, source="src", destination="dst",
        resolution=4, evaluator=lambda _env: True,
    )
    assert result == {"status": "selected"}
    assert calls["worker"].__class__ is Worker
    assert calls["episode_runner"]
