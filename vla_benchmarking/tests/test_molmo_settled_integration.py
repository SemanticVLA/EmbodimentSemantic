"""Independent production-main smoke for the settled opening retry seam."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np


runner = importlib.import_module("run_molmo_sam3_canary")
matrix = importlib.import_module("run_arrow_pick_place_matrix")
episode = importlib.import_module("run_arrow_pick_place_eval")
settling = importlib.import_module("molmo_sam3.settling")
molmopoint = importlib.import_module("molmo_sam3.molmopoint")


@dataclass
class _Calibration:
    camera_name: str = "agentview"
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "same_sim_render_call"
    intrinsic: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 1.0]))
    world_from_camera: np.ndarray = field(default_factory=lambda: np.eye(4))


class _Capture:
    def __init__(self, serial: int):
        self.serial = serial
        self.rgb = np.full((8, 8, 3), serial % 255, dtype=np.uint8)
        self.metric_depth = np.ones((8, 8), dtype=np.float32)
        self.calibration = _Calibration()


class _Worker:
    def __init__(self, *_args, **_kwargs):
        self.requests = []
        self.robot_calibration = None

    def propose(self, request):
        self.requests.append(request)
        index = len(self.requests)
        return {"candidates": [{
            "candidate_id": f"candidate-{index}",
            "position_world_m": [0.1, 0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "opening_m": 0.04,
            "source_pixel_xy": [2.0, 2.0],
            "score": 1.0,
        }]}


class _Env:
    def __init__(self):
        self._molmo_sam3_action_count = 0


def test_production_main_settled_retry_refreshes_plan_and_capture(tmp_path, monkeypatch):
    """The real canary main must wire settled retry work before the second proposal."""
    events: list[str] = []
    hover_targets: list[float] = []
    settling_targets: list[float] = []
    capture_serial = 0
    run_count = 0

    def capture(_env, *, resolution, camera_name):
        nonlocal capture_serial
        assert resolution == 8
        assert camera_name == runner.AGENTVIEW
        capture_serial += 1
        events.append(f"capture:{capture_serial}")
        return _Capture(capture_serial)

    def build_env(*_args, **_kwargs):
        events.append("build_env")
        return _Env()

    def calibration(_env):
        return SimpleNamespace(name="calibration"), np.eye(3), {
            "gripper_geometry": {"measured_opening_m": 0.04},
            "passed": True,
        }

    def open_gripper(_env, **_kwargs):
        events.append("gripper_open")
        return {"status": "completed", "steps": 1}

    def hover(_env, capture, _source, **_kwargs):
        events.append("hover")
        target_z = 0.5 + float(capture.serial) * 0.001
        hover_targets.append(target_z)
        audit = {
            "status": "completed", "hover_world_m": [0.1, 0.2, target_z],
            "region_q90_world_z_m": 0.4,
        }
        _env._molmo_sam3_observation_hover = audit
        return audit

    def settle(_env, **kwargs):
        events.append("settle")
        settling_targets.append(float(kwargs["hover_audit"]["hover_world_m"][2]))
        return {"status": "completed", "actions_sent": 1}

    def preshape(_env, **_kwargs):
        events.append("preshape")
        return {"status": "completed", "final_opening_m": 0.04}

    def render(rgb, *_args, **_kwargs):
        return rgb, {}

    def decode(_rgb, _rendered):
        return (2.0, 2.0), (3.0, 3.0)

    def raw(_env):
        return {"eef_pos": np.asarray([0.1, 0.2, 0.5]), "eef_quat": np.asarray([0, 0, 0, 1.0])}

    def fake_run_episode(**_kwargs):
        nonlocal run_count
        run_count += 1
        events.append(f"run_episode:{run_count}")
        if run_count == 1:
            raise RuntimeError("synthetic candidate contact failure")
        return {
            "evaluator_success": True,
            "phases": [],
            "status": "placed",
        }

    def fake_matrix_run(**kwargs):
        # Drive the production episode_runner closure; do not replace the
        # canary retry loop or its before_capture callback.
        nonlocal run_count
        run_count = 0
        env = kwargs["env_builder"](4, 1000, 8, suite_mode=kwargs["suite_mode"], controller_variant=kwargs["controller_variant"])
        result = kwargs["episode_runner"](
            env=env, task_id=4, seed=1000, output_dir=tmp_path / kwargs["suite_mode"],
            variant="molmo_dense_agentview", resolution=8, dry_run=False,
            suite_mode=kwargs["suite_mode"], controller_variant=kwargs["controller_variant"],
            bboxes={"bowl": [1, 1, 4, 4]}, subject="bowl", goal_object="plate",
        )
        assert result["canary_manifest"]["attempts"][0]["status"] == "candidate_failed"
        assert result["canary_manifest"]["attempts"][1]["status"] == "selected"
        return {"cells": [result]}

    config = molmopoint.MolmoPointRuntimeConfig()
    fake_runtime = SimpleNamespace(config=config, loaded=True)
    monkeypatch.setattr(runner, "_execution_provenance", lambda **_kwargs: {
        "execution_sha": "a" * 40, "checkout_clean": True, "live_verified": True,
        "provenance_status": "live_verified", "dirty_paths": [],
    })
    monkeypatch.setattr(runner, "_resolved_controller_config_digest", lambda _path: "b" * 64)
    monkeypatch.setattr(runner, "preflight_local_molmo_runtime", lambda *_args, **_kwargs: {
        "model_id": runner.MOLMOPOINT_MODEL_ID,
        "model_revision": runner.MOLMOPOINT_MODEL_REVISION,
        "prompt_id": "rim_clearance", "prompt": "fake", "models_loaded": True,
    })
    monkeypatch.setattr(runner, "ModelPerceptionWorker", _Worker)
    monkeypatch.setattr(runner, "probe_robot_calibration", calibration)
    monkeypatch.setattr(runner, "_perform_gripper_open", open_gripper)
    monkeypatch.setattr(runner, "_perform_observation_hover", hover)
    monkeypatch.setattr(episode, "build_libero_env", build_env)
    monkeypatch.setattr(episode, "capture_agentview", capture)
    monkeypatch.setattr(episode, "render_exactly_one_arrow", render)
    monkeypatch.setattr(episode, "decode_arrow_pixels", decode)
    monkeypatch.setattr(episode, "_raw_observation", raw)
    monkeypatch.setattr(episode, "run_episode", fake_run_episode)
    monkeypatch.setattr(runner, "_recover_after_failed_candidate", lambda *_args, **_kwargs: {
        "retreat_complete": True,
    })
    monkeypatch.setattr(matrix, "run_matrix", fake_matrix_run)
    def refreshed_arrow_inputs(*_args):
        events.append("provider")
        return {"bboxes": {"bowl": [1, 1, 4, 4]}, "subject": "bowl", "goal_object": "plate"}

    monkeypatch.setattr(matrix, "_default_arrow_inputs", refreshed_arrow_inputs)
    monkeypatch.setattr(settling, "_settle_to_original_hover", settle)
    preshape_module = importlib.import_module("molmo_sam3.preshape")
    monkeypatch.setattr(preshape_module, "perform_preshape", preshape)

    assert runner.main([
        "--region-backend", "rgbd", "--variant", "molmo_dense_agentview",
        "--output-dir", str(tmp_path / "main"), "--phase", "prefix",
        "--episodes-per-task", "1", "--opening-profile", "preshape40mm_settled",
    ], molmo_runtime=fake_runtime) == 0
    assert events.count("run_episode:1") == 2
    assert events.count("run_episode:2") == 2
    assert events.count("settle") == 4
    assert events.count("preshape") == 4
    assert capture_serial >= 8
    # Every second proposal follows a fresh planning capture, hover, settle,
    # preshape, and final capture; the callback was not silently ignored.
    assert events.index("settle") < events.index("run_episode:1")
    captures = [i for i, item in enumerate(events) if item.startswith("capture:")]
    retry_capture = captures[2]
    retry_settle = [i for i, item in enumerate(events) if item == "settle"][1]
    assert retry_capture < retry_settle < captures[3], events
    providers = [i for i, item in enumerate(events) if item == "provider"]
    hovers = [i for i, item in enumerate(events) if item == "hover"]
    assert len(providers) == 2
    assert all(provider < hover for provider, hover in zip(providers, hovers[1::2]))
    assert len(hover_targets) == len(set(hover_targets)) == 4
    assert settling_targets == hover_targets
