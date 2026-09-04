from __future__ import annotations

import numpy as np
import pytest

import run_molmo_opening_probe as probe
from run_arrow_pick_place_eval import _ActionBudget


class _Env:
    def __init__(self):
        self.observation = {"eef_pos": np.array([0.1, 0.2, 0.5]), "eef_quat": np.array([0.0, 0.0, 0.0, 1.0])}
        self._molmo_sam3_action_budget = _ActionBudget(1200)
        self._molmo_sam3_action_count = 0
        self.actions = []
        self.openings = [0.040]

    def _get_observations(self, force_update=True):
        return self.observation

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.observation, 0.0, False, {}

    def close(self):
        pass


def test_four_fixed_order_model_free_cases_and_treatment_settling(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.episode, "render_exactly_one_arrow", lambda *args, **kwargs: (np.zeros((4, 4, 3), dtype=np.uint8), {}))
    monkeypatch.setattr(probe.episode, "decode_arrow_pixels", lambda *args, **kwargs: ((1.0, 1.0), (2.0, 2.0)))
    monkeypatch.setattr(probe.episode, "normalized_action_for_waypoint", lambda *_args, gripper, **_kwargs: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]))
    monkeypatch.setattr(probe.episode, "_experimental_controller_telemetry", lambda _env: {"controller_ee_pos_m": [0.1, 0.2, 0.5], "controller_goal_pos_m": [0.1, 0.2, 0.5]})

    def build(_task, _seed, _resolution):
        return _Env()

    def capture(_env, *, resolution, camera_name):
        class Capture:
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        return Capture()

    def arrow(_env, _task, _resolution):
        return {"bboxes": {"bowl": [0, 0, 2, 2]}, "subject": "bowl", "goal_object": "plate"}

    def opening_probe(env):
        return object(), np.eye(3), {"gripper_geometry": {"measured_opening_m": 0.040}}

    def open_fn(_env, **_kwargs):
        return {"status": "completed", "steps": 0}

    def hover_fn(_env, _capture, _source, **_kwargs):
        return {"status": "completed", "hover_world_m": [0.1, 0.2, 0.5], "region_q90_world_z_m": 0.42}

    def preshape_fn(_env, **_kwargs):
        return {"status": "completed", "final_opening_m": 0.040}

    cases = []
    for mode in ("control", "treatment"):
        for seed in (1000, 1001):
            cases.append(probe.run_case(
                mode=mode, seed=seed, output_dir=tmp_path, env_builder=build,
                capture_fn=capture, arrow_input_builder=arrow, open_fn=open_fn,
                hover_fn=hover_fn, probe_fn=opening_probe, preshape_fn=preshape_fn,
            ))
    assert [(item["mode"], item["seed"]) for item in cases] == [
        ("control", 1000), ("control", 1001), ("treatment", 1000), ("treatment", 1001)
    ]
    assert all(item["status"] == "completed" for item in cases)
    assert all("settling" not in item for item in cases[:2])
    assert all(item["settling"]["actions_sent"] == 5 for item in cases[2:])
    assert all(item["candidate_actions"] == 0 and item["evaluator_calls"] == 0 for item in cases)


def test_unsafe_settling_sample_is_recorded_before_fail_closed(tmp_path, monkeypatch):
    env = _Env()
    monkeypatch.setattr(probe.episode, "normalized_action_for_waypoint", lambda *_args, **_kwargs: np.zeros(7))

    def drift_step(_action):
        env.observation["eef_pos"] = np.array([0.13, 0.2, 0.5])
        return env.observation, 0.0, False, {}

    env.step = drift_step
    with pytest.raises(RuntimeError, match="safety envelope"):
        probe._settle_to_original_hover(
            env,
            hover_audit={"hover_world_m": [0.1, 0.2, 0.5], "region_q90_world_z_m": 0.42},
            initial_hover_observation={"eef_pos": [0.1, 0.2, 0.5], "eef_quat": [0.0, 0.0, 0.0, 1.0]},
            output_dir=tmp_path,
            motion_settings={},
        )
    assert env._molmo_opening_settling_audit["actions"][0]["safety_failure"] == "hover_envelope"


def test_control_reproduced_means_the_expected_immediate_failure():
    outcomes = probe._diagnostic_outcomes([
        {"mode": "control", "status": "failed", "preshape": {"failure_reason": "pose_drift"}},
        {"mode": "control", "status": "failed", "preshape": {"failure_reason": "pose_drift"}},
        {"mode": "treatment", "status": "completed"},
        {"mode": "treatment", "status": "completed"},
    ])
    assert outcomes == {"treatment_passed": True, "control_reproduced": True, "control_completed": False}
