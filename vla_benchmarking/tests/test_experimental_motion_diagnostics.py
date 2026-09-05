"""Focused contracts for opt-in experimental motion diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from vla_benchmarking.evaluation import run_arrow_pick_place_eval as runner


def _proprio(position=(0.0, 0.0, 0.0)):
    return {
        "robot0_eef_pos": np.asarray(position, dtype=np.float64),
        "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
        "robot0_gripper_qpos": np.zeros(2),
    }


class _DiagnosticEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64))
        return _proprio((0.001, 0.0, 0.0)), 0.0, False, {}


def test_diagnostics_are_opt_in_and_align_nominal_commanded_action_and_poses(monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    env = _DiagnosticEnv()
    waypoints = np.zeros((6, 3), dtype=np.float64)
    waypoints[0] = (0.1, 0.0, 0.0)
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env, waypoints, _proprio(), phase_timeout_steps=1, gripper_dwell_steps=2,
            stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
            experimental_motion_diagnostics=True,
        )
    assert len(env.actions) == 1
    entry = env._arrow_motion_trace[0]
    assert entry["nominal_target_position_m"] == [0.1, 0.0, 0.0]
    assert entry["commanded_target_position_m"] == [0.1, 0.0, 0.0]
    assert entry["action_normalized"] == [0.0] * 7
    assert entry["eef_pos_before_m"] == [0.0, 0.0, 0.0]
    assert entry["eef_pos_after_m"] == [0.001, 0.0, 0.0]
    assert entry["correction_active"] is False
    assert all(value["status"] == "unavailable" for value in entry["controller_telemetry"].values())


def test_default_trace_has_no_experimental_diagnostic_fields(monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    env = _DiagnosticEnv()
    waypoints = np.zeros((6, 3), dtype=np.float64)
    runner._run_motion(
        env, waypoints, _proprio(), phase_timeout_steps=1, gripper_dwell_steps=2,
        stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
    )
    entry = env._arrow_motion_trace[0]
    assert "nominal_target_position_m" not in entry
    assert "controller_telemetry" not in entry


def _plateau_policy():
    return runner.MicroCorrectionPolicy(
        enabled=True, phases=("pregrasp",), plateau_window_steps=1,
        plateau_delta_m=0.0, residual_max_m=0.005, correction_gain=1.0,
        burst_steps=8, max_rounds=1, max_actions=16,
    )


def test_experimental_plateau_bias_is_bounded_and_audited(monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    env = _DiagnosticEnv()
    waypoints = np.zeros((6, 3), dtype=np.float64)
    waypoints[0] = (0.1, 0.0, 0.0)
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env, waypoints, _proprio(), phase_timeout_steps=10, gripper_dwell_steps=2,
            stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
            micro_correction=_plateau_policy(), micro_action_budget=runner._ActionBudget(16),
            action_budget=runner._ActionBudget(1200), experimental_motion_correction=True,
            experimental_motion_diagnostics=True,
        )
    event = env._arrow_micro_correction_audit[0]
    assert event["trigger"] == "residual_plateau"
    assert np.linalg.norm(np.asarray(event["correction_target_m"]) - np.asarray((0.1, 0.0, 0.0))) == pytest.approx(0.005)
    assert max(np.linalg.norm(np.asarray(item["commanded_target_position_m"]) - np.asarray(item["nominal_target_position_m"])) for item in env._arrow_motion_trace if item["correction_active"]) <= 0.005 + 1e-9


def test_nominal_target_convergence_does_not_apply_bias(monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))

    class ConvergingEnv(_DiagnosticEnv):
        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float64))
            return _proprio((0.1, 0.0, 0.0)), 0.0, False, {}

    env = ConvergingEnv()
    waypoints = np.zeros((6, 3), dtype=np.float64)
    waypoints[0] = (0.1, 0.0, 0.0)
    audit = runner._run_motion(
        env, waypoints, _proprio(), phase_timeout_steps=4, gripper_dwell_steps=2,
        stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
        micro_correction=_plateau_policy(), micro_action_budget=runner._ActionBudget(16),
        action_budget=runner._ActionBudget(1200), experimental_motion_correction=True,
        experimental_motion_diagnostics=True,
    )
    assert audit[0]["status"] == "stop"
    assert env._arrow_micro_correction_audit == []
    assert len(env.actions) == 1


def test_experimental_correction_requires_full_burst_and_workspace(monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    waypoints = np.zeros((6, 3), dtype=np.float64)
    waypoints[0] = (0.1, 0.0, 0.0)
    env = _DiagnosticEnv()
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env, waypoints, _proprio(), phase_timeout_steps=8, gripper_dwell_steps=2,
            stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
            micro_correction=_plateau_policy(), micro_action_budget=runner._ActionBudget(16),
            action_budget=runner._ActionBudget(1200), experimental_motion_correction=True,
        )
    assert env._arrow_micro_correction_audit[0]["trigger"] == "insufficient_remaining_budget"
    assert not any(item.get("correction_active") for item in env._arrow_motion_trace)

    env = _DiagnosticEnv()
    with pytest.raises(ValueError, match="workspace validation failed"):
        runner._run_motion(
            env, waypoints, _proprio(), phase_timeout_steps=10, gripper_dwell_steps=2,
            stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
            micro_correction=_plateau_policy(), micro_action_budget=runner._ActionBudget(16),
            action_budget=runner._ActionBudget(1200), experimental_motion_correction=True,
            motion_workspace_bounds={"x": (-1.0, 0.102), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        )
    assert len(env.actions) == 1
