from __future__ import annotations

import numpy as np
import pytest

from grasp_controller import preshape
from run_arrow_pick_place_eval import _ActionBudget


class _Env:
    def __init__(self):
        self.observation = {
            "eef_pos": np.array([0.1, 0.2, 0.4]),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        self._grasp_controller_action_budget = _ActionBudget(1200)
        self.actions = []

    def _get_observations(self, force_update=True):
        return self.observation

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.observation, 0.0, False, {}


def test_fixed_opening_sends_full_close_then_five_zero_holds(tmp_path, monkeypatch):
    env = _Env()
    monkeypatch.setattr(
        preshape.episode, "normalized_action_for_waypoint",
        lambda *_args, gripper, **_kwargs: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]),
    )
    readings = iter([0.050, 0.044, 0.0400, 0.0401, 0.0399, 0.0400, 0.0400])
    callbacks = []
    audit = preshape.perform_preshape(
        env, measure_opening_fn=lambda _env: next(readings), output_dir=tmp_path,
        motion_started_callback=lambda: callbacks.append(True),
    )
    assert audit["status"] == "completed"
    assert audit["actions_sent"] == 6
    assert callbacks == [True]
    assert [float(action[-1]) for action in env.actions] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert env._grasp_controller_action_budget.used == 6
    assert (tmp_path / "preshape_audit.json").exists()


def test_below_band_fails_closed_without_opening_or_action(tmp_path):
    env = _Env()
    with pytest.raises(preshape.PreshapeError) as error:
        preshape.perform_preshape(
            env, measure_opening_fn=lambda _env: 0.030, output_dir=tmp_path,
        )
    assert error.value.category == "below_band"
    audit = error.value.audit
    assert audit["failure_reason"] == "below_band"
    assert env.actions == []
    assert env._grasp_controller_action_budget.used == 0
    assert (tmp_path / "preshape_audit.json").exists()


def test_failed_step_is_charged_and_audit_is_returned(tmp_path, monkeypatch):
    env = _Env()
    monkeypatch.setattr(
        preshape.episode, "normalized_action_for_waypoint",
        lambda *_args, **_kwargs: np.zeros(7),
    )

    def fail_step(_action):
        raise RuntimeError("transport")

    env.step = fail_step
    readings = iter([0.050, 0.044])
    with pytest.raises(preshape.PreshapeError) as error:
        preshape.perform_preshape(
            env, measure_opening_fn=lambda _env: next(readings), output_dir=tmp_path,
        )
    audit = error.value.audit
    assert error.value.category == "motion_failed"
    assert audit["actions_sent"] == 1
    assert env._grasp_controller_action_budget.used == 1
    assert audit["actions"][0]["command"] == "close"
