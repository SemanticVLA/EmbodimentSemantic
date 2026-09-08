from __future__ import annotations

import numpy as np

from vla_benchmarking.robocasa.evaluation.live import RoboCasaControllerEnv


def _quat_z(angle: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])


class _MovingRaw:
    action_dim = 12
    action_space = None

    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.actions: list[np.ndarray] = []

    def _observation(self) -> dict[str, np.ndarray]:
        if self.reset_count == 1 and self.step_count == 0:
            base = np.asarray([0.0, 0.0, 0.0])
            quat = _quat_z(0.0)
        elif self.reset_count == 1:
            base = np.asarray([1.0, 2.0, 0.0])
            quat = _quat_z(np.pi / 2.0)
        else:
            base = np.asarray([10.0, 0.0, 0.0])
            quat = _quat_z(0.0)
        return {
            "robot0_base_pos": base,
            "robot0_base_quat": quat,
            "robot0_base_to_eef_pos": np.asarray([1.0, 0.0, 0.0]),
            "robot0_base_to_eef_quat": _quat_z(0.0),
            # Deliberately provide world values that would be wrong if they
            # were selected instead of composing the current-base sensor.
            "robot0_eef_pos": base + np.asarray([1.0, 0.0, 0.0]),
            "robot0_eef_quat": quat,
        }

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return self._observation()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float))
        self.step_count += 1
        return self._observation(), 0.0, False, {}


def test_current_base_motion_is_composed_into_frozen_b0() -> None:
    raw = _MovingRaw()
    env = RoboCasaControllerEnv(raw)

    reset = env.reset()
    assert np.allclose(reset["robot0_base_to_eef_pos"], [1.0, 0.0, 0.0])
    assert np.allclose(reset["robot0_eef_pos"], [1.0, 0.0, 0.0])

    step_observation = env.step(np.zeros(7))[0]
    # T_B0G = T_B0W * T_WBt * T_BtG.
    assert np.allclose(step_observation["robot0_base_to_eef_pos"], [1.0, 3.0, 0.0])
    assert np.allclose(step_observation["robot0_eef_pos"], [1.0, 3.0, 0.0])
    assert np.allclose(step_observation["robot0_base_to_eef_quat"], _quat_z(np.pi / 2.0))


def test_b0_delta_is_rotated_into_current_osc_base_and_provenanced() -> None:
    raw = _MovingRaw()
    env = RoboCasaControllerEnv(raw)
    env.reset()
    env.step(np.zeros(7))  # establish the translated, +90-degree current base

    env.step(np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]))
    sent = raw.actions[-1]
    # R_BtB0 = R_WBt.T R_WB0 = Rz(-90 deg).
    assert np.allclose(sent[:3], [0.0, -1.0, 0.0], atol=1e-8)
    assert np.allclose(sent[3:6], [0.0, -1.0, 0.0], atol=1e-8)
    assert sent[6] == -1.0
    record = env._action_history[-1]
    assert record["controller_action_b0"][:6] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert np.allclose(record["controller_action_current_base"][:6], [0.0, -1.0, 0.0, 0.0, -1.0, 0.0])
    assert record["action_frame_provenance"]["input_frame"].endswith("base_B0")
    assert record["action_frame_provenance"]["osc_frame"].endswith("base_current")


def test_zero_base_drift_keeps_canonical_action_unchanged() -> None:
    raw = _MovingRaw()
    env = RoboCasaControllerEnv(raw)
    env.reset()
    action = np.asarray([0.2, -0.3, 0.4, -0.1, 0.3, -0.5, 1.0])
    env.step(action)
    assert np.allclose(raw.actions[-1][:7], action)
    assert np.allclose(env._action_history[-1]["controller_action_current_base"], action)


def test_reset_refreezes_b0_and_clears_prior_episode_history() -> None:
    raw = _MovingRaw()
    env = RoboCasaControllerEnv(raw)
    env.reset()
    env.step(np.zeros(7))
    assert env.steps == 1
    env.reset()

    assert env.steps == 0
    assert env._action_history == []
    assert np.allclose(env._world_from_base_B0[:3, 3], [10.0, 0.0, 0.0])
    assert np.allclose(env._last_observation["robot0_eef_pos"], [1.0, 0.0, 0.0])
