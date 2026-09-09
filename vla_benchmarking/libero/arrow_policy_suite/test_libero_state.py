from __future__ import annotations

import sys
import random
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.libero_state import LiberoRollbackUnavailable, OffScreenRenderState, _state_equal


class _Sim:
    def __init__(self):
        self.value = 0.0

    def get_state(self):
        return self.value

    def set_state(self, state):
        self.value = float(state)

    def forward(self):
        return None


class _Env:
    def __init__(self):
        self.sim = _Sim()
        self.problem_name = "pick_place"
        self.domain_name = "libero"
        self.language_instruction = "pick up the object"
        self._elapsed_steps = 0
        self._done = False
        self._observation = None
        self._arrow_phase_audit = []

    def observe(self):
        self._observation = {"state": [self.sim.value] + [0.0] * 7}
        return self._observation

    def step(self, action):
        self.sim.value += float(action[0])
        self._elapsed_steps += 1
        return {"observation": self.observe(), "done": False}


def test_offscreen_snapshot_restores_sim_wrapper_cache_and_digest():
    env = _Env()
    provider = OffScreenRenderState(env)
    snapshot = provider.snapshot()
    env.step([0.4] + [0.0] * 6)
    env._done = True
    env._arrow_phase_audit.append({"phase": "close"})
    provider.restore(snapshot)
    assert env.sim.value == 0.0
    assert env._elapsed_steps == 0
    assert env._done is False
    assert env._arrow_phase_audit == []
    assert env.observe()["state"][0] == 0.0


def test_offscreen_snapshot_fails_closed_without_sim_state_contract():
    class NoState:
        def __init__(self):
            self.sim = object()

        def observe(self):
            return {"state": [0.0] * 8}

    with pytest.raises(LiberoRollbackUnavailable, match="sim"):
        OffScreenRenderState(NoState())


def test_offscreen_snapshot_treats_libero_task_metadata_as_immutable():
    env = _Env()
    provider = OffScreenRenderState(env)
    snapshot = provider.snapshot()

    assert snapshot.immutable_metadata == {
        "problem_name": "pick_place",
        "domain_name": "libero",
        "language_instruction": "pick up the object",
    }

    # Metadata is not rollback payload: changing it invalidates the episode
    # rather than being silently copied over during simulator restore.
    env.language_instruction = "place the object"
    with pytest.raises(LiberoRollbackUnavailable, match="immutable environment metadata changed"):
        provider.restore(snapshot)


def test_offscreen_restore_rejects_hidden_state_mismatch():
    env = _Env()
    env.hidden = 0
    provider = OffScreenRenderState(env)
    # Hidden state is not part of the explicit rollback contract.  Strict
    # construction fails before a native host can issue an action.
    with pytest.raises(LiberoRollbackUnavailable, match="hidden"):
        provider.snapshot()


def test_offscreen_restore_retains_rng_state():
    env = _Env()
    provider = OffScreenRenderState(env)
    state = random.getstate()
    try:
        random.seed(12345)
        snapshot = provider.snapshot()
        expected_next = random.random()
        random.random()

        provider.restore(snapshot)

        assert random.random() == expected_next
    finally:
        random.setstate(state)


def test_offscreen_restore_ignores_nondeterministic_render_bytes_but_checks_state():
    env = _Env()
    render_counter = {"value": 0}

    def observation_with_redraw(_environment):
        render_counter["value"] += 1
        return {
            "state": [env.sim.value] + [0.0] * 7,
            "agentview": np.array([render_counter["value"]], dtype=np.uint8),
        }

    provider = OffScreenRenderState(env, observation_fn=observation_with_redraw)
    snapshot = provider.snapshot()
    env.step([0.4] + [0.0] * 6)
    provider.restore(snapshot)
    assert env.sim.value == 0.0


def test_offscreen_restore_rejects_simulator_state_corruption_even_when_observation_is_stable():
    class CorruptSim(_Sim):
        def set_state(self, state):
            self.value = float(state) + 1.0

    env = _Env()
    env.sim = CorruptSim()
    provider = OffScreenRenderState(env)
    snapshot = provider.snapshot()
    env.step([0.4] + [0.0] * 6)
    with pytest.raises(LiberoRollbackUnavailable, match="simulator state differs"):
        provider.restore(snapshot)


def test_simulator_state_value_objects_compare_authoritative_fields():
    class SimState:
        __slots__ = ("time", "qpos", "qvel")

        def __init__(self, time, qpos, qvel):
            self.time = time
            self.qpos = qpos
            self.qvel = qvel

    left = SimState(0.5, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    right = SimState(0.5, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    changed = SimState(0.5, np.array([1.0, 3.0]), np.array([0.1, 0.2]))
    assert _state_equal(left, right)
    assert not _state_equal(left, changed)


def test_offscreen_restore_rejects_component_hook_mutating_simulator_after_restore():
    class MutatingComponent:
        def snapshot_state(self):
            return "component-state"

        def restore_state(self, _state):
            env.sim.value += 1.0

    env = _Env()
    env.controller = MutatingComponent()
    provider = OffScreenRenderState(env, observation_fn=lambda _environment: {"state": [0.0] * 8})
    snapshot = provider.snapshot()
    env.step([0.4] + [0.0] * 6)
    with pytest.raises(LiberoRollbackUnavailable, match="simulator state differs"):
        provider.restore(snapshot)
