from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.libero_state import LiberoRollbackUnavailable, OffScreenRenderState


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


def test_offscreen_restore_rejects_hidden_state_mismatch():
    env = _Env()
    env.hidden = 0
    provider = OffScreenRenderState(env)
    # Hidden state is not part of the explicit rollback contract.  Strict
    # construction fails before a native host can issue an action.
    with pytest.raises(LiberoRollbackUnavailable, match="hidden"):
        provider.snapshot()
