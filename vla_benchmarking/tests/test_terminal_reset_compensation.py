from __future__ import annotations

import pytest


pytest.importorskip("lerobot")
import run_lerobot_eval_with_context as runtime


class _PinnedLiberoEnv:
    def __init__(self, success: bool):
        self.init_state_id = 8
        self._reset_stride = 1
        self.success = success

    def step(self, _action):
        if self.success:
            # Simulate the older pinned LiberoEnv implementation which calls
            # reset() internally before returning a terminal transition.
            self.init_state_id += self._reset_stride
        return None, 0.0, self.success, False, {}


def test_terminal_success_counter_is_compensated(monkeypatch):
    monkeypatch.setattr(runtime.lerobot_libero, "LiberoEnv", _PinnedLiberoEnv)
    runtime._patch_libero_env_terminal_reset_compensation()
    env = _PinnedLiberoEnv(success=True)
    env.step(None)
    assert env.init_state_id == 8
    assert env._paired_reset_compensation["detected"] is True


def test_terminal_failure_without_internal_reset_is_unchanged(monkeypatch):
    monkeypatch.setattr(runtime.lerobot_libero, "LiberoEnv", _PinnedLiberoEnv)
    runtime._patch_libero_env_terminal_reset_compensation()
    env = _PinnedLiberoEnv(success=False)
    env.step(None)
    assert env.init_state_id == 8
    assert not hasattr(env, "_paired_reset_compensation")
