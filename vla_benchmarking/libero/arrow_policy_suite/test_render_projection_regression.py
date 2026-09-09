from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arrow_policy_suite.libero_adapter import LiberoEnvironmentAdapter
from arrow_policy_suite.libero_state import OffScreenRenderState
from arrow_policy_suite.native_host import _proposal_equal, _proposal_mismatch


class _Sim:
    def __init__(self):
        self.value = 0.0

    def get_state(self):
        return self.value

    def set_state(self, value):
        self.value = float(value)

    def forward(self):
        return None


class _ProductionShapeEnv:
    def __init__(self):
        self.sim = _Sim()
        self._elapsed_steps = 0
        self._last_obs = None


def _provider_and_env():
    environment = _ProductionShapeEnv()
    redraw = {"count": 0}

    def observation_fn(_environment):
        redraw["count"] += 1
        value = redraw["count"]
        observation = {
            "agentview": np.full((2, 2, 3), value, dtype=np.uint8),
            "wrist": np.full((2, 2, 3), value + 1, dtype=np.uint8),
            "state": [float(environment._elapsed_steps)] + [0.0] * 7,
        }
        environment._last_obs = observation
        return observation

    return OffScreenRenderState(environment, observation_fn=observation_fn), environment


def test_render_projection_ignores_agentview_wrist_redraw_but_detects_state_mutation():
    provider, environment = _provider_and_env()
    first = provider.snapshot()
    redrawn = provider.snapshot()

    assert _proposal_equal(first, redrawn)

    environment._elapsed_steps = 1
    changed = provider.snapshot()
    assert not _proposal_equal(first, changed)
    mismatch = _proposal_mismatch(first, changed, path="environment")
    assert mismatch is not None
    assert "wrist" not in mismatch[0]
    assert "agentview" not in mismatch[0]


def test_render_projection_allows_deterministic_restore_after_image_redraw():
    provider, environment = _provider_and_env()
    snapshot = provider.snapshot()
    environment._elapsed_steps = 1
    environment._last_obs = {
        "agentview": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": [1.0] + [0.0] * 7,
    }

    provider.restore(snapshot)

    assert environment._elapsed_steps == 0
    assert environment._last_obs["state"][0] == 0.0


def test_adapter_snapshot_projection_ignores_production_wrist_redraw():
    class RawEnvironment:
        def __init__(self):
            self.render_count = 0
            self.state_value = 0.0

        def observe(self):
            self.render_count += 1
            return {
                "agentview": np.full((2, 2, 3), self.render_count, dtype=np.uint8),
                "wrist": np.full((2, 2, 3), self.render_count + 1, dtype=np.uint8),
                "state": [self.state_value] + [0.0] * 7,
            }

        def step(self, _action):
            return {"observation": self.observe()}

    raw = RawEnvironment()
    adapter = LiberoEnvironmentAdapter(
        raw,
        require_images=True,
        snapshot_hook=lambda: "raw-snapshot",
        restore_hook=lambda _payload: None,
    )
    first = adapter.snapshot()
    adapter.observe()
    redrawn = adapter.snapshot()
    assert _proposal_equal(first, redrawn)

    raw.state_value = 1.0
    adapter.observe()
    changed = adapter.snapshot()
    assert not _proposal_equal(first, changed)
    mismatch = _proposal_mismatch(first, changed, path="environment")
    assert mismatch is not None
    assert "wrist" not in mismatch[0]
