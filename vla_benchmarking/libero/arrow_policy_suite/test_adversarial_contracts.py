from __future__ import annotations

import math

import pytest

from arrow_policy_suite import (
    ActionProposal,
    GeometryAnchors,
    ObservationFrame,
    TogetherPolicy,
    TransactionalCoordinator,
    clip_action,
    state_only_routes,
)
from arrow_policy_suite.runtime import make_proposal


class _Env:
    def __init__(self) -> None:
        self.steps = 0

    def observe(self):
        return {"state": [0.0] * 8}

    def step(self, _action):
        self.steps += 1
        return {"success": False, "terminal": False}

    def snapshot(self):
        return self.steps

    def restore(self, snapshot):
        self.steps = snapshot


class _VLA:
    def reset(self):
        return None

    def propose(self, frame):
        return make_proposal((0.0,) * 7, "vla", frame)

    def commit(self, _record):
        return None


class _StaleTeacher:
    def reset(self):
        return None

    def propose(self, _frame):
        return ActionProposal((0.0,) * 7, "arrow", "stale-digest")

    def commit(self, _record):
        return None


def test_stale_teacher_is_rejected_before_environment_step():
    env = _Env()
    with pytest.raises(ValueError, match="stale"):
        TransactionalCoordinator(env, _VLA(), _StaleTeacher()).run(TogetherPolicy(), max_steps=1)
    assert env.steps == 0


def test_clip_action_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        clip_action((math.nan,) * 7)
    with pytest.raises(ValueError, match="finite"):
        clip_action((math.inf,) * 7)


def test_trace_rejects_ground_truth_provenance_even_with_nonoracle_provider_name():
    with pytest.raises(ValueError, match="ground-truth"):
        GeometryAnchors(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            "world",
            "calibration-1",
            "arrow_rgbd",
            {"source": "sim_ground_truth"},
        )


def test_state_only_routes_accepts_canonical_observation_state_key_and_drops_modalities():
    frame = ObservationFrame(
        {
            "observation.state": [0.0] * 8,
            "action": [0.2] * 7,
            "image": "must-be-dropped",
        },
        0,
        metadata={"episode_id": "ep-1"},
    )
    base = make_proposal((0.0,) * 7, "vla", frame)
    decision = make_proposal((0.0,) * 7, "unused", frame)
    # state_only_routes only requires StepRecord's shape; its source should
    # structurally retain state and omit actions/images.
    from arrow_policy_suite import PolicyDecision, StepRecord

    policy_decision = PolicyDecision(decision.action, "arrow_together", frame.digest)
    record = StepRecord(frame, base, None, policy_decision, frame, success=True)
    routes = state_only_routes([record])
    assert routes == ((("state", (0.0,) * 8),),)
