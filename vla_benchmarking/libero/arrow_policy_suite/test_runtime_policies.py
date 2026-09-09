from __future__ import annotations

import random

import pytest

from arrow_policy_suite.branching import BranchRunner
from arrow_policy_suite.contracts import ActionProposal, ObservationFrame, StepRecord
from arrow_policy_suite.policies import MinimalPolicy, OnCallPolicy, TogetherPolicy
from arrow_policy_suite.runtime import ProgressTracker, make_proposal


def _frame(step: int = 0) -> ObservationFrame:
    return ObservationFrame({"state": [0.0] * 8}, timestep=step)


def test_together_logged_unavailable_is_a_complete_vla_fallback():
    frame = _frame()
    base = make_proposal((0.8, 0.2, 0.1, 0.4, -0.2, 0.3, -1.0), "vla", frame)
    teacher = ActionProposal(
        (0.0,) * 7,
        "arrow",
        frame.timestep,
        metadata={"logged_unavailable": True},
        observation_digest=frame.digest,
    )
    decision = TogetherPolicy().decide(frame, base, teacher)
    assert decision.action == base.action
    assert decision.teacher_used is False
    assert decision.metadata["fallback_reason"] == "teacher_logged_unavailable"
    assert decision.metadata["base_proposal"]["action"] == base.action
    assert decision.metadata["teacher_proposal"]["policy_id"] == "arrow"


def test_progress_trigger_requires_twenty_steps_and_less_than_ten_percent():
    tracker = ProgressTracker()
    teacher = type("Teacher", (), {"metadata": {"phase_error": 1.0}})()
    for _ in range(19):
        assert tracker.should_trigger(teacher) is False
        tracker.update(_frame(tracker.updates), teacher)
    tracker.update(_frame(tracker.updates), teacher)
    assert tracker.window == 20
    assert tracker.should_trigger(teacher) is True

    exact = ProgressTracker()
    for index in range(19):
        exact.update(_frame(index), teacher)
    exact.update(_frame(19), type("Teacher", (), {"metadata": {"phase_error": 0.9}})())
    assert exact.should_trigger(type("Teacher", (), {"metadata": {"phase_error": 0.9}})()) is False

    dwell_only = type("Teacher", (), {"metadata": {"gripper_dwell": True, "phase_error": 1.0}})()
    assert ProgressTracker().should_trigger(dwell_only) is False


def test_on_call_holds_teacher_for_twenty_steps_and_hands_back_at_milestone():
    policy = OnCallPolicy()
    frame = _frame()
    base = make_proposal((0.0,) * 7, "vla", frame)
    conflict = make_proposal((1.0,) * 7, "arrow", frame, gripper_conflict=True, phase_error=1.0)
    decision = policy.decide(frame, base, conflict)
    assert decision.metadata["takeover_started"] is True
    assert decision.teacher_used is True
    for step in range(20):
        current = _frame(step)
        teacher = make_proposal(
            (1.0,) * 7,
            "arrow",
            current,
            phase_error=1.0,
            milestone_complete=(step == 19),
            gripper_conflict=False,
            gripper_dwell=False,
        )
        b = make_proposal((0.0,) * 7, "vla", current)
        d = policy.decide(current, b, teacher)
        record = StepRecord(current, b, teacher, d, _frame(step + 1))
        policy.commit(record)
    next_frame = _frame(21)
    next_base = make_proposal((0.0,) * 7, "vla", next_frame)
    next_teacher = make_proposal((1.0,) * 7, "arrow", next_frame, phase_error=1.0)
    returned = policy.decide(next_frame, next_base, next_teacher)
    assert returned.teacher_used is False
    assert returned.metadata["handback"] is True
    assert returned.metadata["handback_after_teacher_steps"] == 20


class _BranchEnv:
    def __init__(self):
        self.value = 0
        self.steps = 0

    def snapshot(self):
        return (self.value, self.steps)

    def restore(self, state):
        self.value, self.steps = state

    def step(self, action):
        self.value += action[0]
        self.steps += 1
        return {"reward": action[0], "done": False}


class _Stateful:
    def __init__(self):
        self.value = 0

    def snapshot_state(self):
        return self.value

    def restore_state(self, value):
        self.value = value


def test_branch_runner_restores_all_policy_state_rng_and_counts_real_clones():
    env = _BranchEnv()
    vla, teacher, minimal = _Stateful(), _Stateful(), _Stateful()
    random.seed(17)
    before_rng = random.getstate()
    runner = BranchRunner(
        env,
        policies=(vla, teacher, minimal),
        horizon=20,
        action_selector=lambda _mask, _index, _baseline: (0.0,) * 7,
        rng_snapshot=random.getstate,
        rng_restore=random.setstate,
        require_state_isolation=True,
        outcome_fn=lambda mask, _actions, _raws: {"sufficient": mask == 1, "progress": float(mask == 1)},
    )
    results = runner.run_all()
    assert len(results) == 8
    assert runner.cloned_steps == 160
    assert env.snapshot() == (0, 0)
    assert (vla.value, teacher.value, minimal.value) == (0, 0, 0)
    assert random.getstate() == before_rng
    assert results[1].metadata["sufficient"] is True


def test_minimal_runtime_uses_one_selected_twenty_step_burst():
    env = _BranchEnv()
    runner = BranchRunner(
        env,
        horizon=20,
        action_selector=lambda mask, _index, _baseline: ((1.0 if mask == 1 else 0.0),) + (0.0,) * 6,
        outcome_fn=lambda mask, _actions, _raws: {"sufficient": mask == 1, "progress": float(mask == 1)},
    )
    policy = MinimalPolicy(branch_runner=runner)
    frame = _frame()
    base = make_proposal((0.0,) * 7, "vla", frame)
    teacher = make_proposal((1.0,) * 7, "arrow", frame)
    decision = policy.decide(frame, base, teacher)
    assert decision.metadata["branch_masks_evaluated"] == 8
    assert decision.metadata["branch_steps"] == 160
    assert decision.metadata["mask"] == ("translation",)
    for step in range(19):
        policy.commit(type("Record", (), {})())
        current = _frame(step + 1)
        policy.decide(current, make_proposal((0.0,) * 7, "vla", current), make_proposal((1.0,) * 7, "arrow", current))
    # The final commit ends the selected real burst; the next decision is free
    # to branch again rather than silently carrying stale ownership.
    policy.commit(type("Record", (), {})())
    assert policy._burst_remaining == 0
