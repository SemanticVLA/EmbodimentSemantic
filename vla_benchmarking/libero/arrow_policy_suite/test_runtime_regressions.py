from __future__ import annotations

import pytest

from .branching import BranchRunner
from .contracts import ActionProposal, ContractError, ObservationFrame, PolicyDecision
from .runtime import ProgressTracker, TransactionalCoordinator, make_proposal


class _Env:
    def __init__(self, result):
        self.value = 0
        self.result = result

    def snapshot(self):
        return self.value

    def restore(self, value):
        self.value = value

    def observe(self):
        return {"state": [float(self.value)] + [0.0] * 7}

    def step(self, _action):
        self.value += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _VLA:
    def propose(self, frame):
        return make_proposal((0.0,) * 7, "vla", frame)

    def commit(self, _record):
        return None


class _Teacher:
    def propose(self, frame):
        return None

    def commit(self, _record):
        return None


class _Policy:
    def decide(self, _frame, base, _teacher):
        return PolicyDecision(base.action, "runtime", base.observation_digest)

    def commit(self, _record):
        return None


def test_gym_terminated_and_truncated_stop_rollout():
    env = _Env(({"state": [1.0] + [0.0] * 7}, 0.0, False, True, {}))
    result = TransactionalCoordinator(env, _VLA(), _Teacher()).run(_Policy(), max_steps=5)
    assert result.stats.terminal is True
    assert result.stats.steps == 1
    assert env.value == 1


def test_step_failure_restores_snapshot_and_fails_closed_without_snapshot():
    env = _Env(RuntimeError("boom"))
    coordinator = TransactionalCoordinator(env)
    frame = coordinator.observe()
    proposal = ActionProposal((0.0,) * 7, timestep=frame.timestep)
    with pytest.raises(RuntimeError):
        coordinator.step(proposal)
    assert env.value == 0


def test_commit_hook_failure_restores_environment_before_propagating():
    class FailingPolicy(_Policy):
        def commit(self, _record):
            raise RuntimeError("commit failed")

    env = _Env({"success": False, "terminal": False})
    coordinator = TransactionalCoordinator(env, _VLA(), _Teacher())
    with pytest.raises(RuntimeError, match="commit failed"):
        coordinator.run(FailingPolicy(), max_steps=1)
    assert env.value == 0
    assert coordinator.current is not None and coordinator.current.timestep == 0


def test_make_proposal_preserves_explicit_timestep_and_digest():
    frame = ObservationFrame({"state": [0.0] * 8}, timestep=3)
    proposal = make_proposal((0.0,) * 7, "vla", frame, timestep=9, observation_digest="explicit")
    assert proposal.timestep == 9
    assert proposal.observation_digest == "explicit"


def test_progress_tracker_waits_for_window_and_allows_safety_interrupt():
    tracker = ProgressTracker(window=2)
    teacher = type("Teacher", (), {"metadata": {"phase_error": 1.0}})()
    assert tracker.should_trigger(teacher) is False
    tracker.update(ObservationFrame({"state": [0.0] * 8}), teacher)
    assert tracker.should_trigger(teacher) is False
    tracker.update(ObservationFrame({"state": [0.0] * 8}, timestep=1), teacher)
    assert tracker.should_trigger(teacher) is True
    safety = type("Teacher", (), {"metadata": {"gripper_conflict": True}})()
    assert tracker.should_trigger(safety) is True


def test_branch_runner_restores_sandbox_after_branch_failure():
    env = _Env({"reward": 1.0})
    runner = BranchRunner(env, action_selector=lambda _mask, index, _state: (float(index),) + (0.0,) * 6)
    with pytest.raises(ContractError):
        runner.run_all()
    assert env.value == 0
