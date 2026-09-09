from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.contracts import ActionProposal, ContractError
from arrow_policy_suite.native_runtime import NativeCanaryError, run_native_canary
from arrow_policy_suite.smolvla_adapter import SmolVLAAdapter


def _proposal(frame, policy_id):
    return ActionProposal(
        (0.25,) + (0.0,) * 6, policy_id=policy_id, timestep=frame.timestep,
        observation_digest=frame.digest,
    )


class _Env:
    def __init__(self, *, privileged=False):
        self.value = 0.0
        self.steps = 0
        self.restores = 0
        self.closed = 0
        self.privileged = privileged

    def observe(self):
        result = {"state": [self.value] + [0.0] * 7}
        if self.privileged:
            result["object_pose_gt"] = [0.0, 0.0, 0.0]
        return result

    def step(self, action):
        self.steps += 1
        self.value += action[0]
        return {"success": False, "terminal": self.steps >= 1}

    def snapshot(self):
        return self.value, self.steps

    def restore(self, value):
        self.restores += 1
        self.value, self.steps = value

    def close(self):
        self.closed += 1

class _Producer:
    def __init__(self, policy_id, *, fail_commit=False):
        self.policy_id = policy_id
        self.frames = []
        self.commits = 0
        self.closed = 0
        self.fail_commit = fail_commit

    def reset(self):
        return None

    def propose(self, frame):
        self.frames.append(frame)
        return _proposal(frame, self.policy_id)

    def commit(self, record):
        self.commits += 1
        if self.fail_commit:
            raise RuntimeError("commit failed")

    def close(self):
        self.closed += 1

    def snapshot_state(self):
        return self.commits, len(self.frames)

    def restore_state(self, value):
        self.commits, frame_count = value
        del self.frames[frame_count:]


def test_native_canary_uses_one_shared_frame_and_one_environment_step():
    env = _Env()
    vla = _Producer("smolvla")
    teacher = _Producer("arrow")
    receipt = run_native_canary(env, vla, teacher, max_steps=3)
    assert receipt.status == "completed"
    assert receipt.environment_steps == 1
    assert env.steps == 1
    assert env.closed == 1
    assert vla.closed == 1 and teacher.closed == 1
    assert receipt.terminal is True
    assert vla.frames[0] is teacher.frames[0]
    assert isinstance(receipt.steps[0].frame_digest, str)
    assert receipt.steps[0].proposal_state_unchanged is True


def test_native_canary_rolls_back_when_commit_fails():
    env = _Env()
    with pytest.raises(NativeCanaryError) as exc_info:
        run_native_canary(env, _Producer("smolvla", fail_commit=True), _Producer("arrow"))
    assert exc_info.value.receipt.status == "failed"
    assert exc_info.value.receipt.rollback_count == 1
    assert env.restores == 1
    assert env.steps == 0
    assert env.closed == 1


def test_native_canary_rejects_privileged_observation_before_step():
    env = _Env(privileged=True)
    with pytest.raises(NativeCanaryError, match="privileged"):
        run_native_canary(env, _Producer("smolvla"), _Producer("arrow"))
    assert env.steps == 0
    assert env.closed == 1


def test_native_canary_requires_rollback_hooks():
    class NoRollback:
        def observe(self):
            return {"state": [0.0] * 8}

        def step(self, action):
            return {"terminal": True}

    with pytest.raises(NativeCanaryError, match="snapshot"):
        run_native_canary(NoRollback(), _Producer("smolvla"), _Producer("arrow"))


def test_native_canary_closes_constructed_components_when_factory_fails():
    env = _Env()

    def failing_vla_factory():
        raise RuntimeError("VLA construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        run_native_canary(env, failing_vla_factory, _Producer("arrow"))
    assert env.closed == 1


def test_native_canary_deduplicates_close_for_shared_component():
    env = _Env()
    # A factory may intentionally share one stateful object across boundaries;
    # lifecycle cleanup must still be exactly once.
    class Shared(_Producer):
        def observe(self):
            return {"state": [0.0] * 8}

        def step(self, action):
            return {"terminal": True}

        def snapshot(self):
            return None

        def restore(self, value):
            return None

    shared = Shared("shared", fail_commit=True)
    with pytest.raises(NativeCanaryError):
        run_native_canary(shared, shared, shared)
    assert shared.closed == 1


def test_native_canary_prefers_state_hooks_for_environment_and_components():
    class StateOnlyEnv(_Env):
        snapshot = None
        restore = None

        def snapshot_state(self):
            return self.value, self.steps

        def restore_state(self, value):
            self.restores += 1
            self.value, self.steps = value

    class StatefulProducer(_Producer):
        snapshot = None
        restore = None

        def snapshot_state(self):
            return self.commits, len(self.frames)

        def restore_state(self, value):
            self.commits, frame_count = value
            del self.frames[frame_count:]

    env = StateOnlyEnv()
    vla = StatefulProducer("smolvla", fail_commit=True)
    teacher = StatefulProducer("arrow")
    with pytest.raises(NativeCanaryError):
        run_native_canary(env, vla, teacher)
    assert env.restores == 1
    assert vla.commits == 0
    assert vla.frames == []
    assert teacher.frames == []


def test_native_canary_fails_closed_on_one_sided_component_state_hook():
    class OneSided(_Producer):
        restore_state = None

        def snapshot_state(self):
            return 1

    with pytest.raises(NativeCanaryError, match="only one"):
        run_native_canary(_Env(), OneSided("smolvla"), _Producer("arrow"))


def test_native_canary_captures_optional_policy_state():
    class PolicyState:
        def __init__(self):
            self.value = 0
            self.restored = 0

        def snapshot_state(self):
            return self.value

        def restore_state(self, value):
            self.value = value
            self.restored += 1

        def commit(self, _record):
            self.value += 1
            raise RuntimeError("policy commit failed")

    policy = PolicyState()
    with pytest.raises(NativeCanaryError):
        run_native_canary(_Env(), _Producer("smolvla"), _Producer("arrow"), policy=policy)
    assert policy.value == 0
    assert policy.restored == 1


def test_native_canary_rejects_incomplete_smolvla_mutable_state():
    class MutablePolicy:
        def __call__(self, _observation, _step):
            return [0.0] * 7

    smolvla = SmolVLAAdapter(policy=MutablePolicy())
    assert smolvla.rollback_complete is False
    with pytest.raises(NativeCanaryError, match="incomplete"):
        run_native_canary(_Env(), smolvla, _Producer("arrow"))


def test_smolvla_requires_explicit_stateless_declaration_for_callable_components():
    def inference(_observation, _step):
        return [0.0] * 7

    undeclared = SmolVLAAdapter(inference=inference)
    assert undeclared.rollback_complete is False

    inference.__arrow_stateless__ = True
    declared = SmolVLAAdapter(inference=inference)
    assert declared.rollback_complete is True


def test_native_canary_rejects_noop_environment_restore_during_replay_probe():
    class NoopRestore(_Env):
        def restore(self, _value):
            self.restores += 1

    env = NoopRestore()
    with pytest.raises(NativeCanaryError, match="restore"):
        run_native_canary(env, _Producer("smolvla"), _Producer("arrow"))
    assert env.steps == 1
    assert env.restores >= 1


def test_native_canary_rejects_hidden_environment_state_restore_mismatch():
    class HiddenStateEnv(_Env):
        def __init__(self):
            super().__init__()
            self.hidden = 0

        def observe(self):
            # Deliberately hide mutable state from the student observation.
            return {"state": [self.value] + [0.0] * 7}

        def step(self, action):
            self.hidden += 1
            return super().step(action)

        def snapshot(self):
            return self.value, self.steps, self.hidden

        def restore(self, value):
            self.restores += 1
            self.value, self.steps, _hidden = value
            # Simulate an incomplete restore whose visible observation still
            # matches while privileged/internal state remains changed.

    env = HiddenStateEnv()
    with pytest.raises(NativeCanaryError, match="pre-proposal state"):
        run_native_canary(env, _Producer("smolvla"), _Producer("arrow"))
    assert env.steps == 0
    assert env.hidden == 1


def test_native_canary_rejects_missing_or_stale_proposal_digest():
    env = _Env()

    class MissingDigest(_Producer):
        def propose(self, frame):
            self.frames.append(frame)
            return ActionProposal((0.0,) * 7, policy_id=self.policy_id, timestep=frame.timestep)

    with pytest.raises(NativeCanaryError, match="observation_digest"):
        run_native_canary(env, MissingDigest("smolvla"), _Producer("arrow"))

    class StaleDigest(_Producer):
        def propose(self, frame):
            self.frames.append(frame)
            return ActionProposal(
                (0.0,) * 7, policy_id=self.policy_id, timestep=frame.timestep,
                observation_digest="stale",
            )

    with pytest.raises(NativeCanaryError, match="does not match"):
        run_native_canary(env, StaleDigest("smolvla"), _Producer("arrow"))


def test_native_canary_records_and_rejects_environment_mutation_during_proposals():
    env = _Env()

    class Mutating(_Producer):
        def propose(self, frame):
            env.value += 1.0
            return super().propose(frame)

    with pytest.raises(NativeCanaryError, match="advanced environment state"):
        run_native_canary(env, Mutating("smolvla"), _Producer("arrow"))
    assert env.steps == 0
    assert env.restores == 1
