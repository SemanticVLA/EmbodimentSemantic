from __future__ import annotations

import pytest

from .contracts import ContractError, EpisodeSpec, EpisodeStatus, SourceState, TeacherRecoveryResult, TransitionRecord
from .experiment import EnvironmentHandle, _step_result, initial_state_hash, run_episode


class Env:
    def __init__(self):
        self.t = 0
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1
        return {"state": [0]}

    def observe(self):
        return {"state": [self.t]}

    def step(self, _action):
        self.t += 1
        return {"observation": {"state": [self.t]}, "done": False}

    def close(self):
        self.close_calls += 1


class Policy:
    def reset(self, _description, _seed):
        pass

    def act(self, _observation):
        return [0.0] * 7


def _spec():
    return EpisodeSpec("lifecycle-0", 0, 1, "task", "policy")


def test_handle_uses_factory_observation_without_second_reset_and_closes_once():
    env = Env()
    initial = {"state": [0]}
    handle = EnvironmentHandle(
        env, initial, initial_state_hash(initial), "env-0",
        runtime_identity_verifier=lambda live_env, declared: live_env.observe() == declared,
    )
    outcome = run_episode(
        _spec(), environment_factory=lambda _spec: handle, policy=Policy(),
        success_fn=lambda _env, _result: True, teacher=None, vla_budget=1,
    )
    assert outcome.transitions[0].success is True
    assert outcome.transitions[0].done is True
    assert env.reset_calls == 0 and env.close_calls == 1


def test_teacher_requires_explicit_source_classifier():
    env = Env()

    class Teacher:
        def recover(self, _view, _request):
            raise AssertionError("source classifier should be checked first")

    with pytest.raises(ContractError, match="source_state_fn is mandatory"):
        run_episode(
            _spec(), environment_factory=lambda _spec: env, policy=Policy(),
            teacher=Teacher(), source_state_fn=None, vla_budget=1,
        )
    assert env.close_calls == 1


def test_string_evaluator_success_metadata_is_rejected():
    env = Env()

    class Teacher:
        def recover(self, view, _request):
            view.step([0.0] * 7)
            return TeacherRecoveryResult(
                transitions=view.executed_transitions,
                success=True,
                status=EpisodeStatus.TEACHER_SUCCESS,
                metadata={"evaluator_success": "false"},
            )

    with pytest.raises(ContractError, match="evaluator_success must be a boolean"):
        run_episode(
            _spec(), environment_factory=lambda _spec: env, policy=Policy(),
            teacher=Teacher(), source_state_fn=lambda _env, _obs: SourceState.SOURCE_UNHELD,
            success_fn=lambda _env, _result: False, vla_budget=1, teacher_budget=1,
        )


def test_handle_fails_closed_when_live_state_differs_from_claimed_state():
    env = Env()
    env.t = 999
    initial = {"state": [0]}
    handle = EnvironmentHandle(
        env, initial, initial_state_hash(initial), "env-0",
        runtime_identity_verifier=lambda live_env, declared: live_env.observe() == declared,
    )
    with pytest.raises(ContractError, match="live environment state"):
        run_episode(
            _spec(), environment_factory=lambda _spec: handle, policy=Policy(),
            success_fn=lambda _env, _result: True, teacher=None, vla_budget=1,
        )
    assert env.close_calls == 1


def test_handle_without_runtime_identity_verifier_is_rejected():
    env = Env()
    initial = {"state": [0]}
    handle = EnvironmentHandle(env, initial, initial_state_hash(initial), "env-0")
    with pytest.raises(ContractError, match="runtime_identity_verifier"):
        run_episode(
            _spec(), environment_factory=lambda _spec: handle, policy=Policy(),
            success_fn=lambda _env, _result: True, teacher=None, vla_budget=1,
        )
    assert env.close_calls == 1


def test_malformed_success_flags_are_rejected_instead_of_truthified():
    with pytest.raises(ContractError, match="success"):
        _step_result({"observation": {"state": [0]}, "done": False, "success": "false"})
    with pytest.raises(ContractError, match="success"):
        _step_result(({"state": [0]}, 0.0, False, {"success": "false"}))
