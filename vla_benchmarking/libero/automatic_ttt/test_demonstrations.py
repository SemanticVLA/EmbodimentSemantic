from __future__ import annotations

import pytest

from .contracts import Actor, ContractError, SourceState, TransitionRecord
from .demonstrations import DemonstrationValidationError, validate_and_build_demonstration


def _row(episode, timestep, actor, observation, next_observation, *, success=False, done=False, source=None, chunk_id=None):
    return TransitionRecord(
        episode_id=episode,
        timestep=timestep,
        actor=actor,
        observation=observation,
        action=(0.0,) * 7,
        next_observation=next_observation,
        done=done,
        success=success,
        training_eligible=actor is Actor.TEACHER,
        source_state=source,
        action_chunk_id=chunk_id or f"chunk-{timestep}",
        action_chunk_index=0,
        action_chunk_horizon=1,
    )


def _valid(*, teacher_success=True, evaluator_success=True):
    return validate_and_build_demonstration(
        (
            _row("ep", 0, Actor.VLA, {"state": [0]}, {"state": [1]}),
            _row("ep", 1, Actor.TEACHER, {"state": [1]}, {"state": [2]}, success=teacher_success, done=True, source=SourceState.SOURCE_UNHELD),
        ),
        task_id=0, seed=1, environment_identity="env-1",
        teacher_success=teacher_success, evaluator_success=evaluator_success,
    )


def test_valid_demonstration_returns_immutable_receipt():
    artifact = _valid()
    assert artifact.receipt.vla_transition_count == 1
    assert artifact.receipt.teacher_transition_count == 1
    assert artifact.receipt.accepted_correction_chunks[0]["executed"] == 1
    assert len(artifact.receipt.transitions_sha256) == 64


def test_first_teacher_boundary_must_match_vla_next_observation():
    with pytest.raises(DemonstrationValidationError, match="first teacher observation"):
        validate_and_build_demonstration(
            (
                _row("ep", 0, Actor.VLA, {"state": [0]}, {"state": [1]}),
                _row("ep", 1, Actor.TEACHER, {"state": [9]}, {"state": [2]}, source=SourceState.SOURCE_UNHELD),
            ),
            task_id=0, seed=1, environment_identity="env-1", teacher_success=False, evaluator_success=False,
        )


def test_successful_teacher_requires_final_success_and_evaluator():
    with pytest.raises(DemonstrationValidationError, match="final teacher transition success"):
        validate_and_build_demonstration(
            (
                _row("ep", 0, Actor.VLA, {"state": [0]}, {"state": [1]}),
                _row("ep", 1, Actor.TEACHER, {"state": [1]}, {"state": [2]}, success=False, done=True, source=SourceState.SOURCE_UNHELD),
            ),
            task_id=0, seed=1, environment_identity="env-1", teacher_success=True, evaluator_success=True,
        )
    with pytest.raises(DemonstrationValidationError, match="evaluator_success"):
        _valid(teacher_success=True, evaluator_success=False)


def test_failed_teacher_demonstration_is_valid_but_not_successful():
    artifact = _valid(teacher_success=False, evaluator_success=False)
    assert artifact.record.teacher_success is False
    assert artifact.receipt.evaluator_success is False


def test_gaps_and_privileged_observations_are_rejected():
    with pytest.raises(DemonstrationValidationError, match="contiguous"):
        validate_and_build_demonstration(
            (
                _row("ep", 0, Actor.VLA, {"state": [0]}, {"state": [1]}),
                _row("ep", 2, Actor.TEACHER, {"state": [1]}, {"state": [2]}, source=SourceState.SOURCE_UNHELD),
            ),
            task_id=0, seed=1, environment_identity="env-1", teacher_success=False, evaluator_success=False,
        )
    with pytest.raises(DemonstrationValidationError, match="privileged"):
        try:
            validate_and_build_demonstration(
                (
                    _row("ep", 0, Actor.VLA, {"state": [0], "simulator_bbox": [1]}, {"state": [1]}),
                    _row("ep", 1, Actor.TEACHER, {"state": [1]}, {"state": [2]}, source=SourceState.SOURCE_UNHELD),
                ),
                task_id=0, seed=1, environment_identity="env-1", teacher_success=False, evaluator_success=False,
            )
        except ContractError as exc:
            raise DemonstrationValidationError((str(exc),)) from exc
