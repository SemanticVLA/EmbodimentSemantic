"""Regression tests preventing privileged/raw simulator fields in training."""

from __future__ import annotations

import numpy as np
import pytest

from .dataset import (
    CANONICAL_OBSERVATION_SCHEMA,
    ObservationSchemaError,
    validate_student_observation_schema,
)


def _observation():
    return {
        "agentview": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "instruction": "pick up the object",
    }


def test_canonical_projection_accepts_expected_fields():
    validate_student_observation_schema(_observation(), require_complete=True)


def test_unknown_raw_object_state_is_rejected():
    observation = _observation()
    observation["object-state"] = np.zeros((32,), dtype=np.float32)
    with pytest.raises(ObservationSchemaError, match="unknown/non-student"):
        validate_student_observation_schema(observation)


def test_wrong_state_width_is_rejected():
    observation = _observation()
    observation["state"] = np.zeros((9,), dtype=np.float32)
    with pytest.raises(ObservationSchemaError, match="exactly 8"):
        validate_student_observation_schema(observation)


def test_adapter_must_declare_schema_before_dataset_boundary():
    from .contracts import Actor, TransitionRecord as ExecutedTransition
    from .dataset import episode_from_executed_records

    executed = ExecutedTransition(
        episode_id="e",
        timestep=0,
        actor=Actor.VLA,
        observation=_observation(),
        action=(0.0,) * 7,
        next_observation=_observation(),
        done=False,
        success=False,
        training_eligible=False,
    )
    with pytest.raises(ObservationSchemaError, match="observation_schema must be supplied"):
        episode_from_executed_records([executed], task_id=0, seed=1000)
    episode_from_executed_records(
        [executed], task_id=0, seed=1000, observation_schema=CANONICAL_OBSERVATION_SCHEMA
    )

