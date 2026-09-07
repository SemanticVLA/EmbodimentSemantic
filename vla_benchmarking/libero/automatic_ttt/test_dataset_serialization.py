"""Focused regression tests for lossless dataset persistence."""

from __future__ import annotations

import numpy as np
import pytest

from .dataset import EpisodeRecord, TTTDataset, TransitionRecord


def _episode(observation: object, episode_id: str = "episode") -> EpisodeRecord:
    row = TransitionRecord(
        episode_id=episode_id,
        timestep=0,
        observation={"agentview": observation},
        robot_action=(0.0,) * 7,
        teacher_action=None,
        context_loss_mask=1.0,
        action_loss_mask=0.0,
        source="robot",
        task_id=0,
        seed=1000,
    )
    return EpisodeRecord(
        episode_id=episode_id,
        task_id=0,
        seed=1000,
        outcome="failed",
        transitions=(row,),
        teacher_used=False,
        environment_identity="test-env",
    )


def test_ndarray_round_trip_preserves_dtype_shape_and_values(tmp_path):
    observation = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    dataset = TTTDataset([_episode(observation)])
    path = tmp_path / "dataset.jsonl"
    dataset.write_jsonl(path)
    restored = TTTDataset.read_jsonl(path)
    actual = restored.episodes[0].transitions[0].observation["agentview"]
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == observation.dtype
    assert actual.shape == observation.shape
    np.testing.assert_array_equal(actual, observation)
    assert restored.episodes[0].content_hash() == dataset.episodes[0].content_hash()


def test_distinct_array_content_has_distinct_episode_hash():
    first = _episode(np.zeros((4, 4, 3), dtype=np.uint8), "first")
    changed = np.zeros((4, 4, 3), dtype=np.uint8)
    changed[-1, -1, -1] = 1
    second = _episode(changed, "second")
    assert first.content_hash() != second.content_hash()


def test_non_finite_observation_is_rejected(tmp_path):
    dataset = TTTDataset([_episode(np.full((2, 3, 3), float("nan"), dtype=np.float32))])
    with pytest.raises(ValueError, match="non-finite"):
        dataset.write_jsonl(tmp_path / "invalid.jsonl")


def test_final_dataset_artifact_is_not_overwritten_by_default(tmp_path):
    dataset = TTTDataset([_episode(np.zeros((2, 3, 3), dtype=np.uint8))])
    path = tmp_path / "final.jsonl"
    dataset.write_jsonl(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        dataset.write_jsonl(path)
    assert path.read_bytes() == original
    dataset.write_jsonl(path, overwrite=True)
