from __future__ import annotations

import pytest

from .dataset import EpisodeRecord, TTTDataset, TransitionRecord


def _dataset(episode_id="task0_seed1", state_hash="hash-a"):
    row = TransitionRecord(
        episode_id=episode_id,
        timestep=0,
        observation={
            "agentview": [[[0, 0, 0]]], "wrist": [[[0, 0, 0]]],
            "state": [0.0] * 8, "instruction": "task",
        },
        robot_action=(0.0,) * 7,
        teacher_action=None,
        context_loss_mask=1.0,
        action_loss_mask=0.0,
        source="robot",
        task_id=0,
        seed=1,
    )
    return TTTDataset([EpisodeRecord(
        episode_id=episode_id, task_id=0, seed=1, outcome="teacher_failed",
        transitions=(row,), teacher_used=False, environment_identity="env",
        provenance={"init_state_hash": state_hash},
    )])


def test_unregistered_train_episode_is_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        _dataset().validate_against_split({"train_episode_ids": ["task0_seed2"], "eval_episode_ids": ["task0_seed3"]})


def test_registered_episode_binds_task_seed_and_initial_hash():
    manifest = {
        "train_episode_ids": ["task0_seed1"],
        "eval_episode_ids": ["task0_seed2"],
        "episode_identities": {"task0_seed1": {"task_id": 0, "seed": 1, "init_state_hash": "hash-a"}},
    }
    _dataset().validate_against_split(manifest)
    bad = dict(manifest)
    bad["episode_identities"] = {"task0_seed1": {"task_id": 0, "seed": 1, "init_state_hash": "hash-b"}}
    with pytest.raises(ValueError, match="initial-state hash"):
        _dataset().validate_against_split(bad)

