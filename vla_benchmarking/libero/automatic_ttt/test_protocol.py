from __future__ import annotations

import dataclasses

import pytest

from .protocol import (
    CostReceipt,
    ProtocolError,
    ResetStateEntry,
    count_accepted_arrow_trajectories,
    make_default_protocol,
    require_arrow_target,
)


def _receipt() -> CostReceipt:
    return CostReceipt("run-1", "GB200", 8, 10.0, None, "runs/run-1", 1000, 20000)


def _reset_entry(task_id: int, seed: int, query_index: int, *, episode_id: str | None = None) -> ResetStateEntry:
    return ResetStateEntry(
        task_id=task_id,
        episode_id=episode_id or f"task{task_id:02d}_seed{seed}_query{query_index:02d}",
        seed=seed,
        reset_index=1,
        simulator_state_sha256="a" * 64,
        simulator_replay_key=f"replay://task{task_id:02d}/{seed}/{query_index:02d}",
        observation_sha256="b" * 64,
        environment_fingerprint="libero-test-env-v1",
        query_index=query_index,
    )


def _full_scored_manifest() -> list[ResetStateEntry]:
    return [
        _reset_entry(task_id, seed, query_index)
        for task_id in range(10)
        for seed in (17, 29, 43)
        for query_index in range(50)
    ]


def test_frozen_counts_and_digest_are_stable():
    protocol = make_default_protocol(_receipt())
    protocol.validate()
    assert protocol.count_report()["base_pool_total"] == 100
    assert protocol.count_report()["base_pool_per_policy"] == 50
    assert protocol.count_report()["accepted_arrow_corrections_per_round"] == 100
    assert protocol.count_report()["expected_rollouts_per_arm"] == 6000
    assert protocol.digest() == protocol.digest()


def test_protocol_digest_changes_on_any_design_change():
    protocol = make_default_protocol(_receipt())
    renamed = dataclasses.replace(protocol, protocol_id="automatic-robottt-libero-v1-revision")
    assert renamed.digest() != protocol.digest()
    changed = dataclasses.replace(protocol, query_episodes_per_task=51)
    with pytest.raises(ProtocolError):
        changed.validate()
    # Digest is intentionally unavailable for invalid protocols.


def test_invalid_task_overlap_rejected():
    protocol = make_default_protocol(_receipt())
    invalid = dataclasses.replace(protocol, transfer_tasks=(0, 3, 5, 7, 9))
    with pytest.raises(ProtocolError, match="overlap"):
        invalid.validate()


def test_protocol_lock_requires_authoritative_reset_identity():
    protocol = make_default_protocol(_receipt())
    with pytest.raises(ProtocolError, match="reset manifest"):
        protocol.lock([])
    entry = ResetStateEntry(
        task_id=0,
        episode_id="task00_seed17_query00",
        seed=17,
        reset_index=1,
        simulator_state_sha256="a" * 64,
        simulator_replay_key="replay://task00/17",
        observation_sha256="b" * 64,
        environment_fingerprint="libero-test-env-v1",
    )
    with pytest.raises(ProtocolError, match="scored reset manifest requires"):
        protocol.lock([entry])
    receipt = protocol.lock([entry], locked_at_utc="2026-09-07T00:00:00Z", pilot=True)
    assert receipt.scope == "pilot"
    assert receipt.protocol_digest
    assert receipt.reset_manifest_digest


def test_only_evaluator_confirmed_trajectories_count_as_demos():
    receipts = [
        {"episode_id": "accepted", "teacher_success": True, "evaluator_success": True},
        {"episode_id": "failed", "teacher_success": False, "evaluator_success": False},
    ]
    assert count_accepted_arrow_trajectories(receipts) == 1
    complete_round = [
        {
            "episode_id": f"task{task_id:02d}_attempt{attempt_index:02d}",
            "task_id": task_id,
            "seed": 17,
            "round": 0,
            "attempt_index": attempt_index,
            "teacher_success": task_id == 0 and attempt_index == 0,
            "evaluator_success": task_id == 0 and attempt_index == 0,
        }
        for task_id in (0, 2, 4, 6, 8)
        for attempt_index in range(20)
    ]
    with pytest.raises(ProtocolError, match="target is 100"):
        require_arrow_target(complete_round, target=100)


def test_scored_lock_requires_exact_task_seed_query_grid():
    protocol = make_default_protocol(_receipt())
    receipt = protocol.lock(_full_scored_manifest(), locked_at_utc="2026-09-07T00:00:00Z")
    assert receipt.scope == "scored"


def test_length_matching_all_task_zero_manifest_is_rejected():
    protocol = make_default_protocol(_receipt())
    # Deliberately retain 1,500 unique episode IDs, while every authoritative
    # task field is zero.  A row-count/episode-ID-only lock would accept this.
    adversarial = [
        _reset_entry(
            0,
            (17, 29, 43)[row % 3],
            (row // 3) % 50,
            episode_id=f"task00_seed{(17, 29, 43)[row % 3]}_query{(row // 3) % 50:02d}_copy{row:04d}",
        )
        for row in range(1500)
    ]
    with pytest.raises(ProtocolError, match="duplicate scored reset coverage key|exact task×seed×query grid"):
        protocol.lock(adversarial)


def test_scored_lock_rejects_opaque_query_ids_without_explicit_index():
    protocol = make_default_protocol(_receipt())
    manifest = _full_scored_manifest()
    manifest[0] = _reset_entry(0, 17, 0, episode_id="opaque-episode-id")
    manifest[0] = dataclasses.replace(manifest[0], query_index=None)
    with pytest.raises(ProtocolError, match="explicit query_index"):
        protocol.lock(manifest)


def test_reset_state_digests_must_be_lowercase_hex():
    with pytest.raises(ProtocolError, match="observation digest"):
        dataclasses.replace(_reset_entry(0, 17, 0), observation_sha256="A" * 64).validate()
    with pytest.raises(ProtocolError, match="simulator state digest"):
        dataclasses.replace(_reset_entry(0, 17, 0), simulator_state_sha256="B" * 64).validate()
