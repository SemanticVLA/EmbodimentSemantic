from __future__ import annotations

import pytest

from .collection_cache import DurableEpisodeCache
from .contracts import ContractError


def _contract(**overrides):
    value = {
        "collection_mode": "fresh_arrow",
        "task_id": 0,
        "task_description": "pick up the bowl",
        "policy_id": "smolvla",
        "accepted_target": 1,
        "adaptation_seed_start": 3000,
        "max_attempts": 500,
        "teacher_step_budget": 1200,
        "controller_config_hash": "a" * 64,
        "provenance": {"controller": "arrow-v1"},
    }
    value.update(overrides)
    return value


def _row():
    return {
        "episode_id": "arrow-fresh-task0-seed3000",
        "task_id": 0,
        "seed": 3000,
        "source_kind": "arrow_grasp_controller_fresh_demonstration",
        "transitions": [{"actor": "arrow_grasp_controller", "action": [0.0] * 7}],
        "evaluator_receipt": {"evaluator_success": True, "teacher_success": True},
        "reset_identity": {"task_id": 0, "selected_init_state_index": 10, "init_state_sha256": "b" * 64},
    }


def test_cache_index_retains_only_scalar_ref_not_episode_payload(tmp_path):
    cache = DurableEpisodeCache(tmp_path, contract=_contract(), target=1)
    cache.begin_attempt(0)
    cache.add_success(_row())

    ref = cache.refs[0]
    assert not hasattr(ref, "transitions")
    assert ref.path == "seed-3000.json"
    assert list(cache.iter_rows())[0]["transitions"]
    index_text = (tmp_path / ".accepted_episode_cache" / "collection_index.json").read_text()
    assert "transitions" not in index_text


def test_cache_survives_restart_after_finalization_failure(tmp_path):
    cache = DurableEpisodeCache(tmp_path, contract=_contract(), target=1)
    cache.begin_attempt(0)
    cache.add_success(_row())
    # Simulate the process dying while the final dataset/manifest is written.
    restarted = DurableEpisodeCache(tmp_path, contract=_contract(), target=1)
    assert restarted.accepted_count == 1
    assert restarted.attempted_count == 1
    assert next(restarted.iter_rows())["episode_id"] == "arrow-fresh-task0-seed3000"


def test_cache_rejects_exact_contract_mismatch(tmp_path):
    cache = DurableEpisodeCache(tmp_path, contract=_contract(), target=1)
    cache.begin_attempt(0)
    cache.add_success(_row())
    with pytest.raises(ContractError, match="contract mismatch"):
        DurableEpisodeCache(tmp_path, contract=_contract(task_id=1), target=1)


def test_cache_rejects_unbound_legacy_success_file(tmp_path):
    cache_root = tmp_path / ".accepted_episode_cache"
    cache_root.mkdir()
    (cache_root / "seed-3000.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="unbound legacy"):
        DurableEpisodeCache(tmp_path, contract=_contract(), target=1)
