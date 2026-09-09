from __future__ import annotations

import json
import hashlib

import pytest

from arrow_policy_suite.contracts import ContractError, digest
from arrow_policy_suite.learning import load_persisted_training_view
from arrow_policy_suite.native_learned_training import load_native_residual, train_editor, train_minimal_learned


def _row(*, source: str, minimal: bool = False) -> dict:
    observation = {"state": [0.0] * 8}
    action = [0.2] + [0.0] * 6
    result = {
        "task_id": 3,
        "reset_id": "reset-1",
        "episode_id": "episode-1",
        "timestep": 0,
        "observation": observation,
        "observation_digest": digest(observation),
        "base_action": [0.0] * 7,
        "teacher_action": action,
        "executed_action": action,
        "decision": {"action": action, "teacher_used": source == "on_call"},
        "source": source,
        "outcome": {"success": False, "terminal": True},
    }
    if minimal:
        result["minimal_branch"] = {
            "selected_hybrid_action": [0.1] + [0.0] * 6,
            "selected_mask": 1,
            "label_source": "minimal_branch_runtime",
        }
    return result


def test_persisted_editor_view_is_hashed_and_fixed(tmp_path):
    path = tmp_path / "master.jsonl"
    path.write_text(json.dumps({"schema": "arrow_policy_suite.training_transitions.v1", "rows": [_row(source="on_call")]}) + "\n")
    view = load_persisted_training_view(path, variant="arrow_editor", task_ids=[3])
    assert view.rows[0].target_kind == "teacher_residual"
    assert view.rows[0].executed_action == view.rows[0].teacher_action
    assert view.manifest.source_sha256 == view.source_sha256


def test_training_source_jsonl_contract_is_consumed(tmp_path):
    observation = {"state": [0.0] * 8}
    row = {
        "schema": "arrow_policy_suite.training_source.v1",
        "task_id": 3,
        "reset_id": "reset-1",
        "episode_id": "episode-1",
        "timestep": 0,
        "observation": observation,
        "observation_digest": digest(observation),
        "base_proposal": {"action": [0.0] * 7},
        "teacher_proposal": {"action": [0.2] + [0.0] * 6},
        "decision": {"action": [0.2] + [0.0] * 6, "policy_id": "arrow_on_call", "teacher_used": True},
        "executed_action": [0.2] + [0.0] * 6,
        "outcome": {"success": False, "terminal": True},
        "eligible": True,
        "hashes": {"observation_sha256": digest(observation)},
    }
    row["transition_sha256"] = hashlib.sha256(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    path = tmp_path / "training-source.jsonl"
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    view = load_persisted_training_view(path, variant="arrow_editor")
    assert view.rows[0].metadata["source_policy_family"] == "arrow_on_call"


def test_minimal_loader_requires_real_branch_labels(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({"schema": "arrow_policy_suite.minimal_branch_labels.v1", "rows": [_row(source="minimal_runtime")]}) + "\n")
    with pytest.raises(ContractError, match="selected_hybrid_action|branch"):
        load_persisted_training_view(path, variant="arrow_minimal_learned")


def test_minimal_training_uses_persisted_target_and_mask_and_reload_is_immutable(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({"schema": "arrow_policy_suite.minimal_branch_labels.v1", "rows": [_row(source="minimal_runtime", minimal=True)]}) + "\n")
    view = load_persisted_training_view(path, variant="arrow_minimal_learned")
    checkpoint = tmp_path / "minimal.pt"
    train_minimal_learned(view.rows, output_path=checkpoint, base_vla_sha256="a" * 64, manifest=view.manifest)
    model = load_native_residual(checkpoint, base_vla_sha256="a" * 64, variant="arrow_minimal_learned")
    residual, corrected = model.predict_correction([0.0] * 32, [0.0] * 7)
    assert len(residual) == len(corrected) == 7
    with pytest.raises(ContractError, match="immutable"):
        train_minimal_learned(view.rows, output_path=checkpoint, base_vla_sha256="a" * 64, manifest=view.manifest)


def test_editor_checkpoint_rejects_wrong_variant(tmp_path):
    pytest.importorskip("torch")
    from arrow_policy_suite.learning import InterventionRow, manifest_for_rows

    rows = (InterventionRow("episode-1", 3, 0, {"state": [0.0] * 8}, (0.0,) * 7, (0.2,) + (0.0,) * 6, False),)
    manifest = manifest_for_rows(rows, parent_artifact="archive")
    checkpoint = tmp_path / "editor.pt"
    train_editor(rows, output_path=checkpoint, base_vla_sha256="b" * 64, manifest=manifest)
    with pytest.raises(ContractError, match="variant"):
        load_native_residual(checkpoint, base_vla_sha256="b" * 64, variant="arrow_minimal_learned")
