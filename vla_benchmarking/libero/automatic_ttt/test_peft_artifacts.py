from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from .peft_artifacts import (
    COLLECTION_SOURCE_KIND,
    FRESH_COLLECTION_SOURCE_KIND,
    PEFTArtifactError,
    load_arrow_collection_manifest,
    load_peft_manifest,
    save_peft_adapter,
)


def _inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (source / "tokenizer.json").write_text('{"version":"test"}\n', encoding="utf-8")
    base = tmp_path / "base.bin"
    base.write_bytes(b"base checkpoint")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True, exist_ok=True)
    (dataset / "meta" / "info.json").write_text('{"total_frames": 10}\n', encoding="utf-8")
    accepted = tmp_path / "accepted_episodes.jsonl"
    accepted_rows = []
    for i in range(50):
        episode_id = f"t-{i}"
        accepted_rows.append({
            "episode_id": episode_id,
            "task_id": 0,
            "seed": 3000 + i,
            "source_kind": COLLECTION_SOURCE_KIND,
            "transitions": [{"actor": "vla"}, {"actor": "arrow_grasp_controller"}],
            "evaluator_receipt": {
                "episode_id": episode_id,
                "seed": 3000 + i,
                "source_controller": "arrow_grasp_controller",
                "teacher_success": True,
                "evaluator_success": True,
            },
        })
    accepted.write_text("".join(json.dumps(row) + "\n" for row in accepted_rows), encoding="utf-8")
    dataset_manifest = tmp_path / "dataset_manifest.json"
    info = dataset / "meta" / "info.json"
    dataset_manifest.write_text(json.dumps({
        "schema_version": 1,
        "dataset_root": str(dataset.resolve()),
        "source_accepted_episodes": str(accepted.resolve()),
        "source_accepted_episodes_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
        "files": [{
            "path": "meta/info.json",
            "sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
            "bytes": info.stat().st_size,
        }],
    }) + "\n", encoding="utf-8")
    collection = tmp_path / "collection.json"
    payload = {
        "schema_version": 1,
        "source_kind": COLLECTION_SOURCE_KIND,
        "method_label": "peft_lora",
        "task_id": 0,
        "task_ids": [0],
        "evaluator_confirmed_successes": 50,
        "successful_trajectories": [
            {"trajectory_id": f"t-{i}", "seed": 3000 + i, "evaluator_success": True} for i in range(50)
        ],
        "adaptation_seeds": list(range(3000, 3050)),
        "controller_config_hash": "a" * 64,
        "accepted_episodes_jsonl": str(accepted),
        "accepted_episodes_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
        "dataset_root": str(dataset),
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
    }
    collection.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return source, base, collection


def _runtime_evidence():
    return {
        "schema_version": 1, "updates_observed": 20000,
        "all_losses_finite": True, "all_grad_norms_finite": True,
        "min_loss": 0.1, "max_loss": 1.0, "last_loss": 0.2,
        "last_grad_norm": 1.0, "min_learning_rate": 2.5e-6,
        "max_learning_rate": 5e-5, "last_learning_rate": 2.5e-6,
        "optimizer": {"optimizer": "AdamW", "weight_decay": 1e-5,
                       "peak_learning_rate": 5e-5, "betas": [0.9, 0.95],
                       "epsilon": 1e-8, "gradient_clip_norm": 10.0},
        "scheduler": {"scheduler": "cosine_decay_with_warmup", "warmup_steps": 666,
                      "decay_steps": 20000, "decay_lr": 2.5e-6},
    }


def _fresh_inputs(tmp_path):
    source, base, collection = _inputs(tmp_path)
    accepted = tmp_path / "accepted_episodes.jsonl"
    rows = [json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines()]
    accepted_reset_identities = []
    for i, row in enumerate(rows):
        reset_identity = {
            "task_id": 0,
            "selected_init_state_index": 10 + i,
            "init_state_sha256": f"{100 + i:064x}",
        }
        row["source_kind"] = FRESH_COLLECTION_SOURCE_KIND
        row["method_label"] = "fresh_arrow_behavior_cloning_peft"
        row["collection_mode"] = "fresh_arrow"
        row["transitions"] = [{"actor": "arrow_grasp_controller"}]
        row["reset_identity"] = reset_identity
        accepted_reset_identities.append(reset_identity)
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    info = tmp_path / "dataset" / "meta" / "info.json"
    info.write_text('{"total_frames": 17}\n', encoding="utf-8")
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset_payload["source_accepted_episodes_sha256"] = hashlib.sha256(accepted.read_bytes()).hexdigest()
    dataset_payload["files"][0]["sha256"] = hashlib.sha256(info.read_bytes()).hexdigest()
    dataset_payload["files"][0]["bytes"] = info.stat().st_size
    dataset_manifest.write_text(json.dumps(dataset_payload) + "\n", encoding="utf-8")

    payload = json.loads(collection.read_text(encoding="utf-8"))
    payload.update({
        "schema": "automatic_ttt.arrow_fresh_demonstration_collection.v1",
        "source_kind": FRESH_COLLECTION_SOURCE_KIND,
        "method_label": "fresh_arrow_behavior_cloning_peft",
        "collection_mode": "fresh_arrow",
        "starts_from_reset": True,
        "vla_called": False,
        "reserved_eval_init_state_indices": list(range(10)),
        "reserved_eval_init_state_hashes": [f"{i + 1:064x}" for i in range(10)],
        "accepted_reset_identities": accepted_reset_identities,
        "accepted_episodes_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
    })
    for item, identity in zip(payload["successful_trajectories"], accepted_reset_identities):
        item["reset_identity"] = identity
    collection.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return source, base, collection


def _fresh_runtime_evidence(*, steps=11, warmup=0):
    evidence = _runtime_evidence()
    evidence["updates_observed"] = steps
    evidence["expected_updates"] = steps
    evidence["scheduler"]["warmup_steps"] = warmup
    evidence["scheduler"]["decay_steps"] = steps
    return evidence


def _fresh_save(tmp_path, run_id="fresh-run"):
    source, base, collection = _fresh_inputs(tmp_path)
    steps, frames, batch = 11, 17, 8
    optimizer = {
        "optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 5e-5,
        "betas": [0.9, 0.95], "epsilon": 1e-8, "gradient_clip_norm": 10.0,
        "scheduler": "cosine_decay_with_warmup", "warmup_steps": 0,
        "decay_steps": steps, "decay_lr": 2.5e-6,
    }
    return save_peft_adapter(
        source, tmp_path / "fresh_outputs", vla="smolvla", task_id=0, run_id=run_id,
        base_checkpoint=base, base_checkpoint_revision="base-rev",
        collection_manifest=collection, collection_success_count=50,
        controller_config_hash="a" * 64, checkpoint_step=steps, seed=1000,
        train_counts={
            "successful_trajectories": 50, "steps": steps, "requested_epochs": 5,
            "batch_size": batch, "global_batch_size": batch, "dataset_frames": frames,
            "epoch_equivalent": steps * batch / frames, "save_freq": steps,
            "seed": 1000, "training_scope": "task_specific",
        },
        eval_counts={
            "baseline": {"episodes": 10, "seeds": list(range(1000, 1010))},
            "adapted": {"episodes": 10, "seeds": list(range(1000, 1010))},
            "seeds": list(range(1000, 1010)),
        },
        optimizer=optimizer, runtime_evidence=_fresh_runtime_evidence(),
        git_commit="a" * 40, runtime_versions={"python": "3.12"},
    )


def _save(tmp_path, run_id="run-1"):
    source, base, collection = _inputs(tmp_path)
    return save_peft_adapter(
        source, tmp_path / "outputs", vla="smolvla", task_id=0, run_id=run_id,
        base_checkpoint=base, base_checkpoint_revision="base-rev",
        collection_manifest=collection, collection_success_count=50,
        controller_config_hash="a" * 64, checkpoint_step=20000, seed=1000,
        train_counts={"successful_trajectories": 50, "steps": 20000,
                      "batch_size": 8, "global_batch_size": 8,
                      "dataset_frames": 10, "epoch_equivalent": 16000.0,
                      "save_freq": 2000, "seed": 1000,
                      "training_scope": "task_specific"},
        eval_counts={
            "baseline": {"episodes": 10, "successes": 2, "seeds": list(range(1000, 1010))},
            "adapted": {"episodes": 10, "successes": 2, "seeds": list(range(1000, 1010))},
            "seeds": list(range(1000, 1010)),
        },
        optimizer={"optimizer": "AdamW", "weight_decay": 1e-5,
                   "peak_learning_rate": 5e-5, "betas": [0.9, 0.95],
                   "epsilon": 1e-8, "gradient_clip_norm": 10.0,
                   "scheduler": "cosine_decay_with_warmup", "warmup_steps": 666,
                   "decay_steps": 20000, "decay_lr": 2.5e-6},
        runtime_evidence=_runtime_evidence(),
        git_commit="a" * 40, runtime_versions={"python": "3.12", "torch": "2.x"},
    )


def test_collection_contract_requires_one_task_50_confirmed_and_disjoint_seeds(tmp_path):
    _, _, collection = _inputs(tmp_path)
    normalized = load_arrow_collection_manifest(collection, task_id=0)
    assert normalized["source_kind"] == COLLECTION_SOURCE_KIND
    assert normalized["task_ids"] == [0]
    assert normalized["evaluator_confirmed_successes"] == 50
    assert normalized["adaptation_seeds"] == list(range(3000, 3050))

    payload = json.loads(collection.read_text(encoding="utf-8"))
    payload["task_ids"] = [0, 1]
    collection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="exactly one"):
        load_arrow_collection_manifest(collection, task_id=0)


def test_collection_contract_rejects_eval_seed_overlap_and_unconfirmed_rows(tmp_path):
    _, _, collection = _inputs(tmp_path)
    payload = json.loads(collection.read_text(encoding="utf-8"))
    payload["adaptation_seeds"][0] = 1000
    collection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="overlap"):
        load_arrow_collection_manifest(collection, task_id=0)
    payload["adaptation_seeds"][0] = 3000
    payload["successful_trajectories"][0]["evaluator_success"] = False
    collection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="evaluator-confirmed"):
        load_arrow_collection_manifest(collection, task_id=0)


def test_collection_contract_binds_dataset_to_exact_accepted_trace(tmp_path):
    _, _, collection = _inputs(tmp_path)
    payload = json.loads(collection.read_text(encoding="utf-8"))
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset_payload["source_accepted_episodes_sha256"] = "b" * 64
    dataset_manifest.write_text(json.dumps(dataset_payload) + "\n", encoding="utf-8")
    payload["dataset_manifest_sha256"] = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    collection.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="source_accepted_episodes_sha256"):
        load_arrow_collection_manifest(collection, task_id=0)


def test_collection_contract_rejects_non_vla_prefix_or_arrow_suffix(tmp_path):
    _, _, collection = _inputs(tmp_path)
    payload = json.loads(collection.read_text(encoding="utf-8"))
    trace = tmp_path / "accepted_episodes.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["transitions"] = [{"actor": "arrow_grasp_controller"}, {"actor": "vla"}]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload["accepted_episodes_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset_payload["source_accepted_episodes_sha256"] = payload["accepted_episodes_sha256"]
    dataset_manifest.write_text(json.dumps(dataset_payload) + "\n", encoding="utf-8")
    payload["dataset_manifest_sha256"] = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    collection.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="VLA prefix"):
        load_arrow_collection_manifest(collection, task_id=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("betas", [0.9, 0.9], "betas"),
        ("epsilon", 1e-7, "epsilon"),
        ("gradient_clip_norm", 1.0, "gradient_clip_norm"),
        ("warmup_steps", 0, "warmup_steps"),
        ("decay_lr", 0.0, "decay_lr"),
    ],
)
def test_artifact_requires_full_paper_optimizer_mapping(tmp_path, field, value, message):
    source, base, collection = _inputs(tmp_path)
    optimizer = {
        "optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 5e-5,
        "betas": [0.9, 0.95], "epsilon": 1e-8, "gradient_clip_norm": 10.0,
        "scheduler": "cosine_decay_with_warmup", "warmup_steps": 666,
        "decay_steps": 20000, "decay_lr": 2.5e-6,
    }
    optimizer[field] = value
    with pytest.raises(PEFTArtifactError, match=message):
        save_peft_adapter(
            source, tmp_path / "outputs", vla="smolvla", task_id=0, run_id="run-optimizer",
            base_checkpoint=base, base_checkpoint_revision="base-rev", collection_manifest=collection,
            collection_success_count=50, controller_config_hash="a" * 64, checkpoint_step=20000,
            seed=1000,
            train_counts={"successful_trajectories": 50, "steps": 20000, "batch_size": 8,
                          "global_batch_size": 8, "dataset_frames": 10, "epoch_equivalent": 16000.0,
                          "save_freq": 2000, "seed": 1000},
            eval_counts={"baseline": {"episodes": 10}, "adapted": {"episodes": 10},
                         "seeds": list(range(1000, 1010))},
            optimizer=optimizer, runtime_evidence=_runtime_evidence(),
            git_commit="a" * 40, runtime_versions={"python": "3.12"},
        )


def test_artifact_layout_provenance_hashes_and_round_trip(tmp_path):
    source, _, _ = _inputs(tmp_path)
    (source / "artifact_manifest.json").write_text("stale\n", encoding="utf-8")
    manifest = _save(tmp_path)
    expected = tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1"
    assert manifest.artifact_path == str(expected.resolve())
    assert {expected / name for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json")} <= {
        path for path in expected.iterdir() if path.name != "artifact_manifest.json"
    }
    assert (expected / "lineage").is_dir()
    restored = load_peft_manifest(expected)
    assert restored.schema_version == 2
    assert restored.training_scope == "task_specific"
    assert restored.trained_task_ids == [0]
    assert restored.collection_success_count == 50
    assert restored.checkpoint_step == 20000
    assert restored.base_checkpoint_path == "lineage/base_checkpoint.json"
    assert restored.collection_manifest_path == "lineage/collection_manifest.json"
    assert restored.dataset_manifest_path == "lineage/dataset_manifest.json"
    assert restored.runtime_evidence_path == "lineage/runtime_evidence.json"
    assert restored.artifact_tree_sha256 == manifest.artifact_tree_sha256
    assert "artifact_manifest.json" not in restored.files
    payload = json.loads((expected / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert payload["source_kind"] == COLLECTION_SOURCE_KIND


def test_existing_final_artifact_cannot_be_overwritten(tmp_path):
    _save(tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        _save(tmp_path)


def test_source_requirements_and_digest_validation(tmp_path):
    source, base, collection = _inputs(tmp_path)
    (source / "adapter_model.safetensors").unlink()
    with pytest.raises(PEFTArtifactError, match="required"):
        save_peft_adapter(
            source, tmp_path / "outputs", vla="smolvla", task_id=0, run_id="run-1",
            base_checkpoint=base, base_checkpoint_revision="base-rev", collection_manifest=collection,
            collection_success_count=50, controller_config_hash="a" * 64, checkpoint_step=20000,
            seed=1000, train_counts={"steps": 20000}, eval_counts={"episodes": 10},
            optimizer={"optimizer": "AdamW", "weight_decay": 1e-5, "peak_learning_rate": 5e-5},
            runtime_evidence=_runtime_evidence(),
            git_commit="a" * 40, runtime_versions={"python": "3.12"},
        )


def test_tampered_artifact_fails_manifest_validation(tmp_path):
    manifest = _save(tmp_path)
    path = tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1" / "adapter_model.safetensors"
    path.write_bytes(b"tampered")
    with pytest.raises(PEFTArtifactError, match="digest mismatch"):
        load_peft_manifest(manifest.artifact_path)


def test_bundled_lineage_remains_loadable_after_scratch_deletion(tmp_path):
    manifest = _save(tmp_path)
    for scratch_name in ("accepted_episodes.jsonl", "collection.json", "dataset_manifest.json"):
        scratch = tmp_path / scratch_name
        if scratch.exists():
            scratch.unlink()
    dataset = tmp_path / "dataset"
    for child in sorted(dataset.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
    restored = load_peft_manifest(manifest.artifact_path)
    assert restored.runtime_evidence_path == "lineage/runtime_evidence.json"
    assert restored.collection_manifest_path == "lineage/collection_manifest.json"


def test_tampered_bundled_runtime_evidence_is_rejected(tmp_path):
    manifest = _save(tmp_path)
    runtime = tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1" / "lineage" / "runtime_evidence.json"
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload["updates_observed"] = 19999
    runtime.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="digest mismatch"):
        load_peft_manifest(manifest.artifact_path)


def test_unlisted_bundled_lineage_file_is_rejected(tmp_path):
    manifest = _save(tmp_path)
    lineage = tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1" / "lineage"
    (lineage / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="inventory"):
        load_peft_manifest(manifest.artifact_path)


def test_fresh_artifact_enforces_dynamic_five_epoch_budget(tmp_path):
    manifest = _fresh_save(tmp_path)
    restored = load_peft_manifest(manifest.artifact_path)
    assert restored.source_kind == FRESH_COLLECTION_SOURCE_KIND
    assert restored.checkpoint_step == 11
    assert restored.train_counts["requested_epochs"] == 5
    assert restored.train_counts["epoch_equivalent"] == pytest.approx(88 / 17)
    assert restored.optimizer["warmup_steps"] == 0
    assert restored.optimizer["decay_steps"] == 11


def test_fresh_artifact_relocates_after_all_original_inputs_are_deleted(tmp_path):
    manifest = _fresh_save(tmp_path)
    original = Path(manifest.artifact_path)
    relocated = tmp_path / "downloaded_elsewhere" / "adapter"
    shutil.copytree(original, relocated)

    shutil.rmtree(tmp_path / "fresh_outputs")
    shutil.rmtree(tmp_path / "source")
    shutil.rmtree(tmp_path / "dataset")
    for name in ("base.bin", "accepted_episodes.jsonl", "dataset_manifest.json", "collection.json"):
        path = tmp_path / name
        if path.exists():
            path.unlink()

    restored = load_peft_manifest(relocated)
    assert restored.artifact_path == "."
    assert (relocated / "lineage" / "dataset_tree" / "meta" / "info.json").is_file()

    dataset_file = relocated / "lineage" / "dataset_tree" / "meta" / "info.json"
    dataset_file.write_text('{"total_frames": 999}\n', encoding="utf-8")
    with pytest.raises(PEFTArtifactError, match="digest mismatch"):
        load_peft_manifest(relocated)
