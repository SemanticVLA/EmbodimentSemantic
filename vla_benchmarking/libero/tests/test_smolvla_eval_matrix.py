from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows import run_smolvla_eval_matrix as matrix


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / f"smolvla_libero-{matrix.SEALED_REVISION}"
    base.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (base / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": matrix.SEALED_REVISION, "files": {"config.json": _sha(base / "config.json")}}) + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    adapter = run / "checkpoints" / matrix.SEALED_CHECKPOINT_ID / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "train_config.json").write_text("{}\n", encoding="utf-8")
    plan = run / "training_plan.json"
    plan.write_text(
        json.dumps({
            "experiment": "smolvla_lora_no_arrow_treatment_training",
            "training_variant": "no_arrow_treatment",
            "dataset_variant": "control",
            "dataset_repo_id": "local/libero_spatial_control",
            "trained_on_visual_condition": "no_arrows",
            "base_policy": str(base.resolve()),
            "base_policy_revision": matrix.SEALED_REVISION,
            "flags": {"steps": matrix.SEALED_STEPS, "save_freq": matrix.SEALED_SAVE_FREQ, "batch_size": 32, "seed": 1000, "peft_r": 16},
        }) + "\n",
        encoding="utf-8",
    )
    pair_manifest = tmp_path / "sealed_lora_pair_manifest.json"
    pair_manifest.write_text(json.dumps({"pair_kind": "sealed_lora_control_treatment", "full_experiment_ready": True, "launch_eligibility": "full_experiment_ready"}) + "\n", encoding="utf-8")
    pair_sentinel = tmp_path / "sealed_lora_pair_verified.json"
    pair_sentinel.write_text(json.dumps({"pair_kind": "sealed_lora_control_treatment", "full_experiment_ready": True, "launch_eligibility": "full_experiment_ready"}) + "\n", encoding="utf-8")
    training = run / "training_manifest.json"
    training.write_text(
        json.dumps({
            "experiment": "smolvla_lora_no_arrow_treatment_training",
            "training_variant": "no_arrow_treatment",
            "dataset_variant": "control",
            "dataset_repo_id": "local/libero_spatial_control",
            "trained_on_visual_condition": "no_arrows",
            "base_policy": str(base.resolve()),
            "base_policy_revision": matrix.SEALED_REVISION,
            "training_plan": str(plan.resolve()),
            "training_plan_sha256": _sha(plan),
            "pair_manifest": str(pair_manifest.resolve()),
            "pair_manifest_sha256": _sha(pair_manifest),
            "pair_sentinel": str(pair_sentinel.resolve()),
            "pair_sentinel_sha256": _sha(pair_sentinel),
            "pair_kind": "sealed_lora_control_treatment",
            "resume_audits": [],
            "resume_chain_digest": hashlib.sha256(b"[]").hexdigest(),
            "final_checkpoint_id": matrix.SEALED_CHECKPOINT_ID,
            "flags": {"steps": matrix.SEALED_STEPS, "save_freq": matrix.SEALED_SAVE_FREQ, "batch_size": 32, "seed": 1000, "peft_r": 16},
            "no_arrow_treatment_adapter": {"path": str(adapter.resolve()), "sha256": _sha(adapter / "adapter_model.safetensors")},
        }) + "\n",
        encoding="utf-8",
    )
    return base, adapter, training


def test_matrix_smoke_has_fixed_cells_and_two_task_schedule(tmp_path: Path):
    base, adapter, training = _fixture(tmp_path)
    manifest = matrix.build_manifest(
        adapter_checkpoint=str(adapter), training_manifest=str(training),
        output_root=tmp_path / "outputs", protocol="smoke",
    )
    assert [cell["cell_id"] for cell in manifest["cells"]] == list(matrix.CELL_IDS)
    assert [cell["suite_mode"] for cell in manifest["cells"]] == ["vanilla", "sealed_randomized", "vanilla"]
    assert manifest["cells"][0]["checkpoint"] == str(base.resolve())
    assert manifest["cells"][2]["checkpoint"] == str(adapter.resolve())
    assert manifest["planned_episodes_total"] == 6
    assert len(manifest["schedule"]["cells"]) == 6
    assert all(plan["schedule"]["cells"][0]["task_id"] == 0 for plan in manifest["plans"])


def test_matrix_full_has_100_rows_per_cell_and_binds_ft_artifact(tmp_path: Path):
    _base, adapter, training = _fixture(tmp_path)
    manifest = matrix.build_manifest(
        adapter_checkpoint=str(adapter), training_manifest=str(training),
        output_root=tmp_path / "outputs", protocol="full",
    )
    assert manifest["planned_episodes_per_cell"] == 100
    assert manifest["planned_episodes_total"] == 300
    assert [len(plan["schedule"]["cells"]) for plan in manifest["plans"]] == [100, 100, 100]
    assert manifest["cells"][2]["adapter_sha256"] == _sha(adapter / "adapter_model.safetensors")
    assert manifest["training_contract"]["validated"] is True


def test_matrix_rejects_base_checkpoint_that_is_not_training_snapshot(tmp_path: Path):
    _base, adapter, training = _fixture(tmp_path)
    with pytest.raises(ValueError, match="exact base snapshot"):
        matrix.build_manifest(
            adapter_checkpoint=str(adapter), training_manifest=str(training),
            output_root=tmp_path / "outputs", protocol="smoke",
            base_checkpoint=str(tmp_path / "wrong-base"),
        )


def test_immutable_schedule_rejects_changed_payload(tmp_path: Path):
    _base, adapter, training = _fixture(tmp_path)
    manifest = matrix.build_manifest(
        adapter_checkpoint=str(adapter), training_manifest=str(training),
        output_root=tmp_path / "outputs", protocol="smoke",
    )
    path = tmp_path / "schedule.json"
    assert matrix.write_immutable_schedule(path, manifest["schedule"]) == manifest["schedule"]["schedule_sha256"]
    changed = dict(manifest["schedule"])
    changed["tasks"] = [0]
    with pytest.raises(ValueError, match="schedule hash"):
        matrix.write_immutable_schedule(tmp_path / "changed.json", changed)
