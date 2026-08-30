from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import legacy_action_only_evidence as evidence


REVISION = evidence.SEALED_REVISION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    base = tmp_path / f"smolvla_libero-{REVISION}"
    base.mkdir()
    (base / "config.json").write_text("{\"model\":\"pinned\"}\n", encoding="utf-8")
    (base / "base_snapshot_manifest.json").write_text(
        json.dumps({"revision": REVISION, "files": {"config.json": _sha(base / "config.json")}}) + "\n",
        encoding="utf-8",
    )
    (base / ".cache" / "huggingface").mkdir(parents=True)
    (base / ".cache" / "huggingface" / "hub.lock").write_text("runtime metadata\n", encoding="utf-8")
    checkpoint = tmp_path / "run" / "checkpoints" / evidence.FINAL_CHECKPOINT_ID / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"historical adapter")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 16, "target_modules": "historical"}), encoding="utf-8"
    )

    data_root = tmp_path / "data"
    data_root.mkdir()
    pair_manifest = data_root / evidence.PAIR_MANIFEST_NAME
    pair_manifest.write_text(json.dumps({"pair_kind": "sealed_lora_control_treatment"}) + "\n", encoding="utf-8")
    pair_sentinel = data_root / evidence.PAIR_SENTINEL_NAME
    pair_sentinel.write_text(json.dumps({"pair_kind": "sealed_lora_control_treatment"}) + "\n", encoding="utf-8")

    plan = tmp_path / "run" / "training_plan.json"
    plan.write_text(
        json.dumps(
            {
                "training_variant": "no_arrow_treatment",
                "base_policy_revision": REVISION,
                "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "run" / "training_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "experiment": "smolvla_lora_no_arrow_treatment_training",
                "training_variant": "no_arrow_treatment",
                "dataset_variant": "control",
                "dataset_repo_id": "local/libero_spatial_control",
                "trained_on_visual_condition": "no_arrows",
                "base_policy": str(base),
                "base_policy_revision": REVISION,
                "training_plan": str(plan),
                "training_plan_sha256": _sha(plan),
                "pair_manifest": str(pair_manifest),
                "pair_manifest_sha256": _sha(pair_manifest),
                "pair_sentinel": str(pair_sentinel),
                "pair_sentinel_sha256": _sha(pair_sentinel),
                "final_checkpoint_id": evidence.FINAL_CHECKPOINT_ID,
                "flags": {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16},
                "no_arrow_treatment_adapter": {
                    "path": str(checkpoint),
                    "sha256": _sha(checkpoint / "adapter_model.safetensors"),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    expected = {"inventory_sha256": "expected-inventory", "policy_id": evidence.ACTION_ONLY_LORA_V1}
    monkeypatch.setattr(evidence, "build_expected_inventory", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        evidence,
        "audit_adapter_checkpoint",
        lambda checkpoint_path, **kwargs: {
            "schema_version": 1,
            "checkpoint_tree_sha256": evidence._audit_tree_digest(Path(checkpoint_path)),
            "adapter_sha256": _sha(Path(checkpoint_path) / "adapter_model.safetensors"),
        },
    )
    monkeypatch.setattr(
        evidence,
        "validate_verified_pair",
        lambda root, require_full_experiment=True: {
            "pair_kind": "sealed_lora_control_treatment",
            "dataset_fingerprints": {"control": {"sha256": "control"}, "treatment": {"sha256": "treatment"}},
            "verified_at_utc": "volatile",
            "manifest_path": "/old/location/sealed_lora_pair_manifest.json",
            "raw_hdf5_sha256": "volatile",
        },
    )
    return {
        "base": base,
        "checkpoint": checkpoint,
        "data": data_root,
        "manifest": manifest,
        "plan": plan,
    }


def test_build_and_validate_external_action_only_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    built = evidence.build_legacy_action_only_evidence(
        paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output
    )
    assert built == output
    assert (output / "expected_adapter_inventory.json").is_file()
    assert (output / "adapter_audit.json").is_file()
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["evidence_kind"] == "legacy_action_only_evidence_v1"
    assert bundle["policy_id"] == "action_only_lora_v1"
    assert bundle["base_snapshot_identity_sha256"]
    assert bundle["pair"]["pair_identity"]
    assert bundle["pair"]["historical_sentinel"]["status"] == "unavailable_at_recorded_digest"
    assert bundle["pair"]["historical_sentinel"]["recorded_sha256"] == _sha(paths["data"] / evidence.PAIR_SENTINEL_NAME)
    assert bundle["pair"]["historical_sentinel"]["observed_current_sha256"] == _sha(paths["data"] / evidence.PAIR_SENTINEL_NAME)
    assert bundle["contemporaneous_sidecars"]["run_provenance.json"]["status"] == "unavailable"
    result = evidence.validate_legacy_action_only_evidence(
        output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"]
    )
    assert result["valid"] is True


def test_bundle_is_no_overwrite_and_rejects_bundle_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    original = (output / "bundle.json").read_bytes()
    with pytest.raises(FileExistsError):
        evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    assert (output / "bundle.json").read_bytes() == original
    (output / "bundle.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree seal|file changed"):
        evidence.validate_legacy_action_only_evidence(output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"])


def test_relocated_sources_use_sibling_training_plan_and_stable_pair_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "original-evidence"
    evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    original_identity = json.loads((output / "bundle.json").read_text(encoding="utf-8"))["pair"]["pair_identity"]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    manifest = relocated / "training_manifest.json"
    shutil.copyfile(paths["manifest"], manifest)
    # The manifest still names the old absolute plan.  The sibling fallback
    # must authenticate the relocated plan by its recorded hash.
    shutil.copyfile(paths["plan"], relocated / "training_plan.json")
    base = relocated / paths["base"].name
    shutil.copytree(paths["base"], base)
    checkpoint = relocated / "run" / "checkpoints" / evidence.FINAL_CHECKPOINT_ID / "pretrained_model"
    checkpoint.parent.mkdir(parents=True)
    shutil.copytree(paths["checkpoint"], checkpoint)
    data = relocated / "data"
    shutil.copytree(paths["data"], data)
    relocated_output = tmp_path / "relocated-evidence"
    evidence.build_legacy_action_only_evidence(manifest, checkpoint, base, data, relocated_output)
    relocated_identity = json.loads((relocated_output / "bundle.json").read_text(encoding="utf-8"))["pair"]["pair_identity"]
    assert relocated_identity == original_identity
    result = evidence.validate_legacy_action_only_evidence(relocated_output, manifest, checkpoint, base, data)
    assert result["valid"] is True


def test_source_and_plan_drift_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    (paths["checkpoint"] / "adapter_config.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint tree"):
        evidence.validate_legacy_action_only_evidence(output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"])


def test_declared_base_file_drift_is_rejected_but_cache_metadata_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    (paths["base"] / "config.json").write_text("changed declared base\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base snapshot hash mismatch"):
        evidence.validate_legacy_action_only_evidence(output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"])


def test_public_base_snapshot_identity_ignores_benign_cache_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    identity = evidence.validate_base_snapshot_identity(paths["base"])
    (paths["base"] / ".cache" / "huggingface" / "new.lock").write_text("another runtime file\n", encoding="utf-8")
    assert evidence.validate_base_snapshot_identity(paths["base"]) == identity


def test_reverified_sentinel_drift_is_accepted_but_pair_manifest_drift_is_hard_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original_sentinel_sha = _sha(paths["data"] / evidence.PAIR_SENTINEL_NAME)
    (paths["data"] / evidence.PAIR_SENTINEL_NAME).write_text("reverified sentinel\n", encoding="utf-8")
    output = tmp_path / "reverified-evidence"
    evidence.build_legacy_action_only_evidence(paths["manifest"], paths["checkpoint"], paths["base"], paths["data"], output)
    pair = json.loads((output / "bundle.json").read_text(encoding="utf-8"))["pair"]
    assert pair["historical_sentinel"]["status"] == "unavailable_at_recorded_digest"
    assert pair["historical_sentinel"]["recorded_sha256"] == original_sentinel_sha
    assert pair["historical_sentinel"]["observed_current_sha256"] == _sha(paths["data"] / evidence.PAIR_SENTINEL_NAME)
    assert evidence.validate_legacy_action_only_evidence(output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"])["valid"] is True

    (paths["data"] / evidence.PAIR_MANIFEST_NAME).write_text("changed pair manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pair manifest differs"):
        evidence.validate_legacy_action_only_evidence(output, paths["manifest"], paths["checkpoint"], paths["base"], paths["data"])
