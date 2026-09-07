from __future__ import annotations

import json

import pytest

from .peft_artifacts import PEFTArtifactError, load_peft_manifest, save_peft_adapter


def _inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (source / "tokenizer.json").write_text('{"version":"test"}\n', encoding="utf-8")
    base = tmp_path / "base.bin"
    base.write_bytes(b"base checkpoint")
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"episodes": 1}\n', encoding="utf-8")
    return source, base, dataset


def _save(tmp_path, run_id="run-1"):
    source, base, dataset = _inputs(tmp_path)
    return save_peft_adapter(
        source,
        tmp_path / "outputs",
        vla="smolvla",
        task_id=0,
        run_id=run_id,
        base_checkpoint=base,
        dataset_manifest=dataset,
        seed=17,
        train_counts={"episodes": 10, "steps": 20000},
        eval_counts={"episodes": 50, "successes": 20},
        git_commit="a" * 40,
        runtime_versions={"python": "3.12", "torch": "2.x"},
    )


def test_artifact_layout_provenance_hashes_and_round_trip(tmp_path):
    source, _, _ = _inputs(tmp_path)
    # A stale source-side receipt is not payload and must never be copied into
    # the published tree (otherwise the receipt would recursively hash itself).
    (source / "artifact_manifest.json").write_text("stale\n", encoding="utf-8")
    manifest = _save(tmp_path)
    expected = tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1"
    assert manifest.artifact_path == str(expected.resolve())
    assert set((expected / "adapter_config.json", expected / "adapter_model.safetensors", expected / "tokenizer.json")) == {
        expected / name for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json")
    }
    restored = load_peft_manifest(expected)
    assert restored.artifact_tree_sha256 == manifest.artifact_tree_sha256
    assert "artifact_manifest.json" not in restored.files
    assert restored.base_checkpoint_sha256 == manifest.base_checkpoint_sha256
    assert restored.dataset_manifest_sha256 == manifest.dataset_manifest_sha256
    payload = json.loads((expected / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert payload["method"] == "peft_lora"


def test_existing_final_artifact_cannot_be_overwritten(tmp_path):
    _save(tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        _save(tmp_path)


def test_source_requirements_and_digest_validation(tmp_path):
    source, base, dataset = _inputs(tmp_path)
    (source / "adapter_model.safetensors").unlink()
    with pytest.raises(PEFTArtifactError, match="required"):
        save_peft_adapter(
            source, tmp_path / "outputs", vla="smolvla", task_id=0, run_id="run-1",
            base_checkpoint=base, dataset_manifest=dataset, seed=17,
            train_counts={"steps": 1}, eval_counts={"episodes": 1}, git_commit="a" * 40,
            runtime_versions={"python": "3.12"},
        )


def test_tampered_artifact_fails_manifest_validation(tmp_path):
    manifest = _save(tmp_path)
    path = (tmp_path / "outputs" / "smolvla" / "task_0" / "peft_adapter" / "run-1" / "adapter_model.safetensors")
    path.write_bytes(b"tampered")
    with pytest.raises(PEFTArtifactError, match="digest mismatch"):
        load_peft_manifest(manifest.artifact_path)
