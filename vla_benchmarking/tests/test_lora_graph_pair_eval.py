from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import run_lora_graph_pair_eval as graph_eval
from scene_graph_formats import GRAPH_TOKENIZER_CONTRACT


def _adapter_fixture(tmp_path: Path) -> Path:
    adapter = tmp_path / "checkpoint" / "pretrained_model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "tokenizer").mkdir()
    return adapter


def _training_manifest_fixture(tmp_path: Path, adapter: Path) -> Path:
    audit = tmp_path / "graph_tokenizer_audit.json"
    audit.write_text("{}", encoding="utf-8")
    pair = tmp_path / "sealed_lora_graph_pair_manifest.json"
    sentinel = tmp_path / "sealed_lora_graph_pair_verified.json"
    pair.write_text(
        json.dumps(
            {
                "pair_kind": "sealed_lora_graph_treatment_arrow_graph_treatment",
                "graph_contract_sha256": "graph-contract",
                "tokenizer_contract_sha256": hashlib.sha256(
                    json.dumps(GRAPH_TOKENIZER_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                ).hexdigest(),
                "graph_formatter_sha256": "formatter",
            }
        ),
        encoding="utf-8",
    )
    sentinel.write_text(
        json.dumps(
            {
                "pair_kind": "sealed_lora_graph_treatment_arrow_graph_treatment",
                "manifest_sha256": hashlib.sha256(pair.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "experiment": graph_eval.TRAINING_EXPERIMENT,
        "base_policy_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "training_variant": "graph_treatment",
        "dataset_variant": "graph_treatment",
        "trained_on_visual_condition": "no_arrows",
        "trained_on_text_condition": "target_natural_v1",
        "graph_treatment_adapter": {
            "path": str(adapter),
            "sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        },
        "flags": {
            "steps": 29190,
            "save_freq": 1946,
            "batch_size": 32,
            "seed": 1000,
            "peft_r": 16,
            "tokenizer_max_length": 96,
        },
        "graph_tokenizer_audit": {
            "path": str(audit),
            "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        },
        "tokenizer_contract_sha256": hashlib.sha256(
            json.dumps(GRAPH_TOKENIZER_CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
        "graph_contract_sha256": "graph-contract",
        "graph_formatter_sha256": "formatter",
        "pair_manifest": str(pair),
        "pair_manifest_sha256": hashlib.sha256(pair.read_bytes()).hexdigest(),
        "pair_sentinel": str(sentinel),
        "pair_sentinel_sha256": hashlib.sha256(sentinel.read_bytes()).hexdigest(),
        "pair_kind": "sealed_lora_graph_treatment_arrow_graph_treatment",
        "base_policy": str(adapter.parent),
    }
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_graph_cells_are_exactly_graph_context_and_standard_context_without_arrows(tmp_path, monkeypatch):
    adapter = _adapter_fixture(tmp_path)
    cells = graph_eval._build_cells(adapter, tmp_path / "eval")
    assert [cell["cell_id"] for cell in cells] == list(graph_eval.CELL_IDS)
    assert [(cell["context_mode"], cell["context_format"]) for cell in cells] == [
        ("scene_graph", "target_natural_v1"),
        ("standard", "standard"),
    ]
    captured = []

    def fake_run(command, *, cwd, env, stdout, stderr):
        del command, cwd, stdout, stderr
        captured.append(env)
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(graph_eval.subprocess, "run", fake_run)
    args = argparse.Namespace(episodes=10, videos=False, max_videos=1, device="cpu")
    for cell in cells:
        assert graph_eval._run_cell(cell, args, Path("vla_benchmarking")) == 0
    assert len(captured) == 2
    assert all(env["TRAINING_PROFILE"] == "graph_treatment" for env in captured)
    assert all(env["VISUAL_CONDITION"] == "none" and env["VISUAL_ARROWS"] == "0" for env in captured)


def test_graph_training_lineage_rejects_visual_arrow_training(monkeypatch, tmp_path):
    adapter = _adapter_fixture(tmp_path)
    manifest = _training_manifest_fixture(tmp_path, adapter)
    monkeypatch.setattr(graph_eval, "validate_serialized_graph_preprocessor", lambda *args, **kwargs: {})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["trained_on_visual_condition"] = "arrows"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="visual lineage"):
        graph_eval._training_provenance(manifest, adapter)


def test_graph_training_lineage_rejects_changed_audit_and_adapter(monkeypatch, tmp_path):
    adapter = _adapter_fixture(tmp_path)
    manifest = _training_manifest_fixture(tmp_path, adapter)
    monkeypatch.setattr(graph_eval, "validate_serialized_graph_preprocessor", lambda *args, **kwargs: {})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = Path(payload["graph_tokenizer_audit"]["path"])
    audit.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        graph_eval._training_provenance(manifest, adapter)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["graph_tokenizer_audit"]["sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
    payload["graph_treatment_adapter"]["sha256"] = "wrong"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter artifact"):
        graph_eval._training_provenance(manifest, adapter)


def test_graph_training_lineage_rejects_wrong_tokenizer_contract_digest(monkeypatch, tmp_path):
    adapter = _adapter_fixture(tmp_path)
    manifest = _training_manifest_fixture(tmp_path, adapter)
    monkeypatch.setattr(graph_eval, "validate_serialized_graph_preprocessor", lambda *args, **kwargs: {})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tokenizer_contract_sha256"] = "wrong-contract"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer contract"):
        graph_eval._training_provenance(manifest, adapter)


def test_graph_training_lineage_rejects_base_revision_or_pair_artifact_drift(monkeypatch, tmp_path):
    adapter = _adapter_fixture(tmp_path)
    manifest = _training_manifest_fixture(tmp_path, adapter)
    monkeypatch.setattr(graph_eval, "validate_serialized_graph_preprocessor", lambda *args, **kwargs: {})

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["base_policy_revision"] = "wrong-base-revision"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="base policy revision"):
        graph_eval._training_provenance(manifest, adapter)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["base_policy_revision"] = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    pair = Path(payload["pair_manifest"])
    pair.write_bytes(b"drifted pair")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pair manifest"):
        graph_eval._training_provenance(manifest, adapter)


def test_graph_training_lineage_rejects_sentinel_contract_drift(monkeypatch, tmp_path):
    adapter = _adapter_fixture(tmp_path)
    manifest = _training_manifest_fixture(tmp_path, adapter)
    monkeypatch.setattr(graph_eval, "validate_serialized_graph_preprocessor", lambda *args, **kwargs: {})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sentinel = Path(payload["pair_sentinel"])
    sentinel_data = json.loads(sentinel.read_text(encoding="utf-8"))
    sentinel_data["graph_contract_sha256"] = "wrong-graph-contract"
    sentinel.write_text(json.dumps(sentinel_data), encoding="utf-8")
    payload["pair_sentinel_sha256"] = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sentinel"):
        graph_eval._training_provenance(manifest, adapter)


def _write_reset_audit(root: Path, *, selected_row: str = "a" , include_state: bool = True) -> None:
    details = {
        "removed": ["cookies_1"],
        "projection": {"success": True},
        "protected": {"akita_black_bowl_1": True, "plate_1": True},
        "init_state": {"selected_index": 3, "selected_row_sha256": selected_row * 64},
    }
    if include_state:
        details["sim_state_sha256"] = "b" * 64
    (root / "randomization_audit.jsonl").write_text(
        json.dumps({"task_id": 0, "env_index": 0, "reset_sequence": 1, "details": details}) + "\n",
        encoding="utf-8",
    )


def test_graph_reset_pair_requires_matching_init_and_final_sim_state(monkeypatch, tmp_path):
    left = tmp_path / "graph"
    right = tmp_path / "standard"
    left.mkdir(); right.mkdir()
    _write_reset_audit(left)
    _write_reset_audit(right)
    monkeypatch.setattr(graph_eval, "validate_randomization_audit", lambda *args, **kwargs: None)
    graph_eval._paired_reset_audit(
        [{"output_dir": str(left)}, {"output_dir": str(right)}], {}
    )

    (right / "randomization_audit.jsonl").write_text(
        json.dumps({"task_id": 0, "env_index": 0, "reset_sequence": 1, "details": {
            "removed": ["cookies_1"], "projection": {"success": True},
            "protected": {"akita_black_bowl_1": True, "plate_1": True},
            "init_state": {"selected_index": 4, "selected_row_sha256": "a" * 64},
            "sim_state_sha256": "b" * 64,
        }}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identical reset/randomization identities"):
        graph_eval._paired_reset_audit(
            [{"output_dir": str(left)}, {"output_dir": str(right)}], {}
        )


def test_graph_reset_pair_rejects_missing_sim_state(monkeypatch, tmp_path):
    left = tmp_path / "graph"
    right = tmp_path / "standard"
    left.mkdir(); right.mkdir()
    _write_reset_audit(left)
    _write_reset_audit(right, include_state=False)
    monkeypatch.setattr(graph_eval, "validate_randomization_audit", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="simulator-state hash"):
        graph_eval._paired_reset_audit(
            [{"output_dir": str(left)}, {"output_dir": str(right)}], {}
        )


def _results_sentinel_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cell_root = tmp_path / "cell"
    cell_root.mkdir()
    for name, content in (("eval_info.json", "{}"), ("randomization_audit.jsonl", "reset\n"), ("prompt_audit.jsonl", "prompt\n")):
        (cell_root / name).write_text(content, encoding="utf-8")
    summary = tmp_path / graph_eval.SUMMARY_FILENAME
    summary.write_text("cell_id,success_rate\ngraph,0\n", encoding="utf-8")
    manifest_path = tmp_path / graph_eval.MANIFEST_FILENAME
    manifest = {
        "cells": [{"cell_id": "graph", "output_dir": str(cell_root)}],
        "results_sentinel": str(tmp_path / "results_verified.json"),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sentinel = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": graph_eval._sha256(manifest_path),
        "summary": str(summary.resolve()),
        "summary_sha256": graph_eval._sha256(summary),
        "cells": [{"cell_id": "graph", "artifact_tree_sha256": graph_eval._json_hash(graph_eval._tree_inventory(cell_root))}],
    }
    sentinel_path = Path(manifest["results_sentinel"])
    sentinel_path.write_text(json.dumps(sentinel), encoding="utf-8")
    return manifest_path, summary, cell_root


def test_graph_results_sentinel_rejects_eval_evidence_tampering(tmp_path):
    manifest, _summary, cell_root = _results_sentinel_fixture(tmp_path)
    graph_eval.validate_results_sentinel(manifest)
    (cell_root / "eval_info.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="cell evidence changed"):
        graph_eval.validate_results_sentinel(manifest)


def test_graph_results_sentinel_rejects_summary_tampering(tmp_path):
    manifest, summary, _cell_root = _results_sentinel_fixture(tmp_path)
    summary.write_text("cell_id,success_rate\ngraph,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary changed"):
        graph_eval.validate_results_sentinel(manifest)
