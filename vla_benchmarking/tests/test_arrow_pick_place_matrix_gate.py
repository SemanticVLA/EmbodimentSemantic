"""Focused offline tests for v9 matrix promotion evidence."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _record(task: int, suite: str, seed: int, *, success: bool, config_hash: str) -> dict:
    canonical = {"name": "test-controller"}
    return {
        "cell_index": seed,
        "task_id": task,
        "suite_mode": suite,
        "seed": seed,
        "status": "completed",
        "dry_run": False,
        "contract_hash": f"contract-{suite}",
        "controller_config_hash": config_hash,
        "controller_config_observed": {
            "canonical": canonical,
            "config_hash": config_hash,
        },
        "protocol": {"source_hashes": {"controller.py": "a" * 64}},
        "audit": {"evaluator_success": success},
        "diagnostics": {
            "capture_contract": {"valid": True, "rgb_shape": [256, 256, 3]},
            "phases": [{"phase": "pregrasp", "status": "reached"}],
            "grasp_search": [],
            "micro_corrections": [],
        },
    }


def test_gate_loads_dual_summary_and_enforces_per_stratum_threshold(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    config_hash = "b" * 64
    per_suite = {}
    for suite in ("vanilla", "sealed_randomized"):
        suite_root = tmp_path / suite
        suite_root.mkdir()
        records = [
            _record(task, suite, 1000 + episode, success=episode < 9, config_hash=config_hash)
            for task in (4, 6, 9)
            for episode in range(10)
        ]
        manifest = suite_root / "arrow_pick_place_matrix_manifest.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        summary = suite_root / "arrow_pick_place_matrix_summary.json"
        summary.write_text(json.dumps({"manifest_path": str(manifest)}), encoding="utf-8")
        per_suite[suite] = {"summary_path": str(summary)}
    dual = tmp_path / "arrow_pick_place_dual_matrix_summary.json"
    dual.write_text(
        json.dumps({"controller_config": {"config_hash": config_hash}, "per_suite": per_suite}),
        encoding="utf-8",
    )

    report = gate.evaluate_gate(
        dual,
        expected_task_ids=(4, 6, 9),
        expected_seeds=tuple(range(1000, 1010)),
        expected_controller_config_hash=config_hash,
    )
    assert report["passed"] is True
    assert report["cell_count"] == 60
    assert report["conservative_successes"] == 54


def test_gate_rejects_missing_diagnostics_and_config_hash_drift(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    records = [_record(4, "vanilla", 1000, success=True, config_hash="a" * 64)]
    records[0]["diagnostics"].pop("grasp_search")
    records[0]["diagnostics"]["capture_contract"] = {"valid": True}
    records[0]["diagnostics"]["phases"] = []
    artifact = tmp_path / "status.json"
    artifact.write_text(
        json.dumps({
            "controller_config": {"config_hash": "b" * 64},
            "cells": records,
        }),
        encoding="utf-8",
    )

    report = gate.evaluate_gate(
        artifact,
        expected_task_ids=(4,),
        expected_seeds=(1000,),
        expected_suites=("vanilla",),
        hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("diagnostics incomplete" in error for error in report["errors"])
    assert any("config hash" in error for error in report["errors"])


def test_factor_gate_rejects_any_stratum_regression(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    config_hash = "c" * 64
    baseline_records = [
        _record(4, "vanilla", 1000 + index, success=index < 9, config_hash=config_hash)
        for index in range(10)
    ]
    candidate_records = [
        _record(4, "vanilla", 1000 + index, success=index < 8, config_hash=config_hash)
        for index in range(10)
    ]
    # Add a second task so total successes improve while task 4 regresses.
    baseline_records += [
        _record(6, "vanilla", 1000 + index, success=index < 5, config_hash=config_hash)
        for index in range(10)
    ]
    candidate_records += [
        _record(6, "vanilla", 1000 + index, success=index < 8, config_hash=config_hash)
        for index in range(10)
    ]
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"cells": baseline_records}), encoding="utf-8")
    candidate.write_text(json.dumps({"cells": candidate_records}), encoding="utf-8")

    report = gate.evaluate_gate(
        candidate,
        baseline=baseline,
        expected_task_ids=(4, 6),
        expected_seeds=tuple(range(1000, 1010)),
        expected_suites=("vanilla",),
        hard_tasks=(),
    )
    assert report["conservative_successes"] > report["baseline"]["conservative_successes"]
    assert any("stratum drop in task_4__vanilla" in error for error in report["errors"])


def test_gate_defaults_to_closed_canary_inventory_when_expectations_are_omitted(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    artifact = tmp_path / "one-cell.jsonl"
    artifact.write_text(json.dumps(_record(4, "vanilla", 1000, success=True, config_hash="d" * 64)) + "\n", encoding="utf-8")

    report = gate.evaluate_gate(artifact)

    assert report["passed"] is False
    assert report["conservative_denominator"] == 60
    assert any("exact denominator" in error for error in report["errors"])
    assert any("missing expected cell identities" in error for error in report["errors"])


def test_gate_rejects_missing_hard_stratum_even_when_other_cells_fill_denominator(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    records = [
        _record(task, suite, seed, success=True, config_hash="e" * 64)
        for task in (4, 6)
        for suite in ("vanilla", "sealed_randomized")
        for seed in range(1000, 1010)
    ]
    artifact = tmp_path / "missing-task-9.json"
    artifact.write_text(json.dumps({"cells": records}), encoding="utf-8")

    report = gate.evaluate_gate(
        artifact,
        expected_task_ids=(4, 6, 9),
        expected_seeds=range(1000, 1010),
        expected_controller_config_hash="e" * 64,
    )

    assert report["passed"] is False
    assert "task_9__vanilla" in report["hard_task_failures"]
    assert "task_9__sealed_randomized" in report["hard_task_failures"]


def test_gate_requires_substantive_capture_or_partial_failure_evidence(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    record = _record(4, "vanilla", 1000, success=True, config_hash="f" * 64)
    record["diagnostics"] = {
        "capture_contract": {"valid": True},
        "phases": [],
        "grasp_search": [],
        "micro_corrections": [],
    }
    artifact = tmp_path / "placeholder-diagnostics.json"
    artifact.write_text(json.dumps({"cells": [record]}), encoding="utf-8")
    report = gate.evaluate_gate(
        artifact, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["passed"] is False
    assert "1000" in str(report["diagnostics_incomplete_cells"])

    record["status"] = "failed"
    record["partial_audit"] = {"capture_contract": {"valid": False, "reason": "depth unavailable"}}
    artifact.write_text(json.dumps({"cells": [record]}), encoding="utf-8")
    report = gate.evaluate_gate(
        artifact, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["diagnostics_incomplete_cells"] == []


def test_summary_prefers_authoritative_status_and_jsonl_dedupes_latest_attempt(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    old = _record(4, "vanilla", 1000, success=False, config_hash="1" * 64)
    new = _record(4, "vanilla", 1000, success=True, config_hash="1" * 64)
    new["attempts"] = [{"attempt_index": 2, "finished_unix": 2.0}]
    manifest = tmp_path / "history.jsonl"
    manifest.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n", encoding="utf-8")
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"cells": [new]}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"status_path": "status.json", "manifest_path": "history.jsonl"}), encoding="utf-8")

    report = gate.evaluate_gate(
        summary, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["cell_count"] == 1
    assert report["conservative_successes"] == 1

    report = gate.evaluate_gate(
        manifest, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["cell_count"] == 1
    assert report["conservative_successes"] == 1


def test_gate_checks_runtime_hash_and_canonical_but_allows_legacy_no_config(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    record = _record(4, "vanilla", 1000, success=True, config_hash="2" * 64)
    artifact = tmp_path / "configured.json"
    artifact.write_text(json.dumps({
        "controller_config": {
            "config_hash": "2" * 64,
            "runtime_hash": "3" * 64,
            "canonical": {"name": "expected"},
        },
        "cells": [record],
    }), encoding="utf-8")
    report = gate.evaluate_gate(
        artifact, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["passed"] is False
    assert any("runtime hash mismatch" in error for error in report["errors"])
    assert any("canonical mismatch" in error for error in report["errors"])

    record.pop("controller_config_observed")
    record.pop("controller_config_hash")
    record["audit"].pop("controller_variant", None)
    artifact.write_text(json.dumps({"cells": [record]}), encoding="utf-8")
    report = gate.evaluate_gate(
        artifact, expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=()
    )
    assert report["passed"] is True
