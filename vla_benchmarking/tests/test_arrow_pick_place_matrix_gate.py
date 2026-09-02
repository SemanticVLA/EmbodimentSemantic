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


def _comparison_payload(name: str, config_hash: str, runtime_hash: str, *, seed: int = 1000, success: bool = True, external_runtime: dict | None = None) -> dict:
    config_hash = {
        "v9-hash": "1" * 64,
        "c10-hash": "2" * 64,
        "c11-hash": "3" * 64,
        "other-hash": "4" * 64,
    }.get(config_hash, config_hash)
    record = _record(4, "vanilla", seed, success=success, config_hash=config_hash)
    canonical = {"name": name}
    record["controller_config_observed"] = {"canonical": canonical, "config_hash": runtime_hash}
    record["protocol"] = {
        "name": "libero_arrow_pick_place_matrix",
        "schema_version": "arrow_pick_place_matrix.v1",
        "camera": "agentview",
        "suite_contract": "matched-suite",
        "seed_policy": "seed=seed_base+episode_index",
        "seed_base": 1000,
        "resolution": 256,
        "motion_mode": "execute_motion",
        "source_hashes": {"controller.py": "a" * 64},
    }
    record["provenance"] = {
        "launcher_path": "/repo/run_matrix.py",
        "repository_root": "/repo",
        "episode_runner": "run_episode",
        "git": {"commit": "same"},
        "python": "python-test",
        "platform": "platform-test",
        "dependency_versions": {"libero": "same"},
    }
    payload = {
        "controller_variant": name,
        "controller_config": {
            "config_hash": config_hash,
            "runtime_hash": runtime_hash,
            "canonical": canonical,
        },
        "protocol": record["protocol"],
        "provenance": record["provenance"],
        "cells": [record],
    }
    if external_runtime is not None:
        payload["zerograsp_runtime"] = external_runtime
    return payload


def _comparison_graph() -> dict:
    return {
        "C10": {
            "candidate": {"controller_name": "c10", "canonical_config_hash": "2" * 64},
            "baseline": {"label": "v9_patient_base/control", "controller_name": "v9", "canonical_config_hash": "1" * 64},
            "held_constant": ("task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes"),
            "candidate_only": ("runtime_provenance",),
        },
        "C11": {
            "candidate": {"controller_name": "c11", "canonical_config_hash": "3" * 64},
            "baseline": {"label": "C10", "controller_name": "c10", "canonical_config_hash": "2" * 64},
            "held_constant": ("task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes", "runtime_provenance"),
        },
    }


def _write_comparison(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _add_zerograsp_audit(payload: dict, *, source_px: list[float], destination_px: list[float]) -> dict:
    canonical = {
        "name": payload["controller_variant"],
        "grasp_provider": "zerograsp",
        "placement_provider": "classical",
    }
    payload["controller_config"]["canonical"] = canonical
    record = payload["cells"][0]
    record["controller_config_observed"]["canonical"] = canonical
    record["audit"]["zerograsp"] = {
        "status": "completed",
        "fallback": False,
        "runtime_hash": "zg-runtime",
        "frame_metadata": {
            "camera_frame": "opencv_optical_x_right_y_down_z_forward",
            "world_frame": "libero_mujoco_world",
            "eef_frame": "robot0_eef_pos_grip_site",
            "eef_position_frame": "world_grip_site",
            "eef_orientation_frame": "world_right_hand",
            "rgb_shape": [256, 256, 3],
            "depth_shape": [256, 256],
            "source_px": source_px,
            "destination_px": destination_px,
            "transform": "T_world_camera",
        },
    }
    return payload


def test_v10_c10_comparison_requires_declared_v9_pair_and_reports_held_constants(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "v9.json", _comparison_payload("v9", "v9-hash", "v9-runtime", success=False))
    candidate = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}))
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is True
    assert report["comparison"]["baseline"]["controller_name"] == "v9"
    assert set(report["comparison"]["held_constant"]) == {
        "task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes"
    }


def test_v10_comparison_rejects_wrong_baseline_identity_and_c11_requires_c10(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    wrong_baseline = _write_comparison(tmp_path / "wrong.json", _comparison_payload("other", "other-hash", "other-runtime"))
    candidate = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}))
    report = gate.evaluate_gate(
        candidate, baseline=wrong_baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("baseline controller name mismatch" in error for error in report["errors"])

    c10 = _write_comparison(tmp_path / "c10-base.json", _comparison_payload("c10", "c10-hash", "c10-runtime", success=False, external_runtime={"worker": "zg-v1"}))
    c11 = _write_comparison(tmp_path / "c11.json", _comparison_payload("c11", "c11-hash", "c11-runtime", external_runtime={"worker": "zg-v1"}))
    report = gate.evaluate_gate(
        c11, baseline=c10, comparison_id="C11", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is True


def test_v10_comparison_rejects_held_constant_provenance_drift(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline_payload = _comparison_payload("v9", "v9-hash", "v9-runtime")
    baseline_payload["provenance"]["git"] = {"commit": "different"}
    baseline = _write_comparison(tmp_path / "v9-drift.json", baseline_payload)
    candidate = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}))
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("held-constant provenance mismatch: harness" in error for error in report["errors"])


def test_v10_c10_requires_candidate_only_zerograsp_runtime(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "v9.json", _comparison_payload("v9", "v9-hash", "v9-runtime", success=False))
    candidate = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime"))
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("candidate-only provenance missing: runtime_provenance" in error for error in report["errors"])


def test_v10_c11_rejects_zero_grasp_runtime_drift(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    c10 = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime", success=False, external_runtime={"worker": "zg-v1"}))
    c11 = _write_comparison(tmp_path / "c11.json", _comparison_payload("c11", "c11-hash", "c11-runtime", external_runtime={"worker": "zg-v2"}))
    report = gate.evaluate_gate(
        c11, baseline=c10, comparison_id="C11", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("held-constant provenance mismatch: runtime_provenance" in error for error in report["errors"])


def test_v10_comparison_requires_operator_manifest_hashes(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "v9.json", _comparison_payload("v9", "v9-hash", "v9-runtime", success=False))
    candidate = _write_comparison(
        tmp_path / "c10.json",
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
    )
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10",
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("operator-supplied manifest" in error for error in report["errors"])

    graph = _comparison_graph()
    graph["C10"]["candidate"].pop("canonical_config_hash")
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=graph,
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("missing candidate calibrated canonical config hash" in error for error in report["errors"])

    for malformed_hash in ("a", "A" * 64, "g" * 64):
        graph = _comparison_graph()
        graph["C10"]["candidate"]["canonical_config_hash"] = malformed_hash
        report = gate.evaluate_gate(
            candidate, baseline=baseline, comparison_id="C10", comparison_graph=graph,
            expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
        )
        assert report["passed"] is False
        assert any("not lowercase SHA-256" in error for error in report["errors"])


def test_v10_c11_rejects_empty_runtime_on_both_sides(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "c10.json", _comparison_payload("c10", "c10-hash", "c10-runtime", success=False))
    candidate = _write_comparison(tmp_path / "c11.json", _comparison_payload("c11", "c11-hash", "c11-runtime"))
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C11", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("runtime provenance missing or empty for candidate" in error for error in report["errors"])
    assert any("runtime provenance missing or empty for baseline" in error for error in report["errors"])


def test_comparison_rejects_structurally_corrupt_baseline_but_accepts_low_success_baseline(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline_payload = _comparison_payload("v9", "v9-hash", "v9-runtime", success=False)
    baseline_payload["cells"][0]["controller_config_observed"]["config_hash"] = "corrupt-runtime"
    baseline = _write_comparison(tmp_path / "corrupt-v9.json", baseline_payload)
    candidate = _write_comparison(
        tmp_path / "c10.json",
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
    )
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("baseline integrity failure" in error for error in report["errors"])

    # The same low-success baseline is valid evidence once its structural
    # identity and provenance are repaired; it need not meet the promotion
    # threshold itself.
    baseline_payload["cells"][0]["controller_config_observed"]["config_hash"] = "v9-runtime"
    baseline = _write_comparison(tmp_path / "valid-low-success-v9.json", baseline_payload)
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is True
    assert report["baseline"]["conservative_successes"] == 0


def test_comparison_rejects_non_terminal_baseline_cell(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline_payload = _comparison_payload("v9", "v9-hash", "v9-runtime", success=False)
    baseline_payload["cells"][0]["status"] = "running"
    baseline = _write_comparison(tmp_path / "running-v9.json", baseline_payload)
    candidate = _write_comparison(
        tmp_path / "c10.json",
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
    )
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("baseline integrity failure: non-terminal cell statuses" in error for error in report["errors"])


def test_zerograsp_frame_schema_is_invariant_but_endpoints_may_differ(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline_payload = _comparison_payload("v9", "v9-hash", "v9-runtime", success=False)
    baseline = _write_comparison(tmp_path / "v9.json", baseline_payload)
    candidate_payload = _add_zerograsp_audit(
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
        source_px=[20.0, 30.0], destination_px=[220.0, 230.0],
    )
    second_payload = _add_zerograsp_audit(
        _comparison_payload("c10", "c10-hash", "c10-runtime", seed=1001, external_runtime={"worker": "zg-v1"}),
        source_px=[80.0, 90.0], destination_px=[180.0, 190.0],
    )
    candidate_payload["cells"].append(second_payload["cells"][0])
    candidate_payload["protocol"]["seed_base"] = 1000
    candidate = _write_comparison(tmp_path / "c10.json", candidate_payload)

    # Make the control cover the same two seeds without adding ZeroGrasp.
    baseline_payload["cells"].append(_comparison_payload("v9", "v9-hash", "v9-runtime", seed=1001, success=False)["cells"][0])
    baseline_payload["protocol"]["seed_base"] = 1000
    baseline = _write_comparison(tmp_path / "v9.json", baseline_payload)
    graph = _comparison_graph()
    graph["C10"]["held_constant"] = ("task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes")
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=graph,
        expected_task_ids=(4,), expected_seeds=(1000, 1001), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is True


def test_zerograsp_frame_pixels_must_be_finite_and_in_bounds(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "v9.json", _comparison_payload("v9", "v9-hash", "v9-runtime", success=False))
    candidate_payload = _add_zerograsp_audit(
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
        source_px=[float("nan"), 30.0], destination_px=[256.0, 230.0],
    )
    candidate = _write_comparison(tmp_path / "c10.json", candidate_payload)
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("source_px invalid or out of bounds" in error for error in report["errors"])
    assert any("destination_px invalid or out of bounds" in error for error in report["errors"])


def test_zerograsp_frame_requires_exact_tags_and_coherent_shapes(tmp_path: Path):
    gate = importlib.import_module("arrow_pick_place_matrix_gate")
    baseline = _write_comparison(tmp_path / "v9.json", _comparison_payload("v9", "v9-hash", "v9-runtime", success=False))
    candidate_payload = _add_zerograsp_audit(
        _comparison_payload("c10", "c10-hash", "c10-runtime", external_runtime={"worker": "zg-v1"}),
        source_px=[20.0, 30.0], destination_px=[220.0, 230.0],
    )
    frame = candidate_payload["cells"][0]["audit"]["zerograsp"]["frame_metadata"]
    frame.pop("eef_frame")
    frame["world_frame"] = "wrong-world-frame"
    frame["depth_shape"] = [128, 256]
    candidate = _write_comparison(tmp_path / "c10.json", candidate_payload)
    report = gate.evaluate_gate(
        candidate, baseline=baseline, comparison_id="C10", comparison_graph=_comparison_graph(),
        expected_task_ids=(4,), expected_seeds=(1000,), expected_suites=("vanilla",), hard_tasks=(),
    )
    assert report["passed"] is False
    assert any("eef_frame missing or invalid" in error for error in report["errors"])
    assert any("world_frame missing or invalid" in error for error in report["errors"])
    assert any("RGB/depth shape mismatch" in error for error in report["errors"])
