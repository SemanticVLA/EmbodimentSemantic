#!/usr/bin/env python3
"""Offline integrity and canary gate for arrow pick/place matrix outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CANARY_TASK_IDS = (4, 6, 9)
DEFAULT_CANARY_SEEDS = tuple(range(1000, 1010))
DEFAULT_EXPECTED_SUITES = ("vanilla", "sealed_randomized")


def _resolve_reference(source: Path, value: Any, fallback: str) -> Path:
    candidate = Path(str(value or fallback)).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return candidate.resolve()


def _cell_identity(record: Mapping[str, Any]) -> tuple[int, str, int] | None:
    try:
        return int(record["task_id"]), str(record["suite_mode"]), int(record["seed"])
    except (KeyError, TypeError, ValueError):
        return None


def _attempt_rank(record: Mapping[str, Any], line_number: int) -> tuple[int, float, int]:
    attempts = record.get("attempts")
    attempt_index = 0
    finished = 0.0
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        if isinstance(last, Mapping):
            try:
                attempt_index = int(last.get("attempt_index", len(attempts)))
            except (TypeError, ValueError):
                attempt_index = len(attempts)
            try:
                finished = float(last.get("finished_unix", 0.0) or 0.0)
            except (TypeError, ValueError):
                finished = 0.0
    try:
        attempt_index = max(attempt_index, int(record.get("attempt_index", 0) or 0))
    except (TypeError, ValueError):
        pass
    return attempt_index, finished, line_number


def _dedupe_append_only(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest attempt for each cell in append-only history."""
    selected: dict[tuple[Any, ...], tuple[tuple[int, float, int], dict[str, Any]]] = {}
    for line_number, item in enumerate(records):
        record = dict(item)
        identity = _cell_identity(record)
        key: tuple[Any, ...] = ("cell",) + identity if identity is not None else ("invalid", line_number)
        ranked = (_attempt_rank(record, line_number), record)
        previous = selected.get(key)
        if previous is None or ranked[0] >= previous[0]:
            selected[key] = ranked
    return [item for _, item in sorted(selected.values(), key=lambda value: value[0][2])]


def _read_records(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid matrix JSONL at line {line_number}") from exc
            if isinstance(value, Mapping):
                records.append(dict(value))
        return {}, _dedupe_append_only(records)

    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("matrix artifact root must be an object")

    per_suite = payload.get("per_suite")
    if isinstance(per_suite, Mapping):
        combined_records: list[dict[str, Any]] = []
        suite_metadata: dict[str, dict[str, Any]] = {}
        for suite_mode, entry in per_suite.items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid per_suite entry for {suite_mode}")
            summary_path = _resolve_reference(
                source, entry.get("summary_path"),
                f"{suite_mode}/arrow_pick_place_matrix_summary.json",
            )
            if not summary_path.is_file() and entry.get("status_path"):
                summary_path = _resolve_reference(source, entry["status_path"], "")
            child_metadata, suite_records = _read_records(summary_path)
            suite_metadata[str(suite_mode)] = child_metadata
            combined_records.extend(suite_records)
        result = dict(payload)
        result["_suite_metadata"] = suite_metadata
        return result, combined_records

    # A summary is an aggregate; its final status inventory is authoritative
    # whenever available, rather than append-only JSONL history.
    status_value = payload.get("status_path")
    if status_value:
        status_path = _resolve_reference(source, status_value, "")
        if status_path.is_file() and status_path != source:
            _status_metadata, status_records = _read_records(status_path)
            return dict(payload), status_records

    cells = payload.get("cells")
    if isinstance(cells, list):
        return dict(payload), [dict(item) for item in cells if isinstance(item, Mapping)]

    manifest_value = payload.get("manifest_path")
    if manifest_value:
        manifest_path = _resolve_reference(source, manifest_value, "")
        if manifest_path.is_file() and manifest_path != source:
            _manifest_metadata, records = _read_records(manifest_path)
            return dict(payload), records
    return dict(payload), []


def _bool_success(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "completed" or record.get("dry_run") is True:
        return False
    audit = record.get("audit")
    value = audit.get("evaluator_success") if isinstance(audit, Mapping) else record.get("evaluator_result")
    return value is True


def _substantive(value: Any) -> bool:
    if value is None or value is False or value == "":
        return False
    if isinstance(value, Mapping):
        return any(_substantive(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value) and any(_substantive(item) for item in value)
    return True


def _meaningful_capture(diagnostics: Mapping[str, Any]) -> bool:
    capture = diagnostics.get("capture_contract")
    if not isinstance(capture, Mapping):
        return False
    return capture.get("valid") is True and any(
        key != "valid" and _substantive(value) for key, value in capture.items()
    )


def _meaningful_phases(diagnostics: Mapping[str, Any]) -> bool:
    phases = diagnostics.get("phases")
    if isinstance(phases, list) and any(
        isinstance(item, Mapping)
        and _substantive(item.get("phase"))
        and _substantive(item.get("status"))
        for item in phases
    ):
        return True
    frames = diagnostics.get("phase_frames")
    return isinstance(frames, list) and any(_substantive(item) for item in frames)


def _diagnostics_complete(record: Mapping[str, Any]) -> bool:
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    if _meaningful_capture(diagnostics) or _meaningful_phases(diagnostics):
        return True
    if record.get("status") == "completed":
        return False
    partial = record.get("partial_audit")
    return isinstance(partial, Mapping) and _substantive(partial)


def _config_declaration(metadata: Mapping[str, Any], suite_mode: str) -> Mapping[str, Any] | None:
    suite_metadata = metadata.get("_suite_metadata")
    candidates: list[Any] = []
    if isinstance(suite_metadata, Mapping):
        child = suite_metadata.get(str(suite_mode))
        if isinstance(child, Mapping):
            candidates.extend((
                child.get("controller_config"),
                child.get("protocol", {}).get("controller_config") if isinstance(child.get("protocol"), Mapping) else None,
                child.get("provenance", {}).get("controller_config") if isinstance(child.get("provenance"), Mapping) else None,
            ))
    candidates.extend((
        metadata.get("controller_config"),
        metadata.get("protocol", {}).get("controller_config") if isinstance(metadata.get("protocol"), Mapping) else None,
        metadata.get("provenance", {}).get("controller_config") if isinstance(metadata.get("provenance"), Mapping) else None,
    ))
    return next((item for item in candidates if isinstance(item, Mapping)), None)


def _observed_config(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("controller_config_observed")
    if value is None and isinstance(record.get("audit"), Mapping):
        value = record["audit"].get("controller_variant")
    return value if isinstance(value, Mapping) else None


def _canonical_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))
    return left == right


def evaluate_gate(
    artifact: str | Path,
    *,
    expected_task_ids: Sequence[int] | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_suites: Sequence[str] = DEFAULT_EXPECTED_SUITES,
    expected_hash: str | None = None,
    expected_controller_config_hash: str | None = None,
    hard_tasks: Sequence[int] = DEFAULT_CANARY_TASK_IDS,
    minimum_hard_task_successes: int = 9,
    baseline: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a matrix artifact and return a machine-readable gate report."""
    metadata, records = _read_records(artifact)
    errors: list[str] = []
    expected_tasks = {int(value) for value in (DEFAULT_CANARY_TASK_IDS if expected_task_ids is None else expected_task_ids)}
    expected_seed_set = {int(value) for value in (DEFAULT_CANARY_SEEDS if expected_seeds is None else expected_seeds)}
    expected_suite_set = {str(value) for value in expected_suites}
    hard_task_set = {int(value) for value in hard_tasks}
    identities: list[tuple[int, str, int]] = []
    invalid_records: list[Any] = []
    for index, record in enumerate(records):
        identity = _cell_identity(record)
        if identity is None:
            invalid_records.append(record.get("cell_index", index))
        else:
            identities.append(identity)
    if invalid_records:
        errors.append(f"invalid cell identities for records {invalid_records[:10]}")
    expected_identities = {
        (task_id, suite_mode, seed)
        for task_id in expected_tasks for suite_mode in expected_suite_set for seed in expected_seed_set
    }
    actual_identities = set(identities)
    if len(identities) != len(actual_identities):
        errors.append("duplicate cell identities")
    missing = sorted(expected_identities - actual_identities)
    unexpected = sorted(actual_identities - expected_identities)
    if missing:
        errors.append(f"missing expected cell identities: {missing[:10]}")
    if unexpected:
        errors.append(f"unexpected cell identities: {unexpected[:10]}")
    if len(records) != len(expected_identities):
        errors.append(f"expected exact denominator {len(expected_identities)}, found {len(records)}")
    if not hard_task_set.issubset(expected_tasks):
        errors.append("hard-task IDs must be included in expected task IDs")

    observed_hashes = {str(r.get("contract_hash")) for r in records if r.get("contract_hash")}
    observed_config_hashes = {str(r.get("controller_config_hash")) for r in records if r.get("controller_config_hash")}
    observed_runtime_hashes: set[str] = set()
    source_hash_documents = {
        json.dumps(r["protocol"]["source_hashes"], sort_keys=True, separators=(",", ":"))
        for r in records
        if isinstance(r.get("protocol"), Mapping) and isinstance(r["protocol"].get("source_hashes"), Mapping)
    }
    if records and not source_hash_documents:
        errors.append("source hashes are missing from terminal records")
    elif len(source_hash_documents) != 1:
        errors.append("source hashes are inconsistent across terminal records")
    for value in (
        metadata.get("contract_hash"),
        metadata.get("provenance", {}).get("contract_hash") if isinstance(metadata.get("provenance"), Mapping) else None,
    ):
        if value:
            observed_hashes.add(str(value))
    if expected_hash and observed_hashes != {str(expected_hash)}:
        errors.append("contract hash mismatch or inconsistency")

    declared_configs: dict[str, Mapping[str, Any]] = {}
    for suite_mode in expected_suite_set:
        declaration = _config_declaration(metadata, suite_mode)
        if declaration is not None:
            declared_configs[suite_mode] = declaration
    declared_hashes = {str(item.get("config_hash")) for item in declared_configs.values() if item.get("config_hash")}
    if expected_controller_config_hash and declared_hashes != {str(expected_controller_config_hash)}:
        errors.append("controller config hash mismatch")
    for record in records:
        identity = _cell_identity(record)
        if identity is None:
            continue
        declaration = declared_configs.get(identity[1])
        observed = _observed_config(record)
        if declaration is None:
            continue  # legacy v1-v8 records have no config declaration
        if observed is None:
            errors.append(f"controller config observation missing for cell {record.get('cell_index')}")
            continue
        runtime_hash = observed.get("config_hash")
        if runtime_hash:
            observed_runtime_hashes.add(str(runtime_hash))
        expected_runtime_hash = declaration.get("runtime_hash") or declaration.get("config_hash")
        if str(runtime_hash) != str(expected_runtime_hash):
            errors.append(f"controller config runtime hash mismatch for cell {record.get('cell_index')}")
        if "canonical" in declaration and not _canonical_equal(observed.get("canonical"), declaration.get("canonical")):
            errors.append(f"controller config canonical mismatch for cell {record.get('cell_index')}")
    metadata_config = metadata.get("controller_config")
    if isinstance(metadata_config, Mapping) and metadata_config.get("config_hash"):
        declared_config_hash = str(metadata_config["config_hash"])
        if observed_config_hashes and observed_config_hashes != {declared_config_hash}:
            errors.append("controller config hash is missing or inconsistent across cells")

    incomplete = [r.get("cell_index") for r in records if not _diagnostics_complete(r)]
    if incomplete:
        errors.append(f"diagnostics incomplete for cells {incomplete[:10]}")
    successes = sum(_bool_success(r) for r in records)
    strata: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "successes": 0})
    for record in records:
        identity = _cell_identity(record)
        if identity is None:
            continue
        key = f"task_{identity[0]}__{identity[1]}"
        strata[key]["total"] += 1
        strata[key]["successes"] += int(_bool_success(record))
    hard_failures: dict[str, dict[str, int]] = {}
    for task_id in sorted(hard_task_set):
        for suite_mode in sorted(expected_suite_set):
            key = f"task_{task_id}__{suite_mode}"
            value = strata.get(key, {"total": 0, "successes": 0})
            if value["total"] != len(expected_seed_set) or value["successes"] < int(minimum_hard_task_successes):
                hard_failures[key] = dict(value)
    if hard_failures:
        errors.append("hard-task stratum gate failed")
    baseline_report = None
    if baseline is not None:
        baseline_report = evaluate_gate(
            baseline, expected_task_ids=expected_task_ids, expected_seeds=expected_seeds,
            expected_suites=expected_suites, expected_hash=None,
            expected_controller_config_hash=None, hard_tasks=hard_tasks,
            minimum_hard_task_successes=minimum_hard_task_successes,
        )
        if successes <= int(baseline_report["conservative_successes"]):
            errors.append("factor positive-signal rule failed: no strict conservative improvement")
        for key, current in strata.items():
            prior = baseline_report["strata"].get(key)
            if prior and current["successes"] < prior["successes"]:
                errors.append(f"factor positive-signal rule failed: stratum drop in {key}")
    return {
        "passed": not errors,
        "artifact": str(Path(artifact).expanduser().resolve()),
        "cell_count": len(records),
        "conservative_successes": int(successes),
        "conservative_denominator": len(expected_identities),
        "strata": dict(strata),
        "hard_task_failures": hard_failures,
        "observed_contract_hashes": sorted(observed_hashes),
        "observed_controller_config_hashes": sorted(observed_config_hashes),
        "observed_controller_runtime_hashes": sorted(observed_runtime_hashes),
        "diagnostics_incomplete_cells": incomplete,
        "errors": errors,
        "baseline": baseline_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expected-task-ids", default=None)
    parser.add_argument("--expected-seeds", default=None)
    parser.add_argument("--expected-contract-hash", default=None)
    parser.add_argument("--expected-controller-config-hash", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    parse_ints = lambda value: None if value is None else [int(part) for part in value.split(",") if part]
    report = evaluate_gate(
        args.artifact, baseline=args.baseline,
        expected_task_ids=parse_ints(args.expected_task_ids), expected_seeds=parse_ints(args.expected_seeds),
        expected_hash=args.expected_contract_hash,
        expected_controller_config_hash=args.expected_controller_config_hash,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
