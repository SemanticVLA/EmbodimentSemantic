"""Offline, archive-only forensics for arrow pick/place failures.

The input is one or more audit JSON/JSONL files or archive directories.  This
module never constructs a simulator, moves a robot, or queries an evaluator.
Evaluator values already present in an audit are retained only as offline
outcome labels; controller-stage classification uses controller evidence only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


FAILURE_STATUSES = {
    "failed",
    "failure",
    "error",
    "stall",
    "timeout",
    "aborted",
    "exception",
}
PHASE_TO_STAGE = {
    "pregrasp": "pregrasp",
    "descend": "grasp",
    "close": "grasp",
    "lift": "lift",
    "preplace": "transport",
    "pre_place": "transport",
    "descend_place": "place",
    "place": "place",
    "open": "place",
    "retreat": "retreat",
}
STAGE_ORDER = ("pregrasp", "grasp", "lift", "transport", "place", "retreat", "unknown")
CSV_FIELDS = (
    "task_id",
    "episode_index",
    "seed",
    "cell_index",
    "suite_mode",
    "status",
    "source_path",
    "controller_failure",
    "controller_stage",
    "controller_confidence",
    "controller_evidence",
    "evaluator_outcome",
    "motion_began",
    "failed_phase",
    "phase_status",
    "frame_count",
    "existing_frame_count",
    "depth_file_count",
    "valid_depth_file_count",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def _input_files(inputs: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in inputs:
        path = _canonical(Path(raw))
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            found.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}:
                    found.add(_canonical(candidate))
    return sorted(found, key=lambda item: item.as_posix())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item, _source_path=source.as_posix()) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    if isinstance(payload.get("cells"), list):
        return [dict(item, _source_path=source.as_posix()) for item in payload["cells"] if isinstance(item, Mapping)]
    # Matrix summaries contain aggregate ``failed_cells`` but not per-cell
    # traces. Do not mistake those summaries for audit records.
    if "failed_cells" in payload and "task_id" not in payload and "audit" not in payload:
        return []
    return [dict(payload, _source_path=source.as_posix())]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            records.extend(_records_from_payload(payload, path))
        return records
    return _records_from_payload(_load_json(path), path)


def _resolve_reference(value: Any, source: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = source.parent / candidate
    return _canonical(candidate)


def _nested_values(container: Mapping[str, Any], keys: Iterable[str]) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        if key in container:
            value = container[key]
            values.extend(value if isinstance(value, list) else [value])
    return values


def _load_referenced_audit(record: Mapping[str, Any], source: Path) -> tuple[dict[str, Any], Path]:
    audit = record.get("audit")
    if isinstance(audit, Mapping):
        return dict(audit), source
    audit_path = _resolve_reference(record.get("audit_path"), source)
    if audit_path is not None and audit_path.is_file():
        payload = _load_json(audit_path)
        if isinstance(payload, Mapping):
            return dict(payload), audit_path
    # A direct run_episode audit is itself the evidence source.
    if "schema_version" in record or "phases" in record or "phase_frames" in record:
        return dict(record), source
    partial = record.get("partial_audit")
    if isinstance(partial, Mapping):
        return dict(partial), source
    diagnostics = record.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics), source
    return {}, source


def _phase_records(record: Mapping[str, Any], audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    for container in (audit, record.get("partial_audit"), record.get("diagnostics"), record):
        if isinstance(container, Mapping) and isinstance(container.get("phases"), list):
            return [dict(item) for item in container["phases"] if isinstance(item, Mapping)]
    return []


def _evaluator_outcome(record: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    for container in (audit, record):
        if not isinstance(container, Mapping):
            continue
        value = container.get("evaluator_success")
        if value is None:
            value = container.get("evaluator_result")
        if isinstance(value, bool):
            return "success" if value else "failure"
    return "not_available"


def _phase_failure(phases: list[Mapping[str, Any]]) -> tuple[str | None, str | None, str | None]:
    for phase in reversed(phases):
        name = str(phase.get("phase", "")).strip().lower()
        status = str(phase.get("status", "")).strip().lower()
        if name and status in FAILURE_STATUSES:
            return name, status, PHASE_TO_STAGE.get(name, "unknown")
    return None, None, None


def _error_phase(error: Any) -> str | None:
    text = str(error or "").lower()
    match = re.search(r"phase\s+([a-z_]+)", text)
    if match:
        return match.group(1)
    for phase in sorted(PHASE_TO_STAGE, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phase)}\b", text):
            return phase
    return None


def _classify_controller_failure(record: Mapping[str, Any], audit: Mapping[str, Any], phases: list[Mapping[str, Any]]) -> dict[str, Any]:
    evaluator = _evaluator_outcome(record, audit)
    explicit_class = str(record.get("failure_class", ""))
    status = str(record.get("status", ""))
    phase_name, phase_status, stage = _phase_failure(phases)
    evidence: list[str] = []
    confidence = "none"
    if evaluator == "failure" and status == "completed":
        return {"controller_failure": False, "stage": None, "confidence": "none", "evidence": [], "failed_phase": None, "phase_status": None}
    if phase_name is not None:
        evidence.append(f"phase:{phase_name}:{phase_status}")
        confidence = "high" if bool(record.get("motion_began", audit.get("motion_executed", False))) else "medium"
    else:
        error_phase = _error_phase(record.get("error") or audit.get("error"))
        if error_phase is not None:
            phase_name = error_phase
            stage = PHASE_TO_STAGE.get(error_phase, "unknown")
            evidence.append(f"error_phase:{error_phase}")
            confidence = "medium"
        elif explicit_class == "controller_failure" or status == "failed" and str(record.get("stage", "")) == "run_episode":
            evidence.append("record:controller_failure")
            stage = "unknown"
            confidence = "low"
    controller = bool(evidence) and evaluator != "failure"
    return {
        "controller_failure": controller,
        "stage": stage if controller else None,
        "confidence": confidence if controller else "none",
        "evidence": evidence if controller else [],
        "failed_phase": phase_name if controller else None,
        "phase_status": phase_status if controller else None,
    }


def _depth_summary(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    summaries: list[dict[str, Any]] = []
    valid_count = 0
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    for path in paths:
        item: dict[str, Any] = {"path": path.as_posix(), "exists": path.is_file()}
        if path.is_file() and path.suffix.lower() == ".npy" and np is not None:
            try:
                array = np.asarray(np.load(path, allow_pickle=False))
                finite = array[np.isfinite(array)]
                item.update({"shape": list(array.shape), "dtype": str(array.dtype), "finite_count": int(finite.size)})
                if finite.size:
                    valid_count += 1
                    item.update({"min": float(np.min(finite)), "max": float(np.max(finite)), "median": float(np.median(finite))})
            except (OSError, ValueError, TypeError) as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        summaries.append(item)
    return summaries, valid_count


def _image_references(record: Mapping[str, Any], audit: Mapping[str, Any], source: Path) -> list[dict[str, Any]]:
    values: list[Any] = []
    for container in (audit, record.get("partial_audit"), record.get("diagnostics"), record):
        if isinstance(container, Mapping):
            values.extend(_nested_values(container, ("phase_frames", "frames")))
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        path = _resolve_reference(value, source)
        if path is None or path.as_posix() in seen:
            continue
        seen.add(path.as_posix())
        references.append({"path": path.as_posix(), "exists": path.is_file()})
    return references


def _depth_references(record: Mapping[str, Any], audit: Mapping[str, Any], source: Path) -> list[Path]:
    paths: list[Path] = []
    for container in (audit, record.get("partial_audit"), record.get("diagnostics"), record):
        if not isinstance(container, Mapping):
            continue
        for value in _nested_values(container, ("depth_input", "metric_depth", "metric_depth_m", "depth_path", "depth")):
            path = _resolve_reference(value, source)
            if path is not None and path not in paths:
                paths.append(path)
    return paths


def analyse_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source = _canonical(Path(str(record.get("_source_path", "audit.json"))))
    audit, audit_source = _load_referenced_audit(record, source)
    phases = _phase_records(record, audit)
    controller = _classify_controller_failure(record, audit, phases)
    images = _image_references(record, audit, audit_source)
    depth_paths = _depth_references(record, audit, audit_source)
    depth, valid_depth_count = _depth_summary(depth_paths)
    evaluator = _evaluator_outcome(record, audit)
    row = {
        "task_id": record.get("task_id", audit.get("task_id")),
        "episode_index": record.get("episode_index", audit.get("episode_index")),
        "seed": record.get("seed", audit.get("seed")),
        "cell_index": record.get("cell_index"),
        "suite_mode": record.get("suite_mode", audit.get("suite_mode")),
        "status": record.get("status", "completed" if audit else "unknown"),
        "source_path": audit_source.as_posix(),
        "controller_failure": controller["controller_failure"],
        "controller_stage": controller["stage"],
        "controller_confidence": controller["confidence"],
        "controller_evidence": controller["evidence"],
        "evaluator_outcome": evaluator,
        "motion_began": bool(record.get("motion_began", audit.get("motion_executed", False))),
        "failed_phase": controller["failed_phase"],
        "phase_status": controller["phase_status"],
        "phases": _json_safe(phases),
        "frames": images,
        "depth": depth,
        "endpoint_depths_m": _json_safe(audit.get("endpoint_depths_m")),
        "error": record.get("error", audit.get("error")),
        "failure_class": record.get("failure_class"),
        "controller_variant": record.get("controller_variant", audit.get("controller_variant")),
        "frame_count": len(images),
        "existing_frame_count": sum(bool(item["exists"]) for item in images),
        "depth_file_count": len(depth),
        "valid_depth_file_count": valid_depth_count,
    }
    return _json_safe(row)


def _stable_row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in ("suite_mode", "task_id", "episode_index", "seed", "cell_index", "source_path"))


def _make_contact_sheet(rows: list[Mapping[str, Any]], output: Path) -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    items: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in rows:
        for frame in row.get("frames", []):
            if isinstance(frame, Mapping) and frame.get("exists"):
                items.append((row, frame))
    if not items:
        return []
    items.sort(key=lambda pair: (str(pair[1].get("path", "")), _stable_row_key(pair[0])))
    cell_w, cell_h, label_h, columns = 256, 288, 32, 3
    canvas = Image.new("RGB", (columns * cell_w, math.ceil(len(items) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    references: list[dict[str, Any]] = []
    for index, (row, frame) in enumerate(items):
        path = Path(str(frame["path"]))
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w, cell_h - label_h))
                x = (index % columns) * cell_w + (cell_w - image.width) // 2
                y = (index // columns) * cell_h
                canvas.paste(image, (x, y))
        except (OSError, ValueError):
            continue
        label = f"task={row.get('task_id')} ep={row.get('episode_index')} {row.get('failed_phase') or 'frame'}"
        draw.text(((index % columns) * cell_w + 3, (index // columns) * cell_h + cell_h - label_h + 5), label[:48], fill="black")
        references.append({"path": path.as_posix(), "contact_sheet_index": index})
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)
    return references


def build_summary(inputs: Iterable[str | Path], output_dir: str | Path) -> dict[str, Any]:
    output = _canonical(Path(output_dir))
    files = _input_files(inputs)
    raw_records: list[dict[str, Any]] = []
    for path in files:
        raw_records.extend(_load_records(path))
    rows = [analyse_record(record) for record in raw_records]
    rows.sort(key=_stable_row_key)
    # A terminal JSONL row and its referenced audit are the same episode. Keep
    # the most complete record deterministically when both were supplied.
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        variant_key = json.dumps(row.get("controller_variant"), sort_keys=True, default=str)
        key = (str(row.get("task_id", "")), str(row.get("episode_index", "")), str(row.get("seed", "")), str(row.get("cell_index", "")), str(row.get("suite_mode", "")), variant_key)
        previous = unique.get(key)
        if previous is None or (len(row.get("phases", [])), len(row.get("frames", []))) > (len(previous.get("phases", [])), len(previous.get("frames", []))):
            unique[key] = row
    rows = sorted(unique.values(), key=_stable_row_key)
    output.mkdir(parents=True, exist_ok=True)
    contact_sheet = output / "contact_sheet.png"
    contact_refs = _make_contact_sheet(rows, contact_sheet)
    summary = {
        "schema_version": "arrow_failure_forensics.v1",
        "input_files": [path.as_posix() for path in files],
        "record_count": len(rows),
        "controller_failure_count": sum(bool(row["controller_failure"]) for row in rows),
        "controller_failure_by_stage": {stage: sum(row["controller_stage"] == stage for row in rows if row["controller_failure"]) for stage in STAGE_ORDER if any(row["controller_stage"] == stage for row in rows)},
        "controller_confidence": {confidence: sum(row["controller_confidence"] == confidence for row in rows if row["controller_failure"]) for confidence in ("high", "medium", "low") if any(row["controller_confidence"] == confidence for row in rows if row["controller_failure"])},
        "evaluator_outcomes_offline": {outcome: sum(row["evaluator_outcome"] == outcome for row in rows) for outcome in ("success", "failure", "not_available")},
        "contact_sheet": contact_sheet.as_posix() if contact_refs else None,
        "contact_sheet_references": contact_refs,
        "episodes": rows,
    }
    summary = _json_safe(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), sort_keys=True) if isinstance(row.get(field), (list, dict)) else row.get(field) for field in CSV_FIELDS})
    (output / "image_references.json").write_text(json.dumps({"references": contact_refs, "contact_sheet": summary["contact_sheet"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="audit JSON/JSONL files or archive directories")
    parser.add_argument("--output-dir", required=True, help="new or dedicated offline output directory")
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.inputs, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"arrow failure forensics: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": str((_canonical(Path(args.output_dir)) / 'summary.json')), "record_count": summary["record_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
