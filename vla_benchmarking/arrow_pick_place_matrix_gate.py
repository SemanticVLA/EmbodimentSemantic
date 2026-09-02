#!/usr/bin/env python3
"""Offline integrity and canary gate for arrow pick/place matrix outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CANARY_TASK_IDS = (4, 6, 9)
DEFAULT_CANARY_SEEDS = tuple(range(1000, 1010))
DEFAULT_EXPECTED_SUITES = ("vanilla", "sealed_randomized")

# This is an explicit comparison graph, rather than a positional convention
# inferred from filenames.  Hashes are the semantic (canonical JSON) hashes
# emitted in ``controller_config.config_hash``.  A caller may supply an
# equivalent graph to ``evaluate_gate`` for a relocated/custom config, while
# these defaults make accidental C10/C11 baseline swaps fail closed.
V10_COMPARISON_GRAPH = {
    "C10": {
        "candidate": {
            "controller_name": "libero_spatial_akita_bowl_agentview_v10_zg_grasp_only",
            "canonical_config_hash": None,
        },
        "baseline": {
            "label": "v9_patient_base/control",
            "controller_name": "libero_spatial_akita_bowl_agentview_v9_patient_control",
            "canonical_config_hash": None,
        },
        # C10 introduces ZeroGrasp.  Its runtime is therefore candidate-only;
        # LIBERO protocol, harness, dependencies, and source hashes remain
        # held constant against the v9 control.
        "held_constant": ("task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes"),
        "candidate_only": ("runtime_provenance",),
    },
    "C11": {
        "candidate": {
            "controller_name": "libero_spatial_akita_bowl_agentview_v10_zg_grasp_recon_place",
            "canonical_config_hash": None,
        },
        "baseline": {
            "label": "C10",
            "controller_name": "libero_spatial_akita_bowl_agentview_v10_zg_grasp_only",
            "canonical_config_hash": None,
        },
        "held_constant": ("task_ids", "suite_modes", "seeds", "protocol", "harness", "source_hashes", "runtime_provenance"),
    },
}


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


def _non_terminal_cells(records: Sequence[Mapping[str, Any]]) -> list[Any]:
    terminal_statuses = {"completed", "failed", "interrupted"}
    return [
        record.get("cell_index", index)
        for index, record in enumerate(records)
        if record.get("status") not in terminal_statuses
    ]


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


_P4_EVIDENCE_FIELDS = (
    "motion_trace", "motion_trace_path", "motion_trace_sha256",
    "motion_trace_max_steps", "motion_trace_truncated", "failure_snapshots",
    "failure_bundle", "post_lift_retention_gate", "retention_gate",
    "source_approach", "support_plane", "placement_observable",
    "placement_after_retention", "canary_video", "input_budget", "forbidden_input_audit",
)
_P4_FORBIDDEN_KEYS = {
    "object_pose", "object_poses", "sim_state", "simulator_state",
    "scene_graph", "evaluator", "evaluator_result", "bbox", "bboxes",
}
_P4_PLACEMENT_PHASES = {"preplace", "pre_place", "descend_place", "place", "open", "retreat"}


def _evidence_value(record: Mapping[str, Any], field: str) -> Any:
    """Read additive evidence from final, partial, diagnostic, or flat records."""
    found: Any = None
    for container in (
        record.get("audit"), record.get("partial_audit"),
        record.get("diagnostics"), record,
    ):
        if isinstance(container, Mapping) and field in container:
            value = container[field]
            if found is None:
                found = value
            if _substantive(value):
                return value
    return found


def _p4_active(metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> bool:
    for key in ("evidence_contract", "p4_evidence", "persistence_contract"):
        if _substantive(metadata.get(key)):
            return True
    return any(
        _substantive(_evidence_value(record, field))
        for record in records for field in _P4_EVIDENCE_FIELDS
    )


def _trace_from_record(record: Mapping[str, Any], artifact: Path) -> tuple[list[Any] | None, Path | None]:
    value = _evidence_value(record, "motion_trace")
    if isinstance(value, list):
        return value, None
    raw_path = _evidence_value(record, "motion_trace_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = artifact.parent / path
    path = path.resolve()
    if not path.is_file():
        return None, path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, path
    steps = payload.get("steps") if isinstance(payload, Mapping) else None
    return steps if isinstance(steps, list) else None, path


def _p4_trace_errors(record: Mapping[str, Any], artifact: Path) -> list[str]:
    trace, trace_path = _trace_from_record(record, artifact)
    cell = record.get("cell_index")
    if trace is None:
        return [f"motion trace missing or unreadable for cell {cell}"]
    if not trace:
        return [f"motion trace empty for cell {cell}"]
    by_phase: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item in trace:
        if not isinstance(item, Mapping):
            return [f"motion trace contains invalid entry for cell {cell}"]
        phase = str(item.get("phase", ""))
        try:
            step = int(item["step"])
        except (KeyError, TypeError, ValueError):
            return [f"motion trace step missing for cell {cell}"]
        if step < 1:
            return [f"motion trace step invalid for cell {cell}"]
        segment = str(item.get("segment", "initial"))
        by_phase[(phase, segment)].append(step)
    errors: list[str] = []
    for (phase, segment), steps in by_phase.items():
        ordered = sorted(set(steps))
        if len(ordered) != len(steps) or ordered != list(range(1, ordered[-1] + 1)):
            errors.append(f"motion trace has missing or gapped steps for cell {cell} phase {phase} segment {segment}")
    if _evidence_value(record, "motion_trace_truncated") is True:
        errors.append(f"motion trace truncated for cell {cell}")
    maximum = _evidence_value(record, "motion_trace_max_steps")
    if maximum is not None:
        try:
            if int(maximum) <= 0 or len(trace) > int(maximum):
                errors.append(f"motion trace exceeds declared budget for cell {cell}")
        except (TypeError, ValueError):
            errors.append(f"motion trace max_steps invalid for cell {cell}")
    expected_hash = _evidence_value(record, "motion_trace_sha256")
    if expected_hash is not None:
        if not _is_sha256_hex(expected_hash):
            errors.append(f"motion trace hash invalid for cell {cell}")
        elif trace_path is not None:
            actual = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            if actual != expected_hash:
                errors.append(f"motion trace hash mismatch for cell {cell}")
    return errors


def _walk_forbidden(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if name in _P4_FORBIDDEN_KEYS:
                found.append(prefix + str(key))
            found.extend(_walk_forbidden(item, prefix + str(key) + "."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_forbidden(item, prefix + f"[{index}]."))
    return found


def _p4_evidence_errors(metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]], artifact: Path) -> list[str]:
    if not _p4_active(metadata, records):
        return []
    errors: list[str] = []
    trace_active = any(
        _substantive(_evidence_value(record, field))
        for record in records for field in ("motion_trace", "motion_trace_path", "motion_trace_sha256")
    )
    for record in records:
        cell = record.get("cell_index")
        if trace_active:
            errors.extend(_p4_trace_errors(record, artifact))
            video = _evidence_value(record, "canary_video")
            if not isinstance(video, Mapping) or video.get("status") != "complete" or not isinstance(video.get("video_path"), str):
                errors.append(f"canary video missing or incomplete for cell {cell}")
            else:
                video_path = Path(video["video_path"]).expanduser()
                if not video_path.is_absolute():
                    video_path = artifact.parent / video_path
                if not video_path.resolve().is_file():
                    errors.append(f"canary video unreadable for cell {cell}")
        status = str(record.get("status", ""))
        failure_class = str(record.get("failure_class") or "")
        needs_failure_bundle = (
            status != "completed"
            or failure_class in {"controller_failure", "runtime_failure", "evidence_unavailable"}
            or failure_class.startswith("controller_")
            or failure_class.startswith("runtime_")
        )
        if needs_failure_bundle:
            bundle = _evidence_value(record, "failure_bundle")
            if bundle is None:
                bundle = _evidence_value(record, "failure_snapshots")
            if not _substantive(bundle):
                errors.append(f"failure bundle missing for cell {cell}")
            elif isinstance(bundle, str):
                path = Path(bundle).expanduser()
                if not path.is_absolute():
                    path = artifact.parent / path
                if not path.resolve().is_file():
                    errors.append(f"failure bundle unreadable for cell {cell}")
        support = _evidence_value(record, "support_plane")
        if support is not None:
            if not isinstance(support, Mapping) or support.get("valid") is not True:
                errors.append(f"invalid support plane for cell {cell}")
            elif "normal" in support or "plane_normal" in support:
                normal = support.get("normal", support.get("plane_normal"))
                try:
                    values = [float(item) for item in normal]
                    if len(values) != 3 or not all(math.isfinite(item) for item in values) or math.sqrt(sum(item * item for item in values)) <= 1e-9:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"invalid support plane normal for cell {cell}")
        source_approach = _evidence_value(record, "source_approach")
        if source_approach is not None:
            if not isinstance(source_approach, Mapping) or not _substantive(source_approach):
                errors.append(f"source approach evidence missing for cell {cell}")
            forbidden = _walk_forbidden(source_approach)
            if forbidden:
                errors.append(f"forbidden controller input in source approach for cell {cell}: {forbidden[0]}")
        forbidden_audit = _evidence_value(record, "forbidden_input_audit")
        if _substantive(forbidden_audit):
            errors.append(f"forbidden controller input audit is non-empty for cell {cell}")
        budget = _evidence_value(record, "input_budget")
        if isinstance(budget, Mapping):
            try:
                used = float(budget.get("used", budget.get("actions_used", 0)))
                limit = float(budget.get("limit", budget.get("max_actions")))
                if not math.isfinite(used) or not math.isfinite(limit) or used < 0 or limit < 0 or used > limit:
                    errors.append(f"input budget exceeded or invalid for cell {cell}")
            except (TypeError, ValueError):
                errors.append(f"input budget incomplete for cell {cell}")
        retention = _evidence_value(record, "post_lift_retention_gate")
        if retention is None:
            retention = _evidence_value(record, "retention_gate")
        if retention is not None:
            values = retention.get("records", []) if isinstance(retention, Mapping) else retention
            values = values if isinstance(values, list) else [values]
            decision = values[-1] if values and isinstance(values[-1], Mapping) else None
            retained = decision.get("retained") if decision else None
            if retained is not True:
                phases = record.get("audit", {}).get("phases", []) if isinstance(record.get("audit"), Mapping) else record.get("phases", [])
                if any(isinstance(item, Mapping) and str(item.get("phase")) in _P4_PLACEMENT_PHASES and int(item.get("steps", 0) or 0) > 0 for item in phases):
                    errors.append(f"placement after retention failure or unobservable for cell {cell}")
                if record.get("evaluator_result") is True or (isinstance(record.get("audit"), Mapping) and record["audit"].get("evaluator_success") is True):
                    errors.append(f"placement/evaluator success after retention failure for cell {cell}")
    return errors


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


def _zerograsp_observed(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for container in (record, record.get("audit"), record.get("diagnostics"), record.get("partial_audit")):
        if isinstance(container, Mapping) and isinstance(container.get("zerograsp"), Mapping):
            return container["zerograsp"]
    return None


def _zerograsp_configured(declarations: Sequence[Mapping[str, Any]]) -> bool:
    for declaration in declarations:
        canonical = declaration.get("canonical", declaration)
        if not isinstance(canonical, Mapping):
            continue
        if canonical.get("grasp_provider") == "zerograsp" or canonical.get("placement_provider") == "zerograsp_reconstruction":
            return True
    return False


_FRAME_INVARIANT_FIELDS = (
    "camera_frame", "world_frame", "eef_frame", "eef_position_frame",
    "eef_orientation_frame", "transform", "rgb_shape", "depth_shape",
)
_EXPECTED_FRAME_TAGS = {
    "camera_frame": "opencv_optical_x_right_y_down_z_forward",
    "world_frame": "libero_mujoco_world",
    "eef_frame": "robot0_eef_pos_grip_site",
    "eef_position_frame": "world_grip_site",
    "eef_orientation_frame": "world_right_hand",
    "transform": "T_world_camera",
}


def _frame_invariants(frame_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: frame_metadata.get(key)
        for key in _FRAME_INVARIANT_FIELDS
    }


def _finite_in_bounds_pixel(value: Any, frame_metadata: Mapping[str, Any]) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return False
    try:
        u, v = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    if not math.isfinite(u) or not math.isfinite(v):
        return False
    shape = frame_metadata.get("rgb_shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) < 2:
        return False
    try:
        height, width = int(shape[0]), int(shape[1])
    except (TypeError, ValueError):
        return False
    return width > 0 and height > 0 and 0.0 <= u < width and 0.0 <= v < height


def _frame_metadata_errors(frame_metadata: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in _EXPECTED_FRAME_TAGS.items():
        if frame_metadata.get(key) != expected:
            errors.append(f"ZeroGrasp frame metadata {key} missing or invalid for cell {record.get('cell_index')}")
    rgb_shape = frame_metadata.get("rgb_shape")
    depth_shape = frame_metadata.get("depth_shape")
    if (
        not isinstance(rgb_shape, Sequence) or isinstance(rgb_shape, (str, bytes)) or len(rgb_shape) != 3
        or not isinstance(depth_shape, Sequence) or isinstance(depth_shape, (str, bytes)) or len(depth_shape) != 2
    ):
        errors.append(f"ZeroGrasp RGB/depth shapes incomplete for cell {record.get('cell_index')}")
        return errors
    try:
        rgb_height, rgb_width, channels = (int(rgb_shape[0]), int(rgb_shape[1]), int(rgb_shape[2]))
        depth_height, depth_width = int(depth_shape[0]), int(depth_shape[1])
    except (TypeError, ValueError):
        errors.append(f"ZeroGrasp RGB/depth shapes invalid for cell {record.get('cell_index')}")
        return errors
    if channels != 3 or rgb_height <= 0 or rgb_width <= 0 or depth_height != rgb_height or depth_width != rgb_width:
        errors.append(f"ZeroGrasp RGB/depth shape mismatch for cell {record.get('cell_index')}")
    protocol = record.get("protocol")
    resolution = protocol.get("resolution") if isinstance(protocol, Mapping) else None
    try:
        if isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)):
            expected_height, expected_width = int(resolution[0]), int(resolution[1])
        elif resolution is not None:
            expected_height = expected_width = int(resolution)
        else:
            expected_height = expected_width = rgb_height
        if (rgb_height, rgb_width) != (expected_height, expected_width):
            errors.append(f"ZeroGrasp frame resolution disagrees with protocol for cell {record.get('cell_index')}")
    except (TypeError, ValueError, IndexError):
        errors.append(f"ZeroGrasp protocol resolution invalid for cell {record.get('cell_index')}")
    return errors


def _canonical_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))
    return left == right


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


_HELD_PROTOCOL_FIELDS = (
    "name", "schema_version", "camera", "suite_contract", "seed_policy",
    "seed_base", "resolution", "motion_mode", "allow_unvalidated_profile",
    "continue_on_motion_failure",
)
_HELD_PROVENANCE_FIELDS = (
    "launcher_path", "repository_root", "episode_runner", "git", "python",
    "platform", "dependency_versions", "runtime_provenance",
)


def _artifact_identity(metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract comparison identity without trusting output filenames."""
    suites = sorted({identity[1] for record in records if (identity := _cell_identity(record)) is not None})
    declarations = [_config_declaration(metadata, suite) for suite in suites]
    declarations = [item for item in declarations if item is not None]
    controller_name = metadata.get("controller_variant")
    if not isinstance(controller_name, str) and declarations:
        canonical = declarations[0].get("canonical")
        controller_name = canonical.get("name") if isinstance(canonical, Mapping) else None
    config_hashes = {str(item.get("config_hash")) for item in declarations if item.get("config_hash")}
    if not config_hashes:
        config_hashes = {str(record.get("controller_config_hash")) for record in records if record.get("controller_config_hash")}
    return {
        "controller_name": controller_name,
        "canonical_config_hashes": sorted(config_hashes),
        "configured": bool(declarations or config_hashes),
    }


def _comparison_snapshot(metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the material identity that must remain fixed across a factor."""
    identities = sorted(identity for record in records if (identity := _cell_identity(record)) is not None)
    suite_metadata = metadata.get("_suite_metadata")
    metadata_by_suite: dict[str, Mapping[str, Any]] = {}
    if isinstance(suite_metadata, Mapping):
        metadata_by_suite.update({str(key): value for key, value in suite_metadata.items() if isinstance(value, Mapping)})
    if not metadata_by_suite:
        for suite in {identity[1] for identity in identities}:
            metadata_by_suite[suite] = metadata

    protocol: dict[str, Any] = {}
    harness: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for suite in sorted(metadata_by_suite):
        item = metadata_by_suite[suite]
        item_protocol = item.get("protocol") if isinstance(item.get("protocol"), Mapping) else item
        item_provenance = item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
        protocol[suite] = {key: item_protocol.get(key) for key in _HELD_PROTOCOL_FIELDS if key in item_protocol}
        harness[suite] = {key: item_provenance.get(key) for key in _HELD_PROVENANCE_FIELDS if key in item_provenance}
        runtime_value = item.get("runtime_provenance")
        if runtime_value is None:
            runtime_value = {
                key: item[key]
                for key in ("zerograsp_runtime", "external_runtime", "runtime")
                if key in item
            }
        runtime[suite] = runtime_value
        if isinstance(item_protocol.get("source_hashes"), Mapping):
            sources[suite] = item_protocol["source_hashes"]
    # If protocol metadata is absent, terminal records still carry a source
    # hash and it remains part of the comparison contract.
    if not sources:
        source_values = {
            json.dumps(record["protocol"]["source_hashes"], sort_keys=True, separators=(",", ":"))
            for record in records
            if isinstance(record.get("protocol"), Mapping) and isinstance(record["protocol"].get("source_hashes"), Mapping)
        }
        sources["records"] = sorted(source_values)
    # Dual summaries keep shared external-runtime provenance at the root;
    # retain it even though per-suite summaries are used for suite-specific
    # protocol/harness fields.
    root_runtime = {
        key: metadata[key]
        for key in ("zerograsp_runtime", "external_runtime", "runtime")
        if key in metadata
    }
    if root_runtime:
        runtime["__root__"] = root_runtime
    return {
        "task_suite_seed": identities,
        "protocol": protocol,
        "harness": harness,
        "source_hashes": sources,
        "runtime_provenance": runtime,
    }


def _comparison_check(
    *, candidate_metadata: Mapping[str, Any], candidate_records: Sequence[Mapping[str, Any]],
    baseline_metadata: Mapping[str, Any], baseline_records: Sequence[Mapping[str, Any]],
    comparison_id: str, graph: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    spec = graph.get(comparison_id)
    if not isinstance(spec, Mapping):
        return [f"unknown comparison graph node {comparison_id!r}"], {"comparison_id": comparison_id}
    candidate_spec = spec.get("candidate") if isinstance(spec.get("candidate"), Mapping) else {}
    baseline_spec = spec.get("baseline") if isinstance(spec.get("baseline"), Mapping) else {}
    for label, item in (("candidate", candidate_spec), ("baseline", baseline_spec)):
        if not isinstance(item.get("controller_name"), str) or not item.get("controller_name"):
            errors.append(f"{comparison_id} comparison manifest missing {label} controller name")
        if not isinstance(item.get("canonical_config_hash"), str) or not item.get("canonical_config_hash"):
            errors.append(f"{comparison_id} comparison manifest missing {label} calibrated canonical config hash")
        elif not _is_sha256_hex(item["canonical_config_hash"]):
            errors.append(f"{comparison_id} comparison manifest {label} canonical config hash is not lowercase SHA-256")
    candidate_identity = _artifact_identity(candidate_metadata, candidate_records)
    baseline_identity = _artifact_identity(baseline_metadata, baseline_records)
    for label, observed, expected in (
        ("candidate controller name", candidate_identity.get("controller_name"), candidate_spec.get("controller_name")),
        ("baseline controller name", baseline_identity.get("controller_name"), baseline_spec.get("controller_name")),
    ):
        if expected is not None and observed != expected:
            errors.append(f"{comparison_id} {label} mismatch: expected {expected!r}, observed {observed!r}")
    for label, observed, expected in (
        ("candidate canonical config hash", candidate_identity.get("canonical_config_hashes"), candidate_spec.get("canonical_config_hash")),
        ("baseline canonical config hash", baseline_identity.get("canonical_config_hashes"), baseline_spec.get("canonical_config_hash")),
    ):
        if any(not _is_sha256_hex(value) for value in observed):
            errors.append(f"{comparison_id} {label} observed canonical config hash is not lowercase SHA-256")
        if expected is not None and observed != [str(expected)]:
            errors.append(f"{comparison_id} {label} mismatch")

    candidate_snapshot = _comparison_snapshot(candidate_metadata, candidate_records)
    baseline_snapshot = _comparison_snapshot(baseline_metadata, baseline_records)
    held_constant = spec.get("held_constant", _HELD_PROVENANCE_FIELDS)
    held_constant = tuple(str(item) for item in held_constant) if isinstance(held_constant, Sequence) and not isinstance(held_constant, (str, bytes)) else tuple(_HELD_PROVENANCE_FIELDS)
    snapshot_fields = {
        "task_ids": candidate_snapshot["task_suite_seed"],
        "suite_modes": sorted({item[1] for item in candidate_snapshot["task_suite_seed"]}),
        "seeds": sorted({item[2] for item in candidate_snapshot["task_suite_seed"]}),
        "protocol": candidate_snapshot["protocol"],
        "harness": candidate_snapshot["harness"],
        "source_hashes": candidate_snapshot["source_hashes"],
        "runtime_provenance": candidate_snapshot["runtime_provenance"],
    }
    baseline_fields = {
        "task_ids": baseline_snapshot["task_suite_seed"],
        "suite_modes": sorted({item[1] for item in baseline_snapshot["task_suite_seed"]}),
        "seeds": sorted({item[2] for item in baseline_snapshot["task_suite_seed"]}),
        "protocol": baseline_snapshot["protocol"],
        "harness": baseline_snapshot["harness"],
        "source_hashes": baseline_snapshot["source_hashes"],
        "runtime_provenance": baseline_snapshot["runtime_provenance"],
    }
    for field in held_constant:
        if snapshot_fields.get(field) != baseline_fields.get(field):
            errors.append(f"{comparison_id} held-constant provenance mismatch: {field}")
        if field == "runtime_provenance":
            if not _substantive(snapshot_fields.get(field)):
                errors.append(f"{comparison_id} runtime provenance missing or empty for candidate")
            if not _substantive(baseline_fields.get(field)):
                errors.append(f"{comparison_id} runtime provenance missing or empty for baseline")
    candidate_only = spec.get("candidate_only", ())
    candidate_only = tuple(str(item) for item in candidate_only) if isinstance(candidate_only, Sequence) and not isinstance(candidate_only, (str, bytes)) else ()
    for field in candidate_only:
        candidate_value = snapshot_fields.get(field)
        baseline_value = baseline_fields.get(field)
        if not _substantive(candidate_value):
            errors.append(f"{comparison_id} candidate-only provenance missing: {field}")
        if _substantive(baseline_value):
            errors.append(f"{comparison_id} candidate-only provenance present in baseline: {field}")
    return errors, {
        "comparison_id": comparison_id,
        "candidate": candidate_identity,
        "baseline": baseline_identity,
        "held_constant": {field: {"candidate": snapshot_fields.get(field), "baseline": baseline_fields.get(field)} for field in held_constant},
        "candidate_only": {field: {"candidate": snapshot_fields.get(field), "baseline": baseline_fields.get(field)} for field in candidate_only},
    }


def _load_comparison_manifest(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    """Load an operator-declared comparison graph, including calibrated hashes."""
    if isinstance(value, Mapping):
        payload: Any = value
    else:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("comparison manifest must be an object")
    graph = payload.get("comparison_graph", payload.get("graph", payload))
    if not isinstance(graph, Mapping):
        raise ValueError("comparison manifest graph must be an object")
    return graph


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
    comparison_id: str | None = None,
    comparison: str | None = None,
    comparison_graph: Mapping[str, Any] | None = None,
    comparison_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate a matrix artifact and return a machine-readable gate report."""
    metadata, records = _read_records(artifact)
    errors: list[str] = []
    artifact_path = Path(artifact).expanduser().resolve()
    errors.extend(_p4_evidence_errors(metadata, records, artifact_path))
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
    contract_hashes_by_suite: dict[str, set[str]] = defaultdict(set)
    for record in records:
        identity = _cell_identity(record)
        if identity is not None and record.get("contract_hash"):
            contract_hashes_by_suite[identity[1]].add(str(record["contract_hash"]))
    observed_config_hashes = {str(r.get("controller_config_hash")) for r in records if r.get("controller_config_hash")}
    observed_runtime_hashes: set[str] = set()
    observed_zerograsp_runtime_hashes: set[str] = set()
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
    elif any(len(values) > 1 for values in contract_hashes_by_suite.values()):
        errors.append("contract hashes are inconsistent within a suite")

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
    # ZeroGrasp is a learned external runtime.  A matrix is not gateable if
    # any cell can silently use a fallback or if its model/frame provenance is
    # missing.  These checks are conditional so v1-v9 artifacts retain their
    # historical gate behavior.
    if _zerograsp_configured(list(declared_configs.values())):
        zg_runtime_hashes: set[str] = set()
        zg_frame_documents: set[str] = set()
        zg_pixel_count = 0
        for record in records:
            observed_zg = _zerograsp_observed(record)
            if observed_zg is None:
                errors.append(f"ZeroGrasp audit missing for cell {record.get('cell_index')}")
                continue
            if observed_zg.get("fallback") is not False:
                errors.append(f"ZeroGrasp fallback must be false for cell {record.get('cell_index')}")
            runtime_hash = observed_zg.get("runtime_hash")
            frame_metadata = observed_zg.get("frame_metadata")
            if not isinstance(runtime_hash, str) or not runtime_hash:
                errors.append(f"ZeroGrasp runtime hash missing for cell {record.get('cell_index')}")
            else:
                zg_runtime_hashes.add(runtime_hash)
                observed_zerograsp_runtime_hashes.add(runtime_hash)
            if not isinstance(frame_metadata, Mapping):
                errors.append(f"ZeroGrasp frame metadata incomplete for cell {record.get('cell_index')}")
            else:
                errors.extend(_frame_metadata_errors(frame_metadata, record))
                # Frame schema/convention is invariant, but pixel seeds are
                # episode-specific.  Comparing the complete object would
                # incorrectly reject a valid matrix whose arrows differ.
                zg_frame_documents.add(json.dumps(_frame_invariants(frame_metadata), sort_keys=True, separators=(",", ":")))
                for pixel_name in ("source_px", "destination_px"):
                    if pixel_name not in frame_metadata or not _finite_in_bounds_pixel(frame_metadata[pixel_name], frame_metadata):
                        errors.append(f"ZeroGrasp {pixel_name} invalid or out of bounds for cell {record.get('cell_index')}")
                    else:
                        zg_pixel_count += 1
        if len(zg_runtime_hashes) > 1:
            errors.append("ZeroGrasp runtime hashes are inconsistent across cells")
        if len(zg_frame_documents) > 1:
            errors.append("ZeroGrasp frame metadata are inconsistent across cells")
        if records and zg_pixel_count != 2 * len(records):
            errors.append("ZeroGrasp per-cell pixel provenance is incomplete")
    metadata_config = metadata.get("controller_config")
    if isinstance(metadata_config, Mapping) and metadata_config.get("config_hash"):
        declared_config_hash = str(metadata_config["config_hash"])
        if observed_config_hashes and observed_config_hashes != {declared_config_hash}:
            errors.append("controller config hash is missing or inconsistent across cells")

    incomplete = [r.get("cell_index") for r in records if not _diagnostics_complete(r)]
    if incomplete:
        errors.append(f"diagnostics incomplete for cells {incomplete[:10]}")
    non_terminal = _non_terminal_cells(records)
    if non_terminal:
        errors.append(f"non-terminal cell statuses for cells {non_terminal[:10]}")
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
    if comparison_id is not None and comparison is not None and comparison_id != comparison:
        errors.append("comparison_id and comparison disagree")
    comparison_id = comparison_id or comparison
    baseline_report = None
    comparison_report = None
    if baseline is not None:
        baseline_report = evaluate_gate(
            baseline, expected_task_ids=expected_task_ids, expected_seeds=expected_seeds,
            expected_suites=expected_suites, expected_hash=None,
            expected_controller_config_hash=None, hard_tasks=hard_tasks,
            minimum_hard_task_successes=minimum_hard_task_successes,
        )
        # A comparison may use a weak but valid baseline.  It may not use a
        # structurally corrupt one.  The hard-task threshold is deliberately
        # excluded here: that threshold is the candidate promotion criterion,
        # not a prerequisite for a historical control artifact.
        baseline_structural_errors = [
            error for error in baseline_report["errors"]
            if error != "hard-task stratum gate failed"
        ]
        for error in baseline_structural_errors:
            errors.append(f"baseline integrity failure: {error}")
        if successes <= int(baseline_report["conservative_successes"]):
            errors.append("factor positive-signal rule failed: no strict conservative improvement")
        for key, current in strata.items():
            prior = baseline_report["strata"].get(key)
            if prior and current["successes"] < prior["successes"]:
                errors.append(f"factor positive-signal rule failed: stratum drop in {key}")
        if comparison_id is not None:
            baseline_metadata, baseline_records = _read_records(baseline)
            if comparison_manifest is None and comparison_graph is None:
                errors.append(f"{comparison_id} comparison requires an operator-supplied manifest with calibrated hashes")
            else:
                try:
                    graph = _load_comparison_manifest(comparison_manifest or comparison_graph)  # type: ignore[arg-type]
                    comparison_errors, comparison_report = _comparison_check(
                        candidate_metadata=metadata, candidate_records=records,
                        baseline_metadata=baseline_metadata, baseline_records=baseline_records,
                        comparison_id=str(comparison_id), graph=graph,
                    )
                    errors.extend(comparison_errors)
                except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{comparison_id} comparison manifest invalid: {exc}")
    elif comparison_id is not None:
        errors.append(f"{comparison_id} comparison requires a baseline artifact")
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
        "observed_zerograsp_runtime_hashes": sorted(observed_zerograsp_runtime_hashes),
        "diagnostics_incomplete_cells": incomplete,
        "errors": errors,
        "baseline": baseline_report,
        "comparison": comparison_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expected-task-ids", default=None)
    parser.add_argument("--expected-seeds", default=None)
    parser.add_argument("--expected-contract-hash", default=None)
    parser.add_argument("--expected-controller-config-hash", default=None)
    parser.add_argument("--comparison-id", choices=tuple(V10_COMPARISON_GRAPH), default=None)
    parser.add_argument("--comparison-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    parse_ints = lambda value: None if value is None else [int(part) for part in value.split(",") if part]
    report = evaluate_gate(
        args.artifact, baseline=args.baseline,
        expected_task_ids=parse_ints(args.expected_task_ids), expected_seeds=parse_ints(args.expected_seeds),
        expected_hash=args.expected_contract_hash,
        expected_controller_config_hash=args.expected_controller_config_hash,
        comparison_id=args.comparison_id,
        comparison_manifest=args.comparison_manifest,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
