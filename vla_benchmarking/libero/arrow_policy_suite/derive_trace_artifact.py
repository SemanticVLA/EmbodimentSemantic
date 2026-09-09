"""Create-only conversion from retained LeRobot parquet rows to Trace routes.

The converter is deliberately narrower than a general dataset loader.  It
reads only the task/episode/state/index columns needed to produce a Trace
artifact.  The source parquet files remain untouched, and actions/images are
never materialized into the output.  Route anchors are inferred from the EEF
state at the first close and subsequent reopen finger-qpos milestones, or
supplied explicitly by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractError, state8
from .trace_native_factory import TRACE_ROUTE_ARTIFACT_SCHEMA


TRANSFORMATION_REVISION = "trace-state-route-from-lerobot-parquet-v1"
_TASK_COLUMNS = ("task_index", "task_id", "task")
_EPISODE_COLUMNS = ("episode_index", "episode_id", "episode")
_STATE_COLUMNS = ("observation.state", "state")
_FRAME_COLUMNS = ("frame_index", "frame", "index")


@dataclass(frozen=True)
class GripperMilestoneConfig:
    """Explicit finger-qpos transition thresholds in source units."""

    close_delta: float = 0.01
    reopen_delta: float = 0.01
    min_dwell_frames: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.close_delta)) or self.close_delta <= 0.0:
            raise ContractError("close_delta must be finite and positive")
        if not math.isfinite(float(self.reopen_delta)) or self.reopen_delta <= 0.0:
            raise ContractError("reopen_delta must be finite and positive")
        if isinstance(self.min_dwell_frames, bool) or int(self.min_dwell_frames) < 1:
            raise ContractError("min_dwell_frames must be a positive integer")


@dataclass(frozen=True)
class TraceDerivationResult:
    output_path: Path
    output_sha256: str
    artifact: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _finite_point(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be a numeric three-vector")
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a numeric three-vector") from exc
    if len(point) != 3 or any(not math.isfinite(item) for item in point):
        raise ContractError(f"{name} must be a finite three-vector")
    return point


def _normalise_selector(value: Any, name: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ContractError(f"{name} is required")
    return str(value)


def _matches(value: Any, requested: Any) -> bool:
    if str(value) == str(requested):
        return True
    try:
        return int(value) == int(requested)
    except (TypeError, ValueError):
        return False


def _find_column(columns: Sequence[str], candidates: Sequence[str], name: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ContractError(f"LeRobot parquet is missing required {name} column")


def _load_parent_manifest(value: str | Path) -> tuple[str, str | None]:
    """Return a verified parent hash and optional source path."""
    text = str(value)
    if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text):
        return text.lower(), None
    path = Path(value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise ContractError("parent collection manifest must be an existing regular file or SHA-256")
    return _sha256(path), str(path.resolve())


def _parquet_files(dataset_dir: str | Path) -> tuple[Path, ...]:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"LeRobot dataset directory does not exist: {root}")
    files = tuple(sorted(path for path in root.rglob("*.parquet") if path.is_file() and not path.is_symlink()))
    if not files:
        raise ContractError("LeRobot dataset directory contains no regular parquet files")
    return files


def _read_parquet_rows(
    files: Sequence[Path],
    *,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - CI tests inject pyarrow
        raise ContractError("Trace parquet derivation requires pyarrow") from exc
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            # ``ParquetSchema.names`` exposes physical leaf names.  A
            # fixed-size-list column such as LeRobot's ``observation.state``
            # therefore appears there as ``element``.  The Arrow schema keeps
            # the logical top-level field names that ``ParquetFile.read``
            # accepts and must be the source of the projection contract.
            logical_schema = getattr(parquet, "schema_arrow", None)
            if logical_schema is None:
                logical_schema = parquet.schema
            columns = tuple(str(name) for name in logical_schema.names)
            task_column = next((name for name in _TASK_COLUMNS if name in columns), None)
            episode_column = next((name for name in _EPISODE_COLUMNS if name in columns), None)
            state_column = next((name for name in _STATE_COLUMNS if name in columns), None)
            missing = [
                name for name, column in (
                    ("task", task_column), ("episode", episode_column), ("state", state_column)
                ) if column is None
            ]
            try:
                relative_path = path.resolve().relative_to(dataset_root)
            except ValueError as exc:
                raise ContractError(f"parquet path is outside dataset root: {path}") from exc
            relative = relative_path.as_posix()
            # Only the dataset's conventional metadata tree is allowed to
            # contain parquet tables without frame columns. A malformed data
            # shard (or an unexpected parquet elsewhere) must fail closed.
            is_metadata = len(relative_path.parts) > 1 and relative_path.parts[0].lower() == "meta"
            metadata = getattr(parquet, "metadata", None)
            row_count = None if metadata is None else int(metadata.num_rows)
            if missing:
                if not is_metadata:
                    raise ContractError(
                        f"data parquet {relative} is missing required column(s): {', '.join(missing)}"
                    )
                # LeRobot stores metadata tables (for example tasks.parquet and
                # episodes.parquet) beside the frame table.  They are valid
                # parquet inputs but not Trace data inputs.  Record them
                # explicitly so a successful artifact remains auditable.
                reason = "missing_required_columns"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                audits.append({
                    "path": relative,
                    "sha256": _sha256(path),
                    "bytes": int(path.stat().st_size),
                    "row_count": row_count,
                    "selected_row_count": 0,
                    "status": "skipped",
                    "skip_reason": reason,
                    "missing_columns": missing,
                    "available_columns": list(columns),
                    "columns_read": [],
                })
                continue
            frame_column = next((name for name in _FRAME_COLUMNS if name in columns), None)
            requested_columns = [task_column, episode_column, state_column]
            if frame_column is not None:
                requested_columns.append(frame_column)
            table = parquet.read(columns=requested_columns)
            file_rows = table.to_pylist()
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"cannot read LeRobot parquet {path}") from exc
        audits.append({
            "path": relative,
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "row_count": int(parquet.metadata.num_rows),
            "selected_row_count": 0,
            "status": "data",
            "columns_read": [task_column, episode_column, state_column] + ([frame_column] if frame_column else []),
        })
        for row_index, raw in enumerate(file_rows):
            rows.append({
                "task": raw.get(task_column), "episode": raw.get(episode_column),
                "state": raw.get(state_column),
                "frame": row_index if frame_column is None else raw.get(frame_column),
                "source_file": relative, "source_row": row_index,
            })
    data_file_count = sum(1 for audit in audits if audit["status"] == "data")
    if data_file_count == 0:
        reason_summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(skipped_reasons.items())
        ) or "none"
        raise ContractError(
            "no parquet file with required task, episode, and state column(s); "
            f"scanned={len(files)}, skipped_reasons={reason_summary}"
        )
    return rows, audits, rejection_counts, {
        "scanned_parquet_files": len(files),
        "data_parquet_files": data_file_count,
        "skipped_non_data_parquet_files": len(audits) - data_file_count,
        "skipped_non_data_reasons": dict(sorted(skipped_reasons.items())),
    }


def _state_from_row(value: Any) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        return state8({"state": value.get("state", value.get("observation.state"))})
    return state8({"state": value})


def _episode_events(states: Sequence[tuple[float, ...]], cfg: GripperMilestoneConfig) -> tuple[int, int]:
    """Find first close then first later reopen from Panda finger separation.

    LIBERO records the two finger qpos with opposite signs. Their sum
    therefore cancels near zero; the physical opening/separation is the
    signed difference (up to the symmetric 0.5 factor).  The threshold is a
    cumulative excursion rather than an adjacent-frame velocity: close is
    detected after the running open reference falls by ``close_delta`` and
    reopen after the post-close minimum rises by ``reopen_delta``.
    """
    widths = [0.5 * (state[6] - state[7]) for state in states]
    close_index: int | None = None
    reopen_index: int | None = None
    open_reference = widths[0]
    post_close_minimum: float | None = None
    for index in range(1, len(widths)):
        width = widths[index]
        if close_index is None:
            open_reference = max(open_reference, width)
            if open_reference - width >= float(cfg.close_delta):
                close_index = index
                post_close_minimum = width
            continue
        post_close_minimum = min(float(post_close_minimum), width)
        if index - close_index >= int(cfg.min_dwell_frames) and width - post_close_minimum >= float(cfg.reopen_delta):
            reopen_index = index
            break
    if close_index is None or reopen_index is None:
        raise ContractError("route has no close-then-reopen finger-qpos milestone pattern")
    return close_index, reopen_index


def _stable_route_id(task_id: Any, episode_id: Any, states: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(_canonical_json(states)).hexdigest()[:16]
    return f"trace_t{task_id}_e{episode_id}_{digest}"


def derive_trace_artifact(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    task_id: int | str,
    episode_ids: Sequence[int | str],
    graph_triplet: Sequence[str],
    coordinate_frame: str,
    parent_collection_manifest: str | Path,
    source_role: str | None = None,
    destination_role: str | None = None,
    units: str = "m",
    source_anchor: Sequence[float] | None = None,
    destination_anchor: Sequence[float] | None = None,
    milestone_config: GripperMilestoneConfig | None = None,
    transformation_revision: str = TRANSFORMATION_REVISION,
) -> TraceDerivationResult:
    """Derive one immutable Trace artifact from explicit task/episode rows."""
    output = Path(output_path).expanduser()
    if output.exists():
        raise ContractError(f"refusing to overwrite existing Trace artifact: {output}")
    if not transformation_revision.strip():
        raise ContractError("transformation_revision is required")
    if not isinstance(graph_triplet, Sequence) or isinstance(graph_triplet, (str, bytes)) or len(graph_triplet) != 3:
        raise ContractError("graph_triplet must contain exactly three text nodes")
    triplet = tuple(str(value).strip() for value in graph_triplet)
    if any(not value for value in triplet):
        raise ContractError("graph_triplet nodes must be non-empty")
    if not str(coordinate_frame).strip():
        raise ContractError("coordinate_frame is required")
    if str(units).lower() not in {"m", "meter", "meters"}:
        raise ContractError("Trace routes must use metric metres")
    selected_episodes = tuple(str(value) for value in episode_ids)
    if not selected_episodes or len(set(selected_episodes)) != len(selected_episodes):
        raise ContractError("episode_ids must be non-empty and unique")
    source_role = str(source_role or triplet[0]).strip()
    destination_role = str(destination_role or triplet[2]).strip()
    if not source_role or not destination_role:
        raise ContractError("source_role and destination_role are required")
    if (source_anchor is None) != (destination_anchor is None):
        raise ContractError("source_anchor and destination_anchor must be supplied together")
    override_source = None if source_anchor is None else _finite_point(source_anchor, "source_anchor")
    override_destination = None if destination_anchor is None else _finite_point(destination_anchor, "destination_anchor")
    cfg = milestone_config or GripperMilestoneConfig()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    files = _parquet_files(dataset_root)
    rows, file_audits, rejection_counts, parquet_audit = _read_parquet_rows(
        files, dataset_root=dataset_root
    )
    parent_hash, parent_path = _load_parent_manifest(parent_collection_manifest)
    requested = {str(value) for value in selected_episodes}
    selected_rows = [row for row in rows if _matches(row["task"], task_id) and str(row["episode"]) in requested]
    for audit in file_audits:
        audit["selected_row_count"] = sum(1 for row in selected_rows if row["source_file"] == audit["path"])
    present = {str(row["episode"]) for row in selected_rows}
    for episode in selected_episodes:
        if episode not in present:
            rejection_counts["missing_requested_episode"] = rejection_counts.get("missing_requested_episode", 0) + 1
    if not selected_rows:
        raise ContractError("no rows matched explicit task_id and episode_ids")

    grouped: dict[str, list[dict[str, Any]]] = {episode: [] for episode in selected_episodes}
    for row in selected_rows:
        grouped[str(row["episode"])].append(row)
    routes: list[dict[str, Any]] = []
    rejected_routes = 0
    for episode in selected_episodes:
        episode_rows = grouped[episode]
        if not episode_rows:
            continue
        try:
            episode_rows.sort(key=lambda row: (int(row["frame"]), row["source_file"], int(row["source_row"])))
            frame_values = [int(row["frame"]) for row in episode_rows]
            if len(set(frame_values)) != len(frame_values):
                raise ContractError("duplicate frame index")
            states = [_state_from_row(row["state"]) for row in episode_rows]
            close_index, reopen_index = _episode_events(states, cfg)
            inferred_source = tuple(float(value) for value in states[close_index][:3])
            inferred_destination = tuple(float(value) for value in states[reopen_index][:3])
            route_source = override_source or inferred_source
            route_destination = override_destination or inferred_destination
            if math.dist(route_source, route_destination) <= 1e-9:
                raise ContractError("source and destination anchors are degenerate")
            state_payload: list[dict[str, Any]] = [{"state": list(state)} for state in states]
            state_payload[close_index]["event"] = "close"
            state_payload[reopen_index]["event"] = "reopen"
            route_id = _stable_route_id(task_id, episode, state_payload)
            # Validate the persisted route fields locally.  Do not call the
            # resampling helper here: conversion must remain independent of
            # waypoint interpolation and must preserve every source state.
            if len(state_payload) < 2 or close_index >= reopen_index:
                raise ContractError("route state/event ordering is invalid")
            routes.append({
                "route_id": route_id, "task_id": task_id, "episode_id": episode,
                "source_role": source_role, "destination_role": destination_role,
                "graph_triplet": list(triplet), "coordinate_frame": coordinate_frame, "units": "m",
                "source_anchor": list(route_source), "destination_anchor": list(route_destination),
                "states": state_payload, "action_free": True,
                "milestones": {"close_frame_index": frame_values[close_index], "reopen_frame_index": frame_values[reopen_index],
                                "close_delta": cfg.close_delta, "reopen_delta": cfg.reopen_delta,
                                "min_dwell_frames": cfg.min_dwell_frames},
            })
        except (ContractError, TypeError, ValueError, OverflowError) as exc:
            rejected_routes += 1
            reason = str(exc).lower()
            category = "duplicate_frame_index" if "duplicate frame" in reason else (
                "malformed_state" if "state" in reason else "route_rejected")
            rejection_counts[category] = rejection_counts.get(category, 0) + 1

    if not routes:
        raise ContractError("no requested episodes produced a valid close/reopen Trace route")
    artifact: dict[str, Any] = {
        "schema": TRACE_ROUTE_ARTIFACT_SCHEMA,
        "create_only": True,
        "transformation_revision": transformation_revision,
        "selection": {"task_id": task_id, "episode_ids": list(selected_episodes)},
        "graph_triplet": list(triplet), "source_role": source_role, "destination_role": destination_role,
        "coordinate_frame": coordinate_frame, "units": "m",
        "parent_collection_manifest_sha256": parent_hash,
        "parent_collection_manifest_path": parent_path,
        "input": {
            "dataset_directory": str(Path(dataset_dir).expanduser().resolve()),
            "parquet_file_count": len(file_audits), "files": file_audits,
            "parquet_audit": parquet_audit,
            "row_count": len(rows), "selected_row_count": len(selected_rows),
        },
        "counts": {
            "input_rows": len(rows), "selected_rows": len(selected_rows),
            "accepted_routes": len(routes), "rejected_routes": rejected_routes,
            "requested_episodes": len(selected_episodes),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "routes": routes,
    }
    encoded = _canonical_json(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return TraceDerivationResult(output, hashlib.sha256(encoded).hexdigest(), artifact)


__all__ = [
    "TRANSFORMATION_REVISION", "GripperMilestoneConfig", "TraceDerivationResult",
    "derive_trace_artifact", "main",
]


def _parse_json_argument(value: str, name: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{name} must be valid JSON") from exc
    return parsed


def _parse_task_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive an action-free Arrow Trace route artifact from LeRobot parquet")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-id", action="append", required=True, dest="episode_ids")
    parser.add_argument("--graph-triplet", required=True, help="JSON array: [source, relation, destination]")
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--parent-manifest", required=True, dest="parent_collection_manifest")
    parser.add_argument("--source-role")
    parser.add_argument("--destination-role")
    parser.add_argument("--source-anchor", help="JSON three-vector override")
    parser.add_argument("--destination-anchor", help="JSON three-vector override")
    parser.add_argument("--close-delta", type=float, default=0.01)
    parser.add_argument("--reopen-delta", type=float, default=0.01)
    parser.add_argument("--min-dwell-frames", type=int, default=1)
    parser.add_argument("--transformation-revision", default=TRANSFORMATION_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        triplet = _parse_json_argument(args.graph_triplet, "--graph-triplet")
        source_anchor = None if args.source_anchor is None else _parse_json_argument(args.source_anchor, "--source-anchor")
        destination_anchor = None if args.destination_anchor is None else _parse_json_argument(args.destination_anchor, "--destination-anchor")
        result = derive_trace_artifact(
            args.dataset_dir, args.output, task_id=_parse_task_id(args.task_id), episode_ids=args.episode_ids,
            graph_triplet=triplet, coordinate_frame=args.coordinate_frame,
            parent_collection_manifest=args.parent_collection_manifest,
            source_role=args.source_role, destination_role=args.destination_role,
            source_anchor=source_anchor, destination_anchor=destination_anchor,
            milestone_config=GripperMilestoneConfig(
                close_delta=args.close_delta, reopen_delta=args.reopen_delta,
                min_dwell_frames=args.min_dwell_frames,
            ), transformation_revision=args.transformation_revision,
        )
    except ContractError as exc:
        parser.error(str(exc))
    receipt = {
        "status": "COMPLETED", "output": str(result.output_path),
        "output_sha256": result.output_sha256,
        "transformation_revision": result.artifact["transformation_revision"],
        "counts": result.artifact["counts"],
        "input": {"parquet_file_count": result.artifact["input"]["parquet_file_count"],
                  "parquet_audit": result.artifact["input"]["parquet_audit"],
                  "row_count": result.artifact["input"]["row_count"],
                  "selected_row_count": result.artifact["input"]["selected_row_count"]},
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
