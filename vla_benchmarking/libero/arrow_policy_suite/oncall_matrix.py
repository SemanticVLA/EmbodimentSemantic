"""Deterministic Arrow On-Call 10-task x 10-episode matrix.

Array workers write disjoint task directories. A create-only plan at the
matrix root seals shared provenance before rollouts; finalization accepts the
matrix only after all task and episode artifacts pass validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .artifacts import write_artifact, write_json_artifact
from .config import StudyConfig
from .contracts import ContractError
from .control_video import inspect_control_video, write_control_video
from .native_executor import _git_revision, _hash_path, execute_native, import_callable

MATRIX_SCHEMA = "arrow_policy_suite.oncall_matrix.v2"
TASK_IDS = tuple(range(10))
EPISODE_INDICES = tuple(range(10))
HORIZON = 280
BASE_SEED = 1000
VIDEO_EPISODE_INDEX = 0
_LOCK_TIMEOUT_S = 120.0


class EpisodeSpec:
    def __init__(self, task_id: int, episode_index: int, seed: int, init_state_index: int,
                 horizon: int = HORIZON, policy_id: str = "arrow_on_call") -> None:
        self.task_id, self.episode_index = task_id, episode_index
        self.seed, self.init_state_index = seed, init_state_index
        self.horizon, self.policy_id = horizon, policy_id

    @property
    def identity(self) -> str:
        return f"task-{self.task_id:02d}-episode-{self.episode_index:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "task_id": self.task_id,
                "episode_index": self.episode_index, "seed": self.seed,
                "init_state_index": self.init_state_index,
                "horizon": self.horizon, "policy_id": self.policy_id}


def matrix_schedule(*, task_ids: Sequence[int] = TASK_IDS,
                    episode_indices: Sequence[int] = EPISODE_INDICES) -> tuple[EpisodeSpec, ...]:
    tasks, episodes = tuple(map(int, task_ids)), tuple(map(int, episode_indices))
    if tasks != TASK_IDS or episodes != EPISODE_INDICES:
        raise ContractError("the exploratory matrix is sealed to tasks 0..9 and episodes 0..9")
    return tuple(EpisodeSpec(task, episode, BASE_SEED + episode, episode)
                 for task in tasks for episode in episodes)


def video_selection() -> tuple[tuple[int, int], ...]:
    return tuple((task, VIDEO_EPISODE_INDEX) for task in TASK_IDS)


def _load_matrix_config(path: str | os.PathLike[str]) -> StudyConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and payload.get("schema") == MATRIX_SCHEMA:
        payload = payload.get("study_config", payload.get("config", {}))
    if not isinstance(payload, Mapping):
        raise ContractError("matrix config must be a JSON object")
    if payload.get("schema") == "arrow_policy_suite.study_config.v1":
        StudyConfig.verify_manifest(payload)
        return StudyConfig.from_manifest(payload)
    kwargs = dict(payload)
    for field in ("task_ids", "learned_seeds", "camera_names", "image_resolution"):
        if field in kwargs:
            kwargs[field] = tuple(kwargs[field])
    for field in ("test_reset_ids", "validation_reset_ids"):
        if field in kwargs:
            kwargs[field] = {int(key): tuple(value) for key, value in dict(kwargs[field]).items()}
    kwargs.pop("schema", None); kwargs.pop("config_sha256", None); kwargs.pop("identity_seal_sha256", None)
    config = StudyConfig(**kwargs); config.validate()
    return config


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"JSON artifact must be an object: {path}")
    return value


def _write_or_verify(path: Path, value: Mapping[str, Any], *, kind: str) -> None:
    if path.exists():
        if dict(_read_json(path)) != dict(value):
            raise ContractError(f"immutable artifact provenance disagreement: {path}")
        return
    try:
        write_json_artifact(path, value, kind=kind)
    except Exception:
        if not path.exists() or dict(_read_json(path)) != dict(value):
            raise


def _write_bytes_or_verify(path: Path, data: bytes, *, kind: str) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ContractError(f"immutable artifact disagreement: {path}")
        return
    try:
        write_artifact(path, data, kind=kind)
    except Exception:
        if not path.exists() or path.read_bytes() != data:
            raise


def _write_json_atomic_create_only(path: Path, value: Mapping[str, Any], *, kind: str) -> None:
    """Publish a complete JSON artifact with one atomic directory entry."""
    if path.exists():
        _write_or_verify(path, value, kind=kind)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
    temporary = path.parent / f".{path.name}.{token}.tmp"
    try:
        write_json_artifact(temporary, value, kind=kind)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _write_or_verify(path, value, kind=kind)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _acquire_lock(path: Path, *, timeout: float = _LOCK_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            path.mkdir(parents=False)
            (path / "owner").write_text(f"{os.getpid()}\n", encoding="utf-8")
            return
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ContractError(f"timed out waiting for matrix lock: {path}")
            time.sleep(0.05)


def _release_lock(path: Path) -> None:
    try:
        (path / "owner").unlink()
    except FileNotFoundError:
        pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def _hash_optional(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "sha256": None}
    target = Path(path)
    return {"path": str(target), "sha256": _hash_path(target)}


def matrix_plan(config: StudyConfig, *, factory_spec: str | None = None,
                checkpoint: str | os.PathLike[str] | None = None,
                controller: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config.validate()
    schedule = matrix_schedule()
    return {"schema": MATRIX_SCHEMA, "experiment_evidence": False,
            "policy_id": "arrow_on_call", "tasks": list(TASK_IDS),
            "episodes_per_task": 10, "episode_count": 100, "horizon": HORIZON,
            "seed_rule": "seed=1000+episode_index",
            "init_state_rule": "init_state_index=episode_index",
            "video_rule": "exactly one video per task: episode_index=0",
            "video_selection": [{"task_id": t, "episode_index": e} for t, e in video_selection()],
            "git_revision": _git_revision(), "factory": factory_spec,
            "checkpoint": _hash_optional(checkpoint), "controller": _hash_optional(controller),
            "config_sha256": config.config_sha256(), "identity_seal_sha256": config.identity_seal_sha256(),
            "schedule": [item.to_dict() for item in schedule]}


def _ensure_plan(root: Path, config: StudyConfig, *, factory_spec: str,
                 checkpoint: str | os.PathLike[str] | None,
                 controller: str | os.PathLike[str] | None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    expected = matrix_plan(config, factory_spec=factory_spec, checkpoint=checkpoint, controller=controller)
    lock = root / ".matrix_plan.lock"
    _acquire_lock(lock)
    try:
        _write_or_verify(root / "matrix_plan.json", expected, kind="arrow-oncall-matrix-plan")
        return expected
    finally:
        _release_lock(lock)


def _set_identity_environment(spec: EpisodeSpec) -> dict[str, str | None]:
    names = ("ARROW_SUITE_TASK_ID", "ARROW_SUITE_SEED", "ARROW_SUITE_INIT_STATE_INDEX")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({"ARROW_SUITE_TASK_ID": str(spec.task_id), "ARROW_SUITE_SEED": str(spec.seed),
                       "ARROW_SUITE_INIT_STATE_INDEX": str(spec.init_state_index)})
    return previous


def _restore_identity_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None: os.environ.pop(name, None)
        else: os.environ[name] = value


def _owner_stats(step_digests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    owners = {owner: 0 for owner in ("vla", "arrow", "hybrid")}; segments = []
    previous = None; start = None; takeover_count = 0
    for item in step_digests:
        owner = str(item.get("executed_by", ""))
        if owner not in owners: continue
        step = int(item.get("timestep", sum(owners.values())))
        owners[owner] += 1
        if owner != previous:
            if previous is not None and start is not None:
                segments.append({"owner": previous, "start_step": start, "end_step": step - 1, "frames": step - start})
            if owner == "arrow" and previous not in (None, "arrow"): takeover_count += 1
            previous, start = owner, step
    if previous is not None and start is not None:
        end = int(step_digests[-1].get("timestep", start)) if step_digests else start
        segments.append({"owner": previous, "start_step": start, "end_step": end, "frames": end - start + 1})
    return {"owner_counts": owners, "segments": segments, "takeover_count": takeover_count,
            "takeover_duration": sum(int(s["frames"]) for s in segments if s["owner"] == "arrow")}


def _receipt_hash(path: Path) -> str:
    return _canonical_digest(_read_json(path))


def _canonical_episode_row(spec: EpisodeSpec, receipt: Mapping[str, Any], receipt_path: Path,
                 *, video: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = dict(receipt.get("manifest", {}) or {})
    return {**spec.to_dict(), "status": str(receipt.get("status", "UNKNOWN")),
            "success": bool(receipt.get("success", False)), "terminal": bool(receipt.get("terminal", False)),
            "steps": int(receipt.get("steps", 0)), "error": receipt.get("error"),
            "receipt_sha256": _receipt_hash(receipt_path),
            **_owner_stats(manifest.get("step_digests", [])),
            "video": None if video is None else dict(video)}


def _episode_row(spec: EpisodeSpec, receipt: Any, receipt_path: Path,
                 *, video: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if hasattr(receipt, "to_dict"):
        value = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        value = {"status": getattr(receipt, "status", "UNKNOWN"),
                 "success": getattr(receipt, "success", False),
                 "terminal": getattr(receipt, "terminal", False),
                 "steps": getattr(receipt, "steps", 0),
                 "error": getattr(receipt, "error", None),
                 "manifest": getattr(receipt, "manifest", {})}
    return _canonical_episode_row(spec, value, receipt_path, video=video)


def _task_summary(task: int, rows: list[dict[str, Any]], skipped: list[int]) -> dict[str, Any]:
    owners = {owner: sum(int(row.get("owner_counts", {}).get(owner, 0)) for row in rows) for owner in ("vla", "arrow", "hybrid")}
    return {"schema": f"{MATRIX_SCHEMA}.task_summary.v1", "experiment_evidence": False, "task_id": task,
            "episodes": rows, "attempted_episodes": len(rows), "skipped_episode_indices": skipped,
            "episode_count": len(rows), "scientific_successes": sum(bool(row.get("success", False)) for row in rows),
            "success_rate": (sum(bool(row.get("success", False)) for row in rows) / len(rows)) if rows else 0.0,
            "owner_counts": owners, "takeover_count": sum(int(row.get("takeover_count", 0)) for row in rows),
            "takeover_duration": sum(int(row.get("takeover_duration", 0)) for row in rows),
            "infrastructure_failures": sum(row.get("status") != "COMPLETED" for row in rows),
            "status": "COMPLETED" if len(rows) == 10 and not any(row.get("status") != "COMPLETED" for row in rows) else "FAILED_INFRASTRUCTURE"}


def run_task(*, config_path: str | os.PathLike[str], output_root: str | os.PathLike[str], task_id: int,
             factory_spec: str, checkpoint: str | os.PathLike[str] | None = None,
             controller: str | os.PathLike[str] | None = None, graph_context_revision: str | None = None,
             trace_geometry_variant: str | None = None, protocol_seal: Any | None = None) -> dict[str, Any]:
    config = _load_matrix_config(config_path); task = int(task_id)
    if task not in TASK_IDS: raise ContractError("task_id must be one of 0..9")
    root = Path(output_root)
    plan = _ensure_plan(root, config, factory_spec=factory_spec, checkpoint=checkpoint, controller=controller)
    task_root = root / "tasks" / f"task_{task:02d}"
    if task_root.exists(): raise ContractError(f"refusing to reuse existing task output: {task_root}")
    task_root.mkdir(parents=True)
    write_json_artifact(task_root / "task_plan.json", {"schema": f"{MATRIX_SCHEMA}.task_plan.v1", "experiment_evidence": False,
        "task_id": task, "matrix_plan_sha256": _canonical_digest(plan),
        "episodes": [item for item in plan["schedule"] if item["task_id"] == task], "video_episode_index": VIDEO_EPISODE_INDEX}, kind="arrow-oncall-task-plan")
    factory = import_callable(factory_spec); rows: list[dict[str, Any]] = []; skipped: list[int] = []
    for spec in matrix_schedule():
        if spec.task_id != task: continue
        episode_root = task_root / "episodes" / f"episode_{spec.episode_index:02d}"; episode_root.mkdir(parents=True)
        run_dir = episode_root / "native_run"; receipt_path = episode_root / "execution_receipt.json"; video_metadata = None
        previous_env = _set_identity_environment(spec)
        try:
            def consume(records: tuple[Any, ...], receipt: Any) -> None:
                nonlocal video_metadata
                if spec.episode_index != VIDEO_EPISODE_INDEX: return
                video_path = root / "videos" / f"task_{task:02d}_episode_00.mp4"
                video_metadata = write_control_video(records, video_path, task_id=task, episode_index=0, receipt=receipt)
                video_metadata = {**video_metadata, "path": str(video_path)}
            receipt = execute_native(factory, config=config, protocol_seal=protocol_seal, operation="evaluate",
                policy_id=spec.policy_id, run_dir=run_dir, output=receipt_path, max_steps=spec.horizon,
                graph_context_revision=graph_context_revision, trace_geometry_variant=trace_geometry_variant,
                checkpoint=checkpoint, controller=controller, records_consumer=consume)
        finally: _restore_identity_environment(previous_env)
        row = _episode_row(spec, receipt, receipt_path, video=video_metadata); rows.append(row)
        write_json_artifact(episode_root / "episode_result.json", row, kind="arrow-oncall-episode-result")
        if spec.episode_index == VIDEO_EPISODE_INDEX and (row["status"] != "COMPLETED" or video_metadata is None):
            skipped = list(range(1, 10)); break
    summary = _task_summary(task, rows, skipped)
    write_json_artifact(task_root / "task_summary.json", summary, kind="arrow-oncall-task-summary")
    if summary["status"] == "COMPLETED": write_artifact(task_root / "COMPLETED", b"completed\n", kind="arrow-oncall-task-complete")
    _write_json_atomic_create_only(root / "worker_status" / f"task_{task:02d}.json", {
        "schema": "arrow_policy_suite.oncall_worker_terminal.v1",
        "experiment_evidence": False, "task_id": task,
        "task_status": summary["status"],
        "task_exit_code": 0 if summary["status"] == "COMPLETED" else 2,
        "workload_exit_code": 0 if summary["status"] == "COMPLETED" else 2,
    }, kind="arrow-oncall-worker-terminal")
    return summary


def _validate_receipt(path: Path, expected: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
    receipt = _read_json(path); failures: list[str] = []
    if receipt.get("status") != "COMPLETED": failures.append(f"receipt not completed: {path}")
    if receipt.get("experiment_evidence") is not False: failures.append(f"receipt experiment_evidence is not false: {path}")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, Mapping): return receipt, failures + [f"receipt manifest missing: {path}"]
    checks = {"task_id": expected["task_id"], "seed": expected["seed"], "init_state_index": expected["init_state_index"],
              "policy_id": "arrow_on_call", "max_steps": HORIZON, "config_sha256": plan["config_sha256"],
              "identity_seal_sha256": plan["identity_seal_sha256"], "git_revision": plan["git_revision"],
              "checkpoint_sha256": plan["checkpoint"]["sha256"], "controller_sha256": plan["controller"]["sha256"],
              "experiment_evidence": False}
    for key, value in checks.items():
        if manifest.get(key) != value: failures.append(f"receipt {path} {key} mismatch")
    for key in ("policy_id", "steps", "success", "terminal"):
        if receipt.get(key) != manifest.get(key):
            failures.append(f"receipt {path} top-level {key} mismatch")
    if receipt.get("policy_id") != "arrow_on_call": failures.append(f"receipt {path} policy mismatch")
    if receipt.get("operation") != "evaluate": failures.append(f"receipt {path} operation mismatch")
    return receipt, failures


def _publish_archive(root: Path, archive_root: Path, sources: Sequence[Path]) -> dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True); published = []
    for source in sources:
        destination = archive_root / source.relative_to(root); destination.parent.mkdir(parents=True, exist_ok=True); data = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != data: raise ContractError(f"archive artifact disagreement: {destination}")
        else: write_artifact(destination, data, kind="arrow-oncall-archive-artifact")
        published.append({"path": destination.relative_to(archive_root).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    counts = {
        "execution_receipts": sum(item["path"].startswith("tasks/") and item["path"].endswith("/execution_receipt.json") for item in published),
        "episode_results": sum(item["path"].startswith("tasks/") and item["path"].endswith("/episode_result.json") for item in published),
        "task_summaries": sum(item["path"].startswith("tasks/") and item["path"].endswith("/task_summary.json") for item in published),
        "worker_terminal_receipts": sum(item["path"].startswith("worker_status/") and item["path"].endswith(".json") for item in published),
        "videos": sum(item["path"].startswith("videos/") and item["path"].endswith(".mp4") for item in published),
    }
    required_counts = {"execution_receipts": 100, "episode_results": 100,
                       "task_summaries": 10, "worker_terminal_receipts": 10, "videos": 10}
    if counts != required_counts:
        raise ContractError(f"global archive required-count mismatch: {counts}")
    inventory_sha256 = _canonical_digest(published)
    marker = archive_root / "COMPLETED"
    marker_data = b"completed\n"
    _write_bytes_or_verify(marker, marker_data, kind="arrow-oncall-archive-complete")
    published.append({"path": "COMPLETED", "sha256": hashlib.sha256(marker_data).hexdigest(), "bytes": len(marker_data)})
    status = {"schema": f"{MATRIX_SCHEMA}.archive_status.v1", "experiment_evidence": False,
              "status": "VERIFIED", "artifacts": published, "required_counts": counts,
              "inventory_sha256": inventory_sha256}
    _write_or_verify(archive_root / "matrix_status.json", status, kind="arrow-oncall-archive-status")
    return status


def _archive_sources(root: Path) -> list[Path]:
    sources = [root / "matrix_plan.json", root / "summary.json", root / "summary.csv"]
    for directory in (root / "tasks", root / "worker_status"):
        if directory.is_dir():
            sources.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    for task in TASK_IDS:
        sources.extend((root / "videos" / f"task_{task:02d}_episode_00.mp4",
                        root / "videos" / f"task_{task:02d}_episode_00.mp4.json"))
    return sources


def _worker_terminal_statuses(root: Path) -> tuple[dict[int, Mapping[str, Any]], list[str], int]:
    """Read terminal worker receipts without treating absent workers as failures."""
    statuses: dict[int, Mapping[str, Any]] = {}
    failures: list[str] = []
    expected_paths = {root / "worker_status" / f"task_{task:02d}.json" for task in TASK_IDS}
    actual_paths = set((root / "worker_status").glob("task_*.json")) if (root / "worker_status").is_dir() else set()
    for path in sorted(actual_paths):
        try:
            value = _read_json(path)
        except ContractError as exc:
            failures.append(str(exc)); continue
        task = value.get("task_id")
        if (value.get("schema") != "arrow_policy_suite.oncall_worker_terminal.v1"
                or value.get("experiment_evidence") is not False
                or isinstance(task, bool) or not isinstance(task, int) or task not in TASK_IDS):
            failures.append(f"invalid worker terminal receipt: {path}"); continue
        if task in statuses:
            failures.append(f"duplicate worker terminal receipt: task={task}"); continue
        if value.get("task_status") not in {"COMPLETED", "FAILED_INFRASTRUCTURE"}:
            failures.append(f"invalid worker task status: task={task}")
        for field in ("task_exit_code", "workload_exit_code"):
            if isinstance(value.get(field), bool) or not isinstance(value.get(field), int):
                failures.append(f"invalid worker exit code: task={task} field={field}")
        statuses[task] = value
    if len(statuses) == len(TASK_IDS) and actual_paths != expected_paths:
        failures.append("worker terminal receipt set mismatch")
    return statuses, failures, len(actual_paths)


def _persist_finalization_failure(root: Path, archive_root: str | os.PathLike[str] | None,
                                  failures: Sequence[str]) -> None:
    _write_or_verify(root / "finalization_failure.json",
                     {"schema": f"{MATRIX_SCHEMA}.failure.v1", "failures": list(failures)},
                     kind="arrow-oncall-finalization-failure")
    _write_bytes_or_verify(root / "FAILED", b"failed_infrastructure\n",
                           kind="arrow-oncall-matrix-failed")
    if archive_root is not None:
        _publish_archive_failure(Path(archive_root), failures)


def _publish_archive_failure(archive_root: Path, failures: Sequence[str]) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    failure = {"schema": f"{MATRIX_SCHEMA}.archive_failure.v1",
               "experiment_evidence": False, "status": "FAILED_INFRASTRUCTURE",
               "failures": list(failures)}
    _write_or_verify(archive_root / "finalization_failure.json", failure,
                     kind="arrow-oncall-archive-finalization-failure")
    status = {"schema": f"{MATRIX_SCHEMA}.archive_status.v1",
              "experiment_evidence": False, "status": "FAILED_INFRASTRUCTURE",
              "failure_artifact": "finalization_failure.json",
              "failures": list(failures)}
    _write_or_verify(archive_root / "matrix_status.json", status,
                     kind="arrow-oncall-archive-status")


def finalize(*, config_path: str | os.PathLike[str], output_root: str | os.PathLike[str], archive_root: str | os.PathLike[str] | None = None,
             factory_spec: str | None = None, checkpoint: str | os.PathLike[str] | None = None, controller: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config = _load_matrix_config(config_path); root = Path(output_root)
    if not root.is_dir(): raise ContractError(f"matrix output root is missing: {root}")
    plan_path = root / "matrix_plan.json"
    if not plan_path.is_file():
        if factory_spec is None:
            # A finalizer may be invoked while the array has not yet created
            # its first worker artifact. Treat that as an ordinary pending
            # state; do not seal a permanent failure for an absent plan.
            worker_statuses, worker_failures, worker_terminal_count = _worker_terminal_statuses(root)
            if worker_terminal_count < len(TASK_IDS):
                return {"schema": f"{MATRIX_SCHEMA}.pending.v1", "status": "PENDING",
                        "experiment_evidence": False, "task_summaries": 0,
                        "worker_terminal_receipts": worker_terminal_count,
                        "missing_matrix_plan": True}
            lock = root / ".finalize.lock"; _acquire_lock(lock)
            try:
                failures = ["missing matrix plan", *worker_failures]
                for task, status in sorted(worker_statuses.items()):
                    if status.get("task_status") != "COMPLETED": failures.append(f"worker task failed: {task}")
                _persist_finalization_failure(root, archive_root, failures)
            finally:
                _release_lock(lock)
            raise ContractError("matrix finalization failed: " + "; ".join(failures[:5]))
        plan = _ensure_plan(root, config, factory_spec=factory_spec, checkpoint=checkpoint, controller=controller)
    else:
        plan = dict(_read_json(plan_path)); expected = matrix_plan(
            config,
            factory_spec=factory_spec if factory_spec is not None else plan.get("factory"),
            checkpoint=checkpoint if checkpoint is not None else (plan.get("checkpoint") or {}).get("path"),
            controller=controller if controller is not None else (plan.get("controller") or {}).get("path"),
        )
        if plan != expected: raise ContractError("matrix_plan provenance disagreement")
    completed = root / "COMPLETED"
    if completed.exists():
        summary = dict(_read_json(root / "summary.json"))
        if summary.get("status") != "COMPLETED":
            raise ContractError("COMPLETED marker has a non-completed summary")
        if archive_root is not None:
            lock = root / ".finalize.lock"
            _acquire_lock(lock)
            try:
                # Crash recovery: the run marker may have been sealed after a
                # prior process published the summary but before it finished
                # archive publication. Re-run the create-only publication.
                _publish_archive(root, Path(archive_root), _archive_sources(root))
            finally:
                _release_lock(lock)
        return summary
    summary_paths = [root / "tasks" / f"task_{task:02d}" / "task_summary.json" for task in TASK_IDS]
    missing = [path for path in summary_paths if not path.is_file()]
    worker_statuses, worker_failures, worker_terminal_count = _worker_terminal_statuses(root)
    if worker_terminal_count < len(TASK_IDS):
        return {"schema": f"{MATRIX_SCHEMA}.pending.v1", "status": "PENDING", "experiment_evidence": False,
                "task_summaries": 10 - len(missing), "worker_terminal_receipts": worker_terminal_count,
                "missing_task_summaries": [str(path) for path in missing]}
    if (missing or worker_failures or len(worker_statuses) != len(TASK_IDS)
            or any(status.get("task_status") != "COMPLETED" for status in worker_statuses.values())):
        lock = root / ".finalize.lock"; _acquire_lock(lock)
        try:
            failures = list(worker_failures) + [f"missing task summary: {path}" for path in missing]
            for task, status in sorted(worker_statuses.items()):
                if status.get("task_status") != "COMPLETED":
                    failures.append(f"worker task failed: {task}")
            if len(worker_statuses) != len(TASK_IDS):
                failures.append("worker terminal receipt set is incomplete or malformed")
            _persist_finalization_failure(root, archive_root, failures)
        finally:
            _release_lock(lock)
        raise ContractError("matrix finalization failed: " + "; ".join(failures[:5]))
    lock = root / ".finalize.lock"; _acquire_lock(lock)
    try:
        if completed.exists(): return dict(_read_json(root / "summary.json"))
        failures: list[str] = list(worker_failures); rows: list[dict[str, Any]] = []; expected_schedule = {(x.task_id, x.episode_index): x.to_dict() for x in matrix_schedule()}
        if len(worker_statuses) == len(TASK_IDS):
            for task, status in sorted(worker_statuses.items()):
                if status.get("task_status") != "COMPLETED": failures.append(f"worker task failed: {task}")
        for task, summary_path in zip(TASK_IDS, summary_paths):
            task_summary = _read_json(summary_path)
            if int(task_summary.get("task_id", -1)) != task or task_summary.get("status") != "COMPLETED": failures.append(f"task summary failed: {task}")
            for episode in EPISODE_INDICES:
                episode_root = root / "tasks" / f"task_{task:02d}" / "episodes" / f"episode_{episode:02d}"; receipt_path = episode_root / "execution_receipt.json"; result_path = episode_root / "episode_result.json"
                if not receipt_path.is_file() or not result_path.is_file(): failures.append(f"missing episode artifact: task={task} episode={episode}"); continue
                receipt, receipt_failures = _validate_receipt(receipt_path, expected_schedule[(task, episode)], plan); failures.extend(receipt_failures)
                video_metadata = None
                if episode == VIDEO_EPISODE_INDEX:
                    video_path = root / "videos" / f"task_{task:02d}_episode_00.mp4"; sidecar_path = Path(str(video_path) + ".json")
                    if video_path.is_file() and sidecar_path.is_file():
                        sidecar = dict(_read_json(sidecar_path)); video_metadata = {**sidecar, "path": str(video_path)}
                        if sidecar.get("task_id") != task or sidecar.get("episode_index") != 0: failures.append(f"video identity mismatch: task={task}")
                        if sidecar.get("format") != "H264/yuv420p": failures.append(f"video format mismatch: task={task}")
                        if sidecar.get("source_receipt_sha256") != _receipt_hash(receipt_path): failures.append(f"video receipt digest mismatch: task={task}")
                        if sidecar.get("video_sha256") != hashlib.sha256(video_path.read_bytes()).hexdigest(): failures.append(f"video digest mismatch: task={task}")
                result = dict(_read_json(result_path))
                spec = EpisodeSpec(task, episode, int(expected_schedule[(task, episode)]["seed"]), int(expected_schedule[(task, episode)]["init_state_index"]))
                canonical = _canonical_episode_row(spec, receipt, receipt_path, video=video_metadata)
                if result != canonical: failures.append(f"episode result disagrees with receipt-derived row: {task}/{episode}")
                rows.append(canonical)
        expected_receipts = {root / "tasks" / f"task_{task:02d}" / "episodes" / f"episode_{episode:02d}" / "execution_receipt.json" for task in TASK_IDS for episode in EPISODE_INDICES}
        if set(root.glob("tasks/task_*/episodes/episode_*/execution_receipt.json")) != expected_receipts: failures.append("receipt set mismatch")
        expected_videos = {root / "videos" / f"task_{task:02d}_episode_00.mp4" for task in TASK_IDS}
        if not (root / "videos").is_dir() or set((root / "videos").glob("task_*.mp4")) != expected_videos: failures.append("video set mismatch")
        for task in TASK_IDS:
            video = root / "videos" / f"task_{task:02d}_episode_00.mp4"; sidecar = Path(str(video) + ".json")
            if not video.is_file() or not sidecar.is_file(): failures.append(f"missing video/sidecar: task={task}"); continue
            metadata = _read_json(sidecar); receipt_path = root / "tasks" / f"task_{task:02d}" / "episodes" / "episode_00" / "execution_receipt.json"
            if metadata.get("task_id") != task or metadata.get("episode_index") != 0: failures.append(f"video identity mismatch: task={task}")
            if metadata.get("format") != "H264/yuv420p": failures.append(f"video format mismatch: task={task}")
            if metadata.get("source_receipt_sha256") != _receipt_hash(receipt_path): failures.append(f"video receipt digest mismatch: task={task}")
            if metadata.get("video_sha256") != hashlib.sha256(video.read_bytes()).hexdigest(): failures.append(f"video digest mismatch: task={task}")
            try:
                inspect_control_video(video, metadata, expected_steps=int(_read_json(receipt_path).get("steps", 0)))
            except Exception as exc:
                failures.append(f"video decode/metadata mismatch: task={task}: {type(exc).__name__}: {exc}")
        rows.sort(key=lambda row: (int(row.get("task_id", -1)), int(row.get("episode_index", -1)))); owners = {owner: sum(int(row.get("owner_counts", {}).get(owner, 0)) for row in rows) for owner in ("vla", "arrow", "hybrid")}
        per_task = []
        for task in TASK_IDS:
            selected = [row for row in rows if int(row.get("task_id", -1)) == task]; per_task.append({"task_id": task, "successes": sum(bool(row.get("success", False)) for row in selected), "episodes": len(selected), "owner_counts": {owner: sum(int(row.get("owner_counts", {}).get(owner, 0)) for row in selected) for owner in ("vla", "arrow", "hybrid")}, "takeover_count": sum(int(row.get("takeover_count", 0)) for row in selected), "takeover_duration": sum(int(row.get("takeover_duration", 0)) for row in selected)})
        summary = {"schema": f"{MATRIX_SCHEMA}.summary.v1", "experiment_evidence": False, "status": "COMPLETED" if not failures else "FAILED_INFRASTRUCTURE", "config_sha256": plan["config_sha256"], "identity_seal_sha256": plan["identity_seal_sha256"], "git_revision": plan["git_revision"], "policy_id": "arrow_on_call", "horizon": HORIZON, "tasks": list(TASK_IDS), "episode_count": len(rows), "scientific_successes": sum(bool(row.get("success", False)) for row in rows), "scientific_success_rate": (sum(bool(row.get("success", False)) for row in rows) / 100.0) if len(rows) == 100 else None, "owner_counts": owners, "takeover_count": sum(int(row.get("takeover_count", 0)) for row in rows), "takeover_duration": sum(int(row.get("takeover_duration", 0)) for row in rows), "per_task": per_task, "episodes": rows, "infrastructure_failures": failures}
        if failures:
            _persist_finalization_failure(root, archive_root, failures)
            raise ContractError("matrix finalization failed: " + "; ".join(failures[:5]))
        _write_or_verify(root / "summary.json", summary, kind="arrow-oncall-summary")
        csv = ["task_id,episode_index,seed,init_state_index,status,success,steps,vla_steps,arrow_steps,hybrid_steps"]
        for row in rows:
            owners_row = row.get("owner_counts", {}); csv.append(",".join(str(row.get(k, "")) for k in ("task_id", "episode_index", "seed", "init_state_index", "status", "success", "steps")) + "," + ",".join(str(owners_row.get(owner, 0)) for owner in ("vla", "arrow", "hybrid")))
        _write_bytes_or_verify(root / "summary.csv", ("\n".join(csv) + "\n").encode(), kind="arrow-oncall-summary-csv")
        if archive_root is not None:
            # Archive completion is sealed before the run-root completion
            # marker. A crash now leaves a safely repairable completed archive.
            _publish_archive(root, Path(archive_root), _archive_sources(root))
        _write_bytes_or_verify(completed, b"completed\n", kind="arrow-oncall-matrix-complete")
        return summary
    finally: _release_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arrow On-Call 10x10 exploratory matrix"); sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-task"); run.add_argument("--config", required=True); run.add_argument("--output-root", required=True); run.add_argument("--task-id", type=int, required=True); run.add_argument("--factory", required=True); run.add_argument("--checkpoint"); run.add_argument("--controller"); run.add_argument("--graph-context-revision"); run.add_argument("--trace-geometry-variant")
    fin = sub.add_parser("finalize"); fin.add_argument("--config", required=True); fin.add_argument("--output-root", required=True); fin.add_argument("--archive-root"); fin.add_argument("--factory"); fin.add_argument("--checkpoint"); fin.add_argument("--controller")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-task":
            result = run_task(config_path=args.config, output_root=args.output_root, task_id=args.task_id, factory_spec=args.factory, checkpoint=args.checkpoint, controller=args.controller, graph_context_revision=args.graph_context_revision, trace_geometry_variant=args.trace_geometry_variant)
            return 0 if result.get("status") == "COMPLETED" else 2
        result = finalize(config_path=args.config, output_root=args.output_root, archive_root=args.archive_root, factory_spec=args.factory, checkpoint=args.checkpoint, controller=args.controller)
        # PENDING is an expected non-terminal state while array workers are
        # still producing task summaries and must not fail the dependency.
        return 0 if result.get("status") in {"COMPLETED", "PENDING"} else 2
    except Exception as exc:
        print(f"oncall matrix: {type(exc).__name__}: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["EpisodeSpec", "matrix_schedule", "matrix_plan", "video_selection", "run_task", "finalize", "main"]
