"""Durable, observation-only trace artifacts for one arrow episode.

The trace package is deliberately independent of LIBERO and controller code.
It records RGB-D/proprioceptive observations and bounded metadata; it never
constructs actions, invokes a controller, or reads simulator/evaluator state.
All durable files are write-once.  ``steps.jsonl`` is the sole append-only
file and every write is flushed and fsynced before the call returns.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - import failure is explicit at use time
    np = None  # type: ignore[assignment]


SCHEMA_VERSION = "arrow_episode_trace.v1"
STEPS_FILENAME = "steps.jsonl"
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "artifact_manifest.json"
SNAPSHOT_DIRNAME = "snapshots"
FAILURE_FILENAME = "failure_bundle.json"


class TraceError(RuntimeError):
    """Raised when a trace violates its append-only or integrity contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise TraceError(f"refusing to overwrite trace artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_descriptor(value: Any) -> dict[str, Any]:
    if np is None:
        raise TraceError("numpy is required for RGB-D trace snapshots")
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _safe_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        encoded = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TraceError("trace metadata must be JSON serializable") from exc
    if not isinstance(encoded, dict):
        raise TraceError("trace metadata must be a JSON object")
    return encoded


class ArrowEpisodeTrace:
    """Write-once trace writer with an observation-only callback surface."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        episode_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if self.root.exists():
            raise TraceError(f"trace output already exists; refusing overwrite: {self.root}")
        self.root.mkdir(parents=True, exist_ok=False)
        self.snapshot_root = self.root / SNAPSHOT_DIRNAME
        self.snapshot_root.mkdir()
        self.steps_path = self.root / STEPS_FILENAME
        self.steps_path.touch(exist_ok=False)
        self._steps = self.steps_path.open("a", encoding="utf-8", newline="\n")
        self._next_seq = 0
        self._started_unix = time.time()
        self._finalized = False
        self._episode_id = episode_id
        self._metadata = _safe_metadata(metadata)
        self._snapshot_refs: list[dict[str, Any]] = []

    def _ensure_open(self) -> None:
        if self._finalized or self._steps.closed:
            raise TraceError("trace is already finalized")

    def append_step(
        self,
        event: Mapping[str, Any],
        *,
        seq: int | None = None,
    ) -> dict[str, Any]:
        """Append one JSONL event and return its serialized record.

        A supplied sequence number is checked, never silently normalized;
        duplicate, skipped, or regressed sequence numbers therefore fail at
        the writer boundary as well as during offline validation.
        """
        self._ensure_open()
        if not isinstance(event, Mapping):
            raise TraceError("trace step must be a JSON object")
        expected = self._next_seq
        actual = expected if seq is None else int(seq)
        if actual != expected:
            raise TraceError(f"non-monotonic trace sequence: expected {expected}, got {actual}")
        record = dict(event)
        if "seq" in record and int(record["seq"]) != actual:
            raise TraceError("event seq disagrees with append seq")
        record["seq"] = actual
        try:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TraceError("trace step must be JSON serializable") from exc
        self._steps.write(encoded + "\n")
        self._steps.flush()
        os.fsync(self._steps.fileno())
        self._next_seq += 1
        return record

    def record_observation(
        self,
        rgb: Any,
        depth: Any,
        proprio: Any,
        *,
        phase: str | None = None,
        timestamp_unix: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist RGB-D/proprio and append an observation event.

        This is the only callback intended for a live loop.  It returns an
        immutable reference record and never returns an action or controller
        decision.
        """
        self._ensure_open()
        if np is None:
            raise TraceError("numpy is required for RGB-D trace snapshots")
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth)
        proprio_array = np.asarray(proprio)
        if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
            raise TraceError("RGB observation must have shape HxWx3")
        if depth_array.ndim != 2 or tuple(depth_array.shape) != tuple(rgb_array.shape[:2]):
            raise TraceError("depth observation must be aligned HxW with RGB")
        if proprio_array.ndim == 0:
            raise TraceError("proprioception must be an array-like vector")
        seq = self._next_seq
        relative = Path(SNAPSHOT_DIRNAME) / f"step_{seq:06d}.npz"
        target = self.root / relative
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
        temporary = Path(temporary_name)
        try:
            os.close(fd)
            np.savez_compressed(temporary, rgb=rgb_array, depth=depth_array, proprio=proprio_array)
            # numpy appends .npz when given a path without that suffix.
            generated = temporary if temporary.suffix == ".npz" else Path(f"{temporary}.npz")
            with generated.open("rb") as handle:
                payload = handle.read()
            digest = _sha256_bytes(payload)
            if target.exists():
                raise TraceError(f"refusing to overwrite snapshot: {target}")
            os.replace(generated, target)
        finally:
            if temporary.exists():
                temporary.unlink()
            generated = Path(f"{temporary}.npz")
            if generated.exists():
                generated.unlink()
        reference = {
            "path": relative.as_posix(),
            "sha256": digest,
            "rgb": _array_descriptor(rgb_array),
            "depth": _array_descriptor(depth_array),
            "proprio": _array_descriptor(proprio_array),
        }
        self._snapshot_refs.append(reference)
        event = {
            "kind": "observation",
            "snapshot": reference,
            "phase": phase,
            "timestamp_unix": float(time.time() if timestamp_unix is None else timestamp_unix),
            "metadata": _safe_metadata(metadata),
        }
        self.append_step(event, seq=seq)
        return dict(reference)

    # Explicit callback aliases keep the live integration observation-only.
    observe = record_observation
    on_observation = record_observation

    def record_failure(
        self,
        error: BaseException | str,
        *,
        stage: str | None = None,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        self._ensure_open()
        if isinstance(error, BaseException):
            bundle = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        else:
            bundle = {"type": "TraceFailure", "message": str(error), "traceback": None}
        bundle.update({"stage": stage, "interrupted": bool(interrupted), "last_seq": self._next_seq - 1})
        _atomic_write(self.root / FAILURE_FILENAME, _json_bytes(bundle))
        return bundle

    def finalize(
        self,
        *,
        status: str = "completed",
        summary: Mapping[str, Any] | None = None,
        failure: BaseException | str | None = None,
        stage: str | None = None,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        """Close the append stream and write immutable manifest/summary files."""
        self._ensure_open()
        if status not in {"completed", "failed", "interrupted"}:
            raise TraceError(f"invalid terminal trace status: {status}")
        failure_bundle = None
        if failure is not None:
            failure_bundle = self.record_failure(failure, stage=stage, interrupted=interrupted)
            status = "interrupted" if interrupted else "failed"
        self._steps.flush()
        os.fsync(self._steps.fileno())
        self._steps.close()
        artifacts: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name not in {SUMMARY_FILENAME, MANIFEST_FILENAME}:
                artifacts[path.relative_to(self.root).as_posix()] = _sha256_file(path)
        manifest = {"schema_version": SCHEMA_VERSION, "artifacts": artifacts}
        _atomic_write(self.root / MANIFEST_FILENAME, _json_bytes(manifest))
        manifest_hash = _sha256_file(self.root / MANIFEST_FILENAME)
        artifacts[MANIFEST_FILENAME] = manifest_hash
        finished = time.time()
        result = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self._episode_id,
            "status": status,
            "started_unix": self._started_unix,
            "finished_unix": finished,
            "step_count": self._next_seq,
            "snapshot_count": len(self._snapshot_refs),
            "snapshots": list(self._snapshot_refs),
            "metadata": dict(self._metadata),
            "failure": failure_bundle,
            "artifacts": artifacts,
        }
        result.update(_safe_metadata(summary))
        _atomic_write(self.root / SUMMARY_FILENAME, _json_bytes(result))
        self._finalized = True
        return result

    close = finalize

    def __enter__(self) -> "ArrowEpisodeTrace":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self._finalized:
            if exc is None:
                self.finalize()
            else:
                self.finalize(status="interrupted" if exc_type is KeyboardInterrupt else "failed", failure=exc, interrupted=exc_type is KeyboardInterrupt)
        return False


def validate_trace(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate hashes, JSONL sequencing, snapshot references, and summary."""
    base = Path(root).expanduser().resolve()
    manifest_path = base / MANIFEST_FILENAME
    summary_path = base / SUMMARY_FILENAME
    steps_path = base / STEPS_FILENAME
    if not all(path.is_file() for path in (manifest_path, summary_path, steps_path)):
        raise TraceError("trace is missing manifest, summary, or steps.jsonl")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError("trace manifest or summary is unreadable") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION or summary.get("schema_version") != SCHEMA_VERSION:
        raise TraceError("trace schema version mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TraceError("trace manifest artifacts must be an object")
    for relative, expected in artifacts.items():
        path = base / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise TraceError(f"trace artifact hash mismatch: {relative}")
    try:
        lines = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError("steps.jsonl is unreadable or malformed") from exc
    for expected_seq, record in enumerate(lines):
        if not isinstance(record, Mapping) or record.get("seq") != expected_seq:
            raise TraceError(f"trace sequence is not strictly monotonic at line {expected_seq + 1}")
        snapshot = record.get("snapshot")
        if isinstance(snapshot, Mapping):
            relative = str(snapshot.get("path", ""))
            if relative not in artifacts:
                raise TraceError(f"snapshot is absent from artifact manifest: {relative}")
            if not (base / relative).is_file():
                raise TraceError(f"snapshot reference is missing: {relative}")
    if int(summary.get("step_count", -1)) != len(lines):
        raise TraceError("summary step_count disagrees with steps.jsonl")
    if summary.get("artifacts", {}).get(MANIFEST_FILENAME) != _sha256_file(manifest_path):
        raise TraceError("summary manifest hash disagrees with artifact_manifest.json")
    return {"valid": True, "root": base.as_posix(), "step_count": len(lines), "artifacts": dict(artifacts)}


__all__ = [
    "ArrowEpisodeTrace", "TraceError", "validate_trace", "SCHEMA_VERSION",
    "STEPS_FILENAME", "SUMMARY_FILENAME", "MANIFEST_FILENAME", "FAILURE_FILENAME",
]
