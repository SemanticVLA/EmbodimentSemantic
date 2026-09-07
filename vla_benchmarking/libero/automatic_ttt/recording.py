"""Crash-safe transition recording and experiment provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import os
from typing import Any, Iterable, Mapping

from .contracts import ContractError, SCHEMA_VERSION, TransitionRecord, _json_safe


@dataclass(frozen=True)
class ExperimentProvenance:
    """Identity of every artifact that can affect an experiment result."""

    method_variant: str
    paper_reference: str
    reference_artifact_id: str
    reference_artifact_sha256: str | None
    policy_checkpoint_id: str
    policy_checkpoint_sha256: str | None
    teacher_id: str
    controller_config_id: str
    controller_config_sha256: str | None
    environment_id: str
    environment_config_sha256: str | None
    code_revision: str
    exact_fidelity_status: str

    def __post_init__(self) -> None:
        required = {
            "method_variant": self.method_variant,
            "paper_reference": self.paper_reference,
            "reference_artifact_id": self.reference_artifact_id,
            "policy_checkpoint_id": self.policy_checkpoint_id,
            "teacher_id": self.teacher_id,
            "controller_config_id": self.controller_config_id,
            "environment_id": self.environment_id,
            "code_revision": self.code_revision,
            "exact_fidelity_status": self.exact_fidelity_status,
        }
        if any(not value for value in required.values()):
            raise ContractError("provenance contains an empty required identity")
        for name, digest in (
            ("reference_artifact_sha256", self.reference_artifact_sha256),
            ("policy_checkpoint_sha256", self.policy_checkpoint_sha256),
            ("controller_config_sha256", self.controller_config_sha256),
            ("environment_config_sha256", self.environment_config_sha256),
        ):
            if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
                raise ContractError(f"{name} must be a lowercase or uppercase SHA-256 digest")

    def to_json(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class EpisodeManifest:
    episode_id: str
    task_id: int
    seed: int
    split: str
    task_description: str
    policy_id: str
    provenance: ExperimentProvenance
    observation_keys: tuple[str, ...]
    action_dim: int = 7
    action_range: tuple[float, float] = (-1.0, 1.0)
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_description or not self.policy_id:
            raise ContractError("manifest identity fields are required")
        if self.split not in {"train", "validation", "test"}:
            raise ContractError("manifest split is invalid")
        if self.action_dim != 7 or self.action_range != (-1.0, 1.0):
            raise ContractError("LIBERO automatic TTT requires 7D normalized actions")
        _json_safe(self.metadata)

    def to_json(self) -> dict[str, Any]:
        return _json_safe(self)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JSONLTransitionWriter:
    """Write records to a same-directory partial file and atomically publish it."""

    def __init__(self, path: str | os.PathLike[str], manifest: EpisodeManifest, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        if self.manifest_path.exists() and not overwrite:
            raise FileExistsError(self.manifest_path)
        if self.partial_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"incomplete transition artifact exists at {self.partial_path}; "
                    "choose a new path or pass overwrite=True explicitly"
                )
            self.partial_path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.partial_path.open("w", encoding="utf-8", newline="\n")
        self._closed = False
        self._count = 0
        self._digest = hashlib.sha256()

    @property
    def count(self) -> int:
        return self._count

    def write(self, record: TransitionRecord) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if record.episode_id != self.manifest.episode_id:
            raise ContractError("record belongs to another episode")
        payload = record.to_json()
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        self._handle.write(line)
        self._digest.update(line.encode("utf-8"))
        self._count += 1

    def write_many(self, records: Iterable[TransitionRecord]) -> None:
        for record in records:
            self.write(record)

    def finalize(self, *, status: str, metadata: Mapping[str, Any] | None = None) -> Path:
        if self._closed:
            raise RuntimeError("writer is already closed")
        if not status:
            raise ContractError("final status is required")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self.partial_path, self.path)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.manifest.episode_id,
            "status": status,
            "transition_count": self._count,
            "transitions_sha256": self._digest.hexdigest(),
            "manifest": self.manifest.to_json(),
            "metadata": _json_safe(metadata or {}),
        }
        temporary_manifest = self.manifest_path.with_suffix(self.manifest_path.suffix + ".partial")
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, self.manifest_path)
        self._closed = True
        return self.path

    def abort(self) -> None:
        if self._closed:
            return
        self._handle.close()
        self._closed = True
        # Keep the partial file as evidence of interruption; never overwrite data.

    def __enter__(self) -> "JSONLTransitionWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.abort()


__all__ = ["EpisodeManifest", "ExperimentProvenance", "JSONLTransitionWriter", "sha256_file"]
