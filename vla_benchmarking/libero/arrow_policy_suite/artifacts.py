"""Create-only, content-addressed experiment artifacts with lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ContractError, _safe


class ArtifactError(ContractError):
    """Raised on artifact mutation, invalid lineage, or digest mismatch."""


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    size: int
    kind: str
    lineage: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path or not self.kind or self.size < 0:
            raise ArtifactError("artifact path, kind, and non-negative size are required")
        if len(self.sha256) != 64 or self.sha256 != self.sha256.lower() or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ArtifactError("artifact sha256 must be lowercase hexadecimal")
        for parent in self.lineage:
            if len(parent) != 64 or parent != parent.lower() or any(c not in "0123456789abcdef" for c in parent):
                raise ArtifactError("lineage entries must be lowercase SHA-256 digests")
        _safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    """Canonical bytes used for deterministic source hashes."""
    return (json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(str(path), flags)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd != -1:
                os.close(fd)
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite immutable artifact: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"could not create artifact {path}") from exc


def write_artifact(
    path: str | os.PathLike[str], data: bytes | bytearray | memoryview,
    *, kind: str = "blob", lineage: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Write bytes exactly once and return a verifiable content reference."""
    payload = bytes(data)
    parents = tuple(str(item) for item in lineage)
    ref = ArtifactRef(str(Path(path)), _digest(payload), len(payload), kind, parents, metadata or {})
    _create_only(Path(path), payload)
    return ref


def write_json_artifact(
    path: str | os.PathLike[str], value: Any, *, kind: str = "json",
    lineage: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    payload = (json.dumps(_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return write_artifact(path, payload, kind=kind, lineage=lineage, metadata=metadata)


def write_jsonl_artifact(
    path: str | os.PathLike[str], records: Iterable[Any], *, kind: str = "jsonl",
    lineage: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Write a deterministic JSONL view exactly once."""
    payload = b"".join(canonical_bytes(record) for record in records)
    return write_artifact(path, payload, kind=kind, lineage=lineage, metadata=metadata)


def verify_artifact(ref: ArtifactRef | Mapping[str, Any]) -> ArtifactRef:
    """Re-hash the referenced path and fail if bytes or size changed."""
    normalized = ref if isinstance(ref, ArtifactRef) else ArtifactRef(**ref)
    target = Path(normalized.path)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"artifact is unreadable: {target}") from exc
    actual = _digest(data)
    if len(data) != normalized.size or actual != normalized.sha256:
        raise ArtifactError(f"artifact digest/size mismatch: {target}")
    return normalized


@dataclass(frozen=True)
class LineageManifest:
    artifacts: tuple[ArtifactRef, ...]
    schema: str = "arrow_policy_suite.artifact_lineage.v1"
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != "arrow_policy_suite.artifact_lineage.v1":
            raise ArtifactError("unsupported lineage manifest schema")
        digests = [item.sha256 for item in self.artifacts]
        if len(digests) != len(set(digests)):
            raise ArtifactError("lineage manifest contains duplicate artifact digests")
        calculated = _digest(json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if self.manifest_sha256 and self.manifest_sha256 != calculated:
            raise ArtifactError("lineage manifest digest does not match contents")
        object.__setattr__(self, "manifest_sha256", calculated)

    def payload(self) -> dict[str, Any]:
        return {"schema": self.schema, "artifacts": [item.to_dict() for item in self.artifacts]}

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}


def write_lineage_manifest(path: str | os.PathLike[str], manifest: LineageManifest) -> ArtifactRef:
    """Persist a lineage manifest once, itself as an immutable artifact."""
    return write_json_artifact(path, manifest.to_dict(), kind="lineage-manifest")


def read_lineage_manifest(path: str | os.PathLike[str]) -> LineageManifest:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read lineage manifest {target}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactError("lineage manifest must be a JSON object")
    try:
        artifacts = tuple(ArtifactRef(**item) for item in value["artifacts"])
        return LineageManifest(artifacts, str(value["schema"]), str(value.get("manifest_sha256", "")))
    except (KeyError, TypeError) as exc:
        raise ArtifactError("malformed lineage manifest") from exc


__all__ = [
    "ArtifactError", "ArtifactRef", "LineageManifest", "canonical_bytes", "write_artifact", "write_json_artifact", "write_jsonl_artifact",
    "verify_artifact", "write_lineage_manifest", "read_lineage_manifest",
]
