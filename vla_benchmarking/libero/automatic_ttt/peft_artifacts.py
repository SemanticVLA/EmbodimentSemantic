"""Immutable PEFT adapter artifacts for automatic-TTT experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


class PEFTArtifactError(ValueError):
    pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return tree_sha256(target)
    raise PEFTArtifactError(f"cannot hash missing/non-file checkpoint: {target}")


def tree_sha256(root: str | Path, *, exclude: set[str] | None = None) -> str:
    """Hash relative paths and bytes deterministically; excludes no files by default."""
    base = Path(root)
    if not base.is_dir():
        raise PEFTArtifactError(f"artifact tree is not a directory: {base}")
    excluded = exclude or set()
    digest = hashlib.sha256()
    files = sorted(path for path in base.rglob("*") if path.is_file() and path.relative_to(base).as_posix() not in excluded)
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PEFTArtifactManifest:
    schema_version: int
    artifact_path: str
    vla: str
    task_id: int
    run_id: str
    method: str
    files: Mapping[str, Mapping[str, Any]]
    artifact_tree_sha256: str
    base_checkpoint_path: str
    base_checkpoint_sha256: str
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    seed: int
    train_counts: Mapping[str, Any]
    eval_counts: Mapping[str, Any]
    git_commit: str
    runtime_versions: Mapping[str, Any]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != "peft_lora":
            raise PEFTArtifactError("unsupported PEFT artifact schema or method")
        if not self.vla or self.task_id < 0 or not self.run_id:
            raise PEFTArtifactError("invalid VLA/task/run identity")
        if any(name not in self.files for name in REQUIRED_FILES):
            raise PEFTArtifactError("manifest is missing required adapter files")
        for relative, record in self.files.items():
            candidate = Path(str(relative))
            if candidate.is_absolute() or ".." in candidate.parts or not str(relative):
                raise PEFTArtifactError(f"unsafe artifact file path in manifest: {relative!r}")
            if not isinstance(record, Mapping) or not record.get("sha256"):
                raise PEFTArtifactError(f"invalid digest record for artifact file: {relative!r}")
        for mapping_name, mapping in (("train_counts", self.train_counts), ("eval_counts", self.eval_counts), ("runtime_versions", self.runtime_versions)):
            if not isinstance(mapping, Mapping) or not mapping:
                raise PEFTArtifactError(f"{mapping_name} must be a non-empty mapping")


def _validate_identity(vla: str, task_id: int, run_id: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(vla) or not _SAFE_COMPONENT.fullmatch(run_id):
        raise PEFTArtifactError("vla and run_id must be single safe path components")
    if int(task_id) < 0:
        raise PEFTArtifactError("task_id must be non-negative")


def _resolve_digest_input(value: str | Path | Mapping[str, Any], label: str) -> tuple[str, str]:
    if isinstance(value, Mapping):
        path_value = value.get("path")
        expected = value.get("sha256", value.get("digest"))
    else:
        path_value, expected = value, None
    if not path_value:
        raise PEFTArtifactError(f"{label} path is required")
    path = Path(path_value).expanduser().resolve()
    actual = sha256_path(path)
    if expected is not None and str(expected).lower() != actual.lower():
        raise PEFTArtifactError(f"{label} digest mismatch: expected {expected}, got {actual}")
    return str(path), actual


def save_peft_adapter(
    source_adapter: str | Path,
    output_root: str | Path,
    *,
    vla: str,
    task_id: int,
    run_id: str,
    base_checkpoint: str | Path | Mapping[str, Any],
    dataset_manifest: str | Path | Mapping[str, Any],
    seed: int,
    train_counts: Mapping[str, Any],
    eval_counts: Mapping[str, Any],
    git_commit: str,
    runtime_versions: Mapping[str, Any],
) -> PEFTArtifactManifest:
    """Publish a complete adapter directory exactly once using an atomic rename."""
    _validate_identity(vla, task_id, run_id)
    source = Path(source_adapter).expanduser().resolve()
    if not source.is_dir():
        raise PEFTArtifactError(f"source adapter directory does not exist: {source}")
    source_files = sorted(
        path for path in source.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"
    )
    for required in REQUIRED_FILES:
        candidate = source / required
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise PEFTArtifactError(f"source adapter is missing required non-empty file: {candidate}")
    base_path, base_digest = _resolve_digest_input(base_checkpoint, "base checkpoint")
    dataset_path, dataset_digest = _resolve_digest_input(dataset_manifest, "dataset manifest")
    if not git_commit or not isinstance(runtime_versions, Mapping) or not runtime_versions:
        raise PEFTArtifactError("git_commit and runtime_versions are required provenance")
    if not isinstance(train_counts, Mapping) or not train_counts or not isinstance(eval_counts, Mapping) or not eval_counts:
        raise PEFTArtifactError("train_counts and eval_counts must be non-empty mappings")

    parent = Path(output_root).expanduser().resolve() / vla / f"task_{int(task_id)}" / "peft_adapter"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / run_id
    if final.exists():
        raise FileExistsError(f"refusing to overwrite immutable PEFT artifact: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".partial", dir=parent))
    try:
        for source_file in source_files:
            relative = source_file.relative_to(source)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        files = {
            path.relative_to(temporary).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*")) if path.is_file()
        }
        manifest = PEFTArtifactManifest(
            schema_version=SCHEMA_VERSION,
            artifact_path=str(final),
            vla=vla,
            task_id=int(task_id),
            run_id=run_id,
            method="peft_lora",
            files=files,
            artifact_tree_sha256=tree_sha256(temporary),
            base_checkpoint_path=base_path,
            base_checkpoint_sha256=base_digest,
            dataset_manifest_path=dataset_path,
            dataset_manifest_sha256=dataset_digest,
            seed=int(seed),
            train_counts=dict(train_counts),
            eval_counts=dict(eval_counts),
            git_commit=git_commit,
            runtime_versions=dict(runtime_versions),
        )
        manifest.validate()
        manifest_path = temporary / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(asdict(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # A second atomic manifest is intentionally not included in the payload
        # tree hash, so verification hashes exactly the copied adapter files.
        if final.exists():
            raise FileExistsError(f"immutable PEFT artifact appeared during save: {final}")
        os.rename(temporary, final)
        temporary = Path()
        return manifest
    finally:
        if str(temporary) not in {"", "."} and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def load_peft_manifest(artifact_path: str | Path) -> PEFTArtifactManifest:
    path = Path(artifact_path).expanduser().resolve()
    manifest_path = path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise PEFTArtifactError(f"artifact manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = PEFTArtifactManifest(**payload)
    manifest.validate()
    if Path(manifest.artifact_path).resolve() != path:
        raise PEFTArtifactError("manifest artifact_path does not match requested directory")
    for relative, record in manifest.files.items():
        candidate = path / relative
        if not candidate.is_file() or sha256_file(candidate) != record.get("sha256"):
            raise PEFTArtifactError(f"artifact file digest mismatch: {candidate}")
    if tree_sha256(path, exclude={"artifact_manifest.json"}) != manifest.artifact_tree_sha256:
        raise PEFTArtifactError("artifact tree digest mismatch")
    return manifest


__all__ = [
    "PEFTArtifactError", "PEFTArtifactManifest", "REQUIRED_FILES", "load_peft_manifest",
    "save_peft_adapter", "sha256_file", "sha256_path", "tree_sha256",
]
