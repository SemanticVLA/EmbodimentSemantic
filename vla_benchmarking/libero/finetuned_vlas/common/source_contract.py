"""Canonical source-data identity for matched VLA experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SOURCE_SCHEMA = "vla_dataset_source_contract.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: str | Path) -> str:
    """Hash a file or deterministic relative file inventory for a directory."""

    target = Path(path).expanduser().resolve()
    if target.is_file():
        return hash_file(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    inventory = []
    for child in sorted(item for item in target.rglob("*") if item.is_file()):
        relative = child.relative_to(target).as_posix()
        inventory.append({"path": relative, "sha256": hash_file(child)})
    return hash_json({"root": target.name, "files": inventory})


@dataclass(frozen=True)
class DatasetSourceContract:
    """Immutable identity of the demonstrations before model conversion."""

    source_id: str
    source_revision: str
    episode_count: int
    timestep_count: int
    schema_sha256: str
    source_sha256: str
    split_sha256: str
    arrow_absent: bool = True
    preprocessing_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source_id).strip() or not str(self.source_revision).strip():
            raise ValueError("source_id and source_revision must be non-empty")
        if int(self.episode_count) <= 0 or int(self.timestep_count) <= 0:
            raise ValueError("episode_count and timestep_count must be positive")
        for name in ("schema_sha256", "source_sha256", "split_sha256", "preprocessing_sha256"):
            value = getattr(self, name)
            if value is not None:
                text = str(value)
                if len(text) != 64 or text != text.lower() or any(ch not in "0123456789abcdef" for ch in text):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.arrow_absent is not True:
            raise ValueError("matched no-arrow source contracts must set arrow_absent=true")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_SCHEMA,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "episode_count": int(self.episode_count),
            "timestep_count": int(self.timestep_count),
            "schema_sha256": self.schema_sha256,
            "source_sha256": self.source_sha256,
            "split_sha256": self.split_sha256,
            "arrow_absent": True,
            "preprocessing_sha256": self.preprocessing_sha256,
            "extra": dict(self.extra),
        }

    @property
    def sha256(self) -> str:
        return hash_json(self.as_dict())


__all__ = ["SOURCE_SCHEMA", "DatasetSourceContract", "hash_file", "hash_json", "hash_path"]
