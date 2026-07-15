from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any
from uuid import uuid4


class ArtifactPathError(ValueError):
    pass


class ArtifactStore:
    """The only write surface used by this sub-demo."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str | Path, *, create_parent: bool = True) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactPathError(
                f"Refusing to write outside the SO101 artifact root: {candidate}"
            ) from exc
        if candidate == self.root:
            raise ArtifactPathError("An artifact path must name a file below the artifact root")
        if create_parent:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write_json(self, relative: str | Path, payload: Mapping[str, Any] | list[Any]) -> Path:
        path = self.path(relative)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def write_jsonl(self, relative: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
        path = self.path(relative)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def ensure_layout(self) -> None:
        for name in (
            "index",
            "metadata",
            "bboxes",
            "models",
            "proxy_graphs",
            "reports",
            "audit",
            "cache",
        ):
            self.path(f"{name}/.keep").touch(exist_ok=True)
