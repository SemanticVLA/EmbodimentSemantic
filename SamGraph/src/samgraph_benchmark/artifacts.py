"""Constrained artifact paths for reproducible benchmark outputs."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path = Path(__file__).parents[2] / "artifacts") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"artifact path escapes store: {relative}")
        return candidate

    def write_json(self, relative: str | Path, value: Any) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
