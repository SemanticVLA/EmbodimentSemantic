from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from ..proxy.schemas import BBoxObject


class VisionDependencyError(RuntimeError):
    pass


class BBoxDetector(Protocol):
    name: str

    def detect(self, image: Image.Image, image_path: Path | None = None) -> dict[str, BBoxObject]:
        ...
