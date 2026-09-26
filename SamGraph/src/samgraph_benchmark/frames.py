"""Deterministic readers for the existing LIBERO demo-frame ZIP cache.

The VLM demo cache stores agentview JPEGs after applying a 180 degree display
rotation.  This module reverses that transform before a model sees the image;
it never opens an HDF5 file and therefore cannot leak graph ground truth into
prediction generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
from typing import Iterator
import zipfile

import numpy as np
from PIL import Image


_FRAME_RE = re.compile(r"(?:^|/)(\d+)(?:\.[^.]+)?$")


@dataclass(frozen=True)
class FrameRecord:
    task: str
    demo: str
    frame: int
    rgb: np.ndarray
    source_name: str
    source_sha256: str
    original_size: tuple[int, int]


def _frame_number(name: str) -> int:
    match = _FRAME_RE.search(name.replace("\\", "/"))
    if not match:
        raise ValueError(f"frame archive member is not a numbered image: {name!r}")
    return int(match.group(1))


def ordered_members(archive: zipfile.ZipFile) -> list[str]:
    """Return numbered image members in frame-index order, rejecting ambiguity."""

    members = [item.filename for item in archive.infolist() if not item.is_dir()]
    image_members = [name for name in members if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}]
    numbered = sorted(image_members, key=lambda name: (_frame_number(name), name))
    if not numbered:
        raise ValueError("frame archive contains no numbered JPEG/PNG images")
    numbers = [_frame_number(name) for name in numbered]
    if len(set(numbers)) != len(numbers):
        raise ValueError("frame archive contains duplicate frame numbers")
    expected = list(range(len(numbers)))
    if numbers != expected:
        raise ValueError(f"frame archive has non-contiguous frame numbers: first={numbers[:4]} last={numbers[-4:]}")
    return numbered


def undo_agentview_rotation(rgb: np.ndarray) -> np.ndarray:
    """Undo the cache's 180-degree ``np.rot90(image, 2)`` display transform."""

    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] != value.shape[1]:
        raise ValueError(f"RGB image must be square HxWx3, got {value.shape}")
    return np.ascontiguousarray(np.rot90(value, 2))


def resize_rgb(rgb: np.ndarray, resolution: int | None) -> np.ndarray:
    """Resize RGB to a square SAM input while preserving RGB/channel order."""

    value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] != value.shape[1]:
        raise ValueError(f"RGB image must be square HxWx3, got {value.shape}")
    if resolution is None:
        return value
    if not isinstance(resolution, int) or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    if value.shape[:2] == (resolution, resolution):
        return value
    image = Image.fromarray(value, mode="RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def iter_zip_frames(
    archive_path: str | Path,
    *,
    task: str | None = None,
    demo: str | None = None,
    resolution: int | None = None,
    undo_rotation: bool = True,
) -> Iterator[FrameRecord]:
    """Yield every frame exactly once in numeric order.

    Errors decoding a member are raised rather than silently dropping that
    frame.  The prediction runner catches model failures separately and emits
    an explicit empty prediction for those frames.
    """

    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    task_name = task or path.parent.name
    demo_name = demo or (path.stem if re.fullmatch(r"demo_\d+", path.stem) else "demo_0")
    with zipfile.ZipFile(path, "r") as archive:
        for index, member in enumerate(ordered_members(archive)):
            encoded = archive.read(member)
            with Image.open(io.BytesIO(encoded)) as image:
                original = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
            raw = undo_agentview_rotation(original) if undo_rotation else original
            transformed = resize_rgb(raw, resolution)
            yield FrameRecord(
                task=task_name,
                demo=demo_name,
                frame=index,
                rgb=transformed,
                source_name=member,
                source_sha256=hashlib.sha256(encoded).hexdigest(),
                original_size=(int(original.shape[1]), int(original.shape[0])),
            )


def parse_episode_selection(value: str) -> tuple[int, ...] | None:
    """Select indices, a half-open range, or all available episodes."""
    if value == "all":
        return None
    if re.fullmatch(r"\d+:\d+", value):
        start, stop = map(int, value.split(":"))
        if start < stop:
            return tuple(range(start, stop))
    elif re.fullmatch(r"\d+(,\d+)*", value):
        result = tuple(map(int, value.split(",")))
        if len(set(result)) == len(result):
            return result
    raise ValueError("episodes must be all, an index/list (0,1), or a half-open range (0:50)")


def discover_archives(frames_root: str | Path, *, episodes: tuple[int, ...] | None = (0,)) -> list[Path]:
    """Discover one ``demo_0.zip`` per task without importing demo code."""

    root = Path(frames_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates = [p for p in root.glob("*/demo_*.zip") if re.fullmatch(r"demo_\d+", p.stem)]
    if episodes is not None:
        if not episodes or len(set(episodes)) != len(episodes) or any(type(e) is not int or e < 0 for e in episodes):
            raise ValueError("episodes must contain unique nonnegative integers")
        for task in {p.parent for p in candidates}:
            missing = [e for e in episodes if not (task / f"demo_{e}.zip").is_file()]
            if missing:
                raise FileNotFoundError(f"missing requested episodes for {task.name}: {missing}")
        candidates = [p for p in candidates if int(p.stem[5:]) in episodes]
    paths = sorted(candidates, key=lambda p: (p.parent.name, int(p.stem[5:])))
    if not paths:
        raise FileNotFoundError(f"no requested task/demo_N.zip archives under {root}")
    return paths
