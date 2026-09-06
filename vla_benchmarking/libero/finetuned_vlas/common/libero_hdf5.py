"""Canonical clean LIBERO HDF5 reader for the new VLA integrations.

The existing SmolVLA converter established the stored-frame convention used by
this repository: resize the raw RGB frame to 256x256, then rotate it 180
degrees exactly once. This module exposes that source contract without
importing LeRobot, TensorFlow, Octo, or a simulator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class LiberoFrame:
    """One canonical no-arrow source transition with stable lineage."""

    task_id: int
    task_name: str
    episode_id: str
    frame_id: int
    image_primary: np.ndarray
    image_wrist: np.ndarray
    state: np.ndarray
    action: np.ndarray
    language_instruction: str
    source_path: str
    source_sha256: str

    def as_policy_frame(self) -> dict[str, Any]:
        return {
            "observation.images.image": self.image_primary,
            "observation.images.image2": self.image_wrist,
            "observation.state": self.state,
            "action": self.action,
            "task": self.language_instruction,
            "episode_id": self.episode_id,
            "step_index": self.frame_id,
            "frame_id": self.frame_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "arrow_condition": "none",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flip180(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"LIBERO RGB frame must be HxWx3, got {frame.shape}")
    return np.ascontiguousarray(frame[::-1, ::-1])


def _resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[:2] == (size, size):
        return np.ascontiguousarray(frame.copy())
    try:
        import cv2  # type: ignore

        return np.ascontiguousarray(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    except ImportError:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise RuntimeError("resizing LIBERO frames requires cv2 or Pillow") from exc
        return np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _decode_json_blob(dataset: Any) -> list[Any]:
    raw = dataset[()]
    value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    if not isinstance(value, list):
        raise ValueError("LIBERO HDF5 JSON metadata must contain a list")
    return value


def _task_metadata(task_id: int, hdf5_file: Any) -> tuple[str, str]:
    try:
        from vla_benchmarking.libero.shared.config import TASK_NAMES, TASK_PROMPT_OVERRIDE
        task_name = str(TASK_NAMES[int(task_id)])
        override = TASK_PROMPT_OVERRIDE.get(int(task_id))
    except (ImportError, KeyError):
        task_name, override = str(task_id), None
    problem_info = json.loads(hdf5_file["data"].attrs["problem_info"])
    instruction = str(override or problem_info["language_instruction"])
    if not instruction.strip():
        raise ValueError(f"LIBERO task {task_id} has an empty language instruction")
    return task_name, instruction


def iter_hdf5_frames(
    path: str | Path,
    *,
    task_id: int,
    demo_keys: Sequence[str] | None = None,
    image_size: int = 256,
) -> Iterator[LiberoFrame]:
    """Yield canonical frames from one LIBERO task HDF5 file."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required to read LIBERO HDF5 demonstrations") from exc
    source_sha256 = _sha256_file(target)
    with h5py.File(target, "r") as hdf5_file:
        task_name, instruction = _task_metadata(task_id, hdf5_file)
        data = hdf5_file["data"]
        available = sorted(
            (str(key) for key in data.keys() if str(key).startswith("demo_")),
            key=lambda key: int(key.split("_", 1)[1]),
        )
        selected = list(demo_keys) if demo_keys is not None else available
        for demo_key in selected:
            if demo_key not in data:
                raise KeyError(f"demo {demo_key!r} is missing from {target}")
            demo = data[demo_key]
            obs = demo["obs"]
            agentview, wrist = obs["agentview_rgb"], obs["eye_in_hand_rgb"]
            ee_pos, ee_ori, gripper = obs["ee_pos"], obs["ee_ori"], obs["gripper_states"]
            actions = demo["actions"]
            lengths = (len(agentview), len(wrist), len(actions), len(ee_pos), len(ee_ori), len(gripper))
            if len(set(lengths)) != 1:
                raise ValueError(f"demo {demo_key!r} has inconsistent transition lengths: {lengths}")
            for frame_id in range(lengths[0]):
                main = _flip180(_resize_rgb(np.asarray(agentview[frame_id], dtype=np.uint8), image_size))
                wrist_frame = _flip180(_resize_rgb(np.asarray(wrist[frame_id], dtype=np.uint8), image_size))
                state = np.concatenate([ee_pos[frame_id], ee_ori[frame_id], gripper[frame_id]]).astype(np.float32)
                action = np.asarray(actions[frame_id], dtype=np.float32)
                if state.shape != (8,) or action.shape != (7,):
                    raise ValueError(f"demo {demo_key!r} frame {frame_id} has invalid state/action shape")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"demo {demo_key!r} frame {frame_id} contains non-finite state/action")
                yield LiberoFrame(
                    task_id=int(task_id), task_name=task_name,
                    episode_id=f"task{int(task_id)}-{demo_key}", frame_id=frame_id,
                    image_primary=main, image_wrist=wrist_frame,
                    state=state, action=action, language_instruction=instruction,
                    source_path=str(target), source_sha256=source_sha256,
                )


def iter_source_frames(
    data_dir: str | Path,
    *, task_ids: Iterable[int] = range(10), demos_per_task: int | None = None,
    image_size: int = 256,
) -> Iterator[LiberoFrame]:
    """Yield the canonical source in deterministic task/demo/frame order."""

    root = Path(data_dir).expanduser().resolve()
    try:
        from vla_benchmarking.libero.shared.config import TASK_NAMES
    except ImportError as exc:
        raise RuntimeError("LIBERO task configuration is unavailable") from exc
    for task_id in sorted(int(item) for item in task_ids):
        keys = None if demos_per_task is None else [f"demo_{i}" for i in range(int(demos_per_task))]
        if keys is not None and int(demos_per_task) <= 0:
            raise ValueError("demos_per_task must be positive")
        yield from iter_hdf5_frames(
            root / f"{TASK_NAMES[task_id]}_demo.hdf5",
            task_id=task_id, demo_keys=keys, image_size=image_size,
        )


__all__ = ["LiberoFrame", "iter_hdf5_frames", "iter_source_frames"]
