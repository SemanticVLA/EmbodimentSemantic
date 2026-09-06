"""Small, dependency-free Pi0.5 frame serializer.

The actual LeRobot writer is deliberately kept at the workflow boundary.  This
module validates and serializes canonical source frames so a conversion job can
be audited without importing LeRobot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import PI05_IO


REQUIRED_KEYS = (*PI05_IO.camera_keys, "observation.state", "action", "task")
ARROW_MARKERS = ("arrow_overlay", "visual_arrow", "has_arrows", "arrow_mask", "arrows")


def _present(marker: Any) -> bool:
    if isinstance(marker, np.ndarray):
        return bool(marker.any())
    if isinstance(marker, (list, tuple, set, dict)):
        return bool(marker)
    return bool(marker)


def _assert_arrow_free(frame: Mapping[str, Any]) -> None:
    if frame.get("arrow_condition") not in (None, "none"):
        raise ValueError("Pi0.5 no-arrow serializer received a non-none arrow_condition")
    for key in ARROW_MARKERS:
        if key in frame and _present(frame[key]):
            raise ValueError(f"Pi0.5 no-arrow serializer received {key}")


def _digest_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _all_fields_digest(frame: Mapping[str, Any], image_digests: Mapping[str, str]) -> str:
    """Hash every source field, including image bytes, deterministically."""
    normalized: dict[str, Any] = {}
    for key in sorted(frame):
        value = frame[key]
        if key in PI05_IO.camera_keys:
            normalized[key] = {"array_sha256": image_digests[key]}
        elif isinstance(value, np.ndarray):
            normalized[key] = {"array_sha256": _digest_array(value)}
        else:
            try:
                json.dumps(value)
                normalized[key] = value
            except (TypeError, ValueError):
                normalized[key] = repr(value)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def serialize_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    _assert_arrow_free(frame)
    missing = [key for key in REQUIRED_KEYS if key not in frame]
    if missing:
        raise ValueError(f"Pi0.5 frame is missing required keys: {missing}")
    images: dict[str, dict[str, Any]] = {}
    for key in PI05_IO.camera_keys:
        image = np.asarray(frame[key])
        if image.ndim != 3 or 3 not in image.shape:
            raise ValueError(f"{key} must be an RGB image, got shape {image.shape}")
        if image.shape[-3:] not in {(256, 256, 3), (3, 256, 256)}:
            raise ValueError(f"{key} must be 256x256 RGB, got {image.shape}")
        images[key] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "sha256": _digest_array(image),
        }
    state = np.asarray(frame["observation.state"], dtype=np.float32)
    action = np.asarray(frame["action"], dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"observation.state must have shape (8,), got {state.shape}")
    if action.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action contains non-finite values")
    return {
        "images": images,
        "state": state.tolist(),
        "action": action.tolist(),
        "task": str(frame["task"]),
        "arrow_condition": "none",
        "episode_id": None if "episode_id" not in frame else str(frame["episode_id"]),
        "step_index": None if "step_index" not in frame else int(frame["step_index"]),
        "fields_sha256": _all_fields_digest(
            frame, {key: value["sha256"] for key, value in images.items()}
        ),
    }


def build_manifest(
    frames: Iterable[Mapping[str, Any]],
    *,
    source_id: str = "canonical_500_demo_source",
) -> dict[str, Any]:
    count = 0
    digest = hashlib.sha256()
    episodes: dict[str, set[int]] = {}
    for frame in frames:
        if "episode_id" not in frame or "step_index" not in frame:
            raise ValueError("Pi0.5 manifest input must include episode_id and step_index for every frame")
        serialized = serialize_frame(frame)
        episode_id = str(frame["episode_id"])
        step_index = int(frame["step_index"])
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        steps = episodes.setdefault(episode_id, set())
        if step_index in steps:
            raise ValueError(f"duplicate step_index={step_index} in episode {episode_id!r}")
        steps.add(step_index)
        payload = json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise ValueError("cannot build a dataset manifest from zero frames")
    for episode_id, steps in episodes.items():
        if sorted(steps) != list(range(len(steps))):
            raise ValueError(f"episode {episode_id!r} has non-contiguous step indices")
    return {
        "schema_version": 1,
        "model": "pi05",
        "source_id": source_id,
        "frames": count,
        "episodes": len(episodes),
        "episode_ids": sorted(episodes),
        "episode_lengths": {key: len(value) for key, value in sorted(episodes.items())},
        "state_dim": 8,
        "action_dim": 7,
        "cameras": list(PI05_IO.camera_keys),
        "arrow_condition": "none",
        "content_sha256": digest.hexdigest(),
    }


def write_manifest(manifest: Mapping[str, Any], output: str | Path) -> Path:
    if not manifest.get("content_sha256") or not manifest.get("episodes"):
        raise ValueError("refusing to write an incomplete Pi0.5 manifest")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_lerobot_dataset(
    frames: Iterable[Mapping[str, Any]],
    root: str | Path,
    *,
    repo_id: str = "local/pi05_libero_no_arrows",
    fps: int = 20,
) -> Path:
    """Write a real LeRobot dataset, or fail closed if its API is unavailable."""
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot write a LeRobot dataset from zero frames")
    canonical_frames = [canonical_frame(frame) for frame in frame_list]
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required to write Pi0.5 datasets; use build_manifest for a dependency-free dry run"
        ) from exc
    features = {
        "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
        "task": {"dtype": "string", "shape": (1,), "names": ["task"]},
    }
    if not hasattr(LeRobotDataset, "create"):
        raise RuntimeError("installed LeRobot lacks LeRobotDataset.create; pin the verified dataset API")
    try:
        dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=Path(root))
        current_episode: str | None = None
        for source_frame, frame in zip(frame_list, canonical_frames):
            episode_id = str(source_frame.get("episode_id", ""))
            if current_episode is None:
                current_episode = episode_id
            elif episode_id != current_episode:
                dataset.save_episode()
                current_episode = episode_id
            dataset.add_frame(frame)
        if current_episode is not None:
            dataset.save_episode()
        dataset.finalize()
    except Exception as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("LeRobot dataset write failed; no partial dataset is considered valid") from exc
    return Path(root)


def canonical_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the validated LeRobot fields, excluding source metadata."""
    serialize_frame(frame)
    canonical: dict[str, Any] = {}
    for key in PI05_IO.camera_keys:
        image = np.asarray(frame[key])
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        canonical[key] = np.ascontiguousarray(image)
    canonical.update(
        {
            "observation.state": np.asarray(frame["observation.state"], dtype=np.float32).copy(),
            "action": np.asarray(frame["action"], dtype=np.float32).copy(),
            "task": str(frame["task"]),
        }
    )
    return canonical
