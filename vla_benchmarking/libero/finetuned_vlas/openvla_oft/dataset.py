"""Dependency-light serializer for native OpenVLA-OFT RLDS records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import OPENVLA_DATASET_NAME, OPENVLA_IO

REQUIRED_KEYS = (*OPENVLA_IO.camera_keys, "observation.state", "action", "task")
ARROW_MARKERS = ("arrow_overlay", "visual_arrow", "has_arrows", "arrow_mask", "arrows")
NOOP_FILTER_NAME = "pinned_openvla_oft_zero_action_filter"
NOOP_FILTER_THRESHOLD = 1e-5


def _present(marker: Any) -> bool:
    if isinstance(marker, np.ndarray):
        return bool(marker.any())
    if isinstance(marker, (list, tuple, set, dict)):
        return bool(marker)
    return bool(marker)


def _assert_arrow_free(frame: Mapping[str, Any]) -> None:
    if frame.get("arrow_condition") not in (None, "none"):
        raise ValueError("OpenVLA-OFT no-arrow serializer received a non-none arrow_condition")
    for key in ARROW_MARKERS:
        if key in frame and _present(frame[key]):
            raise ValueError(f"OpenVLA-OFT no-arrow serializer received {key}")


def is_noop_action(
    action: Any,
    previous_action: Any | None = None,
    *,
    threshold: float = NOOP_FILTER_THRESHOLD,
) -> bool:
    """Match pinned ``zero_action_filter`` on normalized OpenVLA actions.

    The pinned fork checks only the six relative pose dimensions and keeps the
    seventh gripper dimension in the action contract.  ``previous_action`` is
    retained as a source-compatible, intentionally ignored argument because
    earlier local callers supplied it; no-op status must not depend on episode
    history.
    """
    del previous_action
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("OpenVLA-OFT no-op filtering requires a finite 7-D action")
    return not bool(np.any(np.abs(values[:6]) > float(threshold)))


def filter_noop_transitions(
    frames: Iterable[Mapping[str, Any]], *, threshold: float = NOOP_FILTER_THRESHOLD
) -> list[dict[str, Any]]:
    """Drop no-op transitions and reindex each retained RLDS episode.

    The returned records are explicitly marked as filtered.  Callers that
    materialize ``libero_spatial_no_noops`` must use this function (or an
    equivalent upstream regeneration receipt) before writing the source
    JSONL; merely naming a dataset ``*_no_noops`` does not apply the filter.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for frame in frames:
        episode_id = str(frame.get("episode_id", ""))
        if not episode_id:
            raise ValueError("no-op filtering requires episode_id on every frame")
        grouped.setdefault(episode_id, []).append(frame)
    filtered: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        source = sorted(grouped[episode_id], key=lambda item: int(item.get("step_index", -1)))
        retained: list[Mapping[str, Any]] = []
        for frame in source:
            if is_noop_action(frame["action"], threshold=threshold):
                continue
            retained.append(frame)
        if not retained:
            raise ValueError(f"no-op filtering removed every transition from episode {episode_id!r}")
        episode_length = len(retained)
        for step_index, frame in enumerate(retained):
            updated = dict(frame)
            updated.update(
                episode_id=episode_id,
                step_index=step_index,
                episode_length=episode_length,
                native_noop_filter_applied=True,
            )
            filtered.append(updated)
    return filtered


def _lineage_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array_sha256": _digest_array(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _lineage_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_lineage_value(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _lineage_digest(frames: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    normalized = []
    for frame in frames:
        normalized.append(
            {
                str(key): _lineage_value(value)
                for key, value in frame.items()
                if key != "native_noop_filter_applied"
            }
        )
    normalized.sort(key=lambda item: (str(item.get("episode_id", "")), int(item.get("step_index", -1))))
    for item in normalized:
        payload = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return len(normalized), digest.hexdigest()


def noop_filter_receipt_path(source: str | Path) -> Path:
    """Return the sidecar receipt path for a materialized source JSONL."""

    return Path(source).with_suffix(".noop_filter_receipt.json")


def _verified_noop_receipt(
    receipt: Mapping[str, Any] | str | Path | None,
    *,
    output_frames: int,
) -> dict[str, Any]:
    if receipt is None:
        raise ValueError(
            f"{OPENVLA_DATASET_NAME} manifest requires the persisted pinned no-op filter receipt"
        )
    if isinstance(receipt, (str, Path)):
        path = Path(receipt).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no-op filter receipt does not exist: {path}")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid no-op filter receipt: {path}") from exc
    result = dict(receipt)
    if result.get("dataset_name") != OPENVLA_DATASET_NAME or result.get("filter") != NOOP_FILTER_NAME:
        raise ValueError("manifest receipt does not prove the pinned OpenVLA-OFT no-op filter")
    if float(result.get("threshold", -1)) != NOOP_FILTER_THRESHOLD:
        raise ValueError("manifest receipt has an unexpected no-op threshold")
    if not all(isinstance(result.get(key), str) and len(result[key]) == 64 for key in ("source_sha256", "output_sha256")):
        raise ValueError("manifest receipt must persist source and output SHA-256 values")
    if int(result.get("output_frames", -1)) != output_frames:
        raise ValueError("manifest frame count does not match the verified filter receipt")
    source_frames = int(result.get("source_frames", -1))
    if source_frames < output_frames or int(result.get("dropped_frames", -1)) != source_frames - output_frames:
        raise ValueError("manifest receipt has inconsistent before/after counts")
    return result


def _digest_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _all_fields_digest(frame: Mapping[str, Any], image_digests: Mapping[str, str]) -> str:
    normalized: dict[str, Any] = {}
    for key in sorted(frame):
        value = frame[key]
        if key in OPENVLA_IO.camera_keys:
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


def _image_record(value: Any, key: str) -> dict[str, Any]:
    image = np.asarray(value)
    if image.shape[-3:] not in {(256, 256, 3), (3, 256, 256)}:
        raise ValueError(f"{key} must be 256x256 RGB (HWC or CHW), got {image.shape}")
    return {"shape": list(image.shape), "dtype": str(image.dtype), "sha256": _digest_array(image)}


def serialize_transition(
    frame: Mapping[str, Any], *, episode_id: str, step_index: int, episode_length: int
) -> dict[str, Any]:
    """Create one validated RLDS transition contract with content digests."""
    _assert_arrow_free(frame)
    missing = [key for key in REQUIRED_KEYS if key not in frame]
    if missing:
        raise ValueError(f"OpenVLA-OFT frame is missing required keys: {missing}")
    state = np.asarray(frame["observation.state"], dtype=np.float32)
    action = np.asarray(frame["action"], dtype=np.float32)
    if state.shape != (8,) or action.shape != (7,):
        raise ValueError(f"expected state (8,) and action (7,), got {state.shape} and {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action contains non-finite values")
    if not 0 <= step_index < episode_length or episode_length <= 0:
        raise ValueError("step_index must be inside a non-empty episode")
    image_digests = {
        OPENVLA_IO.camera_keys[0]: _digest_array(frame[OPENVLA_IO.camera_keys[0]]),
        OPENVLA_IO.camera_keys[1]: _digest_array(frame[OPENVLA_IO.camera_keys[1]]),
    }
    return {
        "episode_id": str(episode_id),
        "observation": {
            "image_primary": _image_record(frame[OPENVLA_IO.camera_keys[0]], "image_primary"),
            "image_wrist": _image_record(frame[OPENVLA_IO.camera_keys[1]], "image_wrist"),
            "proprio": state.tolist(),
            "natural_language_instruction": str(frame["task"]),
        },
        "action": action.tolist(),
        "is_first": step_index == 0,
        "is_last": step_index == episode_length - 1,
        "is_terminal": step_index == episode_length - 1,
        "no_arrow_condition": True,
        "native_noop_filter_applied": bool(frame.get("native_noop_filter_applied", False)),
        "fields_sha256": _all_fields_digest(frame, image_digests),
    }


def build_manifest(
    frames: Iterable[Mapping[str, Any]],
    *,
    source_id: str = "canonical_500_demo_source",
    noop_filter_receipt: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest only from frames and a verified filter receipt.

    This strict boundary prevents ``libero_spatial_no_noops`` from becoming a
    naming-only claim: all frames must carry the filter marker and the receipt
    must reconcile its output count and persisted hashes.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot build an RLDS manifest from zero frames")
    if not all(bool(frame.get("native_noop_filter_applied")) for frame in frame_list):
        raise ValueError(
            f"{OPENVLA_DATASET_NAME} manifest requires every frame to carry verified no-op filtering"
        )
    verified_receipt = _verified_noop_receipt(noop_filter_receipt, output_frames=len(frame_list))
    count = 0
    digest = hashlib.sha256()
    episodes: dict[str, set[int]] = {}
    episode_lengths: dict[str, int] = {}
    for frame in frame_list:
        if not {"episode_id", "step_index", "episode_length"}.issubset(frame):
            raise ValueError("RLDS manifest input must include episode_id, step_index, and episode_length")
        episode_id = str(frame["episode_id"])
        step_index = int(frame["step_index"])
        episode_length = int(frame["episode_length"])
        if step_index < 0 or episode_length <= 0 or step_index >= episode_length:
            raise ValueError("episode step metadata is invalid")
        if episode_id in episode_lengths and episode_lengths[episode_id] != episode_length:
            raise ValueError(f"episode_length changed inside episode {episode_id!r}")
        episode_lengths[episode_id] = episode_length
        steps = episodes.setdefault(episode_id, set())
        if step_index in steps:
            raise ValueError(f"duplicate step_index={step_index} in episode {episode_id!r}")
        steps.add(step_index)
        record = serialize_transition(frame, episode_id=episode_id, step_index=step_index, episode_length=episode_length)
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise ValueError("cannot build an RLDS manifest from zero frames")
    for episode_id, steps in episodes.items():
        if sorted(steps) != list(range(episode_lengths[episode_id])):
            raise ValueError(f"episode {episode_id!r} has missing or non-contiguous steps")
    return {
        "schema_version": 1,
        "model": "openvla_oft",
        "format": "rlds",
        "dataset_name": OPENVLA_DATASET_NAME,
        "source_id": source_id,
        "frames": count,
        "episodes": len(episodes),
        "timesteps": count,
        "episode_ids": sorted(episodes),
        "episode_lengths": {key: episode_lengths[key] for key in sorted(episode_lengths)},
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": 8,
        "cameras": list(OPENVLA_IO.camera_keys),
        "arrow_condition": "none",
        "native_noop_filter": "verified_pinned_openvla_oft_zero_action_filter",
        "noop_filter": verified_receipt,
        "content_sha256": digest.hexdigest(),
    }


def write_manifest(manifest: Mapping[str, Any], output: str | Path) -> Path:
    if not manifest.get("content_sha256") or not manifest.get("episodes"):
        raise ValueError("refusing to write an incomplete OpenVLA-OFT manifest")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
