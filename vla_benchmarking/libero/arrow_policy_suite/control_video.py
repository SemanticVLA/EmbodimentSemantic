"""Annotated ownership videos for native Arrow/SmolVLA rollouts.

The video is deliberately a post-rollout artifact.  ``NativeStep`` already
retains the exact observation and the owner of each environment transition,
so recording frames here cannot add a second environment step or perturb the
policy.  Every frame carries a large, unambiguous owner banner:
blue is VLA control and orange is Arrow control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import write_artifact, write_json_artifact


@dataclass(frozen=True)
class OwnershipSegment:
    owner: str
    start_step: int
    end_step: int

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "start_step": self.start_step, "end_step": self.end_step,
                "frames": self.end_step - self.start_step + 1}


def _receipt_digest(receipt: Any) -> str:
    if hasattr(receipt, "to_dict"):
        value = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        raise TypeError("receipt must expose to_dict() or be a mapping")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_rgb(image: Any):
    """Convert common RGB payloads to a uint8 HWC NumPy array."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("control videos require numpy") from exc

    if image is None:
        raise ValueError("agentview image is missing")
    if hasattr(image, "detach") and callable(image.detach):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "convert") and callable(image.convert):
        image = np.asarray(image.convert("RGB"))
    elif isinstance(image, (bytes, bytearray, memoryview)):
        from PIL import Image
        import io
        image = np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"))
    else:
        image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError(f"agentview image must be HWC/CHW, got shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"agentview image must have three channels, got shape {image.shape}")
    if image.dtype.kind == "f":
        maximum = float(np.nanmax(image)) if image.size else 1.0
        if maximum <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _frame_image(step: Any):
    frame = getattr(step, "frame", step)
    observation = getattr(frame, "observation", None)
    if not isinstance(observation, Mapping):
        observation = getattr(frame, "student_observation", None)
    if not isinstance(observation, Mapping):
        raise ValueError("NativeStep frame has no observation mapping")
    image = observation.get("agentview")
    if image is None:
        image = observation.get("observation.images.image")
    return _to_rgb(image)


def _owner(step: Any) -> str:
    owner = str(getattr(step, "executed_by", "")).lower()
    if owner not in {"vla", "arrow", "hybrid"}:
        raise ValueError(f"unsupported ownership label {owner!r}")
    return owner


def ownership_segments(records: Sequence[Any]) -> tuple[OwnershipSegment, ...]:
    if not records:
        return ()
    segments: list[OwnershipSegment] = []
    start = int(getattr(getattr(records[0], "frame", records[0]), "timestep", 0))
    prior = _owner(records[0])
    for item in records[1:]:
        step = int(getattr(getattr(item, "frame", item), "timestep", start + 1))
        current = _owner(item)
        if current != prior:
            segments.append(OwnershipSegment(prior, start, step - 1))
            start, prior = step, current
    last = int(getattr(getattr(records[-1], "frame", records[-1]), "timestep", start))
    segments.append(OwnershipSegment(prior, start, last))
    return tuple(segments)


def _transition(previous: str | None, current: str) -> str | None:
    if previous is None or previous == current:
        return None
    if current == "arrow" and previous != "arrow":
        return "TAKEOVER: ARROW"
    if previous == "arrow" and current == "vla":
        return "HANDBACK: VLA"
    return f"CONTROL: {current.upper()}"


def _annotate(image: Any, *, owner: str, task_id: int, episode_index: int, step: int,
              transition: str | None, success: bool, terminal: bool):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    image = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # Keep the default font dependency-free and make the banner tall enough
    # to be readable in a small browser preview.
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, width // 32))
        small = ImageFont.truetype("DejaVuSans.ttf", max(10, width // 48))
    except OSError:  # pragma: no cover - platform font variation
        font = ImageFont.load_default()
        small = font
    palette = {"vla": (24, 91, 220), "arrow": (230, 116, 20), "hybrid": (126, 62, 180)}
    label = {"vla": "VLA CONTROL", "arrow": "ARROW TAKEOVER", "hybrid": "HYBRID CONTROL"}[owner]
    draw.rectangle((0, 0, width, max(34, height // 9)), fill=palette[owner])
    draw.text((12, 7), label, fill=(255, 255, 255), font=font)
    line_y = max(36, height // 9) + 5
    meta = f"task={task_id:02d}  episode={episode_index:02d}  step={step:03d}"
    draw.rectangle((0, line_y, width, line_y + max(24, height // 15)), fill=(0, 0, 0))
    draw.text((12, line_y + 4), meta, fill=(255, 255, 255), font=small)
    if transition:
        box_top = line_y + max(30, height // 15) + 4
        bbox = draw.textbbox((0, 0), transition, font=font)
        box_width = min(width, bbox[2] - bbox[0] + 28)
        draw.rectangle((0, box_top, box_width, box_top + max(32, height // 10)), fill=(0, 0, 0))
        draw.text((12, box_top + 7), transition, fill=(255, 240, 80), font=font)
    legend_h = max(24, height // 14)
    y0 = height - legend_h
    draw.rectangle((0, y0, width, height), fill=(0, 0, 0))
    draw.rectangle((12, y0 + 7, 28, y0 + legend_h - 7), fill=palette["vla"])
    draw.text((36, y0 + 5), "VLA", fill=(255, 255, 255), font=small)
    offset = 100 if width >= 240 else width // 3
    draw.rectangle((offset, y0 + 7, offset + 16, y0 + legend_h - 7), fill=palette["arrow"])
    draw.text((offset + 24, y0 + 5), "ARROW", fill=(255, 255, 255), font=small)
    if success or terminal:
        status = "SUCCESS" if success else "TERMINAL"
        draw.text((max(0, width - 115), y0 + 5), status, fill=(100, 255, 130), font=small)
    return np.asarray(image)


def write_control_video(
    records: Sequence[Any], output: str | os.PathLike[str], *, task_id: int,
    episode_index: int, receipt: Any, fps: int = 10,
) -> dict[str, Any]:
    """Encode one immutable annotated MP4 and its immutable JSON sidecar."""
    if not records:
        raise ValueError("cannot encode a control video with zero records")
    if isinstance(fps, bool) or int(fps) <= 0:
        raise ValueError("fps must be a positive integer")
    output_path = Path(output)
    sidecar = Path(str(output_path) + ".json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"control video artifact already exists: {output_path}")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("control videos require imageio with an ffmpeg backend") from exc

    segments = ownership_segments(records)
    frames = []
    previous: str | None = None
    for item in records:
        owner = _owner(item)
        frame = getattr(item, "frame", item)
        timestep = int(getattr(frame, "timestep", len(frames)))
        frames.append(_annotate(
            _frame_image(item), owner=owner, task_id=int(task_id),
            episode_index=int(episode_index), step=timestep,
            transition=_transition(previous, owner),
            success=bool(getattr(item, "success", False)),
            terminal=bool(getattr(item, "terminal", False)),
        ))
        previous = owner
    height, width = frames[0].shape[:2]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("all agentview frames must have identical dimensions")
    # Keep the .mp4 suffix on the temporary file: imageio's legacy FFMPEG
    # plugin uses the extension to select its writer.
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    if temporary.exists():
        raise FileExistsError(f"temporary control video already exists: {temporary}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(
            str(temporary), format="FFMPEG", mode="I", fps=int(fps),
            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_log_level="error",
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        encoded = temporary.read_bytes()
        write_artifact(output_path, encoded, kind="arrow-oncall-control-video")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    video_sha = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "schema": "arrow_policy_suite.control_video.v1",
        "task_id": int(task_id), "episode_index": int(episode_index),
        "fps": int(fps), "frames": len(frames), "width": int(width), "height": int(height),
        "format": "H264/yuv420p",
        "video_sha256": video_sha,
        "source_receipt_sha256": _receipt_digest(receipt),
        "owner_counts": {owner: sum(1 for item in records if _owner(item) == owner)
                         for owner in ("vla", "arrow", "hybrid")},
        "segments": [segment.to_dict() for segment in segments],
        "transition_count": max(0, len(segments) - 1),
    }
    write_json_artifact(sidecar, manifest, kind="arrow-oncall-control-video-sidecar")
    return manifest


def inspect_control_video(path: str | os.PathLike[str], metadata: Mapping[str, Any], *,
                          expected_steps: int | None = None) -> dict[str, Any]:
    """Decode and validate one encoded control video against its sidecar.

    Hashing proves that bytes did not change; it does not prove that the bytes
    are a playable H264 rollout video. This helper performs the latter check
    with the same FFMPEG backend used by the writer.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("video inspection requires imageio with an ffmpeg backend") from exc
    target = Path(path)
    reader = imageio.get_reader(str(target), format="FFMPEG")
    try:
        info = dict(reader.get_meta_data() or {})
        codec = str(info.get("codec", "")).lower()
        pixel_format = str(info.get("pix_fmt", info.get("pixelformat", ""))).lower()
        if codec != "h264":
            raise ValueError(f"video codec is {codec!r}, expected h264")
        if not pixel_format.startswith("yuv420p"):
            raise ValueError(f"video pixel format is {pixel_format!r}, expected yuv420p")
        fps = float(info.get("fps", 0.0))
        if abs(fps - float(metadata.get("fps", 0.0))) > 1e-6:
            raise ValueError(f"video FPS {fps} disagrees with sidecar {metadata.get('fps')}")
        expected_width, expected_height = int(metadata.get("width", 0)), int(metadata.get("height", 0))
        source_size = info.get("source_size", info.get("size", (0, 0)))
        if tuple(map(int, source_size)) != (expected_width, expected_height):
            raise ValueError(f"video dimensions {source_size!r} disagree with sidecar {(expected_width, expected_height)!r}")
        first = reader.get_data(0)
        if first.shape[:2] != (expected_height, expected_width):
            raise ValueError(f"decoded first frame shape {first.shape!r} disagrees with sidecar")
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = sum(1 for _ in reader)
        expected_frames = int(metadata.get("frames", 0))
        if frame_count != expected_frames:
            raise ValueError(f"decoded frame count {frame_count} disagrees with sidecar {expected_frames}")
        if expected_steps is not None and frame_count != int(expected_steps):
            raise ValueError(f"decoded frame count {frame_count} disagrees with receipt steps {expected_steps}")
        last = reader.get_data(frame_count - 1)
        if last.shape[:2] != (expected_height, expected_width):
            raise ValueError(f"decoded last frame shape {last.shape!r} disagrees with sidecar")
        return {"codec": codec, "pix_fmt": pixel_format, "fps": fps,
                "frames": frame_count, "width": expected_width, "height": expected_height}
    finally:
        reader.close()


__all__ = ["OwnershipSegment", "ownership_segments", "write_control_video", "inspect_control_video"]
