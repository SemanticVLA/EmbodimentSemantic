"""RoboCasa-local RGB-D capture contract.

This is deliberately kept inside the RoboCasa package.  It uses only the
official robosuite camera hooks and does not import LIBERO capture helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_name: str
    width: int
    height: int
    intrinsic: list[list[float]]
    world_from_camera: list[list[float]]
    world_frame: str = "robocasa_mujoco_world"
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    extrinsic_direction: str = "world_from_camera"


@dataclass
class CapturedRGBD:
    rgb: np.ndarray
    normalized_depth: np.ndarray
    metric_depth: np.ndarray
    calibration: CameraCalibration
    observation: Mapping[str, Any] | None = None
    depth_conversion_mode: str = "normalized"
    depth_sanitization: Mapping[str, Any] | None = None

    @property
    def raw_depth(self) -> np.ndarray:
        return self.normalized_depth


def _as_rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {image.shape}")
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8, copy=False))


def _as_depth(value: Any, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(value)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape != shape:
        raise ValueError(f"RGB/depth are not aligned: RGB={shape}, depth={depth.shape}")
    if not np.isfinite(depth).any():
        raise ValueError("normalized depth contains no finite pixels")
    return np.ascontiguousarray(depth.astype(np.float32, copy=False))


def sanitize_normalized_depth(depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(depth, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"depth must be HxW, got {source.shape}")
    valid = np.isfinite(source) & (source >= 0.0) & (source <= 1.0)
    positive = source[valid & (source > 0.0)]
    if positive.size == 0:
        raise ValueError("normalized depth contains no positive finite in-range pixels")
    masked = ~valid
    fallback = float(np.median(positive))
    sanitized = source.copy()
    sanitized[masked] = fallback
    return sanitized, {
        "input_encoding": "normalized",
        "masked_pixel_count": int(np.count_nonzero(masked)),
        "total_pixel_count": int(source.size),
        "masked_fraction": float(np.count_nonzero(masked) / source.size),
        "fallback_value": fallback,
    }


def build_camera_calibration(sim: Any, camera_name: str, width: int, height: int) -> CameraCalibration:
    try:
        from robosuite.utils import camera_utils
    except ImportError as exc:  # pragma: no cover - live runtime boundary
        raise RuntimeError("robosuite camera_utils is required for RoboCasa calibration") from exc
    intrinsic = np.asarray(
        camera_utils.get_camera_intrinsic_matrix(
            sim, camera_name, camera_height=height, camera_width=width
        ),
        dtype=np.float64,
    )
    extrinsic = np.asarray(camera_utils.get_camera_extrinsic_matrix(sim, camera_name), dtype=np.float64)
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"unexpected calibration shapes K={intrinsic.shape}, T={extrinsic.shape}")
    # RoboSuite's camera utility returns the positive OpenCV intrinsic matrix
    # for the image after the native MuJoCo render is vertically flipped.
    # Keep K unchanged: changing fy/cy here would mirror projected roles a
    # second time relative to the post-flip RGB/depth frame.
    image_intrinsic = intrinsic.copy()
    return CameraCalibration(
        camera_name=str(camera_name), width=int(width), height=int(height),
        intrinsic=image_intrinsic.tolist(), world_from_camera=extrinsic.tolist(),
    )


def normalized_depth_to_metric(sim: Any, depth: np.ndarray) -> np.ndarray:
    try:
        from robosuite.utils import camera_utils
    except ImportError as exc:  # pragma: no cover - live runtime boundary
        raise RuntimeError("robosuite camera_utils is required for RoboCasa depth") from exc
    sanitized, _ = sanitize_normalized_depth(depth)
    metric = np.asarray(camera_utils.get_real_depth_map(sim, sanitized), dtype=np.float32)
    if metric.shape != np.asarray(depth).shape:
        raise ValueError(f"metric depth changed shape from {np.asarray(depth).shape} to {metric.shape}")
    invalid = ~(np.isfinite(depth) & (depth >= 0.0) & (depth <= 1.0))
    if np.any(invalid):
        metric = metric.copy()
        metric[invalid] = np.nan
    return np.ascontiguousarray(metric)


def capture_robocasa_rgbd(env: Any, *, camera_name: str, resolution: int) -> CapturedRGBD:
    """Capture one aligned, post-flip RoboCasa RGB-D frame."""

    raw = getattr(env, "_env", env)
    sim = getattr(raw, "sim", None)
    if sim is None:
        sim = getattr(env, "sim", None)
    if sim is None:
        raise RuntimeError("RoboCasa environment does not expose sim for capture")
    render = getattr(sim, "render", None)
    result = None
    if callable(render):
        try:
            result = render(
                camera_name=camera_name, width=int(resolution), height=int(resolution), depth=True
            )
        except (TypeError, NotImplementedError):
            result = None
    observation = getattr(env, "_last_observation", None)
    from_render = isinstance(result, tuple) and len(result) >= 2
    if from_render:
        rgb_raw, depth_raw = result[0], result[1]
    elif isinstance(observation, Mapping):
        rgb_raw = observation.get(f"video.{camera_name}")
        if rgb_raw is None:
            rgb_raw = observation.get(f"{camera_name}_image")
        depth_raw = observation.get(f"video.{camera_name}_depth")
        if depth_raw is None:
            depth_raw = observation.get(f"{camera_name}_depth")
    else:
        raise RuntimeError(f"could not capture aligned {camera_name} RGB/depth")
    if rgb_raw is None or depth_raw is None:
        raise RuntimeError("RoboCasa capture did not provide both RGB and depth")
    # Only raw ``sim.render`` is bottom-left-origin.  Observation dictionaries
    # are already wrapper-normalized top-left images and must not be flipped.
    if from_render:
        rgb_raw = np.asarray(rgb_raw)[::-1]
        depth_raw = np.asarray(depth_raw)[::-1]
    rgb = _as_rgb(rgb_raw)
    normalized = _as_depth(depth_raw, rgb.shape[:2])
    _, sanitization = sanitize_normalized_depth(normalized)
    calibration = build_camera_calibration(sim, camera_name, rgb.shape[1], rgb.shape[0])
    metric = normalized_depth_to_metric(sim, normalized)
    return CapturedRGBD(
        rgb=rgb, normalized_depth=normalized, metric_depth=metric,
        calibration=calibration,
        observation=observation if isinstance(observation, Mapping) else None,
        depth_conversion_mode=("normalized_masked" if sanitization["masked_pixel_count"] else "normalized"),
        depth_sanitization=sanitization,
    )


__all__ = [
    "CameraCalibration", "CapturedRGBD", "build_camera_calibration",
    "capture_robocasa_rgbd", "normalized_depth_to_metric", "sanitize_normalized_depth",
]
