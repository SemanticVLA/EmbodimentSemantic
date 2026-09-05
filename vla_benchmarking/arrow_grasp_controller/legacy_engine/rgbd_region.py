"""SAM-free RGB-D region masks for the Molmo canary.

The only object localization input is an arrow-decoded pixel in an observed
RGB-D frame.  Wrist association is deliberately strict: agentview support is
projected through measured depth and calibration, then retained only where the
current wrist depth agrees with the projected surface.  There is no detection
or nearest-object fallback in this module.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .arrow_controller import derive_rgbd_region_mask
except ImportError:  # pragma: no cover - direct script use
    from vla_benchmarking.arrow_grasp_controller.legacy_engine.arrow_controller import derive_rgbd_region_mask


def _calibration(capture: Any) -> tuple[np.ndarray, np.ndarray]:
    calibration = getattr(capture, "calibration", None)
    if calibration is None:
        raise ValueError("capture is missing calibration")
    K = np.asarray(getattr(calibration, "intrinsic", None), dtype=np.float64)
    T = np.asarray(getattr(calibration, "world_from_camera", None), dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.all(np.isfinite(K)) or not np.all(np.isfinite(T)):
        raise ValueError("capture calibration has invalid intrinsic/extrinsic matrices")
    if abs(K[0, 0]) <= 1e-9 or abs(K[1, 1]) <= 1e-9:
        raise ValueError("capture calibration has zero focal length")
    if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError("capture extrinsic must be homogeneous")
    return K, T


def _aligned(capture: Any) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(getattr(capture, "rgb", None))
    depth = np.asarray(getattr(capture, "metric_depth", None), dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("capture RGB must be uint8 HxWx3")
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("capture RGB and metric depth are not aligned")
    return rgb, depth


def _deproject_pixels(depth: np.ndarray, uv: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    z = depth[uv[:, 1], uv[:, 0]]
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    pixels = uv[valid]
    values = z[valid]
    camera = np.column_stack(((pixels[:, 0] - K[0, 2]) * values / K[0, 0],
                              (pixels[:, 1] - K[1, 2]) * values / K[1, 1], values))
    return (T[:3, :3] @ camera.T).T + T[:3, 3]


def _project_world(points: np.ndarray, K: np.ndarray, T: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (T[:3, :3].T @ (points - T[:3, 3]).T).T
    z = camera[:, 2]
    finite = np.isfinite(camera).all(axis=1) & (z > 1e-8)
    u = K[0, 0] * camera[:, 0] / z + K[0, 2]
    v = K[1, 1] * camera[:, 1] / z + K[1, 2]
    h, w = shape
    inside = finite & np.isfinite(u) & np.isfinite(v) & (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
    return np.column_stack((u, v)), z, inside


def derive_observed_region_mask(
    capture: Any,
    source_uv: Sequence[float],
    *,
    profile_target_world: Sequence[float] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an arrow-seeded mask from one observed RGB-D capture."""
    rgb, depth = _aligned(capture)
    K, T = _calibration(capture)
    if profile_target_world is not None:
        profile = np.asarray(profile_target_world, dtype=np.float64).reshape(-1)
        if profile.shape != (3,) or not np.all(np.isfinite(profile)):
            raise ValueError("profile_target_world must be a finite 3-vector")
    mask, audit = derive_rgbd_region_mask(rgb, depth, K, T, source_uv, **kwargs)
    return mask, {"camera_name": str(getattr(capture.calibration, "camera_name", "")), **audit}


def project_observed_region_to_wrist(
    agentview_capture: Any,
    wrist_capture: Any,
    agentview_mask: np.ndarray,
    *,
    depth_tolerance_m: float = 0.025,
    min_support_pixels: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project observed agentview support onto wrist depth with strict identity."""
    agent_rgb, agent_depth = _aligned(agentview_capture)
    wrist_rgb, wrist_depth = _aligned(wrist_capture)
    if np.asarray(agentview_mask).shape != agent_rgb.shape[:2]:
        raise ValueError("agentview mask does not match its RGB frame")
    mask = np.asarray(agentview_mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_support_pixels:
        raise ValueError("insufficient observed agentview support for wrist association")
    agent_K, agent_T = _calibration(agentview_capture)
    wrist_K, wrist_T = _calibration(wrist_capture)
    pixels = np.column_stack((xs, ys)).astype(int)
    world = _deproject_pixels(agent_depth, pixels, agent_K, agent_T)
    if len(world) == 0:
        raise ValueError("agentview region has no valid metric-depth support")
    projected, expected_z, inside = _project_world(world, wrist_K, wrist_T, wrist_depth.shape)
    projected = projected[inside]
    expected_z = expected_z[inside]
    if len(projected) == 0:
        raise ValueError("projected agentview region is outside wrist image")
    wrist_uv = np.rint(projected).astype(int)
    h, w = wrist_depth.shape
    rounded_inside = (
        (wrist_uv[:, 0] >= 0) & (wrist_uv[:, 0] < w)
        & (wrist_uv[:, 1] >= 0) & (wrist_uv[:, 1] < h)
    )
    wrist_uv = wrist_uv[rounded_inside]
    expected_z = expected_z[rounded_inside]
    if len(wrist_uv) == 0:
        raise ValueError("projected agentview region rounds outside wrist image")
    current_z = wrist_depth[wrist_uv[:, 1], wrist_uv[:, 0]]
    agree = np.isfinite(current_z) & (current_z > 0.0) & (np.abs(current_z - expected_z) <= float(depth_tolerance_m))
    if int(np.count_nonzero(agree)) < min_support_pixels:
        raise ValueError("wrist RGB-D support does not strictly match projected agentview region")
    result = np.zeros(wrist_depth.shape, dtype=bool)
    result[wrist_uv[agree, 1], wrist_uv[agree, 0]] = True
    if int(np.count_nonzero(result)) < min_support_pixels:
        raise ValueError("projected wrist RGB-D region is too small")
    audit = {
        "method": "observed_agentview_to_wrist_metric_depth_v1",
        "agentview_region_area_px": int(np.count_nonzero(mask)),
        "projected_support_px": int(len(projected)),
        "depth_agreement_px": int(np.count_nonzero(agree)),
        "wrist_region_area_px": int(np.count_nonzero(result)),
        "depth_tolerance_m": float(depth_tolerance_m),
        "region_mask_sha256": hashlib.sha256(np.ascontiguousarray(result.astype(np.uint8)).tobytes()).hexdigest(),
    }
    return result, audit


__all__ = ["derive_observed_region_mask", "project_observed_region_to_wrist"]
