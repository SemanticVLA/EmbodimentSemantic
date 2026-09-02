"""RGB-D adapter and geometry utilities for an external ZeroGrasp worker.

This module is deliberately importable in a normal LIBERO environment.  It
contains no model/runtime imports; those live behind the JSONL worker process.
"""

from __future__ import annotations

import json
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .zerograsp_contracts import (
        ArrowMasks,
        GRASPNET_CAMERA_FRAME,
        GRASPNET_TRANSLATION_REFERENCE,
        GraspCandidate,
        MaskDiagnostics,
        PlacementPlan,
        PreparedZeroGraspInput,
        ReconstructionSummary,
        ZeroGraspConfig,
        ZeroGraspInferenceResult,
        ZeroGraspObservation,
        decode_array,
        encode_array,
        hash_array,
        serialize_audit,
        stable_json_hash,
        validate_config,
        validate_se3,
    )
except ImportError:  # direct invocation from the vla_benchmarking directory
    from zerograsp_contracts import (
        ArrowMasks,
        GRASPNET_CAMERA_FRAME,
        GRASPNET_TRANSLATION_REFERENCE,
        GraspCandidate,
        MaskDiagnostics,
        PlacementPlan,
        PreparedZeroGraspInput,
        ReconstructionSummary,
        ZeroGraspConfig,
        ZeroGraspInferenceResult,
        ZeroGraspObservation,
        decode_array,
        encode_array,
        hash_array,
        serialize_audit,
        stable_json_hash,
        validate_config,
        validate_se3,
    )


def _check_observation(observation: ZeroGraspObservation | Mapping[str, Any]) -> ZeroGraspObservation:
    if not isinstance(observation, ZeroGraspObservation):
        try:
            observation = ZeroGraspObservation(**dict(observation))
        except TypeError as exc:
            raise ValueError("observation contains fields outside the controller contract") from exc
    return observation


def _nearest_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.copy()
    yy = np.minimum(np.floor(np.arange(height) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    xx = np.minimum(np.floor(np.arange(width) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[np.ix_(yy, xx)]


def _bilinear_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Small dependency-free RGB bilinear resize (pixel-center convention)."""
    if image.shape[:2] == (height, width):
        return image.copy()
    source_h, source_w = image.shape[:2]
    ys = (np.arange(height, dtype=float) + 0.5) * source_h / height - 0.5
    xs = (np.arange(width, dtype=float) + 0.5) * source_w / width - 0.5
    y0 = np.floor(ys).astype(int).clip(0, source_h - 1); y1 = (y0 + 1).clip(0, source_h - 1)
    x0 = np.floor(xs).astype(int).clip(0, source_w - 1); x1 = (x0 + 1).clip(0, source_w - 1)
    wy = (ys - np.floor(ys))[:, None]
    wx = (xs - np.floor(xs))[None, :]
    top = image[y0[:, None], x0[None, :]] * (1 - wx[..., None]) + image[y0[:, None], x1[None, :]] * wx[..., None]
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx[..., None]) + image[y1[:, None], x1[None, :]] * wx[..., None]
    return np.clip(top * (1 - wy[..., None]) + bottom * wy[..., None], 0, 255).astype(image.dtype)


def make_vertical_flip_affine(height: int, *, flip: bool) -> np.ndarray:
    """Return exact image-pixel homogeneous transform for a vertical flip."""
    if height <= 0:
        raise ValueError("height must be positive")
    if not flip:
        return np.eye(3, dtype=np.float64)
    return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])


def make_letterbox_affine(source_shape: tuple[int, int], target_shape: tuple[int, int] = (1024, 1280)) -> np.ndarray:
    """Map source pixels into a centered, aspect-preserving target image."""
    source_h, source_w = (int(x) for x in source_shape)
    target_h, target_w = (int(x) for x in target_shape)
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(target_w / source_w, target_h / source_h)
    pad_x = (target_w - scale * source_w) / 2.0
    pad_y = (target_h - scale * source_h) / 2.0
    return np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]], dtype=np.float64)


def map_pixel(pixel: Sequence[float], affine: np.ndarray) -> tuple[float, float]:
    point = np.asarray([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    out = np.asarray(affine, dtype=np.float64) @ point
    return float(out[0] / out[2]), float(out[1] / out[2])


def preprocess_rgbd(observation: ZeroGraspObservation, config: ZeroGraspConfig) -> PreparedZeroGraspInput:
    """Apply exact sign-aware flip then 1024x1280 centered letterbox.

    ``fy < 0`` is an image-orientation contract, not permission to take an
    absolute value.  Flipping both RGB/depth and pixels first preserves every
    camera ray; the returned intrinsic is the corresponding transformed K.
    """
    obs = _check_observation(observation)
    cfg = validate_config(config)
    h, w = obs.clean_rgb.shape[:2]
    need_flip = float(obs.K[1, 1]) < 0
    flip_affine = make_vertical_flip_affine(h, flip=need_flip)
    if need_flip:
        rgb = np.flip(obs.clean_rgb, axis=0).copy()
        depth = np.flip(obs.depth_m, axis=0).copy()
    else:
        rgb, depth = obs.clean_rgb.copy(), obs.depth_m.copy()
    K_flip = flip_affine @ np.asarray(obs.K, dtype=np.float64)
    letterbox = make_letterbox_affine((h, w), (cfg.model_height, cfg.model_width))
    affine = letterbox @ flip_affine
    out_h, out_w = cfg.model_height, cfg.model_width
    resized_rgb = _bilinear_resize(rgb, int(round((h * letterbox[1, 1]))), int(round((w * letterbox[0, 0]))))
    resized_depth = _nearest_resize(depth, resized_rgb.shape[0], resized_rgb.shape[1])
    canvas_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas_depth = np.full((out_h, out_w), np.nan, dtype=np.float32)
    y0 = int(round(letterbox[1, 2]))
    x0 = int(round(letterbox[0, 2]))
    canvas_rgb[y0:y0 + resized_rgb.shape[0], x0:x0 + resized_rgb.shape[1]] = resized_rgb
    canvas_depth[y0:y0 + resized_depth.shape[0], x0:x0 + resized_depth.shape[1]] = resized_depth.astype(np.float32, copy=False)
    source = map_pixel(obs.source_px, affine)
    destination = map_pixel(obs.destination_px, affine)
    K_model = letterbox @ K_flip
    request_material = {
        "clean_rgb": hash_array(canvas_rgb),
        "depth_m": hash_array(canvas_depth),
        "source_px_model": source,
        "destination_px_model": destination,
        "K_model": hash_array(K_model),
        "pixel_affine": affine.tolist(),
    }
    return PreparedZeroGraspInput(
        clean_rgb=canvas_rgb,
        depth_m=canvas_depth,
        source_mask=np.zeros((out_h, out_w), dtype=bool),
        destination_mask=np.zeros((out_h, out_w), dtype=bool),
        K_model=K_model,
        source_px_model=source,
        destination_px_model=destination,
        pixel_affine=affine,
        vertically_flipped=need_flip,
        request_hash=stable_json_hash(request_material),
    )


def _seeded_region(rgb: np.ndarray, depth: np.ndarray, seed: tuple[float, float], config: ZeroGraspConfig) -> tuple[np.ndarray, float | None, float]:
    """Grow a deterministic local RGB-D region from one arrow endpoint."""
    h, w = depth.shape
    u, v = int(round(seed[0])), int(round(seed[1]))
    radius = int(config.mask_seed_radius_px)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    local_depth = depth[y0:y1, x0:x1]
    valid = np.isfinite(local_depth) & (local_depth > 0)
    if not valid.any():
        return np.zeros((h, w), dtype=bool), None, 0.0
    depth_seed = float(np.median(local_depth[valid]))
    local_rgb = rgb[y0:y1, x0:x1].astype(np.float64)
    rgb_seed = local_rgb[valid].reshape(-1, 3).mean(axis=0) if valid.any() else np.zeros(3)
    color_distance = np.linalg.norm(rgb.astype(np.float64) - rgb_seed, axis=2)
    depth_distance = np.abs(np.where(np.isfinite(depth), depth, depth_seed) - depth_seed)
    candidate = (color_distance <= float(config.mask_color_tolerance)) & (depth_distance <= float(config.mask_depth_tolerance_m)) & np.isfinite(depth) & (depth > 0)
    # Enforce spatial containment by connected flood from the seed.  No object
    # class, task constant, or bbox participates in this operation.
    if not candidate[min(max(v, 0), h - 1), min(max(u, 0), w - 1)]:
        # permit the seed when it is a valid depth sample but slightly outside
        # the adaptive RGB threshold, then use the same flood rule.
        if not (np.isfinite(depth[v, u]) and depth[v, u] > 0):
            return np.zeros((h, w), dtype=bool), depth_seed, 0.0
        candidate[v, u] = True
    mask = np.zeros((h, w), dtype=bool)
    stack = [(u, v)]
    while stack:
        x, y = stack.pop()
        if mask[y, x] or not candidate[y, x]:
            continue
        mask[y, x] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                        stack.append((nx, ny))
    area = int(mask.sum())
    confidence = min(1.0, area / max(1.0, float(config.mask_min_area_px)))
    confidence *= min(1.0, valid.sum() / max(1.0, float((2 * radius + 1) ** 2)))
    if area > h * w * float(config.mask_max_area_fraction):
        return np.zeros((h, w), dtype=bool), depth_seed, 0.0
    return mask, depth_seed, float(confidence)


def build_arrow_seeded_masks(observation: ZeroGraspObservation, config: ZeroGraspConfig | None = None) -> ArrowMasks:
    """Create disjoint source/destination RGB-D masks from arrow endpoint seeds."""
    cfg = validate_config(config)
    obs = _check_observation(observation)
    distance = float(np.linalg.norm(np.asarray(obs.source_px) - np.asarray(obs.destination_px)))
    if distance <= max(2.0, 2 * cfg.mask_seed_radius_px):
        d = MaskDiagnostics(False, "arrow endpoint seeds are too close to form disjoint roles", 0, 0, 0.0, 0.0, None, None, False)
        return ArrowMasks(np.zeros(obs.depth_m.shape, bool), np.zeros(obs.depth_m.shape, bool), d)
    source, source_depth, source_conf = _seeded_region(obs.clean_rgb, obs.depth_m, obs.source_px, cfg)
    dest, dest_depth, dest_conf = _seeded_region(obs.clean_rgb, obs.depth_m, obs.destination_px, cfg)
    overlap = source & dest
    if overlap.any():
        # Assign ambiguous pixels to the closer endpoint; ties are assigned to
        # source, giving a stable, disjoint partition.
        yy, xx = np.nonzero(overlap)
        s = (xx - obs.source_px[0]) ** 2 + (yy - obs.source_px[1]) ** 2
        d = (xx - obs.destination_px[0]) ** 2 + (yy - obs.destination_px[1]) ** 2
        source[yy[d < s], xx[d < s]] = False
        dest[yy[s < d], xx[s < d]] = False
        dest[yy[s == d], xx[s == d]] = False
    source_area, dest_area = int(source.sum()), int(dest.sum())
    ok = source_area >= cfg.mask_min_area_px and dest_area >= cfg.mask_min_area_px and source_conf > 0 and dest_conf > 0 and not np.any(source & dest)
    reason = None if ok else "one or both arrow-seeded RGB-D masks are too small or low-confidence"
    diagnostics = MaskDiagnostics(ok, reason, source_area, dest_area, source_conf, dest_conf, source_depth, dest_depth, not np.any(source & dest))
    return ArrowMasks(source, dest, diagnostics)


def prepare_zero_grasp_input(observation: ZeroGraspObservation, config: ZeroGraspConfig | None = None) -> PreparedZeroGraspInput:
    cfg = validate_config(config)
    prepared = preprocess_rgbd(_check_observation(observation), cfg)
    masks = build_arrow_seeded_masks(_check_observation(observation), cfg)
    h, w = observation.depth_m.shape
    def map_mask(mask: np.ndarray) -> np.ndarray:
        work = np.flip(mask, axis=0) if prepared.vertically_flipped else mask
        scale = min(cfg.model_width / w, cfg.model_height / h)
        inner_h = int(round(h * scale))
        inner_w = int(round(w * scale))
        resized = _nearest_resize(work.astype(np.uint8), inner_h, inner_w)
        canvas = np.zeros((cfg.model_height, cfg.model_width), dtype=bool)
        x0 = int(round((cfg.model_width - inner_w) / 2.0))
        y0 = int(round((cfg.model_height - inner_h) / 2.0))
        canvas[y0:y0 + inner_h, x0:x0 + inner_w] = resized.astype(bool)
        return canvas
    source_mask = map_mask(masks.source_mask)
    destination_mask = map_mask(masks.destination_mask)
    material = dict(prepared.__dict__)
    material["source_mask"] = hash_array(source_mask)
    material["destination_mask"] = hash_array(destination_mask)
    rejection = () if masks.diagnostics.ok else ((masks.diagnostics.reason or "mask validation failed"),)
    return PreparedZeroGraspInput(**{**prepared.__dict__, "source_mask": source_mask, "destination_mask": destination_mask, "request_hash": stable_json_hash(material), "source_mask_hash": hash_array(source_mask), "destination_mask_hash": hash_array(destination_mask), "source_mask_area": int(masks.source_mask.sum()), "destination_mask_area": int(masks.destination_mask.sum()), "mask_rejections": rejection})


def parse_grasp_array(raw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Strictly parse model grasps as finite homogeneous 4x4 matrices + scores."""
    scores: Any = None
    if isinstance(raw, Mapping):
        if set(raw) - {"grasps", "scores"} or "grasps" not in raw:
            raise ValueError("grasp mapping may contain only grasps and scores")
        scores = raw.get("scores")
        raw = raw["grasps"]
    arr = np.asarray(raw)
    if arr.size == 0:
        arr = np.empty((0, 4, 4), dtype=np.float64)
    if not np.issubdtype(arr.dtype, np.number) or arr.ndim != 3 or arr.shape[1:] != (4, 4) or not np.all(np.isfinite(arr)):
        raise ValueError("grasps must be a finite numeric array with shape (N, 4, 4)")
    matrices = arr.astype(np.float64, copy=False)
    if not np.allclose(matrices[:, 3, :], np.array([0, 0, 0, 1]), atol=1e-5):
        raise ValueError("every grasp must be a homogeneous transform")
    for rotation in matrices[:, :3, :3]:
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError("every grasp rotation must be proper orthonormal")
    if scores is None:
        score_array = np.ones((len(matrices),), dtype=np.float64)
    else:
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_array.shape != (len(matrices),) or not np.all(np.isfinite(score_array)):
            raise ValueError("scores must be one finite value per grasp")
    return matrices, score_array


def transform_T_W_G(T_W_E: Any, T_E_G: Any) -> np.ndarray:
    """Compose diagnostic world-from-grasp geometry; never an EEF command."""
    return validate_se3(T_W_E, "T_W_E") @ validate_se3(T_E_G, "T_E_G")


def transform_T_W_E(T_W_G: Any, T_G_E: Any) -> np.ndarray:
    """Convert intermediate grasp geometry to an EEF pose with a validated transform."""
    return validate_se3(T_W_G, "T_W_G") @ validate_se3(T_G_E, "T_G_E")


def compose_T_W_G_from_eef(T_W_E: Any, T_E_G: Any) -> np.ndarray:
    """Compose world-from-EEF with EEF-from-grasp explicitly."""
    return transform_T_W_G(T_W_E, T_E_G)


def compose_T_W_E_from_grasp(T_W_G: Any, T_G_E: Any) -> np.ndarray:
    """Compose world-from-grasp with grasp-from-EEF explicitly."""
    return transform_T_W_E(T_W_G, T_G_E)


def dynamic_T_G_E(candidate: GraspCandidate, config: ZeroGraspConfig) -> np.ndarray:
    """Build candidate-dependent grasp→EEF calibration from verified evidence.

    The released convention stores grasp depth along local +X from grasp
    center to fingertip.  A fixed full transform is intentionally not used.
    """
    cfg = validate_config(config)
    rotation = cfg.R_grasp_eef
    if not cfg.eef_calibration_verified or rotation is None:
        raise ValueError("verified grasp-to-EEF rotation calibration is required")
    if candidate.depth_m is None or candidate.depth_m <= 0:
        raise ValueError("candidate depth_m is required for dynamic EEF translation")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    residual_e = np.zeros(3, dtype=float) if cfg.tip_to_eef_residual_m is None else np.asarray(cfg.tip_to_eef_residual_m, dtype=float)
    # Residual is calibrated in EEF coordinates, then expressed in grasp
    # coordinates for the homogeneous composition below.
    T[:3, 3] = np.array([float(candidate.depth_m), 0.0, 0.0]) + T[:3, :3] @ residual_e
    return T


def transform_camera_grasps_to_world(candidates: Sequence[GraspCandidate], T_world_camera: Any) -> tuple[np.ndarray, ...]:
    T_W_C = validate_se3(T_world_camera, "T_world_camera")
    return tuple(T_W_C @ validate_se3(candidate.T_camera_gripper, "T_camera_gripper") for candidate in candidates)


def make_pregrasp(T_W_G: Any, distance_m: float, approach_axis_local: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    pose = validate_se3(T_W_G, "T_W_G")
    axis = np.asarray(approach_axis_local, dtype=np.float64).reshape(-1)
    if axis.size != 3 or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-9 or distance_m < 0:
        raise ValueError("approach axis must be nonzero finite 3-vector and distance non-negative")
    axis = axis / np.linalg.norm(axis)
    out = pose.copy()
    # A pregrasp is behind the grasp along the approach direction.  For the
    # released GraspNet convention this is ``t - d * R[:, 0]`` (+X approach).
    out[:3, 3] = pose[:3, 3] - pose[:3, :3] @ axis * float(distance_m)
    return out


def _world_point_cloud(observation: ZeroGraspObservation, *, exclude_mask: np.ndarray | None = None) -> np.ndarray:
    depth = observation.depth_m
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    if exclude_mask is not None:
        valid &= ~np.asarray(exclude_mask, dtype=bool)
    z = depth[valid]
    pixels = np.stack([xx[valid], yy[valid], np.ones_like(z)], axis=0)
    camera = np.linalg.inv(observation.K) @ pixels
    camera *= z / camera[2]
    world = observation.T_world_camera @ np.vstack([camera, np.ones_like(z)])
    return world[:3].T


def observed_depth_swept_path_clear(observation: ZeroGraspObservation, start_world: Sequence[float], end_world: Sequence[float], clearance_m: float, *, exclude_mask: np.ndarray | None = None, samples: int = 24) -> tuple[bool, float]:
    """Check a translational EEF sweep against observed depth points."""
    start, end = np.asarray(start_world, dtype=np.float64).reshape(-1), np.asarray(end_world, dtype=np.float64).reshape(-1)
    if start.size != 3 or end.size != 3 or clearance_m < 0 or samples < 2:
        raise ValueError("invalid swept-path arguments")
    points = _world_point_cloud(_check_observation(observation), exclude_mask=exclude_mask)
    if len(points) == 0:
        return True, float("inf")
    trajectory = np.linspace(start, end, int(samples))
    min_distance = float(np.min(np.linalg.norm(points[:, None, :] - trajectory[None, :, :], axis=2)))
    return bool(min_distance >= float(clearance_m)), min_distance


def _candidate_projects_into_source(candidate: GraspCandidate, observation: ZeroGraspObservation, source_mask: np.ndarray | None, reconstruction: ReconstructionSummary | None, margin_m: float) -> bool:
    translation_camera = np.asarray(candidate.T_camera_gripper, dtype=float)[:3, 3]
    if translation_camera[2] <= 0:
        return False
    if source_mask is not None:
        mask = np.asarray(source_mask, dtype=bool)
        projected = observation.K @ translation_camera
        u, v = int(round(projected[0] / projected[2])), int(round(projected[1] / projected[2]))
        if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
            if mask[v, u]:
                return True
            yy, xx = np.nonzero(mask)
            if len(xx):
                pixel_margin = max(1, int(round(observation.K[0, 0] * margin_m / max(translation_camera[2], 1e-6))))
                if np.any((np.abs(xx - u) <= pixel_margin) & (np.abs(yy - v) <= pixel_margin)):
                    return True
    if reconstruction is not None and reconstruction.bounds_camera_m is not None:
        bounds = np.asarray(reconstruction.bounds_camera_m, dtype=float)
        return bool(np.all(translation_camera >= bounds[:3] - margin_m) and np.all(translation_camera <= bounds[3:] + margin_m))
    return False


def _quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


def _eef_orientation_bonus(T_W_G: np.ndarray, observation: ZeroGraspObservation | None, config: ZeroGraspConfig) -> float:
    """Optional orientation tie-break using separately calibrated H-to-E rotation."""
    grasp_to_eef = config.R_grasp_eef
    if not config.eef_calibration_verified or observation is None or observation.eef_quaternion_right_hand_xyzw is None or config.R_H_E is None or grasp_to_eef is None:
        return 0.0
    target_R_W_E = _quaternion_xyzw_to_rotation(observation.eef_quaternion_right_hand_xyzw) @ np.asarray(config.R_H_E, dtype=float)
    candidate_R_W_E = T_W_G[:3, :3] @ np.asarray(grasp_to_eef, dtype=float)
    return float((np.trace(target_R_W_E.T @ candidate_R_W_E) - 1.0) / 2.0)


def rank_grasps(candidates: Sequence[GraspCandidate], T_world_camera: Any, config: ZeroGraspConfig, *, eef_position_world_m: Any | None = None, observation: ZeroGraspObservation | None = None, source_mask: np.ndarray | None = None, reconstruction: ReconstructionSummary | None = None, audit: dict[str, Any] | None = None) -> tuple[tuple[GraspCandidate, np.ndarray, float], ...]:
    """Filter/rank intermediate camera-grasp geometry deterministically.

    Returned ``T_W_G`` values are diagnostic geometry only; execution must use
    the EEF-explicit output of :func:`select_grasp_result`.
    """
    cfg = validate_config(config)
    T_W_C = validate_se3(T_world_camera, "T_world_camera")
    lo, hi = np.asarray(cfg.grasp_workspace_min_m), np.asarray(cfg.grasp_workspace_max_m)
    ranked: list[tuple[GraspCandidate, np.ndarray, float]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        T_W_G = T_W_C @ validate_se3(candidate.T_camera_gripper, "T_camera_gripper")
        position = T_W_G[:3, 3]
        if np.any(position < lo) or np.any(position > hi):
            rejected.append({"index": index, "reason": "workspace"})
            continue
        if not candidate.collision_free:
            rejected.append({"index": index, "reason": "collision"})
            continue
        if candidate.depth_m is None or not (0.0 < float(candidate.depth_m) <= 0.04):
            rejected.append({"index": index, "reason": "missing_or_invalid_depth"})
            continue
        if candidate.width_m is None or not (0.005 <= float(candidate.width_m) <= 0.08):
            rejected.append({"index": index, "reason": "missing_or_invalid_width"})
            continue
        if observation is not None and not _candidate_projects_into_source(candidate, observation, source_mask, reconstruction, max(float(cfg.swept_path_clearance_m), float(candidate.depth_m or 0.0))):
            rejected.append({"index": index, "reason": "source_association"})
            continue
        if not (cfg.grasp_width_range_m[0] <= candidate.width_m <= cfg.grasp_width_range_m[1]):
            rejected.append({"index": index, "reason": "width"})
            continue
        if not (cfg.grasp_depth_range_m[0] <= candidate.depth_m <= cfg.grasp_depth_range_m[1]):
            rejected.append({"index": index, "reason": "depth"})
            continue
        if candidate.height_m is not None and not (cfg.grasp_height_range_m[0] <= candidate.height_m <= cfg.grasp_height_range_m[1]):
            rejected.append({"index": index, "reason": "height"})
            continue
        if observation is not None and source_mask is not None:
            source_points = _world_point_cloud(observation, exclude_mask=~np.asarray(source_mask, dtype=bool))
            if len(source_points):
                # Candidate translation must remain in the observed source
                # envelope, with a margin derived from candidate geometry and
                # configured clearance rather than a task-specific offset.
                margin = max(float(cfg.swept_path_clearance_m), float(candidate.depth_m or 0.0), float(candidate.width_m or 0.0))
                if np.any(position < source_points.min(axis=0) - margin) or np.any(position > source_points.max(axis=0) + margin):
                    rejected.append({"index": index, "reason": "source_envelope"})
                    continue
        # The official grasp convention is parameterized explicitly (default
        # +X); no unspoken gripper-axis assumption is made in the adapter.
        approach_world = T_W_G[:3, :3] @ (np.asarray(cfg.approach_axis_local, dtype=float) / np.linalg.norm(cfg.approach_axis_local))
        min_cos = max(cfg.approach_axis_cos_min, float(np.cos(np.deg2rad(cfg.max_approach_tilt_deg))))
        # +X points palm-to-object in the released convention; a top-down
        # approach therefore aligns with world -Z.
        if float(np.dot(approach_world, np.array([0.0, 0.0, -1.0]))) < min_cos - 1e-9:
            rejected.append({"index": index, "reason": "approach_tilt"})
            continue
        min_clearance = float(candidate.clearance_m)
        derived_eef_world: np.ndarray | None = None
        derived_pregrasp_world: np.ndarray | None = None
        if cfg.eef_calibration_verified:
            try:
                T_G_E = dynamic_T_G_E(candidate, cfg)
                derived_eef_world = compose_T_W_E_from_grasp(T_W_G, T_G_E)
                derived_pregrasp_world = compose_T_W_E_from_grasp(make_pregrasp(T_W_G, cfg.pregrasp_distance_m, cfg.approach_axis_local), T_G_E)
            except ValueError:
                rejected.append({"index": index, "reason": "eef_calibration"})
                continue
        if observation is not None and eef_position_world_m is not None:
            if not cfg.eef_calibration_verified:
                rejected.append({"index": index, "reason": "unverified_eef_for_swept_path"})
                continue
            eef_position = np.asarray(eef_position_world_m, dtype=float).reshape(-1)
            if eef_position.size != 3 or not np.all(np.isfinite(eef_position)):
                rejected.append({"index": index, "reason": "invalid_eef_position"})
                continue
            # The controller first travels to the calibrated pregrasp and only
            # then performs a bounded contact descent.  Checking a direct EEF
            # sweep into the grasp pose incorrectly treats the intended final
            # approach near the support surface as a transit collision.
            target_position = derived_pregrasp_world[:3, 3] if derived_pregrasp_world is not None else position
            clear, observed_clearance = observed_depth_swept_path_clear(observation, eef_position, target_position, cfg.swept_path_clearance_m, exclude_mask=source_mask)
            if not clear:
                rejected.append({"index": index, "reason": "pregrasp_swept_path", "observed_clearance_m": observed_clearance, "required_clearance_m": float(cfg.swept_path_clearance_m)})
                continue
            min_clearance = min(min_clearance, observed_clearance)
        if cfg.eef_calibration_verified and derived_eef_world is not None and derived_pregrasp_world is not None:
            if np.any(derived_eef_world[:3, 3] < lo) or np.any(derived_eef_world[:3, 3] > hi) or np.any(derived_pregrasp_world[:3, 3] < lo) or np.any(derived_pregrasp_world[:3, 3] > hi):
                rejected.append({"index": index, "reason": "eef_workspace"})
                continue
        ranked.append((candidate, T_W_G, min_clearance))
    ranked.sort(key=lambda item: (-(float(item[0].score) + 1e-6 * _eef_orientation_bonus(item[1], observation, cfg)), -float(item[2]), int(item[0].source_index)))
    if audit is not None:
        audit.update({"candidate_count": len(candidates), "accepted_count": len(ranked), "candidate_rejections": rejected})
    return tuple(ranked[:cfg.max_candidates])


def _surface_normal_and_point(observation: ZeroGraspObservation, mask: np.ndarray | None, destination_px: Sequence[float]) -> tuple[np.ndarray, np.ndarray, float]:
    obs = _check_observation(observation)
    valid = np.isfinite(obs.depth_m) & (obs.depth_m > 0)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if int(valid.sum()) < 3:
        u, v = int(round(destination_px[0])), int(round(destination_px[1]))
        if not valid[v, u]:
            raise ValueError("destination has insufficient valid depth")
        pixels = np.array([[u], [v], [1.0]], dtype=float)
        camera = np.linalg.inv(obs.K) @ pixels
        camera *= float(obs.depth_m[v, u]) / camera[2]
        point = (obs.T_world_camera @ np.r_[camera[:, 0], 1.0])[:3]
        return np.array([0.0, 0.0, 1.0]), point, 0.25
    yy, xx = np.nonzero(valid)
    z = obs.depth_m[yy, xx]
    camera = np.linalg.inv(obs.K) @ np.vstack([xx, yy, np.ones_like(z)])
    camera *= z / camera[2]
    world = (obs.T_world_camera @ np.vstack([camera, np.ones_like(z)]))[:3].T
    center = np.median(world, axis=0)
    _, _, vh = np.linalg.svd(world - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    residual = np.abs((world - center) @ normal)
    confidence = float(min(1.0, len(world) / 64.0) * min(1.0, 0.01 / max(1e-6, float(np.median(residual)))))
    return normal / np.linalg.norm(normal), center, confidence


def reconstruction_assisted_placement(observation: ZeroGraspObservation, reconstruction: ReconstructionSummary, *, selected_T_world_eef: np.ndarray, destination_mask: np.ndarray | None = None, destination_px: Sequence[float] | None = None, clearance_m: float = 0.004) -> PlacementPlan:
    """Place the reconstructed footprint on an observed destination surface.

    The arrow supplies the destination seed; reconstruction supplies only size,
    never an object pose.  A deterministic tangent is derived from the arrow
    direction projected onto the support plane.
    """
    obs = _check_observation(observation)
    if reconstruction.source_frame != "camera":
        raise ValueError("placement requires reconstruction.source_frame exactly camera")
    if destination_px is None:
        destination_px = obs.destination_px
    normal, support, surface_conf = _surface_normal_and_point(obs, destination_mask, destination_px)
    footprint = np.asarray(reconstruction.dimensions_m[:2], dtype=float) + 2 * float(clearance_m)
    selected = validate_se3(selected_T_world_eef, "selected_T_world_eef")
    if reconstruction.centroid_camera_m is None or reconstruction.bounds_camera_m is None:
        raise ValueError("placement requires camera-frame reconstruction centroid and bounds")
    source_centroid_camera = np.asarray(reconstruction.centroid_camera_m, dtype=float)
    source_centroid_world = (obs.T_world_camera @ np.r_[source_centroid_camera, 1.0])[:3]
    bounds = np.asarray(reconstruction.bounds_camera_m, dtype=float)
    if bounds.shape != (6,) or np.any(bounds[3:] < bounds[:3]):
        raise ValueError("reconstruction bounds_camera_m must be ordered min/max values")
    corners_camera = np.asarray([[x, y, z] for x in (bounds[0], bounds[3]) for y in (bounds[1], bounds[4]) for z in (bounds[2], bounds[5])], dtype=float)
    corners_world = (obs.T_world_camera @ np.c_[corners_camera, np.ones(8)].T)[:3].T
    support_relative_min = float(np.min((corners_world - source_centroid_world) @ normal))
    desired_centroid_world = support + normal * (float(clearance_m) - support_relative_min)
    centroid_offset_eef = selected[:3, :3].T @ (source_centroid_world - selected[:3, 3])
    pose = selected.copy()
    pose[:3, 3] = desired_centroid_world - selected[:3, :3] @ centroid_offset_eef
    confidence = float(min(surface_conf, reconstruction.confidence))
    return PlacementPlan(pose, tuple(support), tuple(normal), tuple(footprint), confidence)


def deproject_pixel(observation: ZeroGraspObservation, pixel: Sequence[float]) -> np.ndarray:
    u, v = float(pixel[0]), float(pixel[1])
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < observation.depth_m.shape[1] and 0 <= iv < observation.depth_m.shape[0]):
        raise ValueError("pixel outside image")
    z = float(observation.depth_m[iv, iu])
    if not np.isfinite(z) or z <= 0:
        raise ValueError("pixel has no valid metric depth")
    ray = np.linalg.inv(observation.K) @ np.array([u, v, 1.0], dtype=float)
    point_camera = ray * (z / ray[2])
    return (observation.T_world_camera @ np.r_[point_camera, 1.0])[:3]


class ZeroGraspSelectionError(ValueError):
    """Selection failure that retains deterministic filter evidence."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


def select_grasp_result(result: ZeroGraspInferenceResult, observation: ZeroGraspObservation, config: ZeroGraspConfig | None = None) -> dict[str, Any]:
    cfg = validate_config(config)
    masks = build_arrow_seeded_masks(observation, cfg)
    selection_audit: dict[str, Any] = {}
    selection_audit["mask_diagnostics"] = serialize_audit(masks.diagnostics)
    ranked = rank_grasps(result.candidates, observation.T_world_camera, cfg, eef_position_world_m=observation.eef_position_world_m, observation=observation, source_mask=masks.source_mask, reconstruction=result.reconstruction, audit=selection_audit)
    if not ranked:
        raise ZeroGraspSelectionError("no valid ZeroGrasp candidate remains after deterministic filters", selection_audit)
    candidate, T_W_G, clearance = ranked[0]
    T_W_pregrasp_G = make_pregrasp(T_W_G, cfg.pregrasp_distance_m, cfg.approach_axis_local)
    out = {"candidate": candidate, "clearance_m": clearance, "eef_pose_ready": False, "selection_audit": selection_audit}
    # The robot-specific grasp↔EEF translation is not inferred.  Unless an
    # independently verified transform is supplied, return a diagnostic-only
    # grasp selection and force the execution layer to fail closed.
    try:
        T_G_E = dynamic_T_G_E(candidate, cfg)
    except ValueError as exc:
        out["selection_audit"]["eef_rejection"] = str(exc)
        T_G_E = None
    if T_G_E is not None:
        out["T_W_E"] = compose_T_W_E_from_grasp(T_W_G, T_G_E)
        out["T_W_E_pregrasp"] = compose_T_W_E_from_grasp(T_W_pregrasp_G, T_G_E)
        out["eef_pose_ready"] = True
        if result.reconstruction is not None and masks.diagnostics.ok:
            out["placement"] = reconstruction_assisted_placement(observation, result.reconstruction, selected_T_world_eef=out["T_W_E"], destination_mask=masks.destination_mask, clearance_m=cfg.placement_clearance_m)
    return out


def parse_inference_result(payload: Mapping[str, Any], request_hash: str) -> ZeroGraspInferenceResult:
    allowed = {"type", "grasps", "scores", "width_m", "height_m", "depth_m", "clearance_m", "collision_free", "source_index", "eef_frame", "translation_reference", "translation_frame", "rotation_frame", "reconstruction", "diagnostics", "request_hash", "output_hash"}
    if set(payload) - allowed:
        raise ValueError("worker output contains unsupported fields")
    if payload.get("request_hash") != request_hash:
        raise ValueError("worker returned a request hash mismatch")
    matrices, scores = parse_grasp_array({"grasps": payload.get("grasps", []), "scores": payload.get("scores")})
    count = len(matrices)
    def candidate_meta(name: str, default: Any, *, required: bool = False) -> list[Any]:
        if required and (name not in payload or payload[name] is None):
            raise ValueError(f"worker output requires per-grasp {name}")
        value = payload.get(name)
        if value is None:
            return [default] * count
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must contain one value per grasp")
        values = list(value)
        if len(values) != count:
            raise ValueError(f"{name} must contain one value per grasp")
        return values
    widths, heights, depths = candidate_meta("width_m", None), candidate_meta("height_m", None), candidate_meta("depth_m", None)
    clearances = candidate_meta("clearance_m", 0.0)
    collisions = candidate_meta("collision_free", False)
    if any(not isinstance(value, (bool, np.bool_)) for value in collisions):
        raise ValueError("collision_free entries must be actual booleans")
    indexes = candidate_meta("source_index", None)
    eef_frames = candidate_meta("eef_frame", None)
    references = candidate_meta("translation_reference", None, required=True)
    translation_frames = candidate_meta("translation_frame", None, required=True)
    rotation_frames = candidate_meta("rotation_frame", None, required=True)
    for name, values, expected in (
        ("translation_reference", references, GRASPNET_TRANSLATION_REFERENCE),
        ("translation_frame", translation_frames, GRASPNET_CAMERA_FRAME),
        ("rotation_frame", rotation_frames, GRASPNET_CAMERA_FRAME),
    ):
        if any(value != expected for value in values):
            raise ValueError(f"{name} must be exactly {expected}")
    candidates = tuple(GraspCandidate(matrices[i], float(scores[i]), float(clearances[i]), i if indexes[i] is None else int(indexes[i]), None if widths[i] is None else float(widths[i]), None if heights[i] is None else float(heights[i]), None if depths[i] is None else float(depths[i]), bool(collisions[i]), "source", eef_frames[i], references[i], translation_frames[i], rotation_frames[i]) for i in range(count))
    reconstruction = None
    if payload.get("reconstruction") is not None:
        reconstruction_payload = payload["reconstruction"]
        if not isinstance(reconstruction_payload, Mapping):
            raise ValueError("reconstruction must be a mapping")
        if "source_frame" not in reconstruction_payload or reconstruction_payload["source_frame"] is None:
            raise ValueError("reconstruction requires explicit source_frame")
        if reconstruction_payload["source_frame"] != "camera":
            raise ValueError("reconstruction source_frame must be exactly camera")
        reconstruction = ReconstructionSummary(tuple(float(x) for x in reconstruction_payload["dimensions_m"]), float(reconstruction_payload.get("confidence", 0.0)), "camera", reconstruction_payload.get("centroid_camera_m"), reconstruction_payload.get("bounds_camera_m"))
    output_hash = str(payload.get("output_hash") or stable_json_hash(serialize_audit(payload)))
    return ZeroGraspInferenceResult(candidates, reconstruction, request_hash, output_hash, payload.get("diagnostics", {}))


def preflight_zerograsp(config: ZeroGraspConfig | Mapping[str, Any] | None = None, command: Sequence[str] | None = None) -> dict[str, Any]:
    cfg = validate_config(config)
    errors: list[str] = []
    paths = resolve_external_paths(cfg)
    if not command:
        errors.append("an explicit external worker command is required")
    for name, path in paths.items():
        if path is not None and not Path(path).exists():
            errors.append(f"{name} does not exist: {path}")
    if not cfg.entrypoint and not all(paths.values()):
        errors.append("entrypoint is required for the external worker")
    effective_command = tuple(str(part) for part in (command or ()))
    return {"ok": not errors, "errors": errors, "seed": cfg.fixed_seed, "model_input": [cfg.model_height, cfg.model_width], **paths, "entrypoint": cfg.entrypoint, "runtime_hash": runtime_identity_hash("zerograsp-jsonl-v1", effective_command, cfg)}


def resolve_external_paths(config: ZeroGraspConfig | Mapping[str, Any] | None = None) -> dict[str, str | None]:
    """Resolve runtime locations from explicit ctor/config values or env vars."""
    cfg = validate_config(config)
    return {
        "external_repo": cfg.external_repo or os.environ.get("ZERO_GRASP_ROOT"),
        "checkpoint": cfg.checkpoint or os.environ.get("ZERO_GRASP_CHECKPOINT"),
        "runtime_config": cfg.runtime_config or os.environ.get("ZERO_GRASP_CONFIG"),
    }


def _path_identity(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return {"path": str(target), "exists": False}
    if target.is_file():
        digestor = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digestor.update(chunk)
        digest = digestor.hexdigest()
        return {"path": str(target), "kind": "file", "sha256": digest, "size": target.stat().st_size}
    stat = target.stat()
    return {"path": str(target), "kind": "directory", "mtime_ns": stat.st_mtime_ns}


def runtime_identity_hash(protocol: str, command: Sequence[str], config: ZeroGraspConfig | Mapping[str, Any] | None = None) -> str:
    cfg = validate_config(config)
    paths = resolve_external_paths(cfg)
    identity = {"protocol": str(protocol), "command": [str(part) for part in command], "external_repo": _path_identity(paths["external_repo"]), "checkpoint": _path_identity(paths["checkpoint"]), "runtime_config": _path_identity(paths["runtime_config"]), "entrypoint": cfg.entrypoint, "calibration_sha256": cfg.calibration_sha256, "probe_sha256": cfg.probe_sha256, "calibration_source": cfg.calibration_source, "translation_rule": cfg.translation_rule, "R_grasp_eef": None if cfg.R_grasp_eef is None else hash_array(cfg.R_grasp_eef), "R_H_E": None if cfg.R_H_E is None else hash_array(cfg.R_H_E)}
    return stable_json_hash(identity)


class ZeroGraspProcessAdapter:
    """Long-lived JSONL process adapter with bounded requests and fail-closed errors."""

    protocol = "zerograsp-jsonl-v1"

    def __init__(self, command: Sequence[str], config: ZeroGraspConfig | Mapping[str, Any] | None = None, *, cwd: str | os.PathLike[str] | None = None, timeout_s: float | None = None):
        self.config = validate_config(config)
        if not command:
            raise ValueError("command must be a non-empty external worker command")
        self.command = tuple(str(part) for part in command)
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self.timeout_s = float(timeout_s if timeout_s is not None else self.config.request_timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self.stderr_tail: list[str] = []
        self._request_counter = 0
        self.last_audit: dict[str, Any] = {}
        self.runtime_hash = runtime_identity_hash(self.protocol, self.command, self.config)

    def preflight(self) -> dict[str, Any]:
        return preflight_zerograsp(self.config, self.command)

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        env = os.environ.copy()
        env["ZERO_GRASP_SEED"] = str(self.config.fixed_seed)
        env["ZEROGRASP_FIXED_SEED"] = str(self.config.fixed_seed)
        env["PYTHONHASHSEED"] = str(self.config.fixed_seed)
        self._process = subprocess.Popen(self.command, cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()
        self.stderr_tail = []
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()
        self._send({"type": "handshake", "protocol": self.protocol, "seed": self.config.fixed_seed, "config": serialize_audit(self.config)})
        reply = self._receive()
        if reply.get("type") != "ready" or reply.get("protocol") != self.protocol:
            self.close()
            raise RuntimeError(f"ZeroGrasp worker handshake failed: {reply}")

    def _read_lines(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                if len(line.encode("utf-8", errors="replace")) > self.config.max_json_line_bytes:
                    self._lines.put("__ZEROGRASP_OVERSIZE__")
                    return
                self._lines.put(line)
        finally:
            self._lines.put("")

    def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        for line in self._process.stderr:
            self.stderr_tail.append(line.rstrip()[:1000])
            if len(self.stderr_tail) > 64:
                del self.stderr_tail[0]

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.poll() is not None or self._process.stdin is None:
            raise RuntimeError("ZeroGrasp worker is not running")
        self._process.stdin.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("ZeroGrasp worker response timed out") from exc
        if not line:
            code = None if self._process is None else self._process.poll()
            self.close()
            raise RuntimeError(f"ZeroGrasp worker exited before responding (returncode={code})")
        if line == "__ZEROGRASP_OVERSIZE__":
            self.close()
            raise RuntimeError("ZeroGrasp worker response exceeded max_json_line_bytes")
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise RuntimeError("ZeroGrasp worker emitted invalid JSONL") from exc
        if not isinstance(result, dict):
            raise RuntimeError("ZeroGrasp worker response must be a JSON object")
        return result

    def prepare(self, observation: ZeroGraspObservation) -> PreparedZeroGraspInput:
        return prepare_zero_grasp_input(_check_observation(observation), self.config)

    def infer(self, observation: ZeroGraspObservation) -> ZeroGraspInferenceResult:
        if not self.config.eef_calibration_verified:
            raise RuntimeError("ZeroGraspProcessAdapter.infer requires verified EEF calibration before worker I/O")
        obs = _check_observation(observation)
        prepared = self.prepare(obs)
        self.start()
        self._request_counter += 1
        payload = {"type": "infer", "request_id": self._request_counter, "request_hash": prepared.request_hash, "clean_rgb": encode_array(prepared.clean_rgb), "depth_m": encode_array(prepared.depth_m), "source_mask": encode_array(prepared.source_mask), "destination_mask": encode_array(prepared.destination_mask), "K_model": encode_array(prepared.K_model), "source_px_model": prepared.source_px_model, "destination_px_model": prepared.destination_px_model, "fixed_seed": self.config.fixed_seed}
        try:
            encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if len(encoded_payload.encode("utf-8")) > self.config.max_json_line_bytes:
                raise ValueError("ZeroGrasp request exceeds max_json_line_bytes")
            self.last_audit = {"protocol": self.protocol, "request_id": self._request_counter, "request_hash": prepared.request_hash, "runtime_hash": self.runtime_hash, "worker_command": self.command, "fixed_seed": self.config.fixed_seed, "preprocessing": {"pixel_affine": prepared.pixel_affine.tolist(), "vertically_flipped": prepared.vertically_flipped, "source_mask_hash": prepared.source_mask_hash, "destination_mask_hash": prepared.destination_mask_hash, "source_mask_area": prepared.source_mask_area, "destination_mask_area": prepared.destination_mask_area, "mask_rejections": list(prepared.mask_rejections)}}
            self._send(payload)
            reply = self._receive()
            if reply.get("type") == "error":
                raise RuntimeError(str(reply.get("error", "external ZeroGrasp error")))
            if reply.get("type") != "result":
                raise RuntimeError(f"unexpected ZeroGrasp response type: {reply.get('type')}")
            result = parse_inference_result(reply, prepared.request_hash)
            self.last_audit.update({"output_hash": result.output_hash, "worker_diagnostics": serialize_audit(result.diagnostics), "candidate_count": len(result.candidates), "status": "ok"})
            return result
        except Exception as exc:
            self.last_audit.update({"status": "failed", "error": str(exc)})
            if self._process is not None and self._process.poll() is not None:
                self.close()
            raise

    def select(self, result: ZeroGraspInferenceResult, observation: ZeroGraspObservation) -> dict[str, Any]:
        try:
            selected = select_grasp_result(result, _check_observation(observation), self.config)
        except ZeroGraspSelectionError as exc:
            self.last_audit.update({"status": "failed", "error": str(exc), "selection_audit": serialize_audit(exc.audit)})
            raise
        self.last_audit.update(serialize_audit(selected))
        return selected

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for thread in (self._reader, self._stderr_reader):
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "ZeroGraspProcessAdapter":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_zerograsp_adapter(command: Sequence[str] | None = None, config: ZeroGraspConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> ZeroGraspProcessAdapter:
    cfg = validate_config(config)
    if command is None:
        python = os.environ.get("ZERO_GRASP_PYTHON")
        if not python:
            raise ValueError("command is required, or ZERO_GRASP_PYTHON must point to the external runtime")
        command = [python, "-m", "vla_benchmarking.zerograsp_worker"]
        if cfg.entrypoint:
            command += ["--entrypoint", cfg.entrypoint]
        paths = resolve_external_paths(cfg)
        for flag, key in (("--root", "external_repo"), ("--checkpoint", "checkpoint"), ("--config", "runtime_config")):
            if paths[key]:
                command += [flag, str(paths[key])]
    return ZeroGraspProcessAdapter(command, cfg, **kwargs)


__all__ = [name for name in globals() if not name.startswith("_")]
