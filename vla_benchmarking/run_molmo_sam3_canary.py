#!/usr/bin/env python3
"""Experimental MolmoPoint/SAM3 grasp canary runner.

This module is deliberately outside the frozen v9d entry points.  It provides
the small adapter contract needed by a model worker and owns only experimental
bookkeeping, camera provenance, candidate ordering, and bounded retries.  The
worker is given RGB-D captures and calibration, never a simulator handle,
object pose, or evaluator.

The production episode adapter is injected by the caller.  This keeps ordinary
v9d runs model-free and makes the first Legion canary runnable with a fake
worker before loading either large model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    from .controller_configs import ACTIVE_CONTROLLER_CONFIG_FILENAME
except ImportError:  # pragma: no cover - direct script use
    from controller_configs import ACTIVE_CONTROLLER_CONFIG_FILENAME

BASELINE_COMMIT = "fd24a4c5cf8da4991013ab18b15704523ad0836b"
SAM_EXPERIMENT_SCHEMA = "molmo_sam3_canary.v1"
RGBD_EXPERIMENT_SCHEMA = "molmo_rgbd_region_canary.v1"
EXPERIMENT_SCHEMA = SAM_EXPERIMENT_SCHEMA
AGENTVIEW = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"
MAX_CANDIDATES = 128
MAX_ATTEMPTS = 4
YAW_OFFSETS_DEG = (-15.0, 0.0, 15.0)
INSERTION_OFFSETS_M = (0.0, 0.004, 0.008)
OBSERVATION_HOVER_OFFSET_M = 0.10
GRIPPER_OPEN_TIMEOUT_STEPS = 160
DEFAULT_STALL_DELTA_M = 1e-4
SAM3_SOURCE_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
SAM3_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
MOLMOPOINT_MODEL_ID = "allenai/MolmoPoint-8B"
MOLMOPOINT_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
MOLMOPOINT_PROMPT_IDS = ("rim_contact", "rim_downward_approach", "rim_clearance")

MOTION_PROFILE_NAMES = (
    "baseline", "placement_micro5mm", "release_plus20mm", "release20_visual_xy",
)
SCIENTIFIC_IDENTITY_SCHEMA = "molmo_sam3_scientific_identity.v1"
PLACEMENT_MICRO_CORRECTION_PARAMS = {
    "enabled": True,
    "phases": ("preplace", "descend_place"),
    "plateau_window_steps": 10,
    "plateau_delta_m": 0.0001,
    "residual_max_m": 0.005,
    "correction_gain": 1.0,
    "burst_steps": 8,
    "max_rounds": 1,
    "max_actions": 16,
}

OBSERVATION_PROFILE_NAMES = ("baseline", "hover20mm")
OBSERVATION_PROFILE_PARAMS = {
    "baseline": {
        "phase_tolerance_m": 0.015,
        "orientation_tolerance_rad": 0.12,
        "target_offset_m": OBSERVATION_HOVER_OFFSET_M,
        "min_actual_height_margin_m": None,
        "require_actual_height_margin": False,
    },
    "hover20mm": {
        "phase_tolerance_m": 0.020,
        "orientation_tolerance_rad": 0.12,
        "target_offset_m": OBSERVATION_HOVER_OFFSET_M,
        "min_actual_height_margin_m": 0.080,
        "require_actual_height_margin": True,
    },
}


def resolve_motion_profile(name: str, *, region_backend: str) -> dict[str, Any]:
    """Resolve the bounded motion treatment without changing baseline knobs."""
    if name not in MOTION_PROFILE_NAMES:
        raise ValueError(f"motion profile must be one of {MOTION_PROFILE_NAMES}")
    if name == "placement_micro5mm" and region_backend != "rgbd":
        raise ValueError("placement_micro5mm requires --region-backend rgbd")
    return {
        "name": name,
        "region_backend": region_backend,
        "micro_correction": (
            dict(PLACEMENT_MICRO_CORRECTION_PARAMS)
            if name == "placement_micro5mm" else None
        ),
        "release_height_offset_m": 0.020 if name in {"release_plus20mm", "release20_visual_xy"} else 0.0,
        "transfer_xy_policy": "visual_endpoints" if name == "release20_visual_xy" else "legacy_displacement",
    }


def _candidate_motion_kwargs(
    resolved_motion_profile: Mapping[str, Any],
    *,
    experimental_micro_correction: Any | None = None,
    motion_diagnostics: bool = False,
) -> dict[str, Any]:
    """Build candidate-only experimental arguments from a resolved profile."""
    kwargs: dict[str, Any] = {}
    if experimental_micro_correction is not None:
        kwargs["experimental_micro_correction"] = experimental_micro_correction
    if motion_diagnostics:
        kwargs["experimental_motion_diagnostics"] = True
    release_offset = float(resolved_motion_profile.get("release_height_offset_m", 0.0))
    if release_offset != 0.0:
        kwargs["experimental_release_height_offset_m"] = release_offset
    if resolved_motion_profile.get("transfer_xy_policy") == "visual_endpoints":
        kwargs["experimental_transfer_xy_policy"] = "visual_endpoints"
    return kwargs


def resolve_observation_profile(name: str) -> dict[str, Any]:
    """Resolve the bounded observation-hover policy independently of grasp motion."""
    if name not in OBSERVATION_PROFILE_NAMES:
        raise ValueError(f"observation profile must be one of {OBSERVATION_PROFILE_NAMES}")
    return {"name": name, **dict(OBSERVATION_PROFILE_PARAMS[name])}


def _stable_model_identity(model_provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep model identity fields while excluding paths, timings, and load state."""
    source = model_provenance if isinstance(model_provenance, Mapping) else {}
    keys = (
        "sam3_source_commit", "sam3_checkpoint_sha256",
        "molmopoint_model", "molmopoint_revision", "molmopoint_prompt_id",
        "molmopoint_prompt", "model_id", "model_revision", "prompt_id", "prompt",
        "transformers_version", "dtype", "device", "device_map", "max_new_tokens",
        "max_points", "padding_side", "config_sha256",
    )
    return {key: _json_safe(source[key]) for key in keys if key in source}


def build_scientific_identity(
    *,
    execution_sha: str,
    controller_config_digest: str,
    model_provenance: Mapping[str, Any] | None,
    variant: str,
    candidate_policy: str,
    camera_name: str,
    region_backend: str,
    backend: str,
    motion_profile: str,
    motion_profile_params: Mapping[str, Any],
    motion_diagnostics: bool,
    observation_profile: str,
    observation_profile_params: Mapping[str, Any],
    task_ids: Sequence[int],
    seed_base: int,
    suite_modes: Sequence[str],
) -> tuple[dict[str, Any], str]:
    """Build a stable scientific identity, excluding operational run metadata."""
    payload = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "execution_sha": str(execution_sha),
        "controller_config_digest": str(controller_config_digest),
        "model": _stable_model_identity(model_provenance),
        "variant": {
            "name": str(variant), "candidate_policy": str(candidate_policy),
            "camera_name": str(camera_name),
        },
        "backend": {"region_backend": str(region_backend), "backend": str(backend)},
        "motion": {
            "profile": str(motion_profile),
            "params": _json_safe(motion_profile_params),
            "diagnostics": bool(motion_diagnostics),
        },
        "observation": {
            "profile": str(observation_profile),
            "params": _json_safe(observation_profile_params),
        },
        "task_seed": {"task_ids": [int(value) for value in task_ids], "seed_base": int(seed_base)},
        "suite_modes": [str(value) for value in suite_modes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def _execution_provenance(*, repo_root: Path | None = None, require_clean: bool) -> dict[str, Any]:
    """Read the actual worktree SHA and enforce cleanliness for live execution."""
    root = (repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root,
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to resolve execution worktree provenance: {exc}") from exc
    sha = sha_result.stdout.strip()
    status_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise RuntimeError("execution worktree did not provide a valid HEAD SHA")
    if require_clean and status_lines:
        raise RuntimeError("live execution requires a clean checkout")
    return {
        "execution_sha": sha.lower(),
        "checkout_clean": not bool(status_lines),
        "live_verified": bool(require_clean and not status_lines),
        "provenance_status": "live_verified" if require_clean else "dry_run_unverified",
        "dirty_paths": status_lines if not require_clean else [],
    }


@dataclass(frozen=True)
class CanaryVariant:
    """One intentionally explicit treatment in the canary screen."""

    name: str
    camera_name: str
    policy: str
    uses_molmo: bool
    region_backend: str = "sam3"

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("variant name must be a safe non-empty label")
        if self.camera_name not in {AGENTVIEW, WRIST_CAMERA}:
            raise ValueError(f"unsupported canary camera {self.camera_name!r}")
        if self.policy not in {"geometry_only", "molmo_local", "molmo_dense"}:
            raise ValueError(f"unsupported canary policy {self.policy!r}")
        if self.region_backend not in {"sam3", "rgbd"}:
            raise ValueError(f"unsupported region backend {self.region_backend!r}")


VARIANTS: dict[str, CanaryVariant] = {
    "sam3_geometry_agentview": CanaryVariant("sam3_geometry_agentview", AGENTVIEW, "geometry_only", False),
    "molmo_local_agentview": CanaryVariant("molmo_local_agentview", AGENTVIEW, "molmo_local", True),
    "molmo_dense_agentview": CanaryVariant("molmo_dense_agentview", AGENTVIEW, "molmo_dense", True),
    "molmo_dense_wrist": CanaryVariant("molmo_dense_wrist", WRIST_CAMERA, "molmo_dense", True),
    "rgbd_geometry_agentview": CanaryVariant("rgbd_geometry_agentview", AGENTVIEW, "geometry_only", False, "rgbd"),
}


@dataclass(frozen=True)
class GraspCandidate:
    """A fully specified, executable grasp proposal in world coordinates."""

    candidate_id: str
    position_world_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    opening_m: float
    source_pixel_xy: tuple[float, float] | None = None
    score: float = 0.0
    feasible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Explicit approach target from the geometry worker.  ``None`` is retained
    # for old test fixtures; live Molmo/SAM3 candidates populate this field.
    pregrasp_world_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.position_world_m, dtype=np.float64)
        q = np.asarray(self.orientation_xyzw, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("candidate position must be a finite world-frame 3-vector")
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
            raise ValueError("candidate orientation must be a finite nonzero xyzw quaternion")
        if not math.isfinite(float(self.opening_m)) or not 0.0 <= float(self.opening_m) <= 0.20:
            raise ValueError("candidate opening_m must be in [0, 0.20]")
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.source_pixel_xy is not None:
            uv = np.asarray(self.source_pixel_xy, dtype=np.float64)
            if uv.shape != (2,) or not np.isfinite(uv).all():
                raise ValueError("source_pixel_xy must be a finite 2-vector")
        if self.pregrasp_world_m is not None:
            pregrasp = np.asarray(self.pregrasp_world_m, dtype=np.float64)
            if pregrasp.shape != (3,) or not np.isfinite(pregrasp).all():
                raise ValueError("pregrasp_world_m must be a finite world-frame 3-vector")

    def canonical(self) -> dict[str, Any]:
        q = np.asarray(self.orientation_xyzw, dtype=np.float64)
        q = q / np.linalg.norm(q)
        result = {
            "candidate_id": self.candidate_id,
            "position_world_m": [float(v) for v in self.position_world_m],
            "orientation_xyzw": [float(v) for v in q],
            "opening_m": float(self.opening_m),
            "source_pixel_xy": None if self.source_pixel_xy is None else [float(v) for v in self.source_pixel_xy],
            "score": float(self.score),
            "feasible": bool(self.feasible),
            "metadata": _json_safe(self.metadata),
        }
        if self.pregrasp_world_m is not None:
            result["pregrasp_world_m"] = [float(v) for v in self.pregrasp_world_m]
        return result


@dataclass(frozen=True)
class PerceptionRequest:
    """Worker input; all image/calibration data belong to the same capture."""

    variant: CanaryVariant
    agentview_capture: Any
    source_capture: Any
    source_uv: tuple[float, float]
    destination_uv: tuple[float, float] | None
    previous_candidate_ids: tuple[str, ...]
    output_dir: Path
    placement_displacement_world_m: tuple[float, float, float] | None = None


class PerceptionWorker(Protocol):
    def propose(self, request: PerceptionRequest) -> Any:
        """Return executable candidates or ``{candidates, diagnostics}``."""


class EpisodeRunner(Protocol):
    def __call__(self, *, context: "CanaryEpisodeContext", evaluator: Callable[[Any], bool] | None) -> Mapping[str, Any]:
        """Execute one candidate and call ``context.mark_retreat_complete`` first."""


@dataclass
class CanaryEpisodeContext:
    variant: CanaryVariant
    candidate: GraspCandidate
    agentview_capture: Any
    source_capture: Any
    source_uv: tuple[float, float]
    destination_uv: tuple[float, float] | None
    placement_displacement_world_m: tuple[float, float, float] | None
    output_dir: Path
    attempt_index: int
    candidate_index: int
    _retreat_complete: bool = False

    def mark_retreat_complete(self) -> None:
        self._retreat_complete = True


class _RetreatGatedEvaluator:
    def __init__(self, evaluator: Callable[[Any], bool], context: CanaryEpisodeContext):
        self._evaluator = evaluator
        self._context = context
        self.called = False

    def __call__(self, env: Any) -> bool:
        if not self._context._retreat_complete:
            raise RuntimeError("evaluator called before retreat completed")
        self.called = True
        result = self._evaluator(env)
        if not isinstance(result, bool):
            raise TypeError(f"evaluator must return bool, got {type(result).__name__}")
        return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _capture_provenance(capture: Any) -> dict[str, Any]:
    calibration = getattr(capture, "calibration", None)
    if calibration is None:
        raise ValueError("capture is missing calibration provenance")
    camera_name = str(getattr(calibration, "camera_name", ""))
    if not camera_name:
        raise ValueError("capture calibration is missing camera_name")
    rgb = np.asarray(getattr(capture, "rgb", None))
    metric = np.asarray(getattr(capture, "metric_depth", None))
    if rgb.ndim != 3 or metric.ndim != 2 or rgb.shape[:2] != metric.shape:
        raise ValueError("capture RGB and metric depth are not aligned")
    return {
        "camera_name": camera_name,
        "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "pixel_origin": getattr(calibration, "pixel_origin", None),
        "camera_frame": getattr(calibration, "camera_frame", None),
        "world_frame": getattr(calibration, "world_frame", None),
        "extrinsic_direction": getattr(calibration, "extrinsic_direction", None),
        "rgb_depth_alignment": getattr(calibration, "rgb_depth_alignment", None),
    }


def _observation_capture_provenance(capture: Any) -> dict[str, Any]:
    """Record the observed capture and the exact world-from-camera transform."""
    calibration = getattr(capture, "calibration", None)
    if calibration is None:
        raise ValueError("observation capture is missing calibration")
    K = np.asarray(getattr(calibration, "intrinsic", None), dtype=np.float64)
    T = np.asarray(getattr(calibration, "world_from_camera", None), dtype=np.float64)
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.all(np.isfinite(K)) or not np.all(np.isfinite(T)):
        raise ValueError("observation capture calibration has invalid K/T")
    rgb = np.asarray(getattr(capture, "rgb", None))
    depth = np.asarray(getattr(capture, "metric_depth", None))
    if rgb.ndim != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("observation capture RGB and metric depth are not aligned")
    return {
        "camera_name": getattr(calibration, "camera_name", None),
        "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
        "pixel_origin": getattr(calibration, "pixel_origin", None),
        "camera_frame": getattr(calibration, "camera_frame", None),
        "world_frame": getattr(calibration, "world_frame", None),
        "extrinsic_direction": getattr(calibration, "extrinsic_direction", "world_from_camera"),
        "rgb_depth_alignment": getattr(calibration, "rgb_depth_alignment", None),
        "intrinsic_matrix": K.tolist(),
        "world_from_camera": T.tolist(),
        "transform_contract": {
            "source_frame": "camera",
            "destination_frame": "world",
            "direction": "world_from_camera",
            "units": "meters",
            "pixel_origin": getattr(calibration, "pixel_origin", "top_left"),
        },
    }


def _rotation_matrix(value: Any) -> np.ndarray:
    """Convert a proprioceptive quaternion or rotation matrix to SO(3)."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        matrix = array
    elif array.size == 9:
        matrix = array.reshape(3, 3)
    else:
        flat = array.reshape(-1)
        if flat.size != 4:
            raise ValueError("orientation must be a quaternion or 3x3 rotation")
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("orientation quaternion is non-finite or zero")
        x, y, z, w = flat / norm
        matrix = np.asarray(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                             (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                             (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("orientation rotation is non-finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-4):
        raise ValueError("orientation must be a proper rotation")
    return matrix


def _orientation_error_rad(actual: Any, target: Any) -> float:
    relative = _rotation_matrix(actual).T @ _rotation_matrix(target)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def _normalise_candidate(value: Any, index: int) -> GraspCandidate:
    if isinstance(value, GraspCandidate):
        return value
    # ``molmo_sam3.grasp_candidates.GraspCandidate`` is intentionally a richer
    # geometry result.  Keep this adapter local so the frozen controller never
    # imports the experimental package, while preserving its complete audit.
    if hasattr(value, "grip_site_world_m"):
        position = getattr(value, "grip_site_world_m")
        pregrasp = getattr(value, "pregrasp_world_m", None)
        orientation = getattr(value, "quaternion_world_grip_site_xyzw")
        audit = getattr(value, "audit", {})
        metadata = dict(audit) if isinstance(audit, Mapping) else {}
        for key in ("yaw_deg", "insertion_depth_m", "clearance_m", "depth_support_count"):
            if hasattr(value, key):
                metadata[key] = getattr(value, key)
        return GraspCandidate(
            candidate_id=str(getattr(value, "candidate_id", f"candidate_{index:03d}")),
            position_world_m=tuple(float(v) for v in position),
            orientation_xyzw=tuple(float(v) for v in orientation),
            opening_m=float(getattr(value, "required_aperture_m", 0.04)),
            source_pixel_xy=tuple(float(v) for v in getattr(value, "source_pixel_uv")),
            score=float(getattr(value, "score", 0.0)),
            feasible=True,
            metadata=metadata,
            pregrasp_world_m=None if pregrasp is None else tuple(float(v) for v in pregrasp),
        )
    if not isinstance(value, Mapping):
        raise TypeError(f"candidate {index} must be a mapping or GraspCandidate")
    position = value.get("position_world_m", value.get("position_world"))
    orientation = value.get("orientation_xyzw", value.get("quaternion_xyzw", value.get("orientation")))
    if position is None or orientation is None:
        raise ValueError(f"candidate {index} must include position_world_m and orientation_xyzw")
    return GraspCandidate(
        candidate_id=str(value.get("candidate_id", f"candidate_{index:03d}")),
        position_world_m=tuple(float(v) for v in position),
        orientation_xyzw=tuple(float(v) for v in orientation),
        opening_m=float(value.get("opening_m", value.get("jaw_width_m", 0.04))),
        source_pixel_xy=None if value.get("source_pixel_xy") is None else tuple(float(v) for v in value["source_pixel_xy"]),
        score=float(value.get("score", 0.0)),
        feasible=bool(value.get("feasible", True)),
        metadata=value.get("metadata", {}),
        pregrasp_world_m=(
            None if value.get("pregrasp_world_m") is None
            else tuple(float(v) for v in value["pregrasp_world_m"])
        ),
    )


def rank_candidates(candidates: Sequence[GraspCandidate]) -> list[GraspCandidate]:
    """Stable feasibility-first ordering; never depends on model list order."""
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique within one proposal")
    return sorted(
        candidates,
        key=lambda c: (
            not bool(c.feasible),
            -float(c.score),
            -float(c.metadata.get("clearance_m", 0.0)),
            -float(c.metadata.get("depth_support", 0.0)),
            float(c.metadata.get("position_distance_m", c.metadata.get("motion_distance_m", 0.0))),
            float(c.metadata.get("rotation_distance_rad", 0.0)),
            c.candidate_id,
        ),
    )


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray((aw * bx + ax * bw + ay * bz - az * by,
                       aw * by - ax * bz + ay * bw + az * bx,
                       aw * bz + ax * by - ay * bx + az * bw,
                       aw * bw - ax * bx - ay * by - az * bz), dtype=np.float64)


def expand_grasp_candidates(
    seeds: Sequence[GraspCandidate],
    *,
    yaw_offsets_deg: Sequence[float] = YAW_OFFSETS_DEG,
    insertion_offsets_m: Sequence[float] = INSERTION_OFFSETS_M,
    max_candidates: int = MAX_CANDIDATES,
) -> list[GraspCandidate]:
    """Expand up to 16 worker seeds into bounded yaw/insertion proposals.

    The first campaign assumes a downward approach, so insertion is expressed
    along world ``+Z`` in this adapter.  A worker that has a calibrated local
    approach axis may supply already-expanded candidates instead.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    expanded: list[GraspCandidate] = []
    for seed in seeds[:16]:
        p = np.asarray(seed.position_world_m, dtype=np.float64)
        q = np.asarray(seed.orientation_xyzw, dtype=np.float64)
        for yaw in yaw_offsets_deg:
            if not math.isfinite(float(yaw)):
                raise ValueError("yaw offsets must be finite")
            half = math.radians(float(yaw)) / 2.0
            yaw_q = np.asarray((0.0, 0.0, math.sin(half), math.cos(half)), dtype=np.float64)
            rotated = _quaternion_multiply(yaw_q, q)
            for insertion in insertion_offsets_m:
                if not math.isfinite(float(insertion)) or float(insertion) < 0.0:
                    raise ValueError("insertion offsets must be finite and non-negative")
                meta = dict(seed.metadata)
                meta.update({"seed_id": seed.candidate_id, "yaw_offset_deg": float(yaw), "insertion_offset_m": float(insertion)})
                expanded.append(GraspCandidate(
                    candidate_id=f"{seed.candidate_id}__yaw{yaw:g}__ins{insertion:g}",
                    position_world_m=tuple((p + np.asarray((0.0, 0.0, float(insertion)))).tolist()),
                    orientation_xyzw=tuple(rotated.tolist()),
                    opening_m=seed.opening_m,
                    source_pixel_xy=seed.source_pixel_xy,
                    score=seed.score,
                    feasible=seed.feasible,
                    metadata=meta,
                    pregrasp_world_m=seed.pregrasp_world_m,
                ))
                if len(expanded) >= max_candidates:
                    return rank_candidates(expanded)
    return rank_candidates(expanded)


def _worker_candidates(worker: PerceptionWorker, request: PerceptionRequest) -> tuple[list[GraspCandidate], Mapping[str, Any]]:
    started = time.perf_counter()
    raw = worker.propose(request)
    perception_latency_s = float(time.perf_counter() - started)
    diagnostics: Mapping[str, Any] = {}
    if hasattr(raw, "candidates"):
        diagnostics = getattr(raw, "audit", {}) if isinstance(getattr(raw, "audit", {}), Mapping) else {}
        raw = getattr(raw, "candidates")
    if isinstance(raw, Mapping):
        diagnostics = raw.get("diagnostics", {}) if isinstance(raw.get("diagnostics", {}), Mapping) else {}
        raw = raw.get("candidates", ())
    values = list(raw or ())
    if len(values) > MAX_CANDIDATES:
        values = values[:MAX_CANDIDATES]
    candidates = [_normalise_candidate(value, i) for i, value in enumerate(values)]
    attempted = set(request.previous_candidate_ids)
    if attempted:
        candidates = [candidate for candidate in candidates if candidate.candidate_id not in attempted]
        diagnostics = {**dict(diagnostics), "filtered_attempted_candidates": len(values) - len(candidates)}
    diagnostics = {**dict(diagnostics), "perception_latency_s": perception_latency_s}
    return rank_candidates(candidates), diagnostics


def _failed_grasp(result: Mapping[str, Any]) -> bool:
    if "grasp_retained" in result:
        return not bool(result["grasp_retained"])
    status = str(result.get("status", "")).lower()
    return status in {
        "grasp_failed", "empty_gripper_likely", "post_lift_retention_failed",
        "candidate_failed", "runner_error", "recovery_failed", "retry",
    }


def _completed_motion_phases(phases: Any) -> list[str]:
    """Return only phases that actually reached their completion boundary."""
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        return []
    return [
        str(item.get("phase"))
        for item in phases
        if isinstance(item, Mapping)
        and item.get("status") in {"reached", "dwell", "stop"}
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolved_controller_config_digest(path: Path) -> str:
    """Return the semantic digest of the fully resolved controller config."""
    try:
        try:
            from .controller_configs import load_controller_config
        except ImportError:  # pragma: no cover - direct script use
            from controller_configs import load_controller_config
        resolved = load_controller_config(path)
    except Exception as exc:
        raise RuntimeError(f"unable to resolve controller config for identity: {path}") from exc
    digest = str(resolved.get("config_hash", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise RuntimeError("resolved controller config did not provide a valid semantic digest")
    return digest.lower()


def run_canary_episode(
    *,
    env: Any,
    task_id: int,
    seed: int,
    output_dir: str | Path,
    variant: CanaryVariant | str,
    worker: PerceptionWorker,
    episode_runner: EpisodeRunner,
    source_uv: Sequence[float],
    destination_uv: Sequence[float] | None = None,
    placement_displacement_world_m: Sequence[float] | None = None,
    evaluator: Callable[[Any], bool] | None = None,
    capture_fn: Callable[..., Any] | None = None,
    resolution: int = 256,
    dry_run: bool = True,
    arrow_rgb: np.ndarray | None = None,
    arrow_refresh_fn: Callable[[Any], tuple[np.ndarray, Sequence[float], Sequence[float] | None]] | None = None,
    experiment_schema: str = SAM_EXPERIMENT_SCHEMA,
    region_backend: str = "sam3",
    before_propose_callback: Callable[[Any], Any] | None = None,
    motion_profile: str = "baseline",
    motion_profile_params: Mapping[str, Any] | None = None,
    motion_diagnostics: bool = False,
    observation_profile: str = "baseline",
    observation_profile_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded canary episode, regenerating candidates after failure."""
    selected_variant = VARIANTS[variant] if isinstance(variant, str) else variant
    if selected_variant.name not in VARIANTS:
        raise ValueError(f"unknown canary variant {selected_variant.name!r}")
    if capture_fn is None:
        try:
            from .run_arrow_pick_place_eval import capture_agentview
        except ImportError:  # pragma: no cover - direct script use
            from run_arrow_pick_place_eval import capture_agentview
        capture_fn = capture_agentview
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    av_capture = capture_fn(env, resolution=resolution, camera_name=AGENTVIEW)
    if arrow_refresh_fn is not None:
        arrow_rgb, source_uv, destination_uv = arrow_refresh_fn(av_capture)
    source_capture = av_capture if selected_variant.camera_name == AGENTVIEW else capture_fn(
        env, resolution=resolution, camera_name=selected_variant.camera_name
    )
    av_provenance = _capture_provenance(av_capture)
    source_provenance = _capture_provenance(source_capture)
    if source_provenance["camera_name"] != selected_variant.camera_name:
        raise ValueError("source capture camera does not match selected variant")
    source = tuple(float(v) for v in source_uv)
    if len(source) != 2 or not np.isfinite(source).all():
        raise ValueError("source_uv must be finite")
    destination = None if destination_uv is None else tuple(float(v) for v in destination_uv)
    displacement = None if placement_displacement_world_m is None else tuple(float(v) for v in placement_displacement_world_m)
    if displacement is not None and (len(displacement) != 3 or not np.isfinite(displacement).all()):
        raise ValueError("placement_displacement_world_m must be finite and length three")
    attempts: list[dict[str, Any]] = []
    previous_ids: tuple[str, ...] = ()
    final_result: Mapping[str, Any] | None = None
    for attempt_index in range(1, MAX_ATTEMPTS + 1):
        if attempt_index > 1:
            av_capture = capture_fn(env, resolution=resolution, camera_name=AGENTVIEW)
            source_capture = av_capture if selected_variant.camera_name == AGENTVIEW else capture_fn(
                env, resolution=resolution, camera_name=selected_variant.camera_name
            )
            if arrow_refresh_fn is not None:
                arrow_rgb, refreshed_source, refreshed_destination = arrow_refresh_fn(av_capture)
                source = tuple(float(v) for v in refreshed_source)
                destination = (
                    None if refreshed_destination is None
                    else tuple(float(v) for v in refreshed_destination)
                )
            av_provenance = _capture_provenance(av_capture)
            source_provenance = _capture_provenance(source_capture)
        request = PerceptionRequest(
            variant=selected_variant,
            agentview_capture=av_capture,
            source_capture=source_capture,
            source_uv=source,
            destination_uv=destination,
            previous_candidate_ids=previous_ids,
            output_dir=root,
            placement_displacement_world_m=displacement,
        )
        # Candidate geometry must use a no-motion probe synchronized with the
        # exact capture that feeds proposal generation.  This is called for
        # the initial proposal and every post-recovery retry.
        if before_propose_callback is not None:
            before_propose_callback(env)
        candidates, diagnostics = _worker_candidates(worker, request)
        candidates = candidates[:MAX_CANDIDATES]
        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "source_camera": selected_variant.camera_name,
            "capture_provenance": {"agentview": av_provenance, "source": source_provenance},
            "candidate_count": len(candidates),
            "candidate_diagnostics": diagnostics,
            "candidates": [candidate.canonical() for candidate in candidates],
            "results": [],
        }
        if not candidates:
            attempt_record["status"] = "no_candidates"
            attempts.append(attempt_record)
            break
        # Exactly one candidate is sent per fresh capture.  This makes the
        # retry contract unambiguous: a failed contact never reuses a stale
        # RGB-D frame or silently tries a second target from that frame.
        candidate = candidates[0]
        context = CanaryEpisodeContext(
            variant=selected_variant, candidate=candidate,
            agentview_capture=av_capture, source_capture=source_capture,
            source_uv=source, destination_uv=destination,
            placement_displacement_world_m=displacement, output_dir=root,
            attempt_index=attempt_index, candidate_index=0,
        )
        gated = None if dry_run or evaluator is None else _RetreatGatedEvaluator(evaluator, context)
        runner_kwargs: dict[str, Any] = {"context": context, "evaluator": gated}
        try:
            parameters = inspect.signature(episode_runner).parameters
            accepts_retreat_callback = (
                any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
                or "retreat_completed_callback" in parameters
            )
        except (TypeError, ValueError):
            accepts_retreat_callback = False
        if accepts_retreat_callback:
            runner_kwargs["retreat_completed_callback"] = context.mark_retreat_complete
        try:
            result = dict(episode_runner(**runner_kwargs))
        except Exception as exc:
            item = {"candidate_id": candidate.candidate_id, "status": "runner_error", "error_type": type(exc).__name__, "error": str(exc)}
            attempt_record["results"].append(item)
            _write_json(root / "attempts" / f"attempt_{attempt_index:02d}.json", attempt_record)
            raise
        if result.get("retreat_complete"):
            context.mark_retreat_complete()
        # Keep the phase boundary explicit in the attempt record.  The runner
        # may append recovery phases after a failed candidate; callers need the
        # candidate's own completed phases for gating and diagnosis.
        result.setdefault(
            "motion_phases_reached",
            _completed_motion_phases(result.get("audit", {}).get("phases", []) if isinstance(result.get("audit"), Mapping) else ()),
        )
        if result.get("evaluator_called") and not context._retreat_complete:
            raise RuntimeError("episode runner reports evaluator use before retreat")
        item = {"candidate_id": candidate.candidate_id, **result}
        attempt_record["results"].append(item)
        attempt_record["motion_phases_reached"] = list(result.get("motion_phases_reached", []))
        final_result = result
        if str(result.get("status", "")).lower() == "recovery_failed":
            attempt_record["status"] = "recovery_failed"
            attempts.append(attempt_record)
            # Fail closed while preserving a complete manifest for the matrix;
            # no subsequent candidate is allowed on an uncertain robot state.
            break
        if not _failed_grasp(result):
            attempt_record["status"] = "selected"
            attempts.append(attempt_record)
            break
        previous_ids = tuple(list(previous_ids) + [candidate.candidate_id])
        attempt_record["status"] = "candidate_failed"
        attempts.append(attempt_record)
    manifest = {
        "schema_version": str(experiment_schema),
        "experiment_id": f"{selected_variant.name}__task{int(task_id)}__seed{int(seed)}",
        "baseline_commit": BASELINE_COMMIT,
        "frozen_controller": "v9d",
        "variant": asdict(selected_variant),
        "task_id": int(task_id), "seed": int(seed), "resolution": int(resolution),
        "dry_run": bool(dry_run), "max_attempts": MAX_ATTEMPTS,
        "camera_provenance": {"agentview": av_provenance, "source": source_provenance},
        "placement_displacement_world_m": displacement,
        "attempts": attempts,
        "final_result": final_result,
        "total_actions": (int(final_result["total_actions"]) if isinstance(final_result, Mapping) and final_result.get("total_actions") is not None else None),
        "evaluator_timing": "after_retreat_only",
        "region_backend": str(region_backend),
        "backend": "v9d_rgbd_region" if region_backend == "rgbd" else "sam3",
        "sam3_used": bool(region_backend == "sam3"),
        "motion_profile": motion_profile,
        "motion_profile_params": _json_safe(motion_profile_params or {}),
        "motion_diagnostics": bool(motion_diagnostics),
        "observation_profile": observation_profile,
        "observation_profile_params": _json_safe(observation_profile_params or {}),
    }
    digest = hashlib.sha256(json.dumps(_json_safe(manifest), sort_keys=True).encode()).hexdigest()
    manifest["experiment_config_hash"] = digest
    _write_json(root / "molmo_sam3_canary_manifest.json", manifest)
    return manifest


def build_local_model_runtimes(
    *,
    sam3_source: str | Path,
    sam3_checkpoint: str | Path,
    sam3_checkpoint_sha256: str,
    molmopoint_model: str,
    molmopoint_revision: str,
    molmopoint_prompt_id: str = "rim_downward_approach",
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Construct the two project-owned runtimes, rejecting Omnis paths.

    Construction is intentionally separate from inference.  Legion can use
    this function as a cheap, deterministic model preflight, while the worker
    loads each model once when its first frame arrives.
    """
    source = Path(sam3_source).expanduser().resolve()
    checkpoint = Path(sam3_checkpoint).expanduser().resolve()
    if any("omnis" in part.lower() for part in source.parts + checkpoint.parts):
        raise RuntimeError("project-local canary runtimes must not use an Omnis path")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(sam3_checkpoint_sha256)):
        raise ValueError("sam3 checkpoint SHA-256 must be a full 64-character hexadecimal digest")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(molmopoint_revision)):
        raise ValueError("MolmoPoint revision must be a pinned 40-character commit SHA")
    if str(molmopoint_model) != MOLMOPOINT_MODEL_ID:
        raise ValueError("MolmoPoint model must be the pinned canary model")
    if str(molmopoint_revision).lower() != MOLMOPOINT_MODEL_REVISION:
        raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
    try:
        from .molmo_sam3.runtime import Sam3Runtime, Sam3RuntimeConfig
        from .molmo_sam3.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    except ImportError:  # pragma: no cover - direct script use
        from molmo_sam3.runtime import Sam3Runtime, Sam3RuntimeConfig
        from molmo_sam3.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    sam3 = Sam3Runtime(Sam3RuntimeConfig(
        sam3_source_dir=source, checkpoint_path=checkpoint,
        checkpoint_sha256=str(sam3_checkpoint_sha256).lower(), device=device,
    ))
    molmo = MolmoPointRuntime(MolmoPointRuntimeConfig(
        model_id=str(molmopoint_model), model_revision=str(molmopoint_revision), device=device,
        prompt_id=str(molmopoint_prompt_id), prompt=PROMPT_VARIANTS[str(molmopoint_prompt_id)],
    ))
    return sam3, molmo


def preflight_local_model_runtimes(sam3: Any, molmo: Any, *, load_models: bool = False) -> dict[str, Any]:
    """Verify local SAM3 assets and optionally load both models once."""
    validate_install = getattr(sam3, "_validate_local_install", None)
    if not callable(validate_install):
        raise RuntimeError("SAM3 runtime does not expose its local-install validation contract")
    validate_install()
    if load_models:
        try:
            getattr(sam3, "_load")()
            getattr(molmo, "_load")()
        except Exception as exc:
            raise RuntimeError(f"local MolmoPoint/SAM3 model load failed: {exc}") from exc
    return {
        "sam3_source": str(sam3.config.sam3_source_dir.resolve()),
        "sam3_checkpoint": str(sam3.config.checkpoint_path.resolve()),
        "sam3_checkpoint_sha256": str(sam3.config.checkpoint_sha256),
        "molmopoint_model": str(molmo.config.model_id),
        "molmopoint_revision": str(molmo.config.model_revision),
        "molmopoint_prompt_id": str(getattr(molmo.config, "prompt_id", "unknown")),
        "molmopoint_prompt": str(getattr(molmo.config, "prompt", "")),
        "models_loaded": bool(load_models),
    }


def build_local_molmo_runtime(
    *,
    molmopoint_model: str = MOLMOPOINT_MODEL_ID,
    molmopoint_revision: str = MOLMOPOINT_MODEL_REVISION,
    molmopoint_prompt_id: str = "rim_downward_approach",
    device: str = "cuda",
) -> Any:
    """Construct the persistent Molmo runtime without touching SAM assets."""
    if str(molmopoint_model) != MOLMOPOINT_MODEL_ID:
        raise ValueError("MolmoPoint model must be the pinned canary model")
    if str(molmopoint_revision).lower() != MOLMOPOINT_MODEL_REVISION:
        raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
    try:
        from .molmo_sam3.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    except ImportError:  # pragma: no cover - direct script use
        from molmo_sam3.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    if str(molmopoint_prompt_id) not in PROMPT_VARIANTS:
        raise ValueError(f"unknown MolmoPoint prompt id: {molmopoint_prompt_id}")
    return MolmoPointRuntime(MolmoPointRuntimeConfig(
        model_id=str(molmopoint_model), model_revision=str(molmopoint_revision), device=device,
        prompt_id=str(molmopoint_prompt_id), prompt=PROMPT_VARIANTS[str(molmopoint_prompt_id)],
    ))


def preflight_local_molmo_runtime(molmo: Any, *, load_models: bool = False) -> dict[str, Any]:
    """Validate/load Molmo only; importantly, this performs no SAM preflight."""
    config = getattr(molmo, "config", None)
    if config is None:
        raise RuntimeError("Molmo runtime is missing config provenance")
    if str(getattr(config, "model_id", "")) != MOLMOPOINT_MODEL_ID:
        raise ValueError("MolmoPoint model must be the pinned canary model")
    if str(getattr(config, "model_revision", "")).lower() != MOLMOPOINT_MODEL_REVISION:
        raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
    if load_models and not bool(getattr(molmo, "loaded", False)):
        try:
            getattr(molmo, "_load")()
        except Exception as exc:
            raise RuntimeError(f"local MolmoPoint model load failed: {exc}") from exc
    provenance = config.provenance() if callable(getattr(config, "provenance", None)) else {
        "model_id": str(config.model_id), "model_revision": str(config.model_revision),
        "prompt_id": str(getattr(config, "prompt_id", "unknown")),
        "prompt": str(getattr(config, "prompt", "")),
    }
    provenance["models_loaded"] = bool(getattr(molmo, "loaded", False))
    return provenance


def _capture_calibration(capture: Any) -> Any:
    """Adapt the evaluator capture contract to the pure geometry contract."""
    try:
        from .molmo_sam3.grasp_candidates import CameraCalibration
    except ImportError:  # pragma: no cover - direct script use
        from molmo_sam3.grasp_candidates import CameraCalibration
    calibration = capture.calibration
    return CameraCalibration(
        width=int(calibration.width), height=int(calibration.height),
        intrinsic=tuple(tuple(float(v) for v in row) for row in calibration.intrinsic),
        world_from_camera=tuple(tuple(float(v) for v in row) for row in calibration.world_from_camera),
        camera_name=str(calibration.camera_name),
    )


def _depth_near(capture: Any, uv: Sequence[float], radius: int = 2) -> float:
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("metric depth must be two-dimensional")
    u, v = (int(round(float(uv[0]))), int(round(float(uv[1]))))
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("pixel lies outside depth frame")
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    values = depth[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        raise ValueError("no positive metric depth near pixel")
    return float(np.median(valid))


def _deproject_capture(capture: Any, uv: Sequence[float]) -> np.ndarray:
    calibration = capture.calibration
    K = np.asarray(calibration.intrinsic, dtype=np.float64)
    T = np.asarray(calibration.world_from_camera, dtype=np.float64)
    u, v = float(uv[0]), float(uv[1])
    z = _depth_near(capture, (u, v))
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.all(np.isfinite(K)) or not np.all(np.isfinite(T)):
        raise ValueError("capture calibration has invalid K/T")
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    point = T[:3, :3] @ np.asarray((x, y, z), dtype=np.float64) + T[:3, 3]
    if not np.all(np.isfinite(point)):
        raise ValueError("deprojected point is non-finite")
    return point


def _project_capture(capture: Any, world_point: np.ndarray) -> tuple[float, float]:
    calibration = capture.calibration
    K = np.asarray(calibration.intrinsic, dtype=np.float64)
    T = np.asarray(calibration.world_from_camera, dtype=np.float64)
    camera_point = T[:3, :3].T @ (np.asarray(world_point, dtype=np.float64).reshape(3) - T[:3, 3])
    if camera_point[2] <= 1e-8:
        raise ValueError("projected point is behind the camera")
    uv = np.asarray((K[0, 0] * camera_point[0] / camera_point[2] + K[0, 2], K[1, 1] * camera_point[1] / camera_point[2] + K[1, 2]), dtype=np.float64)
    height, width = np.asarray(capture.rgb).shape[:2]
    if not np.all(np.isfinite(uv)) or not (0.0 <= uv[0] < width and 0.0 <= uv[1] < height):
        raise ValueError("projected association point is outside the source image")
    return float(uv[0]), float(uv[1])


def _projected_mask_region(agent_capture: Any, source_capture: Any, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Project observed agentview mask support into the wrist image."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        raise ValueError("observed agentview target mask is empty")
    # Bound work while retaining spatial coverage and only use pixels with
    # valid metric depth; this is an observed RGB-D association, not simulator
    # object state.
    stride = max(1, int(math.ceil(len(xs) / 256)))
    projected: list[tuple[float, float]] = []
    for u, v in zip(xs[::stride], ys[::stride]):
        try:
            projected.append(_project_capture(source_capture, _deproject_capture(agent_capture, (u, v))))
        except (ValueError, IndexError, TypeError):
            continue
    if len(projected) < 4:
        raise ValueError("insufficient observed target support for wrist association")
    values = np.asarray(projected, dtype=np.float64)
    return (float(values[:, 0].min()), float(values[:, 1].min()),
            float(values[:, 0].max()), float(values[:, 1].max()))


def _select_detection(
    detections: Sequence[Any], uv: Sequence[float], *, expected_depth_m: float | None = None,
    depth_image: np.ndarray | None = None, strict: bool = False,
    expected_region: tuple[float, float, float, float] | None = None,
) -> Any:
    if not detections:
        raise ValueError("SAM3 returned no bowl detections")
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    containing = [item for item in detections if 0 <= v < item.mask.shape[0] and 0 <= u < item.mask.shape[1] and bool(item.mask[v, u])]
    if strict and expected_region is not None:
        x0, y0, x1, y1 = expected_region
        region_mask = np.zeros_like(np.asarray(detections[0].mask, dtype=bool))
        ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        ix1, iy1 = min(region_mask.shape[1], int(math.ceil(x1)) + 1), min(region_mask.shape[0], int(math.ceil(y1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            raise ValueError("projected agentview target footprint is outside wrist image")
        region_mask[iy0:iy1, ix0:ix1] = True
        region_area = int(np.count_nonzero(region_mask))
        overlaps = []
        overlap_fractions = []
        for item in detections:
            item_mask = np.asarray(item.mask, dtype=bool)
            overlap = int(np.count_nonzero(item_mask & region_mask))
            overlaps.append(overlap)
            item_area = int(np.count_nonzero(item_mask))
            denominator = max(1, min(item_area, region_area))
            overlap_fractions.append(float(overlap) / float(denominator))
        min_support = max(4, int(math.ceil(0.05 * max(1, region_area))))
        meaningful = [
            item for item, overlap, fraction in zip(detections, overlaps, overlap_fractions)
            if overlap >= min_support and fraction >= 0.10
        ]
        if not meaningful:
            raise ValueError("no wrist SAM3 mask overlaps the observed agentview target footprint")
        containing = meaningful
    if strict and not containing:
        raise ValueError("SAM3 wrist mask does not contain the projected agentview target")
    pool = containing or list(detections)
    selected = None
    if strict and len(containing) > 1 and expected_region is not None:
        def association_key(item: Any) -> tuple[float, float, int]:
            item_mask = np.asarray(item.mask, dtype=bool)
            overlap = int(np.count_nonzero(item_mask & region_mask))
            denominator = max(1, min(int(np.count_nonzero(item_mask)), region_area))
            return (float(overlap) / float(denominator), float(item.score), int(np.count_nonzero(item_mask)))
        ranked = sorted(containing, key=association_key, reverse=True)
        if association_key(ranked[0])[0] - association_key(ranked[1])[0] < 0.10 and float(ranked[0].score) - float(ranked[1].score) < 0.05:
            raise ValueError("projected wrist target overlaps ambiguous SAM3 masks")
        selected = ranked[0]
    if not containing:
        distances = []
        for item in pool:
            ys, xs = np.nonzero(item.mask)
            if len(xs) == 0:
                distances.append(float("inf"))
            else:
                distances.append(float(np.hypot(np.mean(xs) - u, np.mean(ys) - v)))
        nearest = int(np.argmin(distances))
        if not np.isfinite(distances[nearest]) or distances[nearest] > max(item.mask.shape) * 0.45:
            raise ValueError("SAM3 bowl mask is not associated with the observed target")
        return pool[nearest]
    if selected is None:
        selected = sorted(pool, key=lambda item: (-float(item.score), -int(np.count_nonzero(item.mask))))[0]
    if expected_depth_m is not None and np.isfinite(expected_depth_m):
        values = (
            np.asarray(depth_image, dtype=np.float64)[np.asarray(selected.mask, dtype=bool)]
            if depth_image is not None else np.asarray(getattr(selected, "mask_depth_m", []), dtype=np.float64)
        )
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size and abs(float(np.median(values)) - float(expected_depth_m)) > 0.08:
            raise ValueError("SAM3 mask depth disagrees with projected agentview target")
    return selected


class ModelPerceptionWorker:
    """Persistent perception worker for SAM3 compatibility and RGB-D canaries."""

    def __init__(self, sam3: Any, molmo: Any, robot_calibration: Any, *, region_backend: str = "sam3") -> None:
        if region_backend not in {"sam3", "rgbd"}:
            raise ValueError("region_backend must be sam3 or rgbd")
        if region_backend == "sam3" and sam3 is None:
            raise ValueError("SAM3 runtime is required for the sam3 backend")
        self.sam3 = sam3
        self.molmo = molmo
        self.robot_calibration = robot_calibration
        self.region_backend = region_backend

    @staticmethod
    def _write_overlay(
        request: PerceptionRequest,
        *,
        mask: np.ndarray,
        source_uv: Sequence[float],
        molmo_points: Sequence[Any],
        selected_candidate_uv: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Persist a same-frame visual audit for every perception call."""
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:  # pragma: no cover - Pillow is a runtime dep
            return {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}
        image = np.asarray(request.source_capture.rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
            return {"status": "invalid_frame"}
        canvas = Image.fromarray(image, mode="RGB").convert("RGBA")
        tint = Image.new("RGBA", canvas.size, (220, 32, 32, 0))
        tint.putalpha(Image.fromarray(np.where(mask, 72, 0).astype(np.uint8), mode="L"))
        canvas = Image.alpha_composite(canvas, tint)
        draw = ImageDraw.Draw(canvas)

        def dot(uv: Sequence[float], colour: tuple[int, int, int], radius: int = 4) -> None:
            u, v = float(uv[0]), float(uv[1])
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=colour, width=2)

        dot(source_uv, (255, 255, 0), 5)
        for point in molmo_points:
            dot((float(point.u), float(point.v)), (0, 255, 255), 4)
        if selected_candidate_uv is not None:
            dot(selected_candidate_uv, (0, 255, 0), 6)
        overlay_dir = request.output_dir / "candidate_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        frame_hash = hashlib.sha256(image.tobytes()).hexdigest()[:12]
        name = f"{request.variant.name}__{frame_hash}__{len(request.previous_candidate_ids):02d}.png"
        path = overlay_dir / name
        canvas.convert("RGB").save(path, format="PNG")
        return {"status": "saved", "path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def propose(self, request: PerceptionRequest) -> Any:
        if self.region_backend == "rgbd":
            return self._propose_rgbd(request)
        try:
            from .molmo_sam3.grasp_candidates import (
                CandidatePolicy, MolmoPoint, RGBDObservation, generate_grasp_candidates,
            )
            from .molmo_sam3.molmopoint import MolmoPointRequest
            from .molmo_sam3.runtime import Sam3Request
        except ImportError:  # pragma: no cover - direct script use
            from molmo_sam3.grasp_candidates import CandidatePolicy, MolmoPoint, RGBDObservation, generate_grasp_candidates
            from molmo_sam3.molmopoint import MolmoPointRequest
            from molmo_sam3.runtime import Sam3Request
        started = time.perf_counter()
        source_uv = tuple(float(v) for v in request.source_uv)
        expected_region = None
        if request.variant.camera_name == WRIST_CAMERA:
            observed_world = _deproject_capture(request.agentview_capture, source_uv)
            source_uv = _project_capture(request.source_capture, observed_world)
            observed_sam = self.sam3.predict(Sam3Request(
                np.asarray(request.agentview_capture.rgb, dtype=np.uint8), prompt="bowl"
            ))
            observed_detection = _select_detection(
                observed_sam.detections, request.source_uv,
                expected_depth_m=_depth_near(request.agentview_capture, request.source_uv),
                depth_image=np.asarray(request.agentview_capture.metric_depth, dtype=np.float64),
            )
            expected_region = _projected_mask_region(
                request.agentview_capture, request.source_capture,
                np.asarray(observed_detection.mask, dtype=bool),
            )
        sam_result = self.sam3.predict(Sam3Request(np.asarray(request.source_capture.rgb, dtype=np.uint8), prompt="bowl"))
        source_depth = _depth_near(request.source_capture, source_uv)
        detection = _select_detection(
            sam_result.detections, source_uv,
            expected_depth_m=source_depth,
            depth_image=np.asarray(request.source_capture.metric_depth, dtype=np.float64),
            strict=request.variant.camera_name == WRIST_CAMERA,
            expected_region=expected_region,
        )
        molmo_points: list[Any] = []
        molmo_provenance: Mapping[str, Any] | None = None
        if request.variant.uses_molmo:
            molmo_result = self.molmo.predict(MolmoPointRequest(
                np.asarray(request.source_capture.rgb, dtype=np.uint8), mask=np.asarray(detection.mask, dtype=np.bool_)
            ))
            molmo_points = [MolmoPoint(float(point.x), float(point.y), label="molmo_rim") for point in molmo_result.points]
            molmo_provenance = dict(molmo_result.provenance)
        policy = CandidatePolicy(name=request.variant.policy, obstruction_clearance_m=0.006)
        result = generate_grasp_candidates(
            rgb=np.asarray(request.source_capture.rgb, dtype=np.uint8),
            metric_depth_m=np.asarray(request.source_capture.metric_depth, dtype=np.float64),
            sam_mask=np.asarray(detection.mask, dtype=np.bool_),
            molmo_points=molmo_points,
            calibration=_capture_calibration(request.source_capture),
            robot_calibration=self.robot_calibration,
            policy=policy,
        )
        diagnostics = {
            "source_camera": request.variant.camera_name,
            "source_uv": list(source_uv),
            "sam3_detection_count": len(sam_result.detections),
            "sam3_selected_score": float(detection.score),
            "molmopoint_count": len(molmo_points),
            "molmopoint_provenance": molmo_provenance,
            "returned_candidate_count": len(result.candidates),
            "rejection_count": len(result.rejected),
            "latency_s": float(time.perf_counter() - started),
            "geometry_audit": result.audit,
            "rejections": [
                {"seed_index": item.seed_index, "yaw_deg": item.yaw_deg, "insertion_depth_m": item.insertion_depth_m, "reason": item.reason, "details": _json_safe(item.details)}
                for item in result.rejected
            ],
        }
        diagnostics["candidate_overlay"] = self._write_overlay(
            request, mask=np.asarray(detection.mask, dtype=np.bool_), source_uv=source_uv,
            molmo_points=molmo_points,
            selected_candidate_uv=(result.candidates[0].source_pixel_uv if result.candidates else None),
        )
        return {"candidates": result.candidates, "diagnostics": diagnostics}

    def _propose_rgbd(self, request: PerceptionRequest) -> Any:
        """Generate candidates from arrow-seeded RGB-D support only."""
        try:
            from .molmo_sam3.grasp_candidates import CandidatePolicy, MolmoPoint, generate_grasp_candidates
            from .rgbd_region import derive_observed_region_mask, project_observed_region_to_wrist
            from .molmo_sam3.molmopoint import MolmoPointRequest
        except ImportError:  # pragma: no cover - direct script use
            from molmo_sam3.grasp_candidates import CandidatePolicy, MolmoPoint, generate_grasp_candidates
            from rgbd_region import derive_observed_region_mask, project_observed_region_to_wrist
            from molmo_sam3.molmopoint import MolmoPointRequest
        started = time.perf_counter()
        # The destination arrow endpoint is an observed profile hint; when it
        # is unavailable, the source point remains a valid deterministic axis.
        profile_world = _deproject_capture(request.agentview_capture, request.destination_uv or request.source_uv)
        region_mask, region_audit = derive_observed_region_mask(
            request.agentview_capture, request.source_uv, profile_target_world=profile_world,
        )
        source_capture = request.source_capture
        source_uv = tuple(float(v) for v in request.source_uv)
        association_audit: dict[str, Any] = {}
        if request.variant.camera_name == WRIST_CAMERA:
            projected_support, association_audit = project_observed_region_to_wrist(
                request.agentview_capture, source_capture, region_mask,
                depth_tolerance_m=float(region_audit.get("depth_tolerance_m", 0.025)),
            )
            # The observed source support must be projected into the current
            # wrist frame; stale agentview coordinates are never reused.
            seed_world = _deproject_capture(request.agentview_capture, request.source_uv)
            source_uv = _project_capture(source_capture, seed_world)
            profile_world = _deproject_capture(source_capture, source_uv)
            # Projection is used only for strict camera identity.  Candidate
            # geometry consumes a dense component grown from current wrist
            # RGB-D, gated by overlap with the projected observed support.
            wrist_region, wrist_audit = derive_observed_region_mask(
                source_capture, source_uv, profile_target_world=profile_world,
            )
            overlap = int(np.count_nonzero(wrist_region & projected_support))
            projected_area = max(1, int(np.count_nonzero(projected_support)))
            overlap_fraction = float(overlap / projected_area)
            if overlap < 4 or overlap_fraction < 0.20:
                raise ValueError("current wrist RGB-D region does not overlap projected agentview support")
            region_mask = wrist_region
            association_audit = {
                **association_audit,
                "current_wrist_region_area_px": int(np.count_nonzero(wrist_region)),
                "current_wrist_region_audit": wrist_audit,
                "projected_current_overlap_px": overlap,
                "projected_current_overlap_fraction": overlap_fraction,
            }
        molmo_points: list[Any] = []
        molmo_provenance: Mapping[str, Any] | None = None
        if request.variant.uses_molmo:
            molmo_result = self.molmo.predict(MolmoPointRequest(
                np.asarray(source_capture.rgb, dtype=np.uint8), mask=np.asarray(region_mask, dtype=np.bool_)
            ))
            molmo_points = [MolmoPoint(float(point.x), float(point.y), label="molmo_rim") for point in molmo_result.points]
            molmo_provenance = dict(molmo_result.provenance)
        policy = CandidatePolicy(name=request.variant.policy, obstruction_clearance_m=0.006)
        result = generate_grasp_candidates(
            rgb=np.asarray(source_capture.rgb, dtype=np.uint8),
            metric_depth_m=np.asarray(source_capture.metric_depth, dtype=np.float64),
            sam_mask=np.asarray(region_mask, dtype=np.bool_),
            molmo_points=molmo_points,
            calibration=_capture_calibration(source_capture),
            robot_calibration=self.robot_calibration,
            policy=policy,
        )
        diagnostics = {
            "backend": "v9d_rgbd_region", "region_backend": "rgbd", "sam3_used": False,
            "source_camera": request.variant.camera_name, "source_uv": list(source_uv),
            "molmopoint_count": len(molmo_points), "molmopoint_provenance": molmo_provenance,
            "returned_candidate_count": len(result.candidates), "rejection_count": len(result.rejected),
            "latency_s": float(time.perf_counter() - started),
            "region_audit": region_audit, "wrist_association_audit": association_audit,
            "geometry_audit": result.audit,
            "rejections": [
                {"seed_index": item.seed_index, "yaw_deg": item.yaw_deg,
                 "insertion_depth_m": item.insertion_depth_m, "reason": item.reason,
                 "details": _json_safe(item.details)}
                for item in result.rejected
            ],
        }
        diagnostics["candidate_overlay"] = self._write_overlay(
            request, mask=np.asarray(region_mask, dtype=np.bool_), source_uv=source_uv,
            molmo_points=molmo_points,
            selected_candidate_uv=(result.candidates[0].source_pixel_uv if result.candidates else None),
        )
        return {"candidates": result.candidates, "diagnostics": diagnostics}


def probe_robot_calibration(env: Any) -> tuple[Any, np.ndarray, Mapping[str, Any]]:
    """Run the no-motion Panda grip-site probe and derive the action transform."""
    try:
        from .sanity_checks.probe_panda_grip_site_frame import probe_grip_site_frame
        from .molmo_sam3.grasp_candidates import RobotGraspCalibration
    except ImportError:  # pragma: no cover - direct script use
        from sanity_checks.probe_panda_grip_site_frame import probe_grip_site_frame
        from molmo_sam3.grasp_candidates import RobotGraspCalibration
    record = probe_grip_site_frame(env.sim)
    if not bool(record.get("passed")):
        raise RuntimeError("Panda grip-site frame probe failed")
    transform = np.asarray(record["observed_body_to_site_rotation_matrix"], dtype=np.float64)
    sim = getattr(env, "sim", env)
    model, data = getattr(sim, "model", None), getattr(sim, "data", None)
    if model is None or data is None:
        raise RuntimeError("Panda calibration probe requires MuJoCo model/data")

    def named_id(kind: str, names: Sequence[str]) -> tuple[int, str]:
        resolver = getattr(model, f"{kind}_name2id", None)
        for name in names:
            if callable(resolver):
                try:
                    idx = int(resolver(name))
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
                if idx >= 0:
                    return idx, name
        raise RuntimeError(f"Panda calibration probe could not resolve {kind} names {tuple(names)!r}")

    def optional_named_id(kind: str, names: Sequence[str]) -> tuple[int, str] | None:
        """Resolve a model name when present (fixtures omit some Panda geoms)."""
        resolver = getattr(model, f"{kind}_name2id", None)
        if not callable(resolver):
            return None
        for name in names:
            try:
                idx = int(resolver(name))
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            if idx >= 0:
                return idx, name
        return None

    # Robosuite's Panda XML exposes the two finger bodies under these stable
    # names.  Resolve them as an identity check, but use contact-pad geoms
    # below for all physical aperture and translation measurements.
    left_id, left_name = named_id("body", ("gripper0_leftfinger", "gripper0_left_finger", "leftfinger", "left_finger"))
    right_id, right_name = named_id("body", ("gripper0_rightfinger", "gripper0_right_finger", "rightfinger", "right_finger"))

    # Use the MuJoCo contact-pad geoms, not finger-body origins.  Body origins
    # are useful as a fallback diagnostic but are not a calibrated contact
    # point and must never determine aperture or the grasp-site translation.
    left_pad_id, left_pad_name = named_id("geom", (
        "gripper0_finger1_pad_collision", "finger1_pad_collision",
        "gripper0_leftfinger_pad_collision", "leftfinger_pad_collision",
    ))
    right_pad_id, right_pad_name = named_id("geom", (
        "gripper0_finger2_pad_collision", "finger2_pad_collision",
        "gripper0_rightfinger_pad_collision", "rightfinger_pad_collision",
    ))
    left = np.asarray(data.geom_xpos[left_pad_id], dtype=np.float64).reshape(-1)
    right = np.asarray(data.geom_xpos[right_pad_id], dtype=np.float64).reshape(-1)
    site_id = int(record["site_id"])
    site = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(-1)
    site_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    if left.shape != (3,) or right.shape != (3,) or site.shape != (3,) or not np.all(np.isfinite(np.vstack((left, right, site, site_rotation)))):
        raise RuntimeError("Panda calibration probe returned invalid finger/site geometry")
    jaw_vector = right - left
    center_aperture = float(np.linalg.norm(jaw_vector))
    if center_aperture <= 1e-4:
        raise RuntimeError("Panda calibration probe measured a closed/degenerate aperture")
    jaw_axis = jaw_vector / center_aperture
    midpoint = (left + right) * 0.5
    contact_offset_site = site_rotation.T @ (site - midpoint)
    # Build the site basis from live geometry.  The approach direction is the
    # palm/right-hand to pad-midpoint vector, projected off the measured jaw
    # axis; this avoids assuming a particular local +Y convention.  M is
    # R_site_grasp (grasp axes expressed in site coordinates), so the
    # transform consumed by pose generation is R_ge=M.T.
    body_positions = np.asarray(getattr(data, "xpos", getattr(data, "body_xpos", [])), dtype=np.float64)
    if body_positions.ndim != 2 or body_positions.shape[1:] != (3,):
        raise RuntimeError("Panda calibration probe lacks body world positions for approach basis")
    # Reuse the exact hand identity resolved by the shared grip-site probe;
    # prefixed robosuite models expose this as ``robot0_right_hand``.
    try:
        palm_id = int(record["body_id"])
        palm_name = str(record.get("resolved_body_name") or record.get("body_frame") or "right_hand")
        if palm_id < 0 or palm_id >= len(body_positions):
            raise ValueError("body_id outside MuJoCo body-position array")
    except (KeyError, TypeError, ValueError):
        palm_id, palm_name = named_id("body", ("right_hand", "robot0_right_hand", "panda_hand", "gripper0_hand"))
    palm = body_positions[palm_id]
    approach_world = midpoint - palm
    approach_world -= jaw_axis * float(np.dot(approach_world, jaw_axis))
    if not np.isfinite(approach_world).all() or np.linalg.norm(approach_world) <= 1e-8:
        raise RuntimeError("Panda calibration probe has degenerate palm-to-pad approach vector")
    approach_world /= np.linalg.norm(approach_world)
    measured_jaw_site = site_rotation.T @ jaw_axis
    measured_jaw_site /= np.linalg.norm(measured_jaw_site)
    approach_site = site_rotation.T @ approach_world
    approach_site /= np.linalg.norm(approach_site)
    completion_site = np.cross(approach_site, measured_jaw_site)
    if np.linalg.norm(completion_site) <= 1e-8:
        raise RuntimeError("Panda calibration probe has degenerate approach/jaw basis")
    completion_site /= np.linalg.norm(completion_site)
    M = np.column_stack((approach_site, measured_jaw_site, completion_site))
    grasp_to_grip_site = M.T
    if not np.isclose(np.linalg.det(grasp_to_grip_site), 1.0, atol=1e-5):
        raise RuntimeError("derived Panda grasp-to-grip-site basis is not a proper rotation")
    jaw_alignment = float(np.dot(measured_jaw_site, grasp_to_grip_site[:, 1]))

    def geom_half_extent_and_normal(geom_id: int, expected_inward: np.ndarray) -> tuple[float, np.ndarray]:
        sizes = getattr(model, "geom_size", None)
        rotations = getattr(data, "geom_xmat", None)
        if sizes is None or rotations is None:
            raise RuntimeError("Panda calibration probe requires model.geom_size and data.geom_xmat")
        size = np.asarray(sizes[geom_id], dtype=np.float64).reshape(-1)
        rotation = np.asarray(rotations[geom_id], dtype=np.float64).reshape(3, 3)
        if size.size < 3 or not np.isfinite(size[:3]).all() or not np.isfinite(rotation).all():
            raise RuntimeError("Panda calibration probe returned invalid contact-pad dimensions")
        expected_inward = np.asarray(expected_inward, dtype=np.float64)
        expected_inward /= np.linalg.norm(expected_inward)
        local_jaw = rotation.T @ jaw_axis
        face_axis = int(np.argmax(np.abs(local_jaw)))
        if abs(float(local_jaw[face_axis])) < 0.9:
            raise RuntimeError("Panda contact-pad face axis is not aligned with measured jaw")
        face_normal = rotation[:, face_axis] * (1.0 if np.dot(rotation[:, face_axis], expected_inward) >= 0 else -1.0)
        if float(np.dot(face_normal, expected_inward)) < 0.9:
            raise RuntimeError("Panda contact-pad face normal is inconsistent with inward jaw direction")
        return float(abs(local_jaw[face_axis]) * size[face_axis]), face_normal

    left_half_extent, left_face_normal = geom_half_extent_and_normal(left_pad_id, right - left)
    right_half_extent, right_face_normal = geom_half_extent_and_normal(right_pad_id, left - right)
    usable_aperture = center_aperture - left_half_extent - right_half_extent
    if usable_aperture <= 1e-4:
        raise RuntimeError("Panda contact-pad usable aperture is closed after pad extent correction")
    contact_offset_grasp = grasp_to_grip_site @ contact_offset_site

    # Capture conservative collision primitives for the live hand/finger
    # geometry.  Centers are measured in the contact-midpoint grasp frame, so
    # the geometry worker can transform them relative to each candidate without
    # reading object/evaluator state.  ``geom_rbound`` is preferred; the
    # Euclidean half-size bound is a safe fallback for lightweight MuJoCo fakes.
    # The collision mesh for the Panda palm is owned by the right-gripper
    # body, not by the ``robot0_right_hand`` site body.  Include both owners
    # when available, and explicitly resolve the stable palm collision geom
    # because some MuJoCo models omit body-parent metadata on lightweight
    # wrappers.  Finger collision meshes are likewise included by name.
    right_gripper = optional_named_id("body", ("gripper0_right_gripper", "right_gripper"))
    hand_body_ids = {int(left_id), int(right_id), int(palm_id)}
    if right_gripper is not None:
        hand_body_ids.add(int(right_gripper[0]))
    geom_body_ids = getattr(model, "geom_bodyid", None)
    geom_positions = getattr(data, "geom_xpos", None)
    geom_sizes = getattr(model, "geom_size", None)
    geom_rbounds = getattr(model, "geom_rbound", None)
    collision_spheres: list[dict[str, Any]] = []
    collision_boxes: list[dict[str, Any]] = []
    candidate_geom_ids = {int(left_pad_id), int(right_pad_id)}
    for geom_names in (
        ("gripper0_hand_collision", "hand_collision"),
        (
            "gripper0_leftfinger_collision", "gripper0_rightfinger_collision",
            "gripper0_finger1_collision", "gripper0_finger2_collision",
            "leftfinger_collision", "rightfinger_collision", "finger1_collision", "finger2_collision",
        ),
    ):
        resolved = optional_named_id("geom", geom_names)
        if resolved is not None:
            candidate_geom_ids.add(int(resolved[0]))
    geom_name_resolver = getattr(model, "geom_id2name", None)
    model_geom_names = getattr(model, "names", {}).get("geom", ()) if isinstance(getattr(model, "names", {}), Mapping) else ()
    unrepresented_collision_geoms: list[str] = []
    collision_geom_ids: set[int] = set()
    has_live_geom_metadata = (
        geom_body_ids is not None and getattr(model, "geom_type", None) is not None
    )

    def geom_name_for(geom_id: int) -> str:
        if callable(geom_name_resolver):
            try:
                return str(geom_name_resolver(int(geom_id)))
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        if int(geom_id) < len(model_geom_names):
            return str(model_geom_names[int(geom_id)])
        return str(geom_id)

    def local_bounds_for_geom(geom_id: int) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return compiled local center/half-extents for one live collision geom."""
        sizes = np.asarray(geom_sizes, dtype=np.float64) if geom_sizes is not None else None
        geom_type_values = getattr(model, "geom_type", None)
        geom_data_values = getattr(model, "geom_dataid", None)
        # MuJoCo's compiled mesh vertices are already in geom-local units.
        # Use the mesh's enclosing AABB, retaining a non-zero extent even for
        # a degenerate axis so the downstream box validator remains strict.
        if geom_type_values is not None and geom_data_values is not None:
            try:
                geom_type = int(np.asarray(geom_type_values).reshape(-1)[geom_id])
                mesh_id = int(np.asarray(geom_data_values).reshape(-1)[geom_id])
            except (IndexError, TypeError, ValueError):
                geom_type, mesh_id = -1, -1
            mesh_vertices = getattr(model, "mesh_vert", None)
            mesh_adrs = getattr(model, "mesh_vertadr", None)
            mesh_counts = getattr(model, "mesh_vertnum", None)
            if geom_type == 7 and mesh_vertices is not None and mesh_adrs is not None and mesh_counts is not None and mesh_id >= 0:
                try:
                    start = int(np.asarray(mesh_adrs).reshape(-1)[mesh_id])
                    count = int(np.asarray(mesh_counts).reshape(-1)[mesh_id])
                    all_vertices = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
                    if start < 0 or count <= 0 or start + count > len(all_vertices):
                        raise ValueError("compiled mesh vertex range is invalid")
                    vertices = all_vertices[start:start + count]
                except (IndexError, TypeError, ValueError):
                    vertices = np.empty((0, 3), dtype=np.float64)
                if len(vertices) and np.isfinite(vertices).all():
                    local_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
                    half_extents = (vertices.max(axis=0) - vertices.min(axis=0)) * 0.5
                    half_extents = np.maximum(half_extents, 1e-6)
                    return local_center, half_extents, "compiled_mesh_vertices_aabb"
            # A mesh must be represented by its compiled vertices (or by the
            # explicit geom_rbound fallback below); do not silently infer a
            # mesh box from geom_size, whose meaning is model-dependent.
            if geom_type == 7:
                return None
            if sizes is not None:
                try:
                    size = sizes[geom_id].reshape(-1)
                    if size.size >= 3 and np.isfinite(size[:3]).all():
                        # MuJoCo primitive geom sizes are local half-extents
                        # (sphere/capsule/cylinder use radius and half-length).
                        if geom_type == 2:       # sphere
                            extents = np.repeat(abs(float(size[0])), 3)
                        elif geom_type == 3:     # capsule, local Z axis
                            radius, half_length = abs(float(size[0])), abs(float(size[1]))
                            extents = np.asarray((radius, radius, radius + half_length))
                        elif geom_type == 4:     # ellipsoid
                            extents = np.abs(size[:3])
                        elif geom_type == 5:     # cylinder, local Z axis
                            extents = np.asarray((abs(float(size[0])), abs(float(size[0])), abs(float(size[1]))))
                        elif geom_type == 6:     # box
                            extents = np.abs(size[:3])
                        else:
                            return None
                        if np.all(np.isfinite(extents)) and np.all(extents > 0.0):
                            return np.zeros(3, dtype=np.float64), extents, "mujoco_geom_size"
                except (IndexError, TypeError, ValueError):
                    pass
        # If type metadata is unavailable, retain an explicit conservative
        # sphere in the legacy sphere audit; no fabricated box is emitted.
        return None

    if geom_body_ids is not None and geom_positions is not None:
        body_ids = np.asarray(geom_body_ids).reshape(-1)
        for geom_id, body_id in enumerate(body_ids):
            if int(body_id) in hand_body_ids:
                candidate_geom_ids.add(int(geom_id))
    for geom_id in sorted(candidate_geom_ids):
        try:
            # Match MuJoCo's collision filtering: visual-only geoms have both
            # masks cleared and must not become conservative hand obstacles.
            # Missing masks are accepted only for minimal test doubles.
            contypes = getattr(model, "geom_contype", None)
            conaffinities = getattr(model, "geom_conaffinity", None)
            if contypes is not None and conaffinities is not None:
                contype = int(np.asarray(contypes).reshape(-1)[geom_id])
                conaffinity = int(np.asarray(conaffinities).reshape(-1)[geom_id])
                if contype == 0 and conaffinity == 0:
                    continue
            collision_geom_ids.add(int(geom_id))
            center_world = np.asarray(geom_positions[geom_id], dtype=np.float64).reshape(-1)
            if center_world.shape != (3,) or not np.isfinite(center_world).all():
                continue
            if geom_rbounds is not None:
                radius = float(np.asarray(geom_rbounds).reshape(-1)[geom_id])
            elif geom_sizes is not None:
                size = np.asarray(geom_sizes[geom_id], dtype=np.float64).reshape(-1)
                radius = float(np.linalg.norm(size[:3])) if size.size >= 3 else float("nan")
            else:
                radius = float("nan")
            radius_valid = bool(np.isfinite(radius) and radius >= 0.0)
            center_site = site_rotation.T @ (center_world - midpoint)
            center_grasp = grasp_to_grip_site @ center_site
            geom_name = geom_name_for(geom_id)
            if radius_valid:
                collision_spheres.append({
                    "center_grasp_m": center_grasp.tolist(),
                    "radius_m": radius,
                    "geom_id": int(geom_id),
                    "geom_name": geom_name,
                })
            geom_rotation_values = getattr(data, "geom_xmat", None)
            if geom_rotation_values is not None:
                try:
                    geom_rotation = np.asarray(geom_rotation_values[geom_id], dtype=np.float64).reshape(3, 3)
                except (IndexError, TypeError, ValueError):
                    geom_rotation = None
                if geom_rotation is not None and np.all(np.isfinite(geom_rotation)) and np.allclose(geom_rotation.T @ geom_rotation, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(geom_rotation), 1.0, atol=1e-5):
                    bounds = local_bounds_for_geom(geom_id)
                    if bounds is not None:
                        local_center, half_extents, bounds_source = bounds
                        center_with_local_offset = center_world + geom_rotation @ local_center
                        box_center = grasp_to_grip_site @ (site_rotation.T @ (center_with_local_offset - midpoint))
                        box_rotation = grasp_to_grip_site @ (site_rotation.T @ geom_rotation)
                        collision_boxes.append({
                            "center_grasp_m": box_center.tolist(),
                            "rotation_grasp_box": box_rotation.tolist(),
                            "half_extents_m": half_extents.tolist(),
                            "geom_id": int(geom_id),
                            "geom_name": geom_name,
                            "source": bounds_source,
                        })
                    elif has_live_geom_metadata:
                        # The sphere audit is retained for diagnostics, but
                        # live consumers prefer complete AABB coverage.  A
                        # verified MuJoCo rbound is a conservative fallback;
                        # otherwise fail closed below instead of dropping a
                        # palm/finger collision piece silently.
                        if geom_rbounds is not None:
                            try:
                                fallback_radius = float(np.asarray(geom_rbounds).reshape(-1)[geom_id])
                            except (IndexError, TypeError, ValueError):
                                fallback_radius = float("nan")
                            if np.isfinite(fallback_radius) and fallback_radius > 0.0:
                                collision_boxes.append({
                                    "center_grasp_m": center_grasp.tolist(),
                                    "rotation_grasp_box": (grasp_to_grip_site @ (site_rotation.T @ geom_rotation)).tolist(),
                                    "half_extents_m": np.repeat(fallback_radius, 3).tolist(),
                                    "geom_id": int(geom_id),
                                    "geom_name": geom_name,
                                    "source": "verified_geom_rbound_sphere_box",
                                })
                            else:
                                unrepresented_collision_geoms.append(geom_name)
                        else:
                            unrepresented_collision_geoms.append(geom_name)
        except (IndexError, TypeError, ValueError):
            continue
    if not collision_spheres:
        raise RuntimeError("Panda calibration probe found no finite hand collision geometry")
    if has_live_geom_metadata:
        represented_box_ids = {int(item["geom_id"]) for item in collision_boxes}
        missing_box_ids = sorted(collision_geom_ids - represented_box_ids)
        unrepresented_collision_geoms.extend(geom_name_for(geom_id) for geom_id in missing_box_ids)
    if unrepresented_collision_geoms:
        raise RuntimeError(
            "Panda calibration probe could not conservatively represent collision geoms: "
            + ", ".join(sorted(set(unrepresented_collision_geoms)))
        )
    record = dict(record)
    record["gripper_geometry"] = {
        "left_body": left_name, "right_body": right_name,
        "left_pad_geom": left_pad_name, "right_pad_geom": right_pad_name,
        "left_pad_center_world_m": left.tolist(), "right_pad_center_world_m": right.tolist(),
        "jaw_axis_world": jaw_axis.tolist(), "contact_midpoint_world_m": midpoint.tolist(),
         "palm_body": palm_name,
         "jaw_axis_grip_site_local": measured_jaw_site.tolist(),
         "approach_axis_grip_site_local": approach_site.tolist(),
         "completion_axis_grip_site_local": completion_site.tolist(),
         "jaw_axis_alignment": jaw_alignment,
         "grasp_to_grip_site": grasp_to_grip_site.tolist(),
         "center_aperture_m": center_aperture,
         "pad_half_extent_left_m": left_half_extent,
         "pad_half_extent_right_m": right_half_extent,
         "left_inward_face_normal_world": left_face_normal.tolist(),
         "right_inward_face_normal_world": right_face_normal.tolist(),
         "measured_opening_m": usable_aperture,
         "contact_to_grip_site_site_m": contact_offset_site.tolist(),
        "contact_to_grip_site_grasp_m": contact_offset_grasp.tolist(),
        "current_grip_site_world_m": site.tolist(),
        "current_rotation_world_grip_site": site_rotation.tolist(),
        "hand_collision_spheres_grasp": collision_spheres,
        "hand_collision_boxes_grasp": collision_boxes,
        "hand_collision_geom_count": len(collision_spheres),
        "hand_collision_geom_names": [item["geom_name"] for item in collision_spheres],
        "hand_collision_box_count": len(collision_boxes),
        "hand_collision_box_names": [item["geom_name"] for item in collision_boxes],
        "source": "no_motion_mujoco_contact_pad_geom_probe",
    }
    calibration = RobotGraspCalibration(
        grasp_to_grip_site=grasp_to_grip_site, contact_to_grip_site_m=contact_offset_grasp,
         max_aperture_m=max(0.005, usable_aperture * 0.98),
         calibration_source="panda_grip_site_frame_and_contact_pad_probe_no_motion",
         calibration_sha256=str(record.get("calibration_sha256") or ""),
         current_grip_site_world_m=site,
         current_rotation_world_grip_site=site_rotation,
         hand_collision_spheres_grasp=collision_spheres,
         hand_collision_boxes_grasp=collision_boxes,
    )
    setattr(env, "_molmo_sam3_robot_calibration_probe", record)
    return calibration, transform, record


def _recover_after_failed_candidate(
    env: Any,
    *,
    output_dir: Path,
    orientation_transform: np.ndarray | None,
    motion_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Open and retreat with bounded actions before a fresh perception retry."""
    try:
        try:
            from . import run_arrow_pick_place_eval as episode
        except ImportError:  # pragma: no cover - direct script use
            import run_arrow_pick_place_eval as episode
        observation = episode._raw_observation(env)
        proprio = episode._proprioception(observation)
        current = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise RuntimeError("failed-candidate recovery lacks finite EEF position")
        retreat = current + np.asarray((0.0, 0.0, 0.10), dtype=np.float64)
        waypoints = np.vstack((current, current, current, current, current, retreat))
        prior_actions = int(getattr(env, "_molmo_sam3_action_count", 0))
        budget = getattr(env, "_molmo_sam3_action_budget", None)
        if budget is not None:
            remaining = int(budget.remaining)
        else:
            remaining = max(0, 1200 - prior_actions)
        if remaining <= 0:
            raise RuntimeError("candidate retry action budget exhausted before recovery")
        budget = getattr(env, "_molmo_sam3_action_budget", None)
        hover_budget = min(160, int(budget.remaining)) if budget is not None else 160
        if hover_budget <= 0:
            raise RuntimeError("1200-action retry budget exhausted before observation hover")
        motion_diag_kwargs = {"experimental_motion_diagnostics": True} if (motion_settings or {}).get("motion_diagnostics") else {}
        phases = episode._run_motion(
            env, waypoints, observation,
            phase_timeout_steps=int((motion_settings or {}).get("phase_timeout_steps", 160)),
            gripper_dwell_steps=int((motion_settings or {}).get("gripper_dwell_steps", 20)),
            stop_after_phase="retreat", start_phase="open", dry_run=False,
            action_budget=budget if budget is not None else remaining,
            eef_orientation_transform=orientation_transform,
            motion_trace_path=output_dir / "recovery_motion_trace.json",
            stall_window_steps=int((motion_settings or {}).get("stall_window_steps", 0)),
            stall_delta_m=float((motion_settings or {}).get("stall_delta_m", DEFAULT_STALL_DELTA_M)),
            osc_position_scale_m=(motion_settings or {}).get("osc_position_scale_m"),
            phase_policies=(motion_settings or {}).get("phase_policies"),
            **motion_diag_kwargs,
        )
        recovery_steps = int(sum(1 for item in (getattr(env, "_arrow_motion_trace", []) or []) if isinstance(item, Mapping) and item.get("action_sent")))
        total_actions = int(budget.used) if budget is not None else prior_actions + recovery_steps
        if total_actions > 1200:
            raise RuntimeError("candidate retry action budget exceeded 1200 steps")
        setattr(env, "_molmo_sam3_action_count", total_actions)
        return {"retreat_complete": True, "prior_actions": prior_actions, "recovery_steps": recovery_steps, "total_actions": total_actions, "recovery_phases": phases}
    except Exception as exc:
        return {"retreat_complete": False, "status": "recovery_failed", "recovery_error_type": type(exc).__name__, "recovery_error": str(exc)}


def retention_gate(_record: Mapping[str, Any], proprio: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless lift-time gripper proprioception shows contact."""
    try:
        qpos = np.asarray(proprio.get("gripper_qpos"), dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        qpos = np.asarray([], dtype=np.float64)
    if not qpos.size or not np.isfinite(qpos).all():
        return {"decision": "unobservable", "retained": False, "source": "lift_gripper_qpos"}
    threshold = 0.0015
    retained = bool(float(np.max(np.abs(qpos))) >= threshold)
    return {
        "decision": "retained" if retained else "not_retained",
        "retained": retained,
        "threshold": threshold,
        "final_abs_qpos": np.abs(qpos).tolist(),
        "source": "lift_gripper_qpos",
    }


def _newly_sent_actions(env: Any, trace_before: Any) -> int:
    """Count only actions from the current run_episode motion trace."""
    trace_after = getattr(env, "_arrow_motion_trace", None)
    if trace_after is trace_before:
        return 0
    return int(sum(
        1 for item in (trace_after or ())
        if isinstance(item, Mapping) and item.get("action_sent")
    ))


def _perform_gripper_open(
    env: Any,
    *,
    output_dir: Path,
    motion_started_callback: Callable[[], None] | None = None,
    motion_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Open at the current EEF pose before measuring contact-pad geometry."""
    try:
        try:
            from . import run_arrow_pick_place_eval as episode
        except ImportError:  # pragma: no cover - direct script use
            import run_arrow_pick_place_eval as episode
        observation = episode._raw_observation(env)
        proprio = episode._proprioception(observation)
        current = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise RuntimeError("gripper-open preflight lacks finite EEF position")
        budget = getattr(env, "_molmo_sam3_action_budget", None)
        prior_actions = int(getattr(env, "_molmo_sam3_action_count", 0))
        remaining = int(budget.remaining) if budget is not None else max(0, 1200 - prior_actions)
        if remaining <= 0:
            raise RuntimeError("1200-action budget exhausted before gripper open")
        waypoints = np.tile(current, (6, 1))
        motion_diag_kwargs = {"experimental_motion_diagnostics": True} if (motion_settings or {}).get("motion_diagnostics") else {}
        phases = episode._run_motion(
            env, waypoints, observation,
            phase_timeout_steps=int((motion_settings or {}).get("phase_timeout_steps", GRIPPER_OPEN_TIMEOUT_STEPS)),
            gripper_dwell_steps=int((motion_settings or {}).get("gripper_dwell_steps", 20)),
            stop_after_phase="open", start_phase="open",
            dry_run=False, action_budget=budget if budget is not None else remaining,
            motion_started_callback=motion_started_callback,
            motion_trace_path=output_dir / "gripper_open_trace.json",
            motion_trace_segment="gripper_open_preflight",
            stall_window_steps=int((motion_settings or {}).get("stall_window_steps", 0)),
            stall_delta_m=float((motion_settings or {}).get("stall_delta_m", DEFAULT_STALL_DELTA_M)),
            osc_position_scale_m=(motion_settings or {}).get("osc_position_scale_m"),
            phase_policies=(motion_settings or {}).get("phase_policies"),
            **motion_diag_kwargs,
        )
        steps = int(sum(int(item.get("steps", 0)) for item in phases))
        total_actions = int(budget.used) if budget is not None else prior_actions + steps
        if total_actions > 1200:
            raise RuntimeError("gripper-open preflight exceeded 1200-action budget")
        setattr(env, "_molmo_sam3_action_count", total_actions)
        audit = {
            "status": "completed", "eef_world_m": current.tolist(),
            "steps": steps, "total_actions": total_actions,
            "phase_statuses": [{"phase": item.get("phase"), "status": item.get("status")} for item in phases],
            "stop_after_phase": "open", "gripper_command": -1.0,
        }
        setattr(env, "_molmo_sam3_gripper_open", audit)
        return audit
    except Exception as exc:
        audit = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        setattr(env, "_molmo_sam3_gripper_open", audit)
        raise


def _perform_observation_hover(
    env: Any,
    capture: Any,
    source_uv: Sequence[float],
    *,
    output_dir: Path,
    motion_started_callback: Callable[[], None] | None = None,
    motion_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Move to the fixed 10-cm observation hover used by every source arm."""
    requested_profile = (motion_settings or {}).get("observation_profile", "baseline")
    requested_name = requested_profile.get("name", "baseline") if isinstance(requested_profile, Mapping) else requested_profile
    observation_policy = resolve_observation_profile(str(requested_name))
    profile_audit: dict[str, Any] = {
        "observation_profile": observation_policy["name"],
        "observation_profile_params": observation_policy,
    }
    try:
        try:
            from . import run_arrow_pick_place_eval as episode
        except ImportError:  # pragma: no cover - direct script use
            import run_arrow_pick_place_eval as episode
        observed_world = _deproject_capture(capture, source_uv)
        # Derive the same arrow-seeded RGB-D support used by the experimental
        # worker and use its observed world-Z upper quantile for clearance.
        # This remains image/depth/calibration evidence only; no simulator or
        # object pose is consulted.
        try:
            from .rgbd_region import derive_observed_region_mask
        except ImportError:  # pragma: no cover - direct script use
            from rgbd_region import derive_observed_region_mask
        region_mask, region_audit = derive_observed_region_mask(
            capture, source_uv, profile_target_world=observed_world,
        )
        depth = np.asarray(capture.metric_depth, dtype=np.float64)
        calibration = capture.calibration
        K = np.asarray(calibration.intrinsic, dtype=np.float64)
        T = np.asarray(calibration.world_from_camera, dtype=np.float64)
        capture_provenance = _observation_capture_provenance(capture)
        ys, xs = np.nonzero(np.asarray(region_mask, dtype=bool))
        if len(xs) == 0:
            raise RuntimeError("observation hover region has no RGB-D support")
        depths = depth[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.0)
        if not np.any(valid):
            raise RuntimeError("observation hover region has no valid metric depth")
        xs_valid, ys_valid, z_valid = xs[valid].astype(np.float64), ys[valid].astype(np.float64), depths[valid]
        camera_points = np.column_stack(((xs_valid - K[0, 2]) * z_valid / K[0, 0],
                                         (ys_valid - K[1, 2]) * z_valid / K[1, 1], z_valid))
        world_points = (T[:3, :3] @ camera_points.T).T + T[:3, 3]
        world_z = world_points[:, 2]
        world_z = world_z[np.isfinite(world_z)]
        if world_z.size == 0:
            raise RuntimeError("observation hover region has no finite world-Z support")
        region_q90_world_z = float(np.percentile(world_z, 90.0))
        anchor_z = float(observed_world[2])
        target_offset_m = float(observation_policy["target_offset_m"])
        hover_z = max(anchor_z, region_q90_world_z) + target_offset_m
        hover = np.asarray((observed_world[0], observed_world[1], hover_z), dtype=np.float64)
        observation = episode._raw_observation(env)
        proprio = episode._proprioception(observation)
        current = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        held_rotation = np.asarray(proprio.get("eef_quat"), dtype=np.float64).reshape(-1)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise RuntimeError("observation hover lacks finite current EEF position")
        if held_rotation.size not in (3, 4, 9) or not np.all(np.isfinite(held_rotation)):
            raise RuntimeError("observation hover lacks finite EEF orientation proprioception")
        clearance_z = max(float(current[2]), hover_z)
        raise_waypoint = episode._pose_waypoint((current[0], current[1], clearance_z), held_rotation)
        translate_waypoint = episode._pose_waypoint((observed_world[0], observed_world[1], clearance_z), held_rotation)
        lower_waypoint = episode._pose_waypoint(hover, held_rotation)
        episode.validate_workspace_points({
            "hover_raise": raise_waypoint["position"],
            "hover_translate": translate_waypoint["position"],
            "hover_lower": lower_waypoint["position"],
        })
        if not np.all(np.isfinite(hover)) or hover[2] > 1.75:
            raise RuntimeError("observation hover lies outside the calibrated workspace")
        waypoints = {
            "hover_raise": raise_waypoint,
            "hover_translate": translate_waypoint,
            "hover_lower": lower_waypoint,
        }
        budget = getattr(env, "_molmo_sam3_action_budget", None)
        prior_actions = int(getattr(env, "_molmo_sam3_action_count", 0))
        remaining = int(budget.remaining) if budget is not None else max(0, 1200 - prior_actions)
        if remaining <= 0:
            raise RuntimeError("1200-action budget exhausted before observation hover")
        policies = {name: dict(policy) for name, policy in (motion_settings or {}).get("phase_policies", {}).items()}
        staging_policy = policies.get("pregrasp", {"tolerance_m": 0.015})
        phase_tolerance_m = (
            float(observation_policy["phase_tolerance_m"])
            if observation_policy["name"] == "hover20mm"
            else float(staging_policy.get("tolerance_m", 0.015))
        )
        for name in ("hover_raise", "hover_translate", "hover_lower"):
            policies[name] = {"role": "experimental_hover", "tolerance_m": phase_tolerance_m}
        motion_diag_kwargs = {"experimental_motion_diagnostics": True} if (motion_settings or {}).get("motion_diagnostics") else {}
        phases = episode._run_motion(
            env, waypoints, observation,
            phase_timeout_steps=int((motion_settings or {}).get("phase_timeout_steps", 160)),
            gripper_dwell_steps=int((motion_settings or {}).get("gripper_dwell_steps", 20)),
            stop_after_phase="hover_lower", start_phase="hover_raise", dry_run=False,
            phase_order=("hover_raise", "hover_translate", "hover_lower"),
            phase_timeout_steps_by_phase={name: 160 for name in ("hover_raise", "hover_translate", "hover_lower")},
            action_budget=budget if budget is not None else remaining,
            motion_started_callback=motion_started_callback,
            motion_trace_path=output_dir / "observation_hover_trace.json",
            motion_trace_segment="observation_hover",
            # Hover has a fixed target and must not trigger the legacy
            # ten-sample stall detector.  Keep this explicit in the call.
            stall_window_steps=0,
            stall_delta_m=float((motion_settings or {}).get("stall_delta_m", DEFAULT_STALL_DELTA_M)),
            osc_position_scale_m=(motion_settings or {}).get("osc_position_scale_m"),
            phase_policies=policies,
            orientation_tolerance_rad=float(observation_policy["orientation_tolerance_rad"]),
            **motion_diag_kwargs,
        )
        phase_records: list[dict[str, Any]] = []
        required_height_margin_m = observation_policy["min_actual_height_margin_m"]
        require_height = bool(observation_policy["require_actual_height_margin"])
        motion_trace = list(getattr(env, "_arrow_motion_trace", []) or [])
        for phase_name in ("hover_raise", "hover_translate", "hover_lower"):
            matches = [item for item in phases if isinstance(item, Mapping) and item.get("phase") == phase_name]
            record = matches[-1] if matches else None
            commanded = np.asarray(waypoints[phase_name]["position"], dtype=np.float64).reshape(-1)
            if commanded.shape != (3,) or not np.all(np.isfinite(commanded)):
                raise RuntimeError(f"observation hover {phase_name} has invalid commanded position")
            actual_value = None if record is None else record.get("eef_pos_m")
            actual = np.asarray(actual_value, dtype=np.float64).reshape(-1) if actual_value is not None else np.asarray([], dtype=np.float64)
            position_error = None if record is None else record.get("position_error_norm_m")
            if position_error is None and actual.shape == (3,) and np.all(np.isfinite(actual)):
                position_error = float(np.linalg.norm(commanded - actual))
            orientation_error = None if record is None else record.get("orientation_error_rad")
            if orientation_error is None:
                trace_records = [
                    item for item in motion_trace
                    if isinstance(item, Mapping) and item.get("phase") == phase_name
                ]
                if trace_records:
                    trace_orientation = trace_records[-1].get("eef_quat_xyzw")
                    if trace_orientation is not None:
                        try:
                            orientation_error = _orientation_error_rad(trace_orientation, held_rotation)
                        except (TypeError, ValueError, FloatingPointError):
                            orientation_error = None
            if orientation_error is not None:
                try:
                    orientation_error = float(orientation_error)
                except (TypeError, ValueError):
                    raise RuntimeError(f"observation hover {phase_name} has invalid orientation error")
                if not np.isfinite(orientation_error):
                    raise RuntimeError(f"observation hover {phase_name} has non-finite orientation error")
            detail = {
                "phase": phase_name,
                "status": None if record is None else record.get("status"),
                "commanded_position_m": commanded.tolist(),
                "actual_position_m": actual.tolist() if actual.shape == (3,) and np.all(np.isfinite(actual)) else None,
                "position_error_norm_m": float(position_error) if position_error is not None and np.isfinite(position_error) else None,
                "orientation_error_rad": float(orientation_error) if orientation_error is not None and np.isfinite(orientation_error) else None,
            }
            phase_records.append(detail)
            if require_height:
                if record is None or detail["status"] not in {"reached", "stop"}:
                    raise RuntimeError(f"observation hover {phase_name} did not complete: {detail}")
                if detail["actual_position_m"] is None:
                    raise RuntimeError(f"observation hover {phase_name} has no finite completed EEF position: {detail}")
                if detail["position_error_norm_m"] is None or detail["position_error_norm_m"] > phase_tolerance_m:
                    raise RuntimeError(
                        f"observation hover {phase_name} position error exceeds {phase_tolerance_m:.6f} m: {detail}"
                    )
                if detail["orientation_error_rad"] is not None and detail["orientation_error_rad"] > float(observation_policy["orientation_tolerance_rad"]):
                    raise RuntimeError(
                        f"observation hover {phase_name} orientation error exceeds "
                        f"{float(observation_policy['orientation_tolerance_rad']):.6f} rad: {detail}"
                    )
                height_margin = float(actual[2] - region_q90_world_z)
                detail["actual_height_margin_m"] = height_margin
                if height_margin < float(required_height_margin_m):
                    raise RuntimeError(
                        f"observation hover {phase_name} actual height margin {height_margin:.6f} m "
                        f"below {float(required_height_margin_m):.6f} m: {detail}"
                    )
        fresh_proprio_detail: dict[str, Any] | None = None
        if require_height:
            fresh_observation = episode._raw_observation(env)
            fresh_proprio = episode._proprioception(fresh_observation)
            fresh_position = np.asarray(fresh_proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
            fresh_orientation = np.asarray(fresh_proprio.get("eef_quat"), dtype=np.float64).reshape(-1)
            if fresh_position.shape != (3,) or not np.all(np.isfinite(fresh_position)):
                raise RuntimeError("observation hover final fresh proprioception lacks finite EEF position")
            if fresh_orientation.size not in (3, 4, 9) or not np.all(np.isfinite(fresh_orientation)):
                raise RuntimeError("observation hover final fresh proprioception lacks finite EEF orientation")
            try:
                final_orientation_error = _orientation_error_rad(fresh_orientation, held_rotation)
            except (TypeError, ValueError, FloatingPointError) as exc:
                raise RuntimeError("observation hover final fresh proprioception has invalid orientation") from exc
            final_height_margin = float(fresh_position[2] - region_q90_world_z)
            final_position_error = float(np.linalg.norm(fresh_position - hover))
            if final_position_error > phase_tolerance_m:
                raise RuntimeError(
                    f"observation hover final EEF position error {final_position_error:.6f} m "
                    f"exceeds {phase_tolerance_m:.6f} m"
                )
            if final_orientation_error > float(observation_policy["orientation_tolerance_rad"]):
                raise RuntimeError(
                    f"observation hover final orientation error {final_orientation_error:.6f} rad "
                    f"exceeds {float(observation_policy['orientation_tolerance_rad']):.6f} rad"
                )
            if final_height_margin < float(required_height_margin_m):
                raise RuntimeError(
                    f"observation hover final EEF height margin {final_height_margin:.6f} m "
                    f"below {float(required_height_margin_m):.6f} m"
                )
            fresh_proprio_detail = {
                "eef_pos_m": fresh_position.tolist(),
                "eef_orientation": fresh_orientation.tolist(),
                "orientation_error_rad": final_orientation_error,
                "position_error_norm_m": final_position_error,
                "actual_height_margin_m": final_height_margin,
            }
        audit = {
            "status": "completed",
            "hover_world_m": hover.tolist(),
            "source_world_m": observed_world.tolist(),
            "anchor_world_z_m": anchor_z,
            "region_q90_world_z_m": region_q90_world_z,
            "region_area_px": int(np.count_nonzero(region_mask)),
            "region_audit": region_audit,
            "clearance_z_m": clearance_z,
            "steps": int(sum(int(item.get("steps", 0)) for item in phases)),
            "phase_statuses": [{"phase": item.get("phase"), "status": item.get("status")} for item in phases],
            "fixed_offset_m": target_offset_m,
            "observation_profile": observation_policy["name"],
            "observation_profile_params": observation_policy,
            "phase_tolerance_m": phase_tolerance_m,
            "orientation_tolerance_rad": float(observation_policy["orientation_tolerance_rad"]),
            "commanded_positions_m": {item["phase"]: item["commanded_position_m"] for item in phase_records},
            "actual_positions_m": {item["phase"]: item["actual_position_m"] for item in phase_records},
            "phase_motion_audit": phase_records,
            "capture_provenance": capture_provenance,
            "height_margin_threshold_m": required_height_margin_m,
            "final_fresh_proprio": fresh_proprio_detail,
        }
        if budget is not None:
            setattr(env, "_molmo_sam3_action_count", int(budget.used))
        else:
            setattr(env, "_molmo_sam3_action_count", int(getattr(env, "_molmo_sam3_action_count", 0)) + audit["steps"])
        setattr(env, "_molmo_sam3_observation_hover", audit)
        return audit
    except Exception as exc:
        audit = {"status": "failed", **profile_audit, "error_type": type(exc).__name__, "error": str(exc)}
        setattr(env, "_molmo_sam3_observation_hover", audit)
        raise


def main(
    argv: Sequence[str] | None = None,
    *,
    molmo_runtime: Any | None = None,
    cell_completed_callback: Callable[[Mapping[str, Any]], Any] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="molmo_sam3_canary")
    parser.add_argument("--phase", choices=("prefix", "full60"), default="prefix")
    parser.add_argument("--task-ids", default="4,6,9")
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--suite-modes", default="vanilla,sealed_randomized")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--region-backend", choices=("sam3", "rgbd"), default="sam3")
    parser.add_argument("--motion-profile", choices=MOTION_PROFILE_NAMES, default="baseline")
    parser.add_argument("--motion-diagnostics", action="store_true", help="record bounded motion diagnostics")
    parser.add_argument("--observation-profile", choices=OBSERVATION_PROFILE_NAMES, default="baseline")
    parser.add_argument("--sam3-source", type=Path, required=False)
    parser.add_argument("--sam3-checkpoint", type=Path, required=False)
    parser.add_argument("--sam3-source-commit", required=False)
    parser.add_argument("--sam3-checkpoint-sha256", required=False)
    parser.add_argument("--molmopoint-model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--molmopoint-revision", default=MOLMOPOINT_MODEL_REVISION)
    parser.add_argument("--molmopoint-prompt-id", choices=MOLMOPOINT_PROMPT_IDS, default="rim_downward_approach")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate local model assets without motion")
    args = parser.parse_args(argv)
    if args.episodes_per_task is None:
        args.episodes_per_task = 2 if args.phase == "prefix" else 10
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    try:
        resolved_motion_profile = resolve_motion_profile(args.motion_profile, region_backend=args.region_backend)
        resolved_observation_profile = resolve_observation_profile(args.observation_profile)
        if VARIANTS[args.variant].region_backend == "rgbd" and args.region_backend != "rgbd":
            raise ValueError("rgbd_geometry_agentview requires --region-backend rgbd")
        tasks = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
        suites = tuple(item.strip() for item in str(args.suite_modes).split(",") if item.strip())
        if tasks != (4, 6, 9):
            raise ValueError("canary task IDs must be exactly 4,6,9")
        if suites != ("vanilla", "sealed_randomized"):
            raise ValueError("canary suite modes must be exactly vanilla,sealed_randomized")
        if args.seed_base != 1000:
            raise ValueError("canary seed base must be 1000")
        if args.molmopoint_model != MOLMOPOINT_MODEL_ID:
            raise ValueError("MolmoPoint model must be the pinned canary model")
        if args.molmopoint_revision != MOLMOPOINT_MODEL_REVISION:
            raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
        if args.config is not None and not args.config.is_file():
            raise ValueError(f"canary config is unavailable: {args.config}")
        if args.region_backend == "rgbd":
            # Campaigns inject one runtime and reuse its loaded weights across
            # sequential arms.  Prompt text remains an explicit per-arm config.
            uses_molmo = bool(VARIANTS[args.variant].uses_molmo)
            if not uses_molmo:
                # The geometry control is intentionally model-free, even when
                # a campaign has a runtime available for neighboring arms.
                sam3 = None
                molmo_runtime = None
                molmo = None
                model_provenance = {
                    "model_id": None, "model_revision": None,
                    "prompt_id": None, "prompt": None, "models_loaded": False,
                }
            else:
                if molmo_runtime is None:
                    molmo_runtime = build_local_molmo_runtime(
                        molmopoint_model=args.molmopoint_model,
                        molmopoint_revision=args.molmopoint_revision,
                        molmopoint_prompt_id=args.molmopoint_prompt_id,
                    )
                try:
                    try:
                        from .molmo_sam3.molmopoint import PROMPT_VARIANTS
                    except ImportError:  # pragma: no cover - direct script use
                        from molmo_sam3.molmopoint import PROMPT_VARIANTS
                    config = getattr(molmo_runtime, "config", None)
                    if config is None:
                        raise RuntimeError("injected Molmo runtime is missing config")
                    molmo_runtime.config = replace(
                        config, prompt_id=args.molmopoint_prompt_id,
                        prompt=PROMPT_VARIANTS[args.molmopoint_prompt_id],
                    )
                except (KeyError, TypeError, AttributeError) as exc:
                    raise RuntimeError("injected Molmo runtime cannot update prompt config") from exc
                sam3 = None
                model_provenance = preflight_local_molmo_runtime(molmo_runtime, load_models=not args.dry_run)
            molmo = molmo_runtime
        else:
            missing = [name for name in ("sam3_source", "sam3_checkpoint", "sam3_source_commit", "sam3_checkpoint_sha256") if getattr(args, name) is None]
            if missing:
                raise ValueError(f"SAM3 backend requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
            if args.sam3_source_commit != SAM3_SOURCE_COMMIT:
                raise ValueError("SAM3 source revision does not match the pinned project-local revision")
            if str(args.sam3_checkpoint_sha256).lower() != SAM3_CHECKPOINT_SHA256:
                raise ValueError("SAM3 checkpoint digest does not match the pinned canary artifact")
            sam3, molmo = build_local_model_runtimes(
                sam3_source=args.sam3_source, sam3_checkpoint=args.sam3_checkpoint,
                sam3_checkpoint_sha256=args.sam3_checkpoint_sha256,
                molmopoint_model=args.molmopoint_model, molmopoint_revision=args.molmopoint_revision,
                molmopoint_prompt_id=args.molmopoint_prompt_id,
            )
            model_provenance = preflight_local_model_runtimes(sam3, molmo, load_models=not args.dry_run)
    except Exception as exc:
        print(f"molmo/sam3 canary preflight failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    try:
        controller_config = args.config or (Path(__file__).resolve().parent / "controller_configs" / ACTIVE_CONTROLLER_CONFIG_FILENAME)
        if not controller_config.is_file():
            raise RuntimeError(f"canary controller config is unavailable: {controller_config}")
        controller_config_digest = _resolved_controller_config_digest(controller_config)
        execution_provenance = _execution_provenance(require_clean=not args.dry_run)
    except Exception as exc:
        print(f"molmo/sam3 execution provenance failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    scientific_identity_payload, scientific_identity_hash = build_scientific_identity(
        execution_sha=execution_provenance["execution_sha"],
        controller_config_digest=controller_config_digest,
        model_provenance=model_provenance,
        variant=args.variant,
        candidate_policy=VARIANTS[args.variant].policy,
        camera_name=VARIANTS[args.variant].camera_name,
        region_backend=args.region_backend,
        backend="v9d_rgbd_region" if args.region_backend == "rgbd" else "sam3",
        motion_profile=resolved_motion_profile["name"],
        motion_profile_params=resolved_motion_profile,
        motion_diagnostics=bool(args.motion_diagnostics),
        observation_profile=resolved_observation_profile["name"],
        observation_profile_params=resolved_observation_profile,
        task_ids=tasks,
        seed_base=args.seed_base,
        suite_modes=suites,
    )
    manifest = {
        "schema_version": RGBD_EXPERIMENT_SCHEMA if args.region_backend == "rgbd" else SAM_EXPERIMENT_SCHEMA,
        "experiment_id": str(args.label), "baseline_commit": BASELINE_COMMIT,
        "variant": asdict(replace(VARIANTS[args.variant], region_backend=args.region_backend)), "phase": args.phase,
        "task_ids": list(tasks), "episodes_per_task": int(args.episodes_per_task),
        "seed_base": int(args.seed_base), "suite_modes": list(suites),
        "output_dir": args.output_dir.expanduser().resolve().as_posix(),
        "model_provenance": model_provenance,
        "region_backend": args.region_backend,
        "backend": "v9d_rgbd_region" if args.region_backend == "rgbd" else "sam3",
        "sam3_used": bool(args.region_backend == "sam3"),
        "molmopoint_prompt_id": args.molmopoint_prompt_id,
        "dry_run": bool(args.dry_run),
        "motion_profile": resolved_motion_profile["name"],
        "motion_profile_params": _json_safe(resolved_motion_profile),
        "motion_diagnostics": bool(args.motion_diagnostics),
        "observation_profile": resolved_observation_profile["name"],
        "observation_profile_params": _json_safe(resolved_observation_profile),
        "execution_provenance": execution_provenance,
        "scientific_identity_payload": scientific_identity_payload,
        "scientific_identity_hash": scientific_identity_hash,
    }
    manifest["experiment_configuration_hash"] = scientific_identity_hash
    if args.dry_run:
        _write_json(args.output_dir.expanduser().resolve() / "molmo_sam3_canary_preflight.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0

    # Live canary path.  The matrix remains the existing evaluator/motion
    # implementation; this wrapper injects only the project-local perception
    # worker, candidate pose, and fresh-frame retry policy.
    try:
        try:
            from . import run_arrow_pick_place_eval as episode
            from . import run_arrow_pick_place_matrix as matrix
        except ImportError:  # pragma: no cover - direct script use
            import run_arrow_pick_place_eval as episode
            import run_arrow_pick_place_matrix as matrix
    except ImportError as exc:  # pragma: no cover - dependency-free host
        print(f"live canary imports unavailable: {exc}", file=sys.stderr)
        return 2

    experimental_micro_correction = None
    if resolved_motion_profile["micro_correction"] is not None:
        experimental_micro_correction = episode.MicroCorrectionPolicy(
            **dict(resolved_motion_profile["micro_correction"])
        )

    controller_config = args.config or (Path(__file__).resolve().parent / "controller_configs" / ACTIVE_CONTROLLER_CONFIG_FILENAME)
    if not controller_config.is_file():
        print(f"live canary controller config is unavailable: {controller_config}", file=sys.stderr)
        return 2
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_state: dict[str, Any] = {"worker": None, "transform": None, "robot_calibration": None, "opening_m": None}

    def env_builder(task_id: int, seed: int, resolution: int, *, suite_mode: str | None = None, controller_variant: Any = None) -> Any:
        env = episode.build_libero_env(
            task_id, seed, resolution, suite_mode=suite_mode,
            controller_variant=controller_variant,
            extra_camera_names=(WRIST_CAMERA,),
        )
        calibration, transform, probe = probe_robot_calibration(env)
        worker_state["robot_calibration"] = calibration
        worker_state["transform"] = transform
        geometry = probe.get("gripper_geometry") if isinstance(probe, Mapping) else None
        worker_state["opening_m"] = (
            float(geometry["measured_opening_m"])
            if isinstance(geometry, Mapping) and geometry.get("measured_opening_m") is not None
            else None
        )
        if worker_state["worker"] is None:
            worker_state["worker"] = ModelPerceptionWorker(
                sam3, molmo, calibration, region_backend=args.region_backend,
            )
        else:
            worker_state["worker"].robot_calibration = calibration
        setattr(env, "_molmo_sam3_model_provenance", model_provenance)
        setattr(env, "_molmo_sam3_probe_sha256", hashlib.sha256(json.dumps(_json_safe(probe), sort_keys=True).encode()).hexdigest())
        # One shared budget covers the observation hover, every candidate
        # motion, and failed-candidate recovery.  It is initialized before the
        # first capture because the hover itself sends actions.
        try:
            from . import run_arrow_pick_place_eval as _budget_module
        except ImportError:  # pragma: no cover - direct script use
            import run_arrow_pick_place_eval as _budget_module
        setattr(env, "_molmo_sam3_action_budget", _budget_module._ActionBudget(1200))
        setattr(env, "_molmo_sam3_action_count", 0)
        return env

    def episode_runner(**kwargs: Any) -> Mapping[str, Any]:
        env = kwargs["env"]
        task_id, seed = int(kwargs["task_id"]), int(kwargs["seed"])
        output_dir = Path(kwargs["output_dir"]).expanduser().resolve()
        resolution = int(kwargs.get("resolution", 256))
        dry_run = bool(kwargs.get("dry_run", False))
        suite_mode = kwargs.get("suite_mode") or "vanilla"
        variant_name = args.variant
        variant = VARIANTS[variant_name]
        # CLI backend selection is part of the experimental identity and is
        # reflected in every cell, including legacy variant names.
        variant = replace(variant, region_backend=args.region_backend)
        frozen_motion_variant = episode._resolve_controller_variant(
            kwargs.get("controller_variant"), suite_mode=suite_mode
        )
        motion_settings = {
            "phase_timeout_steps": int(frozen_motion_variant.phase_timeout_steps),
            "gripper_dwell_steps": int(frozen_motion_variant.gripper_dwell_steps),
            "stall_window_steps": int(frozen_motion_variant.stall_window_steps),
            "stall_delta_m": float(frozen_motion_variant.stall_delta_m),
            "osc_position_scale_m": frozen_motion_variant.osc_position_scale_m,
            "motion_diagnostics": bool(args.motion_diagnostics),
        }
        phase_policies = {name: dict(policy) for name, policy in episode.PHASE_POLICIES.items()}
        if frozen_motion_variant.approach_tolerance_m is not None:
            for name in ("descend", "descend_place"):
                phase_policies[name]["tolerance_m"] = float(frozen_motion_variant.approach_tolerance_m)
        if frozen_motion_variant.waypoint_tolerance_m is not None:
            for name in episode.PHASES:
                if name not in {"close", "open"}:
                    phase_policies[name]["tolerance_m"] = float(frozen_motion_variant.waypoint_tolerance_m)
        staging_tolerance = float(
            frozen_motion_variant.waypoint_tolerance_m
            if frozen_motion_variant.waypoint_tolerance_m is not None
            else 0.015
        )
        for name in ("vertical_clearance", "rotate", "translate_clearance", "pregrasp"):
            phase_policies[name] = {
                "role": "experimental_staging",
                "tolerance_m": staging_tolerance,
            }
        motion_settings["phase_policies"] = phase_policies
        motion_settings["motion_profile"] = resolved_motion_profile["name"]
        motion_settings["motion_profile_params"] = _json_safe(resolved_motion_profile)
        motion_settings["observation_profile"] = resolved_observation_profile
        bboxes = kwargs.get("bboxes")
        if not isinstance(bboxes, Mapping):
            raise ValueError("canary episode requires existing arrow input-generation bboxes")
        initial = episode.capture_agentview(env, resolution=resolution, camera_name=AGENTVIEW)
        provisional_arrow, _ = episode.render_exactly_one_arrow(
            initial.rgb, bboxes, subject=kwargs.get("subject", "bowl"),
            goal_object=kwargs.get("goal_object", "plate"), anchor_policy="bbox_center",
        )
        provisional_source, _ = episode.decode_arrow_pixels(initial.rgb, provisional_arrow)
        hover_output = output_dir / "observation_hover"
        motion_callback = kwargs.get("motion_started_callback")
        # Contact-pad aperture is measured only after a full open dwell.  The
        # shared budget includes this preflight before any perception request.
        _perform_gripper_open(
            env, output_dir=output_dir / "gripper_open",
            motion_started_callback=motion_callback,
            motion_settings=motion_settings,
        )
        calibration, transform, probe = probe_robot_calibration(env)
        worker_state["robot_calibration"] = calibration
        worker_state["transform"] = transform
        geometry = probe.get("gripper_geometry") if isinstance(probe, Mapping) else None
        worker_state["opening_m"] = (
            float(geometry["measured_opening_m"])
            if isinstance(geometry, Mapping) and geometry.get("measured_opening_m") is not None
            else None
        )
        if worker_state.get("worker") is None:
            raise RuntimeError("gripper-open preflight completed without an initialized perception worker")
        worker_state["worker"].robot_calibration = calibration
        _perform_observation_hover(
            env, initial, provisional_source, output_dir=hover_output,
            motion_started_callback=motion_callback,
            motion_settings=motion_settings,
        )
        fresh = episode.capture_agentview(env, resolution=resolution, camera_name=AGENTVIEW)
        hover_audit = getattr(env, "_molmo_sam3_observation_hover", None)
        if isinstance(hover_audit, dict):
            hover_audit["fresh_capture_provenance"] = _observation_capture_provenance(fresh)
        arrow_state: dict[str, Any] = {"rgb": None, "source_uv": None, "destination_uv": None}
        first_capture: dict[str, Any] = {AGENTVIEW: fresh}

        def refresh_arrow(capture: Any) -> tuple[np.ndarray, Sequence[float], Sequence[float] | None]:
            rendered, _ = episode.render_exactly_one_arrow(
                capture.rgb, bboxes, subject=kwargs.get("subject", "bowl"),
                goal_object=kwargs.get("goal_object", "plate"), anchor_policy="bbox_center",
            )
            source_point, target_point = episode.decode_arrow_pixels(capture.rgb, rendered)
            arrow_state.update({"rgb": rendered, "source_uv": source_point, "destination_uv": target_point})
            return rendered, source_point, target_point

        refresh_arrow(fresh)

        def capture_fn(capture_env: Any, *, resolution: int, camera_name: str) -> Any:
            cached = first_capture.pop(camera_name, None)
            if cached is not None:
                return cached
            return episode.capture_agentview(capture_env, resolution=resolution, camera_name=camera_name)

        def refresh_robot_calibration(_env: Any) -> None:
            # This callback runs after the fresh RGB-D capture and immediately
            # before worker.propose, including every post-recovery retry.
            calibration, refreshed_transform, probe = probe_robot_calibration(_env)
            worker_state["robot_calibration"] = calibration
            worker_state["transform"] = refreshed_transform
            geometry = probe.get("gripper_geometry") if isinstance(probe, Mapping) else None
            worker_state["opening_m"] = (
                float(geometry["measured_opening_m"])
                if isinstance(geometry, Mapping) and geometry.get("measured_opening_m") is not None
                else None
            )
            worker_state["worker"].robot_calibration = calibration
            setattr(_env, "_molmo_sam3_robot_calibration_probe", probe)

        transform = worker_state.get("transform")
        if transform is None or worker_state.get("worker") is None:
            raise RuntimeError("Panda calibration/model worker was not initialized")

        def run_one(*, context: CanaryEpisodeContext, evaluator: Callable[[Any], bool] | None,
                    retreat_completed_callback: Callable[[], None] | None = None) -> Mapping[str, Any]:
            action_count_before = int(getattr(env, "_molmo_sam3_action_count", 0))
            trace_before = getattr(env, "_arrow_motion_trace", None)
            # Do not let a previous candidate's phase audit leak into a
            # pre-motion failure or into the recovery snapshot for this one.
            setattr(env, "_arrow_phase_audit", [])
            attempt_output_dir = output_dir / "attempts" / f"attempt_{int(context.attempt_index):02d}"
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                from . import run_arrow_pick_place_eval as _episode_budget_module
            except ImportError:  # pragma: no cover - direct script use
                import run_arrow_pick_place_eval as _episode_budget_module
            budget = getattr(env, "_molmo_sam3_action_budget", None)
            if budget is None:
                budget = _episode_budget_module._ActionBudget(1200, used=action_count_before)
                setattr(env, "_molmo_sam3_action_budget", budget)

            if action_count_before >= 1200:
                return {"status": "recovery_failed", "grasp_retained": False, "retreat_complete": False,
                        "total_actions": action_count_before,
                        "error": "1200-action retry budget exhausted before candidate"}
            transform = worker_state.get("transform")
            if transform is None or worker_state.get("worker") is None:
                return {"status": "recovery_failed", "grasp_retained": False,
                        "retreat_complete": False, "total_actions": action_count_before,
                        "error": "live contact calibration unavailable before candidate"}
            try:
                candidate_motion_kwargs = _candidate_motion_kwargs(
                    resolved_motion_profile,
                    experimental_micro_correction=experimental_micro_correction,
                    motion_diagnostics=bool(args.motion_diagnostics),
                )
                audit = episode.run_episode(
                    env=env, task_id=task_id, seed=seed, output_dir=attempt_output_dir,
                    arrow_rgb=arrow_state["rgb"], dry_run=dry_run, resolution=resolution,
                    goal_object=kwargs.get("goal_object", "plate"), subject=kwargs.get("subject", "bowl"),
                    allow_unvalidated_profile=True, controller_variant=kwargs.get("controller_variant"),
                    suite_mode=suite_mode, evaluator=evaluator,
                    capture=context.agentview_capture,
                    motion_started_callback=kwargs.get("motion_started_callback"),
                    canary_video_dir=kwargs.get("canary_video_dir"),
                    experimental_candidate=context.candidate,
                    experimental_eef_orientation_transform=transform,
                    experimental_gripper_opening_m=worker_state.get("opening_m"),
                    post_lift_retention_gate=retention_gate,
                    retreat_completed_callback=retreat_completed_callback,
                    experimental_action_budget=getattr(env, "_molmo_sam3_action_budget", None),
                    **candidate_motion_kwargs,
                )
                trace_steps = _newly_sent_actions(env, trace_before)
                if getattr(env, "_molmo_sam3_action_budget", None) is not None:
                    setattr(env, "_molmo_sam3_action_count", int(env._molmo_sam3_action_budget.used))
                else:
                    setattr(env, "_molmo_sam3_action_count", action_count_before + trace_steps)
                if int(getattr(env, "_molmo_sam3_action_count", 0)) > 1200:
                    raise RuntimeError("candidate action budget exceeded 1200 steps")
                return {
                    "status": "placed" if audit.get("evaluator_success") is not False else "task_failure",
                    "grasp_retained": True, "retreat_complete": True,
                    "total_actions": int(getattr(env, "_molmo_sam3_action_count", 0)),
                    "evaluator_called": bool(evaluator is not None),
                    "evaluator_success": audit.get("evaluator_success"),
                    "audit": audit,
                    "motion_phases_reached": _completed_motion_phases(audit.get("phases", [])),
                }
            except Exception as exc:
                if "evaluator failure" in str(exc).lower():
                    raise
                trace_steps = _newly_sent_actions(env, trace_before)
                if getattr(env, "_molmo_sam3_action_budget", None) is not None:
                    setattr(env, "_molmo_sam3_action_count", int(env._molmo_sam3_action_budget.used))
                else:
                    setattr(env, "_molmo_sam3_action_count", action_count_before + trace_steps)
                attempt_phases = list(getattr(env, "_arrow_phase_audit", []) or [])
                reached = _completed_motion_phases(attempt_phases)
                recovery = _recover_after_failed_candidate(
                    env, output_dir=attempt_output_dir,
                    orientation_transform=transform, motion_settings=motion_settings,
                )
                return {
                    "status": "grasp_failed" if recovery.get("retreat_complete") else "recovery_failed",
                    "grasp_retained": False, "retreat_complete": bool(recovery.get("retreat_complete")),
                    "total_actions": int(getattr(env, "_molmo_sam3_action_count", 0)),
                    "error_type": type(exc).__name__, "error": str(exc), "recovery": recovery,
                    "motion_phases_reached": reached,
                    "attempt_phases": attempt_phases,
                }

        result = run_canary_episode(
            env=env, task_id=task_id, seed=seed, output_dir=output_dir,
            variant=variant, worker=worker_state["worker"], episode_runner=run_one,
            source_uv=arrow_state["source_uv"], destination_uv=arrow_state["destination_uv"],
            evaluator=kwargs.get("evaluator"), capture_fn=capture_fn,
            resolution=resolution, dry_run=dry_run, arrow_rgb=arrow_state["rgb"],
            arrow_refresh_fn=refresh_arrow,
            experiment_schema=(RGBD_EXPERIMENT_SCHEMA if args.region_backend == "rgbd" else SAM_EXPERIMENT_SCHEMA),
            region_backend=args.region_backend,
            before_propose_callback=refresh_robot_calibration,
            motion_profile=resolved_motion_profile["name"],
            motion_profile_params=resolved_motion_profile,
            motion_diagnostics=bool(args.motion_diagnostics),
            observation_profile=resolved_observation_profile["name"],
            observation_profile_params=resolved_observation_profile,
        )
        final = result.get("final_result") if isinstance(result.get("final_result"), Mapping) else {}
        audit = final.get("audit") if isinstance(final, Mapping) else None
        return {
            "audit_path": (output_dir / "molmo_sam3_canary_manifest.json").as_posix(),
            "evaluator_success": final.get("evaluator_success") if isinstance(final, Mapping) else None,
            "total_actions": int(getattr(env, "_molmo_sam3_action_count", 0)),
            "gripper_open_preflight": getattr(env, "_molmo_sam3_gripper_open", None),
            "observation_hover": getattr(env, "_molmo_sam3_observation_hover", None),
            "phases": audit.get("phases", []) if isinstance(audit, Mapping) else [],
            "motion_phases_reached": final.get("motion_phases_reached", []) if isinstance(final, Mapping) else [],
            "grasp_search": [], "canary_manifest": result,
            "experimental_identity": (RGBD_EXPERIMENT_SCHEMA if args.region_backend == "rgbd" else SAM_EXPERIMENT_SCHEMA),
        }

    suite_summaries: dict[str, Any] = {}
    try:
        for suite_mode in suites:
            suite_root = root / suite_mode
            suite_summaries[suite_mode] = matrix.run_matrix(
                output_root=suite_root, task_ids=tasks,
                episodes_per_task=int(args.episodes_per_task), seed_base=int(args.seed_base),
                resolution=256, suite_mode=suite_mode,
                controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT, controller_config=controller_config,
                dry_run=False, execute_motion=True, allow_unvalidated_profile=True,
                env_builder=env_builder, episode_runner=episode_runner,
                arrow_input_builder=matrix._default_arrow_inputs,
                continue_on_motion_failure=True,
                resume=bool(args.phase == "full60" and suite_root.exists()),
                resume_terminal=bool(args.phase == "full60" and suite_root.exists()),
                cell_completed_callback=cell_completed_callback,
                experiment_metadata={
                    "schema_version": (RGBD_EXPERIMENT_SCHEMA if args.region_backend == "rgbd" else SAM_EXPERIMENT_SCHEMA),
                    "variant": args.variant,
                    "region_backend": args.region_backend,
                    "backend": "v9d_rgbd_region" if args.region_backend == "rgbd" else "sam3",
                    "sam3_used": bool(args.region_backend == "sam3"),
                    "baseline_commit": BASELINE_COMMIT,
                    "molmopoint_model": args.molmopoint_model,
                    "molmopoint_revision": args.molmopoint_revision,
                    "molmopoint_prompt_id": args.molmopoint_prompt_id,
                    "candidate_policy": VARIANTS[args.variant].policy,
                    "source_camera": VARIANTS[args.variant].camera_name,
                    "motion_profile": resolved_motion_profile["name"],
                    "motion_profile_params": _json_safe(resolved_motion_profile),
                    "motion_diagnostics": bool(args.motion_diagnostics),
                    "experiment_configuration_hash": manifest["experiment_configuration_hash"],
                    "scientific_identity_hash": scientific_identity_hash,
                    "scientific_identity_payload": scientific_identity_payload,
                    "observation_profile": resolved_observation_profile["name"],
                    "observation_profile_params": _json_safe(resolved_observation_profile),
                    **({
                        "sam3_source_commit": args.sam3_source_commit,
                        "sam3_checkpoint_sha256": args.sam3_checkpoint_sha256,
                    } if args.region_backend == "sam3" else {}),
                },
            )
    except Exception as exc:
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["suite_summaries"] = suite_summaries
        _write_json(root / "molmo_sam3_canary_failed.json", manifest)
        print(f"molmo/sam3 canary failed: {exc}", file=sys.stderr)
        return 2
    manifest["suite_summaries"] = suite_summaries
    _write_json(root / "molmo_sam3_canary_summary.json", manifest)
    print(json.dumps({"summary_path": (root / "molmo_sam3_canary_summary.json").as_posix(), "suite_summaries": suite_summaries}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
