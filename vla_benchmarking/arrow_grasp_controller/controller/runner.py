#!/usr/bin/env python3
"""Internal execution engine for the canonical RGB-D/MolmoPoint controller.

The public operator entrypoint fixes the policy and this module retains the
existing episode/matrix motion adapter.  The worker receives only observed
RGB-D captures and calibration, never simulator object state or evaluator
signals.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VLA_ROOT = REPOSITORY_ROOT / "vla_benchmarking"

try:
    from vla_benchmarking.arrow_grasp_controller.configs import ACTIVE_CONTROLLER_CONFIG_FILENAME
except ImportError:  # pragma: no cover - direct script use
    from vla_benchmarking.arrow_grasp_controller.configs import ACTIVE_CONTROLLER_CONFIG_FILENAME
try:
    from .policy import (
        ARROW_LONG_HEAD_LENGTH_PX,
        ARROW_LONG_LINE_WIDTH,
        ARROW_SHORT_HEAD_FRACTION,
        ARROW_SHORT_HEAD_MAX_PX,
        ARROW_SHORT_HEAD_MIN_PX,
        ARROW_SHORT_LINE_WIDTH,
        ARROW_SHORT_SPAN_THRESHOLD_PX,
        EMPTY_GRIPPER_THRESHOLD,
        FINGER_CLEARANCE_M,
        INSERTION_OFFSETS_M,
        LEGACY_DESTINATION_OFFSET_M,
        LEGACY_SOURCE_OFFSET_M,
        LIFT_CLEARANCE_M,
        MAX_APERTURE_M,
        MAX_CANDIDATES,
        MAX_GRASP_ATTEMPTS,
        MAX_SPATIAL_SEEDS,
        MIN_APERTURE_M,
        MOLMOPOINT_MAX_NEW_TOKENS,
        MOLMOPOINT_MAX_POINTS,
        MOLMOPOINT_MODEL_ID,
        MOLMOPOINT_MODEL_REVISION,
        MOLMOPOINT_PADDING_SIDE,
        MOLMOPOINT_PROMPT_ID,
        MODEL_DEVICE,
        MODEL_DEVICE_MAP,
        MODEL_DTYPE,
        OBSERVATION_PROFILE,
        PHASE_TIMEOUT_STEPS,
        PREGRASP_DISTANCE_M,
        PRESHAPE_BAND_M,
        PRESHAPE_MAX_ACTIONS,
        PRESHAPE_SETTLE_STEPS,
        PRESHAPE_STABILITY_M,
        PRESHAPE_TARGET_M,
        RELEASE_HEIGHT_OFFSET_M,
        RETREAT_HEIGHT_OFFSET_M,
        RETREAT_TOLERANCE_M,
        RETRY_RETREAT_DISTANCE_M,
        SHARED_ACTION_BUDGET,
        SOURCE_RELEASE_COMMIT,
        TRANSFORMERS_VERSION,
        WORKSPACE_MAX_M,
        WORKSPACE_MIN_M,
        YAW_OFFSETS_DEG,
        canonical_candidate_policy,
        canonical_region_kwargs,
    )
except ImportError:  # pragma: no cover - direct package use
    from vla_benchmarking.arrow_grasp_controller.controller.policy import (
        ARROW_LONG_HEAD_LENGTH_PX,
        ARROW_LONG_LINE_WIDTH,
        ARROW_SHORT_HEAD_FRACTION,
        ARROW_SHORT_HEAD_MAX_PX,
        ARROW_SHORT_HEAD_MIN_PX,
        ARROW_SHORT_LINE_WIDTH,
        ARROW_SHORT_SPAN_THRESHOLD_PX,
        EMPTY_GRIPPER_THRESHOLD,
        FINGER_CLEARANCE_M,
        INSERTION_OFFSETS_M,
        LEGACY_DESTINATION_OFFSET_M,
        LEGACY_SOURCE_OFFSET_M,
        LIFT_CLEARANCE_M,
        MAX_APERTURE_M,
        MAX_CANDIDATES,
        MAX_GRASP_ATTEMPTS,
        MAX_SPATIAL_SEEDS,
        MIN_APERTURE_M,
        MOLMOPOINT_MAX_NEW_TOKENS,
        MOLMOPOINT_MAX_POINTS,
        MOLMOPOINT_MODEL_ID,
        MOLMOPOINT_MODEL_REVISION,
        MOLMOPOINT_PADDING_SIDE,
        MOLMOPOINT_PROMPT_ID,
        MODEL_DEVICE,
        MODEL_DEVICE_MAP,
        MODEL_DTYPE,
        OBSERVATION_PROFILE,
        PHASE_TIMEOUT_STEPS,
        PREGRASP_DISTANCE_M,
        PRESHAPE_BAND_M,
        PRESHAPE_MAX_ACTIONS,
        PRESHAPE_SETTLE_STEPS,
        PRESHAPE_STABILITY_M,
        PRESHAPE_TARGET_M,
        RELEASE_HEIGHT_OFFSET_M,
        RETREAT_HEIGHT_OFFSET_M,
        RETREAT_TOLERANCE_M,
        RETRY_RETREAT_DISTANCE_M,
        SHARED_ACTION_BUDGET,
        SOURCE_RELEASE_COMMIT,
        TRANSFORMERS_VERSION,
        WORKSPACE_MAX_M,
        WORKSPACE_MIN_M,
        YAW_OFFSETS_DEG,
        canonical_candidate_policy,
        canonical_region_kwargs,
    )

BASELINE_COMMIT = SOURCE_RELEASE_COMMIT
RGBD_EXPERIMENT_SCHEMA = "canonical_rgbd_grasp_controller.v1"
EXPERIMENT_SCHEMA = RGBD_EXPERIMENT_SCHEMA
AGENTVIEW = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"
MAX_ATTEMPTS = MAX_GRASP_ATTEMPTS
GRIPPER_OPEN_TIMEOUT_STEPS = PHASE_TIMEOUT_STEPS
DEFAULT_STALL_DELTA_M = 1e-4
MOLMOPOINT_PROMPT_IDS = (MOLMOPOINT_PROMPT_ID,)

MOTION_PROFILE_NAMES = ("release20_retreat80mm",)
SCIENTIFIC_IDENTITY_SCHEMA = "canonical_grasp_controller_identity.v1"
OBSERVATION_PROFILE_NAMES = (OBSERVATION_PROFILE,)
OBSERVATION_PROFILE_PARAMS = {
    "parked": {
        # Parked skips the observation-hover controller path entirely.  The
        # explicit fields keep the profile identity auditable without
        # fabricating a completed hover audit.
        "skip_hover": True,
        "target_offset_m": None,
        "phase_tolerance_m": None,
        "orientation_tolerance_rad": None,
        "min_actual_height_margin_m": None,
        "require_actual_height_margin": False,
        "arrow_refresh": "current_matrix_inputs_after_opening",
        "arrow_render_policy": "adaptive_short_v1",
        "arrow_short_span_threshold_px": ARROW_SHORT_SPAN_THRESHOLD_PX,
        "arrow_short_line_width": ARROW_SHORT_LINE_WIDTH,
        "arrow_short_head_length_rule": (
            f"max({ARROW_SHORT_HEAD_MIN_PX},min({ARROW_SHORT_HEAD_MAX_PX},"
            f"round({ARROW_SHORT_HEAD_FRACTION}*span_px)))"
        ),
        "arrow_long_line_width": ARROW_LONG_LINE_WIDTH,
        "arrow_long_head_length": ARROW_LONG_HEAD_LENGTH_PX,
    },
}

OPENING_PROFILE_NAMES = ("preshape40mm",)
PRESHAPE40MM_PARAMS = {
    "target_opening_m": PRESHAPE_TARGET_M,
    "accepted_opening_band_m": PRESHAPE_BAND_M,
    "settled_range_m": PRESHAPE_STABILITY_M,
    "settle_steps": PRESHAPE_SETTLE_STEPS,
    "close_pulse": 1.0,
    "max_actions": PRESHAPE_MAX_ACTIONS,
    "shared_action_budget": SHARED_ACTION_BUDGET,
}
def resolve_opening_profile(
    name: str,
    *,
    region_backend: str,
    camera_name: str,
) -> dict[str, Any]:
    """Return the immutable 40 mm preshape contract."""
    if name not in OPENING_PROFILE_NAMES:
        raise ValueError(f"opening profile must be one of {OPENING_PROFILE_NAMES}")
    if region_backend != "rgbd" or camera_name != AGENTVIEW:
        raise ValueError("preshape40mm requires the RGB-D agentview path")
    return {"name": name, **dict(PRESHAPE40MM_PARAMS)}


def resolve_motion_profile(name: str, *, region_backend: str) -> dict[str, Any]:
    """Return the immutable +20 mm release / +80 mm retreat contract."""
    if name not in MOTION_PROFILE_NAMES:
        raise ValueError(f"motion profile must be one of {MOTION_PROFILE_NAMES}")
    if region_backend != "rgbd":
        raise ValueError("canonical motion requires the RGB-D region backend")
    return {
        "name": name,
        "region_backend": region_backend,
        "micro_correction": None,
        "release_height_offset_m": RELEASE_HEIGHT_OFFSET_M,
        "transfer_xy_policy": "legacy_displacement",
        "retreat_height_offset_m": RETREAT_HEIGHT_OFFSET_M,
        "retreat_tolerance_m": RETREAT_TOLERANCE_M,
        "retreat_reference": "open_after_release_plus20mm",
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
    if resolved_motion_profile.get("name") in {"release20_retreat80mm", "release_plus40mm"}:
        kwargs["experimental_motion_profile"] = str(resolved_motion_profile["name"])
    retreat_offset = float(resolved_motion_profile.get("retreat_height_offset_m", 0.0))
    if retreat_offset != 0.0:
        kwargs["experimental_retreat_height_offset_m"] = retreat_offset
    if resolved_motion_profile.get("transfer_xy_policy") == "visual_endpoints":
        kwargs["experimental_transfer_xy_policy"] = "visual_endpoints"
    return kwargs


def resolve_observation_profile(name: str) -> dict[str, Any]:
    """Resolve the bounded observation-hover policy independently of grasp motion."""
    if name not in OBSERVATION_PROFILE_NAMES:
        raise ValueError(f"observation profile must be one of {OBSERVATION_PROFILE_NAMES}")
    return {"name": name, **dict(OBSERVATION_PROFILE_PARAMS[name])}


def _parked_arrow_render_params(
    episode_module: Any,
    bboxes: Mapping[str, Sequence[float]],
    *,
    subject: str,
    goal_object: str,
    image_shape: Sequence[int],
) -> dict[str, Any]:
    """Resolve the parked short-arrow rendering from renderer anchor semantics."""
    anchor_fn = getattr(episode_module, "_arrow_anchor_bboxes", None)
    if not callable(anchor_fn):
        raise RuntimeError("parked short-arrow policy requires the canonical arrow anchor helper")
    anchored = anchor_fn(
        bboxes, subject=subject, image_shape=image_shape, policy="bbox_center"
    )
    if subject not in anchored or goal_object not in anchored:
        raise ValueError("parked short-arrow policy requires subject and goal bboxes")
    # Import lazily so dependency-light config/preflight commands do not load
    # OpenCV. Every policy arrow still uses this one canonical center helper.
    from vla_benchmarking.evaluation.visual_scene_graph import bbox_center

    source_center = bbox_center(anchored[subject])
    target_center = bbox_center(anchored[goal_object])
    span = float(np.linalg.norm(np.asarray(target_center, dtype=np.float64) - np.asarray(source_center, dtype=np.float64)))
    if not math.isfinite(span):
        raise ValueError("parked arrow endpoint span is non-finite")
    if span < ARROW_SHORT_SPAN_THRESHOLD_PX:
        return {
            "line_width": ARROW_SHORT_LINE_WIDTH,
            "head_length": max(
                ARROW_SHORT_HEAD_MIN_PX,
                min(ARROW_SHORT_HEAD_MAX_PX, round(ARROW_SHORT_HEAD_FRACTION * span)),
            ),
            "endpoint_span_px": span,
            "rounded_source_center": list(source_center),
            "rounded_target_center": list(target_center),
            "render_policy": "adaptive_short_v1",
        }
    return {
        "line_width": ARROW_LONG_LINE_WIDTH,
        "head_length": ARROW_LONG_HEAD_LENGTH_PX,
        "endpoint_span_px": span,
        "rounded_source_center": list(source_center),
        "rounded_target_center": list(target_center),
        "render_policy": "adaptive_short_v1_long_default",
    }


def _stable_model_identity(model_provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep model identity fields while excluding paths, timings, and load state."""
    source = model_provenance if isinstance(model_provenance, Mapping) else {}
    keys = (
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
    opening_profile: str = "preshape40mm",
    opening_profile_params: Mapping[str, Any] | None = None,
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
        "opening": {
            "profile": str(opening_profile),
            "params": _json_safe(opening_profile_params or {}),
        },
        "task_seed": {"task_ids": [int(value) for value in task_ids], "seed_base": int(seed_base)},
        "suite_modes": [str(value) for value in suite_modes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def _execution_provenance(*, repo_root: Path | None = None, require_clean: bool) -> dict[str, Any]:
    """Read the actual worktree SHA and enforce cleanliness for live execution."""
    root = (repo_root or REPOSITORY_ROOT).expanduser().resolve()
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
    """Immutable internal description of the canonical policy."""

    name: str
    camera_name: str
    policy: str
    uses_molmo: bool
    region_backend: str = "rgbd"

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("variant name must be a safe non-empty label")
        if self.camera_name != AGENTVIEW:
            raise ValueError("the canonical policy uses agentview")
        if self.policy != "molmo_dense" or not self.uses_molmo:
            raise ValueError("the canonical policy uses Molmo-dense candidates")
        if self.region_backend != "rgbd":
            raise ValueError("the canonical controller uses the RGB-D region backend")


VARIANTS: dict[str, CanaryVariant] = {
    "canonical": CanaryVariant("canonical", AGENTVIEW, "molmo_dense", True, "rgbd"),
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
    # for lightweight test fixtures; live RGB-D/Molmo candidates populate it.
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
    # ``grasp_controller.grasp_candidates.GraspCandidate`` is intentionally a richer
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
            opening_m=float(getattr(value, "required_aperture_m", PRESHAPE_TARGET_M)),
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
        opening_m=float(value.get("opening_m", value.get("jaw_width_m", PRESHAPE_TARGET_M))),
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
    for seed in seeds[:MAX_SPATIAL_SEEDS]:
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
            from vla_benchmarking.arrow_grasp_controller.configs import load_controller_config
        except ImportError:  # pragma: no cover - direct script use
            from vla_benchmarking.arrow_grasp_controller.configs import load_controller_config
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
    experiment_schema: str = RGBD_EXPERIMENT_SCHEMA,
    region_backend: str = "rgbd",
    before_propose_callback: Callable[[Any], Any] | None = None,
    motion_profile: str = "release20_retreat80mm",
    motion_profile_params: Mapping[str, Any] | None = None,
    motion_diagnostics: bool = False,
    observation_profile: str = "parked",
    observation_profile_params: Mapping[str, Any] | None = None,
    opening_profile: str = "preshape40mm",
    opening_profile_params: Mapping[str, Any] | None = None,
    before_capture_fn: Callable[[Any, int], Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded canary episode, regenerating candidates after failure."""
    selected_variant = VARIANTS[variant] if isinstance(variant, str) else variant
    if selected_variant.name not in VARIANTS:
        raise ValueError(f"unknown canary variant {selected_variant.name!r}")
    if capture_fn is None:
        from vla_benchmarking.evaluation.run_arrow_pick_place_eval import capture_agentview
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
            if before_capture_fn is not None:
                refreshed_capture = before_capture_fn(env, attempt_index)
                if refreshed_capture is None:
                    raise RuntimeError("before_capture_fn must return a fresh agentview capture")
                av_capture = refreshed_capture
            else:
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
        "frozen_controller": "canonical_molmo_rgbd_grasp",
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
        "backend": "rgbd_region",
        "sam3_used": False,
        "motion_profile": motion_profile,
        "motion_profile_params": _json_safe(motion_profile_params or {}),
        "motion_diagnostics": bool(motion_diagnostics),
        "observation_profile": observation_profile,
        "observation_profile_params": _json_safe(observation_profile_params or {}),
        "opening_profile": str(opening_profile),
        "opening_profile_params": _json_safe(opening_profile_params or {}),
    }
    digest = hashlib.sha256(json.dumps(_json_safe(manifest), sort_keys=True).encode()).hexdigest()
    manifest["experiment_config_hash"] = digest
    _write_json(root / "canonical_grasp_manifest.json", manifest)
    return manifest


def build_local_molmo_runtime(
    *,
    molmopoint_model: str = MOLMOPOINT_MODEL_ID,
    molmopoint_revision: str = MOLMOPOINT_MODEL_REVISION,
    molmopoint_prompt_id: str = MOLMOPOINT_PROMPT_ID,
    device: str = MODEL_DEVICE,
) -> Any:
    """Construct the persistent Molmo runtime without touching SAM assets."""
    if str(molmopoint_model) != MOLMOPOINT_MODEL_ID:
        raise ValueError("MolmoPoint model must be the pinned canary model")
    if str(molmopoint_revision).lower() != MOLMOPOINT_MODEL_REVISION:
        raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
    try:
        from .molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    except ImportError:  # pragma: no cover - direct script use
        from vla_benchmarking.arrow_grasp_controller.controller.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig, PROMPT_VARIANTS
    if str(molmopoint_prompt_id) not in PROMPT_VARIANTS:
        raise ValueError(f"unknown MolmoPoint prompt id: {molmopoint_prompt_id}")
    return MolmoPointRuntime(MolmoPointRuntimeConfig(
        model_id=str(molmopoint_model), model_revision=str(molmopoint_revision),
        transformers_version=TRANSFORMERS_VERSION, dtype=MODEL_DTYPE,
        device=device, device_map=MODEL_DEVICE_MAP,
        max_new_tokens=MOLMOPOINT_MAX_NEW_TOKENS,
        max_points=MOLMOPOINT_MAX_POINTS,
        padding_side=MOLMOPOINT_PADDING_SIDE,
        prompt_id=str(molmopoint_prompt_id), prompt=PROMPT_VARIANTS[str(molmopoint_prompt_id)],
    ))


def preflight_local_molmo_runtime(molmo: Any, *, load_models: bool = False) -> dict[str, Any]:
    """Validate/load the exact canonical Molmo runtime and configuration."""
    try:
        from .molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig
    except ImportError:  # pragma: no cover - direct script use
        from vla_benchmarking.arrow_grasp_controller.controller.molmopoint import MolmoPointRuntime, MolmoPointRuntimeConfig
    if type(molmo) is not MolmoPointRuntime:
        raise TypeError("canonical execution requires the project MolmoPointRuntime")
    config = getattr(molmo, "config", None)
    expected = MolmoPointRuntimeConfig()
    if config != expected:
        raise ValueError("MolmoPoint runtime configuration is not canonical")
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
        from .grasp_candidates import CameraCalibration
    except ImportError:  # pragma: no cover - direct script use
        from vla_benchmarking.arrow_grasp_controller.controller.grasp_candidates import CameraCalibration
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


class ModelPerceptionWorker:
    """Persistent MolmoPoint worker with deterministic RGB-D geometry."""

    def __init__(self, molmo: Any, robot_calibration: Any) -> None:
        self.molmo = molmo
        self.robot_calibration = robot_calibration

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
        """Generate candidates from the canonical arrow-seeded RGB-D frame."""
        return self._propose_rgbd(request)

    def _propose_rgbd(self, request: PerceptionRequest) -> Any:
        """Generate candidates from arrow-seeded RGB-D support only."""
        try:
            from .grasp_candidates import MolmoPoint, generate_grasp_candidates
            from vla_benchmarking.arrow_grasp_controller.legacy_engine.rgbd_region import derive_observed_region_mask
            from .molmopoint import MolmoPointRequest
        except ImportError:  # pragma: no cover - direct script use
            from vla_benchmarking.arrow_grasp_controller.controller.grasp_candidates import MolmoPoint, generate_grasp_candidates
            from vla_benchmarking.arrow_grasp_controller.legacy_engine.rgbd_region import derive_observed_region_mask
            from vla_benchmarking.arrow_grasp_controller.controller.molmopoint import MolmoPointRequest
        started = time.perf_counter()
        # The destination arrow endpoint is an observed profile hint; when it
        # is unavailable, the source point remains a valid deterministic axis.
        profile_world = _deproject_capture(request.agentview_capture, request.destination_uv or request.source_uv)
        region_mask, region_audit = derive_observed_region_mask(
            request.agentview_capture, request.source_uv, profile_target_world=profile_world,
            **canonical_region_kwargs(),
        )
        source_capture = request.source_capture
        source_uv = tuple(float(v) for v in request.source_uv)
        molmo_points: list[Any] = []
        molmo_provenance: Mapping[str, Any] | None = None
        if request.variant.uses_molmo:
            molmo_result = self.molmo.predict(MolmoPointRequest(
                np.asarray(source_capture.rgb, dtype=np.uint8), mask=np.asarray(region_mask, dtype=np.bool_)
            ))
            molmo_points = [MolmoPoint(float(point.x), float(point.y), label="molmo_rim") for point in molmo_result.points]
            molmo_provenance = dict(molmo_result.provenance)
        policy = canonical_candidate_policy()
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
            "backend": "rgbd_region", "region_backend": "rgbd", "sam3_used": False,
            "source_camera": request.variant.camera_name, "source_uv": list(source_uv),
            "molmopoint_count": len(molmo_points), "molmopoint_provenance": molmo_provenance,
            "returned_candidate_count": len(result.candidates), "rejection_count": len(result.rejected),
            "latency_s": float(time.perf_counter() - started),
            "region_audit": region_audit,
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
        from vla_benchmarking.tools.sanity_checks.probe_panda_grip_site_frame import probe_grip_site_frame
        from .grasp_candidates import RobotGraspCalibration
    except ImportError:  # pragma: no cover - direct script use
        from vla_benchmarking.tools.sanity_checks.probe_panda_grip_site_frame import probe_grip_site_frame
        from vla_benchmarking.arrow_grasp_controller.controller.grasp_candidates import RobotGraspCalibration
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
         min_aperture_m=MIN_APERTURE_M,
         max_aperture_m=min(MAX_APERTURE_M, max(MIN_APERTURE_M, usable_aperture * 0.98)),
         finger_clearance_m=FINGER_CLEARANCE_M,
         pregrasp_distance_m=PREGRASP_DISTANCE_M,
         workspace_min_m=WORKSPACE_MIN_M,
         workspace_max_m=WORKSPACE_MAX_M,
         calibration_source="panda_grip_site_frame_and_contact_pad_probe_no_motion",
         calibration_sha256=str(record.get("calibration_sha256") or ""),
         current_grip_site_world_m=site,
         current_rotation_world_grip_site=site_rotation,
         hand_collision_spheres_grasp=collision_spheres,
         hand_collision_boxes_grasp=collision_boxes,
    )
    setattr(env, "_grasp_controller_robot_calibration_probe", record)
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
        from vla_benchmarking.evaluation import run_arrow_pick_place_eval as episode
        observation = episode._raw_observation(env)
        proprio = episode._proprioception(observation)
        current = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise RuntimeError("failed-candidate recovery lacks finite EEF position")
        retreat = current + np.asarray((0.0, 0.0, RETRY_RETREAT_DISTANCE_M), dtype=np.float64)
        waypoints = np.vstack((current, current, current, current, current, retreat))
        prior_actions = int(getattr(env, "_grasp_controller_action_count", 0))
        budget = getattr(env, "_grasp_controller_action_budget", None)
        if budget is not None:
            remaining = int(budget.remaining)
        else:
            remaining = max(0, SHARED_ACTION_BUDGET - prior_actions)
        if remaining <= 0:
            raise RuntimeError("candidate retry action budget exhausted before recovery")
        budget = getattr(env, "_grasp_controller_action_budget", None)
        hover_budget = (
            min(PHASE_TIMEOUT_STEPS, int(budget.remaining))
            if budget is not None
            else PHASE_TIMEOUT_STEPS
        )
        if hover_budget <= 0:
            raise RuntimeError("shared action budget exhausted before observation hover")
        motion_diag_kwargs = {"experimental_motion_diagnostics": True} if (motion_settings or {}).get("motion_diagnostics") else {}
        phases = episode._run_motion(
            env, waypoints, observation,
            phase_timeout_steps=int((motion_settings or {}).get("phase_timeout_steps", PHASE_TIMEOUT_STEPS)),
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
        if total_actions > SHARED_ACTION_BUDGET:
            raise RuntimeError("candidate retry exceeded the shared action budget")
        setattr(env, "_grasp_controller_action_count", total_actions)
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
    threshold = EMPTY_GRIPPER_THRESHOLD
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
        from vla_benchmarking.evaluation import run_arrow_pick_place_eval as episode
        observation = episode._raw_observation(env)
        proprio = episode._proprioception(observation)
        current = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        if current.shape != (3,) or not np.all(np.isfinite(current)):
            raise RuntimeError("gripper-open preflight lacks finite EEF position")
        budget = getattr(env, "_grasp_controller_action_budget", None)
        prior_actions = int(getattr(env, "_grasp_controller_action_count", 0))
        remaining = (
            int(budget.remaining)
            if budget is not None
            else max(0, SHARED_ACTION_BUDGET - prior_actions)
        )
        if remaining <= 0:
            raise RuntimeError("shared action budget exhausted before gripper open")
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
        if total_actions > SHARED_ACTION_BUDGET:
            raise RuntimeError("gripper-open preflight exceeded the shared action budget")
        setattr(env, "_grasp_controller_action_count", total_actions)
        audit = {
            "status": "completed", "eef_world_m": current.tolist(),
            "steps": steps, "total_actions": total_actions,
            "phase_statuses": [{"phase": item.get("phase"), "status": item.get("status")} for item in phases],
            "stop_after_phase": "open", "gripper_command": -1.0,
        }
        setattr(env, "_grasp_controller_gripper_open", audit)
        return audit
    except Exception as exc:
        audit = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        setattr(env, "_grasp_controller_gripper_open", audit)
        raise


def main(
    argv: Sequence[str] | None = None,
    *,
    molmo_runtime: Any | None = None,
    cell_completed_callback: Callable[[Mapping[str, Any]], Any] | None = None,
    execution_cell_identities_by_suite: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="canonical_grasp_controller")
    parser.add_argument("--phase", choices=("prefix", "full60"), default="prefix")
    parser.add_argument(
        "--sealed-100", action="store_true",
        help="run a complete 100-cell sealed-randomized evaluation",
    )
    parser.add_argument(
        "--sealed-100-profile", choices=("canonical_molmo_rgbd_grasp",), default="canonical_molmo_rgbd_grasp",
        help="sealed-100 policy identity (immutable)",
    )
    parser.add_argument("--task-ids", default="4,6,9")
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--suite-modes", default="vanilla,sealed_randomized")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--region-backend", choices=("rgbd",), default="rgbd")
    parser.add_argument("--motion-profile", choices=MOTION_PROFILE_NAMES, default="release20_retreat80mm")
    parser.add_argument("--motion-diagnostics", action="store_true", help="record bounded motion diagnostics")
    parser.add_argument("--observation-profile", choices=OBSERVATION_PROFILE_NAMES, default="parked")
    parser.add_argument("--opening-profile", choices=OPENING_PROFILE_NAMES, default="preshape40mm")
    parser.add_argument("--molmopoint-model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--molmopoint-revision", default=MOLMOPOINT_MODEL_REVISION)
    parser.add_argument("--molmopoint-prompt-id", choices=MOLMOPOINT_PROMPT_IDS, default="rim_clearance")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate local model assets without motion")
    args = parser.parse_args(argv)
    if args.episodes_per_task is None:
        # A failure-stratified prefix selects arbitrary seeds from the full
        # canonical 10-episode shard; materialize all 30 cells and filter only
        # execution.  Ordinary prefixes retain the historical 2-episode plan.
        args.episodes_per_task = (
            10
            if args.phase == "prefix" and execution_cell_identities_by_suite is not None
            else 2 if args.phase == "prefix" else 10
        )
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    try:
        resolved_motion_profile = resolve_motion_profile(args.motion_profile, region_backend=args.region_backend)
        resolved_observation_profile = resolve_observation_profile(args.observation_profile)
        resolved_opening_profile = resolve_opening_profile(
            args.opening_profile, region_backend=args.region_backend,
            camera_name=VARIANTS[args.variant].camera_name,
        )
        if args.observation_profile == "parked":
            if args.region_backend != "rgbd" or VARIANTS[args.variant].camera_name != AGENTVIEW:
                raise ValueError("parked observation profile requires the RGB-D agentview path")
        if VARIANTS[args.variant].region_backend != args.region_backend:
            raise ValueError("canonical policy requires --region-backend rgbd")
        tasks = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
        suites = tuple(item.strip() for item in str(args.suite_modes).split(",") if item.strip())
        if args.sealed_100:
            if args.phase != "full60":
                raise ValueError("--sealed-100 requires --phase full60")
            if args.region_backend != "rgbd" or args.variant != "canonical":
                raise ValueError("--sealed-100 requires the canonical RGB-D agentview policy")
            if tasks != tuple(range(10)):
                raise ValueError("--sealed-100 task IDs must be exactly 0 through 9")
            if suites != ("sealed_randomized",):
                raise ValueError("--sealed-100 requires suite mode sealed_randomized only")
            if args.episodes_per_task != 10:
                raise ValueError("--sealed-100 requires exactly 10 episodes per task")
            expected_profile = ("release20_retreat80mm", "parked", "preshape40mm")
            if (args.motion_profile, args.observation_profile, args.opening_profile) != expected_profile:
                raise ValueError(
                    f"--sealed-100 profile {args.sealed_100_profile} requires "
                    f"motion={expected_profile[0]}, observation={expected_profile[1]}, opening={expected_profile[2]}"
                )
            if args.molmopoint_prompt_id != "rim_clearance":
                raise ValueError("--sealed-100 requires the pinned rim_clearance prompt")
        else:
            if not tasks or len(set(tasks)) != len(tasks) or any(task < 0 or task > 9 for task in tasks):
                raise ValueError("task IDs must be unique values from 0 through 9")
            if not suites or len(set(suites)) != len(suites) or any(
                suite not in {"vanilla", "sealed_randomized"} for suite in suites
            ):
                raise ValueError("suite modes must be unique vanilla and/or sealed_randomized values")
        if args.seed_base < 0:
            raise ValueError("seed base must be non-negative")
        if args.molmopoint_model != MOLMOPOINT_MODEL_ID:
            raise ValueError("MolmoPoint model must be the pinned canary model")
        if args.molmopoint_revision != MOLMOPOINT_MODEL_REVISION:
            raise ValueError("MolmoPoint revision does not match the pinned canary artifact")
        if args.config is not None and not args.config.is_file():
            raise ValueError(f"canonical config is unavailable: {args.config}")
        if molmo_runtime is None:
            molmo_runtime = build_local_molmo_runtime(
                molmopoint_model=args.molmopoint_model,
                molmopoint_revision=args.molmopoint_revision,
                molmopoint_prompt_id=args.molmopoint_prompt_id,
            )
        try:
            try:
                from .molmopoint import PROMPT_VARIANTS
            except ImportError:  # pragma: no cover - direct script use
                from vla_benchmarking.arrow_grasp_controller.controller.molmopoint import PROMPT_VARIANTS
            config = getattr(molmo_runtime, "config", None)
            if config is None:
                raise RuntimeError("injected Molmo runtime is missing config")
            molmo_runtime.config = replace(
                config, prompt_id=args.molmopoint_prompt_id,
                prompt=PROMPT_VARIANTS[args.molmopoint_prompt_id],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError("injected Molmo runtime cannot update prompt config") from exc
        model_provenance = preflight_local_molmo_runtime(
            molmo_runtime, load_models=not args.dry_run
        )
        molmo = molmo_runtime
    except Exception as exc:
        print(f"canonical grasp preflight failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    try:
        controller_config = args.config or (VLA_ROOT / "arrow_grasp_controller" / "configs" / ACTIVE_CONTROLLER_CONFIG_FILENAME)
        if not controller_config.is_file():
            raise RuntimeError(f"canonical controller config is unavailable: {controller_config}")
        controller_config_digest = _resolved_controller_config_digest(controller_config)
        execution_provenance = _execution_provenance(require_clean=not args.dry_run)
    except Exception as exc:
        print(f"canonical grasp execution provenance failed: {exc}", file=sys.stderr)
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
        backend="rgbd_region",
        motion_profile=resolved_motion_profile["name"],
        motion_profile_params=resolved_motion_profile,
        motion_diagnostics=bool(args.motion_diagnostics),
        observation_profile=resolved_observation_profile["name"],
        observation_profile_params=resolved_observation_profile,
        opening_profile=resolved_opening_profile["name"],
        opening_profile_params=resolved_opening_profile,
        task_ids=tasks,
        seed_base=args.seed_base,
        suite_modes=suites,
    )
    manifest = {
        "schema_version": RGBD_EXPERIMENT_SCHEMA,
        "experiment_id": str(args.label), "baseline_commit": BASELINE_COMMIT,
        "variant": asdict(replace(VARIANTS[args.variant], region_backend=args.region_backend)), "phase": args.phase,
        "task_ids": list(tasks), "episodes_per_task": int(args.episodes_per_task),
        "seed_base": int(args.seed_base), "suite_modes": list(suites),
        "output_dir": args.output_dir.expanduser().resolve().as_posix(),
        "model_provenance": model_provenance,
        "region_backend": args.region_backend,
        "backend": "rgbd_region",
        "sam3_used": False,
        "molmopoint_prompt_id": args.molmopoint_prompt_id,
        "evaluation_scope": "sealed_100_cells" if args.sealed_100 else "canary",
        "sealed_100_profile": args.sealed_100_profile if args.sealed_100 else None,
        "dry_run": bool(args.dry_run),
        "motion_profile": resolved_motion_profile["name"],
        "motion_profile_params": _json_safe(resolved_motion_profile),
        "motion_diagnostics": bool(args.motion_diagnostics),
        "observation_profile": resolved_observation_profile["name"],
        "observation_profile_params": _json_safe(resolved_observation_profile),
        "opening_profile": resolved_opening_profile["name"],
        "opening_profile_params": _json_safe(resolved_opening_profile),
        "execution_provenance": execution_provenance,
        "scientific_identity_payload": scientific_identity_payload,
        "scientific_identity_hash": scientific_identity_hash,
    }
    manifest["experiment_configuration_hash"] = scientific_identity_hash
    if args.dry_run:
        _write_json(args.output_dir.expanduser().resolve() / "canonical_grasp_preflight.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0

    # Live canary path.  The matrix remains the existing evaluator/motion
    # implementation; this wrapper injects only the project-local perception
    # worker, candidate pose, and fresh-frame retry policy.
    try:
        try:
            from vla_benchmarking.evaluation import run_arrow_pick_place_eval as episode
            from vla_benchmarking.evaluation import run_arrow_pick_place_matrix as matrix
        except ImportError:  # pragma: no cover - direct script use
            from vla_benchmarking.evaluation import run_arrow_pick_place_eval as episode
            from vla_benchmarking.evaluation import run_arrow_pick_place_matrix as matrix
    except ImportError as exc:  # pragma: no cover - dependency-free host
        print(f"live canary imports unavailable: {exc}", file=sys.stderr)
        return 2

    experimental_micro_correction = None
    if resolved_motion_profile["micro_correction"] is not None:
        experimental_micro_correction = episode.MicroCorrectionPolicy(
            **dict(resolved_motion_profile["micro_correction"])
        )

    controller_config = args.config or (VLA_ROOT / "arrow_grasp_controller" / "configs" / ACTIVE_CONTROLLER_CONFIG_FILENAME)
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
            worker_state["worker"] = ModelPerceptionWorker(molmo, calibration)
        else:
            worker_state["worker"].robot_calibration = calibration
        setattr(env, "_grasp_controller_model_provenance", model_provenance)
        setattr(env, "_grasp_controller_probe_sha256", hashlib.sha256(json.dumps(_json_safe(probe), sort_keys=True).encode()).hexdigest())
        # One shared budget covers the observation hover, every candidate
        # motion, and failed-candidate recovery.  It is initialized before the
        # first capture because the hover itself sends actions.
        try:
            from vla_benchmarking.evaluation import run_arrow_pick_place_eval as _budget_module
        except ImportError:  # pragma: no cover - direct script use
            from vla_benchmarking.evaluation import run_arrow_pick_place_eval as _budget_module
        setattr(
            env,
            "_grasp_controller_action_budget",
            _budget_module._ActionBudget(SHARED_ACTION_BUDGET),
        )
        setattr(env, "_grasp_controller_action_count", 0)
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
        motion_settings["opening_profile"] = resolved_opening_profile
        bboxes = kwargs.get("bboxes")
        if not isinstance(bboxes, Mapping):
            raise ValueError("canary episode requires existing arrow input-generation bboxes")
        # The promoted treatment keeps the post-open parked pose; there is no
        # selectable observation-hover mode in the canonical entrypoint.
        parked_observation = True
        initial = episode.capture_agentview(env, resolution=resolution, camera_name=AGENTVIEW)
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
        parked_audit = {
            "status": "skipped_by_parked_profile",
            "observation_profile": resolved_observation_profile["name"],
            "reason": "canonical policy keeps the robot at its post-open parked pose",
        }
        setattr(env, "_grasp_controller_observation_hover", parked_audit)

        opening_audit: Mapping[str, Any] | None = None

        def measure_opening(measure_env: Any = env) -> float:
            # The shaping helper measures the real contact-pad geometry before
            # and after its bounded pulses.  Do not derive this from commands
            # or from the candidate's requested aperture.
            measured_calibration, measured_transform, measured_probe = probe_robot_calibration(measure_env)
            worker_state["robot_calibration"] = measured_calibration
            worker_state["transform"] = measured_transform
            geometry = measured_probe.get("gripper_geometry") if isinstance(measured_probe, Mapping) else None
            if not isinstance(geometry, Mapping) or geometry.get("measured_opening_m") is None:
                raise RuntimeError("preshape opening measurement is unavailable")
            opening = float(geometry["measured_opening_m"])
            if not math.isfinite(opening):
                raise RuntimeError("preshape opening measurement is non-finite")
            worker_state["opening_m"] = opening
            if worker_state.get("worker") is not None:
                worker_state["worker"].robot_calibration = measured_calibration
            setattr(measure_env, "_grasp_controller_robot_calibration_probe", measured_probe)
            return opening

        def run_preshape(target_env: Any, target_output: Path) -> Mapping[str, Any]:
            try:
                from .preshape import perform_preshape
            except ImportError:  # pragma: no cover - direct script use
                from vla_benchmarking.arrow_grasp_controller.controller.preshape import perform_preshape
            try:
                audit = perform_preshape(
                    target_env, measure_opening_fn=measure_opening,
                    output_dir=target_output,
                    motion_started_callback=motion_callback,
                    motion_settings=motion_settings,
                )
            except Exception as exc:
                # Keep retry/stop accounting stable across helper versions:
                # backend failures are categorized by the helper, not by the
                # incidental numeric details in their exception text.
                audit = getattr(exc, "audit", None)
                if not isinstance(audit, Mapping):
                    raise RuntimeError("preshape_failed:motion_failed") from exc
                setattr(target_env, "_grasp_controller_opening_preshape", dict(audit))
                category = str(getattr(exc, "category", audit.get("failure_reason", "motion_failed")))
                raise RuntimeError(f"preshape_failed:{category}") from exc
            if not isinstance(audit, Mapping):
                raise RuntimeError("preshape helper returned an invalid audit")
            if str(audit.get("status", "")).lower() != "completed":
                category = str(audit.get("failure_reason", "motion_failed"))
                setattr(target_env, "_grasp_controller_opening_preshape", dict(audit))
                raise RuntimeError(f"preshape_failed:{category}")
            setattr(target_env, "_grasp_controller_opening_preshape", dict(audit))
            return audit

        opening_audit = run_preshape(env, output_dir / "opening_preshape")
        fresh = episode.capture_agentview(env, resolution=resolution, camera_name=AGENTVIEW)
        hover_audit = getattr(env, "_grasp_controller_observation_hover", None)
        if isinstance(hover_audit, dict):
            hover_audit["fresh_capture_provenance"] = _observation_capture_provenance(fresh)
        arrow_state: dict[str, Any] = {"rgb": None, "source_uv": None, "destination_uv": None}
        arrow_input_state: dict[str, Any] = {
            "bboxes": bboxes, "subject": kwargs.get("subject", "bowl"),
            "goal_object": kwargs.get("goal_object", "plate"),
        }
        first_capture: dict[str, Any] = {AGENTVIEW: fresh}
        arrow_refresh_index = 0
        arrow_refresh_attempt = 1

        def refresh_current_arrow_inputs(target_env: Any) -> None:
            """Refresh privileged matrix arrow inputs at the current robot pose."""
            try:
                try:
                        from vla_benchmarking.evaluation import run_arrow_pick_place_matrix as current_matrix
                except ImportError:  # pragma: no cover - direct script use
                    from vla_benchmarking.evaluation import run_arrow_pick_place_matrix as current_matrix
                current_inputs = current_matrix._default_arrow_inputs(target_env, task_id, resolution)
            except Exception as exc:
                raise RuntimeError("parked opening arrow refresh failed") from exc
            if not isinstance(current_inputs, Mapping) or not isinstance(current_inputs.get("bboxes"), Mapping):
                raise RuntimeError("parked opening arrow inputs are invalid")
            arrow_input_state.update({
                "bboxes": current_inputs["bboxes"],
                "subject": current_inputs.get("subject", "bowl"),
                "goal_object": current_inputs.get("goal_object", "plate"),
            })

        def refresh_arrow(capture: Any) -> tuple[np.ndarray, Sequence[float], Sequence[float] | None]:
            nonlocal arrow_refresh_index
            arrow_refresh_index += 1
            arrow_state.update({"rgb": None, "source_uv": None, "destination_uv": None})
            render_params: dict[str, Any] = {}
            refresh_audit: dict[str, Any] = {
                "refresh_index": arrow_refresh_index,
                "attempt_index": arrow_refresh_attempt,
            }
            refresh_dir = output_dir / "arrow_refreshes"
            render_params = _parked_arrow_render_params(
                episode, arrow_input_state["bboxes"],
                subject=arrow_input_state["subject"], goal_object=arrow_input_state["goal_object"],
                image_shape=np.asarray(capture.rgb).shape[:2],
            )
            refresh_audit.update(render_params)
            clean_rgb = np.asarray(capture.rgb)
            refresh_audit["capture_provenance"] = _observation_capture_provenance(capture)
            refresh_audit["clean_rgb_sha256"] = hashlib.sha256(clean_rgb.tobytes()).hexdigest()
            try:
                from PIL import Image
                refresh_dir.mkdir(parents=True, exist_ok=True)
                clean_path = refresh_dir / f"refresh_{arrow_refresh_index:02d}_clean.png"
                Image.fromarray(np.asarray(capture.rgb, dtype=np.uint8), mode="RGB").save(clean_path)
                refresh_audit["clean_rgb_path"] = clean_path.as_posix()
            except Exception as exc:  # evidence persistence cannot alter rendering semantics
                refresh_audit["clean_rgb_persist_error"] = type(exc).__name__
            try:
                rendered, renderer_audit = episode.render_exactly_one_arrow(
                    capture.rgb, arrow_input_state["bboxes"], subject=arrow_input_state["subject"],
                    goal_object=arrow_input_state["goal_object"], anchor_policy="bbox_center",
                    line_width=int(render_params["line_width"]), head_length=int(render_params["head_length"]),
                )
            except Exception as exc:
                refresh_audit.update({"render_status": "failed", "render_error_type": type(exc).__name__, "render_error": str(exc)})
                try:
                    _write_json(refresh_dir / f"refresh_{arrow_refresh_index:02d}.json", refresh_audit)
                except Exception:
                    pass
                raise
            refresh_audit.update({"render_status": "completed", "renderer_audit": renderer_audit})
            refresh_audit["rendered_rgb_sha256"] = hashlib.sha256(np.asarray(rendered).tobytes()).hexdigest()
            try:
                from PIL import Image
                refresh_dir.mkdir(parents=True, exist_ok=True)
                rendered_path = refresh_dir / f"refresh_{arrow_refresh_index:02d}_arrow.png"
                Image.fromarray(np.asarray(rendered, dtype=np.uint8), mode="RGB").save(rendered_path)
                refresh_audit["rendered_rgb_path"] = rendered_path.as_posix()
            except Exception as exc:
                refresh_audit["rendered_rgb_persist_error"] = type(exc).__name__
            try:
                source_point, target_point = episode.decode_arrow_pixels(capture.rgb, rendered)
            except Exception as exc:
                refresh_audit.update({"decode_status": "failed", "decode_error_type": type(exc).__name__, "decode_error": str(exc)})
                arrow_state["render_audit"] = refresh_audit
                try:
                    _write_json(refresh_dir / f"refresh_{arrow_refresh_index:02d}.json", refresh_audit)
                except Exception:
                    pass
                raise
            refresh_audit["decode_status"] = "completed"
            refresh_audit["decoded_source_uv"] = np.asarray(source_point, dtype=np.float64).tolist()
            refresh_audit["decoded_target_uv"] = np.asarray(target_point, dtype=np.float64).tolist()
            arrow_state.update({"rgb": rendered, "source_uv": source_point, "destination_uv": target_point, "render_audit": refresh_audit})
            try:
                _write_json(refresh_dir / f"refresh_{arrow_refresh_index:02d}.json", refresh_audit)
            except Exception:
                pass
            return rendered, source_point, target_point

        refresh_current_arrow_inputs(env)
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
            setattr(_env, "_grasp_controller_robot_calibration_probe", probe)

        def before_capture(_env: Any, attempt_index: int) -> Any:
            # Recovery has already completed. Re-preshape, capture a fresh
            # frame, and refresh arrow inputs before generating candidates.
            nonlocal opening_audit, arrow_refresh_attempt
            arrow_refresh_attempt = int(attempt_index)
            retry_output = output_dir / "attempts" / f"attempt_{int(attempt_index):02d}"
            opening_audit = run_preshape(_env, retry_output / "opening_preshape")
            retry_capture = episode.capture_agentview(_env, resolution=resolution, camera_name=AGENTVIEW)
            refresh_current_arrow_inputs(_env)
            refresh_arrow(retry_capture)
            return retry_capture

        transform = worker_state.get("transform")
        if transform is None or worker_state.get("worker") is None:
            raise RuntimeError("Panda calibration/model worker was not initialized")

        def run_one(*, context: CanaryEpisodeContext, evaluator: Callable[[Any], bool] | None,
                    retreat_completed_callback: Callable[[], None] | None = None) -> Mapping[str, Any]:
            action_count_before = int(getattr(env, "_grasp_controller_action_count", 0))
            trace_before = getattr(env, "_arrow_motion_trace", None)
            # Do not let a previous candidate's phase audit leak into a
            # pre-motion failure or into the recovery snapshot for this one.
            setattr(env, "_arrow_phase_audit", [])
            attempt_output_dir = output_dir / "attempts" / f"attempt_{int(context.attempt_index):02d}"
            attempt_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                from vla_benchmarking.evaluation import run_arrow_pick_place_eval as _episode_budget_module
            except ImportError:  # pragma: no cover - direct script use
                from vla_benchmarking.evaluation import run_arrow_pick_place_eval as _episode_budget_module
            budget = getattr(env, "_grasp_controller_action_budget", None)
            if budget is None:
                budget = _episode_budget_module._ActionBudget(
                    SHARED_ACTION_BUDGET, used=action_count_before
                )
                setattr(env, "_grasp_controller_action_budget", budget)

            if action_count_before >= SHARED_ACTION_BUDGET:
                return {"status": "recovery_failed", "grasp_retained": False, "retreat_complete": False,
                        "total_actions": action_count_before,
                        "error": "shared action budget exhausted before candidate"}
            transform = worker_state.get("transform")
            if transform is None or worker_state.get("worker") is None:
                return {"status": "recovery_failed", "grasp_retained": False,
                        "retreat_complete": False, "total_actions": action_count_before,
                        "error": "live contact calibration unavailable before candidate"}
            if (resolved_opening_profile.get("preshape_required") or resolved_opening_profile["name"] == "preshape40mm"):
                try:
                    actual_opening = float(measure_opening(env))
                    low, high = (float(value) for value in resolved_opening_profile["accepted_opening_band_m"])
                    required_opening = float(context.candidate.opening_m)
                    if not math.isfinite(actual_opening) or not low <= actual_opening <= high or actual_opening < required_opening:
                        raise ValueError(
                            f"measured opening {actual_opening:.6f} m outside [{low:.6f}, {high:.6f}] "
                            f"or below candidate requirement {required_opening:.6f} m"
                        )
                except Exception as exc:
                    return {
                        "status": "recovery_failed", "grasp_retained": False,
                        "retreat_complete": False, "total_actions": action_count_before,
                        "error_type": "PreshapeOpeningGuardError",
                        "error": f"preshape_failed:opening_guard: {exc}",
                    }
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
                    source_grasp_offset=LEGACY_SOURCE_OFFSET_M,
                    destination_release_offset=LEGACY_DESTINATION_OFFSET_M,
                    clearance_m=LIFT_CLEARANCE_M,
                    capture=context.agentview_capture,
                    motion_started_callback=kwargs.get("motion_started_callback"),
                    canary_video_dir=kwargs.get("canary_video_dir"),
                    experimental_candidate=context.candidate,
                    experimental_eef_orientation_transform=transform,
                    experimental_gripper_opening_m=(actual_opening if (resolved_opening_profile.get("preshape_required") or resolved_opening_profile["name"] == "preshape40mm") else worker_state.get("opening_m")),
                    post_lift_retention_gate=retention_gate,
                    retreat_completed_callback=retreat_completed_callback,
                    experimental_action_budget=getattr(env, "_grasp_controller_action_budget", None),
                    **candidate_motion_kwargs,
                )
                trace_steps = _newly_sent_actions(env, trace_before)
                if getattr(env, "_grasp_controller_action_budget", None) is not None:
                    setattr(env, "_grasp_controller_action_count", int(env._grasp_controller_action_budget.used))
                else:
                    setattr(env, "_grasp_controller_action_count", action_count_before + trace_steps)
                if int(getattr(env, "_grasp_controller_action_count", 0)) > SHARED_ACTION_BUDGET:
                    raise RuntimeError("candidate exceeded the shared action budget")
                return {
                    "status": "placed" if audit.get("evaluator_success") is not False else "task_failure",
                    "grasp_retained": True, "retreat_complete": True,
                    "total_actions": int(getattr(env, "_grasp_controller_action_count", 0)),
                    "evaluator_called": bool(evaluator is not None),
                    "evaluator_success": audit.get("evaluator_success"),
                    "audit": audit,
                    "motion_phases_reached": _completed_motion_phases(audit.get("phases", [])),
                }
            except Exception as exc:
                if "evaluator failure" in str(exc).lower():
                    raise
                trace_steps = _newly_sent_actions(env, trace_before)
                if getattr(env, "_grasp_controller_action_budget", None) is not None:
                    setattr(env, "_grasp_controller_action_count", int(env._grasp_controller_action_budget.used))
                else:
                    setattr(env, "_grasp_controller_action_count", action_count_before + trace_steps)
                attempt_phases = list(getattr(env, "_arrow_phase_audit", []) or [])
                reached = _completed_motion_phases(attempt_phases)
                recovery = _recover_after_failed_candidate(
                    env, output_dir=attempt_output_dir,
                    orientation_transform=transform, motion_settings=motion_settings,
                )
                return {
                    "status": "grasp_failed" if recovery.get("retreat_complete") else "recovery_failed",
                    "grasp_retained": False, "retreat_complete": bool(recovery.get("retreat_complete")),
                    "total_actions": int(getattr(env, "_grasp_controller_action_count", 0)),
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
            experiment_schema=RGBD_EXPERIMENT_SCHEMA,
            region_backend=args.region_backend,
            before_propose_callback=refresh_robot_calibration,
            motion_profile=resolved_motion_profile["name"],
            motion_profile_params=resolved_motion_profile,
            motion_diagnostics=bool(args.motion_diagnostics),
            observation_profile=resolved_observation_profile["name"],
            observation_profile_params=resolved_observation_profile,
            opening_profile=resolved_opening_profile["name"],
            opening_profile_params=resolved_opening_profile,
            before_capture_fn=before_capture,
        )
        final = result.get("final_result") if isinstance(result.get("final_result"), Mapping) else {}
        audit = final.get("audit") if isinstance(final, Mapping) else None
        return {
            "audit_path": (output_dir / "canonical_grasp_manifest.json").as_posix(),
            "evaluator_success": final.get("evaluator_success") if isinstance(final, Mapping) else None,
            "total_actions": int(getattr(env, "_grasp_controller_action_count", 0)),
            "gripper_open_preflight": getattr(env, "_grasp_controller_gripper_open", None),
            "observation_hover": getattr(env, "_grasp_controller_observation_hover", None),
            "opening_profile": resolved_opening_profile["name"],
            "opening_profile_params": _json_safe(resolved_opening_profile),
            "opening_preshape": _json_safe(opening_audit),
            "arrow_refresh_audit": _json_safe(arrow_state.get("render_audit")),
            "phases": audit.get("phases", []) if isinstance(audit, Mapping) else [],
            "motion_phases_reached": final.get("motion_phases_reached", []) if isinstance(final, Mapping) else [],
            "grasp_search": [], "canary_manifest": result,
            "experimental_identity": RGBD_EXPERIMENT_SCHEMA,
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
                execution_cell_identities=(
                    tuple(execution_cell_identities_by_suite[suite_mode])
                    if args.phase == "prefix"
                    and execution_cell_identities_by_suite is not None
                    and suite_mode in execution_cell_identities_by_suite
                    else None
                ),
                cell_completed_callback=cell_completed_callback,
                experiment_metadata={
                    "schema_version": RGBD_EXPERIMENT_SCHEMA,
                    "variant": args.variant,
                    "region_backend": args.region_backend,
                    "backend": "rgbd_region",
                    "sam3_used": False,
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
                    "opening_profile": resolved_opening_profile["name"],
                    "opening_profile_params": _json_safe(resolved_opening_profile),
                },
            )
    except Exception as exc:
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["suite_summaries"] = suite_summaries
        _write_json(root / "canonical_grasp_failed.json", manifest)
        print(f"canonical grasp controller failed: {exc}", file=sys.stderr)
        return 2
    manifest["suite_summaries"] = suite_summaries
    _write_json(root / "canonical_grasp_summary.json", manifest)
    print(json.dumps({"summary_path": (root / "canonical_grasp_summary.json").as_posix(), "suite_summaries": suite_summaries}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - operator guard
    raise SystemExit(
        "arrow_grasp_controller.controller.runner is internal; use "
        "vla_benchmarking/arrow_grasp_controller/run_grasp_controller.py"
    )
