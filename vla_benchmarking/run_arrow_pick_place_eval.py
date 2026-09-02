#!/usr/bin/env python3
"""Arrow-only RGB-D pick/place MVP for LIBERO.

The runner deliberately keeps perception and motion separate.  The only data
passed to :mod:`arrow_controller` are the clean RGB frame, the one-arrow RGB
frame, aligned metric depth, and camera calibration.  LIBERO object poses,
meshes, bboxes, and evaluator state never cross that boundary.

Motion is opt-in.  ``python run_arrow_pick_place_eval.py`` captures and audits
one episode without calling ``step``; pass ``--execute-motion`` only after
reviewing the generated artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:  # Optional outside the LIBERO environment (all pure seams stay importable).
    from robosuite.utils import camera_utils
except ImportError:  # pragma: no cover - exercised by dependency-free tests
    camera_utils = None

try:
    from radomize_scenes import settle_physics as _settle_physics
except ImportError:  # pragma: no cover - dependency-free import guard
    _settle_physics = None

try:
    # Prefer package-relative imports so ``vla_benchmarking`` callers use the
    # same pure controller seam as the direct Legion script entrypoint.
    from .arrow_controller import (
        build_bowl_waypoints,
        decode_arrow,
        decode_arrow_diagnostics,
        deproject_endpoint,
        refine_rgbd_endpoint,
        normalized_osc_action,
    )
except (ImportError, ValueError):  # pragma: no cover - direct script fallback
    try:
        from arrow_controller import (
            build_bowl_waypoints,
            decode_arrow,
            decode_arrow_diagnostics,
            deproject_endpoint,
            refine_rgbd_endpoint,
            normalized_osc_action,
        )
    except ImportError:  # pragma: no cover - a clear error is raised on use
        build_bowl_waypoints = decode_arrow = decode_arrow_diagnostics = None
        deproject_endpoint = refine_rgbd_endpoint = normalized_osc_action = None


CAMERA_NAME = "agentview"
# The only resolution covered by the live-validated calibration/profile.  Keep
# this conservative default explicit; other resolutions are dry-run friendly
# but require an opt-in before sending motion to LIBERO.
DEFAULT_RESOLUTION = 256
DEFAULT_GOAL_OBJECT = "plate_1"
DEFAULT_SUBJECT = "akita_black_bowl_1"
# Fixed, camera/profile-specific offsets from visual endpoints to a robust
# bowl-rim grasp/release point. These are constant transforms, never runtime
# simulator object coordinates.
DEFAULT_PROFILE_NAME = "libero_spatial_akita_bowl_agentview_v1"
CANDIDATE_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v1"
CANDIDATE_V2_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v2"
CANDIDATE_V3_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v3"
CANDIDATE_V4_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_lowerq_gain_v4"
CANDIDATE_V5_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_visible_anchor_v5"
CANDIDATE_V6_CONTROLLER_VARIANT_NAME = "libero_spatial_akita_bowl_agentview_candidate_directional_approach_v6"
DEFAULT_SOURCE_GRASP_OFFSET_M = (0.0146, 0.0432, 0.0244)
DEFAULT_DESTINATION_RELEASE_OFFSET_M = (-0.0057, 0.0484, 0.0310)
DEFAULT_GRIPPER_DWELL_STEPS = 20
# The OSC controller can settle within a few millimetres of a waypoint while
# contact dynamics keep the final error from crossing exactly 1 cm.  Keep this
# explicit and conservative so task-specific diagnostics can distinguish a
# reachable near-stop from a true runaway.
WAYPOINT_POSITION_TOLERANCE_M = 0.015
VERIFIED_PROFILE_TASK_ID = 0
VERIFIED_PROFILE_SEED = 1000
VERIFIED_PROFILE_RESOLUTION = 256
VERIFIED_PROFILE_CONDITIONS = {
    "task_id": VERIFIED_PROFILE_TASK_ID,
    "seed": VERIFIED_PROFILE_SEED,
    "resolution": VERIFIED_PROFILE_RESOLUTION,
}
# Coarse finite safety volume for the adjusted controller endpoints.  This is
# a workspace sanity check only, not object detection or collision validation.
# The limits cover the validated LIBERO tabletop scene and intentionally leave
# a generous margin around its nominal bowl/plate region.
WORKSPACE_BOUNDS_M = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (0.0, 1.6),
}
OSC_ACTION_DIM = 7
OSC_LOW = -1.0
OSC_HIGH = 1.0
PHASES = (
    "pregrasp",
    "descend",
    "close",
    "lift",
    "preplace",
    "descend_place",
    "open",
    "retreat",
)
PHASE_WAYPOINT_INDEX = {
    "pregrasp": 0,
    "descend": 1,
    "close": 1,
    "lift": 2,
    "preplace": 3,
    "descend_place": 4,
    "open": 4,
    "retreat": 5,
}
# Named policy metadata is kept separate from controller geometry.  Callers can
# still override the common timeout, while the audit records the intended
# phase-specific tolerance and action role.
PHASE_POLICIES = {
    "pregrasp": {"role": "transit", "tolerance_m": 0.015},
    "descend": {"role": "approach", "tolerance_m": 0.010},
    "close": {"role": "grasp_dwell", "tolerance_m": None},
    "lift": {"role": "carry_clearance", "tolerance_m": 0.015},
    "preplace": {"role": "transit", "tolerance_m": 0.015},
    "descend_place": {"role": "approach", "tolerance_m": 0.010},
    "open": {"role": "release_dwell", "tolerance_m": None},
    "retreat": {"role": "retreat", "tolerance_m": 0.015},
}
DEFAULT_OSC_SCALES = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
SUITE_MODES = ("vanilla", "sealed_randomized")
DEFAULT_SUITE_MODE = "vanilla"
DEFAULT_STALL_WINDOW_STEPS = 10
DEFAULT_STALL_DELTA_M = 1e-4
DEFAULT_RECOVERY_ATTEMPTS = 1
DEFAULT_RECOVERY_STEPS = 3
MAX_NORMALIZED_MASK_FRACTION = 0.25
ENDPOINT_DEPTH_STATISTICS = ("median", "lower_quantile", "nearest_valid")
DEFAULT_ENDPOINT_DEPTH_STATISTIC = "median"
DEFAULT_ENDPOINT_DEPTH_QUANTILE = 0.25
CANDIDATE_ENDPOINT_DEPTH_STATISTIC = "lower_quantile"
CANDIDATE_ENDPOINT_DEPTH_QUANTILE = 0.25
CANDIDATE_APPROACH_TOLERANCE_M = 0.015
CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION = 0.40
CANDIDATE_V3_WAYPOINT_TOLERANCE_M = 0.025
CANDIDATE_V4_OSC_POSITION_SCALE_M = 0.035
ARROW_ANCHOR_POLICIES = ("bbox_center", "visible_inset")
CANDIDATE_V6_APPROACH_LATERAL_OFFSET_M = 0.04
CANDIDATE_V2_WORKSPACE_BOUNDS_M = MappingProxyType({
    "x": WORKSPACE_BOUNDS_M["x"],
    "y": WORKSPACE_BOUNDS_M["y"],
    "z": (WORKSPACE_BOUNDS_M["z"][0], 1.8),
})
MAX_APPROACH_TOLERANCE_M = 0.025
MAX_MASK_FRACTION_FOR_MOTION = 0.50
MAX_OSC_POSITION_SCALE_M = 0.10


@dataclass(frozen=True)
class ControllerVariantConfig:
    """Canonical, hashable controller/runtime configuration provenance.

    The hash covers control policy knobs, not observations or simulator state.
    This makes paired runs auditable while preserving the old function-level
    arguments as the compatibility surface.
    """

    name: str = DEFAULT_PROFILE_NAME
    controller: str = "OSC_POSE"
    suite_mode: str = DEFAULT_SUITE_MODE
    phase_timeout_steps: int = 80
    gripper_dwell_steps: int = DEFAULT_GRIPPER_DWELL_STEPS
    stall_window_steps: int = DEFAULT_STALL_WINDOW_STEPS
    stall_delta_m: float = DEFAULT_STALL_DELTA_M
    recovery_attempts: int = DEFAULT_RECOVERY_ATTEMPTS
    recovery_steps: int = DEFAULT_RECOVERY_STEPS
    endpoint_depth_statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC
    endpoint_depth_quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE
    approach_tolerance_m: float | None = None
    waypoint_tolerance_m: float | None = None
    osc_position_scale_m: float | None = None
    arrow_anchor_policy: str = "bbox_center"
    approach_lateral_offset_m: float | None = None
    max_mask_fraction_for_motion: float | None = None
    workspace_bounds_m: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if self.suite_mode not in SUITE_MODES:
            raise ValueError(f"suite_mode must be one of {SUITE_MODES}, got {self.suite_mode!r}")
        if self.phase_timeout_steps <= 0 or self.gripper_dwell_steps <= 0:
            raise ValueError("controller phase and gripper step limits must be positive")
        if self.stall_window_steps < 0 or self.stall_delta_m < 0 or self.recovery_attempts < 0 or self.recovery_steps < 0:
            raise ValueError("stall/recovery limits must be non-negative")
        if self.endpoint_depth_statistic not in ENDPOINT_DEPTH_STATISTICS:
            raise ValueError(
                f"endpoint_depth_statistic must be one of {ENDPOINT_DEPTH_STATISTICS}"
            )
        if not 0.0 <= float(self.endpoint_depth_quantile) <= 1.0:
            raise ValueError("endpoint_depth_quantile must be in [0, 1]")
        if self.approach_tolerance_m is not None and (
            not np.isfinite(float(self.approach_tolerance_m))
            or float(self.approach_tolerance_m) <= 0.0
            or float(self.approach_tolerance_m) > MAX_APPROACH_TOLERANCE_M
        ):
            raise ValueError(
                f"approach_tolerance_m must be in (0, {MAX_APPROACH_TOLERANCE_M}]"
            )
        if self.approach_tolerance_m is not None and self.name != CANDIDATE_CONTROLLER_VARIANT_NAME:
            if self.name not in {
                CANDIDATE_V2_CONTROLLER_VARIANT_NAME,
                CANDIDATE_V3_CONTROLLER_VARIANT_NAME,
                CANDIDATE_V4_CONTROLLER_VARIANT_NAME,
                CANDIDATE_V5_CONTROLLER_VARIANT_NAME,
                CANDIDATE_V6_CONTROLLER_VARIANT_NAME,
            }:
                raise ValueError("relaxed approach tolerance is reserved for the candidate controller variant")
        if self.waypoint_tolerance_m is not None and (
            not np.isfinite(float(self.waypoint_tolerance_m))
            or float(self.waypoint_tolerance_m) <= 0.0
            or float(self.waypoint_tolerance_m) > MAX_APPROACH_TOLERANCE_M
        ):
            raise ValueError(
                f"waypoint_tolerance_m must be in (0, {MAX_APPROACH_TOLERANCE_M}]"
            )
        if self.waypoint_tolerance_m is not None and self.name != CANDIDATE_V3_CONTROLLER_VARIANT_NAME:
            raise ValueError("waypoint tolerance is reserved for the v3 candidate controller variant")
        if self.osc_position_scale_m is not None and (
            not np.isfinite(float(self.osc_position_scale_m))
            or float(self.osc_position_scale_m) <= 0.0
            or float(self.osc_position_scale_m) > MAX_OSC_POSITION_SCALE_M
        ):
            raise ValueError(
                f"osc_position_scale_m must be in (0, {MAX_OSC_POSITION_SCALE_M}]"
            )
        if self.osc_position_scale_m is not None and self.name != CANDIDATE_V4_CONTROLLER_VARIANT_NAME:
            raise ValueError("OSC position scale is reserved for the v4 candidate controller variant")
        if self.arrow_anchor_policy not in ARROW_ANCHOR_POLICIES:
            raise ValueError(
                f"arrow_anchor_policy must be one of {ARROW_ANCHOR_POLICIES}"
            )
        if self.arrow_anchor_policy != "bbox_center" and self.name != CANDIDATE_V5_CONTROLLER_VARIANT_NAME:
            raise ValueError("non-center arrow anchors are reserved for the v5 candidate controller variant")
        if self.approach_lateral_offset_m is not None and (
            not np.isfinite(float(self.approach_lateral_offset_m))
            or float(self.approach_lateral_offset_m) <= 0.0
            or float(self.approach_lateral_offset_m) > 0.10
        ):
            raise ValueError("approach_lateral_offset_m must be in (0, 0.10]")
        if self.approach_lateral_offset_m is not None and self.name != CANDIDATE_V6_CONTROLLER_VARIANT_NAME:
            raise ValueError("directional approach offset is reserved for the v6 candidate controller variant")
        if self.max_mask_fraction_for_motion is not None and (
            not np.isfinite(float(self.max_mask_fraction_for_motion))
            or not 0.0 <= float(self.max_mask_fraction_for_motion) <= MAX_MASK_FRACTION_FOR_MOTION
        ):
            raise ValueError(
                f"max_mask_fraction_for_motion must be in [0, {MAX_MASK_FRACTION_FOR_MOTION}]"
            )
        if self.workspace_bounds_m is not None:
            for axis in ("x", "y", "z"):
                limits = np.asarray(self.workspace_bounds_m.get(axis, ()), dtype=np.float64).reshape(-1)
                if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
                    raise ValueError(f"workspace_bounds_m[{axis!r}] must be two finite ordered values")
            if float(self.workspace_bounds_m["z"][1]) > 1.8:
                raise ValueError("workspace_bounds_m z max cannot exceed 1.8 m")

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "controller": self.controller,
            "suite_mode": self.suite_mode,
            "phase_timeout_steps": int(self.phase_timeout_steps),
            "gripper_dwell_steps": int(self.gripper_dwell_steps),
            "stall_window_steps": int(self.stall_window_steps),
            "stall_delta_m": float(self.stall_delta_m),
            "recovery_attempts": int(self.recovery_attempts),
            "recovery_steps": int(self.recovery_steps),
            "endpoint_depth_statistic": self.endpoint_depth_statistic,
            "endpoint_depth_quantile": float(self.endpoint_depth_quantile),
            "approach_tolerance_m": (
                None if self.approach_tolerance_m is None else float(self.approach_tolerance_m)
            ),
            "waypoint_tolerance_m": (
                None if self.waypoint_tolerance_m is None else float(self.waypoint_tolerance_m)
            ),
            "osc_position_scale_m": (
                None if self.osc_position_scale_m is None else float(self.osc_position_scale_m)
            ),
            "arrow_anchor_policy": self.arrow_anchor_policy,
            "approach_lateral_offset_m": (
                None
                if self.approach_lateral_offset_m is None
                else float(self.approach_lateral_offset_m)
            ),
            "max_mask_fraction_for_motion": (
                None if self.max_mask_fraction_for_motion is None
                else float(self.max_mask_fraction_for_motion)
            ),
            "workspace_bounds_m": (
                None if self.workspace_bounds_m is None else {
                    axis: [float(value) for value in self.workspace_bounds_m[axis]]
                    for axis in ("x", "y", "z")
                }
            ),
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def hash(self) -> str:
        """Short compatibility spelling for callers storing variant hashes."""
        return self.config_hash

    def canonical_json(self) -> str:
        return json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))

    def provenance(self) -> dict[str, Any]:
        return {"canonical": self.canonical(), "config_hash": self.config_hash}


def _resolve_controller_variant(
    value: ControllerVariantConfig | str | None,
    *,
    suite_mode: str,
    phase_timeout_steps: int = 80,
    gripper_dwell_steps: int = DEFAULT_GRIPPER_DWELL_STEPS,
    recovery_attempts: int = DEFAULT_RECOVERY_ATTEMPTS,
    recovery_steps: int = DEFAULT_RECOVERY_STEPS,
    endpoint_depth_statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC,
    endpoint_depth_quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE,
    approach_tolerance_m: float | None = None,
    max_mask_fraction_for_motion: float | None = None,
    workspace_bounds_m: Mapping[str, Sequence[float]] | None = None,
) -> ControllerVariantConfig:
    if isinstance(value, ControllerVariantConfig):
        return value
    name = str(value) if value is not None else DEFAULT_PROFILE_NAME
    if name in {
        CANDIDATE_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V2_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V3_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V4_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V5_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V6_CONTROLLER_VARIANT_NAME,
    }:
        endpoint_depth_statistic = CANDIDATE_ENDPOINT_DEPTH_STATISTIC
        endpoint_depth_quantile = CANDIDATE_ENDPOINT_DEPTH_QUANTILE
        approach_tolerance_m = CANDIDATE_APPROACH_TOLERANCE_M
    waypoint_tolerance_m = None
    if name == CANDIDATE_V3_CONTROLLER_VARIANT_NAME:
        waypoint_tolerance_m = CANDIDATE_V3_WAYPOINT_TOLERANCE_M
    if name == CANDIDATE_V2_CONTROLLER_VARIANT_NAME:
        max_mask_fraction_for_motion = CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        workspace_bounds_m = CANDIDATE_V2_WORKSPACE_BOUNDS_M
    if name == CANDIDATE_V3_CONTROLLER_VARIANT_NAME:
        max_mask_fraction_for_motion = CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        workspace_bounds_m = CANDIDATE_V2_WORKSPACE_BOUNDS_M
    osc_position_scale_m = None
    if name == CANDIDATE_V4_CONTROLLER_VARIANT_NAME:
        max_mask_fraction_for_motion = CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        workspace_bounds_m = CANDIDATE_V2_WORKSPACE_BOUNDS_M
        endpoint_depth_statistic = CANDIDATE_ENDPOINT_DEPTH_STATISTIC
        endpoint_depth_quantile = CANDIDATE_ENDPOINT_DEPTH_QUANTILE
        approach_tolerance_m = CANDIDATE_APPROACH_TOLERANCE_M
        osc_position_scale_m = CANDIDATE_V4_OSC_POSITION_SCALE_M
    arrow_anchor_policy = "visible_inset" if name == CANDIDATE_V5_CONTROLLER_VARIANT_NAME else "bbox_center"
    if name == CANDIDATE_V5_CONTROLLER_VARIANT_NAME:
        max_mask_fraction_for_motion = CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        workspace_bounds_m = CANDIDATE_V2_WORKSPACE_BOUNDS_M
        endpoint_depth_statistic = CANDIDATE_ENDPOINT_DEPTH_STATISTIC
        endpoint_depth_quantile = CANDIDATE_ENDPOINT_DEPTH_QUANTILE
        approach_tolerance_m = CANDIDATE_APPROACH_TOLERANCE_M
    approach_lateral_offset_m = (
        CANDIDATE_V6_APPROACH_LATERAL_OFFSET_M
        if name == CANDIDATE_V6_CONTROLLER_VARIANT_NAME
        else None
    )
    if name == CANDIDATE_V6_CONTROLLER_VARIANT_NAME:
        max_mask_fraction_for_motion = CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        workspace_bounds_m = CANDIDATE_V2_WORKSPACE_BOUNDS_M
        endpoint_depth_statistic = CANDIDATE_ENDPOINT_DEPTH_STATISTIC
        endpoint_depth_quantile = CANDIDATE_ENDPOINT_DEPTH_QUANTILE
        approach_tolerance_m = CANDIDATE_APPROACH_TOLERANCE_M
    return ControllerVariantConfig(
        name=name,
        suite_mode=suite_mode,
        phase_timeout_steps=phase_timeout_steps,
        gripper_dwell_steps=gripper_dwell_steps,
        recovery_attempts=recovery_attempts,
        recovery_steps=recovery_steps,
        endpoint_depth_statistic=endpoint_depth_statistic,
        endpoint_depth_quantile=endpoint_depth_quantile,
        approach_tolerance_m=approach_tolerance_m,
        waypoint_tolerance_m=waypoint_tolerance_m,
        osc_position_scale_m=osc_position_scale_m,
        arrow_anchor_policy=arrow_anchor_policy,
        approach_lateral_offset_m=approach_lateral_offset_m,
        max_mask_fraction_for_motion=max_mask_fraction_for_motion,
        workspace_bounds_m=workspace_bounds_m,
    )


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration and orientation provenance for one RGB-D capture."""

    camera_name: str
    width: int
    height: int
    intrinsic: list[list[float]]
    world_from_camera: list[list[float]]
    raw_projection_intrinsic: list[list[float]] | None = None
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "same_sim_render_call"
    source: str = "robosuite.camera_utils"
    projection_vertical_axis: str = "robosuite_projection_v_up_converted_to_image_v_down"


@dataclass
class CapturedRGBD:
    rgb: np.ndarray
    normalized_depth: np.ndarray
    metric_depth: np.ndarray
    calibration: CameraCalibration
    observation: Mapping[str, Any] | None = None
    depth_conversion_mode: str = "unknown"
    depth_sanitization: Mapping[str, Any] | None = None

    @property
    def raw_depth(self) -> np.ndarray:
        """Depth pixels exactly as produced by the RGB-D capture source."""
        return self.normalized_depth


def _require(function: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if function is None:
        raise RuntimeError(
            f"arrow controller function {name} is unavailable; install/provide arrow_controller"
        )
    return function


def settle_libero_env(env: Any, *, max_steps: int) -> dict[str, Any]:
    """Settle the inner LIBERO env with the robot frozen and retain safe diagnostics."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if _settle_physics is None:
        raise RuntimeError("radomize_scenes.settle_physics is required before motion")
    inner = getattr(env, "env", env)
    result = _settle_physics(inner, max_steps=int(max_steps))
    if not isinstance(result, Mapping):
        raise RuntimeError("settle_physics returned a non-mapping diagnostic")
    try:
        diagnostics = {
            "steps": int(result["steps_taken"]),
            "final_max_velocity_m_s": float(result["final_max_vel"]),
            "settled": bool(result["settled"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("settle_physics returned incomplete diagnostics") from exc
    setattr(env, "_arrow_settle_diagnostics", diagnostics)
    return diagnostics


def _close_environment_quietly(env: Any) -> None:
    """Best-effort close used when construction fails before the caller owns env.

    Cleanup must also run for ``KeyboardInterrupt``/``SystemExit`` paths, but a
    close failure must never mask the construction exception.  This helper is
    intentionally local to the direct runner rather than changing LIBERO's
    lifecycle contract.
    """
    close = getattr(env, "close", None)
    if close is None:
        return
    try:
        close()
    except BaseException:
        pass


def _as_rgb(array: Any) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # Renderer outputs may be views into a reusable MuJoCo/robosuite buffer.
    # Own the pixels before a subsequent observation read can reuse that buffer.
    return np.array(image, dtype=np.uint8, order="C", copy=True)


def _as_depth(array: Any, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(array)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape != shape:
        raise ValueError(f"RGB/depth are not aligned: RGB={shape}, depth={depth.shape}")
    if not np.isfinite(depth).any():
        raise ValueError("normalized depth contains no finite pixels")
    # As with RGB, force ownership: ``capture_agentview`` may read a second
    # observation after this point for proprioception, and some renderers reuse
    # their depth buffer in-place.  Retaining a view can silently turn a valid
    # frame into uninitialized/overwritten values (the sealed LIBERO symptom).
    return np.array(depth, dtype=np.float32, order="C", copy=True)


def _raw_observation(env: Any) -> Mapping[str, Any] | None:
    """Read one simulator observation without reading object state."""
    for owner in (getattr(env, "env", None), env):
        getter = getattr(owner, "_get_observations", None)
        if getter is None:
            continue
        try:
            return getter(force_update=True)
        except TypeError:
            return getter()
    return None


def _declared_depth_encoding(env: Any, *, source: str, camera_name: str) -> str:
    """Resolve an explicit producer declaration; never infer units by magnitude."""
    for owner in (env, getattr(env, "env", None)):
        if owner is None:
            continue
        for name in ("_arrow_depth_encoding", "depth_encoding", "depth_units"):
            declared = getattr(owner, name, None)
            if declared is not None:
                value = str(declared).lower()
                if value in {"normalized", "metric"}:
                    return value
                raise ValueError(f"unsupported {camera_name} depth encoding {declared!r}")
    # These are the only implicit producer contracts accepted by the direct
    # runner.  Other fakes/wrappers must declare their units explicitly.
    owners = (env, getattr(env, "env", None))
    if source == "observation" and any(
        owner is not None and (
            owner.__class__.__name__ in {"OffScreenRenderEnv", "ControlEnv"}
        )
        for owner in owners
    ):
        return "normalized"
    raise ValueError(
        f"depth producer {source} for {camera_name} is unknown; declare depth_encoding="
        "'normalized' or 'metric'"
    )


def _render_pair_with_encoding(
    env: Any, camera_name: str, width: int, height: int
) -> tuple[Any, Any, Mapping[str, Any] | None, str]:
    """Return one aligned RGB/depth pair plus explicit producer units."""
    render = getattr(env, "render", None)
    if render is not None:
        try:
            result = render(camera_name=camera_name, width=width, height=height, depth=True)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1], None, _declared_depth_encoding(
                    env, source="render", camera_name=camera_name
                )
        except (TypeError, NotImplementedError):
            pass

    observation = _raw_observation(env)
    if observation is not None:
        rgb = observation.get(f"{camera_name}_image")
        depth = observation.get(f"{camera_name}_depth")
        if rgb is not None and depth is not None:
            return rgb, depth, observation, _declared_depth_encoding(
                env, source="observation", camera_name=camera_name
            )
    raise RuntimeError(
        f"could not capture aligned {camera_name} RGB+depth; "
        "construct LIBERO with camera_depths=True"
    )


def _render_pair(
    env: Any, camera_name: str, width: int, height: int
) -> tuple[Any, Any, Mapping[str, Any] | None]:
    """Backward-compatible three-value render seam; units are resolved internally."""
    rgb, depth, observation, _encoding = _render_pair_with_encoding(
        env, camera_name, width, height
    )
    return rgb, depth, observation


def build_camera_calibration(sim: Any, camera_name: str, width: int, height: int) -> CameraCalibration:
    """Build image-aligned calibration from robosuite's canonical hooks.

    robosuite returns ``K`` for projection coordinates whose vertical axis is
    positive upward.  The rendered agentview arrays and arrow pixels use image
    coordinates with ``v`` positive downward, matching the existing
    ``world_to_pixel`` implementation, so fy/cy are converted explicitly.
    """
    if camera_utils is None:
        raise RuntimeError("robosuite camera_utils is required for metric deprojection")
    intrinsic = camera_utils.get_camera_intrinsic_matrix(
        sim, camera_name, camera_height=height, camera_width=width
    )
    extrinsic = camera_utils.get_camera_extrinsic_matrix(sim, camera_name)
    raw_intrinsic = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if raw_intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"unexpected calibration shapes K={raw_intrinsic.shape}, T={extrinsic.shape}")
    image_intrinsic = raw_intrinsic.copy()
    image_intrinsic[1, 1] = -raw_intrinsic[1, 1]
    image_intrinsic[1, 2] = float(height) - raw_intrinsic[1, 2]
    return CameraCalibration(
        camera_name=camera_name,
        width=int(width),
        height=int(height),
        intrinsic=image_intrinsic.tolist(),
        world_from_camera=extrinsic.tolist(),
        raw_projection_intrinsic=raw_intrinsic.tolist(),
    )


def _normalized_valid_mask(depth: np.ndarray) -> np.ndarray:
    source = np.asarray(depth)
    if source.ndim == 3 and source.shape[-1] == 1:
        source = source[..., 0]
    finite = np.isfinite(source)
    return finite & (source >= 0.0) & (source <= 1.0)


def sanitize_normalized_depth(depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask invalid normalized sentinels without changing the preserved raw input."""
    source = np.asarray(depth)
    if source.ndim == 3 and source.shape[-1] == 1:
        source = source[..., 0]
    if source.ndim != 2:
        raise ValueError(f"depth must be HxW, got {source.shape}")
    in_range = _normalized_valid_mask(source)
    candidates = source[in_range & (source > 0.0)]
    if candidates.size == 0:
        raise ValueError("normalized depth contains no positive finite in-range pixels")
    masked = ~in_range
    fallback = float(np.median(candidates))
    sanitized = source if not np.any(masked) else np.array(source, dtype=np.float32, copy=True)
    if np.any(masked):
        sanitized[masked] = fallback
    return np.ascontiguousarray(sanitized.astype(np.float32, copy=False)), {
        "input_encoding": "normalized",
        "masked_pixel_count": int(np.count_nonzero(masked)),
        "total_pixel_count": int(source.size),
        "masked_fraction": float(np.count_nonzero(masked) / source.size),
        "max_mask_fraction_for_motion": MAX_NORMALIZED_MASK_FRACTION,
        "fallback_value": fallback,
        "fallback_rule": "median_positive_finite_in_range_0_1",
        "metric_mask_restore": "masked_pixels_restored_to_nan",
    }


def normalized_depth_to_metric(
    sim: Any, depth: np.ndarray, *, encoding: str | None = None
) -> np.ndarray:
    """Convert depth under an explicit producer encoding contract.

    ``encoding='normalized'`` delegates to robosuite's conversion hook;
    ``encoding='metric'`` preserves metres exactly.  Unknown units fail closed
    rather than being inferred from magnitude (metric depths can be below one,
    and normalized frames can contain outliers).
    """
    if encoding not in {"normalized", "metric"}:
        raise ValueError(
            "depth encoding is unknown; pass encoding='normalized' or encoding='metric'"
        )
    source = np.asarray(depth)
    if source.ndim == 3 and source.shape[-1] == 1:
        source = source[..., 0]
    if source.ndim != 2:
        raise ValueError(f"depth must be HxW, got {source.shape}")
    if encoding == "metric":
        if np.any(np.isfinite(source) & (source < 0)):
            raise ValueError("depth contains negative finite pixels")
        if not np.isfinite(source).any() or not np.isfinite(source[source > 0]).any():
            raise ValueError("depth contains no positive finite pixels")
        converted = source
    else:
        if camera_utils is None:
            raise RuntimeError("robosuite camera_utils.get_real_depth_map is required")
        sanitized, _sanitization = sanitize_normalized_depth(source)
        converted = camera_utils.get_real_depth_map(sim, sanitized)
    metric = np.asarray(converted, dtype=np.float32)
    if metric.ndim == 3 and metric.shape[-1] == 1:
        metric = metric[..., 0]
    if metric.shape != source.shape:
        raise ValueError(
            f"metric depth changed shape from {source.shape} to {metric.shape}"
        )
    if np.any(np.isfinite(metric) & (metric < 0)):
        raise ValueError("metric depth contains negative finite pixels")
    if not np.isfinite(metric).any() or not np.isfinite(metric[metric > 0]).any():
        raise ValueError("metric depth contains no positive finite pixels")
    if encoding == "normalized":
        invalid = ~_normalized_valid_mask(source)
        if np.any(invalid):
            metric = np.array(metric, dtype=np.float32, copy=True)
            metric[invalid] = np.nan
    return np.ascontiguousarray(metric)


def capture_agentview(
    env: Any,
    *,
    resolution: int = DEFAULT_RESOLUTION,
    camera_name: str = CAMERA_NAME,
) -> CapturedRGBD:
    """Capture exactly one aligned clean RGB + normalized/metric depth pair."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    rgb_raw, normalized_raw, observation, depth_encoding = _render_pair_with_encoding(
        env, camera_name, resolution, resolution
    )
    rgb = _as_rgb(rgb_raw)
    normalized = _as_depth(normalized_raw, rgb.shape[:2])
    depth_sanitization: dict[str, Any] | None = None
    if depth_encoding == "normalized":
        _sanitized, depth_sanitization = sanitize_normalized_depth(normalized)
    # ``render(depth=True)`` implementations may return only pixels.  A
    # second non-visual read is allowed for EEF/gripper proprioception; it is
    # never used to reconstruct object geometry and does not affect RGB-D
    # alignment.
    if observation is None:
        observation = _raw_observation(env)
    sim = getattr(env, "sim", None)
    if sim is None:
        raise RuntimeError("capture environment does not expose sim for calibration/depth conversion")
    calibration = build_camera_calibration(sim, camera_name, rgb.shape[1], rgb.shape[0])
    # Pass the raw normalized frame so conversion can restore masked pixels to
    # NaN in metric output; camera_utils still receives a sanitized copy.
    metric = normalized_depth_to_metric(sim, normalized, encoding=depth_encoding)
    depth_mode = (
        "normalized_masked"
        if depth_encoding == "normalized"
        and depth_sanitization is not None
        and depth_sanitization["masked_pixel_count"]
        else depth_encoding
    )
    return CapturedRGBD(
        rgb=rgb,
        normalized_depth=normalized,
        metric_depth=metric,
        calibration=calibration,
        observation=observation,
        depth_conversion_mode=depth_mode,
        depth_sanitization=depth_sanitization,
    )


# Descriptive aliases make the seam convenient for external smoke tests and
# keep the capture contract discoverable without duplicating implementation.
capture_synchronized_frame = capture_agentview
convert_depth_to_metric = normalized_depth_to_metric


def render_exactly_one_arrow(
    clean_rgb: np.ndarray,
    bboxes: Mapping[str, Sequence[float]],
    *,
    subject: str = DEFAULT_SUBJECT,
    goal_object: str = DEFAULT_GOAL_OBJECT,
    line_width: int = 2,
    head_length: int = 16,
    anchor_policy: str = "bbox_center",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render only the subject→goal arrow on a copy of the clean frame.

    Ground-truth bbox generation is deliberately kept on the input-generation
    side.  The returned audit is not passed to the controller.
    """
    if subject not in bboxes or goal_object not in bboxes:
        raise ValueError("exactly-one-arrow rendering requires subject and goal bboxes")
    try:
        from visual_scene_graph import draw_scene_graph_arrows
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("visual_scene_graph is required to render the input arrow") from exc
    if anchor_policy not in ARROW_ANCHOR_POLICIES:
        raise ValueError(f"anchor_policy must be one of {ARROW_ANCHOR_POLICIES}")
    relation = (subject, "goal", goal_object)
    arrow_bboxes = _arrow_anchor_bboxes(
        bboxes, subject=subject, image_shape=_as_rgb(clean_rgb).shape[:2], policy=anchor_policy
    )
    arrow = draw_scene_graph_arrows(
        _as_rgb(clean_rgb), arrow_bboxes, [relation], line_width=line_width,
        head_length=head_length, copy_image=True,
    )
    if np.array_equal(arrow, clean_rgb):
        raise ValueError("one-arrow renderer produced no visible pixels")
    return arrow, {
        "relation_count": 1,
        "relation": list(relation),
        "line_width": int(line_width),
        "head_length": int(head_length),
        "anchor_policy": anchor_policy,
        "input_generation_only": True,
    }


def _arrow_anchor_bboxes(
    bboxes: Mapping[str, Sequence[float]],
    *,
    subject: str,
    image_shape: Sequence[int],
    policy: str,
) -> dict[str, list[float]]:
    """Return input-only bbox anchors, nudging clipped subjects inward.

    The controller still receives only rendered pixels.  This policy addresses
    a clipped source bbox whose mathematical center can lie on an occluded or
    off-frame portion of an object; it never reads simulator state.
    """
    if policy not in ARROW_ANCHOR_POLICIES:
        raise ValueError(f"anchor policy must be one of {ARROW_ANCHOR_POLICIES}")
    result = {str(name): [float(value) for value in bbox] for name, bbox in bboxes.items()}
    if policy == "bbox_center" or subject not in result:
        return result
    height, width = (int(image_shape[0]), int(image_shape[1]))
    x1, y1, x2, y2 = result[subject]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid subject bbox for arrow anchor: {result[subject]}")
    # Preserve the visible bbox extent while moving only the synthetic center
    # toward the in-frame side when the bbox touches an image boundary.
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if x1 <= 0.0:
        cx = x1 + 0.25 * (x2 - x1)
    elif x2 >= float(width):
        cx = x2 - 0.25 * (x2 - x1)
    if y1 <= 0.0:
        cy = y1 + 0.25 * (y2 - y1)
    elif y2 >= float(height):
        cy = y2 - 0.25 * (y2 - y1)
    result[subject] = [2.0 * cx - x2, 2.0 * cy - y2, x2, y2]
    return result


def _apply_directional_pregrasp_offset(
    waypoints: Any,
    source_visual_point: Sequence[float],
    destination_visual_point: Sequence[float],
    offset_m: float | None,
) -> Any:
    """Move only the pregrasp XY waypoint along the visual arrow direction."""
    if offset_m is None:
        return waypoints
    offset = float(offset_m)
    if not np.isfinite(offset) or offset <= 0.0 or offset > 0.10:
        raise ValueError("directional pregrasp offset must be in (0, 0.10]")
    source = np.asarray(source_visual_point, dtype=np.float64).reshape(-1)
    destination = np.asarray(destination_visual_point, dtype=np.float64).reshape(-1)
    if source.shape != (3,) or destination.shape != (3,):
        raise ValueError("visual endpoints must be finite 3D points")
    direction = destination[:2] - source[:2]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-9:
        return waypoints
    adjusted = np.asarray(waypoints, dtype=np.float64)
    if adjusted.ndim != 2 or adjusted.shape[1] != 3 or adjusted.shape[0] < 1:
        raise ValueError("directional pregrasp offset requires an Nx3 waypoint array")
    adjusted = adjusted.copy()
    adjusted[0, :2] += (direction / norm) * offset
    return adjusted


def _call_controller(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a controller seam while allowing a compact positional test fake."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    accepted = {
        key: value for key, value in kwargs.items()
        if any(p.kind == p.VAR_KEYWORD or name == key for name, p in signature.parameters.items())
    }
    return function(**accepted)


def _endpoint(value: Any, names: Sequence[str]) -> np.ndarray:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return np.asarray(value[name], dtype=np.float64)
    for name in names:
        if hasattr(value, name):
            return np.asarray(getattr(value, name), dtype=np.float64)
    if isinstance(value, Sequence) and len(value) >= 2:
        return np.asarray(value[:2], dtype=np.float64)
    raise ValueError(f"arrow controller did not return an endpoint with fields {names}")


def decode_arrow_pixels(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    *,
    encoding: str | Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode source-tail and destination-head pixels from the arrow controller."""
    kwargs = {"clean_rgb": clean_rgb, "arrow_rgb": arrow_rgb}
    if encoding is not None:
        kwargs["encoding"] = encoding
    decoded = _call_controller(_require(decode_arrow, "decode_arrow"), **kwargs)
    source = _endpoint(decoded, ("source_uv", "tail_uv", "source", "tail", "source_xy"))
    target = _endpoint(decoded, ("target_uv", "head_uv", "destination_uv", "target", "head", "target_xy"))
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError(f"arrow endpoints must be (u,v), got {source.shape} and {target.shape}")
    return source, target


def _depth_at_with_audit(
    depth: np.ndarray,
    uv: np.ndarray,
    radius: int = 2,
    *,
    statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC,
    quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE,
) -> tuple[float, dict[str, Any]]:
    if statistic not in ENDPOINT_DEPTH_STATISTICS:
        raise ValueError(f"endpoint_depth_statistic must be one of {ENDPOINT_DEPTH_STATISTICS}")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("endpoint_depth_quantile must be in [0, 1]")
    u, v = np.rint(uv).astype(int)
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError(f"arrow endpoint {(u, v)} lies outside depth frame {depth.shape[::-1]}")
    patch = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError(f"no valid metric depth near arrow endpoint {(u, v)}")
    if statistic == "median":
        selected = float(np.median(valid))
    elif statistic == "lower_quantile":
        selected = float(np.quantile(valid, float(quantile)))
    else:  # nearest_valid
        yy, xx = np.where(np.isfinite(patch) & (patch > 0))
        center = np.asarray([v - max(0, v - radius), u - max(0, u - radius)], dtype=np.float64)
        distances = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        selected = float(valid[int(np.argmin(distances))])
    return selected, {
        "statistic": statistic,
        "quantile": float(quantile),
        "patch_radius": int(radius),
        "valid_sample_count": int(valid.size),
        "selected_depth_m": selected,
    }


def _depth_at(
    depth: np.ndarray,
    uv: np.ndarray,
    radius: int = 2,
    *,
    statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC,
    quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE,
) -> float:
    """Robust endpoint depth estimator with median-compatible defaults."""
    selected, _audit = _depth_at_with_audit(
        depth, uv, radius, statistic=statistic, quantile=quantile
    )
    return selected


def assess_depth_sanitization_for_motion(
    capture: CapturedRGBD,
    endpoint_pixels: Sequence[Sequence[float]],
    *,
    max_mask_fraction: float = MAX_NORMALIZED_MASK_FRACTION,
) -> dict[str, Any]:
    """Apply the conservative normalized-depth validity gate before motion."""
    details = dict(getattr(capture, "depth_sanitization", None) or {})
    mode = str(getattr(capture, "depth_conversion_mode", "unknown"))
    if mode != "normalized_masked":
        return {
            "status": "not_applicable",
            "depth_conversion_mode": mode,
            "masked_pixel_count": int(details.get("masked_pixel_count", 0)),
            "masked_fraction": float(details.get("masked_fraction", 0.0)),
            "max_mask_fraction": float(max_mask_fraction),
            "endpoint_patch_valid": [True for _ in endpoint_pixels],
        }
    masked_fraction = float(details.get("masked_fraction", 1.0))
    endpoint_valid: list[bool] = []
    for pixel in endpoint_pixels:
        try:
            _depth_at(capture.metric_depth, np.asarray(pixel, dtype=np.float64))
        except (ValueError, IndexError, TypeError):
            endpoint_valid.append(False)
        else:
            endpoint_valid.append(True)
    passed = masked_fraction <= float(max_mask_fraction) and all(endpoint_valid)
    return {
        "status": "passed" if passed else "rejected",
        "depth_conversion_mode": mode,
        "masked_pixel_count": int(details.get("masked_pixel_count", 0)),
        "masked_fraction": masked_fraction,
        "max_mask_fraction": float(max_mask_fraction),
        "endpoint_patch_valid": endpoint_valid,
        "rejection_reason": None if passed else (
            "masked_fraction_exceeds_limit"
            if masked_fraction > float(max_mask_fraction)
            else "endpoint_patch_has_no_valid_metric_depth"
        ),
    }


def _refine_or_deproject_endpoint(
    pixel: np.ndarray,
    depth: float,
    K: Any,
    T_world_camera: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the stable RGBD refinement API when available, preserving test seams."""
    if (
        refine_rgbd_endpoint is not None
        and str(getattr(deproject_endpoint, "__module__", "")).endswith("arrow_controller")
    ):
        refined = refine_rgbd_endpoint(pixel, depth, K, T_world_camera)
        world = np.asarray(refined.world_xyz, dtype=np.float64).reshape(-1)
        return world, {
            "method": refined.method,
            "pixel_provenance": refined.pixel_provenance,
            "depth_provenance": refined.depth_provenance,
        }
    world = np.asarray(
        _require(deproject_endpoint, "deproject_endpoint")(
            pixel, depth, K, T_world_camera
        ),
        dtype=np.float64,
    ).reshape(-1)
    return world, {
        "method": "runner_deproject_endpoint_with_robust_depth_scalar",
        "pixel_provenance": "runner_decoded_arrow_endpoint",
        "depth_provenance": "runner_depth_at_median_patch",
    }


def validate_capture_contract(
    capture: CapturedRGBD, *, resolution: int, camera_name: str = CAMERA_NAME
) -> dict[str, Any]:
    """Validate RGB-D/calibration shape, camera, and requested-size provenance."""
    rgb = np.asarray(capture.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"capture contract violation: RGB must be HxWx3, got {rgb.shape}")
    height, width = (int(rgb.shape[0]), int(rgb.shape[1]))

    def depth_shape(value: Any, name: str) -> tuple[int, int]:
        shape = np.asarray(value).shape
        if len(shape) == 3 and shape[-1] == 1:
            shape = shape[:2]
        if len(shape) != 2:
            raise ValueError(f"capture contract violation: {name} must be HxW, got {shape}")
        return int(shape[0]), int(shape[1])

    normalized_shape = depth_shape(capture.normalized_depth, "normalized depth")
    metric_shape = depth_shape(capture.metric_depth, "metric depth")
    if normalized_shape != (height, width) or metric_shape != (height, width):
        raise ValueError(
            "capture contract violation: RGB/depth shapes disagree "
            f"(RGB={(height, width)}, normalized={normalized_shape}, metric={metric_shape})"
        )
    calibration = capture.calibration
    if int(calibration.width) != width or int(calibration.height) != height:
        raise ValueError(
            "capture contract violation: calibration dimensions "
            f"={(calibration.width, calibration.height)} do not match RGB={(width, height)}"
        )
    if (width, height) != (int(resolution), int(resolution)):
        raise ValueError(
            "capture contract violation: requested resolution "
            f"={resolution} does not match captured RGB={(width, height)}"
        )
    if calibration.camera_name != camera_name:
        raise ValueError(
            "capture contract violation: calibration camera "
            f"{calibration.camera_name!r} is not {camera_name!r}"
        )
    return {
        "valid": True,
        "camera_name": str(calibration.camera_name),
        "depth_conversion_mode": str(getattr(capture, "depth_conversion_mode", "unknown")),
        "depth_sanitization": dict(getattr(capture, "depth_sanitization", None) or {}),
        "requested_resolution": int(resolution),
        "raw_depth_shape": list(normalized_shape),
        "depth_input_shape": list(normalized_shape),
        "rgb_shape": [height, width, 3],
        "normalized_depth_shape": list(normalized_shape),
        "metric_depth_shape": list(metric_shape),
        "calibration_shape": [int(calibration.height), int(calibration.width)],
    }


def _profile_conditions(task_id: int, seed: int, resolution: int) -> dict[str, Any]:
    """Return a stable audit record for the live-validated profile gate."""
    actual = {
        "task_id": int(task_id),
        "seed": int(seed),
        "resolution": int(resolution),
    }
    return {
        "verified": dict(VERIFIED_PROFILE_CONDITIONS),
        "actual": actual,
        "conditions_match": actual == VERIFIED_PROFILE_CONDITIONS,
    }


def validate_workspace_points(
    points: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    bounds: Mapping[str, Sequence[float]] = WORKSPACE_BOUNDS_M,
) -> None:
    """Reject non-finite/out-of-volume controller targets before motion.

    ``bounds`` is deliberately a named, coarse safety volume.  It does not
    establish object identity, collision freedom, or task success.
    """
    items = points.items() if isinstance(points, Mapping) else points
    for name, point in items:
        value = np.asarray(point, dtype=np.float64).reshape(-1)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"workspace validation failed for {name}: expected finite 3D point")
        for axis, coordinate in zip(("x", "y", "z"), value):
            limits = np.asarray(bounds[axis], dtype=np.float64).reshape(-1)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError(f"workspace bounds for {axis} are invalid")
            if coordinate < limits[0] or coordinate > limits[1]:
                raise ValueError(
                    f"workspace validation failed for {name}: {axis}={coordinate:.6f} "
                    f"outside [{limits[0]:.6f}, {limits[1]:.6f}] m"
                )


def _proprioception(observation: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    """Extract only EEF/gripper proprioception needed by the motion loop."""
    if observation is None:
        return {}
    aliases = {
        "eef_pos": ("robot0_eef_pos", "eef_pos"),
        "eef_quat": ("robot0_eef_quat", "eef_quat"),
        "gripper_qpos": ("robot0_gripper_qpos", "gripper_qpos"),
    }
    result = {}
    for output, candidates in aliases.items():
        for key in candidates:
            if key in observation:
                result[output] = np.asarray(observation[key], dtype=np.float64)
                break
    return result


def _position(waypoint: Any) -> np.ndarray:
    if isinstance(waypoint, Mapping):
        for key in ("position", "pos", "eef_pos"):
            if key in waypoint:
                return np.asarray(waypoint[key], dtype=np.float64)
    for key in ("position", "pos", "eef_pos"):
        if hasattr(waypoint, key):
            return np.asarray(getattr(waypoint, key), dtype=np.float64)
    return np.asarray(waypoint, dtype=np.float64)[:3]


def _phase_waypoint(waypoints: Any, phase: str) -> Any:
    if isinstance(waypoints, Mapping):
        if phase in waypoints:
            return waypoints[phase]
        aliases = {"descend_place": "place", "preplace": "pre_place"}
        if aliases.get(phase) in waypoints:
            return waypoints[aliases[phase]]
    if isinstance(waypoints, (np.ndarray, Sequence)) and not isinstance(waypoints, (str, bytes)):
        index = PHASE_WAYPOINT_INDEX[phase]
        if index < len(waypoints):
            return waypoints[index]
    raise ValueError(f"controller did not provide waypoint for phase {phase!r}")


def normalized_action_for_waypoint(
    current_proprio: Mapping[str, np.ndarray], waypoint: Any, *, gripper: float,
    held_rotation: np.ndarray | None = None,
    osc_position_scale_m: float | None = None,
) -> np.ndarray:
    """Produce one finite, seven-dimensional normalized OSC_POSE action."""
    current = current_proprio.get("eef_pos")
    if current is None:
        raise ValueError("EEF position proprioception is required for OSC action generation")
    target = _position(waypoint)
    current_rot = current_proprio.get("eef_quat")
    if current_rot is None:
        raise ValueError("EEF orientation proprioception is required for OSC action generation")
    if held_rotation is None:
        held_rotation = current_rot
    scales = DEFAULT_OSC_SCALES
    if osc_position_scale_m is not None:
        position_scale = float(osc_position_scale_m)
        if not np.isfinite(position_scale) or position_scale <= 0.0:
            raise ValueError("osc_position_scale_m must be finite and positive")
        scales = (position_scale, position_scale, position_scale, *DEFAULT_OSC_SCALES[3:])
    # Waypoints are positions only by design.  Holding the initial EEF
    # orientation makes the bowl transfer a top-down, deterministic primitive.
    action = _require(normalized_osc_action, "normalized_osc_action")(
        current_pos=current[:3],
        current_rot=current_rot,
        target_pos=target[:3],
        target_rot=held_rotation,
        gripper=float(gripper),
        scales=scales,
    )
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (OSC_ACTION_DIM,):
        raise ValueError(f"OSC action must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("OSC action contains non-finite values")
    if np.any(action < OSC_LOW) or np.any(action > OSC_HIGH):
        raise ValueError("OSC action exceeded normalized [-1, 1] bounds")
    return action.astype(np.float32)


def _step_once(env: Any, action: np.ndarray) -> tuple[Mapping[str, Any] | None, bool, dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("LIBERO step must return observation and status")
    observation = result[0] if isinstance(result[0], Mapping) else None
    if len(result) == 4:  # robosuite/LIBERO legacy: obs, reward, done, info
        done = bool(result[2])
        info = result[3] if isinstance(result[3], Mapping) else {}
    else:  # Gymnasium: obs, reward, terminated, truncated, info
        done = bool(result[2]) or bool(result[3])
        info = result[4] if isinstance(result[4], Mapping) else {}
    return observation, done, dict(info)


def _run_motion(
    env: Any,
    waypoints: Any,
    initial_observation: Mapping[str, Any] | None,
    *,
    phase_timeout_steps: int,
    gripper_dwell_steps: int,
    stop_after_phase: str,
    dry_run: bool,
    phase_frame_callback: Callable[[str, int], str | Path | None] | None = None,
    motion_started_callback: Callable[[], None] | None = None,
    stall_window_steps: int = DEFAULT_STALL_WINDOW_STEPS,
    stall_delta_m: float = DEFAULT_STALL_DELTA_M,
    phase_policies: Mapping[str, Mapping[str, Any]] | None = None,
    osc_position_scale_m: float | None = None,
) -> list[dict[str, Any]]:
    if phase_timeout_steps <= 0:
        raise ValueError("phase_timeout_steps must be positive")
    if gripper_dwell_steps <= 0:
        raise ValueError("gripper_dwell_steps must be positive")
    if stop_after_phase not in PHASES:
        raise ValueError(f"stop_after_phase must be one of {PHASES}, got {stop_after_phase!r}")
    if stall_window_steps < 0 or stall_delta_m < 0 or not np.isfinite(stall_delta_m):
        raise ValueError("stall_window_steps and stall_delta_m must be finite and non-negative")
    policies = phase_policies or PHASE_POLICIES
    # Preserve a failure-safe motion marker for batch diagnostics.  A phase can
    # fail during its first simulator step, before an episode audit is written;
    # the matrix runner must still distinguish "motion never began" from a
    # controller/environment failure after an action was sent.
    setattr(env, "_arrow_motion_began", False)
    motion_notified = False
    proprio = _proprioception(initial_observation)
    held_rotation = proprio.get("eef_quat")
    if held_rotation is None:
        raise ValueError("initial observation lacks EEF orientation proprioception")
    phase_audit: list[dict[str, Any]] = []
    # Keep a live reference so a timeout or other controller exception can be
    # serialized by the matrix runner even though no final audit is produced.
    setattr(env, "_arrow_phase_audit", phase_audit)
    for phase in PHASES:
        waypoint = _phase_waypoint(waypoints, phase)
        gripper = 1.0 if phase == "close" else -1.0 if phase == "open" else 0.0
        is_gripper_phase = phase in {"close", "open"}
        phase_tolerance = policies.get(phase, {}).get("tolerance_m", WAYPOINT_POSITION_TOLERANCE_M)
        if phase_tolerance is None:
            phase_tolerance = WAYPOINT_POSITION_TOLERANCE_M
        phase_tolerance = float(phase_tolerance)
        required_steps = gripper_dwell_steps if is_gripper_phase else phase_timeout_steps
        record = {
            "phase": phase,
            "steps": 0,
            "status": "dry_run" if dry_run else "pending",
            "gripper_command": float(gripper),
            "dwell_steps": int(gripper_dwell_steps) if is_gripper_phase else 0,
            "policy": dict(policies.get(phase, {})),
        }
        error_history: list[float] = []
        for step in range(required_steps):
            action = normalized_action_for_waypoint(
                proprio,
                waypoint,
                gripper=gripper,
                held_rotation=held_rotation,
                osc_position_scale_m=osc_position_scale_m,
            )
            record["last_action"] = action.tolist()
            record["steps"] = step + 1
            if dry_run:
                break
            if not motion_notified:
                # Let a batch coordinator durably record the transition to
                # physical motion before the first simulator action is sent.
                if motion_started_callback is not None:
                    motion_started_callback()
                motion_notified = True
            setattr(env, "_arrow_motion_began", True)
            try:
                observation, done, _info = _step_once(env, action)
            except TimeoutError as exc:
                # Preserve a partial gripper record so run_episode can offer a
                # bounded, explicitly-approved RGB-D recovery.  This is the
                # real reachability seam for recovery: a transport/controller
                # timeout during close/open, never an inferred object state.
                if is_gripper_phase:
                    record["status"] = "timeout"
                    record["error_type"] = type(exc).__name__
                    record["error"] = str(exc)
                    phase_audit.append(record)
                    setattr(env, "_arrow_phase_audit", phase_audit)
                raise
            proprio = _proprioception(observation)
            if observation is None or "eef_pos" not in proprio:
                raise RuntimeError(f"phase {phase} lost EEF proprioception; failing closed")
            # LIBERO's ``done`` includes evaluator success.  It is intentionally
            # ignored while executing the bounded state machine: reading it or
            # changing phase based on it would make evaluator state influence
            # motion.  Success is queried only after retreat completes.
            if is_gripper_phase:
                if step + 1 >= gripper_dwell_steps:
                    record["status"] = "dwell"
                    break
            elif np.linalg.norm(_position(waypoint) - proprio["eef_pos"][:3]) < phase_tolerance:
                record["status"] = "reached"
                break
            elif stall_window_steps and "eef_pos" in proprio:
                error = float(np.linalg.norm(_position(waypoint)[:3] - proprio["eef_pos"][:3]))
                error_history.append(error)
                if (
                    len(error_history) >= stall_window_steps
                    and max(error_history[-stall_window_steps:])
                    - min(error_history[-stall_window_steps:]) <= stall_delta_m
                ):
                    record["status"] = "stall"
                    record["stall_window_steps"] = int(stall_window_steps)
                    record["stall_delta_m"] = float(stall_delta_m)
                    record["position_error_norm_m"] = error
                    phase_audit.append(record)
                    setattr(env, "_arrow_phase_audit", phase_audit)
                    raise TimeoutError(f"phase {phase} stalled for {stall_window_steps} steps")
        else:
            record["status"] = "timeout"
            if "eef_pos" in proprio:
                eef_pos = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
                record["eef_pos_m"] = eef_pos.tolist()
                record["position_error_norm_m"] = float(
                    np.linalg.norm(_position(waypoint)[:3] - eef_pos)
                )
            phase_audit.append(record)
            if is_gripper_phase:
                raise TimeoutError(f"phase {phase} failed to complete {gripper_dwell_steps}-step dwell")
            raise TimeoutError(f"phase {phase} exceeded {phase_timeout_steps} steps")
        # Record only robot proprioception observed at phase end.  These
        # diagnostics are not fed back into the controller or evaluator.
        if "eef_pos" in proprio:
            eef_pos = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
            record["eef_pos_m"] = eef_pos.tolist()
            record["position_error_norm_m"] = float(
                np.linalg.norm(_position(waypoint)[:3] - eef_pos)
            )
        if "gripper_qpos" in proprio:
            record["gripper_qpos"] = np.asarray(proprio["gripper_qpos"], dtype=np.float64).tolist()
        if phase == stop_after_phase:
            record["stop_after_phase"] = True
            if dry_run:
                record["status"] = "dry_run_stop"
            else:
                record["status"] = "stop"
        phase_audit.append(record)
        setattr(env, "_arrow_phase_audit", phase_audit)
        if not dry_run and phase_frame_callback is not None:
            try:
                frame_path = phase_frame_callback(phase, len(phase_audit) - 1)
            except Exception as exc:  # diagnostics are best-effort, never control-critical
                record["diagnostic_frame"] = None
                record["diagnostic_frame_error"] = str(exc)
            else:
                if frame_path is not None:
                    record["diagnostic_frame"] = (
                        Path(frame_path).expanduser().resolve().as_posix()
                    )
        if phase == stop_after_phase:
            break
    return phase_audit


def _save_phase_snapshot(
    env: Any,
    output_dir: Path,
    phase_index: int,
    phase: str,
    *,
    width: int,
    height: int,
) -> Path:
    """Save a clean post-phase RGB render without feeding it to control."""
    render = getattr(env, "render", None)
    if render is None:
        # OffScreenRenderEnv exposes camera pixels through its observation
        # helper, not a public render() method.  This is a post-step diagnostic
        # read and does not feed back into the controller.
        rendered, _depth, _observation = _render_pair(env, CAMERA_NAME, width, height)
    else:
        try:
            rendered = render(camera_name=CAMERA_NAME, width=width, height=height, depth=False)
        except TypeError:
            try:
                rendered = render(camera_name=CAMERA_NAME, width=width, height=height)
            except Exception:
                rendered = None
        except Exception:
            rendered = None
        if rendered is None:
            observation = _raw_observation(env)
            rendered = observation.get(f"{CAMERA_NAME}_image") if observation is not None else None
    if isinstance(rendered, tuple):
        rendered = rendered[0]
    if isinstance(rendered, Mapping):
        rendered = rendered.get(f"{CAMERA_NAME}_image")
    if rendered is None:
        raise RuntimeError(f"phase {phase} render returned no RGB image")
    image = _as_rgb(rendered)
    path = output_dir / "phase_frames" / f"{phase_index:02d}_{phase}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(image).save(path)
    return path


def _save_capture(capture: CapturedRGBD, arrow_rgb: np.ndarray, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    depth_mode = str(getattr(capture, "depth_conversion_mode", "unknown"))
    if depth_mode.startswith("normalized"):
        raw_depth_path = output_dir / "agentview_depth_normalized.npy"
    elif depth_mode == "metric":
        raw_depth_path = output_dir / "agentview_depth_metric_input_m.npy"
    else:
        raw_depth_path = output_dir / "agentview_depth_unknown_input.npy"
    paths = {
        "clean_rgb": output_dir / "clean_agentview.png",
        "arrow_rgb": output_dir / "one_arrow_agentview.png",
        "depth_input": raw_depth_path,
        "metric_depth": output_dir / "agentview_depth_metric_m.npy",
    }
    # Keep the historical normalized_depth path only for genuinely normalized
    # producer data; metric input is never mislabeled as normalized.
    if depth_mode.startswith("normalized"):
        paths["normalized_depth"] = raw_depth_path
    elif depth_mode == "metric":
        paths["metric_depth_input"] = raw_depth_path
    Image.fromarray(capture.rgb).save(paths["clean_rgb"])
    Image.fromarray(_as_rgb(arrow_rgb)).save(paths["arrow_rgb"])
    np.save(paths["depth_input"], capture.raw_depth)
    np.save(paths["metric_depth"], capture.metric_depth)
    return {name: path.as_posix() for name, path in paths.items()}


def _persist_recovery_audit(output_dir: Path, recovery_audit: Sequence[Mapping[str, Any]]) -> str | None:
    """Persist bounded-recovery diagnostics without masking the motion error."""
    path = output_dir / "arrow_pick_place_recovery_audit.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(recovery_audit), indent=2), encoding="utf-8")
        return path.as_posix()
    except Exception:
        return None


def recover_grasp_or_release(
    env: Any,
    arrow_rgb: np.ndarray | None,
    *,
    phase: str,
    resolution: int,
    arrow_encoding: str | Any | None = None,
    endpoint_pixels: tuple[Sequence[float], Sequence[float]] | None = None,
    endpoint_depth_statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC,
    endpoint_depth_quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE,
    source_grasp_offset: Sequence[float] = DEFAULT_SOURCE_GRASP_OFFSET_M,
    destination_release_offset: Sequence[float] = DEFAULT_DESTINATION_RELEASE_OFFSET_M,
) -> dict[str, Any]:
    """Re-observe a stalled grasp/release using only RGB-D, arrow pixels, and EEF state.

    This helper deliberately returns a bounded recovery proposal; it never reads
    simulator object poses or evaluator state. A caller may approve the proposal
    through ``recovery_callback`` in :func:`run_episode`.
    """
    if phase not in {"close", "open"}:
        raise ValueError("RGB-D recovery is only defined for close/open phases")
    recapture = capture_agentview(env, resolution=resolution)
    contract = validate_capture_contract(recapture, resolution=resolution)
    if endpoint_pixels is None:
        raise ValueError(
            "recovery requires endpoint_pixels from the original decoded arrow; "
            "re-decoding an old overlay against a new RGB frame is unsafe"
        )
    source_uv = np.asarray(endpoint_pixels[0], dtype=np.float64)
    target_uv = np.asarray(endpoint_pixels[1], dtype=np.float64)
    source_depth, source_depth_audit = _depth_at_with_audit(
        recapture.metric_depth, source_uv,
        statistic=endpoint_depth_statistic,
        quantile=endpoint_depth_quantile,
    )
    target_depth, target_depth_audit = _depth_at_with_audit(
        recapture.metric_depth, target_uv,
        statistic=endpoint_depth_statistic,
        quantile=endpoint_depth_quantile,
    )
    source_point, source_refinement = _refine_or_deproject_endpoint(
        source_uv, source_depth, recapture.calibration.intrinsic,
        recapture.calibration.world_from_camera,
    )
    target_point, target_refinement = _refine_or_deproject_endpoint(
        target_uv, target_depth, recapture.calibration.intrinsic,
        recapture.calibration.world_from_camera,
    )
    proprio = _proprioception(recapture.observation)
    if source_point.shape != (3,) or target_point.shape != (3,):
        raise ValueError("recovery deprojection must return finite 3D points")
    return {
        "phase": phase,
        "capture_contract": contract,
        "arrow_endpoints_uv": {
            "source_tail": source_uv.tolist(), "destination_head": target_uv.tolist()
        },
        "endpoint_depths_m": {
            "source_tail": float(source_depth), "destination_head": float(target_depth)
        },
        "endpoint_depth_statistics": {
            "source_tail": source_depth_audit,
            "destination_head": target_depth_audit,
        },
        "deprojected_visual_endpoint_world_points_m": {
            "source_tail": source_point.tolist(), "destination_head": target_point.tolist()
        },
        "control_targets_world_m": {
            "source_grasp": (source_point + np.asarray(source_grasp_offset, dtype=np.float64)).tolist(),
            "destination_release": (target_point + np.asarray(destination_release_offset, dtype=np.float64)).tolist(),
        },
        "endpoint_refinement": {
            "source_tail": source_refinement,
            "destination_head": target_refinement,
        },
        "eef_pos_m": proprio.get("eef_pos", np.asarray([], dtype=np.float64)).tolist(),
        "eef_quat": proprio.get("eef_quat", np.asarray([], dtype=np.float64)).tolist(),
        "gripper_qpos": proprio.get("gripper_qpos", np.asarray([], dtype=np.float64)).tolist(),
        "source": "rgbd_recapture_arrow_roi_proprioception",
    }


def run_episode(
    *,
    env: Any,
    task_id: int,
    seed: int,
    output_dir: str | Path,
    arrow_rgb: np.ndarray | None = None,
    bboxes: Mapping[str, Sequence[float]] | None = None,
    dry_run: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
    goal_object: str = DEFAULT_GOAL_OBJECT,
    subject: str = DEFAULT_SUBJECT,
    arrow_encoding: str | Any | None = None,
    phase_timeout_steps: int = 80,
    gripper_dwell_steps: int = DEFAULT_GRIPPER_DWELL_STEPS,
    stop_after_phase: str = "retreat",
    source_grasp_offset: Sequence[float] = DEFAULT_SOURCE_GRASP_OFFSET_M,
    destination_release_offset: Sequence[float] = DEFAULT_DESTINATION_RELEASE_OFFSET_M,
    clearance_m: float = 0.08,
    evaluator: Callable[[Any], bool] | None = None,
    capture: CapturedRGBD | None = None,
    allow_unvalidated_profile: bool = False,
    motion_started_callback: Callable[[], None] | None = None,
    controller_variant: ControllerVariantConfig | str | None = None,
    suite_mode: str | None = None,
    recovery_attempts: int = DEFAULT_RECOVERY_ATTEMPTS,
    recovery_steps: int = DEFAULT_RECOVERY_STEPS,
    recovery_callback: Callable[[str, Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Capture, decode, and optionally execute one bounded arrow episode."""
    variant_suite_mode = (
        controller_variant.suite_mode
        if isinstance(controller_variant, ControllerVariantConfig)
        else DEFAULT_SUITE_MODE
    )
    requested_suite_mode = suite_mode or variant_suite_mode
    variant = _resolve_controller_variant(
        controller_variant,
        suite_mode=requested_suite_mode,
        phase_timeout_steps=phase_timeout_steps,
        gripper_dwell_steps=gripper_dwell_steps,
        recovery_attempts=recovery_attempts,
        recovery_steps=recovery_steps,
    )
    suite_mode = requested_suite_mode
    if suite_mode not in SUITE_MODES:
        raise ValueError(f"suite_mode must be one of {SUITE_MODES}, got {suite_mode!r}")
    if variant.suite_mode != suite_mode:
        raise ValueError("controller_variant.suite_mode must match suite_mode")
    phase_timeout_steps = int(variant.phase_timeout_steps)
    gripper_dwell_steps = int(variant.gripper_dwell_steps)
    recovery_attempts = int(variant.recovery_attempts)
    recovery_steps = int(variant.recovery_steps)
    endpoint_depth_statistic = variant.endpoint_depth_statistic
    endpoint_depth_quantile = float(variant.endpoint_depth_quantile)
    max_mask_fraction_for_motion = (
        MAX_NORMALIZED_MASK_FRACTION
        if variant.max_mask_fraction_for_motion is None
        else float(variant.max_mask_fraction_for_motion)
    )
    workspace_bounds = {
        axis: tuple(limits)
        for axis, limits in (
            variant.workspace_bounds_m.items()
            if variant.workspace_bounds_m is not None
            else WORKSPACE_BOUNDS_M.items()
        )
    }
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if gripper_dwell_steps <= 0:
        raise ValueError("gripper_dwell_steps must be positive")
    if stop_after_phase not in PHASES:
        raise ValueError(f"stop_after_phase must be one of {PHASES}, got {stop_after_phase!r}")
    settle_diagnostics = getattr(env, "_arrow_settle_diagnostics", None)
    if not dry_run and (
        not isinstance(settle_diagnostics, Mapping)
        or not bool(settle_diagnostics.get("settled", False))
    ):
        raise RuntimeError("refusing motion: LIBERO physics was not confirmed settled")
    np.random.seed(seed)
    capture = capture if capture is not None else capture_agentview(env, resolution=resolution)
    # A caller-supplied capture is treated exactly like a live capture: its
    # actual pixels and calibration must agree with the requested agentview
    # contract before profile validation or any possible motion.
    capture_contract = validate_capture_contract(capture, resolution=resolution)
    setattr(env, "_arrow_capture_contract", dict(capture_contract))
    profile = _profile_conditions(task_id, seed, capture_contract["rgb_shape"][1])
    # Task/seed come from the seeded LIBERO episode setup; camera and
    # resolution are taken from the validated capture, never inferred solely
    # from CLI arguments.
    profile["verified"]["camera_name"] = CAMERA_NAME
    profile["actual"]["camera_name"] = capture_contract["camera_name"]
    if not dry_run and not allow_unvalidated_profile and not profile["conditions_match"]:
        raise RuntimeError(
            "refusing motion outside the verified LIBERO arrow profile "
            f"(expected task={VERIFIED_PROFILE_TASK_ID}, seed={VERIFIED_PROFILE_SEED}, "
            f"resolution={VERIFIED_PROFILE_RESOLUTION}; got task={task_id}, seed={seed}, "
            f"resolution={capture_contract['rgb_shape'][1]}); pass "
            "allow_unvalidated_profile=True only after review"
        )
    if arrow_rgb is None:
        if bboxes is None:
            raise ValueError("provide arrow_rgb or input-generation bboxes")
        arrow_rgb, arrow_audit = render_exactly_one_arrow(
            capture.rgb,
            bboxes,
            subject=subject,
            goal_object=goal_object,
            anchor_policy=variant.arrow_anchor_policy,
        )
    else:
        arrow_rgb = _as_rgb(arrow_rgb)
        if arrow_rgb.shape != capture.rgb.shape:
            raise ValueError("clean RGB and arrow RGB must have identical shape")
        arrow_audit = {"controller_input": "caller_supplied_one_arrow"}
    arrow_audit["encoding"] = getattr(arrow_encoding, "version", arrow_encoding or "legacy")

    # Controller receives no bboxes, names, evaluator, or env handle.
    source_uv, target_uv = decode_arrow_pixels(
        capture.rgb, arrow_rgb, encoding=arrow_encoding
    )
    arrow_decode_audit: dict[str, Any] = {"source": "decode_arrow"}
    if decode_arrow_diagnostics is not None:
        try:
            decoded_diagnostics = decode_arrow_diagnostics(
                capture.rgb, arrow_rgb, encoding=arrow_encoding
            )
            arrow_decode_audit.update({
                "success": bool(
                    getattr(
                        decoded_diagnostics,
                        "ok",
                        getattr(decoded_diagnostics, "success", False),
                    )
                ),
                "reason": getattr(decoded_diagnostics, "reason", None),
                "encoding": getattr(decoded_diagnostics, "encoding_version", None),
                "changed_pixel_count": getattr(decoded_diagnostics, "changed_pixel_count", None),
                "component_count": getattr(decoded_diagnostics, "component_count", None),
            })
        except Exception as exc:
            # Diagnostics must not change the established decoder contract.
            arrow_decode_audit.update({"success": None, "error": str(exc)})
    # Publish visual-input diagnostics immediately after decoding.  These are
    # provenance only: the controller still receives clean RGB, one-arrow RGB,
    # aligned metric depth, and calibration, never bboxes or scene-graph data.
    setattr(env, "_arrow_endpoints_uv", {
        "source_tail": np.asarray(source_uv, dtype=np.float64).tolist(),
        "destination_head": np.asarray(target_uv, dtype=np.float64).tolist(),
    })
    setattr(env, "_arrow_decode_audit", dict(arrow_decode_audit))
    setattr(env, "_arrow_input_arrow_audit", dict(arrow_audit))
    depth_sanitization_policy = assess_depth_sanitization_for_motion(
        capture, (source_uv, target_uv), max_mask_fraction=max_mask_fraction_for_motion
    )
    setattr(env, "_arrow_depth_sanitization_policy", dict(depth_sanitization_policy))
    if not dry_run and depth_sanitization_policy["status"] == "rejected":
        raise RuntimeError(
            "refusing motion: normalized depth sanitization policy rejected capture "
            f"({depth_sanitization_policy['rejection_reason']})"
        )
    source_depth, source_depth_audit = _depth_at_with_audit(
        capture.metric_depth, source_uv,
        statistic=endpoint_depth_statistic,
        quantile=endpoint_depth_quantile,
    )
    target_depth, target_depth_audit = _depth_at_with_audit(
        capture.metric_depth, target_uv,
        statistic=endpoint_depth_statistic,
        quantile=endpoint_depth_quantile,
    )
    setattr(env, "_arrow_endpoint_depths_m", {
        "source_tail": float(source_depth),
        "destination_head": float(target_depth),
    })
    setattr(env, "_arrow_endpoint_depth_statistics", {
        "source_tail": dict(source_depth_audit),
        "destination_head": dict(target_depth_audit),
    })
    calibration = asdict(capture.calibration)
    K = capture.calibration.intrinsic
    T_world_camera = capture.calibration.world_from_camera
    # Deproject with the exact robust scalar depth audited below.  Passing the
    # whole depth image would allow a controller/helper to silently choose a
    # different neighborhood than the one used for provenance.
    source_visual_point, source_refinement = _refine_or_deproject_endpoint(
        source_uv, source_depth, K, T_world_camera
    )
    destination_visual_point, destination_refinement = _refine_or_deproject_endpoint(
        target_uv, target_depth, K, T_world_camera
    )
    if source_visual_point.shape != (3,) or destination_visual_point.shape != (3,):
        raise ValueError("deproject_endpoint must return finite 3D points")
    source_offset = np.asarray(source_grasp_offset, dtype=np.float64).reshape(-1)
    destination_offset = np.asarray(destination_release_offset, dtype=np.float64).reshape(-1)
    if source_offset.shape != (3,) or destination_offset.shape != (3,):
        raise ValueError("source_grasp_offset and destination_release_offset must each have 3 values")
    if not np.isfinite(source_offset).all() or not np.isfinite(destination_offset).all():
        raise ValueError("source/destination offsets must be finite")
    source_offset_overridden = not np.allclose(source_offset, DEFAULT_SOURCE_GRASP_OFFSET_M, atol=1e-12)
    destination_offset_overridden = not np.allclose(
        destination_offset, DEFAULT_DESTINATION_RELEASE_OFFSET_M, atol=1e-12
    )
    bowl_point = source_visual_point + source_offset
    destination_point = destination_visual_point + destination_offset
    setattr(env, "_arrow_deprojected_visual_endpoint_world_points_m", {
        "source_tail": source_visual_point.tolist(),
        "destination_head": destination_visual_point.tolist(),
    })
    setattr(env, "_arrow_control_targets_world_m", {
        "source_grasp": bowl_point.tolist(),
        "destination_release": destination_point.tolist(),
    })
    if not np.isfinite(source_visual_point).all() or not np.isfinite(destination_visual_point).all():
        raise ValueError("deproject_endpoint returned non-finite visual endpoint")
    workspace_validation = {
        "status": "not_run_dry_run" if dry_run else "pending",
        "kind": "coarse_finite_volume_only",
        "bounds_m": {axis: list(limits) for axis, limits in WORKSPACE_BOUNDS_M.items()},
        "points": {
            "source_grasp_target": bowl_point.tolist(),
            "destination_release_target": destination_point.tolist(),
        },
    }
    setattr(env, "_arrow_workspace_validation", dict(workspace_validation))
    initial_proprio = _proprioception(capture.observation)
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    waypoints = _require(build_bowl_waypoints, "build_bowl_waypoints")(
        bowl_point, destination_point,
        initial_proprio.get("eef_quat"),
        {"lift_height_m": float(clearance_m)},
    )
    waypoints = _apply_directional_pregrasp_offset(
        waypoints,
        source_visual_point,
        destination_visual_point,
        variant.approach_lateral_offset_m,
    )
    if isinstance(waypoints, Mapping):
        waypoint_points = {
            f"waypoint_{name}": _position(value)[:3] for name, value in waypoints.items()
        }
    else:
        try:
            waypoint_values = list(waypoints)
        except TypeError as exc:
            raise ValueError("controller waypoints must be a sequence or mapping") from exc
        waypoint_points = {
            f"waypoint_{index}": _position(value)[:3]
            for index, value in enumerate(waypoint_values)
        }
    workspace_validation["points"].update(
        {name: np.asarray(point, dtype=np.float64).tolist() for name, point in waypoint_points.items()}
    )
    workspace_validation["bounds_m"] = {
        axis: list(limits) for axis, limits in workspace_bounds.items()
    }
    setattr(env, "_arrow_workspace_validation", dict(workspace_validation))
    if not dry_run:
        validate_workspace_points(
            {
                "source_grasp_target": bowl_point,
                "destination_release_target": destination_point,
                **waypoint_points,
            },
            bounds=workspace_bounds,
        )
        workspace_validation["status"] = "passed"
    setattr(env, "_arrow_workspace_validation", dict(workspace_validation))
    setattr(env, "_arrow_waypoints_world_m", {
        name: np.asarray(point, dtype=np.float64).tolist()
        for name, point in waypoint_points.items()
    })

    phase_policies = {name: dict(policy) for name, policy in PHASE_POLICIES.items()}
    if variant.approach_tolerance_m is not None:
        for approach_phase in ("descend", "descend_place"):
            phase_policies[approach_phase]["tolerance_m"] = float(variant.approach_tolerance_m)
    if variant.waypoint_tolerance_m is not None:
        for positional_phase in PHASES:
            if positional_phase not in {"close", "open"}:
                phase_policies[positional_phase]["tolerance_m"] = float(
                    variant.waypoint_tolerance_m
                )

    output_root = Path(output_dir).expanduser().resolve()
    frame_paths = _save_capture(capture, arrow_rgb, output_root)
    phase_frame_paths: list[str] = []

    def phase_frame_callback(phase: str, phase_index: int) -> Path:
        path = _save_phase_snapshot(
            env,
            output_root,
            phase_index,
            phase,
            width=capture.rgb.shape[1],
            height=capture.rgb.shape[0],
        )
        phase_frame_paths.append(path.as_posix())
        return path

    recovery_audit: list[dict[str, Any]] = []
    try:
        phase_audit = _run_motion(
            env,
            waypoints,
            capture.observation,
            phase_timeout_steps=phase_timeout_steps,
            gripper_dwell_steps=gripper_dwell_steps,
            stop_after_phase=stop_after_phase,
            dry_run=dry_run,
            phase_frame_callback=phase_frame_callback if not dry_run else None,
            motion_started_callback=motion_started_callback if not dry_run else None,
            stall_window_steps=variant.stall_window_steps,
            stall_delta_m=variant.stall_delta_m,
            phase_policies=phase_policies,
            osc_position_scale_m=variant.osc_position_scale_m,
        )
    except TimeoutError as exc:
        if (
            not dry_run
            and recovery_attempts > 0
            and isinstance(getattr(env, "_arrow_phase_audit", None), list)
            and env._arrow_phase_audit
            and env._arrow_phase_audit[-1].get("phase") in {"close", "open"}
        ):
            phase = str(env._arrow_phase_audit[-1]["phase"])
            for attempt in range(int(recovery_attempts)):
                proposal: dict[str, Any] | None = None
                approved = False
                try:
                    # Endpoint pixels are retained from the original decode.
                    # Never decode the original overlay against this new clean
                    # frame: that could silently choose a different command.
                    proposal = recover_grasp_or_release(
                        env,
                        arrow_rgb,
                        phase=phase,
                        resolution=resolution,
                        arrow_encoding=arrow_encoding,
                        endpoint_pixels=(source_uv, target_uv),
                        endpoint_depth_statistic=endpoint_depth_statistic,
                        endpoint_depth_quantile=endpoint_depth_quantile,
                        source_grasp_offset=source_offset,
                        destination_release_offset=destination_offset,
                    )
                    proposal["attempt"] = attempt + 1
                    approved = bool(recovery_callback(phase, proposal)) if recovery_callback else False
                    proposal["approved"] = approved
                    proposal["attempted_steps"] = 0
                    proposal["executed_steps"] = 0
                    # Every recovery action is explicitly opt-in.  A proposal
                    # or callback is not permission by itself; only a truthy
                    # callback result may send bounded gripper commands.
                    if approved and recovery_steps > 0:
                        current_pos = np.asarray(proposal.get("eef_pos_m", []), dtype=np.float64)
                        current_quat = np.asarray(proposal.get("eef_quat", []), dtype=np.float64)
                        if current_pos.shape == (3,) and current_quat.size:
                            recovery_proprio = {"eef_pos": current_pos, "eef_quat": current_quat}
                            recovery_waypoint = {"position": current_pos}
                            recovery_gripper = 1.0 if phase == "close" else -1.0
                            for _ in range(recovery_steps):
                                action = normalized_action_for_waypoint(
                                    recovery_proprio, recovery_waypoint,
                                    gripper=recovery_gripper, held_rotation=current_quat,
                                )
                                setattr(env, "_arrow_motion_began", True)
                                if motion_started_callback is not None:
                                    motion_started_callback()
                                proposal["attempted_steps"] += 1
                                observation, _done, _info = _step_once(env, action)
                                recovery_proprio = _proprioception(observation)
                                if "eef_pos" not in recovery_proprio or "eef_quat" not in recovery_proprio:
                                    break
                                proposal["executed_steps"] += 1
                    recovery_audit.append(proposal)
                except Exception as recovery_exc:
                    # Preserve the proposal (including approval and any
                    # completed steps) when an approved action fails.  Do not
                    # replace it with a misleading zero-step error record.
                    if proposal is None:
                        proposal = {"phase": phase, "attempt": attempt + 1}
                    proposal.setdefault("approved", approved)
                    proposal.setdefault("attempted_steps", 0)
                    proposal.setdefault("executed_steps", 0)
                    proposal["error_type"] = type(recovery_exc).__name__
                    proposal["error"] = str(recovery_exc)
                    recovery_audit.append(proposal)
                finally:
                    # Make the diagnostic durable even when recapture, callback,
                    # action construction, or an action step fails.
                    setattr(env, "_arrow_recovery_audit", recovery_audit)
                    _persist_recovery_audit(output_root, recovery_audit)
                if recovery_audit and recovery_audit[-1].get("approved"):
                    break
        raise exc
    # Evaluator state is intentionally queried only after all motion phases.
    full_execution = stop_after_phase == "retreat"
    evaluator_error: dict[str, str] | None = None
    if dry_run or not full_execution or evaluator is None:
        success = None
    else:
        try:
            success = bool(evaluator(env))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            success = None
            evaluator_error = {"type": type(exc).__name__, "error": str(exc)}
    audit = {
        "schema_version": "arrow_pick_place_mvp.v1",
        "task_id": int(task_id),
        "seed": int(seed),
        "camera": CAMERA_NAME,
        "resolution": [capture.rgb.shape[1], capture.rgb.shape[0]],
        "dry_run": bool(dry_run),
        "motion_executed": not bool(dry_run),
        "stop_after_phase": stop_after_phase,
        "full_state_machine_completed": bool(full_execution and not dry_run),
        "gripper_dwell_steps": int(gripper_dwell_steps),
        "controller_variant": variant.provenance(),
        "phase_policies": phase_policies,
        "suite_mode": suite_mode,
        "recovery": recovery_audit,
        "settle_diagnostics": settle_diagnostics,
        "environment_audit": getattr(env, "_arrow_environment_audit", None),
        "capture_contract": capture_contract,
        "depth_sanitization_policy": depth_sanitization_policy,
        "workspace_bounds_m": {axis: list(limits) for axis, limits in workspace_bounds.items()},
        "profile": {
            "name": DEFAULT_PROFILE_NAME,
            "verified_conditions": profile["verified"],
            "actual_conditions": profile["actual"],
            "conditions_match": bool(profile["conditions_match"]),
            "allow_unvalidated_profile": bool(allow_unvalidated_profile),
        },
        "profile_validated": bool(profile["conditions_match"]),
        "allow_unvalidated_profile": bool(allow_unvalidated_profile),
        "controller_input": "pixels_depth_calibration",
        "arrow": arrow_audit,
        "arrow_decode_diagnostics": arrow_decode_audit,
        "arrow_endpoints_uv": {"source_tail": source_uv.tolist(), "destination_head": target_uv.tolist()},
        "endpoint_depths_m": {
            "source_tail": float(source_depth),
            "destination_head": float(target_depth),
        },
        "endpoint_depth_statistics": {
            "source_tail": source_depth_audit,
            "destination_head": target_depth_audit,
        },
        "endpoint_refinement": {
            "source_tail": source_refinement,
            "destination_head": destination_refinement,
        },
        "deprojected_visual_endpoint_world_points_m": {
            "source_tail": source_visual_point.tolist(),
            "destination_head": destination_visual_point.tolist(),
        },
        "control_targets_world_m": {
            "source_grasp": bowl_point.tolist(),
            "destination_release": destination_point.tolist(),
        },
        "waypoints_world_m": np.asarray(waypoints, dtype=np.float64).tolist(),
        "source_grasp_offset_m": source_offset.tolist(),
        "destination_release_offset_m": destination_offset.tolist(),
        "offset_profile": DEFAULT_PROFILE_NAME,
        "offsets_overridden": {
            "source_grasp": bool(source_offset_overridden),
            "destination_release": bool(destination_offset_overridden),
        },
        "clearance_m": float(clearance_m),
        "workspace_validation": workspace_validation,
        "calibration": calibration,
        "frame_orientation": {
            "pixel_origin": "top_left",
            "rgb_depth": "aligned_same_capture",
            "intrinsic_vertical_conversion": "K_image[1,1]=-K_raw[1,1]; K_image[1,2]=H-K_raw[1,2]",
            "world_from_camera": "robosuite_camera_utils_with_axis_correction",
        },
        "frames": frame_paths,
        "phase_frames": phase_frame_paths,
        "phase_frame_errors": {
            record["phase"]: record["diagnostic_frame_error"]
            for record in phase_audit
            if "diagnostic_frame_error" in record
        },
        "phases": phase_audit,
        "evaluator_success": success,
        "evaluator_error": evaluator_error,
        "evaluator_read_after_action": bool(not dry_run and full_execution),
        "timestamp_unix": time.time(),
    }
    audit_path = output_root / "arrow_pick_place_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["audit_path"] = audit_path.as_posix()
    if evaluator_error is not None:
        raise RuntimeError(f"evaluator failure: {evaluator_error['error']}")
    return audit


def _apply_direct_swaps(inner_env: Any, task_id: int) -> dict[str, Any]:
    """Apply configured distractor swaps and return explicit audit evidence."""
    from config import TASK_SWAP_CONFIG

    requested = [list(item) for item in TASK_SWAP_CONFIG.get(int(task_id), [])]
    if not requested:
        return {"requested": [], "applied": [], "skipped": []}
    try:
        from preview_visual_arrows import _apply_task_swaps

        _apply_task_swaps(inner_env, int(task_id))
        return {
            "requested": requested,
            "applied": [label for pair in requested for label in pair],
            "skipped": [],
            "source": "preview_visual_arrows._apply_task_swaps",
        }
    except ImportError:
        from radomize_scenes import swap_objects

        applied: list[str] = []
        skipped: list[Any] = []
        for left, right in requested:
            result = swap_objects(inner_env, left, right, verbose=False)
            applied.extend(result.get("applied", []))
            skipped.extend(result.get("skipped", []))
        if skipped or sorted(applied) != sorted(label for pair in requested for label in pair):
            raise RuntimeError(f"configured swaps were not fully applied: {skipped}")
        return {"requested": requested, "applied": applied, "skipped": skipped, "source": "radomize_scenes.swap_objects"}


def _randomization_dimensions(
    suite_mode: str,
    removals: Sequence[str],
    swaps: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Report task-specific transformations that were actually applied."""
    applied_swaps = list((swaps or {}).get("applied", []))
    return {
        "scene_layout": bool(suite_mode == "sealed_randomized" and applied_swaps),
        "object_removal": bool(suite_mode == "sealed_randomized" and removals),
        "prompt_variant": False,
    }


def build_libero_env(
    task_id: int,
    seed: int,
    resolution: int,
    *,
    suite_mode: str | None = None,
    controller_variant: ControllerVariantConfig | str | None = None,
) -> Any:
    """Construct direct LIBERO OffScreenRenderEnv with explicit suite semantics."""
    if suite_mode is None:
        suite_mode = (
            controller_variant.suite_mode
            if isinstance(controller_variant, ControllerVariantConfig)
            else DEFAULT_SUITE_MODE
        )
    if suite_mode not in SUITE_MODES:
        raise ValueError(f"suite_mode must be one of {SUITE_MODES}, got {suite_mode!r}")
    if isinstance(controller_variant, ControllerVariantConfig) and controller_variant.suite_mode != suite_mode:
        raise ValueError("controller_variant.suite_mode must match suite_mode")
    variant = _resolve_controller_variant(controller_variant, suite_mode=suite_mode)
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("LIBERO and robosuite are required for live execution") from exc
    from config import BENCHMARK_NAME

    suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    canonical_bddl_file = suite.get_task_bddl_file_path(int(task_id))
    removals: list[str] = []
    bddl_file = canonical_bddl_file
    if suite_mode == "sealed_randomized":
        from config import TASK_REMOVE_CONFIG
        from bddl_utils import make_filtered_bddl

        removals = list(TASK_REMOVE_CONFIG.get(int(task_id), []))
        bddl_file = make_filtered_bddl(canonical_bddl_file, removals)
    np.random.seed(seed)
    # MuJoCo/robosuite owns a process-global offscreen rendering context.  Do
    # not construct the canonical schema env after the live filtered env: its
    # close() can tear down or reuse the live renderer, yielding black RGB and
    # uninitialized depth only in the sealed condition.  Extract the canonical
    # joint schema first, then close it before creating the env that will render.
    canonical_source_schema = None
    if suite_mode == "sealed_randomized":
        canonical_schema_env = None
        try:
            from bddl_utils import extract_joint_schema

            canonical_schema_env = OffScreenRenderEnv(
                bddl_file_name=canonical_bddl_file,
                camera_names=[CAMERA_NAME],
                camera_heights=int(resolution),
                camera_widths=int(resolution),
                camera_depths=True,
                controller=variant.controller,
            )
            canonical_schema_env.reset()
            canonical_source_schema = extract_joint_schema(canonical_schema_env.sim.model)
        finally:
            if canonical_schema_env is not None:
                _close_environment_quietly(canonical_schema_env)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[CAMERA_NAME],
        camera_heights=int(resolution),
        camera_widths=int(resolution),
        camera_depths=True,
        controller=variant.controller,
    )
    # Match render_visual_arrow_pair.py's seed/init-state setup while avoiding
    # LeRobot's terminal autoreset.  The direct wrapper owns the same init-state
    # files and lets us preserve the terminal observation through retreat.
    try:
        seed_fn = getattr(env, "seed", None)
        if seed_fn is not None:
            seed_fn(int(seed))
        env.reset()
    except BaseException:
        _close_environment_quietly(env)
        raise
    environment_audit: dict[str, Any] = {
        "suite_mode": suite_mode,
        # The sealed treatment changes only the scene/BDDL realization.  This
        # direct runner does not consume language prompts, so prompt provenance
        # is deliberately explicit rather than implying a prompt treatment.
        "scene_randomization": "sealed_randomized" if suite_mode == "sealed_randomized" else "vanilla",
        "randomization_dimensions": _randomization_dimensions(suite_mode, removals),
        "prompt_provenance": "not_applicable_direct_runner",
        "canonical_bddl_file": str(canonical_bddl_file),
        "applied_bddl_file": str(bddl_file),
        "requested_removals": removals,
        "applied_removals": list(removals) if suite_mode == "sealed_randomized" else [],
        "removal_source": "bddl_filter" if suite_mode == "sealed_randomized" else None,
        "requested_swaps": [],
        "applied_swaps": [],
        "skipped_swaps": [],
    }
    init_state_diagnostics = {
        "source": "seeded_reset_fallback",
        "available_count": None,
        "selected_index": None,
        "fallback": True,
    }
    try:
        from lerobot.envs.libero import get_task_init_states

        init_states = get_task_init_states(suite, int(task_id))
        available_count = len(init_states)
        init_state_diagnostics.update({
            "source": "lerobot.get_task_init_states",
            "available_count": int(available_count),
        })
        if available_count:
            selected_index = int(seed) % int(available_count)
            selected_state = init_states[selected_index]
            projection = {"required": False, "projected": False, "source": "canonical_task_init_states"}
            if suite_mode == "sealed_randomized":
                from bddl_utils import extract_joint_schema, project_init_states_by_joint_name

                if canonical_source_schema is None:
                    raise RuntimeError("canonical init-state schema was not prepared")
                source_schema = canonical_source_schema
                target_schema = extract_joint_schema(env.sim.model)
                projected_states = project_init_states_by_joint_name(
                    np.asarray(init_states), source_schema, target_schema
                )
                selected_state = projected_states[selected_index]
                projection = {
                    "required": True,
                    "projected": True,
                    "source": "bddl_utils.project_init_states_by_joint_name",
                    "canonical_nq": source_schema.nq,
                    "filtered_nq": target_schema.nq,
                    "canonical_nv": source_schema.nv,
                    "filtered_nv": target_schema.nv,
                }
            env.set_init_state(selected_state)
            init_state_diagnostics.update({
                "selected_index": int(selected_index),
                "fallback": False,
                "projection": projection,
            })
    except (ImportError, FileNotFoundError, AttributeError, TypeError) as exc:
        # Some lightweight LIBERO installs omit LeRobot's torch loader; the
        # seeded reset remains a valid fallback and is recorded by the caller.
        if suite_mode == "sealed_randomized":
            _close_environment_quietly(env)
            raise RuntimeError(
                "sealed_randomized mode requires canonical init states and "
                "joint-name projection; refusing seeded fallback"
            ) from exc
        pass
    except BaseException:
        # Any unexpected init-state/projection failure occurs before the
        # caller receives ownership of env; close it before propagating.
        _close_environment_quietly(env)
        raise
    if suite_mode == "sealed_randomized":
        try:
            swaps = _apply_direct_swaps(getattr(env, "env", env), int(task_id))
        except BaseException:
            _close_environment_quietly(env)
            raise
        environment_audit.update({
            "requested_swaps": swaps.get("requested", []),
            "applied_swaps": swaps.get("applied", []),
            "skipped_swaps": swaps.get("skipped", []),
            "swap_source": swaps.get("source"),
        })
        environment_audit["randomization_dimensions"] = _randomization_dimensions(
            suite_mode, removals, swaps
        )
    setattr(env, "_arrow_init_state_diagnostics", init_state_diagnostics)
    try:
        from radomize_scenes import sim_state_sha256

        environment_audit["state_hash_sha256_pre_settle"] = sim_state_sha256(env.sim)
    except Exception as exc:
        environment_audit["state_hash_sha256_pre_settle"] = None
        environment_audit["state_hash_pre_settle_error"] = str(exc)
    from config import SETTLE_STEPS_INIT

    try:
        settle_libero_env(env, max_steps=SETTLE_STEPS_INIT)
    except BaseException:
        _close_environment_quietly(env)
        raise
    try:
        from radomize_scenes import sim_state_sha256

        environment_audit["state_hash_sha256"] = sim_state_sha256(env.sim)
    except Exception as exc:
        environment_audit["state_hash_sha256"] = None
        environment_audit["state_hash_error"] = str(exc)
    environment_audit["settle_diagnostics"] = getattr(env, "_arrow_settle_diagnostics", None)
    setattr(env, "_arrow_environment_audit", environment_audit)
    return env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output-dir", type=Path, default=Path("arrow_pick_place_outputs"))
    parser.add_argument("--phase-timeout-steps", type=int, default=80)
    parser.add_argument("--gripper-dwell-steps", type=int, default=DEFAULT_GRIPPER_DWELL_STEPS)
    parser.add_argument("--stall-window-steps", type=int, default=DEFAULT_STALL_WINDOW_STEPS)
    parser.add_argument("--stall-delta-m", type=float, default=DEFAULT_STALL_DELTA_M)
    parser.add_argument("--recovery-attempts", type=int, default=DEFAULT_RECOVERY_ATTEMPTS)
    parser.add_argument("--recovery-steps", type=int, default=DEFAULT_RECOVERY_STEPS)
    parser.add_argument(
        "--controller-variant",
        choices=(
            "default",
            CANDIDATE_CONTROLLER_VARIANT_NAME,
            CANDIDATE_V2_CONTROLLER_VARIANT_NAME,
            CANDIDATE_V3_CONTROLLER_VARIANT_NAME,
            CANDIDATE_V4_CONTROLLER_VARIANT_NAME,
            CANDIDATE_V5_CONTROLLER_VARIANT_NAME,
            CANDIDATE_V6_CONTROLLER_VARIANT_NAME,
        ),
        default="default",
    )
    parser.add_argument(
        "--endpoint-depth-statistic",
        choices=ENDPOINT_DEPTH_STATISTICS,
        default=None,
    )
    parser.add_argument("--endpoint-depth-quantile", type=float, default=None)
    parser.add_argument("--approach-tolerance-m", type=float, default=None)
    parser.add_argument("--suite-mode", choices=SUITE_MODES, default=DEFAULT_SUITE_MODE)
    parser.add_argument("--stop-after-phase", choices=PHASES, default="retreat")
    parser.add_argument("--source-grasp-offset", type=float, nargs=3, default=DEFAULT_SOURCE_GRASP_OFFSET_M, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--destination-release-offset", type=float, nargs=3, default=DEFAULT_DESTINATION_RELEASE_OFFSET_M, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument(
        "--allow-unvalidated-profile",
        action="store_true",
        help="allow motion outside task=0/seed=1000/resolution=256 after manual review",
    )
    parser.add_argument("--dry-run", "--no-motion", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--execute-motion", dest="dry_run", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_variant = args.controller_variant in {
        CANDIDATE_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V2_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V3_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V4_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V5_CONTROLLER_VARIANT_NAME,
        CANDIDATE_V6_CONTROLLER_VARIANT_NAME,
    }
    candidate_v2 = args.controller_variant == CANDIDATE_V2_CONTROLLER_VARIANT_NAME
    candidate_v3 = args.controller_variant == CANDIDATE_V3_CONTROLLER_VARIANT_NAME
    candidate_v4 = args.controller_variant == CANDIDATE_V4_CONTROLLER_VARIANT_NAME
    candidate_v5 = args.controller_variant == CANDIDATE_V5_CONTROLLER_VARIANT_NAME
    candidate_v6 = args.controller_variant == CANDIDATE_V6_CONTROLLER_VARIANT_NAME
    variant_name = args.controller_variant if candidate_variant else DEFAULT_PROFILE_NAME
    endpoint_depth_statistic = args.endpoint_depth_statistic or (
        CANDIDATE_ENDPOINT_DEPTH_STATISTIC if candidate_variant else DEFAULT_ENDPOINT_DEPTH_STATISTIC
    )
    endpoint_depth_quantile = args.endpoint_depth_quantile
    if endpoint_depth_quantile is None:
        endpoint_depth_quantile = (
            CANDIDATE_ENDPOINT_DEPTH_QUANTILE if candidate_variant else DEFAULT_ENDPOINT_DEPTH_QUANTILE
        )
    approach_tolerance_m = args.approach_tolerance_m
    if approach_tolerance_m is None and candidate_variant:
        approach_tolerance_m = CANDIDATE_APPROACH_TOLERANCE_M
    waypoint_tolerance_m = (
        CANDIDATE_V3_WAYPOINT_TOLERANCE_M if candidate_v3 else None
    )
    max_mask_fraction_for_motion = (
        CANDIDATE_V2_MAX_MASK_FRACTION_FOR_MOTION
        if candidate_v2 or candidate_v3 or candidate_v4 or candidate_v5 or candidate_v6
        else None
    )
    workspace_bounds_m = (
        CANDIDATE_V2_WORKSPACE_BOUNDS_M
        if candidate_v2 or candidate_v3 or candidate_v4 or candidate_v5 or candidate_v6
        else None
    )
    osc_position_scale_m = CANDIDATE_V4_OSC_POSITION_SCALE_M if candidate_v4 else None
    arrow_anchor_policy = "visible_inset" if candidate_v5 else "bbox_center"
    approach_lateral_offset_m = (
        CANDIDATE_V6_APPROACH_LATERAL_OFFSET_M if candidate_v6 else None
    )
    env = build_libero_env(
        args.task, args.seed, args.resolution, suite_mode=args.suite_mode
    )
    try:
        # Input-generation integration is intentionally explicit: a caller can
        # replace this with a human/annotated arrow PNG while the controller
        # contract remains unchanged.  The built-in path uses GT bboxes only
        # to create the one-arrow input, never for control.
        from libero_live_semantic_context import LiveSemanticContextGenerator
        from config import SCENE_GRAPH_SUBJECT_FILTER, TASK_GOAL_OBJECT_CONFIG
        from types import SimpleNamespace

        generator = LiveSemanticContextGenerator()
        generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
        capture = capture_agentview(env, resolution=args.resolution)
        # The semantic helper is input-generation-only and expects the small
        # LeRobot wrapper surface.  It reads bboxes for rendering the arrow;
        # none of this namespace is passed to the controller.
        task_text = getattr(env, "language_instruction", "")
        context_env = SimpleNamespace(
            _env=env.env,
            observation_height=args.resolution,
            observation_width=args.resolution,
            task=task_text,
            task_id=args.task,
        )
        context = generator.observe_visual_graph(context_env, camera=CAMERA_NAME)
        goal = TASK_GOAL_OBJECT_CONFIG.get(args.task, DEFAULT_GOAL_OBJECT)
        result = run_episode(
            env=env,
            task_id=args.task,
            seed=args.seed,
            output_dir=args.output_dir,
            bboxes=context["bboxes"],
            dry_run=args.dry_run,
            resolution=args.resolution,
            goal_object=goal,
            subject=SCENE_GRAPH_SUBJECT_FILTER,
            phase_timeout_steps=args.phase_timeout_steps,
            gripper_dwell_steps=args.gripper_dwell_steps,
            recovery_attempts=args.recovery_attempts,
            recovery_steps=args.recovery_steps,
            controller_variant=ControllerVariantConfig(
                name=variant_name,
                suite_mode=args.suite_mode,
                phase_timeout_steps=args.phase_timeout_steps,
                gripper_dwell_steps=args.gripper_dwell_steps,
                stall_window_steps=args.stall_window_steps,
                stall_delta_m=args.stall_delta_m,
                recovery_attempts=args.recovery_attempts,
                recovery_steps=args.recovery_steps,
                endpoint_depth_statistic=endpoint_depth_statistic,
                endpoint_depth_quantile=endpoint_depth_quantile,
                approach_tolerance_m=approach_tolerance_m,
                waypoint_tolerance_m=waypoint_tolerance_m,
                osc_position_scale_m=osc_position_scale_m,
                arrow_anchor_policy=arrow_anchor_policy,
                approach_lateral_offset_m=approach_lateral_offset_m,
                max_mask_fraction_for_motion=max_mask_fraction_for_motion,
                workspace_bounds_m=workspace_bounds_m,
            ),
            stop_after_phase=args.stop_after_phase,
            source_grasp_offset=args.source_grasp_offset,
            destination_release_offset=args.destination_release_offset,
            clearance_m=args.clearance_m,
            allow_unvalidated_profile=args.allow_unvalidated_profile,
            capture=capture,
            evaluator=lambda candidate: bool(candidate.check_success()),
        )
        print(json.dumps({"audit_path": result["audit_path"], "success": result["evaluator_success"]}))
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
