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
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
        arrow_world_xy_basis,
        derive_rgbd_region_grasp_candidates,
        assess_grasp_retention,
        derive_rgbd_source_approach_candidates,
        estimate_destination_support_plane,
        release_point_on_support_plane,
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
            arrow_world_xy_basis,
            derive_rgbd_region_grasp_candidates,
            assess_grasp_retention,
            derive_rgbd_source_approach_candidates,
            estimate_destination_support_plane,
            release_point_on_support_plane,
        )
    except ImportError:  # pragma: no cover - a clear error is raised on use
        build_bowl_waypoints = decode_arrow = decode_arrow_diagnostics = None
        deproject_endpoint = refine_rgbd_endpoint = normalized_osc_action = None
        arrow_world_xy_basis = None
        derive_rgbd_region_grasp_candidates = None
        assess_grasp_retention = None
        derive_rgbd_source_approach_candidates = None
        estimate_destination_support_plane = None
        release_point_on_support_plane = None

try:
    from .controller_configs import (
        ACTIVE_CONTROLLER_CONFIG_FILENAME,
        ACTIVE_CONTROLLER_NAME,
        ControllerConfigError,
        controller_config_hash,
        load_controller_config,
    )
except (ImportError, ValueError):  # pragma: no cover - direct script fallback
    from controller_configs import (
        ACTIVE_CONTROLLER_CONFIG_FILENAME,
        ACTIVE_CONTROLLER_NAME,
        ControllerConfigError,
        controller_config_hash,
        load_controller_config,
    )

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
DEFAULT_PROFILE_NAME = ACTIVE_CONTROLLER_NAME
DEFAULT_CONTROLLER_CONFIG_FILENAME = ACTIVE_CONTROLLER_CONFIG_FILENAME
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
DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.12
SUITE_MODES = ("vanilla", "sealed_randomized")
DEFAULT_SUITE_MODE = "vanilla"
DEFAULT_STALL_WINDOW_STEPS = 10
DEFAULT_STALL_DELTA_M = 1e-4
DEFAULT_RECOVERY_ATTEMPTS = 1
DEFAULT_RECOVERY_STEPS = 3
DEFAULT_MOTION_TRACE_MAX_STEPS = 512
DEFAULT_CANARY_VIDEO_MAX_FRAMES = 512
MAX_NORMALIZED_MASK_FRACTION = 0.25
ENDPOINT_DEPTH_STATISTICS = ("median", "lower_quantile", "nearest_valid")
DEFAULT_ENDPOINT_DEPTH_STATISTIC = "median"
DEFAULT_ENDPOINT_DEPTH_QUANTILE = 0.25
ARROW_ANCHOR_POLICIES = ("bbox_center", "visible_inset")
MAX_APPROACH_TOLERANCE_M = 0.025
MAX_MASK_FRACTION_FOR_MOTION = 0.50
MAX_OSC_POSITION_SCALE_M = 0.10


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or (minimum is not None and number < minimum):
        suffix = f" >= {minimum}" if minimum is not None else " finite"
        raise ValueError(f"{name} must be{suffix}")
    return number


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class GraspSearchPolicy:
    """Bounded, arrow-derived source search policy.

    Offsets are ``(forward, lateral, vertical)`` metres in the world frame.
    They are interpreted against the visual arrow basis by the runner; no
    object or simulator information is accepted here.
    """

    enabled: bool = False
    strategy: str = "fixed_offsets"
    empty_gripper_threshold: float = 0.0015
    offsets_m: Sequence[Sequence[float]] = ()
    region_radius_m: float = 0.075
    region_depth_tolerance_m: float = 0.025
    region_min_pixels: int = 48
    region_max_fraction: float = 0.15
    region_height_quantile: float = 0.70
    region_profile_quantiles: Sequence[float] = (0.80, 0.60, 0.40)
    region_candidate_height_quantiles: Sequence[float] = ()
    region_seed_radius_px: int = 3
    max_attempts: int = 0
    phase_timeout_steps: int = 80
    max_actions: int = 900
    trigger_on: Sequence[str] = (
        "empty_gripper_likely", "close_stall", "close_timeout", "lift_stall", "lift_timeout"
    )

    def __post_init__(self) -> None:
        enabled = _strict_bool(self.enabled, "grasp_search.enabled")
        strategy = str(self.strategy)
        if strategy not in {"fixed_offsets", "rgbd_region"}:
            raise ValueError("grasp_search.strategy must be fixed_offsets or rgbd_region")
        threshold = _finite_number(self.empty_gripper_threshold, "grasp_search.empty_gripper_threshold")
        if threshold <= 0.0 or threshold > 0.01:
            raise ValueError("grasp_search.empty_gripper_threshold must be in (0, 0.01]")
        max_attempts = _strict_int(self.max_attempts, "grasp_search.max_attempts")
        phase_steps = _strict_int(self.phase_timeout_steps, "grasp_search.phase_timeout_steps")
        max_actions = _strict_int(self.max_actions, "grasp_search.max_actions")
        if max_attempts < 0 or phase_steps <= 0 or max_actions < 0:
            raise ValueError("grasp_search attempt/step limits are invalid")
        offsets = tuple(tuple(_finite_number(v, "grasp_search.offset") for v in row) for row in self.offsets_m)
        if any(len(row) != 3 or any(abs(v) > 0.05 for v in row) for row in offsets):
            raise ValueError("grasp_search.offsets_m must contain 3-vectors within +/-0.05 m")
        triggers = tuple(str(item) for item in self.trigger_on)
        allowed = {
            "empty_gripper_likely", "close_stall", "close_timeout", "lift_stall",
            "lift_timeout", "post_lift_retention",
        }
        if not set(triggers) <= allowed:
            raise ValueError(f"grasp_search.trigger_on must contain only {sorted(allowed)}")
        region_radius = _finite_number(self.region_radius_m, "grasp_search.region_radius_m")
        region_depth_tolerance = _finite_number(
            self.region_depth_tolerance_m, "grasp_search.region_depth_tolerance_m"
        )
        region_min_pixels = _strict_int(self.region_min_pixels, "grasp_search.region_min_pixels")
        region_max_fraction = _finite_number(
            self.region_max_fraction, "grasp_search.region_max_fraction"
        )
        region_height_quantile = _finite_number(
            self.region_height_quantile, "grasp_search.region_height_quantile"
        )
        region_profile_quantiles = tuple(
            _finite_number(value, "grasp_search.region_profile_quantile")
            for value in self.region_profile_quantiles
        )
        region_candidate_height_quantiles = tuple(
            _finite_number(value, "grasp_search.region_candidate_height_quantile")
            for value in self.region_candidate_height_quantiles
        )
        region_seed_radius = _strict_int(
            self.region_seed_radius_px, "grasp_search.region_seed_radius_px"
        )
        if not 0.01 <= region_radius <= 0.15:
            raise ValueError("grasp_search.region_radius_m must be in [0.01, 0.15]")
        if not 0.001 <= region_depth_tolerance <= 0.10:
            raise ValueError("grasp_search.region_depth_tolerance_m must be in [0.001, 0.10]")
        if region_min_pixels < 4:
            raise ValueError("grasp_search.region_min_pixels must be >= 4")
        if not 0.0 < region_max_fraction <= 0.5:
            raise ValueError("grasp_search.region_max_fraction must be in (0, 0.5]")
        if not 0.0 < region_height_quantile < 1.0:
            raise ValueError("grasp_search.region_height_quantile must be in (0, 1)")
        if not region_profile_quantiles or len(region_profile_quantiles) > 8 or any(
            not 0.0 < value < 1.0 for value in region_profile_quantiles
        ):
            raise ValueError(
                "grasp_search.region_profile_quantiles must contain one to eight values in (0, 1)"
            )
        if region_candidate_height_quantiles and (
            len(region_candidate_height_quantiles) != len(region_profile_quantiles)
            or any(not 0.0 < value < 1.0 for value in region_candidate_height_quantiles)
        ):
            raise ValueError(
                "grasp_search.region_candidate_height_quantiles must be empty or match "
                "region_profile_quantiles with values in (0, 1)"
            )
        if not 1 <= region_seed_radius <= 12:
            raise ValueError("grasp_search.region_seed_radius_px must be in [1, 12]")
        object.__setattr__(self, "empty_gripper_threshold", threshold)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "offsets_m", offsets)
        object.__setattr__(self, "max_attempts", max_attempts)
        object.__setattr__(self, "phase_timeout_steps", phase_steps)
        object.__setattr__(self, "max_actions", max_actions)
        object.__setattr__(self, "trigger_on", triggers)
        object.__setattr__(self, "region_radius_m", region_radius)
        object.__setattr__(self, "region_depth_tolerance_m", region_depth_tolerance)
        object.__setattr__(self, "region_min_pixels", region_min_pixels)
        object.__setattr__(self, "region_max_fraction", region_max_fraction)
        object.__setattr__(self, "region_height_quantile", region_height_quantile)
        object.__setattr__(self, "region_profile_quantiles", region_profile_quantiles)
        object.__setattr__(
            self, "region_candidate_height_quantiles", region_candidate_height_quantiles
        )
        object.__setattr__(self, "region_seed_radius_px", region_seed_radius)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "GraspSearchPolicy" | None) -> "GraspSearchPolicy | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("grasp_search must be an object")
        allowed = {field.name for field in dataclass_fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown grasp_search keys: {sorted(unknown)}")
        return cls(**dict(value))

    def canonical(self) -> dict[str, Any]:
        result = {
            "enabled": bool(self.enabled),
            "strategy": self.strategy,
            "empty_gripper_threshold": float(self.empty_gripper_threshold),
            "offsets_m": [[float(v) for v in row] for row in self.offsets_m],
            "region_radius_m": float(self.region_radius_m),
            "region_depth_tolerance_m": float(self.region_depth_tolerance_m),
            "region_min_pixels": int(self.region_min_pixels),
            "region_max_fraction": float(self.region_max_fraction),
            "region_height_quantile": float(self.region_height_quantile),
            "region_profile_quantiles": [float(value) for value in self.region_profile_quantiles],
            "region_candidate_height_quantiles": [
                float(value) for value in self.region_candidate_height_quantiles
            ],
            "region_seed_radius_px": int(self.region_seed_radius_px),
            "max_attempts": int(self.max_attempts),
            "phase_timeout_steps": int(self.phase_timeout_steps),
            "max_actions": int(self.max_actions),
            "trigger_on": list(self.trigger_on),
        }
        return result


@dataclass(frozen=True)
class MicroCorrectionPolicy:
    """Finite residual plateau correction policy for configured phases."""

    enabled: bool = False
    phases: Sequence[str] = ("descend", "lift", "preplace", "descend_place")
    plateau_window_steps: int = 10
    plateau_delta_m: float = 0.0001
    residual_max_m: float = 0.005
    correction_gain: float = 1.0
    burst_steps: int = 8
    max_rounds: int = 2
    max_actions: int = 16

    def __post_init__(self) -> None:
        enabled = _strict_bool(self.enabled, "micro_correction.enabled")
        phases = tuple(str(item) for item in self.phases)
        if not set(phases) <= set(PHASES) - {"close", "open"}:
            raise ValueError("micro_correction.phases must name positional phases")
        window = _strict_int(self.plateau_window_steps, "micro_correction.plateau_window_steps")
        burst = _strict_int(self.burst_steps, "micro_correction.burst_steps")
        rounds = _strict_int(self.max_rounds, "micro_correction.max_rounds")
        actions = _strict_int(self.max_actions, "micro_correction.max_actions")
        if window <= 0 or burst <= 0 or rounds < 0 or actions < 0:
            raise ValueError("micro_correction step limits are invalid")
        delta = _finite_number(self.plateau_delta_m, "micro_correction.plateau_delta_m", minimum=0.0)
        residual = _finite_number(self.residual_max_m, "micro_correction.residual_max_m")
        gain = _finite_number(self.correction_gain, "micro_correction.correction_gain")
        if residual <= 0.0 or gain <= 0.0:
            raise ValueError("micro_correction residual_max_m and correction_gain must be positive")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "plateau_window_steps", window)
        object.__setattr__(self, "plateau_delta_m", delta)
        object.__setattr__(self, "residual_max_m", residual)
        object.__setattr__(self, "correction_gain", gain)
        object.__setattr__(self, "burst_steps", burst)
        object.__setattr__(self, "max_rounds", rounds)
        object.__setattr__(self, "max_actions", actions)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "MicroCorrectionPolicy" | None) -> "MicroCorrectionPolicy | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("micro_correction must be an object")
        allowed = {field.name for field in dataclass_fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown micro_correction keys: {sorted(unknown)}")
        return cls(**dict(value))

    def canonical(self) -> dict[str, Any]:
        result = {
            "enabled": bool(self.enabled), "phases": list(self.phases),
            "plateau_window_steps": int(self.plateau_window_steps),
            "plateau_delta_m": float(self.plateau_delta_m),
            "residual_max_m": float(self.residual_max_m),
            "correction_gain": float(self.correction_gain),
            "burst_steps": int(self.burst_steps), "max_rounds": int(self.max_rounds),
            "max_actions": int(self.max_actions),
        }
        return result


def dataclass_fields(cls: type[Any]) -> tuple[Any, ...]:
    """Local wrapper keeps policy parsing import-light and testable."""
    from dataclasses import fields
    return fields(cls)


def _optional_rgbd_policy(value: Mapping[str, Any] | None, name: str, allowed: set[str]) -> dict[str, Any] | None:
    """Validate an additive RGB-D policy without changing legacy defaults."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
    result = dict(value)
    if type(result.get("enabled")) is not bool:
        raise ValueError(f"{name}.enabled must be a boolean")
    if name == "source_approach":
        offsets = result.get("offsets_arrow_frame_m")
        if not isinstance(offsets, list) or not offsets or len(offsets) > 8:
            raise ValueError(f"{name}.offsets_arrow_frame_m must contain 1-8 offsets")
        normalized: list[list[float]] = []
        for row in offsets:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{name}.offsets_arrow_frame_m entries must be 3-vectors")
            values = [float(item) for item in row]
            if not np.isfinite(values).all() or any(abs(item) > 0.04 for item in values):
                raise ValueError(f"{name}.offsets_arrow_frame_m entries exceed +/-0.04 m")
            normalized.append(values)
        result["offsets_arrow_frame_m"] = normalized
        radius = int(result.get("patch_radius_px", 2))
        fraction = float(result.get("min_valid_fraction", 0.5))
        if radius < 1 or radius > 8 or not 0.0 < fraction <= 1.0:
            raise ValueError(f"{name} depth support settings are invalid")
        result["patch_radius_px"] = radius
        result["min_valid_fraction"] = fraction
    else:
        radius = int(result.get("patch_radius_px", 3))
        fraction = float(result.get("min_valid_fraction", 0.5))
        residual = float(result.get("max_residual_m", 0.01))
        release_clearance = float(result.get("release_clearance_m", 0.015))
        if radius < 1 or radius > 8 or not 0.0 < fraction <= 1.0 or not np.isfinite(residual) or residual <= 0.0 or residual > 0.05 or not np.isfinite(release_clearance) or not 0.003 <= release_clearance <= 0.08:
            raise ValueError(f"{name} depth support settings are invalid")
        result.update({"patch_radius_px": radius, "min_valid_fraction": fraction, "max_residual_m": residual, "release_clearance_m": release_clearance})
    return result


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
    grasp_contact_threshold: float | None = None
    grasp_retry_offsets_m: Sequence[float] | None = None
    max_mask_fraction_for_motion: float | None = None
    workspace_bounds_m: Mapping[str, Sequence[float]] | None = None
    grasp_search: GraspSearchPolicy | Mapping[str, Any] | None = None
    micro_correction: MicroCorrectionPolicy | Mapping[str, Any] | None = None
    source_approach: Mapping[str, Any] | None = None
    destination_placement: Mapping[str, Any] | None = None
    config_source: str | None = None
    external_config_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasp_search", GraspSearchPolicy.from_value(self.grasp_search))
        object.__setattr__(self, "micro_correction", MicroCorrectionPolicy.from_value(self.micro_correction))
        object.__setattr__(self, "source_approach", _optional_rgbd_policy(
            self.source_approach, "source_approach", {"enabled", "offsets_arrow_frame_m", "patch_radius_px", "min_valid_fraction"}
        ))
        object.__setattr__(self, "destination_placement", _optional_rgbd_policy(
            self.destination_placement, "destination_placement", {"enabled", "patch_radius_px", "min_valid_fraction", "max_residual_m", "release_clearance_m"}
        ))
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
        if self.waypoint_tolerance_m is not None and (
            not np.isfinite(float(self.waypoint_tolerance_m))
            or float(self.waypoint_tolerance_m) <= 0.0
            or float(self.waypoint_tolerance_m) > MAX_APPROACH_TOLERANCE_M
        ):
            raise ValueError(
                f"waypoint_tolerance_m must be in (0, {MAX_APPROACH_TOLERANCE_M}]"
            )
        if self.osc_position_scale_m is not None and (
            not np.isfinite(float(self.osc_position_scale_m))
            or float(self.osc_position_scale_m) <= 0.0
            or float(self.osc_position_scale_m) > MAX_OSC_POSITION_SCALE_M
        ):
            raise ValueError(
                f"osc_position_scale_m must be in (0, {MAX_OSC_POSITION_SCALE_M}]"
            )
        if self.arrow_anchor_policy not in ARROW_ANCHOR_POLICIES:
            raise ValueError(
                f"arrow_anchor_policy must be one of {ARROW_ANCHOR_POLICIES}"
            )
        if self.approach_lateral_offset_m is not None and (
            not np.isfinite(float(self.approach_lateral_offset_m))
            or float(self.approach_lateral_offset_m) <= 0.0
            or float(self.approach_lateral_offset_m) > 0.10
        ):
            raise ValueError("approach_lateral_offset_m must be in (0, 0.10]")
        if self.grasp_contact_threshold is not None and (
            not np.isfinite(float(self.grasp_contact_threshold))
            or float(self.grasp_contact_threshold) <= 0.0
            or float(self.grasp_contact_threshold) > 0.01
        ):
            raise ValueError("grasp_contact_threshold must be in (0, 0.01]")
        if self.grasp_retry_offsets_m is not None:
            offsets = tuple(float(value) for value in self.grasp_retry_offsets_m)
            if len(offsets) > 3 or any(
                not np.isfinite(value) or abs(value) > 0.03 for value in offsets
            ):
                raise ValueError("grasp_retry_offsets_m must contain at most three offsets within +/-0.03 m")
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
        result = {
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
            "grasp_contact_threshold": (
                None
                if self.grasp_contact_threshold is None
                else float(self.grasp_contact_threshold)
            ),
            "grasp_retry_offsets_m": (
                None
                if self.grasp_retry_offsets_m is None
                else [float(value) for value in self.grasp_retry_offsets_m]
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
        # Keep the historical v1--v8 canonical payload byte-for-byte stable;
        # nested policies are semantic additions only for external v9 configs.
        if self.grasp_search is not None:
            result["grasp_search"] = self.grasp_search.canonical()
        if self.micro_correction is not None:
            result["micro_correction"] = self.micro_correction.canonical()
        if self.source_approach is not None:
            result["source_approach"] = json.loads(json.dumps(self.source_approach, sort_keys=True))
        if self.destination_placement is not None:
            result["destination_placement"] = json.loads(json.dumps(self.destination_placement, sort_keys=True))
        return result

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
        result = {"canonical": self.canonical(), "config_hash": self.config_hash}
        if self.config_source is not None:
            result["config_source"] = self.config_source
        if self.external_config_hash is not None:
            result["external_config_hash"] = self.external_config_hash
        return result


def controller_variant_from_config(
    config: Mapping[str, Any], *, suite_mode: str = DEFAULT_SUITE_MODE
) -> ControllerVariantConfig:
    """Build a validated runtime variant from a fully expanded JSON mapping."""
    if not isinstance(config, Mapping):
        raise ControllerConfigError("expanded controller config must be an object")
    data = dict(config)
    source = data.pop("config_source", None)
    external_hash = data.pop("config_hash", None)
    declared_suite = data.pop("suite_mode", None)
    if declared_suite is not None and str(declared_suite) != suite_mode:
        raise ControllerConfigError(
            f"controller config suite_mode {declared_suite!r} does not match {suite_mode!r}"
        )
    allowed = {
        "name", "controller", "phase_timeout_steps", "gripper_dwell_steps",
        "stall_window_steps", "stall_delta_m", "recovery_attempts", "recovery_steps",
        "endpoint_depth_statistic", "endpoint_depth_quantile", "approach_tolerance_m",
        "waypoint_tolerance_m", "osc_position_scale_m", "arrow_anchor_policy",
        "approach_lateral_offset_m", "grasp_contact_threshold", "grasp_retry_offsets_m",
        "max_mask_fraction_for_motion", "workspace_bounds_m", "grasp_search",
        "micro_correction",
        "source_approach", "destination_placement",
    }
    retired_keys = {
        "grasp_provider", "placement_provider", "zerograsp", "zerograsp_policy"
    }
    if retired_keys.intersection(data):
        raise ControllerConfigError(
            "ZeroGrasp is a retired side experiment and cannot be selected by "
            "the active arrow runtime"
        )
    unknown = set(data) - allowed
    if unknown:
        raise ControllerConfigError(f"unknown controller config keys: {sorted(unknown)}")
    data["suite_mode"] = suite_mode
    data["config_source"] = source
    data["external_config_hash"] = external_hash
    return ControllerVariantConfig(**data)


def _resolve_controller_variant(
    value: ControllerVariantConfig | str | None,
    *,
    suite_mode: str,
    phase_timeout_steps: int = 80,
    gripper_dwell_steps: int = DEFAULT_GRIPPER_DWELL_STEPS,
    stall_window_steps: int = DEFAULT_STALL_WINDOW_STEPS,
    stall_delta_m: float = DEFAULT_STALL_DELTA_M,
    recovery_attempts: int = DEFAULT_RECOVERY_ATTEMPTS,
    recovery_steps: int = DEFAULT_RECOVERY_STEPS,
    endpoint_depth_statistic: str = DEFAULT_ENDPOINT_DEPTH_STATISTIC,
    endpoint_depth_quantile: float = DEFAULT_ENDPOINT_DEPTH_QUANTILE,
    approach_tolerance_m: float | None = None,
    max_mask_fraction_for_motion: float | None = None,
    workspace_bounds_m: Mapping[str, Sequence[float]] | None = None,
) -> ControllerVariantConfig:
    def require_active(variant: ControllerVariantConfig) -> ControllerVariantConfig:
        if variant.name != DEFAULT_PROFILE_NAME:
            raise ControllerConfigError(
                "only the active v9d controller is executable; "
                f"got retired controller {variant.name!r}"
            )
        expected = controller_variant_from_config(
            load_controller_config(), suite_mode=suite_mode
        )
        if variant.canonical() != expected.canonical():
            raise ControllerConfigError(
                "active v9d controller payload does not match the checked-in policy"
            )
        return variant

    if isinstance(value, ControllerVariantConfig):
        return require_active(value)
    if isinstance(value, Mapping):
        return require_active(controller_variant_from_config(value, suite_mode=suite_mode))
    if value is None:
        # The active v9d document is the single source of truth for the
        # default.  Loading it here keeps nested RGB-D policy defaults and
        # provenance identical for direct episodes and matrix runs.
        return require_active(controller_variant_from_config(
            load_controller_config(), suite_mode=suite_mode
        ))
    name = str(value)
    if name in {"default", DEFAULT_PROFILE_NAME}:
        return require_active(controller_variant_from_config(
            load_controller_config(DEFAULT_CONTROLLER_CONFIG_FILENAME),
            suite_mode=suite_mode,
        ))
    if name.endswith(".json") or "/" in name or "\\" in name:
        return require_active(controller_variant_from_config(
            load_controller_config(name), suite_mode=suite_mode
        ))
    raise ControllerConfigError(
        "only the active v9d controller is executable; "
        f"got retired controller {name!r}"
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
    if isinstance(waypoints, Mapping):
        adjusted = dict(waypoints)
        first_key = "pregrasp" if "pregrasp" in adjusted else 0
        first = adjusted.get(first_key)
        if first is None:
            return waypoints
        first_position = _position(first).copy()
        first_position[:2] += (direction / norm) * offset
        if isinstance(first, Mapping):
            replacement = dict(first)
            replacement["position"] = first_position
        else:
            replacement = _pose_waypoint(first_position, _orientation(first), _orientation_frame(first))
        adjusted[first_key] = replacement
        return adjusted
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


def _pose_waypoint(position: Any, orientation: Any | None = None, orientation_frame: str | None = None) -> dict[str, Any]:
    result = {"position": np.asarray(position, dtype=np.float64).reshape(3)}
    if orientation is not None:
        result["orientation"] = np.asarray(orientation, dtype=np.float64)
    if orientation_frame is not None:
        result["orientation_frame"] = str(orientation_frame)
    return result


def _experimental_candidate_pose(candidate: Any) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None, dict[str, Any]]:
    """Validate the model/geometry candidate seam without importing its package.

    Experimental candidates are deliberately duck-typed so the frozen runner
    remains independent of MolmoPoint/SAM3.  The position is the measured
    ``grip_site`` target (not the legacy v9d source anchor); the orientation is
    a world-frame grip-site rotation and the aperture is audit-only because
    Panda gripper commands remain signed incremental actions.
    """
    position = getattr(candidate, "grip_site_world_m", None)
    if position is None:
        position = getattr(candidate, "position_world_m", None)
    if position is None and isinstance(candidate, Mapping):
        position = candidate.get("grip_site_world_m", candidate.get("position_world_m"))
    orientation = getattr(candidate, "rotation_world_grip_site", None)
    if orientation is None:
        orientation = getattr(candidate, "orientation_matrix", None)
    if orientation is None:
        quaternion = getattr(candidate, "orientation_xyzw", None)
        if quaternion is not None:
            qx, qy, qz, qw = np.asarray(quaternion, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm((qx, qy, qz, qw)))
            if norm > 1e-9:
                qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
                orientation = np.asarray((
                    (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
                    (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
                    (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
                ), dtype=np.float64)
    if orientation is None and isinstance(candidate, Mapping):
        orientation = candidate.get("rotation_world_grip_site", candidate.get("orientation_matrix"))
    opening = getattr(candidate, "required_aperture_m", None)
    if opening is None:
        opening = getattr(candidate, "opening_m", None)
    if opening is None and isinstance(candidate, Mapping):
        opening = candidate.get("required_aperture_m", candidate.get("opening_m"))
    if position is None or orientation is None or opening is None:
        raise ValueError("experimental candidate requires grip_site_world_m, rotation_world_grip_site, and required_aperture_m")
    point = np.asarray(position, dtype=np.float64).reshape(-1)
    pregrasp = getattr(candidate, "pregrasp_world_m", None)
    if pregrasp is None and isinstance(candidate, Mapping):
        pregrasp = candidate.get("pregrasp_world_m")
    rotation = np.asarray(orientation, dtype=np.float64)
    aperture = float(opening)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("experimental candidate grip-site position must be a finite world 3-vector")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("experimental candidate orientation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("experimental candidate orientation must be a proper SO(3) matrix")
    if not np.isfinite(aperture) or not 0.0 < aperture <= 0.20:
        raise ValueError("experimental candidate aperture must be in (0, 0.20] metres")
    if pregrasp is not None:
        pregrasp = np.asarray(pregrasp, dtype=np.float64).reshape(-1)
        if pregrasp.shape != (3,) or not np.isfinite(pregrasp).all():
            raise ValueError("experimental candidate pregrasp must be a finite world 3-vector")
    audit = getattr(candidate, "audit", {})
    if not audit:
        audit = getattr(candidate, "metadata", {})
    if not isinstance(audit, Mapping):
        audit = {}
    return point, rotation, aperture, pregrasp, dict(audit)


def _serialize_waypoints(waypoints: Any) -> list[Any] | dict[str, Any]:
    values = waypoints.items() if isinstance(waypoints, Mapping) else enumerate(waypoints)
    result: dict[str, Any] = {}
    for key, waypoint in values:
        item: dict[str, Any] = {"position": _position(waypoint)[:3].tolist()}
        orientation = _orientation(waypoint)
        if orientation is not None:
            item["orientation"] = orientation.tolist()
        frame = _orientation_frame(waypoint)
        if frame is not None:
            item["orientation_frame"] = str(frame)
        result[str(key)] = item
    return result if isinstance(waypoints, Mapping) else [result[str(index)] for index in range(len(result))]


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


def _gripper_contact_likely(
    phase_audit: Sequence[Mapping[str, Any]], threshold: float
) -> bool:
    """Use gripper proprioception only to detect a likely empty close."""
    close = next(
        (record for record in reversed(phase_audit) if record.get("phase") == "close"),
        None,
    )
    if close is None or "gripper_qpos" not in close:
        return False
    qpos = np.asarray(close["gripper_qpos"], dtype=np.float64).reshape(-1)
    return bool(qpos.size and np.isfinite(qpos).all() and np.max(np.abs(qpos)) <= float(threshold))


def _empty_gripper_likely(
    phase_audit: Sequence[Mapping[str, Any]], threshold: float
) -> bool:
    """Proprioception-only empty-gripper diagnostic used by v9 policies."""
    return _gripper_contact_likely(phase_audit, threshold)


def _post_lift_retention_decision(
    trace: Sequence[Mapping[str, Any]],
    current_proprio: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Classify lift retention from bounded gripper proprioception only.

    The trace is populated by the motion loop and contains no object/contact
    or evaluator state.  Missing or malformed qpos is explicitly
    ``unobservable`` so callers cannot silently turn absent evidence into a
    retry or a success.
    """
    samples: list[np.ndarray] = []
    for item in trace:
        if not isinstance(item, Mapping) or item.get("phase") != "lift":
            continue
        values = item.get("gripper_qpos")
        try:
            array = np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if array.size and np.isfinite(array).all():
            samples.append(array)
    try:
        current = np.asarray(current_proprio.get("gripper_qpos"), dtype=np.float64).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        current = np.asarray([], dtype=np.float64)
    if current.size and np.isfinite(current).all():
        if not samples or samples[-1].shape == current.shape:
            samples.append(current)
    if not samples or assess_grasp_retention is None:
        return {
            "decision": "unobservable",
            "retained": False,
            "sample_count": len(samples),
            "threshold": float(threshold),
        }
    try:
        evidence = assess_grasp_retention(
            np.vstack(samples),
            closed_threshold=float(threshold),
            min_samples=1,
            min_closed_fraction=0.80,
        )
    except (TypeError, ValueError):
        return {
            "decision": "unobservable",
            "retained": False,
            "sample_count": len(samples),
            "threshold": float(threshold),
        }
    final_closed = bool(evidence.final_abs_qpos.size and np.max(evidence.final_abs_qpos) >= float(threshold))
    retained = bool(evidence.retained and final_closed)
    return {
        "decision": "retained" if retained else "not_retained",
        "retained": retained,
        "sample_count": int(len(samples)),
        "closed_fraction": float(evidence.closed_fraction),
        "final_abs_qpos": evidence.final_abs_qpos.tolist(),
        "threshold": float(threshold),
        "source": "lift_proprioception_trace",
    }


def _observed_gripper_qpos(
    phase_audit: Sequence[Mapping[str, Any]],
) -> list[float] | None:
    for record in reversed(phase_audit):
        if record.get("phase") == "close" and "gripper_qpos" in record:
            values = np.asarray(record["gripper_qpos"], dtype=np.float64).reshape(-1)
            if values.size and np.isfinite(values).all():
                return values.tolist()
    return None


class _GraspSearchRequested(RuntimeError):
    """Internal control-flow marker; no simulator state is attached."""

    def __init__(self, trigger: str):
        super().__init__(trigger)
        self.trigger = trigger


class ControllerMotionTimeout(TimeoutError):
    """Timeout/stall raised by this controller, distinct from env failures."""


@dataclass
class _ActionBudget:
    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, int(self.limit) - int(self.used))

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.used += 1
        return True


def _diagnostic_array(value: Any, *, limit: int | None = None) -> list[float] | None:
    """Serialize a small numeric diagnostic without exposing simulator state."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if limit is not None:
        array = array[:limit]
    if not array.size or not np.isfinite(array).all():
        return None
    return array.tolist()


def _motion_trace_entry(
    phase: str,
    step: int,
    proprio: Mapping[str, Any] | None,
    target: Any,
    *,
    action_sent: bool,
    correction_active: bool,
    experimental_motion_diagnostics: bool = False,
    nominal_target: Any | None = None,
    commanded_target: Any | None = None,
    action: Any | None = None,
    before_proprio: Mapping[str, Any] | None = None,
    env: Any | None = None,
) -> dict[str, Any]:
    """Build one bounded, proprioception-only motion trace record."""
    observed = _proprioception(proprio)
    eef_pos = _diagnostic_array(observed.get("eef_pos"), limit=3)
    target_pos = _diagnostic_array(_position(target), limit=3)
    residual = None
    residual_norm = None
    if eef_pos is not None and target_pos is not None and len(eef_pos) == 3:
        residual_array = np.asarray(target_pos, dtype=np.float64) - np.asarray(eef_pos, dtype=np.float64)
        residual = residual_array.tolist()
        residual_norm = float(np.linalg.norm(residual_array))
    result = {
        "phase": str(phase),
        "step": int(step),
        "action_sent": bool(action_sent),
        "eef_pos_m": eef_pos,
        "eef_quat_xyzw": _diagnostic_array(observed.get("eef_quat"), limit=4),
        "gripper_qpos": _diagnostic_array(observed.get("gripper_qpos")),
        "target_position_m": target_pos,
        "residual_vector_m": residual,
        "residual_norm_m": residual_norm,
        "correction_active": bool(correction_active),
    }
    if experimental_motion_diagnostics:
        before = _proprioception(before_proprio)
        result.update({
            "nominal_target_position_m": _diagnostic_array(_position(nominal_target), limit=3) if nominal_target is not None else None,
            "commanded_target_position_m": _diagnostic_array(_position(commanded_target), limit=3) if commanded_target is not None else None,
            "action_normalized": _diagnostic_array(action),
            "eef_pos_before_m": _diagnostic_array(before.get("eef_pos"), limit=3),
            "eef_pos_after_m": eef_pos,
            "controller_telemetry": _experimental_controller_telemetry(env),
        })
    return result


def _experimental_controller_telemetry(env: Any | None) -> dict[str, Any]:
    """Read optional robot/controller telemetry without touching simulator state."""
    unavailable = {"status": "unavailable", "reason": "not_exposed"}
    sources: list[Any] = []
    if env is not None:
        sources.append(env)
        for name in ("controller", "robot", "robots"):
            try:
                value = getattr(env, name, None)
            except Exception:
                value = None
            if isinstance(value, (list, tuple)):
                sources.extend(value[:2])
            elif value is not None:
                sources.append(value)
        # Robosuite robots expose the controller below each robot wrapper.
        for source in tuple(sources):
            try:
                controller = getattr(source, "controller", None)
            except Exception:
                controller = None
            if controller is not None:
                sources.append(controller)

    def lookup(names: Sequence[str]) -> list[float] | dict[str, str]:
        for source in sources:
            for name in names:
                try:
                    value = getattr(source, name, None)
                except Exception:
                    value = None
                converted = _diagnostic_array(value)
                if converted is not None:
                    return converted
        return dict(unavailable)

    return {
        "controller_ee_pos_m": lookup(("ee_pos", "eef_pos", "ee_position", "eef_position")),
        "controller_goal_pos_m": lookup(("goal_pos", "goal_position", "ee_goal_pos", "eef_goal_pos")),
        "requested_torque": lookup(("requested_torque", "commanded_torque", "torque_command", "torque")),
        "applied_torque": lookup(("applied_torque", "applied_joint_torque", "torque_applied")),
        "eef_force": lookup(("eef_force", "ee_force", "force")),
        "eef_torque": lookup(("eef_torque", "ee_torque", "wrench_torque")),
    }


def _position(waypoint: Any) -> np.ndarray:
    if isinstance(waypoint, Mapping):
        for key in ("position", "pos", "eef_pos"):
            if key in waypoint:
                return np.asarray(waypoint[key], dtype=np.float64)
    for key in ("position", "pos", "eef_pos"):
        if hasattr(waypoint, key):
            return np.asarray(getattr(waypoint, key), dtype=np.float64)
    return np.asarray(waypoint, dtype=np.float64)[:3]


def _as_waypoint_rotation(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        arr = arr[:3, :3]
    if arr.shape == (3, 3):
        if not np.isfinite(arr).all() or not np.allclose(arr.T @ arr, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(arr), 1.0, atol=1e-4):
            raise ValueError("waypoint orientation must be a proper orthonormal rotation")
        return arr
    flat = arr.reshape(-1)
    if flat.size != 4 or not np.isfinite(flat).all() or np.linalg.norm(flat) <= 1e-9:
        raise ValueError("waypoint orientation must be a 3x3 rotation or quaternion")
    x, y, z, w = flat / np.linalg.norm(flat)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), dtype=np.float64)


def _orientation(waypoint: Any) -> np.ndarray | None:
    """Return an optional waypoint rotation in the controller's accepted form."""
    value = None
    if isinstance(waypoint, Mapping):
        for key in ("orientation", "rotation", "target_rot", "eef_quat", "quaternion"):
            if key in waypoint:
                value = waypoint[key]
                break
    else:
        for key in ("orientation", "rotation", "target_rot", "eef_quat", "quaternion"):
            if hasattr(waypoint, key):
                value = getattr(waypoint, key)
                break
    if value is None:
        return None
    return _as_waypoint_rotation(value)


def _orientation_frame(waypoint: Any) -> str | None:
    if isinstance(waypoint, Mapping):
        return waypoint.get("orientation_frame")
    return getattr(waypoint, "orientation_frame", None)



def _rotation_error_rad(current: Any, target: Any) -> float:
    """Compute shortest SO(3) error for orientation convergence auditing."""
    relative = _as_waypoint_rotation(target) @ _as_waypoint_rotation(current).T
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


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
    eef_orientation_transform: np.ndarray | None = None,
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
    target_rotation = _orientation(waypoint)
    if target_rotation is None:
        target_rotation = held_rotation
    if target_rotation is not None and _orientation_frame(waypoint) == "grip_site":
        if eef_orientation_transform is None:
            raise ValueError("grip_site waypoint requires explicit right_hand-to-grip_site calibration")
        current_rot = _as_waypoint_rotation(current_rot) @ _as_waypoint_rotation(eef_orientation_transform)
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
        target_rot=target_rotation,
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
    micro_correction: MicroCorrectionPolicy | None = None,
    post_phase_callback: Callable[[str, Mapping[str, Any], Mapping[str, np.ndarray]], None] | None = None,
    start_phase: str | None = None,
    action_budget: int | _ActionBudget | None = None,
    micro_action_budget: _ActionBudget | None = None,
    orientation_tolerance_rad: float = DEFAULT_ORIENTATION_TOLERANCE_RAD,
    eef_orientation_transform: np.ndarray | None = None,
    motion_trace_max_steps: int = DEFAULT_MOTION_TRACE_MAX_STEPS,
    motion_trace_path: str | Path | None = None,
    motion_trace_state: list[dict[str, Any]] | None = None,
    motion_trace_segment: str = "initial",
    experimental_motion_diagnostics: bool = False,
    motion_workspace_bounds: Mapping[str, Sequence[float]] | None = None,
    experimental_motion_correction: bool = False,
    failure_snapshot_callback: Callable[[str, int, Mapping[str, Any]], str | Path | None] | None = None,
    motion_frame_callback: Callable[[str, int], str | Path | None] | None = None,
    post_lift_retention_gate: Callable[[Mapping[str, Any], Mapping[str, np.ndarray]], Any] | None = None,
    phase_order: Sequence[str] | None = None,
    phase_timeout_steps_by_phase: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    if phase_timeout_steps <= 0:
        raise ValueError("phase_timeout_steps must be positive")
    if gripper_dwell_steps <= 0:
        raise ValueError("gripper_dwell_steps must be positive")
    ordered_phases = tuple(PHASES if phase_order is None else phase_order)
    if not ordered_phases or any(str(phase) == "" for phase in ordered_phases):
        raise ValueError("phase_order must contain at least one non-empty phase")
    if len(set(ordered_phases)) != len(ordered_phases):
        raise ValueError("phase_order must contain unique phases")
    if stop_after_phase not in ordered_phases:
        raise ValueError(f"stop_after_phase must be one of {ordered_phases}, got {stop_after_phase!r}")
    if start_phase is not None and start_phase not in ordered_phases:
        raise ValueError(f"start_phase must be one of {ordered_phases}, got {start_phase!r}")
    phase_limits = dict(phase_timeout_steps_by_phase or {})
    if any(str(name) not in ordered_phases for name in phase_limits):
        raise ValueError("phase_timeout_steps_by_phase contains an unknown phase")
    if any(type(value) is not int or value <= 0 for value in phase_limits.values()):
        raise ValueError("phase_timeout_steps_by_phase values must be positive integers")
    if stall_window_steps < 0 or stall_delta_m < 0 or not np.isfinite(stall_delta_m):
        raise ValueError("stall_window_steps and stall_delta_m must be finite and non-negative")
    if action_budget is not None:
        # A shared budget may legitimately have zero remaining after an
        # earlier phase.  Let the phase loop emit ControllerMotionTimeout so
        # the exhaustion is audited consistently; reject only an invalid
        # configured limit.
        configured_limit = (
            int(action_budget.limit)
            if isinstance(action_budget, _ActionBudget)
            else int(action_budget)
        )
        if configured_limit <= 0:
            raise ValueError("action_budget must be positive when supplied")
    if not np.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0:
        raise ValueError("orientation_tolerance_rad must be finite and positive")
    if type(motion_trace_max_steps) is not int or motion_trace_max_steps <= 0:
        raise ValueError("motion_trace_max_steps must be a positive integer")
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
    motion_trace: list[dict[str, Any]] = motion_trace_state if motion_trace_state is not None else []
    trace_path = Path(motion_trace_path).expanduser().resolve() if motion_trace_path is not None else None
    trace_truncated = False

    def persist_motion_trace() -> None:
        if trace_path is None:
            return
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({
                    "schema_version": "arrow_motion_trace.v1",
                    "max_steps": int(motion_trace_max_steps),
                    "truncated": bool(trace_truncated),
                    "steps": motion_trace,
                }, indent=2)
            temporary = trace_path.with_name(f".{trace_path.name}.{os.getpid()}.partial")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, trace_path)
            setattr(env, "_arrow_motion_trace_sha256", hashlib.sha256(trace_path.read_bytes()).hexdigest())
            setattr(env, "_arrow_motion_trace_persist_error", None)
        except Exception as exc:  # diagnostic persistence cannot alter control
            setattr(env, "_arrow_motion_trace_persist_error", str(exc))

    def append_motion_trace(entry: Mapping[str, Any]) -> None:
        nonlocal trace_truncated
        if len(motion_trace) < motion_trace_max_steps:
            item = dict(entry)
            item["segment"] = str(motion_trace_segment)
            motion_trace.append(item)
        else:
            trace_truncated = True
        setattr(env, "_arrow_motion_trace", motion_trace)
        setattr(env, "_arrow_motion_trace_truncated", trace_truncated)
        persist_motion_trace()

    def append_failure_record(record: dict[str, Any], phase: str) -> None:
        """Publish a phase failure and best-effort durable visual snapshot."""
        if not phase_audit or phase_audit[-1] is not record:
            phase_audit.append(record)
        setattr(env, "_arrow_phase_audit", phase_audit)
        if failure_snapshot_callback is not None and "failure_snapshot" not in record:
            try:
                path = failure_snapshot_callback(phase, len(phase_audit) - 1, record)
            except Exception as exc:  # snapshots are diagnostics, never recovery
                record["failure_snapshot"] = None
                record["failure_snapshot_error"] = str(exc)
            else:
                record["failure_snapshot"] = (
                    Path(path).expanduser().resolve().as_posix() if path is not None else None
                )
                if path is not None:
                    snapshots = list(getattr(env, "_arrow_failure_snapshots", []) or [])
                    snapshots.append(record["failure_snapshot"])
                    setattr(env, "_arrow_failure_snapshots", snapshots)
        persist_motion_trace()

    setattr(env, "_arrow_motion_trace", motion_trace)
    setattr(env, "_arrow_motion_trace_truncated", False)
    setattr(env, "_arrow_motion_trace_path", trace_path.as_posix() if trace_path is not None else None)
    setattr(env, "_arrow_failure_snapshots", [])
    sent_actions = 0
    # Keep a live reference so a timeout or other controller exception can be
    # serialized by the matrix runner even though no final audit is produced.
    setattr(env, "_arrow_phase_audit", phase_audit)
    correction_policy = micro_correction
    if correction_policy is not None and correction_policy.enabled and micro_action_budget is None:
        micro_action_budget = _ActionBudget(correction_policy.max_actions)
    live_corrections = getattr(env, "_arrow_micro_correction_audit", None)
    if not isinstance(live_corrections, list):
        live_corrections = []
        setattr(env, "_arrow_micro_correction_audit", live_corrections)
    started = start_phase is None
    for phase in ordered_phases:
        if not started:
            if phase == start_phase:
                started = True
            else:
                continue
        waypoint = _phase_waypoint(waypoints, phase)
        gripper = 1.0 if phase == "close" else -1.0 if phase == "open" else 0.0
        is_gripper_phase = phase in {"close", "open"}
        phase_tolerance = policies.get(phase, {}).get("tolerance_m", WAYPOINT_POSITION_TOLERANCE_M)
        if phase_tolerance is None:
            phase_tolerance = WAYPOINT_POSITION_TOLERANCE_M
        phase_tolerance = float(phase_tolerance)
        waypoint_orientation = _orientation(waypoint)
        required_steps = gripper_dwell_steps if is_gripper_phase else int(
            phase_limits.get(phase, phase_timeout_steps)
        )
        record = {
            "phase": phase,
            "steps": 0,
            "status": "dry_run" if dry_run else "pending",
            "gripper_command": float(gripper),
            "dwell_steps": int(gripper_dwell_steps) if is_gripper_phase else 0,
            "policy": dict(policies.get(phase, {})),
        }
        if phase in phase_limits:
            record["phase_timeout_steps"] = int(required_steps)
        if waypoint_orientation is not None:
            record["target_orientation_matrix"] = waypoint_orientation.tolist()
            record["orientation_tolerance_rad"] = float(orientation_tolerance_rad)
        error_history: list[float] = []
        correction_target = _position(waypoint)[:3]
        correction_remaining = 0
        correction_rounds = 0
        correction_actions = 0
        phase_correction_events: list[dict[str, Any]] = []
        insufficient_budget_audited = False
        for step in range(required_steps):
            correction_was_active = correction_remaining > 0
            # Micro-correction changes only the positional target.  Preserve
            # any learned pose orientation (and its explicit frame) when
            # rebuilding the per-step waypoint; otherwise the action path
            # would silently command the legacy held orientation while the
            # convergence check still waited for the learned one.
            action_waypoint: dict[str, Any] = {"position": correction_target}
            if motion_workspace_bounds is not None:
                validate_workspace_points(
                    {"commanded_target": correction_target}, bounds=motion_workspace_bounds
                )
            if waypoint_orientation is not None:
                action_waypoint["orientation"] = waypoint_orientation
                frame = _orientation_frame(waypoint)
                if frame is not None:
                    action_waypoint["orientation_frame"] = frame
            try:
                action = normalized_action_for_waypoint(
                    proprio,
                    action_waypoint,
                    gripper=gripper,
                    held_rotation=held_rotation,
                    osc_position_scale_m=osc_position_scale_m,
                    eef_orientation_transform=eef_orientation_transform,
                )
            except BaseException as exc:
                record["status"] = "exception"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                append_motion_trace(_motion_trace_entry(
                    phase, step + 1, proprio, action_waypoint,
                    action_sent=False, correction_active=correction_was_active,
                    experimental_motion_diagnostics=experimental_motion_diagnostics,
                    nominal_target=waypoint, commanded_target=action_waypoint,
                    action=None, before_proprio=proprio, env=env,
                ))
                append_failure_record(record, phase)
                raise
            record["last_action"] = action.tolist()
            record["steps"] = step + 1
            if dry_run:
                break
            budget_exhausted = (
                action_budget is not None
                and (action_budget.remaining <= 0 if isinstance(action_budget, _ActionBudget) else sent_actions >= int(action_budget))
            )
            if budget_exhausted:
                record["status"] = "timeout"
                record["action_budget"] = (
                    int(action_budget.limit)
                    if isinstance(action_budget, _ActionBudget)
                    else int(action_budget)
                )
                append_motion_trace(_motion_trace_entry(
                    phase, step + 1, proprio, action_waypoint,
                    action_sent=False, correction_active=correction_was_active,
                    experimental_motion_diagnostics=experimental_motion_diagnostics,
                    nominal_target=waypoint, commanded_target=action_waypoint,
                    action=action, before_proprio=proprio, env=env,
                ))
                append_failure_record(record, phase)
                budget_limit = (
                    int(action_budget.limit)
                    if isinstance(action_budget, _ActionBudget)
                    else int(action_budget)
                )
                raise ControllerMotionTimeout(f"phase {phase} exhausted action budget {budget_limit}")
            if not motion_notified:
                # Let a batch coordinator durably record the transition to
                # physical motion before the first simulator action is sent.
                if motion_started_callback is not None:
                    motion_started_callback()
                motion_notified = True
            setattr(env, "_arrow_motion_began", True)
            try:
                observation, done, _info = _step_once(env, action)
                sent_actions += 1
                if isinstance(action_budget, _ActionBudget) and not action_budget.consume():
                    # The pre-step check above is authoritative; this branch is
                    # defensive against a concurrently mutated budget.
                    raise ControllerMotionTimeout("action budget exhausted after action dispatch")
            except BaseException as exc:
                append_motion_trace(_motion_trace_entry(
                    phase, step + 1, proprio, action_waypoint,
                    action_sent=False, correction_active=correction_was_active,
                    experimental_motion_diagnostics=experimental_motion_diagnostics,
                    nominal_target=waypoint, commanded_target=action_waypoint,
                    action=action, before_proprio=proprio, env=env,
                ))
                # Preserve a partial gripper record so run_episode can offer a
                # bounded, explicitly-approved RGB-D recovery.  This is the
                # real reachability seam for recovery: a transport/controller
                # timeout during close/open, never an inferred object state.
                record["status"] = "timeout" if isinstance(exc, TimeoutError) else "exception"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                append_failure_record(record, phase)
                raise
            append_motion_trace(_motion_trace_entry(
                phase, step + 1, observation, action_waypoint,
                action_sent=True, correction_active=correction_was_active,
                experimental_motion_diagnostics=experimental_motion_diagnostics,
                nominal_target=waypoint, commanded_target=action_waypoint,
                action=action, before_proprio=proprio, env=env,
            ))
            if motion_frame_callback is not None:
                try:
                    motion_frame_callback(phase, len(motion_trace) - 1)
                except Exception as exc:  # frame capture is optional diagnostics
                    record.setdefault("motion_frame_errors", []).append(str(exc))
            proprio = _proprioception(observation)
            if observation is None or "eef_pos" not in proprio:
                record["status"] = "exception"
                record["error_type"] = "RuntimeError"
                record["error"] = f"phase {phase} lost EEF proprioception; failing closed"
                append_failure_record(record, phase)
                raise RuntimeError(record["error"])
            # Charge a correction action as soon as its simulator step
            # succeeds.  This must happen before the nominal tolerance check:
            # the final correction step may itself reach the waypoint.
            if correction_was_active:
                correction_actions += 1
                if micro_action_budget is not None:
                    micro_action_budget.consume()
            # LIBERO's ``done`` includes evaluator success.  It is intentionally
            # ignored while executing the bounded state machine: reading it or
            # changing phase based on it would make evaluator state influence
            # motion.  Success is queried only after retreat completes.
            if is_gripper_phase:
                if step + 1 >= gripper_dwell_steps:
                    record["status"] = "dwell"
                    break
            elif (
                np.linalg.norm(_position(waypoint) - proprio["eef_pos"][:3]) < phase_tolerance
                and (
                    waypoint_orientation is None
                    or _rotation_error_rad(
                        _as_waypoint_rotation(proprio.get("eef_quat")) @ _as_waypoint_rotation(eef_orientation_transform)
                        if _orientation_frame(waypoint) == "grip_site" and eef_orientation_transform is not None
                        else proprio.get("eef_quat"), waypoint_orientation
                    )
                    <= float(orientation_tolerance_rad)
                )
            ):
                record["status"] = "reached"
                break
            elif "eef_pos" in proprio:
                error = float(np.linalg.norm(_position(waypoint)[:3] - proprio["eef_pos"][:3]))
                error_history.append(error)
                # A correction burst gets a bounded chance to move beyond a
                # nominal target before the historical stall gate fires.
                # The policy is entirely numeric and phase-configured.
                if (
                    correction_policy is not None
                    and correction_policy.enabled
                    and phase in correction_policy.phases
                    and correction_remaining > 0
                ):
                    correction_remaining -= 1
                    if correction_remaining == 0:
                        correction_target = _position(waypoint)[:3]
                        if phase_correction_events:
                            event = phase_correction_events[-1]
                            final_residual = np.asarray(_position(waypoint)[:3], dtype=np.float64) - np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
                            event["post_residual_vector_m"] = final_residual.tolist()
                            event["post_residual_norm_m"] = float(np.linalg.norm(final_residual))
                            event["status"] = "completed"
                if (
                    correction_policy is not None
                    and correction_policy.enabled
                    and phase in correction_policy.phases
                    and correction_rounds < correction_policy.max_rounds
                    and micro_action_budget is not None
                    and micro_action_budget.remaining > 0
                    and len(error_history) >= correction_policy.plateau_window_steps
                    and max(error_history[-correction_policy.plateau_window_steps:])
                    - min(error_history[-correction_policy.plateau_window_steps:])
                    <= correction_policy.plateau_delta_m
                    and error > phase_tolerance
                    and correction_remaining == 0
                ):
                    current = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
                    nominal = np.asarray(_position(waypoint)[:3], dtype=np.float64)
                    residual = nominal - current
                    residual_norm = float(np.linalg.norm(residual))
                    if np.isfinite(residual_norm) and residual_norm > 0.0:
                        requested_burst = int(correction_policy.burst_steps)
                        if experimental_motion_correction:
                            phase_remaining = int(required_steps - (step + 1))
                            global_remaining = (
                                int(action_budget.remaining)
                                if isinstance(action_budget, _ActionBudget)
                                else int(action_budget) - sent_actions
                                if action_budget is not None
                                else None
                            )
                            micro_remaining = (
                                int(micro_action_budget.remaining)
                                if micro_action_budget is not None else 0
                            )
                            insufficient = (
                                phase_remaining < requested_burst
                                or (global_remaining is not None and global_remaining < requested_burst)
                                or micro_remaining < requested_burst
                            )
                            if insufficient:
                                if not insufficient_budget_audited:
                                    event = {
                                        "phase": phase,
                                        "trigger": "insufficient_remaining_budget",
                                        "trigger_step": int(step + 1),
                                        "status": "skipped",
                                        "required_burst_actions": requested_burst,
                                        "phase_remaining_steps": phase_remaining,
                                        "global_remaining_actions": global_remaining,
                                        "micro_remaining_actions": micro_remaining,
                                        "reason": "experimental_correction_requires_full_burst_budget",
                                    }
                                    live_corrections.append(event)
                                    phase_correction_events.append(event)
                                    record.setdefault("micro_corrections", []).append(event)
                                    insufficient_budget_audited = True
                                continue
                        distance = min(
                            float(correction_policy.residual_max_m),
                            residual_norm * float(correction_policy.correction_gain),
                        )
                        correction_target = nominal + residual / residual_norm * distance
                        correction_remaining = min(
                            int(correction_policy.burst_steps),
                            int(micro_action_budget.remaining),
                        )
                        if correction_remaining > 0:
                            correction_rounds += 1
                            event = {
                                "phase": phase,
                                "trigger": "residual_plateau",
                                "trigger_step": int(step + 1),
                                "round": int(correction_rounds),
                                "residual_vector_m": residual.tolist(),
                                "residual_norm_m": residual_norm,
                                "correction_target_m": correction_target.tolist(),
                                "burst_steps": int(correction_remaining),
                                "status": "triggered",
                                "post_residual_vector_m": None,
                                "post_residual_norm_m": None,
                            }
                            live_corrections.append(event)
                            phase_correction_events.append(event)
                            record.setdefault("micro_corrections", []).append(event)
                if (
                    correction_policy is not None
                    and correction_policy.enabled
                    and phase in correction_policy.phases
                    and correction_remaining > 0
                ):
                    # A correction target is active for the next bounded burst.
                    continue
                if (
                    stall_window_steps
                    and
                    len(error_history) >= stall_window_steps
                    and max(error_history[-stall_window_steps:])
                    - min(error_history[-stall_window_steps:]) <= stall_delta_m
                ):
                    record["status"] = "stall"
                    record["stall_window_steps"] = int(stall_window_steps)
                    record["stall_delta_m"] = float(stall_delta_m)
                    record["position_error_norm_m"] = error
                    append_failure_record(record, phase)
                    raise ControllerMotionTimeout(f"phase {phase} stalled for {stall_window_steps} steps")
        else:
            record["status"] = "timeout"
            if "eef_pos" in proprio:
                eef_pos = np.asarray(proprio["eef_pos"][:3], dtype=np.float64)
                record["eef_pos_m"] = eef_pos.tolist()
                record["position_error_norm_m"] = float(
                    np.linalg.norm(_position(waypoint)[:3] - eef_pos)
                )
                if waypoint_orientation is not None and proprio.get("eef_quat") is not None:
                    timeout_orientation = proprio["eef_quat"]
                    if _orientation_frame(waypoint) == "grip_site":
                        if eef_orientation_transform is None:
                            record["status"] = "exception"
                            record["error_type"] = "ValueError"
                            record["error"] = (
                                "grip_site timeout audit requires explicit "
                                "right_hand-to-grip_site calibration"
                            )
                            append_failure_record(record, phase)
                            raise ValueError(record["error"])
                        timeout_orientation = (
                            _as_waypoint_rotation(proprio["eef_quat"])
                            @ _as_waypoint_rotation(eef_orientation_transform)
                        )
                    record["orientation_error_rad"] = _rotation_error_rad(
                        timeout_orientation, waypoint_orientation
                    )
            append_failure_record(record, phase)
            if is_gripper_phase:
                raise ControllerMotionTimeout(f"phase {phase} failed to complete {gripper_dwell_steps}-step dwell")
            raise ControllerMotionTimeout(f"phase {phase} exceeded {phase_timeout_steps} steps")
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
        for event in phase_correction_events:
            if event.get("status") == "triggered":
                final_residual = np.asarray(_position(waypoint)[:3], dtype=np.float64) - np.asarray(proprio.get("eef_pos", _position(waypoint))[:3], dtype=np.float64)
                event["post_residual_vector_m"] = final_residual.tolist()
                event["post_residual_norm_m"] = float(np.linalg.norm(final_residual))
                event["status"] = "phase_end"
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
        if post_phase_callback is not None:
            try:
                post_phase_callback(phase, record, proprio)
            except BaseException as exc:
                record.setdefault("post_phase_error_type", type(exc).__name__)
                record.setdefault("post_phase_error", str(exc))
                append_failure_record(record, phase)
                raise
        if phase == "lift" and post_lift_retention_gate is not None and not dry_run:
            try:
                decision = post_lift_retention_gate(record, proprio)
                retained = bool(
                    decision.get("retained") if isinstance(decision, Mapping) else decision
                )
            except BaseException as exc:
                record["retention_gate"] = {
                    "enabled": True,
                    "retained": False,
                    "status": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_failure_record(record, phase)
                raise
            record["retention_gate"] = {
                "enabled": True,
                "retained": retained,
                "status": "passed" if retained else "rejected",
            }
            if isinstance(decision, Mapping):
                record["retention_gate"].update({
                    str(key): value for key, value in decision.items()
                    if key not in {"retained", "enabled", "status"}
                })
            if not retained:
                append_failure_record(record, phase)
                if isinstance(decision, Mapping) and decision.get("decision") == "unobservable":
                    raise RuntimeError("post_lift_retention_unobservable")
                raise _GraspSearchRequested("post_lift_retention")
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
    post_lift_retention_gate: Callable[[Mapping[str, Any], Mapping[str, np.ndarray]], Any] | None = None,
    motion_trace_max_steps: int = DEFAULT_MOTION_TRACE_MAX_STEPS,
    canary_video_dir: str | Path | None = None,
    canary_video_max_frames: int = DEFAULT_CANARY_VIDEO_MAX_FRAMES,
    experimental_candidate: Any | None = None,
    experimental_eef_orientation_transform: Sequence[Sequence[float]] | np.ndarray | None = None,
    experimental_gripper_opening_m: float | None = None,
    experimental_motion_diagnostics: bool = False,
    experimental_micro_correction: MicroCorrectionPolicy | None = None,
    retreat_completed_callback: Callable[[], None] | None = None,
    experimental_action_budget: _ActionBudget | None = None,
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
    experimental_candidate_audit: dict[str, Any] | None = None
    experimental_candidate_pose: tuple[np.ndarray, np.ndarray, float, np.ndarray | None] | None = None
    experimental_empty_gripper_threshold: float | None = None
    if experimental_candidate is not None:
        # Preserve the resolved policy threshold before disabling its legacy
        # candidate-search branch.  Experimental close detection remains
        # active, but must not invent a second threshold.
        if variant.grasp_search is not None:
            experimental_empty_gripper_threshold = float(
                variant.grasp_search.empty_gripper_threshold
            )
        else:
            experimental_empty_gripper_threshold = 0.0015
        # The experimental runner owns bounded candidate retries.  Disable the
        # legacy v9d RGB-D search for this call so a failed Molmo/SAM3 contact
        # can never silently switch to the old anchor policy.
        variant = replace(variant, grasp_search=None, grasp_retry_offsets_m=None, grasp_contact_threshold=None)
        candidate_point, candidate_rotation, candidate_aperture, candidate_pregrasp, experimental_candidate_audit = _experimental_candidate_pose(experimental_candidate)
        experimental_candidate_pose = (candidate_point, candidate_rotation, candidate_aperture, candidate_pregrasp)
        # Experimental candidate retries retain the close-contact guard even
        # though the legacy grasp_search policy is disabled below.
        if experimental_eef_orientation_transform is None:
            raise ValueError("experimental candidate motion requires explicit right_hand-to-grip_site calibration")
        eef_transform = np.asarray(experimental_eef_orientation_transform, dtype=np.float64)
        if eef_transform.shape != (3, 3) or not np.isfinite(eef_transform).all() or not np.allclose(eef_transform.T @ eef_transform, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(eef_transform), 1.0, atol=1e-5):
            raise ValueError("experimental_eef_orientation_transform must be a proper SO(3) matrix")
        experimental_eef_orientation_transform = eef_transform
        if experimental_action_budget is None:
            experimental_action_budget = _ActionBudget(1200)
        elif not isinstance(experimental_action_budget, _ActionBudget):
            raise TypeError("experimental_action_budget must be an _ActionBudget")
    elif experimental_micro_correction is not None:
        raise ValueError("experimental_micro_correction requires an experimental candidate")
    if experimental_micro_correction is not None and not isinstance(experimental_micro_correction, MicroCorrectionPolicy):
        experimental_micro_correction = MicroCorrectionPolicy.from_value(experimental_micro_correction)
    if experimental_candidate is not None and experimental_micro_correction is not None:
        if experimental_micro_correction.enabled:
            if not set(experimental_micro_correction.phases) <= {"preplace", "descend_place"}:
                raise ValueError("experimental micro-correction is limited to placement phases")
            if experimental_micro_correction.residual_max_m > 0.005:
                raise ValueError("experimental micro-correction residual_max_m must be <= 0.005 m")
            if experimental_micro_correction.burst_steps > 8 or experimental_micro_correction.max_rounds > 1 or experimental_micro_correction.max_actions > 16:
                raise ValueError("experimental micro-correction exceeds bounded placement profile")
        variant = replace(variant, micro_correction=experimental_micro_correction)
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
    experimental_correction_bounds = (
        workspace_bounds
        if experimental_micro_correction is not None and experimental_micro_correction.enabled
        else None
    )
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
    classical_destination_point = np.asarray(destination_point, dtype=np.float64).copy()
    classical_source_point = np.asarray(bowl_point, dtype=np.float64).copy()
    experimental_orientation_matrix: np.ndarray | None = None
    experimental_aperture_m: float | None = None
    if experimental_candidate_pose is not None:
        # Preserve the frozen transfer displacement while replacing only the
        # physical source contact.  The v9d source offset is never applied to
        # the candidate itself.
        candidate_point, candidate_rotation, candidate_aperture, candidate_pregrasp = experimental_candidate_pose
        transfer_displacement = classical_destination_point - classical_source_point
        bowl_point = candidate_point.copy()
        destination_point = bowl_point + transfer_displacement
        experimental_orientation_matrix = candidate_rotation.copy()
        experimental_aperture_m = float(candidate_aperture)
        setattr(env, "_arrow_experimental_candidate_pregrasp_world_m", None if candidate_pregrasp is None else candidate_pregrasp.tolist())
        setattr(env, "_arrow_experimental_candidate", {
            "candidate_id": str(getattr(experimental_candidate, "candidate_id", "unknown")),
            "grip_site_world_m": bowl_point.tolist(),
            "required_aperture_m": experimental_aperture_m,
            "transfer_displacement_world_m": transfer_displacement.tolist(),
            "orientation_world_grip_site": experimental_orientation_matrix.tolist(),
            "audit": dict(experimental_candidate_audit or {}),
            "no_legacy_source_offset": True,
        })
    source_approach_audit: dict[str, Any] | None = None
    support_plane_audit: dict[str, Any] | None = None
    source_policy = variant.source_approach
    if source_policy is not None and bool(source_policy.get("enabled")):
        if derive_rgbd_source_approach_candidates is None:
            raise RuntimeError("RGB-D source approach policy is unavailable")
        try:
            candidates = derive_rgbd_source_approach_candidates(
                source_uv,
                target_uv,
                np.asarray(capture.metric_depth, dtype=np.float64),
                K,
                T_world_camera,
                source_policy["offsets_arrow_frame_m"],
                workspace_bounds,
                patch_radius_px=int(source_policy["patch_radius_px"]),
                min_valid_fraction=float(source_policy["min_valid_fraction"]),
            )
            selected = candidates[0]
            bowl_point = np.asarray(selected.approach_world_m, dtype=np.float64).copy()
            source_approach_audit = {
                "enabled": True,
                "candidate_count": len(candidates),
                "selected_index": 0,
                "selected_point_world_m": bowl_point.tolist(),
                "offset_arrow_frame_m": selected.offset_arrow_frame_m.tolist(),
                "source_world_m": selected.source_world_m.tolist(),
                "forward_world_unit": selected.forward_world_unit.tolist(),
                "lateral_world_unit": selected.lateral_world_unit.tolist(),
                "source": "metric_rgbd_arrow_basis",
            }
        except (TypeError, ValueError, RuntimeError) as exc:
            source_approach_audit = {"enabled": True, "status": "rejected", "error_type": type(exc).__name__, "error": str(exc)}
            setattr(env, "_arrow_source_approach", dict(source_approach_audit))
            raise RuntimeError(f"RGB-D source approach rejected: {exc}") from exc
        setattr(env, "_arrow_source_approach", dict(source_approach_audit))
    destination_policy = variant.destination_placement
    if destination_policy is not None and bool(destination_policy.get("enabled")):
        if estimate_destination_support_plane is None or release_point_on_support_plane is None:
            raise RuntimeError("RGB-D destination support-plane policy is unavailable")
        try:
            support_plane = estimate_destination_support_plane(
                np.asarray(capture.metric_depth, dtype=np.float64),
                target_uv,
                K,
                T_world_camera,
                patch_radius_px=int(destination_policy["patch_radius_px"]),
                min_valid_fraction=float(destination_policy["min_valid_fraction"]),
                max_residual_m=float(destination_policy["max_residual_m"]),
            )
            support_point = release_point_on_support_plane(
                support_plane,
                np.asarray(destination_visual_point[:2], dtype=np.float64),
                workspace_bounds_m=workspace_bounds,
            )
            # The fitted surface is a support reference, not a collision target:
            # retain an explicit tool clearance along its observed normal.
            destination_point = support_point + support_plane.normal_world_unit * float(destination_policy["release_clearance_m"])
            validate_workspace_points({"support_plane_release_target": destination_point}, bounds=workspace_bounds)
            support_plane_audit = {
                "enabled": True,
                "valid": True,
                "status": "accepted",
                "origin_world_m": support_plane.origin_world_m.tolist(),
                "normal_world_unit": support_plane.normal_world_unit.tolist(),
                "residual_rms_m": float(support_plane.residual_rms_m),
                "residual_max_m": float(support_plane.residual_max_m),
                "valid_point_count": int(support_plane.valid_point_count),
                "release_point_world_m": destination_point.tolist(),
                "support_point_world_m": support_point.tolist(),
                "release_clearance_m": float(destination_policy["release_clearance_m"]),
                "source": "metric_rgbd_destination_neighborhood",
            }
        except (TypeError, ValueError, RuntimeError) as exc:
            support_plane_audit = {"enabled": True, "valid": False, "status": "rejected", "error_type": type(exc).__name__, "error": str(exc)}
            setattr(env, "_arrow_support_plane", dict(support_plane_audit))
            setattr(env, "_arrow_placement_observable", {"enabled": True, "status": "rejected"})
            raise RuntimeError(f"RGB-D destination support plane rejected: {exc}") from exc
        setattr(env, "_arrow_support_plane", dict(support_plane_audit))
        setattr(env, "_arrow_placement_observable", {"enabled": True, "status": "accepted"})
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
    if experimental_candidate_pose is not None:
        if experimental_gripper_opening_m is None:
            raise ValueError("experimental candidate motion requires measured gripper opening")
        measured_opening = float(experimental_gripper_opening_m)
        if not np.isfinite(measured_opening) or measured_opening <= 0.0:
            raise ValueError("experimental_gripper_opening_m must be finite and positive")
        if experimental_aperture_m is None or measured_opening + 1e-6 < experimental_aperture_m:
            raise RuntimeError(
                "measured Panda aperture is smaller than the candidate requirement; refusing descent"
            )
    else:
        measured_opening = None
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    waypoints = _require(build_bowl_waypoints, "build_bowl_waypoints")(
        bowl_point, destination_point,
        initial_proprio.get("eef_quat"),
        {"lift_height_m": float(clearance_m)},
    )
    motion_phase_order: Sequence[str] | None = None
    motion_phase_limits: Mapping[str, int] | None = None
    motion_execution_timeout_steps = phase_timeout_steps
    if experimental_orientation_matrix is not None:
        # Experimental candidates carry their own approach target.  Stage at
        # a vertical clearance above the current EEF XY, rotate while there,
        # translate at that clearance, then use the supplied pregrasp before
        # descending.  Rotation is commanded only at this observed elevated
        # safe_z; it is a bounded clearance condition, not collision proof.
        current_eef = np.asarray(initial_proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
        if current_eef.shape != (3,) or not np.isfinite(current_eef).all():
            raise ValueError("experimental candidate motion requires finite current EEF position")
        if candidate_pregrasp is None:
            if not dry_run:
                raise ValueError(
                    "experimental candidate requires explicit pregrasp_world_m for motion"
                )
            # Keep dependency-free dry-run fixtures predating the richer
            # geometry candidate importable without authorizing live motion.
            candidate_pregrasp = candidate_point + np.asarray((0.0, 0.0, 0.03), dtype=np.float64)
            pregrasp_source = "dry_run_fixture_fallback_0.03m"
        else:
            pregrasp_source = "candidate.pregrasp_world_m"
        if not np.isfinite(candidate_pregrasp).all():
            raise ValueError("experimental candidate pregrasp must be finite")
        safe_z = max(float(current_eef[2]), float(candidate_pregrasp[2]))
        clearance_start = np.asarray((current_eef[0], current_eef[1], safe_z), dtype=np.float64)
        clearance_xy = np.asarray((candidate_pregrasp[0], candidate_pregrasp[1], safe_z), dtype=np.float64)
        staged = {
            "vertical_clearance": _pose_waypoint(clearance_start),
            "rotate": _pose_waypoint(clearance_start, experimental_orientation_matrix, "grip_site"),
            "translate_clearance": _pose_waypoint(clearance_xy, experimental_orientation_matrix, "grip_site"),
            "pregrasp": _pose_waypoint(candidate_pregrasp, experimental_orientation_matrix, "grip_site"),
            "descend": _pose_waypoint(candidate_point, experimental_orientation_matrix, "grip_site"),
            "close": _pose_waypoint(waypoints[1], experimental_orientation_matrix, "grip_site"),
            "lift": _pose_waypoint(waypoints[2], experimental_orientation_matrix, "grip_site"),
            "preplace": _pose_waypoint(waypoints[3], experimental_orientation_matrix, "grip_site"),
            "descend_place": _pose_waypoint(waypoints[4], experimental_orientation_matrix, "grip_site"),
            "open": _pose_waypoint(waypoints[4], experimental_orientation_matrix, "grip_site"),
            "retreat": _pose_waypoint(waypoints[5], experimental_orientation_matrix, "grip_site"),
        }
        waypoints = staged
        motion_phase_order = (
            "vertical_clearance", "rotate", "translate_clearance", "pregrasp",
            "descend", "close", "lift", "preplace", "descend_place", "open", "retreat",
        )
        motion_phase_limits = {
            name: 160 for name in ("vertical_clearance", "rotate", "translate_clearance", "pregrasp", "descend")
        }
        motion_execution_timeout_steps = 160
        experimental_candidate_audit = dict(experimental_candidate_audit or {})
        experimental_candidate_audit.update({
            "pregrasp_world_m": candidate_pregrasp.tolist(),
            "pregrasp_source": pregrasp_source,
            "clearance_start_world_m": clearance_start.tolist(),
            "clearance_xy_world_m": clearance_xy.tolist(),
            "clearance_z_m": safe_z,
            "phase_order": list(motion_phase_order),
        })
        experimental_record = getattr(env, "_arrow_experimental_candidate", None)
        if isinstance(experimental_record, Mapping):
            experimental_record = dict(experimental_record)
            experimental_record["audit"] = dict(experimental_candidate_audit)
            setattr(env, "_arrow_experimental_candidate", experimental_record)
    else:
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
    if type(motion_trace_max_steps) is not int or motion_trace_max_steps <= 0:
        raise ValueError("motion_trace_max_steps must be a positive integer")
    if type(canary_video_max_frames) is not int or canary_video_max_frames <= 0:
        raise ValueError("canary_video_max_frames must be a positive integer")
    canary_video_root = (
        Path(canary_video_dir).expanduser().resolve() if canary_video_dir is not None else None
    )
    canary_video_frame_paths: list[str] = []
    canary_video_truncated = False

    def persist_canary_video_manifest() -> None:
        if canary_video_root is None:
            return
        manifest_path = canary_video_root / "canary_video_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "schema_version": "arrow_canary_video.v1",
            "max_frames": int(canary_video_max_frames),
            "truncated": bool(canary_video_truncated),
            "frames": canary_video_frame_paths,
        }, indent=2), encoding="utf-8")

    def canary_motion_frame_callback(phase: str, frame_index: int) -> Path | None:
        nonlocal canary_video_truncated
        if canary_video_root is None:
            return None
        if len(canary_video_frame_paths) >= canary_video_max_frames:
            canary_video_truncated = True
            persist_canary_video_manifest()
            return None
        path = _save_phase_snapshot(
            env,
            canary_video_root,
            len(canary_video_frame_paths),
            f"{phase}_{int(frame_index):04d}",
            width=capture.rgb.shape[1],
            height=capture.rgb.shape[0],
        )
        canary_video_frame_paths.append(path.as_posix())
        persist_canary_video_manifest()
        return path

    def finalize_canary_video() -> dict[str, Any] | None:
        if canary_video_root is None:
            return None
        persist_canary_video_manifest()
        result: dict[str, Any] = {
            "enabled": True,
            "frames": list(canary_video_frame_paths),
            "frame_count": len(canary_video_frame_paths),
            "max_frames": int(canary_video_max_frames),
            "truncated": bool(canary_video_truncated),
            "video_path": None,
            "video_sha256": None,
            "status": "no_frames",
        }
        if not canary_video_frame_paths:
            setattr(env, "_arrow_canary_video", dict(result))
            return result
        try:
            import imageio.v2 as imageio

            # MP4 is intentionally assembled only from the post-step RGB
            # snapshots.  These frames are diagnostics and never re-enter
            # controller input or evaluator timing.
            video_path = canary_video_root / "motion.mp4"
            with imageio.get_writer(
                video_path,
                format="FFMPEG",
                mode="I",
                fps=20,
                codec="libx264",
                macro_block_size=1,
            ) as writer:
                for path in canary_video_frame_paths:
                    writer.append_data(imageio.imread(path))
            digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
            result.update({
                "video_path": video_path.as_posix(),
                "video_sha256": digest,
                "status": "complete",
            })
        except Exception as exc:  # canary artifact failure must remain visible
            result.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        setattr(env, "_arrow_canary_video", dict(result))
        return result

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

    def failure_snapshot_callback(
        phase: str, phase_index: int, _record: Mapping[str, Any]
    ) -> Path:
        path = _save_phase_snapshot(
            env,
            output_root / "failure_snapshots",
            phase_index,
            phase,
            width=capture.rgb.shape[1],
            height=capture.rgb.shape[0],
        )
        return path

    def grasp_search_frame_callback(
        attempt_index: int, segment: str
    ) -> Callable[[str, int], Path]:
        """Keep retry diagnostics separate so repeated phase names never overwrite."""
        def callback(phase: str, phase_index: int) -> Path:
            path = _save_phase_snapshot(
                env,
                output_root / "grasp_search_attempts" / f"attempt_{attempt_index:02d}" / segment,
                phase_index,
                phase,
                width=capture.rgb.shape[1],
                height=capture.rgb.shape[0],
            )
            phase_frame_paths.append(path.as_posix())
            return path

        return callback

    recovery_audit: list[dict[str, Any]] = []
    grasp_search_audit: list[dict[str, Any]] = []
    micro_correction_audit: list[dict[str, Any]] = []
    if (
        experimental_micro_correction is not None
        and experimental_micro_correction.enabled
    ):
        shared_micro_budget = getattr(env, "_arrow_experimental_micro_action_budget", None)
        if not isinstance(shared_micro_budget, _ActionBudget) or shared_micro_budget.limit != experimental_micro_correction.max_actions:
            shared_micro_budget = _ActionBudget(experimental_micro_correction.max_actions)
            setattr(env, "_arrow_experimental_micro_action_budget", shared_micro_budget)
        micro_action_budget = shared_micro_budget
    else:
        micro_action_budget = (
            _ActionBudget(variant.micro_correction.max_actions)
            if variant.micro_correction is not None and variant.micro_correction.enabled
            else None
        )
    setattr(env, "_arrow_grasp_search_audit", grasp_search_audit)
    setattr(env, "_arrow_micro_correction_audit", micro_correction_audit)

    effective_retention_gate = post_lift_retention_gate
    if (
        effective_retention_gate is None
        and not dry_run
        and variant.grasp_search is not None
        and variant.grasp_search.enabled
        and "post_lift_retention" in variant.grasp_search.trigger_on
    ):
        def effective_retention_gate(record: Mapping[str, Any], proprio: Mapping[str, np.ndarray]) -> dict[str, Any]:
            del record  # phase identity is supplied by the motion loop
            return _post_lift_retention_decision(
                list(getattr(env, "_arrow_motion_trace", []) or []),
                proprio,
                variant.grasp_search.empty_gripper_threshold,
            )
        setattr(env, "_arrow_post_lift_retention_gate", {"enabled": True, "source": "lift_proprioception_trace"})

    def _after_motion_phase(
        phase: str, record: Mapping[str, Any], _proprio: Mapping[str, np.ndarray]
    ) -> None:
        # An empty gripper is a proprioceptive signal only.  Raising here
        # stops immediately after close, before lift/place can proceed.
        policy = variant.grasp_search
        close_threshold = (
            experimental_empty_gripper_threshold
            if experimental_empty_gripper_threshold is not None
            else (policy.empty_gripper_threshold if policy is not None else None)
        )
        if (
            close_threshold is not None
            and (experimental_candidate_pose is not None or (policy is not None and policy.enabled))
            and not dry_run
            and stop_after_phase == "retreat"
            and phase == "close"
            and (experimental_candidate_pose is not None or "empty_gripper_likely" in policy.trigger_on)
            and _empty_gripper_likely([record], close_threshold)
        ):
            raise _GraspSearchRequested("empty_gripper_likely")

    try:
        phase_audit = _run_motion(
            env,
            waypoints,
            capture.observation,
            phase_timeout_steps=motion_execution_timeout_steps,
            gripper_dwell_steps=gripper_dwell_steps,
            stop_after_phase=stop_after_phase,
            dry_run=dry_run,
            phase_frame_callback=phase_frame_callback if not dry_run else None,
            motion_started_callback=motion_started_callback if not dry_run else None,
            stall_window_steps=variant.stall_window_steps,
            stall_delta_m=variant.stall_delta_m,
            phase_policies=phase_policies,
            osc_position_scale_m=variant.osc_position_scale_m,
            micro_correction=variant.micro_correction,
            micro_action_budget=micro_action_budget,
            action_budget=experimental_action_budget,
            post_phase_callback=_after_motion_phase,
            eef_orientation_transform=experimental_eef_orientation_transform,
            motion_trace_max_steps=motion_trace_max_steps,
            motion_trace_path=output_root / "motion_trace.json",
            experimental_motion_diagnostics=experimental_motion_diagnostics,
            motion_workspace_bounds=experimental_correction_bounds,
            experimental_motion_correction=(experimental_micro_correction is not None and experimental_micro_correction.enabled),
            failure_snapshot_callback=failure_snapshot_callback,
            motion_frame_callback=(canary_motion_frame_callback if canary_video_root is not None else None),
            post_lift_retention_gate=effective_retention_gate,
            phase_order=motion_phase_order,
            phase_timeout_steps_by_phase=motion_phase_limits,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            finalize_canary_video()
            raise
        phase_audit = list(getattr(env, "_arrow_phase_audit", []) or [])
        search_policy = variant.grasp_search
        last_phase = str(phase_audit[-1].get("phase")) if phase_audit else ""
        last_status = str(phase_audit[-1].get("status")) if phase_audit else ""
        if isinstance(exc, _GraspSearchRequested):
            trigger = exc.trigger
        elif isinstance(exc, ControllerMotionTimeout) and last_phase in {"close", "lift"} and last_status in {"stall", "timeout"}:
            trigger = f"{last_phase}_{last_status}"
        else:
            trigger = None
        can_search = bool(
            not dry_run
            and stop_after_phase == "retreat"
            and search_policy is not None
            and search_policy.enabled
            and trigger in search_policy.trigger_on
            and search_policy.max_attempts > 0
            and (
                search_policy.strategy == "rgbd_region"
                or bool(search_policy.offsets_m)
            )
        )
        search_completed = False
        if can_search:
            search_candidates: list[tuple[np.ndarray, dict[str, Any]]] = []
            if search_policy.strategy == "rgbd_region":
                candidate_fn = _require(
                    derive_rgbd_region_grasp_candidates,
                    "derive_rgbd_region_grasp_candidates",
                )
                try:
                    candidate_targets, candidate_diagnostics = candidate_fn(
                        np.asarray(capture.rgb, dtype=np.uint8),
                        np.asarray(capture.metric_depth, dtype=np.float64),
                        K,
                        T_world_camera,
                        source_uv,
                        bowl_point,
                        region_radius_m=search_policy.region_radius_m,
                        depth_tolerance_m=search_policy.region_depth_tolerance_m,
                        min_region_pixels=search_policy.region_min_pixels,
                        max_region_fraction=search_policy.region_max_fraction,
                        height_quantile=search_policy.region_height_quantile,
                        profile_quantiles=search_policy.region_profile_quantiles,
                        candidate_height_quantiles=(
                            search_policy.region_candidate_height_quantiles or None
                        ),
                        seed_radius_px=search_policy.region_seed_radius_px,
                    )
                except Exception as candidate_exc:
                    grasp_search_audit.append({
                        "stage": "candidate_generation",
                        "strategy": search_policy.strategy,
                        "status": "failed",
                        "trigger": trigger,
                        "error_type": type(candidate_exc).__name__,
                        "error": str(candidate_exc),
                        "selected": False,
                    })
                    raise exc
                candidate_diagnostics = dict(candidate_diagnostics)
                grasp_search_audit.append({
                    "stage": "candidate_generation",
                    "strategy": search_policy.strategy,
                    "status": "generated",
                    "trigger": trigger,
                    "candidate_count": int(len(candidate_targets)),
                    "diagnostics": candidate_diagnostics,
                    "selected": False,
                })
                selected_quantiles = list(
                    candidate_diagnostics.get("selected_profile_quantiles", [])
                )
                selected_pixels = list(candidate_diagnostics.get("selected_pixels_xy", []))
                selected_height_quantiles = list(
                    candidate_diagnostics.get("selected_height_quantiles", [])
                )
                for index, candidate_target in enumerate(candidate_targets):
                    search_candidates.append((
                        np.asarray(candidate_target, dtype=np.float64),
                        {
                            "offset_frame": "arrow_seeded_rgbd_region_world",
                            "profile_quantile": (
                                selected_quantiles[index]
                                if index < len(selected_quantiles) else None
                            ),
                            "selected_pixel_xy": (
                                selected_pixels[index]
                                if index < len(selected_pixels) else None
                            ),
                            "height_quantile": (
                                selected_height_quantiles[index]
                                if index < len(selected_height_quantiles) else None
                            ),
                        },
                    ))
            else:
                basis_fn = _require(arrow_world_xy_basis, "arrow_world_xy_basis")
                forward, lateral = basis_fn(source_visual_point, destination_visual_point)
                for offset_value in search_policy.offsets_m:
                    offset_array = np.asarray(offset_value, dtype=np.float64)
                    target = (
                        np.asarray(bowl_point, dtype=np.float64)
                        + forward * offset_array[0]
                        + lateral * offset_array[1]
                        + np.array((0.0, 0.0, offset_array[2]), dtype=np.float64)
                    )
                    search_candidates.append((target, {
                        "offset_frame": "arrow_world_xy_forward_lateral_vertical",
                        "configured_offset_m": offset_array.tolist(),
                    }))
            search_actions = 0
            selected_waypoints = None
            for attempt_index, (retry_bowl_point, candidate_metadata) in enumerate(
                search_candidates[: search_policy.max_attempts], start=1
            ):
                if search_actions >= search_policy.max_actions:
                    break
                retry_bowl_point = np.asarray(retry_bowl_point, dtype=np.float64)
                effective_offset = retry_bowl_point - np.asarray(bowl_point, dtype=np.float64)
                attempt_record: dict[str, Any] = {
                    "attempt": int(attempt_index),
                    "trigger": trigger,
                    "strategy": search_policy.strategy,
                    "offset_frame": candidate_metadata["offset_frame"],
                    "offset_m": effective_offset.tolist(),
                    "target_m": retry_bowl_point.tolist(),
                    "status": "pending",
                    "selected": False,
                    "actions_before": int(search_actions),
                }
                attempt_record.update({
                    key: value for key, value in candidate_metadata.items()
                    if key != "offset_frame" and value is not None
                })
                base_phase_audit = list(phase_audit)
                placement_started = False
                try:
                    validate_workspace_points(
                        {"grasp_search_source": retry_bowl_point}, bounds=workspace_bounds
                    )
                    retry_waypoints = _require(build_bowl_waypoints, "build_bowl_waypoints")(
                        retry_bowl_point,
                        destination_point,
                        initial_proprio.get("eef_quat"),
                        {"lift_height_m": float(clearance_m)},
                    )
                    # Validate every generated candidate before sending action.
                    retry_points = {
                        f"waypoint_{idx}": _position(value)[:3]
                        for idx, value in enumerate(retry_waypoints)
                    }
                    validate_workspace_points(retry_points, bounds=workspace_bounds)
                    # Explicitly open and retreat from the previous grasp
                    # before every candidate.  A retry never begins while the
                    # gripper is still closed at the failed source point.
                    current_proprio = _proprioception(_raw_observation(env))
                    current_pos = current_proprio.get("eef_pos")
                    current_quat = current_proprio.get("eef_quat")
                    if current_pos is None or current_quat is None:
                        raise RuntimeError("grasp search requires EEF proprioception for open/retreat")
                    reset_waypoints = np.tile(np.asarray(current_pos[:3], dtype=np.float64), (6, 1))
                    reset_waypoints[5] = _position(retry_waypoints[0])[:3]
                    reset_audit = _run_motion(
                        env, reset_waypoints, _raw_observation(env),
                        phase_timeout_steps=search_policy.phase_timeout_steps,
                        gripper_dwell_steps=gripper_dwell_steps,
                        stop_after_phase="retreat", start_phase="open", dry_run=False,
                        phase_frame_callback=grasp_search_frame_callback(
                            attempt_index, "reset"
                        ),
                        motion_started_callback=motion_started_callback,
                        stall_window_steps=variant.stall_window_steps,
                        stall_delta_m=variant.stall_delta_m,
                        phase_policies=phase_policies,
                        osc_position_scale_m=variant.osc_position_scale_m,
                        micro_correction=None,
                        action_budget=search_policy.max_actions - search_actions,
                        motion_trace_path=output_root / "motion_trace.json",
                        experimental_motion_diagnostics=experimental_motion_diagnostics,
                        motion_workspace_bounds=experimental_correction_bounds,
                        experimental_motion_correction=(experimental_micro_correction is not None and experimental_micro_correction.enabled),
                        motion_trace_state=getattr(env, "_arrow_motion_trace", None),
                        motion_trace_segment=f"attempt_{attempt_index}_reset",
                        failure_snapshot_callback=failure_snapshot_callback,
                        motion_frame_callback=canary_motion_frame_callback if canary_video_root is not None else None,
                    )
                    search_actions += sum(int(item.get("steps", 0)) for item in reset_audit)
                    for item in reset_audit:
                        item["grasp_search_attempt"] = int(attempt_index)
                    phase_audit.extend(reset_audit)
                    setattr(env, "_arrow_phase_audit", phase_audit)
                    retry_audit = _run_motion(
                        env, retry_waypoints, _raw_observation(env),
                        phase_timeout_steps=search_policy.phase_timeout_steps,
                        gripper_dwell_steps=gripper_dwell_steps,
                        stop_after_phase="lift", dry_run=False,
                        phase_frame_callback=grasp_search_frame_callback(
                            attempt_index, "grasp"
                        ),
                        motion_started_callback=motion_started_callback,
                        stall_window_steps=variant.stall_window_steps,
                        stall_delta_m=variant.stall_delta_m,
                        phase_policies=phase_policies,
                        osc_position_scale_m=variant.osc_position_scale_m,
                        micro_correction=variant.micro_correction,
                        micro_action_budget=micro_action_budget,
                        action_budget=search_policy.max_actions - search_actions,
                        motion_trace_path=output_root / "motion_trace.json",
                        experimental_motion_diagnostics=experimental_motion_diagnostics,
                        motion_workspace_bounds=experimental_correction_bounds,
                        experimental_motion_correction=(experimental_micro_correction is not None and experimental_micro_correction.enabled),
                        motion_trace_state=getattr(env, "_arrow_motion_trace", None),
                        motion_trace_segment=f"attempt_{attempt_index}_grasp",
                        failure_snapshot_callback=failure_snapshot_callback,
                        motion_frame_callback=canary_motion_frame_callback if canary_video_root is not None else None,
                        post_lift_retention_gate=effective_retention_gate,
                    )
                    search_actions += sum(int(item.get("steps", 0)) for item in retry_audit)
                    for item in retry_audit:
                        item["grasp_search_attempt"] = int(attempt_index)
                    phase_audit.extend(retry_audit)
                    if _empty_gripper_likely(retry_audit, search_policy.empty_gripper_threshold):
                        attempt_record["status"] = "empty_gripper_likely"
                        attempt_record["gripper_qpos"] = _observed_gripper_qpos(retry_audit)
                        attempt_record["actions_after"] = int(search_actions)
                        attempt_record["phase_statuses"] = [
                            {"phase": item.get("phase"), "status": item.get("status")}
                            for item in retry_audit
                        ]
                        grasp_search_audit.append(attempt_record)
                        setattr(env, "_arrow_phase_audit", phase_audit)
                        continue
                    selected_waypoints = retry_waypoints
                    attempt_record["status"] = "selected"
                    attempt_record["selected"] = True
                    attempt_record["gripper_qpos"] = _observed_gripper_qpos(retry_audit)
                    # Continue from preplace only after a proprioception-gated
                    # grasp has been obtained; placement is never evaluated in
                    # a failed candidate.
                    placement_started = True
                    placement_audit = _run_motion(
                        env, retry_waypoints, _raw_observation(env),
                        phase_timeout_steps=search_policy.phase_timeout_steps,
                        gripper_dwell_steps=gripper_dwell_steps,
                        stop_after_phase="retreat", start_phase="preplace", dry_run=False,
                        phase_frame_callback=grasp_search_frame_callback(
                            attempt_index, "placement"
                        ),
                        motion_started_callback=motion_started_callback,
                        stall_window_steps=variant.stall_window_steps,
                        stall_delta_m=variant.stall_delta_m,
                        phase_policies=phase_policies,
                        osc_position_scale_m=variant.osc_position_scale_m,
                        micro_correction=variant.micro_correction,
                        micro_action_budget=micro_action_budget,
                        action_budget=search_policy.max_actions - search_actions,
                        motion_trace_path=output_root / "motion_trace.json",
                        experimental_motion_diagnostics=experimental_motion_diagnostics,
                        motion_workspace_bounds=experimental_correction_bounds,
                        experimental_motion_correction=(experimental_micro_correction is not None and experimental_micro_correction.enabled),
                        motion_trace_state=getattr(env, "_arrow_motion_trace", None),
                        motion_trace_segment=f"attempt_{attempt_index}_placement",
                        failure_snapshot_callback=failure_snapshot_callback,
                        motion_frame_callback=canary_motion_frame_callback if canary_video_root is not None else None,
                    )
                    search_actions += sum(int(item.get("steps", 0)) for item in placement_audit)
                    attempt_record["actions_after"] = int(search_actions)
                    for item in placement_audit:
                        item["grasp_search_attempt"] = int(attempt_index)
                    phase_audit.extend(placement_audit)
                    setattr(env, "_arrow_phase_audit", phase_audit)
                    grasp_search_audit.append(attempt_record)
                    break
                except _GraspSearchRequested as search_exc:
                    # A retry candidate is still subject to the retention
                    # gate.  Losing contact on that candidate is a bounded
                    # candidate failure, not a runtime failure: preserve its
                    # partial trace and continue with the next RGB-D-derived
                    # candidate.  Previously this exception fell through to
                    # the generic handler, aborting the search after candidate
                    # 1 and making the retention treatment look worse than it
                    # was.
                    partial = list(getattr(env, "_arrow_phase_audit", []) or [])
                    search_actions += sum(int(item.get("steps", 0)) for item in partial)
                    phase_audit = base_phase_audit + partial
                    setattr(env, "_arrow_phase_audit", phase_audit)
                    attempt_record["status"] = "post_lift_retention_failed"
                    attempt_record["error_type"] = type(search_exc).__name__
                    attempt_record["error"] = str(search_exc)
                    attempt_record["actions_after"] = int(search_actions)
                    attempt_record["gripper_qpos"] = _observed_gripper_qpos(partial)
                    attempt_record["phase_statuses"] = [
                        {"phase": item.get("phase"), "status": item.get("status")}
                        for item in partial
                    ]
                    grasp_search_audit.append(attempt_record)
                    continue
                except ControllerMotionTimeout as search_exc:
                    partial = list(getattr(env, "_arrow_phase_audit", []) or [])
                    search_actions += sum(int(item.get("steps", 0)) for item in partial)
                    phase_audit = base_phase_audit + partial
                    setattr(env, "_arrow_phase_audit", phase_audit)
                    attempt_record["status"] = "failed"
                    attempt_record["error_type"] = type(search_exc).__name__
                    attempt_record["error"] = str(search_exc)
                    attempt_record["actions_after"] = int(search_actions)
                    grasp_search_audit.append(attempt_record)
                    if locals().get("placement_started", False):
                        raise
                    continue
                except Exception as search_exc:
                    # Environment/runtime failures are not controller stalls;
                    # continuing with another candidate could hide corruption
                    # or an unsafe partially executed trajectory.
                    attempt_record["status"] = "failed"
                    attempt_record["error_type"] = type(search_exc).__name__
                    attempt_record["error"] = str(search_exc)
                    attempt_record["selected"] = bool(placement_started)
                    attempt_record["actions_after"] = int(search_actions)
                    grasp_search_audit.append(attempt_record)
                    raise
            if selected_waypoints is not None:
                search_completed = True
                waypoints = selected_waypoints
                setattr(env, "_arrow_waypoints_world_m", {
                    f"waypoint_{idx}": _position(value)[:3].tolist()
                    for idx, value in enumerate(waypoints)
                })
            else:
                grasp_search_audit.append({
                    "status": "exhausted", "trigger": trigger, "selected": False,
                    "actions": int(search_actions), "actions_after": int(search_actions),
                    "max_actions": int(search_policy.max_actions),
                })
                raise exc
        if (
            not search_completed
            and isinstance(exc, TimeoutError)
            and not can_search
            and not dry_run
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
        if not search_completed:
            finalize_canary_video()
            raise exc
    grasp_retry_audit: list[dict[str, Any]] = []
    # Publish retry evidence on the environment as soon as the motion phase
    # begins.  The matrix runner samples environment diagnostics even when a
    # controller timeout prevents construction of the final audit object.
    setattr(env, "_arrow_grasp_retry_audit", grasp_retry_audit)
    if (
        not dry_run
        and stop_after_phase == "retreat"
        and variant.grasp_contact_threshold is not None
        and variant.grasp_retry_offsets_m
        and _gripper_contact_likely(phase_audit, variant.grasp_contact_threshold)
    ):
        for retry_index, z_offset in enumerate(variant.grasp_retry_offsets_m, start=1):
            retry_record: dict[str, Any] = {
                "attempt": int(retry_index),
                "z_offset_m": float(z_offset),
                "contact_before_retry": True,
                "status": "pending",
            }
            retry_bowl_point = np.asarray(bowl_point, dtype=np.float64).copy()
            retry_bowl_point[2] += float(z_offset)
            try:
                retry_waypoints = _require(build_bowl_waypoints, "build_bowl_waypoints")(
                    retry_bowl_point,
                    destination_point,
                    initial_proprio.get("eef_quat"),
                    {"lift_height_m": float(clearance_m)},
                )
                validate_workspace_points(
                    {"retry_source_grasp_target": retry_bowl_point}, bounds=workspace_bounds
                )
                retry_observation = _raw_observation(env)
                retry_phase_audit = _run_motion(
                    env,
                    retry_waypoints,
                    retry_observation,
                    phase_timeout_steps=phase_timeout_steps,
                    gripper_dwell_steps=gripper_dwell_steps,
                    stop_after_phase="retreat",
                    dry_run=False,
                    phase_frame_callback=None,
                    motion_started_callback=motion_started_callback,
                    stall_window_steps=variant.stall_window_steps,
                    stall_delta_m=variant.stall_delta_m,
                    phase_policies=phase_policies,
                    osc_position_scale_m=variant.osc_position_scale_m,
                    motion_trace_path=output_root / "motion_trace.json",
                    experimental_motion_diagnostics=experimental_motion_diagnostics,
                    motion_workspace_bounds=experimental_correction_bounds,
                    experimental_motion_correction=(experimental_micro_correction is not None and experimental_micro_correction.enabled),
                    motion_trace_state=getattr(env, "_arrow_motion_trace", None),
                    motion_trace_segment=f"grasp_retry_{retry_index}",
                    failure_snapshot_callback=failure_snapshot_callback,
                    motion_frame_callback=canary_motion_frame_callback if canary_video_root is not None else None,
                    post_lift_retention_gate=effective_retention_gate,
                )
                for record in retry_phase_audit:
                    record["grasp_retry_attempt"] = int(retry_index)
                phase_audit.extend(retry_phase_audit)
                setattr(env, "_arrow_phase_audit", phase_audit)
                retry_record["phase_statuses"] = [
                    {"phase": record.get("phase"), "status": record.get("status")}
                    for record in retry_phase_audit
                ]
                retry_record["contact_after_retry"] = _gripper_contact_likely(
                    retry_phase_audit, variant.grasp_contact_threshold
                )
                retry_record["status"] = (
                    "contact_likely" if retry_record["contact_after_retry"] else "completed_no_contact"
                )
                grasp_retry_audit.append(retry_record)
                if retry_record["contact_after_retry"]:
                    retry_record["selected"] = True
                    break
            except Exception as retry_exc:
                retry_record["status"] = "failed"
                retry_record["error_type"] = type(retry_exc).__name__
                retry_record["error"] = str(retry_exc)
                grasp_retry_audit.append(retry_record)
                # A failed retry can leave the simulator at a phase boundary;
                # do not hide the original completed trajectory behind it.
                break
    # Evaluator state is intentionally queried only after all motion phases.
    full_execution = stop_after_phase == "retreat"
    canary_video_audit = finalize_canary_video()
    evaluator_error: dict[str, str] | None = None
    if full_execution and not dry_run and retreat_completed_callback is not None:
        # This callback is the explicit handoff from motion to evaluation.  It
        # must run before evaluator() so an experimental adapter can gate the
        # evaluator without inferring completion from a returned audit.
        retreat_completed_callback()
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
    waypoints_have_pose = any(
        _orientation(value) is not None
        for value in (
            waypoints.values() if isinstance(waypoints, Mapping) else waypoints
        )
    )
    # Keep the historical list-of-XYZ audit schema for classical v1-v9
    # trajectories.  Pose waypoints use the additive serialized form so V10
    # orientation/frame provenance is retained without changing old hashes or
    # consumers.
    serialized_waypoints = _serialize_waypoints(waypoints)
    if not waypoints_have_pose and not isinstance(waypoints, Mapping):
        serialized_waypoints = np.asarray(waypoints, dtype=np.float64).tolist()
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
        "grasp_retries": grasp_retry_audit,
        "grasp_search": grasp_search_audit,
        "micro_corrections": micro_correction_audit,
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
        "experimental_grasp": getattr(env, "_arrow_experimental_candidate", None),
        "experimental_action_budget": (
            {"limit": int(experimental_action_budget.limit), "used": int(experimental_action_budget.used),
             "remaining": int(experimental_action_budget.remaining)}
            if experimental_action_budget is not None else None
        ),
        "experimental_eef_orientation_transform": (
            np.asarray(experimental_eef_orientation_transform, dtype=np.float64).tolist()
            if experimental_eef_orientation_transform is not None else None
        ),
        "experimental_aperture_m": experimental_aperture_m,
        "experimental_measured_opening_m": measured_opening,
        "source_approach": source_approach_audit,
        "support_plane": support_plane_audit,
        "placement_observable": (
            getattr(env, "_arrow_placement_observable", None)
            if destination_policy is not None and bool(destination_policy.get("enabled"))
            else None
        ),
        "classical_destination_release_target_m": classical_destination_point.tolist(),
        "waypoints_world_m": serialized_waypoints,
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
        "motion_trace": list(getattr(env, "_arrow_motion_trace", []) or []),
        "motion_trace_max_steps": int(motion_trace_max_steps),
        "motion_trace_truncated": bool(getattr(env, "_arrow_motion_trace_truncated", False)),
        "motion_trace_path": getattr(env, "_arrow_motion_trace_path", None),
        "failure_snapshots": list(getattr(env, "_arrow_failure_snapshots", []) or []),
        "post_lift_retention_gate": {
            "enabled": bool(effective_retention_gate is not None),
            "records": [
                record.get("retention_gate") for record in phase_audit
                if isinstance(record.get("retention_gate"), Mapping)
            ],
        },
        "canary_video": canary_video_audit,
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
    extra_camera_names: Sequence[str] = (),
) -> Any:
    """Construct direct LIBERO OffScreenRenderEnv with explicit suite semantics."""
    if isinstance(controller_variant, ControllerVariantConfig):
        if controller_variant.name != DEFAULT_PROFILE_NAME:
            raise ControllerConfigError(
                "only the active v9d controller may construct a runtime environment; "
                f"got retired controller {controller_variant.name!r}"
            )
    elif controller_variant is not None:
        requested_name = str(controller_variant).strip()
        if requested_name not in {"default", DEFAULT_PROFILE_NAME, DEFAULT_CONTROLLER_CONFIG_FILENAME}:
            raise ControllerConfigError(
                "only the active v9d controller may construct a runtime environment; "
                f"got retired controller {requested_name!r}"
            )
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
    camera_names = [CAMERA_NAME]
    for extra_camera_name in extra_camera_names:
        name = str(extra_camera_name).strip()
        if not name:
            raise ValueError("extra_camera_names must not contain empty names")
        if name not in camera_names:
            camera_names.append(name)
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
                camera_names=camera_names,
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
        camera_names=camera_names,
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
            DEFAULT_PROFILE_NAME,
        ),
        default="default",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help="external JSON controller config (supports relative extends); mutually exclusive with a non-default --controller-variant",
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
    if args.controller_config is not None and args.controller_variant != "default":
        raise SystemExit("--controller-config cannot be combined with a non-default --controller-variant")
    external_variant = None
    if args.controller_config is not None:
        # Resolve and validate before constructing an environment or sending
        # any action.  Invalid explicit configs never fall back to v8.
        external_variant = controller_variant_from_config(
            load_controller_config(args.controller_config), suite_mode=args.suite_mode
        )
        if external_variant.name != DEFAULT_PROFILE_NAME:
            raise SystemExit(
                "retired controller selection rejected; use "
                f"{DEFAULT_CONTROLLER_CONFIG_FILENAME}"
            )
    elif args.controller_variant in {"default", DEFAULT_PROFILE_NAME}:
        # No-config execution must use the same fully expanded v9d policy as
        # an explicit config path, including its real source and semantic hash.
        external_variant = controller_variant_from_config(
            load_controller_config(), suite_mode=args.suite_mode
        )
    else:
        # Historical v1-v9/v10 labels remain useful in archived provenance,
        # but are not executable runtime selections anymore.
        raise SystemExit(
            "retired controller selection rejected; use "
            f"{DEFAULT_CONTROLLER_CONFIG_FILENAME}"
        )
    # The checked-in v9d document is the sole executable policy.  Its expanded
    # values are authoritative; command-line knobs remain for historical
    # parsing compatibility but cannot silently create a second controller.
    selected_variant = external_variant
    if selected_variant is None:  # defensive: both resolution branches above must set it
        raise ControllerConfigError("active v9d controller could not be resolved")
    env = build_libero_env(
        args.task, args.seed, args.resolution, suite_mode=args.suite_mode,
        controller_variant=selected_variant,
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
            controller_variant=selected_variant,
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
