"""Single-source access to the frozen canonical grasp policy."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Final

try:
    from vla_benchmarking.arrow_grasp_controller.configs import (
        ACTIVE_CONTROLLER_NAME,
        ACTIVE_POLICY_LOCK_PATH,
        load_controller_config,
    )
except ImportError:  # pragma: no cover - direct package use
    from vla_benchmarking.arrow_grasp_controller.configs import (
        ACTIVE_CONTROLLER_NAME,
        ACTIVE_POLICY_LOCK_PATH,
        load_controller_config,
    )


def _load_frozen_policy() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_controller_config()
    lock = json.loads(ACTIVE_POLICY_LOCK_PATH.read_text(encoding="utf-8"))
    if config["name"] != ACTIVE_CONTROLLER_NAME:
        raise RuntimeError("canonical policy name does not match the active controller")
    if config["config_hash"] != lock.get("canonical_config_sha256"):
        raise RuntimeError("canonical policy hash does not match active_policy.lock.json")
    metadata = config.get("policy_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("canonical policy_metadata is missing")
    if metadata.get("policy_id") != lock.get("policy_id"):
        raise RuntimeError("canonical policy identity does not match active_policy.lock.json")
    return config, lock


_CONFIG, _LOCK = _load_frozen_policy()
_SETTINGS = deepcopy(_CONFIG["policy_metadata"])

CANONICAL_POLICY_NAME: Final = str(_CONFIG["name"])
CANONICAL_POLICY_ID: Final = str(_SETTINGS["policy_id"])
SOURCE_TREATMENT_ID: Final = str(_SETTINGS["source_treatment_id"])
SOURCE_RELEASE_COMMIT: Final = str(_LOCK["source_release_commit"])
CANONICAL_RESULT: Final = str(_SETTINGS["result"])
CANONICAL_CAMERA: Final = str(_SETTINGS["camera"])
CANONICAL_REGION_BACKEND: Final = str(_SETTINGS["region_backend"])
CANONICAL_CANDIDATE_POLICY: Final = str(_SETTINGS["candidate_policy"])
MOLMOPOINT_MODEL_ID: Final = str(_SETTINGS["molmopoint_model"])
MOLMOPOINT_MODEL_REVISION: Final = str(_SETTINGS["molmopoint_revision"])
MOLMOPOINT_PROMPT_ID: Final = str(_SETTINGS["molmopoint_prompt_id"])
MOLMOPOINT_PROMPT: Final = str(_SETTINGS["molmopoint_prompt"])
TRANSFORMERS_VERSION: Final = str(_SETTINGS["transformers_version"])
MODEL_DTYPE: Final = str(_SETTINGS["model_dtype"])
MODEL_DEVICE: Final = str(_SETTINGS["model_device"])
MODEL_DEVICE_MAP: Final = str(_SETTINGS["model_device_map"])
MOLMOPOINT_MAX_NEW_TOKENS: Final = int(_SETTINGS["molmopoint_max_new_tokens"])
MOLMOPOINT_MAX_POINTS: Final = int(_SETTINGS["molmopoint_max_points"])
MOLMOPOINT_PADDING_SIDE: Final = str(_SETTINGS["molmopoint_padding_side"])
MASK_HIGHLIGHT_ALPHA: Final = float(_SETTINGS["mask_highlight_alpha"])

YAW_OFFSETS_DEG: Final = tuple(float(v) for v in _SETTINGS["yaw_offsets_deg"])
INSERTION_OFFSETS_M: Final = tuple(float(v) for v in _SETTINGS["insertion_offsets_m"])
MAX_SPATIAL_SEEDS: Final = int(_SETTINGS["max_spatial_seeds"])
MAX_CANDIDATES: Final = int(_SETTINGS["max_candidates"])
MAX_GRASP_ATTEMPTS: Final = int(_SETTINGS["max_grasp_attempts"])
PHASE_TIMEOUT_STEPS: Final = int(_SETTINGS["phase_timeout_steps"])
SHARED_ACTION_BUDGET: Final = int(_SETTINGS["shared_action_budget"])
PRESHAPE_TARGET_M: Final = float(_SETTINGS["preshape_target_m"])
PRESHAPE_BAND_M: Final = tuple(float(v) for v in _SETTINGS["preshape_band_m"])
PRESHAPE_STABILITY_M: Final = float(_SETTINGS["preshape_stability_m"])
PRESHAPE_SETTLE_STEPS: Final = int(_SETTINGS["preshape_settle_steps"])
PRESHAPE_MAX_ACTIONS: Final = int(_SETTINGS["preshape_max_actions"])
PRESHAPE_MAX_POSE_DRIFT_M: Final = float(_SETTINGS["preshape_max_pose_drift_m"])
RELEASE_HEIGHT_OFFSET_M: Final = float(_SETTINGS["release_height_offset_m"])
RETREAT_HEIGHT_OFFSET_M: Final = float(_SETTINGS["retreat_height_offset_m"])
RETREAT_TOLERANCE_M: Final = float(_SETTINGS["retreat_tolerance_m"])
RETRY_RETREAT_DISTANCE_M: Final = float(_SETTINGS["retry_retreat_distance_m"])
OBSERVATION_PROFILE: Final = str(_SETTINGS["observation_profile"])
ARROW_REFRESH_POLICY: Final = str(_SETTINGS["arrow_refresh_policy"])
ARROW_SHORT_SPAN_THRESHOLD_PX: Final = int(_SETTINGS["arrow_short_span_threshold_px"])
ARROW_SHORT_LINE_WIDTH: Final = int(_SETTINGS["arrow_short_line_width"])
ARROW_SHORT_HEAD_MIN_PX: Final = int(_SETTINGS["arrow_short_head_min_px"])
ARROW_SHORT_HEAD_MAX_PX: Final = int(_SETTINGS["arrow_short_head_max_px"])
ARROW_SHORT_HEAD_FRACTION: Final = float(_SETTINGS["arrow_short_head_fraction"])
ARROW_LONG_LINE_WIDTH: Final = int(_SETTINGS["arrow_long_line_width"])
ARROW_LONG_HEAD_LENGTH_PX: Final = int(_SETTINGS["arrow_long_head_length_px"])
MOLMO_SNAP_RADIUS_PX: Final = int(_SETTINGS["molmo_snap_radius_px"])
LOCAL_SEED_RADIUS_PX: Final = int(_SETTINGS["local_seed_radius_px"])
RIM_SUPPORT_RADIUS_PX: Final = int(_SETTINGS["rim_support_radius_px"])
RIM_HEIGHT_QUANTILE: Final = float(_SETTINGS["rim_height_quantile"])
RIM_HEIGHT_BAND_M: Final = float(_SETTINGS["rim_height_band_m"])
RIM_LOCAL_RADIUS_M: Final = float(_SETTINGS["rim_local_radius_m"])
MIN_RIM_SUPPORT_PIXELS: Final = int(_SETTINGS["min_rim_support_pixels"])
MIN_DEPTH_SUPPORT_PIXELS: Final = int(_SETTINGS["min_depth_support_pixels"])
DEDUPE_POSITION_M: Final = float(_SETTINGS["dedupe_position_m"])
DEDUPE_ROTATION_DEG: Final = float(_SETTINGS["dedupe_rotation_deg"])
OBSTRUCTION_CLEARANCE_M: Final = float(_SETTINGS["obstruction_clearance_m"])
PREGRASP_DISTANCE_M: Final = float(_SETTINGS["pregrasp_distance_m"])
LIFT_CLEARANCE_M: Final = float(_SETTINGS["lift_clearance_m"])
MIN_APERTURE_M: Final = float(_SETTINGS["min_aperture_m"])
MAX_APERTURE_M: Final = float(_SETTINGS["max_aperture_m"])
FINGER_CLEARANCE_M: Final = float(_SETTINGS["finger_clearance_m"])
CONTACT_MODE: Final = str(_SETTINGS["contact_mode"])
TERMINAL_CONTACT_ALLOWANCE_M: Final = float(_SETTINGS["terminal_contact_allowance_m"])
LEGACY_SOURCE_OFFSET_M: Final = tuple(float(v) for v in _SETTINGS["legacy_source_offset_m"])
LEGACY_DESTINATION_OFFSET_M: Final = tuple(float(v) for v in _SETTINGS["legacy_destination_offset_m"])
EMPTY_GRIPPER_THRESHOLD: Final = float(_CONFIG["grasp_search"]["empty_gripper_threshold"])
WORKSPACE_MIN_M: Final = tuple(
    float(_CONFIG["workspace_bounds_m"][axis][0]) for axis in ("x", "y", "z")
)
WORKSPACE_MAX_M: Final = tuple(
    float(_CONFIG["workspace_bounds_m"][axis][1]) for axis in ("x", "y", "z")
)


def canonical_settings() -> dict[str, object]:
    """Return a detached JSON-safe snapshot for manifests and handoffs."""

    return deepcopy(_SETTINGS)


def canonical_candidate_policy() -> Any:
    """Build geometry settings without relying on dataclass defaults."""

    from .grasp_candidates import CandidatePolicy

    return CandidatePolicy(
        name=CANONICAL_CANDIDATE_POLICY,
        max_seeds=MAX_SPATIAL_SEEDS,
        max_candidates=MAX_CANDIDATES,
        molmo_snap_radius_px=MOLMO_SNAP_RADIUS_PX,
        local_seed_radius_px=LOCAL_SEED_RADIUS_PX,
        rim_support_radius_px=RIM_SUPPORT_RADIUS_PX,
        contact_mode=CONTACT_MODE,
        rim_height_quantile=RIM_HEIGHT_QUANTILE,
        rim_height_band_m=RIM_HEIGHT_BAND_M,
        rim_local_radius_m=RIM_LOCAL_RADIUS_M,
        min_rim_support_pixels=MIN_RIM_SUPPORT_PIXELS,
        min_depth_support_pixels=MIN_DEPTH_SUPPORT_PIXELS,
        dedupe_position_m=DEDUPE_POSITION_M,
        dedupe_rotation_deg=DEDUPE_ROTATION_DEG,
        yaw_offsets_deg=YAW_OFFSETS_DEG,
        insertion_depths_m=INSERTION_OFFSETS_M,
        obstruction_clearance_m=OBSTRUCTION_CLEARANCE_M,
        robot_exclusion_clearance_m=None,
        terminal_contact_allowance_m=TERMINAL_CONTACT_ALLOWANCE_M,
    )


def canonical_region_kwargs() -> dict[str, object]:
    """Map the canonical controller's arrow-seeded RGB-D region settings."""

    search = _CONFIG["grasp_search"]
    return {
        "region_radius_m": float(search["region_radius_m"]),
        "depth_tolerance_m": float(search["region_depth_tolerance_m"]),
        "min_region_pixels": int(search["region_min_pixels"]),
        "max_region_fraction": float(search["region_max_fraction"]),
        "seed_radius_px": int(search["region_seed_radius_px"]),
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_candidate_policy",
    "canonical_region_kwargs",
    "canonical_settings",
]
