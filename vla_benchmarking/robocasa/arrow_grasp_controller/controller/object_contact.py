"""Molmo contact-seed adaptation for the RoboCasa RGB-D controller.

The object-contact arm changes only proposal derivation.  It never receives
task state, simulator poses, masks, or evaluator information: Molmo points
are admitted against the arrow-tail RGB-D anchor and each admitted point gets
its own small observed RGB-D support mask before the canonical geometry engine
is called.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .grasp_candidates import (
    CandidatePolicy,
    CandidateRejection,
    GraspCandidate,
    GraspCandidateResult,
    MolmoPoint,
    _quaternion_xyzw,
    _support_points,
    generate_grasp_candidates,
)
from .policy import MAX_SPATIAL_SEEDS, canonical_candidate_policy


PROFILE_NAME = "object_contact_v1"
_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "object_contact_v1.json"
PROFILE_V2_NAME = "object_contact_v2"
_PROFILE_V2_PATH = Path(__file__).resolve().parents[1] / "configs" / "object_contact_v2.json"
PROFILE_V3_NAME = "object_contact_v3"
_PROFILE_V3_PATH = Path(__file__).resolve().parents[1] / "configs" / "object_contact_v3.json"
PROFILE_V4_NAME = "object_contact_v4"
_PROFILE_V4_PATH = Path(__file__).resolve().parents[1] / "configs" / "object_contact_v4.json"
PROFILE_V5_NAME = "object_contact_v5"
_PROFILE_V5_PATH = Path(__file__).resolve().parents[1] / "configs" / "object_contact_v5.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def object_contact_profile_path() -> Path:
    return _PROFILE_PATH


def object_contact_profile_sha256(path: str | Path | None = None) -> str:
    """Hash the immutable profile bytes for experiment provenance."""

    profile_path = Path(path) if path is not None else _PROFILE_PATH
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def object_contact_profile() -> dict[str, Any]:
    with _PROFILE_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("name") != PROFILE_NAME:
        raise ValueError("object contact profile has an invalid name")
    return value


def _validate_object_contact_profile(prompt: str) -> dict[str, Any]:
    """Fail closed if the descriptive profile drifts from executable policy."""

    profile = object_contact_profile()
    if profile.get("base_profile") != "canonical_molmo_rgbd_grasp":
        raise ValueError("object contact profile base_profile is not canonical")
    try:
        from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
        expected_prompt_template = object_contact_prompt("{noun}")
    except ImportError:  # pragma: no cover - direct script use
        expected_prompt_template = (
            "Point to visible grasp contact locations on the {noun} at the start of the green arrow "
            "where a parallel-jaw gripper can descend from above, straddle the object, and close "
            "without touching nearby objects or support surfaces. Point only on the {noun}."
        )
    if profile.get("prompt") != expected_prompt_template:
        raise ValueError("object contact profile prompt does not match the executable prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("object_contact_v1 requires an explicit object-contact prompt")
    template_parts = expected_prompt_template.split("{noun}")
    if len(template_parts) != 3:
        raise ValueError("object contact executable prompt template must contain two noun slots")
    prompt_pattern = (
        re.escape(template_parts[0])
        + r"(?P<noun>[^\r\n]+?)"
        + re.escape(template_parts[1])
        + r"(?P=noun)"
        + re.escape(template_parts[2])
    )
    prompt_match = re.fullmatch(prompt_pattern, prompt)
    if prompt_match is None or " ".join(prompt_match.group("noun").split()) != prompt_match.group("noun"):
        raise ValueError("object contact prompt must instantiate the validated noun template")
    expected_labels = {
        "model_seed": "molmo_contact",
        "empty_model_fallback": "arrow_tail_anchor",
        "local_policy": "molmo_local",
    }
    if profile.get("derivation_labels") != expected_labels:
        raise ValueError("object contact profile derivation labels do not match runtime")
    expected_factors = [
        "MolmoPoint is queried on the same-frame green-arrow RGB image without a mask",
        "decoded points are admitted by a finite RGB-D world-distance gate from the arrow tail",
        "each admitted point receives an independent 0.015 m local RGB-D support patch",
    ]
    if profile.get("changed_factors") != expected_factors:
        raise ValueError("object contact profile changed factors do not match runtime")
    if profile.get("arrow_color_rgb") != [0, 166, 107]:
        raise ValueError("object contact profile arrow color does not match the green arrow")
    local = profile.get("local_geometry")
    canonical = canonical_candidate_policy()
    if not isinstance(local, Mapping):
        raise ValueError("object contact profile local_geometry is missing")
    for key in ("rim_local_radius_m", "rim_height_band_m"):
        if not np.isclose(float(local.get(key)), float(getattr(canonical, key))):
            raise ValueError(f"object contact profile {key} does not match canonical policy")
    for key in ("min_rim_support_pixels", "min_depth_support_pixels"):
        if int(local.get(key)) != int(getattr(canonical, key)):
            raise ValueError(f"object contact profile {key} does not match canonical policy")
    gate = profile.get("admission_gate")
    if not isinstance(gate, Mapping) or gate.get("distance") != "finite same-frame RGB-D world distance" or gate.get("threshold") != "0.5 * robot_calibration.max_aperture_m + robot_calibration.finger_clearance_m":
        raise ValueError("object contact profile admission gate does not match runtime")
    if profile.get("candidate_engine") != "generate_grasp_candidates" or profile.get("training") != "none":
        raise ValueError("object contact profile candidate engine or training marker is invalid")
    return profile


def object_contact_v2_profile_path() -> Path:
    return _PROFILE_V2_PATH


def object_contact_v2_profile_sha256(path: str | Path | None = None) -> str:
    profile_path = Path(path) if path is not None else _PROFILE_V2_PATH
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def object_contact_v2_profile() -> dict[str, Any]:
    with _PROFILE_V2_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("name") != PROFILE_V2_NAME:
        raise ValueError("object contact v2 profile has an invalid name")
    return value


def _validate_object_contact_v2_profile(prompt: str) -> dict[str, Any]:
    profile = object_contact_v2_profile()
    if profile.get("base_profile") != "canonical_molmo_rgbd_grasp":
        raise ValueError("object contact v2 base_profile is not canonical")
    from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
    expected_prompt = object_contact_prompt("{noun}")
    if profile.get("prompt") != expected_prompt:
        raise ValueError("object contact v2 prompt does not match executable prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("object_contact_v2 requires an explicit object-contact prompt")
    parts = expected_prompt.split("{noun}")
    pattern = re.escape(parts[0]) + r"(?P<noun>[^\r\n]+?)" + re.escape(parts[1]) + r"(?P=noun)" + re.escape(parts[2])
    match = re.fullmatch(pattern, prompt)
    if match is None or " ".join(match.group("noun").split()) != match.group("noun"):
        raise ValueError("object contact v2 prompt must instantiate the validated noun template")
    expected_labels = {
        "model_seed": "molmo_contact",
        "empty_model_fallback": "arrow_tail_anchor",
        "all_model_points_rejected_fallback": "arrow_tail_anchor_all_model_points_rejected",
        "local_policy": "molmo_local",
        "approach_hypotheses": ["world_down", "camera_ray"],
    }
    if profile.get("derivation_labels") != expected_labels:
        raise ValueError("object contact v2 derivation labels do not match runtime")
    approach = profile.get("approach_hypotheses")
    if not isinstance(approach, Mapping) or approach.get("camera_ray") != "camera_origin_to_contact" or float(approach.get("max_tilt_deg", -1)) != 55.0:
        raise ValueError("object contact v2 approach hypotheses do not match runtime")
    gate = profile.get("admission_gate")
    if not isinstance(gate, Mapping) or gate.get("distance") != "finite same-frame RGB-D world distance":
        raise ValueError("object contact v2 admission gate does not match runtime")
    if profile.get("candidate_engine") != "generate_grasp_candidates" or profile.get("training") != "none":
        raise ValueError("object contact v2 candidate engine or training marker is invalid")
    return profile


def object_contact_v3_profile_path() -> Path:
    return _PROFILE_V3_PATH


def object_contact_v3_profile_sha256(path: str | Path | None = None) -> str:
    profile_path = Path(path) if path is not None else _PROFILE_V3_PATH
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def object_contact_v3_profile() -> dict[str, Any]:
    with _PROFILE_V3_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("name") != PROFILE_V3_NAME:
        raise ValueError("object contact v3 profile has an invalid name")
    return value


def _validate_object_contact_v3_profile(prompt: str) -> dict[str, Any]:
    profile = object_contact_v3_profile()
    if profile.get("base_profile") != "canonical_molmo_rgbd_grasp":
        raise ValueError("object contact v3 base_profile is not canonical")
    from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
    expected_prompt = object_contact_prompt("{noun}")
    if profile.get("prompt") != expected_prompt:
        raise ValueError("object contact v3 prompt does not match executable prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("object_contact_v3 requires an explicit object-contact prompt")
    parts = expected_prompt.split("{noun}")
    pattern = re.escape(parts[0]) + r"(?P<noun>[^\r\n]+?)" + re.escape(parts[1]) + r"(?P=noun)" + re.escape(parts[2])
    match = re.fullmatch(pattern, prompt)
    if match is None or " ".join(match.group("noun").split()) != match.group("noun"):
        raise ValueError("object contact v3 prompt must instantiate the validated noun template")
    labels = profile.get("derivation_labels")
    expected_labels = {
        "model_seed": "molmo_contact",
        "empty_model_fallback": "arrow_tail_anchor",
        "all_model_points_rejected_fallback": "arrow_tail_anchor_all_model_points_rejected",
        "surface_promotion": "promoted_upper_surface",
        "approach_hypotheses": ["world_down", "camera_ray"],
    }
    if labels != expected_labels:
        raise ValueError("object contact v3 derivation labels do not match runtime")
    if profile.get("candidate_engine") != "generate_grasp_candidates" or profile.get("training") != "none":
        raise ValueError("object contact v3 candidate engine or training marker is invalid")
    promotion = profile.get("surface_promotion")
    if not isinstance(promotion, Mapping):
        raise ValueError("object contact v3 surface promotion is missing")
    expected = {
        "max_snap_radius_px": 12,
        "max_displacement_m": 0.015,
        "connectivity": 8,
        "upper_band_m": 0.008,
        "component_radius": "0.5 * robot_calibration.max_aperture_m",
        "terminal_contact_allowance": "0.5 * robot_calibration.max_aperture_m",
        "max_component_pixels": 512,
        "max_component_fraction": 0.05,
        "reject_outer_annulus": True,
        "reject_image_boundary": True,
        "adaptive_retries": [
            {"component_radius_m": "min(0.5 * robot_calibration.max_aperture_m, 0.012)", "upper_band_m": 0.004},
            {"component_radius_m": "min(0.5 * robot_calibration.max_aperture_m, 0.009)", "upper_band_m": 0.003},
        ],
    }
    if any(promotion.get(key) != value for key, value in expected.items()):
        raise ValueError("object contact v3 surface promotion does not match runtime")
    approach = profile.get("approach_hypotheses")
    if not isinstance(approach, Mapping) or approach.get("camera_ray") != "camera_origin_to_contact" or float(approach.get("max_tilt_deg", -1)) != 55.0:
        raise ValueError("object contact v3 approach hypotheses do not match runtime")
    return profile


def object_contact_v4_profile_path() -> Path:
    return _PROFILE_V4_PATH


def object_contact_v4_profile_sha256(path: str | Path | None = None) -> str:
    profile_path = Path(path) if path is not None else _PROFILE_V4_PATH
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def object_contact_v4_profile() -> dict[str, Any]:
    with _PROFILE_V4_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("name") != PROFILE_V4_NAME:
        raise ValueError("object contact v4 profile has an invalid name")
    return value


def _validate_object_contact_v4_profile(prompt: str) -> dict[str, Any]:
    """Validate the new geometry/contact-mask split against its profile."""
    profile = object_contact_v4_profile()
    if profile.get("base_profile") != "object_contact_v3":
        raise ValueError("object contact v4 base_profile is not object_contact_v3")
    from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
    expected_prompt = object_contact_prompt("{noun}")
    if profile.get("prompt") != expected_prompt:
        raise ValueError("object contact v4 prompt does not match executable prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("object_contact_v4 requires an explicit object-contact prompt")
    parts = expected_prompt.split("{noun}")
    pattern = re.escape(parts[0]) + r"(?P<noun>[^\r\n]+?)" + re.escape(parts[1]) + r"(?P=noun)" + re.escape(parts[2])
    match = re.fullmatch(pattern, prompt)
    if match is None or " ".join(match.group("noun").split()) != match.group("noun"):
        raise ValueError("object contact v4 prompt must instantiate the validated noun template")
    expected_labels = {
        "model_seed": "molmo_contact",
        "empty_model_fallback": "arrow_tail_anchor",
        "all_model_points_rejected_fallback": "arrow_tail_anchor_all_model_points_rejected",
        "surface_promotion": "promoted_upper_surface",
        "approach_hypotheses": ["world_down", "camera_ray", "blocker_away"],
        "geometry_support": "outer_annulus_geometry_only",
        "contact_mask": "strict_seed_connected_component",
        "blocker_away_approach": "same_frame_rgbd_non_target_rejection",
    }
    if profile.get("derivation_labels") != expected_labels:
        raise ValueError("object contact v4 derivation labels do not match runtime")
    if profile.get("candidate_engine") != "generate_grasp_candidates" or profile.get("training") != "none":
        raise ValueError("object contact v4 candidate engine or training marker is invalid")
    split = profile.get("geometry_contact_split")
    if not isinstance(split, Mapping) or split.get("geometry_support") != "bounded_same_frame_outer_ring" or split.get("contact_mask") != "strict_seed_connected_component" or split.get("contact_exemption") != "measured_pad_prism_and_local_support_only":
        raise ValueError("object contact v4 geometry/contact split is invalid")
    approach = profile.get("approach_hypotheses")
    if (
        not isinstance(approach, Mapping)
        or approach.get("camera_ray") != "camera_origin_to_contact"
        or approach.get("blocker_away") != "same_frame_rgbd_non_target_rejection"
        or float(approach.get("max_tilt_deg", -1)) != 55.0
    ):
        raise ValueError("object contact v4 approach hypotheses do not match runtime")
    return profile


def object_contact_v5_profile_path() -> Path:
    return _PROFILE_V5_PATH


def object_contact_v5_profile_sha256(path: str | Path | None = None) -> str:
    profile_path = Path(path) if path is not None else _PROFILE_V5_PATH
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def object_contact_v5_profile() -> dict[str, Any]:
    with _PROFILE_V5_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("name") != PROFILE_V5_NAME:
        raise ValueError("object contact v5 profile has an invalid name")
    return value


def _validate_object_contact_v5_profile(prompt: str) -> dict[str, Any]:
    profile = object_contact_v5_profile()
    if profile.get("base_profile") != "object_contact_v4":
        raise ValueError("object contact v5 base_profile is not object_contact_v4")
    from vla_benchmarking.robocasa.evaluation.prompt import object_contact_prompt
    expected_prompt = object_contact_prompt("{noun}")
    if profile.get("prompt") != expected_prompt:
        raise ValueError("object contact v5 prompt does not match executable prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("object_contact_v5 requires an explicit object-contact prompt")
    parts = expected_prompt.split("{noun}")
    pattern = re.escape(parts[0]) + r"(?P<noun>[^\r\n]+?)" + re.escape(parts[1]) + r"(?P=noun)" + re.escape(parts[2])
    match = re.fullmatch(pattern, prompt)
    if match is None or " ".join(match.group("noun").split()) != match.group("noun"):
        raise ValueError("object contact v5 prompt must instantiate the validated noun template")
    expected_labels = {
        "model_seed": "molmo_contact",
        "empty_model_fallback": "arrow_tail_anchor",
        "all_model_points_rejected_fallback": "arrow_tail_anchor_all_model_points_rejected",
        "surface_promotion": "promoted_upper_surface",
        "approach_hypotheses": ["world_down", "camera_ray", "blocker_away_ladder"],
        "geometry_support": "outer_annulus_geometry_only",
        "contact_mask": "strict_seed_connected_component",
        "blocker_away_approach": "same_frame_rgbd_non_target_rejection",
    }
    if profile.get("derivation_labels") != expected_labels:
        raise ValueError("object contact v5 derivation labels do not match runtime")
    gate = profile.get("admission_gate")
    if not isinstance(gate, Mapping) or gate.get("distance") != "finite same-frame RGB-D image-metric ellipse" or gate.get("threshold") != "0.5 * robot_calibration.max_aperture_m + robot_calibration.finger_clearance_m":
        raise ValueError("object contact v5 admission gate does not match runtime")
    approach = profile.get("approach_hypotheses")
    if not isinstance(approach, Mapping) or approach.get("camera_ray") != "camera_origin_to_contact" or approach.get("blocker_away") != "same_frame_rgbd_non_target_rejection" or approach.get("blocker_away_tilts_deg") != [15.0, 30.0, 45.0] or float(approach.get("max_tilt_deg", -1)) != 45.0:
        raise ValueError("object contact v5 approach hypotheses do not match runtime")
    split = profile.get("geometry_contact_split")
    if not isinstance(split, Mapping) or split.get("geometry_support") != "bounded_same_frame_outer_ring" or split.get("contact_mask") != "strict_seed_connected_component" or split.get("contact_exemption") != "measured_pad_prism_and_local_support_only":
        raise ValueError("object contact v5 geometry/contact split is invalid")
    if profile.get("candidate_engine") != "generate_grasp_candidates" or profile.get("training") != "none":
        raise ValueError("object contact v5 candidate engine or training marker is invalid")
    return profile


def _camera_arrays(capture: Any) -> tuple[np.ndarray, np.ndarray]:
    calibration = capture.calibration
    K = np.asarray(calibration.intrinsic, dtype=np.float64)
    T = np.asarray(calibration.world_from_camera, dtype=np.float64)
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or T.shape != (4, 4):
        raise ValueError("object contact requires a calibrated RGB-D frame")
    if not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError("object contact calibration must be finite")
    if abs(float(K[0, 0])) <= 1e-9 or abs(float(K[1, 1])) <= 1e-9:
        raise ValueError("object contact calibration has zero focal length")
    return K, T


def _depth_at(capture: Any, uv: Sequence[float]) -> float:
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("contact seed lies outside the RGB-D frame")
    value = float(depth[v, u])
    if np.isfinite(value) and value > 0.0:
        return value
    y0, y1 = max(0, v - 2), min(depth.shape[0], v + 3)
    x0, x1 = max(0, u - 2), min(depth.shape[1], u + 3)
    values = depth[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values > 0.0)]
    if len(valid) == 0:
        raise ValueError("contact seed has no finite metric depth")
    return float(np.median(valid))


def _deproject(capture: Any, uv: Sequence[float]) -> np.ndarray:
    K, T = _camera_arrays(capture)
    u, v = float(uv[0]), float(uv[1])
    z = _depth_at(capture, (u, v))
    camera = np.asarray(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
    point = T[:3, :3] @ camera + T[:3, 3]
    if not np.isfinite(point).all():
        raise ValueError("contact seed deprojection is non-finite")
    return point


def _camera_ray_approach(
    capture: Any,
    point_world: Sequence[float],
    *,
    max_tilt_deg: float = 55.0,
) -> tuple[np.ndarray, float]:
    """Return the calibrated camera-to-contact ray when it is admissible.

    The ray is expressed in the frozen B0/world frame.  Upward and overly
    oblique rays are rejected before they enter the canonical geometry
    generator; this keeps the exploratory hypothesis bounded and auditable.
    """

    _, calibration_world_from_camera = _camera_arrays(capture)
    camera_origin = calibration_world_from_camera[:3, 3]
    ray = np.asarray(point_world, dtype=np.float64).reshape(3) - camera_origin
    ray_norm = float(np.linalg.norm(ray))
    if not np.isfinite(ray_norm) or ray_norm <= 1e-9:
        raise ValueError("degenerate_camera_ray")
    ray /= ray_norm
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(float(ray[2])), -1.0, 1.0))))
    if not np.isfinite(max_tilt_deg) or max_tilt_deg < 0.0:
        raise ValueError("camera_ray_max_tilt_invalid")
    if float(ray[2]) >= 0.0 or tilt_deg > float(max_tilt_deg) + 1e-9:
        raise ValueError("camera_ray_over_tilted_or_upward")
    return ray, tilt_deg


def _blocker_away_approach(
    capture: Any,
    *,
    contact_world: np.ndarray,
    rejection_details: Sequence[Mapping[str, Any]],
    max_tilt_deg: float = 55.0,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Derive one bounded approach from a measured non-target blocker."""
    for details in rejection_details:
        if details.get("reason") != "target_aware_approach_obstruction":
            continue
        if details.get("offender_in_target_component") is not False:
            continue
        primitive_name = str(details.get("primitive_name", "")).lower()
        if not any(token in primitive_name for token in ("finger", "pad")):
            continue
        try:
            segment_fraction = float(details.get("segment_fraction", float("nan")))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(segment_fraction) or segment_fraction > 0.75 + 1e-9:
            continue
        uv = details.get("closest_offending_pixel_uv")
        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
            continue
        try:
            blocker_world = _deproject(capture, (float(uv[0]), float(uv[1])))
        except (TypeError, ValueError):
            continue
        # The generator defines ``pregrasp = grip_site - approach * d``.
        # Therefore the approach axis must point from the contact toward the
        # blocker so that the pregrasp is displaced in the opposite direction,
        # away from the measured obstacle.
        delta = blocker_world - np.asarray(contact_world, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm <= 1e-6:
            continue
        # Point toward the blocker while adding a descending component. Since
        # pregrasp = grip - approach*d, this places pregrasp away from it.
        # This turns a horizontal rejection into a bounded tilted hypothesis
        # rather than admitting an unsafe sideways insertion.
        horizontal = np.asarray(delta, dtype=np.float64).copy()
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm > 1e-6:
            horizontal /= horizontal_norm
            desired_tilt = float(np.degrees(np.arctan2(horizontal_norm, max(abs(float(delta[2])), 1e-9))))
            tilt = min(float(max_tilt_deg), max(15.0, desired_tilt))
            tilt_rad = np.radians(tilt)
            axis = horizontal * np.sin(tilt_rad)
            axis[2] = -np.cos(tilt_rad)
        else:
            axis = delta / norm
            if float(axis[2]) >= 0.0:
                continue
            tilt = float(np.degrees(np.arccos(np.clip(abs(float(axis[2])), -1.0, 1.0))))
            if not np.isfinite(tilt) or tilt > float(max_tilt_deg) + 1e-9:
                continue
        return axis, {
            "approach_label": "blocker_away",
            "blocker_pixel_uv": [int(round(float(uv[0]))), int(round(float(uv[1])))],
            "blocker_world_m": blocker_world.tolist(),
            "approach_axis_world": axis.tolist(),
            "approach_tilt_deg": tilt,
            "max_approach_tilt_deg": float(max_tilt_deg),
            "basis": "same_frame_rgbd_non_target_rejection",
            "temporal_gate": "same_frame_rgbd",
        }
    return None


def _blocker_away_recovery_allowed(outcomes: Sequence[Mapping[str, Any]]) -> bool:
    """Require both v4 primary hypotheses to be executed and blocked."""
    by_label = {str(item.get("approach_label")): item for item in outcomes}
    required = ("world_down", "camera_ray")
    if any(label not in by_label for label in required):
        return False
    return all(
        bool(by_label[label].get("executed"))
        and int(by_label[label].get("survivor_count", 0)) == 0
        and bool(by_label[label].get("non_target_obstruction"))
        for label in required
    )


def _blocker_away_ladder(
    capture: Any,
    *,
    contact_world: np.ndarray,
    rejection_details: Sequence[Mapping[str, Any]],
    tilts_deg: Sequence[float] = (15.0, 30.0, 45.0),
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Build deterministic low-tilt blocker-away hypotheses."""
    blocker = _blocker_away_approach(
        capture, contact_world=contact_world,
        rejection_details=rejection_details, max_tilt_deg=45.0,
    )
    if blocker is None:
        return []
    base_axis, base_audit = blocker
    horizontal = np.asarray(base_axis, dtype=np.float64).copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    if not np.isfinite(horizontal_norm) or horizontal_norm <= 1e-6:
        return []
    horizontal /= horizontal_norm
    result: list[tuple[np.ndarray, dict[str, Any]]] = []
    for tilt_value in tilts_deg:
        tilt = float(tilt_value)
        if not np.isfinite(tilt) or tilt <= 0.0 or tilt > 45.0:
            continue
        angle = np.radians(tilt)
        axis = horizontal * np.sin(angle)
        axis[2] = -np.cos(angle)
        result.append((axis, {
            **base_audit,
            "approach_label": "blocker_away_ladder",
            "ladder_tilt_deg": tilt,
            "approach_axis_world": axis.tolist(),
            "max_approach_tilt_deg": 45.0,
        }))
    return result


def _point_cloud(capture: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return all finite RGB-D world points and their (v,u) pixels."""

    K, T = _camera_arrays(capture)
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    pixels = np.column_stack(np.nonzero(valid))
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float64), pixels
    v = pixels[:, 0].astype(np.float64)
    u = pixels[:, 1].astype(np.float64)
    z = depth[pixels[:, 0], pixels[:, 1]]
    camera = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
    world = camera @ T[:3, :3].T + T[:3, 3]
    keep = np.isfinite(world).all(axis=1)
    return world[keep], pixels[keep]


def _local_rgbd_mask(
    capture: Any,
    seed_world: np.ndarray,
    *,
    radius_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points, pixels = _point_cloud(capture)
    mask = np.zeros(np.asarray(capture.metric_depth).shape, dtype=bool)
    if len(points):
        local = np.linalg.norm(points - seed_world[None, :], axis=1) <= float(radius_m) + 1e-12
        chosen = pixels[local]
        mask[chosen[:, 0], chosen[:, 1]] = True
    return mask, {
        "point_count": int(len(points)),
        "local_support_pixels": int(mask.sum()),
        "radius_m": float(radius_m),
    }


def _connected_component_8(seed_vu: tuple[int, int], eligible: np.ndarray) -> np.ndarray:
    """Return one deterministic 8-connected component from a boolean image."""
    if eligible.ndim != 2 or not (0 <= seed_vu[0] < eligible.shape[0] and 0 <= seed_vu[1] < eligible.shape[1]):
        return np.zeros_like(eligible, dtype=bool)
    if not bool(eligible[seed_vu]):
        return np.zeros_like(eligible, dtype=bool)
    result = np.zeros_like(eligible, dtype=bool)
    stack = [seed_vu]
    result[seed_vu] = True
    while stack:
        v, u = stack.pop()
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dv == 0 and du == 0:
                    continue
                nv, nu = v + dv, u + du
                if 0 <= nv < eligible.shape[0] and 0 <= nu < eligible.shape[1] and eligible[nv, nu] and not result[nv, nu]:
                    result[nv, nu] = True
                    stack.append((nv, nu))
    return result


def _promote_upper_surface(
    capture: Any,
    *,
    seed_uv: Sequence[float],
    seed_world: np.ndarray,
    max_snap_radius_px: int = 12,
    max_displacement_m: float = 0.015,
    upper_band_m: float = 0.008,
    component_radius_m: float = 0.0195,
    max_component_pixels: int = 512,
    max_component_fraction: float = 0.05,
    reject_outer_annulus: bool = True,
    select_generator_anchor: bool = False,
) -> tuple[MolmoPoint, np.ndarray, np.ndarray, dict[str, Any]]:
    """Promote an arrow/model seed to a bounded visible upper component.

    All operations use the current RGB-D capture. The component is deliberately
    local and top-surface conditioned, so a broad coplanar counter cannot become
    the target through nearest-point or simulator-state recovery.
    """
    K, T = _camera_arrays(capture)
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    height, width = depth.shape
    u0, v0 = float(seed_uv[0]), float(seed_uv[1])
    if not np.isfinite((u0, v0)).all() or not (0.0 <= u0 < width and 0.0 <= v0 < height):
        raise ValueError("surface promotion seed lies outside the RGB-D frame")
    if not np.isfinite(seed_world).all():
        raise ValueError("surface promotion seed world point is non-finite")
    if max_snap_radius_px <= 0 or max_component_pixels <= 0 or not 0.0 < max_component_fraction <= 1.0:
        raise ValueError("surface promotion bounds are invalid")
    if not np.isfinite((max_displacement_m, upper_band_m, component_radius_m)).all() or min(max_displacement_m, upper_band_m, component_radius_m) <= 0:
        raise ValueError("surface promotion metric bounds are invalid")
    points, pixels = _point_cloud(capture)
    if len(points) == 0:
        raise ValueError("surface promotion has no finite RGB-D points")
    # Build a same-frame world-point image for deterministic pixel-local tests.
    world_image = np.full((height, width, 3), np.nan, dtype=np.float64)
    world_image[pixels[:, 0], pixels[:, 1]] = points
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(world_image).all(axis=2)
    yy, xx = np.indices((height, width))
    pixel_distance = np.hypot(xx - u0, yy - v0)
    snap = valid & (pixel_distance <= float(max_snap_radius_px) + 1e-9)
    snap_pixels = np.column_stack(np.nonzero(snap))
    if len(snap_pixels) == 0:
        raise ValueError("surface promotion has no valid pixels within snap radius")
    snap_world = world_image[snap]
    displacement = np.linalg.norm(snap_world - seed_world[None, :], axis=1)
    admitted = np.isfinite(displacement) & (displacement <= float(max_displacement_m) + 1e-12)
    if not np.any(admitted):
        raise ValueError("surface promotion exceeds metric displacement cap")
    # Highest world-z point is the observable upper surface under the inherited
    # RoboCasa vertical convention. Ties use row/column order for determinism.
    candidates = snap_pixels[admitted]
    candidate_world = snap_world[admitted]
    candidate_disp = displacement[admitted]
    ordering = np.lexsort((candidates[:, 1], candidates[:, 0], candidate_disp, -candidate_world[:, 2]))
    promoted_v, promoted_u = (int(value) for value in candidates[int(ordering[0])])
    promoted_world = world_image[promoted_v, promoted_u].copy()
    promoted_displacement = float(np.linalg.norm(promoted_world - seed_world))
    if not np.isfinite(promoted_displacement) or promoted_displacement > float(max_displacement_m) + 1e-12:
        raise ValueError("surface promotion exceeds metric displacement cap")
    radius = np.linalg.norm(world_image - promoted_world[None, None, :], axis=2)
    upper = world_image[:, :, 2] >= float(promoted_world[2]) - float(upper_band_m) - 1e-12
    neighborhood = valid & np.isfinite(radius) & (radius <= float(component_radius_m) + 1e-12)
    eligible = neighborhood & upper
    component = _connected_component_8((promoted_v, promoted_u), eligible)
    if not component[promoted_v, promoted_u]:
        raise ValueError("surface promotion component does not contain promoted seed")
    area = int(component.sum())
    fraction = float(area / max(height * width, 1))
    if area > int(max_component_pixels) or fraction > float(max_component_fraction):
        raise ValueError("surface promotion component exceeds strict area or fraction cap")
    touches_image_boundary = bool(
        np.any(component[0]) or np.any(component[-1])
        or np.any(component[:, 0]) or np.any(component[:, -1])
    )
    if touches_image_boundary:
        raise ValueError("surface promotion component touches image boundary")
    # Reject only a component that both reaches its radial bound and has
    # same-height support continuing immediately outside it.  Merely seeing a
    # coplanar pixel somewhere in the annulus is insufficient: a nearby
    # object, counter edge, or depth speckle can legitimately occupy that
    # annulus while the promoted component remains bounded.
    component_radius = np.linalg.norm(world_image - promoted_world[None, None, :], axis=2)
    touches_radial_boundary = bool(np.any(component & (component_radius >= 0.95 * float(component_radius_m))))
    outer = valid & np.isfinite(radius) & (radius > float(component_radius_m) + 1e-12) & (radius <= float(component_radius_m + upper_band_m) + 1e-12) & upper
    padded_component = np.pad(component, 1, mode="constant", constant_values=False)
    adjacent_to_component = np.zeros_like(component, dtype=bool)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if dv == 0 and du == 0:
                continue
            adjacent_to_component |= padded_component[1 + dv : 1 + dv + height, 1 + du : 1 + du + width]
    outer_continuation = outer & adjacent_to_component
    outer_count = int(outer.sum())
    outer_adjacent_count = int(outer_continuation.sum())
    outer_annulus_ambiguous = bool(touches_radial_boundary and outer_adjacent_count > 0)
    if reject_outer_annulus and outer_annulus_ambiguous:
        raise ValueError("surface promotion detected coplanar support in outer annulus")
    # A sparse solid can put the highest observed pixel at the edge of its
    # component.  Choose the pixel with the most same-frame upper support in
    # the generator's 15 mm local disk, while retaining the original 15 mm
    # displacement cap from the model/arrow seed.  This is a deterministic
    # anchor selection, not nearest-object recovery.
    component_pixels = np.column_stack(np.nonzero(component))
    component_world = world_image[component]
    anchor_support = np.sum(
        np.linalg.norm(component_world[:, None, :] - component_world[None, :, :], axis=2)
        <= 0.015 + 1e-12,
        axis=1,
    )
    anchor_seed_distance = np.linalg.norm(component_world - seed_world[None, :], axis=1)
    if select_generator_anchor:
        anchor_ordering = np.lexsort((
            component_pixels[:, 1], component_pixels[:, 0], anchor_seed_distance,
            -component_world[:, 2], -anchor_support,
        ))
        anchor_index = int(anchor_ordering[0])
    else:
        anchor_index = int(np.flatnonzero((component_pixels[:, 0] == promoted_v) & (component_pixels[:, 1] == promoted_u))[0])
    anchor_v, anchor_u = (int(value) for value in component_pixels[anchor_index])
    anchor_world = world_image[anchor_v, anchor_u].copy()
    anchor_displacement = float(np.linalg.norm(anchor_world - seed_world))
    if not np.isfinite(anchor_displacement) or anchor_displacement > float(max_displacement_m) + 1e-12:
        raise ValueError("surface promotion anchor exceeds metric displacement cap")
    mask_hash = hashlib.sha256(component.tobytes()).hexdigest()
    audit = {
        "promotion": "promoted_upper_surface",
        "seed_uv": [u0, v0],
        "promoted_uv": [promoted_u, promoted_v],
        "pixel_displacement": float(pixel_distance[promoted_v, promoted_u]),
        "spatial_distance_m": promoted_displacement,
        "max_snap_radius_px": int(max_snap_radius_px),
        "max_displacement_m": float(max_displacement_m),
        "upper_band_m": float(upper_band_m),
        "component_radius_m": float(component_radius_m),
        "connectivity": 8,
        "component_pixels": area,
        "component_fraction": fraction,
        "outer_annulus_pixels": outer_count,
        "outer_annulus_adjacent_pixels": outer_adjacent_count,
        "outer_annulus_ambiguous": outer_annulus_ambiguous,
        "reject_outer_annulus": bool(reject_outer_annulus),
        "generator_anchor_selection": (
            "max_local_upper_support_15mm" if select_generator_anchor else "promoted_upper_surface"
        ),
        "generator_anchor_uv": [anchor_u, anchor_v],
        "generator_anchor_support_pixels": int(anchor_support[anchor_index]),
        "generator_anchor_spatial_distance_m": anchor_displacement,
        "touches_radial_boundary": touches_radial_boundary,
        "touches_image_boundary": touches_image_boundary,
        "mask_sha256": mask_hash,
        "mask_shape": [height, width],
        "temporal_gate": "same_frame_rgbd",
    }
    return MolmoPoint(float(anchor_u), float(anchor_v), label="promoted_upper_surface"), anchor_world, component, audit


def _geometry_support_mask_for_promotion(
    capture: Any,
    *,
    contact_mask: np.ndarray,
    promoted_world: np.ndarray,
    promotion_audit: Mapping[str, Any],
    max_component_pixels: int = 512,
    max_component_fraction: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return bounded generator support separately from the collision mask.

    The contact mask remains the strict seed-connected component.  When its
    radial annulus is ambiguous, the frozen generator may see only a one-pixel
    outer ring of finite, same-frame upper support to satisfy its visibility
    contract.  The ring is never used as semantic target contact or collision
    exemption; the target-aware sweep receives ``contact_mask`` unchanged.
    """
    mask = np.asarray(contact_mask, dtype=bool)
    if mask.ndim != 2 or not np.isfinite(promoted_world).all():
        raise ValueError("geometry support promotion inputs are invalid")
    geometry = mask.copy()
    if not bool(promotion_audit.get("outer_annulus_ambiguous", False)):
        return geometry, {
            "status": "same_as_contact_mask",
            "contact_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "geometry_support_mask_sha256": hashlib.sha256(geometry.tobytes()).hexdigest(),
            "geometry_support_pixels": int(geometry.sum()),
            "geometry_support_outer_pixels": 0,
            "geometry_support_temporal_gate": "same_frame_rgbd",
        }
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    K, T = _camera_arrays(capture)
    valid = np.isfinite(depth) & (depth > 0.0)
    pixels = np.column_stack(np.nonzero(valid))
    points, kept = _support_points(pixels, depth, K, T)
    if not len(points):
        raise ValueError("geometry support promotion has no finite RGB-D points")
    world_image = np.full((*depth.shape, 3), np.nan, dtype=np.float64)
    world_image[kept[:, 0], kept[:, 1]] = points
    radius = np.linalg.norm(world_image - np.asarray(promoted_world)[None, None, :], axis=2)
    upper_band = float(promotion_audit.get("upper_band_m", 0.008))
    component_radius = float(promotion_audit.get("component_radius_m", 0.0195))
    eligible = valid & np.isfinite(radius)
    eligible &= radius > component_radius + 1e-12
    eligible &= radius <= component_radius + upper_band + 1e-12
    adjacency = np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if dv == 0 and du == 0:
                continue
            adjacency |= padded[1 + dv : 1 + dv + mask.shape[0], 1 + du : 1 + du + mask.shape[1]]
    outer = eligible & adjacency
    outer_pixels = np.column_stack(np.nonzero(outer))
    if len(outer_pixels):
        room = min(
            int(max_component_pixels) - int(mask.sum()),
            int(max_component_fraction * mask.size) - int(mask.sum()),
        )
        if room < len(outer_pixels):
            # Keep the nearest outer support pixels deterministically; an
            # oversized continuation stays ambiguous and is not admitted.
            outer_world = world_image[outer]
            order = np.lexsort((outer_pixels[:, 1], outer_pixels[:, 0], np.linalg.norm(outer_world - promoted_world[None, :], axis=1)))
            outer_pixels = outer_pixels[order[: max(room, 0)]]
        if len(outer_pixels):
            geometry[outer_pixels[:, 0], outer_pixels[:, 1]] = True
    audit = {
        "status": "outer_annulus_geometry_only",
        "contact_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "geometry_support_mask_sha256": hashlib.sha256(geometry.tobytes()).hexdigest(),
        "geometry_support_pixels": int(geometry.sum()),
        "geometry_support_outer_pixels": int(np.count_nonzero(geometry & ~mask)),
        "geometry_support_temporal_gate": "same_frame_rgbd",
        "geometry_support_contact_exempted": False,
    }
    return geometry, audit


def _promote_upper_surface_adaptively(
    capture: Any,
    *,
    seed_uv: Sequence[float],
    seed_world: np.ndarray,
    max_aperture_m: float,
    allow_geometry_support: bool = False,
    select_generator_anchor: bool = False,
) -> tuple[MolmoPoint, np.ndarray, np.ndarray, dict[str, Any]]:
    """Promote a seed while keeping the local support bounded on flat counters.

    RoboCasa places many target solids directly on a counter.  At 256 px the
    target's top surface and the counter can touch the same 19.5 mm support
    disk even though a smaller disk is a valid grasp patch.  Keep the original
    v3 promotion as the first hypothesis, then retry only the specific
    coplanar-annulus failure with progressively smaller, tighter local bands.
    The RGB-D component, area cap, image-boundary check, and target-aware
    collision filter remain unchanged for every accepted hypothesis.
    """
    primary_radius = float(max_aperture_m) * 0.5
    attempts = (
        (primary_radius, 0.008, "nominal"),
        (min(primary_radius, 0.012), 0.004, "compact"),
        (min(primary_radius, 0.009), 0.003, "tight"),
    )
    last_error: ValueError | None = None
    for index, (radius, upper_band, label) in enumerate(attempts):
        try:
            point, world, mask, audit = _promote_upper_surface(
                capture,
                seed_uv=seed_uv,
                seed_world=seed_world,
                max_snap_radius_px=12,
                max_displacement_m=0.015,
                upper_band_m=upper_band,
                component_radius_m=radius,
                max_component_pixels=512,
                max_component_fraction=0.05,
                select_generator_anchor=select_generator_anchor,
            )
        except ValueError as exc:
            last_error = exc
            if "coplanar support in outer annulus" not in str(exc):
                raise
            continue
        audit = {
            **audit,
            "adaptive_promotion": label,
            "adaptive_attempt_index": index,
            "adaptive_attempt_count": len(attempts),
            "nominal_component_radius_m": primary_radius,
            "nominal_upper_band_m": 0.008,
        }
        return point, world, mask, audit
    # Preserve a strict collision/contact mask while allowing the frozen
    # generator to inspect a bounded outer support ring.  This is reached only
    # after all compact retries fail for the specific coplanar-annulus guard.
    # The caller records the split masks; no support point is later exempted
    # from the measured RGB-D sweep by this geometry-only concession.
    if not allow_geometry_support:
        assert last_error is not None
        raise last_error
    try:
        point, world, mask, audit = _promote_upper_surface(
            capture,
            seed_uv=seed_uv,
            seed_world=seed_world,
            max_snap_radius_px=12,
            max_displacement_m=0.015,
            upper_band_m=0.008,
            component_radius_m=primary_radius,
            max_component_pixels=512,
            max_component_fraction=0.05,
            reject_outer_annulus=False,
            select_generator_anchor=select_generator_anchor,
        )
    except ValueError:
        assert last_error is not None
        raise last_error
    audit = {
        **audit,
        "adaptive_promotion": "geometry_only_outer_annulus",
        "adaptive_attempt_index": len(attempts),
        "adaptive_attempt_count": len(attempts) + 1,
        "nominal_component_radius_m": primary_radius,
        "nominal_upper_band_m": 0.008,
    }
    return point, world, mask, audit


def _robot_occludes_anchor(robot_calibration: Any, point_world: np.ndarray, *, clearance_m: float = 0.0) -> bool:
    """Classify an arrow anchor against the measured current hand envelope."""
    try:
        from .grasp_candidates import _current_hand_world_boxes, _current_hand_world_spheres, _points_box_signed_clearance, _validate_live_robot_geometry
        current_grip, current_rotation, spheres, boxes = _validate_live_robot_geometry(robot_calibration)
        if current_grip is None or current_rotation is None:
            return False
        grasp_to_grip = np.asarray(robot_calibration.grasp_to_grip_site, dtype=np.float64).reshape(3, 3)
        contact_offset = np.asarray(robot_calibration.contact_to_grip_site_m, dtype=np.float64).reshape(3)
        if boxes:
            for box in _current_hand_world_boxes(current_grip, current_rotation, grasp_to_grip, contact_offset, boxes):
                if float(_points_box_signed_clearance(point_world[None, :], box["center_world_m"], box["rotation_world_box"], box["half_extents_m"])[0]) <= float(clearance_m):
                    return True
        else:
            for center, radius in _current_hand_world_spheres(current_grip, current_rotation, grasp_to_grip, contact_offset, spheres):
                if float(np.linalg.norm(point_world - center)) <= float(radius) + float(clearance_m):
                    return True
    except (AttributeError, KeyError, TypeError, ValueError):
        # Missing optional live envelope is not evidence of occlusion; the
        # generator's own validated live geometry remains authoritative.
        return False
    return False


def _v3_image_metric_offset(
    point_uv: Sequence[float], source_uv: Sequence[float], point_depth_m: float,
    K: np.ndarray,
) -> tuple[float, float, float]:
    if not np.isfinite(point_depth_m) or point_depth_m <= 0.0:
        raise ValueError("v3 model point has no finite metric depth")
    du = (float(point_uv[0]) - float(source_uv[0])) * float(point_depth_m) / abs(float(K[0, 0]))
    dv = (float(point_uv[1]) - float(source_uv[1])) * float(point_depth_m) / abs(float(K[1, 1]))
    return float(du), float(dv), float(np.hypot(du, dv))


def _save_input_image(request: Any, image: np.ndarray) -> dict[str, Any]:
    # Content-address the encoded image so retries never overwrite an audit
    # artifact from an earlier fresh capture.  The returned hash is the exact
    # PNG content hash embedded in the filename.
    try:
        from PIL import Image
        encoded = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(encoded, format="PNG")
        payload = encoded.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        path = Path(request.output_dir) / "object_contact_inputs" / f"molmo_object_contact_input_{digest}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
    except Exception as exc:  # pragma: no cover - Pillow is an optional test dependency
        return {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}
    return {"status": "saved", "path": path.as_posix(), "sha256": digest}


def _point_from_model(value: Any) -> MolmoPoint:
    if isinstance(value, MolmoPoint):
        return value
    if isinstance(value, Mapping):
        return MolmoPoint.from_value(value)
    if hasattr(value, "x") and hasattr(value, "y"):
        return MolmoPoint(
            float(value.x), float(value.y),
            float(getattr(value, "confidence", getattr(value, "score", 1.0))),
            str(getattr(value, "label", "contact")),
        )
    return MolmoPoint.from_value(value)


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(np.asarray(first).T @ np.asarray(second)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_from_to(source: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Return a proper rotation that maps one unit axis onto another."""

    first = np.asarray(source, dtype=np.float64).reshape(3)
    second = np.asarray(target, dtype=np.float64).reshape(3)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if not np.isfinite(first_norm) or not np.isfinite(second_norm) or first_norm <= 1e-9 or second_norm <= 1e-9:
        raise ValueError("approach axes must be finite and non-zero")
    first /= first_norm
    second /= second_norm
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1e-9:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        # Anti-parallel axes need a deterministic half-turn around any axis
        # orthogonal to the source.  Choose the least-aligned basis vector.
        basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(first)))]
        axis = np.cross(first, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = np.array(((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)))
    return np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def _rotated_workspace_bounds(robot_calibration: Any, world_to_virtual: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build a conservative virtual-frame AABB for the original workspace."""

    lo = np.asarray(robot_calibration.workspace_min_m, dtype=np.float64).reshape(3)
    hi = np.asarray(robot_calibration.workspace_max_m, dtype=np.float64).reshape(3)
    corners = np.asarray(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=np.float64,
    )
    rotated = corners @ world_to_virtual.T
    return tuple(float(value) for value in np.min(rotated, axis=0)), tuple(float(value) for value in np.max(rotated, axis=0))


def _virtual_camera_calibration(capture: Any, world_to_virtual: np.ndarray) -> Any:
    """Express the same RGB-D camera in the temporary virtual world frame."""

    _, T = _camera_arrays(capture)
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = world_to_virtual
    return _capture_calibration_with_transform(capture, homogeneous @ T)


def _capture_calibration_with_transform(capture: Any, world_from_camera: np.ndarray) -> Any:
    from .grasp_candidates import CameraCalibration

    calibration = capture.calibration
    return CameraCalibration(
        width=int(calibration.width), height=int(calibration.height),
        intrinsic=tuple(tuple(float(v) for v in row) for row in np.asarray(calibration.intrinsic, dtype=np.float64)),
        world_from_camera=tuple(tuple(float(v) for v in row) for row in np.asarray(world_from_camera, dtype=np.float64)),
        camera_name=str(getattr(calibration, "camera_name", "unknown")),
    )


def _virtual_robot_calibration(robot_calibration: Any, world_to_virtual: np.ndarray) -> Any:
    """Adapt world-valued live geometry for the unchanged upright generator."""

    workspace_min, workspace_max = _rotated_workspace_bounds(robot_calibration, world_to_virtual)
    updates: dict[str, Any] = {
        "approach_axis_world": (0.0, 0.0, -1.0),
        "workspace_min_m": workspace_min,
        "workspace_max_m": workspace_max,
    }
    current_grip = getattr(robot_calibration, "current_grip_site_world_m", None)
    current_rotation = getattr(robot_calibration, "current_rotation_world_grip_site", None)
    if current_grip is not None:
        updates["current_grip_site_world_m"] = tuple(float(value) for value in world_to_virtual @ np.asarray(current_grip, dtype=np.float64))
    if current_rotation is not None:
        updates["current_rotation_world_grip_site"] = tuple(tuple(float(value) for value in row) for row in world_to_virtual @ np.asarray(current_rotation, dtype=np.float64))
    return replace(robot_calibration, **updates)


def _inverse_transform_candidate(candidate: GraspCandidate, world_to_virtual: np.ndarray, approach_label: str) -> GraspCandidate:
    """Map a candidate generated in the virtual frame back to B0/world."""

    virtual_to_world = world_to_virtual.T
    rotation_world_grasp = virtual_to_world @ np.asarray(candidate.rotation_world_grasp, dtype=np.float64)
    rotation_world_grip = virtual_to_world @ np.asarray(candidate.rotation_world_grip_site, dtype=np.float64)
    audit = dict(candidate.audit)
    audit["approach_label"] = approach_label
    audit["approach_axis_world"] = [float(value) for value in virtual_to_world @ np.array((0.0, 0.0, -1.0))]
    audit["approach_tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(float(audit["approach_axis_world"][2])), -1.0, 1.0))))
    audit["geometry_frame"] = "virtual_upright_world_to_b0"
    audit["world_to_virtual_rotation"] = world_to_virtual.tolist()
    # The canonical generator's audit is retained verbatim; this marker makes
    # the rigid-frame provenance explicit without inventing a second audit
    # schema for the virtual geometry.
    return replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}__approach_{approach_label}",
        contact_world_m=virtual_to_world @ np.asarray(candidate.contact_world_m, dtype=np.float64),
        grip_site_world_m=virtual_to_world @ np.asarray(candidate.grip_site_world_m, dtype=np.float64),
        pregrasp_world_m=virtual_to_world @ np.asarray(candidate.pregrasp_world_m, dtype=np.float64),
        release_world_m=(None if candidate.release_world_m is None else virtual_to_world @ np.asarray(candidate.release_world_m, dtype=np.float64)),
        rotation_world_grasp=rotation_world_grasp,
        rotation_world_grip_site=rotation_world_grip,
        quaternion_world_grip_site_xyzw=_quaternion_xyzw(rotation_world_grip),
        jaw_axis_world=virtual_to_world @ np.asarray(candidate.jaw_axis_world, dtype=np.float64),
        rim_tangent_world=virtual_to_world @ np.asarray(candidate.rim_tangent_world, dtype=np.float64),
        audit=audit,
    )


def _generate_for_approach(
    *, capture: Any, robot_calibration: Any, policy: CandidatePolicy,
    point: MolmoPoint, mask: np.ndarray, approach_axis_world: np.ndarray,
    approach_label: str,
) -> GraspCandidateResult:
    """Run the byte-identical upright generator in a rotated temporary frame."""

    world_to_virtual = _rotation_from_to(approach_axis_world, (0.0, 0.0, -1.0))
    virtual_robot = _virtual_robot_calibration(robot_calibration, world_to_virtual)
    result = generate_grasp_candidates(
        rgb=np.asarray(capture.rgb, dtype=np.uint8),
        metric_depth_m=np.asarray(capture.metric_depth, dtype=np.float64),
        sam_mask=np.asarray(mask, dtype=bool),
        molmo_points=(point,),
        calibration=_virtual_camera_calibration(capture, world_to_virtual),
        robot_calibration=virtual_robot,
        policy=policy,
    )
    original_lo = np.asarray(robot_calibration.workspace_min_m, dtype=np.float64).reshape(3)
    original_hi = np.asarray(robot_calibration.workspace_max_m, dtype=np.float64).reshape(3)
    transformed_candidates: list[GraspCandidate] = []
    workspace_rejected: list[CandidateRejection] = []
    for item in result.candidates:
        transformed = _inverse_transform_candidate(item, world_to_virtual, approach_label)
        points = {
            "contact_world_m": np.asarray(transformed.contact_world_m, dtype=np.float64),
            "grip_site_world_m": np.asarray(transformed.grip_site_world_m, dtype=np.float64),
            "pregrasp_world_m": np.asarray(transformed.pregrasp_world_m, dtype=np.float64),
        }
        if any(
            np.any(point < original_lo - 1e-9) or np.any(point > original_hi + 1e-9)
            for point in points.values()
        ):
            workspace_rejected.append(
                CandidateRejection(
                    seed_index=transformed.seed_index,
                    yaw_deg=transformed.yaw_deg,
                    insertion_depth_m=transformed.insertion_depth_m,
                    reason="workspace_original_frame",
                    details={
                        "approach_label": approach_label,
                        "geometry_frame": "b0_world",
                        "workspace_min_m": original_lo.tolist(),
                        "workspace_max_m": original_hi.tolist(),
                        "points": {key: value.tolist() for key, value in points.items()},
                    },
                )
            )
            continue
        transformed_candidates.append(transformed)
    transformed = tuple(transformed_candidates)
    audit = {**dict(result.audit), "geometry_frame": "virtual_upright", "approach_label": approach_label, "approach_axis_world": [float(value) for value in approach_axis_world], "world_to_virtual_rotation": world_to_virtual.tolist()}
    # Rejections retain the virtual-frame geometry details and are explicitly
    # labeled so no virtual coordinate can be mistaken for B0/world evidence.
    rejected = tuple(replace(item, details={**dict(item.details), "approach_label": approach_label, "geometry_frame": "virtual_upright"}) for item in result.rejected) + tuple(workspace_rejected)
    return GraspCandidateResult(transformed, rejected, result.seeds_uv, result.policy, audit)


def _aggregate(
    generated: Sequence[tuple[str, int, GraspCandidateResult]],
    policy: CandidatePolicy,
) -> tuple[tuple[GraspCandidate, ...], tuple[CandidateRejection, ...], dict[str, Any]]:
    candidates: list[GraspCandidate] = []
    rejected: list[CandidateRejection] = []
    for source, seed_index, result in generated:
        rejected.extend(
            replace(item, seed_index=seed_index, details={**dict(item.details), "seed_source": source})
            for item in result.rejected
        )
        for candidate in result.candidates:
            candidate_id = f"{source}_{seed_index}__{candidate.candidate_id}"
            audit = {**dict(candidate.audit), "seed_source": source, "seed_index": seed_index}
            candidates.append(replace(candidate, candidate_id=candidate_id, seed_index=seed_index, audit=audit))
    candidates.sort(key=lambda item: (-float(item.score), item.candidate_id))
    unique: list[GraspCandidate] = []
    for candidate in candidates:
        duplicate = any(
            np.linalg.norm(candidate.contact_world_m - prior.contact_world_m) <= policy.dedupe_position_m
            and _rotation_distance_deg(candidate.rotation_world_grasp, prior.rotation_world_grasp) <= policy.dedupe_rotation_deg
            for prior in unique
        )
        if duplicate:
            rejected.append(CandidateRejection(candidate.seed_index, candidate.yaw_deg, candidate.insertion_depth_m, "duplicate_candidate", {"global": True}))
        else:
            unique.append(candidate)
    return tuple(unique[: policy.max_candidates]), tuple(rejected), {
        "generated_candidate_count": len(candidates),
        "deduped_candidate_count": len(unique),
        "returned_count": min(len(unique), policy.max_candidates),
    }


def _propose_object_contact_profile(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
    profile_name: str,
    profile_sha256: str,
    allow_rejected_fallback: bool,
    camera_ray_approach: bool,
) -> dict[str, Any]:
    """Query Molmo on the unmasked arrow frame and derive local candidates."""

    image = np.asarray(request.arrow_rgb)
    source_rgb = np.asarray(request.source_capture.rgb)
    depth = np.asarray(request.source_capture.metric_depth)
    calibration = getattr(request.source_capture, "calibration", None)
    calibration_shape = (
        getattr(calibration, "height", None), getattr(calibration, "width", None)
    )
    if (
        image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
        or source_rgb.ndim != 3 or source_rgb.shape[2] != 3 or source_rgb.dtype != np.uint8
        or image.shape != source_rgb.shape
        or depth.ndim != 2 or depth.shape != image.shape[:2]
        or calibration_shape != image.shape[:2]
    ):
        raise ValueError(
            f"{profile_name} requires same-frame RGB-D arrow/source images "
            "with matching HxWx3 uint8 and calibrated depth shape"
        )
    if profile_name == PROFILE_NAME:
        _validate_object_contact_profile(prompt)
    elif profile_name == PROFILE_V2_NAME:
        _validate_object_contact_v2_profile(prompt)
    elif profile_name == PROFILE_V3_NAME:
        _validate_object_contact_v3_profile(prompt)
    elif profile_name == PROFILE_V4_NAME:
        _validate_object_contact_v4_profile(prompt)
    elif profile_name == PROFILE_V5_NAME:
        _validate_object_contact_v5_profile(prompt)
    else:  # pragma: no cover - internal profile dispatch guard
        raise ValueError(f"unknown object contact profile: {profile_name}")

    # Deliberately no exception fallback around predict: model/runtime errors
    # are execution failures.  Only a valid decoded empty point list admits the
    # deterministic arrow-tail anchor below.
    from .molmopoint import MolmoPointRequest
    input_audit = _save_input_image(request, image)
    result = molmo.predict(MolmoPointRequest(image, mask=None, prompt=prompt))
    decoded = tuple(getattr(result, "points", ()) or ())
    source_uv = tuple(float(v) for v in request.source_uv)
    source_world = _deproject(request.source_capture, source_uv)
    anchor_status = "visible"
    if profile_name in {PROFILE_V3_NAME, PROFILE_V4_NAME, PROFILE_V5_NAME} and _robot_occludes_anchor(robot_calibration, source_world, clearance_m=0.0):
        anchor_status = "robot_occluded"
    max_aperture = float(getattr(robot_calibration, "max_aperture_m", 0.080))
    finger_clearance = float(getattr(robot_calibration, "finger_clearance_m", 0.004))
    # v3's occlusion-aware admission is an image-metric ellipse, with a
    # profile-fixed half-aperture radius for both decoded and fallback paths.
    # v1/v2 retain their historical world-distance gate and finger margin.
    v3_family = profile_name in {PROFILE_V3_NAME, PROFILE_V4_NAME, PROFILE_V5_NAME}
    v4_geometry_split = profile_name in {PROFILE_V4_NAME, PROFILE_V5_NAME}
    v5_reflex = profile_name == PROFILE_V5_NAME
    admission_threshold = (
        max_aperture * 0.5 + finger_clearance
        if v5_reflex
        else max_aperture * 0.5
        if v3_family
        else max_aperture * 0.5 + finger_clearance
    )
    admission_distance_frame = (
        "same_frame_rgbd_image_metric_ellipse"
        if v3_family
        else "same_frame_rgbd_world"
    )
    canonical = canonical_candidate_policy()
    local_policy = replace(canonical, name="molmo_local", max_seeds=1)
    if v3_family:
        # Target-aware collision is applied after unchanged candidate geometry.
        # Disabling the generator's historical tiny-mask terminal allowance
        # avoids conflating the full hand sweep with the promoted component.
        local_policy = replace(local_policy, obstruction_clearance_m=None, terminal_contact_allowance_m=max_aperture * 0.5)
    generated: list[tuple[str, int, GraspCandidateResult]] = []
    seed_diagnostics: list[dict[str, Any]] = []
    seeds: list[tuple[str, MolmoPoint, np.ndarray]] = []
    if not decoded and anchor_status != "robot_occluded":
        seeds.append(("arrow_tail_anchor", MolmoPoint(source_uv[0], source_uv[1], label="arrow_tail_anchor"), source_world))
    elif not decoded:
        seed_diagnostics.append({"source": "arrow_tail_anchor", "status": "rejected", "reason": "robot_occluded", "anchor_status": anchor_status, "temporal_gate": "same_frame_rgbd"})
    else:
        for index, value in enumerate(decoded):
            point = _point_from_model(value)
            try:
                point_world = _deproject(request.source_capture, (point.u, point.v))
                distance = float(np.linalg.norm(point_world - source_world))
            except ValueError as exc:
                seed_diagnostics.append({"source": "molmo", "index": index, "status": "rejected", "reason": str(exc)})
                continue
            if v3_family:
                K, _ = _camera_arrays(request.source_capture)
                metric_dx, metric_dy, image_metric_distance = _v3_image_metric_offset(
                    (point.u, point.v), source_uv, _depth_at(request.source_capture, (point.u, point.v)), K
                )
                distance = image_metric_distance
                if _robot_occludes_anchor(robot_calibration, point_world, clearance_m=0.0):
                    seed_diagnostics.append({"source": "molmo", "index": index, "status": "rejected", "reason": "robot_occluded", "anchor_status": "robot_occluded", "distance_m": distance, "projected_metric_offset_m": [metric_dx, metric_dy]})
                    continue
            if not np.isfinite(distance) or distance > admission_threshold:
                seed_diagnostics.append({"source": "molmo", "index": index, "status": "rejected", "reason": "outside_arrow_anchor_gate", "distance_m": distance, "threshold_m": admission_threshold})
                continue
            seeds.append(("molmo_contact", point, point_world))
            seed_diagnostics.append({"source": "molmo", "index": index, "status": "admitted", "distance_m": distance, "threshold_m": admission_threshold, "uv": [point.u, point.v]})
    if decoded and not seeds and allow_rejected_fallback and anchor_status != "robot_occluded":
        # The only nonempty-decode fallback is the deterministic generated
        # arrow tail.  It is intentionally distinct from nearest-object or
        # simulator-assisted recovery and remains subject to RGB-D depth.
        seeds.append((
            "arrow_tail_anchor_all_model_points_rejected",
            MolmoPoint(source_uv[0], source_uv[1], label="arrow_tail_anchor_all_model_points_rejected"),
            source_world,
        ))
        seed_diagnostics.append({
            "source": "arrow_tail_anchor_all_model_points_rejected",
            "status": "fallback",
            "reason": "all_model_points_rejected",
            "model_point_count": len(decoded),
        })
    elif decoded and not seeds and allow_rejected_fallback:
        seed_diagnostics.append({"source": "arrow_tail_anchor_all_model_points_rejected", "status": "rejected", "reason": "robot_occluded", "anchor_status": anchor_status, "temporal_gate": "same_frame_rgbd"})
    semantic_seed_count = sum(source == "molmo_contact" for source, _, _ in seeds)
    dropped_seed_count = max(0, semantic_seed_count - MAX_SPATIAL_SEEDS)
    if dropped_seed_count:
        # Preserve the canonical spatial budget and model ordering.  The
        # arrow-tail fallback is exclusive with semantic seeds when decoding
        # is empty, so the reserved fallback never expands this budget.
        seeds = seeds[:MAX_SPATIAL_SEEDS]
        seed_diagnostics.append({
            "source": "molmo", "status": "dropped_seed_budget",
            "dropped_count": dropped_seed_count,
            "budget": MAX_SPATIAL_SEEDS,
        })
    for seed_index, (source, point, point_world) in enumerate(seeds):
        promotion_audit: dict[str, Any] = {}
        if v3_family:
            try:
                point, point_world, local_mask, promotion_audit = _promote_upper_surface_adaptively(
                    request.source_capture,
                    seed_uv=(point.u, point.v), seed_world=point_world,
                    max_aperture_m=max_aperture,
                    allow_geometry_support=v4_geometry_split,
                    select_generator_anchor=v4_geometry_split,
                )
            except ValueError as exc:
                seed_diagnostics.append({
                    "source": source, "index": seed_index, "status": "rejected",
                    "reason": str(exc), "temporal_gate": "same_frame_rgbd",
                    "offender_in_target_component": None,
                })
                continue
            if v4_geometry_split:
                # The compact promoted component is the only semantic/contact
                # mask. An ambiguous coplanar annulus may contribute a bounded
                # same-frame support ring to the frozen geometry engine, but
                # that ring is never passed to collision exemption logic.
                generator_mask, geometry_audit = _geometry_support_mask_for_promotion(
                    request.source_capture,
                    contact_mask=local_mask,
                    promoted_world=point_world,
                    promotion_audit=promotion_audit,
                )
                promotion_audit = {
                    **promotion_audit,
                    "geometry_support": geometry_audit,
                    "contact_mask_sha256": geometry_audit["contact_mask_sha256"],
                    "generator_mask_sha256": geometry_audit["geometry_support_mask_sha256"],
                    "generator_support_pixels": geometry_audit["geometry_support_pixels"],
                }
            else:
                generator_mask = local_mask
        else:
            local_mask, local_audit = _local_rgbd_mask(
                request.source_capture, point_world, radius_m=local_policy.rim_local_radius_m
            )
            generator_mask = local_mask
        if v3_family:
            local_audit = promotion_audit
        local_audit = {
            **local_audit,
            "mask_sha256": hashlib.sha256(local_mask.tobytes()).hexdigest(),
            "generator_mask_sha256": hashlib.sha256(generator_mask.tobytes()).hexdigest(),
            "generator_support_pixels": int(generator_mask.sum()),
        }
        seed_diagnostics.append({"source": source, "index": seed_index, "status": "local_patch", **local_audit, "rim_height_band_m": local_policy.rim_height_band_m, "min_rim_support_pixels": local_policy.min_rim_support_pixels})
        approach_jobs: list[tuple[str, np.ndarray]] = [
            ("world_down", np.asarray((0.0, 0.0, -1.0), dtype=np.float64)),
        ]
        if camera_ray_approach:
            try:
                ray, downward_tilt_deg = _camera_ray_approach(capture=request.source_capture, point_world=point_world)
            except ValueError as exc:
                reason = str(exc)
                seed_diagnostics.append({"source": source, "index": seed_index, "status": "rejected", "reason": reason, "max_approach_tilt_deg": 55.0})
                camera_ray_unavailable = True
            else:
                approach_jobs.append(("camera_ray", ray))
                camera_ray_unavailable = False
        else:
            camera_ray_unavailable = True
        approach_rejections: list[Mapping[str, Any]] = []
        approach_candidate_count = 0
        approach_outcomes: list[dict[str, Any]] = []
        for approach_label, axis in approach_jobs:
            # Keep the v1 source/candidate identity byte-for-byte compatible
            # with its historical single world-down hypothesis.  v2 carries
            # the approach label because it intentionally evaluates more than
            # one rigid-frame hypothesis.
            generation_source = (
                source
                if not camera_ray_approach and approach_label == "world_down"
                else f"{source}__{approach_label}"
            )
            if approach_label == "world_down":
                generated_result = generate_grasp_candidates(
                    rgb=np.asarray(request.source_capture.rgb, dtype=np.uint8),
                    metric_depth_m=np.asarray(request.source_capture.metric_depth, dtype=np.float64),
                    sam_mask=generator_mask,
                    molmo_points=(point,),
                    calibration=_capture_calibration(request.source_capture),
                    robot_calibration=robot_calibration,
                    policy=local_policy,
                )
            else:
                generated_result = _generate_for_approach(
                    capture=request.source_capture,
                    robot_calibration=robot_calibration,
                    policy=local_policy,
                    point=point,
                    mask=generator_mask,
                    approach_axis_world=axis,
                    approach_label=approach_label,
                )
            if v3_family:
                from .object_contact_collision import filter_target_aware_candidates
                generated_result = filter_target_aware_candidates(
                    generated_result, capture=request.source_capture,
                    target_mask=local_mask, robot_calibration=robot_calibration,
                    clearance_m=0.006,
                )
            approach_candidate_count += len(generated_result.candidates)
            approach_rejections.extend(
                {"reason": item.reason, **dict(item.details)}
                for item in generated_result.rejected
            )
            approach_outcomes.append({
                "approach_label": approach_label,
                "executed": True,
                "survivor_count": len(generated_result.candidates),
                "non_target_obstruction": any(
                    item.reason == "target_aware_approach_obstruction"
                    and item.details.get("offender_in_target_component") is False
                    for item in generated_result.rejected
                ),
            })
            generated.append((generation_source, seed_index, generated_result))
        if v4_geometry_split and camera_ray_unavailable:
            approach_outcomes.append({
                "approach_label": "camera_ray",
                "executed": False,
                "survivor_count": 0,
                "non_target_obstruction": False,
            })
        if v4_geometry_split:
            seed_diagnostics.append({
                "source": source, "index": seed_index,
                "status": "approach_hypotheses",
                "outcomes": approach_outcomes,
                "blocker_away_authorized": _blocker_away_recovery_allowed(approach_outcomes),
            })
        if v4_geometry_split and _blocker_away_recovery_allowed(approach_outcomes):
            blocker_hypotheses = (
                _blocker_away_ladder(
                    request.source_capture,
                    contact_world=point_world,
                    rejection_details=approach_rejections,
                )
                if v5_reflex else [
                    _blocker_away_approach(
                        request.source_capture,
                        contact_world=point_world,
                        rejection_details=approach_rejections,
                        max_tilt_deg=55.0,
                    )
                ]
            )
            for reflex_index, blocker_item in enumerate(blocker_hypotheses):
                if blocker_item is None:
                    continue
                blocker_axis, blocker_audit = blocker_item
                generated_result = _generate_for_approach(
                    capture=request.source_capture,
                    robot_calibration=robot_calibration,
                    policy=local_policy,
                    point=point,
                    mask=generator_mask,
                    approach_axis_world=blocker_axis,
                    approach_label=("blocker_away_ladder" if v5_reflex else "blocker_away"),
                )
                from .object_contact_collision import filter_target_aware_candidates
                generated_result = filter_target_aware_candidates(
                    generated_result, capture=request.source_capture,
                    target_mask=local_mask, robot_calibration=robot_calibration,
                    clearance_m=0.006,
                )
                generated_result = replace(
                    generated_result,
                    audit={**dict(generated_result.audit), "blocker_away": blocker_audit},
                )
                seed_diagnostics.append({
                    "source": source, "index": seed_index,
                    "status": "blocker_away_hypothesis",
                    "approach_outcomes": approach_outcomes,
                    **blocker_audit,
                })
                generated.append((f"{source}__blocker_away_{reflex_index}", seed_index, generated_result))
                if generated_result.candidates:
                    break
    candidates, rejected, aggregate_audit = _aggregate(generated, canonical)
    diagnostics = {
        "profile": profile_name,
        "profile_sha256": profile_sha256,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "molmopoint_count": len(decoded),
        "molmopoint_provenance": dict(getattr(result, "provenance", {}) or {}),
        "arrow_tail_source_uv": list(source_uv),
        "anchor_status": anchor_status,
        "admission_gate": {"max_aperture_m": max_aperture, "finger_clearance_m": finger_clearance, "threshold_m": admission_threshold, "distance_frame": admission_distance_frame},
        "seed_diagnostics": seed_diagnostics,
        "seed_budget": {
            "canonical_max_spatial_seeds": MAX_SPATIAL_SEEDS,
            "accepted_before_cap": semantic_seed_count,
            "accepted_after_cap": sum(source == "molmo_contact" for source, _, _ in seeds),
            "dropped_count": dropped_seed_count,
            "fallback_mode": "empty_decoded_only" if not allow_rejected_fallback else "empty_or_all_model_points_rejected",
        },
        "input_image": input_audit,
        "local_policy": {"name": local_policy.name, "rim_local_radius_m": local_policy.rim_local_radius_m, "rim_height_band_m": local_policy.rim_height_band_m, "min_rim_support_pixels": local_policy.min_rim_support_pixels, "min_depth_support_pixels": local_policy.min_depth_support_pixels, "obstruction_clearance_m": local_policy.obstruction_clearance_m, "terminal_contact_allowance_m": local_policy.terminal_contact_allowance_m},
        "aggregate": aggregate_audit,
        "rejection_count": len(rejected),
        "rejections": [
            {
                "seed_index": item.seed_index,
                "yaw_deg": item.yaw_deg,
                "insertion_depth_m": item.insertion_depth_m,
                "reason": item.reason,
                "details": _json_safe({
                    **dict(item.details),
                    **({
                        "offender_in_target_component": item.details.get("offender_in_target_component"),
                        "temporal_gate": item.details.get("temporal_gate", "same_frame_rgbd"),
                        "spatial_distance_m": item.details.get("spatial_distance_m", item.details.get("distance_m")),
                    } if v3_family else {}),
                }),
            }
            for item in rejected
        ],
        "geometry_audits": [dict(item[2].audit) for item in generated],
    }
    return {"candidates": candidates, "diagnostics": diagnostics}


def propose_object_contact(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
) -> dict[str, Any]:
    """Faithful v1 object contact proposal (empty-only fallback)."""
    return _propose_object_contact_profile(
        molmo=molmo, request=request, robot_calibration=robot_calibration,
        prompt=prompt, profile_name=PROFILE_NAME,
        profile_sha256=object_contact_profile_sha256(),
        allow_rejected_fallback=False, camera_ray_approach=False,
    )


def propose_object_contact_v2(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
) -> dict[str, Any]:
    """Experimental v2 with world-down plus bounded camera-ray hypotheses."""
    return _propose_object_contact_profile(
        molmo=molmo, request=request, robot_calibration=robot_calibration,
        prompt=prompt, profile_name=PROFILE_V2_NAME,
        profile_sha256=object_contact_v2_profile_sha256(),
        allow_rejected_fallback=True, camera_ray_approach=True,
    )


def propose_object_contact_v3(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
) -> dict[str, Any]:
    """Experimental v3 with upper-surface promotion and v2 approaches."""
    return _propose_object_contact_profile(
        molmo=molmo, request=request, robot_calibration=robot_calibration,
        prompt=prompt, profile_name=PROFILE_V3_NAME,
        profile_sha256=object_contact_v3_profile_sha256(),
        allow_rejected_fallback=True, camera_ray_approach=True,
    )


def propose_object_contact_v4(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
) -> dict[str, Any]:
    """Experimental v4 with split geometry and contact support masks."""
    return _propose_object_contact_profile(
        molmo=molmo, request=request, robot_calibration=robot_calibration,
        prompt=prompt, profile_name=PROFILE_V4_NAME,
        profile_sha256=object_contact_v4_profile_sha256(),
        allow_rejected_fallback=True, camera_ray_approach=True,
    )


def propose_object_contact_v5(
    *,
    molmo: Any,
    request: Any,
    robot_calibration: Any,
    prompt: str,
) -> dict[str, Any]:
    """Experimental v5 with wider admission and low-tilt reflex ladder."""
    return _propose_object_contact_profile(
        molmo=molmo, request=request, robot_calibration=robot_calibration,
        prompt=prompt, profile_name=PROFILE_V5_NAME,
        profile_sha256=object_contact_v5_profile_sha256(),
        allow_rejected_fallback=True, camera_ray_approach=True,
    )


def _capture_calibration(capture: Any) -> Any:
    from .grasp_candidates import CameraCalibration
    calibration = capture.calibration
    return CameraCalibration(
        width=int(calibration.width), height=int(calibration.height),
        intrinsic=tuple(tuple(float(v) for v in row) for row in calibration.intrinsic),
        world_from_camera=tuple(tuple(float(v) for v in row) for row in calibration.world_from_camera),
        camera_name=str(getattr(calibration, "camera_name", "unknown")),
    )


__all__ = [
    "PROFILE_NAME", "PROFILE_V2_NAME", "PROFILE_V3_NAME", "PROFILE_V4_NAME", "PROFILE_V5_NAME", "object_contact_profile",
    "object_contact_profile_path", "object_contact_profile_sha256",
    "object_contact_v2_profile", "object_contact_v2_profile_path",
    "object_contact_v2_profile_sha256", "object_contact_v3_profile",
    "object_contact_v3_profile_path", "object_contact_v3_profile_sha256",
    "object_contact_v4_profile", "object_contact_v4_profile_path",
    "object_contact_v4_profile_sha256", "propose_object_contact",
    "propose_object_contact_v2", "propose_object_contact_v3",
    "propose_object_contact_v4", "object_contact_v5_profile",
    "object_contact_v5_profile_path", "object_contact_v5_profile_sha256",
    "propose_object_contact_v5",
]
