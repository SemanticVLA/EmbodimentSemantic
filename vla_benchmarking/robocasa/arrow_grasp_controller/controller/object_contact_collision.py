"""Target-aware collision filtering for the experimental object-contact arm.

The canonical candidate generator remains the sole source of grasp geometry.
This module only applies a RoboCasa-local RGB-D sweep after candidate
creation. The target mask is a semantic hint, while every observed point
remains in the sweep until a late, bounded finger/pad contact relation is
geometrically justified.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from .grasp_candidates import (
    CandidateRejection,
    GraspCandidate,
    GraspCandidateResult,
    _current_hand_world_boxes,
    _current_hand_world_spheres,
    _hand_world_boxes,
    _points_box_signed_clearance,
    _support_points,
    _validate_live_robot_geometry,
)


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1e-6)


def _support_surface_exemptions(
    worlds: np.ndarray,
    *,
    target_flags: np.ndarray,
    candidate: GraspCandidate,
    contact_audit: dict[str, Any],
    robot_calibration: Any,
    clearance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify terminal pad-footprint support points below a target surface.

    A support point is eligible only when the pad prism itself was certified
    for this candidate, the point lies inside its jaw-face footprint, and its
    world-height gap is a small, bounded amount below the promoted surface.
    The sweep still requires non-negative geometric clearance at the terminal
    finger/pad pose; support points never receive an exemption during transit.
    """
    result = np.zeros(len(worlds), dtype=bool)
    target_worlds = worlds[np.asarray(target_flags, dtype=bool)]
    if not len(target_worlds):
        return result, {"status": "unavailable", "reason": "no_target_component", "support_exempted_points": 0}
    if not isinstance(contact_audit, dict) or contact_audit.get("status") != "certified":
        return result, {"status": "unavailable", "reason": "uncertified_pad_geometry", "support_exempted_points": 0}
    try:
        x_low, x_high = (float(value) for value in contact_audit["capture_extent_x_m"])
        y_low, y_high = (float(value) for value in contact_audit["capture_extent_y_m"])
        z_low, z_high = (float(value) for value in contact_audit["capture_extent_z_m"])
    except (KeyError, TypeError, ValueError):
        return result, {"status": "unavailable", "reason": "incomplete_pad_capture_extent", "support_exempted_points": 0}
    if not all(np.isfinite(value) for value in (x_low, x_high, y_low, y_high, z_low, z_high)) or x_high <= x_low or y_high <= y_low or z_high <= z_low:
        return result, {"status": "unavailable", "reason": "degenerate_pad_capture_extent", "support_exempted_points": 0}
    try:
        max_aperture = float(robot_calibration.max_aperture_m)
    except (AttributeError, TypeError, ValueError):
        return result, {"status": "unavailable", "reason": "missing_max_aperture", "support_exempted_points": 0}
    if not np.isfinite(max_aperture) or max_aperture <= 0.0:
        return result, {"status": "unavailable", "reason": "invalid_max_aperture", "support_exempted_points": 0}
    contact = np.asarray(candidate.contact_world_m, dtype=np.float64)
    if contact.shape != (3,) or not np.isfinite(contact).all():
        return result, {"status": "unavailable", "reason": "invalid_candidate_contact", "support_exempted_points": 0}
    surface_z = float(np.max(target_worlds[:, 2]))
    # The support is certified only when the observed target component reaches
    # the candidate contact height. This prevents a random low scene patch from
    # becoming a support exemption.
    if not np.isfinite(surface_z) or abs(surface_z - float(contact[2])) > 0.012:
        return result, {"status": "unavailable", "reason": "target_surface_height_misaligned", "support_exempted_points": 0, "surface_z_m": surface_z, "contact_z_m": float(contact[2])}
    non_target = ~np.asarray(target_flags, dtype=bool)
    vertical_gap = surface_z - worlds[:, 2]
    # A support plane must be close to the object.  The bound scales with the
    # measured gripper aperture and is capped at 25 mm so a shelf or wall far
    # below the object cannot become an accidental exemption.
    support_band_m = min(0.5 * max_aperture, 0.025)
    R = np.asarray(candidate.rotation_world_grasp, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return result, {"status": "unavailable", "reason": "invalid_candidate_rotation", "support_exempted_points": 0}
    # The grasp frame's +X axis is the approach axis; its +Z axis is a
    # horizontal completion axis.  Build a symmetric world-XY footprint from
    # all eight measured pad-prism corners instead of assigning "down" to one
    # arbitrary local axis.  The resulting set is invariant to the equivalent
    # jaw-flip R @ diag(1,-1,-1).
    local_corners = np.asarray([
        [x, y, z]
        for x in (x_low, x_high)
        for y in (y_low, y_high)
        for z in (z_low, z_high)
    ], dtype=np.float64)
    world_corners = local_corners @ R.T + contact[None, :]
    support_radius = float(np.max(np.linalg.norm(world_corners[:, :2] - contact[None, :2], axis=1)))
    if not np.isfinite(support_radius) or support_radius <= 0.0:
        return result, {"status": "unavailable", "reason": "degenerate_world_pad_footprint", "support_exempted_points": 0}
    horizontal_distance = np.linalg.norm(worlds[:, :2] - contact[None, :2], axis=1)
    in_pad_footprint = horizontal_distance <= support_radius + 1e-6
    result = non_target & in_pad_footprint & (vertical_gap >= float(clearance_m)) & (vertical_gap <= support_band_m)
    return result, {
        "status": "applied",
        "reason": "terminal_pad_footprint_support_below_promoted_contact",
        "support_exempted_points": int(result.sum()),
        "support_band_m": float(support_band_m),
        "world_pad_footprint_radius_m": support_radius,
        "surface_z_m": surface_z,
        "contact_z_m": float(contact[2]),
        "min_vertical_gap_m": (float(np.min(vertical_gap[result])) if np.any(result) else None),
        "max_vertical_gap_m": (float(np.max(vertical_gap[result])) if np.any(result) else None),
        "max_aperture_m": max_aperture,
        "clearance_m": float(clearance_m),
        "temporal_gate": "terminal_only",
        "contact_relation": "measured_pad_footprint_support",
    }


def _terminal_pad_capture_mask(
    worlds: np.ndarray,
    *,
    candidate: GraspCandidate,
    primitives: list[tuple[str, Any]],
    target_flags: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Certify target points inside the measured terminal pad capture prism.

    The semantic RGB-D component is deliberately not treated as an
    authoritative collision mask.  A point is certifiable only inside the
    overlap of the two inward pad faces and their measured approach and
    completion extents.  Missing or ambiguous pad geometry disables the
    exemption, leaving every target point in the obstacle cloud.
    """
    result = np.zeros(len(worlds), dtype=bool)
    pad_boxes = [
        value for kind, value in primitives
        if kind == "box"
        and "finger" in str(value.get("geom_name", "")).lower()
        and "pad" in str(value.get("geom_name", "")).lower()
    ]
    audit: dict[str, Any] = {
        "status": "unavailable",
        "reason": "missing_or_ambiguous_pad_geometry",
        "pad_count": int(len(pad_boxes)),
        "certified_contact_points": 0,
    }
    if len(pad_boxes) != 2 or not len(worlds):
        return result, audit
    R = np.asarray(candidate.rotation_world_grasp, dtype=np.float64)
    contact = np.asarray(candidate.contact_world_m, dtype=np.float64)
    local_centers: list[np.ndarray] = []
    local_extents: list[np.ndarray] = []
    for box in pad_boxes:
        box_rotation = np.asarray(box["rotation_world_box"], dtype=np.float64)
        half_extents = np.asarray(box["half_extents_m"], dtype=np.float64)
        # Project each OBB onto the candidate grasp axes.  This remains valid
        # for a calibrated pad box whose local axes are not exactly identity.
        extent = np.abs(R.T @ box_rotation) @ half_extents
        center = R.T @ (np.asarray(box["center_world_m"], dtype=np.float64) - contact)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            return result, audit
        local_centers.append(center)
        local_extents.append(extent)
    order = np.argsort([value[1] for value in local_centers])
    low_index, high_index = (int(order[0]), int(order[1]))
    low_center, high_center = local_centers[low_index], local_centers[high_index]
    low_extent, high_extent = local_extents[low_index], local_extents[high_index]
    # The pads must sit on opposite jaw sides with a positive inward gap.
    low_face = float(low_center[1] + low_extent[1])
    high_face = float(high_center[1] - high_extent[1])
    if not (low_center[1] < high_center[1] and high_face > low_face):
        return result, {**audit, "reason": "degenerate_pad_gap"}
    x_low = max(float(low_center[0] - low_extent[0]), float(high_center[0] - high_extent[0]))
    x_high = min(float(low_center[0] + low_extent[0]), float(high_center[0] + high_extent[0]))
    z_low = max(float(low_center[2] - low_extent[2]), float(high_center[2] - high_extent[2]))
    z_high = min(float(low_center[2] + low_extent[2]), float(high_center[2] + high_extent[2]))
    if x_high <= x_low or z_high <= z_low:
        return result, {**audit, "reason": "non_overlapping_pad_capture_extents"}
    # This epsilon covers RGB-D quantization without turning the measured pad
    # footprint into a second broad clearance margin.  The full 6 mm margin
    # remains active in the collision test itself.
    contact_epsilon = min(0.001, float(clearance_m))
    local_worlds = (worlds - contact[None, :]) @ R
    within_prism = (
        (local_worlds[:, 1] >= low_face - contact_epsilon)
        & (local_worlds[:, 1] <= high_face + contact_epsilon)
        & (local_worlds[:, 0] >= x_low - contact_epsilon)
        & (local_worlds[:, 0] <= x_high + contact_epsilon)
        & (local_worlds[:, 2] >= z_low - contact_epsilon)
        & (local_worlds[:, 2] <= z_high + contact_epsilon)
    )
    result = np.asarray(target_flags, dtype=bool) & within_prism
    audit = {
        "status": "certified",
        "reason": "inside_terminal_pad_capture_prism",
        "pad_count": 2,
        "pad_gap_m": float(high_face - low_face),
        "capture_extent_y_m": [float(low_face), float(high_face)],
        "capture_extent_x_m": [float(x_low), float(x_high)],
        "capture_extent_z_m": [float(z_low), float(z_high)],
        "certified_contact_points": int(result.sum()),
        "contact_epsilon_m": float(contact_epsilon),
        "clearance_m": float(clearance_m),
    }
    return result, audit


def _hand_sweep(
    candidate: GraspCandidate,
    *,
    worlds: np.ndarray,
    pixels_vu: np.ndarray,
    target_flags: np.ndarray,
    robot_calibration: Any,
    calibration: Any,
    clearance_m: float,
) -> tuple[bool, dict[str, Any]]:
    current_grip, current_rotation, hand_spheres, hand_boxes = _validate_live_robot_geometry(robot_calibration)
    if current_grip is None or current_rotation is None:
        ignored_spheres: tuple[tuple[np.ndarray, float], ...] = ()
        ignored_boxes: tuple[dict[str, Any], ...] = ()
    elif hand_boxes:
        ignored_spheres = ()
        ignored_boxes = _current_hand_world_boxes(
            current_grip, current_rotation,
            np.asarray(robot_calibration.grasp_to_grip_site, dtype=np.float64).reshape(3, 3),
            np.asarray(robot_calibration.contact_to_grip_site_m, dtype=np.float64).reshape(3),
            hand_boxes,
        )
    else:
        ignored_boxes = ()
        ignored_spheres = _current_hand_world_spheres(
            current_grip, current_rotation,
            np.asarray(robot_calibration.grasp_to_grip_site, dtype=np.float64).reshape(3, 3),
            np.asarray(robot_calibration.contact_to_grip_site_m, dtype=np.float64).reshape(3),
            hand_spheres,
        )
    # Keep the same precedence as the frozen generator: calibrated live OBBs
    # describe the compiled hand envelope, so their conservative spheres are
    # not unioned into the sweep when both are present.  Mixing both changes
    # the collision contract and can reject otherwise valid grasps.
    sweep_spheres = () if hand_boxes else hand_spheres
    sweep_boxes = hand_boxes
    primitives: list[tuple[str, Any]] = [("sphere", value) for value in sweep_spheres]
    primitives.extend(("box", value) for value in _hand_world_boxes(
        np.asarray(candidate.contact_world_m, dtype=np.float64),
        np.asarray(candidate.rotation_world_grasp, dtype=np.float64),
        sweep_boxes,
    ))
    direction = np.asarray(candidate.pregrasp_world_m) - np.asarray(candidate.grip_site_world_m)
    path_length = float(np.linalg.norm(direction))
    sample_count = max(2, int(np.ceil(path_length / 0.01)) + 1)
    nearest_clearance = float("inf")
    certified_target, contact_audit = _terminal_pad_capture_mask(
        worlds,
        candidate=candidate,
        primitives=primitives,
        target_flags=target_flags,
        clearance_m=float(clearance_m),
    )
    support_exempt, support_audit = _support_surface_exemptions(
        worlds, target_flags=target_flags, candidate=candidate,
        contact_audit=contact_audit, robot_calibration=robot_calibration,
        clearance_m=float(clearance_m),
    )
    for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, sample_count)):
        translation = direction * (1.0 - float(fraction))
        for primitive_index, (primitive_type, primitive) in enumerate(primitives):
            if primitive_type == "sphere":
                center_grasp, radius = primitive
                center = np.asarray(candidate.contact_world_m) + np.asarray(candidate.rotation_world_grasp) @ center_grasp + translation
                signed = np.linalg.norm(worlds - center[None, :], axis=1) - float(radius) if len(worlds) else np.empty(0)
                primitive_name = None
            else:
                center = np.asarray(primitive["center_world_m"]) + translation
                signed = _points_box_signed_clearance(worlds, center, primitive["rotation_world_box"], primitive["half_extents_m"]) if len(worlds) else np.empty(0)
                primitive_name = primitive.get("geom_name")
            if len(signed):
                finger_or_pad = primitive_type == "box" and any(
                    token in str(primitive_name or "").lower()
                    for token in ("finger", "pad")
                )
                target_allowed = (
                    certified_target
                    & (float(fraction) >= 0.75)
                    & bool(finger_or_pad)
                    & (signed <= float(clearance_m))
                )
                # Local support can be exempted only for the measured fingers
                # or pads. Palm/hand primitives always see the support point.
                support_allowed = (
                    support_exempt
                    & (float(fraction) >= 0.75)
                    & bool(finger_or_pad)
                    # Never forgive a terminal penetration.  The measured
                    # support relation may waive the 6 mm safety band only
                    # when the actual pad OBB remains outside the support.
                    & (signed > 0.0)
                )
                eligible = np.where(target_allowed | support_allowed, np.inf, signed)
                nearest_clearance = min(nearest_clearance, float(np.min(eligible)))
            else:
                eligible = signed
            if len(eligible) and np.any(eligible <= float(clearance_m)):
                point_index = int(np.argmin(eligible))
                return False, {
                    "status": "collision",
                    "reason": "target_aware_approach_obstruction",
                    "offender_in_target_component": bool(target_flags[point_index]),
                    "closest_offending_pixel_uv": [int(pixels_vu[point_index, 1]), int(pixels_vu[point_index, 0])],
                    "spatial_distance_m": float(np.linalg.norm(worlds[point_index] - center)),
                    "primitive_index": primitive_index,
                    "primitive_type": primitive_type,
                    "primitive_name": primitive_name,
                    "segment_fraction": float(fraction),
                    "segment_sample_index": sample_index,
                    "segment_sample_count": sample_count,
                    "threshold_m": float(clearance_m),
                    "target_component_excluded": False,
                    "certified_target_contact": bool(certified_target[point_index]),
                    "target_contact_relation": "late_finger_or_pad_only",
                    "contact_prism": contact_audit,
                    "support_surface": support_audit,
                    "temporal_gate": "same_frame_rgbd",
                }
    # The target may touch fingers/pads at terminal contact, but must not
    # penetrate the palm/hand body. This check is intentionally terminal-only.
    for primitive_index, (primitive_type, primitive) in enumerate(primitives):
        if primitive_type != "box" or "hand" not in str(primitive.get("geom_name", "")).lower():
            continue
        target_worlds = worlds[np.asarray(target_flags, dtype=bool)]
        signed = _points_box_signed_clearance(
            target_worlds,
            np.asarray(primitive["center_world_m"]),
            primitive["rotation_world_box"],
            primitive["half_extents_m"],
        ) if len(target_worlds) else np.empty(0)
        if len(signed) and np.any(signed <= 0.0):
            point_index = int(np.argmin(signed))
            return False, {
                "status": "collision",
                "reason": "target_palm_terminal_collision",
                "offender_in_target_component": True,
                "primitive_index": primitive_index,
                "primitive_type": primitive_type,
                "primitive_name": primitive.get("geom_name"),
                "spatial_distance_m": float(np.linalg.norm(target_worlds[point_index] - np.asarray(primitive["center_world_m"]))),
                "target_component_excluded": False,
                "target_contact_relation": "palm_rejected",
                "temporal_gate": "terminal_contact_only",
            }
    return True, {
        "status": "ok",
        "reason": "target_contact_relation_full_sweep",
        "offender_in_target_component": None,
        "target_component_excluded": False,
        "non_target_point_count": int(np.count_nonzero(~np.asarray(target_flags, dtype=bool))),
        "target_point_count": int(np.count_nonzero(np.asarray(target_flags, dtype=bool))),
        "target_contact_relation": "late_finger_or_pad_only",
        "contact_prism": contact_audit,
        "support_surface": support_audit,
        "clearance_m": (float(nearest_clearance) if np.isfinite(nearest_clearance) else float("inf")),
        "temporal_gate": "same_frame_rgbd",
    }


def filter_target_aware_candidates(
    result: GraspCandidateResult,
    *,
    capture: Any,
    target_mask: np.ndarray,
    robot_calibration: Any,
    clearance_m: float = 0.006,
) -> GraspCandidateResult:
    """Filter obstruction-disabled candidates against non-target RGB-D points."""
    if not np.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("target-aware collision clearance must be finite and non-negative")
    depth = np.asarray(capture.metric_depth, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("target-aware collision requires a same-frame target mask")
    calibration = capture.calibration
    K = np.asarray(calibration.intrinsic, dtype=np.float64)
    T = np.asarray(calibration.world_from_camera, dtype=np.float64)
    pixels = np.column_stack(np.nonzero(_valid_depth(depth)))
    worlds, kept = _support_points(pixels, depth, K, T)
    if len(worlds) == 0:
        raise ValueError("target-aware collision requires finite RGB-D support")
    target_flags = mask[kept[:, 0], kept[:, 1]]
    target_mask_sha256 = hashlib.sha256(mask.tobytes()).hexdigest()
    current_grip, current_rotation, hand_spheres, hand_boxes = _validate_live_robot_geometry(robot_calibration)
    keep = np.ones(len(worlds), dtype=bool)
    if current_grip is not None and current_rotation is not None:
        if hand_boxes:
            ignored = _current_hand_world_boxes(
                current_grip, current_rotation,
                np.asarray(robot_calibration.grasp_to_grip_site).reshape(3, 3),
                np.asarray(robot_calibration.contact_to_grip_site_m).reshape(3), hand_boxes,
            )
            for box in ignored:
                keep &= _points_box_signed_clearance(worlds, box["center_world_m"], box["rotation_world_box"], box["half_extents_m"]) > float(clearance_m)
        else:
            ignored = _current_hand_world_spheres(
                current_grip, current_rotation,
                np.asarray(robot_calibration.grasp_to_grip_site).reshape(3, 3),
                np.asarray(robot_calibration.contact_to_grip_site_m).reshape(3), hand_spheres,
            )
            for center, radius in ignored:
                keep &= np.linalg.norm(worlds - center[None, :], axis=1) > float(radius) + float(clearance_m)
    worlds, kept, target_flags = worlds[keep], kept[keep], target_flags[keep]
    accepted: list[GraspCandidate] = []
    rejected = list(result.rejected)
    for candidate in result.candidates:
        valid, audit = _hand_sweep(
            candidate, worlds=worlds, pixels_vu=kept, target_flags=target_flags,
            robot_calibration=robot_calibration,
            calibration=calibration, clearance_m=float(clearance_m),
        )
        if not valid:
            rejected.append(CandidateRejection(
                candidate.seed_index, candidate.yaw_deg, candidate.insertion_depth_m,
                str(audit.get("reason", "target_aware_collision")), audit,
            ))
            continue
        accepted.append(replace(
            candidate,
            # The frozen generator ran with obstruction disabled, therefore
            # its clearance field is +inf.  Replace it with the B0 target
            # aware sweep result so downstream ranking/audits see the actual
            # executable margin.
            clearance_m=float(audit.get("clearance_m", float("inf"))),
            audit={**dict(candidate.audit), "target_aware_collision": audit},
        ))
    return GraspCandidateResult(
        tuple(accepted), tuple(rejected), result.seeds_uv, result.policy,
        {
            **dict(result.audit),
            "target_aware_collision": {
                "status": "applied",
                "target_component_excluded": False,
                "target_mask_role": "strict_contact_component_only",
                "target_mask_sha256": target_mask_sha256,
                "target_contact_relation": "late_finger_or_pad_only",
                "clearance_m": float(clearance_m),
                "observed_point_count": int(len(worlds)),
                "non_target_point_count": int(np.count_nonzero(~target_flags)),
                "target_point_count": int(np.count_nonzero(target_flags)),
            },
        },
    )


__all__ = ["filter_target_aware_candidates"]
