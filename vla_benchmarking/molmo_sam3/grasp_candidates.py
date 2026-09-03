"""Pure RGB-D geometry for MolmoPoint/SAM3 bowl grasp candidates.

This module deliberately has no simulator, evaluator, or model imports.  The
perception worker supplies an original-image SAM mask and MolmoPoint pixels;
this module turns those observations into executable, fully oriented
``grip_site`` poses.  All transforms are explicit and the signed ``fy`` in
the supplied calibration is preserved (the project uses a Robosuite
projection convention with a converted image vertical axis).

The output is diagnostic geometry until a caller validates it against the
robot's live calibration and motion safety checks.  In particular, a Molmo
point is a proposal for a visible rim location, never a robot target by
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, cos, pi, sin
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


class CameraCalibrationLike(Protocol):
    """Structural camera-calibration contract required by this module."""

    width: int
    height: int
    intrinsic: ArrayLike
    world_from_camera: ArrayLike


@dataclass(frozen=True)
class CameraCalibration:
    """Minimal original-image RGB-D calibration.

    ``intrinsic`` must map the OpenCV optical camera frame (x right, y down,
    z forward) to pixels.  A negative ``intrinsic[1][1]`` is valid and is
    intentionally not normalized away.
    """

    width: int
    height: int
    intrinsic: tuple[tuple[float, float, float], ...]
    world_from_camera: tuple[tuple[float, float, float, float], ...]
    camera_name: str = "unknown"


@dataclass(frozen=True)
class RGBDObservation:
    """One aligned, original-resolution observation used for geometry."""

    rgb: np.ndarray | None
    metric_depth_m: np.ndarray
    calibration: CameraCalibrationLike


@dataclass(frozen=True)
class MolmoPoint:
    """MolmoPoint output in original-image pixel coordinates."""

    u: float
    v: float
    confidence: float = 1.0
    label: str = "rim"

    @classmethod
    def from_value(cls, value: Any) -> "MolmoPoint":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            u = value.get("u", value.get("x"))
            v = value.get("v", value.get("y"))
            confidence = value.get("confidence", value.get("score", 1.0))
            label = value.get("label", "rim")
        else:
            values = np.asarray(value, dtype=float).reshape(-1)
            if values.size < 2:
                raise ValueError("MolmoPoint requires at least u and v")
            u, v = values[:2]
            confidence = values[2] if values.size >= 3 else 1.0
            label = "rim"
        point = cls(float(u), float(v), float(confidence), str(label))
        if not np.all(np.isfinite((point.u, point.v, point.confidence))):
            raise ValueError("MolmoPoint coordinates and confidence must be finite")
        if point.confidence < 0:
            raise ValueError("MolmoPoint confidence must be non-negative")
        return point


@dataclass(frozen=True)
class RobotGraspCalibration:
    """Verified robot-side relation for grasp-frame geometry.

    The local grasp frame is right-handed and has ``+X`` along the palm to
    object approach, ``+Y`` along jaw opening, and ``+Z`` completing the
    frame.  ``grasp_to_grip_site`` maps those axes into the controller's
    ``grip_site`` axes.  ``contact_to_grip_site_m`` is measured in the local
    grasp frame and is zero by default: no legacy v9d source offset is ever
    inserted here.
    """

    grasp_to_grip_site: ArrayLike = field(default_factory=lambda: np.eye(3))
    contact_to_grip_site_m: ArrayLike = (0.0, 0.0, 0.0)
    approach_axis_world: ArrayLike = (0.0, 0.0, -1.0)
    min_aperture_m: float = 0.005
    max_aperture_m: float = 0.080
    finger_clearance_m: float = 0.004
    pregrasp_distance_m: float = 0.080
    workspace_min_m: ArrayLike = (-0.8, -0.8, 0.0)
    workspace_max_m: ArrayLike = (0.8, 0.8, 1.8)
    calibration_source: str = "unverified"
    calibration_sha256: str | None = None
    # These are optional live robot-only probe values.  They are deliberately
    # absent from the legacy pure RGB-D fixtures: when present together they
    # make candidate ranking relative to the measured robot pose, and the
    # contact-relative spheres provide a bounded hand-volume collision check.
    current_grip_site_world_m: ArrayLike | None = None
    current_rotation_world_grip_site: ArrayLike | None = None
    hand_collision_spheres_grasp: Sequence[Any] | None = None
    hand_collision_boxes_grasp: Sequence[Any] | None = None


@dataclass(frozen=True)
class CandidateRejection:
    seed_index: int
    yaw_deg: float
    insertion_depth_m: float
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspCandidate:
    """One feasible candidate expressed in explicit world/robot frames."""

    candidate_id: str
    seed_index: int
    source_pixel_uv: tuple[float, float]
    molmo_distance_px: float | None
    yaw_deg: float
    insertion_depth_m: float
    contact_world_m: np.ndarray
    grip_site_world_m: np.ndarray
    pregrasp_world_m: np.ndarray
    release_world_m: np.ndarray | None
    rotation_world_grasp: np.ndarray
    rotation_world_grip_site: np.ndarray
    quaternion_world_grip_site_xyzw: np.ndarray
    jaw_axis_world: np.ndarray
    rim_tangent_world: np.ndarray
    required_aperture_m: float
    depth_support_count: int
    clearance_m: float
    score: float
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspCandidateResult:
    candidates: tuple[GraspCandidate, ...]
    rejected: tuple[CandidateRejection, ...]
    seeds_uv: tuple[tuple[int, int], ...]
    policy: str
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePolicy:
    """Deterministic knobs shared by the canary ablations."""

    name: str = "molmo_dense"
    max_seeds: int = 16
    max_candidates: int = 128
    molmo_snap_radius_px: int = 12
    local_seed_radius_px: int = 28
    rim_support_radius_px: int = 24
    contact_mode: str = "observed_upper_rim"
    rim_height_quantile: float = 0.90
    rim_height_band_m: float = 0.008
    rim_local_radius_m: float = 0.015
    min_rim_support_pixels: int = 3
    min_depth_support_pixels: int = 3
    dedupe_position_m: float = 0.002
    dedupe_rotation_deg: float = 4.0
    yaw_offsets_deg: tuple[float, ...] = (-15.0, 0.0, 15.0)
    insertion_depths_m: tuple[float, ...] = (0.0, 0.004, 0.008)
    obstruction_clearance_m: float | None = None
    release_world_m: ArrayLike | None = None
    # Target-mask points are allowed only in this small neighborhood of the
    # intended terminal contact.  Scene points outside it remain obstacles.
    terminal_contact_allowance_m: float = 0.012


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source[key]
    return getattr(source, key)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _validate_camera(calibration: CameraCalibrationLike, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    try:
        cal_width = int(_value(calibration, "width"))
        cal_height = int(_value(calibration, "height"))
        K = np.asarray(_value(calibration, "intrinsic"), dtype=np.float64)
        T = np.asarray(_value(calibration, "world_from_camera"), dtype=np.float64)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("calibration must expose width, height, intrinsic, and world_from_camera") from exc
    if (cal_height, cal_width) != (height, width):
        raise ValueError(f"calibration/image shape mismatch: {(cal_height, cal_width)} != {(height, width)}")
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or abs(K[0, 0]) <= 1e-9 or abs(K[1, 1]) <= 1e-9:
        raise ValueError("intrinsic must be finite 3x3 with non-zero signed focal lengths")
    if not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError("intrinsic homogeneous row must be [0, 0, 1]")
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], (0, 0, 0, 1), atol=1e-7):
        raise ValueError("world_from_camera must be a finite homogeneous 4x4 matrix")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise ValueError("world_from_camera rotation must be proper orthonormal")
    return K, T


def _validate_robot(robot: RobotGraspCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_ge = _finite_vector(robot.grasp_to_grip_site, 9, "grasp_to_grip_site").reshape(3, 3)
    if not np.allclose(R_ge.T @ R_ge, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_ge), 1.0, atol=1e-5):
        raise ValueError("grasp_to_grip_site must be proper orthonormal")
    offset = _finite_vector(robot.contact_to_grip_site_m, 3, "contact_to_grip_site_m")
    approach = _finite_vector(robot.approach_axis_world, 3, "approach_axis_world")
    approach /= np.linalg.norm(approach)
    if abs(float(approach[2])) < 0.9:
        raise ValueError("upright grasp approach_axis_world must be within 25.8 degrees of world vertical")
    lo = _finite_vector(robot.workspace_min_m, 3, "workspace_min_m")
    hi = _finite_vector(robot.workspace_max_m, 3, "workspace_max_m")
    if np.any(lo >= hi):
        raise ValueError("workspace_min_m must be strictly below workspace_max_m")
    if not (np.isfinite(robot.min_aperture_m) and np.isfinite(robot.max_aperture_m) and 0 < robot.min_aperture_m <= robot.max_aperture_m):
        raise ValueError("aperture range must be finite, positive, and ordered")
    if not np.isfinite(robot.finger_clearance_m) or robot.finger_clearance_m < 0:
        raise ValueError("finger_clearance_m must be finite and non-negative")
    if not np.isfinite(robot.pregrasp_distance_m) or robot.pregrasp_distance_m < 0:
        raise ValueError("pregrasp_distance_m must be finite and non-negative")
    return R_ge, offset, approach, lo, hi


def _validate_live_robot_geometry(
    robot: RobotGraspCalibration,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[tuple[np.ndarray, float], ...], tuple[dict[str, Any], ...]]:
    """Validate optional live-pose and contact-relative hand primitives."""
    current_position = None
    current_rotation = None
    if robot.current_grip_site_world_m is not None:
        current_position = _finite_vector(robot.current_grip_site_world_m, 3, "current_grip_site_world_m")
    if robot.current_rotation_world_grip_site is not None:
        value = np.asarray(robot.current_rotation_world_grip_site, dtype=np.float64)
        if value.shape != (3, 3) or not np.all(np.isfinite(value)):
            raise ValueError("current_rotation_world_grip_site must be a finite 3x3 matrix")
        if not np.allclose(value.T @ value, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(value), 1.0, atol=1e-5):
            raise ValueError("current_rotation_world_grip_site must be proper orthonormal")
        current_rotation = value
    if (current_position is None) != (current_rotation is None):
        raise ValueError("current_grip_site_world_m and current_rotation_world_grip_site must be supplied together")
    spheres: list[tuple[np.ndarray, float]] = []
    raw_spheres = robot.hand_collision_spheres_grasp
    for index, item in enumerate(() if raw_spheres is None else raw_spheres):
        if isinstance(item, Mapping):
            center_value = item.get("center", item.get("center_grasp", item.get("center_grasp_m", item.get("xyz"))))
            radius_value = item.get("radius_m", item.get("radius", item.get("geom_rbound_m")))
        else:
            # Accept both flat ``(x, y, z, radius)`` records and the natural
            # ``((x, y, z), radius)`` pair used by live probe callers.
            try:
                values = np.asarray(item, dtype=np.float64).reshape(-1)
                center_value = values[:3] if values.size >= 4 else None
                radius_value = values[3] if values.size >= 4 else None
            except (TypeError, ValueError):
                try:
                    center_value, radius_value = item
                except (TypeError, ValueError):
                    center_value = getattr(item, "center_grasp_m", getattr(item, "center_grasp", getattr(item, "center", None)))
                    radius_value = getattr(item, "radius_m", getattr(item, "radius", getattr(item, "geom_rbound_m", None)))
        if center_value is None or radius_value is None:
            raise ValueError(f"hand_collision_spheres_grasp[{index}] must contain center and radius")
        center = _finite_vector(center_value, 3, f"hand_collision_spheres_grasp[{index}].center")
        radius = float(radius_value)
        if not np.isfinite(radius) or radius < 0:
            raise ValueError(f"hand_collision_spheres_grasp[{index}].radius must be finite and non-negative")
        spheres.append((center, radius))
    boxes: list[dict[str, Any]] = []
    raw_boxes = robot.hand_collision_boxes_grasp
    for index, item in enumerate(() if raw_boxes is None else raw_boxes):
        if not isinstance(item, Mapping):
            raise ValueError(f"hand_collision_boxes_grasp[{index}] must be a mapping")
        center = _finite_vector(item.get("center_grasp_m"), 3, f"hand_collision_boxes_grasp[{index}].center_grasp_m")
        rotation_value = np.asarray(item.get("rotation_grasp_box"), dtype=np.float64)
        if rotation_value.shape != (3, 3) or not np.all(np.isfinite(rotation_value)):
            raise ValueError(f"hand_collision_boxes_grasp[{index}].rotation_grasp_box must be a finite 3x3 matrix")
        if not np.allclose(rotation_value.T @ rotation_value, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation_value), 1.0, atol=1e-5):
            raise ValueError(f"hand_collision_boxes_grasp[{index}].rotation_grasp_box must be proper orthonormal")
        half_extents = _finite_vector(item.get("half_extents_m"), 3, f"hand_collision_boxes_grasp[{index}].half_extents_m")
        if np.any(half_extents <= 0):
            raise ValueError(f"hand_collision_boxes_grasp[{index}].half_extents_m must be strictly positive")
        boxes.append({
            "center_grasp_m": center,
            "rotation_grasp_box": rotation_value,
            "half_extents_m": half_extents,
            "geom_id": item.get("geom_id"),
            "geom_name": item.get("geom_name"),
            "source": item.get("source"),
        })
    return current_position, current_rotation, tuple(spheres), tuple(boxes)


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = np.ones_like(mask, dtype=bool)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if du == 0 and dv == 0:
                continue
            neighbors &= padded[1 + dv : 1 + dv + mask.shape[0], 1 + du : 1 + du + mask.shape[1]]
    return mask & ~neighbors


def _valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1e-6)


def _as_points(points: Sequence[Any] | None) -> tuple[MolmoPoint, ...]:
    if points is None:
        return ()
    return tuple(MolmoPoint.from_value(point) for point in points)


def _nearest_boundary(point: MolmoPoint, pixels: np.ndarray, radius: float) -> tuple[int, int] | None:
    if pixels.size == 0:
        return None
    delta = pixels.astype(float) - np.array((point.v, point.u), dtype=float)
    distances = np.sum(delta * delta, axis=1)
    nearest = int(np.argmin(distances))
    if float(np.sqrt(distances[nearest])) > radius:
        return None
    return int(pixels[nearest, 0]), int(pixels[nearest, 1])


def _uniform_pixels(pixels: np.ndarray, count: int) -> list[tuple[int, int]]:
    if len(pixels) == 0 or count <= 0:
        return []
    pixels = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))]
    indices = np.floor(np.linspace(0, len(pixels) - 1, min(count, len(pixels)))).astype(int)
    result: list[tuple[int, int]] = []
    for index in indices:
        item = (int(pixels[index, 0]), int(pixels[index, 1]))
        if item not in result:
            result.append(item)
    return result


def _make_seeds(boundary_pixels: np.ndarray, molmo: tuple[MolmoPoint, ...], policy: CandidatePolicy) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    if policy.name not in {"geometry_only", "molmo_local", "molmo_dense"}:
        raise ValueError("policy.name must be geometry_only, molmo_local, or molmo_dense")
    diagnostics: list[dict[str, Any]] = []
    snapped: list[tuple[int, int]] = []
    for point_index, point in enumerate(molmo):
        nearest = _nearest_boundary(point, boundary_pixels, policy.molmo_snap_radius_px)
        if nearest is None:
            diagnostics.append({"point_index": point_index, "status": "rejected", "reason": "outside_mask_or_snap_radius"})
            continue
        if nearest not in snapped:
            snapped.append(nearest)
            diagnostics.append({"point_index": point_index, "status": "accepted", "snapped_uv": [nearest[1], nearest[0]]})
    if policy.name == "geometry_only":
        return _uniform_pixels(boundary_pixels, policy.max_seeds), diagnostics
    if policy.name == "molmo_local":
        return snapped[: policy.max_seeds], diagnostics
    # Dense ablation: Molmo seeds lead, then deterministic distributed rim
    # samples fill coverage so a single poor point cannot starve retries.
    seeds = list(snapped)
    for item in _uniform_pixels(boundary_pixels, policy.max_seeds):
        if item not in seeds:
            seeds.append(item)
        if len(seeds) >= policy.max_seeds:
            break
    return seeds[: policy.max_seeds], diagnostics


def _deproject(u: float, v: float, depth: float, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    if not np.isfinite(depth) or depth <= 1e-6:
        raise ValueError("depth must be finite and positive")
    camera = np.array(((u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth), dtype=np.float64)
    world_h = T @ np.r_[camera, 1.0]
    if abs(world_h[3]) <= 1e-9:
        raise ValueError("deprojected homogeneous point has zero scale")
    result = world_h[:3] / world_h[3]
    if not np.all(np.isfinite(result)):
        raise ValueError("deprojected point is not finite")
    return result


def _local_pixels(mask: np.ndarray, center: tuple[int, int], radius: int, *, boundary_only: bool = False) -> np.ndarray:
    v, u = center
    yy, xx = np.indices(mask.shape)
    selected = mask & ((yy - v) ** 2 + (xx - u) ** 2 <= radius * radius)
    if boundary_only:
        selected &= _boundary(mask)
    return np.column_stack(np.nonzero(selected))


def _support_points(pixels: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    worlds: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for v, u in pixels:
        value = float(depth[int(v), int(u)])
        if not np.isfinite(value) or value <= 1e-6:
            continue
        try:
            worlds.append(_deproject(float(u), float(v), value, K, T))
            kept.append(np.array((int(v), int(u))))
        except ValueError:
            continue
    if not worlds:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    return np.asarray(worlds), np.asarray(kept, dtype=int)


def _rim_tangent(worlds: np.ndarray, pixels_vu: np.ndarray) -> np.ndarray:
    xy = worlds[:, :2]
    if len(xy) >= 2 and np.max(np.ptp(xy, axis=0)) > 1e-7:
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        tangent_xy = vectors[:, int(np.argmax(values))]
    else:
        centered = pixels_vu[:, ::-1].astype(float) - np.mean(pixels_vu[:, ::-1], axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangent_xy = vh[0] if len(vh) else np.array((1.0, 0.0))
    tangent = np.array((float(tangent_xy[0]), float(tangent_xy[1]), 0.0))
    norm = np.linalg.norm(tangent)
    if norm <= 1e-9:
        tangent = np.array((1.0, 0.0, 0.0))
    else:
        tangent /= norm
    # Fix the otherwise arbitrary PCA sign for reproducible yaw and scores.
    if tangent[0] < -1e-9 or (abs(tangent[0]) <= 1e-9 and tangent[1] < 0):
        tangent = -tangent
    return tangent


def _rotate_about(axis: np.ndarray, vector: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return vector * cos(angle_rad) + np.cross(axis, vector) * sin(angle_rad) + axis * np.dot(axis, vector) * (1.0 - cos(angle_rad))


def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    # Stable branch selection and xyzw convention matching Robosuite APIs.
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-15))
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2], 1e-15))
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(max(1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2], 1e-15))
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(acos(float(cosine)) * 180.0 / pi)


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0)
    return float(acos(float(cosine)))


def _inside_workspace(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.all(point >= lo - 1e-9) and np.all(point <= hi + 1e-9))


def _obstruction_clearance(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    clearance: float,
) -> dict[str, Any]:
    """Return nearest observed point and auditable segment diagnostics.

    This is the conservative legacy fallback when no live hand primitives are
    supplied.  Unlike the old helper it retains all observed RGB-D points and
    their pixel provenance, so a rejection cannot look like an unexplained
    scalar.
    """
    valid = _valid_depth(depth)
    pixels = np.column_stack(np.nonzero(valid))
    worlds, kept = _support_points(pixels, depth, K, T)
    if len(worlds) == 0:
        return {
            "status": "no_observed_scene",
            "clearance_m": float("inf"),
            "observed_point_count": 0,
            "segment": "pregrasp_to_grip_site",
        }
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-12:
        distances = np.linalg.norm(worlds - start[None, :], axis=1)
        factors = np.zeros(len(worlds), dtype=np.float64)
    else:
        factors = np.clip(((worlds - start[None, :]) @ direction) / length_sq, 0.0, 1.0)
        closest = start[None, :] + factors[:, None] * direction[None, :]
        distances = np.linalg.norm(worlds - closest, axis=1)
    nearest = int(np.argmin(distances))
    return {
        "status": "ok",
        "clearance_m": float(distances[nearest]),
        "closest_distance_m": float(distances[nearest]),
        "distance_m": float(distances[nearest]),
        "closest_world_m": worlds[nearest].tolist(),
        "closest_offending_world_m": worlds[nearest].tolist(),
        "closest_pixel_uv": [int(kept[nearest, 1]), int(kept[nearest, 0])],
        "closest_offending_pixel_uv": [int(kept[nearest, 1]), int(kept[nearest, 0])],
        "segment": "pregrasp_to_grip_site",
        "segment_factor": float(factors[nearest]),
        "observed_point_count": int(len(worlds)),
        "threshold_m": float(clearance),
    }


def _current_hand_world_spheres(
    current_grip: np.ndarray,
    current_rotation_grip: np.ndarray,
    grasp_to_grip: np.ndarray,
    contact_offset: np.ndarray,
    spheres: tuple[tuple[np.ndarray, float], ...],
) -> tuple[tuple[np.ndarray, float], ...]:
    """Transform contact-relative primitives into the measured robot frame."""
    current_grasp = current_rotation_grip @ grasp_to_grip.T
    current_contact = current_grip - current_grasp @ contact_offset
    return tuple((current_contact + current_grasp @ center, radius) for center, radius in spheres)


def _hand_world_boxes(
    contact: np.ndarray,
    rotation_world_grasp: np.ndarray,
    boxes: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Transform contact-relative oriented boxes into world coordinates."""
    result: list[dict[str, Any]] = []
    for box in boxes:
        result.append({
            **box,
            "center_world_m": contact + rotation_world_grasp @ box["center_grasp_m"],
            "rotation_world_box": rotation_world_grasp @ box["rotation_grasp_box"],
        })
    return tuple(result)


def _current_hand_world_boxes(
    current_grip: np.ndarray,
    current_rotation_grip: np.ndarray,
    grasp_to_grip: np.ndarray,
    contact_offset: np.ndarray,
    boxes: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    current_grasp = current_rotation_grip @ grasp_to_grip.T
    current_contact = current_grip - current_grasp @ contact_offset
    return _hand_world_boxes(current_contact, current_grasp, boxes)


def _point_box_signed_clearance(
    point_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_box: np.ndarray,
    half_extents: np.ndarray,
) -> float:
    """Signed distance to an OBB: positive outside, negative inside."""
    return float(_points_box_signed_clearance(point_world[None, :], center_world, rotation_world_box, half_extents)[0])


def _points_box_signed_clearance(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_box: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Vectorized signed distances for points against one oriented box."""
    local = (points_world - center_world[None, :]) @ rotation_world_box
    delta = np.abs(local) - half_extents[None, :]
    outside = np.maximum(delta, 0.0)
    return np.linalg.norm(outside, axis=1) + np.minimum(np.max(delta, axis=1), 0.0)


def _hand_volume_obstruction(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    contact: np.ndarray,
    grip_site: np.ndarray,
    pregrasp: np.ndarray,
    rotation_world_grasp: np.ndarray,
    spheres: tuple[tuple[np.ndarray, float], ...],
    clearance: float,
    terminal_allowance_m: float,
    required_aperture_m: float,
    boxes: tuple[dict[str, Any], ...] = (),
    ignored_robot_spheres: tuple[tuple[np.ndarray, float], ...] = (),
    ignored_robot_boxes: tuple[dict[str, Any], ...] = (),
    observed_worlds: np.ndarray | None = None,
    observed_pixels_vu: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check a swept, contact-relative hand envelope against observed RGB-D.

    The object mask is not removed globally.  Only target-mask samples within
    ``terminal_allowance_m`` of the selected contact and at the terminal end
    of the approach are allowed; this preserves bowl/scene collision checks.
    """
    if observed_worlds is None or observed_pixels_vu is None:
        pixels = np.column_stack(np.nonzero(_valid_depth(depth)))
        worlds, kept = _support_points(pixels, depth, K, T)
    else:
        worlds, kept = observed_worlds, observed_pixels_vu
    if len(worlds) == 0:
        return {
            "status": "no_observed_scene",
            "clearance_m": float("inf"),
            "observed_point_count": 0,
            "segment": "pregrasp_to_contact",
            "aperture_m": float(required_aperture_m),
        }

    # The depth image contains the currently measured robot in some setups.
    # Remove only points inside the measured robot envelope; never erase an
    # arbitrary image region or the entire target mask.
    keep = np.ones(len(worlds), dtype=bool)
    for center, radius in ignored_robot_spheres:
        keep &= np.linalg.norm(worlds - center[None, :], axis=1) > radius + clearance
    for box in ignored_robot_boxes:
        keep &= _points_box_signed_clearance(
            worlds, box["center_world_m"], box["rotation_world_box"], box["half_extents_m"]
        ) > clearance
    worlds = worlds[keep]
    kept = kept[keep]
    if len(worlds) == 0:
        return {
            "status": "no_observed_scene_after_robot_exclusion",
            "clearance_m": float("inf"),
            "observed_point_count": 0,
            "ignored_robot_point_count": int(np.count_nonzero(~keep)),
            "segment": "pregrasp_to_contact",
            "aperture_m": float(required_aperture_m),
        }

    direction = pregrasp - grip_site
    path_length = float(np.linalg.norm(direction))
    sample_count = max(2, int(np.ceil(path_length / 0.01)) + 1)
    nearest: dict[str, Any] | None = None
    nearest_gap = float("inf")
    primitives: list[tuple[str, Any]] = [("sphere", sphere) for sphere in spheres]
    primitives.extend(("box", box) for box in _hand_world_boxes(contact, rotation_world_grasp, boxes))
    for sample_index, t in enumerate(np.linspace(0.0, 1.0, sample_count)):
        # At t=1 the hand is in its contact-relative terminal pose.  The
        # selected pregrasp is the same orientation translated along approach.
        translation = direction * (1.0 - float(t))
        for primitive_index, (primitive_type, primitive) in enumerate(primitives):
            if primitive_type == "sphere":
                center_grasp, radius = primitive
                center = contact + rotation_world_grasp @ center_grasp + translation
                distances = np.linalg.norm(worlds - center[None, :], axis=1)
                signed_clearances = distances - radius
                rotation_box = None
                half_extents = None
            else:
                box = primitive
                center = box["center_world_m"] + translation
                rotation_box = box["rotation_world_box"]
                half_extents = box["half_extents_m"]
                signed_clearances = _points_box_signed_clearance(
                    worlds, center, rotation_box, half_extents
                )
                distances = signed_clearances
            target_allowance = (
                mask[kept[:, 0], kept[:, 1]]
                & (t >= 0.75)
                & (np.linalg.norm(worlds - contact[None, :], axis=1) <= terminal_allowance_m)
            )
            eligible_gap = np.where(target_allowance, np.inf, signed_clearances)
            nearest_index = int(np.argmin(eligible_gap))
            if np.isfinite(eligible_gap[nearest_index]) and float(eligible_gap[nearest_index]) < nearest_gap:
                nearest_gap = float(eligible_gap[nearest_index])
                nearest = {
                    "closest_distance_m": float(distances[nearest_index]),
                    "clearance_m": nearest_gap,
                    "distance_m": float(distances[nearest_index]),
                    "closest_world_m": worlds[nearest_index].tolist(),
                    "closest_offending_world_m": worlds[nearest_index].tolist(),
                    "closest_pixel_uv": [int(kept[nearest_index, 1]), int(kept[nearest_index, 0])],
                    "closest_offending_pixel_uv": [int(kept[nearest_index, 1]), int(kept[nearest_index, 0])],
                    "primitive_index": primitive_index,
                    "primitive_center_world_m": center.tolist(),
                    "primitive_radius_m": (float(radius) if primitive_type == "sphere" else None),
                    "primitive_type": primitive_type,
                    "primitive_half_extents_m": (half_extents.tolist() if primitive_type == "box" else None),
                    "primitive_rotation_world_box": (rotation_box.tolist() if primitive_type == "box" else None),
                    "primitive_rotation_grasp_box": (primitive.get("rotation_grasp_box").tolist() if primitive_type == "box" else None),
                    "geom_id": (primitive.get("geom_id") if primitive_type == "box" else None),
                    "geom_name": (primitive.get("geom_name") if primitive_type == "box" else None),
                    "source": (primitive.get("source") if primitive_type == "box" else None),
                    "segment": "pregrasp_to_contact",
                    "segment_sample_index": sample_index,
                    "segment_sample_count": sample_count,
                    "segment_fraction": float(t),
                    "threshold_m": float(clearance),
                    "aperture_m": float(required_aperture_m),
                }
            collision_indices = np.flatnonzero((signed_clearances <= clearance) & ~target_allowance)
            if len(collision_indices):
                point_index = int(collision_indices[np.argmin(signed_clearances[collision_indices])])
                distance = float(distances[point_index])
                gap = float(signed_clearances[point_index])
                # Report this actual blocking point, rather than an earlier
                # target-mask point that was allowed at terminal contact.
                details = {
                    "closest_distance_m": distance,
                    "clearance_m": gap,
                    "distance_m": distance,
                    "closest_world_m": worlds[point_index].tolist(),
                    "closest_offending_world_m": worlds[point_index].tolist(),
                    "closest_pixel_uv": [int(kept[point_index, 1]), int(kept[point_index, 0])],
                    "closest_offending_pixel_uv": [int(kept[point_index, 1]), int(kept[point_index, 0])],
                    "primitive_index": primitive_index,
                    "primitive_center_world_m": center.tolist(),
                    "primitive_radius_m": (float(radius) if primitive_type == "sphere" else None),
                    "primitive_type": primitive_type,
                    "primitive_half_extents_m": (half_extents.tolist() if primitive_type == "box" else None),
                    "primitive_rotation_world_box": (rotation_box.tolist() if primitive_type == "box" else None),
                    "primitive_rotation_grasp_box": (primitive.get("rotation_grasp_box").tolist() if primitive_type == "box" else None),
                    "geom_id": (primitive.get("geom_id") if primitive_type == "box" else None),
                    "geom_name": (primitive.get("geom_name") if primitive_type == "box" else None),
                    "source": (primitive.get("source") if primitive_type == "box" else None),
                    "segment": "pregrasp_to_contact",
                    "segment_sample_index": sample_index,
                    "segment_sample_count": sample_count,
                    "segment_fraction": float(t),
                    "threshold_m": float(clearance),
                    "aperture_m": float(required_aperture_m),
                    "status": "collision",
                    "target_mask_terminal_allowance_m": float(terminal_allowance_m),
                    "target_mask_allowed": False,
                    "observed_point_count": int(len(worlds)),
                    "ignored_robot_point_count": int(np.count_nonzero(~keep)),
                }
                return details
    details = dict(nearest or {})
    details.update({
        "status": "ok",
        "observed_point_count": int(len(worlds)),
        "ignored_robot_point_count": int(np.count_nonzero(~keep)),
        "target_mask_terminal_allowance_m": float(terminal_allowance_m),
        "target_mask_allowed": True,
    })
    return details


def generate_grasp_candidates(
    *,
    rgb: np.ndarray | None,
    metric_depth_m: np.ndarray,
    sam_mask: np.ndarray,
    molmo_points: Sequence[Any] | None,
    calibration: CameraCalibrationLike,
    robot_calibration: RobotGraspCalibration,
    policy: CandidatePolicy | None = None,
) -> GraspCandidateResult:
    """Generate deterministic, filtered and ranked upright grasp candidates.

    ``rgb`` is accepted to make the source RGB-D contract explicit and is
    validated when supplied; geometry uses the mask/depth and never accesses
    simulator or evaluator state.  MolmoPoint pixels are original-image
    coordinates.  The first candidate is the highest deterministic score;
    callers should retain the full tuple and advance to another candidate
    after a failed fresh-frame attempt.
    """
    config = policy or CandidatePolicy()
    if config.max_seeds <= 0 or config.max_candidates <= 0 or config.max_candidates > 128:
        raise ValueError("max_seeds must be positive and max_candidates must be in (0, 128]")
    if len(config.yaw_offsets_deg) != 3 or len(config.insertion_depths_m) != 3:
        raise ValueError("the canary candidate grid requires exactly three yaw and three insertion values")
    if any(not np.isfinite(x) for x in (*config.yaw_offsets_deg, *config.insertion_depths_m)):
        raise ValueError("candidate yaw and insertion values must be finite")
    depth = np.asarray(metric_depth_m, dtype=np.float64)
    mask = np.asarray(sam_mask, dtype=bool)
    if depth.ndim != 2 or mask.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("metric_depth_m and sam_mask must be matching 2-D arrays")
    if np.asarray(metric_depth_m).dtype.kind not in "fiu":
        raise ValueError("metric_depth_m must be numeric")
    if rgb is not None:
        image = np.asarray(rgb)
        if image.ndim < 2 or image.shape[0] != depth.shape[0] or image.shape[1] != depth.shape[1]:
            raise ValueError("rgb and RGB-D depth must share original image height and width")
    K, T = _validate_camera(calibration, depth.shape)
    R_ge, contact_offset, approach, workspace_min, workspace_max = _validate_robot(robot_calibration)
    current_grip, current_rotation, hand_spheres, hand_boxes = _validate_live_robot_geometry(robot_calibration)
    live_motion_available = current_grip is not None and current_rotation is not None
    ignored_robot_spheres: tuple[tuple[np.ndarray, float], ...] = ()
    ignored_robot_boxes: tuple[dict[str, Any], ...] = ()
    # Live oriented boxes are preferred over spheres, never unioned with them.
    collision_boxes = hand_boxes
    collision_spheres = () if collision_boxes else hand_spheres
    observed_worlds: np.ndarray | None = None
    observed_pixels_vu: np.ndarray | None = None
    if config.obstruction_clearance_m is not None and (collision_spheres or collision_boxes):
        observed_pixels_vu = np.column_stack(np.nonzero(_valid_depth(depth)))
        observed_worlds, observed_pixels_vu = _support_points(observed_pixels_vu, depth, K, T)
    if live_motion_available and collision_spheres:
        ignored_robot_spheres = _current_hand_world_spheres(current_grip, current_rotation, R_ge, contact_offset, collision_spheres)
    if live_motion_available and collision_boxes:
        ignored_robot_boxes = _current_hand_world_boxes(current_grip, current_rotation, R_ge, contact_offset, collision_boxes)
    if not np.isfinite(config.terminal_contact_allowance_m) or config.terminal_contact_allowance_m < 0:
        raise ValueError("terminal_contact_allowance_m must be finite and non-negative")
    if config.contact_mode != "observed_upper_rim":
        raise ValueError("contact_mode must be observed_upper_rim")
    if not (0.0 < config.rim_height_quantile <= 1.0) or not np.isfinite(config.rim_height_quantile):
        raise ValueError("rim_height_quantile must be finite in (0, 1]")
    if not np.isfinite(config.rim_height_band_m) or config.rim_height_band_m < 0:
        raise ValueError("rim_height_band_m must be finite and non-negative")
    if not np.isfinite(config.rim_local_radius_m) or config.rim_local_radius_m <= 0:
        raise ValueError("rim_local_radius_m must be finite and positive")
    rim_local_radius_m = config.rim_local_radius_m
    valid = _valid_depth(depth)
    masked_pixels = np.column_stack(np.nonzero(mask & valid))
    masked_world, masked_kept = _support_points(masked_pixels, depth, K, T)
    if len(masked_world) == 0:
        return GraspCandidateResult((), (), (), config.name, {"reason": "no_valid_visible_rim_boundary", "seed_audit": []})
    # Exclude the currently observed robot before estimating rim height.  This
    # is geometry-only self filtering from the live probe, not mask dilation.
    masked_keep = np.ones(len(masked_world), dtype=bool)
    for center, radius in ignored_robot_spheres:
        masked_keep &= np.linalg.norm(masked_world - center[None, :], axis=1) > radius + float(config.obstruction_clearance_m or 0.0)
    for box in ignored_robot_boxes:
        masked_keep &= _points_box_signed_clearance(masked_world, box["center_world_m"], box["rotation_world_box"], box["half_extents_m"]) > float(config.obstruction_clearance_m or 0.0)
    masked_world = masked_world[masked_keep]
    masked_kept = masked_kept[masked_keep]
    if len(masked_world) == 0:
        return GraspCandidateResult((), (), (), config.name, {"reason": "no_visible_target_after_robot_exclusion", "seed_audit": []})
    rim_height = float(np.quantile(masked_world[:, 2], config.rim_height_quantile))
    upper_threshold = rim_height - config.rim_height_band_m
    boundary_mask = _boundary(mask) & valid
    upper_observed = np.zeros(mask.shape, dtype=bool)
    upper_observed[masked_kept[:, 0], masked_kept[:, 1]] = masked_world[:, 2] >= upper_threshold
    # The upper-support pool is intentionally not restricted to silhouette
    # boundary pixels: a visible upper rim can be interior to a mask.
    boundary_pixels = np.column_stack(np.nonzero(upper_observed))
    molmo = _as_points(molmo_points)
    seeds, seed_audit = _make_seeds(boundary_pixels, molmo, config)
    rejected: list[CandidateRejection] = []
    candidates: list[GraspCandidate] = []
    if not len(boundary_pixels):
        return GraspCandidateResult((), (), (), config.name, {"reason": "no_valid_visible_upper_rim_boundary", "seed_audit": seed_audit, "rim_height_m": rim_height, "upper_rim_threshold_m": upper_threshold})
    # Only accepted Molmo proposals influence ranking.  In dense mode an
    # out-of-mask proposal must not pull fallback geometry toward its invalid
    # pixel merely because it was present in the model response.
    accepted_molmo_uv = [
        (point.u, point.v)
        for point in molmo
        if _nearest_boundary(point, boundary_pixels, config.molmo_snap_radius_px) is not None
    ]
    molmo_uv = np.asarray(accepted_molmo_uv, dtype=float).reshape(-1, 2)
    for seed_index, (v, u) in enumerate(seeds):
        seed_depth = float(depth[v, u])
        try:
            base_contact = _deproject(float(u), float(v), seed_depth, K, T)
        except ValueError as exc:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "seed_deprojection", {"error": str(exc)}))
            continue
        distances_to_seed = np.linalg.norm(masked_world - base_contact[None, :], axis=1)
        local_upper = (distances_to_seed <= rim_local_radius_m) & (masked_world[:, 2] >= upper_threshold)
        local_world = masked_world[local_upper]
        local_kept = masked_kept[local_upper]
        rim_world = local_world
        rim_kept = local_kept
        if len(rim_world) < config.min_rim_support_pixels:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "insufficient_visible_rim_support", {"support": int(len(rim_world)), "upper_support": int(len(local_world)), "rim_height_m": rim_height, "upper_rim_threshold_m": upper_threshold}))
            continue
        tangent = _rim_tangent(rim_world, rim_kept)
        jaw = np.cross(approach, tangent)
        jaw_norm = np.linalg.norm(jaw)
        if jaw_norm <= 1e-8:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "degenerate_rim_tangent"))
            continue
        jaw /= jaw_norm
        if len(local_world) < config.min_depth_support_pixels:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "insufficient_depth_support", {"support": int(len(local_world))}))
            continue
        molmo_distance = None
        if len(molmo_uv) and config.name != "geometry_only":
            molmo_distance = float(np.min(np.linalg.norm(molmo_uv - np.array((float(u), float(v))), axis=1)))
        depth_support = int(len(local_world))
        molmo_score = 0.0 if molmo_distance is None else 1.0 / (1.0 + molmo_distance)
        for yaw in config.yaw_offsets_deg:
            yaw_float = float(yaw)
            jaw_yaw = _rotate_about(approach, jaw, yaw_float * pi / 180.0)
            jaw_yaw -= approach * np.dot(approach, jaw_yaw)
            jaw_yaw /= np.linalg.norm(jaw_yaw)
            completion = np.cross(approach, jaw_yaw)
            completion /= np.linalg.norm(completion)
            R_world_grasp = np.column_stack((approach, jaw_yaw, completion))
            if not np.allclose(R_world_grasp.T @ R_world_grasp, np.eye(3), atol=1e-5) or np.linalg.det(R_world_grasp) <= 0:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, yaw_float, float(insertion), "invalid_orientation"))
                continue
            extent = float(np.ptp((local_world - base_contact[None, :]) @ jaw_yaw)) if len(local_world) >= 2 else 0.0
            required_aperture = max(float(robot_calibration.min_aperture_m), extent + 2.0 * float(robot_calibration.finger_clearance_m))
            if required_aperture > float(robot_calibration.max_aperture_m) + 1e-9:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, yaw_float, float(insertion), "aperture_exceeded", {"required_aperture_m": required_aperture, "local_width_support_m": extent, "rim_height_m": rim_height, "upper_rim_threshold_m": upper_threshold}))
                continue
            clearance_score = max(0.0, 1.0 - required_aperture / float(robot_calibration.max_aperture_m))
            # The second branch is physically equivalent for a symmetric jaw:
            # its approach axis is unchanged while jaw/completion are flipped.
            # It is considered only with a live current pose, and is selected
            # per insertion after workspace/obstruction feasibility checks.
            orientation_frames = [(False, R_world_grasp)]
            if live_motion_available:
                orientation_frames.append((True, R_world_grasp @ np.diag((1.0, -1.0, -1.0))))
            for insertion in config.insertion_depths_m:
                insertion_float = float(insertion)
                if insertion_float < 0:
                    rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, "negative_insertion"))
                    continue
                feasible_frames: list[dict[str, Any]] = []
                rejection_details: list[dict[str, Any]] = []
                for jaw_flip, grasp_frame in orientation_frames:
                    grip_frame = grasp_frame @ R_ge
                    if not np.allclose(grip_frame.T @ grip_frame, np.eye(3), atol=1e-5) or np.linalg.det(grip_frame) <= 0:
                        rejection_details.append({"jaw_flip": jaw_flip, "reason": "invalid_orientation"})
                        continue
                    contact = base_contact + approach * insertion_float
                    # Recompute the asymmetric contact offset after choosing
                    # the jaw branch; using the unflipped offset is incorrect.
                    grip_site = contact + grasp_frame @ contact_offset
                    pregrasp = grip_site - approach * float(robot_calibration.pregrasp_distance_m)
                    points_to_check = (contact, grip_site, pregrasp)
                    if any(not _inside_workspace(point, workspace_min, workspace_max) for point in points_to_check):
                        rejection_details.append({"jaw_flip": jaw_flip, "reason": "workspace"})
                        continue
                    clearance = float("inf")
                    obstruction_details: dict[str, Any] = {"status": "disabled"}
                    if config.obstruction_clearance_m is not None:
                        if config.obstruction_clearance_m < 0 or not np.isfinite(config.obstruction_clearance_m):
                            raise ValueError("obstruction_clearance_m must be finite and non-negative")
                        if collision_spheres or collision_boxes:
                            obstruction_details = _hand_volume_obstruction(
                                depth=depth,
                                mask=mask,
                                K=K,
                                T=T,
                                contact=contact,
                                grip_site=grip_site,
                                pregrasp=pregrasp,
                                rotation_world_grasp=grasp_frame,
                                spheres=collision_spheres,
                                boxes=collision_boxes,
                                clearance=float(config.obstruction_clearance_m),
                                terminal_allowance_m=float(config.terminal_contact_allowance_m),
                                required_aperture_m=required_aperture,
                                ignored_robot_spheres=ignored_robot_spheres,
                                ignored_robot_boxes=ignored_robot_boxes,
                                observed_worlds=observed_worlds,
                                observed_pixels_vu=observed_pixels_vu,
                            )
                        else:
                            obstruction_details = _obstruction_clearance(depth, mask, K, T, pregrasp, grip_site, float(config.obstruction_clearance_m))
                        clearance = float(obstruction_details.get("clearance_m", float("inf")))
                        if obstruction_details.get("status") == "no_observed_scene" or obstruction_details.get("status") == "no_observed_scene_after_robot_exclusion":
                            rejection_details.append({"jaw_flip": jaw_flip, "reason": "no_observed_scene", **obstruction_details})
                            continue
                        if clearance < float(config.obstruction_clearance_m):
                            rejection_details.append({"jaw_flip": jaw_flip, "reason": "approach_obstruction", **obstruction_details})
                            continue
                    position_distance = None if not live_motion_available else float(np.linalg.norm(pregrasp - current_grip))
                    rotation_distance = None if not live_motion_available else _rotation_distance_rad(current_rotation, grip_frame)
                    position_motion_score = None if position_distance is None else 1.0 / (1.0 + position_distance)
                    rotation_motion_score = None if rotation_distance is None else 1.0 / (1.0 + rotation_distance)
                    motion_score = 1.0 if position_motion_score is None else 0.5 * (position_motion_score + rotation_motion_score)
                    feasible_frames.append({
                        "jaw_flip": jaw_flip,
                        "grasp_frame": grasp_frame,
                        "grip_frame": grip_frame,
                        "contact": contact,
                        "grip_site": grip_site,
                        "pregrasp": pregrasp,
                        "clearance": clearance,
                        "obstruction_details": obstruction_details,
                        "position_distance": position_distance,
                        "rotation_distance": rotation_distance,
                        "position_motion_score": position_motion_score,
                        "rotation_motion_score": rotation_motion_score,
                        "motion_score": motion_score,
                    })
                if not feasible_frames:
                    # Keep one rejection per candidate grid cell even when
                    # both symmetric jaw frames fail.  Preserve both attempts
                    # in the audit so the discarded branch is still visible.
                    priority = {"approach_obstruction": 0, "no_observed_scene": 1, "workspace": 2, "invalid_orientation": 3}
                    selected_details = min(rejection_details, key=lambda item: priority.get(str(item.get("reason")), 99)) if rejection_details else {"reason": "invalid_orientation"}
                    reason = str(selected_details.get("reason", "invalid_orientation"))
                    details = {key: value for key, value in selected_details.items() if key != "reason"}
                    details["symmetry_attempts"] = rejection_details
                    rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, reason, details))
                    continue
                if live_motion_available:
                    chosen = min(feasible_frames, key=lambda item: (float(item["rotation_distance"]), float(item["position_distance"]), bool(item["jaw_flip"])))
                else:
                    chosen = feasible_frames[0]
                jaw_flip = bool(chosen["jaw_flip"])
                grasp_frame = chosen["grasp_frame"]
                grip_frame = chosen["grip_frame"]
                contact = chosen["contact"]
                grip_site = chosen["grip_site"]
                pregrasp = chosen["pregrasp"]
                clearance = float(chosen["clearance"])
                insertion_score = 1.0 - insertion_float / max(max(config.insertion_depths_m), 1e-9)
                motion_score = float(chosen["motion_score"])
                score = 4.0 + 0.5 * molmo_score + 0.25 * clearance_score + 0.15 * min(depth_support / 64.0, 1.0) + 0.1 * insertion_score + (0.5 if live_motion_available else 0.05) * motion_score
                release = None if config.release_world_m is None else _finite_vector(config.release_world_m, 3, "release_world_m")
                candidate = GraspCandidate(
                    # Jaw symmetry is a frame-selection detail, not a new
                    # proposal.  Keep identity stable across recovery/live
                    # pose changes; the selected branch is audited below.
                    candidate_id=f"seed{seed_index:02d}_yaw{yaw_float:+05.1f}_ins{insertion_float * 1000.0:04.1f}mm",
                    seed_index=seed_index,
                    source_pixel_uv=(float(u), float(v)),
                    molmo_distance_px=molmo_distance,
                    yaw_deg=yaw_float,
                    insertion_depth_m=insertion_float,
                    contact_world_m=contact,
                    grip_site_world_m=grip_site,
                    pregrasp_world_m=pregrasp,
                    release_world_m=release,
                    rotation_world_grasp=grasp_frame,
                    rotation_world_grip_site=grip_frame,
                    quaternion_world_grip_site_xyzw=_quaternion_xyzw(grip_frame),
                    jaw_axis_world=grasp_frame[:, 1],
                    rim_tangent_world=tangent,
                    required_aperture_m=required_aperture,
                    depth_support_count=depth_support,
                    clearance_m=clearance,
                    score=score,
                    audit={
                        "frame": "world_grip_site", "source_frame": "original_image_uv", "no_legacy_source_offset": True,
                        "support_pixels": int(len(rim_world)), "upper_support_pixels": int(len(local_world)),
                        "rim_height_m": rim_height, "upper_rim_threshold_m": upper_threshold,
                        "seed_height_m": float(base_contact[2]), "local_width_support_m": extent,
                        "jaw_flip": jaw_flip,
                        "current_pose_available": live_motion_available,
                        "position_distance_m": chosen["position_distance"],
                        "rotation_distance_rad": chosen["rotation_distance"],
                        "current_to_pregrasp_distance_m": chosen["position_distance"],
                        "current_to_candidate_rotation_rad": chosen["rotation_distance"],
                        "position_motion_score": chosen["position_motion_score"],
                        "rotation_motion_score": chosen["rotation_motion_score"],
                        "motion_score": motion_score,
                        "obstruction": chosen["obstruction_details"],
                        "required_aperture_m": required_aperture,
                    },
                )
                duplicate = False
                for prior in candidates:
                    if np.linalg.norm(prior.contact_world_m - candidate.contact_world_m) <= config.dedupe_position_m and _rotation_distance_deg(prior.rotation_world_grasp, candidate.rotation_world_grasp) <= config.dedupe_rotation_deg:
                        duplicate = True
                        break
                if duplicate:
                    rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, "duplicate_candidate"))
                else:
                    candidates.append(candidate)
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.candidate_id))
    candidates = candidates[: config.max_candidates]
    return GraspCandidateResult(
        tuple(candidates), tuple(rejected), tuple((u, v) for v, u in seeds), config.name,
        {
            "seed_audit": seed_audit,
            "candidate_grid_size": len(seeds) * len(config.yaw_offsets_deg) * len(config.insertion_depths_m),
            "returned_count": len(candidates),
            "current_pose_available": live_motion_available,
            "hand_collision_sphere_count": len(collision_spheres),
            "hand_collision_box_count": len(collision_boxes),
            "terminal_contact_allowance_m": float(config.terminal_contact_allowance_m),
            "contact_mode": config.contact_mode,
            "rim_height_quantile": float(config.rim_height_quantile),
            "rim_height_m": rim_height,
            "upper_rim_threshold_m": upper_threshold,
            "rim_height_band_m": float(config.rim_height_band_m),
            "rim_local_radius_m": float(rim_local_radius_m),
        },
    )


def generate_from_observation(
    observation: RGBDObservation,
    *,
    sam_mask: np.ndarray,
    molmo_points: Sequence[Any] | None,
    robot_calibration: RobotGraspCalibration,
    policy: CandidatePolicy | None = None,
) -> GraspCandidateResult:
    """Convenience wrapper preserving the aligned RGB-D observation contract."""
    return generate_grasp_candidates(
        rgb=observation.rgb,
        metric_depth_m=observation.metric_depth_m,
        sam_mask=sam_mask,
        molmo_points=molmo_points,
        calibration=observation.calibration,
        robot_calibration=robot_calibration,
        policy=policy,
    )


__all__ = [
    "CameraCalibration",
    "CameraCalibrationLike",
    "CandidatePolicy",
    "CandidateRejection",
    "GraspCandidate",
    "GraspCandidateResult",
    "MolmoPoint",
    "RGBDObservation",
    "RobotGraspCalibration",
    "generate_from_observation",
    "generate_grasp_candidates",
]
