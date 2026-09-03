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
    min_rim_support_pixels: int = 3
    min_depth_support_pixels: int = 3
    dedupe_position_m: float = 0.002
    dedupe_rotation_deg: float = 4.0
    yaw_offsets_deg: tuple[float, ...] = (-15.0, 0.0, 15.0)
    insertion_depths_m: tuple[float, ...] = (0.0, 0.004, 0.008)
    obstruction_clearance_m: float | None = None
    release_world_m: ArrayLike | None = None


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
) -> float:
    """Return nearest observed non-object point to the approach segment."""
    valid = _valid_depth(depth) & ~mask
    pixels = np.column_stack(np.nonzero(valid))
    worlds, _ = _support_points(pixels, depth, K, T)
    if len(worlds) == 0:
        return float("inf")
    direction = end - start
    length_sq = float(np.dot(direction, direction))
    if length_sq <= 1e-12:
        return float(np.min(np.linalg.norm(worlds - start[None, :], axis=1)))
    factors = np.clip(((worlds - start[None, :]) @ direction) / length_sq, 0.0, 1.0)
    closest = start[None, :] + factors[:, None] * direction[None, :]
    return float(np.min(np.linalg.norm(worlds - closest, axis=1)))


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
    valid = _valid_depth(depth)
    eligible_boundary = _boundary(mask) & valid
    boundary_pixels = np.column_stack(np.nonzero(eligible_boundary))
    molmo = _as_points(molmo_points)
    seeds, seed_audit = _make_seeds(boundary_pixels, molmo, config)
    rejected: list[CandidateRejection] = []
    candidates: list[GraspCandidate] = []
    if not len(boundary_pixels):
        return GraspCandidateResult((), (), (), config.name, {"reason": "no_valid_visible_rim_boundary", "seed_audit": seed_audit})
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
        rim_pixels = _local_pixels(mask & valid, (v, u), config.rim_support_radius_px, boundary_only=True)
        rim_world, rim_kept = _support_points(rim_pixels, depth, K, T)
        if len(rim_world) < config.min_rim_support_pixels:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "insufficient_visible_rim_support", {"support": int(len(rim_world))}))
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
        local_mask_pixels = _local_pixels(mask & valid, (v, u), config.local_seed_radius_px)
        local_world, _ = _support_points(local_mask_pixels, depth, K, T)
        if len(local_world) < config.min_depth_support_pixels:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "insufficient_depth_support", {"support": int(len(local_world))}))
            continue
        extent = float(np.ptp((local_world - base_contact[None, :]) @ jaw)) if len(local_world) >= 2 else 0.0
        required_aperture = max(float(robot_calibration.min_aperture_m), extent + 2.0 * float(robot_calibration.finger_clearance_m))
        if required_aperture > float(robot_calibration.max_aperture_m) + 1e-9:
            for yaw in config.yaw_offsets_deg:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, float(yaw), float(insertion), "aperture_exceeded", {"required_aperture_m": required_aperture}))
            continue
        molmo_distance = None
        if len(molmo_uv) and config.name != "geometry_only":
            molmo_distance = float(np.min(np.linalg.norm(molmo_uv - np.array((float(u), float(v))), axis=1)))
        depth_support = int(len(local_world))
        clearance_score = max(0.0, 1.0 - required_aperture / float(robot_calibration.max_aperture_m))
        molmo_score = 0.0 if molmo_distance is None else 1.0 / (1.0 + molmo_distance)
        for yaw in config.yaw_offsets_deg:
            yaw_float = float(yaw)
            jaw_yaw = _rotate_about(approach, jaw, yaw_float * pi / 180.0)
            jaw_yaw -= approach * np.dot(approach, jaw_yaw)
            jaw_yaw /= np.linalg.norm(jaw_yaw)
            completion = np.cross(approach, jaw_yaw)
            completion /= np.linalg.norm(completion)
            R_world_grasp = np.column_stack((approach, jaw_yaw, completion))
            R_world_grip = R_world_grasp @ R_ge
            if not np.allclose(R_world_grip.T @ R_world_grip, np.eye(3), atol=1e-5) or np.linalg.det(R_world_grip) <= 0:
                for insertion in config.insertion_depths_m:
                    rejected.append(CandidateRejection(seed_index, yaw_float, float(insertion), "invalid_orientation"))
                continue
            grip_offset_world = R_world_grasp @ contact_offset
            for insertion in config.insertion_depths_m:
                insertion_float = float(insertion)
                if insertion_float < 0:
                    rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, "negative_insertion"))
                    continue
                contact = base_contact + approach * insertion_float
                grip_site = contact + grip_offset_world
                pregrasp = grip_site - approach * float(robot_calibration.pregrasp_distance_m)
                points_to_check = (contact, grip_site, pregrasp)
                if any(not _inside_workspace(point, workspace_min, workspace_max) for point in points_to_check):
                    rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, "workspace"))
                    continue
                clearance = float("inf")
                if config.obstruction_clearance_m is not None:
                    if config.obstruction_clearance_m < 0 or not np.isfinite(config.obstruction_clearance_m):
                        raise ValueError("obstruction_clearance_m must be finite and non-negative")
                    clearance = _obstruction_clearance(depth, mask, K, T, pregrasp, grip_site, float(config.obstruction_clearance_m))
                    if clearance < float(config.obstruction_clearance_m):
                        rejected.append(CandidateRejection(seed_index, yaw_float, insertion_float, "approach_obstruction", {"clearance_m": clearance}))
                        continue
                insertion_score = 1.0 - insertion_float / max(max(config.insertion_depths_m), 1e-9)
                motion_score = 1.0 / (1.0 + float(np.linalg.norm(pregrasp - grip_site)))
                score = 4.0 + 0.5 * molmo_score + 0.25 * clearance_score + 0.15 * min(depth_support / 64.0, 1.0) + 0.1 * insertion_score + 0.05 * motion_score
                release = None if config.release_world_m is None else _finite_vector(config.release_world_m, 3, "release_world_m")
                candidate = GraspCandidate(
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
                    rotation_world_grasp=R_world_grasp,
                    rotation_world_grip_site=R_world_grip,
                    quaternion_world_grip_site_xyzw=_quaternion_xyzw(R_world_grip),
                    jaw_axis_world=jaw_yaw,
                    rim_tangent_world=tangent,
                    required_aperture_m=required_aperture,
                    depth_support_count=depth_support,
                    clearance_m=clearance,
                    score=score,
                    audit={"frame": "world_grip_site", "source_frame": "original_image_uv", "no_legacy_source_offset": True, "support_pixels": int(len(rim_world))},
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
    return GraspCandidateResult(tuple(candidates), tuple(rejected), tuple((u, v) for v, u in seeds), config.name, {"seed_audit": seed_audit, "candidate_grid_size": len(seeds) * len(config.yaw_offsets_deg) * len(config.insertion_depths_m), "returned_count": len(candidates)})


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
