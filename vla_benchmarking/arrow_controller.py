"""Deterministic 2-D visual-arrow to bounded LIBERO control primitives.

This module deliberately has no LIBERO, MuJoCo, or object-state dependency.  A
runner may use a camera image with one rendered arrow, a metric depth image and
calibration to obtain a world-space target.  The coordinate contract is:

* pixels are ``(u, v) == (x, y)`` with origin at the upper-left;
* depth is metres along the camera optical ``+Z`` axis;
* ``K`` is an image-aligned pinhole camera matrix in the *same image
  resolution* as the input.  Signed non-zero focal entries are supported;
  in particular, ``fy < 0`` encodes LIBERO's vertical image flip.  When
  converting a conventional projection ``v = H - (fy*y/z + cy)`` to this
  matrix, use ``K[1, 1] = -fy`` and ``K[1, 2] = H - cy``;
* ``T_world_camera`` maps homogeneous camera points into world coordinates.

The implementation uses only NumPy.  Keeping perception and control here free
of simulator objects is intentional: any state used by a caller must arrive as
an explicit observation/calibration argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float]


def _as_finite_array(value: ArrayLike, *, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _pixel(value: ArrayLike, *, name: str = "pixel") -> np.ndarray:
    point = _as_finite_array(value, name=name).reshape(-1)
    if point.size != 2:
        raise ValueError(f"{name} must contain exactly two values (u, v)")
    return point


@dataclass(frozen=True)
class ArrowObservation:
    """A clean/overlay image pair and optional explicit geometry metadata."""

    clean_rgb: np.ndarray
    arrow_rgb: np.ndarray
    depth_m: np.ndarray | float | None = None
    K: np.ndarray | None = None
    T_world_camera: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_images(self.clean_rgb, self.arrow_rgb)
        if self.depth_m is not None:
            depth = np.asarray(self.depth_m)
            if depth.ndim not in (0, 2):
                raise ValueError("depth_m must be a scalar or an HxW depth image")
        if self.K is not None:
            _validate_intrinsics(self.K)
        if self.T_world_camera is not None:
            _validate_transform(self.T_world_camera)


@dataclass(frozen=True)
class ArrowCommand2D:
    """The source and pointy endpoint decoded from one visual arrow.

    Coordinates are floating point image pixels in ``(u, v)`` order.  The
    aliases ``tail_xy``/``tip_xy`` and ``source_px``/``target_px`` keep the
    command convenient for runners that use either vocabulary.
    """

    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    confidence: float = 1.0
    component_area: int = 1
    image_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        source = _pixel(self.source_xy, name="source_xy")
        target = _pixel(self.target_xy, name="target_xy")
        if float(np.linalg.norm(target - source)) <= 1.0:
            raise ValueError("arrow endpoints must be more than one pixel apart")
        if not np.isfinite(self.confidence) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if self.component_area < 1:
            raise ValueError("component_area must be positive")
        if self.image_shape is not None:
            if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
                raise ValueError("image_shape must be (height, width)")

    @property
    def tail_xy(self) -> tuple[float, float]:
        return self.source_xy

    @property
    def tip_xy(self) -> tuple[float, float]:
        return self.target_xy

    @property
    def source_px(self) -> tuple[float, float]:
        return self.source_xy

    @property
    def target_px(self) -> tuple[float, float]:
        return self.target_xy

    @property
    def source(self) -> tuple[float, float]:
        return self.source_xy

    @property
    def target(self) -> tuple[float, float]:
        return self.target_xy

    @property
    def start_xy(self) -> tuple[float, float]:
        return self.source_xy

    @property
    def end_xy(self) -> tuple[float, float]:
        return self.target_xy

    @property
    def endpoint_xy(self) -> tuple[float, float]:
        """The pointy/destination endpoint (the one used for deprojection)."""
        return self.target_xy

    @property
    def vector_xy(self) -> np.ndarray:
        return np.asarray(self.target_xy, dtype=np.float64) - np.asarray(self.source_xy, dtype=np.float64)


def _validate_images(clean_rgb: np.ndarray, arrow_rgb: np.ndarray) -> None:
    if not isinstance(clean_rgb, np.ndarray) or not isinstance(arrow_rgb, np.ndarray):
        raise TypeError("clean_rgb and arrow_rgb must be NumPy arrays")
    if clean_rgb.shape != arrow_rgb.shape:
        raise ValueError(f"clean_rgb and arrow_rgb must have identical shapes, got {clean_rgb.shape} and {arrow_rgb.shape}")
    if clean_rgb.ndim != 3 or clean_rgb.shape[2] != 3:
        raise ValueError(f"images must have shape HxWx3, got {clean_rgb.shape}")
    if clean_rgb.dtype != np.uint8 or arrow_rgb.dtype != np.uint8:
        raise TypeError("clean_rgb and arrow_rgb must have dtype uint8")


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected component coordinates without a CV dependency."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        components.append(np.asarray(points, dtype=np.int32))
    return components


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _arrow_component(clean_rgb: np.ndarray, arrow_rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    difference = np.max(np.abs(arrow_rgb.astype(np.int16) - clean_rgb.astype(np.int16)), axis=2)
    mask = difference >= threshold
    if not mask.any():
        raise ValueError("no arrow pixels detected in clean/overlay difference")

    # Anti-aliased one-pixel renderings can have tiny pinholes/gaps.  A single
    # close is enough to join those pixels while retaining separate arrows as
    # separate components.
    closed = _erode(_dilate(mask))
    components = _connected_components(closed)
    min_area = max(8, int(mask.size * 2e-6))
    candidates = [component for component in components if len(component) >= min_area]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one unambiguous arrow component, found {len(candidates)}")
    component = candidates[0]
    # Reconstruct the accepted component region.  ``closed`` contains every
    # component, so using it as the selector would accidentally re-introduce
    # isolated original changed pixels (including tiny specks rejected above)
    # into endpoint PCA.
    component_region = np.zeros_like(closed, dtype=bool)
    component_region[component[:, 0], component[:, 1]] = True
    # Use original changed pixels inside only the selected region, excluding
    # morphology-expanded border pixels from endpoint estimation.
    selected_points = np.column_stack(np.nonzero(mask & component_region))
    if selected_points.shape[0] < 8:
        raise ValueError("arrow component has too few changed pixels")
    return selected_points.astype(np.float64), int(selected_points.shape[0])


def _endpoint_and_score(points_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = points_yx[:, ::-1]  # (u, v)
    center = points.mean(axis=0)
    demeaned = points - center
    _, singular_values, vectors_t = np.linalg.svd(demeaned, full_matrices=False)
    if singular_values[0] <= 4.0 or singular_values[0] / max(singular_values[1], 1e-9) < 2.0:
        raise ValueError("arrow geometry is too short or direction is ambiguous")
    axis = vectors_t[0]
    normal = vectors_t[1]
    projection = demeaned @ axis
    lateral = demeaned @ normal
    low, high = float(projection.min()), float(projection.max())
    length = high - low
    if length < 6.0:
        raise ValueError("arrow is too short to determine a pointy endpoint")

    # An arrowhead has a narrow apex followed by a wider wing region.  Compare
    # occupancy near each end with occupancy in the stable shaft.  This works
    # for the sealed LIBERO renderer and for synthetic anti-aliased arrows,
    # without assuming a particular arrow colour or orientation.
    bins = np.linspace(0.0, 1.0, 21)
    normalized = (projection - low) / length
    occupancy = np.array([np.count_nonzero((normalized >= bins[i]) & (normalized < bins[i + 1])) for i in range(20)], dtype=float)
    occupancy[-1] += np.count_nonzero(normalized >= bins[-1])
    shaft = max(float(np.median(occupancy[7:14])), 1.0)

    def side_score(reverse: bool) -> float:
        values = occupancy[::-1] if reverse else occupancy
        # The shaft is a thin line.  Wings cause an excess in bins 2..5;
        # requiring both a narrow apex bin and a wing excess rejects a plain
        # segment, which has no directional information.
        apex = values[0] / shaft
        wing = float(np.sum(np.maximum(values[1:6] - shaft, 0.0))) / shaft
        return wing + 0.15 * max(0.0, 1.5 - apex)

    low_score, high_score = side_score(False), side_score(True)
    best, other = max(low_score, high_score), min(low_score, high_score)
    if best < 0.45 or best - other < 0.18:
        raise ValueError("arrowhead direction is ambiguous")

    tip_is_low = low_score > high_score
    tip_projection = low if tip_is_low else high
    tail_projection = high if tip_is_low else low
    # Median of the extremal pixels is stable against antialiasing and wing
    # pixels, while still returning the actual arrow-tip pixel location.
    tip_band = np.abs(projection - tip_projection) <= max(1.5, length * 0.025)
    tail_band = np.abs(projection - tail_projection) <= max(1.5, length * 0.025)
    tip = points[tip_band].mean(axis=0)
    tail = points[tail_band].mean(axis=0)
    confidence = float(np.clip(0.5 + 0.2 * (best - other) + 0.08 * best, 0.0, 0.99))
    return tail, tip, confidence, length


def decode_arrow(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    *,
    difference_threshold: float = 10.0,
) -> ArrowCommand2D:
    """Decode one clean/overlay pair into source and pointy target pixels.

    Multiple disconnected overlays, missing overlays, shape/dtype mismatch and
    a line without a clear arrowhead fail closed with ``ValueError``.
    """
    _validate_images(clean_rgb, arrow_rgb)
    if not np.isfinite(difference_threshold) or difference_threshold <= 0:
        raise ValueError("difference_threshold must be finite and positive")
    points, area = _arrow_component(clean_rgb, arrow_rgb, float(difference_threshold))
    tail, tip, confidence, _ = _endpoint_and_score(points)
    return ArrowCommand2D(
        source_xy=(float(tail[0]), float(tail[1])),
        target_xy=(float(tip[0]), float(tip[1])),
        confidence=confidence,
        component_area=area,
        image_shape=(clean_rgb.shape[0], clean_rgb.shape[1]),
    )


def estimate_endpoint_depth(
    depth_m: np.ndarray | float,
    pixel_xy: ArrayLike,
    *,
    radius: int = 2,
    min_valid: int = 3,
    min_depth_m: float = 1e-4,
    max_depth_m: float = 100.0,
) -> float:
    """Return a robust local metric depth at ``pixel_xy``.

    Invalid, zero, non-finite and out-of-range samples are ignored.  A median
    absolute deviation filter removes a foreground/background boundary sample
    before taking the final median.  A scalar depth is accepted for synthetic
    tests and calibrated depth-image callers should pass an ``HxW`` array.
    """
    point = _pixel(pixel_xy)
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if not isinstance(min_valid, (int, np.integer)) or min_valid < 1:
        raise ValueError("min_valid must be a positive integer")
    if not np.isfinite(min_depth_m) or not np.isfinite(max_depth_m) or not 0 < min_depth_m < max_depth_m:
        raise ValueError("depth bounds must be finite and satisfy 0 < min < max")
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 0:
        values = depth.reshape(1)
    elif depth.ndim == 2:
        u, v = np.rint(point).astype(int)
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            raise ValueError("pixel is outside the depth image")
        y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
        values = depth[y0:y1, x0:x1].reshape(-1)
    else:
        raise ValueError("depth_m must be a scalar or an HxW array")
    values = values[np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)]
    if values.size < min_valid:
        raise ValueError(f"insufficient valid local depth samples: {values.size} < {min_valid}")
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    if mad > 0:
        inliers = values[deviations <= 3.5 * 1.4826 * mad]
        if inliers.size >= min_valid:
            values = inliers
    return float(np.median(values))


def _validate_intrinsics(K: ArrayLike) -> np.ndarray:
    matrix = _as_finite_array(K, name="K", ndim=2)
    if matrix.shape != (3, 3) or matrix[0, 0] == 0 or matrix[1, 1] == 0:
        raise ValueError("K must be a finite 3x3 matrix with non-zero focal lengths")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("K must have homogeneous bottom row [0, 0, 1]")
    return matrix


def _validate_transform(T_world_camera: ArrayLike) -> np.ndarray:
    transform = _as_finite_array(T_world_camera, name="T_world_camera", ndim=2)
    if transform.shape != (4, 4) or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("T_world_camera must be a homogeneous 4x4 camera-to-world transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) <= 0:
        raise ValueError("T_world_camera rotation must be a proper orthonormal matrix")
    return transform


def deproject_endpoint(
    point_xy_or_command: ArrayLike | ArrowCommand2D,
    depth_m: np.ndarray | float,
    K: ArrayLike,
    T_world_camera: ArrayLike | None = None,
    *,
    depth_radius: int = 2,
    min_valid_depth: int = 3,
) -> np.ndarray:
    """Deproject an arrow target pixel to a world-space 3-D point."""
    point = point_xy_or_command.target_xy if isinstance(point_xy_or_command, ArrowCommand2D) else point_xy_or_command
    pixel = _pixel(point)
    if isinstance(point_xy_or_command, ArrowCommand2D) and point_xy_or_command.image_shape is not None:
        height, width = point_xy_or_command.image_shape
        if not (0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height):
            raise ValueError("arrow endpoint is outside its image shape")
    intrinsics = _validate_intrinsics(K)
    transform = np.eye(4, dtype=np.float64) if T_world_camera is None else _validate_transform(T_world_camera)
    if np.asarray(depth_m).ndim == 2:
        depth = estimate_endpoint_depth(depth_m, pixel, radius=depth_radius, min_valid=min_valid_depth)
    else:
        depth = estimate_endpoint_depth(depth_m, pixel, radius=0, min_valid=1)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_point = np.array(((pixel[0] - cx) * depth / fx, (pixel[1] - cy) * depth / fy, depth), dtype=np.float64)
    world_h = transform @ np.r_[camera_point, 1.0]
    if abs(world_h[3]) <= 1e-9:
        raise ValueError("deprojected homogeneous world point has zero scale")
    world = world_h[:3] / world_h[3]
    if not np.all(np.isfinite(world)):
        raise ValueError("deprojected world point is not finite")
    return world


@dataclass(frozen=True)
class BowlWaypointConfig:
    """Fixed, conservative six-position bowl transfer plan."""

    lift_height_m: float = 0.08
    approach_height_m: float = 0.03
    # LIBERO's Panda convention, verified against a live reset/step probe:
    # +1 closes the gripper and -1 opens it.
    gripper_open: float = -1.0
    gripper_closed: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lift_height_m) or self.lift_height_m <= 0:
            raise ValueError("lift_height_m must be finite and positive")
        if not np.isfinite(self.approach_height_m) or self.approach_height_m <= 0:
            raise ValueError("approach_height_m must be finite and positive")
        if self.approach_height_m > self.lift_height_m:
            raise ValueError("approach_height_m cannot exceed lift_height_m")
        for name in ("gripper_open", "gripper_closed"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")


def build_bowl_waypoints(
    source_world: ArrayLike,
    target_world: ArrayLike,
    current_rotation: ArrayLike | None = None,
    config: BowlWaypointConfig | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Build a fixed six-position bowl transfer plan.

    The rows are ``[pregrasp, grasp, lift, preplace, place, retreat]``.  The
    caller should use :class:`BowlWaypointConfig`'s gripper signs to close at
    ``grasp`` and open at ``place``; the returned positions stay pure 3-D so
    they can be consumed directly by an OSC action builder.

    The lift is relative to world ``+Z``.  ``current_rotation`` is accepted and
    validated as the orientation to hold throughout the transfer; orientation
    actions are produced separately by :func:`normalized_osc_action`.
    """
    # Accepting config as the third positional argument is convenient for
    # runners and remains unambiguous because rotations are array-like.
    if config is None and isinstance(current_rotation, (BowlWaypointConfig, Mapping)):
        config, current_rotation = current_rotation, None
    source = _as_finite_array(source_world, name="source_world").reshape(-1)
    target = _as_finite_array(target_world, name="target_world").reshape(-1)
    if source.size != 3 or target.size != 3:
        raise ValueError("source_world and target_world must each contain three values")
    if current_rotation is not None:
        _rotation_matrix(current_rotation)
    if config is None:
        plan_config = BowlWaypointConfig()
    elif isinstance(config, BowlWaypointConfig):
        plan_config = config
    elif isinstance(config, Mapping):
        allowed = {"lift_height_m", "approach_height_m", "gripper_open", "gripper_closed"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown BowlWaypointConfig keys: {sorted(unknown)}")
        plan_config = BowlWaypointConfig(**dict(config))
    else:
        raise TypeError("config must be BowlWaypointConfig, a mapping, or None")
    approach = np.array((0.0, 0.0, plan_config.approach_height_m), dtype=np.float64)
    # Use one shared transit height.  Independent source/target offsets can
    # descend while carrying the bowl when the source happens to be higher.
    safe_z = max(float(source[2]), float(target[2])) + plan_config.lift_height_m
    source_lift = np.array((source[0], source[1], safe_z), dtype=np.float64)
    target_preplace = np.array((target[0], target[1], safe_z), dtype=np.float64)
    return np.vstack(
        (
            source + approach,
            source,
            source_lift,
            target_preplace,
            target,
            target + approach,
        )
    )


def _rotation_matrix(rotation: ArrayLike) -> np.ndarray:
    arr = _as_finite_array(rotation, name="rotation")
    if arr.shape == (3, 3):
        if not np.allclose(arr.T @ arr, np.eye(3), atol=1e-4) or np.linalg.det(arr) <= 0:
            raise ValueError("rotation matrix must be proper orthonormal")
        return arr
    flat = arr.reshape(-1)
    if flat.size == 4:
        # Quaternion convention is (x, y, z, w), matching common robotics APIs.
        norm = np.linalg.norm(flat)
        if norm <= 1e-9:
            raise ValueError("quaternion must have non-zero norm")
        x, y, z, w = flat / norm
        return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))
    if flat.size == 3:
        # Axis-angle / rotation-vector representation.
        theta = float(np.linalg.norm(flat))
        if theta <= 1e-9:
            return np.eye(3)
        axis = flat / theta
        x, y, z = axis
        skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
        return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
    raise ValueError("rotation must be 3x3, a quaternion (x,y,z,w), or an axis-angle vector")


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < 1e-7:
        return np.array(((rotation[2, 1] - rotation[1, 2]) / 2, (rotation[0, 2] - rotation[2, 0]) / 2, (rotation[1, 0] - rotation[0, 1]) / 2), dtype=np.float64)
    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]), dtype=np.float64)


def _osc_scales(scales: ArrayLike | Mapping[str, float]) -> np.ndarray:
    if isinstance(scales, Mapping):
        if "position" in scales and "rotation" in scales:
            result = np.r_[np.repeat(float(scales["position"]), 3), np.repeat(float(scales["rotation"]), 3)]
        else:
            result = np.asarray([scales.get(k) for k in ("x", "y", "z", "rx", "ry", "rz")], dtype=np.float64)
    else:
        result = _as_finite_array(scales, name="scales").reshape(-1)
    if result.size == 1:
        result = np.repeat(result, 6)
    if result.size == 7:
        result = result[:6]
    if result.size != 6 or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("scales must provide six finite positive values (position xyz, rotation xyz)")
    return result


def normalized_osc_action(
    current_pos: ArrayLike,
    current_rot: ArrayLike,
    target_pos: ArrayLike,
    target_rot: ArrayLike,
    gripper: float,
    scales: ArrayLike | Mapping[str, float],
) -> np.ndarray:
    """Return a finite bounded 7-D OSC action ``[dp(3), dR(3), gripper]``."""
    current_position = _as_finite_array(current_pos, name="current_pos").reshape(-1)
    target_position = _as_finite_array(target_pos, name="target_pos").reshape(-1)
    if current_position.size != 3 or target_position.size != 3:
        raise ValueError("current_pos and target_pos must each contain three values")
    current_rotation = _rotation_matrix(current_rot)
    target_rotation = _rotation_matrix(target_rot)
    scale = _osc_scales(scales)
    try:
        gripper_value = float(gripper)
    except (TypeError, ValueError) as exc:
        raise ValueError("gripper must be a finite scalar in [-1, 1]") from exc
    if not np.isfinite(gripper_value) or not -1.0 <= gripper_value <= 1.0:
        raise ValueError("gripper must be finite and in [-1, 1]")
    position_error = target_position - current_position
    rotation_error = _rotation_vector(target_rotation @ current_rotation.T)
    action = np.r_[position_error, rotation_error] / scale
    action = np.clip(action, -1.0, 1.0)
    return np.r_[action, gripper_value].astype(np.float32)


__all__ = [
    "ArrowObservation",
    "ArrowCommand2D",
    "BowlWaypointConfig",
    "decode_arrow",
    "estimate_endpoint_depth",
    "deproject_endpoint",
    "build_bowl_waypoints",
    "normalized_osc_action",
]
