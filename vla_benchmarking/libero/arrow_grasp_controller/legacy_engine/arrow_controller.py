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

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float]


@dataclass(frozen=True)
class ArrowEncoding:
    """Versioned visual encoding metadata consumed by the decoder.

    ``legacy`` represents the historical arrow-only overlay.  The
    ``color_endpoint`` candidate additionally draws a small endpoint marker in
    ``endpoint_color_rgb``.  The marker is optional at decode time; when it is
    present it provides an unambiguous target pixel while the shaft still
    supplies the source direction.
    """

    version: str = "legacy"
    arrow_color_rgb: tuple[int, int, int] | None = None
    endpoint_color_rgb: tuple[int, int, int] | None = None
    color_tolerance: float = 18.0
    endpoint_radius_px: int = 4

    def __post_init__(self) -> None:
        if self.version not in {"legacy", "color_endpoint"}:
            raise ValueError("unsupported arrow encoding version")
        for name in ("arrow_color_rgb", "endpoint_color_rgb"):
            color = getattr(self, name)
            if color is not None and (len(color) != 3 or any(int(channel) != channel or not 0 <= int(channel) <= 255 for channel in color)):
                raise ValueError(f"{name} must be an RGB triple in [0, 255]")
        if self.version == "color_endpoint" and self.endpoint_color_rgb is None:
            raise ValueError("color_endpoint encoding requires endpoint_color_rgb")
        if not np.isfinite(self.color_tolerance) or self.color_tolerance < 0:
            raise ValueError("color_tolerance must be finite and non-negative")
        if not isinstance(self.endpoint_radius_px, (int, np.integer)) or self.endpoint_radius_px < 1:
            raise ValueError("endpoint_radius_px must be a positive integer")

    @property
    def name(self) -> str:
        """Human-readable stable name used by manifests and diagnostics."""
        return self.version

    @property
    def endpoint_rgb(self) -> tuple[int, int, int] | None:
        return self.endpoint_color_rgb


LEGACY_ARROW_ENCODING = ArrowEncoding()
COLOR_ENDPOINT_ARROW_ENCODING = ArrowEncoding(
    version="color_endpoint", endpoint_color_rgb=(255, 64, 64)
)


def resolve_arrow_encoding(encoding: str | ArrowEncoding | None) -> ArrowEncoding:
    """Resolve a stable encoding name without importing renderer code."""
    if encoding is None or encoding == "legacy":
        return LEGACY_ARROW_ENCODING
    if encoding in {"color_endpoint", "color-endpoint", "v2_color_endpoint", "v2-color-endpoint"}:
        return COLOR_ENDPOINT_ARROW_ENCODING
    if isinstance(encoding, ArrowEncoding):
        return encoding
    raise ValueError("encoding must be 'legacy', 'color_endpoint', or ArrowEncoding")


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
    encoding: ArrowEncoding | str | None = None

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
        if self.encoding is not None:
            resolve_arrow_encoding(self.encoding)


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


@dataclass(frozen=True)
class ArrowDecodeDiagnostics:
    """Deterministic decode outcome, including a stable failure reason."""

    ok: bool
    command: ArrowCommand2D | None
    reason: str | None
    encoding_version: str
    changed_pixel_count: int = 0
    component_count: int = 0

    def __post_init__(self) -> None:
        if self.ok != (self.command is not None):
            raise ValueError("ok must agree with whether command is present")
        if self.ok and self.reason is not None:
            raise ValueError("valid diagnostics cannot have a failure reason")
        if not self.ok and not self.reason:
            raise ValueError("failed diagnostics require a reason")
        if self.changed_pixel_count < 0 or self.component_count < 0:
            raise ValueError("diagnostic counts must be non-negative")

    @property
    def decoded(self) -> ArrowCommand2D | None:
        return self.command

    @property
    def failure_reason(self) -> str | None:
        return self.reason


@dataclass(frozen=True)
class RGBDEndpoint:
    """An endpoint refined from one arrow pixel and a matching depth sample."""

    pixel_xy: tuple[float, float]
    depth_m: float
    world_xyz: np.ndarray
    pixel_provenance: str
    depth_provenance: str
    method: str
    command: ArrowCommand2D | None = None

    def __post_init__(self) -> None:
        _pixel(self.pixel_xy, name="pixel_xy")
        if not np.isfinite(self.depth_m) or self.depth_m <= 0:
            raise ValueError("depth_m must be finite and positive")
        world = _as_finite_array(self.world_xyz, name="world_xyz").reshape(-1)
        if world.size != 3:
            raise ValueError("world_xyz must contain exactly three values")
        if not self.pixel_provenance or not self.depth_provenance or not self.method:
            raise ValueError("RGB-D provenance and method must be non-empty")


@dataclass(frozen=True)
class EndpointChangeEvidence:
    """Pure numeric evidence for a before/after endpoint change.

    No environment, simulator, object, pose, or evaluator is accepted here;
    callers provide only endpoint and optional proprioceptive arrays.
    """

    before_endpoint: np.ndarray
    after_endpoint: np.ndarray
    endpoint_delta: np.ndarray
    endpoint_distance: float
    before_proprioception: np.ndarray | None
    after_proprioception: np.ndarray | None
    proprioception_delta: np.ndarray | None
    proprioception_distance: float | None

    def __post_init__(self) -> None:
        for name in ("before_endpoint", "after_endpoint", "endpoint_delta"):
            value = _as_finite_array(getattr(self, name), name=name).reshape(-1)
            if value.size not in (2, 3):
                raise ValueError(f"{name} must contain two or three values")
        if not np.isfinite(self.endpoint_distance) or self.endpoint_distance < 0:
            raise ValueError("endpoint_distance must be finite and non-negative")
        if (self.before_proprioception is None) != (self.after_proprioception is None):
            raise ValueError("before and after proprioception must be supplied together")
        if self.proprioception_delta is None:
            if self.proprioception_distance is not None:
                raise ValueError("proprioception distance requires a delta")
        elif self.proprioception_distance is None or not np.isfinite(self.proprioception_distance) or self.proprioception_distance < 0:
            raise ValueError("proprioception distance must be finite and non-negative")

    @property
    def delta(self) -> np.ndarray:
        return self.endpoint_delta

    @property
    def distance(self) -> float:
        return self.endpoint_distance


@dataclass(frozen=True)
class GraspRetentionEvidence:
    """Post-lift gripper-joint evidence, expressed only in proprioception.

    ``qpos_trace`` is an ``N x D`` trace of gripper joint positions.  The
    values are intentionally not interpreted as a simulator state: callers
    provide the already exposed proprioceptive signal and its units are kept
    explicit in the result.  A sample is considered closed when the largest
    absolute joint value is at least ``closed_threshold``.
    """

    qpos_trace: np.ndarray
    closed_threshold: float
    min_samples: int
    min_closed_fraction: float
    closed_fraction: float
    final_abs_qpos: np.ndarray
    retained: bool
    qpos_units: str = "proprioceptive gripper joint units"
    frame: str = "post_lift_proprioception_samples"

    def __post_init__(self) -> None:
        trace = _as_finite_array(self.qpos_trace, name="qpos_trace", ndim=2)
        if trace.shape[0] < 1 or trace.shape[1] < 1:
            raise ValueError("qpos_trace must have at least one sample and one joint")
        if type(self.min_samples) is not int or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        threshold = float(self.closed_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("closed_threshold must be finite and positive")
        fraction = float(self.min_closed_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("min_closed_fraction must be finite and in [0, 1]")
        observed_fraction = float(self.closed_fraction)
        if not np.isfinite(observed_fraction) or not 0.0 <= observed_fraction <= 1.0:
            raise ValueError("closed_fraction must be finite and in [0, 1]")
        final = _as_finite_array(self.final_abs_qpos, name="final_abs_qpos").reshape(-1)
        if final.size != trace.shape[1]:
            raise ValueError("final_abs_qpos must match qpos_trace joint count")
        if not isinstance(self.retained, (bool, np.bool_)):
            raise TypeError("retained must be a boolean")
        if not self.qpos_units or not self.frame:
            raise ValueError("qpos_units and frame must be non-empty")


def assess_grasp_retention(
    qpos_trace: ArrayLike,
    *,
    closed_threshold: float = 0.0015,
    min_samples: int = 1,
    min_closed_fraction: float = 1.0,
) -> GraspRetentionEvidence:
    """Assess post-lift retention from a finite gripper qpos trace.

    This is a pure threshold policy.  It accepts no handles or hidden state;
    an empty, malformed, or non-finite trace fails closed with ``ValueError``.
    A one-dimensional input is treated as one sample for convenience.
    """
    raw = _as_finite_array(qpos_trace, name="qpos_trace")
    trace = raw.reshape(1, -1) if raw.ndim == 1 else raw
    if trace.ndim != 2 or trace.shape[0] < 1 or trace.shape[1] < 1:
        raise ValueError("qpos_trace must be a non-empty 1-D or 2-D numeric array")
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    threshold = float(closed_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("closed_threshold must be finite and positive")
    required_fraction = float(min_closed_fraction)
    if not np.isfinite(required_fraction) or not 0.0 <= required_fraction <= 1.0:
        raise ValueError("min_closed_fraction must be finite and in [0, 1]")
    closed = np.max(np.abs(trace), axis=1) >= threshold
    observed_fraction = float(np.mean(closed))
    retained = bool(trace.shape[0] >= min_samples and observed_fraction >= required_fraction)
    return GraspRetentionEvidence(
        qpos_trace=np.array(trace, dtype=np.float64, copy=True),
        closed_threshold=threshold,
        min_samples=min_samples,
        min_closed_fraction=required_fraction,
        closed_fraction=observed_fraction,
        final_abs_qpos=np.abs(trace[-1]).astype(np.float64, copy=True),
        retained=retained,
    )


# Explicit aliases make the evidence primitive discoverable without coupling
# callers to one particular report vocabulary.
evaluate_post_lift_grasp_retention = assess_grasp_retention
post_lift_grasp_retention = assess_grasp_retention
PostLiftGraspRetentionEvidence = GraspRetentionEvidence


@dataclass(frozen=True)
class DepthSupportEvidence:
    """Local metric-depth support around one image pixel.

    Pixel coordinates are ``(u, v)`` in the top-left image frame.  Depth is in
    metres along camera ``+Z`` and the gradient is ``(dZ/du, dZ/dv)`` in
    metres per pixel.  Clipping is reported instead of silently extrapolated.
    """

    pixel_xy: tuple[float, float]
    image_shape: tuple[int, int]
    patch_radius_px: int
    sample_count: int
    valid_sample_count: int
    valid_fraction: float
    clipped: bool
    center_depth_m: float | None
    gradient_m_per_pixel: tuple[float, float] | None
    depth_units: str = "metres"
    pixel_frame: str = "image_top_left_uv"

    def __post_init__(self) -> None:
        _pixel(self.pixel_xy, name="pixel_xy")
        if len(self.image_shape) != 2 or any(type(v) is not int or v < 1 for v in self.image_shape):
            raise ValueError("image_shape must contain two positive integers")
        if type(self.patch_radius_px) is not int or self.patch_radius_px < 0:
            raise ValueError("patch_radius_px must be a non-negative integer")
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if type(self.valid_sample_count) is not int or not 0 <= self.valid_sample_count <= self.sample_count:
            raise ValueError("valid_sample_count must be between zero and sample_count")
        fraction = float(self.valid_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("valid_fraction must be finite and in [0, 1]")
        if not isinstance(self.clipped, (bool, np.bool_)):
            raise TypeError("clipped must be a boolean")
        if self.center_depth_m is not None and (not np.isfinite(self.center_depth_m) or self.center_depth_m <= 0.0):
            raise ValueError("center_depth_m must be finite and positive when present")
        if self.gradient_m_per_pixel is not None:
            gradient = _as_finite_array(self.gradient_m_per_pixel, name="gradient_m_per_pixel").reshape(-1)
            if gradient.size != 2:
                raise ValueError("gradient_m_per_pixel must have two values")
        if not self.depth_units or not self.pixel_frame:
            raise ValueError("depth_units and pixel_frame must be non-empty")


def analyze_depth_support(
    depth_m: np.ndarray,
    pixel_xy: ArrayLike,
    *,
    patch_radius_px: int = 2,
) -> DepthSupportEvidence:
    """Measure valid local depth support and a finite-difference gradient."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[0] < 1 or depth.shape[1] < 1:
        raise ValueError("depth_m must be a non-empty HxW array")
    if type(patch_radius_px) is not int or patch_radius_px < 0:
        raise ValueError("patch_radius_px must be a non-negative integer")
    pixel = _pixel(pixel_xy)
    u, v = np.rint(pixel).astype(int)
    h, w = depth.shape
    # Use the original floating pixel for clipping; rounding an out-of-frame
    # coordinate such as ``-0.1`` to zero must not turn it into valid support.
    clipped = not (0.0 <= pixel[0] < w and 0.0 <= pixel[1] < h)
    cu, cv = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    x0, x1 = max(0, cu - patch_radius_px), min(w, cu + patch_radius_px + 1)
    y0, y1 = max(0, cv - patch_radius_px), min(h, cv + patch_radius_px + 1)
    local = depth[y0:y1, x0:x1]
    valid = np.isfinite(local) & (local > 0.0)
    valid_count = int(valid.sum())
    sample_count = int(local.size)
    center = float(depth[cv, cu]) if np.isfinite(depth[cv, cu]) and depth[cv, cu] > 0.0 else None
    gradient: tuple[float, float] | None = None
    if center is not None:
        neighbours: dict[str, float] = {}
        for name, nx, ny in (("left", cu - 1, cv), ("right", cu + 1, cv), ("up", cu, cv - 1), ("down", cu, cv + 1)):
            if 0 <= nx < w and 0 <= ny < h and np.isfinite(depth[ny, nx]) and depth[ny, nx] > 0.0:
                neighbours[name] = float(depth[ny, nx])
        if "left" in neighbours and "right" in neighbours:
            du = (neighbours["right"] - neighbours["left"]) / 2.0
        elif "right" in neighbours:
            du = neighbours["right"] - center
        elif "left" in neighbours:
            du = center - neighbours["left"]
        else:
            du = None
        if "up" in neighbours and "down" in neighbours:
            dv = (neighbours["down"] - neighbours["up"]) / 2.0
        elif "down" in neighbours:
            dv = neighbours["down"] - center
        elif "up" in neighbours:
            dv = center - neighbours["up"]
        else:
            dv = None
        if du is not None and dv is not None:
            gradient = (float(du), float(dv))
    return DepthSupportEvidence(
        pixel_xy=(float(pixel[0]), float(pixel[1])),
        image_shape=(h, w),
        patch_radius_px=patch_radius_px,
        sample_count=sample_count,
        valid_sample_count=valid_count,
        valid_fraction=float(valid_count / sample_count),
        clipped=clipped,
        center_depth_m=center,
        gradient_m_per_pixel=gradient,
    )


depth_support_evidence = analyze_depth_support


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


def _arrow_component(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    threshold: float,
    *,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    difference = np.max(np.abs(arrow_rgb.astype(np.int16) - clean_rgb.astype(np.int16)), axis=2)
    mask = difference >= threshold
    if exclude_mask is not None:
        if exclude_mask.shape != mask.shape:
            raise ValueError("exclude_mask must match the image height and width")
        mask &= ~exclude_mask
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


def _endpoint_and_score(
    points_yx: np.ndarray,
    *,
    tip_override_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    if tip_override_xy is not None:
        tip = _pixel(tip_override_xy, name="tip_override_xy")
    confidence = float(np.clip(0.5 + 0.2 * (best - other) + 0.08 * best, 0.0, 0.99))
    return tail, tip, confidence, length


def _endpoint_marker_mask(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    encoding: ArrowEncoding,
    threshold: float,
) -> np.ndarray:
    if encoding.endpoint_color_rgb is None:
        raise ValueError("color_endpoint encoding has no endpoint color")
    marker = np.asarray(encoding.endpoint_color_rgb, dtype=np.float64)
    color_error = np.linalg.norm(arrow_rgb.astype(np.float64) - marker, axis=2)
    difference = np.max(np.abs(arrow_rgb.astype(np.int16) - clean_rgb.astype(np.int16)), axis=2)
    return (color_error <= encoding.color_tolerance) & (difference >= threshold)


def _endpoint_marker_xy(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    encoding: ArrowEncoding,
    threshold: float,
) -> np.ndarray:
    marker_mask = _endpoint_marker_mask(clean_rgb, arrow_rgb, encoding, threshold)
    components = _connected_components(marker_mask)
    if not components:
        raise ValueError("color_endpoint marker was not detected")
    component = max(components, key=len)
    if len(component) < 2:
        raise ValueError("color_endpoint marker is too small")
    # A marker is intentionally a local coordinate cue; use its centroid and
    # leave the shaft/PCA responsible for recovering the source endpoint.
    return component[:, ::-1].mean(axis=0)


def _diagnostic_counts(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    threshold: float,
    *,
    exclude_mask: np.ndarray | None = None,
) -> tuple[int, int]:
    difference = np.max(np.abs(arrow_rgb.astype(np.int16) - clean_rgb.astype(np.int16)), axis=2)
    mask = difference >= threshold
    if exclude_mask is not None:
        mask &= ~exclude_mask
    if not mask.any():
        return 0, 0
    closed = _erode(_dilate(mask))
    min_area = max(8, int(mask.size * 2e-6))
    count = sum(len(component) >= min_area for component in _connected_components(closed))
    return int(mask.sum()), int(count)


def decode_arrow_diagnostics(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    *,
    difference_threshold: float = 10.0,
    encoding: str | ArrowEncoding | None = None,
) -> ArrowDecodeDiagnostics:
    """Decode an arrow while returning a deterministic valid/invalid record.

    Unlike :func:`decode_arrow`, this diagnostic surface does not raise for
    malformed or ambiguous inputs.  ``decode_arrow`` remains the compatibility
    API and raises the recorded ``reason``.
    """
    try:
        resolved = resolve_arrow_encoding(encoding)
        _validate_images(clean_rgb, arrow_rgb)
        threshold = float(difference_threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("difference_threshold must be finite and positive")
        marker = None
        marker_mask = None
        if resolved.version == "color_endpoint":
            marker_mask = _endpoint_marker_mask(clean_rgb, arrow_rgb, resolved, threshold)
            marker = _endpoint_marker_xy(clean_rgb, arrow_rgb, resolved, threshold)
        changed_count, component_count = _diagnostic_counts(
            clean_rgb, arrow_rgb, threshold, exclude_mask=marker_mask
        )
        points, area = _arrow_component(
            clean_rgb, arrow_rgb, threshold, exclude_mask=marker_mask
        )
        tail, tip, confidence, _ = _endpoint_and_score(points, tip_override_xy=marker)
        command = ArrowCommand2D(
            source_xy=(float(tail[0]), float(tail[1])),
            target_xy=(float(tip[0]), float(tip[1])),
            confidence=confidence,
            component_area=area,
            image_shape=(clean_rgb.shape[0], clean_rgb.shape[1]),
        )
        return ArrowDecodeDiagnostics(True, command, None, resolved.version, changed_count, component_count)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(clean_rgb, np.ndarray) and isinstance(arrow_rgb, np.ndarray) and clean_rgb.ndim == 3 and arrow_rgb.ndim == 3 and clean_rgb.shape == arrow_rgb.shape and clean_rgb.shape[2] == 3 and clean_rgb.dtype == np.uint8 and arrow_rgb.dtype == np.uint8:
            try:
                changed_count, component_count = _diagnostic_counts(clean_rgb, arrow_rgb, float(difference_threshold))
            except (TypeError, ValueError, OverflowError):
                changed_count, component_count = 0, 0
        else:
            changed_count, component_count = 0, 0
        # Resolve invalid encoding names to a stable textual diagnostic without
        # masking the original reason.
        version = encoding.version if isinstance(encoding, ArrowEncoding) else str(encoding or "legacy")
        return ArrowDecodeDiagnostics(False, None, str(exc), version, changed_count, component_count)


def decode_arrow(
    clean_rgb: np.ndarray,
    arrow_rgb: np.ndarray,
    *,
    difference_threshold: float = 10.0,
    encoding: str | ArrowEncoding | None = None,
) -> ArrowCommand2D:
    """Decode one clean/overlay pair into source and pointy target pixels.

    Multiple disconnected overlays, missing overlays, shape/dtype mismatch and
    a line without a clear arrowhead fail closed with ``ValueError``.
    """
    diagnostics = decode_arrow_diagnostics(
        clean_rgb,
        arrow_rgb,
        difference_threshold=difference_threshold,
        encoding=encoding,
    )
    if diagnostics.command is None:
        raise ValueError(diagnostics.reason or "arrow decode failed")
    return diagnostics.command


def estimate_endpoint_depth(
    depth_m: np.ndarray | float,
    pixel_xy: ArrayLike,
    *,
    radius: int = 2,
    min_valid: int = 3,
    min_depth_m: float = 1e-4,
    max_depth_m: float = 100.0,
    method: str = "mad_median",
) -> float:
    """Return a robust local metric depth at ``pixel_xy``.

    Invalid, zero, non-finite and out-of-range samples are ignored.  A median
    absolute deviation filter removes a foreground/background boundary sample
    before taking the final median.  A scalar depth is accepted for synthetic
    tests and calibrated depth-image callers should pass an ``HxW`` array.
    """
    point = _pixel(pixel_xy)
    method = {"mad": "mad_median", "trimmed": "trimmed_median", "nearest": "nearest_valid"}.get(method, method)
    if method not in {"median", "mad_median", "trimmed_median", "nearest_valid"}:
        raise ValueError("method must be median, mad_median, trimmed_median, or nearest_valid")
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
    if method == "nearest_valid":
        # Values are row-major from the local window.  The central sample is
        # preferred when valid; otherwise choose the nearest valid pixel from
        # the window using a deterministic distance tie-break.
        if depth.ndim == 0:
            return float(values[0])
        u, v = np.rint(point).astype(int)
        y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local = depth[y0:y1, x0:x1]
        valid = np.isfinite(local) & (local >= min_depth_m) & (local <= max_depth_m)
        distances = np.where(valid, (xx - u) ** 2 + (yy - v) ** 2, np.inf)
        nearest = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if not np.isfinite(distances[nearest]):
            raise ValueError("no valid nearest depth sample")
        return float(local[nearest])

    median = float(np.median(values))
    if method == "median":
        return median
    if method == "trimmed_median" and values.size >= 5:
        low, high = np.quantile(values, (0.1, 0.9))
        trimmed = values[(values >= low) & (values <= high)]
        if trimmed.size >= min_valid:
            return float(np.median(trimmed))
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
    depth_method: str = "mad_median",
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
        depth = estimate_endpoint_depth(
            depth_m,
            pixel,
            radius=depth_radius,
            min_valid=min_valid_depth,
            method=depth_method,
        )
    else:
        depth = estimate_endpoint_depth(depth_m, pixel, radius=0, min_valid=1, method=depth_method)
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


def refine_rgbd_endpoint(
    endpoint_or_command: ArrayLike | ArrowCommand2D,
    depth_m: np.ndarray | float,
    K: ArrayLike,
    T_world_camera: ArrayLike | None = None,
    *,
    clean_rgb: np.ndarray | None = None,
    arrow_rgb: np.ndarray | None = None,
    encoding: str | ArrowEncoding | None = None,
    depth_method: str = "mad_median",
    depth_radius: int = 2,
    min_valid_depth: int = 3,
) -> RGBDEndpoint:
    """Refine one arrow endpoint using RGB provenance and local metric depth.

    Either an existing :class:`ArrowCommand2D`/pixel is supplied, or both
    ``clean_rgb`` and ``arrow_rgb`` are supplied and decoded in this function.
    If an image and a depth map are both present, their resolution must match;
    this prevents silently pairing a pixel from one image frame with depth from
    another.  The returned provenance strings retain the exact pixel and depth
    method used for the world point.
    """
    if (clean_rgb is None) != (arrow_rgb is None):
        raise ValueError("clean_rgb and arrow_rgb must be supplied together")
    resolved = resolve_arrow_encoding(encoding)
    if clean_rgb is not None and arrow_rgb is not None:
        command = decode_arrow(clean_rgb, arrow_rgb, encoding=resolved)
        pixel_source = f"arrow_decode:{resolved.version}"
    elif isinstance(endpoint_or_command, ArrowCommand2D):
        command = endpoint_or_command
        pixel_source = "arrow_command:target_xy"
    else:
        pixel = _pixel(endpoint_or_command, name="endpoint_xy")
        command = None
        pixel_source = "explicit_endpoint_xy"

    if command is not None:
        pixel = _pixel(command.target_xy, name="target_xy")
        if command.image_shape is not None and np.asarray(depth_m).ndim == 2:
            if tuple(np.asarray(depth_m).shape) != tuple(command.image_shape):
                raise ValueError(
                    "pixel/depth resolution mismatch: "
                    f"pixel frame {command.image_shape} versus depth {np.asarray(depth_m).shape}"
                )
    if clean_rgb is not None and np.asarray(depth_m).ndim == 2:
        if tuple(clean_rgb.shape[:2]) != tuple(np.asarray(depth_m).shape):
            raise ValueError(
                "pixel/depth resolution mismatch: "
                f"RGB {clean_rgb.shape[:2]} versus depth {np.asarray(depth_m).shape}"
            )
    depth_value = estimate_endpoint_depth(
        depth_m,
        pixel,
        radius=depth_radius,
        min_valid=min_valid_depth if np.asarray(depth_m).ndim == 2 else 1,
        method=depth_method,
    )
    world = deproject_endpoint(
        pixel,
        depth_value,
        K,
        T_world_camera,
        depth_method=depth_method,
    )
    pixel_text = f"pixel_xy=({pixel[0]:.3f},{pixel[1]:.3f})"
    depth_shape = "scalar" if np.asarray(depth_m).ndim == 0 else str(tuple(np.asarray(depth_m).shape))
    depth_source = f"depth_m:{depth_method}:shape={depth_shape}:{pixel_text}"
    return RGBDEndpoint(
        pixel_xy=(float(pixel[0]), float(pixel[1])),
        depth_m=depth_value,
        world_xyz=world,
        pixel_provenance=f"{pixel_source}:{pixel_text}",
        depth_provenance=depth_source,
        method=f"rgbd/{resolved.version}/{depth_method}",
        command=command,
    )


# Natural spelling retained as a small compatibility alias for runner code.
refine_endpoint_rgbd = refine_rgbd_endpoint
refine_endpoint_from_rgbd = refine_rgbd_endpoint
refine_rgbd = refine_rgbd_endpoint


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


def arrow_world_xy_basis(
    source_world: ArrayLike, target_world: ArrayLike, *, min_norm_m: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Return the forward/lateral unit basis induced by a visual arrow.

    Only the horizontal world displacement is used.  The function is pure
    geometry: callers provide deprojected arrow endpoints and receive
    ``(forward, lateral)`` world vectors, both with zero Z component.  A
    degenerate projected arrow fails closed instead of inventing an axis.
    """
    source = _as_finite_array(source_world, name="source_world").reshape(-1)
    target = _as_finite_array(target_world, name="target_world").reshape(-1)
    if source.size != 3 or target.size != 3:
        raise ValueError("source_world and target_world must each contain three values")
    threshold = float(min_norm_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("min_norm_m must be finite and positive")
    displacement = target - source
    displacement[2] = 0.0
    norm = float(np.linalg.norm(displacement))
    if norm < threshold:
        raise ValueError("arrow world-XY displacement is degenerate")
    forward = displacement / norm
    lateral = np.array((-forward[1], forward[0], 0.0), dtype=np.float64)
    return forward, lateral


@dataclass(frozen=True)
class RGBDApproachCandidate:
    """One bounded candidate in the arrow-derived approach frame.

    ``offset_arrow_frame_m`` is ``(forward, lateral, vertical)`` in metres.
    ``candidate_world_m`` and ``offset_world_m`` are in the world frame and
    are also metres.  No candidate is constructed outside the supplied
    workspace bounds.
    """

    candidate_world_m: np.ndarray
    offset_arrow_frame_m: np.ndarray
    offset_world_m: np.ndarray
    workspace_bounds_m: tuple[np.ndarray, np.ndarray]
    frame: str = "world"
    units: str = "metres"

    def __post_init__(self) -> None:
        candidate = _as_finite_array(self.candidate_world_m, name="candidate_world_m").reshape(-1)
        arrow_offset = _as_finite_array(self.offset_arrow_frame_m, name="offset_arrow_frame_m").reshape(-1)
        world_offset = _as_finite_array(self.offset_world_m, name="offset_world_m").reshape(-1)
        if candidate.size != 3 or arrow_offset.size != 3 or world_offset.size != 3:
            raise ValueError("candidate and offsets must each contain three values")
        if not isinstance(self.workspace_bounds_m, tuple) or len(self.workspace_bounds_m) != 2:
            raise ValueError("workspace_bounds_m must be a (minimum, maximum) tuple")
        minimum = _as_finite_array(self.workspace_bounds_m[0], name="workspace_minimum").reshape(-1)
        maximum = _as_finite_array(self.workspace_bounds_m[1], name="workspace_maximum").reshape(-1)
        if minimum.size != 3 or maximum.size != 3 or np.any(minimum > maximum):
            raise ValueError("workspace bounds must be ordered three-vectors")
        if np.any(candidate < minimum) or np.any(candidate > maximum):
            raise ValueError("candidate_world_m lies outside workspace bounds")
        if self.frame != "world" or self.units != "metres":
            raise ValueError("candidate frame must be world and units must be metres")


@dataclass(frozen=True)
class RGBDBoundedApproachGeometry:
    """Arrow-frame candidate set plus the RGB-D evidence used to anchor it."""

    anchor_world_m: np.ndarray
    direction_world_m: np.ndarray
    forward_world_unit: np.ndarray
    lateral_world_unit: np.ndarray
    candidates: tuple[RGBDApproachCandidate, ...]
    anchor_support: DepthSupportEvidence
    direction_support: DepthSupportEvidence
    workspace_bounds_m: tuple[np.ndarray, np.ndarray]
    frame: str = "world"
    units: str = "metres"

    def __post_init__(self) -> None:
        for name in ("anchor_world_m", "direction_world_m", "forward_world_unit", "lateral_world_unit"):
            value = _as_finite_array(getattr(self, name), name=name).reshape(-1)
            if value.size != 3:
                raise ValueError(f"{name} must contain three values")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        if not isinstance(self.anchor_support, DepthSupportEvidence) or not isinstance(self.direction_support, DepthSupportEvidence):
            raise TypeError("anchor_support and direction_support must be DepthSupportEvidence")
        if self.frame != "world" or self.units != "metres":
            raise ValueError("geometry frame must be world and units must be metres")

    @property
    def world_points_m(self) -> np.ndarray:
        """Return candidate points as a defensive ``N x 3`` copy."""
        return np.vstack([candidate.candidate_world_m for candidate in self.candidates]).copy()


def _workspace_bounds(
    bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize either named-axis or minimum/maximum workspace bounds."""
    if isinstance(bounds, Mapping):
        if "minimum" in bounds or "maximum" in bounds:
            if set(bounds) != {"minimum", "maximum"}:
                raise ValueError("workspace bounds mapping may only contain minimum and maximum")
            minimum = _as_finite_array(bounds["minimum"], name="workspace minimum").reshape(-1)
            maximum = _as_finite_array(bounds["maximum"], name="workspace maximum").reshape(-1)
        else:
            if set(bounds) != {"x", "y", "z"}:
                raise ValueError("workspace bounds mapping must contain x, y, and z")
            axis_bounds = [_as_finite_array(bounds[axis], name=f"workspace {axis}").reshape(-1) for axis in ("x", "y", "z")]
            if any(axis.size != 2 for axis in axis_bounds):
                raise ValueError("each workspace axis must contain [minimum, maximum]")
            minimum = np.array([axis[0] for axis in axis_bounds], dtype=np.float64)
            maximum = np.array([axis[1] for axis in axis_bounds], dtype=np.float64)
    else:
        matrix = _as_finite_array(bounds, name="workspace_bounds_m", ndim=2)
        if matrix.shape != (2, 3):
            raise ValueError("workspace_bounds_m must have shape (2, 3): minimum then maximum")
        minimum, maximum = matrix[0].copy(), matrix[1].copy()
    if minimum.size != 3 or maximum.size != 3 or np.any(minimum > maximum):
        raise ValueError("workspace bounds must contain ordered three-vectors")
    return minimum, maximum


def derive_rgbd_bounded_approach_candidates(
    depth_m: np.ndarray,
    anchor_pixel_xy: ArrayLike,
    direction_pixel_xy: ArrayLike,
    K: ArrayLike,
    T_world_camera: ArrayLike,
    offset_arrow_frame_m: Sequence[Sequence[float]],
    workspace_bounds_m: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    patch_radius_px: int = 2,
    min_valid_fraction: float = 0.5,
) -> RGBDBoundedApproachGeometry:
    """Build bounded RGB-D approach/support candidates around an arrow anchor.

    The two pixels are deprojected from the supplied metric depth image.  The
    horizontal world displacement defines a right-handed arrow frame:
    ``forward`` follows the arrow and ``lateral = (-forward_y, forward_x)``.
    Each configured ``(forward, lateral, vertical)`` metre offset is mapped to
    world coordinates and checked against the explicit workspace.  Invalid or
    clipped depth support fails closed before any candidate is returned.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[0] < 1 or depth.shape[1] < 1:
        raise ValueError("depth_m must be a non-empty HxW array")
    intrinsics = _validate_intrinsics(K)
    transform = _validate_transform(T_world_camera)
    anchor_pixel = _pixel(anchor_pixel_xy, name="anchor_pixel_xy")
    direction_pixel = _pixel(direction_pixel_xy, name="direction_pixel_xy")
    anchor_support = analyze_depth_support(depth, anchor_pixel, patch_radius_px=patch_radius_px)
    direction_support = analyze_depth_support(depth, direction_pixel, patch_radius_px=patch_radius_px)
    fraction = float(min_valid_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("min_valid_fraction must be finite and in (0, 1]")
    for label, support in (("anchor", anchor_support), ("direction", direction_support)):
        if support.clipped:
            raise ValueError(f"{label} pixel is clipped by the RGB-D image")
        if support.center_depth_m is None or support.valid_fraction < fraction:
            raise ValueError(f"{label} pixel lacks sufficient valid metric-depth support")
    anchor_world = deproject_endpoint(anchor_pixel, anchor_support.center_depth_m, intrinsics, transform)
    direction_world = deproject_endpoint(direction_pixel, direction_support.center_depth_m, intrinsics, transform)
    forward, lateral = arrow_world_xy_basis(anchor_world, direction_world)
    raw_offsets = np.asarray(offset_arrow_frame_m, dtype=np.float64)
    if raw_offsets.ndim != 2 or raw_offsets.shape[0] < 1 or raw_offsets.shape[1] != 3:
        raise ValueError("offset_arrow_frame_m must be a non-empty N x 3 sequence")
    if not np.all(np.isfinite(raw_offsets)):
        raise ValueError("offset_arrow_frame_m must contain only finite values")
    minimum, maximum = _workspace_bounds(workspace_bounds_m)
    bounds = (minimum.copy(), maximum.copy())
    candidates: list[RGBDApproachCandidate] = []
    world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    for offset in raw_offsets:
        world_offset = float(offset[0]) * forward + float(offset[1]) * lateral + float(offset[2]) * world_up
        candidate_world = anchor_world + world_offset
        if np.any(candidate_world < minimum) or np.any(candidate_world > maximum):
            raise ValueError("configured approach candidate lies outside workspace bounds")
        candidates.append(
            RGBDApproachCandidate(
                candidate_world_m=candidate_world,
                offset_arrow_frame_m=offset.copy(),
                offset_world_m=world_offset,
                workspace_bounds_m=bounds,
            )
        )
    return RGBDBoundedApproachGeometry(
        anchor_world_m=anchor_world,
        direction_world_m=direction_world,
        forward_world_unit=forward,
        lateral_world_unit=lateral,
        candidates=tuple(candidates),
        anchor_support=anchor_support,
        direction_support=direction_support,
        workspace_bounds_m=bounds,
    )


# Compatibility spellings for evidence/reporting callers.
build_rgbd_approach_candidates = derive_rgbd_bounded_approach_candidates
derive_rgbd_approach_candidates = derive_rgbd_bounded_approach_candidates


@dataclass(frozen=True)
class SourceApproachCandidate:
    """A bounded source-side approach point in explicit world metres."""

    approach_world_m: np.ndarray
    offset_arrow_frame_m: np.ndarray
    offset_world_m: np.ndarray
    source_world_m: np.ndarray
    forward_world_unit: np.ndarray
    lateral_world_unit: np.ndarray
    workspace_bounds_m: tuple[np.ndarray, np.ndarray]
    frame: str = "world"
    units: str = "metres"

    def __post_init__(self) -> None:
        for name in (
            "approach_world_m",
            "offset_arrow_frame_m",
            "offset_world_m",
            "source_world_m",
            "forward_world_unit",
            "lateral_world_unit",
        ):
            value = _as_finite_array(getattr(self, name), name=name).reshape(-1)
            if value.size != 3:
                raise ValueError(f"{name} must contain three values")
        if not isinstance(self.workspace_bounds_m, tuple) or len(self.workspace_bounds_m) != 2:
            raise ValueError("workspace_bounds_m must be a (minimum, maximum) tuple")
        minimum = _as_finite_array(self.workspace_bounds_m[0], name="workspace_minimum").reshape(-1)
        maximum = _as_finite_array(self.workspace_bounds_m[1], name="workspace_maximum").reshape(-1)
        if minimum.size != 3 or maximum.size != 3 or np.any(minimum > maximum):
            raise ValueError("workspace bounds must be ordered three-vectors")
        approach = _as_finite_array(self.approach_world_m, name="approach_world_m").reshape(-1)
        if np.any(approach < minimum) or np.any(approach > maximum):
            raise ValueError("approach_world_m lies outside workspace bounds")
        if self.frame != "world" or self.units != "metres":
            raise ValueError("candidate frame must be world and units must be metres")

    @property
    def world_xyz_m(self) -> np.ndarray:
        return self.approach_world_m


def derive_rgbd_source_approach_candidates(
    source_pixel_xy: ArrayLike,
    direction_pixel_xy: ArrayLike,
    depth_m: np.ndarray,
    K: ArrayLike,
    T_world_camera: ArrayLike,
    offset_arrow_frame_m: Sequence[Sequence[float]],
    workspace_bounds_m: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    patch_radius_px: int = 2,
    min_valid_fraction: float = 0.5,
) -> tuple[SourceApproachCandidate, ...]:
    """Derive bounded source-side candidates from two RGB-D arrow pixels."""
    geometry = derive_rgbd_bounded_approach_candidates(
        depth_m,
        source_pixel_xy,
        direction_pixel_xy,
        K,
        T_world_camera,
        offset_arrow_frame_m,
        workspace_bounds_m,
        patch_radius_px=patch_radius_px,
        min_valid_fraction=min_valid_fraction,
    )
    return tuple(
        SourceApproachCandidate(
            approach_world_m=candidate.candidate_world_m.copy(),
            offset_arrow_frame_m=candidate.offset_arrow_frame_m.copy(),
            offset_world_m=candidate.offset_world_m.copy(),
            source_world_m=geometry.anchor_world_m.copy(),
            forward_world_unit=geometry.forward_world_unit.copy(),
            lateral_world_unit=geometry.lateral_world_unit.copy(),
            workspace_bounds_m=(geometry.workspace_bounds_m[0].copy(), geometry.workspace_bounds_m[1].copy()),
        )
        for candidate in geometry.candidates
    )


@dataclass(frozen=True)
class SupportPlaneEstimate:
    """A local destination support plane fit from metric RGB-D points."""

    origin_world_m: np.ndarray
    normal_world_unit: np.ndarray
    residual_rms_m: float
    residual_max_m: float
    valid_point_count: int
    pixel_xy: tuple[float, float]
    support: DepthSupportEvidence
    frame: str = "world"
    units: str = "metres"

    def __post_init__(self) -> None:
        origin = _as_finite_array(self.origin_world_m, name="origin_world_m").reshape(-1)
        normal = _as_finite_array(self.normal_world_unit, name="normal_world_unit").reshape(-1)
        if origin.size != 3 or normal.size != 3:
            raise ValueError("origin_world_m and normal_world_unit must contain three values")
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-4:
            raise ValueError("normal_world_unit must be unit length")
        for name in ("residual_rms_m", "residual_max_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.valid_point_count) is not int or self.valid_point_count < 3:
            raise ValueError("valid_point_count must be an integer >= 3")
        if not isinstance(self.support, DepthSupportEvidence):
            raise TypeError("support must be DepthSupportEvidence")
        if self.frame != "world" or self.units != "metres":
            raise ValueError("support plane frame must be world and units must be metres")

    @property
    def point_world_m(self) -> np.ndarray:
        return self.origin_world_m


def estimate_destination_support_plane(
    depth_m: np.ndarray,
    destination_pixel_xy: ArrayLike,
    K: ArrayLike,
    T_world_camera: ArrayLike,
    *,
    patch_radius_px: int = 2,
    min_valid_fraction: float = 0.5,
    max_residual_m: float = 0.01,
) -> SupportPlaneEstimate:
    """Fit a finite local support plane from deprojected metric depth."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.shape[0] < 1 or depth.shape[1] < 1:
        raise ValueError("depth_m must be a non-empty HxW array")
    intrinsics = _validate_intrinsics(K)
    transform = _validate_transform(T_world_camera)
    pixel = _pixel(destination_pixel_xy, name="destination_pixel_xy")
    support = analyze_depth_support(depth, pixel, patch_radius_px=patch_radius_px)
    fraction = float(min_valid_fraction)
    residual_limit = float(max_residual_m)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("min_valid_fraction must be finite and in (0, 1]")
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("max_residual_m must be finite and positive")
    if support.clipped or support.valid_fraction < fraction:
        raise ValueError("destination pixel lacks sufficient unclipped metric-depth support")
    u, v = np.rint(pixel).astype(int)
    radius = patch_radius_px
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    local = depth[y0:y1, x0:x1]
    valid_y, valid_x = np.nonzero(np.isfinite(local) & (local > 0.0))
    values = local[valid_y, valid_x]
    if values.size < 3:
        raise ValueError("support plane needs at least three valid metric-depth points")
    pixel_x = valid_x + x0
    pixel_y = valid_y + y0
    camera_points = np.column_stack(
        (
            (pixel_x - intrinsics[0, 2]) * values / intrinsics[0, 0],
            (pixel_y - intrinsics[1, 2]) * values / intrinsics[1, 1],
            values,
            np.ones(values.size, dtype=np.float64),
        )
    )
    points = (transform @ camera_points.T).T[:, :3]
    if not np.all(np.isfinite(points)):
        raise ValueError("support plane deprojection produced non-finite points")
    origin = np.mean(points, axis=0)
    centered = points - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    residuals = np.abs(centered @ normal)
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    maximum = float(np.max(residuals))
    if maximum > residual_limit:
        raise ValueError(f"support plane residual exceeds max_residual_m: {maximum:.6g}")
    return SupportPlaneEstimate(
        origin_world_m=origin,
        normal_world_unit=normal,
        residual_rms_m=rms,
        residual_max_m=maximum,
        valid_point_count=int(points.shape[0]),
        pixel_xy=(float(pixel[0]), float(pixel[1])),
        support=support,
    )


def release_point_on_support_plane(
    support_plane: SupportPlaneEstimate,
    world_xy_m: ArrayLike,
    *,
    workspace_bounds_m: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Return the point at requested world XY lying on a fitted support plane."""
    if not isinstance(support_plane, SupportPlaneEstimate):
        raise TypeError("support_plane must be SupportPlaneEstimate")
    xy = _as_finite_array(world_xy_m, name="world_xy_m").reshape(-1)
    if xy.size != 2:
        raise ValueError("world_xy_m must contain exactly two values")
    origin = support_plane.origin_world_m
    normal = support_plane.normal_world_unit
    if abs(float(normal[2])) <= 1e-9:
        raise ValueError("support plane is vertical and has no finite Z at requested XY")
    z = float(origin[2] - (normal[0] * (xy[0] - origin[0]) + normal[1] * (xy[1] - origin[1])) / normal[2])
    point = np.array((xy[0], xy[1], z), dtype=np.float64)
    if workspace_bounds_m is not None:
        minimum, maximum = _workspace_bounds(workspace_bounds_m)
        if np.any(point < minimum) or np.any(point > maximum):
            raise ValueError("release point lies outside workspace bounds")
    return point


def derive_rgbd_region_grasp_candidates(
    clean_rgb: np.ndarray,
    depth_m: np.ndarray,
    K: ArrayLike,
    T_world_camera: ArrayLike,
    source_pixel_xy: ArrayLike,
    profile_target_world: ArrayLike,
    *,
    region_radius_m: float = 0.075,
    depth_tolerance_m: float = 0.025,
    min_region_pixels: int = 48,
    max_region_fraction: float = 0.15,
    height_quantile: float = 0.70,
    profile_quantiles: Sequence[float] = (0.80, 0.60, 0.40),
    candidate_height_quantiles: Sequence[float] | None = None,
    seed_radius_px: int = 3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Derive bounded EEF-position candidates from an arrow-seeded RGB-D region.

    The arrow supplies only ``source_pixel_xy``.  A metric-depth connected
    component is grown around that pixel inside a physically sized image
    neighbourhood.  Candidate XY locations are real observed region samples,
    ordered from the side favoured by the existing camera-profile target
    toward the region centre; candidate Z is an observed surface quantile.

    RGB is deliberately not a hard segmentation gate: the Akita bowl is
    textured and can be partially clipped by the camera.  It is validated and
    retained in the audit hash, while metric depth and camera calibration
    determine all executable geometry.  No bbox, object pose, simulator state,
    task identifier, or evaluator result is accepted by this function.
    """
    _validate_images(clean_rgb, clean_rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have 2 dimensions, got shape {depth.shape}")
    if depth.shape != clean_rgb.shape[:2]:
        raise ValueError("clean_rgb and depth_m must have identical image dimensions")
    intrinsics = _validate_intrinsics(K)
    transform = _validate_transform(T_world_camera)
    source = _pixel(source_pixel_xy, name="source_pixel_xy")
    profile = _as_finite_array(profile_target_world, name="profile_target_world").reshape(-1)
    if profile.size != 3:
        raise ValueError("profile_target_world must contain exactly three values")
    h, w = depth.shape
    u, v = np.rint(source).astype(int)
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("source_pixel_xy is outside the RGB-D image")

    radius_m = float(region_radius_m)
    tolerance_m = float(depth_tolerance_m)
    fraction_limit = float(max_region_fraction)
    quantile_z = float(height_quantile)
    quantiles = tuple(float(value) for value in profile_quantiles)
    height_quantiles = (
        tuple(quantile_z for _ in quantiles)
        if candidate_height_quantiles is None
        else tuple(float(value) for value in candidate_height_quantiles)
    )
    if not np.isfinite(radius_m) or not 0.01 <= radius_m <= 0.15:
        raise ValueError("region_radius_m must be finite and in [0.01, 0.15]")
    if not np.isfinite(tolerance_m) or not 0.001 <= tolerance_m <= 0.10:
        raise ValueError("depth_tolerance_m must be finite and in [0.001, 0.10]")
    if type(min_region_pixels) is not int or min_region_pixels < 4:
        raise ValueError("min_region_pixels must be an integer >= 4")
    if not np.isfinite(fraction_limit) or not 0.0 < fraction_limit <= 0.5:
        raise ValueError("max_region_fraction must be in (0, 0.5]")
    if not np.isfinite(quantile_z) or not 0.0 < quantile_z < 1.0:
        raise ValueError("height_quantile must be in (0, 1)")
    if not quantiles or len(quantiles) > 8 or any(
        not np.isfinite(value) or not 0.0 < value < 1.0 for value in quantiles
    ):
        raise ValueError("profile_quantiles must contain one to eight values in (0, 1)")
    if len(height_quantiles) != len(quantiles) or any(
        not np.isfinite(value) or not 0.0 < value < 1.0 for value in height_quantiles
    ):
        raise ValueError(
            "candidate_height_quantiles must match profile_quantiles and contain values in (0, 1)"
        )
    if type(seed_radius_px) is not int or not 1 <= seed_radius_px <= 12:
        raise ValueError("seed_radius_px must be an integer in [1, 12]")

    y0, y1 = max(0, v - seed_radius_px), min(h, v + seed_radius_px + 1)
    x0, x1 = max(0, u - seed_radius_px), min(w, u + seed_radius_px + 1)
    local = depth[y0:y1, x0:x1]
    local_valid = np.isfinite(local) & (local > 0.0)
    if not local_valid.any():
        raise ValueError("source seed has no valid metric-depth samples")
    seed_depth_m = float(np.median(local[local_valid]))
    focal_px = max(abs(float(intrinsics[0, 0])), abs(float(intrinsics[1, 1])))
    radius_px = int(np.ceil(focal_px * radius_m / seed_depth_m))
    radius_px = max(seed_radius_px + 1, min(radius_px, max(h, w)))

    yy, xx = np.mgrid[:h, :w]
    radial = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    eligible = radial & valid_depth & (np.abs(depth - seed_depth_m) <= tolerance_m)
    seed_locations: list[tuple[int, int, int]] = []
    for seed_y in range(y0, y1):
        for seed_x in range(x0, x1):
            if eligible[seed_y, seed_x]:
                distance_sq = (seed_x - u) ** 2 + (seed_y - v) ** 2
                seed_locations.append((distance_sq, seed_y, seed_x))
    if not seed_locations:
        raise ValueError("source seed has no sample inside the configured depth band")
    _, seed_y, seed_x = min(seed_locations)

    region = np.zeros((h, w), dtype=bool)
    stack = [(seed_x, seed_y)]
    while stack:
        x, y = stack.pop()
        if region[y, x] or not eligible[y, x]:
            continue
        region[y, x] = True
        for dx, dy in (
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),             (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not region[ny, nx]:
                stack.append((nx, ny))

    area = int(region.sum())
    fraction = float(area / (h * w))
    if area < min_region_pixels:
        raise ValueError(
            f"arrow-seeded RGB-D region is too small: {area} < {min_region_pixels} pixels"
        )
    if fraction > fraction_limit:
        raise ValueError(
            "arrow-seeded RGB-D region exceeds max_region_fraction: "
            f"{fraction:.6f} > {fraction_limit:.6f}"
        )

    region_y, region_x = np.nonzero(region)
    region_depth = depth[region_y, region_x]
    camera_points = np.column_stack(
        (
            (region_x - intrinsics[0, 2]) * region_depth / intrinsics[0, 0],
            (region_y - intrinsics[1, 2]) * region_depth / intrinsics[1, 1],
            region_depth,
            np.ones(area, dtype=np.float64),
        )
    )
    world_points = (transform @ camera_points.T).T[:, :3]
    if not np.all(np.isfinite(world_points)):
        raise ValueError("arrow-seeded region deprojection produced non-finite world points")

    center_xy = np.median(world_points[:, :2], axis=0)
    profile_direction = profile[:2] - center_xy
    direction_norm = float(np.linalg.norm(profile_direction))
    if direction_norm <= 1e-6:
        centered = world_points[:, :2] - center_xy
        covariance = centered.T @ centered / max(1, area)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        profile_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if profile_direction[0] < 0.0 or (
            abs(profile_direction[0]) <= 1e-12 and profile_direction[1] < 0.0
        ):
            profile_direction = -profile_direction
    else:
        profile_direction = profile_direction / direction_norm
    transverse = np.array((-profile_direction[1], profile_direction[0]), dtype=np.float64)
    centered_xy = world_points[:, :2] - center_xy
    along = centered_xy @ profile_direction
    across = centered_xy @ transverse
    desired_across = float(np.median(across))
    targets: list[np.ndarray] = []
    selected_pixels: list[list[int]] = []
    selected_quantiles: list[float] = []
    selected_height_quantiles: list[float] = []
    selected_heights_m: list[float] = []
    # Select actual observed XY samples instead of inventing points inside a
    # rectangular/PCA envelope that may lie outside a clipped bowl region.
    scale_along = max(float(np.ptp(along)), 1e-6)
    scale_across = max(float(np.ptp(across)), 1e-6)
    for requested_quantile, requested_height_quantile in zip(quantiles, height_quantiles):
        desired_along = float(np.quantile(along, requested_quantile))
        distance = ((along - desired_along) / scale_along) ** 2
        distance += ((across - desired_across) / scale_across) ** 2
        order = np.argsort(distance, kind="stable")
        selected_index = None
        for index in order:
            candidate_xy = world_points[int(index), :2]
            if all(np.linalg.norm(candidate_xy - prior[:2]) >= 0.003 for prior in targets):
                selected_index = int(index)
                break
        if selected_index is None:
            continue
        target_z = float(np.quantile(world_points[:, 2], requested_height_quantile))
        candidate = np.array(
            (world_points[selected_index, 0], world_points[selected_index, 1], target_z),
            dtype=np.float64,
        )
        targets.append(candidate)
        selected_pixels.append([int(region_x[selected_index]), int(region_y[selected_index])])
        selected_quantiles.append(requested_quantile)
        selected_height_quantiles.append(requested_height_quantile)
        selected_heights_m.append(target_z)
    if not targets:
        raise ValueError("arrow-seeded RGB-D region produced no distinct grasp candidates")

    mask_hash = hashlib.sha256(np.ascontiguousarray(region.astype(np.uint8)).tobytes()).hexdigest()
    diagnostics: dict[str, object] = {
        "method": "arrow_seeded_metric_depth_component_v1",
        "source_pixel_xy": [float(source[0]), float(source[1])],
        "seed_pixel_xy": [int(seed_x), int(seed_y)],
        "seed_depth_m": seed_depth_m,
        "region_radius_m": radius_m,
        "region_radius_px": radius_px,
        "depth_tolerance_m": tolerance_m,
        "region_area_px": area,
        "region_fraction": fraction,
        "region_mask_sha256": mask_hash,
        "touches_image_border": bool(
            np.any(region[0]) or np.any(region[-1]) or np.any(region[:, 0]) or np.any(region[:, -1])
        ),
        "world_bounds_m": {
            "minimum": np.min(world_points, axis=0).tolist(),
            "maximum": np.max(world_points, axis=0).tolist(),
        },
        "world_center_xy_m": center_xy.tolist(),
        "profile_direction_xy": profile_direction.tolist(),
        "height_quantile": quantile_z,
        "target_height_m": (
            selected_heights_m[0]
            if selected_heights_m and len(set(selected_height_quantiles)) == 1 else None
        ),
        "requested_profile_quantiles": list(quantiles),
        "requested_candidate_height_quantiles": list(height_quantiles),
        "selected_profile_quantiles": selected_quantiles,
        "selected_height_quantiles": selected_height_quantiles,
        "selected_heights_m": selected_heights_m,
        "selected_pixels_xy": selected_pixels,
        "targets_world_m": [target.tolist() for target in targets],
        "rgb_sha256": hashlib.sha256(np.ascontiguousarray(clean_rgb).tobytes()).hexdigest(),
        "depth_sha256": hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest(),
    }
    return np.vstack(targets), diagnostics


def derive_rgbd_region_mask(
    clean_rgb: np.ndarray,
    depth_m: np.ndarray,
    K: ArrayLike,
    T_world_camera: ArrayLike,
    source_pixel_xy: ArrayLike,
    *,
    region_radius_m: float = 0.075,
    depth_tolerance_m: float = 0.025,
    min_region_pixels: int = 48,
    max_region_fraction: float = 0.15,
    seed_radius_px: int = 3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the canonical arrow-seeded metric-depth component.

    This is the mask-only companion to :func:`derive_rgbd_region_grasp_candidates`.
    It intentionally duplicates the established seed/growth arithmetic so the
    RGB-D perception path can reuse the established numeric region contract without
    changing the legacy candidate generator's return type or behavior.
    """
    _validate_images(clean_rgb, clean_rgb)
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must have 2 dimensions, got shape {depth.shape}")
    if depth.shape != clean_rgb.shape[:2]:
        raise ValueError("clean_rgb and depth_m must have identical image dimensions")
    intrinsics = _validate_intrinsics(K)
    _validate_transform(T_world_camera)
    source = _pixel(source_pixel_xy, name="source_pixel_xy")
    h, w = depth.shape
    u, v = np.rint(source).astype(int)
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("source_pixel_xy is outside the RGB-D image")
    radius_m = float(region_radius_m)
    tolerance_m = float(depth_tolerance_m)
    fraction_limit = float(max_region_fraction)
    if not np.isfinite(radius_m) or not 0.01 <= radius_m <= 0.15:
        raise ValueError("region_radius_m must be finite and in [0.01, 0.15]")
    if not np.isfinite(tolerance_m) or not 0.001 <= tolerance_m <= 0.10:
        raise ValueError("depth_tolerance_m must be finite and in [0.001, 0.10]")
    if type(min_region_pixels) is not int or min_region_pixels < 4:
        raise ValueError("min_region_pixels must be an integer >= 4")
    if not np.isfinite(fraction_limit) or not 0.0 < fraction_limit <= 0.5:
        raise ValueError("max_region_fraction must be in (0, 0.5]")
    if type(seed_radius_px) is not int or not 1 <= seed_radius_px <= 12:
        raise ValueError("seed_radius_px must be an integer in [1, 12]")
    y0, y1 = max(0, v - seed_radius_px), min(h, v + seed_radius_px + 1)
    x0, x1 = max(0, u - seed_radius_px), min(w, u + seed_radius_px + 1)
    local = depth[y0:y1, x0:x1]
    local_valid = np.isfinite(local) & (local > 0.0)
    if not local_valid.any():
        raise ValueError("source seed has no valid metric-depth samples")
    seed_depth_m = float(np.median(local[local_valid]))
    focal_px = max(abs(float(intrinsics[0, 0])), abs(float(intrinsics[1, 1])))
    radius_px = int(np.ceil(focal_px * radius_m / seed_depth_m))
    radius_px = max(seed_radius_px + 1, min(radius_px, max(h, w)))
    yy, xx = np.mgrid[:h, :w]
    radial = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    eligible = radial & valid_depth & (np.abs(depth - seed_depth_m) <= tolerance_m)
    seed_locations: list[tuple[int, int, int]] = []
    for seed_y in range(y0, y1):
        for seed_x in range(x0, x1):
            if eligible[seed_y, seed_x]:
                seed_locations.append(((seed_x - u) ** 2 + (seed_y - v) ** 2, seed_y, seed_x))
    if not seed_locations:
        raise ValueError("source seed has no sample inside the configured depth band")
    _, seed_y, seed_x = min(seed_locations)
    region = np.zeros((h, w), dtype=bool)
    stack = [(seed_x, seed_y)]
    while stack:
        x, y = stack.pop()
        if region[y, x] or not eligible[y, x]:
            continue
        region[y, x] = True
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not region[ny, nx]:
                stack.append((nx, ny))
    area = int(region.sum())
    fraction = float(area / (h * w))
    if area < min_region_pixels:
        raise ValueError(f"arrow-seeded RGB-D region is too small: {area} < {min_region_pixels} pixels")
    if fraction > fraction_limit:
        raise ValueError(
            "arrow-seeded RGB-D region exceeds max_region_fraction: "
            f"{fraction:.6f} > {fraction_limit:.6f}"
        )
    audit: dict[str, object] = {
        "method": "arrow_seeded_metric_depth_component_v1",
        "source_pixel_xy": [float(source[0]), float(source[1])],
        "seed_pixel_xy": [int(seed_x), int(seed_y)],
        "seed_depth_m": seed_depth_m,
        "region_radius_m": radius_m,
        "region_radius_px": radius_px,
        "depth_tolerance_m": tolerance_m,
        "region_area_px": area,
        "region_fraction": fraction,
        "region_mask_sha256": hashlib.sha256(np.ascontiguousarray(region.astype(np.uint8)).tobytes()).hexdigest(),
        "touches_image_border": bool(np.any(region[0]) or np.any(region[-1]) or np.any(region[:, 0]) or np.any(region[:, -1])),
        "rgb_sha256": hashlib.sha256(np.ascontiguousarray(clean_rgb).tobytes()).hexdigest(),
        "depth_sha256": hashlib.sha256(np.ascontiguousarray(depth).tobytes()).hexdigest(),
    }
    return region, audit


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


def compute_endpoint_change_evidence(
    before_endpoint: ArrayLike,
    after_endpoint: ArrayLike,
    *,
    before_proprioception: ArrayLike | None = None,
    after_proprioception: ArrayLike | None = None,
) -> EndpointChangeEvidence:
    """Compute pure numeric before/after endpoint and proprioception evidence.

    The two endpoint arrays must have the same 2-D or 3-D shape.  Optional
    proprioception arrays must likewise be supplied as a matching pair with
    identical shape.  This function intentionally accepts no object, pose,
    simulator, environment, or evaluator handles.
    """
    before = _as_finite_array(before_endpoint, name="before_endpoint").reshape(-1)
    after = _as_finite_array(after_endpoint, name="after_endpoint").reshape(-1)
    if before.size not in (2, 3) or after.size != before.size:
        raise ValueError("endpoint arrays must have the same length of two or three")
    endpoint_delta = after - before
    proprio_before = None
    proprio_after = None
    proprio_delta = None
    proprio_distance = None
    if (before_proprioception is None) != (after_proprioception is None):
        raise ValueError("before and after proprioception must be supplied together")
    if before_proprioception is not None and after_proprioception is not None:
        proprio_before = _as_finite_array(before_proprioception, name="before_proprioception")
        proprio_after = _as_finite_array(after_proprioception, name="after_proprioception")
        if proprio_before.shape != proprio_after.shape:
            raise ValueError("before and after proprioception must have matching shapes")
        proprio_delta = proprio_after - proprio_before
        proprio_distance = float(np.linalg.norm(proprio_delta.reshape(-1)))
    return EndpointChangeEvidence(
        before_endpoint=before,
        after_endpoint=after,
        endpoint_delta=endpoint_delta,
        endpoint_distance=float(np.linalg.norm(endpoint_delta)),
        before_proprioception=proprio_before,
        after_proprioception=proprio_after,
        proprioception_delta=proprio_delta,
        proprioception_distance=proprio_distance,
    )


# Both names are deliberately explicit; the former reads well in reports and
# the latter is convenient in a small runner utility.
endpoint_change_evidence = compute_endpoint_change_evidence
measure_endpoint_change = compute_endpoint_change_evidence
compute_endpoint_change = compute_endpoint_change_evidence


__all__ = [
    "ArrowObservation",
    "ArrowCommand2D",
    "ArrowEncoding",
    "LEGACY_ARROW_ENCODING",
    "COLOR_ENDPOINT_ARROW_ENCODING",
    "ArrowDecodeDiagnostics",
    "RGBDEndpoint",
    "EndpointChangeEvidence",
    "GraspRetentionEvidence",
    "PostLiftGraspRetentionEvidence",
    "assess_grasp_retention",
    "evaluate_post_lift_grasp_retention",
    "post_lift_grasp_retention",
    "DepthSupportEvidence",
    "analyze_depth_support",
    "depth_support_evidence",
    "BowlWaypointConfig",
    "resolve_arrow_encoding",
    "decode_arrow",
    "decode_arrow_diagnostics",
    "estimate_endpoint_depth",
    "deproject_endpoint",
    "refine_rgbd_endpoint",
    "refine_endpoint_rgbd",
    "refine_endpoint_from_rgbd",
    "refine_rgbd",
    "build_bowl_waypoints",
    "arrow_world_xy_basis",
    "RGBDApproachCandidate",
    "RGBDBoundedApproachGeometry",
    "derive_rgbd_bounded_approach_candidates",
    "build_rgbd_approach_candidates",
    "derive_rgbd_approach_candidates",
    "SourceApproachCandidate",
    "derive_rgbd_source_approach_candidates",
    "SupportPlaneEstimate",
    "estimate_destination_support_plane",
    "release_point_on_support_plane",
    "derive_rgbd_region_grasp_candidates",
    "derive_rgbd_region_mask",
    "normalized_osc_action",
    "compute_endpoint_change_evidence",
    "endpoint_change_evidence",
    "measure_endpoint_change",
    "compute_endpoint_change",
]
