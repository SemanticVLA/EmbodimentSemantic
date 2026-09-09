"""Strict RGB-D geometry providers for Arrow Trace.

The Trace policy is deliberately action-free: its demonstration is a route of
canonical states and its query-time alignment comes from visual arrow
endpoints.  This module owns the one place where pixels become metric points.
It keeps calibration/timestamp/provenance attached to the returned anchors and
fails closed when the observation is not a synchronized RGB-D frame.

The simulator-assisted provider is intentionally separate and loudly named.
It is useful for debugging route alignment, but its provenance cannot be
reported as vision-only by the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, GeometryAnchors, ObservationFrame


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _hash_payload(value: Any) -> str:
    """Hash calibration values without depending on NumPy serialization."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif isinstance(value, Mapping):
        value = {str(key): _hashable_value(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_hashable_value(item) for item in value]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hashable_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _hashable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable_value(item) for item in value]
    return value


def _as_array(value: Any, name: str, shape: tuple[int, ...]):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - native RGB-D requires NumPy
        raise ContractError("Trace RGB-D geometry requires NumPy") from exc
    array = np.asarray(value)
    if array.shape != shape:
        raise ContractError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _validate_calibration(intrinsics: Any, world_from_camera: Any) -> tuple[Any, Any]:
    import numpy as np

    K = _as_array(intrinsics, "intrinsics", (3, 3))
    T = _as_array(world_from_camera, "world_from_camera", (4, 4))
    if abs(float(K[0, 0])) <= 1e-9 or abs(float(K[1, 1])) <= 1e-9:
        raise ContractError("intrinsics must have non-zero focal lengths")
    if not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1e-6):
        raise ContractError("intrinsics homogeneous row must be [0, 0, 1]")
    if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ContractError("world_from_camera must be homogeneous")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or float(np.linalg.det(R)) <= 0.0:
        raise ContractError("world_from_camera rotation must be proper orthonormal")
    return K, T


def _reject_privileged_metadata(value: Any, path: str = "capture") -> None:
    forbidden = (
        "mujoco", "sim_state", "ground_truth", "ground-truth", "object_pose",
        "object_world", "privileged", "oracle", "bbox", "bounding_box",
    )
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(token in key_text for token in forbidden):
                raise ContractError(f"Trace RGB-D provider cannot consume privileged field {path}.{key}")
            _reject_privileged_metadata(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_privileged_metadata(nested, f"{path}[{index}]")


@dataclass(frozen=True)
class ArrowRGBDObservation:
    """A synchronized camera packet consumed by the strict provider.

    ``world_from_camera`` maps OpenCV camera coordinates (x right, y down,
    z forward) into ``frame_name``.  Depth is metric metres; no implicit
    simulator units or renderer scaling are accepted.
    """

    rgb: Any
    depth_m: Any
    intrinsics: Any
    world_from_camera: Any
    frame_name: str
    calibration_revision: str
    timestamp_s: float
    rgb_timestamp_s: float | None = None
    depth_timestamp_s: float | None = None
    camera_id: str = "agentview"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frame_name or not self.calibration_revision or not self.camera_id:
            raise ContractError("RGB-D frame, camera, and calibration provenance are required")
        timestamp = _finite(self.timestamp_s, "timestamp_s")
        rgb_time = timestamp if self.rgb_timestamp_s is None else _finite(self.rgb_timestamp_s, "rgb_timestamp_s")
        depth_time = timestamp if self.depth_timestamp_s is None else _finite(self.depth_timestamp_s, "depth_timestamp_s")
        if abs(rgb_time - depth_time) > 1e-3:
            raise ContractError("RGB and depth timestamps are not synchronized")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "rgb_timestamp_s", rgb_time)
        object.__setattr__(self, "depth_timestamp_s", depth_time)
        _validate_calibration(self.intrinsics, self.world_from_camera)
        try:
            import numpy as np
            rgb = np.asarray(self.rgb)
            depth = np.asarray(self.depth_m)
            if rgb.ndim < 2 or depth.ndim != 2 or tuple(rgb.shape[:2]) != tuple(depth.shape):
                raise ContractError("RGB and depth must share height and width")
            if depth.dtype.kind not in "fiu":
                raise ContractError("depth_m must be numeric metric depth")
            if not np.any(np.isfinite(depth) & (depth > 0.0)):
                raise ContractError("depth frame has no valid metric samples")
        except ImportError as exc:  # pragma: no cover
            raise ContractError("Trace RGB-D geometry requires NumPy") from exc
        _reject_privileged_metadata(self.metadata)


@dataclass(frozen=True)
class ArrowPixelEndpoints:
    source_xy: tuple[float, float]
    destination_xy: tuple[float, float]
    source_role: str | None = None
    destination_role: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, point in (("source_xy", self.source_xy), ("destination_xy", self.destination_xy)):
            if len(point) != 2 or any(not math.isfinite(float(v)) for v in point):
                raise ContractError(f"{name} must contain two finite pixel coordinates")
        if self.source_role is not None and not str(self.source_role).strip():
            raise ContractError("source_role must be non-empty when supplied")
        if self.destination_role is not None and not str(self.destination_role).strip():
            raise ContractError("destination_role must be non-empty when supplied")
        _reject_privileged_metadata(self.provenance, "endpoint")


CaptureFn = Callable[[ObservationFrame], ArrowRGBDObservation | Mapping[str, Any]]
EndpointFn = Callable[[ObservationFrame, ArrowRGBDObservation], ArrowPixelEndpoints | Mapping[str, Any]]


def _coerce_capture(value: ArrowRGBDObservation | Mapping[str, Any]) -> ArrowRGBDObservation:
    if isinstance(value, ArrowRGBDObservation):
        return value
    if not isinstance(value, Mapping):
        raise ContractError("RGB-D capture must be ArrowRGBDObservation or mapping")
    _reject_privileged_metadata(value)
    def get(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in value:
                return value[name]
        return default
    rgb = get("rgb", "clean_rgb", "image")
    depth = get("depth_m", "depth")
    K = get("intrinsics", "K", "camera_intrinsics")
    T = get("world_from_camera", "T_world_camera", "camera_to_world")
    timestamp = get("timestamp_s", "timestamp")
    if any(item is None for item in (rgb, depth, K, T, timestamp)):
        raise ContractError("RGB-D capture requires rgb, depth_m, intrinsics, world_from_camera, timestamp_s")
    return ArrowRGBDObservation(
        rgb=rgb, depth_m=depth, intrinsics=K, world_from_camera=T,
        frame_name=str(get("frame_name", "coordinate_frame", default="world")),
        calibration_revision=str(get("calibration_revision", "calibration_id", default="unresolved")),
        timestamp_s=timestamp,
        rgb_timestamp_s=get("rgb_timestamp_s", "rgb_timestamp"),
        depth_timestamp_s=get("depth_timestamp_s", "depth_timestamp"),
        camera_id=str(get("camera_id", "sensor_id", default="agentview")),
        metadata=get("metadata", default={}),
    )


def _coerce_endpoints(value: ArrowPixelEndpoints | Mapping[str, Any]) -> ArrowPixelEndpoints:
    if isinstance(value, ArrowPixelEndpoints):
        return value
    if not isinstance(value, Mapping):
        raise ContractError("arrow endpoint detector must return ArrowPixelEndpoints or mapping")
    _reject_privileged_metadata(value, "endpoint")
    source = value.get("source_xy", value.get("source", value.get("source_pixel_xy")))
    destination = value.get("destination_xy", value.get("destination", value.get("destination_pixel_xy")))
    if source is None or destination is None:
        raise ContractError("arrow endpoint detector must return source and destination pixels")
    return ArrowPixelEndpoints(
        tuple(float(v) for v in source), tuple(float(v) for v in destination),
        None if value.get("source_role") is None else str(value["source_role"]),
        None if value.get("destination_role") is None else str(value["destination_role"]),
        value.get("provenance", {}),
    )


def _depth_at(depth: Any, pixel: Sequence[float], radius: int = 2) -> float:
    import numpy as np
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2:
        raise ContractError("Trace RGB-D depth must be a 2-D metric image")
    u, v = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
    height, width = array.shape
    if not (0 <= u < width and 0 <= v < height):
        raise ContractError("arrow endpoint pixel lies outside the depth image")
    local = array[max(0, v - radius):min(height, v + radius + 1), max(0, u - radius):min(width, u + radius + 1)]
    valid = local[np.isfinite(local) & (local > 1e-6)]
    if valid.size < 3:
        raise ContractError("arrow endpoint lacks at least three valid depth samples")
    return float(np.median(valid))


def _deproject(pixel: Sequence[float], depth_m: Any, K: Any, T: Any) -> tuple[float, float, float]:
    import numpy as np
    depth = _depth_at(depth_m, pixel)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    camera = np.array(((float(pixel[0]) - cx) * depth / fx, (float(pixel[1]) - cy) * depth / fy, depth, 1.0))
    world = np.asarray(T, dtype=np.float64) @ camera
    if abs(float(world[3])) <= 1e-9:
        raise ContractError("deprojected point has zero homogeneous scale")
    point = world[:3] / world[3]
    if not np.all(np.isfinite(point)):
        raise ContractError("deprojected endpoint is non-finite")
    return tuple(float(v) for v in point)


class ArrowRGBDGeometryProvider:
    """Strict vision-only RGB-D endpoint provider for Trace."""

    provider = "arrow_rgbd"
    provider_revision = "arrow-trace-rgbd-v1"
    provider_hash = hashlib.sha256(provider_revision.encode("utf-8")).hexdigest()

    def __init__(
        self,
        capture_fn: CaptureFn,
        endpoint_fn: EndpointFn | None = None,
        *,
        expected_frame_name: str = "world",
        expected_calibration_revision: str | None = None,
        expected_calibration_hash: str | None = None,
        max_frame_age_s: float = 0.25,
        max_rgb_depth_skew_s: float = 1e-3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(capture_fn):
            raise ContractError("capture_fn must be callable")
        if endpoint_fn is not None and not callable(endpoint_fn):
            raise ContractError("endpoint_fn must be callable")
        if not expected_frame_name or max_frame_age_s < 0 or max_rgb_depth_skew_s < 0:
            raise ContractError("RGB-D frame and freshness configuration is invalid")
        self.capture_fn = capture_fn
        self.endpoint_fn = endpoint_fn
        self.expected_frame_name = expected_frame_name
        self.expected_calibration_revision = expected_calibration_revision
        self.expected_calibration_hash = expected_calibration_hash
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_rgb_depth_skew_s = float(max_rgb_depth_skew_s)
        self.clock = clock

    def _endpoints(self, frame: ObservationFrame, capture: ArrowRGBDObservation) -> ArrowPixelEndpoints:
        if self.endpoint_fn is not None:
            return _coerce_endpoints(self.endpoint_fn(frame, capture))
        metadata = capture.metadata
        source = metadata.get("source_xy", metadata.get("source_pixel_xy"))
        destination = metadata.get("destination_xy", metadata.get("destination_pixel_xy"))
        if source is None or destination is None:
            raise ContractError("strict RGB-D provider needs an Arrow endpoint detector")
        return ArrowPixelEndpoints(tuple(source), tuple(destination),
                                   metadata.get("source_role"), metadata.get("destination_role"),
                                   {"source": "capture_metadata"})

    def anchors(self, frame: ObservationFrame) -> GeometryAnchors:
        capture = _coerce_capture(self.capture_fn(frame))
        if capture.frame_name != self.expected_frame_name:
            raise ContractError(f"RGB-D frame mismatch: expected {self.expected_frame_name!r}, got {capture.frame_name!r}")
        if self.expected_calibration_revision is not None and capture.calibration_revision != self.expected_calibration_revision:
            raise ContractError("RGB-D calibration revision does not match frozen provider configuration")
        frame_timestamp = frame.metadata.get("timestamp_s", frame.metadata.get("timestamp"))
        if frame_timestamp is not None and abs(_finite(frame_timestamp, "frame timestamp") - capture.timestamp_s) > self.max_frame_age_s:
            raise ContractError("RGB-D capture is stale relative to the policy observation")
        if abs(float(capture.rgb_timestamp_s) - float(capture.depth_timestamp_s)) > self.max_rgb_depth_skew_s:
            raise ContractError("RGB-D capture exceeds synchronized timestamp skew")
        K, T = _validate_calibration(capture.intrinsics, capture.world_from_camera)
        endpoints = self._endpoints(frame, capture)
        source = _deproject(endpoints.source_xy, capture.depth_m, K, T)
        destination = _deproject(endpoints.destination_xy, capture.depth_m, K, T)
        if math.dist(source, destination) <= 1e-9:
            raise ContractError("RGB-D arrow anchors are degenerate")
        calibration_hash = _hash_payload({"intrinsics": K, "world_from_camera": T, "frame": capture.frame_name,
                                          "revision": capture.calibration_revision})
        if self.expected_calibration_hash is not None and calibration_hash != self.expected_calibration_hash:
            raise ContractError("RGB-D calibration hash does not match frozen artifact")
        provenance = {
            "source_kind": "arrow_rgbd", "coordinate_frame": capture.frame_name, "units": "m",
            "provider_revision": self.provider_revision, "provider_hash": self.provider_hash,
            "calibration_hash": calibration_hash, "camera_id": capture.camera_id,
            "timestamp_s": capture.timestamp_s, "rgb_timestamp_s": capture.rgb_timestamp_s,
            "depth_timestamp_s": capture.depth_timestamp_s, "frame_timestamp_checked": frame_timestamp is not None,
            "endpoint_source_xy": endpoints.source_xy, "endpoint_destination_xy": endpoints.destination_xy,
            "source_role": endpoints.source_role, "destination_role": endpoints.destination_role,
            "endpoint_provenance": dict(endpoints.provenance), "vision_only": True,
        }
        return GeometryAnchors(source, destination, capture.frame_name, capture.calibration_revision,
                               self.provider, provenance)

    __call__ = anchors


class TraceSimulatorAssistedArrowGeometryProvider:
    """Explicit privileged diagnostic provider; never vision-only."""

    provider = "trace_simulator_assisted_arrow"
    provider_revision = "trace-simulator-assisted-arrow-v1"
    provider_hash = hashlib.sha256(provider_revision.encode("utf-8")).hexdigest()

    def __init__(self, anchors_fn: Callable[[ObservationFrame], GeometryAnchors | Mapping[str, Any]], *, frame_name: str = "world", calibration_revision: str = "sim-calibration") -> None:
        if not callable(anchors_fn):
            raise ContractError("anchors_fn must be callable")
        self.anchors_fn = anchors_fn
        self.frame_name = frame_name
        self.calibration_revision = calibration_revision

    def anchors(self, frame: ObservationFrame) -> GeometryAnchors:
        raw = self.anchors_fn(frame)
        if isinstance(raw, GeometryAnchors):
            source, destination = raw.source, raw.destination
            frame_name, calibration_revision = raw.frame_name, raw.calibration_revision
            extra = dict(raw.provenance)
        elif isinstance(raw, Mapping):
            source, destination = raw.get("source"), raw.get("destination")
            if source is None or destination is None:
                raise ContractError("simulator-assisted provider needs source and destination anchors")
            frame_name = str(raw.get("frame_name", self.frame_name))
            calibration_revision = str(raw.get("calibration_revision", self.calibration_revision))
            extra = dict(raw.get("provenance", {}))
        else:
            raise ContractError("simulator-assisted provider must return GeometryAnchors or mapping")
        extra.update({"source_kind": self.provider, "diagnostic_only": True, "vision_only": False,
                      "provider_revision": self.provider_revision, "provider_hash": self.provider_hash,
                      "coordinate_frame": frame_name, "units": "m"})
        return GeometryAnchors(tuple(float(v) for v in source), tuple(float(v) for v in destination),
                               frame_name, calibration_revision, self.provider, extra)

    __call__ = anchors


class SimulatorAssistedRGBDGeometryProvider:
    """Diagnostic RGB-D provider whose endpoint detector may use simulator hints.

    The capture, calibration, synchronization, and deprojection contracts are
    identical to :class:`ArrowRGBDGeometryProvider`.  Only endpoint selection
    differs: ``endpoint_fn`` may use simulator-derived bounding boxes or arrow
    annotations.  Every anchor is consequently marked diagnostic-only and is
    not eligible for the vision-only Trace result.
    """

    provider = "simulator_assisted_rgbd"
    provider_revision = "simulator-assisted-rgbd-v1"
    provider_hash = hashlib.sha256(provider_revision.encode("utf-8")).hexdigest()

    def __init__(
        self,
        capture_fn: CaptureFn,
        endpoint_fn: EndpointFn,
        *,
        expected_frame_name: str = "world",
        expected_calibration_revision: str | None = None,
        expected_calibration_hash: str | None = None,
        max_frame_age_s: float = 0.25,
        max_rgb_depth_skew_s: float = 1e-3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not callable(capture_fn) or not callable(endpoint_fn):
            raise ContractError("simulator-assisted RGB-D provider requires capture_fn and endpoint_fn")
        if not expected_frame_name or max_frame_age_s < 0 or max_rgb_depth_skew_s < 0:
            raise ContractError("RGB-D frame and freshness configuration is invalid")
        if not callable(clock):
            raise ContractError("RGB-D clock must be callable")
        self.capture_fn = capture_fn
        self.endpoint_fn = endpoint_fn
        self.expected_frame_name = str(expected_frame_name)
        self.expected_calibration_revision = expected_calibration_revision
        self.expected_calibration_hash = expected_calibration_hash
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_rgb_depth_skew_s = float(max_rgb_depth_skew_s)
        self.clock = clock

    @staticmethod
    def _diagnostic_endpoints(value: ArrowPixelEndpoints | Mapping[str, Any]) -> tuple[ArrowPixelEndpoints, Mapping[str, Any]]:
        """Coerce endpoint pixels while retaining privileged provenance.

        This deliberately does not call the strict ``_coerce_endpoints`` path,
        whose recursive rejection of bbox/simulator fields is part of the
        vision-only contract.
        """
        if isinstance(value, ArrowPixelEndpoints):
            source, destination = value.source_xy, value.destination_xy
            source_role, destination_role = value.source_role, value.destination_role
            provenance = dict(value.provenance)
        elif isinstance(value, Mapping):
            source = value.get("source_xy", value.get("source", value.get("source_pixel_xy")))
            destination = value.get("destination_xy", value.get("destination", value.get("destination_pixel_xy")))
            if source is None or destination is None:
                raise ContractError("simulator-assisted endpoint detector needs source and destination pixels")
            source_role = None if value.get("source_role") is None else str(value["source_role"])
            destination_role = None if value.get("destination_role") is None else str(value["destination_role"])
            provenance = dict(value.get("provenance", {}))
        else:
            raise ContractError("simulator-assisted endpoint detector must return endpoint pixels or a mapping")
        try:
            source_xy = tuple(float(item) for item in source)
            destination_xy = tuple(float(item) for item in destination)
        except (TypeError, ValueError) as exc:
            raise ContractError("simulator-assisted endpoint pixels must be numeric") from exc
        if len(source_xy) != 2 or len(destination_xy) != 2 or any(
            not math.isfinite(item) for item in (*source_xy, *destination_xy)
        ):
            raise ContractError("simulator-assisted endpoint pixels must be finite 2-vectors")
        if not isinstance(provenance, Mapping):
            raise ContractError("simulator-assisted endpoint provenance must be a mapping")
        # Rebuild a strict endpoint object only when provenance is non-
        # privileged.  For diagnostic provenance retain the raw fields in the
        # returned mapping and let the provider perform deprojection directly.
        try:
            endpoint = ArrowPixelEndpoints(source_xy, destination_xy, source_role, destination_role, {})
        except ContractError:
            raise
        return endpoint, provenance

    def anchors(self, frame: ObservationFrame) -> GeometryAnchors:
        capture = _coerce_capture(self.capture_fn(frame))
        if capture.frame_name != self.expected_frame_name:
            raise ContractError(
                f"RGB-D frame mismatch: expected {self.expected_frame_name!r}, got {capture.frame_name!r}"
            )
        if self.expected_calibration_revision is not None and capture.calibration_revision != self.expected_calibration_revision:
            raise ContractError("RGB-D calibration revision does not match frozen provider configuration")
        frame_timestamp = frame.metadata.get("timestamp_s", frame.metadata.get("timestamp"))
        if frame_timestamp is not None:
            if abs(_finite(frame_timestamp, "frame timestamp") - capture.timestamp_s) > self.max_frame_age_s:
                raise ContractError("RGB-D capture is stale relative to the policy observation")
        elif abs(_finite(self.clock(), "clock") - capture.timestamp_s) > self.max_frame_age_s:
            raise ContractError("RGB-D capture is stale relative to the current policy time")
        if abs(float(capture.rgb_timestamp_s) - float(capture.depth_timestamp_s)) > self.max_rgb_depth_skew_s:
            raise ContractError("RGB-D capture exceeds synchronized timestamp skew")
        K, T = _validate_calibration(capture.intrinsics, capture.world_from_camera)
        endpoint_value = self.endpoint_fn(frame, capture)
        endpoints, endpoint_provenance = self._diagnostic_endpoints(endpoint_value)
        source = _deproject(endpoints.source_xy, capture.depth_m, K, T)
        destination = _deproject(endpoints.destination_xy, capture.depth_m, K, T)
        if math.dist(source, destination) <= 1e-9:
            raise ContractError("RGB-D arrow anchors are degenerate")
        calibration_hash = _hash_payload({
            "intrinsics": K, "world_from_camera": T,
            "frame": capture.frame_name, "revision": capture.calibration_revision,
        })
        if self.expected_calibration_hash is not None and calibration_hash != self.expected_calibration_hash:
            raise ContractError("RGB-D calibration hash does not match frozen artifact")
        provenance = {
            "source_kind": self.provider,
            "privileged_source": "simulator_bbox_or_arrow",
            "diagnostic_only": True,
            "vision_only": False,
            "provider_revision": self.provider_revision,
            "provider_hash": self.provider_hash,
            "coordinate_frame": capture.frame_name,
            "units": "m",
            "calibration_hash": calibration_hash,
            "calibration_revision": capture.calibration_revision,
            "camera_id": capture.camera_id,
            "timestamp_s": capture.timestamp_s,
            "rgb_timestamp_s": capture.rgb_timestamp_s,
            "depth_timestamp_s": capture.depth_timestamp_s,
            "frame_timestamp_checked": frame_timestamp is not None,
            "endpoint_source_xy": endpoints.source_xy,
            "endpoint_destination_xy": endpoints.destination_xy,
            "source_role": endpoints.source_role,
            "destination_role": endpoints.destination_role,
            "endpoint_provenance": dict(endpoint_provenance),
        }
        return GeometryAnchors(
            source, destination, capture.frame_name, capture.calibration_revision,
            self.provider, provenance,
        )

    __call__ = anchors


def simulator_assisted_rgbd(
    capture_fn: CaptureFn,
    endpoint_fn: EndpointFn,
    **kwargs: Any,
) -> SimulatorAssistedRGBDGeometryProvider:
    """Construct the explicitly diagnostic simulator-assisted RGB-D provider."""
    return SimulatorAssistedRGBDGeometryProvider(capture_fn, endpoint_fn, **kwargs)


# Verbose alias retained for callers that use the Trace naming convention.
TraceSimulatorAssistedRGBDGeometryProvider = SimulatorAssistedRGBDGeometryProvider
trace_simulator_assisted_rgbd = simulator_assisted_rgbd


def trace_simulator_assisted_arrow(
    anchors_fn: Callable[[ObservationFrame], GeometryAnchors | Mapping[str, Any]], **kwargs: Any
) -> TraceSimulatorAssistedArrowGeometryProvider:
    """Named factory used by diagnostic-only canaries and ablations."""
    return TraceSimulatorAssistedArrowGeometryProvider(anchors_fn, **kwargs)


__all__ = [
    "ArrowRGBDObservation", "ArrowPixelEndpoints", "ArrowRGBDGeometryProvider",
    "TraceSimulatorAssistedArrowGeometryProvider", "trace_simulator_assisted_arrow",
    "SimulatorAssistedRGBDGeometryProvider", "TraceSimulatorAssistedRGBDGeometryProvider",
    "simulator_assisted_rgbd", "trace_simulator_assisted_rgbd",
]
