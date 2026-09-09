"""Native-ready Trace artifact loader and policy factory.

``native_legion_factory`` owns model/environment construction.  This module
owns the Trace-specific dependency bundle that factory must pass as
``policy_kwargs``.  Keeping it separate makes the policy independently
testable and prevents a missing route or geometry artifact from silently
turning an ``arrow_trace`` run into a frozen-base run.

The loader is intentionally create-only and read-only: it never writes,
mutates, or synthesizes a route artifact.  A native launcher supplies the
capture/endpoint callbacks after it has created the environment and Arrow
perception worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, GeometryAnchors, ObservationFrame, _safe
from .rgbd_geometry import (
    ArrowRGBDGeometryProvider,
    CaptureFn,
    EndpointFn,
    TraceSimulatorAssistedArrowGeometryProvider,
    _hash_payload,
)
from .trace import TracePolicy, TraceRoute, extract_state_route
from .waypoint_controller import WaypointController, WaypointControllerConfig


TRACE_ROUTE_ARTIFACT_SCHEMA = "arrow_policy_suite.trace_route_artifact.v1"
TRACE_GEOMETRY_VARIANTS = {"rgbd", "simulator_assisted_arrow", "simulator_assisted_rgbd"}


def _reject_action_image_fields(value: Any, path: str = "artifact") -> None:
    """Reject retained actions/images while allowing explicit omitted markers."""
    omitted = {"omitted", "excluded", "dropped", "not_available", "not-applicable", "not-captured"}
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower().replace("-", "_")
            action_key = key in {"action", "actions"} or key.endswith("_action") or key.endswith("_actions")
            image_key = key in {"image", "images"} or key.endswith("_image") or key.endswith("_images")
            if action_key or image_key:
                allowed = nested is None or nested is False or str(nested).lower() in omitted
                if not allowed:
                    raise ContractError(f"Trace artifact must omit actions/images ({path}.{raw_key})")
            _reject_action_image_fields(nested, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_action_image_fields(nested, f"{path}[{index}]")


def _read_create_only_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    target = Path(path).expanduser()
    if not target.exists() or not target.is_file():
        raise ContractError(f"{label} artifact does not exist: {target}")
    if target.is_symlink():
        raise ContractError(f"{label} artifact must not be a symlink")
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} artifact {target}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} artifact root must be an object")
    if value.get("create_only", True) is not True:
        raise ContractError(f"{label} artifact is not create-only")
    expected = value.get("content_sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and str(expected).lower() != actual:
        raise ContractError(f"{label} artifact content_sha256 does not match bytes")
    return value, actual


def _point(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be a three-value numeric point")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a three-value numeric point") from exc
    if len(result) != 3 or any(not __import__("math").isfinite(item) for item in result):
        raise ContractError(f"{name} must be a three-value finite point")
    return result


def _load_calibration(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, content_hash = _read_create_only_json(path, label="Trace calibration")
    for key in ("revision", "frame_name", "intrinsics", "world_from_camera"):
        if key not in payload:
            raise ContractError(f"Trace calibration is missing {key}")
    revision = str(payload["revision"])
    frame_name = str(payload["frame_name"])
    if not revision.strip() or not frame_name.strip():
        raise ContractError("Trace calibration revision and frame_name are required")
    # Reuse provider validation, but do not instantiate a provider or invoke
    # runtime callbacks during loading.
    from .rgbd_geometry import _validate_calibration
    K, T = _validate_calibration(payload["intrinsics"], payload["world_from_camera"])
    computed = _hash_payload({"intrinsics": K, "world_from_camera": T,
                              "frame": frame_name, "revision": revision})
    declared = payload.get("calibration_hash", computed)
    if str(declared) != computed:
        raise ContractError("Trace calibration_hash does not match calibration values")
    return {
        "revision": revision, "frame_name": frame_name,
        "intrinsics": K, "world_from_camera": T,
        "calibration_hash": computed, "content_sha256": content_hash,
    }, content_hash


def _load_routes(path: str | Path) -> tuple[tuple[TraceRoute, ...], str]:
    payload, content_hash = _read_create_only_json(path, label="Trace route")
    if payload.get("schema") != TRACE_ROUTE_ARTIFACT_SCHEMA:
        raise ContractError(f"Trace route artifact schema must be {TRACE_ROUTE_ARTIFACT_SCHEMA!r}")
    _reject_action_image_fields(payload)
    routes_payload = payload.get("routes")
    if not isinstance(routes_payload, list) or not routes_payload:
        raise ContractError("Trace route artifact must contain at least one route")
    routes: list[TraceRoute] = []
    seen: set[str] = set()
    for index, item in enumerate(routes_payload):
        if not isinstance(item, Mapping):
            raise ContractError(f"Trace route {index} must be an object")
        route_id = str(item.get("route_id", f"route-{index}"))
        if route_id in seen:
            raise ContractError(f"duplicate Trace route_id {route_id!r}")
        seen.add(route_id)
        states = item.get("states")
        if not isinstance(states, list) or len(states) < 2:
            raise ContractError(f"Trace route {route_id!r} needs at least two states")
        source_anchor = item.get("source_anchor")
        destination_anchor = item.get("destination_anchor")
        if source_anchor is None or destination_anchor is None:
            raise ContractError(f"Trace route {route_id!r} is missing source/destination anchors")
        source_role = item.get("source_role")
        destination_role = item.get("destination_role")
        graph_triplet = item.get("graph_triplet")
        if not isinstance(source_role, str) or not source_role.strip():
            raise ContractError(f"Trace route {route_id!r} is missing source_role")
        if not isinstance(destination_role, str) or not destination_role.strip():
            raise ContractError(f"Trace route {route_id!r} is missing destination_role")
        if not isinstance(graph_triplet, list) or len(graph_triplet) != 3:
            raise ContractError(f"Trace route {route_id!r} is missing graph_triplet")
        provenance = item.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ContractError(f"Trace route {route_id!r} provenance must be an object")
        frame_name = item.get("coordinate_frame", provenance.get("coordinate_frame"))
        if not isinstance(frame_name, str) or not frame_name.strip():
            raise ContractError(f"Trace route {route_id!r} is missing coordinate_frame")
        if str(item.get("units", provenance.get("units", "m"))).lower() not in {"m", "meter", "meters"}:
            raise ContractError(f"Trace route {route_id!r} must declare metric units")
        route = extract_state_route(
            states, source_anchor=_point(source_anchor, "source_anchor"),
            destination_anchor=_point(destination_anchor, "destination_anchor"), route_id=route_id,
            samples=int(item.get("samples", max(2, min(128, len(states))))),
            close_threshold=float(item.get("close_threshold", 0.0)), source_role=source_role,
            destination_role=destination_role, coordinate_frame=frame_name,
            units="m", graph_triplet=tuple(str(value) for value in graph_triplet),
        )
        if item.get("action_free", True) is not True:
            raise ContractError(f"Trace route {route_id!r} is not action-free")
        routes.append(route)
    return tuple(routes), content_hash


@dataclass(frozen=True)
class TraceNativeComponents:
    """All dependencies needed to construct an executable Trace policy."""

    policy: TracePolicy
    policy_kwargs: Mapping[str, Any]
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | None]
    waypoint_controller: WaypointController
    geometry_provider: Any
    routes: tuple[TraceRoute, ...]
    route_artifact_sha256: str
    calibration_artifact_sha256: str
    geometry_variant: str
    calibration_revision: str
    graph_context_revision: str


def build_trace_policy_components(
    *,
    route_artifact: str | Path,
    calibration_artifact: str | Path,
    geometry_variant: str,
    capture_fn: CaptureFn | None = None,
    endpoint_fn: EndpointFn | None = None,
    simulator_anchors_fn: Callable[[ObservationFrame], GeometryAnchors | Mapping[str, Any]] | None = None,
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | None] | None = None,
    graph_context_revision: str | None = None,
    waypoint_config: WaypointControllerConfig | None = None,
    lookahead: int = 2,
) -> TraceNativeComponents:
    """Load immutable artifacts and construct a real Trace policy.

    No argument is optional merely for convenience: missing route,
    calibration, graph context, or the corresponding geometry callback is a
    construction error.  In particular, this function never returns a base
    policy or a no-op waypoint function.
    """
    variant = str(geometry_variant).strip().lower()
    if variant not in TRACE_GEOMETRY_VARIANTS:
        raise ContractError(f"Trace geometry_variant must be one of {sorted(TRACE_GEOMETRY_VARIANTS)}")
    routes, route_hash = _load_routes(route_artifact)
    calibration, calibration_hash = _load_calibration(calibration_artifact)
    if not callable(graph_context_fn):
        raise ContractError("Trace requires a graph_context_fn for native construction")
    if graph_context_revision is None or not str(graph_context_revision).strip() or str(graph_context_revision).lower() in {"unresolved", "unknown", "latest"}:
        raise ContractError("Trace requires an explicit graph_context_revision")
    if variant == "rgbd":
        if not callable(capture_fn) or not callable(endpoint_fn):
            raise ContractError("Trace RGB-D construction requires capture_fn and endpoint_fn")
        geometry_provider = ArrowRGBDGeometryProvider(
            capture_fn, endpoint_fn, expected_frame_name=calibration["frame_name"],
            expected_calibration_revision=calibration["revision"],
            expected_calibration_hash=calibration["calibration_hash"],
        )
    else:
        if not callable(simulator_anchors_fn):
            raise ContractError("simulator-assisted Trace construction requires simulator_anchors_fn")
        geometry_provider = TraceSimulatorAssistedArrowGeometryProvider(
            simulator_anchors_fn, frame_name=calibration["frame_name"],
            calibration_revision=calibration["revision"],
        )
    controller = WaypointController(waypoint_config)
    policy_kwargs = {
        "routes": routes, "geometry_provider": geometry_provider,
        "waypoint_action": controller, "lookahead": int(lookahead),
    }
    policy = TracePolicy(**policy_kwargs)
    return TraceNativeComponents(
        policy=policy, policy_kwargs=policy_kwargs, graph_context_fn=graph_context_fn,
        waypoint_controller=controller, geometry_provider=geometry_provider, routes=routes,
        route_artifact_sha256=route_hash, calibration_artifact_sha256=calibration_hash,
        geometry_variant=variant, calibration_revision=calibration["revision"],
        graph_context_revision=str(graph_context_revision),
    )


def trace_policy_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return the exact kwargs a native ``NativeHostSpec`` needs for Trace."""
    return dict(build_trace_policy_components(**kwargs).policy_kwargs)


def build_trace_native_fields(**kwargs: Any) -> dict[str, Any]:
    """Return the explicit fields a native factory must thread into its host.

    This is an integration contract, not a second host builder.  A launcher
    should merge these values into ``NativeHostSpec`` while retaining its own
    environment, VLA, and same-frame Arrow teacher.
    """
    components = build_trace_policy_components(**kwargs)
    return {
        "policy_id": "arrow_trace",
        "policy": components.policy,
        "policy_kwargs": dict(components.policy_kwargs),
        "graph_context_fn": components.graph_context_fn,
        "graph_context_revision": components.graph_context_revision,
        "trace_geometry_variant": components.geometry_variant,
        "trace_route_artifact_sha256": components.route_artifact_sha256,
        "trace_calibration_artifact_sha256": components.calibration_artifact_sha256,
    }


def build_trace_policy(**kwargs: Any) -> TracePolicy:
    """Construct the executable Trace policy for injected native components."""
    return build_trace_policy_components(**kwargs).policy


__all__ = [
    "TRACE_ROUTE_ARTIFACT_SCHEMA", "TRACE_GEOMETRY_VARIANTS", "TraceNativeComponents",
    "build_trace_policy_components", "trace_policy_kwargs", "build_trace_policy",
    "build_trace_native_fields",
]
