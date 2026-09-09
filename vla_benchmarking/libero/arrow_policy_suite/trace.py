"""State-only teacher-route guidance (Arrow Trace).

Trace intentionally consumes actual canonical states, not teacher actions or
images.  Geometry is supplied by an explicit RGB-D provider with provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import inspect
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ActionProposal,
    ContractError,
    GeometryAnchors,
    ObservationFrame,
    PolicyDecision,
    StepRecord,
    _safe,
    state8,
    validate_action,
)


@dataclass(frozen=True)
class RoutePoint:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    gripper: float
    arc: float
    event: str | None = None
    # Preserve the two canonical finger-qpos values instead of collapsing the
    # route's gripper state irreversibly to the mean.  ``gripper`` remains the
    # backwards-compatible scalar used by existing waypoint adapters.
    gripper_state: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.rotation) != 3:
            raise ContractError("trace point position and rotation must be three-dimensional")
        if any(not math.isfinite(float(v)) for v in (*self.position, *self.rotation)):
            raise ContractError("trace point position and rotation must be finite")
        if not math.isfinite(float(self.gripper)) or not math.isfinite(float(self.arc)):
            raise ContractError("trace point values must be finite")
        if not 0.0 <= float(self.arc) <= 1.0:
            raise ContractError("trace point arc must lie in [0, 1]")
        if self.event not in {None, "close", "reopen"}:
            raise ContractError(f"unknown trace route event {self.event!r}")
        if self.gripper_state is not None:
            if len(self.gripper_state) != 2 or any(not math.isfinite(float(v)) for v in self.gripper_state):
                raise ContractError("trace gripper_state must contain two finite values")


@dataclass(frozen=True)
class TraceRoute:
    points: tuple[RoutePoint, ...]
    source_anchor: tuple[float, float, float]
    destination_anchor: tuple[float, float, float]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    route_id: str = "route"
    # Optional graph labels retained with the route rather than inferred from
    # geometry.  They make the text-triplet -> visual-arrow -> state-route
    # lineage explicit and are intentionally not required by legacy fixtures.
    source_role: str | None = None
    destination_role: str | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ContractError("a trace route needs at least two points")
        if not self.route_id:
            raise ContractError("route_id is required")
        for name, value in (("source_role", self.source_role), ("destination_role", self.destination_role)):
            if value is not None and not str(value).strip():
                raise ContractError(f"{name} must be non-empty when supplied")
        if any(point.event not in {None, "close", "reopen"} for point in self.points):
            raise ContractError("trace route contains an unknown event")
        arcs = [float(point.arc) for point in self.points]
        if any(right < left for left, right in zip(arcs, arcs[1:])):
            raise ContractError("trace route arcs must be monotonic")
        if len(self.source_anchor) != 3 or len(self.destination_anchor) != 3:
            raise ContractError("trace route anchors must be three-dimensional")
        if any(not math.isfinite(float(v)) for v in (*self.source_anchor, *self.destination_anchor)):
            raise ContractError("trace route anchors must be finite")
        if not isinstance(self.provenance, Mapping):
            raise ContractError("trace route provenance must be a mapping")
        provenance = dict(self.provenance)
        # A manually assembled route gets the same explicit action/image-free
        # contract as extract_state_route.  Positive action provenance is
        # rejected so a caller cannot accidentally label a teacher-action route
        # as Trace.
        provenance.setdefault("representation", "state_only")
        provenance.setdefault("actions", "omitted")
        provenance.setdefault("images", "omitted")
        if self.source_role is not None:
            provenance.setdefault("source_role", self.source_role)
        if self.destination_role is not None:
            provenance.setdefault("destination_role", self.destination_role)
        _validate_action_free_provenance(provenance)
        _safe(provenance)
        object.__setattr__(self, "provenance", provenance)


def _validate_action_free_provenance(provenance: Mapping[str, Any]) -> None:
    """Fail closed when route provenance claims that actions were retained."""
    representation = str(provenance.get("representation", "")).lower()
    if representation not in {"state_only", "state-only", "canonical_state_only"}:
        raise ContractError("Trace route representation must be state_only")
    if "action_free" in provenance and provenance["action_free"] is not True:
        raise ContractError("Trace route provenance must declare action_free=True")
    omitted = {"omitted", "excluded", "dropped", "not_available", "not-applicable", "not-captured"}

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, nested in value.items():
            key = str(raw_key).lower().replace("-", "_")
            action_key = key in {"actions", "action"} or (
                "action_free" not in key and (key.endswith("_actions") or key.endswith("_action"))
            )
            image_key = key in {"images", "image"} or (
                key.endswith("_images") or key.endswith("_image")
            )
            if action_key or image_key:
                allowed = nested is None or nested is False or str(nested).lower() in omitted
                if not allowed:
                    noun = "actions" if action_key else "images"
                    raise ContractError(f"Trace route provenance must omit {noun} ({path}.{raw_key})")
            visit(nested, f"{path}.{raw_key}")

    visit(provenance, "provenance")


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    return tuple(float(x) + t * (float(y) - float(x)) for x, y in zip(a, b))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


_ARROW_RGBD_SOURCE_KINDS = {"arrow_rgbd", "arrow_rgbd_perception"}
_ARROW_SIMULATOR_SOURCE_KINDS = {"trace_simulator_assisted_arrow"}
_METRIC_UNITS = {"m", "meter", "meters"}
_FORBIDDEN_GEOMETRY_TOKENS = (
    "mujoco", "oracle", "simulator", "simulation", "simulated", "sim_ground_truth", "ground_truth",
    "ground-truth", "object_pose_gt", "object_pose", "privileged",
)


def _canonical_units(value: Any) -> str:
    text = str(value).lower()
    return "m" if text in _METRIC_UNITS else text


def _validate_geometry_anchors(anchors: GeometryAnchors) -> GeometryAnchors:
    """Validate Trace's stronger RGB-D geometry boundary.

    ``GeometryAnchors`` is shared by other integrations and therefore only
    enforces generic shape/provenance checks.  Trace additionally requires a
    typed Arrow RGB-D source, explicit world-frame/metric declarations, and
    rejects simulator/oracle labels even when they are hidden in nested
    provenance.
    """
    if not isinstance(anchors, GeometryAnchors):
        raise ContractError("Trace geometry provider must return GeometryAnchors")
    if len(anchors.source) != 3 or len(anchors.destination) != 3:
        raise ContractError("Trace geometry anchors must be three-dimensional")
    if any(not math.isfinite(float(value)) for value in (*anchors.source, *anchors.destination)):
        raise ContractError("Trace geometry anchors must be finite")
    if _dist(anchors.source, anchors.destination) <= 1e-9:
        raise ContractError("Trace geometry anchors must define a non-zero span")
    if not isinstance(anchors.provenance, Mapping):
        raise ContractError("Trace geometry provenance must be a mapping")

    def text_values(value: Any) -> list[str]:
        if isinstance(value, Mapping):
            output: list[str] = []
            for key, nested in value.items():
                output.append(str(key).lower())
                output.extend(text_values(nested))
            return output
        if isinstance(value, (list, tuple)):
            output: list[str] = []
            for nested in value:
                output.extend(text_values(nested))
            return output
        return [str(value).lower()]

    provenance = {str(key).lower(): value for key, value in anchors.provenance.items()}
    provider_kind = str(anchors.provider).lower().replace("-", "_")
    all_text = " ".join([str(anchors.provider).lower(), *text_values(anchors.provenance)])
    if provider_kind in _ARROW_SIMULATOR_SOURCE_KINDS:
        # The diagnostic provider is allowed to say "simulator", but remains
        # forbidden from quietly claiming ground-truth/oracle object labels.
        if any(token in all_text for token in ("ground_truth", "ground-truth", "oracle_pose", "object_pose_gt", "privileged")):
            raise ContractError("Trace diagnostic geometry cannot carry hidden oracle/object ground truth")
    elif any(token in all_text for token in _FORBIDDEN_GEOMETRY_TOKENS):
        raise ContractError("Trace geometry provenance cannot use MuJoCo/oracle geometry")
    if provider_kind not in (_ARROW_RGBD_SOURCE_KINDS | _ARROW_SIMULATOR_SOURCE_KINDS):
        raise ContractError("Trace geometry provider must be Arrow RGB-D or explicit simulator-assisted diagnostic")
    source_kind = str(provenance.get(
        "source_kind", provenance.get("kind", provenance.get("source", ""))
    )).lower().replace("-", "_")
    # Older state-only manifests recorded the typed provider in
    # ``arrow_origin`` rather than a dedicated ``source_kind`` field.  Keep
    # that representation accepted only when it explicitly names the RGB-D
    # deprojection path; arbitrary/missing provenance still fails closed.
    arrow_origin = str(provenance.get("arrow_origin", "")).lower()
    legacy_rgbd = "rgbd" in arrow_origin and "deprojection" in arrow_origin
    if not source_kind and legacy_rgbd:
        source_kind = "arrow_rgbd"
    if provider_kind in _ARROW_RGBD_SOURCE_KINDS:
        if source_kind not in _ARROW_RGBD_SOURCE_KINDS:
            raise ContractError("Trace geometry requires typed Arrow RGB-D provenance")
    else:
        if source_kind not in _ARROW_SIMULATOR_SOURCE_KINDS:
            raise ContractError("simulator-assisted Trace requires explicit diagnostic provenance")
        if provenance.get("diagnostic_only") is not True or provenance.get("vision_only") is not False:
            raise ContractError("simulator-assisted Trace must declare diagnostic_only=True and vision_only=False")
    coordinate_frame = provenance.get(
        "coordinate_frame", provenance.get("frame", provenance.get("frame_name"))
    )
    if (not isinstance(coordinate_frame, str) or not coordinate_frame.strip()) and legacy_rgbd:
        coordinate_frame = anchors.frame_name
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ContractError("Trace geometry provenance must declare coordinate_frame")
    if coordinate_frame != anchors.frame_name:
        raise ContractError("Trace geometry frame does not match anchor frame_name")
    units_value = provenance.get("units", provenance.get("unit", provenance.get("position_units", "")))
    if not units_value and legacy_rgbd:
        units_value = "m"
    units = _canonical_units(units_value)
    if units != "m":
        raise ContractError("Trace geometry provenance must declare metric units")
    return anchors


def extract_state_route(
    states: Sequence[Sequence[float] | Mapping[str, Any]],
    *,
    source_anchor: Sequence[float],
    destination_anchor: Sequence[float],
    route_id: str = "route",
    samples: int = 128,
    close_threshold: float = 0.0,
    source_role: str | None = None,
    destination_role: str | None = None,
    coordinate_frame: str | None = None,
    units: str = "m",
    graph_triplet: Sequence[str] | None = None,
) -> TraceRoute:
    """Build a route from actual state8 values and resample by arc length.

    ``close_threshold`` is interpreted in the dataset's native finger-qpos
    units.  The implementation labels only close/reopen finger events; it never
    claims that a closure proves contact or a successful grasp.
    """
    if samples < 2 or len(states) < 2:
        raise ContractError("at least two states and two output samples are required")
    if close_threshold < 0 or not math.isfinite(float(close_threshold)):
        raise ContractError("close_threshold must be finite and non-negative")
    raw: list[tuple[tuple[float, ...], float, str | None]] = []
    for item in states:
        state_item, explicit_event = _state_item_and_event(item)
        values = state8(state_item)
        length = sum((values[j] - raw[-1][0][j]) ** 2 for j in range(3)) ** 0.5 if raw else 0.0
        raw.append((values, length, explicit_event))
    cumulative = [0.0]
    for _, length, _ in raw[1:]:
        cumulative.append(cumulative[-1] + length)
    total = cumulative[-1]
    if total <= 1e-9:
        raise ContractError("teacher route has no spatial movement")
    widths = [sum(raw_i[0][6:8]) / 2.0 for raw_i in raw]
    events: dict[int, str] = {
        index: event for index, (_, _, event) in enumerate(raw) if event is not None
    }
    for i in range(1, len(widths)):
        if i in events:
            continue
        if widths[i] - widths[i - 1] < -abs(close_threshold):
            events[i] = "close"
        elif widths[i] - widths[i - 1] > abs(close_threshold):
            events[i] = "reopen"
    points: list[RoutePoint] = []
    for k in range(samples):
        target = total * k / (samples - 1)
        upper = next((i for i, value in enumerate(cumulative) if value >= target), len(raw) - 1)
        lower = max(0, upper - 1)
        span = cumulative[upper] - cumulative[lower]
        ratio = 0.0 if span <= 1e-12 else (target - cumulative[lower]) / span
        left, right = raw[lower][0], raw[upper][0]
        pos = _lerp(left[:3], right[:3], ratio)
        rot = _lerp(left[3:6], right[3:6], ratio)
        grip_state = tuple(float(v) for v in _lerp(left[6:8], right[6:8], ratio))
        grip = float(sum(grip_state) / 2.0)
        points.append(RoutePoint(pos, rot, grip, target / total, None, grip_state))
    # Assign each raw event to the nearest resampled point.  Looking only at
    # interpolation bounds can silently drop short close/reopen transitions
    # when ``samples`` is small.
    for raw_index, event in events.items():
        event_arc = cumulative[raw_index] / total
        point_index = min(range(len(points)), key=lambda index: abs(points[index].arc - event_arc))
        if points[point_index].event is not None and points[point_index].event != event:
            # Keep both transitions observable when they land on one sample.
            point_index = min(len(points) - 1, point_index + 1)
        point = points[point_index]
        points[point_index] = RoutePoint(point.position, point.rotation, point.gripper,
                                         point.arc, event, point.gripper_state)
    if coordinate_frame is not None and not str(coordinate_frame).strip():
        raise ContractError("coordinate_frame must be non-empty when supplied")
    if _canonical_units(units) != "m":
        raise ContractError("Trace route units must be metric")
    if graph_triplet is not None and len(tuple(graph_triplet)) != 3:
        raise ContractError("graph_triplet must contain source, relation, and destination")
    provenance = {
        "representation": "state_only", "actions": "omitted", "images": "omitted",
        "action_free": True, "gripper_state": "canonical_finger_qpos", "units": "m",
    }
    if source_role is not None:
        if not str(source_role).strip():
            raise ContractError("source_role must be non-empty when supplied")
        provenance["source_role"] = str(source_role)
    if destination_role is not None:
        if not str(destination_role).strip():
            raise ContractError("destination_role must be non-empty when supplied")
        provenance["destination_role"] = str(destination_role)
    if coordinate_frame is not None:
        provenance["coordinate_frame"] = str(coordinate_frame)
    if graph_triplet is not None:
        provenance["graph_triplet"] = tuple(str(item) for item in graph_triplet)
    return TraceRoute(
        tuple(points), tuple(float(v) for v in source_anchor), tuple(float(v) for v in destination_anchor),
        provenance=provenance, route_id=route_id,
        source_role=source_role, destination_role=destination_role,
    )


def _state_item_and_event(item: Sequence[float] | Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
    """Normalize state records while preserving optional route event labels."""
    if isinstance(item, Mapping):
        state_item: Mapping[str, Any] = item
        raw_event = item.get("event", item.get("gripper_event"))
    else:
        # Keep compatibility with the old ``(("state", values),)`` fixture
        # shape, but reject malformed nested records clearly.
        try:
            if len(item) == 1 and isinstance(item[0], (tuple, list)) and len(item[0]) == 2 and item[0][0] == "state":
                state_item = {"state": item[0][1]}
            else:
                state_item = {"state": item}
        except (TypeError, IndexError) as exc:
            raise ContractError("trace state must be a canonical state sequence or mapping") from exc
        raw_event = None
    event = None if raw_event is None else str(raw_event).lower().replace("_", "-")
    if event in {None, ""}:
        return state_item, None
    if event in {"close", "closing", "grasp", "grip-close", "gripper-close"}:
        return state_item, "close"
    if event in {"reopen", "open", "opening", "release", "grip-open", "gripper-open", "gripper-reopen"}:
        return state_item, "reopen"
    raise ContractError(f"unknown trace route event {raw_event!r}")


def _direction_angle(a: Sequence[float], b: Sequence[float]) -> float:
    return math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def _rotate_z(p: Sequence[float], angle: float) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * float(p[0]) - s * float(p[1]), s * float(p[0]) + c * float(p[1]), float(p[2]))


def _rotvec_matrix(v: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    x, y, z = (float(value) for value in v)
    theta = math.sqrt(x * x + y * y + z * z)
    if theta <= 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    a, b, c = x / theta, y / theta, z / theta
    s, co = math.sin(theta), math.cos(theta)
    t = 1.0 - co
    return ((t * a * a + co, t * a * b - s * c, t * a * c + s * b),
            (t * a * b + s * c, t * b * b + co, t * b * c - s * a),
            (t * a * c - s * b, t * b * c + s * a, t * c * c + co))


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3)) for i in range(3))


def _matrix_rotvec(m: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    trace = max(-1.0, min(3.0, sum(float(m[i][i]) for i in range(3))))
    theta = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if theta <= 1e-9:
        return (0.0, 0.0, 0.0)
    scale = theta / (2.0 * math.sin(theta))
    return (scale * (m[2][1] - m[1][2]), scale * (m[0][2] - m[2][0]), scale * (m[1][0] - m[0][1]))


def warp_route(route: TraceRoute, query: GeometryAnchors) -> TraceRoute:
    """Two-anchor source/destination warp with an arc-length transfer blend."""
    if not isinstance(route, TraceRoute):
        raise ContractError("warp_route requires a TraceRoute")
    query = _validate_geometry_anchors(query)
    if _dist(route.source_anchor, route.destination_anchor) <= 1e-9:
        raise ContractError("Trace route anchors must define a non-zero span")
    route_units = route.provenance.get("units", route.provenance.get("unit"))
    route_frame = route.provenance.get("coordinate_frame", route.provenance.get("frame"))
    query_units = _canonical_units(query.provenance.get(
        "units", query.provenance.get("unit", query.provenance.get("position_units", "m"))
    ))
    if route_units is not None and _canonical_units(route_units) != "m":
        raise ContractError("Trace route units must be metric")
    if route_units is not None and _canonical_units(route_units) != query_units:
        raise ContractError("Trace route and geometry units do not match")
    if route_frame is not None and str(route_frame) != query.frame_name:
        raise ContractError("Trace route and geometry coordinate frames do not match")
    reference_angle = _direction_angle(route.source_anchor, route.destination_anchor)
    query_angle = _direction_angle(query.source, query.destination)
    angle = query_angle - reference_angle
    reference_delta = tuple(route.destination_anchor[i] - route.source_anchor[i] for i in range(3))
    query_delta = tuple(query.destination[i] - query.source[i] for i in range(3))
    close_arc = next((point.arc for point in route.points if point.event == "close"), 0.0)
    reopen_arc = next((point.arc for point in route.points if point.event == "reopen"), 1.0)
    if reopen_arc < close_arc:
        reopen_arc = close_arc
    yaw_matrix = ((math.cos(angle), -math.sin(angle), 0.0),
                  (math.sin(angle), math.cos(angle), 0.0), (0.0, 0.0, 1.0))
    warped: list[RoutePoint] = []
    for point in route.points:
        rotated_delta = _rotate_z(tuple(point.position[i] - route.source_anchor[i] for i in range(3)), angle)
        if point.arc <= close_arc:
            alpha = 0.0
        elif point.arc >= reopen_arc:
            alpha = 1.0
        else:
            alpha = (point.arc - close_arc) / max(1e-9, reopen_arc - close_arc)
        first = tuple(query.source[i] + rotated_delta[i] for i in range(3))
        target_delta = tuple(query_delta[i] - _rotate_z(reference_delta, angle)[i] for i in range(3))
        pos = tuple(first[i] + alpha * target_delta[i] for i in range(3))
        rot = _matrix_rotvec(_matmul(yaw_matrix, _rotvec_matrix(point.rotation)))
        if any(not math.isfinite(float(value)) for value in (*pos, *rot)):
            raise ContractError("Trace warp produced non-finite waypoint values")
        warped.append(RoutePoint(pos, rot, point.gripper, point.arc, point.event, point.gripper_state))
    query_kind = str(query.provenance.get("source_kind", query.provenance.get("kind", ""))).lower().replace("-", "_")
    if query_kind not in (_ARROW_RGBD_SOURCE_KINDS | _ARROW_SIMULATOR_SOURCE_KINDS):
        query_kind = "arrow_rgbd"
    warped_provenance = {
        **dict(route.provenance), "warp_provider": query.provider,
        "frame": query.frame_name, "coordinate_frame": query.frame_name,
        "units": query_units, "calibration": query.calibration_revision,
        "source_kind": query_kind, "action_free": True,
    }
    if query_kind in _ARROW_SIMULATOR_SOURCE_KINDS:
        warped_provenance.update({"diagnostic_only": True, "vision_only": False})
    return TraceRoute(tuple(warped), query.source, query.destination,
                      provenance=warped_provenance, route_id=route.route_id,
                      source_role=route.source_role, destination_role=route.destination_role)


WaypointAction = Callable[[ObservationFrame, RoutePoint], Sequence[float]]


def _trace_decision(
    frame: ObservationFrame,
    action: Sequence[float],
    metadata: Mapping[str, Any],
) -> PolicyDecision:
    """Build a Trace decision for either suite coordinator contract."""
    values = validate_action(action)
    if "proposal" not in inspect.signature(PolicyDecision).parameters:
        return PolicyDecision(values, "arrow_trace", frame.digest, teacher_used=False, metadata=metadata)
    from .contracts import ActionProposal, EpisodeSnapshot
    timestep = getattr(frame, "timestep", getattr(frame, "step", 0))
    proposal = ActionProposal(values, policy_id="arrow_trace", timestep=timestep, metadata=metadata)
    snapshot = EpisodeSnapshot(frame.observation, timestep=timestep)
    return PolicyDecision(proposal, snapshot, metadata=metadata,
                          provenance={"policy_id": "arrow_trace", "teacher_used": False})


class TracePolicy:
    policy_id = "arrow_trace"

    def __init__(self, routes: Sequence[TraceRoute], geometry_provider: Callable[[ObservationFrame], GeometryAnchors],
                 waypoint_action: WaypointAction | None = None, *, lookahead: int = 2) -> None:
        if not routes:
            raise ContractError("Trace requires at least one route")
        if waypoint_action is None or not callable(waypoint_action):
            raise ContractError("Trace requires waypoint_action; base pose fallback is not allowed")
        self.routes = tuple(routes)
        if any(not isinstance(route, TraceRoute) for route in self.routes):
            raise ContractError("Trace routes must be TraceRoute instances")
        for route in self.routes:
            _validate_action_free_provenance(route.provenance)
        self.geometry_provider = geometry_provider
        self.waypoint_action = waypoint_action
        self.lookahead = max(0, int(lookahead))
        self._active: TraceRoute | None = None
        self._index = 0
        self.perception_failures = 0
        self._last_perception_failure: str | None = None

    def reset(self) -> None:
        self._active = None
        self._index = 0
        self.perception_failures = 0
        self._last_perception_failure = None

    @property
    def failure_accounting(self) -> Mapping[str, Any]:
        """Return explicit perception failure accounting for rollout logs.

        Trace never falls back to the VLA when geometry is unavailable.  A
        caller may catch the declared ``ContractError`` and terminate the
        rollout, while this immutable-shaped view records why the route was
        not executable.
        """
        return {
            "perception_failures": int(self.perception_failures),
            "last_perception_failure": self._last_perception_failure,
            "fallback_to_base": False,
        }

    def _choose(self, frame: ObservationFrame) -> TraceRoute:
        try:
            if callable(self.geometry_provider):
                anchors = self.geometry_provider(frame)
            elif callable(getattr(self.geometry_provider, "anchors", None)):
                anchors = self.geometry_provider.anchors(frame)  # type: ignore[union-attr]
            else:
                raise ContractError("Trace geometry provider must be callable or expose anchors(frame)")
        except Exception as exc:
            self.perception_failures += 1
            self._last_perception_failure = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ContractError):
                raise ContractError(f"Trace perception unavailable: {exc}") from exc
            raise ContractError("Trace perception unavailable") from exc
        try:
            anchors = _validate_geometry_anchors(anchors)
        except Exception as exc:
            self.perception_failures += 1
            self._last_perception_failure = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ContractError):
                raise ContractError(f"Trace perception unavailable: {exc}") from exc
            raise ContractError("Trace perception unavailable") from exc
        warped = [warp_route(route, anchors) for route in self.routes]
        current = state8(frame.observation)[:3]
        return min(warped, key=lambda route: (_dist(route.points[0].position, current), route.route_id))

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        if self._active is None:
            self._active = self._choose(frame)
            self._index = 0
        route = self._active
        current = state8(frame.observation)[:3]
        previous_index = self._index
        while self._index + 1 < len(route.points) and _dist(current, route.points[self._index + 1].position) < 0.025:
            self._index += 1
        target = route.points[min(len(route.points) - 1, self._index + self.lookahead)]
        # Trace must execute the state-only route's waypoint controller.  A
        # missing hook would silently turn this policy into a frozen-VLA
        # baseline, invalidating the policy comparison, so construction fails
        # closed above and this call is intentionally unconditional.
        guided = validate_action(self.waypoint_action(frame, target))
        action = tuple(0.5 * base.action[i] + 0.5 * guided[i] for i in range(6)) + (guided[6],)
        crossed_events = [point.event for point in route.points[previous_index:self._index + self.lookahead + 1]
                          if point.event is not None]
        event = crossed_events[-1] if crossed_events else target.event
        if event == "close":
            action = action[:6] + (1.0,)
        elif event == "reopen":
            action = action[:6] + (-1.0,)
        return _trace_decision(
            frame, action,
            {"route_id": route.route_id, "route_index": self._index,
             "route_event": event, "route_gripper_state": target.gripper_state,
             "teacher_free": True, "teacher_ignored": teacher is not None,
             "trace_actions_omitted": True,
             "trace_provenance": dict(route.provenance)},
        )

    def commit(self, record: StepRecord) -> None:
        return None
