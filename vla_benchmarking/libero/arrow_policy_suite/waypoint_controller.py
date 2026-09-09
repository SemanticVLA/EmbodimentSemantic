"""Action-producing controller shared by Trace and its path-only ablation.

The controller is deliberately independent of a VLA implementation.  It maps
the canonical state8 observation and one warped :class:`RoutePoint` to the
same normalized seven-value OSC vector used by LIBERO (xyz delta, rotation
vector delta, gripper).  A single instance owns the gripper latch so route
events cannot be lost between waypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .contracts import ContractError, ObservationFrame, clip_action, state8
from .trace import RoutePoint


@dataclass(frozen=True)
class WaypointControllerConfig:
    """Normalization and safety constants for normalized OSC actions."""

    translation_scale_m: float = 0.05
    rotation_scale_rad: float = 0.50
    gripper_open: float = -1.0
    gripper_closed: float = 1.0
    max_translation_m: float = 0.05
    max_rotation_rad: float = 0.50

    def __post_init__(self) -> None:
        for name in ("translation_scale_m", "rotation_scale_rad", "max_translation_m", "max_rotation_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be finite and positive")
        for name in ("gripper_open", "gripper_closed"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ContractError(f"{name} must lie in [-1, 1]")
        if self.gripper_open == self.gripper_closed:
            raise ContractError("gripper_open and gripper_closed must differ")


class WaypointController:
    """Stateful route waypoint -> normalized OSC action controller.

    ``close`` and ``reopen`` are persistent events.  After a close event the
    controller keeps commanding closed until a later reopen event, including
    while the arm is moving through intermediate route points.  The route's
    two finger-qpos values are retained for analysis but are not mistaken for
    normalized action values.
    """

    def __init__(self, config: WaypointControllerConfig | None = None) -> None:
        self.config = config or WaypointControllerConfig()
        self._gripper_closed = False
        self._events: list[str] = []
        self._last_route_id: str | None = None
        self._last_action: tuple[float, ...] | None = None
        self._last_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self._gripper_closed = False
        self._events.clear()
        self._last_route_id = None
        self._last_action = None
        self._last_metadata = {}

    @property
    def gripper_closed(self) -> bool:
        return self._gripper_closed

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    @property
    def last_metadata(self) -> Mapping[str, Any]:
        return dict(self._last_metadata)

    def _apply_event(self, event: str | None) -> None:
        if event is None:
            return
        if event not in {"close", "reopen"}:
            raise ContractError(f"unknown Trace waypoint event {event!r}")
        # Duplicate events are idempotent.  This matters after resampling: a
        # short event can be represented by adjacent points without creating
        # an impossible open/close transition.
        if event == "close":
            if not self._gripper_closed:
                self._gripper_closed = True
                self._events.append(event)
        elif self._gripper_closed:
            self._gripper_closed = False
            self._events.append(event)

    def __call__(self, frame: ObservationFrame, waypoint: RoutePoint) -> tuple[float, ...]:
        if not isinstance(frame, ObservationFrame):
            raise ContractError("WaypointController requires an ObservationFrame")
        if not isinstance(waypoint, RoutePoint):
            raise ContractError("WaypointController requires a RoutePoint")
        current = state8(frame.observation)
        self._apply_event(waypoint.event)
        cfg = self.config
        translation = tuple(
            max(-cfg.max_translation_m, min(cfg.max_translation_m, float(waypoint.position[i]) - current[i]))
            / cfg.translation_scale_m
            for i in range(3)
        )
        rotation = tuple(
            max(-cfg.max_rotation_rad, min(cfg.max_rotation_rad, float(waypoint.rotation[i]) - current[3 + i]))
            / cfg.rotation_scale_rad
            for i in range(3)
        )
        gripper = cfg.gripper_closed if self._gripper_closed else cfg.gripper_open
        # The controller owns the conversion from metric pose deltas to the
        # canonical normalized action.  ``max_*`` is an independent safety
        # bound and may be wider than the configured normalization scale, so
        # clip only after conversion.  ``clip_action`` still rejects malformed
        # dimensions and non-finite values; the gripper value is already a
        # validated semantic command in [-1, 1] and is left unchanged.
        action = clip_action((*translation, *rotation, gripper))
        self._last_route_id = None
        self._last_action = action
        self._last_metadata = {
            "controller": "trace_waypoint_osc_v1",
            "translation_frame": "route/world_frame",
            "translation_units": "m",
            "rotation_representation": "rotation_vector_rad",
            "action_semantics": "normalized_osc_xyz_rotvec_gripper",
            "lookahead": "caller_selected",
            "gripper_state": "closed" if self._gripper_closed else "open",
            "route_event": waypoint.event,
            "gripper_event_count": len(self._events),
        }
        return action


def waypoint_action(frame: ObservationFrame, waypoint: RoutePoint, *, controller: WaypointController) -> tuple[float, ...]:
    """Small dependency-injection helper for ``TracePolicy``."""
    if not isinstance(controller, WaypointController):
        raise ContractError("waypoint_action controller must be a WaypointController")
    return controller(frame, waypoint)


__all__ = ["WaypointControllerConfig", "WaypointController", "waypoint_action"]
