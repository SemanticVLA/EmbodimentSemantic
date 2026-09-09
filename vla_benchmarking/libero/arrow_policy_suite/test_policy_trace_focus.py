from __future__ import annotations

import pytest

from arrow_policy_suite.contracts import ContractError, GeometryAnchors, ObservationFrame, ActionProposal
from arrow_policy_suite.fast import GraphFastCorrector
from arrow_policy_suite.policies import make_policy
from arrow_policy_suite.trace import RoutePoint, TracePolicy, TraceRoute, extract_state_route, warp_route


def _states():
    return [
        {"state": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1]},
        {"state": [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], "event": "close"},
        {"state": [0.8, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1], "gripper_event": "reopen"},
    ]


def test_trace_preserves_finger_state_and_explicit_events_after_resampling():
    route = extract_state_route(
        _states(), source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0), samples=5,
    )
    assert any(point.event == "close" for point in route.points)
    assert any(point.event == "reopen" for point in route.points)
    assert all(point.gripper_state is not None for point in route.points)
    assert route.provenance["actions"] == "omitted"
    assert route.provenance["action_free"] is True


def test_trace_rejects_action_bearing_nested_provenance():
    points = (
        RoutePoint((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1, 0.0),
        RoutePoint((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1, 1.0),
    )
    with pytest.raises(ContractError, match="omit actions"):
        TraceRoute(
            points, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            provenance={"source": {"actions": [[0.0] * 7]}},
        )


def test_trace_rejects_negative_event_threshold():
    with pytest.raises(ContractError, match="close_threshold"):
        extract_state_route(
            _states(), source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0),
            close_threshold=-1.0,
        )


def test_trace_requires_typed_arrow_rgbd_geometry_and_explicit_units():
    route = extract_state_route(
        _states(), source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0), samples=4,
    )
    bad = GeometryAnchors(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "world", "calibration-1", "arrow_rgbd",
        {"source_kind": "mujoco", "coordinate_frame": "world", "units": "m"},
    )
    with pytest.raises(ContractError, match="MuJoCo|typed Arrow"):
        warp_route(route, bad)

    missing_units = GeometryAnchors(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "world", "calibration-1", "arrow_rgbd",
        {"source_kind": "arrow_rgbd", "coordinate_frame": "world"},
    )
    with pytest.raises(ContractError, match="metric units"):
        warp_route(route, missing_units)


def test_trace_requires_waypoint_action_to_avoid_base_pose_fallback():
    route = extract_state_route(
        _states(), source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0), samples=4,
    )
    provider = lambda _frame: GeometryAnchors(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "world", "calibration-1", "arrow_rgbd",
        {"source_kind": "arrow_rgbd", "coordinate_frame": "world", "units": "m"},
    )
    with pytest.raises(ContractError, match="waypoint_action"):
        TracePolicy((route,), provider)

    policy = TracePolicy((route,), provider, waypoint_action=lambda _frame, _point: (0.0,) * 7)
    frame = ObservationFrame({"state": [0.0] * 8}, 0)
    base = ActionProposal((0.0,) * 7, "vla", 0)
    decision = policy.decide(frame, base, None)
    assert decision.action[:6] == (0.0,) * 6
    assert decision.action[6] == 1.0


def test_fast_policy_factory_preserves_graph_hook():
    corrector = GraphFastCorrector(lambda _frame, _role: (0.0,) * 32)
    graph_fn = lambda frame: frame
    policy = make_policy("arrow_fast", corrector=corrector, graph_fn=graph_fn)
    assert policy.corrector is corrector
    assert policy.graph_fn is graph_fn
