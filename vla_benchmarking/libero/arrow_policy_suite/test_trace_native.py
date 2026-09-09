from __future__ import annotations

import numpy as np
import pytest

from arrow_policy_suite.contracts import ActionProposal, ContractError, GeometryAnchors, ObservationFrame
from arrow_policy_suite.rgbd_geometry import (
    ArrowPixelEndpoints,
    ArrowRGBDGeometryProvider,
    ArrowRGBDObservation,
    SimulatorAssistedRGBDGeometryProvider,
    TraceSimulatorAssistedArrowGeometryProvider,
)
from arrow_policy_suite.trace import TracePolicy, extract_state_route, warp_route
from arrow_policy_suite.waypoint_controller import WaypointController, WaypointControllerConfig


def _frame(timestamp: float = 10.0) -> ObservationFrame:
    return ObservationFrame(
        {"state": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1]},
        timestep=0,
        metadata={"timestamp_s": timestamp},
    )


def _capture(timestamp: float = 10.0, depth: object | None = None) -> ArrowRGBDObservation:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth_image = np.ones((32, 32), dtype=np.float32) if depth is None else depth
    return ArrowRGBDObservation(
        rgb, depth_image,
        np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]),
        np.eye(4), "world", "calib-v1", timestamp,
    )


def test_strict_rgbd_provider_deprojects_endpoints_and_freezes_provenance():
    provider = ArrowRGBDGeometryProvider(
        lambda _frame: _capture(),
        lambda _frame, _capture: ArrowPixelEndpoints((10.0, 10.0), (22.0, 10.0), "cup", "bowl"),
        expected_calibration_revision="calib-v1",
    )
    anchors = provider(_frame())
    assert anchors.provider == "arrow_rgbd"
    assert anchors.provenance["source_kind"] == "arrow_rgbd"
    assert anchors.provenance["vision_only"] is True
    assert anchors.provenance["provider_hash"] == provider.provider_hash
    assert anchors.provenance["calibration_hash"]
    assert anchors.provenance["source_role"] == "cup"
    assert anchors.provenance["destination_role"] == "bowl"
    assert anchors.source != anchors.destination


def test_strict_rgbd_provider_rejects_stale_and_missing_depth():
    provider = ArrowRGBDGeometryProvider(
        lambda _frame: _capture(0.0),
        lambda _frame, _capture: {"source_xy": (10.0, 10.0), "destination_xy": (22.0, 10.0)},
        max_frame_age_s=0.1,
    )
    with pytest.raises(ContractError, match="stale"):
        provider(_frame(10.0))

    invalid = np.zeros((32, 32), dtype=np.float32)
    bad_provider = ArrowRGBDGeometryProvider(
        lambda _frame: _capture(depth=invalid),
        lambda _frame, _capture: {"source_xy": (10.0, 10.0), "destination_xy": (22.0, 10.0)},
    )
    with pytest.raises(ContractError, match="valid metric"):
        bad_provider(_frame())


def test_simulator_assisted_provider_is_explicit_diagnostic_and_warps():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ],
        source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0),
        samples=2, source_role="cup", destination_role="bowl",
    )
    provider = TraceSimulatorAssistedArrowGeometryProvider(
        lambda _frame: {"source": (2.0, 2.0, 0.0), "destination": (3.0, 2.0, 0.0)},
    )
    warped = warp_route(route, provider(_frame()))
    assert warped.provenance["source_kind"] == "trace_simulator_assisted_arrow"
    assert warped.provenance["diagnostic_only"] is True
    assert warped.provenance["vision_only"] is False
    assert warped.source_role == "cup"
    assert warped.destination_role == "bowl"


def test_simulator_assisted_rgbd_crosses_trace_policy_boundary_as_diagnostic():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ],
        source_anchor=(0.0, 0.0, 0.0), destination_anchor=(1.0, 0.0, 0.0),
        samples=2, source_role="cup", destination_role="bowl",
    )
    provider = SimulatorAssistedRGBDGeometryProvider(
        lambda _frame: _capture(),
        lambda _frame, _capture: {
            "source_xy": (10.0, 10.0),
            "destination_xy": (22.0, 10.0),
            "provenance": {"simulator_bbox": True},
        },
        expected_calibration_revision="calib-v1",
    )
    policy = TracePolicy(
        (route,), provider,
        waypoint_action=lambda _frame, _point: (0.0,) * 7,
    )
    frame = _frame()
    decision = policy.decide(frame, ActionProposal((0.0,) * 7, "vla", 0), None)
    assert decision.policy_id == "arrow_trace"
    assert decision.metadata["trace_provenance"]["warp_provider"] == "simulator_assisted_rgbd"
    assert decision.metadata["trace_provenance"]["diagnostic_only"] is True


def test_waypoint_controller_persists_close_until_reopen():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.0, 0.0], "event": "close"},
            {"state": [0.4, 0.0, 0.2, 0, 0, 0, 0.1, 0.1], "event": "reopen"},
        ],
        source_anchor=(0, 0, 0), destination_anchor=(1, 0, 0), samples=3,
    )
    controller = WaypointController()
    frame = _frame()
    open_action = controller(frame, route.points[0])
    close_action = controller(frame, route.points[1])
    hold_action = controller(frame, route.points[1])
    reopen_action = controller(frame, route.points[2])
    assert open_action[6] == -1.0
    assert close_action[6] == hold_action[6] == 1.0
    assert reopen_action[6] == -1.0
    assert controller.events == ("close", "reopen")
    assert all(-1.0 <= value <= 1.0 for value in hold_action)


def test_waypoint_controller_normalizes_wide_metric_delta_and_preserves_gripper():
    # A safety bound wider than one normalization scale is valid configuration
    # but must saturate the continuous command at the action boundary.
    controller = WaypointController(WaypointControllerConfig(
        translation_scale_m=0.05,
        max_translation_m=0.10,
    ))
    frame = _frame()
    point = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ],
        source_anchor=(0.0, 0.0, 0.0),
        destination_anchor=(1.0, 0.0, 0.0),
        samples=2,
    ).points[1]
    action = controller(frame, point)
    assert len(action) == 7
    assert all(-1.0 <= value <= 1.0 for value in action)
    assert action[0] == 1.0
    assert action[1:6] == (0.0,) * 5
    assert action[6] == -1.0


def test_trace_generated_proposal_is_normalized_and_directionally_correct():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ],
        source_anchor=(0.0, 0.0, 0.0),
        destination_anchor=(1.0, 0.0, 0.0),
        samples=2,
    )
    provider = lambda _frame: GeometryAnchors(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "world", "calib-v1", "arrow_rgbd",
        {"source_kind": "arrow_rgbd", "coordinate_frame": "world", "units": "m"},
    )
    controller = WaypointController(WaypointControllerConfig(
        translation_scale_m=0.05,
        max_translation_m=0.10,
    ))
    policy = TracePolicy((route,), provider, waypoint_action=controller, lookahead=1)
    decision = policy.decide(_frame(), ActionProposal((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0), "vla", 0), None)

    assert len(decision.action) == 7
    assert all(-1.0 <= value <= 1.0 for value in decision.action)
    assert decision.action[0] == 1.0
    assert decision.action[0] > decision.action[1]
    assert decision.action[6] == -1.0


def test_trace_counts_perception_unavailable_without_base_fallback():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ], source_anchor=(0, 0, 0), destination_anchor=(1, 0, 0), samples=2,
    )
    policy = TracePolicy((route,), lambda _frame: (_ for _ in ()).throw(RuntimeError("no depth")),
                         waypoint_action=lambda _frame, _point: (0.0,) * 7)
    with pytest.raises(ContractError, match="perception unavailable"):
        policy.decide(_frame(), ActionProposal((0.0,) * 7, "vla", 0), None)
    assert policy.failure_accounting["perception_failures"] == 1
    assert policy.failure_accounting["fallback_to_base"] is False


def test_trace_resampling_clamps_only_endpoint_roundoff_for_realistic_states():
    route = extract_state_route(
        [
            {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.0, 0.0]},
            {"state": [0.3, 0.0, 0.2, 0, 0, 0, 0.0, 0.0]},
            {"state": [0.8, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        ],
        source_anchor=(0.2, 0.0, 0.2), destination_anchor=(0.8, 0.0, 0.2), samples=4,
    )
    assert route.points[0].arc == 0.0
    assert route.points[-1].arc == 1.0
    assert all(0.0 <= point.arc <= 1.0 for point in route.points)
