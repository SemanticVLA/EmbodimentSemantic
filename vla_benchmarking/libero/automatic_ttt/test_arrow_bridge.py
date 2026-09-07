from __future__ import annotations

import pytest

from .arrow_bridge import ArrowCanaryBridge
from .contracts import ContractError, EpisodeSpec, SourceState, TeacherRecoveryRequest


def _bridge() -> ArrowCanaryBridge:
    return ArrowCanaryBridge(
        worker=object(),
        episode_runner=lambda **_kwargs: {},
        source_uv=(1, 2),
        output_root=".",
        transition_getter=lambda _raw: [],
        allow_stale_geometry_for_tests=True,
    )


def _request() -> TeacherRecoveryRequest:
    return TeacherRecoveryRequest(
        episode=EpisodeSpec("episode-0", 0, 1000, "pick up object", "openvla"),
        source_state=SourceState.SOURCE_UNHELD,
        observation={"state": [0]},
        vla_history=(),
        remaining_budget=10,
    )


def test_bridge_uses_nested_evaluator_success_and_controller_status():
    success, status = _bridge()._controller_verdict(
        {"success": False, "status": "selected", "final_result": {"status": "placed", "evaluator_success": True}}
    )
    assert success is True
    assert status == "placed"


def test_bridge_does_not_infer_success_from_motion_status():
    success, status = _bridge()._controller_verdict(
        {"success": True, "final_result": {"status": "task_failure", "evaluator_success": False}}
    )
    assert success is False
    assert status == "task_failure"


def test_bridge_requires_fresh_capture_provenance_after_vla_motion():
    bridge = _bridge()
    with pytest.raises(ContractError, match="capture_provenance"):
        bridge._fresh_geometry({"source_uv": (1, 2), "destination_uv": None, "capture_provenance": {}}, _request())
    geometry = bridge._fresh_geometry(
        {
            "source_uv": (3, 4),
            "destination_uv": (5, 6),
            "capture_provenance": {
                "timestamp": 1.0,
                "camera_id": "agentview",
                "resolution": 256,
                "calibration_revision": "calib-1",
                "captured_after_timestep": 0,
            },
        },
        _request(),
    )
    assert geometry["source_uv"] == (3.0, 4.0)
