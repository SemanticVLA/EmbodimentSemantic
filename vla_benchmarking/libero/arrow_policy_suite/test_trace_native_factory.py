from __future__ import annotations

import json

import numpy as np
import pytest

from arrow_policy_suite.contracts import ActionProposal, ContractError, ObservationFrame
from arrow_policy_suite.rgbd_geometry import ArrowPixelEndpoints, ArrowRGBDObservation
from arrow_policy_suite.trace_native_factory import (
    TRACE_ROUTE_ARTIFACT_SCHEMA,
    build_trace_native_fields,
)


def _write_artifacts(tmp_path):
    route_path = tmp_path / "trace_routes.json"
    route_path.write_text(json.dumps({
        "schema": TRACE_ROUTE_ARTIFACT_SCHEMA,
        "create_only": True,
        "routes": [{
            "route_id": "cup-to-bowl",
            "source_role": "cup",
            "destination_role": "bowl",
            "graph_triplet": ["cup", "inside", "bowl"],
            "coordinate_frame": "world",
            "units": "m",
            "source_anchor": [0.0, 0.0, 0.0],
            "destination_anchor": [1.0, 0.0, 0.0],
            "states": [
                {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
                {"state": [0.2, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
            ],
        }],
    }), encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "revision": "calib-v1",
        "frame_name": "world",
        "intrinsics": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
        "world_from_camera": np.eye(4).tolist(),
    }), encoding="utf-8")
    return route_path, calibration_path


def _frame():
    return ObservationFrame(
        {"state": [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1]},
        metadata={"timestamp_s": 10.0},
    )


def test_trace_loader_builds_real_policy_and_native_fields(tmp_path):
    route_path, calibration_path = _write_artifacts(tmp_path)

    def capture(_frame):
        return ArrowRGBDObservation(
            np.zeros((32, 32, 3), dtype=np.uint8), np.ones((32, 32), dtype=np.float32),
            np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]),
            np.eye(4), "world", "calib-v1", 10.0,
        )

    def endpoints(_frame, _capture):
        return ArrowPixelEndpoints((10.0, 16.0), (22.0, 16.0), "cup", "bowl")

    fields = build_trace_native_fields(
        route_artifact=route_path, calibration_artifact=calibration_path,
        geometry_variant="rgbd", capture_fn=capture, endpoint_fn=endpoints,
        graph_context_fn=lambda _frame: {"triplet": ["cup", "inside", "bowl"]},
        graph_context_revision="graph-v1",
    )
    assert fields["policy_id"] == "arrow_trace"
    assert fields["policy"].policy_id == "arrow_trace"
    assert fields["graph_context_revision"] == "graph-v1"
    assert fields["trace_geometry_variant"] == "rgbd"
    decision = fields["policy"].decide(_frame(), ActionProposal((0.0,) * 7, "vla", 0), None)
    assert len(decision.action) == 7
    assert all(-1.0 <= value <= 1.0 for value in decision.action)


def test_trace_loader_fails_closed_on_missing_context_or_route_geometry(tmp_path):
    route_path, calibration_path = _write_artifacts(tmp_path)
    with pytest.raises(ContractError, match="graph_context_fn"):
        build_trace_native_fields(
            route_artifact=route_path, calibration_artifact=calibration_path,
            geometry_variant="rgbd", capture_fn=lambda _frame: {}, endpoint_fn=lambda *_args: {},
            graph_context_fn=None, graph_context_revision="graph-v1",
        )
    with pytest.raises(ContractError, match="route artifact does not exist"):
        build_trace_native_fields(
            route_artifact=tmp_path / "missing.json", calibration_artifact=calibration_path,
            geometry_variant="rgbd", capture_fn=lambda _frame: {}, endpoint_fn=lambda *_args: {},
            graph_context_fn=lambda _frame: {}, graph_context_revision="graph-v1",
        )


def test_trace_loader_rejects_route_with_actions(tmp_path):
    route_path, calibration_path = _write_artifacts(tmp_path)
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    payload["routes"][0]["teacher_actions"] = [[0.0] * 7]
    route_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="omit actions/images"):
        build_trace_native_fields(
            route_artifact=route_path, calibration_artifact=calibration_path,
            geometry_variant="rgbd", capture_fn=lambda _frame: {}, endpoint_fn=lambda *_args: {},
            graph_context_fn=lambda _frame: {}, graph_context_revision="graph-v1",
        )
