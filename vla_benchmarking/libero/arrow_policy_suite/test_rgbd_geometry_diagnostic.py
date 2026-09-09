from __future__ import annotations

import numpy as np
import pytest

from arrow_policy_suite.contracts import ContractError, ObservationFrame
from arrow_policy_suite.rgbd_geometry import (
    ArrowRGBDGeometryProvider,
    ArrowRGBDObservation,
    SimulatorAssistedRGBDGeometryProvider,
)


def _frame(timestamp: float | None = 100.0) -> ObservationFrame:
    metadata = {} if timestamp is None else {"timestamp_s": timestamp}
    return ObservationFrame({"state": [0.0] * 8}, metadata=metadata)


def _capture(*, frame_name: str = "world", revision: str = "calibration-1", timestamp: float = 100.0):
    return ArrowRGBDObservation(
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth_m=np.full((5, 5), 2.0, dtype=np.float32),
        intrinsics=np.array(((1.0, 0.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.0, 1.0))),
        world_from_camera=np.eye(4),
        frame_name=frame_name,
        calibration_revision=revision,
        timestamp_s=timestamp,
        metadata={"depth_units": "meters"},
    )


def _endpoints(_frame, _capture):
    return {
        "source_xy": (1.0, 2.0),
        "destination_xy": (3.0, 2.0),
        "source_role": "source_object",
        "destination_role": "destination_object",
        "provenance": {
            "source": "simulator_bbox_arrow_overlay",
            "bbox_source": "simulator_bbox",
            "arrow_source": "simulator_arrow_annotation",
        },
    }


def test_simulator_assisted_rgbd_deprojects_and_marks_diagnostic_provenance():
    provider = SimulatorAssistedRGBDGeometryProvider(
        lambda _frame: _capture(), _endpoints,
        expected_frame_name="world",
        expected_calibration_revision="calibration-1",
        clock=lambda: 100.0,
    )
    anchors = provider.anchors(_frame())
    assert anchors.provider == "simulator_assisted_rgbd"
    assert anchors.source == pytest.approx((-2.0, 0.0, 2.0))
    assert anchors.destination == pytest.approx((2.0, 0.0, 2.0))
    assert anchors.provenance["diagnostic_only"] is True
    assert anchors.provenance["vision_only"] is False
    assert anchors.provenance["privileged_source"] == "simulator_bbox_or_arrow"
    assert anchors.provenance["provider_revision"] == provider.provider_revision
    assert anchors.provenance["provider_hash"] == provider.provider_hash
    assert anchors.provenance["endpoint_provenance"]["bbox_source"] == "simulator_bbox"
    assert anchors.provenance["units"] == "m"


def test_simulator_assisted_rgbd_rejects_frame_units_calibration_and_stale_capture():
    provider = SimulatorAssistedRGBDGeometryProvider(
        lambda _frame: _capture(frame_name="camera", revision="calibration-1"), _endpoints,
        expected_frame_name="world", expected_calibration_revision="calibration-1", clock=lambda: 100.0,
    )
    with pytest.raises(ContractError, match="frame mismatch"):
        provider.anchors(_frame())

    bad_revision = SimulatorAssistedRGBDGeometryProvider(
        lambda _frame: _capture(revision="calibration-2"), _endpoints,
        expected_frame_name="world", expected_calibration_revision="calibration-1", clock=lambda: 100.0,
    )
    with pytest.raises(ContractError, match="calibration revision"):
        bad_revision.anchors(_frame())

    stale = SimulatorAssistedRGBDGeometryProvider(
        lambda _frame: _capture(timestamp=98.0), _endpoints,
        expected_frame_name="world", expected_calibration_revision="calibration-1",
        max_frame_age_s=0.25, clock=lambda: 100.0,
    )
    with pytest.raises(ContractError, match="stale"):
        stale.anchors(_frame())


def test_strict_rgbd_provider_still_rejects_privileged_capture_and_endpoint_metadata():
    strict_capture = {
        "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
        "depth_m": np.full((5, 5), 2.0, dtype=np.float32),
        "intrinsics": np.array(((1.0, 0.0, 2.0), (0.0, 1.0, 2.0), (0.0, 0.0, 1.0))),
        "world_from_camera": np.eye(4), "frame_name": "world",
        "calibration_revision": "calibration-1", "timestamp_s": 100.0,
        "metadata": {"simulator_bbox": "forbidden"},
    }
    strict = ArrowRGBDGeometryProvider(lambda _frame: strict_capture, _endpoints, clock=lambda: 100.0)
    with pytest.raises(ContractError, match="privileged"):
        strict.anchors(_frame())

    strict_endpoint = ArrowRGBDGeometryProvider(
        lambda _frame: _capture(),
        lambda _frame, _capture: {
            "source_xy": (1.0, 2.0), "destination_xy": (3.0, 2.0),
            "provenance": {"bbox_source": "simulator_bbox"},
        },
        clock=lambda: 100.0,
    )
    with pytest.raises(ContractError, match="privileged"):
        strict_endpoint.anchors(_frame())


def test_simulator_assisted_rgbd_requires_synchronized_capture():
    with pytest.raises(ContractError, match="synchronized"):
        ArrowRGBDObservation(
            rgb=np.zeros((5, 5, 3), dtype=np.uint8), depth_m=np.full((5, 5), 2.0),
            intrinsics=np.eye(3), world_from_camera=np.eye(4), frame_name="world",
            calibration_revision="calibration-1", timestamp_s=100.0,
            rgb_timestamp_s=100.0, depth_timestamp_s=100.01,
        )

