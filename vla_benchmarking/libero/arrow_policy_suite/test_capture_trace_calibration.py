from dataclasses import dataclass
import json

import pytest

from arrow_policy_suite.capture_trace_calibration import (
    build_calibration_artifact,
    write_create_only,
)
from arrow_policy_suite.contracts import ContractError
from arrow_policy_suite.trace_native_factory import _load_calibration


@dataclass
class Calibration:
    camera_name: str = "agentview"
    world_frame: str = "libero_mujoco_world"
    intrinsic: object = ((100.0, 0.0, 32.0), (0.0, 100.0, 32.0), (0.0, 0.0, 1.0))
    world_from_camera: object = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_captured_calibration_round_trips_native_loader_and_is_create_only(tmp_path):
    payload = build_calibration_artifact(
        Calibration(), revision="simulator-assisted-rgbd-v1", expected_commit="a" * 40,
        task_id=1, seed=1000, init_state_index=1, resolution=256,
    )
    target = tmp_path / "calibration.json"
    digest = write_create_only(target, payload)
    loaded, content_digest = _load_calibration(target)
    assert content_digest == digest
    assert loaded["calibration_hash"] == payload["calibration_hash"]
    assert json.loads(target.read_text())["provenance"]["omitted"] == [
        "rgb", "depth", "actions", "simulator_state"
    ]
    with pytest.raises(ContractError, match="overwrite"):
        write_create_only(target, payload)


def test_captured_calibration_rejects_unsealed_identity():
    with pytest.raises(ContractError, match="Git SHA"):
        build_calibration_artifact(
            Calibration(), revision="simulator-assisted-rgbd-v1", expected_commit="latest",
            task_id=1, seed=1000, init_state_index=1, resolution=256,
        )
