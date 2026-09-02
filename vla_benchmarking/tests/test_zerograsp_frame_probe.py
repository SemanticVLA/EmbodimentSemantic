from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sanity_checks.probe_panda_grip_site_frame import RZ_MINUS_90, probe_grip_site_frame


class FakeModel:
    def site_name2id(self, name: str) -> int:
        assert name == "grip_site"
        return 0

    def body_name2id(self, name: str) -> int:
        assert name == "right_hand"
        return 0


class NoMotionSim:
    def __init__(self, site_rotation: np.ndarray):
        self.model = FakeModel()
        self.data = SimpleNamespace(
            site_xmat=np.asarray([site_rotation], dtype=float),
            body_xmat=np.asarray([np.eye(3)], dtype=float),
            site_xpos=np.asarray([[0.10, 0.20, 0.30]], dtype=float),
            body_xpos=np.asarray([[0.09, 0.20, 0.30]], dtype=float),
        )

    def step(self, *_args, **_kwargs):  # pragma: no cover - catches accidental motion
        raise AssertionError("frame probe must not step the simulator")


class PrefixedModel:
    def site_name2id(self, name: str) -> int:
        if name != "gripper0_grip_site":
            raise KeyError(name)
        return 0

    def body_name2id(self, name: str) -> int:
        if name != "robot0_right_hand":
            raise KeyError(name)
        return 0


def prefixed_env() -> SimpleNamespace:
    sim = SimpleNamespace(
        model=PrefixedModel(),
        data=SimpleNamespace(
            site_xmat=np.asarray([RZ_MINUS_90], dtype=float),
            body_xmat=np.asarray([np.eye(3)], dtype=float),
            site_xpos=np.asarray([[0.10, 0.20, 0.30]], dtype=float),
            body_xpos=np.asarray([[0.09, 0.20, 0.30]], dtype=float),
        ),
    )
    return SimpleNamespace(sim=sim)


def test_probe_compares_grip_site_to_right_hand_rz_minus_90_without_motion():
    calibration = {"camera": "agentview", "resolution": [256, 256]}
    result = probe_grip_site_frame(NoMotionSim(RZ_MINUS_90), calibration_input=calibration)

    assert result["passed"] is True
    assert result["pass"] is True
    assert result["frame_contract"]["expected_rotation"] == "R_right_hand @ Rz(-90deg)"
    assert result["expected_axes"] == result["observed_axes"]
    assert np.allclose(result["observed_body_to_site_rotation_matrix"], RZ_MINUS_90)
    assert result["angular"]["error_deg"] == 0.0
    assert result["position"]["site_minus_right_hand_m"] == [0.010000000000000009, 0.0, 0.0]
    assert result["position"]["status"] == "observed_metadata_only"
    assert result["position"]["center_to_tip_residual_verified"] is False
    encoded = json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert result["calibration_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_probe_fails_when_observed_site_rotation_exceeds_tolerance():
    rotation = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    result = probe_grip_site_frame(NoMotionSim(rotation), angular_tolerance_deg=0.5)
    assert result["passed"] is False
    assert result["angular"]["error_deg"] > 0.5


def test_probe_rejects_scaled_rotation_matrix():
    with pytest.raises(ValueError, match="orthonormal"):
        probe_grip_site_frame(NoMotionSim(2.0 * RZ_MINUS_90))


def test_probe_rejects_reflection_matrix():
    reflected = RZ_MINUS_90.copy()
    reflected[:, 2] *= -1.0
    with pytest.raises(ValueError, match=r"determinant \+1"):
        probe_grip_site_frame(NoMotionSim(reflected))


def test_probe_auto_resolves_robot0_names_and_unwraps_environment_without_motion():
    result = probe_grip_site_frame(prefixed_env())

    assert result["passed"] is True
    assert result["resolved_site_name"] == "gripper0_grip_site"
    assert result["resolved_body_name"] == "robot0_right_hand"
    assert result["frame_contract"]["site_frame"] == "gripper0_grip_site"
    assert result["frame_contract"]["body_frame"] == "robot0_right_hand"


def test_probe_honors_explicit_frame_name_over_auto_resolution():
    sim = NoMotionSim(RZ_MINUS_90)
    result = probe_grip_site_frame(sim, site_name="grip_site", body_name="right_hand")
    assert result["resolved_site_name"] == "grip_site"
    assert result["resolved_body_name"] == "right_hand"


def test_probe_accepts_flattened_mujoco_rotation_and_records_file_hash(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text('{"revision":"probe-1"}\n', encoding="utf-8")
    result = probe_grip_site_frame(NoMotionSim(RZ_MINUS_90.reshape(-1)), calibration_input=calibration_path)

    assert result["passed"] is True
    assert result["calibration"]["source"] == "file"
    assert result["calibration_input"] == calibration_path.resolve().as_posix()
    assert result["calibration_sha256"] == hashlib.sha256(calibration_path.read_bytes()).hexdigest()
