"""Focused contracts for the opt-in experimental release-height treatment."""

from __future__ import annotations

import numpy as np
import pytest

from vla_benchmarking.evaluation import run_arrow_pick_place_eval as runner


class _Env:
    def __init__(self):
        self.actions = []
        self._arrow_settle_diagnostics = {"settled": True}

    def step(self, action):
        self.actions.append(np.asarray(action))
        return {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "robot0_gripper_qpos": np.zeros(2),
        }, 0.0, False, {}


def _capture():
    calibration = runner.CameraCalibration(
        "agentview", 256, 256,
        [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
        "robot0_gripper_qpos": np.zeros(2),
    }
    return runner.CapturedRGBD(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        calibration, observation,
    )


def _candidate():
    return {
        "candidate_id": "c0",
        "grip_site_world_m": (0.0, 0.0, 1.0),
        "rotation_world_grip_site": np.eye(3),
        "required_aperture_m": 0.04,
        "pregrasp_world_m": (0.0, 0.0, 1.05),
    }


def _patch_controller(monkeypatch, *, high_release=False):
    monkeypatch.setattr(
        runner, "decode_arrow",
        lambda **_kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(runner, "deproject_endpoint", lambda point, depth, K, T: np.asarray((float(point[0]) / 100.0, float(point[1]) / 100.0, 1.0)))
    release_z = 1.79 if high_release else 0.70
    base = np.asarray(((0.0, 0.0, 0.5), (0.0, 0.0, 0.8), (0.0, 0.0, 0.9),
                       (0.1, 0.2, 0.8), (0.1, 0.2, release_z), (0.1, 0.2, release_z - 0.1)))
    monkeypatch.setattr(runner, "build_bowl_waypoints", lambda *_args: base.copy())
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))


def test_release_height_zero_preserves_experimental_targets_and_orientation(monkeypatch, tmp_path):
    _patch_controller(monkeypatch)
    audit = runner.run_episode(
        env=_Env(), task_id=0, seed=1000, output_dir=tmp_path, arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_capture(), dry_run=True, experimental_candidate=_candidate(),
        experimental_eef_orientation_transform=np.eye(3), experimental_gripper_opening_m=0.05,
        experimental_release_height_offset_m=0.0,
    )
    treatment = audit["experimental_grasp"]["audit"]["release_height_treatment"]
    assert treatment["offset_m"] == 0.0
    assert treatment["nominal_release_world_m"] == treatment["shifted_release_world_m"]
    assert treatment["nominal_retreat_world_m"] == treatment["shifted_retreat_world_m"]
    assert np.allclose(audit["waypoints_world_m"]["descend_place"]["orientation"], np.eye(3))


def test_release_height_plus_20mm_shifts_only_release_and_retreat(monkeypatch, tmp_path):
    _patch_controller(monkeypatch)
    audit = runner.run_episode(
        env=_Env(), task_id=0, seed=1000, output_dir=tmp_path, arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_capture(), dry_run=True, experimental_candidate=_candidate(),
        experimental_eef_orientation_transform=np.eye(3), experimental_gripper_opening_m=0.05,
        experimental_release_height_offset_m=0.020,
    )
    treatment = audit["experimental_grasp"]["audit"]["release_height_treatment"]
    assert treatment["offset_m"] == pytest.approx(0.020)
    assert np.asarray(treatment["shifted_release_world_m"])[2] - np.asarray(treatment["nominal_release_world_m"])[2] == pytest.approx(0.020)
    assert np.asarray(treatment["shifted_retreat_world_m"])[2] - np.asarray(treatment["nominal_retreat_world_m"])[2] == pytest.approx(0.020)
    assert treatment["rows_shifted"] == [4, 5]
    assert treatment["rows_unchanged"] == [0, 1, 2, 3]
    assert np.allclose(audit["waypoints_world_m"]["preplace"]["position"], (0.1, 0.2, 0.8))
    assert np.allclose(audit["waypoints_world_m"]["descend_place"]["position"], (0.1, 0.2, 0.72))
    assert np.allclose(audit["waypoints_world_m"]["retreat"]["position"], (0.1, 0.2, 0.62))
    assert np.allclose(audit["waypoints_world_m"]["descend_place"]["orientation"], np.eye(3))


@pytest.mark.parametrize("offset", [-0.001, 0.01, float("nan"), float("inf")])
def test_release_height_rejects_invalid_or_non_experimental_values(offset, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="experimental_release_height_offset_m"):
        runner.run_episode(
            env=_Env(), task_id=0, seed=1000, output_dir=tmp_path, arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_capture(), dry_run=True, experimental_release_height_offset_m=offset,
        )


def test_release_height_nonzero_requires_experimental_candidate(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires an experimental candidate"):
        runner.run_episode(
            env=_Env(), task_id=0, seed=1000, output_dir=tmp_path, arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_capture(), dry_run=True, experimental_release_height_offset_m=0.020,
        )


def test_release_height_validates_shifted_actual_waypoints_before_motion(monkeypatch, tmp_path):
    _patch_controller(monkeypatch, high_release=True)
    env = _Env()
    with pytest.raises(ValueError, match="workspace validation failed.*waypoint_descend_place"):
        runner.run_episode(
            env=env, task_id=0, seed=1000, output_dir=tmp_path, arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_capture(), dry_run=False, experimental_candidate=_candidate(),
            experimental_eef_orientation_transform=np.eye(3), experimental_gripper_opening_m=0.05,
            experimental_release_height_offset_m=0.020,
        )
    assert env.actions == []
