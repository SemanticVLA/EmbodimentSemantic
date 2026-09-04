from __future__ import annotations

import numpy as np
import pytest

import run_arrow_pick_place_eval as episode
from run_molmo_sam3_canary import _candidate_motion_kwargs, resolve_motion_profile


class _Env:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "robot0_gripper_qpos": np.zeros(2),
        }, 0.0, False, {}


def _capture():
    calibration = episode.CameraCalibration(
        "agentview", 256, 256,
        [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
        "robot0_gripper_qpos": np.zeros(2),
    }
    return episode.CapturedRGBD(
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


def _run_dry_profile(monkeypatch, tmp_path, profile, base):
    monkeypatch.setattr(
        episode,
        "decode_arrow",
        lambda **_kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(
        episode,
        "deproject_endpoint",
        lambda point, depth, K, T: np.asarray((float(point[0]) / 100.0, float(point[1]) / 100.0, 1.0)),
    )
    monkeypatch.setattr(episode, "build_bowl_waypoints", lambda *_args: np.asarray(base, dtype=float).copy())
    monkeypatch.setattr(episode, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    kwargs = _candidate_motion_kwargs(profile)
    return episode.run_episode(
        env=_Env(), task_id=0, seed=1000, output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8), capture=_capture(), dry_run=True,
        experimental_candidate=_candidate(), experimental_eef_orientation_transform=np.eye(3),
        experimental_gripper_opening_m=0.05, **kwargs,
    )


def test_failure_profiles_are_explicit_and_defaults_remain_unchanged():
    baseline = resolve_motion_profile("baseline", region_backend="sam3")
    assert baseline == {
        "name": "baseline", "region_backend": "sam3", "micro_correction": None,
        "release_height_offset_m": 0.0, "transfer_xy_policy": "legacy_displacement",
    }
    assert _candidate_motion_kwargs(baseline) == {}
    retreat = resolve_motion_profile("release20_retreat80mm", region_backend="rgbd")
    assert retreat["release_height_offset_m"] == pytest.approx(0.020)
    assert retreat["retreat_height_offset_m"] == pytest.approx(0.080)
    assert retreat["retreat_tolerance_m"] == pytest.approx(0.005)
    assert _candidate_motion_kwargs(retreat) == {
        "experimental_release_height_offset_m": pytest.approx(0.020),
        "experimental_retreat_height_offset_m": pytest.approx(0.080),
        "experimental_motion_profile": "release20_retreat80mm",
    }
    plus40 = resolve_motion_profile("release_plus40mm", region_backend="rgbd")
    assert plus40["release_height_offset_m"] == pytest.approx(0.040)
    assert plus40["retreat_height_offset_m"] == 0.0
    assert plus40["retreat_tolerance_m"] == pytest.approx(0.015)
    with pytest.raises(ValueError):
        resolve_motion_profile("unknown_failure_profile", region_backend="rgbd")
    with pytest.raises(ValueError, match="requires an experimental candidate"):
        episode.run_episode(
            env=object(), task_id=0, seed=1000, output_dir=".",
            experimental_motion_profile="release_plus40mm",
        )


def test_release20_retreat80_changes_only_release_and_final_retreat(monkeypatch, tmp_path):
    base = (
        (0.0, 0.0, 0.53), (0.0, 0.0, 0.50), (0.0, 0.0, 0.80),
        (0.2, 0.3, 0.80), (0.2, 0.3, 0.50), (0.2, 0.3, 0.53),
    )
    profile = resolve_motion_profile("release20_retreat80mm", region_backend="rgbd")
    audit = _run_dry_profile(monkeypatch, tmp_path, profile, base)
    treatment = audit["experimental_grasp"]["audit"]["release_height_treatment"]
    assert treatment["motion_profile"] == "release20_retreat80mm"
    assert treatment["shifted_release_world_m"] == pytest.approx([0.2, 0.3, 0.52])
    assert treatment["shifted_retreat_world_m"] == pytest.approx([0.2, 0.3, 0.60])
    assert audit["phase_policies"]["retreat"]["tolerance_m"] == pytest.approx(0.005)
    assert audit["experimental_motion_profile"] == "release20_retreat80mm"


def test_release_plus40_preserves_original_retreat_and_orientation(monkeypatch, tmp_path):
    base = (
        (0.0, 0.0, 0.53), (0.0, 0.0, 0.50), (0.0, 0.0, 0.80),
        (0.2, 0.3, 0.80), (0.2, 0.3, 0.50), (0.2, 0.3, 0.53),
    )
    profile = resolve_motion_profile("release_plus40mm", region_backend="rgbd")
    audit = _run_dry_profile(monkeypatch, tmp_path, profile, base)
    treatment = audit["experimental_grasp"]["audit"]["release_height_treatment"]
    assert treatment["motion_profile"] == "release_plus40mm"
    assert treatment["shifted_release_world_m"] == pytest.approx([0.2, 0.3, 0.54])
    assert treatment["shifted_retreat_world_m"] == pytest.approx([0.2, 0.3, 0.53])
    assert treatment["shifted_retreat_world_m"] == pytest.approx(treatment["nominal_retreat_world_m"])
    assert treatment["rows_preserved"] == [5]
    assert audit["phase_policies"]["retreat"]["tolerance_m"] == pytest.approx(0.015)
    assert np.allclose(audit["waypoints_world_m"]["descend_place"]["orientation"], np.eye(3))
    assert np.asarray(audit["waypoints_world_m"]["preplace"]["position"])[2] - np.asarray(
        audit["waypoints_world_m"]["descend_place"]["position"]
    )[2] >= 0.040


def test_release_profile_workspace_validation_precedes_motion(monkeypatch, tmp_path):
    base = (
        (0.0, 0.0, 0.53), (0.0, 0.0, 0.50), (0.0, 0.0, 0.80),
        (0.2, 0.3, 0.80), (0.2, 0.3, 1.79), (0.2, 0.3, 0.53),
    )
    profile = resolve_motion_profile("release_plus20mm", region_backend="rgbd")
    # Reuse setup but retain the concrete env so action dispatch can be checked.
    monkeypatch.setattr(
        episode,
        "decode_arrow",
        lambda **_kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(episode, "deproject_endpoint", lambda point, depth, K, T: np.asarray((float(point[0]) / 100.0, float(point[1]) / 100.0, 1.0)))
    monkeypatch.setattr(episode, "build_bowl_waypoints", lambda *_args: np.asarray(base, dtype=float).copy())
    monkeypatch.setattr(episode, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    env = _Env()
    env._arrow_settle_diagnostics = {"settled": True}
    with pytest.raises(ValueError, match="workspace validation failed"):
        episode.run_episode(
            env=env, task_id=0, seed=1000, output_dir=tmp_path,
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8), capture=_capture(), dry_run=False,
            experimental_candidate=_candidate(), experimental_eef_orientation_transform=np.eye(3),
            experimental_gripper_opening_m=0.05, **_candidate_motion_kwargs(profile),
        )
    assert env.actions == []


def test_passive_phase_observer_is_noop_for_motion_and_evaluator(monkeypatch):
    env = _Env()
    monkeypatch.setattr(episode, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))
    observed = []
    phases = episode._run_motion(
        env, np.zeros((6, 3)),
        {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0))},
        phase_timeout_steps=2, gripper_dwell_steps=1, stop_after_phase="retreat", dry_run=True,
        phase_observer=lambda phase, record: observed.append((phase, record["status"])),
    )
    assert [phase for phase, _status in observed] == list(episode.PHASES)
    assert [record["phase"] for record in phases] == list(episode.PHASES)
    assert env.actions == []
