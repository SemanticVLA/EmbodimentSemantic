"""Focused contracts for the opt-in experimental transfer-XY ablation."""

from __future__ import annotations

import numpy as np
import pytest

from vla_benchmarking.arrow_grasp_controller.legacy_engine import arrow_controller
from vla_benchmarking.evaluation import run_arrow_pick_place_eval as runner


class _Env:
    def __init__(self):
        self.actions = []

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


def _candidate(position=(0.0, 0.0, 1.0)):
    return {
        "candidate_id": "c0",
        "grip_site_world_m": position,
        "rotation_world_grip_site": np.eye(3),
        "required_aperture_m": 0.04,
        "pregrasp_world_m": (position[0], position[1], position[2] + 0.05),
    }


def _patch_controller(monkeypatch):
    monkeypatch.setattr(
        runner, "decode_arrow",
        lambda **_kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(
        runner, "deproject_endpoint",
        lambda point, depth, K, T: np.asarray((float(point[0]) / 100.0, float(point[1]) / 100.0, 1.0)),
    )
    monkeypatch.setattr(runner, "build_bowl_waypoints", arrow_controller.build_bowl_waypoints)
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **_kwargs: np.zeros(7, dtype=np.float32))


_DEFAULT_CANDIDATE = object()


def _run(monkeypatch, tmp_path, *, policy=None, candidate=_DEFAULT_CANDIDATE):
    _patch_controller(monkeypatch)
    candidate_value = _candidate() if candidate is _DEFAULT_CANDIDATE else candidate
    kwargs = dict(
        env=_Env(), task_id=0, seed=1000, output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8), capture=_capture(), dry_run=True,
        experimental_candidate=candidate_value, experimental_eef_orientation_transform=np.eye(3),
        experimental_gripper_opening_m=0.05,
    )
    if policy is not None:
        kwargs["experimental_transfer_xy_policy"] = policy
    return runner.run_episode(**kwargs)


def test_default_transfer_policy_matches_explicit_legacy(monkeypatch, tmp_path):
    default = _run(monkeypatch, tmp_path / "default")
    explicit = _run(monkeypatch, tmp_path / "explicit", policy="legacy_displacement")
    assert default["waypoints_world_m"] == explicit["waypoints_world_m"]
    assert default["experimental_grasp"]["audit"]["transfer_xy"] == explicit["experimental_grasp"]["audit"]["transfer_xy"]


def test_visual_endpoint_policy_changes_only_transfer_xy(monkeypatch, tmp_path):
    legacy_audit = _run(monkeypatch, tmp_path / "legacy", policy="legacy_displacement")
    audit = _run(monkeypatch, tmp_path / "visual", policy="visual_endpoints")
    transfer = audit["experimental_grasp"]["audit"]["transfer_xy"]
    legacy = np.asarray(transfer["legacy_displacement_world_m"])
    chosen = np.asarray(transfer["chosen_displacement_world_m"])
    visual = np.asarray(transfer["visual_destination_anchor_world_m"]) - np.asarray(transfer["visual_source_anchor_world_m"])
    np.testing.assert_allclose(chosen[:2], visual[:2])
    assert chosen[2] == pytest.approx(legacy[2])
    np.testing.assert_allclose(chosen - legacy, (0.0203, -0.0052, 0.0), atol=1e-9)
    legacy_waypoints = legacy_audit["waypoints_world_m"]
    visual_waypoints = audit["waypoints_world_m"]
    for phase in ("vertical_clearance", "rotate", "translate_clearance", "pregrasp", "descend", "close", "lift"):
        np.testing.assert_allclose(
            visual_waypoints[phase]["position"], legacy_waypoints[phase]["position"],
        )
    for phase in ("preplace", "descend_place", "open", "retreat"):
        np.testing.assert_allclose(
            np.asarray(visual_waypoints[phase]["position"][:2])
            - np.asarray(legacy_waypoints[phase]["position"][:2]),
            (0.0203, -0.0052), atol=1e-9,
        )
        assert visual_waypoints[phase]["position"][2] == pytest.approx(legacy_waypoints[phase]["position"][2])
        assert visual_waypoints[phase]["orientation"] == np.eye(3).tolist()


def test_visual_endpoint_transfer_preserves_candidate_separation(monkeypatch, tmp_path):
    first = _run(monkeypatch, tmp_path / "first", policy="visual_endpoints", candidate=_candidate((0.0, 0.0, 1.0)))
    second = _run(monkeypatch, tmp_path / "second", policy="visual_endpoints", candidate=_candidate((0.03, -0.02, 1.0)))
    first_transfer = np.asarray(first["experimental_grasp"]["audit"]["transfer_xy"]["chosen_displacement_world_m"])
    second_transfer = np.asarray(second["experimental_grasp"]["audit"]["transfer_xy"]["chosen_displacement_world_m"])
    np.testing.assert_allclose(first_transfer, second_transfer)
    first_release = np.asarray(first["control_targets_world_m"]["destination_release"])
    second_release = np.asarray(second["control_targets_world_m"]["destination_release"])
    np.testing.assert_allclose(second_release - first_release, (0.03, -0.02, 0.0), atol=1e-9)


@pytest.mark.parametrize("policy", ["unknown", "visual_endpoints"])
def test_visual_endpoint_policy_validation(policy, monkeypatch, tmp_path):
    candidate = None if policy == "visual_endpoints" else _candidate()
    with pytest.raises(ValueError, match="experimental_transfer_xy_policy"):
        _run(monkeypatch, tmp_path / policy, policy=policy, candidate=candidate)
