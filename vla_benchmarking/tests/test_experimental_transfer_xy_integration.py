"""Real runner/controller seam check for the visual transfer-XY treatment."""

from __future__ import annotations

import numpy as np

import arrow_controller
import run_arrow_pick_place_eval as runner


class _TrackingEnv:
    def __init__(self) -> None:
        self._arrow_settle_diagnostics = {"settled": True}
        self.pending = np.asarray((0.0, 0.0, 0.7), dtype=float)
        self.pose = self.pending.copy()
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.pose = self.pending.copy()
        return {
            "robot0_eef_pos": self.pose.copy(),
            "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
            "robot0_gripper_qpos": np.asarray((0.05, 0.05)),
        }, 0.0, False, {}


def test_visual_xy_real_episode_forwards_targets_and_accounts_actions(monkeypatch, tmp_path):
    calibration = runner.CameraCalibration(
        "agentview", 256, 256,
        [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    observation = {
        "robot0_eef_pos": np.asarray((0.0, 0.0, 0.7)),
        "robot0_eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
        "robot0_gripper_qpos": np.asarray((0.05, 0.05)),
    }
    capture = runner.CapturedRGBD(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        calibration, observation,
    )
    env = _TrackingEnv()
    monkeypatch.setattr(
        runner, "decode_arrow",
        lambda **_kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(
        runner,
        "deproject_endpoint",
        lambda point, depth, K, T: np.asarray((float(point[0]) / 100.0, float(point[1]) / 100.0, 1.0)),
    )
    # Keep the real build_bowl_waypoints and real _run_motion.  This hook only
    # makes the faithful fake environment converge to the commanded target.
    monkeypatch.setattr(runner, "build_bowl_waypoints", arrow_controller.build_bowl_waypoints)

    def fake_action(proprio, waypoint, **_kwargs):
        env.pending = np.asarray(waypoint["position"], dtype=float).copy()
        return np.zeros(7, dtype=np.float32)

    monkeypatch.setattr(runner, "normalized_action_for_waypoint", fake_action)
    candidate = {
        "candidate_id": "visual-integration",
        "grip_site_world_m": (0.1, 0.2, 0.8),
        "rotation_world_grip_site": np.eye(3),
        "required_aperture_m": 0.04,
        "pregrasp_world_m": (0.1, 0.2, 0.85),
    }
    audit = runner.run_episode(
        env=env, task_id=0, seed=1000, output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8), capture=capture,
        dry_run=False, experimental_candidate=candidate,
        experimental_eef_orientation_transform=np.eye(3),
        experimental_gripper_opening_m=0.05,
        experimental_release_height_offset_m=0.020,
        experimental_transfer_xy_policy="visual_endpoints",
        experimental_motion_diagnostics=True,
    )

    transfer = audit["experimental_grasp"]["audit"]["transfer_xy"]
    np.testing.assert_allclose(
        np.asarray(transfer["chosen_displacement_world_m"])
        - np.asarray(transfer["legacy_displacement_world_m"]),
        (0.0203, -0.0052, 0.0), atol=1e-9,
    )
    assert audit["control_targets_world_m"]["destination_release"] == [0.26, 0.36, 0.8066]
    waypoints = audit["waypoints_world_m"]
    assert waypoints["preplace"]["position"][:2] == [0.26, 0.36]
    assert waypoints["descend_place"]["position"] == [0.26, 0.36, 0.8266]
    assert waypoints["open"]["position"] == [0.26, 0.36, 0.8266]
    assert waypoints["retreat"]["position"] == [0.26, 0.36, 0.8566]
    assert np.asarray(audit["experimental_grasp"]["orientation_world_grip_site"]).shape == (3, 3)
    assert audit["experimental_action_budget"]["limit"] == 1200
    assert audit["experimental_action_budget"]["used"] == len(env.actions) == 49
    assert [item["phase"] for item in audit["phases"]] == [
        "vertical_clearance", "rotate", "translate_clearance", "pregrasp",
        "descend", "close", "lift", "preplace", "descend_place", "open", "retreat",
    ]
