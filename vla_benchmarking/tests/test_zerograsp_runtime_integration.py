import json
from pathlib import Path

import numpy as np
import pytest

# ZeroGrasp is intentionally retired from the active arrow runtime. Keep this
# file as historical integration documentation for a future side experiment,
# but do not let provider-specific expectations define the frozen v9d baseline.
pytestmark = pytest.mark.skip(reason="ZeroGrasp is archived and not part of active v9d")

try:
    from vla_benchmarking import run_arrow_pick_place_eval as episode
    from vla_benchmarking import run_arrow_pick_place_matrix as matrix
except ModuleNotFoundError:  # repository intentionally has no package __init__
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import run_arrow_pick_place_eval as episode
    import run_arrow_pick_place_matrix as matrix


def _zg_policy():
    return {
        "fixed_seed": 0,
        "max_candidates": 2,
        "eef_calibration_verified": False,
        "R_grasp_eef": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        "tip_to_eef_residual_m": [0, 0, 0],
        "R_H_E": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        "translation_rule": "center_plus_depth_x_then_R_G_E_delta_E_v1",
        "calibration_source": "UNVERIFIED_PENDING_LIVE_PROBE",
        "calibration_sha256": "0" * 64,
        "probe_sha256": "0" * 64,
    }


def test_zero_grasp_fields_are_optional_and_legacy_canonical_is_unchanged():
    legacy = episode.ControllerVariantConfig()
    assert "grasp_provider" not in legacy.canonical()
    assert "placement_provider" not in legacy.canonical()
    variant = episode.ControllerVariantConfig(
        name="zg", grasp_provider="zerograsp", placement_provider="classical", zerograsp=_zg_policy()
    )
    canonical = variant.canonical()
    assert canonical["grasp_provider"] == "zerograsp"
    assert canonical["placement_provider"] == "classical"
    assert canonical["zerograsp"]["eef_calibration_verified"] is False
    assert "pose_calibration" not in canonical["zerograsp"]


def test_grip_site_orientation_conversion_uses_explicit_h_to_e_rotation():
    r_h_e = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r_w_h = np.eye(3)
    r_w_e = r_w_h @ r_h_e
    assert episode._rotation_error_rad(r_w_h, r_w_e) == pytest.approx(np.pi / 2)


def test_observation_keeps_grip_site_position_and_right_hand_quaternion_separate():
    calibration = episode.CameraCalibration(
        camera_name="agentview", width=2, height=2,
        intrinsic=[[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    capture = episode.CapturedRGBD(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        normalized_depth=np.ones((2, 2), dtype=np.float32),
        metric_depth=np.ones((2, 2), dtype=np.float32),
        calibration=calibration,
        observation={
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.01, 0.02]),
        },
    )
    observation = episode._zerograsp_observation(capture, capture.rgb.copy(), (0, 0), (1, 1))
    np.testing.assert_allclose(observation.eef_position_world_m, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(observation.eef_quaternion_right_hand_xyzw, [0, 0, 0, 1])
    assert not hasattr(observation, "eef_pose")


def test_unverified_flat_calibration_fails_before_adapter_or_evaluator(tmp_path: Path, monkeypatch):
    calibration = episode.CameraCalibration(
        camera_name="agentview", width=2, height=2,
        intrinsic=[[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    capture = episode.CapturedRGBD(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        normalized_depth=np.ones((2, 2), dtype=np.float32),
        metric_depth=np.ones((2, 2), dtype=np.float32),
        calibration=calibration,
        observation={},
    )
    variant = episode.ControllerVariantConfig(
        name="zg", grasp_provider="zerograsp", placement_provider="classical",
        zerograsp=_zg_policy(),
    )
    calls = []

    class Env:
        pass

    class Adapter:
        def infer(self, _observation):
            calls.append("infer")
            raise AssertionError("unverified calibration must stop before adapter inference")

    monkeypatch.setattr(
        episode, "decode_arrow_pixels",
        lambda *_args, **_kwargs: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    )
    with pytest.raises(RuntimeError, match="flat ZeroGrasp EEF calibration"):
        episode.run_episode(
            env=Env(), task_id=0, seed=1000, output_dir=tmp_path,
            arrow_rgb=np.zeros((2, 2, 3), dtype=np.uint8), capture=capture,
            controller_variant=variant, zerograsp_adapter=Adapter(),
            evaluator=lambda _candidate: calls.append("evaluator") or True,
            dry_run=True, resolution=2,
        )
    assert calls == []


def test_matrix_constructs_and_closes_one_adapter(tmp_path: Path):
    config = tmp_path / "zg.json"
    config.write_text(json.dumps({
        "name": "zg",
        "grasp_provider": "zerograsp",
        "placement_provider": "classical",
        "zerograsp": _zg_policy(),
    }), encoding="utf-8")
    seen = []

    class Adapter:
        def close(self):
            seen.append("close")

    def factory(policy, **_kwargs):
        seen.append("create")
        return Adapter()

    class Env:
        _arrow_settle_diagnostics = {"settled": True}

        def close(self):
            pass

    def episode_runner(**kwargs):
        assert kwargs["zerograsp_adapter"] is not None
        return {
            "evaluator_success": None,
            "audit_path": None,
            "controller_variant": kwargs["controller_variant"].provenance(),
            "capture_contract": {"valid": True, "rgb_shape": [2, 2, 3]},
            "phases": [],
            "phase_frames": [],
            "frames": [],
            "zerograsp": {"status": "completed", "fallback": False, "runtime_hash": "r", "frame_metadata": {"camera_frame": "c", "world_frame": "w", "transform": "t"}},
        }

    summary = matrix.run_matrix(
        output_root=tmp_path / "out",
        task_ids=[0], episodes_per_task=1, dry_run=True,
        controller_config=config, env_builder=lambda *_args, **_kwargs: Env(),
        episode_runner=episode_runner, arrow_input_builder=lambda *_args, **_kwargs: {},
        zerograsp_adapter_factory=factory,
    )
    assert summary["total_cells"] == 1
    assert seen == ["create", "close"]


def test_bundled_v10_configs_parse_flat_policy_and_adapter_factory_receives_it():
    for filename in ("v10_zg_grasp_only.json", "v10_zg_grasp_recon_place.json"):
        path = Path(__file__).resolve().parents[1] / "controller_configs" / filename
        _, variant, provenance = matrix._resolve_controller_selection(
            controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
            controller_config=path,
            suite_mode="vanilla",
        )
        assert provenance["config_hash"]
        assert variant.zerograsp["eef_calibration_verified"] is True
        assert variant.zerograsp["calibration_sha256"] == "f50c2d423cfe5573c02f9d63205fd52cdbef5303a4f3ebd9985eb0c888fb4071"
        assert variant.zerograsp["probe_sha256"] == "f879a86027d5b9cb67bfbb62cb045bece3f485f977e7c2bf61d3127358026291"
        assert "pose_calibration" not in variant.zerograsp
        seen = []

        def factory(policy, **_kwargs):
            seen.append(policy)
            return object()

        matrix._build_zerograsp_adapter(variant, "vanilla", factory)
        assert seen and seen[0]["translation_rule"] == "center_plus_depth_x_then_R_G_E_delta_E_v1"


def test_runtime_policy_override_is_rejected_before_factory_construction():
    policy = _zg_policy()
    policy["checkpoint"] = "/unverified/override.ckpt"
    variant = episode.ControllerVariantConfig(
        name="zg", grasp_provider="zerograsp", placement_provider="classical", zerograsp=policy,
    )
    constructed = []

    def factory(_policy, **_kwargs):
        constructed.append(True)
        return object()

    with pytest.raises(RuntimeError, match="policy overrides are forbidden"):
        matrix._build_zerograsp_adapter(variant, "vanilla", factory)
    assert constructed == []


def test_motion_preserves_grip_site_orientation_during_micro_correction(monkeypatch):
    r_h_e = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    calls = []

    def fake_normalized_osc_action(**kwargs):
        calls.append(kwargs)
        return np.zeros(7, dtype=np.float64)

    class Env:
        def __init__(self):
            self.steps = 0

        def step(self, _action):
            self.steps += 1
            return {
                "robot0_eef_pos": np.array([0.1, 0.0, 0.0]) if self.steps >= 2 else np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            }, 0.0, False, {}

    monkeypatch.setattr(episode, "normalized_osc_action", fake_normalized_osc_action)
    policy = episode.MicroCorrectionPolicy(
        enabled=True, phases=("pregrasp",), plateau_window_steps=1,
        plateau_delta_m=1.0, residual_max_m=0.02, burst_steps=1,
        max_rounds=1, max_actions=2,
    )
    waypoints = {
        "pregrasp": {
            "position": np.array([0.1, 0.0, 0.0]),
            "orientation": r_h_e,
            "orientation_frame": "grip_site",
        }
    }
    episode._run_motion(
        Env(), waypoints,
        {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])},
        phase_timeout_steps=3, gripper_dwell_steps=1, stop_after_phase="pregrasp",
        dry_run=False, stall_window_steps=0, micro_correction=policy,
        eef_orientation_transform=r_h_e,
    )
    assert len(calls) == 2
    for call in calls:
        np.testing.assert_allclose(call["target_rot"], r_h_e)
        np.testing.assert_allclose(call["current_rot"], r_h_e)


def test_motion_legacy_position_only_waypoint_keeps_held_orientation(monkeypatch):
    calls = []

    def fake_normalized_osc_action(**kwargs):
        calls.append(kwargs)
        return np.zeros(7, dtype=np.float64)

    class Env:
        pass

    held = np.array([0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr(episode, "normalized_osc_action", fake_normalized_osc_action)
    episode._run_motion(
        Env(), [np.array([0.1, 0.0, 0.0])] * 6,
        {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": held},
        phase_timeout_steps=2, gripper_dwell_steps=1, stop_after_phase="pregrasp",
        dry_run=True, stall_window_steps=0,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["current_rot"], held)
    np.testing.assert_allclose(calls[0]["target_rot"], held)


def test_grip_site_timeout_audit_uses_converted_right_hand_orientation(monkeypatch):
    r_h_e = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def fake_normalized_osc_action(**_kwargs):
        return np.zeros(7, dtype=np.float64)

    class Env:
        def step(self, _action):
            return {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            }, 0.0, False, {}

    monkeypatch.setattr(episode, "normalized_osc_action", fake_normalized_osc_action)
    env = Env()
    with pytest.raises(episode.ControllerMotionTimeout):
        episode._run_motion(
            env, {
                "pregrasp": {
                    "position": np.array([0.1, 0.0, 0.0]),
                    "orientation": r_h_e,
                    "orientation_frame": "grip_site",
                }
            },
            {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0])},
            phase_timeout_steps=1, gripper_dwell_steps=1,
            stop_after_phase="pregrasp", dry_run=False, stall_window_steps=0,
            eef_orientation_transform=r_h_e,
        )
    assert env._arrow_phase_audit[-1]["orientation_error_rad"] == pytest.approx(0.0)


def test_learned_source_height_drives_upward_safe_transfer_clearance(tmp_path: Path, monkeypatch):
    policy = _zg_policy()
    policy["R_grasp_eef"] = np.asarray(policy["R_grasp_eef"], dtype=float).tolist()
    policy["R_H_E"] = np.asarray(policy["R_H_E"], dtype=float).tolist()
    policy["eef_calibration_verified"] = True
    policy["calibration_sha256"] = episode.stable_json_hash({
        "translation_rule": policy["translation_rule"],
        "R_grasp_eef": policy["R_grasp_eef"],
        "R_H_E": policy["R_H_E"],
        "tip_to_eef_residual_m": (0.0, 0.0, 0.0),
        "calibration_source": policy["calibration_source"],
    })
    variant = episode.ControllerVariantConfig(
        name="zg", grasp_provider="zerograsp", placement_provider="classical", zerograsp=policy,
    )
    calibration = episode.CameraCalibration(
        camera_name="agentview", width=2, height=2,
        intrinsic=[[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    capture = episode.CapturedRGBD(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        normalized_depth=np.ones((2, 2), dtype=np.float32),
        metric_depth=np.ones((2, 2), dtype=np.float32),
        calibration=calibration,
        observation={
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },
    )
    source_pose = np.eye(4)
    source_pose[:3, 3] = [0.1, 0.0, 1.4]
    pregrasp_pose = source_pose.copy()
    pregrasp_pose[2, 3] = 1.48

    class Env:
        pass

    class Result:
        request_hash = "request"
        output_hash = "output"

    class Adapter:
        runtime_hash = "runtime"
        last_audit = {}

        def infer(self, _observation):
            return Result()

        def select(self, _result, _observation):
            return {"T_W_E": source_pose, "T_W_E_pregrasp": pregrasp_pose}

    monkeypatch.setattr(
        episode, "decode_arrow_pixels",
        lambda *_args, **_kwargs: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    )
    audit = episode.run_episode(
        env=Env(), task_id=0, seed=1000, output_dir=tmp_path,
        arrow_rgb=np.zeros((2, 2, 3), dtype=np.uint8), capture=capture,
        controller_variant=variant, zerograsp_adapter=Adapter(),
        dry_run=True, resolution=2, clearance_m=0.08,
    )
    waypoints = audit["waypoints_world_m"]
    assert waypoints[1]["position"][2] == pytest.approx(1.4)
    assert waypoints[2]["position"][2] >= 1.48 - 1e-9
    assert waypoints[3]["position"][2] >= 1.48 - 1e-9
    assert waypoints[2]["position"][2] > waypoints[1]["position"][2]


def test_c10_waypoints_keep_selected_orientation_through_release():
    source_rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    classical = [np.array([0.1 * index, 0.0, 0.2]) for index in range(6)]
    values = list(classical)
    values[0] = episode._pose_waypoint(values[0], source_rotation, "grip_site")
    values[1] = episode._pose_waypoint(values[1], source_rotation, "grip_site")
    values[2] = episode._pose_waypoint(values[2], source_rotation, "grip_site")
    values[3] = episode._pose_waypoint(values[3], source_rotation, "grip_site")
    # Mirror the runtime's C10 release construction: classical positions,
    # selected ZeroGrasp orientation, explicit grip-site frame.
    values[4] = episode._pose_waypoint(values[4], source_rotation, "grip_site")
    values[5] = episode._pose_waypoint(values[5], source_rotation, "grip_site")
    for waypoint in values:
        np.testing.assert_allclose(episode._orientation(waypoint), source_rotation)
        assert episode._orientation_frame(waypoint) == "grip_site"


def test_failed_adapter_preserves_partial_runtime_audit(tmp_path: Path, monkeypatch):
    policy = _zg_policy()
    policy["R_grasp_eef"] = np.asarray(policy["R_grasp_eef"], dtype=float).tolist()
    policy["R_H_E"] = np.asarray(policy["R_H_E"], dtype=float).tolist()
    policy["eef_calibration_verified"] = True
    policy["calibration_sha256"] = episode.stable_json_hash({
        "translation_rule": policy["translation_rule"],
        "R_grasp_eef": policy["R_grasp_eef"],
        "R_H_E": policy["R_H_E"],
        "tip_to_eef_residual_m": (0.0, 0.0, 0.0),
        "calibration_source": policy["calibration_source"],
    })
    variant = episode.ControllerVariantConfig(
        name="zg", grasp_provider="zerograsp", placement_provider="classical", zerograsp=policy,
    )
    calibration = episode.CameraCalibration(
        camera_name="agentview", width=2, height=2,
        intrinsic=[[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    capture = episode.CapturedRGBD(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        normalized_depth=np.ones((2, 2), dtype=np.float32),
        metric_depth=np.ones((2, 2), dtype=np.float32),
        calibration=calibration,
        observation={
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },
    )
    class Env:
        pass
    env = Env()
    class Adapter:
        runtime_hash = "runtime-partial"
        last_audit = {"request_hash": "request-partial", "frame_metadata": {"worker": "partial"}}
        def infer(self, _observation):
            raise RuntimeError("worker failed")
        def select(self, _result, _observation):
            raise AssertionError("select must not run after infer failure")
    monkeypatch.setattr(
        episode, "decode_arrow_pixels",
        lambda *_args, **_kwargs: (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        episode.run_episode(
            env=env, task_id=0, seed=1000, output_dir=tmp_path,
            arrow_rgb=np.zeros((2, 2, 3), dtype=np.uint8), capture=capture,
            controller_variant=variant, zerograsp_adapter=Adapter(),
            dry_run=True, resolution=2,
        )
    audit = env._arrow_zerograsp_audit
    assert audit["status"] == "failed"
    assert audit["fallback"] is False
    assert audit["runtime_hash"] == "runtime-partial"
    assert audit["frame_metadata"]["camera_frame"]
    assert audit["frame_metadata"]["worker"] == "partial"
