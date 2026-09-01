"""Contract tests for the arrow-only LIBERO pick/place runner.

These tests intentionally use small fakes rather than importing LIBERO at
collection time.  The runner is expected to keep simulator construction and
the arrow controller behind injectable seams so geometry can be tested on a
machine without MuJoCo or GPU dependencies.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


MODULE_NAME = "run_arrow_pick_place_eval"


@pytest.fixture(scope="module")
def runner():
    """Import the runner without requiring optional LIBERO dependencies."""
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE_NAME or exc.name in {"libero", "robosuite", "lerobot", "mujoco"}:
            pytest.skip(f"optional LIBERO dependency unavailable: {exc.name}")
        raise


class _CameraModel:
    camera_names = ["agentview"]

    def camera_name2id(self, name):
        assert name == "agentview"
        return 0


class _CameraData:
    cam_xpos = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    # MuJoCo stores camera orientation as a row-major world-from-camera
    # rotation.  Identity keeps this fake easy to reason about.
    cam_xmat = np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])


class _Sim:
    model = _CameraModel()
    data = _CameraData()


class _Env:
    sim = _Sim()

    def __init__(self):
        self.render_calls = []

    def render(self, camera_name, width, height, depth=False):
        self.render_calls.append((camera_name, width, height, depth))
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_image = np.full((height, width), 0.5, dtype=np.float32)
        return (rgb, depth_image) if depth else rgb


def _call(runner, names, *args, **kwargs):
    for name in names:
        function = getattr(runner, name, None)
        if function is not None:
            return function(*args, **kwargs)
    raise AssertionError(f"runner did not expose any of {names!r}")


def test_capture_requests_one_synchronized_rgb_depth_pair(runner):
    env = _Env()
    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[10.0, 0.0, camera_width / 2], [0.0, 10.0, camera_height / 2], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            return np.asarray(depth, dtype=np.float32)

    runner.camera_utils = _CameraUtils
    result = _call(
        runner,
        ("capture_agentview", "capture_synchronized_frame", "capture_frame"),
        env,
        resolution=32,
    )
    if isinstance(result, dict):
        rgb = result.get("rgb", result.get("image"))
        depth = result.get("depth")
    elif hasattr(result, "rgb"):
        rgb, depth = result.rgb, result.normalized_depth
    else:
        rgb, depth = result[:2]

    assert rgb.shape == (32, 32, 3)
    assert depth.shape == (32, 32)
    assert env.render_calls == [("agentview", 32, 32, True)]


def test_metric_depth_conversion_delegates_to_robosuite_hook(runner, monkeypatch):
    calls = []

    class _CameraUtils:
        @staticmethod
        def get_real_depth_map(sim, depth):
            calls.append((sim, depth))
            return np.asarray(depth, dtype=np.float32) * 2.0

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    normalized = np.asarray([[0.25]], dtype=np.float32)
    converted = _call(
        runner,
        ("normalized_depth_to_metric", "metric_depth", "convert_depth_to_metric"),
        _Sim(),
        normalized,
    )
    np.testing.assert_allclose(converted, [[0.5]])
    assert calls and calls[0][1] is normalized


def test_camera_calibration_converts_projection_v_up_to_image_v_down(runner):
    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[100.0, 0.0, 13.0], [0.0, 80.0, 7.0], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

    runner.camera_utils = _CameraUtils
    calibration = runner.build_camera_calibration(_Sim(), "agentview", 64, 48)
    np.testing.assert_allclose(calibration.intrinsic, [[100.0, 0.0, 13.0], [0.0, -80.0, 41.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(calibration.raw_projection_intrinsic, [[100.0, 0.0, 13.0], [0.0, 80.0, 7.0], [0.0, 0.0, 1.0]])
    assert calibration.projection_vertical_axis == "robosuite_projection_v_up_converted_to_image_v_down"


def test_arrow_controller_is_pixel_depth_calibration_only(runner):
    controller_cls = getattr(runner, "ArrowController", None)
    if controller_cls is None:
        pytest.skip("runner does not expose an ArrowController class")

    controller = controller_cls()
    forbidden = {"sim", "env", "model", "bboxes", "scene_graph", "world_pose"}
    assert not forbidden.intersection(vars(controller))

    # A controller may retain immutable calibration/configuration, but should
    # receive the visual input explicitly instead of reaching into LIBERO.
    accepted = getattr(controller, "decode", None) or getattr(runner, "decode_arrow")
    assert callable(accepted)


def test_normalized_osc_action_is_finite_and_bounded(runner):
    function = getattr(runner, "normalized_osc_action", None)
    if function is None:
        pytest.skip("runner does not expose normalized_osc_action")

    raw = np.asarray([99.0, -99.0, 0.25])
    bounded = np.asarray(
        function(
            raw,
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            raw,
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            1.0,
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
    )
    assert bounded.shape == (7,)
    assert np.isfinite(bounded).all()
    assert np.all(bounded >= -1.0)
    assert np.all(bounded <= 1.0)


def test_dry_run_writes_auditable_artifact_without_step(runner, tmp_path: Path):
    class _NoMotionEnv(_Env):
        def step(self, action):  # pragma: no cover - failure proves dry-run moved
            raise AssertionError("dry-run must not call env.step")

        def close(self):
            pass

        def _get_observations(self, force_update=True):
            return {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
            }

    run = getattr(runner, "run_episode", None) or getattr(runner, "main_episode", None)
    if run is None:
        pytest.skip("runner does not expose an injectable episode function")

    kwargs = {
        "task_id": 0,
        "seed": 1000,
        "output_dir": tmp_path,
        "dry_run": True,
        "env": _NoMotionEnv(),
        "arrow_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "resolution": 256,
    }
    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[10.0, 0.0, camera_width / 2], [0.0, 10.0, camera_height / 2], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            return np.asarray(depth, dtype=np.float32)

    runner.camera_utils = _CameraUtils
    runner.decode_arrow = lambda **kwargs: type(
        "Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)}
    )()
    runner.deproject_endpoint = lambda point, depth, K, T: np.asarray(
        [point[0] / 10.0, point[1] / 10.0, float(np.asarray(depth).flat[0])]
    )
    runner.build_bowl_waypoints = lambda *args: np.zeros((6, 3), dtype=np.float64)
    runner.normalized_osc_action = lambda **kwargs: np.zeros(7, dtype=np.float32)
    try:
        result = run(**kwargs)
    except TypeError:
        pytest.skip("runner episode seam differs; CLI integration covers this path")

    audit_path = result.get("audit_path") if isinstance(result, dict) else None
    candidates = [Path(audit_path)] if audit_path else list(tmp_path.glob("*.json"))
    assert candidates, "dry-run must produce a JSON audit"
    audit = json.loads(candidates[0].read_text(encoding="utf-8"))
    assert audit["dry_run"] is True
    assert audit["motion_executed"] is False
    assert audit["task_id"] == 0
    assert audit["seed"] == 1000
    assert audit["controller_input"] in {"pixels_depth_calibration", "arrow_pixels_depth_calibration"}


class _MotionEnv:
    def __init__(self):
        self.actions = []
        self._arrow_settle_diagnostics = {
            "steps": 1,
            "final_max_velocity_m_s": 0.0,
            "settled": True,
        }

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }, 0.0, False, {"is_success": True}

    def render(self, **kwargs):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def _motion_proprio():
    return {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


def test_gripper_phases_dwell_exactly_configured_steps(runner, monkeypatch):
    env = _MotionEnv()
    monkeypatch.setattr(
        runner,
        "normalized_osc_action",
        lambda **kwargs: np.r_[np.zeros(6, dtype=np.float32), float(kwargs["gripper"])],
    )
    audit = runner._run_motion(
        env,
        np.zeros((6, 3)),
        _motion_proprio(),
        phase_timeout_steps=2,
        gripper_dwell_steps=4,
        stop_after_phase="retreat",
        dry_run=False,
    )
    assert audit[2]["phase"] == "close"
    assert audit[2]["status"] == "dwell"
    assert audit[2]["steps"] == 4
    assert audit[6]["phase"] == "open"
    assert audit[6]["status"] == "dwell"
    assert audit[6]["steps"] == 4
    assert [action[-1] for action in env.actions].count(1.0) == 4
    assert [action[-1] for action in env.actions].count(-1.0) == 4


def test_partial_stop_skips_later_phases_and_evaluator(runner, monkeypatch, tmp_path: Path):
    env = _MotionEnv()
    monkeypatch.setattr(
        runner,
        "normalized_osc_action",
        lambda **kwargs: np.zeros(7, dtype=np.float32),
    )
    monkeypatch.setattr(
        runner,
        "decode_arrow",
        lambda **kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(runner, "deproject_endpoint", lambda *args: np.asarray([0.0, 0.0, 1.0]))
    monkeypatch.setattr(runner, "build_bowl_waypoints", lambda *args: np.zeros((6, 3)))
    capture = runner.CapturedRGBD(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        runner.CameraCalibration(
            "agentview", 256, 256,
            [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
            np.eye(4).tolist(),
        ),
        _motion_proprio(),
    )
    evaluator_calls = []
    audit = runner.run_episode(
        env=env,
        task_id=0,
        seed=1000,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=capture,
        dry_run=False,
        phase_timeout_steps=2,
        gripper_dwell_steps=3,
        stop_after_phase="pregrasp",
        evaluator=lambda _env: evaluator_calls.append(True) or True,
    )
    assert [record["phase"] for record in audit["phases"]] == ["pregrasp"]
    assert audit["phases"][0]["status"] == "stop"
    assert audit["phases"][0]["eef_pos_m"] == [0.0, 0.0, 0.0]
    assert audit["phases"][0]["position_error_norm_m"] == 0.0
    assert audit["phases"][0]["gripper_qpos"] == [0.0, 0.0]
    assert len(audit["phase_frames"]) == 1
    assert Path(audit["phase_frames"][0]).is_file()
    assert audit["phases"][0]["diagnostic_frame"] == audit["phase_frames"][0]
    assert len(env.actions) == 1
    assert evaluator_calls == []
    assert audit["evaluator_success"] is None
    assert audit["evaluator_read_after_action"] is False
    assert audit["full_state_machine_completed"] is False
    assert audit["offset_profile"] == runner.DEFAULT_PROFILE_NAME
    assert audit["offsets_overridden"] == {"source_grasp": False, "destination_release": False}
    assert audit["profile"]["conditions_match"] is True
    assert audit["profile_validated"] is True
    assert audit["profile"]["actual_conditions"] == {
        "task_id": 0,
        "seed": 1000,
        "resolution": 256,
        "camera_name": "agentview",
    }
    assert audit["capture_contract"]["valid"] is True
    assert audit["deprojected_visual_endpoint_world_points_m"] == {
        "source_tail": [0.0, 0.0, 1.0],
        "destination_head": [0.0, 0.0, 1.0],
    }
    np.testing.assert_allclose(
        audit["control_targets_world_m"]["source_grasp"], np.asarray(runner.DEFAULT_SOURCE_GRASP_OFFSET_M) + np.asarray([0, 0, 1])
    )


def test_invalid_gripper_dwell_is_rejected(runner):
    with pytest.raises(ValueError, match="gripper_dwell_steps must be positive"):
        runner._run_motion(
            _MotionEnv(),
            np.zeros((6, 3)),
            _motion_proprio(),
            phase_timeout_steps=2,
            gripper_dwell_steps=0,
            stop_after_phase="retreat",
            dry_run=False,
        )


def test_cli_defaults_use_verified_rim_profile(runner):
    args = runner.parse_args([])
    assert args.resolution == runner.DEFAULT_RESOLUTION == 256
    assert args.gripper_dwell_steps == runner.DEFAULT_GRIPPER_DWELL_STEPS == 20
    assert tuple(args.source_grasp_offset) == runner.DEFAULT_SOURCE_GRASP_OFFSET_M
    assert tuple(args.destination_release_offset) == runner.DEFAULT_DESTINATION_RELEASE_OFFSET_M
    assert args.stop_after_phase == "retreat"
    assert args.allow_unvalidated_profile is False
    assert runner.parse_args(["--allow-unvalidated-profile"]).allow_unvalidated_profile is True


def test_phase_snapshot_callback_runs_once_per_executed_phase_and_not_dry_run(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    waypoints = np.zeros((6, 3), dtype=np.float64)

    executed_env = _MotionEnv()
    executed_frames = []
    executed_audit = runner._run_motion(
        executed_env,
        waypoints,
        _motion_proprio(),
        phase_timeout_steps=2,
        gripper_dwell_steps=2,
        stop_after_phase="retreat",
        dry_run=False,
        phase_frame_callback=lambda phase, index: executed_frames.append((phase, index)) or f"phase_frames/{index:02d}_{phase}.png",
    )
    assert len(executed_frames) == len(runner.PHASES)
    assert [index for _phase, index in executed_frames] == list(range(len(runner.PHASES)))
    assert all("diagnostic_frame" in record for record in executed_audit)

    dry_env = _MotionEnv()
    dry_frames = []
    runner._run_motion(
        dry_env,
        waypoints,
        _motion_proprio(),
        phase_timeout_steps=2,
        gripper_dwell_steps=2,
        stop_after_phase="retreat",
        dry_run=True,
        phase_frame_callback=lambda phase, index: dry_frames.append((phase, index)),
    )
    assert dry_frames == []
    assert dry_env.actions == []


def test_phase_snapshot_falls_back_to_no_render_observation(runner, tmp_path: Path):
    class _ObservationOnlyEnv:
        def _get_observations(self, force_update=True):
            return {
                "agentview_image": np.full((8, 8, 3), 17, dtype=np.uint8),
                "agentview_depth": np.ones((8, 8), dtype=np.float32),
            }

    path = runner._save_phase_snapshot(
        _ObservationOnlyEnv(), tmp_path, 0, "pregrasp", width=8, height=8
    )
    assert path.as_posix().endswith("phase_frames/00_pregrasp.png")
    assert path.is_file()


def test_settle_libero_env_calls_frozen_physics_on_inner_env(runner, monkeypatch):
    calls = []

    def fake_settle(inner, max_steps):
        calls.append((inner, max_steps))
        return {"steps_taken": max_steps, "final_max_vel": 0.004, "settled": True, "traces": {}}

    inner = object()
    wrapper = type("Wrapper", (), {"env": inner})()
    monkeypatch.setattr(runner, "_settle_physics", fake_settle)
    diagnostics = runner.settle_libero_env(wrapper, max_steps=100)
    assert calls == [(inner, 100)]
    assert diagnostics == {"steps": 100, "final_max_velocity_m_s": 0.004, "settled": True}
    assert wrapper._arrow_settle_diagnostics == diagnostics


def test_motion_refuses_env_without_settled_diagnostics(runner, tmp_path: Path):
    with pytest.raises(RuntimeError, match="physics was not confirmed settled"):
        runner.run_episode(
            env=object(),
            task_id=0,
            seed=1000,
            output_dir=tmp_path,
            dry_run=False,
            arrow_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            capture=runner.CapturedRGBD(
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                runner.CameraCalibration("agentview", 2, 2, np.eye(3).tolist(), np.eye(4).tolist()),
                _motion_proprio(),
            ),
        )


def _episode_capture(runner, *, size: int = 256):
    calibration = runner.CameraCalibration(
        "agentview", size, size,
        [[10.0, 0.0, size / 2], [0.0, -10.0, size / 2], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    return runner.CapturedRGBD(
        np.zeros((size, size, 3), dtype=np.uint8),
        np.ones((size, size), dtype=np.float32),
        np.ones((size, size), dtype=np.float32),
        calibration,
        _motion_proprio(),
    )


def _patch_episode_controller(runner, monkeypatch, *, point=(0.0, 0.0, 1.0)):
    monkeypatch.setattr(
        runner,
        "decode_arrow",
        lambda **kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(runner, "deproject_endpoint", lambda *args: np.asarray(point, dtype=np.float64))
    monkeypatch.setattr(runner, "build_bowl_waypoints", lambda *args: np.zeros((6, 3)))
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))


def test_profile_gate_rejects_task_seed_and_resolution_before_step(runner, monkeypatch, tmp_path: Path):
    _patch_episode_controller(runner, monkeypatch)
    for kwargs in (
        {"task_id": 1, "seed": 1000, "resolution": 256},
        {"task_id": 0, "seed": 1001, "resolution": 256},
        {"task_id": 0, "seed": 1000, "resolution": 128},
        {
            "task_id": 1,
            "seed": 1000,
            "resolution": 256,
            "source_grasp_offset": (0.0, 0.0, 0.0),
            "destination_release_offset": (0.0, 0.0, 0.0),
        },
    ):
        env = _MotionEnv()
        with pytest.raises(RuntimeError, match="outside the verified LIBERO arrow profile"):
            runner.run_episode(
                env=env,
                output_dir=tmp_path / str(kwargs),
                arrow_rgb=np.zeros((kwargs["resolution"], kwargs["resolution"], 3), dtype=np.uint8),
                capture=_episode_capture(runner, size=kwargs["resolution"]),
                dry_run=False,
                stop_after_phase="pregrasp",
                **kwargs,
            )
        assert env.actions == []


def test_unvalidated_profile_opt_in_is_audited(runner, monkeypatch, tmp_path: Path):
    _patch_episode_controller(runner, monkeypatch)
    audit = runner.run_episode(
        env=_MotionEnv(),
        task_id=1,
        seed=1001,
        resolution=256,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_episode_capture(runner),
        dry_run=False,
        stop_after_phase="pregrasp",
        allow_unvalidated_profile=True,
    )
    assert audit["profile"]["conditions_match"] is False
    assert audit["profile"]["allow_unvalidated_profile"] is True
    assert audit["allow_unvalidated_profile"] is True


def test_deprojection_receives_exact_robust_endpoint_depths_and_audit_separates_offsets(
    runner, monkeypatch, tmp_path: Path
):
    calls = []
    monkeypatch.setattr(
        runner,
        "decode_arrow",
        lambda **kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )

    def fake_deproject(point, depth, K, T):
        calls.append((np.asarray(point).tolist(), depth))
        return np.asarray([float(point[0]) / 10.0, float(point[1]) / 10.0, float(depth)])

    monkeypatch.setattr(runner, "deproject_endpoint", fake_deproject)
    monkeypatch.setattr(runner, "build_bowl_waypoints", lambda *args: np.zeros((6, 3)))
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    audit = runner.run_episode(
        env=_MotionEnv(),
        task_id=0,
        seed=1000,
        resolution=256,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_episode_capture(runner),
        dry_run=True,
    )
    assert calls == [([8.0, 8.0], 1.0), ([24.0, 24.0], 1.0)]
    assert audit["endpoint_depths_m"] == {"source_tail": 1.0, "destination_head": 1.0}
    assert audit["deprojected_visual_endpoint_world_points_m"] == {
        "source_tail": [0.8, 0.8, 1.0],
        "destination_head": [2.4, 2.4, 1.0],
    }
    np.testing.assert_allclose(
        audit["control_targets_world_m"]["source_grasp"],
        np.asarray([0.8, 0.8, 1.0]) + np.asarray(runner.DEFAULT_SOURCE_GRASP_OFFSET_M),
    )


def test_workspace_volume_rejects_adjusted_endpoint_before_step(runner, monkeypatch, tmp_path: Path):
    _patch_episode_controller(runner, monkeypatch, point=(2.0, 0.0, 1.0))
    env = _MotionEnv()
    with pytest.raises(ValueError, match="workspace validation failed"):
        runner.run_episode(
            env=env,
            task_id=0,
            seed=1000,
            resolution=256,
            output_dir=tmp_path,
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_episode_capture(runner),
            dry_run=False,
            stop_after_phase="pregrasp",
        )
    assert env.actions == []


def test_capture_contract_rejects_shape_calibration_camera_and_requested_size_mismatch(
    runner, monkeypatch, tmp_path: Path
):
    _patch_episode_controller(runner, monkeypatch)
    base = _episode_capture(runner)
    cases = []

    bad_metric = np.ones((255, 256), dtype=np.float32)
    cases.append(
        runner.CapturedRGBD(base.rgb, base.normalized_depth, bad_metric, base.calibration, base.observation)
    )
    bad_calibration = runner.CameraCalibration(
        "agentview", 255, 256, base.calibration.intrinsic,
        base.calibration.world_from_camera,
    )
    cases.append(
        runner.CapturedRGBD(base.rgb, base.normalized_depth, base.metric_depth, bad_calibration, base.observation)
    )
    bad_camera = runner.CameraCalibration(
        "sideview", 256, 256, base.calibration.intrinsic,
        base.calibration.world_from_camera,
    )
    cases.append(
        runner.CapturedRGBD(base.rgb, base.normalized_depth, base.metric_depth, bad_camera, base.observation)
    )

    for index, capture in enumerate(cases):
        with pytest.raises(ValueError, match="capture contract violation"):
            runner.run_episode(
                env=_MotionEnv(),
                task_id=0,
                seed=1000,
                resolution=256,
                output_dir=tmp_path / f"bad_{index}",
                arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                capture=capture,
                dry_run=True,
            )

    with pytest.raises(ValueError, match="capture contract violation: requested resolution"):
        runner.run_episode(
            env=_MotionEnv(),
            task_id=0,
            seed=1000,
            resolution=128,
            output_dir=tmp_path / "bad_requested_size",
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=base,
            dry_run=True,
        )


def test_large_clearance_rejects_waypoint_before_first_step(runner, monkeypatch, tmp_path: Path):
    # Earlier dependency-light tests replace the runner seams directly; restore
    # the real waypoint builder here so the clearance path is exercised.
    import arrow_controller

    monkeypatch.setattr(runner, "build_bowl_waypoints", arrow_controller.build_bowl_waypoints)
    monkeypatch.setattr(
        runner,
        "decode_arrow",
        lambda **kwargs: type("Arrow", (), {"source_xy": (8.0, 8.0), "target_xy": (24.0, 24.0)})(),
    )
    monkeypatch.setattr(
        runner, "deproject_endpoint", lambda *args: np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    )
    env = _MotionEnv()
    with pytest.raises(ValueError, match="workspace validation failed.*waypoint"):
        runner.run_episode(
            env=env,
            task_id=0,
            seed=1000,
            resolution=256,
            output_dir=tmp_path,
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_episode_capture(runner),
            dry_run=False,
            clearance_m=5.0,
            stop_after_phase="pregrasp",
        )
    assert env.actions == []
