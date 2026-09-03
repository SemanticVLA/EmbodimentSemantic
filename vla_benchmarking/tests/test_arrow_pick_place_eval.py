"""Contract tests for the arrow-only LIBERO pick/place runner.

These tests intentionally use small fakes rather than importing LIBERO at
collection time.  The runner is expected to keep simulator construction and
the arrow controller behind injectable seams so geometry can be tested on a
machine without MuJoCo or GPU dependencies.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
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
    depth_encoding = "normalized"

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


def test_post_lift_retention_uses_only_lift_proprioception(runner):
    trace = [
        {"phase": "lift", "gripper_qpos": [0.004, -0.004]},
        {"phase": "lift", "gripper_qpos": [0.003, -0.003]},
    ]
    decision = runner._post_lift_retention_decision(
        trace, {"gripper_qpos": np.asarray([0.003, -0.003])}, 0.0015
    )
    assert decision["decision"] == "retained"
    assert decision["retained"] is True
    assert decision["sample_count"] == 3


def test_post_lift_retention_fails_closed_when_qpos_is_unobservable(runner):
    decision = runner._post_lift_retention_decision([], {}, 0.0015)
    assert decision["decision"] == "unobservable"
    assert decision["retained"] is False


def test_capture_owns_renderer_buffers_before_second_observation_read(runner, monkeypatch):
    class _ReusingEnv(_Env):
        def render(self, camera_name, width, height, depth=False):
            self.render_calls.append((camera_name, width, height, depth))
            self.render_rgb = np.full((height, width, 3), 7, dtype=np.uint8)
            self.render_depth = np.full((height, width), 0.5, dtype=np.float32)
            return (self.render_rgb, self.render_depth) if depth else self.render_rgb

        def _get_observations(self, force_update=True):
            # Model a renderer/observation implementation that reuses its
            # backing arrays for the subsequent proprioception read.
            self.render_rgb[...] = 255
            self.render_depth[...] = 0.0
            return {"robot0_eef_pos": np.zeros(3)}

    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray(
                [[10.0, 0.0, camera_width / 2],
                 [0.0, 10.0, camera_height / 2],
                 [0.0, 0.0, 1.0]]
            )

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            return np.asarray(depth, dtype=np.float32)

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    capture = runner.capture_agentview(_ReusingEnv(), resolution=4)
    np.testing.assert_array_equal(capture.rgb, 7)
    np.testing.assert_array_equal(capture.normalized_depth, 0.5)


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
        encoding="normalized",
    )
    np.testing.assert_allclose(converted, [[0.5]])
    assert calls and calls[0][1] is normalized


def test_metric_depth_conversion_preserves_already_metric_render(runner, monkeypatch):
    calls = []

    class _CameraUtils:
        @staticmethod
        def get_real_depth_map(sim, depth):
            calls.append(depth)
            raise AssertionError("metric render must not be converted twice")

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    source = np.asarray([[0.25, 0.75]], dtype=np.float32)
    converted = runner.normalized_depth_to_metric(_Sim(), source, encoding="metric")
    np.testing.assert_array_equal(converted, source)
    assert calls == []


def test_depth_encoding_unknown_fails_closed_and_outlier_does_not_select_units(
    runner, monkeypatch
):
    with pytest.raises(ValueError, match="encoding is unknown"):
        runner.normalized_depth_to_metric(_Sim(), np.asarray([[0.5]], dtype=np.float32))

    calls = []

    class _CameraUtils:
        @staticmethod
        def get_real_depth_map(sim, depth):
            calls.append(np.asarray(depth).copy())
            return np.asarray(depth, dtype=np.float32) * 2.0

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    normalized_with_outlier = np.asarray([[0.99, 7.0]], dtype=np.float32)
    converted = runner.normalized_depth_to_metric(
        _Sim(), normalized_with_outlier, encoding="normalized"
    )
    np.testing.assert_allclose(converted, [[1.98, np.nan]], equal_nan=True)
    assert len(calls) == 1


def test_metric_depth_conversion_rejects_negative_or_empty_depth(runner, monkeypatch):
    with pytest.raises(ValueError, match="negative finite"):
        runner.normalized_depth_to_metric(_Sim(), np.asarray([[-1.0]], dtype=np.float32), encoding="metric")
    with pytest.raises(ValueError, match="no positive finite"):
        runner.normalized_depth_to_metric(_Sim(), np.zeros((1, 1), dtype=np.float32), encoding="metric")


def test_capture_audits_already_metric_depth_mode(runner, monkeypatch):
    class _MetricEnv(_Env):
        depth_encoding = "metric"

        def render(self, camera_name, width, height, depth=False):
            self.render_calls.append((camera_name, width, height, depth))
            return (
                np.zeros((height, width, 3), dtype=np.uint8),
                np.full((height, width), 2.5, dtype=np.float32),
            ) if depth else np.zeros((height, width, 3), dtype=np.uint8)

    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[10.0, 0.0, camera_width / 2], [0.0, 10.0, camera_height / 2], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            raise AssertionError("already-metric depth must bypass normalized conversion")

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    capture = runner.capture_agentview(_MetricEnv(), resolution=8)
    assert capture.depth_conversion_mode == "metric"
    np.testing.assert_array_equal(capture.metric_depth, 2.5)


def test_render_producer_requires_explicit_depth_encoding(runner, monkeypatch):
    class _RobosuiteRenderBase:
        sim = _Sim()

        def render(self, camera_name, width, height, depth=False):
            return (
                np.zeros((height, width, 3), dtype=np.uint8),
                np.full((height, width), 0.5, dtype=np.float32),
            ) if depth else np.zeros((height, width, 3), dtype=np.uint8)

    RobosuiteRender = type(
        "RobosuiteRender", (_RobosuiteRenderBase,), {"__module__": "robosuite.fake"}
    )
    with pytest.raises(ValueError, match="depth producer render.*unknown"):
        runner.capture_agentview(RobosuiteRender(), resolution=2)

    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[10.0, 0.0, camera_width / 2], [0.0, 10.0, camera_height / 2], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            assert np.all((depth >= 0) & (depth <= 1))
            return depth + 1.0

    Declared = type(
        "DeclaredRobosuiteRender", (RobosuiteRender,), {"depth_encoding": "normalized"}
    )
    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    capture = runner.capture_agentview(Declared(), resolution=2)
    assert capture.depth_conversion_mode == "normalized"


def test_normalized_depth_outliers_are_masked_for_conversion_but_raw_is_preserved(
    runner, monkeypatch, tmp_path: Path
):
    class _OutlierEnv(_Env):
        depth_encoding = "normalized"

        def render(self, camera_name, width, height, depth=False):
            self.render_calls.append((camera_name, width, height, depth))
            raw = np.asarray([[0.5, -2.0], [np.nan, 1.8e34]], dtype=np.float32)
            return (np.zeros((height, width, 3), dtype=np.uint8), raw) if depth else np.zeros((height, width, 3), dtype=np.uint8)

    seen = []

    class _CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
            return np.asarray([[10.0, 0.0, camera_width / 2], [0.0, 10.0, camera_height / 2], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            seen.append(np.asarray(depth).copy())
            assert np.isfinite(depth).all()
            assert np.all((depth >= 0.0) & (depth <= 1.0))
            return np.asarray(depth, dtype=np.float32) + 1.0

    monkeypatch.setattr(runner, "camera_utils", _CameraUtils, raising=False)
    capture = runner.capture_agentview(_OutlierEnv(), resolution=2)
    assert capture.depth_conversion_mode == "normalized_masked"
    assert capture.depth_sanitization["masked_pixel_count"] == 3
    assert capture.depth_sanitization["fallback_value"] == 0.5
    assert np.isfinite(capture.metric_depth[0, 0])
    assert np.isnan(capture.metric_depth[1, 0])
    assert np.isnan(capture.metric_depth[1, 1])
    assert np.isnan(capture.raw_depth[1, 0])
    assert float(capture.raw_depth[1, 1]) == np.float32(1.8e34)
    np.testing.assert_allclose(seen[0], [[0.5, 0.5], [0.5, 0.5]])

    paths = runner._save_capture(capture, np.zeros((2, 2, 3), dtype=np.uint8), tmp_path)
    assert "normalized_depth" in paths
    saved_raw = np.load(paths["normalized_depth"])
    assert np.array_equal(saved_raw, capture.raw_depth, equal_nan=True)
    contract = runner.validate_capture_contract(capture, resolution=2)
    assert contract["raw_depth_shape"] == [2, 2]
    assert contract["depth_input_shape"] == [2, 2]

    metric_capture = runner.CapturedRGBD(
        capture.rgb,
        np.full((2, 2), 0.5, dtype=np.float32),
        np.full((2, 2), 0.5, dtype=np.float32),
        capture.calibration,
        capture.observation,
        "metric",
    )
    metric_paths = runner._save_capture(
        metric_capture, np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "metric"
    )
    assert "normalized_depth" not in metric_paths
    assert "metric_depth_input" in metric_paths

    unknown_capture = runner.CapturedRGBD(
        capture.rgb,
        capture.raw_depth,
        capture.metric_depth,
        capture.calibration,
        capture.observation,
    )
    unknown_paths = runner._save_capture(
        unknown_capture, np.zeros((2, 2, 3), dtype=np.uint8), tmp_path / "unknown"
    )
    assert "normalized_depth" not in unknown_paths
    assert unknown_paths["depth_input"].endswith("agentview_depth_unknown_input.npy")


def test_depth_sanitization_policy_rejects_near_total_invalid_map(runner):
    capture = runner.CapturedRGBD(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        np.full((8, 8), np.nan, dtype=np.float32),
        runner.CameraCalibration("agentview", 8, 8, np.eye(3).tolist(), np.eye(4).tolist()),
        depth_conversion_mode="normalized_masked",
        depth_sanitization={"masked_pixel_count": 63, "total_pixel_count": 64, "masked_fraction": 63 / 64},
    )
    policy = runner.assess_depth_sanitization_for_motion(capture, ((4.0, 4.0), (5.0, 5.0)))
    assert policy["status"] == "rejected"
    assert policy["rejection_reason"] == "masked_fraction_exceeds_limit"


def test_depth_sanitization_policy_requires_valid_endpoint_neighborhood_and_allows_sparse_away(
    runner,
):
    calibration = runner.CameraCalibration("agentview", 8, 8, np.eye(3).tolist(), np.eye(4).tolist())
    invalid_endpoint = runner.CapturedRGBD(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        np.full((8, 8), np.nan, dtype=np.float32),
        calibration,
        depth_conversion_mode="normalized_masked",
        depth_sanitization={"masked_pixel_count": 63, "total_pixel_count": 64, "masked_fraction": 63 / 64},
    )
    policy = runner.assess_depth_sanitization_for_motion(invalid_endpoint, ((4.0, 4.0),))
    assert policy["endpoint_patch_valid"] == [False]

    metric = np.ones((8, 8), dtype=np.float32)
    metric[0, 0] = np.nan
    sparse = runner.CapturedRGBD(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8), dtype=np.float32),
        metric,
        calibration,
        depth_conversion_mode="normalized_masked",
        depth_sanitization={"masked_pixel_count": 1, "total_pixel_count": 64, "masked_fraction": 1 / 64},
    )
    sparse_policy = runner.assess_depth_sanitization_for_motion(sparse, ((4.0, 4.0), (5.0, 5.0)))
    assert sparse_policy["status"] == "passed"
    assert sparse_policy["endpoint_patch_valid"] == [True, True]


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
    assert env._arrow_capture_contract == audit["capture_contract"]
    assert env._arrow_depth_sanitization_policy["status"] == "not_applicable"
    assert env._arrow_endpoints_uv == {
        "source_tail": [8.0, 8.0],
        "destination_head": [24.0, 24.0],
    }
    assert env._arrow_decode_audit["source"] == "decode_arrow"
    assert env._arrow_input_arrow_audit["controller_input"] == "caller_supplied_one_arrow"
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


def test_motion_timeout_keeps_partial_phase_audit(runner, monkeypatch):
    env = _MotionEnv()
    monkeypatch.setattr(
        runner,
        "normalized_osc_action",
        lambda **kwargs: np.zeros(7, dtype=np.float32),
    )
    with pytest.raises(TimeoutError, match="phase pregrasp exceeded 1 steps"):
        runner._run_motion(
            env,
            np.ones((6, 3), dtype=np.float64),
            _motion_proprio(),
            phase_timeout_steps=1,
            gripper_dwell_steps=2,
            stop_after_phase="retreat",
            dry_run=False,
        )
    assert env._arrow_phase_audit[-1]["phase"] == "pregrasp"
    assert env._arrow_phase_audit[-1]["status"] == "timeout"
    assert env._arrow_phase_audit[-1]["steps"] == 1


def test_motion_trace_is_bounded_and_persisted(runner, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    env = _MotionEnv()
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env,
            np.ones((6, 3), dtype=np.float64),
            _motion_proprio(),
            phase_timeout_steps=4,
            gripper_dwell_steps=2,
            stop_after_phase="pregrasp",
            dry_run=False,
            stall_window_steps=0,
            motion_trace_max_steps=2,
            motion_trace_path=tmp_path / "trace.json",
        )
    assert len(env._arrow_motion_trace) == 2
    assert env._arrow_motion_trace_truncated is True
    persisted = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert persisted["truncated"] is True
    assert len(persisted["steps"]) == 2
    assert {"eef_pos_m", "gripper_qpos", "residual_vector_m"} <= set(persisted["steps"][0])


def test_motion_failure_snapshot_callback_runs_for_timeout(runner, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    snapshots = []

    def save_snapshot(phase, index, record):
        path = tmp_path / f"failure_{index}_{phase}.json"
        path.write_text(json.dumps({"phase": phase, "status": record["status"]}), encoding="utf-8")
        snapshots.append(path)
        return path

    env = _MotionEnv()
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env,
            np.ones((6, 3), dtype=np.float64),
            _motion_proprio(),
            phase_timeout_steps=1,
            gripper_dwell_steps=2,
            stop_after_phase="pregrasp",
            dry_run=False,
            failure_snapshot_callback=save_snapshot,
        )
    assert len(snapshots) == 1 and snapshots[0].is_file()
    assert env._arrow_phase_audit[-1]["failure_snapshot"] == snapshots[0].as_posix()


def test_post_lift_retention_gate_rejects_before_preplace(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    env = _MotionEnv()
    gate_calls = []

    def retention_gate(record, proprio):
        gate_calls.append((record["phase"], dict(proprio)))
        return {"retained": False, "source": "pure_proprio_helper"}

    with pytest.raises(runner._GraspSearchRequested, match="post_lift_retention"):
        runner._run_motion(
            env,
            np.zeros((6, 3), dtype=np.float64),
            _motion_proprio(),
            phase_timeout_steps=2,
            gripper_dwell_steps=2,
            stop_after_phase="retreat",
            dry_run=False,
            post_lift_retention_gate=retention_gate,
        )
    assert gate_calls and gate_calls[0][0] == "lift"
    assert env._arrow_phase_audit[-1]["retention_gate"]["retained"] is False
    assert [record["phase"] for record in env._arrow_phase_audit] == [
        "pregrasp", "descend", "close", "lift"
    ]


def test_legacy_motion_trace_and_retention_are_disabled_by_default(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    env = _MotionEnv()
    audit = runner._run_motion(
        env,
        np.zeros((6, 3), dtype=np.float64),
        _motion_proprio(),
        phase_timeout_steps=2,
        gripper_dwell_steps=2,
        stop_after_phase="pregrasp",
        dry_run=True,
    )
    assert [item["phase"] for item in audit] == ["pregrasp"]
    assert env._arrow_motion_trace == []
    assert not any("retention_gate" in item for item in audit)


def test_canary_video_is_opt_in_and_records_artifact_hash(runner, monkeypatch, tmp_path: Path):
    _patch_episode_controller(runner, monkeypatch)
    canary_dir = tmp_path / "suite_v10" / "variant_zg" / "task_0" / "episode_0"
    audit = runner.run_episode(
        env=_MotionEnv(),
        task_id=0,
        seed=1000,
        output_dir=tmp_path / "episode",
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_episode_capture(runner),
        dry_run=False,
        stop_after_phase="retreat",
        canary_video_dir=canary_dir,
        canary_video_max_frames=4,
        evaluator=lambda _env: True,
    )
    video = audit["canary_video"]
    assert video["enabled"] is True
    assert video["frame_count"] == 4
    assert video["truncated"] is True
    assert Path(video["video_path"]).is_file()
    assert len(video["video_sha256"]) == 64
    assert json.loads((canary_dir / "canary_video_manifest.json").read_text(encoding="utf-8"))["truncated"] is True
    assert audit["evaluator_success"] is True


def test_cli_defaults_use_verified_rim_profile(runner):
    args = runner.parse_args([])
    assert args.resolution == runner.DEFAULT_RESOLUTION == 256
    assert args.gripper_dwell_steps == runner.DEFAULT_GRIPPER_DWELL_STEPS == 20
    assert tuple(args.source_grasp_offset) == runner.DEFAULT_SOURCE_GRASP_OFFSET_M
    assert tuple(args.destination_release_offset) == runner.DEFAULT_DESTINATION_RELEASE_OFFSET_M
    assert args.stop_after_phase == "retreat"
    assert args.allow_unvalidated_profile is False
    assert runner.parse_args(["--allow-unvalidated-profile"]).allow_unvalidated_profile is True


def test_controller_variant_provenance_is_canonical_and_suite_scoped(runner):
    vanilla = runner.ControllerVariantConfig(suite_mode="vanilla")
    sealed = runner.ControllerVariantConfig(suite_mode="sealed_randomized")
    assert vanilla.canonical_json() == vanilla.canonical_json()
    assert vanilla.config_hash == vanilla.hash
    assert vanilla.provenance()["config_hash"] == vanilla.config_hash
    assert vanilla.config_hash != sealed.config_hash
    with pytest.raises(ValueError, match="suite_mode"):
        runner.ControllerVariantConfig(suite_mode="unknown")


def test_retired_controller_variants_are_rejected_before_execution(runner):
    """Historical experiments remain readable in archives, never executable."""
    for retired_name in (
        "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v1",
        "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v2",
        "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v3",
        "libero_spatial_akita_bowl_agentview_candidate_lowerq_gain_v4",
        "libero_spatial_akita_bowl_agentview_candidate_visible_anchor_v5",
        "libero_spatial_akita_bowl_agentview_candidate_directional_approach_v6",
        "libero_spatial_akita_bowl_agentview_candidate_patient_control_v7",
        "libero_spatial_akita_bowl_agentview_candidate_grasp_retry_v8",
    ):
        with pytest.raises(Exception, match="only the active v9d controller"):
            runner._resolve_controller_variant(retired_name, suite_mode="vanilla")

    active = runner._resolve_controller_variant("default", suite_mode="vanilla")
    assert active.name == runner.DEFAULT_PROFILE_NAME
    assert active.endpoint_depth_statistic == "lower_quantile"
    assert active.endpoint_depth_quantile == pytest.approx(0.25)
    assert active.grasp_search is not None and active.grasp_search.enabled is True


def test_osc_position_scale_changes_only_translational_scales(runner, monkeypatch):
    seen = {}

    def fake_osc(**kwargs):
        seen["scales"] = tuple(kwargs["scales"])
        return np.zeros(7, dtype=np.float32)

    monkeypatch.setattr(runner, "normalized_osc_action", fake_osc)
    action = runner.normalized_action_for_waypoint(
        runner._proprioception(_motion_proprio()), np.zeros(3), gripper=0.0,
        osc_position_scale_m=0.035,
    )
    assert action.shape == (7,)
    assert seen["scales"] == pytest.approx((0.035, 0.035, 0.035, 0.5, 0.5, 0.5))


def test_visible_inset_arrow_anchor_moves_only_clipped_subject_center(runner):
    bboxes = {
        "akita_black_bowl_1": [0.0, 107.0, 59.0, 173.0],
        "plate_1": [161.0, 37.0, 223.0, 94.0],
    }
    centered = runner._arrow_anchor_bboxes(
        bboxes, subject="akita_black_bowl_1", image_shape=(256, 256), policy="bbox_center"
    )
    inset = runner._arrow_anchor_bboxes(
        bboxes, subject="akita_black_bowl_1", image_shape=(256, 256), policy="visible_inset"
    )
    assert centered["akita_black_bowl_1"] == [0.0, 107.0, 59.0, 173.0]
    assert inset["akita_black_bowl_1"] == pytest.approx([-29.5, 107.0, 59.0, 173.0])
    assert inset["plate_1"] == pytest.approx(centered["plate_1"])


def test_directional_pregrasp_offset_uses_only_horizontal_arrow_direction(runner):
    waypoints = np.zeros((6, 3), dtype=np.float64)
    adjusted = runner._apply_directional_pregrasp_offset(
        waypoints,
        source_visual_point=(1.0, 2.0, 3.0),
        destination_visual_point=(1.0, 2.2, 4.0),
        offset_m=0.04,
    )
    assert adjusted[0].tolist() == pytest.approx([0.0, 0.04, 0.0])
    assert np.all(adjusted[1:] == 0.0)


def test_gripper_contact_gate_uses_close_proprioception_only(runner):
    assert runner._gripper_contact_likely(
        [{"phase": "close", "gripper_qpos": [0.0004, -0.0007]}], 0.0015
    )
    assert not runner._gripper_contact_likely(
        [{"phase": "close", "gripper_qpos": [0.004, -0.004]}], 0.0015
    )
    assert not runner._gripper_contact_likely([{"phase": "close"}], 0.0015)


def test_endpoint_depth_statistic_selection_is_deterministic(runner):
    depth = np.ones((5, 5), dtype=np.float32)
    depth[0:3, 0:3] = np.linspace(0.2, 0.8, 9, dtype=np.float32).reshape(3, 3)
    median = runner._depth_at(depth, (2, 2), statistic="median")
    lower = runner._depth_at(depth, (2, 2), statistic="lower_quantile", quantile=0.25)
    nearest = runner._depth_at(depth, (2, 2), statistic="nearest_valid")
    assert median == pytest.approx(1.0)
    assert lower < median
    assert nearest == pytest.approx(0.8)


def test_endpoint_depth_statistics_and_candidate_tolerance_are_audited(
    runner, monkeypatch, tmp_path: Path
):
    _patch_episode_controller(runner, monkeypatch)
    capture = _episode_capture(runner)
    audit = runner.run_episode(
        env=_MotionEnv(),
        task_id=0,
        seed=1000,
        resolution=256,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=capture,
        dry_run=True,
    )
    assert audit["endpoint_depth_statistics"]["source_tail"]["statistic"] == "lower_quantile"
    assert audit["controller_variant"]["canonical"]["endpoint_depth_quantile"] == 0.25
    assert audit["phases"][1]["policy"]["tolerance_m"] == 0.015


def test_v3_tolerance_applies_to_all_positional_phases(runner, monkeypatch, tmp_path: Path):
    _patch_episode_controller(runner, monkeypatch)
    with pytest.raises(Exception, match="only the active v9d controller"):
        runner.run_episode(
            env=_MotionEnv(),
            task_id=0,
            seed=1000,
            resolution=256,
            output_dir=tmp_path,
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=_episode_capture(runner),
            dry_run=True,
            controller_variant="libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v3",
        )


def test_v2_mask_gate_and_workspace_override_are_audited(
    runner, monkeypatch, tmp_path: Path
):
    del monkeypatch, tmp_path
    with pytest.raises(Exception, match="only the active v9d controller"):
        runner._resolve_controller_variant(
            "libero_spatial_akita_bowl_agentview_candidate_lowerq_relaxed_v2", suite_mode="vanilla"
        )
    active = runner._resolve_controller_variant("default", suite_mode="vanilla")
    assert active.max_mask_fraction_for_motion == pytest.approx(0.4)
    assert active.workspace_bounds_m["z"] == [0.0, 1.8]


def test_randomization_dimensions_are_task_actual_not_suite_wide(runner):
    swapped = runner._randomization_dimensions(
        "sealed_randomized", ["plate_1"], {"applied": ["bowl_1", "cookies_1"]}
    )
    removal_only = runner._randomization_dimensions(
        "sealed_randomized", ["plate_1"], {"applied": []}
    )
    vanilla = runner._randomization_dimensions(
        "vanilla", ["plate_1"], {"applied": ["bowl_1", "cookies_1"]}
    )
    assert swapped == {"scene_layout": True, "object_removal": True, "prompt_variant": False}
    assert removal_only == {"scene_layout": False, "object_removal": True, "prompt_variant": False}
    assert vanilla == {"scene_layout": False, "object_removal": False, "prompt_variant": False}


@pytest.mark.parametrize(
    ("task_id", "scene_layout"),
    ((0, False), (2, True)),
)
def test_sealed_environment_audit_declares_task_specific_scene_only_and_prompt_not_applicable(
    runner, monkeypatch, tmp_path: Path, task_id: int, scene_layout: bool
):
    _patch_episode_controller(runner, monkeypatch)
    env = _MotionEnv()
    env._arrow_environment_audit = {
        "suite_mode": "sealed_randomized",
        "scene_randomization": "sealed_randomized",
        "randomization_dimensions": {
            "scene_layout": scene_layout,
            "object_removal": True,
            "prompt_variant": False,
        },
        "prompt_provenance": "not_applicable_direct_runner",
    }
    audit = runner.run_episode(
        env=env,
        task_id=task_id,
        seed=1000 + task_id,
        resolution=256,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=_episode_capture(runner),
        dry_run=True,
        suite_mode="sealed_randomized",
    )
    environment = audit["environment_audit"]
    assert environment["scene_randomization"] == "sealed_randomized"
    assert environment["randomization_dimensions"]["prompt_variant"] is False
    assert environment["prompt_provenance"] == "not_applicable_direct_runner"


def test_direct_swap_adapter_reports_requested_and_applied_labels(runner, monkeypatch):
    calls = []
    fake_preview = types.ModuleType("preview_visual_arrows")

    def apply_task_swaps(env, task_id):
        calls.append((env, task_id))

    fake_preview._apply_task_swaps = apply_task_swaps
    monkeypatch.setitem(sys.modules, "preview_visual_arrows", fake_preview)
    result = runner._apply_direct_swaps(object(), 2)
    assert calls and calls[0][1] == 2
    assert result["requested"] == [["akita_black_bowl_2", "cookies_1"]]
    assert result["applied"] == ["akita_black_bowl_2", "cookies_1"]
    assert result["skipped"] == []


def test_cli_exposes_explicit_suite_and_stall_recovery_controls(runner):
    args = runner.parse_args(["--suite-mode", "sealed_randomized", "--stall-window-steps", "4", "--recovery-attempts", "2"])
    assert args.suite_mode == "sealed_randomized"
    assert args.stall_window_steps == 4
    assert args.recovery_attempts == 2


def test_stall_detection_is_phase_bounded_and_records_reason(runner, monkeypatch):
    env = _MotionEnv()
    monkeypatch.setattr(
        runner,
        "normalized_osc_action",
        lambda **kwargs: np.zeros(7, dtype=np.float32),
    )
    with pytest.raises(TimeoutError, match="phase pregrasp stalled"):
        runner._run_motion(
            env,
            np.ones((6, 3), dtype=np.float64),
            _motion_proprio(),
            phase_timeout_steps=8,
            gripper_dwell_steps=2,
            stop_after_phase="retreat",
            dry_run=False,
            stall_window_steps=2,
            stall_delta_m=1e-9,
        )
    assert env._arrow_phase_audit[-1]["status"] == "stall"
    assert env._arrow_phase_audit[-1]["stall_window_steps"] == 2


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
        depth_encoding = "normalized"

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
    # The frozen v9d default enables its RGB-D retry policy.  Keep these
    # controller-fake tests focused on the phase under test by supplying a
    # deterministic, in-workspace retry candidate instead of running the real
    # region planner on the synthetic all-ones depth fixture.
    monkeypatch.setattr(
        runner,
        "derive_rgbd_region_grasp_candidates",
        lambda *args, **kwargs: (
            np.asarray(point, dtype=np.float64).reshape(1, 3),
            {
                "selected_profile_quantiles": [0.8],
                "selected_pixels_xy": [[8, 8]],
            },
        ),
    )
    monkeypatch.setattr(runner, "build_bowl_waypoints", lambda *args: np.zeros((6, 3)))
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    # v9d's bounded retry asks for a fresh proprioception sample before its
    # open/retreat reset.  The tiny fake environment has no observation hook,
    # so provide the contract-level sample explicitly.
    monkeypatch.setattr(runner, "_raw_observation", lambda _env: _motion_proprio())
    # Keep generic motion/video tests out of the v9d retry trigger itself; the
    # dedicated retry test below supplies its own explicit trigger.
    monkeypatch.setattr(runner, "_empty_gripper_likely", lambda *args, **kwargs: False)


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


def test_visual_endpoint_diagnostics_are_published_before_workspace_failure(
    runner, monkeypatch, tmp_path: Path
):
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
    assert "source_tail" in env._arrow_endpoint_depths_m
    assert env._arrow_endpoint_depth_statistics["source_tail"]["statistic"] == "lower_quantile"
    assert env._arrow_deprojected_visual_endpoint_world_points_m["source_tail"] == [2.0, 0.0, 1.0]
    assert env._arrow_control_targets_world_m["source_grasp"] == pytest.approx(
        [2.0146, 0.0432, 1.0244]
    )
    assert env._arrow_workspace_validation["status"] == "pending"
    assert "waypoint_0" in env._arrow_workspace_validation["points"]


def test_run_motion_preserves_gripper_timeout_and_marks_motion(runner, monkeypatch):
    class _GripperTimeoutEnv(_MotionEnv):
        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            if float(np.asarray(action)[-1]) > 0.5:
                raise TimeoutError("gripper transport timeout")
            return super().step(action)

    monkeypatch.setattr(
        runner,
        "normalized_osc_action",
        lambda **kwargs: np.r_[np.zeros(6, dtype=np.float32), float(kwargs["gripper"])],
    )
    env = _GripperTimeoutEnv()
    with pytest.raises(TimeoutError, match="gripper transport timeout"):
        runner._run_motion(
            env,
            np.zeros((6, 3)),
            _motion_proprio(),
            phase_timeout_steps=2,
            gripper_dwell_steps=3,
            stop_after_phase="retreat",
            dry_run=False,
            stall_window_steps=0,
        )
    assert env._arrow_motion_began is True
    assert env._arrow_phase_audit[-1]["phase"] == "close"
    assert env._arrow_phase_audit[-1]["status"] == "timeout"
    assert env._arrow_phase_audit[-1]["steps"] == 1


def test_frozen_v9d_disables_legacy_recovery_overrides(runner):
    active = runner._resolve_controller_variant("default", suite_mode="vanilla")
    assert active.recovery_attempts == 0
    assert active.recovery_steps == 0


def test_external_v9_config_inheritance_and_hash_are_stable(runner, tmp_path: Path):
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text(json.dumps({"name": "base", "phase_timeout_steps": 12, "grasp_search": {"enabled": False}}), encoding="utf-8")
    child.write_text(json.dumps({"extends": "base.json", "name": "child", "grasp_search": {"enabled": True, "offsets_m": [[0, 0, -0.01]], "max_attempts": 1}}), encoding="utf-8")
    expanded = runner.load_controller_config(child)
    variant = runner.controller_variant_from_config(expanded)
    assert expanded["phase_timeout_steps"] == 12
    assert "extends" not in expanded
    assert expanded["config_hash"] == runner.controller_config_hash(expanded)
    assert variant.grasp_search.enabled is True
    assert variant.canonical()["grasp_search"]["offsets_m"] == [[0.0, 0.0, -0.01]]


def test_v9d_rgbd_region_config_is_the_only_active_policy(runner):
    config_root = Path(runner.__file__).parent / "controller_configs"
    v9d_expanded = runner.load_controller_config(config_root / "v9d_rgbd_region_grasp_search.json")
    v9d = runner.controller_variant_from_config(v9d_expanded)
    assert v9d.grasp_search.strategy == "rgbd_region"
    assert v9d.grasp_search.offsets_m == ()
    assert v9d.grasp_search.max_attempts == 3
    assert v9d.grasp_search.region_candidate_height_quantiles == ()
    assert v9d.micro_correction.enabled is False
    for retired_name in (
        "v9e_rgbd_region_with_micro_correction.json",
        "v9f_rgbd_region_height_sweep.json",
        "v10_zg_grasp_only.json",
    ):
        with pytest.raises(Exception, match="retired|not found"):
            runner.load_controller_config(config_root / retired_name)


def test_rgbd_region_retry_uses_only_capture_geometry_and_proprioception(
    runner, monkeypatch, tmp_path: Path
):
    _patch_episode_controller(runner, monkeypatch)
    generated_target = np.asarray((0.04, 0.03, 1.02), dtype=np.float64)
    candidate_calls = []

    def fake_candidates(clean_rgb, depth_m, K, T_world_camera, source_uv, profile_target, **kwargs):
        candidate_calls.append({
            "rgb": np.asarray(clean_rgb).copy(),
            "depth": np.asarray(depth_m).copy(),
            "K": np.asarray(K).copy(),
            "T": np.asarray(T_world_camera).copy(),
            "source_uv": tuple(source_uv),
            "profile_target": np.asarray(profile_target).copy(),
            "kwargs": dict(kwargs),
        })
        return generated_target.reshape(1, 3), {
            "method": "test_rgbd_region",
            "selected_profile_quantiles": [0.8],
            "selected_pixels_xy": [[9, 8]],
        }

    monkeypatch.setattr(runner, "derive_rgbd_region_grasp_candidates", fake_candidates)
    monkeypatch.setattr(runner, "_raw_observation", lambda env: _motion_proprio())
    waypoint_sources = []

    def fake_waypoints(source, destination, *args, **kwargs):
        waypoint_sources.append(np.asarray(source, dtype=np.float64).copy())
        return np.tile(np.asarray(source, dtype=np.float64), (6, 1))

    monkeypatch.setattr(runner, "build_bowl_waypoints", fake_waypoints)
    motion_calls = []

    def fake_motion(env, waypoints, observation, **kwargs):
        motion_calls.append((np.asarray(waypoints).copy(), dict(kwargs)))
        if len(motion_calls) == 1:
            env._arrow_phase_audit = [
                {"phase": "close", "status": "dwell", "steps": 20, "gripper_qpos": [0.0, 0.0]}
            ]
            raise runner._GraspSearchRequested("empty_gripper_likely")
        if kwargs.get("start_phase") == "open":
            return [
                {"phase": "open", "status": "dwell", "steps": 2},
                {"phase": "retreat", "status": "reached", "steps": 2},
            ]
        if kwargs.get("stop_after_phase") == "lift":
            return [
                {"phase": "close", "status": "dwell", "steps": 2, "gripper_qpos": [0.003, -0.003]},
                {"phase": "lift", "status": "reached", "steps": 2},
            ]
        return [
            {"phase": "preplace", "status": "reached", "steps": 2},
            {"phase": "descend_place", "status": "reached", "steps": 2},
            {"phase": "open", "status": "dwell", "steps": 2},
            {"phase": "retreat", "status": "reached", "steps": 2},
        ]

    monkeypatch.setattr(runner, "_run_motion", fake_motion)
    variant = runner.ControllerVariantConfig(
        name="test_rgbd_region",
        phase_timeout_steps=160,
        grasp_search=runner.GraspSearchPolicy(
            enabled=True,
            strategy="rgbd_region",
            offsets_m=(),
            max_attempts=1,
            phase_timeout_steps=160,
            max_actions=100,
        ),
    )
    # This is an isolated retry-mechanics test, not an executable policy
    # selection.  Bypass the production v9d resolver only at this seam so the
    # historical custom fixture can exercise the bounded retry loop.
    monkeypatch.setattr(runner, "_resolve_controller_variant", lambda value, **kwargs: value)
    capture = _episode_capture(runner)
    audit = runner.run_episode(
        env=_MotionEnv(),
        task_id=0,
        seed=1000,
        resolution=256,
        output_dir=tmp_path,
        arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        capture=capture,
        dry_run=False,
        stop_after_phase="retreat",
        controller_variant=variant,
        evaluator=lambda env: True,
    )
    assert len(candidate_calls) == 1
    np.testing.assert_array_equal(candidate_calls[0]["rgb"], capture.rgb)
    np.testing.assert_array_equal(candidate_calls[0]["depth"], capture.metric_depth)
    assert candidate_calls[0]["source_uv"] == (8.0, 8.0)
    assert np.allclose(waypoint_sources[-1], generated_target)
    assert audit["evaluator_success"] is True
    assert audit["grasp_search"][0]["stage"] == "candidate_generation"
    assert audit["grasp_search"][1]["status"] == "selected"
    assert audit["grasp_search"][1]["selected_pixel_xy"] == [9, 8]


def test_external_v9_config_cycle_rejected_before_motion(runner, tmp_path: Path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"extends": "right.json"}), encoding="utf-8")
    right.write_text(json.dumps({"extends": "left.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        runner.load_controller_config(left)


def test_invalid_explicit_controller_config_fails_before_environment_build(runner, monkeypatch, tmp_path: Path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"grasp_search": {"enabled": True, "offsets_m": [[0, 0, 0]], "max_attempts": 1}, "unknown": 1}), encoding="utf-8")
    monkeypatch.setattr(runner, "build_libero_env", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("environment built")))
    with pytest.raises(ValueError, match="unknown"):
        runner.main(["--controller-config", str(invalid)])


def test_controller_timeout_subclass_preserves_raw_environment_timeout(runner, monkeypatch):
    assert issubclass(runner.ControllerMotionTimeout, TimeoutError)

    class RawTimeoutEnv(_MotionEnv):
        def step(self, action):
            raise TimeoutError("transport timeout")

    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    with pytest.raises(TimeoutError) as caught:
        runner._run_motion(
            RawTimeoutEnv(), np.zeros((6, 3)), _motion_proprio(),
            phase_timeout_steps=2, gripper_dwell_steps=2,
            stop_after_phase="pregrasp", dry_run=False,
        )
    assert type(caught.value) is TimeoutError


def test_micro_correction_event_records_trigger_and_post_residual(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    env = _MotionEnv()
    policy = runner.MicroCorrectionPolicy(
        enabled=True, phases=("descend",), plateau_window_steps=1,
        plateau_delta_m=0.0, residual_max_m=0.005, correction_gain=1.0,
        burst_steps=1, max_rounds=1, max_actions=1,
    )
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env, np.ones((6, 3)), _motion_proprio(),
            phase_timeout_steps=3, gripper_dwell_steps=2,
            stop_after_phase="descend", start_phase="descend", dry_run=False,
            micro_correction=policy,
        )
    assert env._arrow_micro_correction_audit[0]["trigger"] == "residual_plateau"
    assert env._arrow_micro_correction_audit[0]["status"] == "completed"
    assert env._arrow_micro_correction_audit[0]["post_residual_norm_m"] is not None


@pytest.mark.parametrize("field,value", [("enabled", "true"), ("max_attempts", 1.5), ("phase_timeout_steps", True)])
def test_v9_policy_rejects_weakly_typed_limits(runner, field, value):
    with pytest.raises(ValueError):
        runner.GraspSearchPolicy(**{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "object_pose"},
        {"region_min_pixels": 1.5},
        {"region_radius_m": 0.5},
        {"region_height_quantile": 1.0},
        {"region_profile_quantiles": []},
        {"region_candidate_height_quantiles": [0.7]},
    ],
)
def test_v9_dynamic_policy_rejects_invalid_geometry_contract(runner, kwargs):
    with pytest.raises(ValueError):
        runner.GraspSearchPolicy(**kwargs)


@pytest.mark.parametrize("field,value", [("enabled", 1), ("max_actions", 2.5), ("max_rounds", False)])
def test_micro_policy_rejects_weakly_typed_limits(runner, field, value):
    with pytest.raises(ValueError):
        runner.MicroCorrectionPolicy(**{field: value})


def test_micro_policy_budget_is_shared_across_phases(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))
    env = _MotionEnv()
    policy = runner.MicroCorrectionPolicy(
        enabled=True, phases=("pregrasp", "descend"), plateau_window_steps=1,
        plateau_delta_m=0.0, residual_max_m=0.005, burst_steps=1,
        max_rounds=2, max_actions=1,
    )
    with pytest.raises(runner.ControllerMotionTimeout):
        runner._run_motion(
            env, np.ones((6, 3)), _motion_proprio(), phase_timeout_steps=2,
            gripper_dwell_steps=2, stop_after_phase="descend", dry_run=False,
            micro_correction=policy,
        )
    assert len(env._arrow_micro_correction_audit) == 1


def test_correction_step_reaching_tolerance_is_charged_once(runner, monkeypatch):
    monkeypatch.setattr(runner, "normalized_osc_action", lambda **kwargs: np.zeros(7, dtype=np.float32))

    class ReachesNominalOnCorrection(_MotionEnv):
        def __init__(self):
            super().__init__()
            self.n = 0

        def step(self, action):
            self.n += 1
            pos = np.zeros(3) if self.n == 1 else np.ones(3)
            return {
                "robot0_eef_pos": pos,
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
            }, 0.0, False, {}

    env = ReachesNominalOnCorrection()
    budget = runner._ActionBudget(1)
    policy = runner.MicroCorrectionPolicy(
        enabled=True, phases=("descend",), plateau_window_steps=1,
        plateau_delta_m=0.0, residual_max_m=0.005, burst_steps=1,
        max_rounds=1, max_actions=1,
    )
    audit = runner._run_motion(
        env, np.ones((6, 3)), _motion_proprio(), phase_timeout_steps=3,
        gripper_dwell_steps=2, stop_after_phase="descend", start_phase="descend",
        dry_run=False, micro_correction=policy, micro_action_budget=budget,
    )
    assert audit[-1]["status"] == "stop"
    assert audit[-1]["position_error_norm_m"] == pytest.approx(0.0)
    assert budget.used == 1
