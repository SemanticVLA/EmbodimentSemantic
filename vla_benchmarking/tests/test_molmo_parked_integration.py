"""Independent parked/no-hover production-main integration checks."""

from __future__ import annotations

import importlib
import math
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest


runner = importlib.import_module("run_molmo_sam3_canary")
matrix = importlib.import_module("run_arrow_pick_place_matrix")
episode = importlib.import_module("run_arrow_pick_place_eval")
molmopoint = importlib.import_module("molmo_sam3.molmopoint")


@dataclass
class _Calibration:
    camera_name: str = "agentview"
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "same_sim_render_call"
    intrinsic: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0, 1.0]))
    world_from_camera: np.ndarray = field(default_factory=lambda: np.eye(4))


class _Capture:
    def __init__(self, serial: int):
        self.serial = serial
        self.rgb = np.full((8, 8, 3), serial % 255, dtype=np.uint8)
        self.metric_depth = np.ones((8, 8), dtype=np.float32)
        self.calibration = _Calibration()


class _Env:
    def __init__(self):
        self._molmo_sam3_action_count = 0


class _Worker:
    def __init__(self, *_args, **_kwargs):
        self.requests = []
        self.robot_calibration = None

    def propose(self, request):
        self.requests.append(request)
        number = len(self.requests)
        return {"candidates": [{
            "candidate_id": f"parked-candidate-{number}",
            "position_world_m": [0.1, 0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "opening_m": 0.04, "source_pixel_xy": [2.0, 2.0], "score": 1.0,
        }]}


@pytest.mark.parametrize("opening_profile, expected_preshape", [("full_open", 0), ("preshape40mm", 4)])
def test_parked_main_skips_hover_and_refreshes_retry_capture(
    tmp_path, monkeypatch, opening_profile, expected_preshape
):
    events: list[str] = []
    capture_serial = 0
    run_count = 0

    def capture(_env, *, resolution, camera_name):
        nonlocal capture_serial
        assert resolution == 8 and camera_name == runner.AGENTVIEW
        capture_serial += 1
        events.append(f"capture:{capture_serial}")
        return _Capture(capture_serial)

    def build_env(*_args, **_kwargs):
        events.append("build_env")
        return _Env()

    def calibration(_env):
        events.append("calibration")
        return SimpleNamespace(name="calibration"), np.eye(3), {
            "gripper_geometry": {"measured_opening_m": 0.04}, "passed": True,
        }

    def open_gripper(_env, **_kwargs):
        events.append("open")
        return {"status": "completed", "steps": 1}

    def hover(*_args, **_kwargs):
        pytest.fail("parked profile must not invoke observation hover")

    def preshape(_env, **_kwargs):
        events.append("preshape")
        return {"status": "completed", "final_opening_m": 0.04}

    def run_episode(**_kwargs):
        nonlocal run_count
        run_count += 1
        events.append(f"run:{run_count}")
        if run_count == 1:
            raise RuntimeError("synthetic contact failure")
        return {"status": "placed", "evaluator_success": True, "phases": []}

    def fake_matrix_run(**kwargs):
        nonlocal run_count
        run_count = 0
        env = kwargs["env_builder"](4, 1000, 8, suite_mode=kwargs["suite_mode"], controller_variant=kwargs["controller_variant"])
        result = kwargs["episode_runner"](
            env=env, task_id=4, seed=1000, output_dir=tmp_path / kwargs["suite_mode"],
            variant="molmo_dense_agentview", resolution=8, dry_run=False,
            suite_mode=kwargs["suite_mode"], controller_variant=kwargs["controller_variant"],
            bboxes=kwargs["arrow_input_builder"](env, 4, 8)["bboxes"], subject="bowl", goal_object="plate",
        )
        manifest = result["canary_manifest"]
        assert [item["status"] for item in manifest["attempts"]] == ["candidate_failed", "selected"]
        assert manifest["attempts"][0]["candidates"][0]["candidate_id"] != manifest["attempts"][1]["candidates"][0]["candidate_id"]
        return {"cells": [result]}

    config = molmopoint.MolmoPointRuntimeConfig()
    fake_runtime = SimpleNamespace(config=config, loaded=True)
    monkeypatch.setattr(runner, "_execution_provenance", lambda **_kwargs: {
        "execution_sha": "a" * 40, "checkout_clean": True, "live_verified": True,
        "provenance_status": "live_verified", "dirty_paths": [],
    })
    monkeypatch.setattr(runner, "_resolved_controller_config_digest", lambda _path: "b" * 64)
    monkeypatch.setattr(runner, "preflight_local_molmo_runtime", lambda *_args, **_kwargs: {
        "model_id": runner.MOLMOPOINT_MODEL_ID, "model_revision": runner.MOLMOPOINT_MODEL_REVISION,
        "prompt_id": "rim_clearance", "prompt": "fake", "models_loaded": True,
    })
    monkeypatch.setattr(runner, "ModelPerceptionWorker", _Worker)
    monkeypatch.setattr(runner, "probe_robot_calibration", calibration)
    monkeypatch.setattr(runner, "_perform_gripper_open", open_gripper)
    monkeypatch.setattr(runner, "_perform_observation_hover", hover)
    monkeypatch.setattr(episode, "build_libero_env", build_env)
    monkeypatch.setattr(episode, "capture_agentview", capture)
    monkeypatch.setattr(episode, "render_exactly_one_arrow", lambda rgb, *_a, **_k: (rgb, {}))
    monkeypatch.setattr(episode, "decode_arrow_pixels", lambda *_a, **_k: ((2.0, 2.0), (3.0, 3.0)))
    monkeypatch.setattr(episode, "_raw_observation", lambda _env: {
        "eef_pos": np.asarray([0.1, 0.2, 0.5]), "eef_quat": np.asarray([0, 0, 0, 1.0])
    })
    monkeypatch.setattr(episode, "run_episode", run_episode)
    monkeypatch.setattr(runner, "_recover_after_failed_candidate", lambda *_a, **_k: {"retreat_complete": True})
    monkeypatch.setattr(matrix, "run_matrix", fake_matrix_run)
    def current_arrow_inputs(*_a):
        events.append("provider")
        return {"bboxes": {"bowl": [1, 1, 4, 4], "plate": [5, 5, 7, 7]}, "subject": "bowl", "goal_object": "plate"}

    monkeypatch.setattr(matrix, "_default_arrow_inputs", current_arrow_inputs)
    preshape_module = importlib.import_module("molmo_sam3.preshape")
    monkeypatch.setattr(preshape_module, "perform_preshape", preshape)

    assert runner.main([
        "--region-backend", "rgbd", "--variant", "molmo_dense_agentview",
        "--output-dir", str(tmp_path / "main"), "--phase", "prefix", "--episodes-per-task", "1",
        "--observation-profile", "parked", "--opening-profile", opening_profile,
    ], molmo_runtime=fake_runtime) == 0
    assert events.count("run:1") == 2 and events.count("run:2") == 2
    assert events.count("preshape") == expected_preshape
    assert capture_serial >= 6
    # Matrix input generation occurs once before each episode and the parked
    # retry refresh may request it again; both suites must exercise that seam.
    assert events.count("provider") >= 4
    assert events.count("calibration") >= 4
    for suite_mode in ("vanilla", "sealed_randomized"):
        refresh_dir = tmp_path / suite_mode / "arrow_refreshes"
        for refresh_index in (1, 2):
            record_path = refresh_dir / f"refresh_{refresh_index:02d}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            assert record["render_status"] == "completed"
            assert record["decode_status"] == "completed"
            assert (refresh_dir / f"refresh_{refresh_index:02d}_clean.png").exists()
            assert (refresh_dir / f"refresh_{refresh_index:02d}_arrow.png").exists()


@pytest.mark.parametrize("angle_degrees", list(range(0, 360, 15)))
def test_actual_arrow_wrapper_and_decoder_preserve_short_arrow_targets(angle_degrees):
    pytest.importorskip("cv2")
    span = 20
    radians = math.radians(angle_degrees)
    source = np.asarray([64, 64], dtype=int)
    target = source + np.rint(span * np.asarray([math.cos(radians), math.sin(radians)])).astype(int)
    bboxes = {
        "bowl": [source[0] - 1, source[1] - 1, source[0] + 1, source[1] + 1],
        "plate": [target[0] - 1, target[1] - 1, target[0] + 1, target[1] + 1],
    }
    clean = np.full((128, 128, 3), 100, dtype=np.uint8)
    render_params = runner._parked_arrow_render_params(
        episode, bboxes, subject="bowl", goal_object="plate", image_shape=clean.shape[:2]
    )
    rendered, audit = episode.render_exactly_one_arrow(
        clean, bboxes, subject="bowl", goal_object="plate",
        line_width=render_params["line_width"], head_length=render_params["head_length"],
    )
    decoded_source, decoded_target = episode.decode_arrow_pixels(clean, rendered)

    assert render_params["line_width"] == 1
    assert render_params["head_length"] == max(3, min(16, round(0.35 * render_params["endpoint_span_px"])))
    assert audit["relation_count"] == 1
    assert np.linalg.norm(np.asarray(decoded_source) - source) <= 2.0
    assert np.linalg.norm(np.asarray(decoded_target) - target) <= 2.0


@pytest.mark.parametrize("source,target", [((204, 68), (194, 66)), ((193, 85), (200, 72))])
def test_actual_arrow_wrapper_and_decoder_preserve_known_bbox_center_pairs(source, target):
    pytest.importorskip("cv2")
    source = np.asarray(source, dtype=int)
    target = np.asarray(target, dtype=int)
    bboxes = {
        "bowl": [source[0] - 1, source[1] - 1, source[0] + 1, source[1] + 1],
        "plate": [target[0] - 1, target[1] - 1, target[0] + 1, target[1] + 1],
    }
    clean = np.full((256, 256, 3), 90, dtype=np.uint8)
    render_params = runner._parked_arrow_render_params(
        episode, bboxes, subject="bowl", goal_object="plate", image_shape=clean.shape[:2]
    )
    rendered, _ = episode.render_exactly_one_arrow(
        clean, bboxes, subject="bowl", goal_object="plate",
        line_width=render_params["line_width"], head_length=render_params["head_length"],
    )
    decoded_source, decoded_target = episode.decode_arrow_pixels(clean, rendered)
    assert np.linalg.norm(np.asarray(decoded_source) - source) <= 2.0
    assert np.linalg.norm(np.asarray(decoded_target) - target) <= 2.0


def test_actual_arrow_decoder_fails_closed_for_tiny_line_and_long_defaults_are_invariant():
    pytest.importorskip("cv2")
    clean = np.full((128, 128, 3), 100, dtype=np.uint8)

    tiny_bboxes = {"bowl": [63, 64, 65, 66], "plate": [69, 64, 71, 66]}
    tiny = clean.copy()
    tiny[65, 64:80, 0] = 255  # plain shaft: no pointy head, so decode must reject it
    with pytest.raises(ValueError):
        episode.decode_arrow_pixels(clean, tiny)
    tiny_params = runner._parked_arrow_render_params(
        episode, tiny_bboxes, subject="bowl", goal_object="plate", image_shape=clean.shape[:2]
    )
    assert tiny_params["line_width"] == 1 and tiny_params["head_length"] == 3

    long_bboxes = {"bowl": [39, 63, 41, 65], "plate": [79, 63, 81, 65]}
    implicit, implicit_audit = episode.render_exactly_one_arrow(
        clean, long_bboxes, subject="bowl", goal_object="plate"
    )
    explicit, explicit_audit = episode.render_exactly_one_arrow(
        clean, long_bboxes, subject="bowl", goal_object="plate", line_width=2, head_length=16
    )
    assert np.array_equal(implicit, explicit)
    assert implicit_audit["line_width"] == explicit_audit["line_width"] == 2
    assert implicit_audit["head_length"] == explicit_audit["head_length"] == 16
    long_params = runner._parked_arrow_render_params(
        episode, long_bboxes, subject="bowl", goal_object="plate", image_shape=clean.shape[:2]
    )
    assert (long_params["line_width"], long_params["head_length"]) == (2, 16)
