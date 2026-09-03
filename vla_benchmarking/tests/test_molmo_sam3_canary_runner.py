"""Focused, dependency-free contracts for the Molmo/SAM3 canary adapter."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest


runner = importlib.import_module("run_molmo_sam3_canary")


def test_rgbd_main_passes_frozen_v9d_matrix_selection(tmp_path, monkeypatch):
    matrix = importlib.import_module("run_arrow_pick_place_matrix")
    selected = []
    monkeypatch.setattr(runner, "_execution_provenance", lambda **_kwargs: {
        "execution_sha": "a" * 40, "checkout_clean": True,
        "live_verified": True, "provenance_status": "live_verified", "dirty_paths": [],
    })

    def validate_selection(**kwargs):
        label, _, _ = matrix._resolve_controller_selection(
            controller_variant=kwargs["controller_variant"],
            controller_config=kwargs["controller_config"],
            suite_mode=kwargs["suite_mode"],
        )
        selected.append(label)
        return {"successes": 0}

    monkeypatch.setattr(matrix, "run_matrix", validate_selection)
    assert runner.main([
        "--region-backend", "rgbd", "--variant", "rgbd_geometry_agentview",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert selected == [matrix.DEFAULT_CONTROLLER_VARIANT] * 2


@dataclass
class Calibration:
    camera_name: str
    pixel_origin: str = "top_left"
    camera_frame: str = "opencv_optical_x_right_y_down_z_forward"
    world_frame: str = "libero_mujoco_world"
    extrinsic_direction: str = "world_from_camera"
    rgb_depth_alignment: str = "same_sim_render_call"


@dataclass
class Capture:
    camera_name: str

    def __post_init__(self):
        self.rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        self.metric_depth = np.ones((16, 16), dtype=np.float32)
        self.calibration = Calibration(self.camera_name)


class Worker:
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        offset = len(self.requests)
        return {"candidates": [{
            "candidate_id": f"c{offset}",
            "position_world_m": [0.1 + offset * 0.001, 0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "opening_m": 0.04,
            "score": 1.0,
        }]}


def capture_fn(_env, *, resolution, camera_name):
    assert resolution == 16
    return Capture(camera_name)


def test_default_isolation_and_variant_inventory(tmp_path: Path):
    assert runner.BASELINE_COMMIT.startswith("fd24a4c")
    assert set(runner.VARIANTS) == {
        "sam3_geometry_agentview", "molmo_local_agentview",
        "molmo_dense_agentview", "molmo_dense_wrist", "rgbd_geometry_agentview",
    }
    assert all(item.camera_name == runner.AGENTVIEW for name, item in runner.VARIANTS.items() if "wrist" not in name)
    assert runner.VARIANTS["molmo_dense_wrist"].camera_name == runner.WRIST_CAMERA
    # Importing the canary must not import the frozen runner's model lifecycle.
    assert "omnis" not in runner.__dict__


def test_retry_uses_fresh_frames_and_a_different_candidate(tmp_path: Path):
    worker = Worker()
    calls = []

    def episode_runner(*, context, evaluator):
        calls.append((context.attempt_index, context.candidate.candidate_id, context.source_capture))
        return {"status": "grasp_failed" if len(calls) == 1 else "placed", "grasp_retained": len(calls) > 1,
                "retreat_complete": True}

    result = runner.run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant="molmo_dense_agentview", worker=worker,
        episode_runner=episode_runner, source_uv=(4, 4),
        capture_fn=capture_fn, resolution=16, dry_run=True,
    )
    assert len(calls) == 2
    assert calls[0][1] != calls[1][1]
    assert calls[0][2] is not calls[1][2]
    assert worker.requests[1].previous_candidate_ids == ("c1",)
    assert result["attempts"][0]["status"] == "candidate_failed"
    assert result["attempts"][1]["status"] == "selected"
    assert (tmp_path / "molmo_sam3_canary_manifest.json").exists()


def test_wrist_source_capture_is_explicit_and_provenanced(tmp_path: Path):
    worker = Worker()
    seen = {}

    def episode_runner(*, context, evaluator):
        seen["source"] = context.source_capture.calibration.camera_name
        seen["agentview"] = context.agentview_capture.calibration.camera_name
        return {"status": "placed", "grasp_retained": True, "retreat_complete": True}

    result = runner.run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant="molmo_dense_wrist", worker=worker,
        episode_runner=episode_runner, source_uv=(4, 4),
        capture_fn=capture_fn, resolution=16, dry_run=True,
    )
    assert seen == {"source": runner.WRIST_CAMERA, "agentview": runner.AGENTVIEW}
    assert result["camera_provenance"]["source"]["camera_name"] == runner.WRIST_CAMERA
    assert result["camera_provenance"]["agentview"]["camera_name"] == runner.AGENTVIEW
    assert worker.requests[0].source_capture.calibration.camera_name == runner.WRIST_CAMERA


def test_evaluator_is_only_callable_after_retreat(tmp_path: Path):
    worker = Worker()
    evaluator_calls = []

    def evaluator(_env):
        evaluator_calls.append(True)
        return True

    def episode_runner(*, context, evaluator):
        assert evaluator is not None
        with pytest.raises(RuntimeError, match="before retreat"):
            evaluator(object())
        context.mark_retreat_complete()
        assert evaluator(object()) is True
        return {"status": "placed", "grasp_retained": True,
                "retreat_complete": True, "evaluator_called": True}

    result = runner.run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant="molmo_dense_agentview", worker=worker,
        episode_runner=episode_runner, source_uv=(4, 4),
        evaluator=evaluator, capture_fn=capture_fn, resolution=16, dry_run=False,
    )
    assert evaluator_calls == [True]
    assert result["evaluator_timing"] == "after_retreat_only"


def test_candidate_expansion_is_bounded_and_deterministically_ranked():
    seed = runner.GraspCandidate("seed", (0.1, 0.2, 0.3), (0, 0, 0, 1), 0.04, score=1.0)
    expanded = runner.expand_grasp_candidates([seed])
    assert len(expanded) == 9
    assert expanded[0].candidate_id == "seed__yaw-15__ins0"
    assert all(candidate.metadata["yaw_offset_deg"] in (-15.0, 0.0, 15.0) for candidate in expanded)
    assert all(candidate.metadata["insertion_offset_m"] in (0.0, 0.004, 0.008) for candidate in expanded)


def test_failed_recovery_must_fail_closed_without_trying_stale_episode_state(tmp_path: Path):
    """A failed recovery cannot authorize another grasp on the same env state."""
    worker = Worker()
    calls = []

    def episode_runner(*, context, evaluator):
        calls.append(context.candidate.candidate_id)
        return {
            "status": "recovery_failed",
            "grasp_retained": False,
            "retreat_complete": False,
        }

    result = runner.run_canary_episode(
        env=object(), task_id=4, seed=1000, output_dir=tmp_path,
        variant="molmo_dense_agentview", worker=worker,
        episode_runner=episode_runner, source_uv=(4, 4),
        capture_fn=capture_fn, resolution=16, dry_run=False,
    )
    assert calls == ["c1"]
    assert result["attempts"][-1]["status"] == "recovery_failed"


def test_robot_probe_uses_contact_pad_geoms_and_nonidentity_mapping():
    # The probe's frame relation is the live Panda contract; pad geoms provide
    # the actual contact midpoint/aperture rather than finger-body origins.
    rz_minus_90 = np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))

    class Model:
        def __init__(self):
            self.names = {
                "site": ["grip_site"],
                "body": ["right_hand", "leftfinger", "rightfinger"],
                "geom": ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
            }
            self.geom_size = np.asarray([[0.0, 0.01, 0.01], [0.0, 0.01, 0.01]])

        def __getattr__(self, name):
            if name.endswith("_name2id"):
                kind = name[:-8]
                return lambda value: self.names[kind].index(value)
            raise AttributeError(name)

    data = SimpleNamespace(
        site_xmat=np.asarray([rz_minus_90]),
        body_xmat=np.asarray([np.eye(3), np.eye(3), np.eye(3)]),
        xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        site_xpos=np.asarray([[0.0, 0.0, 0.1]]),
        body_xpos=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        geom_xpos=np.asarray([[0.0, 0.0, 0.1], [0.04, 0.0, 0.1]]),
        geom_xmat=np.asarray([np.eye(3), np.eye(3)]),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    calibration, _transform, record = runner.probe_robot_calibration(env)
    assert np.linalg.det(np.asarray(calibration.grasp_to_grip_site)) == pytest.approx(1.0)
    assert not np.allclose(calibration.grasp_to_grip_site, np.eye(3))
    assert record["gripper_geometry"]["source"] == "no_motion_mujoco_contact_pad_geom_probe"
    assert record["gripper_geometry"]["measured_opening_m"] == pytest.approx(0.04)


def test_pre_motion_validation_does_not_recount_hover_trace():
    env = SimpleNamespace(_arrow_motion_trace=[{"action_sent": True}] * 7)
    before = env._arrow_motion_trace
    assert runner._newly_sent_actions(env, before) == 0
    env._arrow_motion_trace = [{"action_sent": True}] * 2
    assert runner._newly_sent_actions(env, before) == 2


def test_gripper_open_preflight_runs_before_reprobe_and_counts_shared_actions(tmp_path, monkeypatch):
    calls = []
    fake_episode = ModuleType("run_arrow_pick_place_eval")
    fake_episode._raw_observation = lambda env: {"eef_pos": np.asarray((0.1, 0.2, 0.3))}
    fake_episode._proprioception = lambda observation: observation

    def run_motion(env, waypoints, observation, **kwargs):
        calls.append((np.asarray(waypoints), kwargs))
        return [{"phase": "open", "steps": 20, "status": "stop"}]

    fake_episode._run_motion = run_motion
    monkeypatch.setitem(sys.modules, "run_arrow_pick_place_eval", fake_episode)
    env = SimpleNamespace(_molmo_sam3_action_count=0)
    callback_calls = []
    audit = runner._perform_gripper_open(
        env, output_dir=tmp_path, motion_started_callback=lambda: callback_calls.append(True),
    )
    assert calls[0][1]["start_phase"] == "open"
    assert calls[0][1]["stop_after_phase"] == "open"
    assert calls[0][1]["action_budget"] == 1200
    assert calls[0][1]["motion_started_callback"] is not None
    assert np.allclose(calls[0][0], np.tile((0.1, 0.2, 0.3), (6, 1)))
    assert audit["total_actions"] == 20
    assert env._molmo_sam3_action_count == 20


def test_observation_hover_uses_observed_region_q90_and_staged_motion(tmp_path, monkeypatch):
    fake_episode = ModuleType("run_arrow_pick_place_eval")
    fake_episode._raw_observation = lambda _env: {
        "eef_pos": np.asarray((0.20, 0.20, 0.80)),
        "eef_quat": np.asarray((0.0, 0.0, 0.0, 1.0)),
    }
    fake_episode._proprioception = lambda observation: observation
    fake_episode._pose_waypoint = lambda position, orientation=None: {
        "position": np.asarray(position, dtype=np.float64), "orientation": np.asarray(orientation, dtype=np.float64),
    }
    fake_episode.validate_workspace_points = lambda points: None
    calls = []

    def run_motion(_env, waypoints, _observation, **kwargs):
        calls.append((waypoints, kwargs))
        return [{"phase": phase, "steps": 1, "status": "reached"} for phase in kwargs["phase_order"]]

    fake_episode._run_motion = run_motion
    monkeypatch.setitem(sys.modules, "run_arrow_pick_place_eval", fake_episode)

    @dataclass
    class HoverCalibration:
        intrinsic: list[list[float]]
        world_from_camera: list[list[float]]

    capture = SimpleNamespace(
        rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        metric_depth=np.ones((64, 64), dtype=np.float64),
        calibration=HoverCalibration(
            intrinsic=((20.0, 0.0, 32.0), (0.0, 20.0, 32.0), (0.0, 0.0, 1.0)),
            world_from_camera=np.eye(4).tolist(),
        ),
    )
    env = SimpleNamespace(_molmo_sam3_action_count=0)
    audit = runner._perform_observation_hover(env, capture, (32, 32), output_dir=tmp_path)
    assert audit["region_q90_world_z_m"] == pytest.approx(1.0)
    assert audit["hover_world_m"][2] == pytest.approx(1.10)
    waypoints, kwargs = calls[0]
    assert kwargs["phase_order"] == ("hover_raise", "hover_translate", "hover_lower")
    assert kwargs["phase_timeout_steps_by_phase"] == {name: 160 for name in kwargs["phase_order"]}
    assert np.allclose(waypoints["hover_raise"]["position"], (0.20, 0.20, 1.10))
    assert np.allclose(waypoints["hover_translate"]["position"], (0.0, 0.0, 1.10))
    assert np.allclose(waypoints["hover_lower"]["position"], (0.0, 0.0, 1.10))


def test_robot_probe_emits_nonzero_center_rotated_mesh_collision_box(monkeypatch):
    probe_module = importlib.import_module("sanity_checks.probe_panda_grip_site_frame")
    monkeypatch.setattr(probe_module, "probe_grip_site_frame", lambda _sim: {
        "passed": True, "site_id": 0, "body_id": 0, "resolved_body_name": "right_hand",
        "observed_body_to_site_rotation_matrix": np.eye(3).tolist(),
    })

    class Model:
        names = {
            "site": ["grip_site"],
            "body": ["right_hand", "leftfinger", "rightfinger", "gripper0_right_gripper"],
            "geom": ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision", "gripper0_hand_collision"],
        }
        geom_size = np.asarray(((0.0, 0.01, 0.01), (0.0, 0.01, 0.01), (0.1, 0.1, 0.1)))
        geom_type = np.asarray((6, 6, 7))
        geom_dataid = np.asarray((-1, -1, 0))
        geom_bodyid = np.asarray((1, 2, 3))
        geom_rbound = np.asarray((0.01, 0.01, 0.06))
        geom_contype = np.asarray((1, 1, 1))
        geom_conaffinity = np.asarray((1, 1, 1))
        mesh_vertadr = np.asarray((0,))
        mesh_vertnum = np.asarray((2,))
        mesh_vert = np.asarray(((-0.01, -0.02, -0.03), (0.03, 0.04, 0.05)))

        def __getattr__(self, name):
            if name.endswith("_name2id"):
                kind = name[:-8]
                return lambda value: self.names[kind].index(value)
            raise AttributeError(name)

    hand_rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    data = SimpleNamespace(
        site_xmat=np.asarray((runner.np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),)),
        body_xmat=np.asarray((np.eye(3), np.eye(3), np.eye(3), np.eye(3))),
        body_xpos=np.zeros((4, 3)), xpos=np.zeros((4, 3)),
        site_xpos=np.asarray(((0.0, 0.0, 0.1),)),
        geom_xpos=np.asarray(((0.0, 0.0, 0.1), (0.04, 0.0, 0.1), (0.01, 0.02, 0.11))),
        geom_xmat=np.asarray((np.eye(3), np.eye(3), hand_rotation)),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    calibration, _transform, record = runner.probe_robot_calibration(env)
    boxes = calibration.hand_collision_boxes_grasp
    hand_box = next(item for item in boxes if item["geom_name"] == "gripper0_hand_collision")
    # Mesh local center=(.01,.01,.01), rotated by Rz(+90deg), gives world
    # center=(0,.03,.12); subtract contact midpoint=(.02,0,.10), then apply
    # the probe's measured site/grasp bases to obtain this grasp-frame center.
    site_rotation = np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    jaw_axis = np.asarray((1.0, 0.0, 0.0))
    approach = np.asarray((0.02, 0.0, 0.10))
    approach -= jaw_axis * np.dot(approach, jaw_axis)
    approach /= np.linalg.norm(approach)
    grasp_to_site = np.column_stack((site_rotation.T @ approach, site_rotation.T @ jaw_axis, np.cross(site_rotation.T @ approach, site_rotation.T @ jaw_axis))).T
    expected_center = grasp_to_site @ site_rotation.T @ (np.asarray((0.0, 0.03, 0.12)) - np.asarray((0.02, 0.0, 0.10)))
    assert np.allclose(hand_box["center_grasp_m"], expected_center)
    assert not np.allclose(hand_box["rotation_grasp_box"], np.eye(3))
    assert np.allclose(hand_box["half_extents_m"], (0.02, 0.03, 0.04))
    assert hand_box["source"] == "compiled_mesh_vertices_aabb"
    assert record["gripper_geometry"]["hand_collision_box_count"] == 3
