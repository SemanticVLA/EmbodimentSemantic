"""Focused, dependency-free contracts for the canonical grasp controller."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest


runner = importlib.import_module("vla_benchmarking.arrow_grasp_controller.controller.runner")


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
    assert runner.BASELINE_COMMIT == "b4fb87759ae3a1ea2cd518cd201a1a737bb14e80"
    assert set(runner.VARIANTS) == {"canonical"}
    policy = runner.VARIANTS["canonical"]
    assert policy.camera_name == runner.AGENTVIEW
    assert policy.policy == "molmo_dense" and policy.uses_molmo is True
    assert policy.region_backend == "rgbd"
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
        variant="canonical", worker=worker,
        episode_runner=episode_runner, source_uv=(4, 4),
        capture_fn=capture_fn, resolution=16, dry_run=True,
    )
    assert len(calls) == 2
    assert calls[0][1] != calls[1][1]
    assert calls[0][2] is not calls[1][2]
    assert worker.requests[1].previous_candidate_ids == ("c1",)
    assert result["attempts"][0]["status"] == "candidate_failed"
    assert result["attempts"][1]["status"] == "selected"
    assert (tmp_path / "canonical_grasp_manifest.json").exists()


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
        variant="canonical", worker=worker,
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
        variant="canonical", worker=worker,
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
    fake_episode = ModuleType("vla_benchmarking.evaluation.run_arrow_pick_place_eval")
    fake_episode._raw_observation = lambda env: {"eef_pos": np.asarray((0.1, 0.2, 0.3))}
    fake_episode._proprioception = lambda observation: observation

    def run_motion(env, waypoints, observation, **kwargs):
        calls.append((np.asarray(waypoints), kwargs))
        return [{"phase": "open", "steps": 20, "status": "stop"}]

    fake_episode._run_motion = run_motion
    import vla_benchmarking.evaluation as evaluation_package
    monkeypatch.setattr(evaluation_package, "run_arrow_pick_place_eval", fake_episode, raising=False)
    env = SimpleNamespace(_grasp_controller_action_count=0)
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
    assert env._grasp_controller_action_count == 20


def test_robot_probe_emits_nonzero_center_rotated_mesh_collision_box(monkeypatch):
    probe_module = importlib.import_module("vla_benchmarking.tools.sanity_checks.probe_panda_grip_site_frame")
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
