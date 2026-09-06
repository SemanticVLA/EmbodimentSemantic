"""Focused RoboCasa boundary tests for the copied canonical controller."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import episode_contract, runner
from vla_benchmarking.robocasa.evaluation import adapter, capture
from vla_benchmarking.robocasa.evaluation.live import (
    RoboCasaControllerEnv,
    _official_outcome_from_controller_audit,
)
from vla_benchmarking.robocasa.evaluation.runner import _canonical_phase_timeout


def _fake_panda_env() -> SimpleNamespace:
    body_names = [
        "gripper0_leftfinger", "gripper0_rightfinger", "right_hand",
    ]
    geom_names = [
        "gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision",
        "gripper0_hand_collision",
    ]
    site_names = ["grip_site"]

    class Model:
        geom_size = np.asarray([[0.002, 0.002, 0.002]] * 3, dtype=float)
        geom_rbound = np.asarray([0.004, 0.004, 0.01], dtype=float)
        geom_bodyid = np.asarray([0, 1, 2], dtype=int)
        geom_type = np.asarray([6, 6, 6], dtype=int)
        geom_dataid = np.asarray([-1, -1, -1], dtype=int)
        geom_contype = np.asarray([1, 1, 1], dtype=int)
        geom_conaffinity = np.asarray([1, 1, 1], dtype=int)

        def site_name2id(self, name):
            return self.site_names.index(name)

        def body_name2id(self, name):
            return self.body_names.index(name)

        def geom_name2id(self, name):
            return self.geom_names.index(name)

        def geom_id2name(self, index):
            return self.geom_names[index]

    Model.site_names, Model.body_names, Model.geom_names = site_names, body_names, geom_names
    rz_minus_90 = np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    data = SimpleNamespace(
        site_xmat=np.asarray([rz_minus_90]),
        body_xmat=np.asarray([np.eye(3), np.eye(3), np.eye(3)]),
        site_xpos=np.asarray([[0.0, 0.0, 0.10]]),
        body_xpos=np.asarray([[0.0, -0.03, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, -0.10]]),
        xpos=np.asarray([[0.0, -0.03, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, -0.10]]),
        geom_xpos=np.asarray([[0.0, -0.03, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, -0.05]]),
        geom_xmat=np.asarray([np.eye(3), np.eye(3), np.eye(3)]),
    )
    return SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data), _base_from_world_B0=np.eye(4))


def test_panda_probe_uses_contact_geometry_and_returns_nonidentity_inputs() -> None:
    env = _fake_panda_env()
    calibration, transform, record = runner.probe_robot_calibration(env)
    geometry = record["gripper_geometry"]
    assert np.isclose(geometry["measured_opening_m"], 0.056)
    assert np.linalg.norm(np.asarray(geometry["contact_to_grip_site_site_m"])) > 0.0
    assert np.linalg.det(np.asarray(calibration.grasp_to_grip_site)) > 0.99
    assert geometry["hand_collision_spheres_grasp"]
    assert geometry["hand_collision_boxes_grasp"]
    assert np.allclose(transform, np.asarray(record["observed_body_to_site_rotation_matrix"]))


def test_panda_probe_does_not_write_mujoco_read_only_data_properties() -> None:
    env = _fake_panda_env()
    values = vars(env.sim.data).copy()

    class ReadOnlyData:
        def __getattr__(self, name):
            return values[name]

        def __setattr__(self, name, value):
            raise AttributeError(f"property {name!r} has no setter")

    env.sim.data = ReadOnlyData()
    calibration, transform, record = runner.probe_robot_calibration(env)
    assert record["passed"] is True
    assert calibration.grasp_to_grip_site.shape == (3, 3)
    assert transform.shape == (3, 3)


def test_panda_probe_adapts_mujoco3_named_accessors() -> None:
    env = _fake_panda_env()
    legacy_model = env.sim.model

    class NamedAccessOnlyModel:
        def __getattr__(self, name):
            if name.endswith("_name2id") or name.endswith("_id2name"):
                raise AttributeError(name)
            return getattr(legacy_model, name)

        def site(self, name_or_id):
            index = int(name_or_id) if isinstance(name_or_id, int) else legacy_model.site_name2id(name_or_id)
            return SimpleNamespace(id=index, name=legacy_model.site_names[index])

        def body(self, name_or_id):
            index = int(name_or_id) if isinstance(name_or_id, int) else legacy_model.body_name2id(name_or_id)
            return SimpleNamespace(id=index, name=legacy_model.body_names[index])

        def geom(self, name_or_id):
            index = int(name_or_id) if isinstance(name_or_id, int) else legacy_model.geom_name2id(name_or_id)
            return SimpleNamespace(id=index, name=legacy_model.geom_names[index])

    env.sim.model = NamedAccessOnlyModel()
    calibration, transform, record = runner.probe_robot_calibration(env)
    assert record["resolved_site_name"] == "grip_site"
    assert record["resolved_body_name"] == "right_hand"
    assert calibration.grasp_to_grip_site.shape == (3, 3)
    assert transform.shape == (3, 3)


def test_xyzw_base_quaternion_rotates_world_into_b0() -> None:
    s = np.sqrt(0.5)
    world_from_base, base_from_world = adapter.world_base_transform(
        {"robot0_base_pos": [1.0, 2.0, 0.0], "robot0_base_quat": [0.0, 0.0, s, s]}
    )
    assert np.allclose(world_from_base[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    point_world = world_from_base @ np.asarray([1.0, 0.0, 0.0, 1.0])
    assert np.allclose((base_from_world @ point_world)[:3], [1.0, 0.0, 0.0])


def test_observation_render_fallback_is_not_flipped() -> None:
    class Raw:
        action_space = None

        def render(self, **kwargs):
            return np.asarray([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8), np.asarray([[0.1], [0.2]], dtype=float)

    from vla_benchmarking.robocasa.evaluation.live import RoboCasaControllerEnv
    rgb, depth = RoboCasaControllerEnv(Raw()).render(camera_name="agentview", width=1, height=2, depth=True)
    assert tuple(rgb[0, 0]) == (1, 2, 3)
    assert np.isclose(depth[0, 0], 0.1)


def test_canonical_capture_alias_uses_physical_robocasa_camera(monkeypatch) -> None:
    physical_names: list[str] = []
    calibration = episode_contract.CameraCalibration(
        camera_name="placeholder",
        width=2,
        height=2,
        intrinsic=np.eye(3).tolist(),
        world_from_camera=np.eye(4).tolist(),
    )
    env = SimpleNamespace(
        sim=object(),
        _arrow_depth_encoding="normalized",
        _arrow_physical_camera_name="robot0_agentview_left",
        _last_observation={},
    )
    env.render = lambda **_kwargs: (
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2), 0.5, dtype=np.float32),
    )
    monkeypatch.setattr(
        episode_contract,
        "build_camera_calibration",
        lambda _sim, camera_name, _width, _height: (
            physical_names.append(camera_name) or calibration
        ),
    )
    monkeypatch.setattr(
        episode_contract,
        "normalized_depth_to_metric",
        lambda _sim, depth, **_kwargs: np.asarray(depth),
    )
    captured = episode_contract.capture_agentview(
        env, resolution=2, camera_name="agentview"
    )
    assert physical_names == ["robot0_agentview_left"]
    assert captured.calibration.camera_name == "agentview"
    assert captured.calibration.source.endswith(":robot0_agentview_left")
    assert episode_contract.validate_capture_contract(
        captured, resolution=2, camera_name="agentview"
    )["valid"]


def test_fixture_without_interior_region_fails_closed() -> None:
    from vla_benchmarking.robocasa.evaluation.live import RoboCasaLiveError, _world_points
    from vla_benchmarking.robocasa.shared.task_manifest import RoleSpec

    fixture = SimpleNamespace(
        get_bbox_points=lambda: np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    with pytest.raises(RoboCasaLiveError, match="exterior bbox"):
        _world_points(
            SimpleNamespace(),
            fixture,
            RoleSpec(
                key="self.counter",
                kind="fixture",
                region="counter_surface",
                supports_direct_bbox=False,
            ),
        )


def test_reset_and_step_return_b0_adapted_proprioception() -> None:
    identity_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])

    def observation(world_x: float, base_x: float) -> dict[str, np.ndarray]:
        return {
            "robot0_base_pos": np.asarray([4.0, 2.0, 0.0]),
            "robot0_base_quat": identity_xyzw,
            "robot0_base_to_eef_pos": np.asarray([base_x, 0.0, 0.8]),
            "robot0_base_to_eef_quat": identity_xyzw,
            "robot0_eef_pos": np.asarray([world_x, 2.0, 0.8]),
            "robot0_eef_quat": identity_xyzw,
        }

    class Raw:
        action_space = None

        def reset(self):
            return observation(4.10, 0.10)

        def step(self, _action):
            return observation(4.11, 0.11), 0.0, False, {}

    env = RoboCasaControllerEnv(Raw())
    reset_observation = env.reset()
    step_observation = env.step(np.zeros(7, dtype=float))[0]
    assert np.allclose(reset_observation["robot0_eef_pos"], [0.10, 0.0, 0.8])
    assert np.allclose(step_observation["robot0_eef_pos"], [0.11, 0.0, 0.8])


def test_canonical_timeout_is_read_from_nested_motion_audit() -> None:
    live = {
        "audit": {
            "final_result": {
                "audit": {"controller_variant": {"phase_timeout_steps": 160}}
            }
        }
    }
    assert _canonical_phase_timeout(live) == 160


def test_official_success_is_read_from_nested_canary_result() -> None:
    audit = {
        "final_result": {"evaluator_called": True, "evaluator_success": True}
    }
    assert _official_outcome_from_controller_audit(audit) == (True, True)
    assert _official_outcome_from_controller_audit({"final_result": {}}) == (False, False)


def test_effective_source_noun_reaches_molmopoint_request(monkeypatch, tmp_path) -> None:
    from vla_benchmarking.robocasa.arrow_grasp_controller.controller import grasp_candidates
    from vla_benchmarking.robocasa.arrow_grasp_controller.legacy_engine import rgbd_region
    seen = {}
    class Molmo:
        def predict(self, request):
            seen["prompt"] = request.prompt
            return SimpleNamespace(points=(), provenance={})
    monkeypatch.setattr(rgbd_region, "derive_observed_region_mask", lambda *a, **k: (np.ones((4, 4), dtype=bool), {}))
    monkeypatch.setattr(grasp_candidates, "generate_grasp_candidates", lambda **k: SimpleNamespace(candidates=(), rejected=(), seeds_uv=(), policy="test", audit={}))
    cal = capture.CameraCalibration("agentview", 4, 4, [[1, 0, 2], [0, 1, 2], [0, 0, 1]], np.eye(4).tolist())
    frame = capture.CapturedRGBD(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32), cal)
    worker = runner.ModelPerceptionWorker(Molmo(), object(), effective_prompt="Point to exposed parts of the rim of the mug highlighted in red where robot fingers could grasp it without touching nearby objects.")
    request = runner.PerceptionRequest(runner.VARIANTS["canonical"], frame, frame, (1.0, 1.0), (2.0, 2.0), (), tmp_path)
    worker.propose(request)
    assert seen["prompt"].count("the mug") == 1


def test_episode_contract_arrow_renderer_is_local_and_one_arrow() -> None:
    pytest.importorskip("cv2")
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    arrow, audit = episode_contract.render_exactly_one_arrow(
        image, {"src": (2, 2, 8, 8), "dst": (20, 20, 28, 28)}, subject="src", goal_object="dst"
    )
    assert audit["relation_count"] == 1
    assert arrow.shape == image.shape
    assert np.count_nonzero(arrow) > 0


def test_high_level_path_preshares_capture_and_plumbs_canonical_contract(monkeypatch, tmp_path) -> None:
    env = SimpleNamespace(_last_observation={}, _grasp_controller_action_count=0)
    calibration = capture.CameraCalibration(
        camera_name="agentview", width=4, height=4,
        intrinsic=[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        world_from_camera=np.eye(4).tolist(),
    )
    frame = capture.CapturedRGBD(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32), calibration)
    events = []
    probe = {"gripper_geometry": {"measured_opening_m": 0.04}}
    monkeypatch.setattr(runner, "build_local_molmo_runtime", lambda: object())
    monkeypatch.setattr(runner, "probe_robot_calibration", lambda _env: (object(), np.diag([1.0, 2.0, 1.0]), probe))
    monkeypatch.setattr(runner, "_run_robocasa_preshape", lambda *a, **k: events.append("preshape") or {"status": "completed"})
    monkeypatch.setattr(episode_contract, "capture_agentview", lambda *a, **k: events.append("capture") or frame)
    monkeypatch.setattr(episode_contract, "decode_arrow_pixels", lambda *_a: (np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))
    class Worker:
        def __init__(self, *args): self.robot_calibration = args[1]
    monkeypatch.setattr(runner, "ModelPerceptionWorker", Worker)
    captured = {}
    def fake_canary(**kwargs):
        captured.update(kwargs)
        f = kwargs["capture_fn"](env, resolution=4, camera_name="agentview")
        kwargs["arrow_refresh_fn"](f)
        retry = kwargs["before_capture_fn"](env, 2)
        kwargs["arrow_refresh_fn"](retry)
        candidate = SimpleNamespace(required_aperture_m=0.03)
        context = SimpleNamespace(candidate=candidate, output_dir=tmp_path, source_capture=f)
        captured["episode_result"] = kwargs["episode_runner"](context=context, evaluator=None)
        return {"status": "selected"}
    monkeypatch.setattr(runner, "run_canary_episode", fake_canary)
    monkeypatch.setattr(episode_contract, "run_episode", lambda **kwargs: captured.update(motion_kwargs=kwargs) or {"phases": [{"phase": "retreat", "status": "reached"}], "post_lift_retention_gate": {"records": [{"retained": True}]}, "evaluator_read_after_action": False})
    result = runner.run_episode(env=env, capture=frame, arrow_rgb=np.zeros_like(frame.rgb), bboxes={"src": (0, 0, 1, 1), "dst": (2, 2, 3, 3)}, output_dir=tmp_path, seed=0, source="src", destination="dst", resolution=4, evaluator=lambda _: True)
    assert result == {"status": "selected"}
    assert events == ["preshape", "capture", "preshape", "capture"]
    kwargs = captured["motion_kwargs"]
    assert kwargs["source_grasp_offset"] == runner.LEGACY_SOURCE_OFFSET_M
    assert kwargs["destination_release_offset"] == runner.LEGACY_DESTINATION_OFFSET_M
    assert kwargs["clearance_m"] == runner.LIFT_CLEARANCE_M
    assert kwargs["experimental_action_budget"] is env._grasp_controller_action_budget
    assert kwargs["post_lift_retention_gate"] is runner.retention_gate
    assert np.allclose(kwargs["experimental_eef_orientation_transform"], np.diag([1.0, 2.0, 1.0]))


def test_candidate_exception_dispatches_recovery_before_retry(monkeypatch, tmp_path) -> None:
    env = SimpleNamespace(_last_observation={}, _grasp_controller_action_count=0)
    calibration = capture.CameraCalibration("agentview", 4, 4, [[1, 0, 2], [0, 1, 2], [0, 0, 1]], np.eye(4).tolist())
    frame = capture.CapturedRGBD(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32), calibration)
    monkeypatch.setattr(runner, "build_local_molmo_runtime", lambda: object())
    monkeypatch.setattr(runner, "probe_robot_calibration", lambda _env: (object(), np.eye(3), {"gripper_geometry": {"measured_opening_m": 0.04}}))
    monkeypatch.setattr(runner, "_run_robocasa_preshape", lambda *a, **k: {"status": "completed"})
    monkeypatch.setattr(episode_contract, "capture_agentview", lambda *a, **k: frame)
    monkeypatch.setattr(episode_contract, "decode_arrow_pixels", lambda *_a: (np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))
    monkeypatch.setattr(runner, "ModelPerceptionWorker", lambda *a: SimpleNamespace(robot_calibration=a[1]))
    monkeypatch.setattr(episode_contract, "run_episode", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("candidate")))
    recovered = []
    monkeypatch.setattr(runner, "_recover_after_failed_candidate", lambda *a, **k: recovered.append(True) or {"retreat_complete": True})
    def canary(**kwargs):
        f = kwargs["capture_fn"](env, resolution=4, camera_name="agentview")
        kwargs["arrow_refresh_fn"](f)
        context = SimpleNamespace(candidate=SimpleNamespace(required_aperture_m=0.03), output_dir=tmp_path, source_capture=f)
        return kwargs["episode_runner"](context=context, evaluator=None)
    monkeypatch.setattr(runner, "run_canary_episode", canary)
    result = runner.run_episode(env=env, capture=frame, arrow_rgb=np.zeros_like(frame.rgb), bboxes={"src": (0, 0, 1, 1), "dst": (2, 2, 3, 3)}, output_dir=tmp_path, seed=0, source="src", destination="dst", resolution=4, evaluator=lambda _: True)
    assert result["status"] == "candidate_failed"
    assert recovered == [True]
