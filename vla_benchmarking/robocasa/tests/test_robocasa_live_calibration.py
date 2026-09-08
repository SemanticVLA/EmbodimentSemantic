"""Regression fixtures for the compiled RoboCasa MuJoCo calibration boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vla_benchmarking.robocasa.arrow_grasp_controller.controller import runner
from vla_benchmarking.robocasa.arrow_grasp_controller.calibration import probe_panda_grip_site_frame as probe


RZ_MINUS_90 = probe.RZ_MINUS_90


def _compiled_fixture(*, important_site: int = 3) -> SimpleNamespace:
    site_names = {3: "gripper0_right_grip_site"}
    body_names = {
        0: "gripper0_right_right_gripper",
        1: "robot0_right_hand",
        2: "gripper0_right_leftfinger",
        3: "gripper0_right_rightfinger",
    }
    geom_names = {
        0: "gripper0_right_finger1_pad_collision",
        1: "gripper0_right_finger2_pad_collision",
        2: "gripper0_right_hand_collision",
        3: "gripper0_right_finger1_collision",
        4: "gripper0_right_finger2_collision",
    }

    class Model:
        nsite, nbody, ngeom = 4, 4, 5
        geom_size = np.asarray([[0.002, 0.002, 0.002]] * 5, dtype=float)
        geom_rbound = np.asarray([0.004, 0.004, 0.01, 0.004, 0.004], dtype=float)
        geom_bodyid = np.asarray([2, 3, 0, 2, 3], dtype=int)
        geom_type = np.asarray([6, 6, 6, 6, 6], dtype=int)
        geom_dataid = np.asarray([-1] * 5, dtype=int)
        geom_contype = np.asarray([1] * 5, dtype=int)
        geom_conaffinity = np.asarray([1] * 5, dtype=int)

        def site_name2id(self, _name: str) -> int:
            raise ValueError("compiled MuJoCo wrapper has no legacy site resolver")

        def body_name2id(self, _name: str) -> int:
            raise ValueError("compiled MuJoCo wrapper has no legacy body resolver")

        def geom_name2id(self, _name: str) -> int:
            raise ValueError("compiled MuJoCo wrapper has no legacy geom resolver")

        def site_id2name(self, index: int) -> str | None:
            return site_names.get(int(index))

        def body_id2name(self, index: int) -> str | None:
            return body_names.get(int(index))

        def geom_id2name(self, index: int) -> str | None:
            return geom_names.get(int(index))

    body_rotation_world = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    site_rotation_world = body_rotation_world @ RZ_MINUS_90
    # Only the flat xmat/body aliases used by MuJoCo 3 are provided.
    body_xpos = np.asarray(
        [[0.5, 0.5, 0.2], [1.0, 2.0, 0.35], [1.0, 1.97, 0.5], [1.0, 2.03, 0.5]],
        dtype=float,
    )
    geom_xpos = np.asarray(
        [[1.0, 1.97, 0.5], [1.0, 2.03, 0.5], [1.0, 2.0, 0.45], [1.0, 1.97, 0.5], [1.0, 2.03, 0.5]],
        dtype=float,
    )
    site_xpos = np.zeros((4, 3), dtype=float)
    site_xpos[3] = [1.0, 2.0, 0.6]
    data = SimpleNamespace(
        site_xmat=np.asarray([np.eye(3).reshape(-1)] * 4, dtype=float),
        xmat=np.asarray([np.eye(3).reshape(-1)] * 4, dtype=float),
        site_xpos=site_xpos,
        xpos=body_xpos,
        geom_xpos=geom_xpos,
        geom_xmat=np.asarray([np.eye(3).reshape(-1)] * 5, dtype=float),
    )
    data.site_xmat[3] = site_rotation_world.reshape(-1)
    data.xmat[1] = body_rotation_world.reshape(-1)
    robot = SimpleNamespace(
        eef_site_id={"right": 3},
        robot_model=SimpleNamespace(eef_name="right_hand"),
        gripper={"right": SimpleNamespace(important_sites={"grip_site": important_site})},
    )
    return SimpleNamespace(
        sim=SimpleNamespace(model=Model(), data=data),
        robots=[robot],
        _base_from_world_B0=np.asarray(
            [[0.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, -0.1], [0.0, 0.0, 0.0, 1.0]]
        ),
    )


def test_compiled_names_flattened_xmat_and_nonidentity_b0_are_resolved() -> None:
    env = _compiled_fixture()
    calibration, transform, record = runner.probe_robot_calibration(env)

    assert record["resolved_site_name"] == "gripper0_right_grip_site"
    assert record["resolved_body_name"] == "robot0_right_hand"
    assert record["gripper_geometry"]["left_pad_geom"] == "gripper0_right_finger1_pad_collision"
    assert np.allclose(record["position"]["site_xpos_m"], [0.0, 0.0, 0.5])
    assert np.allclose(transform, RZ_MINUS_90)
    assert calibration.grasp_to_grip_site.shape == (3, 3)


def test_authoritative_eef_and_important_site_mismatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="inconsistent eef_site_id"):
        runner.probe_robot_calibration(_compiled_fixture(important_site=2))


def test_suffix_resolution_rejects_ambiguous_compiled_names() -> None:
    class Model:
        site_names = ["left_grip_site", "right_grip_site"]

        def site_name2id(self, _name: str) -> int:
            raise ValueError("missing alias")

    with pytest.raises(AttributeError, match="ambiguous site suffix"):
        probe._named_id(Model(), "site", ("grip_site",))


def test_mujoco_id2name_fallback_unwraps_robosuite_model(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_model = object()
    seen: list[object] = []

    def id2name(model: object, _object_type: object, index: int) -> str:
        seen.append(model)
        return f"compiled_{index}"

    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_SITE=1),
        mj_id2name=id2name,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    wrapper = SimpleNamespace(_model=raw_model, nsite=2)
    assert probe._model_name_pairs(wrapper, "site") == [(0, "compiled_0"), (1, "compiled_1")]
    assert seen and all(model is raw_model for model in seen)
