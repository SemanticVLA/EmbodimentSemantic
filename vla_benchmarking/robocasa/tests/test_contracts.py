from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import sys
import types

from vla_benchmarking.robocasa.environment.runtime import (
    CameraObservationContract,
    compose_panda_omron_action,
    validate_action_layout,
)
from vla_benchmarking.robocasa.evaluation import (
    capture_robocasa_rgbd,
    PICK_PLACE_TASKS,
    adapt_source_noun,
    bbox_center,
    render_bbox_center_arrow,
)
from vla_benchmarking.robocasa.evaluation.runner import build_rows, selected_tasks
from vla_benchmarking.robocasa.evaluation.live import (
    RoboCasaControllerEnv,
    RoboCasaLiveError,
    _source_label,
    _project_bbox,
    _world_points,
    official_success,
    project_task_bboxes,
)


def _identity_capture(*, width: int = 256, height: int = 256, focal: float = 100.0):
    return type("Capture", (), {
        "rgb": np.zeros((height, width, 3), dtype=np.uint8),
        "calibration": type("Calibration", (), {
            "intrinsic": [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            "world_from_camera": np.eye(4).tolist(),
        })(),
    })()


def test_project_bbox_gives_subpixel_in_frame_role_pixel_support() -> None:
    capture = _identity_capture(focal=100.0)
    # The world footprint is finite and visible, but projects to less than one
    # pixel in both dimensions at this image resolution.
    points = np.asarray([
        [0.0, -1e-4, 1.0], [0.0, 1e-4, 1.0],
    ])
    bbox = _project_bbox(points, capture, width=256, height=256)
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
    assert bbox[0] == pytest.approx(127.5)
    assert bbox[2] == pytest.approx(128.5)


def test_project_bbox_remains_fail_closed_when_role_is_out_of_frame() -> None:
    capture = _identity_capture(focal=100.0)
    points = np.asarray([
        [-10.1, -1e-4, 1.0], [-10.0, 1e-4, 1.0],
    ])
    with pytest.raises(RoboCasaLiveError, match="no visible area"):
        _project_bbox(points, capture, width=256, height=256)


def test_project_bbox_accepts_overlapping_box_with_no_vertex_inside() -> None:
    capture = _identity_capture(focal=100.0)
    # The projected square spans [-72, 328] in both axes.  It covers the
    # 256x256 image, but every vertex is outside the image rectangle.
    points = np.asarray(
        [(-2.0, -2.0, 1.0), (2.0, -2.0, 1.0),
         (2.0, 2.0, 1.0), (-2.0, 2.0, 1.0)],
        dtype=np.float64,
    )
    bbox = _project_bbox(points, capture, width=256, height=256)
    assert bbox == pytest.approx((0.0, 0.0, 255.0, 255.0))
from vla_benchmarking.robocasa.shared.config import TARGET_SPLIT, target_env_kwargs
from vla_benchmarking.robocasa.shared.task_manifest import (
    PICK_PLACE_TASKS as SHARED_TASKS,
    get_task,
)


def test_manifest_has_all_official_tasks_and_make_iced_coffee_selector() -> None:
    assert len(PICK_PLACE_TASKS) == 21
    assert len(SHARED_TASKS) == 21
    assert get_task("MakeIcedCoffee").source_selector
    assert get_task("PackDessert").destination.key == "cooked_food_container"
    assert len({task.name for task in PICK_PLACE_TASKS}) == 21


def test_target_split_is_explicit_and_deterministic() -> None:
    assert TARGET_SPLIT.layout_and_style_ids == tuple((i, i) for i in range(1, 11))
    kwargs = target_env_kwargs(seed=1000)
    assert kwargs["robots"] == "PandaOmron"
    assert kwargs["camera_names"] == ["robot0_agentview_left"]
    assert kwargs["camera_widths"] == kwargs["camera_heights"] == 256
    assert kwargs["split"] == "target"


def test_camera_keys_match_robocasa_gym_wrapper() -> None:
    contract = CameraObservationContract()
    assert contract.rgb_key == "video.robot0_agentview_left"
    assert contract.depth_key == "video.robot0_agentview_left_depth"


def test_robocasa_capture_owns_flip_and_depth_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class CameraUtils:
        @staticmethod
        def get_camera_intrinsic_matrix(sim, camera_name, *, camera_height, camera_width):
            calls["camera_name"] = camera_name
            return np.asarray([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])

        @staticmethod
        def get_camera_extrinsic_matrix(sim, camera_name):
            return np.eye(4)

        @staticmethod
        def get_real_depth_map(sim, depth):
            calls["depth"] = np.asarray(depth).copy()
            return np.asarray(depth, dtype=np.float32) + 10.0

    robosuite = types.ModuleType("robosuite")
    utils = types.ModuleType("robosuite.utils")
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_camera_intrinsic_matrix = CameraUtils.get_camera_intrinsic_matrix
    camera_utils.get_camera_extrinsic_matrix = CameraUtils.get_camera_extrinsic_matrix
    camera_utils.get_real_depth_map = CameraUtils.get_real_depth_map
    utils.camera_utils = camera_utils
    robosuite.utils = utils
    monkeypatch.setitem(sys.modules, "robosuite", robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.utils", utils)
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)

    class Sim:
        def render(self, *, camera_name, width, height, depth):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[0, 0] = (1, 2, 3)
            raw_depth = np.full((height, width), 0.5, dtype=np.float32)
            raw_depth[0, 0] = 0.25
            return rgb, raw_depth

    class Env:
        sim = Sim()

    capture = capture_robocasa_rgbd(Env(), camera_name="robot0_agentview_left", resolution=2)
    assert tuple(capture.rgb[-1, 0]) == (1, 2, 3)
    assert np.isclose(capture.normalized_depth[-1, 0], 0.25)
    assert np.isclose(capture.metric_depth[-1, 0], 10.25)
    assert calls["camera_name"] == "robot0_agentview_left"
    assert capture.calibration.intrinsic == [
        [100.0, 0.0, 2.0],
        [0.0, 100.0, 2.0],
        [0.0, 0.0, 1.0],
    ]


def test_panda_omron_action_embedding_zeroes_mobile_parts() -> None:
    action = compose_panda_omron_action(np.arange(7, dtype=float) / 10.0)
    assert action[:7] == tuple(np.arange(7, dtype=float) / 10.0)
    assert action[7:11] == (0.0, 0.0, 0.0, 0.0)
    assert action[11] == 0.0
    validate_action_layout(action)
    with pytest.raises(ValueError):
        validate_action_layout((*action[:7], 1.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        compose_panda_omron_action((0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0))


def test_prompt_changes_only_source_noun() -> None:
    prompt = adapt_source_noun("apple")
    assert prompt.count("the apple") == 1
    assert "rim" in prompt
    assert "nearby objects" in prompt
    with pytest.raises(ValueError):
        adapt_source_noun("")


def test_bbox_center_arrow_is_deterministic_and_preserves_shape() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    bboxes = {"obj": (2, 4, 10, 12), "plate": (20, 16, 28, 24)}
    assert bbox_center(bboxes["obj"]) == (6, 8)
    arrow_a, audit_a = render_bbox_center_arrow(
        rgb, bboxes, source="obj", destination="plate", allow_fallback=True
    )
    arrow_b, audit_b = render_bbox_center_arrow(
        rgb, bboxes, source="obj", destination="plate", allow_fallback=True
    )
    assert arrow_a.shape == rgb.shape
    assert not np.array_equal(arrow_a, rgb)
    assert np.array_equal(arrow_a, arrow_b)
    assert audit_a["source_center_uv"] == [6, 8]
    assert audit_a["destination_center_uv"] == audit_b["destination_center_uv"]


def test_arrow_strict_mode_fails_without_pinned_renderer() -> None:
    if __import__("importlib.util").util.find_spec("cv2") is not None:
        pytest.skip("strict renderer is available in this interpreter")
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="OpenCV is required"):
        render_bbox_center_arrow(
            rgb,
            {"obj": (2, 4, 10, 12), "plate": (20, 16, 28, 24)},
            source="obj",
            destination="plate",
            allow_fallback=False,
        )


def test_runner_counts_every_registered_cell() -> None:
    rows = build_rows(
        tasks=selected_tasks(), episodes_per_task=10, seed_base=1000,
        split="target", mode="preflight",
    )
    assert len(rows) == 210
    assert len({(row.task, row.seed) for row in rows}) == 210
    assert rows[0].seed == 1000 and rows[-1].seed == 1009


def test_mirrors_libero_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("shared", "environment", "evaluation", "arrow_grasp_controller", "finetuned_vlas", "tools", "tests"):
        assert (root / name).is_dir(), name
    for name in ("configs", "controller", "calibration", "legion"):
        assert (root / "arrow_grasp_controller" / name).is_dir(), name


def test_live_projection_and_panda_omron_step_shim() -> None:
    class Entity:
        def __init__(self, center: tuple[float, float, float]):
            self.center = np.asarray(center, dtype=float)

        def get_bbox_points(self) -> np.ndarray:
            x, y, z = self.center
            return np.asarray([
                [x - 0.05, y - 0.05, z], [x + 0.05, y + 0.05, z],
                [x - 0.05, y + 0.05, z], [x + 0.05, y - 0.05, z],
            ])

    class RawEnv:
        objects = {"cheese": Entity((-0.2, 0.0, 1.0)), "bread": Entity((0.2, 0.0, 1.0))}

        def step(self, action: dict[str, np.ndarray]):
            self.action = action
            return ({}, 0.0, False, False, {"success": True})

    raw = RawEnv()
    env = RoboCasaControllerEnv(raw)
    capture = type("Capture", (), {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "calibration": type("Calibration", (), {
            "intrinsic": [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]],
            "world_from_camera": np.eye(4).tolist(),
        })(),
    })()
    bboxes, source = project_task_bboxes(raw, capture, SHARED_TASKS[0])
    assert source == "cheese"
    assert bboxes["cheese"][0] < bboxes["bread"][0]
    env.step(np.zeros(7, dtype=float))
    assert raw.action.shape == (12,)
    assert np.allclose(raw.action[7:11], 0.0)
    assert official_success(env)

    class RawCheckerEnv:
        def _check_success(self) -> bool:
            return True

    assert official_success(RoboCasaControllerEnv(RawCheckerEnv()))


def test_make_iced_coffee_resolves_each_candidate_and_object_language() -> None:
    class Entity:
        def __init__(self, half_width: float):
            self.half_width = half_width

        def get_bbox_points(self) -> np.ndarray:
            return np.asarray([
                [-self.half_width, -0.02, 1.0],
                [self.half_width, 0.02, 1.0],
            ])

    class RawEnv:
        objects = {
            "ice_cube1": Entity(0.02),
            "ice_cube2": Entity(0.12),
            "cup": Entity(0.04),
        }

        def get_obj_lang(self, obj_name: str = "obj") -> str:
            return {"ice_cube1": "small ice cube", "ice_cube2": "large ice cube", "cup": "cup"}[obj_name]

    capture = type("Capture", (), {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "calibration": type("Calibration", (), {
            "intrinsic": [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]],
            "world_from_camera": np.eye(4).tolist(),
        })(),
    })()
    task = get_task("MakeIcedCoffee")
    bboxes, selected = project_task_bboxes(RawEnv(), capture, task)
    assert bboxes["ice_cube1"] != bboxes["ice_cube2"]
    assert selected == "ice_cube2"
    assert _source_label(RawEnv(), task, selected) == "large ice cube"


def test_object_bbox_uses_current_mujoco_pose() -> None:
    class Entity:
        def get_bbox_points(self, *, trans=None, rot=None) -> np.ndarray:
            local = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            if trans is None or rot is None:
                return local
            # The test quaternion is a 90-degree z rotation in xyzw order.
            rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            return (rotation @ local.T).T + np.asarray(trans)

    class Data:
        body_xpos = np.asarray([[2.0, 3.0, 4.0]])
        body_xquat = np.asarray([[0.70710678, 0.0, 0.0, 0.70710678]])  # wxyz

    class Sim:
        data = Data()

    class Env:
        sim = Sim()
        obj_body_id = {"obj": 0}

    role = get_task("PickPlaceCounterToStove").source
    points = _world_points(Env(), Entity(), role)
    assert np.allclose(points[0], [2.0, 3.0, 4.0])
    assert np.allclose(points[1], [2.0, 4.0, 4.0])


def test_fixture_interior_sites_are_used_instead_of_exterior_bbox() -> None:
    class Fixture:
        def get_int_sites(self, *, all_points=True, relative=False):
            return np.asarray([
                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0], [0.0, 0.0, 2.0],
            ])

        def get_bbox_points(self):
            raise AssertionError("fixture exterior bbox must not be used")

    role = get_task("PickPlaceCounterToCabinet").destination
    points = _world_points(type("Env", (), {})(), Fixture(), role)
    assert points.shape == (4, 3)
    assert np.allclose(points[3], [0.0, 0.0, 2.0])


def test_official_runtime_region_selectors_resolve_toaster_and_fridge_regions() -> None:
    class Fixture:
        def get_int_sites(self, *, all_points=True, relative=False):
            return {
                "rack0": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]]),
                "fridge_shelf0": np.asarray([[2.0, 0.0, 1.0], [3.0, 0.0, 1.0], [2.0, 1.0, 1.0], [2.0, 0.0, 2.0]]),
                "fridge_drawer1": np.asarray([[4.0, 0.0, 1.0], [5.0, 0.0, 1.0], [4.0, 1.0, 1.0], [4.0, 0.0, 2.0]]),
            }

        def get_reset_regions(self, env, *, compartment, reg_type, rack_index):
            if reg_type == "drawer":
                return {"fridge_drawer1": {"offset": (0, 0, 0), "size": (1, 1)}}
            return {
                "fridge_shelf0": {"offset": (0, 0, 0), "size": (1, 1)},
            }

    class Env:
        chosen_toaster_receptacle = "rack0"

    toaster = get_task("PickPlaceCounterToToasterOven").destination
    fridge_drawer = get_task("PickPlaceFridgeShelfToDrawer").destination
    fridge_shelf = get_task("PickPlaceFridgeDrawerToShelf").destination
    fixture = Fixture()
    toaster_points = _world_points(Env(), fixture, toaster)
    drawer_points = _world_points(Env(), fixture, fridge_drawer)
    shelf_points = _world_points(Env(), fixture, fridge_shelf)
    assert np.allclose(toaster_points[0], [0.0, 0.0, 1.0])
    assert np.allclose(drawer_points[0], [4.0, 0.0, 1.0])
    assert np.allclose(shelf_points[0], [2.0, 0.0, 1.0])
