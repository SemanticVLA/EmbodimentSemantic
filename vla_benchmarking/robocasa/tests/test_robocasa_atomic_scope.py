from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_benchmarking.robocasa.evaluation import live
from vla_benchmarking.robocasa.evaluation.runner import selected_tasks
from vla_benchmarking.robocasa.shared.task_manifest import ARM_ONLY_ATOMIC_TASKS
from vla_benchmarking.robocasa.shared.task_manifest import iter_tasks


class _CoffeeMachine:
    pos = np.asarray((1.0, 2.0, 0.5))
    rot = 0.0

    def get_int_sites(self, all_points=False, relative=False):
        return {}

    def get_reset_regions(self, _env=None):
        return {"bottom": {"offset": (0.1, -0.2, 0.3), "size": (0.04, 0.06), "height": 0.02}}


class _Blender:
    def get_lid_closed_pos(self, _env):
        return np.asarray((0.4, -0.2, 0.8))


def test_atomic_scope_is_selectable_without_changing_pickplace21():
    assert len(ARM_ONLY_ATOMIC_TASKS) == 2
    names = [task.name for task in selected_tasks([
        "PickPlaceCounterToCabinet", "CoffeeSetupMug", "CloseBlenderLid",
    ])]
    assert names == ["PickPlaceCounterToCabinet", "CoffeeSetupMug", "CloseBlenderLid"]
    assert [task.name for task in iter_tasks(["CoffeeSetupMug", "CloseBlenderLid"])] == [
        "CoffeeSetupMug", "CloseBlenderLid",
    ]


def test_coffee_bottom_reset_region_projects_as_world_points():
    task = next(item for item in ARM_ONLY_ATOMIC_TASKS if item.name == "CoffeeSetupMug")
    points = live._world_points(SimpleNamespace(), _CoffeeMachine(), task.destination)
    assert points.shape == (8, 3)
    assert np.allclose(points.mean(axis=0), (1.1, 1.8, 0.81))


def test_blender_closed_lid_target_uses_official_fixture_pose():
    task = next(item for item in ARM_ONLY_ATOMIC_TASKS if item.name == "CloseBlenderLid")
    points = live._world_points(SimpleNamespace(), _Blender(), task.destination)
    assert points.shape == (4, 3)
    assert np.allclose(points.mean(axis=0), (0.4, -0.2, 0.8))


def test_blender_lid_source_resolves_from_fixture_child():
    task = next(item for item in ARM_ONLY_ATOMIC_TASKS if item.name == "CloseBlenderLid")
    lid = object()
    env = SimpleNamespace(fixtures={"blender": SimpleNamespace(blender_lid=lid)})
    assert live._role_entity(env, task.source) is lid
