"""Manifest for the 21 official RoboCasa Pick & Place atomic tasks.

The manifest describes only the source/destination roles required to generate
the controller arrow.  Success remains the official RoboCasa task predicate;
the strings below are deliberately not a replacement for that predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

RoleKind = Literal["object", "fixture", "region"]
LabelMode = Literal["object_lang", "static"]


@dataclass(frozen=True)
class RoleSpec:
    """A source or destination role in a task's arrow-generation contract."""

    key: str
    kind: RoleKind
    label_mode: LabelMode = "object_lang"
    static_label: str | None = None
    region: str | None = None
    # Official fixtures sometimes expose a runtime-selected region (for
    # example ``chosen_toaster_receptacle``) or qualified fridge region names.
    # Keep that selector explicit instead of guessing from display labels.
    region_selector: str | None = None
    # Direct object models expose bbox points.  Fixtures and internal regions
    # need a suite-local region/fixture projection adapter.
    supports_direct_bbox: bool = True

    def __post_init__(self) -> None:
        if self.label_mode == "static" and not self.static_label:
            raise ValueError(f"static role {self.key!r} requires static_label")
        if self.label_mode == "object_lang" and self.static_label is not None:
            raise ValueError(f"object_lang role {self.key!r} cannot set static_label")
        if self.kind in {"fixture", "region"} and self.supports_direct_bbox:
            raise ValueError(f"{self.kind} role {self.key!r} must use region projection")


@dataclass(frozen=True)
class PickPlaceTask:
    """Immutable task entry consumed by RoboCasa environment adapters."""

    name: str
    source: RoleSpec
    destination: RoleSpec
    success_rule: str
    # A selector is used only where the official task has a source choice or
    # generated object; it never changes the official success predicate.
    source_selector: str | None = None
    notes: str = ""


def _obj(key: str, *, label_mode: LabelMode = "object_lang", label: str | None = None) -> RoleSpec:
    return RoleSpec(key=key, kind="object", label_mode=label_mode, static_label=label)


def _fixture(key: str) -> RoleSpec:
    return RoleSpec(key=key, kind="fixture", supports_direct_bbox=False)


def _region(key: str, region: str, *, selector: str | None = None) -> RoleSpec:
    return RoleSpec(
        key=key,
        kind="region",
        region=region,
        region_selector=selector,
        supports_direct_bbox=False,
    )


# Keep this tuple in the public atomic-task order: three named kitchen tasks,
# followed by the 18 PickPlace classes in kitchen_pick_place.py.
PICK_PLACE_TASKS: tuple[PickPlaceTask, ...] = (
    PickPlaceTask(
        "CheesyBread", _obj("cheese", label_mode="static", label="cheese"), _obj("bread"),
        "bread in bread_container; cheese contacts bread; gripper far from cheese",
        notes="The generated bread_container is a secondary official goal object.",
    ),
    PickPlaceTask(
        "MakeIcedCoffee", _obj("ice_cube1", label_mode="static", label="ice cube"), _obj("cup"),
        "ice_cube1 or ice_cube2 in cup; gripper far from both cubes",
        source_selector="largest_visible_candidate(ice_cube1,ice_cube2)",
        notes="Either visible ice cube satisfies the official OR predicate.",
    ),
    PickPlaceTask(
        "PackDessert", _obj("dessert"), _obj("cooked_food_container"),
        "dessert and cooked_food in cooked_food_container; gripper far from dessert",
        notes="The cooked_food co-goal is part of the official task predicate.",
    ),
    PickPlaceTask("PickPlaceCounterToCabinet", _obj("obj"), _fixture("self.cab"), "obj inside cab; gripper far"),
    PickPlaceTask("PickPlaceCabinetToCounter", _obj("obj"), _fixture("self.counter"), "obj contacts counter; gripper far"),
    PickPlaceTask("PickPlaceCounterToSink", _obj("obj"), _fixture("self.sink"), "obj inside sink; gripper far"),
    PickPlaceTask("PickPlaceSinkToCounter", _obj("obj"), _obj("container"), "obj in container; container contacts counter; gripper far"),
    PickPlaceTask("PickPlaceCounterToMicrowave", _obj("obj"), _obj("container"), "obj contacts container; container contacts microwave; gripper far"),
    PickPlaceTask("PickPlaceMicrowaveToCounter", _obj("obj"), _obj("container"), "obj in container; gripper far"),
    PickPlaceTask(
        "PickPlaceCounterToOven",
        _obj("obj"),
        RoleSpec(key="oven_tray", kind="object", region="oven_rack"),
        "obj in oven_tray; oven_tray contacts selected rack; gripper far",
        notes="The arrow targets the direct oven_tray object; the selected rack is retained for success auditing.",
    ),
    PickPlaceTask("PickPlaceCounterToStove", _obj("obj"), _obj("container"), "obj in container; gripper far"),
    PickPlaceTask("PickPlaceStoveToCounter", _obj("obj"), _obj("container"), "obj in container; gripper far", notes="Source is placed inside generated obj_container during setup."),
    PickPlaceTask("PickPlaceToasterToCounter", _obj("obj", label_mode="static", label="toasted item"), _obj("plate"), "obj in plate; gripper far"),
    PickPlaceTask(
        "PickPlaceCounterToToasterOven",
        _obj("obj"),
        _region(
            "self.toaster_oven",
            "selected_toaster_receptacle",
            selector="chosen_toaster_receptacle",
        ),
        "obj contacts selected toaster-oven rack; gripper far",
    ),
    PickPlaceTask("PickPlaceToasterOvenToCounter", _obj("obj"), _obj("container"), "obj in container; container contacts counter; gripper far"),
    PickPlaceTask("PickPlaceCounterToStandMixer", _obj("obj"), _region("self.stand_mixer", "bowl"), "obj in stand-mixer bowl; gripper far"),
    PickPlaceTask(
        "PickPlaceFridgeShelfToDrawer",
        _obj("obj"),
        _region("self.fridge", "drawer", selector="fridge_drawer"),
        "obj contacts fridge drawer; gripper far",
    ),
    PickPlaceTask(
        "PickPlaceFridgeDrawerToShelf",
        _obj("obj"),
        _region("self.fridge", "shelf", selector="fridge_shelf"),
        "obj contacts fridge shelf; gripper far",
    ),
    PickPlaceTask("PickPlaceCounterToDrawer", _obj("obj"), _fixture("self.drawer"), "obj inside drawer and off counter; gripper far"),
    PickPlaceTask("PickPlaceDrawerToCounter", _obj("obj"), _fixture("self.counter"), "obj contacts counter; gripper far"),
    PickPlaceTask("PickPlaceCounterToBlender", _obj("obj"), _fixture("self.blender"), "obj inside blender; gripper far"),
)

# A small arm-only atomic extension requested for the exploratory sweep.  The
# canonical PickPlace-21 tuple remains unchanged so historical full-mode
# accounting and its 21-task contract are preserved.
ARM_ONLY_ATOMIC_TASKS: tuple[PickPlaceTask, ...] = (
    PickPlaceTask(
        "CoffeeSetupMug",
        _obj("obj"),
        _region("self.coffee_machine", "bottom"),
        "mug under coffee-machine dispenser; gripper far",
        notes="Official atomic task CoffeeSetupMug; the coffee-machine bottom reset region is the pouring target.",
    ),
    PickPlaceTask(
        "CloseBlenderLid",
        RoleSpec(
            key="blender_lid",
            kind="fixture",
            label_mode="static",
            static_label="blender lid",
            supports_direct_bbox=False,
        ),
        _region("self.blender", "lid_closed"),
        "blender lid on blender; gripper far",
        notes="Official atomic task CloseBlenderLid; the target is the measured closed-lid pose.",
    ),
)


_TASKS_BY_NAME = {task.name: task for task in PICK_PLACE_TASKS}
_TASKS_BY_NAME.update({task.name: task for task in ARM_ONLY_ATOMIC_TASKS})


def iter_tasks(names: Iterable[str] | None = None) -> tuple[PickPlaceTask, ...]:
    """Return manifest entries in stable order, optionally selecting names."""

    registered = (*PICK_PLACE_TASKS, *ARM_ONLY_ATOMIC_TASKS)
    if names is None:
        return PICK_PLACE_TASKS
    selected = set(names)
    unknown = selected.difference(_TASKS_BY_NAME)
    if unknown:
        raise KeyError(f"unknown RoboCasa PickPlace task(s): {sorted(unknown)}")
    return tuple(task for task in registered if task.name in selected)


def get_task(name: str) -> PickPlaceTask:
    try:
        return _TASKS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown RoboCasa PickPlace task: {name}") from exc


def validate_manifest() -> None:
    expected_count = 21
    if len(PICK_PLACE_TASKS) != expected_count:
        raise ValueError(f"expected {expected_count} PickPlace tasks, got {len(PICK_PLACE_TASKS)}")
    names = [task.name for task in PICK_PLACE_TASKS]
    if len(set(names)) != len(names):
        raise ValueError("PickPlace task names must be unique")
    for task in PICK_PLACE_TASKS:
        if not task.name or not task.success_rule:
            raise ValueError("task names and official success descriptions are required")
        if task.source.kind not in {"object", "fixture", "region"}:
            raise ValueError(f"invalid source kind for {task.name}")
        if task.destination.kind not in {"object", "fixture", "region"}:
            raise ValueError(f"invalid destination kind for {task.name}")


validate_manifest()

__all__ = [
    "PICK_PLACE_TASKS",
    "ARM_ONLY_ATOMIC_TASKS",
    "PickPlaceTask",
    "RoleKind",
    "RoleSpec",
    "get_task",
    "iter_tasks",
    "validate_manifest",
]
