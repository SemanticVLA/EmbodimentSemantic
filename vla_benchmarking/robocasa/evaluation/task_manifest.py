"""Compatibility view of the authoritative RoboCasa Pick & Place manifest.

The shared manifest owns task roles and official success descriptions.  This
module keeps the small field-oriented shape used by the LIBERO-style runner,
but derives every entry from that single source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..shared.task_manifest import (
    ARM_ONLY_ATOMIC_TASKS as _SHARED_ARM_ONLY_ATOMIC_TASKS,
    PICK_PLACE_TASKS as _SHARED_PICK_PLACE_TASKS,
    PickPlaceTask as _SharedPickPlaceTask,
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source_key: str | None
    destination_key: str
    source_label_mode: str = "get_obj_lang"
    destination_kind: str = "object"
    source_choices: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_shared(task: _SharedPickPlaceTask) -> TaskSpec:
    source_label_mode = "static" if task.source.label_mode == "static" else "get_obj_lang"
    source_choices = ("ice_cube1", "ice_cube2") if task.source_selector else ()
    return TaskSpec(
        name=task.name,
        source_key=task.source.key,
        destination_key=task.destination.key,
        source_label_mode=source_label_mode,
        destination_kind=str(task.destination.kind),
        source_choices=source_choices,
        notes=task.notes,
    )


PICK_PLACE_TASKS: tuple[TaskSpec, ...] = tuple(
    _from_shared(task) for task in _SHARED_PICK_PLACE_TASKS
)
ARM_ONLY_ATOMIC_TASKS: tuple[TaskSpec, ...] = tuple(
    _from_shared(task) for task in _SHARED_ARM_ONLY_ATOMIC_TASKS
)
TASKS_BY_NAME: Mapping[str, TaskSpec] = {
    task.name: task for task in (*PICK_PLACE_TASKS, *ARM_ONLY_ATOMIC_TASKS)
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS_BY_NAME[str(name)]
    except KeyError as exc:
        raise KeyError(f"unknown RoboCasa Pick & Place task {name!r}") from exc


def manifest() -> list[dict[str, Any]]:
    return [task.as_dict() for task in PICK_PLACE_TASKS]


__all__ = [
    "PICK_PLACE_TASKS", "ARM_ONLY_ATOMIC_TASKS", "TASKS_BY_NAME", "TaskSpec",
    "get_task", "manifest",
]
