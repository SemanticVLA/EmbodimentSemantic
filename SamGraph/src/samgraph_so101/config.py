"""Validated SO101 task, object, and image-space geometry configuration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SO101Object:
    object_id: str
    role: str
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class SO101Task:
    task_id: str
    instruction: str
    objects: tuple[str, ...]


@dataclass(frozen=True)
class SO101Config:
    path: Path
    sha256: str
    schema: str
    objects: tuple[SO101Object, ...]
    tasks: Mapping[str, SO101Task]
    geometry: Mapping[str, Any]
    camera_convention: Mapping[str, Any]
    persistence: Mapping[str, Any]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.objects)

    def prompts_for_tasks(self, task_ids: list[str]) -> dict[str, dict[str, list[str]]]:
        prompts = {item.object_id: list(item.descriptions) for item in self.objects}
        return {task_id: {key: list(values) for key, values in prompts.items()}
                for task_id in task_ids}


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def load_so101_config(path: str | Path) -> SO101Config:
    source = Path(path).resolve()
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("SO101 configuration must be a JSON object")
    schema = _nonempty_string(value.get("schema"), "schema")
    objects_value = value.get("objects")
    tasks_value = value.get("tasks")
    geometry = value.get("geometry")
    camera = value.get("camera_convention")
    persistence = value.get("persistence")
    if not isinstance(objects_value, dict) or not objects_value:
        raise ValueError("objects must be a non-empty object")
    if not isinstance(tasks_value, dict) or not tasks_value:
        raise ValueError("tasks must be a non-empty object")
    if not isinstance(geometry, dict) or not geometry:
        raise ValueError("geometry must be a non-empty object")
    if not isinstance(camera, dict) or not camera:
        raise ValueError("camera_convention must be a non-empty object")
    if not isinstance(persistence, dict) or not persistence:
        raise ValueError("persistence must be a non-empty object")
    if camera.get("depth_claim") is not False:
        raise ValueError("SO101 camera convention must explicitly disable depth claims")
    objects = []
    for object_id, spec in objects_value.items():
        object_id = _nonempty_string(object_id, "object ID")
        if not isinstance(spec, dict):
            raise ValueError(f"object {object_id} must be an object")
        role = _nonempty_string(spec.get("role"), f"objects.{object_id}.role")
        descriptions = spec.get("descriptions")
        if (not isinstance(descriptions, list) or not descriptions
                or any(not isinstance(item, str) or not item.strip() for item in descriptions)):
            raise ValueError(f"objects.{object_id}.descriptions must be non-empty strings")
        if spec.get("count", 1) != 1:
            raise ValueError("SO101 v1 requires one canonical instance per configured object")
        objects.append(SO101Object(object_id, role, tuple(item.strip() for item in descriptions)))
    object_ids = {item.object_id for item in objects}
    manipulated = persistence.get("manipulated_object_ids")
    max_remembered = persistence.get("max_remembered_native_frames")
    if (not isinstance(manipulated, list) or not manipulated
            or any(not isinstance(item, str) for item in manipulated)
            or len(manipulated) != len(set(manipulated))
            or not set(manipulated) <= object_ids):
        raise ValueError("persistence.manipulated_object_ids must be unique configured objects")
    if not isinstance(max_remembered, int) or not 0 <= max_remembered <= 30:
        raise ValueError("persistence.max_remembered_native_frames must be in [0, 30]")
    tasks: dict[str, SO101Task] = {}
    for task_id, spec in tasks_value.items():
        task_id = _nonempty_string(task_id, "task ID")
        if not isinstance(spec, dict):
            raise ValueError(f"task {task_id} must be an object")
        instruction = _nonempty_string(spec.get("instruction"), f"tasks.{task_id}.instruction")
        task_objects = spec.get("objects")
        if not isinstance(task_objects, list) or set(task_objects) != object_ids:
            raise ValueError(
                f"tasks.{task_id}.objects must contain every configured SO101 object exactly once"
            )
        if len(task_objects) != len(set(task_objects)):
            raise ValueError(f"tasks.{task_id}.objects contains duplicates")
        tasks[task_id] = SO101Task(task_id, instruction, tuple(task_objects))
    return SO101Config(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema=schema,
        objects=tuple(objects),
        tasks=tasks,
        geometry=geometry,
        camera_convention=camera,
        persistence=persistence,
    )


__all__ = ["SO101Config", "SO101Object", "SO101Task", "load_so101_config"]
