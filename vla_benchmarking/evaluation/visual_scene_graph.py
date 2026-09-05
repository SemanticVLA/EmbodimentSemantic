"""Arrow-only scene-graph rendering and LeRobot vector-environment integration."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from vla_benchmarking.evaluation.randomize_scenes import _resolve_envs
from vla_benchmarking.evaluation.scene_graph_formats import human_object_name


DEFAULT_ARROW_COLOR_RGB = (0, 166, 107)
# These defaults deliberately preserve the frozen-policy visual-ablation style.
# The sealed LoRA data/eval contract uses the explicit SEALED_* values below.
DEFAULT_ARROW_WIDTH = 1
DEFAULT_ARROW_HEAD_LENGTH = 8
SEALED_LORA_IMAGE_SIZE = 256
SEALED_LORA_ARROW_WIDTH = 1
SEALED_LORA_ARROW_HEAD_LENGTH = 16
SEALED_LORA_VISUAL_CONTRACT = {
    # Width 1 is a new geometry contract; keep it distinct from the prior
    # width-2 v1 artifacts so provenance checks cannot silently mix them.
    "name": "sealed_lora_visual_v2",
    "main_image_size": SEALED_LORA_IMAGE_SIZE,
    "wrist_image_size": SEALED_LORA_IMAGE_SIZE,
    "source_bbox_size": 128,
    "bbox_scaling": "scale_then_clamp_to_image_bounds",
    "arrow_color_rgb": list(DEFAULT_ARROW_COLOR_RGB),
    "line_width": SEALED_LORA_ARROW_WIDTH,
    "head_length": SEALED_LORA_ARROW_HEAD_LENGTH,
    "overlay_order": "resize_main_then_overlay_then_flip180",
}
DEFAULT_GOAL_OBJECT = "plate_1"
VISUAL_ARROWS_CONDITION = "visual_arrows"
VISUAL_GOAL_ARROW_CONDITION = "visual_goal_arrow"
SUPPORTED_VISUAL_CONDITIONS = {
    VISUAL_ARROWS_CONDITION,
    VISUAL_GOAL_ARROW_CONDITION,
}


def goal_arrow_prompt_hint(goal_object: str = DEFAULT_GOAL_OBJECT) -> str:
    goal_name = human_object_name(goal_object)
    return (
        "The green arrow in the image points from the black bowl to "
        f"the {goal_name} where it should be placed."
    )


def bbox_center(bbox: Iterable[int | float]) -> tuple[int, int]:
    """Return the rounded center of an ``(x1, y1, x2, y2)`` bbox."""
    x1, y1, x2, y2 = bbox
    return round((float(x1) + float(x2)) / 2), round((float(y1) + float(y2)) / 2)


def drawable_relations(
    bboxes: dict[str, Iterable[int | float]],
    relations: Iterable[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Split relations into drawable and skipped groups based on bbox availability."""
    drawn: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str, str]] = []
    for relation in relations:
        subject, _, obj = relation
        target = drawn if subject in bboxes and obj in bboxes else skipped
        target.append(relation)
    return drawn, skipped


def select_visual_relations(
    bboxes: dict[str, Iterable[int | float]],
    relations: Iterable[tuple[str, str, str]],
    *,
    condition: str = VISUAL_ARROWS_CONDITION,
    subject: str | None = None,
    goal_object: str = DEFAULT_GOAL_OBJECT,
) -> list[tuple[str, str, str]]:
    """Select which relations the visual overlay should draw."""
    if condition == VISUAL_ARROWS_CONDITION:
        return list(relations)
    if condition != VISUAL_GOAL_ARROW_CONDITION:
        raise ValueError(
            f"unsupported visual condition {condition!r}; expected one of "
            f"{sorted(SUPPORTED_VISUAL_CONDITIONS)}"
        )

    if subject is None:
        for relation_subject, _, _ in relations:
            subject = relation_subject
            break
    if subject is None and bboxes:
        subject = next(iter(sorted(bboxes)))
    if subject is None:
        return []

    return [(subject, "goal", goal_object)]


def resolve_task_goal_object(
    task_id: int | None,
    goal_object: str | Mapping[int, str] | None,
    *,
    default: str = DEFAULT_GOAL_OBJECT,
) -> str:
    if isinstance(goal_object, Mapping):
        if task_id is not None and task_id in goal_object:
            return goal_object[task_id]
        return default
    if goal_object:
        return goal_object
    return default


def draw_scene_graph_arrows(
    image: np.ndarray,
    bboxes: dict[str, Iterable[int | float]],
    relations: Iterable[tuple[str, str, str]],
    *,
    color_rgb: tuple[int, int, int] = DEFAULT_ARROW_COLOR_RGB,
    line_width: int = DEFAULT_ARROW_WIDTH,
    head_length: int = DEFAULT_ARROW_HEAD_LENGTH,
    copy_image: bool = True,
) -> np.ndarray:
    """Draw subject-to-object arrows without labels or bounding boxes.

    The returned array is a ``uint8 HxWx3`` image. It is a copy by default;
    callers on the hot path may set ``copy_image=False`` to draw in place.
    Relations whose subject or object bbox is absent are skipped.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape HxWx3, got {image.shape}")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if head_length <= 0:
        raise ValueError("head_length must be positive")

    canvas = np.ascontiguousarray(image.copy() if copy_image else image)
    drawn, _ = drawable_relations(bboxes, relations)

    for subject, _, obj in drawn:
        start = bbox_center(bboxes[subject])
        end = bbox_center(bboxes[obj])
        if start == end:
            continue

        cv2.line(
            canvas,
            start,
            end,
            color_rgb,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        wing_a = (
            round(end[0] - head_length * math.cos(angle - math.pi / 6)),
            round(end[1] - head_length * math.sin(angle - math.pi / 6)),
        )
        wing_b = (
            round(end[0] - head_length * math.cos(angle + math.pi / 6)),
            round(end[1] - head_length * math.sin(angle + math.pi / 6)),
        )
        cv2.fillConvexPoly(
            canvas,
            np.asarray([end, wing_a, wing_b], dtype=np.int32),
            color_rgb,
            lineType=cv2.LINE_AA,
        )

    return canvas


class VisualRelationAuditLogger:
    """Thread-safe JSONL logger for relations considered by the visual overlay."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | None,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._fh = None
        if not enabled:
            self.path = None
            return

        root = Path(output_dir or ".")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "visual_relation_audit.jsonl"
        self._fh = self.path.open("a", encoding="utf-8", buffering=64 * 1024)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        with self._lock:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class VisualGraphVecEnvWrapper:
    """Overlay live GT scene-graph arrows on the main policy image.

    The wrapper changes only ``pixels/image``. It caches that exact policy
    image and serves it through ``call("render")`` so LeRobot videos show the
    same orientation and pixels received by the policy.
    """

    def __init__(
        self,
        env: Any,
        live_generator: Any,
        *,
        image_key: str = "image",
        condition: str = VISUAL_ARROWS_CONDITION,
        goal_object: str | Mapping[int, str] = DEFAULT_GOAL_OBJECT,
        audit_logger: VisualRelationAuditLogger | None = None,
        line_width: int = DEFAULT_ARROW_WIDTH,
        head_length: int = DEFAULT_ARROW_HEAD_LENGTH,
    ) -> None:
        if condition not in SUPPORTED_VISUAL_CONDITIONS:
            raise ValueError(
                f"unsupported visual condition {condition!r}; expected one of "
                f"{sorted(SUPPORTED_VISUAL_CONDITIONS)}"
            )
        self.env = env
        self.live_generator = live_generator
        self.image_key = image_key
        self.condition = condition
        self.goal_object = goal_object
        self.audit_logger = audit_logger
        if line_width <= 0:
            raise ValueError("line_width must be positive")
        if head_length <= 0:
            raise ValueError("head_length must be positive")
        self.line_width = line_width
        self.head_length = head_length
        self._sub_envs = _resolve_envs(env)
        self._cameras = [
            self._camera_for_slot(sub_env, image_key)
            for sub_env in self._sub_envs
        ]
        self._latest_overlaid: list[np.ndarray | None] = [None] * len(self._sub_envs)
        self._warned_camera_fallbacks: set[tuple[int | None, str]] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @staticmethod
    def _camera_for_slot(sub_env: Any, image_key: str) -> str:
        mapping = getattr(sub_env, "camera_name_mapping", {})
        for raw_camera, mapped_key in mapping.items():
            if mapped_key == image_key:
                return str(raw_camera).removesuffix("_image")
        return "agentview"

    @staticmethod
    def _env_step(sub_env: Any) -> int | None:
        for candidate in (sub_env, getattr(sub_env, "_env", None)):
            if candidate is None:
                continue
            for attribute in ("_elapsed_steps", "timestep"):
                value = getattr(candidate, attribute, None)
                if isinstance(value, (int, np.integer)):
                    return int(value)
        return None

    def _overlay_observation(
        self,
        observation: dict[str, Any],
        env_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        try:
            batch_images = observation["pixels"][self.image_key]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"VisualGraphVecEnvWrapper expected observation['pixels'][{self.image_key!r}]"
            ) from exc

        if env_indices is None:
            env_indices = list(range(len(self._sub_envs)))

        if len(batch_images) != len(env_indices):
            raise ValueError(
                "VisualGraphVecEnvWrapper received a batch whose length does not match "
                f"the selected environments: {len(batch_images)} != {len(env_indices)}"
            )

        for observation_index, env_index in enumerate(env_indices):
            sub_env = self._sub_envs[env_index]
            camera = self._cameras[env_index]
            if camera != "agentview":
                fallback_key = (getattr(sub_env, "task_id", None), camera)
                if fallback_key not in self._warned_camera_fallbacks:
                    self._warned_camera_fallbacks.add(fallback_key)
                    print(
                        "[visual_arrows] Main policy slot is "
                        f"{camera!r}, not 'agentview', for task "
                        f"{getattr(sub_env, 'task_id', 'unknown')}; using matching {camera} bboxes."
                    )

            context = self.live_generator.observe_visual_graph(sub_env, camera=camera)
            source_relations = context["relations"]
            bboxes = context["bboxes"]
            task_id = getattr(sub_env, "task_id", None)
            task_goal_object = resolve_task_goal_object(task_id, self.goal_object)
            relations = select_visual_relations(
                bboxes,
                source_relations,
                condition=self.condition,
                subject=self.live_generator.scene_graph_subject_filter,
                goal_object=task_goal_object,
            )
            drawn, skipped = drawable_relations(bboxes, relations)
            overlaid = draw_scene_graph_arrows(
                batch_images[observation_index],
                bboxes,
                relations,
                line_width=self.line_width,
                head_length=self.head_length,
                copy_image=False,
            )
            batch_images[observation_index] = overlaid
            self._latest_overlaid[env_index] = overlaid.copy()

            if self.audit_logger is not None:
                self.audit_logger.log(
                    {
                        "condition": self.condition,
                        "task_id": task_id,
                        "env_step": self._env_step(sub_env),
                        "camera": camera,
                        "image_slot": f"pixels/{self.image_key}",
                        "line_width": self.line_width,
                        "head_length": self.head_length,
                        "subject_filter": self.live_generator.scene_graph_subject_filter,
                        "goal_object": task_goal_object if self.condition == VISUAL_GOAL_ARROW_CONDITION else None,
                        "visible_bboxes": sorted(bboxes),
                        "relations": [list(item) for item in source_relations],
                        "selected_relations": [list(item) for item in relations],
                        "drawn_relations": [list(item) for item in drawn],
                        "skipped_relations_missing_bbox": [list(item) for item in skipped],
                    }
                )

        return observation

    def reset(self, id=None, **kwargs):
        result = self.env.reset(**kwargs) if id is None else self.env.reset(id=id, **kwargs)
        returns_info = isinstance(result, tuple) and len(result) == 2
        observation = result[0] if returns_info else result

        if id is None:
            env_indices = None
        elif isinstance(id, int):
            env_indices = [id]
        else:
            env_indices = list(id)

        observation = self._overlay_observation(observation, env_indices)
        return (observation, result[1]) if returns_info else observation

    def step(self, actions):
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) < 1:
            raise TypeError("Vector environment step must return a tuple beginning with observations")
        observation = self._overlay_observation(result[0])
        return (observation, *result[1:])

    def call(self, name: str, *args, **kwargs):
        if name != "render":
            return self.env.call(name, *args, **kwargs)

        missing = [index for index, image in enumerate(self._latest_overlaid) if image is None]
        if missing:
            rendered = self.env.call(name, *args, **kwargs)
            return [
                rendered[index] if image is None else image.copy()
                for index, image in enumerate(self._latest_overlaid)
            ]
        return [image.copy() for image in self._latest_overlaid if image is not None]
