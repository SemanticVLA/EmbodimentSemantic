#!/usr/bin/env python3
"""Live LIBERO semantic annotations for prompt augmentation.

This module is adapted from `LIBERO_Semantic_Generation.ipynb`. It computes
object bounding boxes from the current MuJoCo simulator state, then derives
scene-graph triplets from those bboxes and object world poses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from robosuite.utils import camera_utils as CU

from scene_graph_formats import (
    LEGACY_FORMAT,
    dedupe_relations,
    format_scene_context,
    normalize_context_format,
)


BBOX_PADDING = 5
CONTAINMENT_THRESH = 0.8
EXCLUDE_PREFIXES = ("robot0", "gripper")
_BBOX_CORNER_SIGNS = np.asarray(
    [
        (sx, sy, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ],
    dtype=np.float64,
)
# DEFAULT_OBJECTS = {
#     "akita_black_bowl_1",
#     "akita_black_bowl_2",
#     "cookies_1",
#     "plate_1",
#     "glazed_rim_porcelain_ramekin_1",
# }


# def _inner_env(env: Any) -> Any:
#     return getattr(env, "_env", env)


def _camera_name_for_mujoco(camera_name: str) -> str:
    return camera_name.removesuffix("_image")


def _strip_instance_suffix(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _format_scene_graph_triplet(
    subject: str,
    relation: str,
    obj: str,
    *,
    target_subject: str | None = None,
) -> str:
    subject_name = _strip_instance_suffix(subject)
    object_name = _strip_instance_suffix(obj)

    if target_subject is not None and subject == target_subject:
        subject_name = f"target_{subject_name}"
    return f"({subject_name}, {relation}, {object_name})"


def _format_scene_graph_text(
    scene_graph: dict[str, list[tuple[str, str, str]]],
    *,
    target_subject: str | None = None,
) -> str:
    lines = ["Scene graph:"]

    for view_name, triplets in scene_graph.items():
        lines.append(f"View: {view_name}")
        if not triplets:
            lines.append("(none)")
            continue
        for subject, relation, obj in triplets:
            lines.append(
                _format_scene_graph_triplet(
                    subject,
                    relation,
                    obj,
                    target_subject=target_subject,
                )
            )

    return "\n".join(lines)


def _format_bounding_boxes_text(bboxes_by_view: dict[str, dict[str, list[int]]]) -> str:
    return "Bounding boxes:\n" + json.dumps(bboxes_by_view, separators=(",", ":"), sort_keys=True)


def get_all_geoms_for_body(env: Any, body_name: str) -> list[int]:
    # env = _inner_env(env)
    root_id = env.sim.model.body_name2id(body_name)
    descendant_ids = set()
    for bid in range(env.sim.model.nbody):
        current = bid
        while current != 0:
            if current == root_id:
                descendant_ids.add(bid)
                break
            current = env.sim.model.body_parentid[current]
    return [gid for gid in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[gid] in descendant_ids]


def world_to_pixel(
    world_pos: np.ndarray,
    extrinsic_inv: np.ndarray,
    K: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[int, int] | None:
    pt_cam = extrinsic_inv @ np.append(world_pos, 1.0)
    if pt_cam[2] <= 0.01:
        return None
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return int(u), int(img_h - v)

def get_bbox(
    env,
    body_name,
    extrinsic_inv,
    K,
    img_h,
    img_w,
    padding=BBOX_PADDING,
    *,
    geom_ids=None,
):
    """Project a body's geometry corners with one vectorized matrix operation."""
    if geom_ids is None:
        geom_ids = get_all_geoms_for_body(env, body_name)
    if not geom_ids:
        return None

    geom_ids = np.asarray(geom_ids, dtype=np.int32)
    positions = np.asarray(env.sim.data.geom_xpos)[geom_ids]
    sizes = np.asarray(env.sim.model.geom_size)[geom_ids]
    rotations = np.asarray(env.sim.data.geom_xmat)[geom_ids].reshape(-1, 3, 3)
    local_corners = sizes[:, None, :] * _BBOX_CORNER_SIGNS[None, :, :]
    rotated_corners = np.einsum(
        "gij,gkj->gki",
        rotations,
        local_corners,
        optimize=True,
    )
    world_corners = (positions[:, None, :] + rotated_corners).reshape(-1, 3)
    homogeneous = np.concatenate(
        [world_corners, np.ones((world_corners.shape[0], 1), dtype=world_corners.dtype)],
        axis=1,
    )
    camera_points = homogeneous @ np.asarray(extrinsic_inv).T
    camera_points = camera_points[camera_points[:, 2] > 0.01]
    if camera_points.size == 0:
        return None

    us = (
        K[0, 0] * camera_points[:, 0] / camera_points[:, 2]
        + K[0, 2]
    ).astype(np.int64)
    vs = (
        img_h
        - (
            K[1, 1] * camera_points[:, 1] / camera_points[:, 2]
            + K[1, 2]
        )
    ).astype(np.int64)
    rx1, ry1 = int(us.min()) - padding, int(vs.min()) - padding
    rx2, ry2 = int(us.max()) + padding, int(vs.max()) + padding
    if (rx2-rx1)>(img_w*3) or (ry2-ry1)>(img_h*3): return None
    x1, y1, x2, y2 = max(0,rx1), max(0,ry1), min(img_w,rx2), min(img_h,ry2)
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def discover_objects(env: Any) -> dict[str, str]:
    return {env.sim.model.body_id2name(i).replace("_main",""): env.sim.model.body_id2name(i)
           for i in range(env.sim.model.nbody) if env.sim.model.body_id2name(i).endswith("_main")
           and not any(env.sim.model.body_id2name(i).startswith(p) for p in EXCLUDE_PREFIXES)}

def generate_frame_graph(
    bboxes,
    world,
    object_filter=None,
    is_drawer_task=False,
    subject_filter=None,
):
    if object_filter is not None:
        objects = sorted([o for o in bboxes.keys() if o in object_filter])
    else:
        objects = sorted(list(bboxes.keys()))

    triplets = []

    subjects = objects if subject_filter is None else [subject_filter]
    for A in subjects:
        if A not in objects:
            continue
        if A not in world: continue
        pos_a = np.array(world[A]['pos'])
        for B in objects:
            if A == B: continue
            if B not in world: continue
            pos_b = np.array(world[B]['pos'])

            is_stacked = False
            if A in bboxes and B in bboxes:
                tx1,ty1,tx2,ty2 = bboxes[A]
                bx1,by1,bx2,by2 = bboxes[B]
                ix1,iy1 = max(tx1,bx1), max(ty1,by1)
                ix2,iy2 = min(tx2,bx2), min(ty2,by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    a_area = (tx2-tx1)*(ty2-ty1)
                    b_area = (bx2-bx1)*(by2-by1)
                    io_min = inter / min(a_area, b_area)
                    if io_min > CONTAINMENT_THRESH and ("bowl" in A or "bowl" in B):
                        is_stacked = True
                        pair = {A, B} == {"akita_black_bowl_1", "wooden_cabinet_1"}
                        if is_drawer_task and pair:
                            if pos_a[2] >= pos_b[2]:
                                triplets.append((A, "is_inside", B))
                            else:
                                triplets.append((A, "contains", B))
                        else:
                            if pos_a[2] >= pos_b[2]:
                                triplets.append((A, "is_on_top_of", B))
                            else:
                                triplets.append((A, "is_below_of", B))
            if is_stacked: continue

            dx = pos_a[0] - pos_b[0]
            dy = pos_a[1] - pos_b[1]
            if abs(dx) >= abs(dy):
                triplets.append((A, "is_in_front_of" if dx > 0 else "is_behind", B))
            else:
                triplets.append((A, "is_left_of" if dy > 0 else "is_right_of", B))

    return triplets

@dataclass
class LiveSemanticContextGenerator:
    max_json_chars: int = 2500
    scene_graph_subject_filter: str | None = None  # if set, only keep triplets whose subject matches
    _objects_cache: dict[int, tuple[Any, dict[str, str]]] = field(default_factory=dict)
    _intrinsic_cache: dict[
        tuple[int, str, int, int],
        tuple[Any, np.ndarray],
    ] = field(default_factory=dict)
    _extrinsic_cache: dict[
        tuple[int, str],
        tuple[Any, np.ndarray],
    ] = field(default_factory=dict)
    _object_geometry_cache: dict[
        int,
        tuple[Any, dict[str, tuple[int, tuple[int, ...]]]],
    ] = field(default_factory=dict)

    def _objects(self, env: Any) -> dict[str, str]:
        # inner = _inner_env(env)
        key = id(env)
        model = env.sim.model
        cached = self._objects_cache.get(key)
        if cached is None or cached[0] is not model:
            cached = (model, discover_objects(env))
            self._objects_cache[key] = cached
        return cached[1]

    def _intrinsic(self, env: Any, camera: str, img_h: int, img_w: int) -> np.ndarray:
        # inner = _inner_env(env)
        key = (id(env), camera, img_h, img_w)
        model = env.sim.model
        cached = self._intrinsic_cache.get(key)
        if cached is None or cached[0] is not model:
            cached = (
                model,
                CU.get_camera_intrinsic_matrix(env.sim, camera, img_h, img_w),
            )
            self._intrinsic_cache[key] = cached
        return cached[1]

    def _extrinsic_inv(self, env: Any, camera: str) -> np.ndarray:
        """Cache fixed-camera extrinsics; recompute robot-mounted camera poses."""
        model = env.sim.model
        camera_id = model.camera_name2id(camera)
        camera_body_id = int(model.cam_bodyid[camera_id])
        if camera_body_id != 0:
            return np.linalg.inv(CU.get_camera_extrinsic_matrix(env.sim, camera))

        key = (id(env), camera)
        cached = self._extrinsic_cache.get(key)
        if cached is None or cached[0] is not model:
            cached = (
                model,
                np.linalg.inv(CU.get_camera_extrinsic_matrix(env.sim, camera)),
            )
            self._extrinsic_cache[key] = cached
        return cached[1]

    def _object_geometry(
        self,
        env: Any,
    ) -> dict[str, tuple[int, tuple[int, ...]]]:
        """Cache body and descendant-geometry IDs, which are static per model."""
        key = id(env)
        model = env.sim.model
        cached = self._object_geometry_cache.get(key)
        if cached is None or cached[0] is not model:
            geometry = {}
            for label, body in self._objects(env).items():
                geometry[label] = (
                    model.body_name2id(body),
                    tuple(get_all_geoms_for_body(env, body)),
                )
            cached = (model, geometry)
            self._object_geometry_cache[key] = cached
        return cached[1]

    def observe_env(
        self,
        env: Any,
        *,
        include_scene_graph: bool = True,
        include_bboxes: bool = True,
    ) -> dict[str, Any]:
        inner = env._env #_inner_env(env)
        img_h = env.observation_height # int(getattr(env, "observation_height", 256))
        img_w = env.observation_width # int(getattr(env, "observation_width", 256))
        is_drawer_task = "in_the_top_drawer" in env.task
        objects = self._objects(inner)
        object_geometry = self._object_geometry(inner)
        # Bboxes and scene graph: always agentview only, filtered by scene_graph_subject_filter.
        bbox_cameras = ["agentview"]
        all_cameras = ["agentview"]

        world: dict[str, dict[str, Any]] = {}
        for label in objects:
            bid, _ = object_geometry[label]
            world[label] = {
                "pos": inner.sim.data.body_xpos[bid].tolist(),
            }

        bboxes_by_camera: dict[str, dict[str, list[int]]] = {}
        graphs_by_camera: dict[str, list[tuple[str, str, str]]] = {}

        for camera in all_cameras:
            K = self._intrinsic(inner, camera, img_h, img_w)
            ext_inv = self._extrinsic_inv(inner, camera)
            camera_bboxes = {}
            for label, body in objects.items():
                _, geom_ids = object_geometry[label]
                bbox = get_bbox(
                    inner,
                    body,
                    ext_inv,
                    K,
                    img_h,
                    img_w,
                    geom_ids=geom_ids,
                )
                if bbox is not None:
                    camera_bboxes[label] = list(bbox)

            if include_bboxes and camera in bbox_cameras:
                if self.scene_graph_subject_filter is not None:
                    bboxes_by_camera[camera] = {
                        k: v for k, v in camera_bboxes.items()
                        if k == self.scene_graph_subject_filter
                    }
                else:
                    bboxes_by_camera[camera] = camera_bboxes

            if include_scene_graph and camera == "agentview":
                triplets = generate_frame_graph(
                    camera_bboxes,
                    world,
                    object_filter=set(camera_bboxes.keys()),
                    is_drawer_task=is_drawer_task,
                    subject_filter=self.scene_graph_subject_filter,
                )
                graphs_by_camera["agentview"] = triplets

        return {
            "bboxes": bboxes_by_camera,
            "scene_graph": graphs_by_camera,
        }

    def observe_visual_graph(
        self,
        env: Any,
        *,
        camera: str = "agentview",
    ) -> dict[str, Any]:
        """Return all visible bboxes and target-filtered GT relations for one view."""
        inner = env._env
        img_h = env.observation_height
        img_w = env.observation_width
        is_drawer_task = "in_the_top_drawer" in env.task
        objects = self._objects(inner)
        object_geometry = self._object_geometry(inner)

        world: dict[str, dict[str, Any]] = {}
        for label in objects:
            bid, _ = object_geometry[label]
            world[label] = {
                "pos": inner.sim.data.body_xpos[bid].tolist(),
            }

        K = self._intrinsic(inner, camera, img_h, img_w)
        ext_inv = self._extrinsic_inv(inner, camera)
        bboxes: dict[str, list[int]] = {}
        for label, body in objects.items():
            _, geom_ids = object_geometry[label]
            bbox = get_bbox(
                inner,
                body,
                ext_inv,
                K,
                img_h,
                img_w,
                geom_ids=geom_ids,
            )
            if bbox is not None:
                bboxes[label] = list(bbox)

        relations = generate_frame_graph(
            bboxes,
            world,
            object_filter=set(bboxes),
            is_drawer_task=is_drawer_task,
            subject_filter=self.scene_graph_subject_filter,
        )

        return {
            "camera": camera,
            "bboxes": bboxes,
            "relations": relations,
        }

    def build_prompt(
        self,
        env: Any,
        task_text: str,
        mode: str,
        context_format: str = LEGACY_FORMAT,
    ) -> tuple[str, list[tuple[str, str, str]], int]:
        mode = mode.strip().lower()
        context_format = normalize_context_format(context_format)
        if mode == "standard" or context_format == "standard":
            return task_text, [], 0

        include_scene_graph = mode in {"scene_graph", "scene_graph_bounding_boxes"}
        include_bboxes = mode in {"bounding_boxes", "scene_graph_bounding_boxes"}
        context = self.observe_env(
            env,
            include_scene_graph=include_scene_graph,
            include_bboxes=include_bboxes,
        )
        relations = context["scene_graph"].get("agentview", []) if include_scene_graph else []
        retained_relations = dedupe_relations(relations)

        if include_scene_graph and context_format != LEGACY_FORMAT:
            prompt = format_scene_context(task_text, retained_relations, context_format)
            if include_bboxes:
                prompt = (
                    f"{prompt}\nCurrent simulator semantic context:\n"
                    f"{_format_bounding_boxes_text(context['bboxes'])}"
                )
            return prompt, relations, len(retained_relations)

        sections: list[str] = []
        if include_scene_graph:
            sections.append(
                _format_scene_graph_text(
                    context["scene_graph"],
                    target_subject=self.scene_graph_subject_filter,
                )
            )
        if include_bboxes:
            sections.append(_format_bounding_boxes_text(context["bboxes"]))

        text = "\n\n".join(sections)
        if len(text) > self.max_json_chars:
            text = text[: self.max_json_chars] + "...<truncated>"
        return (
            f"{task_text}\nCurrent simulator semantic context:\n{text}",
            relations,
            len(retained_relations),
        )

    def prompt_suffix(self, env: Any, mode: str) -> str:
        mode = mode.strip().lower()
        if mode == "standard":
            return ""
        include_scene_graph = mode in {"scene_graph", "scene_graph_bounding_boxes"}
        include_bboxes = mode in {"bounding_boxes", "scene_graph_bounding_boxes"}
        context = self.observe_env(
            env,
            include_scene_graph=include_scene_graph,
            include_bboxes=include_bboxes,
        )
        sections: list[str] = []
        if include_scene_graph:
            sections.append(
                _format_scene_graph_text(context["scene_graph"],target_subject=self.scene_graph_subject_filter)
            )
        if include_bboxes:
            sections.append(_format_bounding_boxes_text(context["bboxes"]))

        text = "\n\n".join(sections)
        if len(text) > self.max_json_chars:
            text = text[: self.max_json_chars] + "...<truncated>"
        return f"\nCurrent simulator semantic context:\n{text}"
