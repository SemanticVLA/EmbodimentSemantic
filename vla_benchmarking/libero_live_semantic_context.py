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


BBOX_PADDING = 5
CONTAINMENT_THRESH = 0.8
EXCLUDE_PREFIXES = ("robot0", "gripper")
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

def get_bbox(env, body_name, extrinsic_inv, K, img_h, img_w, padding=BBOX_PADDING):
    geom_ids = get_all_geoms_for_body(env, body_name)
    if not geom_ids: return None
    pixels = []
    for gid in geom_ids:
        pos, size = env.sim.data.geom_xpos[gid], env.sim.model.geom_size[gid]
        xmat = env.sim.data.geom_xmat[gid].reshape(3, 3)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    rotated_offset = xmat @ (np.array([sx, sy, sz]) * size)
                    px = world_to_pixel(pos + rotated_offset, extrinsic_inv, K, img_h, img_w)
                    if px is not None: pixels.append(px)
    if not pixels: return None
    us, vs = zip(*pixels)
    rx1, ry1, rx2, ry2 = min(us)-padding, min(vs)-padding, max(us)+padding, max(vs)+padding
    if (rx2-rx1)>(img_w*3) or (ry2-ry1)>(img_h*3): return None
    x1, y1, x2, y2 = max(0,int(rx1)), max(0,int(ry1)), min(img_w,int(rx2)), min(img_h,int(ry2))
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def discover_objects(env: Any) -> dict[str, str]:
    return {env.sim.model.body_id2name(i).replace("_main",""): env.sim.model.body_id2name(i)
           for i in range(env.sim.model.nbody) if env.sim.model.body_id2name(i).endswith("_main")
           and not any(env.sim.model.body_id2name(i).startswith(p) for p in EXCLUDE_PREFIXES)}

def generate_frame_graph(bboxes, world, object_filter=None, is_drawer_task=False):
    if object_filter is not None:
        objects = sorted([o for o in bboxes.keys() if o in object_filter])
    else:
        objects = sorted(list(bboxes.keys()))

    triplets = []

    for A in objects:
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
    _objects_cache: dict[int, dict[str, str]] = field(default_factory=dict)
    _intrinsic_cache: dict[tuple[int, str, int, int], np.ndarray] = field(default_factory=dict)

    def _objects(self, env: Any) -> dict[str, str]:
        # inner = _inner_env(env)
        key = id(env)
        if key not in self._objects_cache:
            self._objects_cache[key] = discover_objects(env)
        return self._objects_cache[key]

    def _intrinsic(self, env: Any, camera: str, img_h: int, img_w: int) -> np.ndarray:
        # inner = _inner_env(env)
        key = (id(env), camera, img_h, img_w)
        if key not in self._intrinsic_cache:
            self._intrinsic_cache[key] = CU.get_camera_intrinsic_matrix(env.sim, camera, img_h, img_w)
        return self._intrinsic_cache[key]

    def observe_env(self, env: Any) -> dict[str, Any]:
        inner = env._env #_inner_env(env)
        img_h = env.observation_height # int(getattr(env, "observation_height", 256))
        img_w = env.observation_width # int(getattr(env, "observation_width", 256))
        is_drawer_task = "in_the_top_drawer" in env.task
        objects = self._objects(inner)
        # Bboxes and scene graph: always agentview only, filtered by scene_graph_subject_filter.
        bbox_cameras = ["agentview"]
        all_cameras = ["agentview"]

        world: dict[str, dict[str, Any]] = {}
        for label, body in objects.items():
            bid = inner.sim.model.body_name2id(body)
            world[label] = {
                "pos": inner.sim.data.body_xpos[bid].tolist(),
                "mat": inner.sim.data.body_xmat[bid].reshape(3, 3).tolist(),
            }

        bboxes_by_camera: dict[str, dict[str, list[int]]] = {}
        graphs_by_camera: dict[str, list[tuple[str, str, str]]] = {}

        for camera in all_cameras:
            K = self._intrinsic(inner, camera, img_h, img_w)
            ext_inv = np.linalg.inv(CU.get_camera_extrinsic_matrix(inner.sim, camera))
            camera_bboxes = {}
            for label, body in objects.items():
                bbox = get_bbox(inner, body, ext_inv, K, img_h, img_w)
                if bbox is not None:
                    camera_bboxes[label] = list(bbox)

            if camera in bbox_cameras:
                if self.scene_graph_subject_filter is not None:
                    bboxes_by_camera[camera] = {
                        k: v for k, v in camera_bboxes.items()
                        if k == self.scene_graph_subject_filter
                    }
                else:
                    bboxes_by_camera[camera] = camera_bboxes

            if camera == "agentview":
                triplets = generate_frame_graph(
                    camera_bboxes,
                    world,
                    object_filter=set(camera_bboxes.keys()),
                    is_drawer_task=is_drawer_task,
                )
                if self.scene_graph_subject_filter is not None:
                    triplets = [t for t in triplets if t[0] == self.scene_graph_subject_filter]
                graphs_by_camera["agentview"] = triplets

        return {
            "bboxes": bboxes_by_camera,
            "scene_graph": graphs_by_camera,
        }

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
