from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import hypot, log
from statistics import median
from typing import Iterable

from .schemas import BBox, BBoxFrame, BBoxObject
from .task_priors import object_role


INVERSE_RELATIONS = {
    "is_left_of": "is_right_of",
    "is_right_of": "is_left_of",
    "is_in_front_of": "is_behind",
    "is_behind": "is_in_front_of",
    "is_on_top_of": "is_below_of",
    "is_below_of": "is_on_top_of",
    "is_inside": "contains",
    "contains": "is_inside",
}

RELATION_FAMILY = {
    "is_left_of": "axis_lr",
    "is_right_of": "axis_lr",
    "is_in_front_of": "axis_fb",
    "is_behind": "axis_fb",
    "is_on_top_of": "support",
    "is_below_of": "support",
    "is_inside": "containment",
    "contains": "containment",
}


def bbox_center(box: BBox, width: int, height: int) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height)


def bbox_center_pixels(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_iomin(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    minimum = min(bbox_area(a), bbox_area(b))
    return inter / minimum if minimum > 0 else 0.0


def bbox_containment(inner: BBox, outer: BBox) -> float:
    area = bbox_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def rotate_bbox_180(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    return width - x2, height - y2, width - x1, height - y1


def axis_margin(a: BBox, b: BBox, width: int, height: int) -> float:
    ua, va = bbox_center(a, width, height)
    ub, vb = bbox_center(b, width, height)
    du, dv = abs(ua - ub), abs(va - vb)
    scale = max(du, dv, 1e-9)
    return abs(du - dv) / scale


def deterministic_axis_relation(a: BBox, b: BBox, width: int, height: int) -> str:
    ua, va = bbox_center(a, width, height)
    ub, vb = bbox_center(b, width, height)
    du, dv = ua - ub, va - vb
    if abs(dv) >= abs(du):
        return "is_in_front_of" if dv > 0 else "is_behind"
    return "is_right_of" if du > 0 else "is_left_of"


def midpoint_arrow_evidence(subject_box: BBox, object_box: BBox, width: int, height: int) -> dict[str, float | str | list[float]]:
    subject_center = bbox_center(subject_box, width, height)
    object_center = bbox_center(object_box, width, height)
    subject_center_px = bbox_center_pixels(subject_box)
    object_center_px = bbox_center_pixels(object_box)
    subject_relative_dx = subject_center[0] - object_center[0]
    subject_relative_dy = subject_center[1] - object_center[1]
    arrow_dx = object_center[0] - subject_center[0]
    arrow_dy = object_center[1] - subject_center[1]
    dominant_axis = "front_back" if abs(subject_relative_dy) >= abs(subject_relative_dx) else "left_right"
    return {
        "basis": "bbox_midpoint_arrow",
        "subject_center": [round(subject_center_px[0], 3), round(subject_center_px[1], 3)],
        "object_center": [round(object_center_px[0], 3), round(object_center_px[1], 3)],
        "subject_center_norm": [round(subject_center[0], 6), round(subject_center[1], 6)],
        "object_center_norm": [round(object_center[0], 6), round(object_center[1], 6)],
        "arrow_dx_norm": round(arrow_dx, 6),
        "arrow_dy_norm": round(arrow_dy, 6),
        "subject_relative_dx_norm": round(subject_relative_dx, 6),
        "subject_relative_dy_norm": round(subject_relative_dy, 6),
        "dominant_axis": dominant_axis,
        "axis_margin": round(axis_margin(subject_box, object_box, width, height), 6),
        "iou": round(bbox_iou(subject_box, object_box), 6),
        "iomin": round(bbox_iomin(subject_box, object_box), 6),
        "subject_in_object": round(bbox_containment(subject_box, object_box), 6),
        "object_in_subject": round(bbox_containment(object_box, subject_box), 6),
    }


def relation_from_family(family: str, a_name: str, b_name: str, a: BBox, b: BBox, width: int, height: int) -> str:
    if family in {"axis_lr", "axis_fb"}:
        ua, va = bbox_center(a, width, height)
        ub, vb = bbox_center(b, width, height)
        if family == "axis_fb":
            return "is_in_front_of" if va - vb > 0 else "is_behind"
        return "is_right_of" if ua - ub > 0 else "is_left_of"

    a_bowl = object_role(a_name) == "bowl"
    b_bowl = object_role(b_name) == "bowl"
    if not (a_bowl or b_bowl):
        return deterministic_axis_relation(a, b, width, height)
    if family == "containment":
        return "is_inside" if a_bowl else "contains"
    return "is_on_top_of" if a_bowl else "is_below_of"


def pair_features(
    a_name: str,
    b_name: str,
    a: BBox,
    b: BBox,
    width: int,
    height: int,
    *,
    support_eligible: bool,
) -> dict[str, float | str | bool]:
    ua, va = bbox_center(a, width, height)
    ub, vb = bbox_center(b, width, height)
    aw, ah = max(a[2] - a[0], 1e-6) / width, max(a[3] - a[1], 1e-6) / height
    bw, bh = max(b[2] - b[0], 1e-6) / width, max(b[3] - b[1], 1e-6) / height
    area_a, area_b = aw * ah, bw * bh
    return {
        "du": ua - ub,
        "dv": va - vb,
        "abs_du": abs(ua - ub),
        "abs_dv": abs(va - vb),
        "axis_margin": axis_margin(a, b, width, height),
        "a_width": aw,
        "a_height": ah,
        "b_width": bw,
        "b_height": bh,
        "log_area_ratio": log(max(area_a, 1e-9) / max(area_b, 1e-9)),
        "iou": bbox_iou(a, b),
        "iomin": bbox_iomin(a, b),
        "a_in_b": bbox_containment(a, b),
        "b_in_a": bbox_containment(b, a),
        "center_distance": hypot(ua - ub, va - vb),
        "a_role": object_role(a_name),
        "b_role": object_role(b_name),
        "support_eligible": bool(support_eligible),
    }


def visible_objects(frame: BBoxFrame, minimum_confidence: float = 0.0) -> dict[str, BBoxObject]:
    return {
        name: item
        for name, item in frame.objects.items()
        if item.visible and item.confidence >= minimum_confidence and bbox_area(item.bbox) > 0
    }


def unordered_visible_pairs(frame: BBoxFrame) -> Iterable[tuple[str, str]]:
    yield from combinations(sorted(visible_objects(frame)), 2)


def smooth_bbox_frames(frames: list[BBoxFrame], window: int = 5) -> list[BBoxFrame]:
    if window <= 1:
        return frames
    ordered = sorted(frames, key=lambda item: item.frame)
    radius = window // 2
    smoothed: list[BBoxFrame] = []
    for index, frame in enumerate(ordered):
        neighbors = ordered[max(0, index - radius):min(len(ordered), index + radius + 1)]
        objects: dict[str, BBoxObject] = {}
        for name, item in frame.objects.items():
            history = [neighbor.objects[name].bbox for neighbor in neighbors if name in neighbor.objects]
            box = tuple(median(values) for values in zip(*history))
            objects[name] = replace(item, bbox=box)
        smoothed.append(replace(frame, objects=objects))
    return smoothed
