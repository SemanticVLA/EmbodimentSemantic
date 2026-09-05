"""Dependency-light canonical scene-graph relation extractor.

This module is shared by offline HDF5 conversion and live semantic context so
the graph prompt/arrow relation set cannot silently diverge between modalities.
It intentionally consumes only projected bboxes, world coordinates, and the
drawer-task flag; it does not inspect actions, rewards, or future frames.
"""

from __future__ import annotations

import numpy as np


CONTAINMENT_THRESH = 0.8


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
    for subject in subjects:
        if subject not in objects or subject not in world:
            continue
        pos_a = np.array(world[subject]["pos"])
        for obj in objects:
            if subject == obj or obj not in world:
                continue
            pos_b = np.array(world[obj]["pos"])
            is_stacked = False
            if subject in bboxes and obj in bboxes:
                tx1, ty1, tx2, ty2 = bboxes[subject]
                bx1, by1, bx2, by2 = bboxes[obj]
                ix1, iy1 = max(tx1, bx1), max(ty1, by1)
                ix2, iy2 = min(tx2, bx2), min(ty2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    a_area = (tx2 - tx1) * (ty2 - ty1)
                    b_area = (bx2 - bx1) * (by2 - by1)
                    io_min = inter / min(a_area, b_area)
                    if io_min > CONTAINMENT_THRESH and ("bowl" in subject or "bowl" in obj):
                        is_stacked = True
                        pair = {subject, obj} == {"akita_black_bowl_1", "wooden_cabinet_1"}
                        if is_drawer_task and pair:
                            triplets.append((subject, "is_inside" if pos_a[2] >= pos_b[2] else "contains", obj))
                        else:
                            triplets.append((subject, "is_on_top_of" if pos_a[2] >= pos_b[2] else "is_below_of", obj))
            if is_stacked:
                continue
            dx = pos_a[0] - pos_b[0]
            dy = pos_a[1] - pos_b[1]
            if abs(dx) >= abs(dy):
                triplets.append((subject, "is_in_front_of" if dx > 0 else "is_behind", obj))
            else:
                triplets.append((subject, "is_left_of" if dy > 0 else "is_right_of", obj))
    return triplets
