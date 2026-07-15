from __future__ import annotations

import pytest

from demo.so101_proxy_demo.proxy.bbox_geometry import (
    bbox_iomin,
    deterministic_axis_relation,
    midpoint_arrow_evidence,
    rotate_bbox_180,
    smooth_bbox_frames,
)
from demo.so101_proxy_demo.proxy.schemas import BBoxFrame, BBoxObject
from demo.so101_proxy_demo.proxy.task_priors import canonical_object_name, task_prior


@pytest.mark.parametrize(
    ("a", "b", "relation"),
    [
        ((0, 40, 10, 50), (30, 40, 40, 50), "is_left_of"),
        ((30, 40, 40, 50), (0, 40, 10, 50), "is_right_of"),
        ((20, 80, 30, 90), (20, 10, 30, 20), "is_in_front_of"),
        ((20, 10, 30, 20), (20, 80, 30, 90), "is_behind"),
    ],
)
def test_deterministic_image_axis_relations(a, b, relation) -> None:
    assert deterministic_axis_relation(a, b, 100, 100) == relation


def test_iomin_matches_libero_support_rule() -> None:
    assert bbox_iomin((20, 20, 40, 40), (0, 0, 100, 100)) == pytest.approx(1.0)
    assert bbox_iomin((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(.25)


def test_midpoint_arrow_evidence_records_geometry() -> None:
    evidence = midpoint_arrow_evidence((0, 40, 10, 50), (30, 40, 40, 50), 100, 100)

    assert evidence["basis"] == "bbox_midpoint_arrow"
    assert evidence["subject_center"] == [5.0, 45.0]
    assert evidence["object_center"] == [35.0, 45.0]
    assert evidence["arrow_dx_norm"] == pytest.approx(0.3)
    assert evidence["arrow_dy_norm"] == pytest.approx(0.0)
    assert evidence["subject_relative_dx_norm"] == pytest.approx(-0.3)
    assert evidence["dominant_axis"] == "left_right"


def test_agentview_rotation() -> None:
    assert rotate_bbox_180((10, 20, 30, 40), 100, 80) == (70, 40, 90, 60)


def test_task_support_priors() -> None:
    drawer_cookie = task_prior(
        "move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie-at-the-le"
    )
    assert drawer_cookie.allowed_supports == frozenset({"red_drawer", "cookie"})
    left_stove = task_prior("place-the-black-bowl-from-the-left-to-the-top-of-the-stove")
    assert left_stove.source_support is None
    assert left_stove.allowed_supports == frozenset({"black_stove"})
    assert not left_stove.containment


def test_object_aliases_reject_distractor() -> None:
    assert canonical_object_name("akita_black_bowl_1") == "black_bowl"
    assert canonical_object_name("wooden_cabinet_1") == "red_drawer"
    assert canonical_object_name("black_cylindrical_container") is None


def test_offline_bbox_smoothing_is_centered_not_delayed() -> None:
    positions = [100, 100, 0, 0, 0]
    frames = [
        BBoxFrame(
            task="task",
            episode="episode_0",
            frame=index * 30,
            timestamp=index,
            camera="agent_view",
            width=640,
            height=480,
            objects={"black_bowl": BBoxObject((x, 10, x + 20, 30))},
        )
        for index, x in enumerate(positions)
    ]

    smoothed = smooth_bbox_frames(frames, window=5)

    assert smoothed[2].objects["black_bowl"].bbox == (0, 10, 20, 30)
