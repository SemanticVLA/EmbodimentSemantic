from __future__ import annotations

from demo.so101_proxy_demo.proxy.schemas import BBoxObject
from demo.so101_proxy_demo.vision.agent_bbox_rules import (
    _needs_flow_candidate,
    select_candidate_path,
    select_endpoint_candidate,
)


def _box(values: tuple[float, float, float, float], confidence: float = 0.2) -> BBoxObject:
    return BBoxObject(values, confidence, confidence, True, "candidate")


def test_drawer_to_stove_endpoints_reject_large_dark_distractors() -> None:
    task = "place-the-black-bowl-from-on-top-of-the-drawer-to-the-stove"
    drawer = _box((316.9, 71.9, 432.2, 229.7), 1.0)
    stove = _box((0.9, 71.8, 225.6, 213.4), 1.0)
    source_candidates = [
        _box((305.3, 256.2, 416.1, 391.4), 0.77),
        _box((225.4, 150.6, 286.0, 207.0), 0.13),
        _box((320.1, 73.1, 400.7, 133.9), 0.11),
    ]
    target_candidates = [
        _box((305.4, 256.1, 416.0, 391.4), 0.58),
        _box((109.7, 128.6, 176.4, 184.0), 0.46),
        _box((224.6, 151.7, 285.8, 207.1), 0.12),
    ]

    source, _ = select_endpoint_candidate(
        source_candidates,
        support=drawer,
        role="source",
        task=task,
        width=640,
        height=480,
        ideal_area_fraction=0.018,
    )
    target, _ = select_endpoint_candidate(
        target_candidates,
        support=stove,
        role="target",
        task=task,
        width=640,
        height=480,
        ideal_area_fraction=0.018,
    )

    assert source is not None and source.bbox == (320.1, 73.1, 400.7, 133.9)
    assert target is not None and target.bbox == (109.7, 128.6, 176.4, 184.0)


def test_left_source_uses_side_relation_instead_of_support_overlap() -> None:
    task = "place-the-black-bowl-from-the-left-to-the-top-of-the-stove"
    stove = _box((200, 100, 500, 400), 1.0)
    beside = _box((100, 180, 170, 240), 0.10)
    overlapping = _box((260, 180, 330, 240), 0.90)

    selected, _ = select_endpoint_candidate(
        [overlapping, beside],
        support=stove,
        role="source",
        task=task,
        width=640,
        height=480,
        ideal_area_fraction=0.014,
    )

    assert selected is not None and selected.bbox == beside.bbox


def test_temporal_path_reproduces_verified_low_threshold_bowl_track() -> None:
    source = _box((320.1, 73.1, 400.7, 133.9), 0.11)
    target = _box((109.7, 128.6, 176.4, 184.0), 0.46)
    expected = {
        210: _box((289.2, 39.2, 367.7, 120.4), 0.21),
        240: _box((89.8, 103.9, 175.8, 172.6), 0.15),
        270: _box((109.5, 113.4, 160.8, 183.3), 0.16),
        300: _box((112.9, 115.9, 176.8, 183.4), 0.19),
    }
    candidates = {
        210: [expected[210], _box((305.0, 255.9, 416.2, 391.5), 0.63)],
        240: [expected[240], _box((222.3, 149.4, 285.9, 206.9), 0.18)],
        270: [
            expected[270],
            _box((101.9, 92.3, 161.1, 185.2), 0.09),
            _box((305.0, 256.0, 416.2, 391.4), 0.71),
        ],
        300: [expected[300], _box((224.1, 151.3, 286.1, 206.9), 0.09)],
    }
    changes = {
        (frame, item.bbox): (0.8 if item is expected[frame] else 0.05)
        for frame, values in candidates.items()
        for item in values
    }

    selected = select_candidate_path(
        list(expected),
        candidates,
        source_frame=180,
        source=source,
        target_frame=330,
        target=target,
        width=640,
        height=480,
        ideal_area_fraction=0.018,
        change_scores=changes,
    )

    assert {frame: item.bbox for frame, item in selected.items()} == {
        frame: item.bbox for frame, item in expected.items()
    }


def test_flow_is_requested_only_for_missing_or_boundary_clipped_boxes() -> None:
    assert _needs_flow_candidate(_box((0.2, 20, 50, 80)), 640, 480)
    assert _needs_flow_candidate(
        BBoxObject((20, 20, 50, 80), source="agent_bowl_linear_fallback"),
        640,
        480,
    )
    assert not _needs_flow_candidate(_box((20, 20, 50, 80)), 640, 480)
