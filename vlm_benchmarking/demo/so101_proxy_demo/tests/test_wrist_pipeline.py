from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from demo.so101_proxy_demo.proxy.bbox_geometry import midpoint_arrow_evidence
from demo.so101_proxy_demo.proxy.fusion import filter_wrist_frame
from demo.so101_proxy_demo.proxy.schemas import (
    BBoxFrame,
    BBoxObject,
    MetadataFrame,
    ProxyFrame,
    RelationRecord,
    SampledFrameRecord,
)
from demo.so101_proxy_demo.proxy.validation import validate_wrist_contract
from demo.so101_proxy_demo.vision.grounded_sam2 import GroundedSam2Detector
from demo.so101_proxy_demo.vision.wrist_bbox_rules import (
    apply_held_bowl_tracking,
    flow_window_is_stable,
    materialize_direct_wrist_frame,
    select_flow_recovery,
)


def _sample(frame: int) -> SampledFrameRecord:
    return SampledFrameRecord(
        task="task",
        episode="episode_0",
        episode_index=0,
        frame=frame,
        timestamp=frame / 30,
        camera="wrist",
        width=640,
        height=480,
        image_path=f"task/frames/wrist/episode_0/frame_{frame:06d}.jpg",
    )


def _signal(
    frame: int,
    *,
    held: bool = False,
    released: bool = False,
    angular_speed: float = 0.0,
) -> MetadataFrame:
    return MetadataFrame(
        task="task",
        episode="episode_0",
        frame=frame,
        timestamp=frame / 30,
        phase="released" if released else ("held" if held else "home"),
        gripper_open=not held,
        held=held,
        lifted=held,
        released=released,
        metadata_reliable=True,
        state_xyz=(0, 0, 0),
        state_rotation=(0, 0, 0),
        state_gripper=0,
        action_gripper=0,
        ee_speed=0,
        angular_speed=angular_speed,
    )


def test_wrist_area_policy_accepts_large_cropped_bowl_but_rejects_full_frame() -> None:
    detector = object.__new__(GroundedSam2Detector)
    detector.maximum_box_area_fraction = 0.65
    detector.maximum_bowl_area_fraction = 0.15
    detector.wrist_minimum_box_area_fraction = 0.001
    detector.wrist_maximum_box_area_fraction = 0.90
    detector.wrist_maximum_bowl_area_fraction = 0.80
    detector.wrist_minimum_bowl_aspect_ratio = 0.70
    detector.wrist_maximum_plate_aspect_ratio = 2.00
    image = Image.new("RGB", (640, 480))
    path = Path("task/frames/wrist/episode_0/frame_000000.jpg")

    assert detector._candidate_area_allowed(
        "black_bowl", BBoxObject((200, 0, 640, 480), 0.8, 0.8), image, path
    )
    assert not detector._candidate_area_allowed(
        "white_plate", BBoxObject((0, 0, 640, 480), 0.8, 0.8), image, path
    )
    assert not detector._candidate_area_allowed(
        "cookie", BBoxObject((0, 0, 5, 5), 0.8, 0.8), image, path
    )
    assert not detector._candidate_area_allowed(
        "black_bowl", BBoxObject((530, 0, 640, 220), 0.8, 0.8), image, path
    )
    assert not detector._candidate_area_allowed(
        "white_plate", BBoxObject((10, 400, 400, 470), 0.8, 0.8), image, path
    )


def test_wrist_cookie_requires_visible_orange_package_evidence() -> None:
    detector = object.__new__(GroundedSam2Detector)
    detector.wrist_minimum_cookie_orange_fraction = 0.008
    path = Path("task/frames/wrist/episode_0/frame_000000.jpg")
    candidate = BBoxObject((0, 0, 100, 100), 0.8, 0.8)
    package = Image.new("RGB", (100, 100), (220, 220, 215))
    package.paste((230, 120, 35), (0, 0, 10, 100))
    distractor = Image.new("RGB", (100, 100), (20, 25, 25))

    assert detector._candidate_appearance_allowed("cookie", candidate, package, path)
    assert not detector._candidate_appearance_allowed(
        "cookie", candidate, distractor, path
    )


def test_direct_wrist_materialization_keeps_only_high_confidence_objects() -> None:
    sample = _sample(0)
    candidates = BBoxFrame(
        task=sample.task,
        episode=sample.episode,
        frame=sample.frame,
        timestamp=sample.timestamp,
        camera="wrist",
        width=640,
        height=480,
        objects={
            "black_bowl": BBoxObject((1, 2, 30, 40), 0.72, 0.72),
            "cookie": BBoxObject((50, 60, 80, 90), 0.31, 0.31),
        },
    )
    result = materialize_direct_wrist_frame(
        sample, candidates, direct_threshold=0.50, detector="wrist-v1"
    )
    assert set(result.objects) == {"black_bowl"}
    assert result.objects["black_bowl"].source == "wrist_dino_direct"


def test_flow_recovery_requires_candidate_agreement() -> None:
    tracked = BBoxObject((10, 10, 40, 40), 0.7, 0.6)
    close = BBoxObject((12, 9, 42, 39), 0.3, 0.3)
    far = BBoxObject((100, 100, 140, 140), 0.4, 0.4)

    confirmed = select_flow_recovery(close, tracked, minimum_iou=0.25)
    assert confirmed is not None
    assert confirmed.source == "wrist_dino_temporal_confirmed"
    assert select_flow_recovery(far, tracked, minimum_iou=0.25) is None
    assert select_flow_recovery(None, tracked, minimum_iou=0.25).source == "wrist_optical_flow_fill"


def test_rapid_wrist_motion_disables_flow_recovery() -> None:
    assert flow_window_is_stable([_signal(0, angular_speed=0.02)], maximum_angular_speed=0.08)
    assert not flow_window_is_stable(
        [_signal(0, angular_speed=0.02), _signal(1, angular_speed=0.12)],
        maximum_angular_speed=0.08,
    )


def test_held_bowl_tracking_fills_between_visual_anchors_but_not_release() -> None:
    samples = [_sample(frame) for frame in (0, 30, 60, 90)]
    anchor_a = BBoxObject((260, 70, 640, 480), 0.8, 0.8, source="wrist_dino_direct")
    anchor_b = BBoxObject((270, 80, 640, 480), 0.75, 0.75, source="wrist_dino_direct")
    records = [
        BBoxFrame(
            task=sample.task,
            episode=sample.episode,
            frame=sample.frame,
            timestamp=sample.timestamp,
            camera="wrist",
            width=sample.width,
            height=sample.height,
            objects=(
                {"black_bowl": anchor_a}
                if sample.frame == 0
                else ({"black_bowl": anchor_b} if sample.frame == 60 else {})
            ),
        )
        for sample in samples
    ]
    metadata = {
        ("task", "episode_0", frame): _signal(
            frame, held=frame < 90, released=frame == 90
        )
        for frame in (0, 30, 60, 90)
    }

    output, fills = apply_held_bowl_tracking(
        records, {}, metadata, minimum_anchor_iou=0.50
    )

    assert fills == 1
    assert output[1].objects["black_bowl"].source == "wrist_held_bowl_track"
    assert "black_bowl" not in output[3].objects


def test_wrist_graph_is_exact_agent_subset_with_display_only_geometry() -> None:
    agent_boxes = {
        "black_bowl": BBoxObject((10, 10, 30, 30), 1, 0.9),
        "black_stove": BBoxObject((40, 40, 90, 90), 1, 0.8),
        "cookie": BBoxObject((5, 70, 20, 90), 1, 0.7),
    }
    relation = RelationRecord(
        "black_bowl",
        "is_left_of",
        "black_stove",
        "bbox_axis",
        0.85,
        evidence=midpoint_arrow_evidence(
            agent_boxes["black_bowl"].bbox,
            agent_boxes["black_stove"].bbox,
            100,
            100,
        ),
    )
    inverse = RelationRecord(
        "black_stove",
        "is_right_of",
        "black_bowl",
        "bbox_axis",
        0.85,
        evidence=midpoint_arrow_evidence(
            agent_boxes["black_stove"].bbox,
            agent_boxes["black_bowl"].bbox,
            100,
            100,
        ),
    )
    agent = ProxyFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        mode="gt",
        visible_objects=tuple(agent_boxes),
        bboxes=agent_boxes,
        relations=(relation, inverse),
        gripper_phase="held",
        metadata_reliable=True,
        model_version="none",
    )
    wrist = BBoxFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="wrist",
        width=100,
        height=100,
        objects={
            "black_bowl": BBoxObject((70, 50, 100, 100), 0.8, 0.6),
            "black_stove": BBoxObject((0, 0, 80, 70), 0.9, 0.7),
        },
    )

    result = filter_wrist_frame(agent, wrist)

    assert {item.triplet() for item in result.relations} == {
        relation.triplet(),
        inverse.triplet(),
    }
    assert all(item.confidence == 0.6 for item in result.relations)
    assert result.relations[0].evidence["semantic_camera"] == "agent_view"
    assert result.relations[0].evidence["visibility_camera"] == "wrist"
    assert result.relations[0].evidence["display_basis"] == "wrist_bbox_midpoint_arrow"
    assert validate_wrist_contract(result, agent) == []
    bad = replace(
        result,
        relations=(replace(result.relations[0], source="wrist_axis"), result.relations[1]),
    )
    assert any("independently inferred" in error for error in validate_wrist_contract(bad, agent))
