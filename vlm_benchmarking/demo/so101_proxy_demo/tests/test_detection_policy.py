from __future__ import annotations

from pathlib import Path
import inspect

from demo.so101_proxy_demo.proxy.schemas import (
    BBoxFrame,
    BBoxObject,
    MetadataFrame,
    SampledFrameRecord,
)
from demo.so101_proxy_demo.vision.detection_cache import DetectionCache
from demo.so101_proxy_demo.vision.optical_flow import _fuse_flow_boxes
from demo.so101_proxy_demo.vision.grounded_sam2 import (
    _compose_agent_record,
    _interpolate_bowl,
    plan_agent_episode,
    select_bowl_model_frames,
)
from demo.so101_proxy_demo.cli import build_parser
from demo.so101_proxy_demo.vision.agent_bbox_rules import generate_agent_bboxes


def _sample(frame: int, camera: str = "agent_view") -> SampledFrameRecord:
    return SampledFrameRecord(
        task="task",
        episode="episode_0",
        episode_index=0,
        frame=frame,
        timestamp=frame / 30,
        camera=camera,
        width=640,
        height=480,
        image_path=f"frame_{frame:06d}.jpg",
    )


def _signal(
    frame: int,
    *,
    held: bool = False,
    lifted: bool = False,
    released: bool = False,
    reliable: bool = True,
) -> MetadataFrame:
    return MetadataFrame(
        task="task",
        episode="episode_0",
        frame=frame,
        timestamp=frame / 30,
        phase="released" if released else ("held" if held else "home"),
        gripper_open=not held,
        held=held,
        lifted=lifted,
        released=released,
        metadata_reliable=reliable,
        state_xyz=(0, 0, 0),
        state_rotation=(0, 0, 0),
        state_gripper=0,
        action_gripper=0,
        ee_speed=0,
        angular_speed=0,
    )


def test_agent_plan_detects_bowl_only_during_motion_and_release() -> None:
    samples = [_sample(frame) for frame in (0, 30, 60, 90, 120, 150)]
    signals = [
        _signal(frame, held=50 <= frame <= 100, lifted=70 <= frame <= 90, released=frame == 110)
        for frame in range(151)
    ]
    plan = plan_agent_episode(samples, signals)
    assert plan.reference_frame == 0
    assert plan.movement_start == 50
    assert plan.movement_end == 100
    assert plan.release_frame == 120
    assert plan.bowl_detection_frames == frozenset({60, 90, 120})
    assert select_bowl_model_frames(plan, 2) == frozenset({60, 90, 120})


def test_agent_plan_falls_back_to_every_sample_when_metadata_is_unreliable() -> None:
    samples = [_sample(frame) for frame in (0, 30, 60)]
    plan = plan_agent_episode(samples, [_signal(0, reliable=False)])
    assert not plan.metadata_reliable
    assert plan.bowl_detection_frames == frozenset({30, 60})
    assert select_bowl_model_frames(plan, 2) == frozenset({30, 60})


def test_reliable_bowl_model_frames_are_strided_but_keep_last_and_release() -> None:
    plan = plan_agent_episode(
        [_sample(frame) for frame in (0, 30, 60, 90, 120, 150, 180)],
        [
            _signal(frame, held=20 <= frame <= 140, released=frame == 160)
            for frame in range(181)
        ],
    )
    assert plan.bowl_detection_frames == frozenset({30, 60, 90, 120, 180})
    assert select_bowl_model_frames(plan, 2) == frozenset({30, 90, 120, 180})


def test_agent_composition_propagates_only_static_objects() -> None:
    reference = BBoxFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        width=640,
        height=480,
        objects={
            "black_bowl": BBoxObject((1, 2, 3, 4)),
            "black_stove": BBoxObject((10, 20, 30, 40)),
            "cookie": BBoxObject((50, 60, 70, 80)),
        },
        detector="legacy",
    )
    dynamic_bowl = BBoxObject((100, 110, 130, 140), .8, .7)
    record = _compose_agent_record(
        _sample(60), reference, dynamic_bowl, "agent_bowl_dynamic_detection", "v4"
    )
    assert record.objects["black_stove"].bbox == (10, 20, 30, 40)
    assert record.objects["black_stove"].source == "agent_static_reference"
    assert record.objects["black_bowl"].bbox == dynamic_bowl.bbox
    assert record.objects["black_bowl"].source == "agent_bowl_dynamic_detection"


def test_detection_cache_is_versioned_and_resumable(tmp_path: Path) -> None:
    frame = BBoxFrame(
        task="task",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera="agent_view",
        width=640,
        height=480,
        objects={"black_bowl": BBoxObject((1, 2, 3, 4))},
        detector="detector-v4",
    )
    path = tmp_path / "cache" / "detection.sqlite3"
    with DetectionCache(path) as cache:
        assert cache.upsert([frame]) == 1
    with DetectionCache(path) as cache:
        assert cache.load("detector-v4")[frame.key()] == frame
        assert cache.detector_counts() == {"detector-v4": 1}
        assert cache.delete_camera("detector-v4", "agent_view") == 1
        assert cache.load("detector-v4") == {}


def test_detection_cli_defaults_to_agent_view() -> None:
    args = build_parser().parse_args(["detect", "--sampled"])
    assert args.camera == "agent_view"


def test_agent_rule_generator_has_no_legacy_bbox_input() -> None:
    assert "bbox_path" not in inspect.signature(generate_agent_bboxes).parameters


def test_detection_cli_supports_resumable_episode_filters() -> None:
    args = build_parser().parse_args(
        ["detect", "--sampled", "--task", "task", "--episode", "episode_4"]
    )
    assert args.task == "task"
    assert args.episode == "episode_4"


def test_missing_dynamic_bowl_interpolates_between_detected_anchors() -> None:
    anchors = {
        240: (BBoxObject((200, 0, 300, 50), .8, .8), "detected"),
        300: (BBoxObject((80, 80, 160, 160), .9, .9), "detected"),
    }
    bowl, source = _interpolate_bowl(270, anchors)
    assert bowl is not None
    assert bowl.bbox == (140.0, 40.0, 230.0, 105.0)
    assert bowl.tracking_confidence == 0.52
    assert source == "agent_bowl_dynamic_interpolation"


def test_bidirectional_flow_fuses_consistent_boxes_and_rejects_disagreement() -> None:
    import numpy as np

    forward = np.asarray((10, 10, 30, 30), dtype=float)
    backward = np.asarray((12, 8, 32, 28), dtype=float)
    fused = _fuse_flow_boxes(forward, backward, minimum_iou=.15)
    assert fused is not None
    assert tuple(fused) == (11.0, 9.0, 31.0, 29.0)
    far = np.asarray((80, 80, 95, 95), dtype=float)
    assert _fuse_flow_boxes(forward, far, minimum_iou=.15) is None
