from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from demo.so101_proxy_demo.proxy.bbox_geometry import INVERSE_RELATIONS
from demo.so101_proxy_demo.proxy.fusion import filter_wrist_frame, fuse_agent_sequence
from demo.so101_proxy_demo.proxy.schemas import BBoxFrame, BBoxObject, MetadataFrame
from demo.so101_proxy_demo.vision.grounded_sam2 import GroundedSam2Detector


def _config() -> dict:
    return {
        "geometry": {
            "center_median_window": 1,
            "relation_persistence_frames": 1,
            "support_iomin_threshold": .8,
            "support_soft_iomin_threshold": .7,
            "support_center_distance_threshold": .08,
        }
    }


def _metadata(held: bool) -> MetadataFrame:
    return MetadataFrame(
        task="place-the-black-bowl-from-the-left-to-the-top-of-the-stove",
        episode="episode_0",
        frame=0,
        timestamp=0,
        phase="held" if held else "released",
        gripper_open=not held,
        held=held,
        lifted=held,
        released=not held,
        metadata_reliable=True,
        state_xyz=(0, 0, .05 if held else .01),
        state_rotation=(0, 0, 0),
        state_gripper=2 if held else 20,
        action_gripper=2 if held else 20,
        ee_speed=0,
        angular_speed=0,
        gates=("held", "lifted") if held else ("released",),
    )


def _frame(camera: str = "agent_view") -> BBoxFrame:
    return BBoxFrame(
        task="place-the-black-bowl-from-the-left-to-the-top-of-the-stove",
        episode="episode_0",
        frame=0,
        timestamp=0,
        camera=camera,
        width=100,
        height=100,
        objects={
            "black_bowl": BBoxObject((20, 20, 40, 40)),
            "black_stove": BBoxObject((0, 0, 80, 80)),
            "cookie": BBoxObject((82, 20, 92, 30)),
        },
    )


def test_metadata_suppresses_projected_support() -> None:
    frame = _frame()
    held = fuse_agent_sequence(
        [frame], {(frame.task, frame.episode, frame.frame): _metadata(True)}, "gt", _config(), frame.task
    )[0]
    assert not any(item.relation == "is_on_top_of" for item in held.relations)

    released = fuse_agent_sequence(
        [frame], {(frame.task, frame.episode, frame.frame): _metadata(False)}, "gt", _config(), frame.task
    )[0]
    assert ("black_bowl", "is_on_top_of", "black_stove") in {item.triplet() for item in released.relations}
    support = next(item for item in released.relations if item.triplet() == ("black_bowl", "is_on_top_of", "black_stove"))
    assert support.evidence["basis"] == "bbox_midpoint_arrow"
    assert support.evidence["iomin"] == 1.0


def test_centered_allowed_support_does_not_fall_back_to_behind() -> None:
    frame = BBoxFrame(
        task="move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie-at-the-le",
        episode="episode_0",
        frame=570,
        timestamp=19,
        camera="agent_view",
        width=640,
        height=480,
        objects={
            "black_bowl": BBoxObject((79.425048828125, 89.3299560546875, 148.434814453125, 140.24388122558594)),
            "cookie": BBoxObject((80.05690002441406, 101.26687622070312, 156.17623901367188, 156.7161865234375)),
        },
    )

    proxy = fuse_agent_sequence(
        [frame], {(frame.task, frame.episode, frame.frame): _metadata(False)}, "gt", _config(), frame.task
    )[0]

    assert ("black_bowl", "is_on_top_of", "cookie") in {item.triplet() for item in proxy.relations}


def test_released_support_updates_without_persistence_delay() -> None:
    config = _config()
    config["geometry"]["relation_persistence_frames"] = 3
    task = "move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie-at-the-le"
    frames = [
        BBoxFrame(
            task=task,
            episode="episode_0",
            frame=0,
            timestamp=0,
            camera="agent_view",
            width=100,
            height=100,
            objects={
                "black_bowl": BBoxObject((10, 10, 30, 30)),
                "cookie": BBoxObject((70, 70, 90, 90)),
            },
        ),
        BBoxFrame(
            task=task,
            episode="episode_0",
            frame=30,
            timestamp=1,
            camera="agent_view",
            width=100,
            height=100,
            objects={
                "black_bowl": BBoxObject((40, 40, 60, 60)),
                "cookie": BBoxObject((40, 40, 60, 60)),
            },
        ),
    ]
    metadata = {
        (task, "episode_0", 0): _metadata(True),
        (task, "episode_0", 30): _metadata(False),
    }

    released = fuse_agent_sequence(frames, metadata, "gt", config, task)[1]

    assert ("black_bowl", "is_on_top_of", "cookie") in {item.triplet() for item in released.relations}


def test_inverse_and_wrist_visibility_filtering() -> None:
    frame = _frame()
    agent = fuse_agent_sequence([frame], {}, "gt", _config(), frame.task)[0]
    triplets = {item.triplet() for item in agent.relations}
    for subject, relation, obj in triplets:
        assert (obj, INVERSE_RELATIONS[relation], subject) in triplets

    wrist = replace(
        frame,
        camera="wrist",
        objects={
            "black_bowl": frame.objects["black_bowl"],
            "black_stove": frame.objects["black_stove"],
        },
    )
    wrist_proxy = filter_wrist_frame(agent, wrist)
    assert wrist_proxy.visible_objects == ("black_bowl", "black_stove")
    assert all("cookie" not in item.triplet() for item in wrist_proxy.relations)


def test_bowl_selection_rejects_black_cylindrical_distractor() -> None:
    cylinder = BBoxObject((0, 30, 65, 133), .43, .43)
    shallow_bowl = BBoxObject((212, 0, 299, 50), .42, .42)
    selected = {
        "red_drawer": BBoxObject((209, 1, 341, 164), .89, .89),
        "black_stove": BBoxObject((209, 178, 637, 478), .84, .84),
    }
    result = GroundedSam2Detector._select_bowl(
        [cylinder, shallow_bowl],
        selected,
        "move-the-black-bowl-from-the-top-of-the-drawer-to-on-top-of-the-cookie-at-the-le",
    )
    assert result == shallow_bowl


def test_detector_canonicalizes_dish_as_white_plate() -> None:
    from demo.so101_proxy_demo.vision.grounded_sam2 import _canonical_label

    assert _canonical_label("white circular dish") == "white_plate"


def test_wrist_filter_prefers_missing_over_low_confidence_hallucinations() -> None:
    detector = object.__new__(GroundedSam2Detector)
    detector.box_threshold = .30
    detector.wrist_box_threshold = .50
    detector.maximum_box_area_fraction = .65
    objects = {
        "red_drawer": BBoxObject((440, 0, 638, 35), .80, .80),
        "black_stove": BBoxObject((125, 314, 444, 479), .38, .38),
        "cookie": BBoxObject((124, 315, 444, 479), .32, .32),
        "white_plate": BBoxObject((3, 2, 636, 478), .36, .36),
    }
    image_path = Path("task/frames/wrist/episode_0/frame_000000.jpg")
    from PIL import Image

    filtered = detector._filter_objects(objects, Image.new("RGB", (640, 480)), image_path)
    assert filtered == {"red_drawer": objects["red_drawer"]}


def test_bowl_filter_rejects_stove_sized_false_detection() -> None:
    detector = object.__new__(GroundedSam2Detector)
    detector.box_threshold = .30
    detector.wrist_box_threshold = .50
    detector.maximum_box_area_fraction = .65
    detector.maximum_bowl_area_fraction = .15
    giant_bowl = BBoxObject((209, 179, 639, 480), .80, .80)
    image_path = Path("task/frames/agent_view/episode_0/frame_000270.jpg")
    from PIL import Image

    filtered = detector._filter_objects(
        {"black_bowl": giant_bowl}, Image.new("RGB", (640, 480)), image_path
    )
    assert filtered == {}


def test_candidate_area_filter_keeps_valid_runner_up_geometry() -> None:
    detector = object.__new__(GroundedSam2Detector)
    detector.maximum_box_area_fraction = .65
    detector.maximum_bowl_area_fraction = .15
    from PIL import Image

    image = Image.new("RGB", (640, 480))
    assert not detector._candidate_area_allowed(
        "white_plate", BBoxObject((2, 108, 636, 477), .58, .58), image
    )
    assert detector._candidate_area_allowed(
        "white_plate", BBoxObject((259, 383, 375, 478), .22, .22), image
    )
