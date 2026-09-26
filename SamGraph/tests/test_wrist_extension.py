import numpy as np

from samgraph_core.automatic_scene import ObjectMemory
from samgraph_wrist.identity import CrossViewBowlResolver, filter_agent_triplets
from samgraph_wrist.export import VLM_WRIST_CAMERA
from samgraph_wrist.wrist_scene import WristSceneController


def _mask(x0, x1):
    value = np.zeros((32, 32), dtype=bool)
    value[8:24, x0:x1] = True
    return value


def _states(prefix, *, wrist):
    status_key = "observation_state" if wrist else "status"
    status_value = "current_observation" if wrist else "observed"
    return [
        {"track_id": f"{prefix}1", "class_id": "black_bowl",
         "output_id": f"{prefix}1", status_key: status_value,
         "identity_method": "provisional", "semantic_id": None},
        {"track_id": f"{prefix}2", "class_id": "black_bowl",
         "output_id": f"{prefix}2", status_key: status_value,
         "identity_method": "provisional", "semantic_id": None},
    ]


def test_libero_wrist_export_uses_existing_vlm_camera_name():
    assert VLM_WRIST_CAMERA == "eye_in_hand"


def test_filter_is_an_exact_ordered_subset():
    source = [
        ["a", "is_left_of", "b"],
        ["b", "is_right_of", "a"],
        ["a", "is_in_front_of", "c"],
    ]
    result = filter_agent_triplets(source, {"a", "b"})
    assert result == source[:2]
    assert result[0] is not source[0]


def test_cross_view_bowls_use_rgb_appearance_not_pixel_order():
    left, right = _mask(2, 10), _mask(22, 30)
    agent_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    agent_rgb[left] = (240, 20, 20)
    agent_rgb[right] = (20, 20, 240)
    # Wrist locations are reversed, while appearance follows physical identity.
    wrist_rgb = np.zeros_like(agent_rgb)
    wrist_rgb[left] = (20, 20, 240)
    wrist_rgb[right] = (240, 20, 20)
    resolver = CrossViewBowlResolver(appearance_margin=0.01)
    evidence = resolver.update(
        frame=0,
        wrist_rgb=wrist_rgb,
        wrist_masks={"w1": left, "w2": right},
        wrist_states=_states("w", wrist=True),
        agent_rgb=agent_rgb,
        agent_masks={"a1": left, "a2": right},
        agent_states=_states("a", wrist=False),
    )
    assert evidence["decision"] == "assigned"
    assert resolver.mapping == {"w1": "a2", "w2": "a1"}
    assert evidence["assignment_method"] == "cross_view_rgb_appearance"


def test_identical_bowls_remain_explicitly_unresolved():
    left, right = _mask(2, 10), _mask(22, 30)
    rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
    resolver = CrossViewBowlResolver(appearance_margin=0.01)
    evidence = resolver.update(
        frame=0,
        wrist_rgb=rgb,
        wrist_masks={"w1": left, "w2": right},
        wrist_states=_states("w", wrist=True),
        agent_rgb=rgb,
        agent_masks={"a1": left, "a2": right},
        agent_states=_states("a", wrist=False),
    )
    assert evidence["decision"] == "unresolved_correspondence"
    assert resolver.mapping == {}


def test_wrist_lost_track_distinguishes_exit_from_occlusion():
    controller = object.__new__(WristSceneController)
    controller._last_rgb_shape = (32, 32)
    controller._centers = {"bowl": [(0, np.array([24.0, 16.0])),
                                     (1, np.array([31.0, 16.0]))]}
    border = np.zeros((32, 32), dtype=bool)
    border[12:20, 29:32] = True
    item = ObjectMemory("bowl", "black_bowl", "bowl", None,
                        last_mask=border, last_observed_frame=1)
    state, evidence = controller._lost_state(item, 2)
    assert state == "out_of_view"
    assert evidence["last_mask_touched_border"] is True

    controller._centers = {"bowl": [(0, np.array([16.0, 16.0])),
                                     (1, np.array([16.0, 16.0]))]}
    middle = np.zeros((32, 32), dtype=bool)
    middle[12:20, 12:20] = True
    item.last_mask = middle
    state, evidence = controller._lost_state(item, 2)
    assert state == "occluded_remembered"
    assert evidence["reason"] == "lost_inside_image_without_exit_evidence"


def test_wrist_quality_accepts_scale_change_against_previous_frame():
    controller = object.__new__(WristSceneController)
    controller._last_frame = 8
    controller.objects = {}
    previous = np.zeros((64, 64), dtype=bool)
    previous[20:28, 20:28] = True
    current = np.zeros((64, 64), dtype=bool)
    current[12:36, 12:36] = True
    rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
    item = ObjectMemory(
        "plate", "plate", "support", "plate",
        last_mask=previous, last_observed_frame=7,
        descriptor=np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32) / 3, 3),
    )
    controller.objects[item.track_id] = item
    assert controller._quality(item, current, rgb, {}) is None


def test_wrist_quality_still_rejects_extreme_temporal_area_jump():
    controller = object.__new__(WristSceneController)
    controller._last_frame = 8
    controller.objects = {}
    previous = np.zeros((64, 64), dtype=bool)
    previous[20:28, 20:28] = True
    current = np.ones((64, 64), dtype=bool)
    rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
    item = ObjectMemory(
        "plate", "plate", "support", "plate",
        last_mask=previous, last_observed_frame=7,
    )
    controller.objects[item.track_id] = item
    assert controller._quality(item, current, rgb, {}) == "wrist_temporal_area_jump"
