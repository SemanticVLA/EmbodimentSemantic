from __future__ import annotations

import numpy as np

from hdf5_to_lerobot_dataset import (
    FEATURES,
    build_frame,
    filter_by_subject,
    flip180,
)


def _frame_kwargs(**overrides):
    base = dict(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        eye_in_hand_rgb=np.full((128, 128, 3), 7, dtype=np.uint8),
        bboxes={"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [80, 80, 100, 100]},
        relations=[
            ["akita_black_bowl_1", "is_left_of", "plate_1"],
            ["plate_1", "is_right_of", "akita_black_bowl_1"],
        ],
        ee_pos=np.array([0.1, 0.2, 0.3]),
        ee_ori=np.array([1.0, -1.0, 0.5]),
        gripper_states=np.array([0.04, -0.04]),
        action=np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, -1.0]),
        task_text="pick up the black bowl and place it on the plate",
    )
    base.update(overrides)
    return base


def test_flip180_rotates_both_axes():
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    flipped = flip180(frame)
    assert flipped.shape == frame.shape
    assert np.array_equal(flipped, frame[::-1, ::-1])
    assert np.array_equal(flipped[-1, -1], frame[0, 0])


def test_filter_by_subject_keeps_only_matching_subject():
    relations = [
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("plate_1", "is_right_of", "akita_black_bowl_1"),
        ("cookies_1", "is_on_top_of", "wooden_cabinet_1"),
    ]
    filtered = filter_by_subject(relations, "akita_black_bowl_1")
    assert filtered == [("akita_black_bowl_1", "is_left_of", "plate_1")]


def test_filter_by_subject_none_keeps_everything():
    relations = [("a", "r", "b"), ("b", "r", "a")]
    assert filter_by_subject(relations, None) == [("a", "r", "b"), ("b", "r", "a")]


def test_build_frame_state_vector_is_ee_pos_ori_gripper_concat():
    frame = build_frame(**_frame_kwargs(), variant="control")
    expected = np.array([0.1, 0.2, 0.3, 1.0, -1.0, 0.5, 0.04, -0.04], dtype=np.float32)
    assert frame["observation.state"].dtype == np.float32
    assert np.allclose(frame["observation.state"], expected)
    assert frame["observation.state"].shape == FEATURES["observation.state"]["shape"]


def test_build_frame_action_dtype_and_shape():
    frame = build_frame(**_frame_kwargs(), variant="control")
    assert frame["action"].dtype == np.float32
    assert frame["action"].shape == FEATURES["action"]["shape"]


def test_build_frame_control_never_draws_arrows():
    kwargs = _frame_kwargs(agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8))
    frame = build_frame(**kwargs, variant="control")
    # Control must be flip180(raw) and nothing else -- no arrow pixels introduced.
    assert np.array_equal(frame["observation.images.image"], flip180(kwargs["agentview_rgb"]))


def test_build_frame_treatment_draws_arrows_only_for_target_subject():
    kwargs = _frame_kwargs(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        relations=[
            ["akita_black_bowl_1", "is_left_of", "plate_1"],
            ["cookies_1", "is_on_top_of", "plate_1"],  # wrong subject, must be dropped
        ],
    )
    control = build_frame(**kwargs, variant="control")
    treatment = build_frame(**kwargs, variant="treatment")

    assert not np.array_equal(control["observation.images.image"], treatment["observation.images.image"])
    # Everything except the main image must stay identical between variants.
    assert np.array_equal(
        control["observation.images.image2"], treatment["observation.images.image2"]
    )
    assert np.allclose(control["observation.state"], treatment["observation.state"])
    assert np.allclose(control["action"], treatment["action"])
    assert control["task"] == treatment["task"]


def test_build_frame_treatment_with_no_target_relations_matches_control():
    kwargs = _frame_kwargs(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        relations=[["cookies_1", "is_on_top_of", "plate_1"]],  # no akita_black_bowl_1 subject
    )
    control = build_frame(**kwargs, variant="control")
    treatment = build_frame(**kwargs, variant="treatment")
    assert np.array_equal(control["observation.images.image"], treatment["observation.images.image"])


def test_build_frame_rejects_unknown_variant():
    try:
        build_frame(**_frame_kwargs(), variant="bogus")
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown variant")


def test_camera_key_mapping_matches_hdf5_naming():
    # obs/eye_in_hand_rgb has NO robot0_ prefix, unlike obs/robot0_eye_in_hand_bboxes /
    # _scene_graph / _world_coords (verified directly against a real HDF5 file). This
    # test pins build_frame's kwarg name to that convention so a future edit that
    # accidentally renames the kwarg to "robot0_eye_in_hand_rgb" fails loudly.
    import inspect

    params = inspect.signature(build_frame).parameters
    assert "eye_in_hand_rgb" in params
    assert "robot0_eye_in_hand_rgb" not in params
