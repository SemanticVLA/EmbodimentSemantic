import pytest

from vlm_bench.eval import FrameResult


def _result(gt, pred):
    return FrameResult(
        task="task",
        demo="demo_0",
        frame=0,
        camera="agentview",
        gt=set(gt),
        pred=set(pred),
    )


def test_swapped_bowl_ids_against_anchor_objects_get_full_credit():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_2", "is_right_of", "plate_1"),
        },
        {
            ("akita_black_bowl_2", "is_left_of", "plate_1"),
            ("akita_black_bowl_1", "is_right_of", "plate_1"),
        },
    )

    assert result.f1 == 1.0
    assert len(result.tp) == 2
    assert not result.fp
    assert not result.fn


def test_missing_one_bowl_relation_still_gets_only_partial_credit():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_2", "is_left_of", "plate_1"),
        },
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
        },
    )

    assert len(result.tp) == 1
    assert not result.fp
    assert len(result.fn) == 1
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 == pytest.approx(2 / 3)


def test_swapped_bowl_to_bowl_edges_get_full_credit():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
            ("akita_black_bowl_2", "is_right_of", "akita_black_bowl_1"),
        },
        {
            ("akita_black_bowl_2", "is_left_of", "akita_black_bowl_1"),
            ("akita_black_bowl_1", "is_right_of", "akita_black_bowl_2"),
        },
    )

    assert result.f1 == 1.0
    assert len(result.tp) == 2
    assert not result.fp
    assert not result.fn


def test_wrong_relation_type_remains_wrong_after_bowl_swap():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
        },
        {
            ("akita_black_bowl_2", "is_behind", "plate_1"),
        },
    )

    assert result.f1 == 0.0
    assert not result.tp
    assert len(result.fp) == 1
    assert len(result.fn) == 1


def test_unsuffixed_bowl_name_does_not_receive_duplicate_bowl_leniency():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
        },
        {
            ("akita_black_bowl", "is_left_of", "plate_1"),
        },
    )

    assert result.f1 == 0.0
    assert not result.tp
    assert len(result.fp) == 1
    assert len(result.fn) == 1


def test_inconsistent_bowl_assignment_is_not_repaired_per_triplet():
    result = _result(
        {
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_2", "is_right_of", "flat_stove_1"),
        },
        {
            ("akita_black_bowl_2", "is_left_of", "plate_1"),
            ("akita_black_bowl_2", "is_right_of", "flat_stove_1"),
        },
    )

    assert len(result.tp) == 1
    assert len(result.fp) == 1
    assert len(result.fn) == 1
