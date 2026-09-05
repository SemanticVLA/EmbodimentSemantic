from vla_benchmarking.evaluation.backends import DIRECT_MATRIX, LEROBOT, validate_backend_condition
from vla_benchmarking.evaluation.contracts import (
    EvaluationCondition,
    EvaluationResult,
    aggregate_results,
    build_task_seed_matrix,
    parse_suite_mode,
)
from vla_benchmarking.evaluation.registry import validate_policy_condition


def test_shared_sealed_schedule_is_row_major_and_reproducible():
    cells = build_task_seed_matrix(suite_mode="sealed_randomized")
    assert len(cells) == 100
    assert [(cell.task_id, cell.episode_index, cell.seed) for cell in cells[:11]] == [
        (0, 0, 1000), (0, 1, 1001), (0, 2, 1002), (0, 3, 1003),
        (0, 4, 1004), (0, 5, 1005), (0, 6, 1006), (0, 7, 1007),
        (0, 8, 1008), (0, 9, 1009), (1, 0, 1000),
    ]
    assert cells[-1].condition.suite_mode == "sealed_randomized"


def test_public_normal_suite_preserves_native_vanilla_name():
    assert parse_suite_mode("normal") == "vanilla"
    cells = build_task_seed_matrix(task_ids=[0], episodes_per_task=1, suite_mode="normal")
    assert cells[0].condition.suite_mode == "vanilla"


def test_visual_and_text_inputs_are_independent_policy_factors():
    combined = EvaluationCondition(
        visual_input="relation_arrows",
        text_context="text_triplet",
        text_format="natural_compact",
    )
    assert validate_policy_condition("lerobot", combined).backend is LEROBOT

    canonical = EvaluationCondition(visual_input="goal_arrow")
    assert validate_policy_condition("canonical_grasp", canonical).backend is DIRECT_MATRIX
    try:
        validate_policy_condition(
            "canonical_grasp",
            EvaluationCondition(visual_input="goal_arrow", text_context="text_triplet"),
        )
    except ValueError as exc:
        assert "text context" in str(exc)
    else:
        raise AssertionError("canonical grasp must reject text augmentation")


def test_backend_capabilities_keep_policy_boundaries_explicit():
    validate_backend_condition(DIRECT_MATRIX, text_context="none", visual_arrow=False)
    validate_backend_condition(LEROBOT, text_context="text_triplet", visual_arrow=True)
    try:
        validate_backend_condition(DIRECT_MATRIX, text_context="scene_graph", visual_arrow=False)
    except ValueError as exc:
        assert "text context" in str(exc)
    else:
        raise AssertionError("direct matrix must reject text context")


def test_result_aggregation_preserves_terminal_failures():
    cells = build_task_seed_matrix(task_ids=[0], episodes_per_task=2)
    result = aggregate_results([
        EvaluationResult(cells[0], True),
        EvaluationResult(cells[1], False),
    ])
    assert result["successes"] == 1
    assert result["planned"] == result["terminal"] == 2
    assert result["per_task"]["0"] == {"successes": 1, "planned": 2, "terminal": 2}
