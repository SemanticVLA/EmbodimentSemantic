from samgraph_benchmark.ground_truth import GroundTruthIndex
from samgraph_benchmark.metrics import evaluate_predictions


def test_duplicate_bowl_primary_and_strict_diagnostic():
    key = ("task", "demo_0", 0)
    gt = GroundTruthIndex({key: frozenset({("akita_black_bowl_1", "is_left_of", "plate_1")})})
    report = evaluate_predictions(gt, {key: [("black_bowl_2", "is_left_of", "plate_1")]})
    assert report.primary["f1"] == 1.0
    assert report.strict["f1"] == 0.0


def test_missing_predictions_are_frames_and_false_negatives():
    key = ("task", "demo_0", 0)
    gt = GroundTruthIndex({key: frozenset({("plate_1", "is_left_of", "cookies_1")})})
    report = evaluate_predictions(gt, {})
    assert report.frames_expected == report.frames_scored == 1
    assert report.missing_predictions == 1
    assert report.primary["fn"] == 1


def test_stride_scores_sampled_frames_only_and_missing_sampled_frames_are_fns():
    gt = GroundTruthIndex({
        ("task", "demo_0", 0): frozenset({("a", "is_left_of", "b")}),
        ("task", "demo_0", 1): frozenset({("a", "is_left_of", "b")}),
        ("task", "demo_0", 2): frozenset({("a", "is_left_of", "b")}),
        ("task", "demo_0", 5): frozenset({("a", "is_left_of", "b")}),
    })
    report = evaluate_predictions(
        gt, {("task", "demo_0", 0): [("a", "is_left_of", "b")]}, frame_stride=5
    )
    assert report.frames_expected == report.frames_scored == 2
    assert report.missing_predictions == 1
    assert report.primary["tp"] == 1
    assert report.primary["fn"] == 1


def test_metrics_reject_nonpositive_frame_stride():
    gt = GroundTruthIndex({("task", "demo_0", 0): frozenset()})
    try:
        evaluate_predictions(gt, {}, frame_stride=0)
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("zero frame stride was accepted")


def test_prediction_relations_use_vlm_aliases_and_drop_invalid_predicates():
    key = ("task", "demo_0", 0)
    gt = GroundTruthIndex({key: frozenset({("plate_1", "is_left_of", "cookies_1")})})
    report = evaluate_predictions(gt, {key: [
        ("plate_1", "left_of", "cookies_1"),
        ("plate_1", "unknown", "cookies_1"),
    ]})
    assert report.primary["f1"] == 1.0
    assert report.primary["fp"] == 0


def test_mean_task_f1_matches_vlm_mean_per_demo_frame_f1():
    gt = GroundTruthIndex({
        ("task", "demo_0", 0): frozenset({("a", "is_left_of", "b")}),
        ("task", "demo_0", 1): frozenset({
            ("a", "is_left_of", "b"), ("c", "is_right_of", "d"),
            ("e", "is_inside", "f"),
        }),
    })
    report = evaluate_predictions(gt, {("task", "demo_0", 0): [("a", "is_left_of", "b")]})
    assert report.primary["f1"] == 0.4  # one TP and three FN over all four GT triplets
    assert report.per_task["task"]["mean_per_demo_f1"] == 0.5
    assert report.mean_task_f1 == 0.5


def test_empty_micro_aggregate_uses_vlm_zero_division_convention():
    key = ("task", "demo_0", 0)
    report = evaluate_predictions(GroundTruthIndex({key: frozenset()}), {key: []})
    assert report.primary["f1"] == 0.0
    assert report.mean_task_f1 == 1.0
