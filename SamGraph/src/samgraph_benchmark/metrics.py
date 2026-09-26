"""Exact directed-triplet metrics matching the VLM evaluator semantics."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re
from typing import Iterable, Mapping

from .ground_truth import GroundTruthIndex, canonical_triplet


DUPLICATE_BOWL_SWAP = {"akita_black_bowl_1": "akita_black_bowl_2", "akita_black_bowl_2": "akita_black_bowl_1"}
VLM_RELATIONS = frozenset({
    "is_left_of", "is_right_of", "is_in_front_of", "is_behind",
    "is_on_top_of", "is_below_of", "is_inside", "contains",
})
VLM_RELATION_ALIASES = {
    "behind": "is_behind", "is_behind_of": "is_behind",
    "in_front_of": "is_in_front_of", "front_of": "is_in_front_of",
    "is_front_of": "is_in_front_of", "is_in_front": "is_in_front_of",
    "left_of": "is_left_of", "is_to_left_of": "is_left_of",
    "is_to_the_left_of": "is_left_of", "right_of": "is_right_of",
    "is_to_right_of": "is_right_of", "is_to_the_right_of": "is_right_of",
    "on_top_of": "is_on_top_of", "is_on_top": "is_on_top_of",
    "below_of": "is_below_of", "is_below": "is_below_of",
    "inside": "is_inside", "is_inside_of": "is_inside", "is_in": "is_inside",
    "is_contained_by": "is_inside", "is_contained_in": "is_inside",
    "is_contains": "contains", "is_contains_of": "contains", "contains_of": "contains",
}
_VLM_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _clean_vlm_token(value: object) -> str:
    text = str(value).strip().strip("`'\"")
    text = text.strip("[](){}").strip("`'\"")
    return text.strip()


def triplet_set(values: Iterable[object]) -> set[tuple[str, str, str]]:
    """Canonicalize SAM predictions like the VLM parser; discard invalid predicates/tokens."""
    result: set[tuple[str, str, str]] = set()
    for value in values:
        item = canonical_triplet(value)
        if item is None:
            continue
        subject, relation, object_ = (_clean_vlm_token(part) for part in item)
        relation = VLM_RELATION_ALIASES.get(relation, relation)
        if (
            relation in VLM_RELATIONS
            and _VLM_TOKEN.fullmatch(subject)
            and _VLM_TOKEN.fullmatch(object_)
        ):
            result.add((subject, relation, object_))
    return result


def ground_truth_triplet_set(values: Iterable[object]) -> set[tuple[str, str, str]]:
    """Keep only the evaluator's canonical GT predicates (GT aliases are not repaired)."""
    result: set[tuple[str, str, str]] = set()
    for value in values:
        item = canonical_triplet(value)
        if item is not None and item[1] in VLM_RELATIONS:
            result.add(item)
    return result


def _swap(values: Iterable[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {(DUPLICATE_BOWL_SWAP.get(a, a), rel, DUPLICATE_BOWL_SWAP.get(b, b)) for a, rel, b in values}


def frame_counts(gt: set[tuple[str, str, str]], pred: set[tuple[str, str, str]]) -> dict[str, int]:
    return {"tp": len(gt & pred), "fp": len(pred - gt), "fn": len(gt - pred)}


def _f1(counts: Mapping[str, int]) -> float:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0


def duplicate_bowl_invariant(gt: set[tuple[str, str, str]], pred: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], bool]:
    swapped = _swap(pred)
    if swapped != pred and _f1(frame_counts(gt, swapped)) > _f1(frame_counts(gt, pred)):
        return swapped, True
    return pred, False


@dataclass(frozen=True)
class EvaluationReport:
    """A benchmark report with explicit and backwards-compatible metric names.

    ``exact`` is the ordered ``(subject, relation, object)`` triplet metric
    requested for geometry tuning.  ``vlm_comparator`` applies the evaluator's
    duplicate-black-bowl identity-swap convention.  The older ``strict`` /
    ``primary`` / ``per_task`` / ``mean_task_f1`` fields are retained as v1
    aliases for the exact and comparator values respectively.
    """

    primary: dict
    strict: dict
    per_task: dict[str, dict]
    mean_task_f1: float
    frames_expected: int
    frames_scored: int
    missing_predictions: int
    frame_stride: int
    exact: dict
    vlm_comparator: dict
    exact_per_task: dict[str, dict]
    vlm_per_task: dict[str, dict]
    exact_mean_task_f1: float
    vlm_mean_task_f1: float

    def as_dict(self) -> dict:
        return {
            "exact": self.exact,
            "vlm_comparator": self.vlm_comparator,
            "exact_per_task": self.exact_per_task,
            "vlm_per_task": self.vlm_per_task,
            "exact_mean_task_f1": self.exact_mean_task_f1,
            "vlm_mean_task_f1": self.vlm_mean_task_f1,
            # Legacy v1 aliases retained for result-schema compatibility.
            "primary": self.primary,
            "strict": self.strict,
            "per_task": self.per_task,
            "mean_task_f1": self.mean_task_f1,
            "frames_expected": self.frames_expected,
            "frames_scored": self.frames_scored,
            "missing_predictions": self.missing_predictions,
            "frame_stride": self.frame_stride,
        }


def _aggregate(counts: Mapping[str, int]) -> dict:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_predictions(
    ground_truth: GroundTruthIndex,
    predictions: Mapping[tuple[str, str, int], Iterable[object]],
    *,
    frame_stride: int = 1,
) -> EvaluationReport:
    """Score zero-based frames at ``frame_stride``; missing predictions are empty sets."""

    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")

    comparator_total = defaultdict(int)
    exact_total = defaultdict(int)
    comparator_task_totals: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    exact_task_totals: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    comparator_demo_frame_f1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    exact_demo_frame_f1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    sampled_keys = [key for key in ground_truth.keys() if key[2] % frame_stride == 0]
    for key in sorted(sampled_keys):
        gt = ground_truth_triplet_set(ground_truth.frames[key])
        if key not in predictions:
            missing += 1
        pred = triplet_set(predictions.get(key, ()))
        comparator_pred, _ = duplicate_bowl_invariant(gt, pred)
        for target, counts in ((comparator_pred, comparator_total), (pred, exact_total)):
            frame = frame_counts(gt, target)
            for name, value in frame.items():
                counts[name] += value
        task = key[0]
        comparator_frame = frame_counts(gt, comparator_pred)
        exact_frame = frame_counts(gt, pred)
        comparator_demo_frame_f1[task][key[1]].append(_f1(comparator_frame))
        exact_demo_frame_f1[task][key[1]].append(_f1(exact_frame))
        for name, value in comparator_frame.items():
            comparator_task_totals[task][name] += value
        for name, value in exact_frame.items():
            exact_task_totals[task][name] += value

    def _per_task(
        totals: Mapping[str, Mapping[str, int]],
        demo_frame_f1: Mapping[str, Mapping[str, list[float]]],
    ) -> tuple[dict[str, dict], float]:
        per_task: dict[str, dict] = {}
        for task, counts in sorted(totals.items()):
            summary = _aggregate(counts)
            demo_f1 = [sum(values) / len(values) for values in demo_frame_f1[task].values() if values]
            summary["mean_per_demo_f1"] = round(sum(demo_f1) / len(demo_f1), 4) if demo_f1 else 0.0
            summary["macro_frame_f1"] = (
                sum(value for values in demo_frame_f1[task].values() for value in values)
                / sum(len(values) for values in demo_frame_f1[task].values())
                if any(demo_frame_f1[task].values()) else 0.0
            )
            summary["macro_frame_f1"] = round(summary["macro_frame_f1"], 4)
            per_task[task] = summary
        mean = round(
            sum(item["mean_per_demo_f1"] for item in per_task.values()) / len(per_task), 4
        ) if per_task else 0.0
        return per_task, mean

    comparator_per_task, comparator_mean = _per_task(comparator_task_totals, comparator_demo_frame_f1)
    exact_per_task, exact_mean = _per_task(exact_task_totals, exact_demo_frame_f1)
    return EvaluationReport(
        primary=_aggregate(comparator_total),
        strict=_aggregate(exact_total),
        per_task=comparator_per_task,
        mean_task_f1=comparator_mean,
        frames_expected=len(sampled_keys),
        frames_scored=len(sampled_keys),
        missing_predictions=missing,
        frame_stride=frame_stride,
        exact=_aggregate(exact_total),
        vlm_comparator=_aggregate(comparator_total),
        exact_per_task=exact_per_task,
        vlm_per_task=comparator_per_task,
        exact_mean_task_f1=exact_mean,
        vlm_mean_task_f1=comparator_mean,
    )
