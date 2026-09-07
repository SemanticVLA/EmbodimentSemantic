"""Paired evaluation bookkeeping for frozen versus adapted VLA checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .experiment import wilson_interval


SUPPORTED_CONDITIONS = frozenset({
    "frozen_baseline", "adapted", "hybrid", "correction_only",
    "full_failure_context", "shuffled_failure_context", "reset_fast_state", "gdn",
})


@dataclass(frozen=True)
class TrialMetric:
    policy_id: str
    vla: str
    task_id: int
    seed: int
    condition: str  # frozen_baseline, adapted, correction_only, ...
    success: bool
    teacher_used: bool = False
    teacher_success: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # A seed is not necessarily unique when several randomized episodes share
    # it.  Keep an explicit episode key so paired reporting never overwrites
    # trials; the seed remains a useful secondary audit field.
    episode_id: str = ""
    initial_state_hash: str = ""
    checkpoint_lineage: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "vla": self.vla,
            "task_id": self.task_id,
            "seed": self.seed,
            "condition": self.condition,
            "success": int(self.success),
            "teacher_used": self.teacher_used,
            "teacher_success": self.teacher_success,
            "episode_id": self.episode_id,
            "initial_state_hash": self.initial_state_hash,
            "checkpoint_lineage": self.checkpoint_lineage,
            "metadata": dict(self.metadata),
        }


def exact_mcnemar_pvalue(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    if not isinstance(improved, int) or not isinstance(regressed, int) or improved < 0 or regressed < 0:
        raise ValueError("discordant counts must be non-negative integers")
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = sum(math.comb(discordant, index) for index in range(min(improved, regressed) + 1)) / (2 ** discordant)
    upper = sum(math.comb(discordant, index) for index in range(max(improved, regressed), discordant + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * min(lower, upper))


def paired_report(
    metrics: Sequence[TrialMetric],
    *,
    alpha: float = 0.05,
    required_conditions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Join baseline/adapted trials using the complete episode identity.

    A missing identity or an unmatched/duplicate condition is an error.  It
    is unsafe for an evaluation report to silently omit a seed or pair a
    different reset state merely because the VLA and task names match.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if not metrics:
        return []
    required = set(required_conditions or ("frozen_baseline", "adapted"))
    required.update({"frozen_baseline", "adapted"})
    unsupported_required = required - SUPPORTED_CONDITIONS
    if unsupported_required:
        raise ValueError(f"unsupported required evaluation conditions: {sorted(unsupported_required)}")
    groups: dict[tuple[str, str, int, int, str, str, str], dict[str, TrialMetric]] = {}
    for metric in metrics:
        for field_name in ("success", "teacher_used", "teacher_success"):
            if type(getattr(metric, field_name)) is not bool:
                raise ValueError(f"{metric.episode_id or '<unknown>'}: {field_name} must be a boolean")
        if not metric.policy_id or not metric.vla or not metric.episode_id:
            raise ValueError("paired evaluation requires policy_id, vla, and episode_id")
        if not metric.initial_state_hash:
            raise ValueError(f"{metric.episode_id}: initial_state_hash is required for strict pairing")
        if not metric.checkpoint_lineage:
            raise ValueError(f"{metric.episode_id}: checkpoint_lineage is required for strict pairing")
        if metric.condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"unsupported paired evaluation condition {metric.condition!r}")
        key = (
            metric.policy_id, metric.vla, metric.task_id, metric.seed,
            metric.episode_id, metric.initial_state_hash, metric.checkpoint_lineage,
        )
        conditions = groups.setdefault(key, {})
        if metric.condition in conditions:
            raise ValueError(f"duplicate {metric.condition} metric for paired identity {key}")
        conditions[metric.condition] = metric
    rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key, conditions in sorted(groups.items()):
        policy_id, vla, task, seed, episode_id, _initial_hash, _lineage = key
        baseline = conditions.get("frozen_baseline")
        adapted = conditions.get("adapted")
        if baseline is None or adapted is None:
            raise ValueError(f"unpaired evaluation identity {key}: frozen_baseline and adapted are required")
        missing_conditions = sorted(required - set(conditions))
        if missing_conditions:
            raise ValueError(f"incomplete evaluation identity {key}: missing conditions {missing_conditions}")
        hybrid = conditions.get("hybrid")
        # All conditions for a paired identity must have equal data exposure.
        # If a producer supplies these audit counters, require every condition
        # to supply the same value; never silently compare unequal controls.
        for field_name in ("support_episode_count", "training_token_count", "context_token_count"):
            supplied = [item.metadata.get(field_name) for item in conditions.values()]
            if any(value is not None for value in supplied):
                if any(value is None for value in supplied) or len(set(supplied)) != 1:
                    raise ValueError(f"data exposure mismatch for {key}: {field_name}")
        rows.setdefault((vla, task), []).append(
            {
                "episode_id": episode_id,
                "policy_id": policy_id,
                "seed": seed,
                "initial_state_hash": baseline.initial_state_hash,
                "checkpoint_lineage": baseline.checkpoint_lineage,
                "baseline_success": int(baseline.success),
                "adapted_success": int(adapted.success),
                "delta": int(adapted.success) - int(baseline.success),
                "teacher_used": int(hybrid.teacher_used) if hybrid is not None else 0,
                "teacher_success": int(hybrid.teacher_success) if hybrid is not None else 0,
                "hybrid_success": int(hybrid.success) if hybrid is not None else None,
                "teacher_eligible": bool(hybrid and hybrid.metadata.get("teacher_eligible", hybrid.teacher_used)),
                "controls": {
                    condition: int(metric.success)
                    for condition, metric in conditions.items()
                    if condition not in {"frozen_baseline", "adapted", "hybrid"}
                },
            }
        )
    report: list[dict[str, Any]] = []
    for (vla, task), pairs in sorted(rows.items()):
        n = len(pairs)
        base = sum(row["baseline_success"] for row in pairs)
        post = sum(row["adapted_success"] for row in pairs)
        improved = sum(row["delta"] == 1 for row in pairs)
        regressed = sum(row["delta"] == -1 for row in pairs)
        paired_p = exact_mcnemar_pvalue(improved, regressed)
        hybrid_pairs = [row for row in pairs if row["hybrid_success"] is not None]
        teacher_eligible = [row for row in hybrid_pairs if row["teacher_eligible"]]
        teacher_used = sum(row["teacher_used"] for row in teacher_eligible)
        teacher_success = sum(row["teacher_success"] for row in teacher_eligible if row["teacher_used"])
        hybrid_values = [row["hybrid_success"] for row in pairs if row["hybrid_success"] is not None]
        control_reports: dict[str, dict[str, Any]] = {}
        for condition in sorted({condition for row in pairs for condition in row["controls"]}):
            control_pairs = [row for row in pairs if condition in row["controls"]]
            control_successes = sum(row["controls"][condition] for row in control_pairs)
            control_improved = sum(
                row["controls"][condition] == 1 and row["baseline_success"] == 0
                for row in control_pairs
            )
            control_regressed = sum(
                row["controls"][condition] == 0 and row["baseline_success"] == 1
                for row in control_pairs
            )
            control_reports[condition] = {
                "paired_trials": len(control_pairs),
                "success_rate": control_successes / len(control_pairs),
                "wilson95": wilson_interval(control_successes, len(control_pairs)),
                "paired_delta_vs_baseline": (
                    sum(row["controls"][condition] - row["baseline_success"] for row in control_pairs)
                    / len(control_pairs)
                ),
                "discordant_improved": control_improved,
                "discordant_regressed": control_regressed,
                "mcnemar_exact_pvalue": exact_mcnemar_pvalue(control_improved, control_regressed),
            }
        report.append(
            {
                "vla": vla,
                "task_id": task,
                "paired_trials": n,
                "baseline_success_rate": base / n,
                "adapted_success_rate": post / n,
                "absolute_improvement": (post - base) / n,
                "improved_trials": improved,
                "regressed_trials": regressed,
                "tie_trials": n - improved - regressed,
                "discordant_improved": improved,
                "discordant_regressed": regressed,
                "paired_mcnemar_exact_pvalue": paired_p,
                "paired_test_alpha": alpha,
                "decision": (
                    "improvement_supported" if improved > regressed and paired_p < alpha
                    else "regression_supported" if regressed > improved and paired_p < alpha
                    else "inconclusive"
                ),
                "baseline_wilson95": wilson_interval(base, n),
                "adapted_wilson95": wilson_interval(post, n),
                # These are reported separately from the causal VLA-only
                # comparison.  ``adapted`` should be teacher-free; a separate
                # ``hybrid`` condition is required to report takeover success
                # without conflating it with learned improvement.
                "vla_only_success_rate": post / n,
                "teacher_used_trials": teacher_used,
                "teacher_recovery_success_trials": teacher_success,
                "teacher_eligible_trials": len(teacher_eligible),
                "teacher_recovery_rate": (teacher_success / teacher_used if teacher_used else None),
                "teacher_recovery_wilson95": (wilson_interval(teacher_success, teacher_used) if teacher_used else None),
                "hybrid_success_rate": (sum(hybrid_values) / len(hybrid_values) if hybrid_values else None),
                "hybrid_trials": len(hybrid_values),
                "control_success_rates": {
                    condition: sum(row["controls"][condition] for row in pairs if condition in row["controls"])
                    / sum(condition in row["controls"] for row in pairs)
                    for condition in sorted({condition for row in pairs for condition in row["controls"]})
                },
                "control_reports": control_reports,
            }
        )
    return report


def write_metrics(metrics: Iterable[TrialMetric], path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable metrics artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [metric.to_json() for metric in metrics]
    target.write_text(json.dumps({"episodes": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def evaluate_from_config(*, config: Any, args: Any | None = None) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NEEDS_EVALUATOR",
        "config_digest": config.digest(),
        "conditions": list(config.controls),
        "vlas": list(config.canonical_dict()["vla_names"]),
        "tasks": list(config.task_ids),
        "message": "Inject the same held-out seeds and policy factories for frozen/adapted conditions; no evaluation was launched.",
    }


__all__ = ["SUPPORTED_CONDITIONS", "TrialMetric", "evaluate_from_config", "exact_mcnemar_pvalue", "paired_report", "write_metrics"]
