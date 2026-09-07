"""Paired hierarchical bootstrap statistics for automatic-TTT evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


class StatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class PairedOutcome:
    task_id: int | str
    run_id: str
    episode_id: str
    baseline_success: bool
    adapted_success: bool
    condition: str = "adapted"

    @property
    def delta(self) -> float:
        return float(self.adapted_success) - float(self.baseline_success)

    @property
    def identity(self) -> tuple[int | str, str, str]:
        return self.task_id, self.run_id, self.episode_id


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    resamples: int
    seed: int
    target: float
    decision: str


def validate_paired_alignment(records: Sequence[PairedOutcome]) -> None:
    if not records:
        raise StatisticsError("no paired outcomes")
    identities = [record.identity for record in records]
    if len(set(identities)) != len(identities):
        raise StatisticsError("duplicate task/run/episode identity")


def paired_success_delta(records: Sequence[PairedOutcome]) -> float:
    validate_paired_alignment(records)
    return _paired_success_delta_unchecked(records)


def _paired_success_delta_unchecked(records: Sequence[PairedOutcome]) -> float:
    # Bootstrap draws contain duplicate identities by construction.
    return sum(record.delta for record in records) / len(records)


def macro_task_success_delta(records: Sequence[PairedOutcome]) -> float:
    """Primary estimand: equal weight per task, then per paired episode."""
    validate_paired_alignment(records)
    by_task: dict[int | str, list[PairedOutcome]] = {}
    for record in records:
        by_task.setdefault(record.task_id, []).append(record)
    return sum(sum(row.delta for row in rows) / len(rows) for rows in by_task.values()) / len(by_task)


def _hierarchical_sample(records: Sequence[PairedOutcome], rng: random.Random) -> list[PairedOutcome]:
    """Sample tasks, then runs within task, then paired episodes.

    Synthetic task/run IDs mark repeated bootstrap draws so the macro-task
    statistic gives every sampled task replicate equal weight.
    """
    by_task: dict[int | str, dict[str, list[PairedOutcome]]] = {}
    for record in records:
        by_task.setdefault(record.task_id, {}).setdefault(record.run_id, []).append(record)
    task_keys = list(by_task)
    sampled: list[PairedOutcome] = []
    for task_draw in range(len(task_keys)):
        task = rng.choice(task_keys)
        runs = by_task[task]
        run_keys = list(runs)
        for run_draw in range(len(run_keys)):
            run = rng.choice(run_keys)
            episodes = runs[run]
            for episode in episodes:
                sampled_episode = rng.choice(episodes)
                sampled.append(
                    PairedOutcome(
                        task_id=f"bootstrap-task-{task_draw}",
                        run_id=f"bootstrap-run-{task_draw}-{run_draw}",
                        episode_id=f"bootstrap-episode-{task_draw}-{run_draw}-{episode.episode_id}",
                        baseline_success=sampled_episode.baseline_success,
                        adapted_success=sampled_episode.adapted_success,
                        condition=sampled_episode.condition,
                    )
                )
    return sampled


def hierarchical_bootstrap(
    records: Sequence[PairedOutcome],
    *,
    statistic: Callable[[Sequence[PairedOutcome]], float] = macro_task_success_delta,
    resamples: int = 2000,
    seed: int = 17,
    target: float = 0.10,
) -> BootstrapResult:
    """Resample task/run clusters, then episodes within each sampled cluster.

    A complete ``PairedOutcome`` is sampled as one unit, so baseline/adapted
    outcomes for an episode can never be separated during resampling.
    """
    validate_paired_alignment(records)
    if resamples < 1000:
        raise StatisticsError("use at least 1000 bootstrap resamples")
    if not 0 < target < 1:
        raise StatisticsError("target must be in (0,1)")
    estimate = float(statistic(records))
    rng = random.Random(seed)
    def draw_statistic(sample: Sequence[PairedOutcome]) -> float:
        if statistic is paired_success_delta:
            return _paired_success_delta_unchecked(sample)
        if statistic is macro_task_success_delta:
            grouped: dict[int | str, list[PairedOutcome]] = {}
            for item in sample:
                grouped.setdefault(item.task_id, []).append(item)
            return sum(sum(row.delta for row in rows) / len(rows) for rows in grouped.values()) / len(grouped)
        return float(statistic(sample))

    draws = sorted(draw_statistic(_hierarchical_sample(records, rng)) for _ in range(resamples))
    lower = _quantile(draws, 0.025)
    upper = _quantile(draws, 0.975)
    decision = "improvement" if estimate >= target and lower > 0 else "inconclusive_or_below_target"
    return BootstrapResult(estimate, lower, upper, resamples, seed, target, decision)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise StatisticsError("cannot compute quantile of empty values")
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


__all__ = [
    "BootstrapResult", "PairedOutcome", "StatisticsError", "hierarchical_bootstrap",
    "macro_task_success_delta", "paired_success_delta", "validate_paired_alignment",
]
