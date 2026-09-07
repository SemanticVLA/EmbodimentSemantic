from __future__ import annotations

import pytest

from .statistics import (
    PairedOutcome,
    StatisticsError,
    hierarchical_bootstrap,
    macro_task_success_delta,
    paired_success_delta,
    validate_paired_alignment,
)


def _records():
    return [
        PairedOutcome(task, f"run-{run}", f"ep-{task}-{run}-{index}", baseline, adapted)
        for task in range(2)
        for run in range(2)
        for index, (baseline, adapted) in enumerate(((False, True), (True, True), (False, False)))
    ]


def test_paired_delta_and_reproducible_hierarchical_bootstrap():
    records = _records()
    assert paired_success_delta(records) == pytest.approx(1 / 3)
    first = hierarchical_bootstrap(records, resamples=1000, seed=29)
    second = hierarchical_bootstrap(records, resamples=1000, seed=29)
    assert first == second
    assert first.resamples == 1000


def test_duplicate_identity_rejected():
    records = _records()
    with pytest.raises(StatisticsError, match="duplicate"):
        validate_paired_alignment(records + [records[0]])


def test_bootstrap_preserves_paired_identity():
    records = [PairedOutcome(0, "run", "episode", False, True)]
    result = hierarchical_bootstrap(records, resamples=1000)
    assert result.estimate == 1.0
    assert result.lower == result.upper == 1.0


def test_macro_task_estimand_does_not_pool_unequal_task_counts():
    records = [
        PairedOutcome(0, "run", "task0-0", False, True),
        PairedOutcome(1, "run", "task1-0", False, False),
        PairedOutcome(1, "run", "task1-1", False, False),
        PairedOutcome(1, "run", "task1-2", False, False),
    ]
    assert paired_success_delta(records) == pytest.approx(0.25)
    assert macro_task_success_delta(records) == pytest.approx(0.50)
