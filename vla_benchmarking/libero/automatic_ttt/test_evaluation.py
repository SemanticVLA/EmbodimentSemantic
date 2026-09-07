from __future__ import annotations

import pytest

from .evaluation import TrialMetric, exact_mcnemar_pvalue, paired_report


def _metric(condition: str, *, seed: int = 1, state: str = "state-a", lineage: str = "lineage-a",
            success: bool = False, teacher_used: bool = False, teacher_success: bool = False,
            teacher_eligible: bool | None = None) -> TrialMetric:
    metadata = {}
    if teacher_eligible is not None:
        metadata["teacher_eligible"] = teacher_eligible
    return TrialMetric(
        policy_id="p", vla="vla", task_id=0, seed=seed, condition=condition,
        success=success, teacher_used=teacher_used, teacher_success=teacher_success,
        episode_id=f"episode-{seed}", initial_state_hash=state,
        checkpoint_lineage=lineage, metadata=metadata,
    )


def test_pairing_rejects_seed_or_initial_state_mismatch():
    with pytest.raises(ValueError, match="unpaired"):
        paired_report([_metric("frozen_baseline"), _metric("adapted", seed=2)])
    with pytest.raises(ValueError, match="unpaired"):
        paired_report([_metric("frozen_baseline"), _metric("adapted", state="state-b")])
    with pytest.raises(ValueError, match="unpaired"):
        paired_report([_metric("frozen_baseline"), _metric("adapted", lineage="lineage-b")])


def test_pairing_rejects_duplicates_and_missing_identity():
    with pytest.raises(ValueError, match="duplicate"):
        paired_report([_metric("frozen_baseline"), _metric("frozen_baseline"), _metric("adapted")])
    with pytest.raises(ValueError, match="initial_state_hash"):
        paired_report([_metric("frozen_baseline", state=""), _metric("adapted", state="")])


def test_teacher_recovery_uses_hybrid_teacher_used_denominator():
    metrics = [
        _metric("frozen_baseline", success=False),
        _metric("adapted", success=True, teacher_used=True, teacher_success=True),
        _metric("hybrid", success=True, teacher_used=True, teacher_success=True),
    ]
    report = paired_report(metrics)[0]
    assert report["vla_only_success_rate"] == 1.0
    assert report["teacher_eligible_trials"] == 1
    assert report["teacher_recovery_rate"] == 1.0
    assert report["teacher_recovery_wilson95"] is not None


def test_exact_mcnemar_is_deterministic_and_report_has_decision_fields():
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    assert exact_mcnemar_pvalue(3, 0) == 0.25
    report = paired_report([
        _metric("frozen_baseline", success=False),
        _metric("adapted", success=True),
    ])[0]
    assert report["discordant_improved"] == 1
    assert report["discordant_regressed"] == 0
    assert report["paired_mcnemar_exact_pvalue"] == 1.0
    assert report["decision"] == "inconclusive"


def test_registered_controls_are_reported_with_equal_data_exposure():
    base = _metric("frozen_baseline")
    adapted = _metric("adapted", success=True)
    control = _metric("correction_only", success=False)
    metrics = [base, adapted, control]
    report = paired_report(metrics)[0]
    assert report["control_success_rates"]["correction_only"] == 0.0
    mismatch = _metric("correction_only", success=False)
    mismatch = TrialMetric(**{**mismatch.__dict__, "metadata": {"training_token_count": 9}})
    with pytest.raises(ValueError, match="data exposure mismatch"):
        paired_report([base, adapted, mismatch])


def test_required_conditions_are_enforced_for_complete_reports():
    with pytest.raises(ValueError, match="missing conditions"):
        paired_report(
            [_metric("frozen_baseline"), _metric("adapted")],
            required_conditions=("frozen_baseline", "adapted", "hybrid", "gdn"),
        )


def test_paired_report_rejects_non_boolean_trial_flags():
    metric = _metric("frozen_baseline")
    metric = TrialMetric(**{**metric.__dict__, "teacher_used": "false"})
    with pytest.raises(ValueError, match="teacher_used must be a boolean"):
        paired_report([metric, _metric("adapted")])
