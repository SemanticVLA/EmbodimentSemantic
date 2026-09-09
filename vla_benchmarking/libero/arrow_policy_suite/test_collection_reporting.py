from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.benchmark import EvaluationRow
from arrow_policy_suite.benchmark import DEFAULT_CASES, PolicyCase, evaluate_suite
from arrow_policy_suite.collection import (
    AttemptIdentity, CollectionAttempt, collect_on_call, derive_trace_view,
)
from arrow_policy_suite.reporting import aggregate_rows
from arrow_policy_suite.runtime import EpisodeResult, EpisodeStats


@dataclass
class _Frame:
    observation: dict
    step: int
    metadata: dict
    episode_id: str = "ep"

    @property
    def digest(self):
        return f"frame-{self.step}"


def _result(success: bool = True) -> EpisodeResult:
    frame = _Frame({"state": [0.0] * 8, "image": "raw"}, 0, {"episode_id": "ep"})
    teacher = type("Proposal", (), {"action": (0.0,) * 7, "policy_id": "arrow", "metadata": {}})()
    decision = type("Decision", (), {
        "action": (0.0,) * 7, "metadata": {"teacher_used": True},
        "teacher_used": True, "policy_id": "arrow_on_call",
    })()
    record = type("Record", (), {
        "frame": frame, "base": teacher, "teacher": teacher, "decision": decision,
        "next_frame": frame, "success": success, "raw_result": {"success": success},
    })()
    return EpisodeResult((record,), EpisodeStats(success, success, 1))


def test_collection_runs_exactly_fifty_attempts_and_keeps_rejections(tmp_path):
    calls = []

    def runner(task, index, identity):
        calls.append((task, index, identity.reset_id))
        return _result(success=index == 0)

    attempts, manifest = collect_on_call(
        [0], runner, reset_ids={0: [f"r-{i}" for i in range(50)]},
        master_log=tmp_path / "master.jsonl", manifest_path=tmp_path / "collection.json",
    )
    assert len(calls) == 50 and len(attempts) == 50
    assert manifest.attempted == 50 and manifest.eligible == 1
    assert len((tmp_path / "master.jsonl").read_text().splitlines()) == 50


def test_trace_view_is_state_only_and_lineage_bound():
    identity = AttemptIdentity(0, "reset-1", "episode-1")
    attempt = CollectionAttempt(identity, _result(), True)
    view = derive_trace_view([attempt], parent_manifest_sha256="a" * 64)
    assert view.parent_manifest_sha256 == "a" * 64
    assert view.routes[0].states == ((0.0,) * 8,)
    assert "image" not in view.as_dict()["routes"][0]


def test_reporting_requires_complete_pairs_and_keeps_both_horizons():
    rows = []
    for reset in ("r-1", "r-2"):
        for case, success in (("base", False), ("adapted", True)):
            rows.append(EvaluationRow(case, "arrow_together", None, success, success, 1, 0, 0, 0,
                                      {}, 0, reset, f"ep-{reset}"))
    report = aggregate_rows(rows, required_cases=("base", "adapted"), required_tasks=(0,))
    assert report["paired_trials"] == 2
    assert report["pairwise"][0]["delta_280"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_rows(rows[:-1], required_cases=("base", "adapted"))


def test_collection_rejects_teacher_labels_from_the_wrong_policy_family(tmp_path):
    def runner(_task, _index, _identity):
        result = _result()
        result.records[0].decision.policy_id = "arrow_editor"
        return result

    attempts, manifest = collect_on_call([0], runner, reset_ids={0: [f"r-{i}" for i in range(50)]})
    assert not attempts[0].eligible
    assert "source_policy_is_not_arrow_on_call" in attempts[0].eligibility_reasons
    assert manifest.eligible == 0


def test_trace_requires_source_manifest_and_factory_family_is_checked():
    identity = AttemptIdentity(0, "reset-1", "episode-1")
    attempt = CollectionAttempt(identity, _result(), True)
    with pytest.raises(ValueError, match="parent_manifest"):
        derive_trace_view([attempt])

    class _Coordinator:
        def run(self, _policy, *, max_steps):
            return _result()

    with pytest.raises(ValueError, match="incomplete policy case"):
        evaluate_suite(DEFAULT_CASES[:-1], lambda _case: _Coordinator(), lambda _case: type("P", (), {"policy_id": _case.family})())
    with pytest.raises(ValueError, match="returned"):
        evaluate_suite(DEFAULT_CASES, lambda _case: _Coordinator(), lambda _case: type("P", (), {"policy_id": "wrong"})())
