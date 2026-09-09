"""Pure aggregation and completeness checks for Arrow suite results."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .benchmark import DEFAULT_CASES, EvaluationRow
from .contracts import ContractError, _safe


REPORT_SCHEMA = "arrow_policy_suite.report.v1"


def _identity(row: EvaluationRow) -> tuple[str, str, str]:
    metadata = row.metadata
    task = row.task_id if row.task_id is not None else metadata.get("task_id", "unknown")
    reset = row.reset_id or str(metadata.get("reset_id", ""))
    episode = row.episode_id or str(metadata.get("episode_id", ""))
    if reset == "":
        raise ContractError(f"evaluation row {row.case!r} lacks reset_id for paired reporting")
    if episode == "":
        episode = f"task-{task}-reset-{reset}"
    return str(task), reset, episode


def _row_dict(row: EvaluationRow) -> dict[str, Any]:
    task, reset, episode = _identity(row)
    return {
        "case": row.case,
        "family": row.family,
        "variant": row.variant,
        "task_id": row.task_id if row.task_id is not None else task,
        "reset_id": reset,
        "episode_id": episode,
        "success_280": bool(row.success_280),
        "success_1200": bool(row.success_1200),
        "steps": int(row.steps),
        "teacher_proposals": int(row.teacher_proposals),
        "teacher_steps": int(row.teacher_steps),
        "branch_steps": int(row.branch_steps),
        "metadata": dict(row.metadata),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trials": 0,
            "successes_280": 0,
            "successes_1200": 0,
            "success_rate_280": None,
            "success_rate_1200": None,
            "teacher_proposals": 0,
            "teacher_steps": 0,
            "branch_steps": 0,
        }
    count = len(rows)
    return {
        "trials": count,
        "successes_280": sum(int(row["success_280"]) for row in rows),
        "successes_1200": sum(int(row["success_1200"]) for row in rows),
        "success_rate_280": sum(int(row["success_280"]) for row in rows) / count,
        "success_rate_1200": sum(int(row["success_1200"]) for row in rows) / count,
        "teacher_proposals": sum(int(row["teacher_proposals"]) for row in rows),
        "teacher_steps": sum(int(row["teacher_steps"]) for row in rows),
        "branch_steps": sum(int(row["branch_steps"]) for row in rows),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [{key: group_key, **_summary(group)} for group_key, group in sorted(groups.items(), key=lambda pair: str(pair[0]))]


def aggregate_rows(
    rows: Iterable[EvaluationRow],
    *,
    required_cases: Sequence[str] | None = None,
    required_tasks: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """Aggregate complete evaluation rows by policy, task, and reset.

    Every required policy case must appear exactly once for every task/reset
    identity.  The check is intentionally strict: a report must never turn an
    incomplete paired run into a plausible-looking success rate.
    """

    normalized = [_row_dict(row) for row in rows]
    if not normalized:
        raise ContractError("cannot report an empty evaluation")
    cases = tuple(required_cases or tuple(case.name for case in DEFAULT_CASES))
    if not cases:
        raise ContractError("paired report requires at least one policy case")
    # If a producer supplies split/checkpoint lineage, require it on every
    # row.  This prevents a partially annotated report from pairing results
    # across reset splits or checkpoints.
    for provenance_key in ("split_manifest_sha256", "checkpoint_lineage"):
        supplied = [row["metadata"].get(provenance_key) for row in normalized]
        if any(value is not None for value in supplied) and any(not value for value in supplied):
            raise ContractError(f"evaluation rows have incomplete {provenance_key} provenance")
    tasks = {str(task) for task in required_tasks} if required_tasks is not None else None
    seen: set[tuple[str, str, str, str]] = set()
    by_identity: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in normalized:
        task, reset, episode = str(row["task_id"]), row["reset_id"], row["episode_id"]
        if tasks is not None and task not in tasks:
            raise ContractError(f"unexpected task in report: {task}")
        if row["case"] not in cases:
            raise ContractError(f"unexpected policy case in report: {row['case']}")
        key = (task, reset, episode, row["case"])
        if key in seen:
            raise ContractError(f"duplicate evaluation row for paired identity {key}")
        seen.add(key)
        identity = (task, reset, episode)
        if row["case"] in by_identity[identity]:
            raise ContractError(f"duplicate evaluation case for paired identity {identity}")
        by_identity[identity][row["case"]] = row
    incomplete: list[dict[str, Any]] = []
    for identity, values in sorted(by_identity.items()):
        missing = sorted(set(cases) - set(values))
        if missing:
            incomplete.append({"identity": identity, "missing_cases": missing})
    if incomplete:
        raise ContractError(f"incomplete paired evaluation identities: {incomplete}")
    if tasks is not None:
        observed_tasks = {str(row["task_id"]) for row in normalized}
        if observed_tasks != tasks:
            raise ContractError(f"task set mismatch: expected {sorted(tasks)}, observed {sorted(observed_tasks)}")

    by_case: list[dict[str, Any]] = []
    for case in cases:
        case_rows = [row for row in normalized if row["case"] == case]
        by_case.append({"case": case, **_summary(case_rows)})
    by_task: list[dict[str, Any]] = []
    for task in sorted({str(row["task_id"]) for row in normalized}):
        task_rows = [row for row in normalized if str(row["task_id"]) == task]
        by_task.append({"task_id": task, "cases": [
            {"case": case, **_summary([row for row in task_rows if row["case"] == case])}
            for case in cases
        ]})
    by_reset: list[dict[str, Any]] = []
    for identity, values in sorted(by_identity.items()):
        task, reset, episode = identity
        by_reset.append({
            "task_id": task,
            "reset_id": reset,
            "episode_id": episode,
            "cases": {
                case: {
                    "success_280": bool(values[case]["success_280"]),
                    "success_1200": bool(values[case]["success_1200"]),
                    "steps": values[case]["steps"],
                }
                for case in cases
            },
        })

    reference = cases[0]
    pairwise: list[dict[str, Any]] = []
    for case in cases[1:]:
        outcomes = [
            (bool(values[reference]["success_280"]), bool(values[case]["success_280"]),
             bool(values[reference]["success_1200"]), bool(values[case]["success_1200"]))
            for values in by_identity.values()
        ]
        count = len(outcomes)
        pairwise.append({
            "reference_case": reference,
            "case": case,
            "paired_trials": count,
            "delta_280": sum(int(after) - int(before) for before, after, _, _ in outcomes) / count,
            "delta_1200": sum(int(after) - int(before) for _, _, before, after in outcomes) / count,
            "improved_280": sum(after and not before for before, after, _, _ in outcomes),
            "regressed_280": sum(before and not after for before, after, _, _ in outcomes),
            "improved_1200": sum(after and not before for _, _, before, after in outcomes),
            "regressed_1200": sum(before and not after for _, _, before, after in outcomes),
        })
    return {
        "schema": REPORT_SCHEMA,
        "cases": list(cases),
        "paired_trials": len(by_identity),
        "by_case": by_case,
        "by_task": by_task,
        "by_reset": by_reset,
        "pairwise": pairwise,
        "rows": normalized,
    }


def write_report(path: str | Path, report: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# Descriptive aliases for callers that prefer the benchmark terminology.
aggregate_evaluation_rows = aggregate_rows
paired_report = aggregate_rows


__all__ = ["REPORT_SCHEMA", "aggregate_rows", "aggregate_evaluation_rows", "paired_report", "write_report"]
