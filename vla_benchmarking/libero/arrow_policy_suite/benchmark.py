"""Small benchmark harness for paired seven-policy evaluation.

It does not choose checkpoints, create reset identities, or launch training.
Those decisions belong to the existing experiment launcher.  This module only
ensures every concrete row is evaluated through the same coordinator and that
both horizon metrics come from one rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .contracts import POLICY_IDS, ContractError
from .runtime import EpisodeResult, TransactionalCoordinator


@dataclass(frozen=True)
class PolicyCase:
    name: str
    family: str
    variant: str | None = None

    def __post_init__(self) -> None:
        if self.family not in POLICY_IDS:
            raise ContractError(f"unknown policy family {self.family!r}")
        if self.family == "arrow_minimal" and self.variant not in {"runtime_oracle", "learned"}:
            raise ContractError("Minimal cases require runtime_oracle or learned variant")
        if self.family != "arrow_minimal" and self.variant is not None:
            raise ContractError("only Minimal has a variant")


DEFAULT_CASES = (
    PolicyCase("arrow_together", "arrow_together"),
    PolicyCase("arrow_on_call", "arrow_on_call"),
    PolicyCase("arrow_apprentice", "arrow_apprentice"),
    PolicyCase("arrow_editor", "arrow_editor"),
    PolicyCase("arrow_minimal_runtime_oracle", "arrow_minimal", "runtime_oracle"),
    PolicyCase("arrow_minimal_learned", "arrow_minimal", "learned"),
    PolicyCase("arrow_fast", "arrow_fast"),
    PolicyCase("arrow_trace", "arrow_trace"),
)


def validate_case_set(
    cases: Sequence[PolicyCase],
    *,
    expected_cases: Sequence[PolicyCase] = DEFAULT_CASES,
    allow_incomplete: bool = False,
) -> tuple[PolicyCase, ...]:
    """Validate a complete frozen policy matrix before any rollout starts."""

    values = tuple(cases)
    names = [case.name for case in values]
    if len(set(names)) != len(names):
        raise ContractError("policy case names must be unique")
    if not values:
        raise ContractError("benchmark requires at least one policy case")
    expected = {case.name: case.family for case in expected_cases}
    observed = {case.name: case.family for case in values}
    if not allow_incomplete:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
        if missing or extra or mismatched:
            raise ContractError(
                f"incomplete policy case matrix: missing={missing}, extra={extra}, family_mismatch={mismatched}"
            )
    return values


def _policy_family(policy: Any) -> str:
    value = getattr(policy, "policy_id", None)
    if value is None:
        from .controller import policy_identifier
        value = policy_identifier(policy)
    return str(value)


def _validate_factory_policy(case: PolicyCase, policy: Any) -> None:
    actual = _policy_family(policy)
    if actual != case.family:
        raise ContractError(f"policy factory returned {actual!r} for case {case.name!r}; expected {case.family!r}")


@dataclass(frozen=True)
class EvaluationRow:
    case: str
    family: str
    variant: str | None
    success_280: bool
    success_1200: bool
    steps: int
    teacher_proposals: int
    teacher_steps: int
    branch_steps: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_id: int | str | None = None
    reset_id: str = ""
    episode_id: str = ""
    scored_steps: int | None = None
    cloned_steps: int = 0
    latency_seconds: float = 0.0
    policy_latency_seconds: float = 0.0
    branch_latency_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "family": self.family,
            "variant": self.variant,
            "task_id": self.task_id,
            "reset_id": self.reset_id,
            "episode_id": self.episode_id,
            "success_280": self.success_280,
            "success_1200": self.success_1200,
            "steps": self.steps,
            "teacher_proposals": self.teacher_proposals,
            "teacher_steps": self.teacher_steps,
            "branch_steps": self.branch_steps,
            "metadata": dict(self.metadata),
            "scored_steps": self.scored_steps if self.scored_steps is not None else self.steps,
            "cloned_steps": self.cloned_steps,
            "latency_seconds": self.latency_seconds,
            "policy_latency_seconds": self.policy_latency_seconds,
            "branch_latency_seconds": self.branch_latency_seconds,
        }


def row_from_result(
    case: PolicyCase,
    result: EpisodeResult,
    *,
    horizon_280: int = 280,
    task_id: int | str | None = None,
    reset_id: str = "",
    episode_id: str = "",
) -> EvaluationRow:
    if horizon_280 <= 0:
        raise ContractError("the secondary horizon must be positive")
    if result.stats.steps > 1200:
        raise ContractError("episode exceeds the primary 1200-step horizon")
    records = tuple(getattr(result, "records", ()))
    # Runtime implementations may either stop at success or retain the full
    # rollout while marking the successful transition.  Derive the secondary
    # checkpoint from the transition evidence when available, not only from
    # the final episode length.
    success_280 = bool(result.stats.success and result.stats.steps <= horizon_280)
    if records:
        success_280 = any(
            bool(getattr(record, "success", False))
            and int(getattr(getattr(record, "frame", None), "step", getattr(getattr(record, "frame", None), "timestep", 0))) + 1 <= horizon_280
            for record in records
        )
    teacher_proposals = sum(int(getattr(record, "teacher", None) is not None) for record in records)
    teacher_steps = sum(
        int(bool(getattr(getattr(record, "decision", None), "teacher_used", False)
                 or getattr(getattr(record, "decision", None), "metadata", {}).get("teacher_used", False)))
        for record in records
    )
    branch_steps = sum(int(getattr(getattr(record, "decision", None), "metadata", {}).get("branch_steps", 0)) for record in records)
    stats_scored = getattr(result.stats, "scored_steps", None)
    stats_cloned = int(getattr(result.stats, "cloned_steps", 0) or 0)
    stats_branch = int(getattr(result.stats, "branch_steps", 0) or 0)
    if stats_branch:
        branch_steps = stats_branch
    cloned_steps = stats_cloned or branch_steps
    return EvaluationRow(
        case.name, case.family, case.variant,
        success_280,
        bool(result.stats.success),
        result.stats.steps, int(getattr(result.stats, "teacher_proposals", teacher_proposals)),
        int(getattr(result.stats, "teacher_steps", teacher_steps)),
        branch_steps, getattr(result, "metadata", {}) or getattr(result.stats, "metadata", {}),
        task_id, reset_id, episode_id,
        stats_scored if stats_scored is not None else result.stats.steps,
        cloned_steps,
        float(getattr(result.stats, "latency_seconds", 0.0) or 0.0),
        float(getattr(result.stats, "policy_latency_seconds", 0.0) or 0.0),
        float(getattr(result.stats, "branch_latency_seconds", 0.0) or 0.0),
    )


def evaluate_suite(
    cases: Sequence[PolicyCase],
    coordinator_factory: Callable[[PolicyCase], TransactionalCoordinator],
    policy_factory: Callable[[PolicyCase], Any],
    *,
    max_steps: int = 1200,
    expected_cases: Sequence[PolicyCase] = DEFAULT_CASES,
    allow_incomplete: bool = False,
) -> tuple[EvaluationRow, ...]:
    if max_steps < 1200:
        raise ContractError("the primary suite horizon must be at least 1200")
    cases = validate_case_set(cases, expected_cases=expected_cases, allow_incomplete=allow_incomplete)
    rows: list[EvaluationRow] = []
    for case in cases:
        policy = policy_factory(case)
        _validate_factory_policy(case, policy)
        result = coordinator_factory(case).run(policy, max_steps=max_steps)
        rows.append(row_from_result(case, result))
    return tuple(rows)


def evaluate_paired_suite(
    cases: Sequence[PolicyCase],
    identities: Sequence[Mapping[str, Any]],
    coordinator_factory: Callable[[PolicyCase, Mapping[str, Any]], TransactionalCoordinator],
    policy_factory: Callable[[PolicyCase, Mapping[str, Any]], Any],
    *,
    max_steps: int = 1200,
    expected_cases: Sequence[PolicyCase] = DEFAULT_CASES,
    allow_incomplete: bool = False,
) -> tuple[EvaluationRow, ...]:
    """Evaluate every case on every supplied reset identity.

    The factory is responsible for constructing a fresh environment at the
    requested reset.  This orchestration layer only guarantees that each case
    receives the same identity and that both horizon checkpoints are derived
    from its single 1,200-step rollout.
    """

    if max_steps < 1200:
        raise ContractError("the primary suite horizon must be at least 1200")
    if not identities:
        raise ContractError("paired evaluation requires at least one reset identity")
    cases = validate_case_set(cases, expected_cases=expected_cases, allow_incomplete=allow_incomplete)
    rows: list[EvaluationRow] = []
    seen: set[tuple[str, str, str]] = set()
    for identity in identities:
        try:
            task_id = identity["task_id"]
            reset_id = str(identity["reset_id"])
        except (KeyError, TypeError) as exc:
            raise ContractError("paired identity requires task_id and reset_id") from exc
        episode_id = str(identity.get("episode_id", f"task-{task_id}-reset-{reset_id}"))
        key = (str(task_id), reset_id, episode_id)
        if key in seen:
            raise ContractError(f"duplicate paired reset identity {key}")
        seen.add(key)
        if "split_manifest_sha256" in identity and not identity["split_manifest_sha256"]:
            raise ContractError(f"paired identity {key} has empty split manifest provenance")
        for case in cases:
            policy = policy_factory(case, identity)
            _validate_factory_policy(case, policy)
            result = coordinator_factory(case, identity).run(policy, max_steps=max_steps)
            rows.append(row_from_result(case, result, task_id=task_id, reset_id=reset_id, episode_id=episode_id))
    return tuple(rows)


def rank_rows(rows: Sequence[EvaluationRow]) -> tuple[EvaluationRow, ...]:
    """Rank by predeclared primary success, then fewer steps and cost."""
    return tuple(sorted(rows, key=lambda row: (-int(row.success_1200), row.steps,
                                                row.teacher_steps, row.branch_steps, row.case)))
