"""Backend-neutral policy evaluation harness.

Native simulator adapters own environment reset/stepping.  The shared runner
owns only lifecycle, action-contract validation, result normalization, and
optional JSONL persistence.  This keeps LeRobot, OpenVLA-OFT, and Octo action
horizons independent while making their accounting comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import EvaluationCell
from .plan import validate_native_schedule, validate_plan
from .policy_adapter import (
    PolicyAdapter,
    PolicyMetadata,
    validate_adapter_metadata,
    validate_arrow_free_observation,
    validate_policy_action,
)

# All native-policy rollouts use the same episode budget unless a sealed plan
# carries an explicit ``episode_step_budget``.  This is deliberately expressed
# in environment steps rather than policy chunks so native action horizons do
# not change the evaluation exposure.
DEFAULT_EPISODE_STEP_BUDGET = 220


def resolve_episode_step_budget(
    plan: Mapping[str, Any], explicit: int | None = None
) -> int:
    """Resolve and validate the sealed per-episode environment-step budget.

    A plan-owned value wins over the shared default.  The nested locations are
    accepted for compatibility with manifests that group evaluation protocol
    fields, while an explicit function argument is allowed only when it agrees
    with the sealed plan value.
    """

    candidates: list[Any] = []
    if "episode_step_budget" in plan:
        candidates.append(plan["episode_step_budget"])
    for section_name in ("evaluation", "evaluation_contract", "protocol"):
        section = plan.get(section_name)
        if isinstance(section, Mapping) and "episode_step_budget" in section:
            candidates.append(section["episode_step_budget"])
    values: list[int] = []
    for candidate in candidates:
        if isinstance(candidate, bool):
            raise ValueError("episode_step_budget must be an integer greater than zero")
        try:
            value = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode_step_budget must be an integer greater than zero") from exc
        if value <= 0 or str(candidate).strip() != str(value):
            raise ValueError("episode_step_budget must be an integer greater than zero")
        values.append(value)
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError("conflicting episode_step_budget values in evaluation plan")
    plan_value = values[0] if values else None
    if explicit is not None:
        if isinstance(explicit, bool):
            raise ValueError("episode_step_budget must be an integer greater than zero")
        try:
            explicit_value = int(explicit)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode_step_budget must be an integer greater than zero") from exc
        if explicit_value <= 0 or str(explicit).strip() != str(explicit_value):
            raise ValueError("episode_step_budget must be an integer greater than zero")
        if plan_value is not None and explicit_value != plan_value:
            raise ValueError(
                "explicit episode_step_budget disagrees with the sealed evaluation plan"
            )
        return explicit_value
    return DEFAULT_EPISODE_STEP_BUDGET if plan_value is None else plan_value


@dataclass(frozen=True)
class EpisodeSpec:
    cell: EvaluationCell
    task_description: str


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    terminal: bool = True
    # Deprecated input fields are retained for callers, but run_policy_eval
    # derives and verifies them from ActionCountingAdapter.
    action_chunks: int | None = None
    executed_actions: int | None = None
    failure_category: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyEpisodeRecord:
    cell: EvaluationCell
    policy: Mapping[str, Any]
    success: bool
    terminal: bool
    action_chunks: int
    executed_actions: int
    failure_category: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    plan_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.cell.as_dict(),
            "policy": dict(self.policy),
            "success": bool(self.success),
            "terminal": bool(self.terminal),
            "action_chunks": int(self.action_chunks),
            "executed_actions": int(self.executed_actions),
            "failure_category": self.failure_category,
            "metadata": dict(self.metadata),
            "plan_sha256": self.plan_sha256,
        }


class ActionCountingAdapter:
    """Adapter view passed to rollout callbacks.

    It is the sole source of action accounting.  Simulator callbacks receive
    this wrapper, so they cannot claim executed chunks without actually
    calling the policy's validated ``act`` method.
    """

    def __init__(self, adapter: PolicyAdapter) -> None:
        self._adapter = adapter
        self.action_calls = 0
        self.executed_actions = 0
        self.predicted_actions = 0
        self._pending_actions = 0

    @property
    def metadata(self) -> PolicyMetadata:
        return self._adapter.metadata

    def reset(self, task_description: str, episode_seed: int) -> None:
        self._adapter.reset(task_description, int(episode_seed))

    def act(self, observation: Mapping[str, Any]):
        if self._pending_actions:
            raise RuntimeError(
                "previous native action chunk was not fully stepped; "
                "call record_environment_step or discard_pending_chunk"
            )
        validate_arrow_free_observation(observation)
        action = validate_policy_action(self._adapter.act(observation), self.metadata, observation=observation)
        self.action_calls += 1
        self.predicted_actions += int(action.shape[0])
        self._pending_actions = int(action.shape[0])
        return action

    def record_environment_step(self) -> None:
        """Record one action after the simulator accepted ``env.step``."""

        if self._pending_actions <= 0:
            raise RuntimeError("environment step was not preceded by a pending policy action")
        self._pending_actions -= 1
        self.executed_actions += 1

    def discard_pending_chunk(self) -> None:
        """Close a partially executed chunk after terminal environment state."""

        self._pending_actions = 0

    def finalize(self) -> None:
        if self._pending_actions:
            raise RuntimeError(
                f"rollout returned with {self._pending_actions} native actions not sent to env.step"
            )


def run_policy_eval(
    adapter: PolicyAdapter,
    plan: Mapping[str, Any],
    episodes: Iterable[EpisodeSpec],
    rollout: Callable[[ActionCountingAdapter, EpisodeSpec], EpisodeOutcome],
    *,
    output_jsonl: str | Path | None = None,
) -> list[PolicyEpisodeRecord]:
    """Evaluate episodes through a native rollout callback.

    ``rollout`` must call the supplied ActionCountingAdapter for every native
    policy query and apply each returned ``[K, 7]`` chunk to its simulator.
    Simulator stepping remains explicit in the callback; action counts are
    derived exclusively by this wrapper.
    """

    checked_plan = validate_plan(plan)
    validate_adapter_metadata(adapter, checked_plan)
    episode_list = list(episodes)
    if not episode_list:
        raise ValueError("episodes must not be empty")
    if any(not isinstance(episode, EpisodeSpec) for episode in episode_list):
        raise TypeError("episodes must contain EpisodeSpec values")
    validate_native_schedule(checked_plan, [episode.cell.as_dict() for episode in episode_list])
    plan_sha256 = str(checked_plan["sha256"])
    records: list[PolicyEpisodeRecord] = []
    handle = None
    if output_jsonl is not None:
        target = Path(output_jsonl)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("w", encoding="utf-8")
    try:
        for episode in episode_list:
            counted = ActionCountingAdapter(adapter)
            counted.reset(episode.task_description, episode.cell.seed)
            outcome = rollout(counted, episode)
            if not isinstance(outcome, EpisodeOutcome):
                raise TypeError("rollout must return EpisodeOutcome")
            counted.finalize()
            observed_chunks = counted.action_calls
            observed_actions = counted.executed_actions
            if outcome.success and observed_chunks == 0:
                raise RuntimeError("rollout returned success without calling adapter.act")
            if outcome.action_chunks is not None and int(outcome.action_chunks) != observed_chunks:
                raise ValueError("rollout action_chunks disagrees with policy action calls")
            if outcome.executed_actions is not None and int(outcome.executed_actions) != observed_actions:
                raise ValueError("rollout executed_actions disagrees with native action chunks")
            record = PolicyEpisodeRecord(
                cell=episode.cell,
                policy=adapter.metadata.as_dict(),
                success=outcome.success,
                terminal=outcome.terminal,
                action_chunks=observed_chunks,
                executed_actions=observed_actions,
                failure_category=outcome.failure_category,
                metadata=dict(outcome.metadata),
                plan_sha256=plan_sha256,
            )
            records.append(record)
            if handle is not None:
                handle.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
                handle.flush()
    finally:
        if handle is not None:
            handle.close()
    return records


def offline_rollout(adapter: PolicyAdapter, episode: EpisodeSpec, observations: Sequence[Mapping[str, Any]]) -> EpisodeOutcome:
    """Run adapter validation on a fixed observation sequence without a simulator."""

    chunks = 0
    executed = 0
    for observation in observations:
        validate_arrow_free_observation(observation)
        action = adapter.act(observation)
        chunks += 1
        executed += int(action.shape[0])
    return EpisodeOutcome(success=False, terminal=True, action_chunks=chunks, executed_actions=executed)


__all__ = [
    "ActionCountingAdapter", "EpisodeOutcome", "EpisodeSpec", "PolicyEpisodeRecord",
    "DEFAULT_EPISODE_STEP_BUDGET", "offline_rollout", "resolve_episode_step_budget", "run_policy_eval",
]
