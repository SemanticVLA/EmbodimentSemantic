"""Small deterministic branch runner for policy selection tests.

Branches are evaluated from a caller-supplied snapshot/restore environment;
this module never assumes a simulator implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from .contracts import ACTION_DIM, ActionProposal, ContractError, validate_action


class BranchEnvironment(Protocol):
    """A dedicated branch sandbox owned exclusively by ``BranchRunner``.

    It must not be the live environment held by a
    :class:`TransactionalCoordinator`; callers should provide a clone or a
    coordinator-owned sandbox adapter.  This keeps branch probing from
    bypassing the episode transaction owner.
    """
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def step(self, action: Sequence[float]) -> Any: ...


class StatefulPolicy(Protocol):
    """Optional policy hooks used to isolate branch-local state."""

    def snapshot_state(self) -> Any: ...
    def restore_state(self, snapshot: Any) -> None: ...


@dataclass(frozen=True)
class BranchResult:
    mask: int
    actions: tuple[tuple[float, ...], ...]
    raw_results: tuple[Any, ...] = ()
    score: float = 0.0
    terminated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BranchRunner:
    """Execute up to eight masks in a private, restorable branch sandbox."""

    def __init__(self, environment: BranchEnvironment, *, masks: Sequence[int] = range(8),
                 horizon: int = 20, action_selector: Callable[[int, int, Any], Sequence[float]] | None = None,
                 proposal_fn: Callable[[int, int, Any], Any] | None = None,
                 policy: StatefulPolicy | None = None,
                 policies: Sequence[StatefulPolicy] | None = None,
                 rng_snapshot: Callable[[], Any] | None = None,
                 rng_restore: Callable[[Any], None] | None = None,
                 composite_snapshot: Callable[[], Any] | None = None,
                 composite_restore: Callable[[Any], None] | None = None,
                 require_state_isolation: bool = False,
                 outcome_fn: Callable[[int, Sequence[Sequence[float]], Sequence[Any]], Mapping[str, Any]] | None = None) -> None:
        if horizon <= 0:
            raise ContractError("branch horizon must be positive")
        self.environment = environment
        self.masks = tuple(int(mask) for mask in masks)
        if any(mask < 0 or mask >= 8 for mask in self.masks) or len(set(self.masks)) != len(self.masks):
            raise ContractError("branch masks must be unique values in [0, 7]")
        self.horizon = horizon
        self.action_selector = action_selector
        self.proposal_fn = proposal_fn
        if (composite_snapshot is None) != (composite_restore is None):
            raise ContractError("composite_snapshot and composite_restore must be provided as a pair")
        self.composite_snapshot = composite_snapshot
        self.composite_restore = composite_restore
        # ``policy`` is the original single-policy spelling.  ``policies`` is
        # the Minimal-Runtime spelling: VLA, Arrow, and Minimal state are all
        # restored before every branch.  Keep one canonical ordered tuple so
        # duplicate references do not get restored twice.
        supplied = list(policies or ())
        if policy is not None:
            supplied.insert(0, policy)
        unique: list[StatefulPolicy] = []
        for component in supplied:
            if not any(component is item for item in unique):
                unique.append(component)
        self.policies = tuple(unique)
        self.policy = policy
        self.rng_snapshot = rng_snapshot
        self.rng_restore = rng_restore
        for component in self.policies:
            if (not callable(getattr(component, "snapshot_state", None)) or
                    not callable(getattr(component, "restore_state", None))):
                raise ContractError("branch policies must provide snapshot_state/restore_state as a pair")
        if (rng_snapshot is None) != (rng_restore is None):
            raise ContractError("rng_snapshot and rng_restore must be provided as a pair")
        if require_state_isolation:
            if not self.policies:
                raise ContractError("Minimal branching requires policy state isolation")
            for component in self.policies:
                if (not callable(getattr(component, "snapshot_state", None)) or
                        not callable(getattr(component, "restore_state", None))):
                    raise ContractError("every branch policy must provide snapshot_state/restore_state")
            if rng_snapshot is None:
                raise ContractError("Minimal branching requires RNG snapshot/restore")
        self.require_state_isolation = bool(require_state_isolation)
        self.outcome_fn = outcome_fn
        self.cloned_steps = 0

    @property
    def is_real(self) -> bool:
        """Whether this runner owns a restorable sandbox and executes branches."""
        return callable(getattr(self.environment, "step", None)) and (
            self.composite_snapshot is not None or callable(getattr(self.environment, "snapshot", None))
        )

    def run_all(self) -> tuple[BranchResult, ...]:
        self.cloned_steps = 0
        if self.composite_snapshot is not None:
            baseline = copy.deepcopy(self.composite_snapshot())
            if baseline is None:
                raise ContractError("composite branch snapshot must return a state")
            policy_baseline = ()
            rng_baseline = None
        else:
            baseline = copy.deepcopy(self.environment.snapshot())
            if baseline is None:
                raise ContractError("branch environment snapshot() must return a state")
            policy_baseline = tuple(copy.deepcopy(component.snapshot_state()) for component in self.policies)
            rng_baseline = copy.deepcopy(self.rng_snapshot()) if self.rng_snapshot is not None else None
        results: list[BranchResult] = []
        try:
            for mask in self.masks:
                if self.composite_restore is not None:
                    self.composite_restore(copy.deepcopy(baseline))
                else:
                    self.environment.restore(copy.deepcopy(baseline))
                    for component, state in zip(self.policies, policy_baseline):
                        component.restore_state(copy.deepcopy(state))
                    if self.rng_restore is not None:
                        self.rng_restore(copy.deepcopy(rng_baseline))
                actions: list[tuple[float, ...]] = []
                raws: list[Any] = []
                score = 0.0
                terminated = False
                started = time.perf_counter()
                for index in range(self.horizon):
                    if self.proposal_fn is not None:
                        proposal = self.proposal_fn(mask, index, baseline)
                        if isinstance(proposal, ActionProposal):
                            action = proposal.action
                        elif isinstance(proposal, Mapping) and "action" in proposal:
                            action = proposal["action"]
                        else:
                            action = proposal
                    elif self.action_selector is None:
                        action = (0.0,) * ACTION_DIM
                    else:
                        action = self.action_selector(mask, index, baseline)
                    action = validate_action(action)
                    raw = self.environment.step(action)
                    self.cloned_steps += 1
                    actions.append(action)
                    raws.append(raw)
                    if isinstance(raw, tuple) and len(raw) >= 3:
                        score += float(raw[1] or 0.0)
                        terminated = bool(raw[2]) or (len(raw) >= 4 and bool(raw[3]))
                    elif isinstance(raw, dict):
                        score += float(raw.get("reward", 0.0) or 0.0)
                        terminated = bool(raw.get("done", False) or raw.get("terminated", False))
                    if terminated:
                        break
                outcome_metadata = dict(self.outcome_fn(mask, actions, raws)) if self.outcome_fn is not None else {}
                results.append(BranchResult(
                    mask, tuple(actions), tuple(raws), score, terminated,
                    metadata={
                        **outcome_metadata,
                        "cloned_steps": len(actions),
                        "fresh_proposals": len(actions),
                        "fresh_actions": len(actions),
                        "mask_hold_steps": len(actions),
                        "branch_latency_seconds": time.perf_counter() - started,
                        "mask": mask,
                    },
                ))
        finally:
            if self.composite_restore is not None:
                self.composite_restore(copy.deepcopy(baseline))
            else:
                self.environment.restore(copy.deepcopy(baseline))
                for component, state in zip(self.policies, policy_baseline):
                    component.restore_state(copy.deepcopy(state))
                if self.rng_restore is not None:
                    self.rng_restore(copy.deepcopy(rng_baseline))
        if not results:
            raise ContractError("no branch masks configured")
        return tuple(results)

    def run(self, *, selector: Callable[[Sequence[BranchResult]], int] | None = None) -> BranchResult:
        results = self.run_all()
        selected_mask = selector(tuple(results)) if selector else min(results, key=lambda item: (-item.score, item.mask)).mask
        for result in results:
            if result.mask == selected_mask:
                return result
        raise ContractError(f"selector chose unavailable branch mask {selected_mask}")


MinimalBranchExecutor = BranchRunner

__all__ = ["BranchEnvironment", "BranchResult", "BranchRunner", "MinimalBranchExecutor", "StatefulPolicy"]
