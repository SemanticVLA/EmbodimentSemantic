"""Small, dependency-free factory seam for native Arrow policy runs.

The suite deliberately does not guess how a user's LIBERO checkpoint or Arrow
perception stack is loaded.  A Legion/native launcher injects those objects
and calls :func:`build_native_host`.  This module owns the *policy selection*
contract, so every policy uses the same-frame proposals and the same
``NativeHost`` step owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .contracts import ActionProposal, ContractError, ObservationFrame, PolicyDecision
from .native_host import NativeHost
from .policies import make_policy


class _ControlPolicy:
    """A named reference control with no mutable state."""

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id

    def snapshot_state(self) -> tuple[str]:
        return (self.policy_id,)

    def restore_state(self, state: Any) -> None:
        if tuple(state) != (self.policy_id,):
            raise ContractError("control policy snapshot belongs to another policy")

    snapshot = snapshot_state
    restore = restore_state

    def commit(self, _record: Any) -> None:
        return None


def _validated_decision(decision: Any, frame: ObservationFrame) -> PolicyDecision:
    if not isinstance(decision, PolicyDecision):
        raise ContractError("policy selector must return a PolicyDecision")
    proposal_digest = decision.observation_digest
    if proposal_digest is not None and proposal_digest != frame.digest:
        raise ContractError("policy decision is stale for the current observation")
    return decision


def action_selector_for(policy_id: str, policy: Any | None = None) -> Callable[..., Any]:
    """Return the arbitration callback used by :class:`NativeHost`.

    Reference controls intentionally return raw action tuples.  Policy-backed
    selectors return the complete validated ``PolicyDecision`` so collection
    retains the policy metadata and provenance alongside the executed action.
    """

    name = str(policy_id)
    if name == "frozen_base":
        return lambda base, _teacher, _frame=None: base.action
    if name == "teacher_only":
        return lambda base, teacher, _frame=None: base.action if teacher is None else teacher.action
    if policy is None or not callable(getattr(policy, "decide", None)):
        raise ContractError(f"policy {name!r} requires a decide(frame, base, teacher) hook")

    def select(base: ActionProposal, teacher: ActionProposal | None, frame: ObservationFrame | None = None) -> Any:
        if frame is None:
            raise ContractError("policy arbitration requires the authoritative frame")
        return _validated_decision(policy.decide(frame, base, teacher), frame)

    return select


def build_policy(policy_id: str, *, hooks: Mapping[str, Callable[..., Any]] | None = None, **kwargs: Any) -> Any:
    """Build a named policy or one of the two mandatory reference controls."""

    name = str(policy_id)
    if name in {"frozen_base", "teacher_only"}:
        return _ControlPolicy(name)
    hook = hooks.get(name) if isinstance(hooks, Mapping) else None
    # Learned rows must carry an actual inference/training artifact hook.  A
    # missing hook must never silently become the frozen VLA baseline.
    if hook is None:
        if name == "arrow_apprentice":
            runner = kwargs.get("action_fn") or kwargs.get("learned_runner")
            if not callable(runner):
                raise ContractError("arrow_apprentice requires an action_fn/learned_runner hook")
            kwargs = {**kwargs, "action_fn": runner}
        elif name == "arrow_editor":
            runner = kwargs.get("residual_fn") or kwargs.get("residual_runner")
            if not callable(runner):
                raise ContractError("arrow_editor requires a residual_fn/residual_runner hook")
            kwargs = {**kwargs, "residual_fn": runner}
    else:
        if not callable(hook):
            raise ContractError(f"learned policy hook for {name!r} is not callable")
        return hook(**kwargs)
    # CLI/benchmark row names are explicit, while ``make_policy`` takes the
    # family plus a Minimal variant.  Keep the alias conversion here so a
    # launcher cannot accidentally evaluate the wrong Minimal row.
    if name == "arrow_minimal_runtime":
        name, kwargs = "arrow_minimal", {**kwargs, "variant": "runtime_oracle"}
    elif name == "arrow_minimal_learned":
        runner = kwargs.get("learned_fn") or kwargs.get("action_fn") or kwargs.get("residual_fn") or kwargs.get("learned_runner")
        if hook is None and not callable(runner):
            raise ContractError("arrow_minimal_learned requires a learned_fn/residual_fn hook")
        name, kwargs = "arrow_minimal", {**kwargs, "variant": "learned"}
        if callable(runner) and "learned_fn" not in kwargs and "residual_fn" not in kwargs:
            kwargs = {**kwargs, "learned_fn": runner}
    return make_policy(name, hooks=hooks, **kwargs)


@dataclass(frozen=True)
class NativeHostSpec:
    """Explicit dependency bundle supplied by a native/Legion launcher."""

    environment: Any
    vla: Any
    teacher: Any | None
    policy_id: str = "frozen_base"
    policy: Any | None = None
    reset_identity: Any | None = None
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | None] | None = None
    success_fn: Callable[[Any], bool] | None = None
    terminal_fn: Callable[[Any], bool] | None = None
    policy_kwargs: Mapping[str, Any] = ()


def build_native_host(spec: NativeHostSpec) -> NativeHost:
    """Construct a transactional host from injected native components."""

    if not isinstance(spec, NativeHostSpec):
        raise TypeError("build_native_host requires NativeHostSpec")
    policy = spec.policy
    if policy is None:
        policy = build_policy(spec.policy_id, **dict(spec.policy_kwargs))
    selector = action_selector_for(spec.policy_id, policy)
    return NativeHost(
        spec.environment,
        spec.vla,
        spec.teacher,
        policy=policy,
        reset_identity=spec.reset_identity,
        graph_context_fn=spec.graph_context_fn,
        action_selector=selector,
        success_fn=spec.success_fn,
        terminal_fn=spec.terminal_fn,
    )


__all__ = ["NativeHostSpec", "action_selector_for", "build_native_host", "build_policy"]
