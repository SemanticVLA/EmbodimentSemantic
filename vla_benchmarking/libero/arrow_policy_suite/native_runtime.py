"""Dependency-injected native canary runtime.

This module is deliberately separate from the policy-suite runtime.  It is a
small production seam for checking that a native VLA and an interruptible
teacher observe the same frame, that exactly one environment step is issued,
and that a failed transaction can be rolled back.  Factories own dependency
loading; this module never imports LIBERO, Torch, LeRobot, or Arrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import (
    ActionProposal,
    ContractError,
    ObservationFrame,
    assert_student_observation,
    validate_action,
)


class NativeEnvironment(Protocol):
    def observe(self) -> Mapping[str, Any]: ...
    def step(self, action: Sequence[float]) -> Any: ...
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...


class NativeVLA(Protocol):
    def propose(self, frame: ObservationFrame) -> ActionProposal: ...
    def commit(self, record: Any) -> None: ...


class NativeTeacher(Protocol):
    def propose(self, frame: ObservationFrame) -> ActionProposal | None: ...
    def commit(self, record: Any) -> None: ...


class NativeEnvironmentFactory(Protocol):
    def __call__(self) -> NativeEnvironment: ...


class NativeVLAFactory(Protocol):
    def __call__(self) -> NativeVLA: ...


class NativeTeacherFactory(Protocol):
    def __call__(self) -> NativeTeacher: ...


class NativeCanaryError(ContractError):
    """Raised when the native canary cannot complete a safe transaction."""

    def __init__(self, message: str, *, receipt: "NativeCanaryReceipt") -> None:
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class NativeStepReceipt:
    timestep: int
    frame_digest: str
    vla_proposal: ActionProposal
    teacher_proposal: ActionProposal
    executed_action: tuple[float, ...]
    raw_result: Any
    next_frame_digest: str
    success: bool
    terminal: bool
    proposal_state_unchanged: bool | None = None


@dataclass(frozen=True)
class NativeCanaryReceipt:
    status: str
    steps: tuple[NativeStepReceipt, ...] = ()
    environment_steps: int = 0
    success: bool = False
    terminal: bool = False
    rollback_count: int = 0
    termination_reason: str = ""
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def steps_executed(self) -> int:
        return self.environment_steps

    @property
    def terminated(self) -> bool:
        return self.terminal

    @property
    def rolled_back(self) -> bool:
        return self.rollback_count > 0

    @property
    def proposal_state_checks(self) -> tuple[bool | None, ...]:
        """Per-step result of the pre-action proposal side-effect check."""
        return tuple(step.proposal_state_unchanged for step in self.steps)


def _build(component_or_factory: Any, *, name: str) -> Any:
    if component_or_factory is None:
        raise NativeCanaryError(f"{name} is required", receipt=NativeCanaryReceipt("failed"))
    # Adapters are objects; factories are zero-argument callables.  Prefer an
    # explicitly supplied component when it already has the required hook.
    if name == "environment" and callable(getattr(component_or_factory, "observe", None)):
        return component_or_factory
    if name in {"vla", "teacher", "policy"} and callable(getattr(component_or_factory, "propose", None)):
        return component_or_factory
    # ``policy`` is an optional already-constructed component, not a factory:
    # callable policy objects (and plain stateless policy functions) receive
    # observations during the canary and must never be invoked at construction.
    # This also keeps state-hook-bearing callable objects available for the
    # fail-closed rollback check below.
    if name == "policy":
        return component_or_factory
    if callable(component_or_factory):
        return component_or_factory()
    raise NativeCanaryError(f"{name} must be an adapter or zero-argument factory", receipt=NativeCanaryReceipt("failed"))


def _state_hooks(component: Any, *, name: str, required: bool) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
    """Resolve canonical mutable-state hooks, preferring the explicit names."""
    completeness = getattr(component, "rollback_complete", None)
    if completeness is False:
        raise ContractError(f"{name} has incomplete mutable component rollback state")
    preferred_snapshot = getattr(component, "snapshot_state", None)
    preferred_restore = getattr(component, "restore_state", None)
    fallback_snapshot = getattr(component, "snapshot", None)
    fallback_restore = getattr(component, "restore", None)
    if (callable(preferred_snapshot) != callable(preferred_restore)):
        raise ContractError(f"{name} exposes only one of snapshot_state()/restore_state()")
    if callable(preferred_snapshot):
        return preferred_snapshot, preferred_restore
    if (callable(fallback_snapshot) != callable(fallback_restore)):
        raise ContractError(f"{name} exposes only one of snapshot()/restore()")
    if callable(fallback_snapshot):
        return fallback_snapshot, fallback_restore
    if required:
        raise ContractError(f"{name} requires snapshot_state()/restore_state() or snapshot()/restore()")
    return None


def _capture_component_state(
    component: Any,
    hooks: tuple[Callable[[], Any], Callable[[Any], None]] | None,
    *,
    name: str,
) -> tuple[Any, Callable[[Any], None]] | None:
    if hooks is None:
        return None
    try:
        return copy.deepcopy(hooks[0]()), hooks[1]
    except BaseException as exc:
        raise ContractError(f"{name} state snapshot failed") from exc


def _capture_hook_value(hook: Callable[[], Any], *, name: str) -> Any:
    try:
        return copy.deepcopy(hook())
    except BaseException as exc:
        raise ContractError(f"{name} state snapshot failed") from exc


def _states_equal(left: Any, right: Any, *, name: str) -> bool:
    try:
        equal = left == right
        if isinstance(equal, bool):
            return equal
        scalar = getattr(equal, "item", None)
        if callable(scalar):
            value = scalar()
            if isinstance(value, bool):
                return value
    except BaseException:
        pass
    raise ContractError(f"{name} rollback state cannot be compared")


def _restore_component_states(states: Sequence[tuple[Any, Callable[[Any], None]] | None]) -> BaseException | None:
    first_error: BaseException | None = None
    for state in states:
        if state is None:
            continue
        payload, restore = state
        try:
            restore(payload)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _check_observation(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("native environment observation must be a mapping")
    # This check runs before constructing a frame.  It rejects privileged raw
    # fields instead of relying on a downstream policy to ignore them.
    assert_student_observation(value)
    return value


def _proposal(value: Any, *, frame: ObservationFrame, name: str) -> ActionProposal:
    if not isinstance(value, ActionProposal):
        raise ContractError(f"{name} must return an ActionProposal")
    if value.timestep != frame.timestep:
        raise ContractError(f"{name} proposal is stale for timestep {frame.timestep}")
    if value.observation_digest is None:
        raise ContractError(f"{name} proposal is missing observation_digest")
    if value.observation_digest != frame.digest:
        raise ContractError(f"{name} proposal observation_digest does not match the shared frame")
    validate_action(value.action)
    return value


def _proposal_state_check(
    environment: Any,
    *,
    snapshot_hook: Callable[[], Any],
    before_snapshot: Any,
    frame: ObservationFrame,
) -> bool | None:
    """Check that proposal calls did not advance the environment.

    Snapshot equality is preferred because it does not require a second raw
    observation.  If an opaque snapshot cannot be compared, compare the
    canonical observation digest.  ``None`` means neither hook exposed a
    meaningful comparison; callers record that uncertainty rather than
    claiming the check passed.
    """
    if before_snapshot is None:
        return None
    try:
        after_snapshot = _capture_hook_value(snapshot_hook, name="environment")
        return _states_equal(before_snapshot, after_snapshot, name="environment")
    except ContractError:
        return None
    except BaseException:
        pass
    try:
        observation = environment.observe()
        if not isinstance(observation, Mapping):
            return None
        after_frame = ObservationFrame(
            observation, timestep=frame.timestep, episode_id=frame.episode_id,
            metadata=frame.metadata, provenance=frame.provenance,
        )
        return after_frame.digest == frame.digest
    except BaseException:
        return None


def _result_flags(
    result: Any,
    *,
    success_fn: Callable[[Any], bool] | None,
    terminal_fn: Callable[[Any], bool] | None,
) -> tuple[bool, bool, str]:
    if success_fn is not None:
        success = bool(success_fn(result))
    else:
        success = False
        if isinstance(result, Mapping):
            success = any(bool(result.get(key)) for key in ("success", "task_success", "is_success"))
        elif isinstance(result, tuple) and len(result) in (4, 5):
            info = result[-1] if isinstance(result[-1], Mapping) else {}
            success = any(bool(info.get(key)) for key in ("success", "task_success", "is_success"))

    if terminal_fn is not None:
        terminal = bool(terminal_fn(result))
    elif isinstance(result, Mapping):
        terminal = bool(result.get("terminal", result.get("done", result.get("terminated", False))))
        terminal = terminal or bool(result.get("truncated", False))
    elif isinstance(result, tuple) and len(result) == 5:
        terminal = bool(result[2]) or bool(result[3])
    elif isinstance(result, tuple) and len(result) == 4:
        terminal = bool(result[2])
    else:
        terminal = False
    reason = "success" if success else ("terminal" if terminal else "")
    return success, terminal, reason


def _next_observation(result: Any, environment: Any) -> Mapping[str, Any]:
    # Prefer the environment's authoritative reader.  This avoids treating a
    # stale observation embedded in a custom result as the current state.
    observe = getattr(environment, "observe", None)
    if not callable(observe):
        raise ContractError("native environment must expose observe() after step")
    return _check_observation(observe())


def _receipt_with_error(receipt: NativeCanaryReceipt, exc: BaseException, rollback_count: int) -> NativeCanaryReceipt:
    return NativeCanaryReceipt(
        status="failed",
        steps=receipt.steps,
        environment_steps=receipt.environment_steps,
        success=receipt.success,
        terminal=receipt.terminal,
        rollback_count=rollback_count,
        termination_reason="error",
        error=f"{type(exc).__name__}: {exc}",
        metadata=receipt.metadata,
    )


def _run_native_canary_components(
    environment: NativeEnvironment,
    vla: NativeVLA,
    teacher: NativeTeacher,
    *,
    max_steps: int = 1,
    episode_id: str | None = None,
    action_selector: Callable[[ActionProposal, ActionProposal], Sequence[float]] | None = None,
    success_fn: Callable[[Any], bool] | None = None,
    terminal_fn: Callable[[Any], bool] | None = None,
    metadata: Mapping[str, Any] | None = None,
    policy: Any | None = None,
) -> NativeCanaryReceipt:
    """Run a bounded same-frame VLA/teacher canary.

    The environment must expose snapshot/restore so every failed proposal,
    step, observation, or commit can be rolled back.  ``action_selector``
    chooses the one action sent to the environment; by default the VLA action
    is used.
    """

    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    max_steps = int(max_steps)
    try:
        environment_hooks = _state_hooks(environment, name="environment", required=True)
    except ContractError as exc:
        raise NativeCanaryError(str(exc), receipt=NativeCanaryReceipt("failed")) from exc
    if environment_hooks is None:  # pragma: no cover - required=True guard
        raise NativeCanaryError(
            "native canary requires environment rollback hooks",
            receipt=NativeCanaryReceipt("failed"),
        )
    components = [(vla, "VLA"), (teacher, "teacher")]
    if policy is not None:
        components.append((policy, "policy"))
    try:
        component_hooks = [(_state_hooks(component, name=name, required=True), name) for component, name in components]
    except ContractError as exc:
        raise NativeCanaryError(str(exc), receipt=NativeCanaryReceipt("failed")) from exc
    for component, name in components:
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()

    receipt = NativeCanaryReceipt("running", metadata=dict(metadata or {}))
    records: list[NativeStepReceipt] = []
    rollback_count = 0
    final_success = final_terminal = False
    termination_reason = "horizon"
    for timestep in range(max_steps):
        frame_snapshot: Any = None
        component_states: list[tuple[Any, Callable[[Any], None]] | None] = []
        try:
            frame_snapshot = _capture_hook_value(environment_hooks[0], name="environment")
            component_states = []
            for (component, name), (hooks, _hook_name) in zip(components, component_hooks):
                component_states.append(_capture_component_state(component, hooks, name=name))
            before_payload = _check_observation(environment.observe())
            frame = ObservationFrame(before_payload, timestep=timestep, episode_id=episode_id, metadata=metadata or {})
            base = _proposal(vla.propose(frame), frame=frame, name="VLA")
            teacher_proposal = _proposal(teacher.propose(frame), frame=frame, name="teacher")
            # Both methods receive this exact frame object.  The timestep check
            # above catches stale proposals even when a producer ignores it.
            proposal_state_unchanged = _proposal_state_check(
                environment,
                snapshot_hook=environment_hooks[0],
                before_snapshot=frame_snapshot,
                frame=frame,
            )
            if proposal_state_unchanged is False:
                raise ContractError("proposal calls advanced environment state before env.step")
            if action_selector is None:
                action = base.action
            else:
                action = validate_action(action_selector(base, teacher_proposal))
            action = validate_action(action)
            raw_result = environment.step(action)
            environment_steps = timestep + 1
            next_payload = _next_observation(raw_result, environment)
            next_frame = ObservationFrame(next_payload, timestep=timestep + 1, episode_id=episode_id, metadata=metadata or {})
            success, terminal, reason = _result_flags(
                raw_result, success_fn=success_fn, terminal_fn=terminal_fn
            )
            vla_commit = getattr(vla, "commit", None)
            teacher_commit = getattr(teacher, "commit", None)
            commit_payload = type(
                "NativeCommit",
                (),
                {"frame": frame, "base": base, "teacher": teacher_proposal,
                 "next_frame": next_frame, "result": raw_result,
                 "success": success, "terminal": terminal},
            )()
            if callable(vla_commit):
                vla_commit(commit_payload)
            if callable(teacher_commit):
                teacher_commit(commit_payload)
            if policy is not None:
                policy_commit = getattr(policy, "commit", None)
                if callable(policy_commit):
                    policy_commit(commit_payload)
            post_frame_snapshot = _capture_hook_value(environment_hooks[0], name="environment")
            post_component_states: list[tuple[Any, Callable[[Any], None]] | None] = []
            for (component, name), (hooks, _hook_name) in zip(components, component_hooks):
                post_component_states.append(_capture_component_state(component, hooks, name=name))

            # Probe rollback/replay without issuing another environment step.
            pre_restore_error = _restore_component_states(component_states)
            environment_hooks[1](frame_snapshot)
            if pre_restore_error is not None:
                raise ContractError("component restore failed during replay probe") from pre_restore_error
            replay_environment_snapshot = _capture_hook_value(
                environment_hooks[0], name="environment replay"
            )
            if not _states_equal(
                frame_snapshot, replay_environment_snapshot, name="environment"
            ):
                raise ContractError("environment restore did not reproduce the pre-proposal state")
            replay_payload = _check_observation(environment.observe())
            replay_frame = ObservationFrame(
                replay_payload, timestep=timestep, episode_id=episode_id,
                metadata=metadata or {},
            )
            if replay_frame.digest != frame.digest:
                raise ContractError("environment restore changed the replay observation frame")
            for index, state in enumerate(component_states):
                if state is not None:
                    current = _capture_component_state(
                        components[index][0], component_hooks[index][0], name=components[index][1]
                    )
                    if current is None or not _states_equal(state[0], current[0], name=components[index][1]):
                        raise ContractError(f"{components[index][1]} restore was a no-op or incomplete")
            replay_base = _proposal(vla.propose(replay_frame), frame=replay_frame, name="VLA replay")
            replay_teacher = _proposal(teacher.propose(replay_frame), frame=replay_frame, name="teacher replay")
            if (
                replay_base.action != base.action
                or replay_base.observation_digest != base.observation_digest
                or replay_teacher.action != teacher_proposal.action
                or replay_teacher.observation_digest != teacher_proposal.observation_digest
            ):
                raise ContractError("proposal replay did not reproduce the committed proposals")

            post_restore_error = _restore_component_states(post_component_states)
            environment_hooks[1](post_frame_snapshot)
            if post_restore_error is not None:
                raise ContractError("component restore failed after replay probe") from post_restore_error
            restored_post_snapshot = _capture_hook_value(
                environment_hooks[0], name="environment post-commit replay"
            )
            if not _states_equal(
                post_frame_snapshot, restored_post_snapshot, name="environment post-commit"
            ):
                raise ContractError("post-commit environment restore did not reproduce state")
            post_observation = _check_observation(environment.observe())
            post_frame = ObservationFrame(
                post_observation, timestep=timestep + 1, episode_id=episode_id,
                metadata=metadata or {},
            )
            if post_frame.digest != next_frame.digest:
                raise ContractError("post-commit restore did not reproduce the post-step state")
            for index, state in enumerate(post_component_states):
                if state is not None:
                    current = _capture_component_state(
                        components[index][0], component_hooks[index][0], name=components[index][1]
                    )
                    if current is None or not _states_equal(state[0], current[0], name=components[index][1]):
                        raise ContractError(f"{components[index][1]} post-commit restore was a no-op")
            records.append(
                NativeStepReceipt(
                    timestep, frame.digest, base, teacher_proposal, tuple(action), raw_result,
                    next_frame.digest, success, terminal, proposal_state_unchanged,
                )
            )
            final_success, final_terminal = success, terminal
            if success or terminal:
                termination_reason = reason
                break
        except BaseException as exc:
            try:
                if frame_snapshot is None:
                    raise ContractError("no rollback snapshot was captured")
                component_restore_error = _restore_component_states(component_states)
                environment_hooks[1](frame_snapshot)
                if component_restore_error is not None:
                    raise ContractError("one or more component rollback hooks failed") from component_restore_error
                rollback_count += 1
            except BaseException as rollback_exc:
                exc = NativeCanaryError(
                    f"native canary failed and rollback failed: {rollback_exc}",
                    receipt=_receipt_with_error(receipt, exc, rollback_count),
                )
            failed = _receipt_with_error(
                NativeCanaryReceipt(
                    "running", tuple(records), timestep, final_success, final_terminal,
                    rollback_count, termination_reason, metadata=dict(metadata or {}),
                ),
                exc,
                rollback_count,
            )
            raise NativeCanaryError(failed.error or "native canary failed", receipt=failed) from exc

    return NativeCanaryReceipt(
        status="completed",
        steps=tuple(records),
        environment_steps=len(records),
        success=final_success,
        terminal=final_terminal,
        rollback_count=rollback_count,
        termination_reason=termination_reason,
        metadata=dict(metadata or {}),
    )


def _close_components_once(components: Sequence[Any]) -> BaseException | None:
    """Close each constructed component at most once, preserving first error."""
    seen: set[int] = set()
    first_error: BaseException | None = None
    for component in components:
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        close = getattr(component, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


def run_native_canary(
    environment_factory: NativeEnvironmentFactory | NativeEnvironment,
    vla_factory: NativeVLAFactory | NativeVLA,
    teacher_factory: NativeTeacherFactory | NativeTeacher,
    *,
    max_steps: int = 1,
    episode_id: str | None = None,
    action_selector: Callable[[ActionProposal, ActionProposal], Sequence[float]] | None = None,
    success_fn: Callable[[Any], bool] | None = None,
    terminal_fn: Callable[[Any], bool] | None = None,
    metadata: Mapping[str, Any] | None = None,
    policy: Any | None = None,
) -> NativeCanaryReceipt:
    """Construct native components, run one bounded canary, and always close them."""
    components: list[Any] = []
    try:
        environment = _build(environment_factory, name="environment")
        components.append(environment)
        vla = _build(vla_factory, name="vla")
        components.append(vla)
        teacher = _build(teacher_factory, name="teacher")
        components.append(teacher)
        policy_component = None
        if policy is not None:
            policy_component = _build(policy, name="policy")
            components.append(policy_component)
        receipt = _run_native_canary_components(
            environment, vla, teacher,
            max_steps=max_steps, episode_id=episode_id,
            action_selector=action_selector, success_fn=success_fn,
            terminal_fn=terminal_fn, metadata=metadata, policy=policy_component,
        )
    except BaseException as exc:
        cleanup_error = _close_components_once(components)
        if cleanup_error is not None and isinstance(exc, NativeCanaryError):
            receipt = replace(
                exc.receipt,
                error=(exc.receipt.error or str(exc)) + f"; cleanup failed: {cleanup_error}",
            )
            raise NativeCanaryError(receipt.error or "native canary failed", receipt=receipt) from exc
        raise
    cleanup_error = _close_components_once(components)
    if cleanup_error is not None:
        failed = NativeCanaryReceipt(
            status="failed", steps=receipt.steps, environment_steps=receipt.environment_steps,
            success=receipt.success, terminal=receipt.terminal,
            rollback_count=receipt.rollback_count, termination_reason="cleanup_error",
            error=f"component cleanup failed: {cleanup_error}", metadata=receipt.metadata,
        )
        raise NativeCanaryError(failed.error or "component cleanup failed", receipt=failed) from cleanup_error
    return receipt


def run_native_host_canary(
    environment: Any,
    vla: Any,
    teacher: Any | None = None,
    *,
    max_steps: int = 3,
    reset_identity: Any | None = None,
    graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | None] | None = None,
    action_selector: Callable[..., Sequence[float]] | None = None,
    success_fn: Callable[[Any], bool] | None = None,
    terminal_fn: Callable[[Any], bool] | None = None,
    policy: Any | None = None,
    reset_environment: bool = False,
) -> NativeCanaryReceipt:
    """Run the real shared-host canary without receipt-only orchestration.

    Unlike the legacy ``run_native_canary`` replay probe, this path executes
    each bounded step exactly once and lets :class:`NativeHost` own proposal,
    arbitration, queue invalidation, commit, and rollback semantics.  It is
    intentionally a separate entry point so existing contract tests and
    historical one-step receipts retain their schema.
    """
    from .native_host import NativeHost

    components = [environment, vla, teacher, policy]
    seen: set[int] = set()
    try:
        host = NativeHost(
            environment, vla, teacher, policy=policy,
            reset_identity=reset_identity, graph_context_fn=graph_context_fn,
            action_selector=action_selector, success_fn=success_fn,
            terminal_fn=terminal_fn,
        )
        records = host.run(max_steps=max_steps, reset_environment=reset_environment)
        receipts = tuple(
            NativeStepReceipt(
                step.frame.timestep, step.frame.digest, step.base, step.teacher,
                step.action, step.result, step.next_frame.digest, step.success,
                step.terminal, step.proposal_state_unchanged,
            )
            for step in records
        )
        terminal = bool(records[-1].terminal) if records else False
        success = bool(records[-1].success) if records else False
        return NativeCanaryReceipt(
            status="completed", steps=receipts, environment_steps=len(records),
            success=success, terminal=terminal,
            termination_reason="success" if success else ("terminal" if terminal else "horizon"),
            metadata={"runtime": "native_host", "teacher_unavailable_steps": sum(int(not step.teacher_status.available) for step in records)},
        )
    except BaseException as exc:
        receipt = NativeCanaryReceipt(status="failed", termination_reason="error", error=f"{type(exc).__name__}: {exc}", metadata={"runtime": "native_host"})
        raise NativeCanaryError(receipt.error or "native host canary failed", receipt=receipt) from exc
    finally:
        for component in components:
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()


__all__ = [
    "NativeCanaryError", "NativeCanaryReceipt", "NativeEnvironment", "NativeEnvironmentFactory",
    "NativeStepReceipt", "NativeTeacher", "NativeTeacherFactory", "NativeVLA", "NativeVLAFactory",
    "run_native_canary",
    "run_native_host_canary",
]
