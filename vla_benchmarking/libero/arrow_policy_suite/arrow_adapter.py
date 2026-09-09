"""Interruptible Arrow proposal/commit boundary.

The existing ``automatic_ttt`` Arrow bridge owns whole recovery episodes and is
therefore not a per-step teacher.  ``ArrowAdapter`` intentionally requires a
controller with explicit propose/commit hooks.  A caller that only has the
legacy recovery API receives a clear fail-closed error instead of silently
executing a nested episode inside ``TransactionalCoordinator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contracts import ActionProposal, ObservationFrame, StepRecord, ContractError, validate_action
from .interruptible_arrow import ArrowPerceptionUnavailable


def _proposal(action: Sequence[float], frame: ObservationFrame, producer: str, **metadata: Any) -> ActionProposal:
    return ActionProposal(action, policy_id=producer, timestep=frame.timestep,
                          metadata=metadata, observation_digest=frame.digest)


def _state_pair(component: Any, *, name: str) -> tuple[Any, Any] | None:
    snapshot_state = getattr(component, "snapshot_state", None)
    restore_state = getattr(component, "restore_state", None)
    snapshot = getattr(component, "snapshot", None)
    restore = getattr(component, "restore", None)
    if callable(snapshot_state) != callable(restore_state):
        raise ContractError(f"{name} exposes only one of snapshot_state()/restore_state()")
    if callable(snapshot_state):
        return snapshot_state, restore_state
    if callable(snapshot) != callable(restore):
        raise ContractError(f"{name} exposes only one of snapshot()/restore()")
    if callable(snapshot):
        return snapshot, restore
    return None


def _is_stateless_callable(component: Any) -> bool:
    # Callability alone does not prove that a controller is stateless: a
    # closure or callable object can carry mutable state outside ``__dict__``.
    # Require an explicit declaration when no paired state hooks exist.
    return callable(component) and bool(getattr(component, "__arrow_stateless__", False))


class InterruptibleArrowController(Protocol):
    def reset(self) -> None: ...
    def propose(self, frame: ObservationFrame) -> Any: ...
    def commit(self, record: StepRecord) -> None: ...
    def interrupt(self) -> None: ...


@dataclass(frozen=True)
class ArrowSnapshot:
    controller_state: Any = None
    pending: ActionProposal | None = None
    interrupted: bool = False
    unavailable_reason: str | None = None


class ArrowAdapter:
    """Wrap an explicitly interruptible Arrow controller as a Teacher."""

    def __init__(
        self,
        controller: Any,
        *,
        producer: str = "arrow",
        require_interrupt_hook: bool = False,
    ) -> None:
        if controller is None:
            raise TypeError("ArrowAdapter requires an injected controller")
        if not callable(getattr(controller, "propose", None)) and not callable(controller):
            raise TypeError(
                "Arrow controller must expose propose(frame); whole-episode recover() APIs are unsupported"
            )
        if not callable(getattr(controller, "commit", None)):
            raise TypeError("interruptible Arrow controller must expose commit(record)")
        if require_interrupt_hook and not self._interrupt_hook(controller):
            raise TypeError("interruptible Arrow controller must expose interrupt() or cancel()")
        if not producer:
            raise ValueError("producer is required")
        self.controller = controller
        self.producer = producer
        self._pending: ActionProposal | None = None
        self._interrupted = False
        self.last_unavailable_reason: str | None = None

    @property
    def rollback_completeness(self) -> Mapping[str, bool]:
        try:
            complete = _state_pair(self.controller, name="Arrow controller") is not None
        except ContractError:
            complete = False
        if not complete:
            complete = _is_stateless_callable(self.controller)
        return {"controller": complete}

    @property
    def rollback_complete(self) -> bool:
        return all(self.rollback_completeness.values())

    @staticmethod
    def _interrupt_hook(controller: Any) -> Any:
        return getattr(controller, "interrupt", None) or getattr(controller, "cancel", None)

    def reset(self) -> None:
        reset = getattr(self.controller, "reset", None)
        if not callable(reset):
            raise ContractError("Arrow controller has no reset() hook")
        reset()
        self._pending = None
        self._interrupted = False
        self.last_unavailable_reason = None

    def _raw_propose(self, frame: ObservationFrame) -> Any:
        method = getattr(self.controller, "propose", None)
        if callable(method):
            return method(frame)
        # A plain callable is allowed for test doubles and function-backed
        # controllers, but it still has to produce one proposal at a time.
        return self.controller(frame)

    def _coerce(self, value: Any, frame: ObservationFrame) -> ActionProposal:
        if value is None:
            # ``None`` is a valid, explicit no-arrow result: perception had no
            # usable source/destination in this frame.  It is intentionally
            # different from an exception, which is a controller fault.
            self.last_unavailable_reason = "controller_returned_none"
            return None  # type: ignore[return-value]
        if isinstance(value, ActionProposal):
            if value.timestep != frame.timestep:
                raise ContractError("Arrow controller returned a stale proposal")
            if value.observation_digest is None:
                return ActionProposal(value.action, policy_id=value.policy_id,
                                      timestep=value.timestep, metadata=value.metadata,
                                      provenance=value.provenance,
                                      interruptible=value.interruptible,
                                      observation_digest=frame.digest)
            return value
        metadata: Mapping[str, Any] = {}
        confidence = 1.0
        valid = True
        action: Any = value
        if isinstance(value, Mapping):
            if "action" not in value:
                raise ContractError("Arrow proposal mapping lacks action")
            action = value["action"]
            raw_metadata = value.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise ContractError("Arrow proposal metadata must be a mapping")
            metadata = dict(raw_metadata)
            confidence = float(value.get("confidence", 1.0))
            valid = bool(value.get("valid", True))
        validate_action(action)
        if not valid:
            raise ContractError("Arrow controller returned an invalid proposal")
        return _proposal(action, frame, self.producer, confidence=confidence, **dict(metadata))

    def propose(self, frame: ObservationFrame) -> ActionProposal:
        if self._interrupted:
            raise ContractError("Arrow adapter is interrupted; call reset() before proposing")
        if self._pending is not None:
            if self._pending.timestep == frame.timestep:
                return self._pending
            raise ContractError("previous Arrow proposal was not committed")
        try:
            raw = self._raw_propose(frame)
        except ArrowPerceptionUnavailable as exc:
            self.last_unavailable_reason = str(exc) or "perception_unavailable"
            self._pending = None
            return None  # type: ignore[return-value]
        self._pending = self._coerce(raw, frame)
        return self._pending

    def commit(self, record: StepRecord) -> None:
        if self._pending is None:
            # No proposal means an explicit unavailable perception frame.  The
            # host must not turn that into a fake action or a commit fault.
            if getattr(self, "last_unavailable_reason", None):
                return
            raise ContractError("Arrow commit has no pending proposal")
        if record.teacher is None or record.teacher.timestep != self._pending.timestep:
            raise ContractError("Arrow commit does not match the pending proposal")
        if record.teacher.action != self._pending.action:
            raise ContractError("Arrow commit action differs from the pending proposal")
        self.controller.commit(record)
        self._pending = None

    def interrupt(
        self,
        observation: ObservationFrame | None = None,
        *,
        reason: str = "interrupted",
        context: Mapping[str, Any] | None = None,
    ) -> ActionProposal | None:
        # The runtime calls this hook with the current frame.  It is a
        # proposal boundary, not permission for the controller to step.
        if observation is not None:
            proposal = self.propose(observation)
            if proposal is None:
                return None
            merged = {**dict(proposal.metadata), "interrupt_reason": reason, **dict(context or {})}
            return ActionProposal(
                proposal.action, policy_id=proposal.policy_id, timestep=proposal.timestep,
                metadata=merged, provenance=proposal.provenance, interruptible=True,
                observation_digest=proposal.observation_digest or observation.digest,
            )
        hook = self._interrupt_hook(self.controller)
        if not callable(hook):
            raise ContractError(
                "Arrow controller has no interrupt/cancel hook; refusing to emulate interruption with reset"
            )
        hook()
        self._pending = None
        self._interrupted = True
        return None

    def snapshot(self) -> ArrowSnapshot:
        hooks = _state_pair(self.controller, name="Arrow controller")
        if hooks is None:
            raise ContractError("Arrow controller has no snapshot_state/restore_state or snapshot/restore hooks")
        return ArrowSnapshot(hooks[0](), self._pending, self._interrupted, self.last_unavailable_reason)

    def restore(self, snapshot: ArrowSnapshot) -> None:
        if not isinstance(snapshot, ArrowSnapshot):
            raise ContractError("Arrow restore requires ArrowSnapshot")
        hooks = _state_pair(self.controller, name="Arrow controller")
        if hooks is None:
            raise ContractError("Arrow controller has no snapshot_state/restore_state or snapshot/restore hooks")
        hooks[1](snapshot.controller_state)
        self._pending = snapshot.pending
        self._interrupted = bool(snapshot.interrupted)
        self.last_unavailable_reason = snapshot.unavailable_reason

    def snapshot_state(self) -> ArrowSnapshot:
        """Alias used by :class:`TransactionalCoordinator` component rollback."""
        return self.snapshot()

    def restore_state(self, snapshot: ArrowSnapshot) -> None:
        """Alias used by :class:`TransactionalCoordinator` component rollback."""
        self.restore(snapshot)

    def close(self) -> None:
        """Release an injected Arrow controller, if it owns native resources."""
        close = getattr(self.controller, "close", None)
        if callable(close):
            close()


class UnavailableArrowAdapter(ArrowAdapter):
    """Explicit fail-closed seam for hosts without an Arrow runtime."""

    def __init__(self, reason: str = "Arrow controller dependencies are unavailable") -> None:
        self.reason = str(reason)
        self.producer = "arrow"
        self.controller = None
        self._pending = None
        self._interrupted = False

    def _raise(self) -> None:
        raise RuntimeError(self.reason)

    def reset(self) -> None:
        self._raise()

    def propose(self, frame: ObservationFrame) -> ActionProposal:
        self._raise()
        raise AssertionError("unreachable")

    def commit(self, record: StepRecord) -> None:
        self._raise()

    def interrupt(self, observation: ObservationFrame | None = None, *, reason: str = "interrupted", context: Mapping[str, Any] | None = None) -> ActionProposal | None:
        self._raise()

    def snapshot(self) -> ArrowSnapshot:
        self._raise()
        raise AssertionError("unreachable")

    def restore(self, snapshot: ArrowSnapshot) -> None:
        self._raise()


__all__ = ["ArrowAdapter", "ArrowSnapshot", "InterruptibleArrowController", "UnavailableArrowAdapter"]
