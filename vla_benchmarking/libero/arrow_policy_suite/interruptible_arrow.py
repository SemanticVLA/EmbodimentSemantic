"""Side-effect-free, per-frame Arrow teacher boundary.

The historical Arrow controller in ``automatic_ttt`` owns a complete recovery
episode.  This module is the smaller contract needed by the policy suite: a
controller may inspect one observation (and an optional graph context), return
one action, and wait for the host to execute it.  It never receives an
environment object and therefore cannot call ``env.step`` accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping

from .contracts import ActionProposal, ObservationFrame, StepRecord, ContractError


class ArrowPerceptionUnavailable(RuntimeError):
    """Expected no-arrow condition, distinct from a controller fault."""


@dataclass(frozen=True)
class ArrowAvailability:
    available: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = ()


class InterruptibleArrow:
    """Adapt perception + Arrow control into a rollback-capable teacher.

    ``perception`` is a pure frame-to-context hook.  It may return ``None`` or
    raise :class:`ArrowPerceptionUnavailable` when the visual arrow is absent.
    Any other exception propagates as a real fault.  ``controller`` is invoked
    with ``(frame, context)`` when supported, then with ``frame`` for simple
    test/native controllers.  Neither hook receives the environment.
    """

    def __init__(
        self,
        controller: Any,
        *,
        perception: Callable[[ObservationFrame], Any] | None = None,
        producer: str = "arrow",
    ) -> None:
        if controller is None:
            raise TypeError("InterruptibleArrow requires a controller")
        self.controller = controller
        self.perception = perception
        self.producer = producer
        self._pending: ActionProposal | None = None
        self.last_availability = ArrowAvailability(True)

    def reset(self) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self._pending = None
        self.last_availability = ArrowAvailability(True)

    def _context(self, frame: ObservationFrame) -> Any:
        if self.perception is None:
            return frame.metadata.get("graph_context")
        try:
            value = self.perception(frame)
        except ArrowPerceptionUnavailable as exc:
            self.last_availability = ArrowAvailability(False, str(exc) or "perception_unavailable")
            return None
        if value is None:
            self.last_availability = ArrowAvailability(False, "perception_returned_none")
            return None
        self.last_availability = ArrowAvailability(True)
        return value

    def propose(self, frame: ObservationFrame) -> ActionProposal | None:
        if self._pending is not None:
            if self._pending.timestep == frame.timestep:
                return self._pending
            raise ContractError("previous Arrow proposal was not committed")
        context = self._context(frame)
        if not self.last_availability.available:
            return None
        method = getattr(self.controller, "propose", None)
        if not callable(method):
            if not callable(self.controller):
                raise ContractError("Arrow controller exposes no propose hook")
            method = self.controller
        try:
            try:
                parameters = inspect.signature(method).parameters
                accepts_context = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()) or len(parameters) >= 2
            except (TypeError, ValueError):
                accepts_context = True
            value = method(frame, context) if accepts_context else method(frame)
        except ArrowPerceptionUnavailable as exc:
            self.last_availability = ArrowAvailability(False, str(exc) or "perception_unavailable")
            return None
        if value is None:
            self.last_availability = ArrowAvailability(False, "controller_returned_none")
            return None
        if isinstance(value, ActionProposal):
            proposal = value
            if proposal.observation_digest is None:
                proposal = ActionProposal(
                    proposal.action, policy_id=proposal.policy_id,
                    timestep=frame.timestep, metadata=proposal.metadata,
                    provenance=proposal.provenance, interruptible=True,
                    observation_digest=frame.digest,
                )
        elif isinstance(value, Mapping):
            if "action" not in value:
                raise ContractError("Arrow controller mapping lacks action")
            metadata = dict(value.get("metadata", {}))
            proposal = ActionProposal(
                value["action"], policy_id=str(value.get("policy_id", self.producer)),
                timestep=frame.timestep, metadata=metadata,
                provenance=dict(value.get("provenance", {})), interruptible=True,
                observation_digest=frame.digest,
            )
        else:
            proposal = ActionProposal(
                value, policy_id=self.producer, timestep=frame.timestep,
                interruptible=True, observation_digest=frame.digest,
            )
        if proposal.timestep != frame.timestep or proposal.observation_digest != frame.digest:
            raise ContractError("Arrow proposal does not match the current observation frame")
        self._pending = proposal
        return proposal

    def commit(self, record: StepRecord | Any) -> None:
        if self._pending is None:
            if not self.last_availability.available:
                return
            raise ContractError("Arrow commit has no pending proposal")
        teacher = getattr(record, "teacher", None)
        if teacher is None or teacher.action != self._pending.action:
            raise ContractError("Arrow commit does not match the pending proposal")
        commit = getattr(self.controller, "commit", None)
        if callable(commit):
            commit(record)
        self._pending = None

    def interrupt(self) -> None:
        hook = getattr(self.controller, "interrupt", None) or getattr(self.controller, "cancel", None)
        if callable(hook):
            hook()
        self._pending = None

    def snapshot_state(self) -> Any:
        snapshot = getattr(self.controller, "snapshot_state", None) or getattr(self.controller, "snapshot", None)
        if not callable(snapshot):
            raise ContractError("Arrow controller has no snapshot/restore state hook")
        return (snapshot(), self._pending, self.last_availability)

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, tuple) or len(state) != 3:
            raise ContractError("invalid InterruptibleArrow snapshot")
        restore = getattr(self.controller, "restore_state", None) or getattr(self.controller, "restore", None)
        if not callable(restore):
            raise ContractError("Arrow controller has no snapshot/restore state hook")
        restore(state[0])
        self._pending = state[1]
        self.last_availability = state[2]

    snapshot = snapshot_state
    restore = restore_state

    @property
    def rollback_complete(self) -> bool:
        return callable(getattr(self.controller, "snapshot_state", None) or getattr(self.controller, "snapshot", None)) and callable(getattr(self.controller, "restore_state", None) or getattr(self.controller, "restore", None))

    def close(self) -> None:
        close = getattr(self.controller, "close", None)
        if callable(close):
            close()


__all__ = ["ArrowAvailability", "ArrowPerceptionUnavailable", "InterruptibleArrow"]
