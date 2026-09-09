"""Policy and teacher protocols for the Arrow policy suite."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .contracts import ActionProposal, ObservationFrame


@runtime_checkable
class Policy(Protocol):
    """A policy proposes actions from immutable observation frames."""

    policy_id: str

    def propose(self, observation: ObservationFrame) -> ActionProposal: ...


@runtime_checkable
class InterruptibleTeacher(Protocol):
    """Teacher that can propose a correction without owning the environment."""

    teacher_id: str

    def interrupt(
        self,
        observation: ObservationFrame,
        *,
        reason: str,
        context: Mapping[str, Any] | None = None,
    ) -> ActionProposal | None: ...


def policy_identifier(policy: Any) -> str:
    value = getattr(policy, "policy_id", None)
    return str(value) if value else type(policy).__name__


__all__ = ["InterruptibleTeacher", "Policy", "policy_identifier"]
