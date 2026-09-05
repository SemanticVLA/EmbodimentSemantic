"""Explicit policy capabilities for the shared LIBERO evaluator.

The registry selects an existing execution backend; it does not create a
universal action loop. Each policy keeps its native inference and motion code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .backends import DIRECT_MATRIX, LEROBOT, BackendCapabilities
from .contracts import EvaluationCondition


@dataclass(frozen=True)
class PolicyCapabilities:
    policy_kind: Literal["canonical_grasp", "arrow_student", "lerobot"]
    backend: BackendCapabilities
    visual_inputs: tuple[str, ...]
    text_contexts: tuple[str, ...]


POLICIES: dict[str, PolicyCapabilities] = {
    "canonical_grasp": PolicyCapabilities(
        "canonical_grasp", DIRECT_MATRIX, ("goal_arrow",), ("none",)
    ),
    "arrow_student": PolicyCapabilities(
        "arrow_student", DIRECT_MATRIX, ("goal_arrow",), ("none",)
    ),
    "lerobot": PolicyCapabilities(
        "lerobot",
        LEROBOT,
        ("none", "goal_arrow", "relation_arrows"),
        ("none", "scene_graph", "text_triplet"),
    ),
}


def get_policy_capabilities(policy_kind: str) -> PolicyCapabilities:
    try:
        return POLICIES[str(policy_kind)]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation policy kind: {policy_kind!r}") from exc


def validate_policy_condition(
    policy_kind: str, condition: EvaluationCondition
) -> PolicyCapabilities:
    capabilities = get_policy_capabilities(policy_kind)
    if condition.visual_input not in capabilities.visual_inputs:
        raise ValueError(
            f"{policy_kind} does not support visual input {condition.visual_input!r}"
        )
    if condition.text_context not in capabilities.text_contexts:
        raise ValueError(
            f"{policy_kind} does not support text context {condition.text_context!r}"
        )
    return capabilities


__all__ = [
    "POLICIES",
    "PolicyCapabilities",
    "get_policy_capabilities",
    "validate_policy_condition",
]
