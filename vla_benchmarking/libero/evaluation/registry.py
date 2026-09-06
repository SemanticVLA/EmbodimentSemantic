"""Explicit policy capabilities for the shared LIBERO evaluator.

The registry selects an existing execution backend; it does not create a
universal action loop. Each policy keeps its native inference and motion code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .backends import DIRECT_MATRIX, LEROBOT, NATIVE, BackendCapabilities
from .contracts import EvaluationCondition


@dataclass(frozen=True)
class PolicyCapabilities:
    policy_kind: Literal[
        "canonical_grasp", "lerobot", "smolvla_base", "smolvla_no_arrow", "smolvla_target_arrow",
        "pi05", "openvla", "openvla_oft", "octo_community_multisuite_190k",
        "octo_base15_spatial_no_arrow_matched",
    ]
    backend: BackendCapabilities
    visual_inputs: tuple[str, ...]
    text_contexts: tuple[str, ...]
    # The prompt is part of the randomization contract for policies that
    # consume task text.  Arrow controllers do not consume the prompt.
    prompt_applicability: dict[str, str]
    text_contract: str


POLICIES: dict[str, PolicyCapabilities] = {
    "canonical_grasp": PolicyCapabilities(
        "canonical_grasp", DIRECT_MATRIX, ("goal_arrow",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "not_applicable"},
        "none",
    ),
    "lerobot": PolicyCapabilities(
        "lerobot",
        LEROBOT,
        ("none", "goal_arrow", "relation_arrows"),
        ("none", "scene_graph", "text_triplet"),
        {"vanilla": "applied", "sealed_randomized": "applied"},
        "policy_defined",
    ),
    "smolvla_no_arrow": PolicyCapabilities(
        "smolvla_no_arrow",
        LEROBOT,
        ("none",),
        ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "smolvla_base": PolicyCapabilities(
        "smolvla_base",
        LEROBOT,
        ("none",),
        ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "smolvla_target_arrow": PolicyCapabilities(
        "smolvla_target_arrow",
        LEROBOT,
        ("none", "goal_arrow"),
        ("none",),
        {"vanilla": "applied", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    # These policies consume the standard task description but no visual
    # arrows or injected context.  OpenVLA-OFT and Octo retain their native
    # inference loops behind the native backend boundary.
    "pi05": PolicyCapabilities(
        "pi05", LEROBOT, ("none",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "openvla_oft": PolicyCapabilities(
        "openvla_oft", NATIVE, ("none",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "openvla": PolicyCapabilities(
        "openvla", NATIVE, ("none",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "octo_community_multisuite_190k": PolicyCapabilities(
        "octo_community_multisuite_190k", NATIVE, ("none",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
    ),
    "octo_base15_spatial_no_arrow_matched": PolicyCapabilities(
        "octo_base15_spatial_no_arrow_matched", NATIVE, ("none",), ("none",),
        {"vanilla": "not_applicable", "sealed_randomized": "applied"},
        "standard_no_extra_text",
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
