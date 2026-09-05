#!/usr/bin/env python3
"""Versioned LoRA policies used by the SmolVLA experiments.

The policy id is deliberately separate from the dataset/training profile.  A
profile describes the visual/text condition in the data; this module describes
which parameters are adapted.  Keeping the two axes separate makes manifests
and future threads unambiguous while retaining the historical action-only
checkpoint contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


ACTION_SIDE_TARGET_REGEX = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)

ACTION_VISUAL_TARGET_REGEX = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
    r"model\.vlm_with_expert\.vlm\.model\.connector\.modality_projection\.proj|"
    r"model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.(8|9|10|11)\.self_attn\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)

POLICY_SCHEMA_VERSION = 1
ACTION_ONLY_LORA_V1 = "action_only_lora_v1"
ACTION_VISUAL_LORA_V1 = "action_visual_lora_v1"
DEFAULT_FINETUNING_POLICY = ACTION_ONLY_LORA_V1
LORA_RANK = 16

ACTION_REQUIRED_MODULES = (
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)
ACTION_REQUIRED_PATTERNS = (
    r"^model\.vlm_with_expert\.lm_expert\..*\.q_proj$",
    r"^model\.vlm_with_expert\.lm_expert\..*\.v_proj$",
)
VISUAL_REQUIRED_MODULES = (
    "model.vlm_with_expert.vlm.model.connector.modality_projection.proj",
)
VISUAL_REQUIRED_PATTERNS = (
    r"^model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.(8|9|10|11)\.self_attn\.q_proj$",
    r"^model\.vlm_with_expert\.vlm\.model\.vision_model\.encoder\.layers\.(8|9|10|11)\.self_attn\.v_proj$",
)


@dataclass(frozen=True)
class FinetuningPolicy:
    policy_id: str
    display_name: str
    purpose: str
    framing: str
    target_regex: str
    allowed_module_patterns: tuple[str, ...]
    required_module_names: tuple[str, ...]
    required_module_patterns: tuple[str, ...]
    expected_target_count: int | None
    expected_trainable_numel: int | None
    permitted_profiles: tuple[str, ...]
    inventory_schema_version: int
    no_full_weight_trainables: bool = True

    @property
    def target_pattern(self) -> re.Pattern[str]:
        return re.compile(r"^" + self.target_regex + r"$")

    def accepts_profile(self, profile: str) -> bool:
        return profile in self.permitted_profiles

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata included in plans, manifests, and audits."""
        return {
            "policy_id": self.policy_id,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "framing": self.framing,
            "target_regex": self.target_regex,
            "allowed_module_patterns": list(self.allowed_module_patterns),
            "required_module_names": list(self.required_module_names),
            "required_module_patterns": list(self.required_module_patterns),
            "expected_target_count": self.expected_target_count,
            "expected_trainable_numel": self.expected_trainable_numel,
            "permitted_profiles": list(self.permitted_profiles),
            "inventory_schema_version": self.inventory_schema_version,
            "no_full_weight_trainables": self.no_full_weight_trainables,
            "rank": LORA_RANK,
        }


ACTION_ONLY_POLICY = FinetuningPolicy(
    policy_id=ACTION_ONLY_LORA_V1,
    display_name="Action-only LoRA v1",
    purpose="Historical baseline adapting the action expert and action/state projections.",
    framing="The VLM and vision pathway remain frozen; only action-side LoRA is trained.",
    target_regex=ACTION_SIDE_TARGET_REGEX,
    allowed_module_patterns=(ACTION_SIDE_TARGET_REGEX,),
    required_module_names=ACTION_REQUIRED_MODULES,
    required_module_patterns=ACTION_REQUIRED_PATTERNS,
    expected_target_count=69,
    expected_trainable_numel=None,
    permitted_profiles=("treatment", "no_arrow_treatment", "graph_treatment", "arrow_graph_treatment"),
    inventory_schema_version=1,
)

ACTION_VISUAL_POLICY = FinetuningPolicy(
    policy_id=ACTION_VISUAL_LORA_V1,
    display_name="Action + visual-path LoRA v1",
    purpose="Test whether frozen visual representations limit spatially grounded action learning.",
    framing=(
        "Retains action-side LoRA and adds LoRA to the VLM connector plus vision "
        "encoder layers 8-11 q/v; all base weights and VLM text weights stay frozen."
    ),
    target_regex=ACTION_VISUAL_TARGET_REGEX,
    allowed_module_patterns=(ACTION_VISUAL_TARGET_REGEX,),
    required_module_names=ACTION_REQUIRED_MODULES + VISUAL_REQUIRED_MODULES,
    required_module_patterns=ACTION_REQUIRED_PATTERNS + VISUAL_REQUIRED_PATTERNS,
    expected_target_count=78,
    expected_trainable_numel=1_585_152,
    permitted_profiles=("treatment", "no_arrow_treatment"),
    inventory_schema_version=2,
)

POLICIES: Mapping[str, FinetuningPolicy] = {
    ACTION_ONLY_LORA_V1: ACTION_ONLY_POLICY,
    ACTION_VISUAL_LORA_V1: ACTION_VISUAL_POLICY,
}


def get_policy(policy_id: str | None) -> FinetuningPolicy:
    """Resolve a policy id, preserving old manifests with no policy field."""
    resolved = DEFAULT_FINETUNING_POLICY if policy_id in (None, "") else str(policy_id)
    try:
        return POLICIES[resolved]
    except KeyError as exc:
        raise ValueError(f"unknown finetuning policy: {resolved}") from exc


def policy_metadata(policy_id: str | None = None) -> dict[str, Any]:
    return get_policy(policy_id).to_metadata()


def validate_policy_for_profile(
    policy_id: str | None,
    profile: str,
    manifest: Mapping[str, Any] | None = None,
) -> FinetuningPolicy:
    """Fail closed when a policy is incompatible with a data profile.

    Graph manifests currently seal ``action_side_only=true``.  That is a
    stronger boundary than a caller's profile string and is checked here too.
    """
    policy = get_policy(policy_id)
    if manifest is not None and profile in ("graph_treatment", "arrow_graph_treatment"):
        contract = manifest.get("training_contract")
        if isinstance(contract, Mapping) and contract.get("action_side_only") is True and policy.policy_id != ACTION_ONLY_LORA_V1:
            raise ValueError("graph profile manifest is sealed action_side_only=true; visual LoRA is not permitted")
    if not policy.accepts_profile(profile):
        raise ValueError(f"finetuning policy {policy.policy_id} is not permitted for profile {profile}")
    return policy


def policy_from_inventory(value: Mapping[str, Any]) -> FinetuningPolicy:
    """Resolve policy identity from an inventory, defaulting schema-1 records."""
    return get_policy(value.get("finetuning_policy_id"))
