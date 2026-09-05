from __future__ import annotations

import pytest

from vla_benchmarking.arrow_finetuned_vla.action_visual_lora.lora_finetuning_policy import (
    ACTION_ONLY_LORA_V1,
    ACTION_VISUAL_LORA_V1,
    DEFAULT_FINETUNING_POLICY,
    get_policy,
    validate_policy_for_profile,
)


def test_policy_ids_and_visual_contract_are_stable() -> None:
    historical = get_policy(None)
    visual = get_policy(ACTION_VISUAL_LORA_V1)
    assert historical.policy_id == DEFAULT_FINETUNING_POLICY == ACTION_ONLY_LORA_V1
    assert visual.expected_target_count == 78
    assert visual.expected_trainable_numel == 1_585_152
    assert "connector\\.modality_projection\\.proj" in visual.target_regex
    assert "vision_model\\.encoder\\.layers\\.(8|9|10|11)" in visual.target_regex
    assert visual.no_full_weight_trainables is True


def test_visual_policy_is_allowed_only_for_arrow_or_no_arrow_treatment() -> None:
    assert validate_policy_for_profile(ACTION_VISUAL_LORA_V1, "treatment").policy_id == ACTION_VISUAL_LORA_V1
    assert validate_policy_for_profile(ACTION_VISUAL_LORA_V1, "no_arrow_treatment").policy_id == ACTION_VISUAL_LORA_V1
    with pytest.raises(ValueError, match="not permitted"):
        validate_policy_for_profile(ACTION_VISUAL_LORA_V1, "graph_treatment")


def test_graph_manifest_action_side_seal_rejects_visual_policy() -> None:
    with pytest.raises(ValueError, match="action_side_only"):
        validate_policy_for_profile(
            ACTION_VISUAL_LORA_V1,
            "graph_treatment",
            {"training_contract": {"action_side_only": True}},
        )


def test_unknown_policy_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown finetuning policy"):
        get_policy("not_a_policy")
