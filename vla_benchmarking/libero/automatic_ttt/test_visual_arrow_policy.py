from __future__ import annotations

import json

import numpy as np
import pytest

from vla_benchmarking.libero.evaluation.visual_scene_graph import (
    SEALED_LORA_ARROW_HEAD_LENGTH,
    SEALED_LORA_ARROW_WIDTH,
    overlay_visual_relations,
)

from .contracts import ContractError
from .visual_arrow_policy import (
    LiveVisualGoalArrowTransform,
    VISUAL_ARROW_POLICY_CONTRACT,
    VISUAL_ARROW_POLICY_CONTRACT_SHA256,
    VISUAL_ARROW_POLICY_ID,
    validate_visual_arrow_collection_manifest,
)


class _FakeGenerator:
    scene_graph_subject_filter = None

    def observe_visual_graph(self, env, *, camera):
        assert env.task_id == 0
        assert env.observation_height == env.observation_width == 256
        assert camera == "agentview"
        return {
            "bboxes": {
                "akita_black_bowl_1": [10, 116, 30, 136],
                "plate_1": [220, 116, 240, 136],
            },
            "relations": [("akita_black_bowl_1", "is_left_of", "plate_1")],
        }


def _observation() -> dict[str, object]:
    return {
        "agentview": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist": np.full((256, 256, 3), 17, dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "instruction": "pick up the bowl and place it on the plate",
    }


def test_live_transform_matches_shared_eval_renderer_and_changes_only_agentview():
    observation = _observation()
    transform = LiveVisualGoalArrowTransform(
        raw_environment=object(),
        task_id=0,
        task_description=str(observation["instruction"]),
        live_generator=_FakeGenerator(),
    )

    actual = transform(observation)
    expected, audit = overlay_visual_relations(
        np.asarray(observation["agentview"]),
        _FakeGenerator().observe_visual_graph(transform._view, camera="agentview")["bboxes"],
        [("akita_black_bowl_1", "is_left_of", "plate_1")],
        condition="visual_goal_arrow",
        subject="akita_black_bowl_1",
        goal_object="plate_1",
        line_width=SEALED_LORA_ARROW_WIDTH,
        head_length=SEALED_LORA_ARROW_HEAD_LENGTH,
    )

    assert np.array_equal(actual["agentview"], expected)
    assert audit["changed_pixels"] > 0
    assert np.array_equal(actual["wrist"], observation["wrist"])
    assert np.array_equal(actual["state"], observation["state"])
    assert actual["instruction"] == observation["instruction"]
    assert not np.asarray(observation["agentview"]).any()
    summary = transform.audit_summary()
    assert summary["frames_seen"] == 1
    assert summary["frames_with_changed_pixels"] == 1
    assert summary["images_retained_by_transform"] == 0


def test_visual_collection_manifest_requires_visible_transform_evidence(tmp_path):
    summary = {
        "policy_id": VISUAL_ARROW_POLICY_ID,
        "visual_contract": VISUAL_ARROW_POLICY_CONTRACT,
        "visual_contract_sha256": VISUAL_ARROW_POLICY_CONTRACT_SHA256,
        "frames_seen": 2,
        "frames_with_drawable_arrow": 2,
        "frames_with_changed_pixels": 2,
        "changed_pixels_total": 30,
        "frame_audit_sha256": "a" * 64,
        "images_retained_by_transform": 0,
    }
    accepted = tmp_path / "accepted.jsonl"
    accepted.write_text(
        json.dumps({
            "transitions": [{"actor": "arrow_grasp_controller"}],
            "metadata": {"student_observation_transform": summary},
        }) + "\n",
        encoding="utf-8",
    )
    payload = {
        "accepted_count": 1,
        "accepted_episodes_jsonl": str(accepted),
        "provenance": {
            "student_policy_id": VISUAL_ARROW_POLICY_ID,
            "student_visual_contract": VISUAL_ARROW_POLICY_CONTRACT,
            "student_visual_contract_sha256": VISUAL_ARROW_POLICY_CONTRACT_SHA256,
        },
    }

    validate_visual_arrow_collection_manifest(payload)
    summary["frames_with_changed_pixels"] = 0
    accepted.write_text(
        json.dumps({
            "transitions": [{"actor": "arrow_grasp_controller"}],
            "metadata": {"student_observation_transform": summary},
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="no visible arrow pixels"):
        validate_visual_arrow_collection_manifest(payload)
