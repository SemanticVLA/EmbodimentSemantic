"""Matched train/eval visual-goal-arrow policy for automatic TTT.

The teacher continues to execute the canonical Arrow controller.  This module
changes only the student-facing main camera: every fresh live observation is
decorated with the same goal-arrow renderer used by online evaluation.  Wrist
pixels, robot state, task text, episode boundaries, and action labels are left
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_benchmarking.libero.evaluation.libero_live_semantic_context import (
    LiveSemanticContextGenerator,
)
from vla_benchmarking.libero.evaluation.visual_scene_graph import (
    DEFAULT_ARROW_COLOR_RGB,
    SEALED_LORA_ARROW_HEAD_LENGTH,
    SEALED_LORA_ARROW_WIDTH,
    VISUAL_GOAL_ARROW_CONDITION,
    overlay_visual_relations,
    resolve_task_goal_object,
)
from vla_benchmarking.libero.shared.config import (
    ARROW_SOURCE_OBJECT,
    SCENE_GRAPH_SUBJECT_FILTER,
    TASK_GOAL_OBJECT_CONFIG,
)

from .contracts import ContractError
from .dataset import CANONICAL_OBSERVATION_SCHEMA, validate_student_observation_schema


VISUAL_ARROW_POLICY_ID = "smolvla_fresh_arrow_visual_goal_peft"
VISUAL_ARROW_POLICY_CONTRACT = {
    "name": "automatic_ttt_live_visual_goal_arrow_v1",
    "condition": VISUAL_GOAL_ARROW_CONDITION,
    "main_camera": "agentview",
    "modified_student_field": "agentview",
    "unchanged_student_fields": ["wrist", "state", "instruction"],
    "image_size": [256, 256],
    "image_orientation": "canonical_smolvla_policy_orientation",
    "bbox_source": "live_simulator_geometry_projected_by_LiveSemanticContextGenerator",
    "pixel_origin": "top_left",
    "pixel_axes": {"x": "right", "y": "down"},
    "arrow_subject": ARROW_SOURCE_OBJECT,
    "task_goal_objects": {str(key): value for key, value in sorted(TASK_GOAL_OBJECT_CONFIG.items())},
    "arrow_color_rgb": list(DEFAULT_ARROW_COLOR_RGB),
    "line_width": SEALED_LORA_ARROW_WIDTH,
    "head_length": SEALED_LORA_ARROW_HEAD_LENGTH,
    "visual_prompt_hint": "disabled",
    "student_action_labels": "executed_arrow_grasp_controller_actions",
}
VISUAL_ARROW_POLICY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        VISUAL_ARROW_POLICY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class _VisualEnvironmentView:
    """Minimal shape expected by ``LiveSemanticContextGenerator``."""

    _env: Any
    observation_height: int
    observation_width: int
    task: str
    task_id: int


@dataclass
class LiveVisualGoalArrowTransform:
    """Overlay one task-goal arrow without retaining any image across frames."""

    raw_environment: Any
    task_id: int
    task_description: str
    resolution: int = 256
    live_generator: Any | None = None
    frames_seen: int = 0
    frames_with_drawable_arrow: int = 0
    frames_with_changed_pixels: int = 0
    changed_pixels_total: int = 0
    _audit_digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.task_id) not in TASK_GOAL_OBJECT_CONFIG:
            raise ContractError(f"visual-arrow policy has no goal object for task {self.task_id}")
        if int(self.resolution) != 256:
            raise ContractError("visual-arrow policy is sealed to 256x256 observations")
        inner = getattr(self.raw_environment, "raw_environment", self.raw_environment)
        self._view = _VisualEnvironmentView(
            _env=inner,
            observation_height=int(self.resolution),
            observation_width=int(self.resolution),
            task=str(self.task_description),
            task_id=int(self.task_id),
        )
        if self.live_generator is None:
            self.live_generator = LiveSemanticContextGenerator()
            self.live_generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_student_observation_schema(
            observation,
            require_complete=True,
            schema=CANONICAL_OBSERVATION_SCHEMA,
        )
        main = np.asarray(observation["agentview"])
        if main.dtype != np.uint8 or main.shape != (256, 256, 3):
            raise ContractError(
                "visual-arrow policy requires canonical uint8 256x256x3 agentview pixels"
            )
        context = self.live_generator.observe_visual_graph(self._view, camera="agentview")
        bboxes = context.get("bboxes")
        source_relations = context.get("relations")
        if not isinstance(bboxes, dict) or not isinstance(source_relations, list):
            raise ContractError("live visual-arrow context lacks bboxes or relations")
        goal_object = resolve_task_goal_object(int(self.task_id), TASK_GOAL_OBJECT_CONFIG)
        overlaid, audit = overlay_visual_relations(
            main,
            bboxes,
            source_relations,
            condition=VISUAL_GOAL_ARROW_CONDITION,
            subject=ARROW_SOURCE_OBJECT,
            goal_object=goal_object,
            line_width=SEALED_LORA_ARROW_WIDTH,
            head_length=SEALED_LORA_ARROW_HEAD_LENGTH,
        )
        result = dict(observation)
        result["agentview"] = overlaid
        for key in ("wrist", "state", "instruction"):
            left, right = observation[key], result[key]
            equal = left == right if isinstance(left, str) else np.array_equal(left, right)
            if not equal:
                raise ContractError(f"visual-arrow policy unexpectedly changed {key}")

        self.frames_seen += 1
        drawn = audit["drawn_relations"]
        changed_pixels = int(audit["changed_pixels"])
        self.frames_with_drawable_arrow += int(bool(drawn))
        self.frames_with_changed_pixels += int(changed_pixels > 0)
        self.changed_pixels_total += changed_pixels
        digest_record = {
            "frame_index": self.frames_seen - 1,
            "goal_object": goal_object,
            "drawn_relations": drawn,
            "changed_pixels": changed_pixels,
        }
        self._audit_digest.update(
            json.dumps(digest_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        return result

    def audit_summary(self) -> dict[str, Any]:
        return {
            "policy_id": VISUAL_ARROW_POLICY_ID,
            "visual_contract": VISUAL_ARROW_POLICY_CONTRACT,
            "visual_contract_sha256": VISUAL_ARROW_POLICY_CONTRACT_SHA256,
            "frames_seen": int(self.frames_seen),
            "frames_with_drawable_arrow": int(self.frames_with_drawable_arrow),
            "frames_with_changed_pixels": int(self.frames_with_changed_pixels),
            "changed_pixels_total": int(self.changed_pixels_total),
            "frame_audit_sha256": self._audit_digest.hexdigest(),
            "images_retained_by_transform": 0,
        }


def make_live_visual_goal_arrow_transform(
    raw_environment: Any,
    *,
    task_id: int,
    task_description: str,
    resolution: int = 256,
) -> LiveVisualGoalArrowTransform:
    return LiveVisualGoalArrowTransform(
        raw_environment=raw_environment,
        task_id=int(task_id),
        task_description=str(task_description),
        resolution=int(resolution),
    )


def validate_visual_arrow_collection_manifest(payload: Mapping[str, Any]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError("visual-arrow collection manifest lacks provenance")
    if provenance.get("student_policy_id") != VISUAL_ARROW_POLICY_ID:
        raise ContractError("collection manifest has the wrong student visual policy")
    if provenance.get("student_visual_contract") != VISUAL_ARROW_POLICY_CONTRACT:
        raise ContractError("collection manifest visual contract differs from the live policy")
    if provenance.get("student_visual_contract_sha256") != VISUAL_ARROW_POLICY_CONTRACT_SHA256:
        raise ContractError("collection manifest visual contract hash differs from the live policy")
    accepted_path = payload.get("accepted_episodes_jsonl")
    if not isinstance(accepted_path, str):
        raise ContractError("visual-arrow collection manifest lacks its accepted trace path")
    try:
        with Path(accepted_path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("visual-arrow accepted traces are unreadable") from exc
    if len(rows) != payload.get("accepted_count") or not rows:
        raise ContractError("visual-arrow accepted trace count differs from the manifest")
    for row in rows:
        transitions = row.get("transitions")
        metadata = row.get("metadata")
        summary = metadata.get("student_observation_transform") if isinstance(metadata, Mapping) else None
        if not isinstance(transitions, list) or not transitions or not isinstance(summary, Mapping):
            raise ContractError("visual-arrow accepted trace lacks transform evidence")
        if summary.get("policy_id") != VISUAL_ARROW_POLICY_ID:
            raise ContractError("accepted trace has the wrong visual policy id")
        if summary.get("visual_contract") != VISUAL_ARROW_POLICY_CONTRACT:
            raise ContractError("accepted trace has the wrong visual contract")
        if summary.get("visual_contract_sha256") != VISUAL_ARROW_POLICY_CONTRACT_SHA256:
            raise ContractError("accepted trace has the wrong visual contract hash")
        if summary.get("frames_seen") != len(transitions) + 1:
            raise ContractError("visual transform did not process every trajectory observation")
        if int(summary.get("frames_with_changed_pixels", 0)) <= 0:
            raise ContractError("visual transform produced no visible arrow pixels")
        if summary.get("images_retained_by_transform") != 0:
            raise ContractError("visual transform retained image payloads across frames")


__all__ = [
    "LiveVisualGoalArrowTransform",
    "VISUAL_ARROW_POLICY_CONTRACT",
    "VISUAL_ARROW_POLICY_CONTRACT_SHA256",
    "VISUAL_ARROW_POLICY_ID",
    "make_live_visual_goal_arrow_transform",
    "validate_visual_arrow_collection_manifest",
]
