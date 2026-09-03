"""Experimental language-free SmolVLA bridge for visual goal arrows.

This package intentionally has no dependency on LeRobot or Transformers.  It
operates on the output of a compatible vision/connector adapter and exposes
the cache contract consumed by the SmolVLA action expert.
"""

from .bridge import ArrowConditioningBridge, BridgeOutput, build_prefix_attention_mask
from .config import (
    ArrowPolicyConfig,
    OptimizerConfig,
    StageConfig,
    StageName,
    STAGE_SCHEDULE,
    apply_stage_trainability,
    build_stage_optimizer,
)
from .contracts import (
    ConditioningCache,
    StudentBatch,
    StudentOutput,
    TeacherConditioningTargets,
)
from .flow import (
    flow_matching_target,
    masked_mse_loss,
    sample_flow_pair,
    teacher_velocity_loss,
)
from .teacher import (
    apply_rope,
    conditioning_feature_loss,
    inverse_rope,
    make_teacher_conditioning_targets,
    pool_teacher_text_kv,
    rms_normalize,
)
from .routing import ExpertLayerConditioning, route_expert_cache
from .student import ActionExpertAdapter, ArrowVisualStudent
from .training import (
    build_stage_scheduler,
    clip_stage_gradients,
    compute_batch_loss,
    compute_stage_loss,
    resolve_stage,
)
from .lerobot_integration import (
    PINNED_LEROBOT_COMMIT,
    ArrowSmolVLAActionExpert,
    ArrowSmolVLAPolicy,
    ArrowStageTrainer,
    AtomicCheckpointStore,
    SmolVLATeacherAdapter,
    TeacherBatch,
    load_pinned_smolvla,
)


def run_smoke(*args, **kwargs):
    """Lazily run the standalone dummy-tensor smoke check."""

    from .smoke import run_smoke as _run_smoke

    return _run_smoke(*args, **kwargs)

__all__ = [
    "ArrowConditioningBridge",
    "ArrowVisualStudent",
    "ActionExpertAdapter",
    "apply_rope",
    "ArrowPolicyConfig",
    "BridgeOutput",
    "ConditioningCache",
    "ExpertLayerConditioning",
    "OptimizerConfig",
    "StageConfig",
    "StageName",
    "STAGE_SCHEDULE",
    "StudentBatch",
    "StudentOutput",
    "TeacherConditioningTargets",
    "apply_stage_trainability",
    "build_stage_optimizer",
    "conditioning_feature_loss",
    "build_stage_scheduler",
    "clip_stage_gradients",
    "compute_batch_loss",
    "compute_stage_loss",
    "build_prefix_attention_mask",
    "flow_matching_target",
    "inverse_rope",
    "make_teacher_conditioning_targets",
    "masked_mse_loss",
    "pool_teacher_text_kv",
    "rms_normalize",
    "route_expert_cache",
    "resolve_stage",
    "run_smoke",
    "sample_flow_pair",
    "teacher_velocity_loss",
    "PINNED_LEROBOT_COMMIT",
    "ArrowSmolVLAActionExpert",
    "ArrowSmolVLAPolicy",
    "ArrowStageTrainer",
    "AtomicCheckpointStore",
    "SmolVLATeacherAdapter",
    "TeacherBatch",
    "load_pinned_smolvla",
]
