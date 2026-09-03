"""Stage losses for the language-free arrow-policy experiment.

This module deliberately stops at the tensor boundary.  A caller supplies the
retained SmolVLA expert adapter and the already sampled flow pair; the helpers
make the four training stages explicit without importing LeRobot or a dataset
implementation.
"""

from __future__ import annotations

from typing import Optional

import math

import torch
from torch import Tensor, nn

from .config import STAGE_SCHEDULE, StageConfig, StageName
from .contracts import ConditioningCache, StudentBatch, StudentOutput, TeacherConditioningTargets
from .flow import flow_matching_target, masked_mse_loss, teacher_velocity_loss
from .teacher import conditioning_feature_loss


def build_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    stage: StageName | StageConfig,
    *,
    warmup_fraction: float = 0.05,
    final_lr_fraction: float = 0.10,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the pinned linear-warmup/cosine-decay schedule for one stage.

    The optimizer's parameter groups retain their distinct bridge, expert and
    vision learning rates; the scheduler scales all groups by the same factor.
    A stage transition should construct a fresh optimizer and scheduler.
    """

    config = resolve_stage(stage)
    if not (0.0 <= warmup_fraction < 1.0):
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not (0.0 <= final_lr_fraction <= 1.0):
        raise ValueError("final_lr_fraction must be in [0, 1]")
    warmup_steps = max(1, int(round(config.updates * warmup_fraction)))
    decay_steps = max(1, config.updates - warmup_steps)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = min(1.0, float(step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_fraction + (1.0 - final_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def clip_stage_gradients(model: nn.Module, max_norm: float = 1.0) -> Tensor:
    """Clip only currently trainable parameters and return the pre-clip norm."""

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def resolve_stage(stage: StageName | StageConfig) -> StageConfig:
    """Return the canonical schedule entry for a stage name or config."""

    if isinstance(stage, StageConfig):
        return stage
    for item in STAGE_SCHEDULE:
        if item.name == stage:
            return item
    raise ValueError(f"unknown stage: {stage}")


def compute_stage_loss(
    stage: StageName | StageConfig,
    student_output: StudentOutput,
    *,
    actions: Optional[Tensor] = None,
    noise: Optional[Tensor] = None,
    action_mask: Optional[Tensor] = None,
    teacher_targets: Optional[TeacherConditioningTargets | ConditioningCache] = None,
    teacher_velocity: Optional[Tensor] = None,
    feature_weight: float = 1.0,
    velocity_weight: float = 1.0,
) -> Tensor:
    """Compute the objective for one staged update.

    Stage A is the bridge distillation objective: per-layer prefix feature KD
    plus velocity KD from the complete text-conditioned teacher.  Stages B-D
    use the demonstrated flow-matching target ``noise - actions``.  The caller
    must draw the same ``noise`` and timestep for teacher and student in stage
    A; this function only combines their already computed outputs.
    """

    config = resolve_stage(stage)
    if not (feature_weight >= 0 and velocity_weight >= 0):
        raise ValueError("loss weights must be non-negative")

    if config.name is StageName.A_DISTILL_BRIDGE:
        if teacher_targets is None or teacher_velocity is None:
            raise ValueError("stage A requires teacher conditioning targets and teacher velocity")
        feature = conditioning_feature_loss(student_output.conditioning, teacher_targets)
        velocity = teacher_velocity_loss(student_output.velocity, teacher_velocity, action_mask)
        return feature_weight * feature + velocity_weight * velocity

    if actions is None or noise is None:
        raise ValueError(f"stage {config.name} requires demonstrated actions and sampled noise")
    target = flow_matching_target(noise, actions)
    return masked_mse_loss(student_output.velocity, target, action_mask)


def compute_batch_loss(
    stage: StageName | StageConfig,
    student_output: StudentOutput,
    batch: StudentBatch,
    *,
    noise: Optional[Tensor] = None,
    teacher_targets: Optional[TeacherConditioningTargets | ConditioningCache] = None,
    teacher_velocity: Optional[Tensor] = None,
    feature_weight: float = 1.0,
    velocity_weight: float = 1.0,
) -> Tensor:
    """Convenience wrapper using a :class:`StudentBatch` action target."""

    return compute_stage_loss(
        stage,
        student_output,
        actions=batch.actions,
        noise=noise,
        action_mask=batch.action_mask,
        teacher_targets=teacher_targets,
        teacher_velocity=teacher_velocity,
        feature_weight=feature_weight,
        velocity_weight=velocity_weight,
    )
