"""Pinned dimensions and explicit staged optimization configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import torch
from torch import nn


class StageName(str, Enum):
    A_DISTILL_BRIDGE = "A_distill_bridge"
    B_ACTION_BRIDGE = "B_action_bridge"
    C_JOINT_ACTION = "C_joint_action"
    D_FULL_STUDENT = "D_full_student"


@dataclass(frozen=True)
class ArrowPolicyConfig:
    """The pinned SmolVLA arrow-prefix contract."""

    image_tokens: int = 64
    image_width: int = 960
    goal_tokens: int = 8
    state_dim: int = 32
    bridge_width: int = 480
    bridge_heads: int = 6
    bridge_ffn_width: int = 1920
    bridge_layers: int = 4
    expert_layers: int = 32
    expert_kv_heads: int = 5
    expert_head_dim: int = 64
    action_dim: int = 7
    action_chunk: int = 50
    flow_steps: int = 10


@dataclass(frozen=True)
class StageConfig:
    name: StageName
    updates: int
    trainable_components: tuple[str, ...]
    objective: str
    learning_rate: float


STAGE_SCHEDULE: tuple[StageConfig, ...] = (
    StageConfig(
        StageName.A_DISTILL_BRIDGE,
        5_000,
        ("bridge",),
        "feature_distillation + teacher_velocity_distillation",
        1e-4,
    ),
    StageConfig(StageName.B_ACTION_BRIDGE, 5_000, ("bridge",), "flow_matching_action", 1e-4),
    StageConfig(
        StageName.C_JOINT_ACTION,
        10_000,
        ("bridge", "action_expert", "state_projection", "action_in_projection", "action_out_projection", "time_projection"),
        "flow_matching_action",
        5e-5,
    ),
    StageConfig(
        StageName.D_FULL_STUDENT,
        10_000,
        ("vision_encoder", "visual_connector", "bridge", "action_expert", "state_projection", "action_in_projection", "action_out_projection", "time_projection"),
        "flow_matching_action",
        5e-5,
    ),
)


@dataclass(frozen=True)
class OptimizerConfig:
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.05
    final_lr_fraction: float = 0.10
    bridge_lr: float = 1e-4
    expert_lr: float = 1e-5
    vision_lr: float = 1e-6


_ALIASES = {
    "state_proj": "state_projection",
    "action_in": "action_in_projection",
    "action_out": "action_out_projection",
    "time_proj": "time_projection",
    "vision": "vision_encoder",
    "connector": "visual_connector",
}


def _top_level(name: str) -> str:
    return name.split(".", 1)[0]


def apply_stage_trainability(
    model: nn.Module, stage: StageName | StageConfig, *, strict: bool = False
) -> tuple[str, ...]:
    """Set ``requires_grad`` from the explicit stage inventory.

    The model is expected to expose the retained student components as
    top-level modules. Unknown components are skipped to permit lightweight
    adapters, but a stage with no matching parameters raises a useful error.
    """

    config = next((item for item in STAGE_SCHEDULE if item.name == stage), None) if isinstance(stage, StageName) else stage
    if config is None:
        raise ValueError(f"unknown stage: {stage}")
    enabled = {_ALIASES.get(component, component) for component in config.trainable_components}
    if strict and config.name in {StageName.C_JOINT_ACTION, StageName.D_FULL_STUDENT}:
        top_levels = {_top_level(name) for name, _ in model.named_parameters()}
        has_nested_state = any(name.startswith("bridge.state_projection.") for name, _ in model.named_parameters())
        required = set(enabled)
        if "state_projection" in required and has_nested_state:
            required.remove("state_projection")
        missing = sorted(required - top_levels)
        if "state_projection" in enabled and "state_projection" not in top_levels and not has_nested_state:
            missing.append("state_projection")
        if missing:
            raise ValueError(
                f"stage {config.name} requires retained student components: {missing}"
            )
    matched: list[str] = []
    for name, parameter in model.named_parameters():
        top_level = _top_level(name)
        component_enabled = top_level in enabled
        # The retained 32 -> 960 state projection lives inside the bridge in
        # the standalone implementation. It is a distinct retained module:
        # bridge-only stages train the new bridge adapters, while C/D may tune
        # this old projection with the state/action projection inventory.
        if name.startswith("bridge.state_projection."):
            component_enabled = "state_projection" in enabled
        parameter.requires_grad = component_enabled
        if parameter.requires_grad:
            matched.append(name)
    if not matched:
        raise ValueError(f"stage {config.name} matched no model parameters: {sorted(enabled)}")
    return tuple(matched)


def build_stage_optimizer(
    model: nn.Module,
    stage: StageName | StageConfig,
    *,
    optimizer_config: OptimizerConfig = OptimizerConfig(),
    strict: bool = True,
) -> torch.optim.Optimizer:
    """Create AdamW over the stage's trainable parameters only."""

    config = next((item for item in STAGE_SCHEDULE if item.name == stage), None) if isinstance(stage, StageName) else stage
    if config is None:
        raise ValueError(f"unknown stage: {stage}")
    apply_stage_trainability(model, config, strict=strict)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or ".norm" in name or "norm." in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    # Learning rates follow the staged experiment contract. Bridge LR changes
    # from 1e-4 in A/B to 5e-5 in C/D; retained expert/projection modules use
    # 1e-5 and vision/connector use 1e-6 when they are unfrozen.
    def parameter_lr(name: str) -> float:
        top = _top_level(name)
        if top in {"vision_encoder", "visual_connector"}:
            return optimizer_config.vision_lr
        if top in {
            "action_expert",
            "state_projection",
            "action_in_projection",
            "action_out_projection",
            "time_projection",
        } or name.startswith("bridge.state_projection."):
            return optimizer_config.expert_lr
        return optimizer_config.bridge_lr if config.name in {
            StageName.A_DISTILL_BRIDGE,
            StageName.B_ACTION_BRIDGE,
        } else config.learning_rate

    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        weight_decay = 0.0 if name.endswith("bias") or ".norm" in name or "norm." in name else optimizer_config.weight_decay
        grouped.setdefault((parameter_lr(name), weight_decay), []).append(parameter)
    groups = [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), parameters in sorted(grouped.items())
    ]
    return torch.optim.AdamW(groups, lr=optimizer_config.bridge_lr, betas=optimizer_config.betas, eps=optimizer_config.eps)
