"""Small, dependency-free data contracts for the arrow policy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ConditioningCache:
    """Prefix K/V conditioning for the 32-layer action expert.

    Keys and values are deliberately kept attached to the bridge computation
    graph.  ``frozen=True`` prevents replacing fields while still allowing
    autograd to train the bridge through a frozen action expert.

    Every layer has layout ``[batch, prefix_tokens, kv_heads, head_dim]``.
    The prefix is exactly 64 scene tokens, 8 visual-goal tokens and one state
    token.  Keys are unrotated until :func:`route_expert_cache` applies RoPE.
    """

    keys: Tuple[Tensor, ...]
    values: Tuple[Tensor, ...]
    attention_mask: Tensor
    position_ids: Tensor
    keys_are_rotated: bool = False

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must contain the same layer count")
        if not self.keys:
            raise ValueError("conditioning cache cannot be empty")
        if self.attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, prefix_tokens]")
        if self.position_ids.ndim not in (1, 2):
            raise ValueError("position_ids must have shape [prefix_tokens] or [batch, prefix_tokens]")
        batch, tokens = self.keys[0].shape[:2]
        if tokens != 73:
            raise ValueError(f"SmolVLA arrow prefix must contain 73 tokens, got {tokens}")
        if self.attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask does not match cache batch/token dimensions")
        if self.position_ids.shape[-1] != tokens:
            raise ValueError("position_ids does not match cache token dimensions")
        for key, value in zip(self.keys, self.values):
            if key.shape != value.shape:
                raise ValueError("each key/value pair must have matching shapes")
            if key.ndim != 4:
                raise ValueError("cache tensors must have shape [batch, tokens, heads, head_dim]")
            if key.shape[:2] != (batch, tokens):
                raise ValueError("all cache layers must share batch/token dimensions")

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def num_tokens(self) -> int:
        return self.keys[0].shape[1]


@dataclass(frozen=True)
class StudentBatch:
    """A single arrowed observation and demonstration action target."""

    image_tokens: Tensor
    state: Tensor
    actions: Optional[Tensor] = None
    action_mask: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.image_tokens.ndim != 3 or self.image_tokens.shape[1:] != (64, 960):
            raise ValueError("image_tokens must have shape [batch, 64, 960]")
        if self.state.ndim != 2 or self.state.shape[0] != self.image_tokens.shape[0]:
            raise ValueError("state must have shape [batch, state_dim]")
        if self.actions is not None and self.actions.ndim != 3:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        if self.action_mask is not None and self.actions is not None:
            if self.action_mask.shape != self.actions.shape[:2]:
                raise ValueError("action_mask must have shape [batch, horizon]")


@dataclass(frozen=True)
class StudentOutput:
    """Outputs returned by a student policy adapter."""

    velocity: Tensor
    conditioning: ConditioningCache


@dataclass(frozen=True)
class TeacherConditioningTargets:
    """Teacher K/V tensors aligned to the student's 73-token prefix."""

    conditioning: ConditioningCache
    scene_mask: Tensor
    goal_mask: Tensor
    state_mask: Tensor

    def __post_init__(self) -> None:
        if self.scene_mask.shape[-1] != 64:
            raise ValueError("teacher scene mask must contain 64 entries")
        if self.goal_mask.shape[-1] != 8:
            raise ValueError("teacher goal mask must contain 8 entries")
        if self.state_mask.shape[-1] != 1:
            raise ValueError("teacher state mask must contain one entry")
