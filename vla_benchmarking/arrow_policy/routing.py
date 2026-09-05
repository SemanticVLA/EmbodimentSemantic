"""Action-expert cache routing with explicit parity and RoPE ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .contracts import ConditioningCache
from .teacher import apply_rope


@dataclass(frozen=True)
class ExpertLayerConditioning:
    """One expert layer's prefix K/V and attention metadata."""

    layer_index: int
    attention_mode: str
    key: Tensor
    value: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    keys_are_rotated: bool


def route_expert_cache(
    cache: ConditioningCache,
    *,
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
    self_attn_every_n_layers: int = 2,
) -> Tuple[ExpertLayerConditioning, ...]:
    """Route an unrotated bridge cache to all expert layers.

    SmolVLA alternates shared-self and cross-attention interfaces with
    ``self_attn_every_n_layers=2``. This helper labels that parity explicitly,
    carries the same masks and positions to every layer, and applies rotary
    position encoding exactly once. Calling it with an already rotated cache
    and RoPE factors is rejected rather than silently double-rotating keys.
    """

    if self_attn_every_n_layers <= 0:
        raise ValueError("self_attn_every_n_layers must be positive")
    if cache.keys_are_rotated and (rope_cos is not None or rope_sin is not None):
        raise ValueError("RoPE would be applied twice to this cache")
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be supplied together")

    keys = cache.keys
    rotated = cache.keys_are_rotated
    if not rotated and rope_cos is not None:
        keys = tuple(apply_rope(key, rope_cos, rope_sin) for key in keys)
        rotated = True
    result: list[ExpertLayerConditioning] = []
    for layer_index, (key, value) in enumerate(zip(keys, cache.values)):
        mode = (
            "shared_self_attention"
            if layer_index % self_attn_every_n_layers == 0
            else "cross_attention"
        )
        result.append(
            ExpertLayerConditioning(
                layer_index=layer_index,
                attention_mode=mode,
                key=key,
                value=value,
                attention_mask=cache.attention_mask,
                position_ids=cache.position_ids,
                keys_are_rotated=rotated,
            )
        )
    return tuple(result)
