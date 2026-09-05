"""Language-free student wrapper and explicit action-expert adapter contract."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from torch import Tensor, nn

from .bridge import ArrowConditioningBridge
from .contracts import StudentOutput
from .routing import ExpertLayerConditioning, route_expert_cache


class ActionExpertAdapter(Protocol):
    """Minimal adapter expected from a pinned SmolVLA action expert.

    The adapter owns the original even/odd attention implementation. It must
    consume the routed 73-token cache without importing tokenizer/text APIs.
    """

    def __call__(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        conditioning: Tuple[ExpertLayerConditioning, ...],
    ) -> Tensor:
        ...


class ArrowVisualStudent(nn.Module):
    """Bridge + retained expert with an image/state-only inference signature."""

    def __init__(self, bridge: ArrowConditioningBridge, action_expert: nn.Module) -> None:
        super().__init__()
        self.bridge = bridge
        self.action_expert = action_expert

    def forward(
        self,
        image_tokens: Tensor,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        *,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> StudentOutput:
        bridge_output = self.bridge(image_tokens, state)
        routed = route_expert_cache(
            bridge_output.conditioning, rope_cos=rope_cos, rope_sin=rope_sin
        )
        # The adapter receives live routed tensors so bridge-only training can
        # propagate gradients through a frozen expert. The adapter owns no
        # tokenizer/text path and receives RoPE'd keys only when factors were
        # supplied to this method.
        velocity = self.action_expert(noisy_actions, timestep, routed)
        return StudentOutput(velocity=velocity, conditioning=bridge_output.conditioning)
