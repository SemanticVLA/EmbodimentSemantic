"""Cheap CPU smoke check using dummy tensors; no external model is loaded."""

from __future__ import annotations

import torch
from torch import nn

from .bridge import ArrowConditioningBridge
from .routing import route_expert_cache


def run_smoke(*, device: str = "cpu") -> dict[str, object]:
    """Run one forward/backward pass through a small bridge configuration."""

    bridge = ArrowConditioningBridge(bridge_layers=1, expert_layers=2).to(device)
    image = torch.randn(1, 64, 960, device=device)
    state = torch.randn(1, 32, device=device)
    output = bridge(image, state)
    routed = route_expert_cache(output.conditioning)
    # An action-expert-shaped probe verifies the graph remains live.
    probe = nn.Linear(5 * 64 * 2, 1, bias=False).to(device)
    probe_input = torch.cat(
        (routed[0].key.mean(1).flatten(1), routed[0].value.mean(1).flatten(1)), dim=-1
    )
    loss = probe(probe_input).square().mean()
    loss.backward()
    bridge_grad = float(bridge.goal_queries.grad.abs().sum().item())
    return {
        "prefix_shape": tuple(output.prefix_tokens.shape),
        "cache_layers": output.conditioning.num_layers,
        "cache_tokens": output.conditioning.num_tokens,
        "routed_modes": tuple(item.attention_mode for item in routed),
        "bridge_goal_gradient_l1": bridge_grad,
    }


if __name__ == "__main__":
    print(run_smoke())
