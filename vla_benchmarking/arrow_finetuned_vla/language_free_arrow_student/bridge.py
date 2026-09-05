"""Visual-arrow conditioning bridge.

The bridge replaces the language-transformer prefix supplied to SmolVLA's
action expert.  It consumes connector features, rather than importing a
particular vision implementation, so this module remains usable without
LeRobot or Transformers installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import ConditioningCache


def build_prefix_attention_mask(
    *, scene_tokens: int = 64, goal_tokens: int = 8, device: torch.device | None = None
) -> Tensor:
    """Return a bool MHA mask for the 64+8+1 arrow prefix.

    PyTorch bool attention masks use ``True`` for a blocked edge.  Scene and
    goal tokens attend to each other and within their own groups.  The state
    token reads all prefix tokens, while neither image nor goal tokens reads
    the state token.
    """

    total = scene_tokens + goal_tokens + 1
    mask = torch.ones((total, total), dtype=torch.bool, device=device)
    visual_end = scene_tokens + goal_tokens
    mask[:visual_end, :visual_end] = False
    mask[visual_end, :] = False
    return mask


class PreLNTransformerBlock(nn.Module):
    """Width-480 pre-layer-normalized transformer block."""

    def __init__(self, width: int = 480, heads: int = 6, ffn_width: int = 1920) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True, bias=True
        )
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        normalized = self.norm_attn(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class VisualGoalCrossAttention(nn.Module):
    """One learned-query cross-attention block producing visual goal tokens."""

    def __init__(self, width: int = 480, heads: int = 6, ffn_width: int = 1920) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.image_norm = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True, bias=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )

    def forward(self, queries: Tensor, image: Tensor) -> Tensor:
        q = self.query_norm(queries)
        memory = self.image_norm(image)
        update, _ = self.cross_attn(q, memory, memory, need_weights=False)
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


@dataclass(frozen=True)
class BridgeOutput:
    """Bridge output with token-level features and expert K/V cache."""

    prefix_tokens: Tensor
    conditioning: ConditioningCache


class ArrowConditioningBridge(nn.Module):
    """Map arrowed agentview connector features to SmolVLA expert conditioning.

    Args use the pinned user checkpoint contract: connector width 960, 64
    image tokens, robot state capacity 32, action-expert width 480, five K/V
    heads of dimension 64, and 32 expert layers.
    """

    image_width: int = 960
    width: int = 480
    scene_tokens: int = 64
    goal_tokens: int = 8
    state_dim: int = 32
    expert_layers: int = 32
    kv_heads: int = 5
    head_dim: int = 64
    bridge_layers: int = 4

    def __init__(
        self,
        *,
        image_width: int = 960,
        width: int = 480,
        scene_tokens: int = 64,
        goal_tokens: int = 8,
        state_dim: int = 32,
        expert_layers: int = 32,
        kv_heads: int = 5,
        head_dim: int = 64,
        bridge_layers: int = 4,
        heads: int = 6,
        ffn_width: int = 1920,
    ) -> None:
        super().__init__()
        if (scene_tokens, goal_tokens) != (64, 8):
            raise ValueError("the arrow experiment requires exactly 64 scene and 8 goal tokens")
        if width % heads or kv_heads * head_dim * 2 != 640:
            raise ValueError("invalid pinned SmolVLA width/head dimensions")
        self.image_width = image_width
        self.width = width
        self.scene_tokens = scene_tokens
        self.goal_tokens = goal_tokens
        self.state_dim = state_dim
        self.expert_layers = expert_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        self.image_projection = nn.Linear(image_width, width)
        # This is the retained SmolVLA state projection (32 -> VLM width).
        # ``state_to_width`` is the new language-free bridge adapter.
        self.state_projection = nn.Linear(state_dim, image_width)
        self.state_to_width = nn.Linear(image_width, width)
        # Fixed sinusoidal positions keep the bridge compatible with the
        # expert's absolute prefix positions without adding another learned
        # language-like embedding table.
        position = torch.arange(scene_tokens + goal_tokens + 1, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / width)
        )
        positional = torch.zeros(scene_tokens + goal_tokens + 1, width)
        positional[:, 0::2] = torch.sin(position[:, None] * frequencies[None])
        positional[:, 1::2] = torch.cos(position[:, None] * frequencies[None])
        self.register_buffer("fixed_position_embedding", positional, persistent=False)
        self.goal_queries = nn.Parameter(torch.randn(goal_tokens, width) * 0.02)
        self.goal_block = VisualGoalCrossAttention(width, heads, ffn_width)
        self.blocks = nn.ModuleList(
            [PreLNTransformerBlock(width, heads, ffn_width) for _ in range(bridge_layers)]
        )
        self.final_norm = nn.LayerNorm(width)
        # Each head emits concatenated K and V, with 5*64 dimensions each.
        self.kv_heads_projection = nn.ModuleList(
            [nn.Linear(width, kv_heads * head_dim * 2, bias=False) for _ in range(expert_layers)]
        )
        self.register_buffer(
            "prefix_attention_mask",
            build_prefix_attention_mask(),
            persistent=False,
        )
        self.register_buffer(
            "prefix_position_ids",
            torch.arange(scene_tokens + goal_tokens + 1, dtype=torch.long),
            persistent=False,
        )

    def forward(self, image_tokens: Tensor, state: Tensor) -> BridgeOutput:
        """Produce the language-free 73-token cache.

        This signature intentionally has no instruction/text argument.  An
        arrow is already embedded in ``image_tokens`` by the upstream visual
        preprocessing pipeline.
        """

        if image_tokens.ndim != 3 or image_tokens.shape[1:] != (
            self.scene_tokens,
            self.image_width,
        ):
            raise ValueError(
                f"image_tokens must have shape [batch, {self.scene_tokens}, {self.image_width}]"
            )
        if state.ndim != 2 or state.shape[0] != image_tokens.shape[0] or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [batch, {self.state_dim}]")

        image = self.image_projection(image_tokens) + self.fixed_position_embedding[: self.scene_tokens]
        queries = self.goal_queries.unsqueeze(0).expand(image.shape[0], -1, -1)
        queries = queries + self.fixed_position_embedding[self.scene_tokens : self.scene_tokens + self.goal_tokens]
        goals = self.goal_block(queries, image)
        state_token = self.state_to_width(self.state_projection(state)).unsqueeze(1)
        state_token = state_token + self.fixed_position_embedding[-1].view(1, 1, -1)
        prefix = torch.cat((image, goals, state_token), dim=1)
        for block in self.blocks:
            prefix = block(prefix, attention_mask=self.prefix_attention_mask)
        prefix = self.final_norm(prefix)

        keys: list[Tensor] = []
        values: list[Tensor] = []
        for projection in self.kv_heads_projection:
            kv = projection(prefix).view(
                prefix.shape[0], prefix.shape[1], self.kv_heads, self.head_dim * 2
            )
            keys.append(kv[..., : self.head_dim])
            values.append(kv[..., self.head_dim :])
        mask = torch.ones(
            prefix.shape[:2], dtype=torch.bool, device=prefix.device
        )
        positions = self.prefix_position_ids.to(prefix.device)
        return BridgeOutput(
            prefix_tokens=prefix,
            conditioning=ConditioningCache(
                keys=tuple(keys),
                values=tuple(values),
                attention_mask=mask,
                position_ids=positions,
                keys_are_rotated=False,
            ),
        )
