"""Explicit backend capabilities for shared evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import VISUAL_INPUTS


@dataclass(frozen=True)
class BackendCapabilities:
    """Describe policy-specific boundaries without sharing action loops."""

    name: str
    kind: Literal["direct_matrix", "lerobot"]
    supports_text_context: bool
    supports_visual_arrow: bool
    evaluator_timing: str


DIRECT_MATRIX = BackendCapabilities(
    name="direct_matrix", kind="direct_matrix", supports_text_context=False,
    supports_visual_arrow=True, evaluator_timing="after_completed_retreat",
)
LEROBOT = BackendCapabilities(
    name="lerobot", kind="lerobot", supports_text_context=True,
    supports_visual_arrow=True, evaluator_timing="policy_defined",
)


def validate_backend_condition(
    capabilities: BackendCapabilities,
    *,
    text_context: str,
    visual_arrow: bool | None = None,
    visual_input: str | None = None,
) -> None:
    if visual_input is None:
        visual_input = "relation_arrows" if visual_arrow else "none"
    if visual_input not in VISUAL_INPUTS:
        raise ValueError(f"unknown visual input: {visual_input!r}")
    if visual_arrow is not None and bool(visual_arrow) != (visual_input != "none"):
        raise ValueError("visual_arrow and visual_input disagree")
    if text_context != "none" and not capabilities.supports_text_context:
        raise ValueError(f"{capabilities.name} backend does not support text context")
    if visual_input != "none" and not capabilities.supports_visual_arrow:
        raise ValueError(f"{capabilities.name} backend does not support visual arrows")


__all__ = ["BackendCapabilities", "DIRECT_MATRIX", "LEROBOT", "validate_backend_condition"]
