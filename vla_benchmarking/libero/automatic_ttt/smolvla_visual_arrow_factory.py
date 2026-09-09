"""Separate live collector factory for the visual-goal-arrow SmolVLA policy."""

from __future__ import annotations

from typing import Any, Mapping

from .smolvla_arrow_factory import (
    DEFAULT_ARROW_STEP_BUDGET,
    DEFAULT_RESOLUTION,
    DEFAULT_VLA_STEP_BUDGET,
    build_collection_factory,
)


def collect(**kwargs: Any) -> Mapping[str, Any]:
    """Collect fresh Arrow actions with arrows in student-facing agentview RGB."""

    return build_collection_factory(
        resolution=int(kwargs.pop("resolution", DEFAULT_RESOLUTION)),
        vla_step_budget=int(kwargs.pop("vla_step_budget", DEFAULT_VLA_STEP_BUDGET)),
        arrow_step_budget=int(kwargs.pop("arrow_step_budget", DEFAULT_ARROW_STEP_BUDGET)),
        collection_mode=str(kwargs.pop("collection_mode", "fresh_arrow")),
        student_visual_condition="visual_goal_arrow",
    )(**kwargs)


__all__ = ["collect"]
