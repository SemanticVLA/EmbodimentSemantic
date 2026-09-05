"""Stable evaluation contracts shared by all LIBERO policy adapters.

This module deliberately contains no simulator, model, or motion code.  It
keeps the common condition and accounting semantics in one place while the
canonical grasp and LeRobot/ArrowStudent backends retain their distinct
execution loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_TASK_IDS = tuple(range(10))
DEFAULT_EPISODES_PER_TASK = 10
DEFAULT_SEED_BASE = 1000
DEFAULT_RESOLUTION = 256
SUITE_MODES = ("vanilla", "sealed_randomized")
PUBLIC_SUITE_MODES = ("normal", "sealed_randomized")
VISUAL_INPUTS = ("none", "goal_arrow", "relation_arrows")
TEXT_CONTEXTS = ("none", "scene_graph", "text_triplet")


def parse_suite_mode(value: str) -> str:
    mode = str(value).strip()
    if mode == "normal":
        # Keep historical native manifests byte-compatible: the public name is
        # ``normal``, while the direct evaluator has always recorded
        # ``vanilla``.
        mode = "vanilla"
    if mode not in SUITE_MODES:
        raise ValueError(f"suite_mode must be one of {', '.join(SUITE_MODES)}; got {value!r}")
    return mode


@dataclass(frozen=True)
class EvaluationCondition:
    """Inputs that define a comparable evaluation condition."""

    suite_mode: str = "vanilla"
    camera: str = "agentview"
    resolution: int = DEFAULT_RESOLUTION
    text_context: str = "none"
    text_format: str | None = None
    visual_input: str = "none"
    # Transitional input accepted by existing callers. It is normalized into
    # ``visual_input`` and remains in serialized manifests for compatibility.
    visual_arrow: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite_mode", parse_suite_mode(self.suite_mode))
        if int(self.resolution) <= 0:
            raise ValueError("resolution must be positive")
        if not self.camera:
            raise ValueError("camera must be non-empty")
        if self.text_context not in TEXT_CONTEXTS:
            raise ValueError("text_context must be none, scene_graph, or text_triplet")
        if self.text_context == "none" and self.text_format is not None:
            raise ValueError("text_format requires a non-none text_context")
        if self.text_format is not None and not str(self.text_format).strip():
            raise ValueError("text_format must be a non-empty name")
        visual_input = str(self.visual_input).strip()
        if self.visual_arrow is not None:
            legacy_input = "relation_arrows" if self.visual_arrow else "none"
            if visual_input not in {"none", legacy_input}:
                raise ValueError("visual_arrow and visual_input disagree")
            visual_input = legacy_input
        if visual_input not in VISUAL_INPUTS:
            raise ValueError(
                f"visual_input must be one of {', '.join(VISUAL_INPUTS)}"
            )
        object.__setattr__(self, "visual_input", visual_input)
        object.__setattr__(self, "visual_arrow", visual_input != "none")

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_mode": self.suite_mode,
            "camera": self.camera,
            "resolution": int(self.resolution),
            "text_context": self.text_context,
            "text_format": self.text_format,
            "visual_input": self.visual_input,
            "visual_arrow": bool(self.visual_arrow),
        }


@dataclass(frozen=True)
class EvaluationCell:
    """One deterministic task/episode/seed cell in a matrix."""

    cell_index: int
    task_id: int
    episode_index: int
    seed: int
    init_state_index: int
    condition: EvaluationCondition

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_index": int(self.cell_index),
            "task_id": int(self.task_id),
            "episode_index": int(self.episode_index),
            "seed": int(self.seed),
            "init_state_index": int(self.init_state_index),
            **self.condition.as_dict(),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Normalized result record used for aggregation and manifests."""

    cell: EvaluationCell
    success: bool
    terminal: bool = True
    actions: int | None = None
    failure_category: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.cell.as_dict(),
            "success": bool(self.success),
            "terminal": bool(self.terminal),
            "actions": self.actions,
            "failure_category": self.failure_category,
            "metadata": dict(self.metadata),
        }


def build_task_seed_matrix(
    *,
    task_ids: Iterable[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    seed_base: int = DEFAULT_SEED_BASE,
    suite_mode: str = "vanilla",
    resolution: int = DEFAULT_RESOLUTION,
    camera: str = "agentview",
    text_context: str = "none",
    text_format: str | None = None,
    visual_input: str = "none",
    visual_arrow: bool | None = None,
) -> list[EvaluationCell]:
    """Create the canonical row-major task/episode/seed schedule.

    ``init_state_index`` is intentionally the episode index, matching the
    existing LIBERO runner contract.  Policy backends may add richer runtime
    diagnostics, but must preserve this schedule identity.
    """

    condition = EvaluationCondition(
        suite_mode=suite_mode, camera=camera, resolution=resolution,
        text_context=text_context, text_format=text_format,
        visual_input=visual_input, visual_arrow=visual_arrow,
    )
    tasks = tuple(int(task) for task in task_ids)
    if not tasks:
        raise ValueError("task_ids must not be empty")
    if int(episodes_per_task) <= 0:
        raise ValueError("episodes_per_task must be positive")
    if int(seed_base) < 0:
        raise ValueError("seed_base must be non-negative")
    cells: list[EvaluationCell] = []
    for task_id in tasks:
        for episode_index in range(int(episodes_per_task)):
            cells.append(EvaluationCell(
                cell_index=len(cells), task_id=task_id,
                episode_index=episode_index, seed=int(seed_base) + episode_index,
                init_state_index=episode_index, condition=condition,
            ))
    return cells


def aggregate_results(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    """Aggregate normalized results without dropping terminal failures."""

    by_task: dict[int, dict[str, int]] = {}
    successes = terminal = running = 0
    for result in results:
        task = by_task.setdefault(int(result.cell.task_id), {"successes": 0, "planned": 0, "terminal": 0})
        task["planned"] += 1
        if result.success:
            successes += 1
            task["successes"] += 1
        if result.terminal:
            terminal += 1
            task["terminal"] += 1
        else:
            running += 1
    planned = len(results)
    return {
        "successes": successes,
        "planned": planned,
        "terminal": terminal,
        "running": running,
        "success_rate": (successes / planned) if planned else None,
        "per_task": {str(key): value for key, value in sorted(by_task.items())},
    }


__all__ = [
    "DEFAULT_TASK_IDS", "DEFAULT_EPISODES_PER_TASK", "DEFAULT_SEED_BASE",
    "DEFAULT_RESOLUTION", "SUITE_MODES", "PUBLIC_SUITE_MODES",
    "VISUAL_INPUTS", "TEXT_CONTEXTS", "EvaluationCondition",
    "EvaluationCell", "EvaluationResult", "aggregate_results",
    "build_task_seed_matrix", "parse_suite_mode",
]
