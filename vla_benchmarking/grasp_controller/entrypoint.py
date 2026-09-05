"""Single operator entrypoint for the frozen canonical grasp policy.

Only operational output and execution mode are exposed.  Policy, model,
camera, prompt, candidate geometry, opening, retreat, and evaluator timing
are immutable in the canonical implementation and configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .policy import (
    CANONICAL_POLICY_ID,
    MOLMOPOINT_MODEL_ID,
    MOLMOPOINT_MODEL_REVISION,
    MOLMOPOINT_PROMPT_ID,
)
from . import runner as _engine


def build_engine_argv(
    *, output_dir: Path, label: str = "canonical_grasp_controller",
    task_ids: str = "0,1,2,3,4,5,6,7,8,9",
    episodes_per_task: int = 10, seed_base: int = 1000,
    suite_modes: str = "sealed_randomized", dry_run: bool = False,
) -> list[str]:
    """Build the private-engine command from the immutable policy."""

    args = [
        "--variant", "canonical",
        "--output-dir", str(output_dir),
        "--label", label,
        "--phase", "full60",
        "--task-ids", task_ids,
        "--episodes-per-task", str(episodes_per_task),
        "--seed-base", str(seed_base),
        "--suite-modes", suite_modes,
        "--region-backend", "rgbd",
        "--motion-profile", "release20_retreat80mm",
        "--observation-profile", "parked",
        "--opening-profile", "preshape40mm",
        "--molmopoint-model", MOLMOPOINT_MODEL_ID,
        "--molmopoint-revision", MOLMOPOINT_MODEL_REVISION,
        "--molmopoint-prompt-id", MOLMOPOINT_PROMPT_ID,
    ]
    if (
        task_ids == "0,1,2,3,4,5,6,7,8,9"
        and episodes_per_task == 10
        and suite_modes == "sealed_randomized"
    ):
        args.extend(["--sealed-100", "--sealed-100-profile", CANONICAL_POLICY_ID])
    if dry_run:
        args.append("--dry-run")
    return args


def run(
    *, output_dir: Path, label: str = "canonical_grasp_controller",
    task_ids: str = "0,1,2,3,4,5,6,7,8,9", episodes_per_task: int = 10,
    seed_base: int = 1000, suite_modes: str = "sealed_randomized",
    dry_run: bool = False,
    cell_completed_callback: Callable[[Mapping[str, Any]], Any] | None = None,
    execution_cell_identities_by_suite: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> int:
    """Run one immutable policy with injectable runtime/test seams."""
    return _engine.main(
        build_engine_argv(
            output_dir=output_dir, label=label, task_ids=task_ids,
            episodes_per_task=episodes_per_task, seed_base=seed_base,
            suite_modes=suite_modes, dry_run=dry_run,
        ),
        cell_completed_callback=cell_completed_callback,
        execution_cell_identities_by_suite=execution_cell_identities_by_suite,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    cell_completed_callback: Callable[[Mapping[str, Any]], Any] | None = None,
    execution_cell_identities: (
        Mapping[str, Sequence[tuple[int, int]]] | Sequence[tuple[int, int]] | None
    ) = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="canonical_grasp_controller")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--suite-modes", default="sealed_randomized")
    parser.add_argument("--dry-run", action="store_true", help="validate policy/config without motion")
    args = parser.parse_args(argv)
    suites = tuple(item.strip() for item in args.suite_modes.split(",") if item.strip())
    if execution_cell_identities is None:
        identities_by_suite = None
    elif isinstance(execution_cell_identities, Mapping):
        identities_by_suite = execution_cell_identities
    else:
        identities_by_suite = {suite: tuple(execution_cell_identities) for suite in suites}
    return run(
        output_dir=args.output_dir, label=args.label, task_ids=args.task_ids,
        episodes_per_task=args.episodes_per_task, seed_base=args.seed_base,
        suite_modes=args.suite_modes, dry_run=args.dry_run,
        cell_completed_callback=cell_completed_callback,
        execution_cell_identities_by_suite=identities_by_suite,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_engine_argv", "main", "run"]
