"""Deterministic, policy-independent evaluation plans.

The two production evaluators keep their native execution loops, but they must
agree on the cells they claim to evaluate.  This module is intentionally
dependency-light: it only serializes the policy/condition contract, the
row-major task/episode/seed schedule, and the configured randomization payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    DEFAULT_RESOLUTION,
    DEFAULT_SEED_BASE,
    DEFAULT_TASK_IDS,
    EvaluationCell,
    EvaluationCondition,
    build_task_seed_matrix,
)
from .randomization_contract import randomization_config_payload
from .registry import validate_policy_condition


PLAN_SCHEMA = "shared_evaluation_plan.v1"
SCHEDULE_SCHEMA = "shared_evaluation_schedule.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_source_hashes(repo_root: str | Path | None = None) -> dict[str, str | None]:
    """Hash the source files that define the shared evaluation contract.

    Policy-specific source hashes remain in each native manifest.  These four
    hashes are deliberately identical for matched controller/VLA plans and
    make it possible to tell whether the common contract changed.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    paths = (
        "vla_benchmarking/evaluation/contracts.py",
        "vla_benchmarking/evaluation/plan.py",
        "vla_benchmarking/evaluation/registry.py",
        "vla_benchmarking/evaluation/randomization_contract.py",
        "vla_benchmarking/shared/config.py",
    )
    return {relative: _sha256_file(root / relative) for relative in paths}


def _schedule_cells(cells: Sequence[EvaluationCell]) -> list[dict[str, int]]:
    return [
        {
            "cell_index": int(cell.cell_index),
            "task_id": int(cell.task_id),
            "episode_index": int(cell.episode_index),
            "seed": int(cell.seed),
            "init_state_index": int(cell.init_state_index),
        }
        for cell in cells
    ]


def schedule_hash(cells: Sequence[EvaluationCell] | Sequence[Mapping[str, Any]]) -> str:
    """Hash only row-major cell identity, independent of policy or inputs."""

    normalized: list[dict[str, int]] = []
    for position, cell in enumerate(cells):
        if isinstance(cell, EvaluationCell):
            normalized.append(
                {
                    "cell_index": int(cell.cell_index),
                    "task_id": int(cell.task_id),
                    "episode_index": int(cell.episode_index),
                    "seed": int(cell.seed),
                    "init_state_index": int(cell.init_state_index),
                }
            )
            continue
        normalized.append(
            {
                "cell_index": int(cell.get("cell_index", position)),
                "task_id": int(cell["task_id"]),
                "episode_index": int(cell["episode_index"]),
                "seed": int(cell["seed"]),
                "init_state_index": int(cell.get("init_state_index", cell.get("episode_index", 0))),
            }
        )
    return canonical_sha256({"schema": SCHEDULE_SCHEMA, "cells": normalized})


def _prompt_applicability(policy_kind: str, condition: EvaluationCondition) -> str:
    capabilities = validate_policy_condition(policy_kind, condition)
    if hasattr(capabilities, "prompt_applicability"):
        value = capabilities.prompt_applicability.get(condition.suite_mode, "not_applicable")
        return str(value)
    return "not_applicable"


def build_evaluation_plan(
    *,
    policy_kind: str,
    suite_mode: str,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed_base: int = DEFAULT_SEED_BASE,
    camera: str = "agentview",
    resolution: int = DEFAULT_RESOLUTION,
    text_context: str = "none",
    text_format: str | None = None,
    visual_input: str = "none",
    visual_arrow: bool | None = None,
    randomization: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build the single serialized plan consumed by either native evaluator."""

    condition = EvaluationCondition(
        suite_mode=suite_mode,
        camera=camera,
        resolution=resolution,
        text_context=text_context,
        text_format=text_format,
        visual_input=visual_input,
        visual_arrow=visual_arrow,
    )
    capabilities = validate_policy_condition(policy_kind, condition)
    cells = build_task_seed_matrix(
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        seed_base=seed_base,
        suite_mode=condition.suite_mode,
        resolution=condition.resolution,
        camera=condition.camera,
        text_context=condition.text_context,
        text_format=condition.text_format,
        visual_input=condition.visual_input,
    )
    payload = dict(randomization if randomization is not None else randomization_config_payload())
    schedule = _schedule_cells(cells)
    plan = {
        "schema": PLAN_SCHEMA,
        "policy_kind": str(policy_kind),
        "condition": condition.as_dict(),
        "text_contract": capabilities.text_contract,
        "prompt_applicability": _prompt_applicability(policy_kind, condition),
        "randomization": {
            "payload": payload,
            "sha256": canonical_sha256(payload),
            "enabled": condition.suite_mode == "sealed_randomized",
        },
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "cells": schedule,
            "sha256": schedule_hash(cells),
        },
        "source_hashes": dict(sorted((source_hashes or shared_source_hashes()).items())),
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized plan and its internal hashes before execution."""

    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported evaluation plan schema: {plan.get('schema')!r}")
    schedule = plan.get("schedule")
    if not isinstance(schedule, Mapping) or schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("evaluation plan schedule schema is invalid")
    cells = schedule.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("evaluation plan must contain non-empty schedule cells")
    if schedule.get("sha256") != schedule_hash(cells):
        raise ValueError("evaluation plan schedule hash is invalid")
    randomization = plan.get("randomization")
    if not isinstance(randomization, Mapping) or randomization.get("sha256") != canonical_sha256(randomization.get("payload")):
        raise ValueError("evaluation plan randomization hash is invalid")
    expected = dict(plan)
    observed_hash = expected.pop("sha256", None)
    if observed_hash != canonical_sha256(expected):
        raise ValueError("evaluation plan hash is invalid")
    return dict(plan)


def validate_native_schedule(
    plan: Mapping[str, Any], native_cells: Sequence[Mapping[str, Any]]
) -> None:
    """Fail closed if a native runner's planned task/seed schedule drifts."""

    checked = validate_plan(plan)
    expected = checked["schedule"]["cells"]
    actual = [
        {
            "cell_index": int(cell.get("cell_index", position)),
            "task_id": int(cell["task_id"]),
            "episode_index": int(cell["episode_index"]),
            "seed": int(cell["seed"]),
            "init_state_index": int(cell.get("init_state_index", cell.get("episode_index", 0))),
        }
        for position, cell in enumerate(native_cells)
    ]
    if actual != expected:
        raise ValueError("native planned cell schedule differs from shared evaluation plan")


__all__ = [
    "PLAN_SCHEMA",
    "SCHEDULE_SCHEMA",
    "canonical_json",
    "canonical_sha256",
    "shared_source_hashes",
    "schedule_hash",
    "build_evaluation_plan",
    "validate_plan",
    "validate_native_schedule",
]
