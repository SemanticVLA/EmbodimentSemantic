#!/usr/bin/env python3
"""Run a bounded LIBERO arrow-only pick/place episode matrix.

The launcher owns batch bookkeeping only.  Perception, one-arrow rendering,
depth/calibration handling, motion safety, and evaluator timing remain in
``run_arrow_pick_place_eval.run_episode``.  Motion is deliberately explicit:
the command must include either ``--dry-run`` or ``--execute-motion``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:  # Script and package-style imports are both useful in LIBERO checkouts.
    from . import run_arrow_pick_place_eval as _episode_module
except ImportError:  # pragma: no cover - direct ``python file.py`` execution
    import run_arrow_pick_place_eval as _episode_module


DEFAULT_TASK_IDS = tuple(range(10))
DEFAULT_EPISODES_PER_TASK = 10
DEFAULT_SEED_BASE = 1000
DEFAULT_RESOLUTION = 256
SUITE_MODES = ("vanilla", "sealed_randomized")
DEFAULT_SUITE_MODE = "vanilla"
DEFAULT_CONTROLLER_VARIANT = "rgbd_arrow_v1"
CAMERA_NAME = "agentview"
MANIFEST_JSONL_FILENAME = "arrow_pick_place_matrix_manifest.jsonl"
# Backwards-compatible descriptive alias for callers that used the first draft.
MANIFEST_FILENAME = MANIFEST_JSONL_FILENAME
SUMMARY_FILENAME = "arrow_pick_place_matrix_summary.json"
MANIFEST_JSON_FILENAME = "arrow_pick_place_matrix_manifest.json"
STATUS_FILENAME = "arrow_pick_place_matrix_status.json"
MATRIX_SCHEMA_VERSION = "arrow_pick_place_matrix.v1"
VERIFIED_TASK_ID = 0
VERIFIED_SEED = 1000
VERIFIED_RESOLUTION = 256


def _validated_label(value: str, *, name: str) -> str:
    """Return a stable path/contract label and reject ambiguous values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    label = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError(
            f"{name} must contain only letters, numbers, '.', '_' or '-': {value!r}"
        )
    return label


def parse_suite_mode(value: str) -> str:
    """Validate the explicit LIBERO scene condition."""
    mode = _validated_label(value, name="suite_mode")
    if mode not in SUITE_MODES:
        raise ValueError(
            f"suite_mode must be one of {', '.join(SUITE_MODES)}; got {value!r}"
        )
    return mode


def parse_controller_variant(value: str) -> str:
    """Validate a controller variant used in provenance and output paths."""
    return _validated_label(value, name="controller_variant")


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably replace a small JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _contract_hash(protocol: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> str:
    # Output roots are operational destinations, not experiment identity.
    identity_cells = [
        {key: value for key, value in cell.items() if key != "output_dir"}
        for cell in cells
    ]
    payload = {"protocol": protocol, "cells": identity_cells}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_hashes() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent
    # Keep this inventory explicit and relative to the package root.  The
    # matrix's suite/control contract depends on the BDDL filter, scene
    # randomizer, direct dual-suite launcher, arrow renderer/context, and
    # controller—not only on this bookkeeping module.  Missing files remain
    # represented as ``None`` so dependency-light dry runs retain a complete,
    # deterministic provenance schema rather than silently dropping a key.
    relative_paths = (
        "run_arrow_pick_place_matrix.py",
        "run_arrow_pick_place_eval.py",
        "arrow_controller.py",
        "config.py",
        "bddl_utils.py",
        "radomize_scenes.py",
        "visual_scene_graph.py",
        "libero_live_semantic_context.py",
        "preview_visual_arrows.py",
        "legion/run_arrow_pick_place_dual_matrix.sbatch",
    )
    return {
        relative_path: _sha256_file(root / relative_path)
        for relative_path in relative_paths
    }


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return {"head": None, "dirty": None}
    return {"head": head or None, "dirty": dirty}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "robosuite", "libero", "mujoco", "lerobot", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_task_ids(value: str | Sequence[int]) -> list[int]:
    """Parse a non-empty duplicate-free task-id list."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid task IDs: {value!r}") from exc
        else:
            parsed = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parsed = list(value)
    try:
        result = [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid task IDs: {value!r}") from exc
    if not result:
        raise ValueError("task IDs must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"task IDs contain duplicates: {result}")
    if any(task_id < 0 for task_id in result):
        raise ValueError("task IDs must be non-negative")
    return result


def plan_cells(
    *,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    seed_base: int = DEFAULT_SEED_BASE,
    output_root: str | Path = "arrow_pick_place_matrix_outputs",
    resolution: int = DEFAULT_RESOLUTION,
    suite_mode: str = DEFAULT_SUITE_MODE,
    controller_variant: str = DEFAULT_CONTROLLER_VARIANT,
) -> list[dict[str, Any]]:
    """Build deterministic task/episode cells without touching LIBERO."""
    tasks = parse_task_ids(task_ids)
    suite_mode = parse_suite_mode(suite_mode)
    controller_variant = parse_controller_variant(controller_variant)
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    root = Path(output_root).expanduser().resolve()
    cells: list[dict[str, Any]] = []
    cell_index = 0
    for task_id in tasks:
        for episode_index in range(int(episodes_per_task)):
            seed = int(seed_base) + episode_index
            # Keep condition labels in every episode path.  This prevents a
            # vanilla and randomized run from being accidentally mixed when a
            # caller shares a parent output directory.
            cell_dir = (
                root
                / suite_mode
                / controller_variant
                / f"task_{int(task_id)}"
                / f"episode_{episode_index}_seed_{seed}"
            )
            validated = (
                int(task_id) == VERIFIED_TASK_ID
                and seed == VERIFIED_SEED
                and int(resolution) == VERIFIED_RESOLUTION
                and suite_mode == DEFAULT_SUITE_MODE
            )
            cells.append(
                {
                    "cell_index": cell_index,
                    "task_id": int(task_id),
                    "episode_index": int(episode_index),
                    "seed": seed,
                    # Compatibility field retained for consumers that used
                    # the original planner.  It is only a candidate: the
                    # actual selected init state is recorded from the env.
                    "init_state_index": int(episode_index),
                    "init_state_index_candidate": int(episode_index),
                    "resolution": int(resolution),
                    "suite_mode": suite_mode,
                    "controller_variant": controller_variant,
                    "condition_label": f"{suite_mode}__{controller_variant}",
                    "output_dir": cell_dir.as_posix(),
                    "profile_label": "validated" if validated else "exploratory_unvalidated",
                    "profile_validated": bool(validated),
                }
            )
            cell_index += 1
    return cells


# Conventional matrix-builder alias for callers that use the existing
# evaluation launchers' vocabulary.
build_cells = plan_cells
build_plan = plan_cells


def validate_motion_authorization(
    cells: Sequence[Mapping[str, Any]], *, execute_motion: bool, dry_run: bool,
    allow_unvalidated_profile: bool,
) -> None:
    """Fail before environment construction when motion authorization is incomplete."""
    if dry_run:
        return
    if not execute_motion:
        raise ValueError("matrix motion requires explicit --execute-motion or dry-run")
    if not allow_unvalidated_profile:
        unvalidated = [cell for cell in cells if not bool(cell.get("profile_validated", False))]
        if unvalidated:
            first = unvalidated[0]
            raise ValueError(
                "--allow-unvalidated-profile is required for exploratory cells; "
                f"first cell task={first['task_id']} seed={first['seed']} "
                f"resolution={first['resolution']}"
            )


def _init_state_preflight(
    *, task_ids: Sequence[int], episodes_per_task: int, seed_base: int,
    injected_builder: bool,
) -> dict[str, Any]:
    """Validate deterministic LIBERO init-state coverage before env creation."""
    if injected_builder:
        return {
            "status": "not_applicable_injected_builder",
            "source": "caller_provided_env_builder",
            "required_count": int(episodes_per_task),
            "available_counts": {},
            "selected_indices": {},
            "fallback_allowed": False,
        }
    try:
        from libero.libero import benchmark
        from lerobot.envs.libero import get_task_init_states
        from config import BENCHMARK_NAME
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "init-state preflight requires LIBERO benchmark and "
            "lerobot.get_task_init_states; refusing fallback"
        ) from exc
    try:
        suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
        available_counts: dict[str, int] = {}
        selected_indices: dict[str, list[int]] = {}
        for task_id in task_ids:
            count = int(len(get_task_init_states(suite, int(task_id))))
            if count < int(episodes_per_task):
                raise RuntimeError(
                    f"init-state preflight task {int(task_id)} has {count} states "
                    f"but requires {int(episodes_per_task)}"
                )
            indices = [
                (int(seed_base) + int(episode_index)) % count
                for episode_index in range(int(episodes_per_task))
            ]
            if len(set(indices)) != len(indices):
                raise RuntimeError(
                    f"init-state preflight task {int(task_id)} selected indices "
                    "are not unique"
                )
            available_counts[str(int(task_id))] = count
            selected_indices[str(int(task_id))] = indices
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"init-state preflight failed: {exc}") from exc
    return {
        "status": "validated",
        "source": "lerobot.get_task_init_states",
        "required_count": int(episodes_per_task),
        "available_counts": available_counts,
        "selected_indices": selected_indices,
        "fallback_allowed": False,
    }


def _default_arrow_inputs(env: Any, task_id: int, resolution: int) -> dict[str, Any]:
    """Use the existing hardcoded subject→goal bbox renderer integration."""
    try:
        from libero_live_semantic_context import LiveSemanticContextGenerator
        from config import SCENE_GRAPH_SUBJECT_FILTER, TASK_GOAL_OBJECT_CONFIG
        from types import SimpleNamespace
    except ImportError as exc:  # pragma: no cover - live LIBERO dependency
        raise RuntimeError("LIBERO arrow input generation dependencies are unavailable") from exc
    generator = LiveSemanticContextGenerator()
    generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
    context_env = SimpleNamespace(
        _env=getattr(env, "env", env),
        observation_height=int(resolution),
        observation_width=int(resolution),
        task=getattr(env, "language_instruction", ""),
        task_id=int(task_id),
    )
    context = generator.observe_visual_graph(context_env, camera=CAMERA_NAME)
    return {
        "bboxes": context["bboxes"],
        "goal_object": TASK_GOAL_OBJECT_CONFIG.get(int(task_id), _episode_module.DEFAULT_GOAL_OBJECT),
        "subject": SCENE_GRAPH_SUBJECT_FILTER,
    }


def _protocol(
    *, task_ids: Sequence[int], episodes_per_task: int, seed_base: int,
    resolution: int, dry_run: bool, execute_motion: bool,
    allow_unvalidated_profile: bool,
    continue_on_motion_failure: bool,
    suite_mode: str,
    controller_variant: str,
    init_state_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": "libero_arrow_pick_place_matrix",
        "schema_version": MATRIX_SCHEMA_VERSION,
        "camera": CAMERA_NAME,
        "suite_mode": suite_mode,
        "controller_variant": controller_variant,
        "condition_label": f"{suite_mode}__{controller_variant}",
        "suite_contract": {
            "vanilla": "canonical_libero_spatial_bddl_without_custom_randomization",
            "sealed_randomized": (
                "repository_sealed_scene_removals_and_layout_swaps_only; "
                "prompt_variant_not_applicable_to_arrow_runner"
            ),
        },
        "task_ids": [int(task_id) for task_id in task_ids],
        "episodes_per_task": int(episodes_per_task),
        "seed_policy": "seed=seed_base+episode_index",
        "seed_base": int(seed_base),
        "resolution": int(resolution),
        "motion_mode": "dry_run" if dry_run else "execute_motion",
        "execute_motion_explicit": bool(execute_motion),
        "allow_unvalidated_profile": bool(allow_unvalidated_profile),
        "continue_on_motion_failure": bool(continue_on_motion_failure),
        "verified_profile": {
            "task_id": VERIFIED_TASK_ID,
            "seed": VERIFIED_SEED,
            "resolution": VERIFIED_RESOLUTION,
        },
        "init_state_policy": {
            "index": "env_reported_selected_index",
            "fallback": "none; never infer from episode_index",
            "source": "runner_seed_modulo_init_state_count",
            "uniqueness_requirement": "selected indices must be unique per task when reported",
        },
        "init_state_preflight": dict(init_state_preflight or {}),
        "arrow_relation": {
            "subject": _episode_module.DEFAULT_SUBJECT,
            "relationship": "goal",
            "object_policy": "existing_task_goal_config",
        },
        "arrow_generation": "existing_hardcoded_subject_goal_bbox_renderer",
        "language_translation": False,
        "prompt_variant": "not_applicable",
    }


def _failure_class(stage: str, exc: BaseException) -> str:
    """Classify a cell failure without hiding the original exception details."""
    message = str(exc).lower()
    if "profile" in message or "unvalidated" in message:
        return "profile_rejected"
    if "not confirmed settled" in message or "settle_physics" in message:
        return "environment_failure"
    if "evaluator" in message or "success" in message and "check" in message:
        return "evaluator_failure"
    if stage == "build_env" or stage == "close_env":
        return "environment_failure"
    if stage == "generate_arrow_inputs":
        return "input_failure"
    if any(token in message for token in ("depth", "capture", "arrow", "calibration", "pixel")):
        return "input_failure"
    if stage == "run_episode":
        return "controller_failure"
    return "unknown_failure"


def _error_record(
    cell: Mapping[str, Any], *, stage: str, exc: BaseException,
    motion_began: bool = False,
) -> dict[str, Any]:
    record = dict(cell)
    record.update({
        "status": "failed",
        "stage": stage,
        "failure_class": _failure_class(stage, exc),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "audit_path": None,
        "audit": None,
        "frames": [],
        "phase_frames": [],
        "phases": [],
        "evaluator_result": None,
        "diagnostics": {
            "frames": [],
            "phase_frames": [],
            "phases": [],
            "capture_contract": None,
            "endpoint_depths_m": None,
            "deprojected_visual_endpoint_world_points_m": None,
            "control_targets_world_m": None,
            "waypoints_world_m": None,
            "recovery": [],
            "evaluator_result": None,
            "settle_diagnostics": None,
        },
        "motion_began": bool(motion_began),
        "init_state_diagnostics": None,
        "settle_diagnostics": None,
    })
    return record


def _audit_diagnostics(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten important per-episode evidence while retaining the full audit."""
    return {
        "frames": audit.get("frames", []),
        "phase_frames": audit.get("phase_frames", []),
        "phases": audit.get("phases", []),
        "capture_contract": audit.get("capture_contract"),
        "endpoint_depths_m": audit.get("endpoint_depths_m"),
        "deprojected_visual_endpoint_world_points_m": audit.get(
            "deprojected_visual_endpoint_world_points_m"
        ),
        "control_targets_world_m": audit.get("control_targets_world_m"),
        "waypoints_world_m": audit.get("waypoints_world_m"),
        "recovery": audit.get("recovery", []),
        "evaluator_result": audit.get("evaluator_success"),
    }


def _partition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    evaluated = [_audit_success(record) for record in completed]
    evaluated = [value for value in evaluated if value is not None]
    successes = sum(value is True for value in evaluated)
    has_evaluator_result = bool(evaluated)
    return {
        "total": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "evaluated": len(evaluated),
        "successes": int(successes),
        # A dry-run has no evaluator denominator; do not turn its null result
        # into a misleading zero-rate outcome.
        "planned_success_rate": (float(successes) / len(records))
        if records and has_evaluator_result else None,
        "evaluable_success_rate": (float(successes) / len(evaluated)) if evaluated else None,
        "conservative_denominator": len(records),
        "conservative_success_rate": (float(successes) / len(records))
        if records and has_evaluator_result else None,
        "evaluable_denominator": len(evaluated),
    }


def _motion_began(env: Any, audit: Mapping[str, Any] | None = None) -> bool:
    if isinstance(audit, Mapping) and "motion_executed" in audit:
        return bool(audit["motion_executed"])
    marker = getattr(env, "_arrow_motion_began", None)
    if marker is not None:
        return bool(marker)
    actions = getattr(env, "actions", None)
    try:
        return len(actions) > 0
    except TypeError:
        return False


def _init_state_diagnostics(env: Any) -> Any:
    value = getattr(env, "_arrow_init_state_diagnostics", None)
    return dict(value) if isinstance(value, Mapping) else value


def _init_state_index(diagnostics: Any) -> int | None:
    if not isinstance(diagnostics, Mapping):
        return None
    for key in ("selected_index", "init_state_index", "state_index", "index"):
        if key in diagnostics:
            try:
                return int(diagnostics[key])
            except (TypeError, ValueError):
                return None
    return None


def _validate_observed_init_state(
    diagnostics: Any, *, cell: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    """Reject default-builder fallback or a mismatched selected state."""
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("environment did not report init-state diagnostics")
    if bool(diagnostics.get("fallback")):
        raise RuntimeError("environment used forbidden init-state fallback")
    try:
        count = int(diagnostics["available_count"])
        selected = int(diagnostics["selected_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("environment init-state diagnostics are incomplete") from exc
    if count < int(preflight.get("required_count", 0)):
        raise RuntimeError("environment reported insufficient init-state count")
    task_key = str(int(cell["task_id"]))
    expected_values = preflight.get("selected_indices", {}).get(task_key, [])
    episode_index = int(cell["episode_index"])
    if episode_index >= len(expected_values) or selected != int(expected_values[episode_index]):
        raise RuntimeError(
            f"environment selected init state {selected}, expected "
            f"{expected_values[episode_index] if episode_index < len(expected_values) else None}"
        )


def _planned_record(cell: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(cell)
    record.update({
        "status": "planned",
        "stage": "planning",
        "failure_class": None,
        "error_type": None,
        "error": None,
        "audit_path": None,
        "audit": None,
        "diagnostics": None,
        "motion_began": False,
        "duration_s": None,
        "attempts": [],
    })
    return record


def _audit_success(record: Mapping[str, Any]) -> bool | None:
    if bool(record.get("dry_run")):
        return None
    protocol = record.get("protocol")
    if isinstance(protocol, Mapping) and protocol.get("motion_mode") == "dry_run":
        return None
    audit = record.get("audit")
    if not isinstance(audit, Mapping):
        return None
    value = audit.get("evaluator_success")
    return value if isinstance(value, bool) else None


def _record_failure_class(record: Mapping[str, Any]) -> str | None:
    explicit = record.get("failure_class")
    if explicit:
        return str(explicit)
    result = _audit_success(record)
    if result is False:
        return "task_failure"
    return None


def _phase_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        audit = record.get("audit")
        if isinstance(audit, Mapping):
            phases = audit.get("phases", [])
        else:
            # Controller failures can occur before the final audit is written;
            # _run_motion still preserves partial phases at record level.
            phases = record.get("phases", [])
        if not isinstance(phases, list):
            continue
        for phase_record in phases:
            if not isinstance(phase_record, Mapping):
                continue
            phase = str(phase_record.get("phase", "unknown"))
            bucket = result.setdefault(phase, {"count": 0, "statuses": {}, "failed": 0})
            bucket["count"] += 1
            status = str(phase_record.get("status", "unknown"))
            bucket["statuses"][status] = bucket["statuses"].get(status, 0) + 1
            if status not in {"reached", "dwell", "dry_run", "dry_run_stop", "stop"}:
                bucket["failed"] += 1
    return result


def _diagnostic_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "capture": {"valid": 0, "missing": 0},
        "parser": {"success": 0, "failure": 0},
        "depth": {"recorded": 0, "failure": 0},
        "settle": {"recorded": 0, "settled": 0, "unsettled": 0, "max_final_velocity_m_s": None},
        "controller": {"success": 0, "failure": 0},
        "evaluator": {"true": 0, "false": 0, "null": 0, "error": 0},
    }
    for record in records:
        settle = record.get("settle_diagnostics")
        if isinstance(settle, Mapping):
            result["settle"]["recorded"] += 1
            if bool(settle.get("settled")):
                result["settle"]["settled"] += 1
            else:
                result["settle"]["unsettled"] += 1
            try:
                velocity = float(settle["final_max_velocity_m_s"])
            except (KeyError, TypeError, ValueError):
                velocity = None
            if velocity is not None:
                current = result["settle"]["max_final_velocity_m_s"]
                result["settle"]["max_final_velocity_m_s"] = (
                    velocity if current is None else max(float(current), velocity)
                )
        audit = record.get("audit")
        if isinstance(audit, Mapping):
            contract = audit.get("capture_contract")
            if isinstance(contract, Mapping) and contract.get("valid") is True:
                result["capture"]["valid"] += 1
            else:
                result["capture"]["missing"] += 1
            if audit.get("arrow_endpoints_uv") is not None:
                result["parser"]["success"] += 1
            else:
                result["parser"]["failure"] += 1
            if audit.get("endpoint_depths_m") is not None:
                result["depth"]["recorded"] += 1
            else:
                result["depth"]["failure"] += 1
        failure_class = _record_failure_class(record)
        if failure_class == "controller_failure":
            result["controller"]["failure"] += 1
        elif record.get("status") == "completed":
            result["controller"]["success"] += 1
        evaluator = _audit_success(record)
        if evaluator is True:
            result["evaluator"]["true"] += 1
        elif evaluator is False:
            result["evaluator"]["false"] += 1
        else:
            result["evaluator"]["null"] += 1
        if failure_class == "evaluator_failure" and record.get("status") != "completed":
            result["evaluator"]["error"] += 1
    return result


def _accepted_optional_keywords(function: Callable[..., Any]) -> set[str] | None:
    """Inspect a seam without requiring legacy injected test fakes to change.

    ``None`` means the callable accepts arbitrary keyword arguments.  A
    signature can be unavailable for some extension callables; in that case
    returning an empty set preserves the old positional contract.
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return set()
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return accepted


def _call_with_optional_keywords(
    function: Callable[..., Any],
    positional: Sequence[Any],
    optional: Mapping[str, Any],
) -> Any:
    """Call an injected seam while passing only supported condition metadata."""
    accepted = _accepted_optional_keywords(function)
    kwargs = dict(optional) if accepted is None else {
        key: value for key, value in optional.items() if key in accepted
    }
    return function(*positional, **kwargs)


def run_matrix(
    *,
    output_root: str | Path,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    seed_base: int = DEFAULT_SEED_BASE,
    resolution: int = DEFAULT_RESOLUTION,
    suite_mode: str = DEFAULT_SUITE_MODE,
    controller_variant: str = DEFAULT_CONTROLLER_VARIANT,
    dry_run: bool = False,
    execute_motion: bool = False,
    allow_unvalidated_profile: bool = False,
    env_builder: Callable[[int, int, int], Any] | None = None,
    episode_runner: Callable[..., Mapping[str, Any]] | None = None,
    arrow_input_builder: Callable[[Any, int, int], Mapping[str, Any]] | None = None,
    evaluator: Callable[[Any], bool] | None = None,
    resume: bool = False,
    retry_motion_began: bool = False,
    continue_on_motion_failure: bool = False,
) -> dict[str, Any]:
    """Execute every planned cell, isolating failures and preserving audits."""
    suite_mode = parse_suite_mode(suite_mode)
    controller_variant = parse_controller_variant(controller_variant)
    cells = plan_cells(
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        seed_base=seed_base,
        output_root=output_root,
        resolution=resolution,
        suite_mode=suite_mode,
        controller_variant=controller_variant,
    )
    validate_motion_authorization(
        cells,
        execute_motion=execute_motion,
        dry_run=dry_run,
        allow_unvalidated_profile=allow_unvalidated_profile,
    )
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_JSONL_FILENAME
    manifest_json_path = root / MANIFEST_JSON_FILENAME
    status_path = root / STATUS_FILENAME
    summary_path = root / SUMMARY_FILENAME
    build_env = env_builder or _episode_module.build_libero_env
    run_episode = episode_runner or _episode_module.run_episode
    build_inputs = arrow_input_builder or _default_arrow_inputs
    if env_builder is None:
        accepted_builder_keywords = _accepted_optional_keywords(build_env)
        if (
            accepted_builder_keywords is not None
            and "suite_mode" not in accepted_builder_keywords
        ):
            raise RuntimeError(
                "production LIBERO environment builder must accept suite_mode; "
                "use a condition-aware build_libero_env or inject a test builder"
            )
    evaluator_fn = evaluator
    if evaluator_fn is None and not dry_run:
        evaluator_fn = lambda candidate: bool(candidate.check_success())
    if evaluator_fn is not None:
        raw_evaluator = evaluator_fn

        def evaluator_fn(candidate: Any) -> bool:
            try:
                result = raw_evaluator(candidate)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                raise RuntimeError(f"evaluator failure: {exc}") from exc
            if not isinstance(result, bool):
                raise RuntimeError(
                    f"evaluator failure: expected bool, got {type(result).__name__}"
                )
            return result
    init_state_preflight = _init_state_preflight(
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        seed_base=seed_base,
        injected_builder=env_builder is not None,
    )
    protocol = _protocol(
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        seed_base=seed_base,
        resolution=resolution,
        dry_run=dry_run,
        execute_motion=execute_motion,
        allow_unvalidated_profile=allow_unvalidated_profile,
        continue_on_motion_failure=continue_on_motion_failure,
        suite_mode=suite_mode,
        controller_variant=controller_variant,
        init_state_preflight=init_state_preflight,
    )
    protocol["source_hashes"] = _source_file_hashes()
    contract_hash = _contract_hash(protocol, cells)
    provenance = {
        "launcher_path": Path(__file__).resolve().as_posix(),
        "repository_root": Path(__file__).resolve().parents[1].as_posix(),
        "episode_runner": "run_arrow_pick_place_eval.run_episode",
        "suite_mode": suite_mode,
        "controller_variant": controller_variant,
        "condition_label": f"{suite_mode}__{controller_variant}",
        "generated_unix": time.time(),
        "git": _git_provenance(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
    }
    planned_records = [_planned_record(cell) for cell in cells]
    if not resume and any(path.exists() for path in (manifest_json_path, manifest_path, status_path, summary_path)):
        raise FileExistsError(
            f"matrix output already exists at {root}; choose a new output root or pass resume=True"
        )
    previous_by_index: dict[int, dict[str, Any]] = {}
    if resume:
        if not status_path.is_file():
            raise FileNotFoundError(f"cannot resume without status file: {status_path}")
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("contract_hash") != contract_hash:
            raise ValueError("resume contract hash does not match the requested matrix")
        for record in previous.get("cells", []):
            if isinstance(record, Mapping) and "cell_index" in record:
                previous_by_index[int(record["cell_index"])] = dict(record)
        unsafe = [
            record for record in previous_by_index.values()
            if (
                record.get("status") == "running"
                or (
                    record.get("status") != "completed"
                    and bool(record.get("motion_began"))
                )
                or any(
                    isinstance(attempt, Mapping)
                    and (
                        attempt.get("status") == "running"
                        or (
                            attempt.get("status") != "completed"
                            and bool(attempt.get("motion_began"))
                        )
                    )
                    for attempt in record.get("attempts", [])
                )
            )
        ]
        if unsafe and not retry_motion_began:
            first = unsafe[0]
            raise RuntimeError(
                "refusing automatic resume for incomplete or motion-started cell "
                f"{first.get('cell_index')}; pass retry_motion_began=True after recovery review"
            )
    else:
        _atomic_write_text(
            manifest_json_path,
            json.dumps(
                {
                    "schema_version": MATRIX_SCHEMA_VERSION,
                    "suite_mode": suite_mode,
                    "controller_variant": controller_variant,
                    "condition_label": f"{suite_mode}__{controller_variant}",
                    "protocol": protocol,
                    "contract_hash": contract_hash,
                    "provenance": provenance,
                    "cells": planned_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    status_records = [previous_by_index.get(int(cell["cell_index"]), _planned_record(cell)) for cell in cells]

    def write_status() -> None:
        _atomic_write_text(
            status_path,
            json.dumps(
                {
                    "schema_version": MATRIX_SCHEMA_VERSION,
                    "suite_mode": suite_mode,
                    "controller_variant": controller_variant,
                    "condition_label": f"{suite_mode}__{controller_variant}",
                    "protocol": protocol,
                    "contract_hash": contract_hash,
                    "provenance": provenance,
                    "cells": status_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    if not resume:
        write_status()

    def write_initial_manifest() -> None:
        """Initialize terminal-only JSONL; plan/status hold the full inventory."""
        _atomic_write_text(manifest_path, "")

    def append_manifest(record: Mapping[str, Any]) -> None:
        """Append terminal/retry history without erasing earlier attempts."""
        line = dict(record)
        line.setdefault("protocol", protocol)
        line.setdefault("contract_hash", contract_hash)
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(line, sort_keys=True) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())

    # Keep all planned cells visible even if the process is interrupted before
    # the first environment is constructed.
    if not resume:
        write_initial_manifest()
    elif not manifest_path.exists():
        # A recoverable status snapshot is authoritative if a prior process
        # died before creating the append-only JSONL file.
        write_initial_manifest()
    if resume:
        # Status is the durable inventory; reconcile terminal cells that were
        # written to status immediately before a process died while appending
        # JSONL.  Existing history is never rewritten or discarded.
        manifest_history: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid matrix manifest JSONL at line {line_number}"
                ) from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(f"matrix manifest line {line_number} is not an object")
            if parsed.get("contract_hash") not in (None, contract_hash):
                raise ValueError("resume manifest contract hash does not match status")
            manifest_history.append(dict(parsed))
        for prior_record in previous_by_index.values():
            if prior_record.get("status") not in {"completed", "failed", "interrupted"}:
                continue
            expected_attempts = prior_record.get("attempts", [])
            expected_count = len(expected_attempts) if isinstance(expected_attempts, list) else 0
            matching_history = False
            for item in manifest_history:
                if not isinstance(item, Mapping):
                    continue
                try:
                    same_cell = int(item.get("cell_index", -1)) == int(prior_record["cell_index"])
                except (TypeError, ValueError):
                    same_cell = False
                item_attempts = item.get("attempts", [])
                if (
                    same_cell
                    and item.get("status") == prior_record.get("status")
                    and isinstance(item_attempts, list)
                    and len(item_attempts) == expected_count
                ):
                    matching_history = True
                    break
            if matching_history:
                continue
            append_manifest(prior_record)
            manifest_history.append(dict(prior_record))
    records: list[dict[str, Any]] = []
    started = time.time()
    for cell in cells:
            # Reset per-cell state explicitly.  Using ``locals()`` here would
            # see the previous iteration's record after a build/interrupt
            # failure and could attach cleanup errors to the wrong cell.
            cell_record: dict[str, Any] | None = None
            cell_index = int(cell["cell_index"])
            prior = previous_by_index.get(cell_index)
            if resume and prior is not None and prior.get("status") == "completed":
                cell_record = prior
                records.append(cell_record)
                continue
            env = None
            init_state_diagnostics: Mapping[str, Any] | None = None
            settle_diagnostics: Mapping[str, Any] | None = None
            inputs: dict[str, Any] | None = None
            cell_exception: BaseException | None = None
            close_exception: BaseException | None = None
            close_succeeded = False
            stage = "build_env"
            attempt_started = time.time()
            prior_attempts = list(prior.get("attempts", [])) if isinstance(prior, Mapping) else []
            attempt_number = len(prior_attempts) + 1
            attempt_output_dir = Path(cell["output_dir"])
            if attempt_number > 1:
                attempt_output_dir = attempt_output_dir / f"attempt_{attempt_number:02d}"
            running_record = _planned_record(cell)
            running_record.update({
                "status": "running",
                "stage": "build_env",
                "attempt_output_dir": attempt_output_dir.as_posix(),
                "attempts": prior_attempts + [{
                    "attempt_index": attempt_number,
                    "started_unix": attempt_started,
                    "finished_unix": None,
                    "duration_s": None,
                    "status": "running",
                    "stage": "build_env",
                    "motion_began": False,
                }],
                "motion_began": None,
                "protocol": protocol,
                "provenance": provenance,
                "contract_hash": contract_hash,
            })
            status_records[cell_index] = running_record
            write_status()
            def motion_started_callback() -> None:
                running_record["motion_began"] = True
                if running_record.get("attempts"):
                    running_record["attempts"][-1]["motion_began"] = True
                status_records[cell_index] = running_record
                write_status()

            interrupted = False
            interrupt_exc: BaseException | None = None
            try:
                cell_dir = Path(cell["output_dir"])
                cell_dir.mkdir(parents=True, exist_ok=True)
                env = _call_with_optional_keywords(
                    build_env,
                    (int(cell["task_id"]), int(cell["seed"]), int(resolution)),
                    {"suite_mode": suite_mode},
                )
                candidate_init_diagnostics = getattr(env, "_arrow_init_state_diagnostics", None)
                if isinstance(candidate_init_diagnostics, Mapping):
                    init_state_diagnostics = dict(candidate_init_diagnostics)
                candidate_settle_diagnostics = getattr(env, "_arrow_settle_diagnostics", None)
                if isinstance(candidate_settle_diagnostics, Mapping):
                    settle_diagnostics = dict(candidate_settle_diagnostics)
                if env_builder is None:
                    _validate_observed_init_state(
                        init_state_diagnostics,
                        cell=cell,
                        preflight=init_state_preflight,
                    )
            except BaseException as exc:
                cell_exception = exc
                cell_record = _error_record(cell, stage="build_env", exc=exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    interrupted, interrupt_exc = True, exc
            else:
                try:
                    stage = "generate_arrow_inputs"
                    inputs = dict(
                        _call_with_optional_keywords(
                            build_inputs,
                            (env, int(cell["task_id"]), int(resolution)),
                            {
                                "suite_mode": suite_mode,
                                "controller_variant": controller_variant,
                            },
                        )
                    )
                    stage = "run_episode"
                    episode_kwargs = {
                        "env": env,
                        "task_id": int(cell["task_id"]),
                        "seed": int(cell["seed"]),
                            "output_dir": attempt_output_dir.as_posix(),
                        "dry_run": bool(dry_run),
                        "resolution": int(resolution),
                        "allow_unvalidated_profile": bool(allow_unvalidated_profile),
                        "suite_mode": suite_mode,
                        "controller_variant": controller_variant,
                        **inputs,
                    }
                    accepted_runner_keywords = _accepted_optional_keywords(run_episode)
                    supports_motion_callback = (
                        accepted_runner_keywords is None
                        or "motion_started_callback" in accepted_runner_keywords
                    )
                    if supports_motion_callback:
                        episode_kwargs["motion_started_callback"] = motion_started_callback
                    if evaluator_fn is not None:
                        episode_kwargs["evaluator"] = evaluator_fn
                    # Keep the established kwargs call for the runner's core
                    # inputs, but remove newly introduced condition metadata
                    # when a legacy injected fake does not accept it.
                    if accepted_runner_keywords is not None:
                        episode_kwargs = {
                            key: value
                            for key, value in episode_kwargs.items()
                            if key in accepted_runner_keywords
                            or key in {
                                "env", "task_id", "seed", "output_dir", "dry_run",
                                "resolution", "allow_unvalidated_profile", "evaluator",
                                "motion_started_callback",
                            }
                        }
                    audit = dict(run_episode(**episode_kwargs))
                    if dry_run:
                        # A dry-run must never present an evaluator result as
                        # observed motion evidence, even if an injected fake
                        # returns a stale/optimistic value.
                        audit["evaluator_success"] = None
                    cell_record = dict(cell)
                    cell_record.update({
                        "status": "completed",
                        "stage": "run_episode",
                        "failure_class": "task_failure" if audit.get("evaluator_success") is False else None,
                        "error_type": None,
                        "error": None,
                        "audit_path": audit.get("audit_path"),
                        "audit": audit,
                        "frames": audit.get("frames", []),
                        "phase_frames": audit.get("phase_frames", []),
                        "phases": audit.get("phases", []),
                        "evaluator_result": None if dry_run else audit.get("evaluator_success"),
                    })
                except BaseException as exc:
                    cell_exception = exc
                    cell_record = _error_record(cell, stage=stage, exc=exc)
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        interrupted, interrupt_exc = True, exc
            finally:
                # Sample motion before close: real wrappers may clear action
                # buffers or detach simulator state during cleanup.
                motion_began = _motion_began(
                    env, cell_record.get("audit") if cell_record is not None else None
                )
                observed_init_diagnostics = _init_state_diagnostics(env)
                if observed_init_diagnostics is not None:
                    init_state_diagnostics = observed_init_diagnostics
                observed_settle_diagnostics = getattr(env, "_arrow_settle_diagnostics", None)
                if isinstance(observed_settle_diagnostics, Mapping):
                    settle_diagnostics = dict(observed_settle_diagnostics)
                observed_phase_audit = getattr(env, "_arrow_phase_audit", None)
                if isinstance(observed_phase_audit, list):
                    phase_records = [dict(item) for item in observed_phase_audit if isinstance(item, Mapping)]
                    if phase_records:
                        if cell_record is None:
                            cell_record = _error_record(
                                cell, stage=stage, exc=RuntimeError("controller failed")
                            )
                        cell_record["phases"] = phase_records
                        cell_record["phase_frames"] = [
                            item["diagnostic_frame"]
                            for item in phase_records
                            if item.get("diagnostic_frame")
                        ]
                observed_recovery_audit = getattr(env, "_arrow_recovery_audit", None)
                if isinstance(observed_recovery_audit, list):
                    recovery_records = [
                        dict(item) for item in observed_recovery_audit
                        if isinstance(item, Mapping)
                    ]
                    if cell_record is None:
                        cell_record = _error_record(
                            cell, stage=stage, exc=RuntimeError("controller failed")
                        )
                    cell_record["recovery"] = recovery_records
                if env is None:
                    close_succeeded = True
                else:
                    close = getattr(env, "close", None)
                    if close is None:
                        if continue_on_motion_failure:
                            close_exception = RuntimeError("environment close is unavailable")
                        else:
                            # Preserve compatibility for lightweight injected
                            # dry-run fakes; production continuation requires a
                            # callable teardown hook.
                            close_succeeded = True
                    else:
                        try:
                            close()
                        except BaseException as exc:
                            close_exception = exc
                        else:
                            close_succeeded = True
                    if close_exception is not None:
                        exc = close_exception
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            interrupted = True
                            if interrupt_exc is None:
                                interrupt_exc = exc
                        if cell_exception is None:
                            cell_exception = exc
                        if cell_record is None or cell_record.get("status") == "completed":
                            if cell_record is not None and cell_record.get("status") == "completed":
                                # Keep a successful episode's audit even when
                                # cleanup itself fails; classify cleanup as the
                                # terminal cell failure without dropping evidence.
                                cell_record["status"] = "failed"
                                cell_record["stage"] = "close_env"
                                cell_record["failure_class"] = "environment_failure"
                                cell_record["error_type"] = type(exc).__name__
                                cell_record["error"] = str(exc)
                            else:
                                cell_record = _error_record(cell, stage="close_env", exc=exc)
                        else:
                            cell_record["close_error"] = str(exc)
                            cell_record["stage"] = "close_env"
                            cell_record["failure_class"] = "environment_failure"
            # KeyboardInterrupt/SystemExit intentionally propagate after the
            # cleanup above; ordinary failures always initialize a record.
            if cell_record is None:
                cell_record = _error_record(
                    cell, stage="interrupted", exc=RuntimeError("cell interrupted")
                )
            if interrupted:
                cell_record["status"] = "interrupted"
                cell_record["failure_class"] = "interrupted"
                cell_record["stage"] = stage
            finished = time.time()
            cell_record["duration_s"] = float(finished - attempt_started)
            prior_attempts.append(
                {
                    "attempt_index": len(prior_attempts) + 1,
                    "started_unix": attempt_started,
                    "finished_unix": finished,
                    "duration_s": cell_record["duration_s"],
                    "status": cell_record.get("status"),
                    "stage": cell_record.get("stage"),
                    "motion_began": bool(motion_began or running_record.get("motion_began")),
                }
            )
            cell_record["attempts"] = prior_attempts
            cell_record["attempt_output_dir"] = attempt_output_dir.as_posix()
            cell_record["dry_run"] = bool(dry_run)
            cell_record["close_succeeded"] = bool(close_succeeded)
            cell_record["init_state_diagnostics"] = init_state_diagnostics
            cell_record["settle_diagnostics"] = settle_diagnostics
            cell_record["init_state_index"] = _init_state_index(init_state_diagnostics)
            cell_record["motion_began"] = bool(motion_began or running_record.get("motion_began"))
            if isinstance(cell_record.get("audit"), Mapping):
                cell_record["diagnostics"] = _audit_diagnostics(cell_record["audit"])
                if dry_run:
                    cell_record["diagnostics"]["evaluator_result"] = None
            elif cell_record.get("phases"):
                cell_record["diagnostics"]["phases"] = cell_record["phases"]
                cell_record["diagnostics"]["phase_frames"] = cell_record["phase_frames"]
            if isinstance(cell_record.get("recovery"), list):
                cell_record.setdefault("diagnostics", {})["recovery"] = cell_record["recovery"]
            cell_record["protocol"] = protocol
            cell_record["provenance"] = provenance
            cell_record["contract_hash"] = contract_hash
            status_records[cell_index] = cell_record
            records.append(cell_record)
            write_status()
            append_manifest(cell_record)
            if interrupted and interrupt_exc is not None:
                raise interrupt_exc
            if close_exception is not None:
                # Teardown failure invalidates independence even when
                # continuation was explicitly requested; never run the next
                # cell against an environment whose lifecycle is uncertain.
                raise close_exception
            # Persist first, then surface a motion-started exception so the
            # batch cannot continue into an unsafe duplicate manipulation.
            if (
                cell_exception is not None
                and cell_record["motion_began"]
                and not continue_on_motion_failure
            ):
                raise cell_exception

    completed = [record for record in records if record.get("status") == "completed"]
    failures = [record for record in records if record.get("status") != "completed"]
    evaluated = [_audit_success(record) for record in completed]
    evaluated = [value for value in evaluated if value is not None]
    successes = sum(value is True for value in evaluated)
    has_evaluator_result = bool(evaluated)
    class_records = [record for record in records if _record_failure_class(record) is not None]
    by_task_records = {
        task_id: [record for record in records if int(record["task_id"]) == task_id]
        for task_id in sorted({int(record["task_id"]) for record in records})
    }
    by_seed_records = {
        seed: [record for record in records if int(record["seed"]) == seed]
        for seed in sorted({int(record["seed"]) for record in records})
    }
    summary = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "suite_mode": suite_mode,
        "controller_variant": controller_variant,
        "condition_label": f"{suite_mode}__{controller_variant}",
        "protocol": protocol,
        "provenance": provenance,
        "manifest_path": manifest_path.as_posix(),
        "manifest_json_path": manifest_json_path.as_posix(),
        "status_path": status_path.as_posix(),
        "contract_hash": contract_hash,
        "summary_path": summary_path.as_posix(),
        "total_cells": len(records),
        "completed_cells": len(completed),
        "failed_cells_count": len(failures),
        "evaluated_cells": len(evaluated),
        "planned_count": len(records),
        "conservative_denominator": len(records),
        "successes": int(successes),
        # Preserve the original evaluable-only alias while exposing explicit
        # conservative/planned and evaluable denominator metrics below.
        "success_rate": (float(successes) / len(evaluated)) if evaluated else None,
        "planned_success_rate": (float(successes) / len(records))
        if records and evaluated else None,
        "conservative_success_rate": (float(successes) / len(records))
        if records and evaluated else None,
        "evaluable_denominator": len(evaluated),
        "evaluable_success_rate": (float(successes) / len(evaluated)) if evaluated else None,
        "failed_cells": [
            {
                "cell_index": record["cell_index"],
                "task_id": record["task_id"],
                "episode_index": record["episode_index"],
                "seed": record["seed"],
                "output_dir": record["output_dir"],
                "stage": record.get("stage"),
                "failure_class": _record_failure_class(record),
                "error_type": record.get("error_type"),
                "error": record.get("error"),
                "motion_began": bool(record.get("motion_began")),
                "evaluator_result": record.get("evaluator_result"),
            }
            for record in failures
        ],
        "failure_by_stage": {
            stage: sum(record.get("stage") == stage for record in failures)
            for stage in sorted({str(record.get("stage")) for record in failures})
        },
        "failure_by_class": {
            failure_class: sum(_record_failure_class(record) == failure_class for record in class_records)
            for failure_class in sorted({_record_failure_class(record) for record in class_records})
        },
        "phase_aggregates": _phase_aggregates(records),
        "diagnostic_aggregates": _diagnostic_aggregates(records),
        "per_task": {
            str(task_id): _partition_summary(task_records)
            for task_id, task_records in by_task_records.items()
        },
        "per_seed": {
            str(seed): _partition_summary(seed_records)
            for seed, seed_records in by_seed_records.items()
        },
        "started_unix": started,
        "finished_unix": time.time(),
    }
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", default=",".join(str(item) for item in DEFAULT_TASK_IDS))
    parser.add_argument("--episodes-per-task", type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--suite-mode",
        choices=SUITE_MODES,
        default=DEFAULT_SUITE_MODE,
        help="LIBERO scene condition: canonical vanilla or sealed randomized",
    )
    parser.add_argument(
        "--controller-variant",
        default=DEFAULT_CONTROLLER_VARIANT,
        help="stable controller implementation label recorded in outputs",
    )
    parser.add_argument("--output-root", type=Path, default=Path("arrow_pick_place_matrix_outputs"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="run the pipeline without motion")
    mode.add_argument("--execute-motion", action="store_true", help="explicitly authorize motion")
    parser.add_argument(
        "--allow-unvalidated-profile",
        action="store_true",
        help="authorize exploratory cells outside task=0/seed=1000/resolution=256",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume matching planned output, skipping already completed cells",
    )
    parser.add_argument(
        "--retry-motion-began",
        action="store_true",
        help="explicitly authorize retrying a cell whose prior attempt may have moved",
    )
    parser.add_argument(
        "--continue-on-motion-failure",
        action="store_true",
        help="continue independent cells after a motion-started cell failure; each env is closed first",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_matrix(
        output_root=args.output_root,
        task_ids=parse_task_ids(args.task_ids),
        episodes_per_task=args.episodes_per_task,
        seed_base=args.seed_base,
        resolution=args.resolution,
        suite_mode=args.suite_mode,
        controller_variant=args.controller_variant,
        dry_run=args.dry_run,
        execute_motion=args.execute_motion,
        allow_unvalidated_profile=args.allow_unvalidated_profile,
        resume=args.resume,
        retry_motion_began=args.retry_motion_began,
        continue_on_motion_failure=args.continue_on_motion_failure,
    )
    print(json.dumps({"summary_path": summary["summary_path"], "success_rate": summary["success_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
