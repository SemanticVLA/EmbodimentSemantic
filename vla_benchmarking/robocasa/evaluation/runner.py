"""RoboCasa Pick & Place matrix runner with explicit terminal accounting."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .task_manifest import PICK_PLACE_TASKS, TaskSpec


DEFAULT_EPISODES_PER_TASK = 10
DEFAULT_SEED_BASE = 1000
DEFAULT_SPLIT = "target"
_INFRASTRUCTURE_FAILURES = {"dependency_missing", "runtime_backend_unavailable"}

# These markers identify failures at the RoboCasa input/action boundary. They
# are deliberately narrower than all controller errors: a true policy or
# task failure must remain a controller/task failure in the accounting.
_GEOMETRY_CONTRACT_MARKERS = (
    "bbox",
    "projection",
    "visible area",
    "behind the camera",
    "calibration",
    "deproject",
    "arrow endpoint",
    "frame contract",
    "workspace point",
    "workspace bounds",
    "intrinsic",
    "extrinsic",
)
_DEPENDENCY_MARKERS = (
    "no module named",
    "not installed",
    "install the controller dependencies",
    "missing dependency",
)


def _robocasa_controller_identity() -> dict[str, Any]:
    """Resolve stable controller provenance without importing live backends.

    The RoboCasa-local entrypoint is intentionally imported lazily. This keeps
    runner usable for manifests, preflight, and result accounting on hosts
    that do not have MuJoCo/RoboCasa installed. The returned payload is both
    human-readable provenance and part of the experiment identity.
    """

    identity: dict[str, Any] = {
        "schema": "robocasa_controller_identity.v2",
        "availability": "available",
        "canonical_controller": {
            "name": None,
            "config_filename": None,
            "config_hash": None,
            "policy_lock_canonical_config_sha256": None,
            "policy_lock_sha256": None,
        },
        "robocasa_adapter": {
            "module": None,
            "entrypoint": None,
            "source_sha256": None,
            "frame_contract": "robocasa_base_frame_v1",
            "camera_contract": "post_flip_xy_positive_k_world_from_camera_v1",
            "action_contract": "pandaomron_12d_embed_canonical_7d_v1",
        },
        "errors": [],
    }

    try:
        module = importlib.import_module(
            "vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner"
        )
        entrypoint = getattr(module, "run_episode")
        identity["robocasa_adapter"].update({
            "module": str(module.__name__),
            "entrypoint": f"{module.__name__}.{entrypoint.__name__}",
        })
        identity_factory = getattr(module, "controller_identity", None)
        if not callable(identity_factory):
            identity_factory = getattr(module, "get_controller_identity", None)
        if callable(identity_factory):
            declared = identity_factory()
            identity["robocasa_adapter"]["declared_identity"] = _json_safe(declared)
            identity["robocasa_adapter"]["identity_function"] = (
                f"{module.__name__}.{identity_factory.__name__}"
            )
        source_path = inspect.getsourcefile(entrypoint)
        if source_path:
            identity["robocasa_adapter"]["source_sha256"] = hashlib.sha256(
                Path(source_path).read_bytes()
            ).hexdigest()
    except Exception as exc:  # pragma: no cover - exercised on partial installs
        identity["errors"].append({
            "stage": "robocasa_adapter",
            "type": type(exc).__name__,
        })

    try:
        config_module = importlib.import_module(
            "vla_benchmarking.robocasa.arrow_grasp_controller.configs"
        )
        config = config_module.load_controller_config()
        lock_path = Path(config_module.ACTIVE_POLICY_LOCK_PATH)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        identity["canonical_controller"].update({
            "name": str(config_module.ACTIVE_CONTROLLER_NAME),
            "config_filename": str(config_module.ACTIVE_CONTROLLER_CONFIG_FILENAME),
            "config_hash": str(config.get("config_hash")),
            "policy_lock_canonical_config_sha256": lock.get("canonical_config_sha256"),
            "policy_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        })
        if config.get("config_hash") != lock.get("canonical_config_sha256"):
            identity["errors"].append({
                "stage": "canonical_config",
                "type": "ControllerConfigLockMismatch",
            })
    except Exception as exc:  # pragma: no cover - exercised on partial installs
        identity["errors"].append({
            "stage": "canonical_config",
            "type": type(exc).__name__,
        })

    if identity["errors"]:
        identity["availability"] = "partial"
    return identity


@dataclass(frozen=True)
class TerminalRow:
    task: str
    episode_index: int
    seed: int
    split: str
    mode: str
    terminal: bool
    success: bool
    failure_category: str | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return value


def _experiment_identity(
    *, tasks: Sequence[TaskSpec], episodes_per_task: int, seed_base: int,
    split: str, mode: str, resolution: int = 256,
    controller_identity: Mapping[str, Any] | None = None,
) -> str:
    from ..shared.config import CAMERA, ROBOSUITE_COMMIT, ROBOCASA_COMMIT
    from .prompt import canonical_prompt
    resolved_controller_identity = dict(
        controller_identity or _robocasa_controller_identity()
    )
    payload = {
        "schema": "robocasa-pick-place-21.v1",
        "tasks": [task.name for task in tasks],
        "episodes_per_task": int(episodes_per_task),
        "seed_base": int(seed_base),
        "split": str(split),
        "mode": str(mode),
        "resolution": int(resolution),
        "camera": CAMERA.name,
        "depth_encoding": CAMERA.depth_encoding,
        "robocasa_commit": ROBOCASA_COMMIT,
        "robosuite_commit": ROBOSUITE_COMMIT,
        "controller": resolved_controller_identity.get("canonical_controller", {}),
        "controller_source": "robocasa_local_adapter",
        "controller_identity": resolved_controller_identity,
        "controller_camera_seam": CAMERA.name,
        "prompt_sha256": hashlib.sha256(canonical_prompt().encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def selected_tasks(names: Sequence[str] | None = None) -> tuple[TaskSpec, ...]:
    if not names or tuple(names) == ("all",):
        return PICK_PLACE_TASKS
    by_name = {task.name: task for task in PICK_PLACE_TASKS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown RoboCasa task(s): {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def build_rows(
    *,
    tasks: Iterable[TaskSpec],
    episodes_per_task: int,
    seed_base: int,
    split: str,
    mode: str,
    failure_category: str | None = None,
    terminal: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[TerminalRow]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    if seed_base < 0:
        raise ValueError("seed_base must be non-negative")
    if split != DEFAULT_SPLIT:
        raise ValueError("the RoboCasa portability run is locked to split='target'")
    rows: list[TerminalRow] = []
    for task in tasks:
        for episode_index in range(episodes_per_task):
            rows.append(
                TerminalRow(
                    task=task.name,
                    episode_index=episode_index,
                    seed=seed_base + episode_index,
                    split=split,
                    mode=mode,
                    terminal=terminal,
                    success=False,
                    failure_category=failure_category,
                    metadata=dict(metadata or {}),
                )
            )
    return rows


def _cell_key(row: TerminalRow | Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["task"] if isinstance(row, Mapping) else row.task), int(
        row["seed"] if isinstance(row, Mapping) else row.seed
    ), str(row["mode"] if isinstance(row, Mapping) else row.mode)


def _read_existing(path: Path, *, experiment_identity: str) -> dict[tuple[str, int, str], TerminalRow]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, int, str], TerminalRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            row = TerminalRow(
                task=str(payload["task"]),
                episode_index=int(payload["episode_index"]),
                seed=int(payload["seed"]),
                split=str(payload["split"]),
                mode=str(payload["mode"]),
                terminal=bool(payload["terminal"]),
                success=bool(payload["success"]),
                failure_category=payload.get("failure_category"),
                metadata=dict(payload.get("metadata") or {}),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if row.metadata.get("experiment_identity") == experiment_identity:
            existing[_cell_key(row)] = row
    return existing


def _write_outputs(
    output_dir: Path, rows: Sequence[TerminalRow], *, mode: str, experiment_identity: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    temporary_path = output_dir / "results.jsonl.tmp"
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row.as_dict()), sort_keys=True) + "\n")
    temporary_path.replace(result_path)
    by_task: dict[str, dict[str, int]] = {}
    for row in rows:
        stats = by_task.setdefault(row.task, {"episodes": 0, "terminal": 0, "successes": 0})
        stats["episodes"] += 1
        stats["terminal"] += int(row.terminal)
        stats["successes"] += int(row.success)
    failure_categories = {
        category: sum(row.failure_category == category for row in rows)
        for category in sorted({row.failure_category for row in rows if row.failure_category})
    }
    if not all(row.terminal for row in rows):
        evaluation_status = "incomplete"
    elif any(row.failure_category in _INFRASTRUCTURE_FAILURES for row in rows):
        evaluation_status = "infrastructure_unavailable"
    elif any(not row.success for row in rows):
        evaluation_status = "completed_with_failures"
    else:
        evaluation_status = "complete"
    summary = {
        "benchmark": "robocasa_pick_place_21",
        "experiment_identity": experiment_identity,
        "mode": mode,
        "expected": len(rows),
        "planned": len(rows),
        "complete": sum(row.terminal for row in rows),
        "terminal": sum(row.terminal for row in rows),
        "successes": sum(row.success for row in rows),
        "failure_categories": failure_categories,
        "per_task": by_task,
        "evaluation_status": evaluation_status,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _failure_category(
    status: str,
    error: str | None = None,
    *,
    error_type: str | None = None,
) -> str | None:
    if status == "success":
        return None
    normalized_error = (error or "").lower()
    normalized_type = (error_type or "").lower()
    if (
        status in {"runtime_backend_unavailable", "controller_failure"}
        and (
            any(marker in normalized_error for marker in _DEPENDENCY_MARKERS)
            or normalized_type in {"modulenotfounderror", "importerror"}
        )
    ):
        return "dependency_missing"
    if status in {
        "task_failure",
        "controller_failure",
        "horizon_exhaustion",
        "dependency_missing",
        "runtime_backend_unavailable",
        "geometry_contract_failure",
    }:
        if status == "geometry_contract_failure":
            return status
        if status in {"controller_failure", "runtime_backend_unavailable"} and (
            "robocasaliveerror" in normalized_type
            or any(marker in normalized_error for marker in _GEOMETRY_CONTRACT_MARKERS)
        ):
            # Environment import/render failures remain infrastructure errors.
            # Only explicit input/action contract evidence is promoted here.
            if not any(marker in normalized_error for marker in _DEPENDENCY_MARKERS):
                return "geometry_contract_failure"
        return status
    normalized_error = (error or "").lower()
    if any(marker in normalized_error for marker in _DEPENDENCY_MARKERS):
        return "dependency_missing"
    return "runtime_backend_unavailable"


def _canonical_phase_timeout(live: Mapping[str, Any]) -> int | None:
    """Extract canonical controller timeout without conflating task horizon."""
    audit = live.get("audit")
    if not isinstance(audit, Mapping):
        return None
    variant = audit.get("controller_variant")
    if not isinstance(variant, Mapping):
        final_result = audit.get("final_result")
        if isinstance(final_result, Mapping):
            motion_audit = final_result.get("audit")
            if isinstance(motion_audit, Mapping):
                variant = motion_audit.get("controller_variant")
    if not isinstance(variant, Mapping):
        return None
    value = variant.get("phase_timeout_steps")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _run_live_cell(
    *, task: TaskSpec, episode_index: int, seed: int, output_dir: Path,
    mode: str, experiment_identity: str,
    controller_identity: Mapping[str, Any] | None = None,
    execute_motion: bool = True, resolution: int = 256
) -> TerminalRow:
    if importlib.util.find_spec("robocasa") is None:
        return TerminalRow(
            task=task.name, episode_index=episode_index, seed=seed, split=DEFAULT_SPLIT,
            mode=mode, terminal=True, success=False, failure_category="dependency_missing",
            metadata={
                "robocasa_available": False,
                "execution_started": False,
                "experiment_identity": experiment_identity,
                "controller_identity": dict(controller_identity or {}),
                "task_horizon": None,
                "canonical_phase_timeout_steps": None,
            },
        )
    try:
        from .live import run_live_cell
        live = run_live_cell(
            task_name=task.name, seed=seed, output_dir=output_dir,
            resolution=resolution, execute_motion=execute_motion,
        )
    except Exception as exc:  # Runtime boundary: preserve a terminal matrix row.
        live = {
            "status": "runtime_backend_unavailable",
            "terminal_reason": type(exc).__name__,
            "error": str(exc),
            "robocasa_available": True,
            "execution_started": False,
        }
    status = str(live.get("status", "runtime_backend_unavailable"))
    preflight = not execute_motion
    error = str(live.get("error", ""))
    error_type = str(live.get("terminal_reason", ""))
    return TerminalRow(
        task=task.name,
        episode_index=episode_index,
        seed=seed,
        split=DEFAULT_SPLIT,
        mode=mode,
        terminal=not preflight,
        success=status == "success",
        failure_category=(
            None
            if status == "preflight_complete"
            else _failure_category(status, error, error_type=error_type)
        ),
        metadata={
            "robocasa_available": True,
            "execution_started": bool(execute_motion),
            "experiment_identity": experiment_identity,
            "controller_identity": dict(controller_identity or {}),
            "task_horizon": live.get("horizon"),
            "canonical_phase_timeout_steps": _canonical_phase_timeout(live),
            "live": live,
        },
    )


def run(
    *,
    output_dir: Path,
    mode: str = "preflight",
    task_names: Sequence[str] | None = None,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    seed_base: int = DEFAULT_SEED_BASE,
    split: str = DEFAULT_SPLIT,
    execute_motion: bool = False,
) -> int:
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError("mode must be preflight, smoke, or full")
    tasks = selected_tasks(task_names)
    controller_identity = _robocasa_controller_identity()
    experiment_identity = _experiment_identity(
        tasks=tasks, episodes_per_task=episodes_per_task, seed_base=seed_base,
        split=split, mode=mode, controller_identity=controller_identity,
    )
    if mode == "preflight":
        if importlib.util.find_spec("robocasa") is None:
            rows = build_rows(
                tasks=tasks, episodes_per_task=episodes_per_task, seed_base=seed_base,
                split=split, mode=mode,
                metadata={
                    "preflight": True,
                    "runtime_required": False,
                    "experiment_identity": experiment_identity,
                    "controller_identity": controller_identity,
                    "task_horizon": None,
                    "canonical_phase_timeout_steps": None,
                },
            )
        else:
            rows = []
            for task in tasks:
                for episode_index in range(episodes_per_task):
                    rows.append(
                        _run_live_cell(
                            task=task, episode_index=episode_index,
                            seed=seed_base + episode_index,
                            output_dir=output_dir / "cells" / f"{task.name}__seed{seed_base + episode_index}",
                            mode="preflight",
                            experiment_identity=experiment_identity,
                            controller_identity=controller_identity,
                            execute_motion=False,
                        )
                    )
        _write_outputs(output_dir, rows, mode=mode, experiment_identity=experiment_identity)
        print(f"RoboCasa {mode}: wrote {len(rows)} rows to {output_dir / 'results.jsonl'}")
        return 0

    if not execute_motion:
        rows = build_rows(
            tasks=tasks, episodes_per_task=episodes_per_task, seed_base=seed_base,
            split=split, mode=mode, failure_category="motion_not_requested", terminal=True,
            metadata={
                "robocasa_available": importlib.util.find_spec("robocasa") is not None,
                "execution_started": False,
                "requires_explicit_execute_motion": True,
                "experiment_identity": experiment_identity,
                "controller_identity": controller_identity,
                "task_horizon": None,
                "canonical_phase_timeout_steps": None,
            },
        )
    else:
        existing = _read_existing(output_dir / "results.jsonl", experiment_identity=experiment_identity)
        rows = []
        for task in tasks:
            for episode_index in range(episodes_per_task):
                seed = seed_base + episode_index
                key = (task.name, seed, mode)
                prior = existing.get(key)
                # Infrastructure failures are intentionally rerunnable after an
                # environment install; completed task/controller outcomes are not.
                if prior is not None and prior.terminal and prior.failure_category not in _INFRASTRUCTURE_FAILURES:
                    rows.append(prior)
                    continue
                rows.append(
                    _run_live_cell(
                        task=task,
                        episode_index=episode_index,
                        seed=seed,
                        output_dir=output_dir / "cells" / f"{task.name}__seed{seed}",
                        mode=mode,
                        experiment_identity=experiment_identity,
                        controller_identity=controller_identity,
                    )
                )
    _write_outputs(output_dir, rows, mode=mode, experiment_identity=experiment_identity)
    print(f"RoboCasa {mode}: wrote {len(rows)} rows to {output_dir / 'results.jsonl'}")
    if execute_motion and mode == "full" and rows and all(
        row.failure_category in _INFRASTRUCTURE_FAILURES for row in rows
    ):
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--episodes-per-task", type=int, default=DEFAULT_EPISODES_PER_TASK)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--execute-motion", action="store_true", help="authorize live motion; required for full mode")
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.execute_motion:
        parser.error("full mode requires --execute-motion")
    return run(
        output_dir=args.output_dir, mode=args.mode, task_names=args.tasks,
        episodes_per_task=args.episodes_per_task, seed_base=args.seed_base,
        split=args.split, execute_motion=args.execute_motion,
    )


__all__ = ["TerminalRow", "build_rows", "main", "run", "selected_tasks"]


if __name__ == "__main__":  # pragma: no cover - operator entrypoint
    raise SystemExit(main())
