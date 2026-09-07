"""Command line entry points for the automatic RoboTTT LIBERO study.

The CLI is deliberately orchestration-only.  It validates a run declaration,
then hands execution to the self-contained collection/training/evaluation
backends when those modules are installed.  It never silently substitutes a
different model or fills an unresolved RoboTTT hyperparameter.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .config import (
    ExperimentConfig,
    VLA_ALIASES,
    VLA_DISPLAY_NAMES,
    VLA_NAMES,
    config_from_dict,
    load_config,
    save_config,
)

PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_CONFIG = PACKAGE_DIR / "example_config.json"


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load(path: str | None) -> ExperimentConfig:
    return load_config(path or EXAMPLE_CONFIG)


def _preflight(config: ExperimentConfig) -> tuple[bool, dict[str, Any]]:
    errors = config.validate()
    report = {
        "status": "READY" if not errors else "BLOCKED",
        "fidelity_mode": config.fidelity_mode,
        "config_digest": config.digest(),
        "vlas": [VLA_DISPLAY_NAMES.get(v, v) for v in config.canonical_dict()["vla_names"]],
        "tasks": config.task_ids,
        "study_protocol": dict(config.study_protocol),
        "split_errors": config.split.validate(
            task_ids=config.task_ids,
            episodes_per_task=config.episodes_per_task,
        ),
        "errors": errors,
        "exact_mode_policy": (
            "Requires the official RoboTTT artifact manifest and every unresolved "
            "paper detail; no inferred defaults are accepted."
        ),
    }
    return not errors, report


def _run_backend(module_name: str, function_name: str, config: ExperimentConfig, args: argparse.Namespace) -> int:
    """Call a backend without importing heavyweight robotics dependencies at CLI startup."""

    try:
        module = importlib.import_module(f"{__package__}.{module_name}")
    except ImportError as exc:
        print(
            f"BLOCKED: execution backend {module_name!r} is not available ({exc}). "
            "Install the LIBERO/lerobot environment, then rerun this command.",
            file=sys.stderr,
        )
        return 3
    function: Callable[..., Any] | None = getattr(module, function_name, None)
    if function is None:
        print(f"BLOCKED: {module_name}.{function_name} is not implemented", file=sys.stderr)
        return 3
    result = function(config=config, args=args)
    if result is not None:
        _json_dump(result)
        if isinstance(result, dict) and str(result.get("status", "")).startswith(("BLOCKED", "READY_FOR")):
            # A contract receipt is not evidence that episodes were collected
            # or that a checkpoint was trained.  Keep shell/CI callers from
            # treating an injection stub as a successful experiment.
            return 3
    return 0


def _injection_path(config: ExperimentConfig, args: argparse.Namespace, operation: str) -> str:
    """Resolve a host factory from ``--factory`` or the JSON runtime block."""

    command_line = getattr(args, "factory", None)
    configured = getattr(config.runtime, f"{operation}_factory", "")
    return str(command_line or configured or "")


def _run_injected(operation: str, config: ExperimentConfig, args: argparse.Namespace) -> int | None:
    """Run an explicitly supplied real integration; return None when absent."""

    path = _injection_path(config, args, operation)
    if not path:
        return None
    if ":" not in path:
        print(f"BLOCKED: {operation} factory must use python.module:callable notation: {path}", file=sys.stderr)
        return 3
    module_name, callable_name = path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, callable_name)
    except (ImportError, AttributeError) as exc:
        print(f"BLOCKED: cannot import {operation} factory {path!r}: {exc}", file=sys.stderr)
        return 3
    if not callable(factory):
        print(f"BLOCKED: injected {operation} factory is not callable: {path}", file=sys.stderr)
        return 3
    try:
        result = factory(config=config, args=args)
    except Exception as exc:
        print(f"FAILED: injected {operation} factory {path!r}: {exc}", file=sys.stderr)
        return 4
    if result is None:
        print(f"FAILED: injected {operation} factory returned no result", file=sys.stderr)
        return 4
    _json_dump(result)
    if isinstance(result, dict):
        status = str(result.get("status", "")).upper()
        if status.startswith("BLOCKED"):
            return 3
        if status in {"FAILED", "ERROR", "INCOMPLETE"}:
            return 4
    return 0


def _require_ready(config: ExperimentConfig, operation: str) -> int:
    ready, report = _preflight(config)
    if not ready:
        print(f"BLOCKED: cannot {operation} until preflight passes.", file=sys.stderr)
        _json_dump(report)
        return 2
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        config = _load(getattr(args, "config", None))
        ready, report = _preflight(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INVALID CONFIG: {exc}", file=sys.stderr)
        return 2
    _json_dump(report)
    return 0 if ready else 2


def _cmd_init_config(args: argparse.Namespace) -> int:
    target = Path(args.output)
    if target.exists() and not args.force:
        print(f"REFUSED: {target} already exists; pass --force to replace it", file=sys.stderr)
        return 2
    config = _load(None)
    save_config(config, target)
    print(f"Wrote template configuration: {target}")
    print(f"Config digest: {config.digest()}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    config = _load(args.config)
    code = _require_ready(config, "collect")
    if code:
        return code
    if args.dry_run:
        _json_dump({
            "operation": "collect",
            "factory": _injection_path(config, args, "collection"),
            "config_digest": config.digest(),
            "study_protocol": dict(config.study_protocol),
            "accepted_arrow_corrections_target_per_round": config.study_protocol["accepted_arrow_corrections_per_round"],
        })
        return 0
    injected = _run_injected("collection", config, args)
    return injected if injected is not None else _run_backend("experiment", "collect_from_config", config, args)


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load(args.config)
    code = _require_ready(config, "train")
    if code:
        return code
    if args.dry_run:
        _json_dump({
            "operation": "train",
            "factory": _injection_path(config, args, "training"),
            "config_digest": config.digest(),
            "study_protocol": dict(config.study_protocol),
        })
        return 0
    injected = _run_injected("training", config, args)
    return injected if injected is not None else _run_backend("training", "train_from_config", config, args)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load(args.config)
    code = _require_ready(config, "evaluate")
    if code:
        return code
    if args.dry_run:
        _json_dump({
            "operation": "evaluate",
            "factory": _injection_path(config, args, "evaluation"),
            "config_digest": config.digest(),
            "study_protocol": dict(config.study_protocol),
        })
        return 0
    injected = _run_injected("evaluation", config, args)
    return injected if injected is not None else _run_backend("evaluation", "evaluate_from_config", config, args)


def _metric_value(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _strict_bool(record: dict[str, Any], key: str, *, path: Path, index: int) -> bool:
    """Accept JSON booleans (and legacy 0/1), never truth-test strings."""

    value = record.get(key, False)
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{path} episode {index}: {key} must be boolean or integer 0/1, got {value!r}")


def _cmd_report(args: argparse.Namespace) -> int:
    """Produce the strict paired report; incomplete evidence exits nonzero."""

    try:
        config = _load(getattr(args, "config", None))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _json_dump({"status": "INCOMPLETE", "completeness_receipt": {"errors": [str(exc)]}})
        return 2
    root = Path(args.run_root)
    files = sorted(root.glob("**/metrics.json")) if root.is_dir() else []
    errors: list[str] = []
    metrics: list[Any] = []
    try:
        from .evaluation import TrialMetric, paired_report
    except ImportError as exc:
        _json_dump({"status": "INCOMPLETE", "completeness_receipt": {"errors": [str(exc)]}})
        return 2

    if not files:
        errors.append(f"no metrics.json files found under {root}")
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: cannot parse metrics JSON: {exc}")
            continue
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            records = payload["episodes"]
        elif isinstance(payload, list):
            records = payload
        else:
            errors.append(f"{path}: expected an episodes list")
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{path} episode {index}: expected an object")
                continue
            vla_raw = str(record.get("vla", record.get("policy", ""))).lower()
            vla = VLA_ALIASES.get(vla_raw, vla_raw)
            condition = str(record.get("condition", "")).lower()
            if condition not in {
                "frozen_baseline", "adapted", "hybrid", "correction_only",
                "full_failure_context", "shuffled_failure_context", "reset_fast_state", "gdn",
            }:
                errors.append(f"{path} episode {index}: unsupported/missing paired condition {condition!r}")
                continue
            try:
                task_id = int(record["task_id"])
                seed = int(record["seed"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{path} episode {index}: missing task_id/seed/success: {exc}")
                continue
            try:
                success = _strict_bool(record, "success", path=path, index=index)
                teacher_used = _strict_bool(record, "teacher_used", path=path, index=index)
                teacher_success = _strict_bool(record, "teacher_success", path=path, index=index)
                metrics.append(
                    TrialMetric(
                        policy_id=str(record["policy_id"]),
                        vla=vla,
                        task_id=task_id,
                        seed=seed,
                        condition=condition,
                        success=success,
                        teacher_used=teacher_used,
                        teacher_success=teacher_success,
                        metadata=record.get("metadata", {}),
                        episode_id=str(record["episode_id"]),
                        initial_state_hash=str(record["initial_state_hash"]),
                        checkpoint_lineage=str(record["checkpoint_lineage"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{path} episode {index}: missing strict pairing identity: {exc}")
    report: list[dict[str, Any]] = []
    if not errors:
        try:
            required_conditions = set(config.controls)
            required_conditions.update({"frozen_baseline", "adapted"})
            report = paired_report(metrics, required_conditions=sorted(required_conditions))
        except (TypeError, ValueError) as exc:
            errors.append(f"paired_report rejected metrics: {exc}")

    expected_vlas = {VLA_ALIASES.get(name.lower(), name.lower()) for name in config.vla_names}
    expected_tasks = set(config.task_ids)
    observed_cells = {(str(row.get("vla", "")).lower(), int(row.get("task_id", -1))) for row in report}
    expected_cells = {(vla, task) for vla in expected_vlas for task in expected_tasks}
    if not errors and observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        errors.append(f"incomplete VLA/task cells: missing={missing}, extra={extra}")
    if not errors:
        for row in report:
            if row["paired_trials"] != config.episodes_per_task:
                errors.append(
                    f"{row['vla']} task {row['task_id']} has {row['paired_trials']} paired trials; "
                    f"expected exactly {config.episodes_per_task}"
                )
    receipt = {
        "status": "COMPLETE" if not errors else "INCOMPLETE",
        "expected_vla_task_cells": len(expected_cells),
        "observed_vla_task_cells": len(observed_cells),
        "expected_paired_trials_per_cell": config.episodes_per_task,
        "observed_metric_records": len(metrics),
        "metrics_files": len(files),
        "errors": errors,
    }
    _json_dump({"status": receipt["status"], "completeness_receipt": receipt, "paired_report": report if not errors else []})
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic RoboTTT-style test-time training on LIBERO")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="validate exact-artifact, split, and provenance gates")
    preflight.add_argument("--config", help="JSON config (defaults to example_config.json)")
    preflight.set_defaults(func=_cmd_preflight)

    init = sub.add_parser("init-config", help="write a fully explicit blocked template config")
    init.add_argument("output")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=_cmd_init_config)

    for name, function, help_text in (
        ("collect", _cmd_collect, "run VLA attempts and Arrow-controller teacher takeover"),
        ("train", _cmd_train, "run masked correction training with RoboTTT fidelity checks"),
        ("evaluate", _cmd_evaluate, "run paired frozen/adapted evaluation"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", help="JSON config (defaults to example_config.json)")
        command.add_argument(
            "--factory",
            help="override runtime factory as python.module:callable (must accept config= and args=)",
        )
        command.add_argument("--dry-run", action="store_true", help="validate and print backend contract without executing")
        command.set_defaults(func=function)

    report = sub.add_parser("report", help="aggregate paired metrics.json files")
    report.add_argument("run_root")
    report.add_argument("--config", help="JSON config used to define expected VLA/task cells")
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
