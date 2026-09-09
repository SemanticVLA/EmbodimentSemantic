"""Preflight, explicit native execution, and reporting CLI for the suite.

Planning remains the default. A real native rollout requires the explicit
``--execute --factory module:callable`` path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import StudyConfig
from .contracts import ContractError
from .reporting import aggregate_rows, write_report


_LEARNED_POLICY_IDS = {"arrow_apprentice", "arrow_editor", "arrow_minimal_learned"}


def _load_config_bundle(path: str | Path) -> tuple[StudyConfig, Any | None]:
    """Load a raw config, signed config manifest, or frozen-study artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("study config must be a JSON object")
    if payload.get("schema") == "arrow_policy_suite.frozen_study.v1":
        from .config import FrozenStudyManifest
        frozen = FrozenStudyManifest.from_dict(payload)
        return frozen.config, frozen
    if payload.get("schema") == "arrow_policy_suite.protocol_seal.v1":
        from .config import ProtocolSeal
        seal = ProtocolSeal.from_dict(payload)
        return seal.config, seal
    if payload.get("schema") == "arrow_policy_suite.study_config.v1":
        # Verify both config and identity seals before reconstructing.  This
        # also strips digest fields without silently accepting a tampered file.
        StudyConfig.verify_manifest(payload)
        return StudyConfig.from_manifest(payload), None
    kwargs = dict(payload)
    for field in ("task_ids", "learned_seeds"):
        if field in kwargs:
            kwargs[field] = tuple(kwargs[field])
    for field in ("test_reset_ids", "validation_reset_ids"):
        if field in kwargs:
            normalized: dict[int | str, tuple[str, ...]] = {}
            for key, value in dict(kwargs[field]).items():
                try:
                    normalized_key: int | str = int(key)
                except (TypeError, ValueError):
                    normalized_key = str(key)
                normalized[normalized_key] = tuple(str(item) for item in value)
            kwargs[field] = normalized
    kwargs.pop("schema", None)
    kwargs.pop("config_sha256", None)
    kwargs.pop("identity_seal_sha256", None)
    return StudyConfig(**kwargs), None


def _load_config(path: str | Path) -> StudyConfig:
    """Backward-compatible config-only view of :func:`_load_config_bundle`."""

    return _load_config_bundle(path)[0]


def preflight(config: StudyConfig) -> dict[str, Any]:
    """Return a machine-readable readiness receipt without launching work."""

    try:
        manifest = config.manifest()
    except (ContractError, TypeError, ValueError) as exc:
        return {
            "status": "BLOCKED",
            "errors": [str(exc)],
            "operation": "preflight_only",
            "runs_launched": False,
            "written": False,
        }
    return {
        "status": "READY",
        "errors": [],
        "operation": "preflight_only",
        "runs_launched": False,
        "written": False,
        "manifest_digest": manifest["config_sha256"],
        "config": manifest,
        "tasks": list(config.task_ids),
        "attempts_per_task": config.on_call_attempts_per_task,
        "horizons": {"secondary": config.secondary_horizon, "primary": config.primary_horizon},
    }


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("rows", payload.get("episodes", []))
    if not isinstance(payload, list):
        raise ContractError("rows input must be a JSON list or object containing rows")
    return payload


def _row_from_mapping(value: Mapping[str, Any]):
    from .benchmark import EvaluationRow

    required = ("case", "family", "success_280", "success_1200", "steps")
    missing = [key for key in required if key not in value]
    if missing:
        raise ContractError(f"evaluation row missing fields: {missing}")
    return EvaluationRow(
        str(value["case"]), str(value["family"]), value.get("variant"),
        bool(value["success_280"]), bool(value["success_1200"]), int(value["steps"]),
        int(value.get("teacher_proposals", 0)), int(value.get("teacher_steps", 0)),
        int(value.get("branch_steps", 0)), value.get("metadata", {}),
        value.get("task_id"), str(value.get("reset_id", "")), str(value.get("episode_id", "")),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arrow policy suite preflight and reporting")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="validate config; never launch runs")
    pre.add_argument("config")
    pre.add_argument("--output", help="create-only path for the sealed manifest")
    pre.add_argument("--dry-run", action="store_true", help="verify inputs without writing output")
    pre.add_argument("--collection-split", help="typed collection reset-split manifest")
    pre.add_argument("--validation-split", help="typed validation reset-split manifest")
    pre.add_argument("--test-split", help="typed test reset-split manifest")
    pre.add_argument("--retained-training-manifest", action="append", default=[],
                     help="retained training manifest path or SHA-256 digest (repeatable)")
    pre.set_defaults(func=_cmd_preflight)
    collect = sub.add_parser("collect", help="show collection contract; never launch runs")
    collect.add_argument("config")
    collect.set_defaults(func=_cmd_collect)
    report = sub.add_parser("report", help="aggregate completed JSON rows")
    report.add_argument("rows")
    report.add_argument("output", nargs="?")
    report.add_argument("--cases", default="", help="comma-separated required cases")
    report.set_defaults(func=_cmd_report)
    for name, help_text in (
        ("canary", "prepare a simulator/policy canary handoff"),
        ("derive", "prepare derived Trace/dataset-view handoff"),
        ("train-apprentice", "prepare native Apprentice training handoff"),
        ("train-residuals", "prepare Editor/Minimal residual training handoff"),
        ("train-fast-slow", "prepare slow Fast support-training handoff"),
        ("prepare-fast-support", "prepare Fast support examples from eligible rows"),
        ("evaluate", "prepare paired evaluation handoff"),
        ("audit", "prepare artifact/checkpoint/lineage audit handoff"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", help="study config to validate (optional for a receipt-only handoff)")
        command.add_argument("--factory", help="executor path as python.module:callable; never imported here")
        command.add_argument("--input", action="append", default=[], help="input artifact path (repeatable)")
        command.add_argument("--output", help="planned output artifact path")
        command.add_argument("--run-dir", help="planned run directory")
        command.add_argument("--dry-run", action="store_true", help="emit a DRY_RUN receipt")
        command.add_argument("--execute", action="store_true", help="execute through an injected native factory")
        command.add_argument("--policy", default="frozen_base", help="policy/control id for native execution")
        command.add_argument("--steps", type=int, help="bounded native canary/evaluation steps")
        command.add_argument("--checkpoint", help="checkpoint file/directory to hash into run provenance")
        command.add_argument("--controller", help="controller config/file/directory to hash into run provenance")
        command.add_argument("--graph-context-revision", help="sealed graph/arrow context revision")
        command.add_argument("--trace-geometry-variant", choices=("rgbd", "simulator_assisted_rgbd"))
        command.set_defaults(func=_cmd_handoff)
    return parser


def _cmd_preflight(args: argparse.Namespace) -> int:
    try:
        config, loaded_frozen = _load_config_bundle(args.config)
        receipt = preflight(config)
        split_values = {
            "collection": getattr(args, "collection_split", None),
            "validation": getattr(args, "validation_split", None),
            "test": getattr(args, "test_split", None),
        }
        provided_splits = [value for value in split_values.values() if value]
        if provided_splits and len(provided_splits) != 3:
            raise ContractError("collection, validation, and test split manifests must be supplied together")
        if len(provided_splits) == 3:
            from .config import FrozenStudyManifest, ProtocolSeal
            from .splits import read_split_manifest

            manifests = {name: read_split_manifest(path) for name, path in split_values.items()}
            retained = tuple(_training_manifest_digest(path) for path in args.retained_training_manifest)
            if retained:
                frozen = FrozenStudyManifest(config, manifests["collection"], manifests["validation"], manifests["test"], retained)
                receipt["frozen_study"] = frozen.to_dict()
                receipt["manifest_digest"] = frozen.composite_sha256
            else:
                protocol = ProtocolSeal(config, manifests["collection"], manifests["validation"], manifests["test"])
                receipt["protocol_seal"] = protocol.to_dict()
                receipt["manifest_digest"] = protocol.protocol_sha256
        elif loaded_frozen is not None:
            if args.retained_training_manifest:
                raise ContractError("retained training manifests cannot be added to an existing frozen study artifact")
            frozen = loaded_frozen
            if hasattr(frozen, "protocol_sha256"):
                receipt["protocol_seal"] = frozen.to_dict()
                receipt["manifest_digest"] = frozen.protocol_sha256
            else:
                receipt["frozen_study"] = frozen.to_dict()
                receipt["manifest_digest"] = frozen.composite_sha256
        elif args.retained_training_manifest:
            raise ContractError("retained training manifests require all three typed split manifests")
        output = getattr(args, "output", None)
        receipt["mode"] = "DRY_RUN" if args.dry_run else "CREATE_ONLY"
        if output and not args.dry_run and receipt["status"] == "READY":
            from .artifacts import write_json_artifact
            payload = receipt.get("frozen_study", receipt.get("protocol_seal", receipt.get("config", {})))
            kind = "frozen-study-manifest" if "frozen_study" in receipt else ("protocol-seal" if "protocol_seal" in receipt else "study-config-manifest")
            write_json_artifact(output, payload, kind=kind)
            receipt["written"] = True
            receipt["output"] = str(Path(output))
    except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
        receipt = {"status": "BLOCKED", "errors": [str(exc)], "runs_launched": False, "written": False}
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0 if receipt["status"] == "READY" else 2


def _training_manifest_digest(path_or_digest: str) -> str:
    value = str(path_or_digest)
    if len(value) == 64 and value == value.lower() and all(char in "0123456789abcdef" for char in value):
        return value
    data = Path(value).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _cmd_collect(args: argparse.Namespace) -> int:
    try:
        config = _load_config(args.config)
        receipt = preflight(config)
    except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
        receipt = {"status": "BLOCKED", "errors": [str(exc)], "runs_launched": False}
    receipt = {**receipt, "command": "collect", "status": "READY_TO_COLLECT" if receipt["status"] == "READY" else "BLOCKED"}
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    # A successful collect command is only a validated handoff, never evidence
    # that 50 attempts were run.
    return 0 if receipt["status"] == "READY_TO_COLLECT" else 2


def _handoff_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Describe an injected operation without importing or invoking it."""

    operation = str(args.command)
    mode = "DRY_RUN" if bool(getattr(args, "dry_run", False)) else "CREATE_ONLY"
    receipt: dict[str, Any] = {
        "status": mode,
        "operation": operation,
        "runs_launched": False,
        "execution": "not_launched",
        "factory": getattr(args, "factory", None),
        "inputs": list(getattr(args, "input", None) or []),
        "output": getattr(args, "output", None),
        "run_dir": getattr(args, "run_dir", None),
        "config": None,
        "handoff": "Pass --execute and --factory module:callable to run the explicit native executor.",
    }
    config_path = getattr(args, "config", None)
    if config_path:
        try:
            config, loaded_frozen = _load_config_bundle(config_path)
            config_receipt = preflight(config)
            if loaded_frozen is not None:
                if hasattr(loaded_frozen, "protocol_sha256"):
                    config_receipt["protocol_seal"] = loaded_frozen.to_dict()
                    config_receipt["manifest_digest"] = loaded_frozen.protocol_sha256
                else:
                    config_receipt["frozen_study"] = loaded_frozen.to_dict()
                    config_receipt["manifest_digest"] = loaded_frozen.composite_sha256
        except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
            receipt.update(status="BLOCKED", errors=[str(exc)])
            return receipt
        receipt["config"] = config_receipt
        if config_receipt["status"] != "READY":
            receipt.update(status="BLOCKED", errors=config_receipt.get("errors", []))
    else:
        receipt["config"] = {"status": "UNSPECIFIED", "runs_launched": False}
    return receipt


def _cmd_handoff(args: argparse.Namespace) -> int:
    if bool(getattr(args, "execute", False)):
        return _cmd_execute(args)
    receipt = _handoff_receipt(args)
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0 if receipt["status"] in {"DRY_RUN", "CREATE_ONLY"} else 2


def _cmd_execute(args: argparse.Namespace) -> int:
    """Execute a bounded operation only when explicitly requested."""
    if bool(getattr(args, "dry_run", False)):
        print(json.dumps({"status": "BLOCKED", "errors": ["--execute and --dry-run are mutually exclusive"], "runs_launched": False}, indent=2))
        return 2
    if not args.factory:
        print(json.dumps({"status": "BLOCKED", "errors": ["--execute requires --factory module:callable"], "runs_launched": False}, indent=2))
        return 2
    if not args.config:
        print(json.dumps({"status": "BLOCKED", "errors": ["--execute requires --config with a sealed StudyConfig/ProtocolSeal"], "runs_launched": False}, indent=2))
        return 2
    if not args.run_dir:
        print(json.dumps({"status": "BLOCKED", "errors": ["--execute requires a new --run-dir"], "runs_launched": False}, indent=2))
        return 2
    try:
        config, loaded = _load_config_bundle(args.config)
        if args.policy in _LEARNED_POLICY_IDS and not list(getattr(args, "input", ()) or ()):
            raise ContractError(
                f"{args.policy} requires at least one --input learned checkpoint/runner artifact; "
                "refusing a frozen-policy fallback"
            )
        from .native_executor import execute_native, import_callable
        factory = import_callable(args.factory)
        default_steps = {"canary": 3, "evaluate": 1200, "collect": 60}.get(args.command, 3)
        receipt = execute_native(
            factory, config=config, operation=args.command, policy_id=args.policy,
            run_dir=args.run_dir, output=args.output, max_steps=args.steps or default_steps,
            graph_context_revision=args.graph_context_revision,
            trace_geometry_variant=args.trace_geometry_variant,
            checkpoint=args.checkpoint, controller=args.controller,
            learned_artifacts=tuple(getattr(args, "input", ()) or ()),
            protocol_seal=loaded if loaded is not None and hasattr(loaded, "protocol_sha256") else None,
        )
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True, default=str))
        return 0 if receipt.status == "COMPLETED" else 2
    except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
        print(json.dumps({"status": "BLOCKED", "errors": [str(exc)], "runs_launched": False}, indent=2))
        return 2


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        rows = [_row_from_mapping(value) for value in _load_rows(args.rows)]
        cases = tuple(value for value in args.cases.split(",") if value) or None
        report = aggregate_rows(rows, required_cases=cases)
        if args.output:
            write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, ContractError) as exc:
        print(json.dumps({"status": "INCOMPLETE", "errors": [str(exc)]}, indent=2))
        return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "preflight"]
