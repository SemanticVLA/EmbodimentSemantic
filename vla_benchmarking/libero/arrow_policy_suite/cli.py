"""Preflight, explicit native execution, and reporting CLI for the suite.

Planning remains the default. A real native rollout requires the explicit
``--execute --factory module:callable`` path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config import StudyConfig
from .contracts import ContractError
from .reporting import aggregate_rows, write_report


_LEARNED_POLICY_IDS = {"arrow_apprentice", "arrow_editor", "arrow_minimal_learned"}
_TEACHER_REQUIRED_POLICY_IDS = {
    "teacher_only", "arrow_together", "arrow_on_call", "arrow_minimal", "arrow_minimal_runtime", "arrow_fast",
}
_TEACHER_FREE_POLICY_IDS = _LEARNED_POLICY_IDS | {"frozen_base", "arrow_trace"}


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
    collect = sub.add_parser("collect", help="collect bounded native On-Call episodes")
    # Keep the historical positional ``collect CONFIG`` form for receipt-only
    # callers, while accepting the exact executable contract emitted by the
    # Legion sbatch launcher.
    collect.add_argument("config", nargs="?")
    collect.add_argument("--config", dest="config_option", help="study config to validate")
    collect.add_argument("--factory", help="executor path as python.module:callable; never imported here")
    collect.add_argument("--input", action="append", default=[], help="input artifact path (repeatable)")
    collect.add_argument("--output", help="planned output artifact path")
    collect.add_argument("--run-dir", help="planned run directory")
    collect.add_argument("--dry-run", action="store_true", help="emit a DRY_RUN receipt")
    collect.add_argument("--execute", action="store_true", help="execute through an injected native factory")
    collect.add_argument("--policy", default="frozen_base", help="policy/control id for native execution")
    collect.add_argument("--steps", type=int, help="bounded native collection steps")
    collect.add_argument("--checkpoint", help="checkpoint file/directory to hash into run provenance")
    collect.add_argument("--controller", help="controller config/file/directory to hash into run provenance")
    collect.add_argument("--graph-context-revision", help="sealed graph/arrow context revision")
    collect.add_argument("--trace-geometry-variant", choices=("rgbd", "simulator_assisted_rgbd"))
    collect.set_defaults(func=_cmd_handoff)
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


def _immutable_artifact_path(value: str | Path, *, label: str) -> Path:
    """Resolve one input as a safe, existing, non-symlink file or directory.

    Native learned policies deliberately have no fallback artifact.  Keep the
    path contract here (before a factory is imported) so create-only handoffs
    and scored execution validate the same immutable input identity.
    """

    candidate = Path(str(value))
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ContractError(f"{label} must be a safe absolute path")
    if candidate.is_symlink() or not candidate.exists() or not (candidate.is_file() or candidate.is_dir()):
        raise ContractError(f"{label} must be an existing immutable file or directory: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{label} cannot be resolved: {candidate}") from exc
    if resolved.is_symlink() or not (resolved.is_file() or resolved.is_dir()):
        raise ContractError(f"{label} must resolve to an immutable file or directory: {candidate}")
    if resolved.is_dir():
        entries = list(resolved.rglob("*"))
        if any(item.is_symlink() for item in entries):
            raise ContractError(f"{label} directory contains a symlink: {candidate}")
        if any(not (item.is_file() or item.is_dir()) for item in entries):
            raise ContractError(f"{label} directory contains a non-regular entry: {candidate}")
    return resolved


def _artifact_sha256(path: Path) -> str:
    if path.is_dir():
        entries: list[tuple[str, str]] = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix()
            entries.append((relative, _artifact_sha256(child)))
        if not entries:
            raise ContractError(f"learned artifact directory is empty: {path}")
        # Keep this byte-for-byte compatible with native_executor._hash_path.
        return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"learned artifact is unreadable: {path}") from exc
    return digest.hexdigest()


def _learned_artifact_inputs(policy_id: str, values: Any) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Validate and describe repeatable learned inputs for a handoff/execute."""

    raw_values = tuple(str(item) for item in (values or ()))
    if policy_id in _LEARNED_POLICY_IDS and len(raw_values) != 1:
        raise ContractError(
            f"{policy_id} requires exactly one --input learned artifact; "
            "refusing a frozen-policy fallback"
        )
    if policy_id not in _LEARNED_POLICY_IDS and raw_values:
        raise ContractError("learned artifacts are only valid for Apprentice, Editor, and Minimal-Learned")
    paths: list[str] = []
    records: list[dict[str, Any]] = []
    for raw in raw_values:
        path = _immutable_artifact_path(raw, label="learned artifact input")
        record: dict[str, Any] = {"path": str(path), "sha256": _artifact_sha256(path)}
        if policy_id == "arrow_apprentice":
            if not path.is_dir():
                raise ContractError("arrow_apprentice requires an immutable directory bundle")
            manifest = path / "apprentice_manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise ContractError("arrow_apprentice bundle requires apprentice_manifest.json")
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError("arrow_apprentice bundle manifest is unreadable") from exc
            if not isinstance(manifest_payload, Mapping):
                raise ContractError("arrow_apprentice bundle manifest must be a JSON object")
            inventory = manifest_payload.get("checkpoint_inventory", manifest_payload.get("inventory"))
            inventory_hash = manifest_payload.get("checkpoint_sha256", manifest_payload.get("inventory_sha256"))
            if not isinstance(inventory, Mapping) or not inventory:
                raise ContractError("arrow_apprentice bundle manifest has no hashed inventory")
            if not isinstance(inventory_hash, str) or len(inventory_hash) != 64:
                raise ContractError("arrow_apprentice bundle manifest has no inventory hash")
            actual_inventory: dict[str, str] = {}
            for item in sorted(path.rglob("*")):
                if item == manifest:
                    continue
                if item.is_symlink():
                    raise ContractError("arrow_apprentice bundle contains a symlink")
                if item.is_file():
                    actual_inventory[item.relative_to(path).as_posix()] = _artifact_sha256(item)
            if not actual_inventory:
                raise ContractError("arrow_apprentice bundle payload is empty")
            if dict(inventory) != actual_inventory:
                raise ContractError("arrow_apprentice bundle inventory does not match payload")
            expected_inventory_hash = hashlib.sha256(
                (json.dumps(dict(sorted(actual_inventory.items())), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            ).hexdigest()
            if inventory_hash != expected_inventory_hash:
                raise ContractError("arrow_apprentice bundle inventory hash mismatch")
            record["bundle_manifest"] = {
                "path": str(manifest),
                "sha256": _artifact_sha256(manifest),
                "inventory_sha256": inventory_hash,
                "payload_files": len(actual_inventory),
            }
        else:
            if not path.is_file():
                raise ContractError(f"{policy_id} requires an immutable regular checkpoint file")
            sidecar = _immutable_artifact_path(f"{path}.json", label="learned artifact sidecar")
            if not sidecar.is_file():
                raise ContractError(f"{policy_id} requires a regular .json sidecar")
            record["sidecar"] = {"path": str(sidecar), "sha256": _artifact_sha256(sidecar)}
        paths.append(str(path))
        records.append(record)
    return tuple(paths), records


def _runtime_privilege_receipt(policy_id: str) -> dict[str, Any]:
    return {
        "teacher_required": policy_id in _TEACHER_REQUIRED_POLICY_IDS,
        "teacher_free": policy_id in _TEACHER_FREE_POLICY_IDS,
    }


def _launcher_identity_receipt() -> dict[str, int]:
    """Capture only explicitly supplied paired-reset identity values."""

    identity: dict[str, int] = {}
    for env_name, field_name in (
        ("ARROW_SUITE_TASK_ID", "task_id"),
        ("ARROW_SUITE_SEED", "seed"),
        ("ARROW_SUITE_INIT_STATE_INDEX", "init_state_index"),
    ):
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        if not value.isdecimal():
            raise ContractError(f"{env_name} must be a non-negative integer")
        identity[field_name] = int(value)
    return identity


def _cmd_collect(args: argparse.Namespace) -> int:
    if not getattr(args, "config", None):
        receipt = {"status": "BLOCKED", "errors": ["collect requires --config CONFIG or positional CONFIG"], "runs_launched": False}
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
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
    policy_id = str(getattr(args, "policy", "frozen_base"))
    try:
        learned_inputs, learned_records = _learned_artifact_inputs(
            policy_id, getattr(args, "input", None)
        )
        launch_identity = _launcher_identity_receipt()
    except ContractError as exc:
        return {
            "status": "BLOCKED",
            "errors": [str(exc)],
            "runs_launched": False,
            "operation": operation,
            "inputs": list(getattr(args, "input", None) or []),
            "learned_artifacts": [],
            "runtime_privileges": _runtime_privilege_receipt(policy_id),
            "launch_identity": {},
        }
    receipt: dict[str, Any] = {
        "status": mode,
        "operation": operation,
        "runs_launched": False,
        "execution": "not_launched",
        "factory": getattr(args, "factory", None),
        "inputs": list(learned_inputs),
        "learned_artifacts": learned_records,
        "runtime_privileges": _runtime_privilege_receipt(policy_id),
        "launch_identity": launch_identity,
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
    if str(getattr(args, "command", "")) == "collect":
        positional = getattr(args, "config", None)
        option = getattr(args, "config_option", None)
        if positional and option:
            print(json.dumps({"status": "BLOCKED", "errors": ["collect accepts one config path"], "runs_launched": False}, indent=2))
            return 2
        if option:
            args.config = option
        if not bool(getattr(args, "execute", False)):
            return _cmd_collect(args)
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
    if str(args.command) in {"collect", "evaluate"} and getattr(args, "steps", None) is None:
        print(json.dumps({
            "status": "BLOCKED",
            "errors": [f"--execute {args.command} requires explicit --steps; no horizon default is permitted"],
            "runs_launched": False,
        }, indent=2))
        return 2
    if str(args.command) == "evaluate" and int(args.steps) not in {280, 1200}:
        print(json.dumps({
            "status": "BLOCKED",
            "errors": ["--execute evaluate accepts only the predeclared 280 or 1200 step horizons"],
            "runs_launched": False,
        }, indent=2))
        return 2
    if getattr(args, "steps", None) is not None and int(args.steps) < 1:
        print(json.dumps({
            "status": "BLOCKED",
            "errors": ["--steps must be a positive integer"],
            "runs_launched": False,
        }, indent=2))
        return 2
    try:
        learned_inputs, learned_records = _learned_artifact_inputs(
            str(args.policy), getattr(args, "input", None)
        )
        launch_identity = _launcher_identity_receipt()
        # Pass canonical immutable paths to the native factory.  This is the
        # exact contract consumed by native_legion_factory implementations:
        # ``learned_artifacts=tuple(Path(...), ...)``.
        args.input = list(learned_inputs)
        config, loaded = _load_config_bundle(args.config)
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
        output_receipt = receipt.to_dict()
        output_receipt["learned_artifacts"] = learned_records
        output_receipt["runtime_privileges"] = _runtime_privilege_receipt(str(args.policy))
        output_receipt["launch_identity"] = launch_identity
        print(json.dumps(output_receipt, indent=2, sort_keys=True, default=str))
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
