"""Explicit native execution path used by the CLI and Legion canaries.

Importing this module is cheap.  Heavy LIBERO/LeRobot/Arrow loading remains in
the injected factory supplied with ``--factory module:callable``.  The default
CLI path therefore remains a safe plan/dry-run, while ``--execute`` is a real,
fail-closed run rather than another receipt-only handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

from .artifacts import write_json_artifact
from .collection import TrainingSourceWriter
from .config import ProtocolSeal, StudyConfig
from .contracts import ContractError
from .native_host import NativeHost


def import_callable(spec: str) -> Callable[..., Any]:
    if not isinstance(spec, str) or ":" not in spec:
        raise ContractError("factory must be a python.module:callable specification")
    module_name, attr_name = spec.split(":", 1)
    if not module_name or not attr_name or any(part == "" for part in attr_name.split(".")):
        raise ContractError("factory must be a python.module:callable specification")
    try:
        value: Any = importlib.import_module(module_name)
        for part in attr_name.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise ContractError(f"cannot import native factory {spec!r}") from exc
    if not callable(value):
        raise ContractError(f"native factory {spec!r} is not callable")
    return value


def _hash_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return hashlib.sha256(target.read_bytes()).hexdigest()
    if target.is_dir():
        entries = []
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            relative = child.relative_to(target).as_posix()
            entries.append((relative, hashlib.sha256(child.read_bytes()).hexdigest()))
        return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()
    raise ContractError(f"cannot hash missing checkpoint/controller path: {target}")


def _git_revision() -> str:
    try:
        value = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("cannot resolve exact git revision for native run") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError("git revision is not a full lowercase SHA-1")
    return value


def _hooks(component: Any, name: str, *, required: bool = True) -> tuple[Any, Any] | None:
    if component is None:
        if required:
            raise ContractError(f"{name} is required")
        return None
    if getattr(component, "rollback_complete", True) is False:
        raise ContractError(f"{name} reports incomplete rollback hooks")
    pairs = (("snapshot_state", "restore_state"), ("snapshot", "restore"))
    for snapshot_name, restore_name in pairs:
        snapshot = getattr(component, snapshot_name, None)
        restore = getattr(component, restore_name, None)
        if callable(snapshot) != callable(restore):
            raise ContractError(f"{name} exposes only one of {snapshot_name}/{restore_name}")
        if callable(snapshot):
            return snapshot, restore
    if required:
        raise ContractError(f"{name} requires paired snapshot/restore hooks")
    return None


def _validate_fast_lifecycle(receipt: Any) -> dict[str, Any]:
    """Validate the one-shot Fast support receipt before scored execution."""
    if receipt is None:
        raise ContractError("arrow_fast host lacks a verified Fast lifecycle receipt")
    required = (
        "support_attempts", "support_steps", "support_complete", "restored_t0",
        "fast_slot_capacity", "scored_teacher_calls", "teacher_destroyed",
        "non_fast_changed", "slow_manifest_sha256_before", "slow_manifest_sha256_after",
        "vla_manifest_sha256_before", "vla_manifest_sha256_after",
    )
    missing = [name for name in required if not hasattr(receipt, name)]
    if missing:
        raise ContractError(f"arrow_fast lifecycle receipt is missing {missing}")
    if int(receipt.support_attempts) != 1 or int(receipt.support_steps) != 1:
        raise ContractError("arrow_fast requires exactly one complete support observation")
    if receipt.support_complete is not True or receipt.restored_t0 is not True:
        raise ContractError("arrow_fast Fast support was incomplete or did not restore t0")
    if int(receipt.fast_slot_capacity) != 448:
        raise ContractError("arrow_fast lifecycle receipt must report 448 fast slots")
    if int(receipt.scored_teacher_calls) != 0:
        raise ContractError("arrow_fast scored path made teacher calls")
    if receipt.teacher_destroyed is not True:
        raise ContractError("arrow_fast support teacher was not destroyed")
    # Older teacher implementations expose only close(), so detached=False is
    # valid when no explicit detach-support field exists.  If a newer receipt
    # reports that detach is supported, require the stronger invariant.
    if getattr(receipt, "teacher_detach_supported", False) and receipt.teacher_detached is not True:
        raise ContractError("arrow_fast support teacher was not detached")
    if int(receipt.non_fast_changed) != 0:
        raise ContractError("arrow_fast changed non-fast parameters")
    hash_values: dict[str, str] = {}
    for label in ("slow_manifest_sha256", "vla_manifest_sha256"):
        before = getattr(receipt, f"{label}_before")
        after = getattr(receipt, f"{label}_after")
        if not isinstance(before, str) or not before or before != after:
            raise ContractError(f"arrow_fast {label} changed across support adaptation")
        hash_values[label] = before
    return {
        "support_attempts": int(receipt.support_attempts),
        "support_steps": int(receipt.support_steps),
        "support_complete": True,
        "restored_t0": True,
        "fast_slot_capacity": int(receipt.fast_slot_capacity),
        "scored_teacher_calls": int(receipt.scored_teacher_calls),
        "teacher_destroyed": True,
        "teacher_detached": bool(getattr(receipt, "teacher_detached", False)),
        "non_fast_changed": int(receipt.non_fast_changed),
        "slow_manifest_sha256": hash_values["slow_manifest_sha256"],
        "vla_manifest_sha256": hash_values["vla_manifest_sha256"],
    }


def production_preflight(
    host: NativeHost,
    config: StudyConfig,
    *,
    policy_id: str,
    graph_context_revision: str | None = None,
    trace_geometry_variant: str | None = None,
) -> dict[str, Any]:
    """Fail closed before the first simulator transition.

    This is intentionally stricter than the dependency-free tests: a native
    run must be replayable and must expose the graph/geometry provenance it is
    claiming to evaluate.
    """

    if not isinstance(host, NativeHost):
        raise ContractError("native factory must return NativeHost")
    allowed_policy_ids = {
        "frozen_base", "teacher_only", "arrow_together", "arrow_on_call",
        "arrow_apprentice", "arrow_editor", "arrow_minimal", "arrow_minimal_runtime",
        "arrow_minimal_learned", "arrow_fast", "arrow_trace",
    }
    if policy_id not in allowed_policy_ids:
        raise ContractError(f"unknown native policy/control {policy_id!r}")
    config.validate()
    _hooks(host.environment, "LIBERO environment")
    _hooks(host.vla, "SmolVLA")
    if not bool(getattr(host.vla, "rollback_complete", True)):
        raise ContractError("SmolVLA rollback is incomplete")
    n_action_steps = getattr(host.vla, "n_action_steps", None)
    if n_action_steps is None:
        policy = getattr(host.vla, "policy", None)
        cfg = getattr(policy, "config", None)
        n_action_steps = getattr(cfg, "n_action_steps", None)
    if n_action_steps is not None and int(n_action_steps) != 1:
        raise ContractError("native causal arbitration requires SmolVLA n_action_steps=1")
    if host.teacher is not None:
        _hooks(host.teacher, "interruptible Arrow teacher")
        for hook_name in ("propose", "commit", "interrupt"):
            if not callable(getattr(host.teacher, hook_name, None)):
                raise ContractError(f"interruptible Arrow teacher requires {hook_name}()")
    teacher_required = {
        "teacher_only", "arrow_together", "arrow_on_call", "arrow_minimal",
        "arrow_minimal_runtime",
    }
    if policy_id in teacher_required and host.teacher is None:
        raise ContractError(f"{policy_id} requires a same-frame Arrow teacher")
    if host.graph_context_fn is None and policy_id in {"arrow_fast", "arrow_trace"}:
        raise ContractError(f"{policy_id} requires an explicit graph-context callback")
    learned_policy = getattr(host.policy, "policy_id", None)
    if policy_id == "arrow_apprentice":
        if learned_policy != "arrow_apprentice" or not callable(getattr(host.policy, "action_fn", None)):
            raise ContractError("arrow_apprentice host lacks a loaded action runner/checkpoint hook")
    if policy_id == "arrow_editor":
        if learned_policy != "arrow_editor" or not callable(getattr(host.policy, "residual_fn", None)):
            raise ContractError("arrow_editor host lacks a loaded residual runner/checkpoint hook")
    if policy_id == "arrow_minimal_learned":
        if learned_policy != "arrow_minimal" or getattr(host.policy, "variant", None) != "learned":
            raise ContractError("arrow_minimal_learned host lacks its learned variant")
        if not callable(getattr(host.policy, "learned_fn", None)) and not callable(getattr(host.policy, "residual_fn", None)):
            raise ContractError("arrow_minimal_learned host lacks a loaded action/residual runner hook")
    if policy_id == "arrow_minimal_runtime":
        if getattr(host.policy, "policy_id", None) != "arrow_minimal" or getattr(host.policy, "variant", None) != "runtime_oracle":
            raise ContractError("arrow_minimal_runtime host lacks its runtime policy")
        branch_runner = getattr(host.policy, "branch_runner", None)
        if branch_runner is None or not callable(getattr(branch_runner, "run_all", None)):
            raise ContractError("arrow_minimal_runtime requires a real branch runner")
        if getattr(branch_runner, "is_real", False) is not True:
            raise ContractError("arrow_minimal_runtime branch runner is not a real sandbox")
        if (getattr(branch_runner, "composite_snapshot", None) is None
                and getattr(branch_runner, "require_state_isolation", False) is not True):
            raise ContractError("arrow_minimal_runtime requires composite VLA/teacher/policy/RNG isolation")
        if int(getattr(branch_runner, "horizon", 0)) != 20:
            raise ContractError("arrow_minimal_runtime branch horizon must be exactly 20")
        if tuple(getattr(branch_runner, "masks", ())) != tuple(range(8)):
            raise ContractError("arrow_minimal_runtime requires all eight action-group masks")
    if policy_id == "arrow_fast":
        corrector = getattr(host.policy, "corrector", None)
        if learned_policy != "arrow_fast" or not callable(getattr(corrector, "correction", None)):
            raise ContractError("arrow_fast host lacks a loaded fast corrector artifact")
        if host.teacher is not None:
            raise ContractError("arrow_fast scored host must be teacher-free after support adaptation")
        fast_lifecycle = _validate_fast_lifecycle(getattr(host, "fast_lifecycle_receipt", None))
    if graph_context_revision is not None:
        if str(graph_context_revision).strip().lower() in {"", "unresolved", "unknown", "latest"}:
            raise ContractError("graph_context_revision must be explicit")
        if graph_context_revision != config.trace_geometry_provider_revision and policy_id in {"arrow_fast", "arrow_trace"}:
            raise ContractError("graph context revision disagrees with sealed study revision")
    if policy_id == "arrow_trace":
        if trace_geometry_variant not in {"rgbd", "simulator_assisted_rgbd"}:
            raise ContractError("Trace requires explicit RGB-D geometry variant")
        if trace_geometry_variant == "rgbd" and str(config.trace_geometry_provider_revision).lower().startswith("sim"):
            raise ContractError("vision-only Trace cannot use a simulator geometry revision")
    result = {
        "status": "READY",
        "policy_id": policy_id,
        "n_action_steps": 1,
        "rollback": {"environment": True, "vla": True, "teacher": host.teacher is not None},
        "graph_context_revision": graph_context_revision,
        "trace_geometry_variant": trace_geometry_variant,
    }
    if policy_id == "arrow_fast":
        result["fast_lifecycle"] = fast_lifecycle
    return result


@dataclass(frozen=True)
class NativeExecutionReceipt:
    status: str
    operation: str
    policy_id: str
    steps: int
    success: bool
    terminal: bool
    run_dir: str
    output: str | None
    manifest: Mapping[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "arrow_policy_suite.native_execution.v1",
            "status": self.status,
            "operation": self.operation,
            "policy_id": self.policy_id,
            "steps": self.steps,
            "success": self.success,
            "terminal": self.terminal,
            "run_dir": self.run_dir,
            "output": self.output,
            "manifest": dict(self.manifest),
            "error": self.error,
        }


def _resolve_host(factory: Callable[..., Any], **kwargs: Any) -> NativeHost:
    value = factory(**kwargs)
    if isinstance(value, NativeHost):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("host"), NativeHost):
        return value["host"]
    raise ContractError("native factory must return NativeHost or {'host': NativeHost}")


def _native_identity(host: NativeHost, *, operation: str) -> dict[str, Any]:
    """Resolve explicit Legion task/reset identity; never invent scored IDs."""
    required = ("task_id", "seed", "init_state_index")
    missing = [name for name in required if not hasattr(host, name) or getattr(host, name) in (None, "")]
    if missing and operation in {"collect", "evaluate"}:
        raise ContractError(
            f"native {operation} requires explicit host identity fields: {', '.join(missing)}"
        )
    if missing:
        return {}
    task_id = getattr(host, "task_id")
    seed = getattr(host, "seed")
    init_state_index = getattr(host, "init_state_index")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("native host seed must be a non-negative integer")
    if isinstance(init_state_index, bool) or not isinstance(init_state_index, int) or init_state_index < 0:
        raise ContractError("native host init_state_index must be a non-negative integer")
    if isinstance(task_id, bool) or task_id in (None, ""):
        raise ContractError("native host task_id is required")
    reset_identity = getattr(host, "reset_identity", None)
    reset_digest = getattr(reset_identity, "digest", None) if reset_identity is not None else None
    episode_id = getattr(reset_identity, "episode_id", None) if reset_identity is not None else None
    # Native LIBERO hosts expose the explicit init-state index rather than a
    # ResetIdentity object.  This is an identity from the launcher, not a
    # generated reset token or the protocol seal digest.
    reset_id = str(reset_digest) if reset_digest else f"init_state_index:{init_state_index}"
    episode_id = str(episode_id) if episode_id else f"task-{task_id}-seed-{seed}-init-{init_state_index}"
    resolved: dict[str, Any] = {
        "task_id": task_id,
        "seed": seed,
        "init_state_index": init_state_index,
        "reset_id": reset_id,
        "episode_id": episode_id,
    }
    if reset_digest:
        resolved["reset_identity_sha256"] = str(reset_digest)
    return resolved


def execute_native(
    factory: Callable[..., Any], *,
    config: StudyConfig,
    operation: str,
    policy_id: str,
    run_dir: str | Path,
    output: str | Path | None = None,
    max_steps: int = 3,
    graph_context_revision: str | None = None,
    trace_geometry_variant: str | None = None,
    checkpoint: str | Path | None = None,
    controller: str | Path | None = None,
    learned_artifacts: tuple[str, ...] = (),
    protocol_seal: ProtocolSeal | None = None,
    training_source: str | Path | None = None,
) -> NativeExecutionReceipt:
    if operation not in {"canary", "collect", "evaluate"}:
        raise ContractError(
            f"native executor does not implement operation {operation!r}; "
            "use its dedicated training/derivation launcher"
        )
    target = Path(run_dir)
    if target.exists():
        raise ContractError(f"refusing to reuse existing native run directory: {target}")
    if output is not None and Path(output).exists():
        raise ContractError(f"refusing to overwrite native execution receipt: {output}")
    learned_artifact_hashes = tuple(_hash_path(path) for path in learned_artifacts)
    if int(max_steps) <= 0:
        raise ContractError("max_steps must be positive")
    target.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "arrow_policy_suite.native_run.v1",
        "git_revision": _git_revision(),
        "config_sha256": config.config_sha256(),
        "identity_seal_sha256": config.identity_seal_sha256(),
        "protocol_seal_sha256": None if protocol_seal is None else protocol_seal.protocol_sha256,
        "operation": operation,
        "policy_id": policy_id,
        "max_steps": int(max_steps),
        "checkpoint_sha256": None if checkpoint is None else _hash_path(checkpoint),
        "controller_sha256": None if controller is None else _hash_path(controller),
        "learned_artifact_sha256": list(learned_artifact_hashes),
    }
    host: NativeHost | None = None
    try:
        factory_kwargs = {
            "config": config, "operation": operation, "policy_id": policy_id,
            "run_dir": target, "max_steps": int(max_steps),
        }
        if learned_artifacts:
            factory_kwargs["learned_artifacts"] = tuple(Path(path) for path in learned_artifacts)
        host = _resolve_host(factory, **factory_kwargs)
        preflight = production_preflight(
            host, config, policy_id=policy_id,
            graph_context_revision=graph_context_revision,
            trace_geometry_variant=trace_geometry_variant,
        )
        run_identity = _native_identity(host, operation=operation)
        if run_identity:
            manifest["identity"] = dict(run_identity)
            manifest.update({
                "task_id": run_identity["task_id"],
                "seed": run_identity["seed"],
                "init_state_index": run_identity["init_state_index"],
                "reset_id": run_identity["reset_id"],
                "episode_id": run_identity["episode_id"],
            })
        records = host.run(max_steps=int(max_steps), reset_environment=False)
        manifest["preflight"] = preflight
        manifest["steps"] = len(records)
        manifest["success"] = bool(records and records[-1].success)
        manifest["terminal"] = bool(records and records[-1].terminal)
        manifest["step_digests"] = [
            {"timestep": item.frame.timestep, "frame_digest": item.frame.digest,
             "next_frame_digest": item.next_frame.digest, "executed_by": item.executed_by,
             "action": list(item.action), "teacher_available": item.teacher_status.available}
            for item in records
        ]
        if operation == "collect":
            # _native_identity is mandatory for collect/evaluate, so rows can
            # never be attributed to an unknown or protocol-seal-derived reset.
            identity_payload: Mapping[str, Any] = run_identity
            source_path = Path(training_source) if training_source is not None else target / "training_source.jsonl"
            source_writer = TrainingSourceWriter(source_path)
            source_hashes = {
                "config_sha256": manifest["config_sha256"],
                "identity_seal_sha256": manifest["identity_seal_sha256"],
                "protocol_seal_sha256": manifest["protocol_seal_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "controller_sha256": manifest["controller_sha256"],
            }
            rows = [source_writer.append(
                item, identity=identity_payload, source_hashes=source_hashes,
                episode_success=bool(manifest["success"]),
                episode_complete=bool(manifest["terminal"]),
            )
                    for item in records]
            manifest["training_source"] = {
                "schema": "arrow_policy_suite.training_source.v1",
                "path": str(source_path),
                "sha256": source_writer.sha256(),
                "rows": len(rows),
                "eligible_rows": sum(bool(row.get("eligible", False)) for row in rows),
            }
        manifest_path = target / "run_manifest.json"
        write_json_artifact(manifest_path, manifest, kind="native-run-manifest",
                            lineage=tuple(value for value in (manifest["config_sha256"], manifest["identity_seal_sha256"]) if value))
        receipt = NativeExecutionReceipt(
            "COMPLETED", operation, policy_id, len(records), bool(manifest["success"]),
            bool(manifest["terminal"]), str(target), str(output) if output else None, manifest,
        )
    except BaseException as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        error_path = target / "failure_manifest.json"
        try:
            write_json_artifact(error_path, manifest, kind="native-run-failure")
        except Exception:
            pass
        receipt = NativeExecutionReceipt(
            "FAILED", operation, policy_id, int(manifest.get("steps", 0)),
            bool(manifest.get("success", False)), bool(manifest.get("terminal", False)),
            str(target), str(output) if output else None, manifest, manifest["error"],
        )
    finally:
        if host is not None:
            close = getattr(host, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    manifest["close_error"] = f"{type(exc).__name__}: {exc}"
    if output is not None:
        write_json_artifact(output, receipt.to_dict(), kind="native-execution-receipt")
    return receipt


__all__ = ["NativeExecutionReceipt", "execute_native", "import_callable", "production_preflight"]
