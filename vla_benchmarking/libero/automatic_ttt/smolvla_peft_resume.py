"""Fail-closed publication-only recovery for a preserved SmolVLA run.

This module deliberately has no collector, trainer, or evaluator imports.  A
resume is allowed to consume a previously preserved run only after all three
completed ML stages have been independently checked.  Its only write to the
model output is the atomic :func:`save_peft_adapter` call; the source run is
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from .eval_reset_contract import load_eval_reset_contract, validate_adapted_randomization_audit
from .peft_artifacts import (
    FRESH_COLLECTION_MODE,
    FRESH_COLLECTION_SOURCE_KIND,
    PINNED_OPTIMIZER,
    PEFTArtifactError,
    load_arrow_collection_manifest,
    load_peft_manifest,
    save_peft_adapter,
    sha256_file,
    tree_sha256,
)


SOURCE_JOB_ID = 1925947
TRAINING_COMMIT = "01215121028d08cd22ad964a58758a27b04f8f5e"
GLOBAL_BATCH_SIZE = 8
REQUESTED_EPOCHS = 5
TASK_ID = 0
SEALED_SEEDS = list(range(1000, 1010))


class ResumeError(ValueError):
    """The preserved run cannot safely be published as a PEFT artifact."""


@dataclass(frozen=True)
class ResumeSource:
    source_run_root: Path
    archive_status: Path
    collection_manifest: Path
    dataset_root: Path
    dataset_manifest: Path
    checkpoint: Path
    checkpoint_step: int
    runtime_evidence: Path
    training_plan: Path
    adapted_eval: Path
    adapted_audit: Path
    eval_reset_contract: Path
    expected_inventory: Path
    collection_success_count: int
    dataset_frames: int
    controller_config_hash: str
    source_collection_manifest_sha256: str
    training_commit: str
    training_runtime_versions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeStagePlan:
    collection: str
    training: str
    adapted_evaluation: str
    artifact_publication: str


def derive_training_steps(dataset_frames: int, *, epochs: int = REQUESTED_EPOCHS, batch_size: int = GLOBAL_BATCH_SIZE) -> int:
    """Reproduce the launcher step contract from integer inputs only."""
    if isinstance(dataset_frames, bool) or not isinstance(dataset_frames, int) or dataset_frames <= 0:
        raise ResumeError("dataset_frames must be a positive integer")
    if epochs != REQUESTED_EPOCHS or batch_size != GLOBAL_BATCH_SIZE:
        raise ResumeError("resume is sealed to five epochs and global batch size 8")
    return math.ceil(epochs * dataset_frames / batch_size)


def derive_epoch_equivalent(checkpoint_step: int, dataset_frames: int, *, batch_size: int = GLOBAL_BATCH_SIZE) -> float:
    """Return the exact binary float computed from integer counts.

    In particular, do not read the rounded ``epoch_equivalent`` field from the
    failed source plan (the old run serialized ``5.02325581``).
    """
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step <= 0:
        raise ResumeError("checkpoint_step must be a positive integer")
    if isinstance(dataset_frames, bool) or not isinstance(dataset_frames, int) or dataset_frames <= 0:
        raise ResumeError("dataset_frames must be a positive integer")
    if batch_size != GLOBAL_BATCH_SIZE:
        raise ResumeError("resume is sealed to global batch size 8")
    return checkpoint_step * batch_size / dataset_frames


def stage_plan() -> ResumeStagePlan:
    """Describe exactly what a valid resume may do."""
    return ResumeStagePlan(
        collection="VALIDATED_REUSED",
        training="VALIDATED_REUSED",
        adapted_evaluation="VALIDATED_REUSED",
        artifact_publication="PENDING_ATOMIC_PUBLICATION",
    )


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ResumeError(f"{label} must be a JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ResumeError(f"{label} is missing or empty: {path}")
    return path


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ResumeError(f"archive status has malformed line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ResumeError(f"archive status has duplicate/empty key: {key!r}")
        values[key] = value
    return values


def _validate_archive_integrity(archive_root: Path) -> None:
    """Validate the archive's byte inventory before reusing any stage.

    The inventory is intentionally checked as a streaming text file and each
    listed payload is hashed with :func:`sha256_file`; no trajectory or model
    payload is accumulated in memory.
    """
    inventory = _require_file(archive_root / "inventory.sha256", "source archive inventory")
    tree_digest = _require_file(archive_root / "tree_sha256", "source archive tree digest")
    try:
        tree_text = tree_digest.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ResumeError("source archive tree digest is unreadable") from exc
    expected_tree = sha256_file(inventory)
    if tree_text != expected_tree + "\n":
        raise ResumeError("source archive tree digest does not match inventory.sha256")
    reserved_names = {"archive_status.env", "inventory.sha256", "tree_sha256"}
    listed: set[Path] = set()
    digest_re = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")
    try:
        lines = inventory.open("r", encoding="ascii", newline="")
    except (OSError, UnicodeError) as exc:
        raise ResumeError("source archive inventory is unreadable") from exc
    with lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.endswith("\n") or line.endswith("\r\n"):
                raise ResumeError(f"source archive inventory line {line_number} is malformed")
            match = digest_re.fullmatch(line[:-1])
            if match is None:
                raise ResumeError(f"source archive inventory line {line_number} is malformed")
            digest, raw_path = match.groups()
            path = Path(raw_path)
            if not path.is_absolute() or path != path.resolve() or path.name in reserved_names:
                raise ResumeError(f"source archive inventory path is unsafe: {raw_path}")
            try:
                path.relative_to(archive_root.resolve())
            except ValueError as exc:
                raise ResumeError(f"source archive inventory path escapes archive: {raw_path}") from exc
            if path in listed or path.is_symlink() or not path.is_file():
                raise ResumeError(f"source archive inventory entry is not a unique regular file: {raw_path}")
            listed.add(path)
            if sha256_file(path) != digest.lower():
                raise ResumeError(f"source archive inventory digest mismatch: {raw_path}")
    actual: set[Path] = set()
    for candidate in archive_root.rglob("*"):
        if candidate.is_symlink():
            raise ResumeError(f"source archive contains a symlink: {candidate}")
        if candidate.is_file() and candidate.name not in reserved_names:
            actual.add(candidate.resolve())
    if listed != actual:
        missing = sorted(str(path) for path in actual - listed)
        extra = sorted(str(path) for path in listed - actual)
        raise ResumeError(f"source archive inventory set mismatch: missing={missing!r}, extra={extra!r}")


def _safe_source_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise ResumeError("PEFT_RESUME_SOURCE_RUN_ROOT must be an existing absolute non-symlink directory")
    path = path.resolve()
    if not path.is_dir() or path.name != "run" or path.parent.name != "task_0":
        raise ResumeError("resume source must be an archive task_0/run directory")
    return path


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two normalized roots are equal or nested."""
    return first == second or first in second.parents or second in first.parents


def _validate_eval(source_root: Path, task_id: int) -> tuple[Path, Path, Path]:
    eval_root = source_root / "eval_adapted"
    eval_files = sorted(eval_root.rglob("eval_info.json")) if eval_root.is_dir() else []
    if len(eval_files) != 1:
        raise ResumeError(f"expected exactly one preserved adapted eval_info.json, found {len(eval_files)}")
    eval_path = eval_files[0]
    audit_path = eval_path.parent / "randomization_audit.jsonl"
    _require_file(audit_path, "adapted evaluation randomization audit")
    try:
        from vla_benchmarking.libero.evaluation.randomization_contract import randomization_config_payload
        from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import (
            validate_eval_info,
            validate_randomization_audit,
        )
        from vla_benchmarking.libero.shared.config import task_randomization_dimensions

        manifest = {
            "tasks": [task_id],
            "episodes": 10,
            "randomization_dimensions": {str(task_id): task_randomization_dimensions(task_id)},
            "randomization_config": randomization_config_payload(),
        }
        validate_eval_info(eval_path, manifest)
        validate_randomization_audit(eval_path.parent, manifest)
    except Exception as exc:
        raise ResumeError("preserved adapted evaluation failed its sealed output contract") from exc
    reset_contract = source_root / "eval_reset_contract.json"
    _require_file(reset_contract, "sealed evaluation reset contract")
    try:
        load_eval_reset_contract(reset_contract, task_id=task_id)
        validate_adapted_randomization_audit(reset_contract, audit_path)
    except Exception as exc:
        raise ResumeError("preserved adapted evaluation failed its reset-identity contract") from exc
    return eval_path, audit_path, reset_contract


def validate_resume_source(
    source_run_root: str | Path,
    *,
    expected_controller_config_hash: str,
    expected_base_policy_revision: str,
    expected_training_commit: str = TRAINING_COMMIT,
    expected_job_id: int = SOURCE_JOB_ID,
) -> ResumeSource:
    """Validate every source stage before allowing publication."""
    source = _safe_source_root(source_run_root)
    # The archive layout is ``<archive>/task_0/run`` plus the status file at
    # ``<archive>/task_0/archive_status.env``.  Do not walk above task_0: that
    # would inspect a different job's status in an all-task archive.
    archive_status = _require_file(source.parent / "archive_status.env", "source archive_status.env")
    status = _parse_env(archive_status)
    if status.get("status") != "PRESERVED_FAILURE":
        raise ResumeError("resume source must be the preserved failed job, not a verified or live run")
    if status.get("job_id") != str(expected_job_id):
        raise ResumeError(f"source archive job_id must be {expected_job_id}")
    if status.get("task_id") != str(TASK_ID) or status.get("task_scope") != "task_0":
        raise ResumeError("resume source archive identity is not task 0")
    if status.get("vla") != "smolvla" or status.get("batch_size") != str(GLOBAL_BATCH_SIZE):
        raise ResumeError("resume source archive identity is not the sealed SmolVLA pilot")
    if status.get("arrow_demos") != "1":
        raise ResumeError("resume source must contain exactly one accepted Arrow demonstration")
    if not expected_controller_config_hash or status.get("workload_exit_code") in (None, "0"):
        raise ResumeError("source archive does not identify a failed publication-only run")
    _validate_archive_integrity(source.parent)

    collection_manifest = _require_file(source / "collection" / "collection_manifest.json", "collection manifest")
    try:
        collection = load_arrow_collection_manifest(collection_manifest, task_id=TASK_ID, expected_successes=1)
    except Exception as exc:
        raise ResumeError("preserved Arrow collection/dataset failed validation") from exc
    if collection.get("collection_mode") != FRESH_COLLECTION_MODE or collection.get("source_kind") != FRESH_COLLECTION_SOURCE_KIND:
        raise ResumeError("resume requires a fresh-reset Arrow collection")
    if collection.get("controller_config_hash") != expected_controller_config_hash.lower():
        raise ResumeError("source collection controller hash differs from active contract")
    if collection.get("evaluator_confirmed_successes") != 1:
        raise ResumeError("source collection does not contain exactly one evaluator-confirmed success")

    dataset_root = Path(str(collection["dataset_root"])).expanduser().resolve()
    dataset_manifest = Path(str(collection["dataset_manifest_path"])).expanduser().resolve()
    _require_file(dataset_manifest, "dataset manifest")
    info_path = _require_file(dataset_root / "meta" / "info.json", "native dataset info")
    info = _read_json(info_path, "native dataset info")
    frames = info.get("total_frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ResumeError("native dataset total_frames must be a positive integer")
    steps = derive_training_steps(frames)

    training_plan = _require_file(source / "training_metadata" / "training_plan.json", "training plan")
    plan = _read_json(training_plan, "training plan")
    run_context = _read_json(_require_file(source / "run_context.json", "run context"), "run context")
    context_identity = {
        "vla": "smolvla", "task_id": 0, "task_ids": [0],
        "training_scope": "task_specific", "arrow_demos": 1,
        "collection_mode": "fresh_arrow", "evaluation_mode": "adapted_only",
        "collection_success_count": 1, "source_kind": FRESH_COLLECTION_SOURCE_KIND,
        "controller_config_hash": expected_controller_config_hash.lower(),
        "base_policy_revision": expected_base_policy_revision,
        "repo_commit": expected_training_commit,
    }
    for key, expected in context_identity.items():
        if run_context.get(key) != expected:
            raise ResumeError(f"run context {key} does not match the sealed training identity")
    raw_runtime_versions = run_context.get("runtime_versions")
    python_version = run_context.get("python")
    if not isinstance(raw_runtime_versions, Mapping) or not raw_runtime_versions or not isinstance(python_version, str) or not python_version.strip():
        raise ResumeError("run context does not contain training-time runtime versions")
    training_runtime_versions = {"python": python_version, **dict(raw_runtime_versions)}
    if any(not isinstance(key, str) or not key or (value is not None and not isinstance(value, str)) for key, value in training_runtime_versions.items()):
        raise ResumeError("run context runtime_versions are malformed")
    for key, expected in (("task_id", 0), ("steps", steps), ("requested_epochs", 5), ("batch_size", 8), ("global_batch_size", 8), ("dataset_frames", frames), ("seed", 1000)):
        if plan.get(key) != expected:
            raise ResumeError(f"training plan {key} does not match integer-derived source contract")
    if plan.get("base_policy_revision") != expected_base_policy_revision:
        raise ResumeError("training plan base policy revision differs from active base")
    if plan.get("repo_commit") != expected_training_commit:
        raise ResumeError("training plan was not produced by the sealed training commit")
    expected_inventory = _require_file(source / "training" / "expected_adapter_inventory.json", "expected native adapter inventory")
    checkpoints = []
    checkpoint_root = source / "training" / "checkpoints"
    for candidate in checkpoint_root.glob("*/pretrained_model") if checkpoint_root.is_dir() else ():
        try:
            step = int(candidate.parent.name)
        except ValueError:
            continue
        if candidate.is_dir():
            checkpoints.append((step, candidate))
    if len([item for item in checkpoints if item[0] == steps]) != 1:
        raise ResumeError(f"expected exactly one native checkpoint at derived step {steps}")
    checkpoint = checkpoints[0][1]
    if checkpoint.parent.name != str(steps):
        checkpoint = next(path for step, path in checkpoints if step == steps)
    try:
        from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.adapter_audit import audit_adapter_checkpoint

        audit_adapter_checkpoint(checkpoint, expected_inventory=expected_inventory, require_expected_inventory=True)
    except Exception as exc:
        raise ResumeError("native PEFT checkpoint failed its sealed inventory audit") from exc

    runtime_evidence = _require_file(source / "training_metadata" / "runtime_evidence.json", "runtime evidence")
    runtime = _read_json(runtime_evidence, "runtime evidence")
    if runtime.get("attestation_status") != "VERIFIED" or runtime.get("updates_observed") != steps:
        raise ResumeError("runtime evidence does not attest the complete derived training step count")
    if not runtime.get("all_losses_finite") or not runtime.get("all_grad_norms_finite"):
        raise ResumeError("runtime evidence contains non-finite training values")
    if runtime.get("expected_contract", {}).get("updates") != steps:
        raise ResumeError("runtime evidence expected update contract is inconsistent")

    adapted_eval, adapted_audit, reset_contract = _validate_eval(source, TASK_ID)
    baseline_stage = _read_json(_require_file(source / "baseline_stage.json", "baseline stage receipt"), "baseline stage receipt")
    if baseline_stage.get("status") != "SKIPPED" or baseline_stage.get("evaluation_mode") != "adapted_only":
        raise ResumeError("resume source is not the adapted-only pilot")
    if (source / "COMPLETED").exists():
        raise ResumeError("source unexpectedly contains COMPLETED after publication failure")
    return ResumeSource(
        source_run_root=source,
        archive_status=archive_status,
        collection_manifest=collection_manifest,
        dataset_root=dataset_root,
        dataset_manifest=dataset_manifest,
        checkpoint=checkpoint,
        checkpoint_step=steps,
        runtime_evidence=runtime_evidence,
        training_plan=training_plan,
        adapted_eval=adapted_eval,
        adapted_audit=adapted_audit,
        eval_reset_contract=reset_contract,
        expected_inventory=expected_inventory,
        collection_success_count=1,
        dataset_frames=frames,
        controller_config_hash=expected_controller_config_hash.lower(),
        source_collection_manifest_sha256=sha256_file(collection_manifest),
        training_commit=expected_training_commit,
        training_runtime_versions=training_runtime_versions,
    )


def _stats(path: Path) -> dict[str, Any]:
    value = _read_json(path, "adapted eval")
    overall = value.get("overall", value)
    successes = overall.get("n_successes", overall.get("successes"))
    episodes = overall.get("n_episodes", overall.get("episodes"))
    if successes is None:
        successes = sum(bool(item) for task in value.get("per_task", []) for item in task.get("metrics", {}).get("successes", []))
    if episodes is None:
        episodes = sum(len(task.get("metrics", {}).get("successes", [])) for task in value.get("per_task", []))
    if int(episodes) != len(SEALED_SEEDS):
        raise ResumeError("adapted eval must contain exactly ten episodes")
    return {"successes": int(successes), "episodes": int(episodes), "success_rate": float(successes) / float(episodes), "seeds": SEALED_SEEDS, "path": str(path.resolve())}


def _versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("python", "torch", "lerobot", "peft", "accelerate"):
        try:
            result[name] = version(name) if name != "python" else os.sys.version.split()[0]
        except PackageNotFoundError:
            result[name] = None
    return result


def _write_once(path: Path, payload: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ResumeError(f"refusing to overwrite resume output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise ResumeError(f"stale partial output exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def publish_resume(
    source: ResumeSource,
    *,
    output_run_root: str | Path,
    output_archive_root: str | Path,
    current_commit: str,
    base_policy: str | Path,
    base_policy_revision: str,
) -> Path:
    """Publish one immutable adapter and its normal run receipts."""
    run_root = Path(output_run_root).expanduser().resolve()
    archive_root = Path(output_archive_root).expanduser().resolve()
    source_archive_root = source.source_run_root.parent
    if _paths_overlap(run_root, source_archive_root) or _paths_overlap(archive_root, source_archive_root):
        raise ResumeError("resume output roots must be disjoint from the source task archive")
    if _paths_overlap(run_root, archive_root):
        raise ResumeError("resume run and archive roots must be disjoint")
    if not current_commit or not base_policy_revision:
        raise ResumeError("publication commit and base revision are required")
    run_root.mkdir(parents=True, exist_ok=True)
    if any(run_root.iterdir()):
        raise ResumeError("resume output run root must be empty before publication")
    adapted = _stats(source.adapted_eval)
    train_counts = {
        "successful_trajectories": 1,
        "dataset_frames": source.dataset_frames,
        "steps": source.checkpoint_step,
        "requested_epochs": REQUESTED_EPOCHS,
        "batch_size": GLOBAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "epoch_equivalent": derive_epoch_equivalent(source.checkpoint_step, source.dataset_frames),
        "save_freq": min(2000, source.checkpoint_step),
        "seed": 1000,
        "training_scope": "task_specific",
        "runtime_evidence_source_path": str(source.runtime_evidence),
        "runtime_evidence_source_sha256": sha256_file(source.runtime_evidence),
    }
    optimizer = {
        **PINNED_OPTIMIZER,
        "warmup_steps": source.checkpoint_step // 30,
        "decay_steps": source.checkpoint_step,
    }
    eval_counts = {
        "mode": "adapted_only",
        "seeds": SEALED_SEEDS,
        "baseline": {"status": "SKIPPED", "reason": "PEFT_SKIP_BASELINE=1"},
        "adapted": {**adapted, "status": "COMPLETED"},
    }
    run_id = f"smolvla_task0_arrow1_steps{source.checkpoint_step}_resume_job{os.environ.get('SLURM_JOB_ID', 'unknown')}"
    manifest = save_peft_adapter(
        source.checkpoint,
        archive_root,
        vla="smolvla",
        task_id=TASK_ID,
        run_id=run_id,
        base_checkpoint=base_policy,
        base_checkpoint_revision=base_policy_revision,
        collection_manifest=source.collection_manifest,
        collection_success_count=1,
        controller_config_hash=source.controller_config_hash,
        checkpoint_step=source.checkpoint_step,
        seed=1000,
        train_counts=train_counts,
        eval_counts=eval_counts,
        optimizer=optimizer,
        runtime_evidence=source.runtime_evidence,
        # The adapter was trained by the source job.  The recovery checkout is
        # publication code only and must never be recorded as the trainer.
        git_commit=source.training_commit,
        runtime_versions=source.training_runtime_versions,
    )
    # Re-open the detached artifact from disk.  The in-memory manifest
    # validation performed by save_peft_adapter is not sufficient: this
    # verifies the post-rename manifest, file inventory, bundled lineage, and
    # tree digest before any success receipt can authorize task 1.
    # save_peft_adapter returns the concrete path, while load_peft_manifest
    # intentionally returns a relocatable manifest whose artifact_path is
    # ``.``.  Keep the concrete path separately for all receipts/pointers.
    published_artifact_path = Path(manifest.artifact_path).expanduser().resolve()
    published = load_peft_manifest(published_artifact_path)
    if (
        published.task_id != TASK_ID
        or published.checkpoint_step != source.checkpoint_step
        or published.collection_success_count != source.collection_success_count
        or published.train_counts.get("dataset_frames") != source.dataset_frames
        or published.train_counts.get("epoch_equivalent") != train_counts["epoch_equivalent"]
    ):
        raise ResumeError("reloaded artifact identity or integer-derived training contract differs from source")
    reused_stage_hashes = {
        "collection_manifest_sha256": sha256_file(source.collection_manifest),
        "checkpoint_adapter_model_sha256": sha256_file(source.checkpoint / "adapter_model.safetensors"),
        "checkpoint_tree_sha256": tree_sha256(source.checkpoint),
        "runtime_evidence_sha256": sha256_file(source.runtime_evidence),
        "adapted_eval_info_sha256": sha256_file(source.adapted_eval),
        "adapted_randomization_audit_sha256": sha256_file(source.adapted_audit),
        "eval_reset_contract_sha256": sha256_file(source.eval_reset_contract),
    }
    artifact_root = published_artifact_path
    final_artifact = {
        "path": str(artifact_root),
        "manifest_sha256": sha256_file(artifact_root / "artifact_manifest.json"),
        "tree_sha256": published.artifact_tree_sha256,
    }
    published_summary = asdict(published)
    published_summary["artifact_path"] = str(published_artifact_path)
    receipt = {
        "schema": "smolvla_peft_resume_receipt.v1",
        "status": "VERIFIED",
        "source_run_root": str(source.source_run_root),
        "source_archive_status": str(source.archive_status),
        "source_job_id": SOURCE_JOB_ID,
        "training_commit": source.training_commit,
        "publication_commit": current_commit,
        "publication_runtime_versions": _versions(),
        "stages": asdict(stage_plan()) | {"artifact_publication": "VERIFIED"},
        "forbidden_actions": {"collector": "NOT_CALLED", "trainer": "NOT_CALLED", "evaluator": "NOT_CALLED"},
        "checkpoint_step": source.checkpoint_step,
        "dataset_frames": source.dataset_frames,
        "epoch_equivalent": train_counts["epoch_equivalent"],
        "source_collection_manifest_sha256": source.source_collection_manifest_sha256,
        "reused_stage_hashes": reused_stage_hashes,
        "final_artifact": final_artifact,
        "artifact_path": str(published_artifact_path),
    }
    # Keep the normal runner's machine-readable artifact pointer as an
    # atomic file.  The all-task wrapper and downstream evaluators consume
    # this exact receipt before permitting task 1 to start.
    _write_once(run_root / "ADAPTER_ARTIFACT_PATH.json", str(published_artifact_path))
    _write_once(run_root / "publication_recovery_receipt.json", receipt)
    summary = {
        "schema": "smolvla_peft_arrow_task_summary.v5",
        "training_scope": "task_specific", "vla": "smolvla", "task_id": 0, "trained_task_ids": [0],
        "source_kind": FRESH_COLLECTION_SOURCE_KIND, "collection_mode": "fresh_arrow", "evaluation_mode": "adapted_only",
        "collection_success_count": 1, "collection_manifest_sha256": source.source_collection_manifest_sha256,
        "controller_config_hash": source.controller_config_hash, "dataset_frames": source.dataset_frames,
        "optimizer_steps": source.checkpoint_step, "requested_epochs": REQUESTED_EPOCHS,
        "global_batch_size": GLOBAL_BATCH_SIZE, "epoch_equivalent": train_counts["epoch_equivalent"],
        "baseline": {"status": "SKIPPED", "reason": "PEFT_SKIP_BASELINE=1"}, "adapted": adapted,
        "comparison_available": False, "improvement_claim_supported": False, "success_delta": None,
        "training_commit": source.training_commit, "publication_commit": current_commit,
        "resumed_from": str(source.source_run_root), "adapter_artifact": published_summary,
    }
    _write_once(run_root / "experiment_summary.json", summary)
    completed = run_root / "COMPLETED"
    if completed.exists() or completed.is_symlink():
        raise ResumeError("resume output already has COMPLETED")
    completed.touch(exist_ok=False)
    return Path(manifest.artifact_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--output-archive-root", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--base-policy-revision", required=True)
    parser.add_argument("--controller-config-hash", required=True)
    args = parser.parse_args(argv)
    source = validate_resume_source(args.source_run_root, expected_controller_config_hash=args.controller_config_hash, expected_base_policy_revision=args.base_policy_revision)
    artifact = publish_resume(source, output_run_root=args.output_run_root, output_archive_root=args.output_archive_root, current_commit=args.current_commit, base_policy=args.base_policy, base_policy_revision=args.base_policy_revision)
    print(json.dumps({"status": "RESUME_PUBLISHED", "artifact_path": str(artifact), "source_run_root": str(source.source_run_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ResumeError", "ResumeSource", "ResumeStagePlan", "derive_training_steps",
    "derive_epoch_equivalent", "stage_plan", "validate_resume_source", "publish_resume", "main",
]
