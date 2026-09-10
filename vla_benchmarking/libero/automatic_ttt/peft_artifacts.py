"""Immutable, task-specific PEFT artifacts for automatic-TTT experiments.

The PEFT control is separate from the RoboTTT TTT runtime.  Its input is a
manifest emitted by the ArrowGraspController collector.  This module refuses
to infer lineage from an HDF5 directory or from a different task.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

SCHEMA_VERSION = 2
COLLECTION_SCHEMA_VERSION = 1
COLLECTION_SOURCE_KIND = "arrow_grasp_controller_trajectory"
FRESH_COLLECTION_SOURCE_KIND = "arrow_grasp_controller_fresh_demonstration"
FRESH_COLLECTION_MODE = "fresh_arrow"
FRESH_METHOD_LABEL = "fresh_arrow_behavior_cloning_peft"
SEALED_EVAL_SEEDS = tuple(range(1000, 1010))
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")

# RoboTTT Appendix A.2 post-training values.  These are intentionally kept in
# the artifact boundary instead of being trainer defaults: a published
# adapter is invalid unless its run records every value that affects the
# optimizer/schedule contract.
PINNED_OPTIMIZER = {
    "optimizer": "AdamW",
    "weight_decay": 1e-5,
    "peak_learning_rate": 5e-5,
    "betas": [0.9, 0.95],
    "epsilon": 1e-8,
    "gradient_clip_norm": 10.0,
    "scheduler": "cosine_decay_with_warmup",
    "decay_lr": 2.5e-6,
}
PAPER_TRAIN_STEPS = 20000
PAPER_GLOBAL_BATCH_SIZE = 8
PAPER_SAVE_FREQ = 2000
PAPER_TRAIN_SEED = 1000
PAPER_EVAL_SEEDS = tuple(range(1000, 1010))


class PEFTArtifactError(ValueError):
    """Raised when an artifact or its training lineage is invalid."""


def _require_positive_count(value: Any, label: str) -> int:
    """Return a strict positive integer count used by collection contracts."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PEFTArtifactError(f"{label} must be a positive integer")
    count = value
    if count <= 0:
        raise PEFTArtifactError(f"{label} must be a positive integer")
    return count


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if target.is_dir():
        return tree_sha256(target)
    raise PEFTArtifactError(f"cannot hash missing/non-file checkpoint: {target}")


def tree_sha256(root: str | Path, *, exclude: set[str] | None = None) -> str:
    """Hash relative paths and bytes deterministically."""
    base = Path(root)
    if not base.is_dir():
        raise PEFTArtifactError(f"artifact tree is not a directory: {base}")
    excluded = exclude or set()
    digest = hashlib.sha256()
    files = sorted(
        path for path in base.rglob("*")
        if path.is_file() and path.relative_to(base).as_posix() not in excluded
    )
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PEFTArtifactError(f"{label} must be a 64-character SHA-256 digest")
    return value.lower()


def _require_float(value: Any, expected: float, label: str, *, tolerance: float = 1e-12) -> None:
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise PEFTArtifactError(f"{label} must be {expected!r}") from exc
    if abs(actual - expected) > tolerance:
        raise PEFTArtifactError(f"{label} must be {expected!r}, got {actual!r}")


def _resolve_existing_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PEFTArtifactError(f"{label} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise PEFTArtifactError(f"{label} does not exist: {path}")
    return path


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PEFTArtifactError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.as_posix() == value.replace("\\", "/"):
        raise PEFTArtifactError(f"{label} must be an artifact-relative path")
    return path


def _validate_runtime_evidence(
    evidence: Mapping[str, Any], *, expected_updates: int | None = None,
    expected_warmup: int | None = None, expected_decay: int | None = None,
) -> None:
    """Validate process-local training evidence, not just declared settings."""
    try:
        updates = int(evidence["updates_observed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PEFTArtifactError("runtime evidence must record updates_observed") from exc
    declared_updates = evidence.get("expected_updates", expected_updates)
    if declared_updates is None:
        declared_updates = updates
    if updates != int(declared_updates):
        raise PEFTArtifactError(f"runtime evidence must contain exactly {int(declared_updates)} updates")
    for field in ("all_losses_finite", "all_grad_norms_finite"):
        if evidence.get(field) is not True:
            raise PEFTArtifactError(f"runtime evidence {field} must be true")
    for field in ("min_loss", "max_loss", "last_loss", "last_grad_norm", "min_learning_rate", "max_learning_rate", "last_learning_rate"):
        try:
            value = float(evidence[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise PEFTArtifactError(f"runtime evidence {field} must be finite") from exc
        if not math.isfinite(value):
            raise PEFTArtifactError(f"runtime evidence {field} must be finite")
    optimizer = evidence.get("optimizer", evidence.get("resolved_optimizer"))
    scheduler = evidence.get("scheduler", evidence.get("resolved_scheduler"))
    if not isinstance(optimizer, Mapping) or not isinstance(scheduler, Mapping):
        raise PEFTArtifactError("runtime evidence must include resolved optimizer and scheduler mappings")
    if optimizer.get("optimizer", optimizer.get("name")) != "AdamW":
        raise PEFTArtifactError("runtime evidence optimizer must resolve to AdamW")
    if scheduler.get("scheduler", scheduler.get("name")) != "cosine_decay_with_warmup":
        raise PEFTArtifactError("runtime evidence scheduler must resolve to cosine_decay_with_warmup")
    optimizer_aliases = {
        "weight_decay": "weight_decay", "peak_learning_rate": "peak_learning_rate",
        "betas": "betas", "epsilon": "epsilon", "gradient_clip_norm": "gradient_clip_norm",
    }
    for key, alias in optimizer_aliases.items():
        if key not in optimizer and alias not in optimizer:
            raise PEFTArtifactError(f"runtime evidence optimizer is missing {key}")
    _require_float(optimizer.get("weight_decay"), PINNED_OPTIMIZER["weight_decay"], "runtime optimizer.weight_decay")
    _require_float(optimizer.get("peak_learning_rate"), PINNED_OPTIMIZER["peak_learning_rate"], "runtime optimizer.peak_learning_rate")
    betas = optimizer.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise PEFTArtifactError("runtime optimizer.betas must be [0.9, 0.95]")
    _require_float(betas[0], 0.9, "runtime optimizer.betas[0]")
    _require_float(betas[1], 0.95, "runtime optimizer.betas[1]")
    _require_float(optimizer.get("epsilon"), 1e-8, "runtime optimizer.epsilon")
    _require_float(optimizer.get("gradient_clip_norm"), 10.0, "runtime optimizer.gradient_clip_norm")
    for key, expected in (("warmup_steps", expected_warmup), ("decay_steps", expected_decay)):
        if expected is None:
            expected = scheduler.get(key)
        if int(scheduler.get(key, -1)) != expected:
            raise PEFTArtifactError(f"runtime scheduler.{key} must be {expected}")
    _require_float(scheduler.get("decay_lr"), 2.5e-6, "runtime scheduler.decay_lr")


def _validate_accepted_trace(
    trace_path: Path,
    *,
    task_id: int,
    successful_trajectories: list[Mapping[str, Any]],
    expected_successes: int,
    collection_mode: str = "same_episode_takeover",
) -> None:
    """Validate the raw, immutable trace behind the native dataset.

    This is deliberately independent of the dataset's frame count.  The
    dataset contains only Arrow suffix frames, while this trace is the proof
    that each accepted example was a real VLA-prefix/Arrow-suffix takeover.
    """
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    expected_pairs = []
    for item in successful_trajectories:
        try:
            expected_pairs.append((item.get("trajectory_id"), int(item.get("seed"))))
        except (TypeError, ValueError) as exc:
            raise PEFTArtifactError("successful trajectory seed must be an integer") from exc
    expected_by_id = {str(item.get("trajectory_id")): item for item in successful_trajectories}
    actual_pairs = []
    record_count = 0
    try:
        trace_handle = trace_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise PEFTArtifactError(f"cannot read accepted episode trace: {trace_path}") from exc
    with trace_handle:
      for line_number, line in enumerate(trace_handle, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PEFTArtifactError(f"accepted episode trace line {line_number} is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise PEFTArtifactError("accepted episode trace records must be JSON objects")
        record_count += 1
        if record_count > expected_successes:
            raise PEFTArtifactError(
                f"accepted episode trace must contain exactly {expected_successes} records"
            )
        trajectory_id = record.get("episode_id", record.get("trajectory_id"))
        if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in seen_ids:
            raise PEFTArtifactError("accepted trace trajectory ids must be unique and non-empty")
        try:
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PEFTArtifactError("every accepted trace must record its adaptation seed") from exc
        if seed in seen_seeds:
            raise PEFTArtifactError("accepted trace adaptation seeds must be unique")
        try:
            record_task = int(record.get("task_id", -1))
        except (TypeError, ValueError) as exc:
            raise PEFTArtifactError("accepted trace task_id must be an integer") from exc
        if record_task != task_id:
            raise PEFTArtifactError("accepted trace task_id differs from collection task")
        expected_source = FRESH_COLLECTION_SOURCE_KIND if collection_mode == FRESH_COLLECTION_MODE else COLLECTION_SOURCE_KIND
        if record.get("source_kind") != expected_source:
            raise PEFTArtifactError("accepted trace source_kind is not ArrowGraspController data")
        transitions = record.get("transitions")
        if not isinstance(transitions, list) or not transitions:
            raise PEFTArtifactError("every accepted trace must contain transitions")
        actors = [transition.get("actor") for transition in transitions if isinstance(transition, Mapping)]
        if len(actors) != len(transitions):
            raise PEFTArtifactError("accepted trace transitions must be objects")
        first_arrow = next((index for index, actor in enumerate(actors) if actor == "arrow_grasp_controller"), None)
        if collection_mode == FRESH_COLLECTION_MODE:
            if first_arrow != 0 or any(actor != "arrow_grasp_controller" for actor in actors):
                raise PEFTArtifactError("fresh Arrow accepted trace must contain only Arrow transitions")
            reset_identity = record.get("reset_identity")
            if not isinstance(reset_identity, Mapping):
                raise PEFTArtifactError("fresh Arrow accepted trace must include reset_identity")
            if reset_identity.get("task_id") != task_id:
                raise PEFTArtifactError("fresh Arrow reset_identity task_id differs from trace")
            selected_index = reset_identity.get("selected_init_state_index")
            digest = reset_identity.get("init_state_sha256")
            if isinstance(selected_index, bool) or not isinstance(selected_index, int) or selected_index < 10:
                raise PEFTArtifactError("fresh Arrow reset_identity selected index is invalid")
            _require_sha256(digest, "fresh Arrow reset_identity init_state_sha256")
            expected_item = expected_by_id.get(trajectory_id)
            if not isinstance(expected_item, Mapping) or expected_item.get("reset_identity") != dict(reset_identity):
                raise PEFTArtifactError("accepted trace reset_identity does not match successful_trajectories")
        else:
            if first_arrow is None or first_arrow == 0:
                raise PEFTArtifactError("accepted trace must have a non-empty VLA prefix and Arrow suffix")
            if any(actor != "vla" for actor in actors[:first_arrow]) or any(
                actor != "arrow_grasp_controller" for actor in actors[first_arrow:]
            ):
                raise PEFTArtifactError("accepted trace actors must be a VLA prefix followed by an Arrow suffix")
        receipt = record.get("evaluator_receipt")
        if not isinstance(receipt, Mapping):
            raise PEFTArtifactError("accepted trace must include an evaluator receipt")
        if receipt.get("evaluator_success") is not True or receipt.get("teacher_success") is not True:
            raise PEFTArtifactError("accepted trace evaluator receipt is not success-confirmed")
        if receipt.get("source_controller") != "arrow_grasp_controller":
            raise PEFTArtifactError("accepted trace evaluator receipt has the wrong controller")
        if receipt.get("episode_id") not in {None, trajectory_id} or receipt.get("seed") not in {None, seed}:
            raise PEFTArtifactError("accepted trace evaluator receipt identity does not match trace")
        seen_ids.add(trajectory_id)
        seen_seeds.add(seed)
        actual_pairs.append((trajectory_id, seed))
    if record_count != expected_successes:
        raise PEFTArtifactError(
            f"accepted episode trace must contain exactly {expected_successes} records"
        )
    if actual_pairs != expected_pairs:
        raise PEFTArtifactError("accepted trace records do not match successful_trajectories")


def _validate_fresh_reset_contract(
    payload: Mapping[str, Any], successful: list[Mapping[str, Any]], expected_successes: int,
) -> tuple[list[int], list[str], list[Mapping[str, Any]]]:
    """Validate collection/evaluation init-state disjointness by identity."""
    indices = payload.get("reserved_eval_init_state_indices")
    hashes = payload.get("reserved_eval_init_state_hashes")
    accepted = payload.get("accepted_reset_identities")
    if indices != list(range(10)):
        raise PEFTArtifactError("fresh collection must reserve evaluation init-state indices 0..9")
    if not isinstance(hashes, list) or len(hashes) != 10:
        raise PEFTArtifactError("fresh collection must record ten reserved evaluation init-state hashes")
    normalized_hashes = [_require_sha256(value, "reserved evaluation init-state hash") for value in hashes]
    if len(set(normalized_hashes)) != 10:
        raise PEFTArtifactError("reserved evaluation init-state hashes must be unique")
    if not isinstance(accepted, list) or len(accepted) != expected_successes:
        raise PEFTArtifactError("fresh collection must record accepted reset identities in trajectory order")
    if len(successful) != expected_successes:
        raise PEFTArtifactError("fresh collection successful trajectory count is invalid")
    reserved_hash_set = set(normalized_hashes)
    normalized_accepted: list[Mapping[str, Any]] = []
    for position, identity in enumerate(accepted):
        if not isinstance(identity, Mapping):
            raise PEFTArtifactError("accepted reset identity must be an object")
        if identity.get("task_id") != payload.get("task_id"):
            raise PEFTArtifactError("accepted reset identity task_id differs from collection")
        selected_index = identity.get("selected_init_state_index")
        if isinstance(selected_index, bool) or not isinstance(selected_index, int) or selected_index < 10:
            raise PEFTArtifactError("accepted reset identity selected index is invalid")
        digest = _require_sha256(identity.get("init_state_sha256"), "accepted reset identity hash")
        # LIBERO exposes a finite init-state table (normally 50 rows).  After
        # reserving rows 0..9 for evaluation, a 50-success collection must
        # revisit some of the remaining 40 layouts.  Repeated layouts are
        # therefore valid; only overlap with the sealed evaluation identities
        # is forbidden. Seeds and trajectory IDs remain unique elsewhere.
        if selected_index in range(10):
            raise PEFTArtifactError("accepted reset identities contain a reserved evaluation index")
        if digest in reserved_hash_set:
            raise PEFTArtifactError("accepted reset identities contain a reserved evaluation hash")
        expected_identity = successful[position].get("reset_identity")
        if expected_identity != dict(identity):
            raise PEFTArtifactError("successful trajectory reset identity order does not match accepted identities")
        normalized_accepted.append({
            "task_id": int(identity["task_id"]),
            "selected_init_state_index": selected_index,
            "init_state_sha256": digest,
        })
    return list(range(10)), normalized_hashes, normalized_accepted


def load_arrow_collection_manifest(
    manifest_path: str | Path,
    *,
    task_id: int,
    expected_successes: int = 50,
) -> dict[str, Any]:
    """Validate the Arrow collector-to-trainer handoff.

    The collector must provide: ``source_kind``, one ``task_id`` and
    ``task_ids=[task_id]``, exactly ``expected_successes`` evaluator-confirmed
    successful trajectories, unique adaptation seeds disjoint from sealed evaluation
    seeds 1000..1009, a controller SHA-256 hash, and a native LeRobot dataset
    plus immutable dataset manifest.  If the producer records ``accepted_target``
    or ``accepted_count``, those fields must also equal ``expected_successes``.
    Those two producer counters are optional for compatibility with legacy
    manifests that predate them; ``evaluator_confirmed_successes`` is always
    required.  Extra fields are allowed.  No HDF5 overlay conversion is
    accepted at this boundary.
    """
    expected_successes = _require_positive_count(expected_successes, "expected_successes")
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise PEFTArtifactError(f"Arrow collection manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PEFTArtifactError(f"cannot read Arrow collection manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PEFTArtifactError("Arrow collection manifest must be a JSON object")
    if payload.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise PEFTArtifactError("unsupported Arrow collection manifest schema")
    collection_mode = str(payload.get("collection_mode", "same_episode_takeover"))
    if collection_mode not in {"same_episode_takeover", FRESH_COLLECTION_MODE}:
        raise PEFTArtifactError("collection_mode must be same_episode_takeover or fresh_arrow")
    expected_collection_source = FRESH_COLLECTION_SOURCE_KIND if collection_mode == FRESH_COLLECTION_MODE else COLLECTION_SOURCE_KIND
    if payload.get("source_kind") != expected_collection_source:
        raise PEFTArtifactError("collection source_kind does not match collection_mode")
    expected_method = FRESH_METHOD_LABEL if collection_mode == FRESH_COLLECTION_MODE else "peft_lora"
    if payload.get("method_label") != expected_method:
        raise PEFTArtifactError("collection method_label does not match its active source method")
    try:
        manifest_task = int(payload["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PEFTArtifactError("collection manifest task_id is required") from exc
    if manifest_task != int(task_id) or not 0 <= manifest_task <= 9:
        raise PEFTArtifactError("collection manifest task_id does not equal the sole launcher task")
    if payload.get("task_ids") != [manifest_task]:
        raise PEFTArtifactError("collection manifest must contain exactly one task_id")
    if "evaluator_confirmed_successes" not in payload:
        raise PEFTArtifactError("evaluator_confirmed_successes is required")
    successes = _require_positive_count(
        payload["evaluator_confirmed_successes"], "evaluator_confirmed_successes"
    )
    for field in ("accepted_target", "accepted_count"):
        if field in payload:
            producer_count = _require_positive_count(payload[field], field)
            if producer_count != expected_successes:
                raise PEFTArtifactError(
                    f"{field} must equal expected_successes ({expected_successes})"
                )
    successful = payload.get("successful_trajectories")
    if not isinstance(successful, list) or successes != expected_successes or len(successful) != expected_successes:
        raise PEFTArtifactError(f"collection must contain exactly {expected_successes} successful trajectories")
    trajectory_ids: set[str] = set()
    trajectory_seeds: list[int] = []
    for item in successful:
        if not isinstance(item, Mapping) or item.get("evaluator_success") is not True:
            raise PEFTArtifactError("every successful trajectory must be evaluator-confirmed")
        trajectory_id = item.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id or trajectory_id in trajectory_ids:
            raise PEFTArtifactError("successful trajectories must have unique non-empty trajectory_id values")
        trajectory_ids.add(trajectory_id)
        try:
            trajectory_seeds.append(int(item["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PEFTArtifactError("every successful trajectory must record its adaptation seed") from exc
    adaptation_seeds = payload.get("adaptation_seeds")
    if not isinstance(adaptation_seeds, list) or len(adaptation_seeds) != expected_successes:
        raise PEFTArtifactError(f"adaptation_seeds must contain exactly {expected_successes} seeds")
    try:
        adaptation_seeds = [int(seed) for seed in adaptation_seeds]
    except (TypeError, ValueError) as exc:
        raise PEFTArtifactError("adaptation_seeds must contain integers") from exc
    if len(set(adaptation_seeds)) != len(adaptation_seeds):
        raise PEFTArtifactError("adaptation_seeds must be unique")
    if set(adaptation_seeds).intersection(SEALED_EVAL_SEEDS):
        raise PEFTArtifactError("adaptation seeds overlap sealed evaluation seeds 1000..1009")
    if any(seed < 3000 for seed in adaptation_seeds):
        raise PEFTArtifactError("adaptation seeds must use the reserved namespace >=3000")
    if trajectory_seeds != adaptation_seeds:
        raise PEFTArtifactError("trajectory seeds must match adaptation_seeds in manifest order")
    reset_contract = None
    if collection_mode == FRESH_COLLECTION_MODE:
        reset_contract = _validate_fresh_reset_contract(payload, successful, expected_successes)
    controller_hash = _require_sha256(payload.get("controller_config_hash"), "controller_config_hash")
    accepted_trace = _resolve_existing_path(payload.get("accepted_episodes_jsonl"), "accepted_episodes_jsonl")
    accepted_trace_digest = _require_sha256(
        payload.get("accepted_episodes_sha256"), "accepted_episodes_sha256"
    )
    actual_trace_digest = sha256_file(accepted_trace)
    if accepted_trace_digest != actual_trace_digest:
        raise PEFTArtifactError("accepted_episodes_sha256 does not match accepted_episodes_jsonl")
    dataset_root_value = payload.get("dataset_root")
    dataset_manifest_value = payload.get("dataset_manifest_path")
    if not isinstance(dataset_root_value, str) or not dataset_root_value:
        raise PEFTArtifactError("dataset_root is required")
    dataset_root = Path(dataset_root_value).expanduser().resolve()
    if not dataset_root.is_dir() or not (dataset_root / "meta" / "info.json").is_file():
        raise PEFTArtifactError("dataset_root must be a native LeRobot dataset with meta/info.json")
    if not isinstance(dataset_manifest_value, str) or not dataset_manifest_value:
        raise PEFTArtifactError("dataset_manifest_path is required")
    dataset_manifest = Path(dataset_manifest_value).expanduser().resolve()
    if not dataset_manifest.is_file():
        raise PEFTArtifactError(f"dataset_manifest_path does not exist: {dataset_manifest}")
    expected_dataset_hash = _require_sha256(
        payload.get("dataset_manifest_sha256"), "dataset_manifest_sha256"
    )
    actual_dataset_hash = sha256_file(dataset_manifest)
    if expected_dataset_hash != actual_dataset_hash:
        raise PEFTArtifactError("dataset_manifest_sha256 does not match dataset_manifest_path")
    try:
        dataset_inventory = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PEFTArtifactError("dataset manifest is unreadable") from exc
    if not isinstance(dataset_inventory, Mapping):
        raise PEFTArtifactError("dataset manifest must be a JSON object")
    source_trace = _resolve_existing_path(
        dataset_inventory.get("source_accepted_episodes"), "dataset manifest source_accepted_episodes"
    )
    source_trace_digest = _require_sha256(
        dataset_inventory.get("source_accepted_episodes_sha256"),
        "dataset manifest source_accepted_episodes_sha256",
    )
    if source_trace != accepted_trace:
        raise PEFTArtifactError("dataset manifest source_accepted_episodes does not match accepted trace")
    if source_trace_digest != actual_trace_digest:
        raise PEFTArtifactError("dataset manifest source_accepted_episodes_sha256 does not match accepted trace")
    _validate_accepted_trace(
        accepted_trace,
        task_id=manifest_task,
        successful_trajectories=successful,
        expected_successes=expected_successes,
        collection_mode=collection_mode,
    )
    inventory_root = dataset_inventory.get("dataset_root")
    if not isinstance(inventory_root, str) or Path(inventory_root).expanduser().resolve() != dataset_root:
        raise PEFTArtifactError("dataset manifest root does not match collection dataset_root")
    files = dataset_inventory.get("files")
    if not isinstance(files, list) or not files:
        raise PEFTArtifactError("dataset manifest must inventory every dataset file")
    seen_files: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise PEFTArtifactError("dataset manifest file entry must be an object")
        relative_value = record.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            raise PEFTArtifactError("dataset manifest file path is required")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen_files:
            raise PEFTArtifactError("dataset manifest contains an unsafe or duplicate file path")
        seen_files.add(relative.as_posix())
        candidate = dataset_root / relative
        expected_file_hash = _require_sha256(record.get("sha256"), f"dataset file {relative_value}")
        if not candidate.is_file() or sha256_file(candidate) != expected_file_hash:
            raise PEFTArtifactError(f"dataset file digest mismatch: {candidate}")
    actual_files = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    if actual_files != seen_files:
        raise PEFTArtifactError("dataset manifest inventory does not match the dataset tree")
    return {
        "path": str(path), "sha256": sha256_file(path), "source_kind": expected_collection_source,
        "collection_mode": collection_mode,
        "task_id": manifest_task, "task_ids": [manifest_task],
        "evaluator_confirmed_successes": successes, "successful_trajectories": successful,
        "adaptation_seeds": adaptation_seeds, "controller_config_hash": controller_hash,
        **({
            "reserved_eval_init_state_indices": reset_contract[0],
            "reserved_eval_init_state_hashes": reset_contract[1],
            "accepted_reset_identities": reset_contract[2],
        } if reset_contract is not None else {}),
        "dataset_root": str(dataset_root), "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": actual_dataset_hash,
        "accepted_episodes_jsonl": str(accepted_trace),
        "accepted_episodes_sha256": actual_trace_digest,
    }


@dataclass(frozen=True)
class PEFTArtifactManifest:
    schema_version: int
    artifact_path: str
    vla: str
    task_id: int
    run_id: str
    method: str
    training_scope: str
    trained_task_ids: list[int]
    files: Mapping[str, Mapping[str, Any]]
    artifact_tree_sha256: str
    base_checkpoint_path: str
    base_checkpoint_sha256: str
    base_checkpoint_revision: str
    collection_manifest_path: str
    collection_manifest_sha256: str
    source_collection_manifest_sha256: str
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    source_dataset_manifest_sha256: str
    runtime_evidence_path: str
    runtime_evidence_sha256: str
    lineage_tree_sha256: str
    collection_success_count: int
    controller_config_hash: str
    source_kind: str
    checkpoint_step: int
    seed: int
    train_counts: Mapping[str, Any]
    eval_counts: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    git_commit: str
    runtime_versions: Mapping[str, Any]

    def validate(self) -> None:
        expected_method = FRESH_METHOD_LABEL if self.source_kind == FRESH_COLLECTION_SOURCE_KIND else "peft_lora"
        if self.schema_version != SCHEMA_VERSION or self.method != expected_method:
            raise PEFTArtifactError("unsupported PEFT artifact schema or method")
        if self.training_scope != "task_specific" or self.trained_task_ids != [self.task_id]:
            raise PEFTArtifactError("artifact must represent exactly one task-specific training scope")
        if not self.vla or not 0 <= self.task_id <= 9 or not self.run_id:
            raise PEFTArtifactError("invalid VLA/task/run identity")
        if any(name not in self.files for name in REQUIRED_FILES):
            raise PEFTArtifactError("manifest is missing required adapter files")
        for relative, record in self.files.items():
            candidate = Path(str(relative))
            if candidate.is_absolute() or ".." in candidate.parts or not str(relative):
                raise PEFTArtifactError(f"unsafe artifact file path in manifest: {relative!r}")
            if not isinstance(record, Mapping) or not record.get("sha256"):
                raise PEFTArtifactError(f"invalid digest record for artifact file: {relative!r}")
        if self.source_kind not in {COLLECTION_SOURCE_KIND, FRESH_COLLECTION_SOURCE_KIND}:
            raise PEFTArtifactError("artifact source_kind is not ArrowGraspController trajectory data")
        collection_success_count = _require_positive_count(
            self.collection_success_count, "collection_success_count"
        )
        for field in (
            "base_checkpoint_path", "collection_manifest_path", "dataset_manifest_path",
            "runtime_evidence_path",
        ):
            _safe_relative_path(getattr(self, field), field)
        for field in (
            "collection_manifest_sha256", "source_collection_manifest_sha256",
            "dataset_manifest_sha256", "source_dataset_manifest_sha256",
            "runtime_evidence_sha256", "lineage_tree_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        _require_sha256(self.controller_config_hash, "controller_config_hash")
        if self.checkpoint_step <= 0:
            raise PEFTArtifactError("artifact checkpoint_step must be positive")
        if not isinstance(self.optimizer, Mapping):
            raise PEFTArtifactError("artifact optimizer must be a mapping")
        for key, expected in PINNED_OPTIMIZER.items():
            if key not in self.optimizer:
                raise PEFTArtifactError(f"artifact optimizer is missing required field {key!r}")
            actual = self.optimizer[key]
            if isinstance(expected, list):
                if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                    raise PEFTArtifactError(f"artifact optimizer field {key!r} must equal {expected!r}")
                for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
                    _require_float(actual_value, expected_value, f"optimizer.{key}[{index}]")
            elif isinstance(expected, float):
                _require_float(actual, expected, f"optimizer.{key}")
            elif actual != expected:
                raise PEFTArtifactError(f"optimizer.{key} must equal {expected!r}, got {actual!r}")
        required_train = (
            "successful_trajectories", "steps", "batch_size", "global_batch_size",
            "dataset_frames", "epoch_equivalent", "save_freq", "seed",
        )
        if not isinstance(self.train_counts, Mapping) or any(key not in self.train_counts for key in required_train):
            raise PEFTArtifactError("train_counts is missing required paper/experiment fields")
        train_success_count = _require_positive_count(
            self.train_counts.get("successful_trajectories"),
            "train_counts.successful_trajectories",
        )
        if train_success_count != collection_success_count:
            raise PEFTArtifactError(
                "train_counts.successful_trajectories must match collection_success_count"
            )
        if self.train_counts.get("steps") != self.checkpoint_step:
            raise PEFTArtifactError("train_counts.steps must equal checkpoint_step")
        if self.train_counts.get("batch_size") != PAPER_GLOBAL_BATCH_SIZE or self.train_counts.get("global_batch_size") != PAPER_GLOBAL_BATCH_SIZE:
            raise PEFTArtifactError("train_counts batch_size and global_batch_size must both be 8")
        try:
            dataset_frames = int(self.train_counts["dataset_frames"])
        except (TypeError, ValueError) as exc:
            raise PEFTArtifactError("train_counts.dataset_frames must be a positive integer") from exc
        if dataset_frames <= 0:
            raise PEFTArtifactError("train_counts.dataset_frames must be positive")
        if self.source_kind == FRESH_COLLECTION_SOURCE_KIND:
            requested_epochs = _require_positive_count(
                self.train_counts.get("requested_epochs"),
                "fresh Arrow train_counts.requested_epochs",
            )
            expected_steps = math.ceil(requested_epochs * dataset_frames / PAPER_GLOBAL_BATCH_SIZE)
            expected_warmup = expected_steps // 30
            if self.checkpoint_step != expected_steps:
                raise PEFTArtifactError(
                    "fresh Arrow checkpoint does not implement "
                    "ceil(requested_epochs*dataset_frames/8)"
                )
            achieved = self.checkpoint_step * PAPER_GLOBAL_BATCH_SIZE / dataset_frames
            _require_float(self.train_counts["epoch_equivalent"], achieved, "train_counts.epoch_equivalent", tolerance=1e-10)
            if not (
                achieved >= requested_epochs
                and achieved < requested_epochs + PAPER_GLOBAL_BATCH_SIZE / dataset_frames
            ):
                raise PEFTArtifactError(
                    "fresh Arrow achieved epoch equivalent is outside the requested-epoch ceiling"
                )
            if self.train_counts.get("save_freq") != min(PAPER_SAVE_FREQ, self.checkpoint_step):
                raise PEFTArtifactError("fresh Arrow save_freq must be min(2000, steps)")
            if self.optimizer.get("warmup_steps") != expected_warmup:
                raise PEFTArtifactError("fresh Arrow optimizer.warmup_steps must be floor(steps/30)")
            if self.optimizer.get("decay_steps") != expected_steps:
                raise PEFTArtifactError("fresh Arrow optimizer.decay_steps must equal steps")
        else:
            if self.checkpoint_step != PAPER_TRAIN_STEPS:
                raise PEFTArtifactError("legacy takeover checkpoint_step must be 20000")
            if self.optimizer.get("warmup_steps") != 666:
                raise PEFTArtifactError("legacy takeover optimizer.warmup_steps must be 666")
            if self.optimizer.get("decay_steps") != PAPER_TRAIN_STEPS:
                raise PEFTArtifactError("legacy takeover optimizer.decay_steps must be 20000")
            _require_float(self.train_counts["epoch_equivalent"], PAPER_TRAIN_STEPS * PAPER_GLOBAL_BATCH_SIZE / dataset_frames, "train_counts.epoch_equivalent", tolerance=1e-10)
            if self.train_counts.get("save_freq") != PAPER_SAVE_FREQ:
                raise PEFTArtifactError("legacy paper save_freq must be 2000")
        if self.train_counts.get("seed") != PAPER_TRAIN_SEED or self.seed != PAPER_TRAIN_SEED:
            raise PEFTArtifactError("training seed must be 1000")
        if "training_scope" in self.train_counts and self.train_counts["training_scope"] != "task_specific":
            raise PEFTArtifactError("train_counts.training_scope must be task_specific")
        if not isinstance(self.eval_counts, Mapping):
            raise PEFTArtifactError("eval_counts must be a mapping")
        expected_eval_seeds = list(PAPER_EVAL_SEEDS)
        if self.eval_counts.get("seeds") != expected_eval_seeds:
            raise PEFTArtifactError("eval_counts.seeds must be exactly 1000..1009")
        eval_mode = self.eval_counts.get("mode", "paired")
        if eval_mode not in {"paired", "adapted_only"}:
            raise PEFTArtifactError("eval_counts.mode must be paired or adapted_only")
        if eval_mode == "adapted_only":
            baseline = self.eval_counts.get("baseline")
            if not isinstance(baseline, Mapping) or set(baseline) != {"status", "reason"}:
                raise PEFTArtifactError(
                    "adapted_only eval_counts.baseline must contain only status and reason"
                )
            if baseline.get("status") != "SKIPPED":
                raise PEFTArtifactError(
                    "adapted_only eval_counts.baseline.status must be SKIPPED"
                )
            if not isinstance(baseline.get("reason"), str) or not baseline["reason"].strip():
                raise PEFTArtifactError(
                    "adapted_only eval_counts.baseline.reason must be non-empty"
                )
            adapted = self.eval_counts.get("adapted")
            if not isinstance(adapted, Mapping) or adapted.get("status") != "COMPLETED":
                raise PEFTArtifactError(
                    "adapted_only eval_counts.adapted.status must be COMPLETED"
                )
            if adapted.get("episodes") != len(expected_eval_seeds):
                raise PEFTArtifactError("eval_counts.adapted.episodes must be exactly 10")
            if adapted.get("seeds") != expected_eval_seeds:
                raise PEFTArtifactError("eval_counts.adapted.seeds must be exactly 1000..1009")
        else:
            for phase in ("baseline", "adapted"):
                phase_counts = self.eval_counts.get(phase)
                if not isinstance(phase_counts, Mapping) or phase_counts.get("episodes") != len(expected_eval_seeds):
                    raise PEFTArtifactError(f"eval_counts.{phase}.episodes must be exactly 10")
                if "seeds" in phase_counts and phase_counts["seeds"] != expected_eval_seeds:
                    raise PEFTArtifactError(f"eval_counts.{phase}.seeds must be exactly 1000..1009")
        if any(not isinstance(mapping, Mapping) or not mapping for mapping in (self.train_counts, self.eval_counts, self.runtime_versions)):
            raise PEFTArtifactError("train_counts, eval_counts, and runtime_versions must be non-empty mappings")


def _validate_identity(vla: str, task_id: int, run_id: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(vla) or not _SAFE_COMPONENT.fullmatch(run_id):
        raise PEFTArtifactError("vla and run_id must be single safe path components")
    if not 0 <= int(task_id) <= 9:
        raise PEFTArtifactError("task_id must be in 0..9")


def _resolve_digest_input(value: str | Path | Mapping[str, Any], label: str) -> tuple[str, str]:
    if isinstance(value, Mapping):
        path_value = value.get("path")
        expected = value.get("sha256", value.get("digest"))
    else:
        path_value, expected = value, None
    if not path_value:
        raise PEFTArtifactError(f"{label} path is required")
    path = Path(path_value).expanduser().resolve()
    actual = sha256_path(path)
    if expected is not None and str(expected).lower() != actual.lower():
        raise PEFTArtifactError(f"{label} digest mismatch: expected {expected}, got {actual}")
    return str(path), actual


def save_peft_adapter(
    source_adapter: str | Path,
    output_root: str | Path,
    *,
    vla: str,
    task_id: int,
    run_id: str,
    base_checkpoint: str | Path | Mapping[str, Any],
    base_checkpoint_revision: str,
    collection_manifest: str | Path | Mapping[str, Any],
    collection_success_count: int,
    controller_config_hash: str,
    checkpoint_step: int,
    seed: int,
    train_counts: Mapping[str, Any],
    eval_counts: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    runtime_evidence: str | Path | Mapping[str, Any],
    git_commit: str,
    runtime_versions: Mapping[str, Any],
) -> PEFTArtifactManifest:
    """Publish one complete adapter directory exactly once via atomic rename."""
    _validate_identity(vla, task_id, run_id)
    source = Path(source_adapter).expanduser().resolve()
    if not source.is_dir():
        raise PEFTArtifactError(f"source adapter directory does not exist: {source}")
    source_files = sorted(path for path in source.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    for required in REQUIRED_FILES:
        candidate = source / required
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise PEFTArtifactError(f"source adapter is missing required non-empty file: {candidate}")
    collection_success_count = _require_positive_count(
        collection_success_count, "collection_success_count"
    )
    base_path, base_digest = _resolve_digest_input(base_checkpoint, "base checkpoint")
    collection_path, collection_digest = _resolve_digest_input(collection_manifest, "collection manifest")
    collection = load_arrow_collection_manifest(
        collection_path, task_id=int(task_id), expected_successes=collection_success_count
    )
    try:
        collection_payload = json.loads(Path(collection_path).read_text(encoding="utf-8"))
        dataset_source_path = Path(collection["dataset_manifest_path"])
        dataset_payload = json.loads(dataset_source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise PEFTArtifactError("validated collection lineage cannot be bundled") from exc
    if not isinstance(collection_payload, Mapping) or not isinstance(dataset_payload, Mapping):
        raise PEFTArtifactError("collection and dataset manifests must be JSON objects")
    if isinstance(runtime_evidence, Mapping):
        runtime_payload = dict(runtime_evidence)
    else:
        runtime_path = _resolve_existing_path(runtime_evidence, "runtime_evidence")
        try:
            runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PEFTArtifactError("runtime_evidence is unreadable") from exc
        if not isinstance(runtime_payload, Mapping):
            raise PEFTArtifactError("runtime_evidence must be a JSON object")
    _validate_runtime_evidence(
        runtime_payload,
        expected_updates=int(checkpoint_step),
        expected_warmup=int(optimizer.get("warmup_steps", runtime_payload.get("scheduler", {}).get("warmup_steps", 0))) if isinstance(optimizer, Mapping) else None,
        expected_decay=int(optimizer.get("decay_steps", checkpoint_step)) if isinstance(optimizer, Mapping) else int(checkpoint_step),
    )
    _require_sha256(controller_config_hash, "controller_config_hash")
    if checkpoint_step <= 0:
        raise PEFTArtifactError("artifacts require a positive checkpoint step")
    if collection["controller_config_hash"] != controller_config_hash.lower():
        raise PEFTArtifactError("controller_config_hash differs from the collection manifest")
    if not base_checkpoint_revision or not git_commit or not isinstance(runtime_versions, Mapping) or not runtime_versions:
        raise PEFTArtifactError("base revision, git_commit, and runtime_versions are required provenance")
    parent = Path(output_root).expanduser().resolve() / vla / f"task_{int(task_id)}" / "peft_adapter"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / run_id
    if final.exists():
        raise FileExistsError(f"refusing to overwrite immutable PEFT artifact: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".partial", dir=parent))
    try:
        for source_file in source_files:
            relative = source_file.relative_to(source)
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        lineage = temporary / "lineage"
        lineage.mkdir(parents=True, exist_ok=True)
        accepted_source = Path(collection["accepted_episodes_jsonl"])
        dataset_source_digest = sha256_file(dataset_source_path)
        if dataset_source_digest != collection["dataset_manifest_sha256"]:
            raise PEFTArtifactError("collection dataset manifest digest changed before bundling")
        # Preserve exact source documents for audit, and write relocatable
        # copies whose only references are other files under lineage/.
        shutil.copy2(collection_path, lineage / "collection_manifest.source.json")
        shutil.copy2(accepted_source, lineage / "accepted_episodes.jsonl")
        shutil.copy2(dataset_source_path, lineage / "dataset_manifest.source.json")
        dataset_tree_source = Path(collection["dataset_root"])
        dataset_tree_destination = lineage / "dataset_tree"
        if not dataset_tree_source.is_dir():
            raise PEFTArtifactError("validated collection dataset root is missing before bundling")
        shutil.copytree(dataset_tree_source, dataset_tree_destination)
        normalized_collection = dict(collection_payload)
        normalized_collection.update({
            "accepted_episodes_jsonl": "accepted_episodes.jsonl",
            "dataset_root": "dataset_tree",
            "dataset_manifest_path": "dataset_manifest.json",
            "source_collection_manifest_sha256": collection_digest,
            "source_dataset_manifest_sha256": dataset_source_digest,
        })
        normalized_dataset = dict(dataset_payload)
        normalized_dataset.update({
            "dataset_root": "dataset_tree",
            "source_accepted_episodes": "accepted_episodes.jsonl",
            "source_dataset_manifest_sha256": dataset_source_digest,
            "dataset_tree_sha256": tree_sha256(dataset_tree_destination),
        })
        (lineage / "collection_manifest.json").write_bytes(_canonical_json(normalized_collection))
        (lineage / "dataset_manifest.json").write_bytes(_canonical_json(normalized_dataset))
        (lineage / "runtime_evidence.json").write_bytes(_canonical_json(dict(runtime_payload)))
        (lineage / "base_checkpoint.json").write_bytes(_canonical_json({
            "schema_version": 1,
            "filename": Path(base_path).name,
            "sha256": base_digest,
            "revision": str(base_checkpoint_revision),
        }))
        bundled_collection_digest = sha256_file(lineage / "collection_manifest.json")
        bundled_dataset_digest = sha256_file(lineage / "dataset_manifest.json")
        bundled_runtime_digest = sha256_file(lineage / "runtime_evidence.json")
        files = {
            path.relative_to(temporary).as_posix(): {
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            }
            for path in sorted(temporary.rglob("*")) if path.is_file()
        }
        manifest = PEFTArtifactManifest(
            schema_version=SCHEMA_VERSION, artifact_path=".", vla=vla,
            task_id=int(task_id), run_id=run_id,
            method=(FRESH_METHOD_LABEL if collection.get("source_kind") == FRESH_COLLECTION_SOURCE_KIND else "peft_lora"),
            training_scope="task_specific", trained_task_ids=[int(task_id)], files=files,
            artifact_tree_sha256=tree_sha256(temporary), base_checkpoint_path="lineage/base_checkpoint.json",
            base_checkpoint_sha256=base_digest, base_checkpoint_revision=str(base_checkpoint_revision),
            collection_manifest_path="lineage/collection_manifest.json", collection_manifest_sha256=bundled_collection_digest,
            source_collection_manifest_sha256=collection_digest,
            dataset_manifest_path="lineage/dataset_manifest.json", dataset_manifest_sha256=bundled_dataset_digest,
            source_dataset_manifest_sha256=dataset_source_digest,
            runtime_evidence_path="lineage/runtime_evidence.json", runtime_evidence_sha256=bundled_runtime_digest,
            lineage_tree_sha256=tree_sha256(lineage),
            collection_success_count=int(collection_success_count), controller_config_hash=controller_config_hash.lower(),
            source_kind=str(collection.get("source_kind", COLLECTION_SOURCE_KIND)), checkpoint_step=int(checkpoint_step), seed=int(seed),
            train_counts=dict(train_counts), eval_counts=dict(eval_counts), optimizer=dict(optimizer),
            git_commit=git_commit, runtime_versions=dict(runtime_versions),
        )
        manifest.validate()
        # Store only a relocatable token on disk.  The returned object keeps
        # the concrete path as a convenience for the current process.
        disk_manifest = replace(manifest, artifact_path=".")
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(asdict(disk_manifest), allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if final.exists():
            raise FileExistsError(f"immutable PEFT artifact appeared during save: {final}")
        os.rename(temporary, final)
        temporary = Path()
        return replace(manifest, artifact_path=str(final))
    finally:
        if str(temporary) not in {"", "."} and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _bundled_json(root: Path, relative: str, label: str) -> Mapping[str, Any]:
    safe = _safe_relative_path(relative, label)
    path = root / safe
    if not path.is_file():
        raise PEFTArtifactError(f"bundled {label} is missing: {safe.as_posix()}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PEFTArtifactError(f"bundled {label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise PEFTArtifactError(f"bundled {label} must be a JSON object")
    return payload


def _validate_bundled_lineage(path: Path, manifest: PEFTArtifactManifest) -> None:
    """Revalidate detached lineage after the original scratch tree is gone."""
    lineage = path / "lineage"
    if not lineage.is_dir():
        raise PEFTArtifactError("immutable adapter is missing bundled lineage")
    if tree_sha256(lineage) != manifest.lineage_tree_sha256:
        raise PEFTArtifactError("bundled lineage tree digest mismatch")
    collection_path = path / _safe_relative_path(manifest.collection_manifest_path, "collection_manifest_path")
    dataset_path = path / _safe_relative_path(manifest.dataset_manifest_path, "dataset_manifest_path")
    runtime_path = path / _safe_relative_path(manifest.runtime_evidence_path, "runtime_evidence_path")
    base_path = path / _safe_relative_path(manifest.base_checkpoint_path, "base_checkpoint_path")
    for required in (collection_path, dataset_path, runtime_path, base_path):
        if not required.is_file():
            raise PEFTArtifactError(f"bundled lineage file is missing: {required.relative_to(path).as_posix()}")
    if sha256_file(collection_path) != manifest.collection_manifest_sha256:
        raise PEFTArtifactError("bundled collection manifest digest mismatch")
    if sha256_file(dataset_path) != manifest.dataset_manifest_sha256:
        raise PEFTArtifactError("bundled dataset manifest digest mismatch")
    if sha256_file(runtime_path) != manifest.runtime_evidence_sha256:
        raise PEFTArtifactError("bundled runtime evidence digest mismatch")
    source_collection = path / "lineage" / "collection_manifest.source.json"
    source_dataset = path / "lineage" / "dataset_manifest.source.json"
    for required in (source_collection, source_dataset):
        if not required.is_file():
            raise PEFTArtifactError(f"bundled source lineage file is missing: {required.relative_to(path).as_posix()}")
    if sha256_file(source_collection) != manifest.source_collection_manifest_sha256:
        raise PEFTArtifactError("source collection manifest digest mismatch")
    if sha256_file(source_dataset) != manifest.source_dataset_manifest_sha256:
        raise PEFTArtifactError("source dataset manifest digest mismatch")
    collection = _bundled_json(path, manifest.collection_manifest_path, "collection manifest")
    dataset = _bundled_json(path, manifest.dataset_manifest_path, "dataset manifest")
    runtime = _bundled_json(path, manifest.runtime_evidence_path, "runtime evidence")
    base = _bundled_json(path, manifest.base_checkpoint_path, "base checkpoint descriptor")
    if collection.get("source_kind") != manifest.source_kind or collection.get("task_id") != manifest.task_id:
        raise PEFTArtifactError("bundled collection identity does not match adapter")
    expected_method = FRESH_METHOD_LABEL if manifest.source_kind == FRESH_COLLECTION_SOURCE_KIND else "peft_lora"
    if collection.get("method_label") != expected_method or manifest.method != expected_method:
        raise PEFTArtifactError("bundled collection method does not match adapter method")
    expected_successes = _require_positive_count(
        manifest.collection_success_count, "collection_success_count"
    )
    if collection.get("task_ids") != [manifest.task_id] or collection.get("evaluator_confirmed_successes") != expected_successes:
        raise PEFTArtifactError("bundled collection task/success contract does not match adapter")
    accepted_relative = _safe_relative_path(collection.get("accepted_episodes_jsonl"), "bundled accepted trace")
    accepted_path = collection_path.parent / accepted_relative
    if not accepted_path.is_file():
        raise PEFTArtifactError("bundled accepted episode trace is missing")
    accepted_digest = _require_sha256(collection.get("accepted_episodes_sha256"), "bundled accepted trace digest")
    if sha256_file(accepted_path) != accepted_digest:
        raise PEFTArtifactError("bundled accepted episode trace digest mismatch")
    collection_mode = str(collection.get("collection_mode", "same_episode_takeover"))
    _validate_accepted_trace(
        accepted_path,
        task_id=manifest.task_id,
        successful_trajectories=list(collection.get("successful_trajectories", [])),
        expected_successes=expected_successes,
        collection_mode=collection_mode,
    )
    if collection_mode == FRESH_COLLECTION_MODE:
        _validate_fresh_reset_contract(
            collection, list(collection.get("successful_trajectories", [])), expected_successes
        )
    if collection.get("dataset_root") != "dataset_tree":
        raise PEFTArtifactError("bundled collection dataset_root must be detached dataset_tree")
    dataset_relative = _safe_relative_path(collection.get("dataset_manifest_path"), "bundled dataset manifest reference")
    if collection_path.parent / dataset_relative != dataset_path:
        raise PEFTArtifactError("bundled collection dataset manifest reference is inconsistent")
    if dataset.get("dataset_root") != "dataset_tree":
        raise PEFTArtifactError("bundled dataset manifest must use detached dataset_tree")
    dataset_trace = dataset_path.parent / _safe_relative_path(
        dataset.get("source_accepted_episodes"), "bundled dataset trace reference"
    )
    if dataset_trace != accepted_path:
        raise PEFTArtifactError("bundled dataset trace reference is inconsistent")
    if dataset.get("source_accepted_episodes_sha256") != accepted_digest:
        raise PEFTArtifactError("bundled dataset trace digest does not match accepted trace")
    if dataset.get("source_dataset_manifest_sha256") != manifest.source_dataset_manifest_sha256:
        raise PEFTArtifactError("bundled dataset source digest does not match adapter")
    _require_sha256(dataset.get("dataset_tree_sha256"), "bundled dataset_tree_sha256")
    bundled_dataset_tree = lineage / "dataset_tree"
    if not bundled_dataset_tree.is_dir():
        raise PEFTArtifactError("bundled dataset tree is missing")
    if tree_sha256(bundled_dataset_tree) != dataset.get("dataset_tree_sha256"):
        raise PEFTArtifactError("bundled dataset tree digest mismatch")
    inventory = dataset.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise PEFTArtifactError("bundled dataset manifest must retain the full file inventory")
    seen: set[str] = set()
    for record in inventory:
        if not isinstance(record, Mapping):
            raise PEFTArtifactError("bundled dataset inventory entry must be an object")
        relative = _safe_relative_path(record.get("path"), "bundled dataset inventory path")
        if relative.as_posix() in seen:
            raise PEFTArtifactError("bundled dataset inventory contains duplicate paths")
        seen.add(relative.as_posix())
        _require_sha256(record.get("sha256"), f"bundled dataset file {relative.as_posix()}")
        candidate = bundled_dataset_tree / relative
        if not candidate.is_file() or sha256_file(candidate) != record.get("sha256"):
            raise PEFTArtifactError(f"bundled dataset file digest mismatch: {relative.as_posix()}")
    actual_dataset_files = {
        item.relative_to(bundled_dataset_tree).as_posix()
        for item in bundled_dataset_tree.rglob("*") if item.is_file()
    }
    if actual_dataset_files != seen:
        raise PEFTArtifactError("bundled dataset inventory does not match the bundled dataset tree")
    if base.get("sha256") != manifest.base_checkpoint_sha256 or base.get("revision") != manifest.base_checkpoint_revision:
        raise PEFTArtifactError("bundled base checkpoint descriptor does not match adapter")
    _validate_runtime_evidence(
        runtime,
        expected_updates=manifest.checkpoint_step,
        expected_warmup=int(manifest.optimizer.get("warmup_steps", runtime.get("scheduler", {}).get("warmup_steps", 0))),
        expected_decay=int(manifest.optimizer.get("decay_steps", manifest.checkpoint_step)),
    )


def load_peft_manifest(artifact_path: str | Path) -> PEFTArtifactManifest:
    path = Path(artifact_path).expanduser().resolve()
    manifest_path = path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise PEFTArtifactError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PEFTArtifactManifest(**payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise PEFTArtifactError(f"artifact manifest is unreadable: {manifest_path}") from exc
    manifest.validate()
    if manifest.artifact_path != ".":
        raise PEFTArtifactError("manifest artifact_path must be relocatable '.'")
    for relative, record in manifest.files.items():
        candidate = path / relative
        if not candidate.is_file() or sha256_file(candidate) != record.get("sha256"):
            raise PEFTArtifactError(f"artifact file digest mismatch: {candidate}")
    actual_files = {
        file_path.relative_to(path).as_posix()
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.name != "artifact_manifest.json"
    }
    if actual_files != set(manifest.files):
        raise PEFTArtifactError("artifact manifest inventory does not match the artifact tree")
    if tree_sha256(path, exclude={"artifact_manifest.json"}) != manifest.artifact_tree_sha256:
        raise PEFTArtifactError("artifact tree digest mismatch")
    _validate_bundled_lineage(path, manifest)
    return manifest


__all__ = [
    "COLLECTION_SOURCE_KIND", "FRESH_COLLECTION_SOURCE_KIND", "FRESH_COLLECTION_MODE", "FRESH_METHOD_LABEL",
    "PEFTArtifactError", "PEFTArtifactManifest",
    "REQUIRED_FILES", "SEALED_EVAL_SEEDS", "load_arrow_collection_manifest",
    "load_peft_manifest", "save_peft_adapter", "sha256_file", "sha256_path", "tree_sha256",
]
