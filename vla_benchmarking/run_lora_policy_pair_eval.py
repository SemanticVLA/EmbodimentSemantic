#!/usr/bin/env python3
"""Matched clean-image evaluation for action-only versus visual-path LoRA.

This evaluator is deliberately a two-cell retrospective matched-checkpoint
comparison.  Both adapters are trained on the same no-arrow dataset and are
evaluated with standard text and no live visual overlay.  The manifest records
the intended training-policy difference, the fields that were verified as
shared, and the remaining runtime/training confounds.  The reset audit binds
the two cells to the same task order and randomized simulator states before
any exact-checkpoint comparison is made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from adapter_audit import audit_adapter_checkpoint, load_expected_inventory
from config import (
    DEFAULT_CAMERAS,
    LEROBOT_CAMERA_KEYS,
    RANDOMIZATION_DIMENSIONS,
    task_randomization_dimensions,
)
from lora_finetuning_policy import ACTION_ONLY_LORA_V1, ACTION_VISUAL_LORA_V1, get_policy
from legacy_action_only_evidence import (
    validate_base_snapshot_identity,
    validate_legacy_action_only_evidence,
)
from run_lora_2x2_eval import (
    TASK_IDS,
    _randomization_config_payload,
    validate_eval_info,
    validate_randomization_audit,
)


TRAINING_CAMERAS = ",".join(LEROBOT_CAMERA_KEYS)
RAW_TRAINING_CAMERAS = ",".join(DEFAULT_CAMERAS)
CELL_IDS = ("historical_action_only_lora_v1_no_arrows", "action_visual_lora_v1_no_arrows")
POLICY_IDS = (ACTION_ONLY_LORA_V1, ACTION_VISUAL_LORA_V1)
MANIFEST_FILENAME = "action_visual_lora_no_arrow_pair_manifest.json"
SUMMARY_FILENAME = "action_visual_lora_no_arrow_pair_summary.csv"
RESULTS_FILENAME = "action_visual_lora_no_arrow_pair_results.json"
SCHEMA_VERSION = 2
EXPERIMENT = "smolvla_lora_action_visual_policy_no_arrow_matched_eval_v1"
TRAINING_EXPERIMENT = "smolvla_lora_no_arrow_treatment_training"
SEALED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SEALED_STEPS = 29190
SEALED_SAVE_FREQ = 1946
SEALED_BATCH_SIZE = 32
SEALED_SEED = 1000
SEALED_PEFT_R = 16
EPISODES = 10
LEGACY_ACTION_AUDIT_KEYS = frozenset({
    "schema_version", "target_regex", "peft_type", "rank", "modules_to_save",
    "tensor_count", "tensor_keys", "tensor_shapes", "tensor_inventory_sha256",
    "adapter_sha256", "checkpoint_inventory", "checkpoint_tree_sha256",
    "nonzero_lora_b_count", "nonzero_lora_b_required", "expected_inventory_sha256",
    "expected_matched_module_names",
})
EVIDENCE_CLASSES = {
    "historical_action_only": "legacy_action_only_evidence_v1",
    "action_visual_candidate": "native_policy_evidence_v1",
}
CONCLUSION_LIMITS = [
    "The valid comparison is limited to exact-checkpoint matched-evaluation success rates and per-task deltas.",
    "This retrospective result cannot identify a strict causal effect of visual-path LoRA.",
    "The current base-snapshot identity verifies reconstruction compatibility only; historical training-byte identity is unavailable and noncontemporaneous.",
    "Paper claims require a contemporaneously retrained matched-policy comparison.",
]
COMPARABILITY_CONTRACT = {
    "intended_training_difference": "finetuning_policy_target_modules",
    "verified_shared_training_fields": [
        "base_policy_revision",
        "dataset_repo_id",
        "dataset_variant",
        "pair_manifest_sha256",
        "training_schedule",
        "training_seed",
    ],
    "verified_current_compatibility_fields": [
        "reconstruction_and_candidate_base_snapshot_identity",
    ],
    "unverified_or_runtime_confound_fields": [
        "historical_training_base_content_identity",
        "historical_vs_contemporaneous_checkpoint_age",
        "optimizer_and_rng_trajectory",
        "checkpoint_capture_context",
        "runtime_gpu_and_software_environment",
    ],
    "shared_base_revision": SEALED_REVISION,
    "shared_training_seed": SEALED_SEED,
    "shared_schedule": {
        "steps": SEALED_STEPS,
        "epochs": 15,
        "batch_size": SEALED_BATCH_SIZE,
        "peft_r": SEALED_PEFT_R,
    },
    "live_arrows": False,
    "paired_reset_states": True,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _adapter_directory(value: str) -> Path:
    if not value or not value.strip():
        raise ValueError("adapter checkpoint path must not be empty")
    path = Path(value).expanduser().resolve()
    artifact = path / "adapter_model.safetensors"
    if path.name != "pretrained_model" or not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(f"adapter checkpoint must be a non-empty pretrained_model directory: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _audit_evidence_matches(recorded: dict[str, Any], live: dict[str, Any], policy_id: str) -> bool:
    """Accept exact current evidence and the immutable pre-registry action audit.

    The historical action-only launcher wrote schema-1 audit evidence before
    the policy registry existed.  Its report consequently has no policy
    metadata.  Require the complete known legacy key set and compare every
    legacy value exactly; only the new policy metadata is absent by design.
    The manifest's recorded audit hash protects the historical bytes.
    """
    if recorded == live:
        return True
    if policy_id != ACTION_ONLY_LORA_V1 or recorded.get("schema_version") != 1:
        return False
    if set(recorded) != LEGACY_ACTION_AUDIT_KEYS:
        return False
    return all(live.get(key) == value for key, value in recorded.items())


def _validate_training_provenance(path: Path, adapter: Path, policy_id: str) -> dict[str, Any]:
    """Authenticate one training manifest and its policy-specific checkpoint."""
    if not path.is_file():
        raise ValueError(f"training manifest is missing: {path}")
    data = _read_json(path, "training manifest")
    if data.get("experiment") != TRAINING_EXPERIMENT:
        raise ValueError(f"training manifest is not {TRAINING_EXPERIMENT}")
    if data.get("training_variant") != "no_arrow_treatment" or data.get("dataset_variant") != "control":
        raise ValueError("training manifest is not the no_arrow_treatment/control lineage")
    if data.get("dataset_repo_id") != "local/libero_spatial_control":
        raise ValueError("training manifest dataset is not local/libero_spatial_control")
    if data.get("trained_on_visual_condition") != "no_arrows":
        raise ValueError("training manifest is not trained on no_arrows")
    if data.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("training manifest base policy revision is not sealed")
    # The first action-only run predates the versioned policy registry.  Its
    # manifest is intentionally accepted when both policy fields are absent;
    # the adapter inventory/audit still authenticates the actual target set.
    recorded_policy_id = data.get("finetuning_policy_id") or ACTION_ONLY_LORA_V1
    if recorded_policy_id != policy_id:
        raise ValueError(f"training manifest policy is not {policy_id}")
    policy = get_policy(policy_id)
    recorded_target_regex = data.get("finetuning_policy_target_regex")
    if recorded_target_regex is None and policy_id == ACTION_ONLY_LORA_V1:
        recorded_target_regex = policy.target_regex
    if recorded_target_regex != policy.target_regex:
        raise ValueError(f"training manifest target regex is not sealed for {policy_id}")
    flags = data.get("flags")
    expected_flags = {
        "steps": SEALED_STEPS,
        "save_freq": SEALED_SAVE_FREQ,
        "batch_size": SEALED_BATCH_SIZE,
        "seed": SEALED_SEED,
        "peft_r": SEALED_PEFT_R,
    }
    if not isinstance(flags, dict) or any(int(flags.get(key, -1)) != value for key, value in expected_flags.items()):
        raise ValueError("training flags are not the sealed 15-epoch LoRA schedule")
    base_policy = Path(str(data.get("base_policy", ""))).expanduser().resolve()
    if not str(data.get("base_policy", "")) or base_policy.name != f"smolvla_libero-{SEALED_REVISION}":
        raise ValueError("training manifest base policy path is not the sealed snapshot")
    base_snapshot_identity = validate_base_snapshot_identity(base_policy)
    if not isinstance(data.get("pair_manifest_sha256"), str) or not isinstance(data.get("pair_sentinel_sha256"), str):
        raise ValueError("training manifest lacks sealed dataset pair hashes")
    for key in ("pair_manifest", "pair_sentinel"):
        artifact = Path(str(data.get(key, ""))).expanduser().resolve()
        if not artifact.is_file() or _sha256_file(artifact) != data.get(f"{key}_sha256"):
            raise ValueError(f"training manifest {key} is missing or changed")
    adapter_record = data.get("no_arrow_treatment_adapter")
    if not isinstance(adapter_record, dict):
        raise ValueError("training manifest lacks no_arrow_treatment_adapter")
    recorded = Path(str(adapter_record.get("path", ""))).expanduser().resolve()
    if recorded.name == "adapter_model.safetensors":
        recorded = recorded.parent
    if recorded != adapter:
        raise ValueError("adapter checkpoint does not match the training manifest")
    if adapter_record.get("sha256") != _sha256_file(adapter / "adapter_model.safetensors"):
        raise ValueError("adapter artifact hash differs from training manifest")
    expected_ref = data.get("expected_adapter_inventory")
    if not isinstance(expected_ref, dict):
        raise ValueError("training manifest lacks expected_adapter_inventory")
    expected_path = Path(str(expected_ref.get("path", ""))).expanduser().resolve()
    if not expected_path.is_file() or expected_ref.get("sha256") != _sha256_file(expected_path):
        raise ValueError("expected adapter inventory is missing or changed")
    expected = load_expected_inventory(expected_path)
    if expected.get("finetuning_policy_id", ACTION_ONLY_LORA_V1) != policy_id:
        raise ValueError("expected adapter inventory policy differs from training manifest")
    if expected.get("base_policy") != str(base_policy):
        raise ValueError("expected adapter inventory base policy differs from training manifest")
    audit_ref = data.get("adapter_audit")
    if not isinstance(audit_ref, dict):
        raise ValueError("training manifest lacks adapter_audit")
    audit_path = Path(str(audit_ref.get("path", ""))).expanduser().resolve()
    if not audit_path.is_file() or audit_ref.get("sha256") != _sha256_file(audit_path):
        raise ValueError("adapter audit evidence is missing or changed")
    live_audit = audit_adapter_checkpoint(
        adapter,
        expected_inventory=expected,
        require_expected_inventory=True,
        policy_id=policy_id,
    )
    recorded_audit = _read_json(audit_path, "adapter audit")
    if not _audit_evidence_matches(recorded_audit, live_audit, policy_id):
        raise ValueError("adapter audit evidence differs from the live checkpoint")
    return {
        "manifest": data,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_file(path),
        "base_policy": str(base_policy),
        "base_policy_revision": SEALED_REVISION,
        "base_snapshot_identity_sha256": base_snapshot_identity,
        "dataset_repo_id": data["dataset_repo_id"],
        "dataset_variant": data["dataset_variant"],
        "pair_manifest_sha256": data["pair_manifest_sha256"],
        "pair_sentinel_sha256": data["pair_sentinel_sha256"],
        "flags": expected_flags,
        "finetuning_policy_id": policy_id,
        "finetuning_policy_target_regex": policy.target_regex,
        "expected_adapter_inventory": {
            "path": str(expected_path),
            "sha256": _sha256_file(expected_path),
            "policy_id": policy_id,
            "target_regex": policy.target_regex,
            "matched_module_names": expected.get("matched_module_names"),
            "trainable_parameter_numel": expected.get("trainable_parameter_numel"),
        },
        "adapter_audit": {
            "path": str(audit_path),
            "sha256": _sha256_file(audit_path),
            "checkpoint_tree_sha256": live_audit.get("checkpoint_tree_sha256"),
        },
    }


def _validate_legacy_provenance(
    *,
    evidence_bundle: Path,
    training_manifest: Path,
    adapter: Path,
    base_policy: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Validate historical action-only evidence through its canonical API only."""
    if not evidence_bundle.is_dir():
        raise ValueError(f"legacy action-only evidence bundle is missing: {evidence_bundle}")
    if not data_root.is_dir():
        raise ValueError(f"training data root is missing: {data_root}")
    try:
        result = validate_legacy_action_only_evidence(
            evidence_bundle,
            training_manifest,
            adapter,
            base_policy,
            data_root,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"legacy action-only evidence validation failed: {exc}") from exc
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise ValueError("legacy action-only evidence validator did not return valid=true")
    base_snapshot_identity = validate_base_snapshot_identity(base_policy)
    if result.get("base_snapshot_identity_sha256") != base_snapshot_identity:
        raise ValueError("legacy evidence base snapshot identity differs from the supplied base")
    pair_identity = result.get("pair_identity")
    if not isinstance(pair_identity, str) or len(pair_identity) != 64:
        raise ValueError("legacy evidence lacks a stable verified pair identity")
    bundle_metadata = _read_json(evidence_bundle / "bundle.json", "legacy evidence metadata")
    declared_pair_identity = (bundle_metadata.get("pair") or {}).get("pair_identity")
    if declared_pair_identity != pair_identity:
        raise ValueError("legacy evidence stable pair identity differs from validated bundle identity")
    pair_manifest = data_root / "sealed_lora_pair_manifest.json"
    if not pair_manifest.is_file():
        raise ValueError(f"verified pair manifest is missing from training data root: {pair_manifest}")
    snapshots = result.get("source_snapshots")
    snapshot_by_role = {
        item.get("role"): item for item in snapshots if isinstance(item, dict)
    } if isinstance(snapshots, list) else {}
    historical_manifest = _read_json(training_manifest, "historical training manifest")
    return {
        "evidence_bundle": str(evidence_bundle.resolve()),
        "manifest_path": str(training_manifest.resolve()),
        "manifest_sha256": _sha256_file(training_manifest),
        "validator_result": result,
        "pair_identity": pair_identity,
        "bundle_pair_identity": declared_pair_identity,
        "pair_manifest_sha256": _sha256_file(pair_manifest),
        "pair_sentinel_sha256": historical_manifest.get("pair_sentinel_sha256"),
        "base_policy": str(base_policy),
        "base_policy_revision": SEALED_REVISION,
        "dataset_repo_id": historical_manifest.get("dataset_repo_id", "local/libero_spatial_control"),
        "dataset_variant": historical_manifest.get("dataset_variant", "control"),
        "flags": {
            "steps": 29190, "save_freq": 1946, "batch_size": 32,
            "seed": 1000, "peft_r": 16,
        },
        "base_snapshot_identity_sha256": base_snapshot_identity,
        "data_root_tree_sha256": snapshot_by_role.get("data_root", {}).get("sha256"),
    }


def build_cells(
    action_only_checkpoint: str,
    action_visual_checkpoint: str,
    output_root: Path,
    provenance: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    checkpoints = (_adapter_directory(action_only_checkpoint), _adapter_directory(action_visual_checkpoint))
    cells: list[dict[str, Any]] = []
    for index, (cell_id, policy_id, checkpoint) in enumerate(zip(CELL_IDS, POLICY_IDS, checkpoints)):
        cell = {
            "order": index,
            "cell_id": cell_id,
            "policy_id": policy_id,
            "finetuning_policy_target_regex": get_policy(policy_id).target_regex,
            "checkpoint": str(checkpoint),
            "adapter_sha256": _sha256_file(checkpoint / "adapter_model.safetensors"),
            "training_manifest": provenance[policy_id]["manifest_path"],
            "training_manifest_sha256": provenance[policy_id]["manifest_sha256"],
            "live_arrows": False,
            "visual_condition": "none",
            "context_mode": "standard",
            "context_format": "standard",
            "output_dir": str(output_root / f"seed_{SEALED_SEED}" / cell_id),
        }
        if policy_id == ACTION_VISUAL_LORA_V1:
            cell["expected_adapter_inventory"] = provenance[policy_id]["expected_adapter_inventory"]
            cell["adapter_audit"] = provenance[policy_id]["adapter_audit"]
        else:
            cell["legacy_evidence_bundle"] = provenance[policy_id]["evidence_bundle"]
            cell["stable_verified_pair_identity"] = provenance[policy_id]["pair_identity"]
        cells.append(cell)
    return cells


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("experiment") != EXPERIMENT:
        raise ValueError("unsupported or unexpected matched-evaluation manifest")
    if manifest.get("comparison_type") != "retrospective_matched_checkpoint_evaluation":
        raise ValueError("matched evaluation comparison type is not retrospective")
    if manifest.get("causal_ablation_status") != "retrospective_not_strict":
        raise ValueError("matched evaluation must disclose retrospective causal limits")
    if manifest.get("evidence_classes") != EVIDENCE_CLASSES:
        raise ValueError("matched evaluation evidence classes are incomplete")
    limits = manifest.get("conclusion_limits")
    if limits != CONCLUSION_LIMITS:
        raise ValueError("matched evaluation conclusion limits must remain exact-checkpoint-only")
    contract = manifest.get("comparability_contract")
    if not isinstance(contract, dict):
        raise ValueError("matched evaluation comparability contract is missing")
    if "only_intended_difference" in contract:
        raise ValueError("retrospective matched evaluation must not claim only_intended_difference")
    if contract != COMPARABILITY_CONTRACT:
        raise ValueError("matched evaluation comparability contract is not retrospective and explicit")
    base_identity_evidence = manifest.get("base_snapshot_identity_evidence")
    expected_identity_evidence = {
        "historical_training_base_content_identity": {
            "status": "unavailable_noncontemporaneous",
            "sha256": None,
        },
        "reconstruction_and_candidate_base_snapshot_identity": {
            "status": "current_verified_compatibility_evidence",
            "sha256": None,
        },
    }
    if not isinstance(base_identity_evidence, dict):
        raise ValueError("matched evaluation base identity evidence is missing")
    if set(base_identity_evidence) != set(expected_identity_evidence):
        raise ValueError("matched evaluation base identity evidence fields are invalid")
    historical_identity = base_identity_evidence["historical_training_base_content_identity"]
    current_identity = base_identity_evidence["reconstruction_and_candidate_base_snapshot_identity"]
    if historical_identity != expected_identity_evidence["historical_training_base_content_identity"]:
        raise ValueError("historical training base content identity must be unavailable")
    if not isinstance(current_identity, dict) or current_identity.get("status") != expected_identity_evidence["reconstruction_and_candidate_base_snapshot_identity"]["status"]:
        raise ValueError("current base identity is not marked as compatibility evidence")
    if not isinstance(current_identity.get("sha256"), str) or len(current_identity["sha256"]) != 64:
        raise ValueError("current base identity compatibility evidence is malformed")
    sentinel_digests = manifest.get("raw_pair_sentinel_digests")
    if not isinstance(sentinel_digests, dict) or not all(isinstance(sentinel_digests.get(key), str) for key in ("historical_action_only", "action_visual_candidate")):
        raise ValueError("matched evaluation must record both raw pair sentinel digests")
    if not isinstance(manifest.get("known_noncontemporaneous_evidence"), list) or not manifest["known_noncontemporaneous_evidence"]:
        raise ValueError("matched evaluation must disclose noncontemporaneous evidence")
    stable_identity = manifest.get("stable_verified_pair_identity")
    if not isinstance(stable_identity, str) or len(stable_identity) != 64:
        raise ValueError("matched evaluation lacks stable verified pair identity")
    if not isinstance(manifest.get("verified_pair_manifest_sha256"), str) or len(manifest["verified_pair_manifest_sha256"]) != 64:
        raise ValueError("matched evaluation lacks verified pair-manifest digest")
    if manifest.get("tasks") != list(TASK_IDS) or manifest.get("seeds") != [SEALED_SEED]:
        raise ValueError("matched evaluation is sealed to tasks 0..9 and seed 1000")
    if manifest.get("episodes") != EPISODES or manifest.get("batch_size") != 1:
        raise ValueError("matched evaluation is sealed to 10 episodes/task and batch_size=1")
    if manifest.get("randomize_scenes") is not True or manifest.get("visual_condition") != "none":
        raise ValueError("matched evaluation must use randomized clean/no-arrow observations")
    if manifest.get("camera_name") != TRAINING_CAMERAS or manifest.get("raw_camera_names") != RAW_TRAINING_CAMERAS:
        raise ValueError("matched evaluation camera contract differs from training")
    if manifest.get("observation_height") != 256 or manifest.get("observation_width") != 256:
        raise ValueError("matched evaluation must use 256x256 observations")
    if manifest.get("n_action_steps") != "checkpoint":
        raise ValueError("matched evaluation must preserve checkpoint n_action_steps")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise ValueError("matched evaluation must contain exactly two cells")
    if [cell.get("cell_id") for cell in cells] != list(CELL_IDS):
        raise ValueError("matched evaluation cells are not in canonical order")
    if [cell.get("policy_id") for cell in cells] != list(POLICY_IDS):
        raise ValueError("matched evaluation policy order is invalid")
    if any(cell.get("live_arrows") is not False or cell.get("visual_condition") != "none" for cell in cells):
        raise ValueError("matched evaluation contains a live-arrow cell")
    for cell in cells:
        if not isinstance(cell.get("adapter_sha256"), str) or not isinstance(cell.get("training_manifest_sha256"), str):
            raise ValueError("matched evaluation cell lacks immutable adapter/training hashes")
        policy = get_policy(cell["policy_id"])
        if cell.get("finetuning_policy_target_regex") != policy.target_regex:
            raise ValueError(f"cell {cell['cell_id']} target regex is not sealed for {policy.policy_id}")
        if cell["policy_id"] == ACTION_VISUAL_LORA_V1:
            for evidence_name in ("expected_adapter_inventory", "adapter_audit"):
                evidence = cell.get(evidence_name)
                if not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str) or not isinstance(evidence.get("sha256"), str):
                    raise ValueError(f"cell {cell['cell_id']} lacks explicit {evidence_name} evidence")
        else:
            if not isinstance(cell.get("legacy_evidence_bundle"), str) or cell.get("stable_verified_pair_identity") != stable_identity:
                raise ValueError("historical cell lacks validated legacy evidence binding")
    if manifest.get("paired_reset_states") is not True:
        raise ValueError("matched evaluation must require paired reset states")
    dimensions = manifest.get("randomization_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != {str(task) for task in TASK_IDS}:
        raise ValueError("matched evaluation must record randomization dimensions for every task")
    for task, values in dimensions.items():
        if not isinstance(values, dict) or not values.get("object_removal"):
            raise ValueError(f"task {task} does not have the sealed object-removal randomization")


def build_manifest(
    *,
    action_only_checkpoint: str,
    action_visual_checkpoint: str,
    action_only_training_manifest: str,
    action_visual_training_manifest: str,
    action_only_legacy_evidence_bundle: str,
    training_data_root: str,
    output_root: Path,
    episodes: int = EPISODES,
    batch_size: int = 1,
    device: str = "cuda",
    videos: bool = False,
    max_videos: int = 0,
) -> dict[str, Any]:
    if episodes != EPISODES:
        raise ValueError("matched evaluation requires exactly 10 episodes per task")
    if batch_size != 1:
        raise ValueError("matched reset-state evaluation requires batch_size=1")
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    only_path = Path(action_only_training_manifest).expanduser().resolve()
    visual_path = Path(action_visual_training_manifest).expanduser().resolve()
    action_only_adapter = _adapter_directory(action_only_checkpoint)
    action_visual_adapter = _adapter_directory(action_visual_checkpoint)
    if action_only_adapter == action_visual_adapter:
        raise ValueError("matched policy cells must use distinct checkpoint paths")
    if _sha256_file(action_only_adapter / "adapter_model.safetensors") == _sha256_file(action_visual_adapter / "adapter_model.safetensors"):
        raise ValueError("matched policy cells must use distinct adapter artifact identities")
    # The candidate must pass the native policy/inventory/audit path.  The
    # historical action-only checkpoint is intentionally validated only by
    # the portable legacy evidence bundle API.
    candidate = _validate_training_provenance(
        visual_path, action_visual_adapter, ACTION_VISUAL_LORA_V1
    )
    legacy = _validate_legacy_provenance(
        evidence_bundle=Path(action_only_legacy_evidence_bundle).expanduser().resolve(),
        training_manifest=only_path,
        adapter=action_only_adapter,
        base_policy=Path(candidate["base_policy"]),
        data_root=Path(training_data_root).expanduser().resolve(),
    )
    if legacy["pair_manifest_sha256"] != candidate["pair_manifest_sha256"]:
        raise ValueError("historical and candidate pair-manifest digests differ")
    if legacy.get("pair_identity") is None:
        raise ValueError("historical evidence lacks stable verified pair identity")
    if not isinstance(legacy.get("base_snapshot_identity_sha256"), str) or len(legacy["base_snapshot_identity_sha256"]) != 64:
        raise ValueError("historical evidence lacks validated base/content identity")
    if candidate.get("base_snapshot_identity_sha256") != legacy["base_snapshot_identity_sha256"]:
        raise ValueError("historical and candidate base/content identities differ")
    if legacy["validator_result"].get("checkpoint_tree_sha256") is None:
        raise ValueError("historical evidence lacks validated checkpoint identity")
    if legacy["pair_sentinel_sha256"] is not None and not isinstance(legacy["pair_sentinel_sha256"], str):
        raise ValueError("historical evidence raw pair sentinel digest is malformed")
    if legacy["pair_sentinel_sha256"] is None:
        raise ValueError("historical training manifest lacks raw pair sentinel digest")
    for key in ("base_policy", "base_policy_revision", "dataset_repo_id", "dataset_variant", "flags"):
        if legacy.get(key) is not None and legacy[key] != candidate[key]:
            raise ValueError(f"paired training manifests differ in sealed field {key}")
    if only_path == visual_path:
        raise ValueError("matched policies require two distinct training manifests")
    root = output_root.expanduser().resolve()
    dimensions = {str(task): task_randomization_dimensions(task) for task in TASK_IDS}
    config = _randomization_config_payload()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "plan_only" if device == "plan" else "ready",
        "tasks": list(TASK_IDS),
        "seeds": [SEALED_SEED],
        "episodes": EPISODES,
        "batch_size": 1,
        "device": device,
        "videos": bool(videos),
        "max_videos": int(max_videos if videos else 0),
        "randomize_scenes": True,
        "paired_reset_states": True,
        "clean_no_arrow_observations": True,
        "camera_name": TRAINING_CAMERAS,
        "raw_camera_names": RAW_TRAINING_CAMERAS,
        "observation_height": 256,
        "observation_width": 256,
        "context_mode": "standard",
        "context_format": "standard",
        "n_action_steps": "checkpoint",
        "visual_condition": "none",
        "training_conditions": {"training_variant": "no_arrow_treatment", "dataset_variant": "control", "trained_on_visual_condition": "no_arrows"},
        "shared_training_provenance": {
            key: candidate[key]
            for key in (
                "base_policy",
                "base_policy_revision",
                "dataset_repo_id",
                "dataset_variant",
                "flags",
            )
        },
        "base_snapshot_identity_evidence": {
            "historical_training_base_content_identity": {
                "status": "unavailable_noncontemporaneous",
                "sha256": None,
            },
            "reconstruction_and_candidate_base_snapshot_identity": {
                "status": "current_verified_compatibility_evidence",
                "sha256": candidate["base_snapshot_identity_sha256"],
            },
        },
        "stable_verified_pair_identity": legacy["pair_identity"],
        "verified_pair_manifest_sha256": candidate["pair_manifest_sha256"],
        "raw_pair_sentinel_digests": {
            "historical_action_only": legacy["pair_sentinel_sha256"],
            "action_visual_candidate": candidate["pair_sentinel_sha256"],
        },
        "comparison_type": "retrospective_matched_checkpoint_evaluation",
        "evidence_classes": dict(EVIDENCE_CLASSES),
        "known_noncontemporaneous_evidence": [
            "historical action-only checkpoint and legacy evidence bundle predate the action_visual_lora_v1 run",
            "raw pair sentinel digests are recorded independently and are not required to be byte-identical",
            "historical training base-content identity is unavailable; the current base identity is reconstruction compatibility evidence only",
        ],
        "causal_ablation_status": "retrospective_not_strict",
        "conclusion_limits": list(CONCLUSION_LIMITS),
        "randomization_dimensions": dimensions,
        "randomization_dimension_names": list(RANDOMIZATION_DIMENSIONS),
        "randomization_config": config,
        "randomization_config_sha256": hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest(),
        "output_root": str(root),
        "contrast": "action_visual_minus_action_only_clean_no_arrow_pp",
        "cells": build_cells(
            action_only_checkpoint, action_visual_checkpoint, root,
            {ACTION_ONLY_LORA_V1: legacy, ACTION_VISUAL_LORA_V1: candidate},
        ),
        "comparability_contract": json.loads(_canonical_json(COMPARABILITY_CONTRACT)),
    }
    _validate_manifest(manifest)
    return manifest


def _write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["manifest_sha256"] = _manifest_digest(manifest)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"immutable matched evaluation manifest differs: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def validate_existing_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if not root.exists():
        return
    expected = {Path(cell["output_dir"]).resolve() for cell in manifest["cells"]}
    for item in root.rglob("*"):
        if item.is_dir() and item.name.startswith("action_") and item.resolve() not in expected:
            raise ValueError(f"unexpected stale cell output directory: {item}")
    for cell in manifest["cells"]:
        marker = Path(cell["output_dir"]) / "cell_manifest.json"
        if marker.exists():
            actual = _read_json(marker, "cell marker")
            expected_marker = {key: cell[key] for key in ("cell_id", "policy_id", "finetuning_policy_target_regex", "checkpoint", "adapter_sha256", "live_arrows", "visual_condition", "output_dir")}
            if actual != expected_marker:
                raise ValueError(f"stale cell marker does not match manifest: {marker}")


def _write_cell_marker(cell: dict[str, Any]) -> None:
    output = Path(cell["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "cell_manifest.json"
    expected = {key: cell[key] for key in ("cell_id", "policy_id", "finetuning_policy_target_regex", "checkpoint", "adapter_sha256", "live_arrows", "visual_condition", "output_dir")}
    if marker.exists() and _read_json(marker, "cell marker") != expected:
        raise ValueError(f"cell marker mismatch: {marker}")
    if not marker.exists():
        marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cell(cell: dict[str, Any], args: argparse.Namespace, script_dir: Path) -> int:
    _write_cell_marker(cell)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MODELS": cell["checkpoint"],
        "TRAINING_PROFILE": "no_arrow_treatment",
        "PROFILE": "no_arrow_treatment",
        "FINETUNING_POLICY_ID": cell["policy_id"],
        "TASK_IDS": json.dumps(list(TASK_IDS), separators=(",", ":")),
        "N_EPISODES": str(args.episodes),
        "BATCH_SIZE": "1",
        "SEED": str(SEALED_SEED),
        "DEVICE": args.device,
        "CONTEXT_MODE": "standard",
        "CONTEXT_FORMAT": "standard",
        "VISUAL_CONDITION": "none",
        "VISUAL_ARROWS": "0",
        "RANDOMIZE_SCENES": "1",
        "N_ACTION_STEPS": "checkpoint",
        "MAX_EPISODES_RENDERED": str(args.max_videos if args.videos else 0),
        "RENDER_MODE": "rgb_array" if args.videos else "none",
    })
    command = [
        sys.executable, str(script_dir / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false", f"--output_dir={cell['output_dir']}",
        f"--policy.path={cell['checkpoint']}",
        "--env.task_ids=" + json.dumps(list(TASK_IDS), separators=(",", ":")),
        "--env.camera_name=" + TRAINING_CAMERAS,
        "--env.observation_height=256", "--env.observation_width=256",
    ]
    log_path = Path(cell["output_dir"]) / "eval_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=script_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def _validate_no_live_arrows(cell: dict[str, Any]) -> None:
    path = Path(cell["output_dir"]) / "visual_relation_audit.jsonl"
    if path.exists() and path.stat().st_size:
        raise ValueError(f"clean evaluation emitted visual arrow audit evidence: {path}")


def _validate_task_order(info: dict[str, Any]) -> None:
    task_order = [record.get("task_id") for record in info.get("per_task", [])]
    if task_order != list(TASK_IDS):
        raise ValueError(f"evaluation task order differs from sealed order {list(TASK_IDS)}: {task_order}")


def _emit_episode_records(cell: dict[str, Any], info: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in info["per_task"]:
        task_id = int(task["task_id"])
        metrics = task["metrics"]
        for index, (success, total, maximum) in enumerate(zip(metrics["successes"], metrics["sum_rewards"], metrics["max_rewards"])):
            records.append({
                "cell_id": cell["cell_id"], "policy_id": cell["policy_id"], "seed": SEALED_SEED,
                "task_id": task_id, "episode_index": index, "success": bool(success),
                "sum_reward": float(total), "max_reward": float(maximum),
                "live_arrows": False, "visual_condition": "none",
                "training_manifest_sha256": cell["training_manifest_sha256"],
                "adapter_sha256": cell["adapter_sha256"], "paired_reset_states": manifest["paired_reset_states"],
            })
    return records


def _task_results(cell: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in info["per_task"]:
        successes = [bool(value) for value in task["metrics"]["successes"]]
        rows.append({
            "cell_id": cell["cell_id"], "policy_id": cell["policy_id"], "task_id": int(task["task_id"]),
            "successes": int(sum(successes)), "episodes": len(successes),
            "success_rate": 100.0 * sum(successes) / len(successes),
            "adapter_sha256": cell["adapter_sha256"], "output_dir": cell["output_dir"],
        })
    return rows


def _paired_results(task_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_policy = {policy: {row["task_id"]: row for row in rows} for policy, rows in task_rows.items()}
    rows = []
    for task_id in TASK_IDS:
        only = by_policy[ACTION_ONLY_LORA_V1][task_id]
        visual = by_policy[ACTION_VISUAL_LORA_V1][task_id]
        rows.append({
            "task_id": task_id,
            "action_only_success_rate": only["success_rate"],
            "action_visual_success_rate": visual["success_rate"],
            "delta_pp": visual["success_rate"] - only["success_rate"],
        })
    only_total = sum(row["successes"] for row in by_policy[ACTION_ONLY_LORA_V1].values())
    visual_total = sum(row["successes"] for row in by_policy[ACTION_VISUAL_LORA_V1].values())
    total = len(TASK_IDS) * EPISODES
    rows.append({
        "task_id": "all",
        "action_only_success_rate": 100.0 * only_total / total,
        "action_visual_success_rate": 100.0 * visual_total / total,
        "delta_pp": 100.0 * (visual_total - only_total) / total,
    })
    return rows


def _paired_reset_audit(cells: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if len(cells) != 2:
        raise ValueError("paired reset audit requires exactly two cells")
    audits: list[list[dict[str, Any]]] = []
    for cell in cells:
        output = Path(cell["output_dir"])
        validate_randomization_audit(output, manifest)
        path = output / "randomization_audit.jsonl"
        audits.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    def identity(record: dict[str, Any]) -> tuple[Any, ...]:
        details = record.get("details", {})
        state = details.get("sim_state_sha256")
        init_state = details.get("init_state")
        if not isinstance(state, str) or len(state) != 64 or not isinstance(init_state, dict):
            raise ValueError("randomization audit lacks exact simulator/reset-state evidence")
        if not isinstance(init_state.get("selected_index"), int) or not isinstance(init_state.get("selected_row_sha256"), str):
            raise ValueError("randomization audit lacks selected init-state evidence")
        return (
            record.get("task_id"), record.get("env_index"), record.get("reset_sequence"),
            details.get("removed"), details.get("projection"), details.get("protected"),
            details.get("layout"), details.get("swaps"), state,
            init_state.get("selected_index"), init_state.get("selected_row_sha256"),
        )
    left = [identity(record) for record in audits[0]]
    right = [identity(record) for record in audits[1]]
    if left != right:
        raise ValueError("policy cells did not use identical reset/randomization identities")


def _write_summary(path: Path, task_rows: Iterable[dict[str, Any]], paired_rows: Iterable[dict[str, Any]]) -> None:
    fields = ["row_type", "cell_id", "policy_id", "task_id", "successes", "episodes", "success_rate", "action_only_success_rate", "action_visual_success_rate", "delta_pp", "adapter_sha256", "output_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in task_rows:
            writer.writerow({"row_type": "task", **row})
        for row in paired_rows:
            writer.writerow({"row_type": "paired_delta", **row})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-only-checkpoint", required=True)
    parser.add_argument("--action-visual-checkpoint", required=True)
    parser.add_argument("--action-only-training-manifest", required=True)
    parser.add_argument("--action-visual-training-manifest", required=True)
    parser.add_argument("--action-only-legacy-evidence-bundle", required=True)
    parser.add_argument("--training-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--plan-only", action="store_true", help="write and validate the immutable plan without running GPU evaluation")
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-videos", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    try:
        manifest = build_manifest(
            action_only_checkpoint=args.action_only_checkpoint,
            action_visual_checkpoint=args.action_visual_checkpoint,
            action_only_training_manifest=args.action_only_training_manifest,
            action_visual_training_manifest=args.action_visual_training_manifest,
            action_only_legacy_evidence_bundle=args.action_only_legacy_evidence_bundle,
            training_data_root=args.training_data_root,
            output_root=root,
            episodes=args.episodes,
            batch_size=args.batch_size,
            device="plan" if args.plan_only else args.device,
            videos=args.videos,
            max_videos=args.max_videos,
        )
        validate_existing_outputs(root, manifest)
        _write_immutable_manifest(root / MANIFEST_FILENAME, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: matched policy evaluation preflight failed: {exc}")
        return 1
    if args.plan_only:
        print(f"matched policy evaluation plan validated: {root / MANIFEST_FILENAME}")
        return 0

    script_dir = Path(__file__).resolve().parent
    task_rows: dict[str, list[dict[str, Any]]] = {}
    episode_records: list[dict[str, Any]] = []
    infos: dict[str, dict[str, Any]] = {}
    for cell in manifest["cells"]:
        code = _run_cell(cell, args, script_dir)
        if code != 0:
            print(f"ERROR: cell {cell['cell_id']} failed ({code})")
            return code
        try:
            output = Path(cell["output_dir"])
            info = validate_eval_info(output / "eval_info.json", manifest)
            _validate_task_order(info)
            validate_randomization_audit(output, manifest)
            _validate_no_live_arrows(cell)
        except ValueError as exc:
            print(f"ERROR: cell {cell['cell_id']} validation failed: {exc}")
            return 1
        infos[cell["policy_id"]] = info
        task_rows[cell["policy_id"]] = _task_results(cell, info)
        episode_records.extend(_emit_episode_records(cell, info, manifest))
    try:
        _paired_reset_audit(manifest["cells"], manifest)
    except ValueError as exc:
        print(f"ERROR: paired reset validation failed: {exc}")
        return 1
    paired_rows = _paired_results(task_rows)
    root.mkdir(parents=True, exist_ok=True)
    (root / "episode_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_records), encoding="utf-8")
    results = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest": str((root / MANIFEST_FILENAME).resolve()),
        "manifest_sha256": _sha256_file(root / MANIFEST_FILENAME),
        "cells": task_rows,
        "paired_delta": paired_rows,
        "episode_results": str((root / "episode_results.jsonl").resolve()),
    }
    (root / RESULTS_FILENAME).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(root / SUMMARY_FILENAME, [row for rows in task_rows.values() for row in rows], paired_rows)
    print(f"matched clean no-arrow policy evaluation complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
