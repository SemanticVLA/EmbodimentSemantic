#!/usr/bin/env python3
"""Build and validate immutable evidence for the historical action-only run.

The historical run predates the versioned LoRA policy registry.  This module
therefore performs a post-hoc, external inventory/audit and writes a portable
evidence bundle without modifying the original training manifest or
checkpoint.  The bundle is deliberately fail-closed: source bytes, the
sealed dataset pair, the pinned base snapshot, the final checkpoint, and the
bundle's own tree seal must all agree at validation time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vla_benchmarking.arrow_finetuned_vla.action_visual_lora.lora_finetuning_policy import ACTION_ONLY_LORA_V1
from vla_benchmarking.arrow_finetuned_vla.workflows.adapter_audit import (
    audit_adapter_checkpoint,
    build_expected_inventory,
    inventory_sha256,
)


SCHEMA_VERSION = 1
SEALED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SEALED_STEPS = 29190
SEALED_SAVE_FREQ = 1946
SEALED_BATCH_SIZE = 32
SEALED_SEED = 1000
SEALED_PEFT_R = 16
FINAL_CHECKPOINT_ID = "029190"
PAIR_MANIFEST_NAME = "sealed_lora_pair_manifest.json"
PAIR_SENTINEL_NAME = "sealed_lora_pair_verified.json"


def validate_verified_pair(data_root: Path, *, require_full_experiment: bool = True) -> dict[str, Any]:
    """Load the source verifier lazily so evidence metadata remains importable."""
    from vla_benchmarking.arrow_finetuned_vla.workflows.hdf5_to_lerobot_dataset import validate_verified_pair as _validate

    return _validate(data_root, require_full_experiment=require_full_experiment)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _resolve_dir(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a regular directory: {path}")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _adapter_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name == "adapter_model.safetensors":
        path = path.parent
    path = _resolve_dir(path, "checkpoint")
    if path.name != "pretrained_model":
        raise ValueError(f"checkpoint must be a pretrained_model directory: {path}")
    artifact = path / "adapter_model.safetensors"
    if not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(f"checkpoint adapter artifact is missing or empty: {artifact}")
    return path


def _tree_inventory(root: Path) -> dict[str, str]:
    root = _resolve_dir(root, "source tree")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"source tree must not contain symlinks: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha(path)
    return result


def _tree_digest(inventory: dict[str, str]) -> str:
    return hashlib.sha256(_canonical(inventory)).hexdigest()


def _audit_tree_digest(checkpoint: Path) -> str:
    """Match adapter_audit's checkpoint tree (its own JSON is excluded)."""
    inventory = _tree_inventory(checkpoint)
    inventory.pop("adapter_audit.json", None)
    return _tree_digest(inventory)


def _snapshot(path: Path, role: str, *, tree: bool = False) -> dict[str, Any]:
    if tree:
        inventory = _tree_inventory(path)
        if not inventory:
            raise ValueError(f"{role} tree is empty: {path}")
        return {"role": role, "kind": "tree", "path": str(path), "sha256": _tree_digest(inventory), "inventory": inventory}
    path = _resolve_file(path, role)
    return {"role": role, "kind": "file", "path": str(path), "sha256": _sha(path)}


def _base_snapshot(base_policy: Path) -> dict[str, Any]:
    """Snapshot only the manifest-sealed base files, excluding HF caches."""
    sealed = _validate_base_snapshot(base_policy)
    return {
        "role": "base_policy",
        "kind": "sealed_tree",
        "path": str(base_policy),
        "sha256": sealed["identity_sha256"],
        "inventory": sealed["files"],
    }


def _assert_snapshot_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before.get("kind") != after.get("kind") or before.get("sha256") != after.get("sha256"):
        raise ValueError(f"source changed during evidence capture: {before.get('role')}")
    if before.get("kind") == "tree" and before.get("inventory") != after.get("inventory"):
        raise ValueError(f"source tree changed during evidence capture: {before.get('role')}")


def _output_is_safe(output: Path, sources: list[Path]) -> None:
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if _is_within(output, source) or _is_within(source, output):
            raise ValueError(f"evidence output must be outside source paths: {output} overlaps {source}")


def _find_training_plan(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    declared = str(manifest.get("training_plan", "")).strip()
    declared_path = Path(declared).expanduser() if declared else None
    candidates: list[Path] = []
    if declared_path is not None:
        candidates.append(declared_path)
        candidates.append(manifest_path.parent / declared_path.name)
    candidates.extend((manifest_path.parent / name for name in ("training_plan.json", "training_plan.pending.json")))
    candidates.extend(sorted(manifest_path.parent.glob("*training_plan*.json")))
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    expected_hash = manifest.get("training_plan_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("training manifest lacks a sealed training_plan_sha256")
    for candidate in unique:
        if candidate.is_file() and not candidate.is_symlink():
            actual = _sha(candidate)
            if actual != expected_hash:
                # An existing declared path is authoritative.  A fallback is
                # accepted only when the archived declared path is absent.
                if declared_path is not None and candidate == declared_path:
                    raise ValueError("training_plan hash differs from training manifest")
                continue
            return candidate
    raise ValueError("training_plan is missing; sibling relocation fallback did not find matching bytes")


def _validate_plan(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    if plan.get("training_variant") not in (None, "no_arrow_treatment"):
        raise ValueError("training_plan variant is not no_arrow_treatment")
    flags = plan.get("flags", {})
    expected = {"steps": SEALED_STEPS, "save_freq": SEALED_SAVE_FREQ, "batch_size": SEALED_BATCH_SIZE, "seed": SEALED_SEED}
    for key, value in expected.items():
        if flags and int(flags.get(key, -1)) != value:
            raise ValueError(f"training_plan {key} differs from sealed schedule")
    if plan.get("base_policy_revision") not in (None, SEALED_REVISION):
        raise ValueError("training_plan base revision is not pinned")
    manifest_flags = manifest.get("flags", {})
    if flags and manifest_flags and any(int(flags.get(key, -1)) != int(manifest_flags.get(key, -2)) for key in expected):
        raise ValueError("training_plan schedule differs from training manifest")


def _validate_base_snapshot(base_policy: Path) -> dict[str, Any]:
    snapshot_path = base_policy / "base_snapshot_manifest.json"
    snapshot = _json(snapshot_path, "base snapshot manifest")
    if snapshot.get("revision") != SEALED_REVISION:
        raise ValueError("base snapshot revision is not pinned")
    files = snapshot.get("files")
    if not isinstance(files, dict) or not files or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise ValueError("base snapshot manifest has no valid file hashes")
    actual = _tree_inventory(base_policy)
    actual.pop("base_snapshot_manifest.json", None)
    extras = set(actual) - set(files)
    unapproved_extras = sorted(name for name in extras if not name.startswith(".cache/huggingface/"))
    if unapproved_extras:
        raise ValueError(f"base snapshot contains unexpected unsealed files: {unapproved_extras}")
    for name, expected in files.items():
        if actual.get(name) != expected:
            raise ValueError(f"base snapshot hash mismatch: {name}")
    identity = hashlib.sha256(_canonical({"revision": SEALED_REVISION, "files": dict(sorted(files.items()))})).hexdigest()
    return {"path": str(snapshot_path), "sha256": _sha(snapshot_path), "revision": SEALED_REVISION, "files": dict(sorted(files.items())), "identity_sha256": identity}


def validate_base_snapshot_identity(base_policy: str | Path) -> str:
    """Validate the pinned sealed files and return the cache-independent identity."""
    return str(_validate_base_snapshot(_resolve_dir(base_policy, "base policy")).get("identity_sha256"))


def _validate_training_manifest(manifest_path: Path, checkpoint: Path, base_policy: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest = _json(manifest_path, "training manifest")
    if manifest.get("experiment") != "smolvla_lora_no_arrow_treatment_training":
        raise ValueError("training manifest is not the historical no-arrow training experiment")
    if manifest.get("training_variant") != "no_arrow_treatment" or manifest.get("dataset_variant") != "control":
        raise ValueError("training manifest is not the no-arrow control lineage")
    if manifest.get("dataset_repo_id") not in (None, "local/libero_spatial_control"):
        raise ValueError("training manifest dataset is not local/libero_spatial_control")
    if manifest.get("trained_on_visual_condition") not in (None, "no_arrows"):
        raise ValueError("training manifest is not trained on no_arrows")
    if manifest.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("training manifest base revision is not pinned")
    recorded_policy = manifest.get("finetuning_policy_id") or ACTION_ONLY_LORA_V1
    if recorded_policy != ACTION_ONLY_LORA_V1:
        raise ValueError("historical evidence requires action_only_lora_v1")
    flags = manifest.get("flags")
    expected_flags = {"steps": SEALED_STEPS, "save_freq": SEALED_SAVE_FREQ, "batch_size": SEALED_BATCH_SIZE, "seed": SEALED_SEED, "peft_r": SEALED_PEFT_R}
    if not isinstance(flags, dict) or any(int(flags.get(key, -1)) != value for key, value in expected_flags.items()):
        raise ValueError("training manifest does not contain the sealed LoRA schedule")
    final_id = str(manifest.get("final_checkpoint_id", FINAL_CHECKPOINT_ID)).zfill(6)
    if final_id != FINAL_CHECKPOINT_ID or checkpoint.parent.name != FINAL_CHECKPOINT_ID:
        raise ValueError("checkpoint is not the sealed final step 029190")
    adapter = manifest.get("no_arrow_treatment_adapter")
    if not isinstance(adapter, dict):
        raise ValueError("training manifest lacks no_arrow_treatment_adapter")
    adapter_hash = adapter.get("sha256")
    if adapter_hash != _sha(checkpoint / "adapter_model.safetensors"):
        raise ValueError("adapter artifact hash differs from training manifest")
    recorded_base = str(manifest.get("base_policy", ""))
    if Path(recorded_base).name != f"smolvla_libero-{SEALED_REVISION}":
        raise ValueError("training manifest base policy is not the pinned snapshot")
    plan_path = _find_training_plan(manifest_path, manifest)
    plan = _json(plan_path, "training plan")
    _validate_plan(plan, manifest)
    _validate_base_snapshot(base_policy)
    return manifest, plan_path, plan


def _normalised_pair_contract(manifest: dict[str, Any], sentinel: dict[str, Any], pair_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Select stable contract fields, excluding paths/timestamps/raw hashes."""
    keys = (
        "pair_kind", "visual_contract", "storage_contract", "task_ids",
        "total_episodes", "total_frames", "full_experiment_ready", "launch_eligibility",
        "source_snapshot_identity", "training_contract", "comparability_contract",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = (pair_manifest or {}).get(key, sentinel.get(key, manifest.get(key)))
        if value is not None:
            result[key] = value
    fingerprints = sentinel.get("dataset_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("verified pair sentinel lacks dataset_fingerprints")
    return {"manifest_contract": result, "dataset_fingerprints": fingerprints}


def _current_pair_manifest(data_root: Path, training_manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = data_root / PAIR_MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"current sealed pair manifest is missing: {path}")
    expected = training_manifest.get("pair_manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("training manifest lacks a sealed pair_manifest_sha256")
    if _sha(path) != expected:
        raise ValueError("current sealed pair manifest differs from training manifest")
    return path, _json(path, "sealed pair manifest")


def _pair_record(data_root: Path, manifest: dict[str, Any], sentinel: dict[str, Any]) -> dict[str, Any]:
    pair_manifest_path, pair_manifest = _current_pair_manifest(data_root, manifest)
    pair_sentinel_path = data_root / PAIR_SENTINEL_NAME
    contract = _normalised_pair_contract(manifest, sentinel, pair_manifest)
    return {
        "pair_identity": hashlib.sha256(_canonical(contract)).hexdigest(),
        "normalised_contract": contract,
        "historical_sentinel": {
            "path": str(manifest.get("pair_sentinel", "")),
            "sha256": manifest.get("pair_sentinel_sha256") if isinstance(manifest.get("pair_sentinel_sha256"), str) else None,
            "status": "unavailable",
        },
        "current_sentinel": {
            "path": str(pair_sentinel_path),
            "sha256": _sha(pair_sentinel_path) if pair_sentinel_path.is_file() else None,
            "status": "available" if pair_sentinel_path.is_file() else "unavailable",
        },
        "current_manifest": {
            "path": str(pair_manifest_path),
            "sha256": _sha(pair_manifest_path) if pair_manifest_path.is_file() else None,
            "status": "available" if pair_manifest_path.is_file() else "unavailable",
        },
    }


def _finalise_sidecars(record: dict[str, Any], manifest: dict[str, Any], checkpoint: Path) -> None:
    historical = record["historical_sentinel"]
    path_text = str(manifest.get("pair_sentinel", "")).strip()
    if path_text:
        path = Path(path_text).expanduser().resolve()
        if path.is_file():
            actual = _sha(path)
            # The historical path can now point at a later re-verification.
            # Preserve the original digest as provenance, but never relabel
            # those bytes as the contemporaneous historical sentinel.
            historical.update({
                "recorded_sha256": historical.get("sha256"),
                "observed_current_sha256": actual,
                "status": "unavailable_at_recorded_digest",
            })
            historical.pop("sha256", None)
    # Missing old run sidecars are recorded explicitly rather than silently
    # reconstructed.  The current pair sentinel was independently validated.
    declared = {
        "run_provenance.json": [],
        "expected_adapter_inventory.json": [manifest.get("expected_adapter_inventory", {}).get("path", "")],
        "adapter_audit.json": [manifest.get("adapter_audit", {}).get("path", ""), str(checkpoint / "adapter_audit.json")],
    }
    sidecars: dict[str, dict[str, Any]] = {}
    for name, values in declared.items():
        candidates = [Path(value).expanduser().resolve() for value in values if str(value).strip()]
        candidates.extend((Path(manifest.get("training_plan", "")).expanduser().resolve().parent / name,) if manifest.get("training_plan") else ())
        found = next((candidate for candidate in candidates if candidate.is_file() and not candidate.is_symlink()), None)
        sidecars[name] = {"status": "unavailable"} if found is None else {
            "status": "available", "path": str(found), "sha256": _sha(found)
        }
    record["contemporaneous_sidecars"] = sidecars


def _bundle_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name not in {"inventory.sha256", "tree_sha256"}
    }


def _write_seal(root: Path) -> None:
    inventory = _bundle_files(root)
    if not inventory:
        raise ValueError("evidence bundle is empty")
    payload = "".join(f"{digest}  {name}\n" for name, digest in inventory.items())
    (root / "inventory.sha256").write_text(payload, encoding="utf-8")
    (root / "tree_sha256").write_text(hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")


def _validate_seal(root: Path) -> dict[str, str]:
    inventory_path = root / "inventory.sha256"
    tree_path = root / "tree_sha256"
    if not inventory_path.is_file() or not tree_path.is_file():
        raise ValueError("evidence bundle inventory/tree seal is missing")
    payload = inventory_path.read_text(encoding="utf-8")
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != tree_path.read_text(encoding="utf-8").strip():
        raise ValueError("evidence bundle tree seal is invalid")
    listed: dict[str, str] = {}
    for line in payload.splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("evidence bundle inventory line is malformed") from exc
        listed[name] = digest
        path = root / name
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"evidence bundle file changed: {name}")
    actual = _bundle_files(root)
    if actual != listed:
        raise ValueError("evidence bundle inventory is incomplete or contains unexpected files")
    return listed


def _copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _prepare_inputs(training_manifest: str | Path, checkpoint: str | Path, base_policy: str | Path, data_root: str | Path, output_dir: str | Path) -> tuple[Path, Path, Path, Path, Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    manifest_path = _resolve_file(training_manifest, "training manifest")
    checkpoint_path = _adapter_dir(checkpoint)
    base_path = _resolve_dir(base_policy, "base policy")
    data_path = _resolve_dir(data_root, "data root")
    output_path = Path(output_dir).expanduser().resolve()
    _output_is_safe(output_path, [manifest_path, checkpoint_path, base_path, data_path])
    raw_manifest = _json(manifest_path, "training manifest")
    plan_path = _find_training_plan(manifest_path, raw_manifest)
    before_validation = [
        _snapshot(manifest_path, "training_manifest"),
        _snapshot(plan_path, "training_plan"),
        _snapshot(checkpoint_path, "checkpoint", tree=True),
        _base_snapshot(base_path),
        _snapshot(data_path, "data_root", tree=True),
    ]
    manifest, plan_path, plan = _validate_training_manifest(manifest_path, checkpoint_path, base_path)
    # Bind the current pair manifest to the historical training manifest
    # before any sentinel variation can be accepted by the live verifier.
    _current_pair_manifest(data_path, manifest)
    # This is the live, source-grounded gate.  It must run for both build and
    # validate; a stale copied sentinel alone is never sufficient.
    sentinel = validate_verified_pair(data_path, require_full_experiment=True)
    if not isinstance(sentinel, dict):
        raise ValueError("live pair validation did not return a sentinel object")
    pair = _pair_record(data_path, manifest, sentinel)
    _finalise_sidecars(pair, manifest, checkpoint_path)
    after_validation = [
        _snapshot(manifest_path, "training_manifest"),
        _snapshot(plan_path, "training_plan"),
        _snapshot(checkpoint_path, "checkpoint", tree=True),
        _base_snapshot(base_path),
        _snapshot(data_path, "data_root", tree=True),
    ]
    for old, new in zip(before_validation, after_validation):
        _assert_snapshot_unchanged(old, new)
    return manifest_path, checkpoint_path, base_path, data_path, output_path, manifest, plan_path, plan, {"sentinel": sentinel, "pair": pair, "source_snapshots": after_validation}


def build_legacy_action_only_evidence(training_manifest: str | Path, checkpoint: str | Path, base_policy: str | Path, data_root: str | Path, output_dir: str | Path) -> Path:
    """Create a new immutable evidence bundle, refusing to overwrite output."""
    values = _prepare_inputs(training_manifest, checkpoint, base_policy, data_root, output_dir)
    manifest_path, checkpoint_path, base_path, data_path, output_path, manifest, plan_path, plan, pair_values = values
    if output_path.exists():
        raise FileExistsError(f"evidence output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent)))
    try:
        # Snapshot all mutable inputs before external audit/model work.
        before = pair_values["source_snapshots"]
        expected = build_expected_inventory(base_path, policy_id=ACTION_ONLY_LORA_V1)
        audit = audit_adapter_checkpoint(
            checkpoint_path,
            expected_inventory=expected,
            require_expected_inventory=True,
            policy_id=ACTION_ONLY_LORA_V1,
        )
        after = [
            _snapshot(manifest_path, "training_manifest"),
            _snapshot(plan_path, "training_plan"),
            _snapshot(checkpoint_path, "checkpoint", tree=True),
            _base_snapshot(base_path),
            _snapshot(data_path, "data_root", tree=True),
        ]
        for old, new in zip(before, after):
            _assert_snapshot_unchanged(old, new)
        _copy_exact(manifest_path, stage / "training_manifest.json")
        _copy_exact(plan_path, stage / "training_plan.json")
        _copy_exact(base_path / "base_snapshot_manifest.json", stage / "base_snapshot_manifest.json")
        pair_manifest = data_path / PAIR_MANIFEST_NAME
        pair_sentinel = data_path / PAIR_SENTINEL_NAME
        if pair_manifest.is_file():
            _copy_exact(pair_manifest, stage / "pair_manifest.json")
        if pair_sentinel.is_file():
            _copy_exact(pair_sentinel, stage / "pair_sentinel.json")
        (stage / "expected_adapter_inventory.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "adapter_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "evidence_kind": "legacy_action_only_evidence_v1",
            "policy_id": ACTION_ONLY_LORA_V1,
            "training_manifest": {"source_path": str(manifest_path), "sha256": _sha(manifest_path)},
            "training_plan": {"source_path": str(plan_path), "sha256": _sha(plan_path)},
            "checkpoint": {"source_path": str(checkpoint_path), "tree_sha256": _tree_digest(_tree_inventory(checkpoint_path)), "adapter_sha256": _sha(checkpoint_path / "adapter_model.safetensors")},
            "base_policy": {"source_path": str(base_path), "base_snapshot_identity_sha256": _base_snapshot(base_path)["sha256"], "revision": SEALED_REVISION},
            "data_root": {"source_path": str(data_path), "tree_sha256": _tree_digest(_tree_inventory(data_path))},
            "final_checkpoint_id": FINAL_CHECKPOINT_ID,
            "schedule": {"steps": SEALED_STEPS, "save_freq": SEALED_SAVE_FREQ, "batch_size": SEALED_BATCH_SIZE, "seed": SEALED_SEED, "peft_r": SEALED_PEFT_R},
            "base_snapshot_identity_sha256": _base_snapshot(base_path)["sha256"],
            "pair": pair_values["pair"],
            "contemporaneous_sidecars": pair_values["pair"]["contemporaneous_sidecars"],
            "live_pair_sentinel": pair_values["sentinel"],
            "source_snapshots": after,
            "expected_inventory_sha256": expected.get("inventory_sha256"),
            "audit_checkpoint_tree_sha256": audit.get("checkpoint_tree_sha256"),
        }
        (stage / "bundle.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(stage)
        if output_path.exists():
            raise FileExistsError(f"evidence output already exists: {output_path}")
        stage.rename(output_path)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_path


def validate_legacy_action_only_evidence(evidence_dir: str | Path, training_manifest: str | Path, checkpoint: str | Path, base_policy: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Re-authenticate an evidence bundle and its current relocated sources."""
    evidence = _resolve_dir(evidence_dir, "evidence bundle")
    _validate_seal(evidence)
    metadata = _json(evidence / "bundle.json", "evidence bundle metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("evidence_kind") != "legacy_action_only_evidence_v1":
        raise ValueError("unsupported legacy action-only evidence bundle")
    if metadata.get("policy_id") != ACTION_ONLY_LORA_V1:
        raise ValueError("evidence bundle policy is not action_only_lora_v1")
    values = _prepare_inputs(training_manifest, checkpoint, base_policy, data_root, evidence)
    manifest_path, checkpoint_path, base_path, data_path, _evidence, manifest, plan_path, _plan, pair_values = values
    if _sha(manifest_path) != metadata["training_manifest"]["sha256"]:
        raise ValueError("training manifest differs from evidence bundle")
    if _sha(plan_path) != metadata["training_plan"]["sha256"]:
        raise ValueError("training plan differs from evidence bundle")
    if _sha(checkpoint_path / "adapter_model.safetensors") != metadata["checkpoint"]["adapter_sha256"]:
        raise ValueError("checkpoint adapter differs from evidence bundle")
    current_checkpoint_tree = _tree_digest(_tree_inventory(checkpoint_path))
    if current_checkpoint_tree != metadata["checkpoint"]["tree_sha256"]:
        raise ValueError("checkpoint tree differs from evidence bundle")
    current_base_identity = _validate_base_snapshot(base_path)["identity_sha256"]
    expected_base_identity = metadata.get("base_snapshot_identity_sha256", metadata["base_policy"].get("base_snapshot_identity_sha256"))
    if current_base_identity != expected_base_identity:
        raise ValueError("base snapshot identity differs from evidence bundle")
    if _tree_digest(_tree_inventory(data_path)) != metadata["data_root"]["tree_sha256"]:
        raise ValueError("data root tree differs from evidence bundle")
    if pair_values["pair"]["pair_identity"] != metadata["pair"]["pair_identity"]:
        raise ValueError("sealed pair identity differs from evidence bundle")
    expected_path = evidence / "expected_adapter_inventory.json"
    audit_path = evidence / "adapter_audit.json"
    expected = _json(expected_path, "evidence expected inventory")
    audit = _json(audit_path, "evidence adapter audit")
    if isinstance(expected.get("inventory_sha256"), str) and len(expected["inventory_sha256"]) == 64:
        if expected["inventory_sha256"] != inventory_sha256(expected):
            raise ValueError("expected inventory internal digest is invalid")
    if metadata.get("expected_inventory_sha256") and expected.get("inventory_sha256") != metadata["expected_inventory_sha256"]:
        raise ValueError("expected inventory differs from evidence metadata")
    audit_tree = audit.get("checkpoint_tree_sha256")
    if audit_tree is not None and audit_tree != _audit_tree_digest(checkpoint_path):
        raise ValueError("adapter audit is stale for the current checkpoint")
    if metadata.get("audit_checkpoint_tree_sha256") != audit.get("checkpoint_tree_sha256"):
        raise ValueError("adapter audit differs from evidence metadata")
    # Copied originals must remain byte-identical, while their source paths may
    # legitimately relocate between machines/archives.
    if (evidence / "training_manifest.json").read_bytes() != manifest_path.read_bytes():
        raise ValueError("copied training manifest differs from supplied manifest")
    if (evidence / "training_plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("copied training plan differs from supplied plan")
    return {
        "valid": True,
        "evidence_dir": str(evidence),
        "policy_id": ACTION_ONLY_LORA_V1,
        "pair_identity": metadata["pair"]["pair_identity"],
        "historical_sentinel_sha256": metadata["pair"]["historical_sentinel"].get("recorded_sha256"),
        "current_sentinel_sha256": pair_values["pair"]["current_sentinel"].get("sha256"),
        "checkpoint_tree_sha256": current_checkpoint_tree,
        "base_snapshot_identity_sha256": current_base_identity,
        "expected_inventory_sha256": metadata.get("expected_inventory_sha256"),
        "source_snapshots": metadata.get("source_snapshots", []),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--training-manifest", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--base-policy", required=True)
        sub.add_argument("--data-root", required=True)
        sub.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(build_legacy_action_only_evidence(args.training_manifest, args.checkpoint, args.base_policy, args.data_root, args.output_dir))
    else:
        print(json.dumps(validate_legacy_action_only_evidence(args.output_dir, args.training_manifest, args.checkpoint, args.base_policy, args.data_root), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(_cli())
