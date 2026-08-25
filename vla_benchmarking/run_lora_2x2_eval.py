#!/usr/bin/env python3
"""Run a sealed base-vs-treatment 2x2 evaluation matrix.

The matrix is deliberately explicit: the frozen base checkpoint, the one
treatment adapter, and every seed/cell are recorded before the first rollout.
The base checkpoint is the control policy; no control adapter is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import (
    LEROBOT_CAMERA_KEYS,
    DEFAULT_CAMERAS,
    RANDOMIZATION_DIMENSIONS,
    TASK_REMOVE_CONFIG,
    TASK_SWAP_CONFIG,
    TASK_PROMPT_OVERRIDE,
    task_randomization_dimensions,
)


TASK_IDS = tuple(range(10))
TRAINING_CAMERAS = ",".join(LEROBOT_CAMERA_KEYS)
RAW_TRAINING_CAMERAS = ",".join(DEFAULT_CAMERAS)
CELL_IDS = (
    "base_no_arrows",
    "base_live_arrows",
    "treatment_no_arrows",
    "treatment_live_arrows",
)
MANIFEST_FILENAME = "lora_2x2_manifest.json"
SUMMARY_FILENAME = "lora_2x2_summary.csv"
SCHEMA_VERSION = 3


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated or JSON integer lists, rejecting duplicates."""
    text = value.strip()
    try:
        parsed = json.loads(text) if text.startswith("[") else text.split(",")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if isinstance(parsed, (int, float)) or not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    try:
        result = [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer list: {value!r}") from exc
    if not result:
        raise ValueError("integer list must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"integer list contains duplicates: {result}")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_manifest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _randomization_config_payload() -> dict[str, Any]:
    return {
        "remove": {str(task_id): list(values) for task_id, values in sorted(TASK_REMOVE_CONFIG.items())},
        "layout": {
            str(task_id): [list(operation) for operation in values]
            for task_id, values in sorted(TASK_SWAP_CONFIG.items())
        },
        "prompt": {str(task_id): value for task_id, value in sorted(TASK_PROMPT_OVERRIDE.items())},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resume_record_digest(record: dict[str, Any]) -> str:
    """Match the canonical record hash emitted by launch_lora_treatment.sh."""
    body = dict(record)
    body.pop("record_sha256", None)
    encoded = json.dumps(body, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_resume_audit_chain(data: dict[str, Any]) -> None:
    """Validate the immutable, treatment-only resume chain in a training manifest."""
    audits = data.get("resume_audits")
    chain_digest = data.get("resume_chain_digest")
    if not isinstance(audits, list) or not isinstance(chain_digest, str):
        raise ValueError("unified training manifest lacks resume_audits/resume_chain_digest")
    if not isinstance(data.get("dataset_variant"), str) or not data["dataset_variant"]:
        raise ValueError("unified training manifest lacks dataset_variant for resume-audit binding")
    record_hashes: list[str] = []
    previous_step: int | None = None
    for index, record in enumerate(audits, 1):
        if not isinstance(record, dict):
            raise ValueError(f"resume audit record {index} is not an object")
        required = (
            "schema_version", "chain_index", "train_config_path", "train_config_sha256",
            "checkpoint_step", "profile", "dataset_variant", "previous_record_sha256",
            "record_sha256",
        )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"resume audit record {index} is missing keys: {missing}")
        if record["schema_version"] != 1 or record["chain_index"] != index:
            raise ValueError(f"resume audit chain index/schema mismatch at record {index}")
        if record["profile"] != "treatment" or record["dataset_variant"] != data.get("dataset_variant"):
            raise ValueError(f"resume audit profile/dataset mismatch at record {index}")
        try:
            step = int(record["checkpoint_step"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resume audit checkpoint_step is invalid at record {index}") from exc
        if step < 0 or (previous_step is not None and step <= previous_step):
            raise ValueError("resume audit checkpoints are not strictly increasing")
        previous_step = step
        expected_previous = record_hashes[-1] if record_hashes else None
        if record["previous_record_sha256"] != expected_previous:
            raise ValueError(f"resume audit previous-record linkage is broken at record {index}")
        if record["record_sha256"] != _resume_record_digest(record):
            raise ValueError(f"resume audit record hash mismatch at record {index}")
        config_path = Path(record["train_config_path"]).expanduser().resolve()
        checkpoint_indexes = [i for i, part in enumerate(config_path.parts) if part == "checkpoints"]
        if not checkpoint_indexes:
            raise ValueError(f"resume config path is not under checkpoints: {config_path}")
        step_index = checkpoint_indexes[-1] + 1
        if step_index >= len(config_path.parts) or not config_path.parts[step_index].isdigit() or int(config_path.parts[step_index]) != step:
            raise ValueError(f"resume config path/checkpoint_step mismatch: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"resume train_config.json is missing: {config_path}")
        if _sha256_file(config_path) != record["train_config_sha256"]:
            raise ValueError(f"resume train_config.json hash changed: {config_path}")
        record_hashes.append(record["record_sha256"])
    expected_chain = hashlib.sha256(
        json.dumps(record_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if chain_digest != expected_chain:
        raise ValueError("resume audit chain digest mismatch")


def _checkpoint_path(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("checkpoint path must not be empty")
    # Hub IDs are valid policy paths too; local paths are made absolute so a
    # later launch cannot accidentally resolve them from a different cwd.
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.exists() else value.strip()


def build_cells(
    base_checkpoint: str,
    treatment_checkpoint: str,
    seeds: Sequence[int],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Return the complete, duplicate-free 2x2 cell list."""
    base = _checkpoint_path(base_checkpoint)
    treatment = _checkpoint_path(treatment_checkpoint)
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for cell_id, checkpoint, arrows in (
            ("base_no_arrows", base, False),
            ("base_live_arrows", base, True),
            ("treatment_no_arrows", treatment, False),
            ("treatment_live_arrows", treatment, True),
        ):
            cell_dir = output_root / f"seed_{int(seed)}" / cell_id
            cells.append(
                {
                    "cell_id": cell_id,
                    "seed": int(seed),
                    "checkpoint": checkpoint,
                    "live_arrows": arrows,
                    "output_dir": cell_dir.as_posix(),
                }
            )
    return cells


def build_manifest(
    *,
    base_checkpoint: str,
    treatment_checkpoint: str,
    seeds: Sequence[int],
    tasks: Sequence[int] = TASK_IDS,
    episodes: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
    videos: bool = False,
    max_videos: int = 0,
    training_manifest: str | None = None,
    output_root: Path,
) -> dict[str, Any]:
    tasks = [int(task_id) for task_id in tasks]
    if tasks != list(TASK_IDS):
        raise ValueError("sealed LoRA eval requires task IDs exactly 0 through 9")
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("sealed LoRA eval requires one or more unique explicit seeds")
    if episodes <= 0 or batch_size <= 0:
        raise ValueError("episodes and batch_size must be positive")
    if batch_size != 1:
        raise ValueError("sealed randomization audit requires batch_size=1")
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    root = output_root.expanduser().resolve()
    dimensions = {
        str(task_id): task_randomization_dimensions(task_id)
        for task_id in tasks
    }
    incomplete = [task_id for task_id, values in dimensions.items() if not values.get("object_removal")]
    if incomplete:
        raise ValueError(
            "sealed all-task eval requires object_removal for every task; "
            f"incomplete tasks: {incomplete}"
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "smolvla_lora_visual_arrows_base_vs_treatment_2x2",
        "base_checkpoint": _checkpoint_path(base_checkpoint),
        "treatment_checkpoint": _checkpoint_path(treatment_checkpoint),
        "tasks": tasks,
        "seeds": seeds,
        "episodes": int(episodes),
        "batch_size": int(batch_size),
        "device": device,
        "randomize_scenes": True,
        "camera_name": TRAINING_CAMERAS,
        "raw_camera_names": RAW_TRAINING_CAMERAS,
        "observation_height": 256,
        "observation_width": 256,
        "text_context_mode": "standard",
        "text_context_format": "standard",
        "n_action_steps": "checkpoint",
        "visual_arrow_width": 1,
        "visual_arrow_head_length": 16,
        "videos": bool(videos),
        "max_videos": int(max_videos if videos else 0),
        "randomization_dimensions": dimensions,
        "randomization_dimension_names": list(RANDOMIZATION_DIMENSIONS),
        "randomization_config": _randomization_config_payload(),
        "randomization_config_sha256": hashlib.sha256(
            _canonical_json(_randomization_config_payload()).encode("utf-8")
        ).hexdigest(),
        "output_root": root.as_posix(),
        "training_manifest": (
            str(Path(training_manifest).expanduser().resolve())
            if training_manifest else None
        ),
    }
    manifest["cells"] = build_cells(
        base_checkpoint,
        treatment_checkpoint,
        seeds,
        root,
    )
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing manifest schema_version")
    if manifest.get("tasks") != list(TASK_IDS):
        raise ValueError("manifest tasks must contain exactly IDs 0 through 9")
    if manifest.get("randomize_scenes") is not True:
        raise ValueError("manifest must keep RANDOMIZE_SCENES enabled")
    if manifest.get("batch_size") != 1:
        raise ValueError("sealed randomization audit requires batch_size=1")
    if manifest.get("camera_name") != TRAINING_CAMERAS:
        raise ValueError("manifest must use exact LeRobot training camera keys")
    if manifest.get("raw_camera_names") != RAW_TRAINING_CAMERAS:
        raise ValueError("manifest must record exact raw MuJoCo training cameras")
    if manifest.get("observation_height") != 256 or manifest.get("observation_width") != 256:
        raise ValueError("manifest must use 256x256 observations")
    if manifest.get("text_context_mode") != "standard" or manifest.get("text_context_format") != "standard":
        raise ValueError("sealed eval must use standard text context")
    if manifest.get("n_action_steps") != "checkpoint":
        raise ValueError("sealed eval must preserve checkpoint n_action_steps")
    if manifest.get("visual_arrow_width") != 1 or manifest.get("visual_arrow_head_length") != 16:
        raise ValueError("manifest arrow geometry must be width=1 and head=16")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds must be explicit and unique")
    dimensions = manifest.get("randomization_dimensions")
    expected_task_keys = {str(task_id) for task_id in TASK_IDS}
    if not isinstance(dimensions, dict) or set(dimensions) != expected_task_keys:
        raise ValueError("manifest must record randomization dimensions for every task")
    for task_id, values in dimensions.items():
        if not isinstance(values, dict) or not values.get("object_removal"):
            raise ValueError(f"task {task_id} must enable object_removal")
    expected_config = _randomization_config_payload()
    if manifest.get("randomization_config") != expected_config:
        raise ValueError("manifest randomization config does not match sealed config")
    expected_config_hash = hashlib.sha256(
        _canonical_json(expected_config).encode("utf-8")
    ).hexdigest()
    if manifest.get("randomization_config_sha256") != expected_config_hash:
        raise ValueError("manifest randomization config hash does not match sealed config")
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise ValueError("manifest cells must be a list")
    expected = {(int(seed), cell_id) for seed in seeds for cell_id in CELL_IDS}
    actual = {(int(cell.get("seed")), cell.get("cell_id")) for cell in cells}
    if actual != expected or len(cells) != len(actual):
        raise ValueError("manifest has incomplete, duplicate, or unexpected 2x2 cells")
    for cell in cells:
        if cell.get("cell_id") not in CELL_IDS or not cell.get("checkpoint"):
            raise ValueError("manifest cell has invalid cell_id or checkpoint")
        expected_arrows = cell["cell_id"].endswith("live_arrows")
        if bool(cell.get("live_arrows")) != expected_arrows:
            raise ValueError("manifest cell arrow condition does not match cell_id")


def compare_manifests(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return explicit experimental-condition differences between manifests."""
    keys = (
        "base_checkpoint", "treatment_checkpoint", "tasks", "seeds", "randomize_scenes", "camera_name", "raw_camera_names",
        "randomization_dimensions", "text_context_mode", "text_context_format",
        "randomization_config", "randomization_config_sha256",
        "visual_arrow_width", "visual_arrow_head_length",
    )
    return {
        key: {"left": left.get(key), "right": right.get(key)}
        for key in keys if left.get(key) != right.get(key)
    }


def validate_training_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the one-adapter treatment lineage and immutable provenance."""
    if not path.is_file():
        raise ValueError(f"treatment training manifest is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"treatment training manifest is unreadable: {path}") from exc
    if data.get("experiment") != "smolvla_lora_treatment_training":
        raise ValueError("unexpected treatment training manifest experiment")
    if data.get("base_policy_revision") != os.environ.get(
        "BASE_POLICY_REVISION", "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    ):
        raise ValueError("treatment training base policy revision does not match evaluator contract")
    if data.get("training_variant") != "treatment":
        raise ValueError("training manifest is not for the arrow treatment dataset")
    _validate_resume_audit_chain(data)
    plan = Path(data.get("training_plan", ""))
    if not plan.is_file() or _sha256_file(plan) != data.get("training_plan_sha256"):
        raise ValueError("treatment training plan is missing or has changed")
    pair_manifest = Path(data.get("pair_manifest", ""))
    pair_sentinel = Path(data.get("pair_sentinel", ""))
    for item, key in ((pair_manifest, "pair_manifest_sha256"), (pair_sentinel, "pair_sentinel_sha256")):
        if not item.is_file() or _sha256_file(item) != data.get(key):
            raise ValueError(f"treatment training provenance file is missing or changed: {item}")
    treatment = data.get("treatment_adapter", {})
    if not isinstance(treatment, dict):
        raise ValueError("treatment training manifest lacks the treatment adapter record")
    expected = {
        "base": _checkpoint_path(manifest["base_checkpoint"]),
        "treatment": _checkpoint_path(manifest["treatment_checkpoint"]),
    }
    base_path = Path(data.get("base_policy", "")).expanduser()
    expected_base = Path(expected["base"]).expanduser()
    if base_path.exists() and expected_base.exists() and base_path.resolve() != expected_base.resolve():
        raise ValueError("base checkpoint does not match the treatment training manifest")
    adapter = Path(treatment.get("path", "")).expanduser().resolve()
    if not adapter.is_file() or _sha256_file(adapter) != treatment.get("sha256"):
        raise ValueError(f"treatment adapter is missing or has changed: {adapter}")
    checkpoint = Path(expected["treatment"]).expanduser()
    allowed = {adapter, adapter.parent, adapter.parent.parent}
    if checkpoint.exists() and checkpoint.resolve() not in allowed:
        raise ValueError("treatment checkpoint is not the adapter recorded by training")
    return data


# Keep the old import name as a read-only compatibility alias; it now validates
# a treatment-only manifest and never accepts a control adapter.
validate_paired_training_manifest = validate_training_manifest


def write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> str:
    """Create a manifest once, or verify an existing one is byte-equivalent."""
    _validate_manifest(manifest)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_sha256"] = _hash_manifest(manifest)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing manifest is unreadable: {path}") from exc
        if existing.get("manifest_sha256") != payload["manifest_sha256"] or {
            key: value for key, value in existing.items() if key != "manifest_sha256"
        } != manifest:
            raise ValueError(f"existing manifest does not match requested matrix: {path}")
        return payload["manifest_sha256"]
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return payload["manifest_sha256"]


def validate_existing_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    """Reject stale or incomplete cell outputs before any rollout is launched."""
    _validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if not root.exists():
        return
    expected_dirs = {Path(cell["output_dir"]).resolve() for cell in manifest["cells"]}
    for path in root.rglob("*"):
        if not path.is_dir() or path == root:
            continue
        if path.name.startswith("seed_") and path not in {item.parent for item in expected_dirs}:
            raise ValueError(f"unexpected stale seed output directory: {path}")
    for cell in manifest["cells"]:
        cell_dir = Path(cell["output_dir"]).resolve()
        if not cell_dir.exists():
            continue
        marker = cell_dir / "cell_manifest.json"
        if not marker.exists():
            raise ValueError(f"stale cell output lacks immutable marker: {cell_dir}")
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable cell marker: {marker}") from exc
        expected_cell = {key: cell[key] for key in ("cell_id", "seed", "checkpoint", "live_arrows", "output_dir")}
        if marker_data != expected_cell:
            raise ValueError(f"stale cell marker does not match manifest: {marker}")


def validate_randomization_audit(cell_output: Path, manifest: dict[str, Any]) -> None:
    """Require one complete observed reset record per task episode."""
    audit_path = cell_output / "randomization_audit.jsonl"
    if not audit_path.exists():
        raise ValueError(f"missing randomization audit: {audit_path}")
    records: list[dict[str, Any]] = []
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable randomization audit: {audit_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed randomization audit line {line_number}: {audit_path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"randomization audit line {line_number} is not an object")
        records.append(record)
    expected_tasks = set(manifest["tasks"])
    seen_keys: set[tuple[int, int, int]] = set()
    per_task: dict[int, int] = {task_id: 0 for task_id in expected_tasks}
    for record in records:
        required = {"task_id", "env_index", "reset_sequence", "dimensions_enabled", "dimensions_realized", "details", "status"}
        if not required <= set(record):
            raise ValueError("randomization audit record is incomplete")
        try:
            task_id = int(record["task_id"])
            env_index = int(record["env_index"])
            reset_sequence = int(record["reset_sequence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("randomization audit key fields are invalid") from exc
        key = (task_id, env_index, reset_sequence)
        if task_id not in expected_tasks or env_index < 0 or reset_sequence <= 0:
            raise ValueError(f"randomization audit has invalid reset key: {key}")
        if key in seen_keys:
            raise ValueError(f"duplicate randomization audit reset record: {key}")
        seen_keys.add(key)
        per_task[task_id] += 1
        expected_dimensions = manifest["randomization_dimensions"][str(task_id)]
        if record["dimensions_enabled"] != expected_dimensions:
            raise ValueError(f"randomization audit enabled dimensions mismatch for task {task_id}")
        realized = record["dimensions_realized"]
        if set(realized) != set(expected_dimensions) or any(not isinstance(value, bool) for value in realized.values()):
            raise ValueError(f"randomization audit realized dimensions malformed for task {task_id}")
        if record["status"] != "ok":
            raise ValueError(f"randomization audit reset is not complete: {key}")
        if any(bool(enabled) != bool(realized[name]) for name, enabled in expected_dimensions.items()):
            raise ValueError(f"randomization audit dimensions were not fully realized for task {task_id}")
        details = record["details"]
        if not isinstance(details, dict) or "removed" not in details or "projection" not in details:
            raise ValueError(f"randomization audit lacks observed removal/projection evidence for task {task_id}")
        expected_config = manifest["randomization_config"]
        expected_removed = expected_config["remove"][str(task_id)]
        if details["removed"] != expected_removed:
            raise ValueError(f"randomization audit removal evidence mismatch for task {task_id}")
        projection = details["projection"]
        if not isinstance(projection, dict) or projection.get("success") is not True:
            raise ValueError(f"randomization audit projection evidence failed for task {task_id}")
        if projection.get("required") and projection.get("projected") is not True:
            raise ValueError(f"randomization audit projection was not completed for task {task_id}")
        protected = details.get("protected")
        if protected != {"akita_black_bowl_1": True, "plate_1": True}:
            raise ValueError(f"randomization audit protected-object evidence failed for task {task_id}")
        if expected_dimensions.get("scene_layout"):
            layout = details.get("layout")
            expected_layout = expected_config["layout"][str(task_id)]
            if not isinstance(layout, dict) or layout.get("configured") != expected_layout:
                raise ValueError(f"randomization audit layout configuration mismatch for task {task_id}")
            expected_applied = sorted(label for operation in expected_layout for label in operation)
            if sorted(layout.get("applied", [])) != expected_applied or layout.get("skipped"):
                raise ValueError(f"randomization audit layout application evidence failed for task {task_id}")
        elif "layout" in details:
            raise ValueError(f"no-layout task {task_id} must not claim layout evidence")
    wrong_counts = [task_id for task_id, count in per_task.items() if count != int(manifest["episodes"])]
    if wrong_counts:
        raise ValueError(f"randomization audit must contain exactly {manifest['episodes']} resets per task: {wrong_counts}")


def validate_eval_info(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate one successful cell's raw LeRobot result without rewriting it."""
    if not path.is_file():
        raise ValueError(f"missing eval_info.json: {path}")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable eval_info.json: {path}") from exc
    per_task = info.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != len(TASK_IDS):
        raise ValueError(f"eval_info must contain exactly {len(TASK_IDS)} task records: {path}")
    seen: set[int] = set()
    expected_episodes = int(manifest["episodes"])
    for record in per_task:
        if not isinstance(record, dict) or not isinstance(record.get("task_id"), int):
            raise ValueError(f"eval_info has an invalid task record: {path}")
        task_id = record["task_id"]
        if task_id in seen or task_id not in TASK_IDS:
            raise ValueError(f"eval_info has duplicate or unexpected task {task_id}: {path}")
        seen.add(task_id)
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"eval_info task {task_id} lacks metrics: {path}")
        successes = metrics.get("successes")
        sum_rewards = metrics.get("sum_rewards")
        max_rewards = metrics.get("max_rewards")
        if not all(isinstance(values, list) for values in (successes, sum_rewards, max_rewards)):
            raise ValueError(f"eval_info task {task_id} has incomplete metric arrays: {path}")
        if not all(len(values) == expected_episodes for values in (successes, sum_rewards, max_rewards)):
            raise ValueError(f"eval_info task {task_id} does not have {expected_episodes} episodes: {path}")
        if not all(isinstance(value, bool) for value in successes):
            raise ValueError(f"eval_info task {task_id} has invalid success values: {path}")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for values in (sum_rewards, max_rewards) for value in values):
            raise ValueError(f"eval_info task {task_id} has invalid reward values: {path}")
    if seen != set(TASK_IDS):
        raise ValueError(f"eval_info task IDs are incomplete: {path}")
    overall = info.get("overall")
    if overall is not None:
        if not isinstance(overall, dict) or overall.get("n_episodes") != len(TASK_IDS) * expected_episodes:
            raise ValueError(f"eval_info overall episode count is incomplete: {path}")
        pc_success = overall.get("pc_success")
        if not isinstance(pc_success, (int, float)) or not math.isfinite(float(pc_success)) or not 0 <= float(pc_success) <= 100:
            raise ValueError(f"eval_info overall pc_success is invalid: {path}")
    return info


def _write_cell_marker(cell: dict[str, Any]) -> None:
    cell_dir = Path(cell["output_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    marker = cell_dir / "cell_manifest.json"
    expected = {key: cell[key] for key in ("cell_id", "seed", "checkpoint", "live_arrows", "output_dir")}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"cell marker mismatch: {marker}")
        return
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_summary(cell: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(cell["output_dir"])
    eval_info = output_dir / "eval_info.json"
    pc_success: float | None = None
    episodes: int | None = None
    if eval_info.exists():
        data = json.loads(eval_info.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        if isinstance(overall, dict):
            pc_success = overall.get("pc_success")
            episodes = overall.get("n_episodes")
        if pc_success is None and isinstance(data.get("per_task"), list):
            successes = [
                value
                for record in data["per_task"]
                for value in record.get("metrics", {}).get("successes", [])
            ]
            if successes:
                pc_success = 100.0 * sum(successes) / len(successes)
                episodes = len(successes)
    return {
        "row_type": "cell",
        "cell_id": cell["cell_id"],
        "seed": cell["seed"],
        "task_id": "",
        "checkpoint": cell["checkpoint"],
        "live_arrows": cell["live_arrows"],
        "pc_success": pc_success if pc_success is not None else "",
        "successes": "",
        "episodes": episodes if episodes is not None else "",
        "eval_info": eval_info.as_posix() if eval_info.exists() else "",
        "output_dir": output_dir.as_posix(),
        "contrast": "",
    }


def _extract_task_rows(cell: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_record in info["per_task"]:
        metrics = task_record["metrics"]
        successes = metrics["successes"]
        rows.append({
            "row_type": "task",
            "cell_id": cell["cell_id"],
            "seed": cell["seed"],
            "task_id": task_record["task_id"],
            "checkpoint": cell["checkpoint"],
            "live_arrows": cell["live_arrows"],
            "pc_success": 100.0 * sum(successes) / len(successes),
            "successes": sum(successes),
            "episodes": len(successes),
            "eval_info": str(Path(cell["output_dir"]) / "eval_info.json"),
            "output_dir": cell["output_dir"],
            "contrast": "",
        })
    return rows


def _build_contrast_rows(cell_infos: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-seed/task and across-seed 2x2 contrasts and interactions."""
    rows: list[dict[str, Any]] = []
    values: dict[tuple[int, int, str], float] = {}
    for (seed, cell_id), (cell, info) in cell_infos.items():
        for task_record in info["per_task"]:
            successes = task_record["metrics"]["successes"]
            values[(seed, task_record["task_id"], cell_id)] = 100.0 * sum(successes) / len(successes)

    def add_row(seed, task_id, metric, value):
        rows.append({
            "row_type": "contrast",
            "cell_id": "",
            "seed": seed,
            "task_id": task_id,
            "checkpoint": "",
            "live_arrows": "",
            "pc_success": value,
            "successes": "",
            "episodes": "",
            "eval_info": "",
            "output_dir": manifest["output_root"],
            "contrast": metric,
        })

    metrics = {
        "base_arrow_effect": lambda b0, b1, t0, t1: b1 - b0,
        "treatment_arrow_effect": lambda b0, b1, t0, t1: t1 - t0,
        "treatment_effect_no_arrows": lambda b0, b1, t0, t1: t0 - b0,
        "treatment_effect_live_arrows": lambda b0, b1, t0, t1: t1 - b1,
        "arrow_treatment_interaction": lambda b0, b1, t0, t1: (t1 - t0) - (b1 - b0),
    }
    aggregate: dict[tuple[int, str], list[float]] = {}
    for seed in manifest["seeds"]:
        for task_id in TASK_IDS:
            b0 = values[(seed, task_id, "base_no_arrows")]
            b1 = values[(seed, task_id, "base_live_arrows")]
            t0 = values[(seed, task_id, "treatment_no_arrows")]
            t1 = values[(seed, task_id, "treatment_live_arrows")]
            for metric, calculate in metrics.items():
                value = calculate(b0, b1, t0, t1)
                add_row(seed, task_id, metric, value)
                aggregate.setdefault((task_id, metric), []).append(value)
    for (task_id, metric), metric_values in sorted(aggregate.items()):
        add_row("aggregate", task_id, metric, sum(metric_values) / len(metric_values))
    for metric in metrics:
        metric_values = [
            row["pc_success"] for row in rows
            if row["seed"] == "aggregate" and row["contrast"] == metric
        ]
        add_row("aggregate", "all", metric, sum(metric_values) / len(metric_values))
    return rows


def _run_cell(cell: dict[str, Any], args: argparse.Namespace, script_dir: Path) -> int:
    _write_cell_marker(cell)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODELS": cell["checkpoint"],
            "TASK_IDS": json.dumps(list(TASK_IDS), separators=(",", ":")),
            "N_EPISODES": str(args.episodes),
            "BATCH_SIZE": str(args.batch_size),
            "SEED": str(cell["seed"]),
            "DEVICE": args.device,
            "CONTEXT_MODE": "standard",
            "CONTEXT_FORMAT": "standard",
            "VISUAL_CONDITION": "visual_arrows" if cell["live_arrows"] else "none",
            "N_ACTION_STEPS": "checkpoint",
            "VISUAL_ARROW_WIDTH": "1",
            "VISUAL_ARROW_HEAD_LENGTH": "16",
            "MAX_EPISODES_RENDERED": str(args.max_videos if args.videos else 0),
            "RENDER_MODE": "rgb_array" if args.videos else "none",
            "RANDOMIZE_SCENES": "1",
        }
    )
    cmd = [
        sys.executable,
        str(script_dir / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false",
        f"--output_dir={cell['output_dir']}",
        f"--policy.path={cell['checkpoint']}",
        "--env.task_ids=" + json.dumps(list(TASK_IDS), separators=(",", ":")),
        "--env.camera_name=" + TRAINING_CAMERAS,
        "--env.observation_height=256",
        "--env.observation_width=256",
    ]
    log_path = Path(cell["output_dir"]) / "eval_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=script_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process.returncode


def _write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = ["row_type", "cell_id", "seed", "task_id", "checkpoint", "live_arrows", "pc_success", "successes", "episodes", "eval_info", "output_dir", "contrast"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True,
                        help="frozen smolvla_libero checkpoint used as the control policy")
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated or JSON integer seeds")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--training-manifest",
        default=os.environ.get("TRAINING_MANIFEST"),
        help="immutable treatment training manifest produced by launch_lora_treatment.sh",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-videos", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform.startswith("linux"):
        # LIBERO/robosuite runs without an X server on Lambda GPU hosts.
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    seeds = parse_int_list(args.seeds)
    if not args.training_manifest:
        print("ERROR: --training-manifest is required for sealed base-vs-treatment evaluation")
        return 2
    manifest = build_manifest(
        base_checkpoint=args.base_checkpoint,
        treatment_checkpoint=args.treatment_checkpoint,
        seeds=seeds,
        episodes=args.episodes,
        batch_size=args.batch_size,
        device=args.device,
        videos=args.videos,
        max_videos=args.max_videos,
        training_manifest=args.training_manifest,
        output_root=output_root,
    )
    manifest_path = output_root / MANIFEST_FILENAME
    validate_existing_outputs(output_root, manifest)
    write_immutable_manifest(manifest_path, manifest)
    try:
        validate_training_manifest(Path(args.training_manifest).expanduser().resolve(), manifest)
    except ValueError as exc:
        print(f"ERROR: treatment training provenance validation failed: {exc}")
        return 1

    script_dir = Path(__file__).resolve().parent
    rows: list[dict[str, Any]] = []
    cell_infos: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for cell in manifest["cells"]:
        return_code = _run_cell(cell, args, script_dir)
        eval_path = Path(cell["output_dir"]) / "eval_info.json"
        if return_code != 0:
            rows.append(_extract_summary(cell))
            _write_summary(output_root / SUMMARY_FILENAME, rows)
            print(f"ERROR: cell {cell['cell_id']} seed {cell['seed']} failed ({return_code})")
            return return_code
        try:
            info = validate_eval_info(eval_path, manifest)
        except ValueError as exc:
            rows.append(_extract_summary(cell))
            _write_summary(output_root / SUMMARY_FILENAME, rows)
            print(f"ERROR: eval_info validation failed for {cell['cell_id']} seed {cell['seed']}: {exc}")
            return 1
        cell_infos[(cell["seed"], cell["cell_id"])] = (cell, info)
        rows.append(_extract_summary(cell))
        rows.extend(_extract_task_rows(cell, info))
        try:
            validate_randomization_audit(Path(cell["output_dir"]), manifest)
        except ValueError as exc:
            _write_summary(output_root / SUMMARY_FILENAME, rows)
            print(f"ERROR: randomization audit failed for {cell['cell_id']} seed {cell['seed']}: {exc}")
            return 1
    rows.extend(_build_contrast_rows(cell_infos, manifest))
    _write_summary(output_root / SUMMARY_FILENAME, rows)
    print(f"2x2 base-vs-treatment evaluation complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
