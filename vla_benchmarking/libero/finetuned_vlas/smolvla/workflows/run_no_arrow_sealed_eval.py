#!/usr/bin/env python3
"""Evaluate the sealed no-arrow-trained SmolVLA policy without visual arrows.

This is a single-condition expanded regression evaluation.  It intentionally
does not modify or reuse the historical paired arrow/no-arrow runner: the
historical 100-cell contract remains frozen, while this runner supports a
one-episode smoke protocol and a 50-episode-per-task full protocol.
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

# Bootstrap direct script launches before importing the organized package.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import (
    TASK_IDS,
    validate_eval_info,
    validate_randomization_audit,
)
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_lora_no_arrow_pair_eval import (
    DEFAULT_PROFILE,
    SEALED_CHECKPOINT_ID,
    SEALED_PEFT_R,
    SEALED_REVISION,
    SEALED_SAVE_FREQ,
    SEALED_SEED,
    SEALED_STEPS,
    _adapter_directory,
    _canonical_json,
    _sha256_file,
    validate_training_manifest,
)
from vla_benchmarking.libero.evaluation.randomization_contract import randomization_config_payload
from vla_benchmarking.libero.evaluation.plan import (
    build_evaluation_plan,
    shared_source_hashes,
    validate_native_schedule,
    validate_plan,
)
from vla_benchmarking.libero.shared.config import (
    DEFAULT_CAMERAS,
    LEROBOT_CAMERA_KEYS,
    RANDOMIZATION_DIMENSIONS,
    task_randomization_dimensions,
)


TRAINING_CAMERAS = ",".join(LEROBOT_CAMERA_KEYS)
RAW_TRAINING_CAMERAS = ",".join(DEFAULT_CAMERAS)
TRAINING_EXPERIMENT = DEFAULT_PROFILE.experiment
CELL_ID = "no_arrow_trained_no_arrows"
MANIFEST_FILENAME = "no_arrow_trained_sealed_randomized_manifest.json"
SUMMARY_FILENAME = "no_arrow_trained_sealed_randomized_summary.csv"
SCHEMA_VERSION = 1
EVAL_EXPERIMENT = "smolvla_lora_no_arrow_trained_sealed_randomized"
PROTOCOL_EPISODES = {"smoke": 1, "full": 50}
PROTOCOL_TASKS = {"smoke": (0, 4), "full": tuple(TASK_IDS)}
TRAINING_VARIANT = "no_arrow_treatment"
DATASET_VARIANT = "control"


def _source_commit() -> str:
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot resolve evaluation source commit") from exc
    expected = os.environ.get("NO_ARROW_EVAL_EXPECTED_COMMIT", "").strip().lower()
    if expected and expected != observed:
        raise ValueError(f"evaluation source commit differs: {observed}")
    if len(observed) != 40 or any(character not in "0123456789abcdef" for character in observed):
        raise ValueError("evaluation source commit is not a full Git SHA")
    return observed


def _hash_manifest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _cell(adapter_checkpoint: str, output_root: Path, tasks: Sequence[int]) -> dict[str, Any]:
    adapter = _adapter_directory(adapter_checkpoint)
    return {
        "cell_id": CELL_ID,
        "seed": SEALED_SEED,
        "checkpoint": adapter,
        "live_arrows": False,
        "tasks": [int(task_id) for task_id in tasks],
        "output_dir": (output_root / f"seed_{SEALED_SEED}" / CELL_ID).as_posix(),
        "adapter_sha256": _sha256_file(Path(adapter) / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256_file(Path(adapter) / "adapter_config.json"),
        "train_config_sha256": _sha256_file(Path(adapter) / "train_config.json"),
    }


def _native_schedule(tasks: Sequence[int], episodes: int) -> list[dict[str, int]]:
    """Describe the cells the native LeRobot loop will execute."""
    cells: list[dict[str, int]] = []
    for task_id in tasks:
        for episode_index in range(int(episodes)):
            cells.append({
                "cell_index": len(cells),
                "task_id": int(task_id),
                "episode_index": int(episode_index),
                "seed": SEALED_SEED + int(episode_index),
                "init_state_index": int(episode_index),
            })
    return cells


def _training_pair_provenance(training_path: Path) -> dict[str, Any]:
    try:
        training = json.loads(training_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("no-arrow training manifest is unreadable") from exc
    pair_manifest = Path(training.get("pair_manifest", "")).expanduser().resolve()
    pair_sentinel = Path(training.get("pair_sentinel", "")).expanduser().resolve()
    if not pair_manifest.is_file() or not pair_sentinel.is_file():
        raise ValueError("no-arrow training pair provenance files are missing")
    expected_manifest_hash = training.get("pair_manifest_sha256")
    observed_manifest_hash = _sha256_file(pair_manifest)
    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError("sealed no-arrow pair manifest has changed")
    try:
        sentinel = json.loads(pair_sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("no-arrow training pair sentinel is unreadable") from exc
    if (
        sentinel.get("pair_kind") != "sealed_lora_control_treatment"
        or sentinel.get("full_experiment_ready") is not True
        or sentinel.get("launch_eligibility") != "full_experiment_ready"
        or sentinel.get("manifest_sha256") != expected_manifest_hash
    ):
        raise ValueError("current no-arrow pair sentinel cannot revalidate the sealed pair")
    expected_sentinel_hash = training.get("pair_sentinel_sha256")
    observed_sentinel_hash = _sha256_file(pair_sentinel)
    return {
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": observed_manifest_hash,
        "pair_sentinel": str(pair_sentinel),
        "pair_sentinel_training_sha256": expected_sentinel_hash,
        "pair_sentinel_observed_sha256": observed_sentinel_hash,
        "pair_sentinel_status": (
            "original_hash_verified"
            if observed_sentinel_hash == expected_sentinel_hash
            else "semantic_revalidation_after_file_drift"
        ),
    }


def build_manifest(
    *,
    adapter_checkpoint: str,
    training_manifest: str,
    output_root: Path,
    protocol: str,
    episodes: int,
    device: str = "cuda",
    videos: bool = False,
    max_videos: int = 0,
    seed: int = SEALED_SEED,
    tasks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build and validate the immutable one-cell evaluation contract."""
    expected_tasks = list(PROTOCOL_TASKS[protocol]) if protocol in PROTOCOL_TASKS else list(TASK_IDS)
    task_ids = expected_tasks if tasks is None else [int(task_id) for task_id in tasks]
    if task_ids != expected_tasks:
        raise ValueError(
            f"protocol {protocol!r} requires task IDs exactly {expected_tasks}"
        )
    if int(seed) != SEALED_SEED:
        raise ValueError("sealed no-arrow eval requires base seed exactly 1000")
    if protocol not in PROTOCOL_EPISODES:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOL_EPISODES)}")
    if int(episodes) != PROTOCOL_EPISODES[protocol]:
        raise ValueError(
            f"protocol {protocol!r} requires exactly {PROTOCOL_EPISODES[protocol]} episodes"
        )
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")

    root = output_root.expanduser().resolve()
    training_path = Path(training_manifest).expanduser().resolve()
    if not training_path.is_file():
        raise ValueError(f"no-arrow training manifest is missing: {training_path}")
    dimensions = {
        str(task_id): task_randomization_dimensions(task_id) for task_id in task_ids
    }
    incomplete = [
        task_id for task_id, values in dimensions.items() if not values.get("object_removal")
    ]
    if incomplete:
        raise ValueError(
            "sealed all-task eval requires object_removal for every task; "
            f"incomplete tasks: {incomplete}"
        )
    randomization_config = randomization_config_payload()
    cell = _cell(adapter_checkpoint, root, task_ids)
    shared_plan = build_evaluation_plan(
        policy_kind="smolvla_no_arrow",
        suite_mode="sealed_randomized",
        task_ids=task_ids,
        episodes_per_task=int(episodes),
        seed_base=SEALED_SEED,
        camera=TRAINING_CAMERAS,
        resolution=256,
        text_context="none",
        visual_input="none",
        source_hashes=shared_source_hashes(),
    )
    validate_native_schedule(shared_plan, _native_schedule(task_ids, episodes))
    pair_provenance = _training_pair_provenance(training_path)
    adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EVAL_EXPERIMENT,
        "protocol": protocol,
        "model_role": "no_arrow_trained_lora",
        "trained_on_visual_condition": "no_arrows",
        "evaluation_visual_condition": "none",
        "training_experiment": TRAINING_EXPERIMENT,
        "training_variant": TRAINING_VARIANT,
        "dataset_variant": DATASET_VARIANT,
        "base_policy_revision": SEALED_REVISION,
        "checkpoint_step": int(SEALED_CHECKPOINT_ID),
        "adapter_checkpoint": _adapter_directory(adapter_checkpoint),
        "adapter_sha256": _sha256_file(
            adapter_path / "adapter_model.safetensors"
        ),
        "adapter_config_sha256": _sha256_file(adapter_path / "adapter_config.json"),
        "train_config_sha256": _sha256_file(adapter_path / "train_config.json"),
        "training_manifest": str(training_path),
        "training_manifest_sha256": _sha256_file(training_path),
        "tasks": task_ids,
        "seed": SEALED_SEED,
        "seed_base": SEALED_SEED,
        "episode_seed_policy": "seed=seed_base+episode_index",
        "episode_seeds": list(range(SEALED_SEED, SEALED_SEED + int(episodes))),
        "episodes": int(episodes),
        "planned_episodes": len(task_ids) * int(episodes),
        "batch_size": 1,
        "device": device,
        "randomize_scenes": True,
        "camera_name": TRAINING_CAMERAS,
        "raw_camera_names": RAW_TRAINING_CAMERAS,
        "observation_height": 256,
        "observation_width": 256,
        "text_context_mode": "standard",
        "text_context_format": "standard",
        "visual_condition": "none",
        "n_action_steps": "checkpoint",
        "videos": bool(videos),
        "max_videos": int(max_videos if videos else 0),
        "randomization_dimensions": dimensions,
        "randomization_dimension_names": list(RANDOMIZATION_DIMENSIONS),
        "randomization_config": randomization_config,
        "randomization_config_sha256": hashlib.sha256(
            _canonical_json(randomization_config).encode("utf-8")
        ).hexdigest(),
        "output_root": root.as_posix(),
        "contrast": "none_single_condition",
        "cells": [cell],
        "shared_plan": shared_plan,
        "shared_plan_schema": shared_plan["schema"],
        "shared_plan_hash": shared_plan["sha256"],
        "shared_schedule_hash": shared_plan["schedule"]["sha256"],
        "shared_source_hashes": dict(shared_plan["source_hashes"]),
        "provenance": {
            "source_commit": _source_commit(),
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "training_job_id": "1910197",
            "historical_evaluation_job_id": "1910198",
            "historical_no_arrow_result": "43/100",
            "checkpoint_step": int(SEALED_CHECKPOINT_ID),
            "training_steps": SEALED_STEPS,
            "training_save_freq": SEALED_SAVE_FREQ,
            "training_batch_size": 32,
            "training_seed": SEALED_SEED,
            "peft_rank": SEALED_PEFT_R,
            "evaluation_seed": SEALED_SEED,
            "evaluation_source": "vla_benchmarking/libero/evaluation/run_lerobot_eval_with_context.py",
            **pair_provenance,
        },
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing sealed no-arrow manifest schema_version")
    if manifest.get("experiment") != EVAL_EXPERIMENT:
        raise ValueError("unexpected sealed no-arrow evaluation experiment")
    if manifest.get("protocol") not in PROTOCOL_EPISODES:
        raise ValueError("manifest protocol is invalid")
    if manifest.get("episodes") != PROTOCOL_EPISODES[manifest["protocol"]]:
        raise ValueError("manifest episode count does not match protocol")
    expected_tasks = list(PROTOCOL_TASKS[manifest["protocol"]])
    if manifest.get("planned_episodes") != len(expected_tasks) * manifest["episodes"]:
        raise ValueError("manifest planned episode count does not match the task schedule")
    if manifest.get("model_role") != "no_arrow_trained_lora":
        raise ValueError("manifest model role is not the no-arrow trained LoRA")
    if manifest.get("trained_on_visual_condition") != "no_arrows":
        raise ValueError("manifest training condition is not no_arrows")
    if manifest.get("evaluation_visual_condition") != "none" or manifest.get("visual_condition") != "none":
        raise ValueError("sealed no-arrow eval must use VISUAL_CONDITION=none")
    if manifest.get("training_variant") != TRAINING_VARIANT or manifest.get("dataset_variant") != DATASET_VARIANT:
        raise ValueError("manifest training/dataset lineage is invalid")
    if manifest.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("manifest base policy revision is not sealed")
    if manifest.get("checkpoint_step") != int(SEALED_CHECKPOINT_ID):
        raise ValueError("manifest must use final checkpoint 029190")
    if manifest.get("tasks") != expected_tasks:
        raise ValueError(f"manifest tasks must match protocol schedule: {expected_tasks}")
    if manifest.get("seed") != SEALED_SEED:
        raise ValueError("manifest seed must be exactly 1000")
    if manifest.get("seed_base") != SEALED_SEED:
        raise ValueError("manifest seed_base must be exactly 1000")
    if manifest.get("episode_seed_policy") != "seed=seed_base+episode_index":
        raise ValueError("manifest episode seed policy is not canonical")
    expected_episode_seeds = list(
        range(SEALED_SEED, SEALED_SEED + int(manifest.get("episodes", 0)))
    )
    if manifest.get("episode_seeds") != expected_episode_seeds:
        raise ValueError("manifest episode seeds do not match the sealed seed schedule")
    if manifest.get("batch_size") != 1:
        raise ValueError("sealed randomization audit requires batch_size=1")
    if manifest.get("randomize_scenes") is not True:
        raise ValueError("manifest must keep RANDOMIZE_SCENES enabled")
    if manifest.get("camera_name") != TRAINING_CAMERAS or manifest.get("raw_camera_names") != RAW_TRAINING_CAMERAS:
        raise ValueError("manifest camera contract does not match training cameras")
    if manifest.get("observation_height") != 256 or manifest.get("observation_width") != 256:
        raise ValueError("manifest must use 256x256 observations")
    if manifest.get("text_context_mode") != "standard" or manifest.get("text_context_format") != "standard":
        raise ValueError("sealed eval must use standard context")
    if manifest.get("n_action_steps") != "checkpoint":
        raise ValueError("sealed eval must preserve checkpoint n_action_steps")
    if manifest.get("contrast") != "none_single_condition":
        raise ValueError("sealed no-arrow eval must be a single condition")
    shared_plan = manifest.get("shared_plan")
    if not isinstance(shared_plan, dict):
        raise ValueError("manifest shared evaluation plan is missing")
    validate_plan(shared_plan)
    if shared_plan.get("policy_kind") != "smolvla_no_arrow":
        raise ValueError("manifest shared plan policy kind is invalid")
    if shared_plan.get("condition", {}).get("visual_input") != "none":
        raise ValueError("manifest shared plan must disable visual input")
    if shared_plan.get("text_contract") != "standard_no_extra_text":
        raise ValueError("manifest shared plan text contract is invalid")
    if shared_plan.get("prompt_applicability") != "applied":
        raise ValueError("sealed no-arrow plan must apply the task prompt")
    if manifest.get("shared_plan_schema") != shared_plan.get("schema"):
        raise ValueError("manifest shared plan schema does not match")
    if manifest.get("shared_plan_hash") != shared_plan.get("sha256"):
        raise ValueError("manifest shared plan hash does not match")
    if manifest.get("shared_schedule_hash") != shared_plan.get("schedule", {}).get("sha256"):
        raise ValueError("manifest shared schedule hash does not match")
    if manifest.get("shared_source_hashes") != shared_plan.get("source_hashes"):
        raise ValueError("manifest shared source hashes do not match")
    validate_native_schedule(shared_plan, _native_schedule(expected_tasks, manifest["episodes"]))
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("manifest provenance is missing")
    source_commit = provenance.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("manifest source commit is invalid")
    if provenance.get("training_job_id") != "1910197" or provenance.get("historical_evaluation_job_id") != "1910198":
        raise ValueError("manifest historical job provenance is invalid")
    for key in ("pair_manifest_sha256", "pair_sentinel_training_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"manifest {key} provenance is invalid")
    if provenance.get("pair_sentinel_status") not in {
        "original_hash_verified", "semantic_revalidation_after_file_drift"
    }:
        raise ValueError("manifest pair-sentinel revalidation status is invalid")
    observed_sentinel_hash = provenance.get("pair_sentinel_observed_sha256")
    if not isinstance(observed_sentinel_hash, str) or len(observed_sentinel_hash) != 64:
        raise ValueError("manifest observed pair-sentinel hash is invalid")
    adapter = manifest.get("adapter_checkpoint")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError("manifest adapter checkpoint is missing")
    if not isinstance(manifest.get("adapter_sha256"), str) or not isinstance(manifest.get("training_manifest_sha256"), str):
        raise ValueError("manifest lacks sealed artifact hashes")
    if not isinstance(manifest.get("adapter_config_sha256"), str) or not isinstance(manifest.get("train_config_sha256"), str):
        raise ValueError("manifest lacks checkpoint configuration hashes")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise ValueError("manifest must contain exactly one cell")
    cell = cells[0]
    expected = {
        "cell_id": CELL_ID,
        "seed": SEALED_SEED,
        "checkpoint": adapter,
        "live_arrows": False,
        "tasks": expected_tasks,
        "adapter_sha256": manifest["adapter_sha256"],
        "adapter_config_sha256": manifest["adapter_config_sha256"],
        "train_config_sha256": manifest["train_config_sha256"],
    }
    for key, value in expected.items():
        if cell.get(key) != value:
            raise ValueError(f"manifest cell {key} does not match the sealed condition")
    expected_config = randomization_config_payload()
    if manifest.get("randomization_config") != expected_config:
        raise ValueError("manifest randomization config does not match sealed config")
    expected_config_hash = hashlib.sha256(
        _canonical_json(expected_config).encode("utf-8")
    ).hexdigest()
    if manifest.get("randomization_config_sha256") != expected_config_hash:
        raise ValueError("manifest randomization config hash does not match sealed config")
    dimensions = manifest.get("randomization_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != {str(task_id) for task_id in expected_tasks}:
        raise ValueError("manifest must record randomization dimensions for every task")
    for task_id, values in dimensions.items():
        if not isinstance(values, dict) or not values.get("object_removal"):
            raise ValueError(f"task {task_id} must enable object_removal")


def write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> str:
    _validate_manifest(manifest)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_sha256"] = _hash_manifest(manifest)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != payload["manifest_sha256"] or {
            key: value for key, value in existing.items() if key != "manifest_sha256"
        } != manifest:
            raise ValueError(f"existing manifest does not match requested sealed eval: {path}")
        return payload["manifest_sha256"]
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return payload["manifest_sha256"]


def validate_existing_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if not root.exists():
        return
    expected_seed = root / f"seed_{SEALED_SEED}"
    expected_cell = expected_seed / CELL_ID
    for path in root.iterdir():
        if path.is_dir() and path.resolve() != expected_seed.resolve():
            raise ValueError(f"unexpected stale output directory: {path}")
    if not expected_seed.is_dir():
        return
    for path in expected_seed.iterdir():
        if path.is_dir() and path.resolve() != expected_cell.resolve():
            raise ValueError(f"unexpected stale cell output directory: {path}")
    if expected_cell.is_dir():
        marker = expected_cell / "cell_manifest.json"
        if not marker.is_file():
            raise ValueError(f"stale cell output lacks immutable marker: {expected_cell}")
        actual = json.loads(marker.read_text(encoding="utf-8"))
        expected = {key: manifest["cells"][0][key] for key in ("cell_id", "seed", "checkpoint", "live_arrows", "tasks", "output_dir")}
        if actual != expected:
            raise ValueError(f"stale cell marker does not match manifest: {marker}")


def _write_cell_marker(cell: dict[str, Any]) -> None:
    cell_dir = Path(cell["output_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    marker = cell_dir / "cell_manifest.json"
    expected = {key: cell[key] for key in ("cell_id", "seed", "checkpoint", "live_arrows", "tasks", "output_dir")}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"cell marker mismatch: {marker}")
        return
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cell(cell: dict[str, Any], args: argparse.Namespace) -> int:
    _write_cell_marker(cell)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODELS": cell["checkpoint"],
            "TASK_IDS": json.dumps(list(cell["tasks"]), separators=(",", ":")),
            "N_EPISODES": str(args.episodes),
            "BATCH_SIZE": "1",
            "SEED": str(SEALED_SEED),
            "DEVICE": args.device,
            "CONTEXT_MODE": "standard",
            "CONTEXT_FORMAT": "standard",
            "VISUAL_CONDITION": "none",
            "VISUAL_ARROWS": "0",
            "TRAINING_PROFILE": "no_arrow_treatment",
            "PROFILE": "no_arrow_treatment",
            "N_ACTION_STEPS": "checkpoint",
            "MAX_EPISODES_RENDERED": "0",
            "RENDER_MODE": "none",
            "RANDOMIZE_SCENES": "1",
        }
    )
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "vla_benchmarking" / "libero" / "evaluation" / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false",
        f"--output_dir={cell['output_dir']}",
        f"--policy.path={cell['checkpoint']}",
        "--env.task_ids=" + json.dumps(list(cell["tasks"]), separators=(",", ":")),
        "--env.camera_name=" + TRAINING_CAMERAS,
        "--env.observation_height=256",
        "--env.observation_width=256",
    ]
    log_path = Path(cell["output_dir"]) / "eval_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process.returncode


def _extract_task_rows(cell: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_record in info["per_task"]:
        successes = task_record["metrics"]["successes"]
        rows.append(
            {
                "row_type": "task",
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "task_id": task_record["task_id"],
                "checkpoint": cell["checkpoint"],
                "live_arrows": False,
                "pc_success": 100.0 * sum(successes) / len(successes),
                "successes": sum(successes),
                "episodes": len(successes),
                "eval_info": str(Path(cell["output_dir"]) / "eval_info.json"),
                "output_dir": cell["output_dir"],
                "contrast": "",
            }
        )
    return rows


def _extract_summary(cell: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
    eval_info = Path(cell["output_dir"]) / "eval_info.json"
    info = info or {}
    overall = info.get("overall", {}) if isinstance(info, dict) else {}
    return {
        "row_type": "overall",
        "cell_id": cell["cell_id"],
        "seed": cell["seed"],
        "task_id": "all",
        "checkpoint": cell["checkpoint"],
        "live_arrows": False,
        "pc_success": overall.get("pc_success", "") if isinstance(overall, dict) else "",
        "successes": sum(
            sum(record.get("metrics", {}).get("successes", []))
            for record in info.get("per_task", [])
        ) if isinstance(info, dict) else "",
        "episodes": overall.get("n_episodes", "") if isinstance(overall, dict) else "",
        "eval_info": eval_info.as_posix() if eval_info.exists() else "",
        "output_dir": cell["output_dir"],
        "contrast": "",
    }


def _write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = [
        "row_type", "cell_id", "seed", "task_id", "checkpoint", "live_arrows", "pc_success",
        "successes", "episodes", "eval_info", "output_dir", "contrast",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, choices=sorted(set(PROTOCOL_EPISODES.values())), required=True)
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_EPISODES), required=True)
    parser.add_argument("--task-ids", default=None, help="internal protocol-locked task subset")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--no-videos", action="store_true", help="do not render or save episode videos")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args(argv)
    if not args.no_videos:
        print("ERROR: sealed no-arrow evaluation requires --no-videos")
        return 1
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        manifest = build_manifest(
            adapter_checkpoint=args.adapter_checkpoint,
            training_manifest=args.training_manifest,
            output_root=output_root,
            protocol=args.protocol,
            episodes=args.episodes,
            device=args.device,
            videos=not args.no_videos,
            tasks=(
                [int(value) for value in args.task_ids.split(",") if value.strip()]
                if args.task_ids is not None else None
            ),
        )
        validate_existing_outputs(output_root, manifest)
        validate_training_manifest(
            Path(args.training_manifest).expanduser().resolve(),
            manifest,
            allow_revalidated_pair_sentinel=True,
        )
        write_immutable_manifest(output_root / MANIFEST_FILENAME, manifest)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: sealed no-arrow evaluation preflight failed: {exc}")
        return 1

    cell = manifest["cells"][0]
    return_code = _run_cell(cell, args)
    if return_code != 0:
        _write_summary(output_root / SUMMARY_FILENAME, [_extract_summary(cell)])
        print(f"ERROR: no-arrow cell failed ({return_code})")
        return return_code
    try:
        info = validate_eval_info(Path(cell["output_dir"]) / "eval_info.json", manifest)
        validate_randomization_audit(Path(cell["output_dir"]), manifest)
    except ValueError as exc:
        _write_summary(output_root / SUMMARY_FILENAME, [_extract_summary(cell)])
        print(f"ERROR: sealed no-arrow cell validation failed: {exc}")
        return 1
    rows = [_extract_summary(cell, info), *_extract_task_rows(cell, info)]
    _write_summary(output_root / SUMMARY_FILENAME, rows)
    print(f"sealed no-arrow {args.protocol} evaluation complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
