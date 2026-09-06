#!/usr/bin/env python3
"""Run the fixed three-cell SmolVLA evaluation matrix sequentially.

The matrix deliberately keeps the vanilla base, sealed-randomized base, and
vanilla no-arrow fine-tuned adapter as separate immutable cells.  A smoke run
uses tasks 0 and 4 with one episode per task; a full run uses all ten tasks
with ten episodes per task.  The fine-tuned cell is validated through the
existing sealed no-arrow training contract before any evaluator subprocess is
started.
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

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vla_benchmarking.libero.evaluation.contracts import parse_suite_mode
from vla_benchmarking.libero.evaluation.plan import (
    build_evaluation_plan,
    shared_source_hashes,
    validate_plan,
)
from vla_benchmarking.libero.evaluation.randomization_contract import (
    randomization_config_payload,
)
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import (
    TASK_IDS,
    validate_eval_info,
    validate_randomization_audit,
)
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_lora_no_arrow_pair_eval import (
    DEFAULT_PROFILE,
    SEALED_CHECKPOINT_ID,
    SEALED_REVISION,
    SEALED_SAVE_FREQ,
    SEALED_SEED,
    SEALED_STEPS,
    TRAINING_CAMERAS,
    _adapter_directory,
    _canonical_json,
    _sha256_file,
    get_profile,
    validate_training_manifest,
)
from vla_benchmarking.libero.shared.config import (
    DEFAULT_CAMERAS,
    LEROBOT_CAMERA_KEYS,
    RANDOMIZATION_DIMENSIONS,
    task_randomization_dimensions,
)


SCHEMA_VERSION = 1
BASE_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
CELL_IDS = (
    "smolvla_base_vanilla",
    "smolvla_base_sealed_randomized",
    "smolvla_no_arrow_ft_vanilla",
)
CELL_POLICY_KINDS = {
    "smolvla_base_vanilla": "smolvla_base",
    "smolvla_base_sealed_randomized": "smolvla_base",
    "smolvla_no_arrow_ft_vanilla": "smolvla_no_arrow",
}
PROTOCOL_TASKS = {"smoke": (0, 4), "full": tuple(TASK_IDS)}
PROTOCOL_EPISODES = {"smoke": 1, "full": 10}
MANIFEST_FILENAME = "smolvla_eval_matrix_manifest.json"
SCHEDULE_FILENAME = "smolvla_eval_matrix_schedule.json"
SUMMARY_FILENAME = "smolvla_eval_matrix_summary.csv"
SCHEDULE_SCHEMA = "smolvla_eval_matrix_schedule.v1"
EVAL_EXPERIMENT = "smolvla_base_and_no_arrow_ft_matrix"


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_commit() -> str:
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot resolve evaluation source commit") from exc
    expected = os.environ.get("SMOLVLA_MATRIX_EXPECTED_COMMIT", "").strip().lower()
    if expected and observed != expected:
        raise ValueError(f"evaluation source commit differs: {observed}")
    if len(observed) != 40 or any(char not in "0123456789abcdef" for char in observed):
        raise ValueError("evaluation source commit is not a full Git SHA")
    return observed


def _resolve_training_lineage(
    *, adapter_checkpoint: str, training_manifest: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate the existing no-arrow contract and return its base snapshot."""
    training_path = Path(training_manifest).expanduser().resolve()
    if not training_path.is_file():
        raise ValueError(f"no-arrow training manifest is missing: {training_path}")
    adapter = Path(_adapter_directory(adapter_checkpoint)).expanduser().resolve()
    adapter_sha256 = _sha256_file(adapter / "adapter_model.safetensors")
    validation_manifest = {
        "profile": DEFAULT_PROFILE.name,
        "evaluation_scope": "full",
        "training_manifest_sha256": _sha256_file(training_path),
        "adapter_checkpoint": str(adapter),
        "adapter_sha256": adapter_sha256,
        "checkpoint_step": int(SEALED_CHECKPOINT_ID),
        "training_steps": int(SEALED_STEPS),
        "training_save_freq": int(SEALED_SAVE_FREQ),
    }
    validated = validate_training_manifest(
        training_path,
        validation_manifest,
        profile=DEFAULT_PROFILE,
    )
    base_policy = Path(validated["base_policy"]).expanduser().resolve()
    if not base_policy.is_dir():
        raise ValueError(f"validated base policy snapshot is missing: {base_policy}")
    return training_path, base_policy, validated


def _cell(
    *, cell_id: str, checkpoint: str, output_root: Path, tasks: Sequence[int],
    episodes: int, adapter_sha256: str | None, suite_mode: str,
    training_manifest: str | None, base_policy: str,
) -> dict[str, Any]:
    if cell_id not in CELL_IDS:
        raise ValueError(f"unsupported matrix cell: {cell_id}")
    output_dir = output_root / f"seed_{SEALED_SEED}" / cell_id
    return {
        "cell_id": cell_id,
        "policy_kind": CELL_POLICY_KINDS[cell_id],
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "base_model_id": BASE_MODEL_ID,
        "base_policy": base_policy,
        "base_policy_revision": SEALED_REVISION,
        "fine_tuned": cell_id == "smolvla_no_arrow_ft_vanilla",
        "training_manifest": training_manifest,
        "adapter_sha256": adapter_sha256,
        "suite_mode": parse_suite_mode(suite_mode),
        "tasks": [int(task_id) for task_id in tasks],
        "episodes": int(episodes),
        "seed": SEALED_SEED,
        "seed_base": SEALED_SEED,
        "episode_seed_policy": "seed=seed_base+episode_index",
        "output_dir": output_dir.as_posix(),
    }


def _schedule_manifest(*, protocol: str, tasks: Sequence[int], episodes: int) -> dict[str, Any]:
    cells: list[dict[str, int]] = []
    for cell_id in CELL_IDS:
        suite_mode = "sealed_randomized" if cell_id.endswith("sealed_randomized") else "vanilla"
        for task_id in tasks:
            for episode_index in range(int(episodes)):
                cells.append({
                    "cell_index": len(cells),
                    "cell_id": cell_id,
                    "task_id": int(task_id),
                    "episode_index": int(episode_index),
                    "seed": SEALED_SEED + int(episode_index),
                    "init_state_index": int(episode_index),
                    "suite_mode": suite_mode,
                })
    body = {
        "schema": SCHEDULE_SCHEMA,
        "protocol": protocol,
        "tasks": [int(task_id) for task_id in tasks],
        "episodes": int(episodes),
        "seed_base": SEALED_SEED,
        "cells": cells,
    }
    return {**body, "schedule_sha256": _hash_payload(body)}


def build_manifest(
    *, adapter_checkpoint: str, training_manifest: str, output_root: Path,
    protocol: str, base_checkpoint: str | None = None, device: str = "cuda",
    videos: bool = False, max_videos: int = 0,
) -> dict[str, Any]:
    """Build and validate the immutable three-cell matrix contract."""
    if protocol not in PROTOCOL_TASKS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOL_TASKS)}")
    if videos:
        raise ValueError("matrix evaluation is sealed to no videos")
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    tasks = list(PROTOCOL_TASKS[protocol])
    episodes = PROTOCOL_EPISODES[protocol]
    training_path, validated_base, validated_training = _resolve_training_lineage(
        adapter_checkpoint=adapter_checkpoint,
        training_manifest=training_manifest,
    )
    base_policy = validated_base
    if base_checkpoint is not None:
        requested_base = Path(base_checkpoint).expanduser().resolve()
        if requested_base != base_policy:
            raise ValueError(
                "base checkpoint must equal the exact base snapshot referenced by the training manifest"
            )
    adapter = Path(_adapter_directory(adapter_checkpoint)).expanduser().resolve()
    adapter_sha256 = _sha256_file(adapter / "adapter_model.safetensors")
    adapter_config = adapter / "adapter_config.json"
    train_config = adapter / "train_config.json"
    if not adapter_config.is_file() or not train_config.is_file():
        raise ValueError("fine-tuned adapter is missing adapter_config.json or train_config.json")
    root = output_root.expanduser().resolve()
    cells = [
        _cell(
            cell_id="smolvla_base_vanilla", checkpoint=str(base_policy),
            output_root=root, tasks=tasks, episodes=episodes,
            adapter_sha256=None, suite_mode="vanilla",
            training_manifest=None, base_policy=str(base_policy),
        ),
        _cell(
            cell_id="smolvla_base_sealed_randomized", checkpoint=str(base_policy),
            output_root=root, tasks=tasks, episodes=episodes,
            adapter_sha256=None, suite_mode="sealed_randomized",
            training_manifest=None, base_policy=str(base_policy),
        ),
        _cell(
            cell_id="smolvla_no_arrow_ft_vanilla", checkpoint=str(adapter),
            output_root=root, tasks=tasks, episodes=episodes,
            adapter_sha256=adapter_sha256, suite_mode="vanilla",
            training_manifest=str(training_path), base_policy=str(base_policy),
        ),
    ]
    source_hashes = shared_source_hashes()
    plans: list[dict[str, Any]] = []
    for cell in cells:
        plan = build_evaluation_plan(
            policy_kind=cell["policy_kind"], suite_mode=cell["suite_mode"],
            task_ids=tasks, episodes_per_task=episodes, seed_base=SEALED_SEED,
            camera=TRAINING_CAMERAS, resolution=256, text_context="none",
            visual_input="none", source_hashes=source_hashes,
        )
        plans.append(validate_plan(plan))
        cell["plan_sha256"] = plan["sha256"]
        cell["schedule_sha256"] = plan["schedule"]["sha256"]
    schedule = _schedule_manifest(protocol=protocol, tasks=tasks, episodes=episodes)
    randomization_config = randomization_config_payload()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EVAL_EXPERIMENT,
        "protocol": protocol,
        "model": "smolvla",
        "base_model_id": BASE_MODEL_ID,
        "base_policy": str(base_policy),
        "base_policy_revision": SEALED_REVISION,
        "adapter_checkpoint": str(adapter),
        "adapter_sha256": adapter_sha256,
        "adapter_config_sha256": _sha256_file(adapter_config),
        "train_config_sha256": _sha256_file(train_config),
        "training_manifest": str(training_path),
        "training_manifest_sha256": _sha256_file(training_path),
        "training_contract": {
            "checkpoint_step": int(SEALED_CHECKPOINT_ID),
            "steps": int(SEALED_STEPS),
            "save_freq": int(SEALED_SAVE_FREQ),
            "base_snapshot_revision": SEALED_REVISION,
            "validated": True,
        },
        "tasks": tasks,
        "episodes": episodes,
        "planned_episodes_per_cell": len(tasks) * episodes,
        "planned_episodes_total": len(tasks) * episodes * len(cells),
        "seed": SEALED_SEED,
        "seed_base": SEALED_SEED,
        "episode_seed_policy": "seed=seed_base+episode_index",
        "batch_size": 1,
        "device": device,
        "camera_name": TRAINING_CAMERAS,
        "raw_camera_names": ",".join(DEFAULT_CAMERAS),
        "observation_height": 256,
        "observation_width": 256,
        "text_context_mode": "standard",
        "text_context_format": "standard",
        "visual_condition": "none",
        "n_action_steps": "checkpoint",
        "videos": False,
        "max_videos": 0,
        "randomization_dimensions": {
            str(task_id): task_randomization_dimensions(task_id) for task_id in tasks
        },
        "randomization_dimension_names": list(RANDOMIZATION_DIMENSIONS),
        "randomization_config": randomization_config,
        "randomization_config_sha256": _hash_payload(randomization_config),
        "output_root": root.as_posix(),
        "cells": cells,
        "plans": plans,
        "schedule": schedule,
        "schedule_sha256": schedule["schedule_sha256"],
        "provenance": {
            "source_commit": _source_commit(),
            "evaluation_source": "vla_benchmarking/libero/evaluation/run_lerobot_eval_with_context.py",
            "training_manifest_sha256": _sha256_file(training_path),
            "validated_base_policy": str(validated_training["base_policy"]),
        },
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing SmolVLA matrix schema_version")
    protocol = manifest.get("protocol")
    if protocol not in PROTOCOL_TASKS:
        raise ValueError("matrix protocol is invalid")
    tasks = list(PROTOCOL_TASKS[protocol])
    episodes = PROTOCOL_EPISODES[protocol]
    if manifest.get("tasks") != tasks or manifest.get("episodes") != episodes:
        raise ValueError("matrix task/episode contract is not fixed for this protocol")
    if manifest.get("seed_base") != SEALED_SEED or manifest.get("seed") != SEALED_SEED:
        raise ValueError("matrix seed base must be exactly 1000")
    if manifest.get("batch_size") != 1 or manifest.get("visual_condition") != "none":
        raise ValueError("matrix runtime contract is invalid")
    if manifest.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("matrix base policy revision is not the sealed snapshot")
    if manifest.get("planned_episodes_per_cell") != len(tasks) * episodes:
        raise ValueError("matrix per-cell episode count is invalid")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or [cell.get("cell_id") for cell in cells] != list(CELL_IDS):
        raise ValueError("matrix cells must be exactly the fixed three-cell order")
    expected_suites = ["vanilla", "sealed_randomized", "vanilla"]
    expected_fine_tuned = [False, False, True]
    for index, cell in enumerate(cells):
        if cell.get("suite_mode") != expected_suites[index]:
            raise ValueError("matrix cell suite mode is invalid")
        if bool(cell.get("fine_tuned")) != expected_fine_tuned[index]:
            raise ValueError("matrix cell fine-tuning identity is invalid")
        if cell.get("tasks") != tasks or cell.get("episodes") != episodes:
            raise ValueError("matrix cell schedule is incomplete")
        if cell.get("seed") != SEALED_SEED:
            raise ValueError("matrix cell seed is invalid")
    if cells[0].get("checkpoint") != manifest.get("base_policy") or cells[1].get("checkpoint") != manifest.get("base_policy"):
        raise ValueError("base cells do not use the exact validated snapshot")
    if cells[2].get("checkpoint") != manifest.get("adapter_checkpoint"):
        raise ValueError("fine-tuned cell does not use the validated adapter")
    if cells[2].get("adapter_sha256") != manifest.get("adapter_sha256"):
        raise ValueError("fine-tuned cell adapter hash is not bound")
    if manifest.get("randomization_config") != randomization_config_payload():
        raise ValueError("matrix randomization config changed")
    if manifest.get("schedule", {}).get("schedule_sha256") != manifest.get("schedule_sha256"):
        raise ValueError("matrix schedule hash is inconsistent")
    if len(manifest.get("plans", [])) != len(CELL_IDS):
        raise ValueError("matrix must contain one plan per cell")
    for cell, plan in zip(cells, manifest["plans"]):
        if plan.get("policy_kind") != cell.get("policy_kind"):
            raise ValueError("cell plan policy identity is invalid")
        if plan.get("condition", {}).get("suite_mode") != cell.get("suite_mode"):
            raise ValueError("cell plan suite mode is invalid")
        if plan.get("schedule", {}).get("sha256") != cell.get("schedule_sha256"):
            raise ValueError("cell plan schedule hash is invalid")


def write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> str:
    _validate_manifest(manifest)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_sha256"] = _hash_payload(manifest)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != payload["manifest_sha256"] or {
            key: value for key, value in existing.items() if key != "manifest_sha256"
        } != manifest:
            raise ValueError(f"existing matrix manifest does not match: {path}")
        return payload["manifest_sha256"]
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return payload["manifest_sha256"]


def write_immutable_schedule(path: Path, schedule: dict[str, Any]) -> str:
    path = path.expanduser().resolve()
    body = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    expected = _hash_payload(body)
    if schedule.get("schedule_sha256") != expected:
        raise ValueError("schedule hash is invalid")
    payload = dict(schedule)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing matrix schedule does not match: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return expected


def validate_existing_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if not root.exists():
        return
    expected = {Path(cell["output_dir"]).resolve() for cell in manifest["cells"]}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.startswith("seed_"):
            for child in path.iterdir():
                if child.is_dir() and child.resolve() not in expected:
                    raise ValueError(f"unexpected stale matrix cell directory: {child}")
    for cell in manifest["cells"]:
        marker = Path(cell["output_dir"]) / "cell_manifest.json"
        if not marker.exists():
            continue
        actual = json.loads(marker.read_text(encoding="utf-8"))
        expected_marker = {
            key: cell[key] for key in ("cell_id", "checkpoint", "suite_mode", "tasks", "episodes", "seed", "output_dir")
        }
        if actual != expected_marker:
            raise ValueError(f"matrix cell marker does not match: {marker}")


def _write_cell_marker(cell: dict[str, Any]) -> None:
    path = Path(cell["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "cell_manifest.json"
    expected = {
        key: cell[key] for key in ("cell_id", "checkpoint", "suite_mode", "tasks", "episodes", "seed", "output_dir")
    }
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"matrix cell marker does not match: {marker}")
        return
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _run_cell(cell: dict[str, Any], *, device: str, videos: bool) -> int:
    _write_cell_marker(cell)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MODELS": cell["checkpoint"],
        "TASK_IDS": json.dumps(cell["tasks"], separators=(",", ":")),
        "N_EPISODES": str(cell["episodes"]),
        "BATCH_SIZE": "1",
        "SEED": str(cell["seed"]),
        "DEVICE": device,
        "SUITE_MODE": cell["suite_mode"],
        "RANDOMIZE_SCENES": "1" if cell["suite_mode"] == "sealed_randomized" else "0",
        "CONTEXT_MODE": "standard",
        "CONTEXT_FORMAT": "standard",
        "VISUAL_CONDITION": "none",
        "VISUAL_ARROWS": "0",
        "TRAINING_PROFILE": "no_arrow_treatment" if cell["fine_tuned"] else "base",
        "PROFILE": "no_arrow_treatment" if cell["fine_tuned"] else "base",
        "N_ACTION_STEPS": "checkpoint",
        "MAX_EPISODES_RENDERED": str(0 if not videos else 1),
        "RENDER_MODE": "rgb_array" if videos else "none",
    })
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "vla_benchmarking" / "libero" / "evaluation" / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false",
        f"--output_dir={cell['output_dir']}",
        f"--policy.path={cell['checkpoint']}",
        "--env.task_ids=" + json.dumps(cell["tasks"], separators=(",", ":")),
        "--env.camera_name=" + TRAINING_CAMERAS,
        "--env.observation_height=256",
        "--env.observation_width=256",
    ]
    log_path = Path(cell["output_dir"]) / "eval_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def _summary_row(cell: dict[str, Any], *, status: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    info = info or {}
    overall = info.get("overall", {}) if isinstance(info, dict) else {}
    return {
        "row_type": "cell",
        "cell_id": cell["cell_id"],
        "suite_mode": cell["suite_mode"],
        "fine_tuned": cell["fine_tuned"],
        "task_id": "all",
        "seed": cell["seed"],
        "successes": sum(sum(record.get("metrics", {}).get("successes", [])) for record in info.get("per_task", [])) if isinstance(info, dict) else "",
        "episodes": overall.get("n_episodes", "") if isinstance(overall, dict) else "",
        "pc_success": overall.get("pc_success", "") if isinstance(overall, dict) else "",
        "status": status,
        "checkpoint": cell["checkpoint"],
        "eval_info": str(Path(cell["output_dir"]) / "eval_info.json"),
        "output_dir": cell["output_dir"],
    }


def _task_rows(cell: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in info["per_task"]:
        successes = record["metrics"]["successes"]
        rows.append({
            "row_type": "task", "cell_id": cell["cell_id"], "suite_mode": cell["suite_mode"],
            "fine_tuned": cell["fine_tuned"], "task_id": record["task_id"], "seed": cell["seed"],
            "successes": sum(successes), "episodes": len(successes),
            "pc_success": 100.0 * sum(successes) / len(successes), "status": "complete",
            "checkpoint": cell["checkpoint"], "eval_info": str(Path(cell["output_dir"]) / "eval_info.json"),
            "output_dir": cell["output_dir"],
        })
    return rows


def _write_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = ["row_type", "cell_id", "suite_mode", "fine_tuned", "task_id", "seed", "successes", "episodes", "pc_success", "status", "checkpoint", "eval_info", "output_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_TASKS), required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args(argv)
    if not args.no_videos:
        print("ERROR: SmolVLA matrix evaluation requires --no-videos")
        return 1
    root = Path(args.output_root).expanduser().resolve()
    try:
        manifest = build_manifest(
            adapter_checkpoint=args.adapter_checkpoint,
            training_manifest=args.training_manifest,
            output_root=root,
            protocol=args.protocol,
            base_checkpoint=args.base_checkpoint,
            device=args.device,
        )
        validate_existing_outputs(root, manifest)
        write_immutable_manifest(root / MANIFEST_FILENAME, manifest)
        write_immutable_schedule(root / SCHEDULE_FILENAME, manifest["schedule"])
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: SmolVLA matrix preflight failed: {exc}")
        return 1

    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        return_code = _run_cell(cell, device=args.device, videos=False)
        eval_path = Path(cell["output_dir"]) / "eval_info.json"
        if return_code != 0:
            rows.append(_summary_row(cell, status=f"failed:{return_code}"))
            _write_summary(root / SUMMARY_FILENAME, rows)
            print(f"ERROR: matrix cell {cell['cell_id']} failed ({return_code})")
            return return_code
        try:
            info = validate_eval_info(eval_path, {"tasks": cell["tasks"], "episodes": cell["episodes"]})
            if cell["suite_mode"] == "sealed_randomized":
                validate_randomization_audit(Path(cell["output_dir"]), manifest)
        except ValueError as exc:
            rows.append(_summary_row(cell, status="invalid"))
            _write_summary(root / SUMMARY_FILENAME, rows)
            print(f"ERROR: matrix cell {cell['cell_id']} validation failed: {exc}")
            return 1
        rows.append(_summary_row(cell, status="complete", info=info))
        rows.extend(_task_rows(cell, info))

    _write_summary(root / SUMMARY_FILENAME, rows)
    print(f"SmolVLA matrix {args.protocol} evaluation complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
