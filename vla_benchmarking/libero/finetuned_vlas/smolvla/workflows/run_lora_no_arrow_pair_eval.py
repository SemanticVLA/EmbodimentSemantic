#!/usr/bin/env python3
"""Evaluate one no-arrow LoRA adapter with and without live visual arrows.

This is a causal-control evaluation, not the treatment 2x2 matrix.  The same
adapter is loaded for both ordered cells, so the only intended contrast is the
effect of adding live arrows at evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Bootstrap the repository root before organized-package imports so direct
# script launches work with an empty ambient PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vla_benchmarking.libero.shared.config import (
    LEROBOT_CAMERA_KEYS,
    DEFAULT_CAMERAS,
    RANDOMIZATION_DIMENSIONS,
    task_randomization_dimensions,
)
from vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.evaluation_contracts import (
    TASK_IDS,
    parse_int_list,
    validate_eval_info,
    validate_randomization_audit,
)
from vla_benchmarking.libero.evaluation.randomization_contract import (
    randomization_config_payload,
)


TRAINING_CAMERAS = ",".join(LEROBOT_CAMERA_KEYS)
RAW_TRAINING_CAMERAS = ",".join(DEFAULT_CAMERAS)
SCHEMA_VERSION = 1
SEALED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SEALED_CHECKPOINT_ID = "029190"
SEALED_STEPS = 29190
SEALED_SAVE_FREQ = 1946
SEALED_BATCH_SIZE = 32
SEALED_SEED = 1000
SEALED_PEFT_R = 16
SMOKE_CHECKPOINT_ID = "000002"
SMOKE_STEPS = 2
SMOKE_SAVE_FREQ = 2
SMOKE_TASKS = (0, 4)
SMOKE_EPISODES = 1


def _evaluation_contract(scope: str, profile: SmolVLAProfile) -> dict[str, Any]:
    """Return the immutable artifact/evaluation contract for one scope.

    The two-step checkpoint is intentionally available only to the explicit
    target-arrow smoke scope.  In particular, a caller cannot obtain a smoke
    contract merely by changing a checkpoint path or by selecting a different
    profile; the training manifest is checked against this contract below.
    """
    if scope == "full":
        return {
            "scope": "full",
            "checkpoint_id": SEALED_CHECKPOINT_ID,
            "steps": SEALED_STEPS,
            "save_freq": SEALED_SAVE_FREQ,
            "tasks": tuple(TASK_IDS),
            "episodes": 10,
        }
    if scope == "smoke":
        if profile.name != "target_arrow_treatment":
            raise ValueError(
                "smoke evaluation is reserved for the target_arrow_treatment profile"
            )
        return {
            "scope": "smoke",
            "checkpoint_id": SMOKE_CHECKPOINT_ID,
            "steps": SMOKE_STEPS,
            "save_freq": SMOKE_SAVE_FREQ,
            "tasks": SMOKE_TASKS,
            "episodes": SMOKE_EPISODES,
        }
    raise ValueError("evaluation scope must be one of: full, smoke")


@dataclass(frozen=True)
class SmolVLAProfile:
    name: str
    experiment: str
    eval_experiment: str
    trained_visual_condition: str
    model_role: str
    training_variant: str
    dataset_variant: str
    dataset_repo_id: str
    adapter_key: str
    pair_kind: str
    manifest_filename: str
    summary_filename: str
    cells: tuple[tuple[str, bool, str], ...]
    visual_condition_with_arrows: str
    visual_label_with_arrows: str
    contrast: str

    @property
    def cell_ids(self) -> tuple[str, str]:
        return tuple(item[0] for item in self.cells)  # type: ignore[return-value]


PROFILES: dict[str, SmolVLAProfile] = {
    "no_arrow_treatment": SmolVLAProfile(
        name="no_arrow_treatment",
        experiment="smolvla_lora_no_arrow_treatment_training",
        eval_experiment="smolvla_lora_no_arrow_trained_live_vs_none_2cell",
        trained_visual_condition="no_arrows",
        model_role="no_arrow_trained_lora",
        training_variant="no_arrow_treatment",
        dataset_variant="control",
        dataset_repo_id="local/libero_spatial_control",
        adapter_key="no_arrow_treatment_adapter",
        pair_kind="sealed_lora_control_treatment",
        manifest_filename="no_arrow_trained_arrow_pair_manifest.json",
        summary_filename="no_arrow_trained_arrow_pair_summary.csv",
        cells=(("no_arrow_trained_live_arrows", True, "visual_arrows"), ("no_arrow_trained_no_arrows", False, "none")),
        visual_condition_with_arrows="visual_arrows",
        visual_label_with_arrows="visual_arrows",
        contrast="live_arrow_effect_pp",
    ),
    "target_arrow_treatment": SmolVLAProfile(
        name="target_arrow_treatment",
        experiment="smolvla_lora_target_arrow_treatment_training",
        eval_experiment="smolvla_lora_target_arrow_trained_goal_vs_none_2cell",
        trained_visual_condition="target_arrow",
        model_role="target_arrow_trained_lora",
        training_variant="target_arrow_treatment",
        dataset_variant="target_arrow_treatment",
        dataset_repo_id="local/libero_spatial_target_arrow_treatment",
        adapter_key="target_arrow_treatment_adapter",
        pair_kind="sealed_lora_control_target_arrow_treatment",
        manifest_filename="target_arrow_trained_arrow_pair_manifest.json",
        summary_filename="target_arrow_trained_arrow_pair_summary.csv",
        cells=(("target_arrow_trained_goal_arrow", True, "visual_goal_arrow"), ("target_arrow_trained_no_arrows", False, "none")),
        visual_condition_with_arrows="visual_goal_arrow",
        visual_label_with_arrows="visual_goal_arrow",
        contrast="goal_arrow_effect_pp",
    ),
}


def get_profile(name: str | None) -> SmolVLAProfile:
    return PROFILES.get(str(name or "no_arrow_treatment"), PROFILES["no_arrow_treatment"])


DEFAULT_PROFILE = PROFILES["no_arrow_treatment"]
CELL_IDS = DEFAULT_PROFILE.cell_ids
MANIFEST_FILENAME = DEFAULT_PROFILE.manifest_filename
SUMMARY_FILENAME = DEFAULT_PROFILE.summary_filename


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_manifest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_directory(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("adapter checkpoint path must not be empty")
    candidate = Path(value).expanduser().resolve()
    if not candidate.is_dir() or candidate.name != "pretrained_model":
        raise ValueError("adapter checkpoint must be a pretrained_model directory")
    artifact = candidate / "adapter_model.safetensors"
    if not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(f"adapter_model.safetensors is missing or empty: {artifact}")
    return str(candidate)


def build_cells(
    adapter_checkpoint: str,
    seeds: Sequence[int],
    output_root: Path,
    profile: SmolVLAProfile = DEFAULT_PROFILE,
) -> list[dict[str, Any]]:
    """Return exactly two ordered cells per seed using the same adapter."""
    adapter = _adapter_directory(adapter_checkpoint)
    cells: list[dict[str, Any]] = []
    for seed in seeds:
        for cell_id, live_arrows, visual_condition in profile.cells:
            cell_dir = output_root / f"seed_{int(seed)}" / cell_id
            cells.append(
                {
                    "cell_id": cell_id,
                    "seed": int(seed),
                    "checkpoint": adapter,
                    "live_arrows": live_arrows,
                    "visual_condition": visual_condition,
                    "output_dir": cell_dir.as_posix(),
                    "adapter_sha256": _sha256_file(Path(adapter) / "adapter_model.safetensors"),
                }
            )
    return cells


def build_manifest(
    *,
    adapter_checkpoint: str,
    seeds: Sequence[int],
    tasks: Sequence[int] | None = None,
    episodes: int = 10,
    batch_size: int = 1,
    device: str = "cuda",
    videos: bool = False,
    max_videos: int = 0,
    training_manifest: str,
    output_root: Path,
    profile_name: str = "no_arrow_treatment",
    evaluation_scope: str = "full",
) -> dict[str, Any]:
    profile = get_profile(profile_name)
    contract = _evaluation_contract(evaluation_scope, profile)
    if tasks is None:
        tasks = contract["tasks"]
    tasks = [int(task_id) for task_id in tasks]
    if tasks != list(contract["tasks"]):
        raise ValueError(
            f"{evaluation_scope} evaluation requires task IDs exactly "
            f"{list(contract['tasks'])}"
        )
    seeds = [int(seed) for seed in seeds]
    if seeds != [SEALED_SEED]:
        raise ValueError("sealed no-arrow eval requires seeds exactly [1000]")
    if episodes != contract["episodes"]:
        raise ValueError(
            f"{evaluation_scope} evaluation requires exactly "
            f"{contract['episodes']} episodes"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size != 1:
        raise ValueError("sealed randomization audit requires batch_size=1")
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    root = output_root.expanduser().resolve()
    dimensions = {
        str(task_id): task_randomization_dimensions(task_id) for task_id in tasks
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
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile.name,
        "experiment": profile.eval_experiment,
        "trained_on_visual_condition": profile.trained_visual_condition,
        "adapter_checkpoint": _adapter_directory(adapter_checkpoint),
        "adapter_sha256": _sha256_file(
            Path(adapter_checkpoint).expanduser().resolve() / "adapter_model.safetensors"
        ),
        "model_role": profile.model_role,
        "evaluation_scope": contract["scope"],
        "checkpoint_step": int(contract["checkpoint_id"]),
        "training_steps": int(contract["steps"]),
        "training_save_freq": int(contract["save_freq"]),
        "training_variant": profile.training_variant,
        "dataset_variant": profile.dataset_variant,
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
        "randomization_config": randomization_config,
        "randomization_config_sha256": hashlib.sha256(
            _canonical_json(randomization_config).encode("utf-8")
        ).hexdigest(),
        "output_root": root.as_posix(),
        "training_manifest": str(Path(training_manifest).expanduser().resolve()),
        "training_manifest_sha256": _sha256_file(Path(training_manifest).expanduser().resolve()),
        "contrast": profile.contrast,
        "cells": build_cells(adapter_checkpoint, seeds, root, profile),
    }
    _validate_manifest(manifest, profile)
    return manifest


def _validate_manifest(manifest: dict[str, Any], profile: SmolVLAProfile | None = None) -> None:
    profile = profile or get_profile(manifest.get("profile"))
    contract = _evaluation_contract(str(manifest.get("evaluation_scope", "full")), profile)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing no-arrow manifest schema_version")
    if manifest.get("profile") != profile.name or manifest.get("experiment") != profile.eval_experiment:
        raise ValueError("evaluation manifest has the wrong SmolVLA profile")
    if manifest.get("trained_on_visual_condition") != profile.trained_visual_condition:
        raise ValueError("evaluation manifest has the wrong training condition")
    if manifest.get("training_variant") != profile.training_variant or manifest.get("dataset_variant") != profile.dataset_variant:
        raise ValueError("evaluation manifest has the wrong dataset lineage")
    if manifest.get("model_role") != profile.model_role:
        raise ValueError("evaluation manifest has the wrong model role")
    if manifest.get("checkpoint_step") != int(contract["checkpoint_id"]):
        raise ValueError(
            f"{contract['scope']} evaluation manifest must use checkpoint "
            f"{contract['checkpoint_id']}"
        )
    if manifest.get("training_steps") != int(contract["steps"]):
        raise ValueError(
            f"{contract['scope']} evaluation manifest must record training steps "
            f"{contract['steps']}"
        )
    if manifest.get("training_save_freq") != int(contract["save_freq"]):
        raise ValueError(
            f"{contract['scope']} evaluation manifest must record training save_freq "
            f"{contract['save_freq']}"
        )
    if not isinstance(manifest.get("training_manifest_sha256"), str) or not isinstance(manifest.get("adapter_sha256"), str):
        raise ValueError("no-arrow evaluation manifest lacks sealed artifact hashes")
    if manifest.get("tasks") != list(contract["tasks"]):
        raise ValueError(
            f"manifest tasks must match {contract['scope']} scope: "
            f"{list(contract['tasks'])}"
        )
    if manifest.get("randomize_scenes") is not True:
        raise ValueError("manifest must keep RANDOMIZE_SCENES enabled")
    if manifest.get("batch_size") != 1:
        raise ValueError("sealed randomization audit requires batch_size=1")
    if manifest.get("camera_name") != TRAINING_CAMERAS or manifest.get("raw_camera_names") != RAW_TRAINING_CAMERAS:
        raise ValueError("manifest camera contract does not match training cameras")
    if manifest.get("observation_height") != 256 or manifest.get("observation_width") != 256:
        raise ValueError("manifest must use 256x256 observations")
    if manifest.get("text_context_mode") != "standard" or manifest.get("text_context_format") != "standard":
        raise ValueError("sealed eval must use standard text context")
    if manifest.get("n_action_steps") != "checkpoint":
        raise ValueError("sealed eval must preserve checkpoint n_action_steps")
    if manifest.get("visual_arrow_width") != 1 or manifest.get("visual_arrow_head_length") != 16:
        raise ValueError("manifest arrow geometry must be width=1 and head=16")
    if manifest.get("contrast") != profile.contrast:
        raise ValueError("evaluation must expose the configured two-cell contrast")
    seeds = manifest.get("seeds")
    if seeds != [SEALED_SEED]:
        raise ValueError("manifest seeds must be exactly [1000]")
    if manifest.get("episodes") != contract["episodes"]:
        raise ValueError(
            f"manifest must contain exactly {contract['episodes']} episodes "
            f"for {contract['scope']} scope"
        )
    dimensions = manifest.get("randomization_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != {
        str(task_id) for task_id in contract["tasks"]
    }:
        raise ValueError(
            f"manifest must record randomization dimensions for every "
            f"{contract['scope']} task"
        )
    for task_id, values in dimensions.items():
        if not isinstance(values, dict) or not values.get("object_removal"):
            raise ValueError(f"task {task_id} must enable object_removal")
    expected_config = randomization_config_payload()
    if manifest.get("randomization_config") != expected_config:
        raise ValueError("manifest randomization config does not match sealed config")
    expected_config_hash = hashlib.sha256(
        _canonical_json(expected_config).encode("utf-8")
    ).hexdigest()
    if manifest.get("randomization_config_sha256") != expected_config_hash:
        raise ValueError("manifest randomization config hash does not match sealed config")
    adapter = manifest.get("adapter_checkpoint")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError("manifest adapter_checkpoint is missing")
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise ValueError("manifest cells must be a list")
    expected = {(int(seed), cell_id) for seed in seeds for cell_id in profile.cell_ids}
    actual = {(int(cell.get("seed")), cell.get("cell_id")) for cell in cells}
    if actual != expected or len(cells) != len(actual):
        raise ValueError("manifest has incomplete, duplicate, or unexpected no-arrow cells")
    for index, cell in enumerate(cells):
        if cell.get("cell_id") not in profile.cell_ids or cell.get("checkpoint") != adapter:
            raise ValueError("all no-arrow cells must use the same adapter checkpoint")
        if cell.get("adapter_sha256") != manifest.get("adapter_sha256"):
            raise ValueError("all no-arrow cells must use the same adapter identity")
        expected_arrows = profile.cells[profile.cell_ids.index(cell["cell_id"])][1]
        if bool(cell.get("live_arrows")) != expected_arrows:
            raise ValueError("manifest cell arrow condition does not match cell_id")
        expected_index = 2 * list(seeds).index(int(cell["seed"])) + profile.cell_ids.index(cell["cell_id"])
        if index != expected_index:
            raise ValueError("manifest cells are not in the required ordered pair sequence")


def _resume_record_digest(record: dict[str, Any]) -> str:
    body = dict(record)
    body.pop("record_sha256", None)
    return hashlib.sha256((json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def _validate_resume_audit_chain(
    data: dict[str, Any],
    profile: SmolVLAProfile = DEFAULT_PROFILE,
    *,
    evaluation_scope: str = "full",
) -> None:
    contract = _evaluation_contract(evaluation_scope, profile)
    audits = data.get("resume_audits")
    chain_digest = data.get("resume_chain_digest")
    if not isinstance(audits, list) or not isinstance(chain_digest, str):
        raise ValueError("no-arrow training manifest lacks resume audit provenance")
    hashes: list[str] = []
    previous_step: int | None = None
    for index, record in enumerate(audits, 1):
        if not isinstance(record, dict):
            raise ValueError(f"resume audit record {index} is not an object")
        required = {
            "schema_version", "chain_index", "train_config_path", "train_config_sha256",
            "checkpoint_step", "profile", "dataset_variant", "previous_record_sha256", "record_sha256",
        }
        if not required <= set(record):
            raise ValueError(f"resume audit record {index} is incomplete")
        if record["schema_version"] != 1 or record["chain_index"] != index:
            raise ValueError(f"resume audit chain index/schema mismatch at record {index}")
        if record["profile"] != profile.name or record["dataset_variant"] != profile.dataset_variant:
            raise ValueError(f"resume audit profile/dataset mismatch at record {index}")
        step = int(record["checkpoint_step"])
        if step < 0 or (previous_step is not None and step <= previous_step):
            raise ValueError("resume audit checkpoints are not strictly increasing")
        previous_step = step
        if record["previous_record_sha256"] != (hashes[-1] if hashes else None):
            raise ValueError(f"resume audit previous-record linkage is broken at record {index}")
        if record["record_sha256"] != _resume_record_digest(record):
            raise ValueError(f"resume audit record hash mismatch at record {index}")
        config_path = Path(record["train_config_path"]).expanduser().resolve()
        if not config_path.is_file() or _sha256_file(config_path) != record["train_config_sha256"]:
            raise ValueError(f"resume train_config.json is missing or changed: {config_path}")
        checkpoint_parts = config_path.parts
        checkpoint_indexes = [i for i, part in enumerate(checkpoint_parts) if part == "checkpoints"]
        if not checkpoint_indexes:
            raise ValueError(f"resume config is not under checkpoints: {config_path}")
        step_index = checkpoint_indexes[-1] + 1
        if step_index >= len(checkpoint_parts) or not checkpoint_parts[step_index].isdigit() or int(checkpoint_parts[step_index]) != step:
            raise ValueError(f"resume config/checkpoint step mismatch: {config_path}")
        if step > contract["steps"]:
            raise ValueError(
                f"resume audit checkpoint exceeds {contract['scope']} training steps"
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"resume train_config.json is unreadable: {config_path}") from exc
        if config.get("dataset", {}).get("repo_id") != profile.dataset_repo_id:
            raise ValueError("resume train_config dataset lineage does not match selected profile")
        for key, expected in (("steps", contract["steps"]), ("save_freq", contract["save_freq"]), ("batch_size", SEALED_BATCH_SIZE), ("seed", SEALED_SEED)):
            if config.get(key) is None or int(config[key]) != expected:
                raise ValueError(f"resume train_config {key} is not sealed")
        peft = config.get("peft", config.get("policy", {}).get("peft", {}))
        if not isinstance(peft, dict) or peft.get("r") is None or int(peft["r"]) != SEALED_PEFT_R:
            raise ValueError("resume train_config PEFT rank is not sealed")
        hashes.append(record["record_sha256"])
    expected_chain = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if chain_digest != expected_chain:
        raise ValueError("resume audit chain digest mismatch")


def validate_training_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    allow_revalidated_pair_sentinel: bool = False,
    profile: SmolVLAProfile | None = None,
) -> dict[str, Any]:
    """Require the sealed no-arrow adapter lineage and reject treatment lineage."""
    if not path.is_file():
        raise ValueError(f"no-arrow training manifest is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"no-arrow training manifest is unreadable: {path}") from exc
    if manifest.get("training_manifest_sha256") != _sha256_file(path):
        raise ValueError("evaluation training_manifest_sha256 does not match the current training manifest")
    profile = profile or get_profile(data.get("training_variant"))
    evaluation_scope = str(manifest.get("evaluation_scope", "full"))
    contract = _evaluation_contract(evaluation_scope, profile)
    if data.get("experiment") != profile.experiment:
        if profile.name == "no_arrow_treatment":
            raise ValueError("unexpected no-arrow training manifest experiment")
        raise ValueError("unexpected SmolVLA training manifest experiment")
    if data.get("training_variant") != profile.training_variant or data.get("dataset_variant") != profile.dataset_variant:
        raise ValueError("training manifest is not for the selected SmolVLA dataset")
    if data.get("dataset_repo_id") != profile.dataset_repo_id:
        raise ValueError(
            "training manifest dataset_repo_id does not match the selected profile"
        )
    if data.get("trained_on_visual_condition") != profile.trained_visual_condition:
        if profile.name == "no_arrow_treatment":
            raise ValueError("training manifest does not record no_arrows")
        raise ValueError("training manifest visual condition does not match the selected profile")
    if data.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("training manifest base policy revision is not sealed")
    base_policy = Path(data.get("base_policy", "")).expanduser().resolve()
    if base_policy.name != f"smolvla_libero-{SEALED_REVISION}" or not base_policy.is_dir():
        raise ValueError("training manifest base policy path is not the sealed snapshot")
    base_snapshot = base_policy / "base_snapshot_manifest.json"
    if not base_snapshot.is_file():
        raise ValueError("sealed base snapshot manifest is missing")
    try:
        snapshot = json.loads(base_snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("sealed base snapshot manifest is unreadable") from exc
    if snapshot.get("revision") != SEALED_REVISION or not isinstance(snapshot.get("files"), dict) or not snapshot["files"]:
        raise ValueError("sealed base snapshot revision/file inventory is invalid")
    actual_files = {
        item.relative_to(base_policy).as_posix()
        for item in base_policy.rglob("*")
        if item.is_file() and item.name != "base_snapshot_manifest.json" and ".cache" not in item.parts
    }
    if actual_files != set(snapshot["files"]):
        raise ValueError("sealed base snapshot file set differs from its manifest")
    for name, expected_hash in snapshot["files"].items():
        item = base_policy / name
        if not item.is_file() or _sha256_file(item) != expected_hash:
            raise ValueError(f"sealed base snapshot hash mismatch: {name}")
    if profile.name == "no_arrow_treatment" and "treatment_adapter" in data:
        raise ValueError("training manifest must not contain treatment_adapter")
    if profile.adapter_key != "no_arrow_treatment_adapter" and "no_arrow_treatment_adapter" in data:
        raise ValueError("selected training manifest contains a no-arrow adapter record")
    adapter = data.get(profile.adapter_key)
    if not isinstance(adapter, dict):
        raise ValueError(f"training manifest lacks {profile.adapter_key}")
    recorded_adapter = Path(adapter.get("path", "")).expanduser().resolve()
    adapter_path = recorded_adapter.parent if recorded_adapter.name == "adapter_model.safetensors" else recorded_adapter
    expected_adapter = Path(manifest["adapter_checkpoint"]).expanduser().resolve()
    if adapter_path != expected_adapter:
        raise ValueError("evaluation adapter does not match no-arrow training manifest")
    adapter_artifact = adapter_path / "adapter_model.safetensors"
    if not adapter_path.is_dir() or adapter_path.name != "pretrained_model" or not adapter_artifact.is_file() or _sha256_file(adapter_artifact) != adapter.get("sha256"):
        raise ValueError(f"no-arrow adapter is missing or has changed: {adapter_artifact}")
    if manifest.get("adapter_sha256") != _sha256_file(adapter_artifact):
        raise ValueError("evaluation adapter_sha256 does not match the adapter artifact")
    checkpoint_dir = adapter_path.parent
    if checkpoint_dir.name != contract["checkpoint_id"]:
        raise ValueError(
            f"{contract['scope']} adapter is not checkpoint "
            f"{contract['checkpoint_id']}"
        )
    if data.get("final_checkpoint_id") != contract["checkpoint_id"]:
        raise ValueError(
            f"{contract['scope']} training manifest final checkpoint is not "
            f"{contract['checkpoint_id']}"
        )
    if manifest.get("checkpoint_step") != int(contract["checkpoint_id"]):
        raise ValueError("evaluation checkpoint step does not match its scope")
    if data.get("pair_kind") != profile.pair_kind:
        raise ValueError("training manifest pair identity is invalid")
    if data.get("base_policy") != str(base_policy):
        raise ValueError("no-arrow training base policy path is not canonical")
    pair_names = {
        "no_arrow_treatment": {
            "pair_manifest": "sealed_lora_pair_manifest.json",
            "pair_sentinel": "sealed_lora_pair_verified.json",
        },
        "target_arrow_treatment": {
            "pair_manifest": "sealed_lora_target_arrow_pair_manifest.json",
            "pair_sentinel": "sealed_lora_target_arrow_pair_verified.json",
        },
    }[profile.name]
    for key, expected_name in pair_names.items():
        item = Path(data.get(key, "")).expanduser().resolve()
        if item.name != expected_name or not item.is_file():
            raise ValueError(f"no-arrow training {key} identity/hash is invalid")
        observed_hash = _sha256_file(item)
        expected_hash = data.get(f"{key}_sha256")
        hash_matches = observed_hash == expected_hash
        if not hash_matches and not (
            key == "pair_sentinel" and allow_revalidated_pair_sentinel
        ):
            raise ValueError(f"no-arrow training {key} identity/hash is invalid")
        try:
            pair_data = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"no-arrow training {key} is unreadable") from exc
        if pair_data.get("pair_kind") != profile.pair_kind or pair_data.get("full_experiment_ready") is not True or pair_data.get("launch_eligibility") != "full_experiment_ready":
            raise ValueError(f"no-arrow training {key} is not the sealed launchable pair")
        if key == "pair_sentinel" and not hash_matches:
            pair_manifest_hash = data.get("pair_manifest_sha256")
            if pair_data.get("manifest_sha256") != pair_manifest_hash:
                raise ValueError(
                    "revalidated no-arrow pair_sentinel does not identify the sealed pair manifest"
                )
    _validate_resume_audit_chain(data, profile, evaluation_scope=evaluation_scope)
    plan = Path(data.get("training_plan", ""))
    if not plan.is_file() or _sha256_file(plan) != data.get("training_plan_sha256"):
        raise ValueError("no-arrow training plan is missing or has changed")
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    if plan_data.get("experiment") != profile.experiment or plan_data.get("training_variant") != profile.training_variant:
        raise ValueError("training plan lineage is invalid")
    if plan_data.get("trained_on_visual_condition") != profile.trained_visual_condition:
        raise ValueError("training plan visual lineage is invalid")
    if plan_data.get("dataset_variant") != profile.dataset_variant or plan_data.get("dataset_repo_id") != profile.dataset_repo_id:
        raise ValueError("training plan dataset lineage is invalid")
    if plan_data.get("base_policy_revision") != SEALED_REVISION or Path(plan_data.get("base_policy", "")).expanduser().resolve() != base_policy:
        raise ValueError("no-arrow training plan base snapshot lineage is invalid")
    expected_flags = {
        "steps": contract["steps"],
        "save_freq": contract["save_freq"],
        "batch_size": SEALED_BATCH_SIZE,
        "seed": SEALED_SEED,
        "peft_r": SEALED_PEFT_R,
    }
    for source_name, source in (("manifest", data), ("training plan", plan_data)):
        flags = source.get("flags")
        if not isinstance(flags, dict) or any(int(flags.get(key, -1)) != value for key, value in expected_flags.items()):
            raise ValueError(f"{source_name} flags are not sealed for no-arrow training")
    return data


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
            raise ValueError(f"existing manifest does not match requested no-arrow pair: {path}")
        return payload["manifest_sha256"]
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return payload["manifest_sha256"]


def validate_existing_outputs(output_root: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    root = output_root.expanduser().resolve()
    if not root.exists():
        return
    expected_dirs = {Path(cell["output_dir"]).resolve() for cell in manifest["cells"]}
    expected_seed_dirs = {path.parent for path in expected_dirs}
    for path in root.rglob("*"):
        if not path.is_dir() or path == root:
            continue
        if path.name.startswith("seed_") and path not in expected_seed_dirs:
            raise ValueError(f"unexpected stale seed output directory: {path}")
    for seed_dir in expected_seed_dirs:
        if not seed_dir.is_dir():
            continue
        allowed = {path for path in expected_dirs if path.parent == seed_dir}
        for child in seed_dir.iterdir():
            if child.is_dir() and child.resolve() not in allowed:
                raise ValueError(f"unexpected stale cell output directory: {child}")
    for cell in manifest["cells"]:
        marker = Path(cell["output_dir"]) / "cell_manifest.json"
        if not marker.exists():
            continue
        actual = json.loads(marker.read_text(encoding="utf-8"))
        expected = {key: cell[key] for key in ("cell_id", "seed", "checkpoint", "live_arrows", "output_dir")}
        if actual != expected:
            raise ValueError(f"stale cell marker does not match manifest: {marker}")


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


def _run_cell(
    cell: dict[str, Any],
    args: argparse.Namespace,
    script_dir: Path,
    profile: SmolVLAProfile,
    task_ids: Sequence[int],
) -> int:
    _write_cell_marker(cell)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODELS": cell["checkpoint"],
            "TASK_IDS": json.dumps(list(task_ids), separators=(",", ":")),
            "N_EPISODES": str(args.episodes),
            "BATCH_SIZE": str(args.batch_size),
            "SEED": str(cell["seed"]),
            "DEVICE": args.device,
            "CONTEXT_MODE": "standard",
            "CONTEXT_FORMAT": "standard",
            "VISUAL_CONDITION": cell["visual_condition"],
            "VISUAL_ARROWS": "1" if cell["live_arrows"] else "0",
            "DISABLE_VISUAL_PROMPT_HINT": "1" if profile.name == "target_arrow_treatment" else "0",
            "TRAINING_PROFILE": profile.name,
            "PROFILE": profile.name,
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
        str(_REPO_ROOT / "vla_benchmarking" / "libero" / "evaluation" / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false",
        f"--output_dir={cell['output_dir']}",
        f"--policy.path={cell['checkpoint']}",
        "--env.task_ids=" + json.dumps(list(task_ids), separators=(",", ":")),
        "--env.camera_name=" + TRAINING_CAMERAS,
        "--env.observation_height=256",
        "--env.observation_width=256",
    ]
    log_path = Path(cell["output_dir"]) / "eval_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=script_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process.returncode


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
    return {
        "row_type": "cell", "cell_id": cell["cell_id"], "seed": cell["seed"], "task_id": "",
        "checkpoint": cell["checkpoint"], "live_arrows": cell["live_arrows"],
        "pc_success": pc_success if pc_success is not None else "", "successes": "",
        "episodes": episodes if episodes is not None else "", "eval_info": eval_info.as_posix() if eval_info.exists() else "",
        "output_dir": output_dir.as_posix(), "contrast": "",
    }


def _extract_task_rows(cell: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_record in info["per_task"]:
        successes = task_record["metrics"]["successes"]
        rows.append(
            {
                "row_type": "task", "cell_id": cell["cell_id"], "seed": cell["seed"],
                "task_id": task_record["task_id"], "checkpoint": cell["checkpoint"],
                "live_arrows": cell["live_arrows"], "pc_success": 100.0 * sum(successes) / len(successes),
                "successes": sum(successes), "episodes": len(successes),
                "eval_info": str(Path(cell["output_dir"]) / "eval_info.json"),
                "output_dir": cell["output_dir"], "contrast": "",
            }
        )
    return rows


def _build_contrast_rows(
    cell_infos: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only live-arrow minus no-arrow success-rate contrasts."""
    values: dict[tuple[int, int, str], float] = {}
    for (seed, cell_id), (_cell, info) in cell_infos.items():
        for task_record in info["per_task"]:
            successes = task_record["metrics"]["successes"]
            values[(seed, task_record["task_id"], cell_id)] = 100.0 * sum(successes) / len(successes)

    rows: list[dict[str, Any]] = []

    def add(seed: int | str, task_id: int | str, value: float) -> None:
        rows.append(
            {
                "row_type": "contrast", "cell_id": "", "seed": seed, "task_id": task_id,
                "checkpoint": "", "live_arrows": "", "pc_success": value,
                "successes": "", "episodes": "", "eval_info": "",
                "output_dir": manifest["output_root"], "contrast": manifest["contrast"],
            }
        )

    per_task_values: dict[int, list[float]] = {}
    for seed in manifest["seeds"]:
        for task_id in manifest["tasks"]:
            arrow_cell = manifest["cells"][0]["cell_id"]
            no_arrow_cell = manifest["cells"][1]["cell_id"]
            value = values[(seed, task_id, arrow_cell)] - values[(seed, task_id, no_arrow_cell)]
            add(seed, task_id, value)
            per_task_values.setdefault(task_id, []).append(value)
    for task_id in manifest["tasks"]:
        add("aggregate", task_id, sum(per_task_values[task_id]) / len(per_task_values[task_id]))
    all_values = [value for values_for_task in per_task_values.values() for value in values_for_task]
    add("aggregate", "all", sum(all_values) / len(all_values))
    return rows


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
    parser.add_argument("--seeds", required=True, help="comma-separated or JSON integer seeds")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--evaluation-scope", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="comma-separated task IDs; scope contract supplies the default",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="no_arrow_treatment")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args = parse_args(argv)
    profile = get_profile(args.profile)
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        seeds = parse_int_list(args.seeds)
        task_ids = (
            parse_int_list(args.task_ids)
            if args.task_ids is not None
            else list(_evaluation_contract(args.evaluation_scope, profile)["tasks"])
        )
        manifest = build_manifest(
            adapter_checkpoint=args.adapter_checkpoint,
            seeds=seeds,
            tasks=task_ids,
            episodes=args.episodes,
            batch_size=args.batch_size,
            device=args.device,
            videos=args.videos,
            max_videos=args.max_videos,
            training_manifest=args.training_manifest,
            output_root=output_root,
            profile_name=profile.name,
            evaluation_scope=args.evaluation_scope,
        )
        validate_existing_outputs(output_root, manifest)
        validate_training_manifest(Path(args.training_manifest).expanduser().resolve(), manifest, profile=profile)
        write_immutable_manifest(output_root / profile.manifest_filename, manifest)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {profile.name} evaluation preflight failed: {exc}")
        return 1

    script_dir = Path(__file__).resolve().parent
    rows: list[dict[str, Any]] = []
    cell_infos: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for cell in manifest["cells"]:
        return_code = _run_cell(cell, args, script_dir, profile, task_ids)
        eval_path = Path(cell["output_dir"]) / "eval_info.json"
        if return_code != 0:
            rows.append(_extract_summary(cell))
            _write_summary(output_root / profile.summary_filename, rows)
            print(f"ERROR: cell {cell['cell_id']} seed {cell['seed']} failed ({return_code})")
            return return_code
        try:
            info = validate_eval_info(eval_path, manifest)
            validate_randomization_audit(Path(cell["output_dir"]), manifest)
        except ValueError as exc:
            rows.append(_extract_summary(cell))
            _write_summary(output_root / profile.summary_filename, rows)
            print(f"ERROR: {profile.name} cell validation failed for {cell['cell_id']} seed {cell['seed']}: {exc}")
            return 1
        cell_infos[(cell["seed"], cell["cell_id"])] = (cell, info)
        rows.append(_extract_summary(cell))
        rows.extend(_extract_task_rows(cell, info))
    rows.extend(_build_contrast_rows(cell_infos, manifest))
    _write_summary(output_root / profile.summary_filename, rows)
    print(f"{profile.name} adapter arrow-effect evaluation complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
