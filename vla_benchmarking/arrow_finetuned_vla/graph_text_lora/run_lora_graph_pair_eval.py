#!/usr/bin/env python3
"""Sealed pilot evaluation for the graph-text SmolVLA adapter.

This is intentionally a two-cell evaluator: the same graph-trained adapter is
run with target-centric text context and with standard task text.  Both cells
are guaranteed to have no visual arrows.  The arrow-graph dataset is a
preparation artifact and is not an evaluation condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Bootstrap the repository root before organized-package imports so direct
# script launches work with an empty ambient PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vla_benchmarking.shared.config import LEROBOT_CAMERA_KEYS, DEFAULT_CAMERAS, RANDOMIZATION_DIMENSIONS, task_randomization_dimensions
from vla_benchmarking.tools.prompt_audit import validate_serialized_graph_preprocessor
from vla_benchmarking.arrow_finetuned_vla.workflows.adapter_audit import audit_adapter_checkpoint, load_expected_inventory
from vla_benchmarking.arrow_finetuned_vla.workflows.run_lora_2x2_eval import TASK_IDS, validate_eval_info, validate_randomization_audit
from vla_benchmarking.evaluation.randomization_contract import randomization_config_payload
from vla_benchmarking.evaluation.scene_graph_formats import GRAPH_TOKENIZER_CONTRACT, format_target_natural_v1
from vla_benchmarking.arrow_finetuned_vla.workflows.run_lora_no_arrow_pair_eval import SEALED_REVISION

TRAINING_CAMERAS = ",".join(LEROBOT_CAMERA_KEYS)
RAW_TRAINING_CAMERAS = ",".join(DEFAULT_CAMERAS)
CELL_IDS = ("graph_trained_graph_context", "graph_trained_standard")
MANIFEST_FILENAME = "graph_trained_text_pair_manifest.json"
SUMMARY_FILENAME = "graph_trained_text_pair_summary.csv"
EXPERIMENT = "smolvla_lora_graph_treatment_text_context_2cell_pilot"
TRAINING_EXPERIMENT = "smolvla_lora_graph_treatment_training"
SEED = 1000
EPISODES = 10
EVALUATION_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
GRAPH_EXTRACTOR_PATH = EVALUATION_ROOT / "graph_relation_extractor.py"
GRAPH_FORMATTER_PATH = EVALUATION_ROOT / "scene_graph_formats.py"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _adapter(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    artifact = path / "adapter_model.safetensors"
    if path.name != "pretrained_model" or not artifact.is_file() or not artifact.stat().st_size:
        raise ValueError(f"graph adapter must be a non-empty pretrained_model directory: {path}")
    return path


def _training_provenance(path: Path, adapter: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"graph training manifest is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_sha256"):
        body = dict(data); body.pop("manifest_sha256", None)
        if data["manifest_sha256"] != _json_hash(body):
            raise ValueError("graph training manifest self-digest is invalid")
    if data.get("experiment") != TRAINING_EXPERIMENT or data.get("training_variant") != "graph_treatment":
        raise ValueError("training manifest is not the graph_treatment lineage")
    if data.get("dataset_variant") != "graph_treatment" or data.get("trained_on_visual_condition") != "no_arrows":
        raise ValueError("graph training manifest has invalid dataset/visual lineage")
    if data.get("base_policy_revision") != SEALED_REVISION:
        raise ValueError("graph training manifest base policy revision is not sealed")
    if data.get("tokenizer_contract_sha256") != _json_hash(GRAPH_TOKENIZER_CONTRACT):
        raise ValueError("graph training manifest tokenizer contract digest is not canonical")
    audit = data.get("graph_tokenizer_audit", {})
    audit_path = Path(audit.get("path", "")).expanduser().resolve()
    if not audit_path.is_file() or audit.get("sha256") != _sha256(audit_path):
        raise ValueError("graph tokenizer audit is missing or changed")
    recorded = data.get("graph_treatment_adapter", {}).get("path")
    recorded_path = Path(recorded).expanduser().resolve() if recorded else None
    if recorded_path is not None and recorded_path.name == "adapter_model.safetensors":
        recorded_path = recorded_path.parent
    if recorded_path != adapter:
        raise ValueError("evaluation adapter is not graph_treatment_adapter from the training manifest")
    if data.get("graph_treatment_adapter", {}).get("sha256") != _sha256(adapter / "adapter_model.safetensors"):
        raise ValueError("graph adapter artifact hash changed")
    pair = Path(data.get("pair_manifest", "")).expanduser().resolve()
    sentinel = Path(data.get("pair_sentinel", "")).expanduser().resolve()
    if not pair.is_file() or data.get("pair_manifest_sha256") != _sha256(pair):
        raise ValueError("graph pair manifest bytes or recorded digest changed")
    if not sentinel.is_file() or data.get("pair_sentinel_sha256") != _sha256(sentinel):
        raise ValueError("graph pair sentinel bytes or recorded digest changed")
    try:
        pair_data = json.loads(pair.read_text(encoding="utf-8"))
        sentinel_data = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("graph pair manifest/sentinel is unreadable") from exc
    expected_pair_kind = "sealed_lora_graph_treatment_arrow_graph_treatment"
    if pair_data.get("pair_kind") != expected_pair_kind or sentinel_data.get("pair_kind") != expected_pair_kind:
        raise ValueError("graph pair manifest lineage is invalid")
    for key in ("graph_contract_sha256", "tokenizer_contract_sha256", "graph_formatter_sha256", "graph_extractor_sha256"):
        if not isinstance(pair_data.get(key), str) or data.get(key) != pair_data.get(key):
            raise ValueError(f"graph {key} is not bound to the sealed pair manifest")
        if sentinel_data.get(key) != pair_data.get(key):
            raise ValueError(f"graph pair sentinel {key} differs from the sealed pair manifest")
    if pair_data.get("graph_formatter_sha256") != _sha256(GRAPH_FORMATTER_PATH):
        raise ValueError("scene-graph formatter has drifted since graph-pair conversion")
    if pair_data.get("graph_extractor_sha256") != _sha256(GRAPH_EXTRACTOR_PATH):
        raise ValueError("graph relation extractor has drifted since graph-pair conversion")
    for key in ("graph_contract", "tokenizer_contract", "visual_contract", "storage_contract"):
        if key in pair_data and sentinel_data.get(key) != pair_data.get(key):
            raise ValueError(f"graph pair sentinel {key} differs from the sealed pair manifest")
    oracle = (pair_data.get("graph_contract") or {}).get("oracle_disclosure")
    required_oracle = {
        "signal_source": "MuJoCo/HDF5 ground-truth",
        "target_identity_oracle": True,
        "visibility": "projectable_bbox_not_occlusion_aware",
        "privileged_simulator_oracle": True,
        "future_information": False,
        "action_labels": False,
        "reward_labels": False,
        "success_labels": False,
    }
    if oracle != required_oracle or data.get("graph_oracle_disclosure") != required_oracle:
        raise ValueError("graph oracle disclosure is missing or not bound to the sealed pair")
    if sentinel_data.get("manifest_sha256") != _sha256(pair):
        raise ValueError("graph pair sentinel is not bound to the recorded pair manifest")
    if data.get("pair_kind") != pair_data.get("pair_kind"):
        raise ValueError("training manifest graph pair kind differs from sealed pair manifest")
    # Validate the checkpoint only after lineage bytes have been authenticated;
    # this preserves clear fail-fast errors for a tampered pair/sentinel while
    # still requiring a real action-side adapter before any evaluation runs.
    adapter_audit = data.get("adapter_audit", {})
    adapter_audit_path = Path(adapter_audit.get("path", "")).expanduser().resolve()
    if adapter_audit_path.parent != adapter or not adapter_audit_path.is_file():
        raise ValueError("graph action-side LoRA audit is missing or outside the checkpoint")
    if adapter_audit.get("sha256") != _sha256(adapter_audit_path):
        raise ValueError("graph action-side LoRA audit digest changed")
    expected_ref = data.get("expected_adapter_inventory", {})
    expected_path = Path(expected_ref.get("path", "")).expanduser().resolve()
    if not expected_path.is_file() or expected_ref.get("sha256") != _sha256(expected_path):
        raise ValueError("sealed expected live-policy LoRA inventory is missing or changed")
    try:
        expected_inventory = load_expected_inventory(expected_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"expected live-policy LoRA inventory failed validation: {exc}") from exc
    authenticated_base = Path(data.get("base_policy", "")).expanduser().resolve()
    if expected_inventory.get("base_policy") != str(authenticated_base):
        raise ValueError("expected live-policy LoRA inventory is for a different base policy")
    if expected_inventory.get("base_policy_revision") not in (None, SEALED_REVISION):
        raise ValueError("expected live-policy LoRA inventory base revision is not sealed")
    try:
        live_adapter_audit = audit_adapter_checkpoint(
            adapter,
            require_nonzero_lora_b=True,
            expected_inventory=expected_inventory,
            require_expected_inventory=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"graph action-side LoRA audit failed: {exc}") from exc
    recorded_adapter_audit = json.loads(adapter_audit_path.read_text(encoding="utf-8"))
    if recorded_adapter_audit != live_adapter_audit:
        raise ValueError("graph action-side LoRA audit evidence differs from the live checkpoint")
    if live_adapter_audit.get("expected_inventory_sha256") != expected_inventory.get("inventory_sha256"):
        raise ValueError("graph checkpoint audit is not bound to the expected live-policy inventory")
    if data.get("checkpoint_tree_sha256") != live_adapter_audit.get("checkpoint_tree_sha256") or data.get("checkpoint_inventory") != live_adapter_audit.get("checkpoint_inventory"):
        raise ValueError("graph checkpoint consumed-file inventory differs from the sealed training manifest")
    # Resume/evaluation consumes the complete numeric checkpoint root, not just
    # adapter files: training_state contains optimizer/scheduler/RNG state and
    # must be lineage-bound even though it is not loaded for rollout.
    checkpoint_root = adapter.parent
    if not checkpoint_root.name.isdigit() or not (checkpoint_root / "training_state").is_dir():
        raise ValueError("graph checkpoint root lacks the numeric step/training_state layout required for provenance")
    state_files = [candidate for candidate in (checkpoint_root / "training_state").rglob("*") if candidate.is_file()]
    if not state_files or any(candidate.is_symlink() for candidate in state_files):
        raise ValueError("graph checkpoint training_state is missing or contains symlinks")
    state_names = {candidate.name.lower() for candidate in state_files}
    if not (
        any(re.search(r"(^|[._-])optimizer([._-]|$)", name) for name in state_names)
        and any(re.search(r"(^|[._-])scheduler([._-]|$)", name) for name in state_names)
        and any("random_state" in name or "rng" in name for name in state_names)
    ):
        raise ValueError("graph checkpoint training_state lacks optimizer/scheduler/RNG artifacts")
    checkpoint_root_inventory = _tree_inventory(checkpoint_root)
    if data.get("checkpoint_root_inventory") != checkpoint_root_inventory or data.get("checkpoint_root_tree_sha256") != _json_hash(checkpoint_root_inventory):
        raise ValueError("graph checkpoint root inventory differs from the sealed training manifest")
    try:
        adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("graph adapter_config.json is unreadable") from exc
    base_policy_for_adapter = Path(data.get("base_policy", "")).expanduser().resolve()
    base_name = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_name, str) or Path(base_name).expanduser().resolve() != base_policy_for_adapter:
        raise ValueError("graph adapter base_model_name_or_path is not the authenticated base policy")
    if data.get("adapter_config_sha256") != _sha256(adapter / "adapter_config.json"):
        raise ValueError("graph adapter_config.json digest changed")
    flags = data.get("flags", {})
    expected = {"steps": 29190, "save_freq": 1946, "batch_size": 32, "seed": 1000, "peft_r": 16, "tokenizer_max_length": 96}
    if any(int(flags.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("graph training flags are not the sealed 15-epoch LoRA condition")
    if data.get("trained_on_text_condition") != "target_natural_v1":
        raise ValueError("graph training manifest lacks target_natural_v1 provenance")
    base_policy = Path(data.get("base_policy", "")).expanduser().resolve()
    # This is the checkpoint's effective processor, not a generic Hub
    # AutoProcessor.  Require the exact local graph96 asset on the compute
    # runtime and every checkpoint reload.
    validate_serialized_graph_preprocessor(base_policy, require_runtime=True, require_snapshot_manifest=True)
    checkpoint_policy = adapter
    # Checkpoint directories carry their own serialized processor/tokenizer
    # evidence, but not the base-policy tree inventory (their adapter files
    # necessarily differ).  Runtime validation still requires the local pinned
    # asset and exact 96-token step.
    validate_serialized_graph_preprocessor(checkpoint_policy, require_runtime=True, require_snapshot_manifest=False)
    return data


def _build_cells(adapter: Path, root: Path) -> list[dict[str, Any]]:
    return [
        {"cell_id": CELL_IDS[0], "seed": SEED, "checkpoint": str(adapter), "context_mode": "scene_graph", "context_format": "target_natural_v1", "output_dir": str(root / f"seed_{SEED}" / CELL_IDS[0])},
        {"cell_id": CELL_IDS[1], "seed": SEED, "checkpoint": str(adapter), "context_mode": "standard", "context_format": "standard", "output_dir": str(root / f"seed_{SEED}" / CELL_IDS[1])},
    ]


def build_manifest(*, adapter_checkpoint: str, training_manifest: Path, output_root: Path, videos: bool, max_videos: int) -> dict[str, Any]:
    adapter = _adapter(adapter_checkpoint)
    if not training_manifest.is_file():
        raise ValueError(f"training manifest is missing: {training_manifest}")
    data = _training_provenance(training_manifest.resolve(), adapter)
    root = output_root.expanduser().resolve()
    dimensions = {str(task): task_randomization_dimensions(task) for task in TASK_IDS}
    config = randomization_config_payload()
    manifest = {
        "schema_version": 1, "experiment": EXPERIMENT, "status": "pilot",
        "pilot": True, "confirmatory": False,
        "confirmatory_required_for_paper_claims": True,
        "future_confirmatory_requirements": {"training_seeds": [1000, 1001, 1002], "eval_master_seeds": list(range(2000, 2050)), "minimum_episodes_per_task": 50, "minimum_independent_training_seeds": 3, "task_stratified_results": True, "interaction_effect_and_95_percent_ci": True},
        "adapter_checkpoint": str(adapter), "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "adapter_audit": data["adapter_audit"],
        "expected_adapter_inventory": data["expected_adapter_inventory"],
        "adapter_config_sha256": data["adapter_config_sha256"],
        "checkpoint_tree_sha256": data["checkpoint_tree_sha256"],
        "checkpoint_inventory": data["checkpoint_inventory"],
        "checkpoint_root_tree_sha256": data["checkpoint_root_tree_sha256"],
        "checkpoint_root_inventory": data["checkpoint_root_inventory"],
        "training_manifest": str(training_manifest.resolve()), "training_manifest_sha256": _sha256(training_manifest),
        "training_variant": "graph_treatment", "trained_on_visual_condition": "no_arrows", "trained_on_text_condition": "target_natural_v1",
        "tasks": list(TASK_IDS), "seeds": [SEED], "episodes": EPISODES, "batch_size": 1, "randomize_scenes": True,
        "camera_name": TRAINING_CAMERAS, "raw_camera_names": RAW_TRAINING_CAMERAS, "observation_height": 256, "observation_width": 256,
        "tokenizer_max_length": 96, "tokenizer_processor_required": "sealed_local_graph96", "visual_condition": "none",
        "tokenizer_contract_sha256": data["tokenizer_contract_sha256"],
        "graph_contract_sha256": data["graph_contract_sha256"],
        "graph_formatter_sha256": data["graph_formatter_sha256"],
        "graph_extractor_sha256": data["graph_extractor_sha256"],
        "graph_oracle_disclosure": data.get("graph_oracle_disclosure"),
        "randomization_dimensions": dimensions, "randomization_dimension_names": list(RANDOMIZATION_DIMENSIONS),
        "randomization_config": config, "randomization_config_sha256": _json_hash(config), "output_root": str(root),
        "contrast": "graph_context_effect_pp", "cells": _build_cells(adapter, root),
        "arrow_graph_evaluation": "blocked_prepare_only",
    }
    if any(not values.get("object_removal") for values in dimensions.values()):
        raise ValueError("all graph eval tasks must retain object_removal")
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["manifest_sha256"] = _json_hash(manifest)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"immutable graph evaluation manifest differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def _run_cell(cell: dict[str, Any], args: argparse.Namespace, script_dir: Path) -> int:
    out = Path(cell["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    marker = out / "cell_manifest.json"
    expected = {key: cell[key] for key in ("cell_id", "seed", "checkpoint", "context_mode", "context_format", "output_dir")}
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"stale graph cell marker: {marker}")
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = os.environ.copy()
    tokenizer_path = Path(cell["checkpoint"]) / "tokenizer"
    if not tokenizer_path.is_dir():
        raise ValueError(f"graph checkpoint local tokenizer is missing: {tokenizer_path}")
    env.update({"PYTHONUNBUFFERED": "1", "TRAINING_PROFILE": "graph_treatment", "PROFILE": "graph_treatment", "MODELS": cell["checkpoint"], "TOKENIZER_MODEL": str(tokenizer_path), "TASK_IDS": json.dumps(list(TASK_IDS)), "N_EPISODES": str(args.episodes), "BATCH_SIZE": "1", "SEED": str(SEED), "DEVICE": args.device, "CONTEXT_MODE": cell["context_mode"], "CONTEXT_FORMAT": cell["context_format"], "VISUAL_CONDITION": "none", "VISUAL_ARROWS": "0", "RANDOMIZE_SCENES": "1", "N_ACTION_STEPS": "checkpoint", "MAX_EPISODES_RENDERED": str(args.max_videos if args.videos else 0), "RENDER_MODE": "rgb_array" if args.videos else "none"})
    cmd = [sys.executable, str(EVALUATION_ROOT / "run_lerobot_eval_with_context.py"), "--eval.use_async_envs=false", f"--output_dir={out}", f"--policy.path={cell['checkpoint']}", "--env.task_ids=" + json.dumps(list(TASK_IDS), separators=(",", ":")), "--env.camera_name=" + TRAINING_CAMERAS, "--env.observation_height=256", "--env.observation_width=256"]
    with (out / "eval_stdout_stderr.log").open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=script_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def _validate_prompt_audit(cell: dict[str, Any]) -> str:
    path = Path(cell["output_dir"]) / "prompt_audit.jsonl"
    if not path.is_file():
        raise ValueError(f"missing prompt audit: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"prompt audit is empty: {path}")
    task_ids = set()
    for record in records:
        task_ids.add(record.get("task_id"))
        if cell["context_mode"] == "scene_graph":
            if record.get("context_mode") != "scene_graph" or record.get("format_name") != "target_natural_v1":
                raise ValueError("graph prompt audit contains a non-graph context record")
            if not isinstance(record.get("triplets"), list) or not isinstance(record.get("triplet_sha256"), str):
                raise ValueError("graph prompt audit lacks exact triplet evidence")
            task_text = record.get("task_text")
            if not isinstance(task_text, str) or not isinstance(record.get("raw_prompt"), str):
                raise ValueError("graph prompt audit lacks the source task/prompt bytes")
            try:
                canonical_prompt = format_target_natural_v1(
                    task_text,
                    [tuple(item) for item in record["triplets"]],
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("graph prompt audit contains malformed triplet evidence") from exc
            if record["raw_prompt"] != canonical_prompt:
                raise ValueError("graph prompt audit prompt bytes do not match the sealed formatter")
        else:
            if record.get("context_mode") != "standard" or record.get("format_name") != "standard":
                raise ValueError("standard prompt audit contains graph context")
            if record.get("number_of_relations_generated") != 0 or record.get("number_of_relations_retained") != 0 or record.get("triplets") != []:
                raise ValueError("standard prompt audit unexpectedly contains graph relations")
        if record.get("token_audit_enabled") is not True or record.get("truncation_occurred") is not False:
            raise ValueError("prompt audit is missing strict no-truncation evidence")
        if not isinstance(record.get("input_ids"), list) or not isinstance(record.get("attention_mask"), list):
            raise ValueError("prompt audit lacks effective input ids/mask")
        if not isinstance(record.get("input_ids_attention_mask_sha256"), str):
            raise ValueError("prompt audit lacks input id/mask digest")
        if not isinstance(record.get("tokenizer_vocab_sha256"), str) or not isinstance(record.get("tokenizer_tree_sha256"), str):
            raise ValueError("prompt audit lacks local tokenizer provenance")
    if not set(range(10)).issubset({int(value) for value in task_ids if isinstance(value, int)}):
        raise ValueError("prompt audit does not cover all ten tasks")
    return _sha256(path)


def _paired_reset_audit(cells: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    parsed = []
    for cell in cells:
        path = Path(cell["output_dir"]) / "randomization_audit.jsonl"
        if not path.is_file(): raise ValueError(f"missing randomization audit: {path}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_randomization_audit(Path(cell["output_dir"]), manifest)
        parsed.append(records)
    if len(parsed) != 2 or len(parsed[0]) != len(parsed[1]):
        raise ValueError("graph cells do not have paired reset records")
    def reset_shape(record: dict[str, Any]) -> Any:
        details = record.get("details", {})
        # Graph prompt triplets are intentionally absent from the standard
        # cell. Compare only reset/randomization evidence shared by both
        # modalities; graph-specific prompt evidence is checked separately.
        state_hash = details.get("sim_state_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            raise ValueError("randomization audit lacks a valid post-randomization simulator-state hash")
        init_state = details.get("init_state")
        if not isinstance(init_state, dict) or not isinstance(init_state.get("selected_index"), int) or not isinstance(init_state.get("selected_row_sha256"), str) or len(init_state["selected_row_sha256"]) != 64:
            raise ValueError("randomization audit lacks exact selected init-state evidence")
        compensation = details.get("terminal_reset_compensation")
        if compensation is not None:
            if (
                not isinstance(compensation, dict)
                or compensation.get("detected") is not True
                or compensation.get("restored_to") != compensation.get("counter_before")
                or not isinstance(compensation.get("stride"), int)
                or compensation["stride"] <= 0
            ):
                raise ValueError("randomization audit contains invalid terminal-reset compensation evidence")
        return {key: details.get(key) for key in ("removed", "projection", "protected", "layout", "swaps", "sim_state_sha256", "init_state")}
    if [(r.get("task_id"), r.get("env_index"), r.get("reset_sequence"), reset_shape(r)) for r in parsed[0]] != [(r.get("task_id"), r.get("env_index"), r.get("reset_sequence"), reset_shape(r)) for r in parsed[1]]:
        raise ValueError("graph text/no-context cells did not use identical reset/randomization identities")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["cell_id", "task_id", "seed", "successes", "episodes", "success_rate", "output_dir", "prompt_audit_sha256"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def validate_results_sentinel(manifest_path: str | Path) -> None:
    """Fail closed if finalized graph results or any cell evidence changed."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    sentinel_path = Path(manifest.get("results_sentinel", "")).expanduser().resolve()
    if not sentinel_path.is_file():
        raise ValueError("graph results sentinel is missing")
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if sentinel.get("manifest") != str(manifest_file) or sentinel.get("manifest_sha256") != _sha256(manifest_file):
        raise ValueError("graph results sentinel is not bound to the final manifest")
    summary = Path(sentinel.get("summary", "")).expanduser().resolve()
    if sentinel.get("summary_sha256") != _sha256(summary):
        raise ValueError("graph results summary changed after finalization")
    by_id = {cell.get("cell_id"): cell for cell in sentinel.get("cells", [])}
    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id")
        if cell_id not in by_id:
            raise ValueError(f"graph results sentinel lacks cell {cell_id}")
        root = Path(cell["output_dir"]).expanduser().resolve()
        if by_id[cell_id].get("artifact_tree_sha256") != _json_hash(_tree_inventory(root)):
            raise ValueError(f"graph cell evidence changed after finalization: {cell_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True); parser.add_argument("--training-manifest", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--seeds", default="1000")
    parser.add_argument("--episodes", type=int, default=EPISODES); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--device", default="cuda"); parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=False); parser.add_argument("--max-videos", type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes != EPISODES or args.batch_size != 1 or args.seeds.replace("[", "").replace("]", "").strip() != "1000": print("ERROR: graph pilot evaluation is sealed to seed 1000, 10 episodes/task, and batch_size=1", file=sys.stderr); return 2
    try:
        root = Path(args.output_root).expanduser().resolve(); training = Path(args.training_manifest).expanduser().resolve()
        manifest = build_manifest(adapter_checkpoint=args.adapter_checkpoint, training_manifest=training, output_root=root, videos=args.videos, max_videos=args.max_videos)
        finalized_manifest = root / MANIFEST_FILENAME
        finalized_sentinel = root / "graph_trained_text_pair_results_verified.json"
        if finalized_manifest.is_file() or finalized_sentinel.is_file():
            if not finalized_manifest.is_file() or not finalized_sentinel.is_file():
                raise ValueError("graph evaluation has an incomplete finalization pair")
            validate_results_sentinel(finalized_manifest)
            print(f"graph evaluation already finalized and verified: {root}")
            return 0
        for cell in manifest["cells"]:
            cell_root = Path(cell["output_dir"])
            if cell_root.is_dir() and any(cell_root.iterdir()):
                raise ValueError(f"partial graph evaluation output exists; explicit resume is unsupported: {cell_root}")
        script_dir = Path(__file__).resolve().parent; rows = []
        for cell in manifest["cells"]:
            code = _run_cell(cell, args, script_dir)
            if code: print(f"ERROR: graph cell failed: {cell['cell_id']} ({code})", file=sys.stderr); return code
            info = validate_eval_info(Path(cell["output_dir"]) / "eval_info.json", manifest)
            cell["prompt_audit"] = str((Path(cell["output_dir"]) / "prompt_audit.jsonl").resolve())
            cell["prompt_audit_sha256"] = _validate_prompt_audit(cell)
            for task in info["per_task"]:
                successes = task["metrics"]["successes"]
                rows.append({"cell_id": cell["cell_id"], "task_id": task["task_id"], "seed": SEED, "successes": sum(successes), "episodes": len(successes), "success_rate": 100 * sum(successes) / len(successes), "output_dir": cell["output_dir"], "prompt_audit_sha256": cell["prompt_audit_sha256"]})
        _paired_reset_audit(manifest["cells"], manifest)
        for cell in manifest["cells"]:
            cell_root = Path(cell["output_dir"])
            for key, filename in (("eval_info", "eval_info.json"), ("randomization_audit", "randomization_audit.jsonl"), ("prompt_audit", "prompt_audit.jsonl")):
                evidence_path = (cell_root / filename).resolve()
                if not evidence_path.is_file():
                    raise ValueError(f"missing final cell evidence: {evidence_path}")
                cell[f"{key}_sha256"] = _sha256(evidence_path)
            cell["artifact_tree"] = _tree_inventory(cell_root)
            cell["artifact_tree_sha256"] = _json_hash(cell["artifact_tree"])
        summary_path = root / SUMMARY_FILENAME
        _write_summary(summary_path, rows)
        manifest["summary"] = {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)}
        manifest["results_sentinel"] = str((root / "graph_trained_text_pair_results_verified.json").resolve())
        _write_manifest(root / MANIFEST_FILENAME, manifest)
        manifest_path = root / MANIFEST_FILENAME
        sentinel = {
            "schema_version": 1,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "summary": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "cells": [
                {"cell_id": cell["cell_id"], "artifact_tree_sha256": cell["artifact_tree_sha256"],
                 "eval_info_sha256": cell["eval_info_sha256"],
                 "randomization_audit_sha256": cell["randomization_audit_sha256"],
                 "prompt_audit_sha256": cell["prompt_audit_sha256"]}
                for cell in manifest["cells"]
            ],
        }
        sentinel_path = root / "graph_trained_text_pair_results_verified.json"
        sentinel_json = json.dumps(sentinel, indent=2, sort_keys=True) + "\n"
        if sentinel_path.exists() and sentinel_path.read_text(encoding="utf-8") != sentinel_json:
            raise ValueError(f"immutable graph results sentinel differs: {sentinel_path}")
        if not sentinel_path.exists():
            sentinel_path.write_text(sentinel_json, encoding="utf-8")
        validate_results_sentinel(manifest_path)
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"ERROR: sealed graph evaluation preflight/validation failed: {exc}", file=sys.stderr); return 1
    print(f"graph text/no-context pilot evaluation complete: {root}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
