"""Fail-closed Octo dataset/checkpoint preflight.

This module performs no Hub download, JAX initialization, or training.  It
only validates local provenance and derives the matched optimizer schedule
from the verified completion manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .config import COMMUNITY_EVAL_CONFIG, MATCHED_TRAIN_CONFIG, OctoConfig
from .manifest import completion_manifest_sha256, compute_optimizer_updates, load_and_validate_completion_manifest
from vla_benchmarking.libero.evaluation.policy_adapter import derive_io_receipt, derive_runtime_receipt


def config_for_mode(mode: Literal["community_eval", "matched_train"]) -> OctoConfig:
    if mode == "community_eval":
        return COMMUNITY_EVAL_CONFIG
    if mode == "matched_train":
        return MATCHED_TRAIN_CONFIG
    raise ValueError(f"unsupported Octo mode: {mode}")


def _validate_checkpoint_path(path: str | Path, config: OctoConfig) -> tuple[Path, Path]:
    """Resolve an Octo experiment root and its ``step/default/checkpoint`` leaf.

    Octo's published snapshots are rooted at the experiment directory; the
    nested checkpoint is selected by ``load_pretrained(root, step=...)``.  We
    still accept the historical leaf path at the CLI boundary so old receipts
    remain readable, but normalize both forms to the same root/step evidence.
    """
    target = Path(path).expanduser().resolve()
    subpath = config.checkpoint.checkpoint_subpath
    step = config.checkpoint.step
    if not subpath or step is None:
        raise ValueError("Octo preflight requires an explicit pinned checkpoint subpath")
    if not target.exists():
        raise ValueError(f"Octo checkpoint/experiment path does not exist: {target}")
    leaf_suffix = Path(str(step)) / "default" / "checkpoint"
    if target.name == "checkpoint" and target.parent.name == "default" and target.parent.parent.name == str(step):
        checkpoint = target
        root = target.parent.parent.parent
    elif target.is_file() and target.parent.name == "checkpoint" and target.parent.parent.name == "default" and target.parent.parent.parent.name == str(step):
        # Some Hub snapshots expose a single large checkpoint leaf instead of
        # a directory.  Preserve the file as the hash target while deriving
        # the required experiment root from its published layout.
        checkpoint = target
        root = target.parent.parent.parent.parent
    else:
        root = target
        checkpoint = root / leaf_suffix
    if not root.is_dir():
        raise ValueError(f"Octo experiment root must be a directory: {root}")
    expected_root = config.checkpoint.root_identifier()
    if expected_root:
        expected_parts = tuple(part for part in expected_root.strip("/").split("/") if part)
        actual_parts = root.parts
        if len(actual_parts) < len(expected_parts) or actual_parts[-len(expected_parts):] != expected_parts:
            raise ValueError(
                f"Octo experiment root does not match {config.mode} pinned path suffix: "
                f"expected .../{expected_root}, got {root}"
            )
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise ValueError(f"Octo pinned checkpoint leaf is missing: {checkpoint}")
    # Published community checkpoints must contain files before hashing.  A
    # matched-training placeholder is allowed for local preflight so a run can
    # be prepared before the official base snapshot is mounted.
    if config.mode == "community_eval":
        if checkpoint.is_dir() and not any(checkpoint.rglob("*")):
            raise ValueError(f"Octo checkpoint subtree is empty: {checkpoint}")
        if checkpoint.is_dir() and not (checkpoint / "params").is_file():
            raise ValueError(f"published Octo checkpoint lacks required params file: {checkpoint / 'params'}")
    return root, checkpoint


def _checkpoint_tree_sha256(checkpoint: Path) -> str:
    """Hash the selected immutable checkpoint subtree, including file paths."""

    digest = hashlib.sha256()
    if checkpoint.is_file():
        files = [checkpoint]
        base = checkpoint.parent
    else:
        files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
        base = checkpoint
    if not files:
        raise ValueError(f"Octo checkpoint subtree is empty: {checkpoint}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Octo checkpoint subtree contains a symlink: {path}")
        relative = path.relative_to(base).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _find_dataset_statistics(checkpoint: Path, root: Path | None = None) -> Path:
    """Find only the known metadata locations adjacent to this checkpoint."""

    anchor = checkpoint if checkpoint.is_dir() else checkpoint.parent
    root = root or anchor.parent.parent.parent
    candidates = (anchor / "dataset_statistics.json", anchor.parent / "dataset_statistics.json", anchor.parent.parent / "dataset_statistics.json", root / "dataset_statistics.json")
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise ValueError("community Octo checkpoint lacks dataset_statistics.json in its pinned run metadata")


def _find_count_pair(value: Any) -> tuple[int, int] | None:
    """Extract trajectory/transition counts from versioned Octo stats metadata."""

    episode_keys = {"episode_count", "episodes", "num_episodes", "num_trajectories", "trajectory_count"}
    transition_keys = {"step_count", "steps", "timesteps", "transition_count", "num_transitions", "transitions"}
    if isinstance(value, dict):
        episode_values = [value[key] for key in episode_keys if key in value]
        transition_values = [value[key] for key in transition_keys if key in value]
        if episode_values and transition_values:
            try:
                episodes, transitions = int(episode_values[0]), int(transition_values[0])
                if episodes >= 0 and transitions >= 0:
                    return episodes, transitions
            except (TypeError, ValueError):
                pass
        for child in value.values():
            found = _find_count_pair(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_count_pair(child)
            if found is not None:
                return found
    return None


def _validate_community_checkpoint(checkpoint: Path, config: OctoConfig, root: Path) -> dict[str, Any]:
    stats_path = _find_dataset_statistics(checkpoint, root)
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"community Octo dataset_statistics.json is unreadable: {stats_path}") from exc
    counts = _find_count_pair(stats)
    if counts != (config.episodes, config.timesteps):
        raise ValueError(
            f"community checkpoint dataset statistics must report {(config.episodes, config.timesteps)}, got {counts}"
        )
    stats_digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    return {
        "dataset_manifest": None,
        "dataset_manifest_sha256": None,
        "dataset_statistics_path": str(stats_path),
        "dataset_statistics_sha256": stats_digest,
        "dataset_statistics_counts": {"episodes": counts[0], "transitions": counts[1]},
        "checkpoint_tree_sha256": _checkpoint_tree_sha256(checkpoint),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_closure_paths() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root / "vla_benchmarking/libero/evaluation",
        repo_root / "vla_benchmarking/libero/finetuned_vlas/common",
        repo_root / "vla_benchmarking/libero/finetuned_vlas/octo",
    )


def _provenance_digests(*, require_clean: bool = False) -> dict[str, str]:
    """Bind the native runtime and IO contract to every launch receipt."""

    runtime = derive_runtime_receipt(
        closure_paths=runtime_closure_paths(),
        require_clean=require_clean,
    )
    io = derive_io_receipt({
        "action_horizon": 4, "action_dim": 7, "input_resolution": 256,
        "camera_keys": ["image_primary"], "state_dim": None, "visual_input": "none",
    })
    return {"runtime_sha256": runtime["sha256"], "io_sha256": io["sha256"]}


def run_preflight(*, mode: Literal["community_eval", "matched_train"], dataset_manifest: str | Path | None, checkpoint_path: str | Path) -> dict[str, Any]:
    """Validate one local manifest/checkpoint pair and return immutable evidence."""

    config = config_for_mode(mode)
    config.validate()
    checkpoint_root, checkpoint = _validate_checkpoint_path(checkpoint_path, config)
    if mode == "community_eval":
        if dataset_manifest is not None:
            raise ValueError("community evaluation is bound to checkpoint dataset_statistics, not a caller manifest")
        provenance = _validate_community_checkpoint(checkpoint, config, checkpoint_root)
        transition_count = config.timesteps
    else:
        if dataset_manifest is None:
            raise ValueError("matched training/evaluation requires a completed local dataset manifest")
        manifest = load_and_validate_completion_manifest(dataset_manifest, config=config)
        provenance = {
            "dataset_manifest": str(Path(dataset_manifest).expanduser().resolve()),
            "dataset_manifest_sha256": completion_manifest_sha256(manifest),
        }
        transition_count = int(manifest["fingerprint"]["step_count"])
    optimizer_updates = None
    if mode == "matched_train":
        optimizer_updates = compute_optimizer_updates(
            transition_count=transition_count,
            epochs=config.epochs,
        )
    checkpoint_sha256 = None
    if checkpoint.is_file() or any(checkpoint.rglob("*")):
        checkpoint_sha256 = _checkpoint_tree_sha256(checkpoint)
    result = {
        "schema": "octo_preflight.v1",
        "mode": mode,
        "policy_kind": config.policy_kind,
        "dataset_name": config.dataset_name,
        **provenance,
        "episodes": config.episodes,
        "transitions": transition_count,
        "checkpoint_path": str(checkpoint),
        "checkpoint_root_path": str(checkpoint_root),
        "checkpoint_step": config.checkpoint.step,
        "checkpoint_repository": config.checkpoint.repository,
        "checkpoint_revision": config.checkpoint.revision,
        "checkpoint_sha256": checkpoint_sha256,
        # Backward-compatible spelling for archived receipts.  New plan and
        # adapter contracts consume checkpoint_sha256.
        "checkpoint_tree_sha256": checkpoint_sha256,
        "checkpoint_subpath": config.checkpoint.checkpoint_subpath,
        "action_horizon": config.action_horizon,
        "optimizer_updates": optimizer_updates,
        "preflight": "PASS",
        **_provenance_digests(),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("community_eval", "matched_train"), required=True)
    parser.add_argument("--dataset-manifest", required=False)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--preflight-only", action="store_true", help="validate only; never print a launch command")
    parser.add_argument("--print-command", action="store_true", help="print the native command after validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evidence = run_preflight(
        mode=args.mode,
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    # ``preflight.py`` intentionally does not construct a launch command: the
    # train/eval wrappers own their distinct native entrypoints.
    if args.print_command and not args.preflight_only:
        print("command printing is available from train.py or eval.py after this preflight")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
