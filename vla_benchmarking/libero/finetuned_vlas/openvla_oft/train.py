"""Guarded OpenVLA-OFT training command builder and entrypoint."""

from __future__ import annotations

import argparse
import json
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .contracts import OPENVLA_ARTIFACT, OPENVLA_DATASET_NAME, OPENVLA_TRAINING
from .preflight import build_receipt
from .rlds_builder import register_tfds_builder
from .upstream import validate_upstream_checkout


def build_command(
    *,
    python_executable: str,
    dataset_root: str,
    output_root: str,
    checkpoint: str,
    device: str,
    optimizer_updates: int | None = None,
    upstream_repo: str | Path | None = None,
    upstream_commit: str | None = None,
) -> list[str]:
    """Build the official OpenVLA-OFT finetune command without executing it."""
    del device  # The upstream script selects the runtime through its launcher.
    if optimizer_updates is None:
        optimizer_updates = OPENVLA_TRAINING.optimizer_updates
    entrypoint: str | Path = "vla-scripts/finetune.py"
    if upstream_repo is not None or upstream_commit is not None:
        if upstream_repo is None or upstream_commit is None:
            raise ValueError("upstream_repo and upstream_commit must be supplied together")
        entrypoint = validate_upstream_checkout(upstream_repo, expected_commit=upstream_commit).finetune_entrypoint
    return [
        python_executable,
        str(entrypoint),
        "--vla_path", checkpoint,
        "--data_root_dir", dataset_root,
        "--dataset_name", OPENVLA_DATASET_NAME,
        "--run_root_dir", output_root,
        "--lora_rank", str(OPENVLA_TRAINING.lora_rank),
        "--batch_size", str(OPENVLA_TRAINING.microbatch_size),
        "--grad_accumulation_steps", str(OPENVLA_TRAINING.gradient_accumulation_steps),
        "--learning_rate", str(OPENVLA_TRAINING.learning_rate),
        "--max_steps", str(optimizer_updates),
        "--save_freq", str(OPENVLA_TRAINING.checkpoint_frequency),
        "--use_l1_regression",
        "--use_proprio",
        "--num_images_in_input", "2",
        "--image_aug",
        "--seed", str(OPENVLA_TRAINING.seed),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--print-command", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint", default=OPENVLA_ARTIFACT.model_id)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="where to persist the immutable preflight receipt (default: <output-root>/preflight_receipt.json)",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--upstream-repo", type=Path, help="explicit local pinned OpenVLA-OFT checkout")
    parser.add_argument("--upstream-commit", help="expected pinned OpenVLA-OFT checkout commit")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Public parser seam used by smoke tests without importing OpenVLA."""
    return build_parser().parse_args(argv)


def two_update_smoke(step_fn: Callable[[int], Mapping[str, float]]) -> dict[str, float | int]:
    """Run two injected trainer updates without downloading a checkpoint."""
    losses: list[float] = []
    for index in range(2):
        metrics = dict(step_fn(index))
        loss = float(metrics.get("loss", float("nan")))
        if not loss == loss or loss in (float("inf"), float("-inf")):
            raise ValueError("smoke update returned a non-finite loss")
        losses.append(loss)
    return {"updates": 2, "first_loss": losses[0], "last_loss": losses[-1]}


def registration_loader_trainer_smoke(
    *,
    register_fn: Callable[[], str | object],
    load_batch_fn: Callable[[str], Mapping[str, object]],
    train_step_fn: Callable[[Mapping[str, object]], Mapping[str, float]],
) -> dict[str, object]:
    """Exercise the dataset-registration -> loader -> trainer seam once.

    This is intentionally dependency-light: callers inject a mocked TFDS
    registration, loader, and one-batch trainer step.  It proves that the
    launcher passes the *registered* dataset name through to the loader and
    that the trainer receives a finite loss, without importing TensorFlow,
    loading a checkpoint, or touching a GPU.
    """
    registered_name = str(register_fn())
    if registered_name != OPENVLA_DATASET_NAME:
        raise ValueError(
            f"registration hook returned {registered_name!r}; expected {OPENVLA_DATASET_NAME!r}"
        )
    batch = dict(load_batch_fn(registered_name))
    if not batch:
        raise ValueError("registered RLDS loader returned an empty batch")
    metrics = dict(train_step_fn(batch))
    loss = float(metrics.get("loss", float("nan")))
    if not loss == loss or loss in (float("inf"), float("-inf")):
        raise ValueError("registered RLDS one-batch smoke returned a non-finite loss")
    return {
        "dataset_name": registered_name,
        "batch_keys": sorted(str(key) for key in batch),
        "loss": loss,
    }


def _run_pinned_finetune(command: Sequence[str]) -> None:
    """Run the pinned script after registering the local RLDS builder.

    Keeping execution in-process is deliberate: TFDS's in-memory builder
    registry does not cross a subprocess boundary.  The generated command is
    still the exact upstream ``vla-scripts/finetune.py`` invocation.
    """
    if len(command) < 2 or Path(command[1]).name != "finetune.py":
        subprocess.run(command, check=True)
        return
    register_tfds_builder()
    previous_argv = sys.argv
    upstream_root = str(Path(command[1]).resolve().parents[1])
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, upstream_root)
        sys.argv = list(command[1:])
        runpy.run_path(command[1], run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = str(args.checkpoint_path) if args.checkpoint_path is not None else args.checkpoint
    if not args.manifest or not args.checkpoint_revision:
        raise SystemExit("--manifest and --checkpoint-revision are required for preflight/execute/print-command")
    if not args.print_command and (args.upstream_repo is None or args.upstream_commit is None):
        raise SystemExit("preflight/execute requires --upstream-repo and --upstream-commit")
    if args.upstream_repo is not None or args.upstream_commit is not None:
        if args.upstream_repo is None or args.upstream_commit is None:
            raise SystemExit("--upstream-repo and --upstream-commit must be supplied together")
        validate_upstream_checkout(args.upstream_repo, expected_commit=args.upstream_commit)
    receipt = build_receipt(
        manifest_path=args.manifest,
        checkpoint_revision=args.checkpoint_revision,
        checkpoint_path=args.checkpoint_path,
        python_executable=args.python_executable,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    command = build_command(
        python_executable=args.python_executable,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        checkpoint=checkpoint,
        device=args.device,
        optimizer_updates=int(receipt["training"]["optimizer_updates"]),
        upstream_repo=args.upstream_repo,
        upstream_commit=args.upstream_commit,
    )
    if args.print_command:
        print(shlex.join(command))
        return 0
    receipt_path = args.receipt_output or (Path(args.output_root) / "preflight_receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.checkpoint_path is None:
        raise SystemExit("OpenVLA-OFT execution requires --checkpoint-path for the resolved immutable base checkpoint")
    _run_pinned_finetune(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
