"""Guarded Pi0.5 training command builder and entrypoint."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .contracts import PI05_ARTIFACT, PI05_TRAINING, micro_steps_for
from .preflight import build_receipt


def build_command(
    *,
    python_executable: str,
    dataset_root: str,
    output_root: str,
    checkpoint: str,
    device: str,
    micro_steps: int | None = None,
) -> list[str]:
    """Build the explicit Pi0.5 trainer command without executing it.

    LeRobot 0.5.2 has no CLI/config field for gradient accumulation and its
    stock loop steps the optimizer on every batch.  The checked-in
    ``trainer`` wrapper owns accumulation semantics and delegates the rest of
    the run to that pinned LeRobot implementation.  ``--steps`` therefore
    remains the number of micro-batches, while ``--save_freq`` is converted to
    the same unit.
    """
    if micro_steps is None:
        micro_steps = micro_steps_for(
            PI05_TRAINING.timesteps,
            PI05_TRAINING.epochs,
            PI05_TRAINING.effective_batch_size,
            PI05_TRAINING.gradient_accumulation_steps,
        )
    if micro_steps % PI05_TRAINING.gradient_accumulation_steps != 0:
        raise ValueError("micro_steps must be divisible by gradient_accumulation_steps")
    return [
        python_executable,
        "-m",
        "vla_benchmarking.libero.finetuned_vlas.pi05.trainer",
        f"--gradient_accumulation_steps={PI05_TRAINING.gradient_accumulation_steps}",
        f"--policy.path={checkpoint}",
        f"--dataset.repo_id=local/libero_spatial_no_arrows",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_root}",
        f"--steps={micro_steps}",
        f"--save_freq={PI05_TRAINING.checkpoint_frequency * PI05_TRAINING.gradient_accumulation_steps}",
        f"--batch_size={PI05_TRAINING.microbatch_size}",
        f"--policy.device={device}",
        "--policy.dtype=bfloat16",
        "--policy.n_action_steps=50",
        "--policy.chunk_size=50",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=true",
        # These are current PI05Config fields in the pinned trainer.  They
        # intentionally stay under ``policy``; top-level optimizer fields
        # would be ignored while the policy preset remains enabled.
        f"--policy.optimizer_lr={PI05_TRAINING.learning_rate}",
        f"--policy.scheduler_warmup_steps={PI05_TRAINING.warmup_updates}",
        f"--policy.optimizer_weight_decay={PI05_TRAINING.weight_decay}",
        f"--policy.optimizer_grad_clip_norm={PI05_TRAINING.grad_clip_norm}",
        f"--seed={PI05_TRAINING.seed}",
        "--eval_freq=0",
        "--policy.push_to_hub=false",
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
    parser.add_argument("--checkpoint", default=PI05_ARTIFACT.model_id)
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Public parser seam used by smoke tests without importing LeRobot."""
    return build_parser().parse_args(argv)


def two_update_smoke(step_fn: Callable[[int], Mapping[str, float]]) -> dict[str, float | int]:
    """Run two injected trainer updates without downloading a checkpoint.

    This deliberately exercises only the update/metric boundary.  A real
    trainer can inject a one-batch function while tests use a tiny fake policy.
    """
    losses: list[float] = []
    for index in range(2):
        metrics = dict(step_fn(index))
        loss = float(metrics.get("loss", float("nan")))
        if not loss == loss or loss in (float("inf"), float("-inf")):
            raise ValueError("smoke update returned a non-finite loss")
        losses.append(loss)
    return {"updates": 2, "first_loss": losses[0], "last_loss": losses[-1]}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = str(args.checkpoint_path) if args.checkpoint_path is not None else args.checkpoint
    if not args.manifest or not args.checkpoint_revision:
        raise SystemExit("--manifest and --checkpoint-revision are required for preflight/execute/print-command")
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
        micro_steps=int(receipt["training"]["micro_steps"]),
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
        raise SystemExit("Pi0.5 execution requires --checkpoint-path for the resolved immutable base checkpoint")
    # --execute is intentionally the only path that can spawn the trainer, and
    # it always performs the provenance preflight immediately beforehand.
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
