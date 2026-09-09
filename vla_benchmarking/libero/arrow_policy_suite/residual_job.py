"""Executable native Editor/Minimal-Learned residual training bridge.

The bridge deliberately accepts the persisted training-source contract rather
than in-memory records.  It performs no collection, launcher, or runtime
fallback work: a selected variant, immutable source artifact, frozen-base
identity, and explicit training configuration are required before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .contracts import ContractError
from .native_learned_training import (
    NativeResidualTrainingConfig,
    train_editor_from_artifact,
    train_minimal_learned_from_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an immutable Arrow residual checkpoint")
    parser.add_argument("--variant", required=True, choices=("editor", "minimal-learned"),
                        help="exact residual variant to train")
    parser.add_argument("--training-source", "--source", dest="training_source", required=True,
                        help="immutable training_source.v1 JSONL artifact")
    parser.add_argument("--output-checkpoint", "--output", "--checkpoint", dest="output_checkpoint", required=True,
                        help="new checkpoint path; existing checkpoints are never overwritten")
    parser.add_argument("--base-vla-sha256", "--base-sha256", "--frozen-base-sha256", dest="base_vla_sha256", required=True,
                        help="lowercase SHA-256 of the frozen base VLA")
    parser.add_argument("--task-id", action="append", dest="task_ids", type=str,
                        help="restrict training to a task; repeat for multiple tasks")
    parser.add_argument("--reset-id", action="append", dest="reset_ids", type=str,
                        help="restrict training to a reset identity; repeat for multiple resets")
    parser.add_argument("--episode-id", action="append", dest="episode_ids", type=str,
                        help="restrict training to an episode identity; repeat for multiple episodes")
    parser.add_argument("--success-only", action="store_true",
                        help="retain only transitions from successful episodes")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-mode", choices=("residual", "correction_mask"), default="correction_mask")
    return parser


def _output_preflight(path: Path) -> None:
    if path.is_symlink() or path.exists() or Path(str(path) + ".json").exists():
        raise ContractError(f"refusing to overwrite immutable residual checkpoint: {path}")


def _filter_name(args: argparse.Namespace) -> str:
    def values(items: Sequence[str] | None) -> str:
        return "*" if not items else ",".join(sorted(str(item) for item in items))

    return (f"residual_job:{args.variant}:task={values(args.task_ids)}:reset={values(args.reset_ids)}:"
            f"episode={values(args.episode_ids)}:success={bool(args.success_only)}")


def _config(args: argparse.Namespace) -> NativeResidualTrainingConfig:
    return NativeResidualTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        output_mode=args.output_mode,
        target_kind="auto",
    )


def _receipt(args: argparse.Namespace, result: Any) -> dict[str, Any]:
    payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    return {
        "status": "COMPLETED",
        "command": "residual_job",
        "variant": args.variant,
        "training_source": str(Path(args.training_source)),
        "output_checkpoint": str(Path(args.output_checkpoint)),
        "filter_name": _filter_name(args),
        "receipt": payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = Path(args.training_source)
        output = Path(args.output_checkpoint)
        _output_preflight(output)
        config = _config(args)
        filters = {
            "task_ids": args.task_ids,
            "reset_ids": args.reset_ids,
            "episode_ids": args.episode_ids,
            "require_success": bool(args.success_only),
            "filter_name": _filter_name(args),
        }
        if args.variant == "editor":
            result = train_editor_from_artifact(
                source, output_path=output, base_vla_sha256=args.base_vla_sha256,
                config=config, **filters,
            )
        elif args.variant == "minimal-learned":
            result = train_minimal_learned_from_artifact(
                source, output_path=output, base_vla_sha256=args.base_vla_sha256,
                config=config, **filters,
            )
        else:  # argparse choices make this unreachable; keep the boundary explicit.
            raise ContractError(f"unsupported residual variant: {args.variant}")
        print(json.dumps(_receipt(args, result), sort_keys=True, default=str))
        return 0
    except (ContractError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({
            "status": "BLOCKED",
            "command": "residual_job",
            "variant": getattr(args, "variant", None),
            "error": str(exc),
            "runs_launched": False,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
