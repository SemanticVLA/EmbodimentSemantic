"""CLI entry point for the teacher-free SmolVLA Apprentice job.

This wrapper performs strict transition loading, native LeRobot export, sealed
job construction, and (unless ``--dry-run``) one immutable PEFT training run.
It never invents observations/actions and never resumes or overwrites an
existing dataset or output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .apprentice_training import (
    ApprenticeTrainingConfig,
    SmolVLALoRAConfig,
    build_apprentice_training_job,
    export_apprentice_dataset_native,
    load_apprentice_transitions,
    make_export_request,
    run_apprentice_training_job,
)
from .learning import manifest_for_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a teacher-free SmolVLA Apprentice adapter")
    parser.add_argument("--transitions", required=True, help="persisted On-Call transition archive/view")
    parser.add_argument("--parent-artifact", required=True, help="immutable source archive identity")
    parser.add_argument("--dataset-root", required=True, help="new LeRobot dataset destination")
    parser.add_argument("--base-checkpoint", required=True, help="frozen SmolVLA base path or Hub id")
    parser.add_argument("--base-sha256", required=True, help="frozen base file/tree SHA-256")
    parser.add_argument("--model-revision", required=True, help="pinned SmolVLA model revision")
    parser.add_argument("--processor-revision", required=True, help="pinned SmolVLA processor revision")
    parser.add_argument("--output-dir", required=True, help="new training output directory")
    parser.add_argument("--task-id", dest="task_ids", action="append", help="limit to task id (repeat or comma-separate)")
    parser.add_argument("--reset-id", dest="reset_ids", action="append", help="limit to reset id (repeat or comma-separate)")
    parser.add_argument("--episode-id", dest="episode_ids", action="append", help="limit to episode id (repeat or comma-separate)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="export and print command without training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_apprentice_transitions(
        args.transitions,
        task_ids=args.task_ids,
        reset_ids=args.reset_ids,
        episode_ids=args.episode_ids,
    )
    source_manifest = manifest_for_rows(rows, parent_artifact=args.parent_artifact)
    request = make_export_request(rows, source_manifest, destination=args.dataset_root)
    receipt = export_apprentice_dataset_native(request)
    lora = SmolVLALoRAConfig(base_vla_sha256=args.base_sha256,
                              model_revision=args.model_revision,
                              processor_revision=args.processor_revision)
    training = ApprenticeTrainingConfig(seed=args.seed, epochs=args.epochs, effective_batch_size=args.batch_size)
    job = build_apprentice_training_job(
        receipt, base_checkpoint=args.base_checkpoint, output_dir=args.output_dir,
        lora=lora, training=training, device=args.device,
    )
    if args.dry_run:
        print(json.dumps(job.manifest, indent=2, sort_keys=True))
        return 0
    result = run_apprentice_training_job(job)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())
