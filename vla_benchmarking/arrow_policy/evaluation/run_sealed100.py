"""Run the ArrowStudent policy on the exact archived sealed randomized 100 cells.

The command deliberately accepts no task, seed, suite, camera, or resolution
selectors.  A single invocation owns the immutable matrix and can resume it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from ... import run_arrow_pick_place_eval as episode
from ... import run_arrow_pick_place_matrix as matrix
from .reference_protocol import load_reference_protocol
from .runner import ArrowStudentEpisodeRunner, ArrowStudentRuntime


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help=".../run/results/sealed_randomized")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pinned Stage D checkpoint directory")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-actions", type=int, default=1200)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-motion", action="store_true", help="run all 100 cells after the no-motion preflight")
    mode.add_argument("--preflight-only", action="store_true", help="validate the model/protocol/env and emit no actions")
    parser.add_argument("--resume", action="store_true")
    return parser


def _preflight_env(reference, output_root: Path) -> dict[str, Any]:
    preview_root = output_root.parent.parent / "preflight" / "task_0_seed_1000"
    preview_root.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        variant = matrix._resolve_controller_selection(
            controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
            controller_config=None,
            suite_mode="sealed_randomized",
        )[1]
        env = episode.build_libero_env(0, 1000, 256, suite_mode="sealed_randomized", controller_variant=variant)
        binding = reference.validate_environment(env, task_id=0, episode_index=0, seed=1000)
        inputs = matrix._default_arrow_inputs(env, 0, 256)
        from .runner import _rgb
        from PIL import Image
        clean = _rgb(env, 256)
        arrow, audit = episode.render_exactly_one_arrow(
            clean, inputs["bboxes"], subject=inputs["subject"], goal_object=inputs["goal_object"], line_width=1, head_length=16
        )
        arrow = arrow[::-1, ::-1].copy()
        Image.fromarray(arrow).save(preview_root / "arrow_student_input.png")
        _write(preview_root / "preflight.json", {"status": "passed", "reference_binding": binding, "arrow_audit": audit, "actions_sent": 0})
        return {"status": "passed", "preview": (preview_root / "arrow_student_input.png").as_posix(), "actions_sent": 0}
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if close is not None:
                close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_motion and not args.preflight_only:
        raise ValueError("explicit --execute-motion or --preflight-only is required")
    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference = load_reference_protocol(args.reference_root)
    runtime = ArrowStudentRuntime(
        checkpoint=args.checkpoint,
        source_checkpoint=args.source_checkpoint,
        base_policy=args.base_policy,
        device=args.device,
    )
    model_preflight = runtime.preflight()
    env_preflight = _preflight_env(reference, output_root)
    run_root = output_root.parent.parent
    _write(run_root / "reference_protocol.json", reference.as_dict())
    _write(run_root / "model_preflight.json", model_preflight)
    _write(run_root / "provenance.json", {
        "treatment": "arrow_student_language_free_stage_d",
        "checkpoint": runtime.checkpoint.as_posix(),
        "source_checkpoint": runtime.source_checkpoint.as_posix(),
        "base_policy": runtime.base_policy.as_posix(),
        "model_sha256": model_preflight["model_sha256"],
        "checkpoint_manifest_sha256": model_preflight["checkpoint_manifest_sha256"],
        "reference_manifest_sha256": reference.manifest_sha256,
        "reference_contract_hash": reference.contract_hash,
        "camera": "agentview",
        "resolution": 256,
        "schedule": {"tasks": list(range(10)), "episodes_per_task": 10, "seed_base": 1000},
        "language_consumed": False,
        "tokenizer_accessed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(runtime.device),
        "env_preflight": env_preflight,
    })
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "run_root": run_root.as_posix()}))
        return 0
    runner = ArrowStudentEpisodeRunner(runtime, reference, resolution=256, max_actions=args.max_actions)
    summary = matrix.run_matrix(
        output_root=output_root,
        task_ids=tuple(range(10)),
        episodes_per_task=10,
        seed_base=1000,
        resolution=256,
        suite_mode="sealed_randomized",
        controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
        execute_motion=True,
        allow_unvalidated_profile=True,
        env_builder=episode.build_libero_env,
        episode_runner=runner,
        arrow_input_builder=matrix._default_arrow_inputs,
        resume=args.resume,
        continue_on_motion_failure=True,
        experiment_metadata={
            "treatment": "arrow_student_language_free_stage_d",
            "reference_manifest_sha256": reference.manifest_sha256,
            "reference_contract_hash": reference.contract_hash,
            "checkpoint_manifest_sha256": model_preflight["checkpoint_manifest_sha256"],
            "model_sha256": model_preflight["model_sha256"],
            "episode_runner": "vla_benchmarking.arrow_policy.evaluation.runner.ArrowStudentEpisodeRunner",
            "language_consumed": False,
            "tokenizer_accessed": False,
        },
    )
    print(json.dumps({"status": "completed_or_partial", "summary_path": summary["summary_path"], "success_rate": summary["success_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
