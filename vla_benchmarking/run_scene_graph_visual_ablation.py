#!/usr/bin/env python3
"""Run the arrow-only visual scene-graph ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import SCENE_GRAPH_SUBJECT_FILTER, TASK_GOAL_OBJECT_CONFIG
from run_scene_graph_format_ablation import (
    DEFAULT_EPISODES,
    DEFAULT_SMOLVLA_MODEL,
    DEFAULT_TASKS,
    _count_videos,
    _episode_rows_for_info,
    _extract_success,
    _git_commit,
    _parse_tasks,
    _preflight_check,
    _run_with_tee,
    _safe_name,
    _task_ids_arg,
    _task_summary_rows,
    _write_csv,
)
from visual_scene_graph import (
    DEFAULT_GOAL_OBJECT,
    SUPPORTED_VISUAL_CONDITIONS,
    goal_arrow_prompt_hint,
)


DEFAULT_CONDITION = "visual_arrows"


def _resolve_eval_info(candidate: Path) -> Path | None:
    candidate = candidate.expanduser().resolve()
    options = [
        candidate,
        candidate / "eval_info.json",
        candidate / "standard" / "eval_info.json",
    ]
    for option in options:
        if option.is_file() and option.name == "eval_info.json":
            return option
    return None


def _find_baseline(
    ablation_root: Path,
    requested: str | None,
    current_output_root: Path,
) -> Path | None:
    if requested:
        resolved = _resolve_eval_info(Path(requested))
        if resolved is None:
            raise SystemExit(
                "ERROR: --baseline-run must point to eval_info.json, a standard "
                "condition directory, or a run containing standard/eval_info.json."
            )
        return resolved

    candidates = []
    for eval_info in ablation_root.glob("*/standard/eval_info.json"):
        try:
            eval_info.resolve().relative_to(current_output_root.resolve())
        except ValueError:
            candidates.append(eval_info)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _result_row(
    condition: str,
    output_dir: Path,
    eval_info_path: Path | None = None,
) -> dict[str, Any]:
    eval_info_path = eval_info_path or output_dir / "eval_info.json"
    pc_success = None
    successes = None
    episodes = None
    if eval_info_path.exists():
        with eval_info_path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
        pc_success, successes, episodes = _extract_success(info)

    relation_audit = output_dir / "visual_relation_audit.jsonl"
    return {
        "condition": condition,
        "pc_success": "" if pc_success is None else f"{pc_success:.4f}",
        "successes": "" if successes is None else str(successes),
        "episodes": "" if episodes is None else str(episodes),
        "videos": str(_count_videos(output_dir)),
        "output_dir": output_dir.as_posix(),
        "eval_info": eval_info_path.as_posix() if eval_info_path.exists() else "",
        "visual_relation_audit": relation_audit.as_posix() if relation_audit.exists() else "",
    }


def _visual_episode_rows(
    condition: str,
    output_dir: Path,
    eval_info_path: Path,
    seed: int,
) -> list[dict[str, Any]]:
    if not eval_info_path.exists():
        return []
    with eval_info_path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)
    rows = _episode_rows_for_info(condition, output_dir, info, seed)
    for row in rows:
        row["condition"] = row.pop("format")
    return rows


def _write_visual_outputs(
    output_root: Path,
    condition: str,
    condition_dir: Path,
    seed: int,
    baseline_eval_info: Path | None,
) -> None:
    visual_row = _result_row(condition, condition_dir)
    _write_csv(output_root / "visual_summary.csv", [visual_row], list(visual_row))

    episode_rows = _visual_episode_rows(
        condition,
        condition_dir,
        condition_dir / "eval_info.json",
        seed,
    )
    _write_csv(
        output_root / "visual_episodes.csv",
        episode_rows,
        [
            "condition",
            "task_group",
            "task_id",
            "episode_ix",
            "seed",
            "success",
            "sum_reward",
            "max_reward",
            "video_path",
            "predicted_video_path",
            "output_dir",
        ],
    )

    task_rows = _task_summary_rows(
        [
            {"format": row["condition"], **{key: value for key, value in row.items() if key != "condition"}}
            for row in episode_rows
        ]
    )
    for row in task_rows:
        row["condition"] = row.pop("format")
    _write_csv(
        output_root / "visual_task_summary.csv",
        task_rows,
        [
            "condition",
            "task_group",
            "task_id",
            "episodes",
            "successes",
            "pc_success",
            "avg_sum_reward",
            "avg_max_reward",
            "videos",
        ],
    )

    comparison_rows = []
    if baseline_eval_info is not None:
        comparison_rows.append(
            _result_row("standard", baseline_eval_info.parent, baseline_eval_info)
        )
    comparison_rows.append(visual_row)
    _write_csv(
        output_root / "visual_vs_baseline.csv",
        comparison_rows,
        list(visual_row),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run arrow-only visual scene-graph evaluation through LeRobot."
    )
    parser.add_argument("--model", default=DEFAULT_SMOLVLA_MODEL)
    parser.add_argument("--tasks", default=_task_ids_arg(DEFAULT_TASKS))
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("N_EPISODES", DEFAULT_EPISODES)),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1000")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument(
        "--condition",
        choices=sorted(SUPPORTED_VISUAL_CONDITIONS),
        default=DEFAULT_CONDITION,
    )
    parser.add_argument(
        "--goal-object",
        default=None,
        help="Optional override for every task. By default, uses TASK_GOAL_OBJECT_CONFIG.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="Previous run, standard directory, or standard/eval_info.json to compare against.",
    )
    parser.add_argument("--prompt-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Rendered videos per task. Defaults to --episodes.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to run_lerobot_eval_with_context.py. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _preflight_check(args.device, token_audit=False)

    script_dir = Path(__file__).resolve().parent
    ablation_root = script_dir / "ablation_outputs"
    task_ids = _parse_tasks(args.tasks)
    run_id = args.run_id or datetime.now().strftime(
        "sg_visual_ablation_%Y_%m_%d_%H_%M_%S"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else ablation_root / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    condition_dir = output_root / _safe_name(args.condition)
    condition_dir.mkdir(parents=True, exist_ok=True)

    baseline_eval_info = _find_baseline(
        ablation_root,
        args.baseline_run,
        output_root,
    )
    max_videos = args.max_videos if args.max_videos is not None else args.episodes
    goal_objects_by_task = {
        task_id: args.goal_object or TASK_GOAL_OBJECT_CONFIG.get(task_id, DEFAULT_GOAL_OBJECT)
        for task_id in task_ids
    }
    manifest = {
        "run_id": run_id,
        "condition": args.condition,
        "model": args.model,
        "tasks": task_ids,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "text_context_mode": "standard",
        "text_prompt_contains_graph": False,
        "visual_prompt_hint": (
            {
                str(task_id): goal_arrow_prompt_hint(goal_object)
                for task_id, goal_object in goal_objects_by_task.items()
            }
            if args.condition == "visual_goal_arrow"
            else None
        ),
        "visual_overlay": {
            "image_slot": "pixels/image",
            "content": "goal_arrow_only" if args.condition == "visual_goal_arrow" else "arrows_only",
            "direction": "subject_to_object",
            "subject_filter": SCENE_GRAPH_SUBJECT_FILTER,
            "goal_objects_by_task": (
                {str(task_id): goal_object for task_id, goal_object in goal_objects_by_task.items()}
                if args.condition == "visual_goal_arrow"
                else None
            ),
            "camera_source": "camera mapped to the main pixels/image policy slot",
        },
        "prompt_audit": args.prompt_audit,
        "videos": args.videos,
        "max_videos": max_videos if args.videos else 0,
        "baseline_eval_info": (
            baseline_eval_info.as_posix() if baseline_eval_info is not None else None
        ),
        "repo_commit": _git_commit(script_dir),
    }
    manifest_path = output_root / "visual_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODELS": args.model,
            "TASK_IDS": _task_ids_arg(task_ids),
            "N_EPISODES": str(args.episodes),
            "BATCH_SIZE": str(args.batch_size),
            "SEED": str(args.seed),
            "DEVICE": args.device,
            "CONTEXT_MODE": "standard",
            "CONTEXT_FORMAT": "standard",
            "VISUAL_CONDITION": args.condition,
            "PROMPT_AUDIT": "1" if args.prompt_audit else "0",
            "TOKEN_AUDIT": "0",
            "MAX_EPISODES_RENDERED": str(max_videos if args.videos else 0),
            "RENDER_MODE": "rgb_array" if args.videos else "none",
        }
    )
    if args.goal_object:
        env["VISUAL_GOAL_OBJECT"] = args.goal_object

    cmd = [
        sys.executable,
        str(script_dir / "run_lerobot_eval_with_context.py"),
        "--eval.use_async_envs=false",
        f"--output_dir={condition_dir.as_posix()}",
        *args.extra_arg,
    ]
    log_path = condition_dir / "eval_stdout_stderr.log"
    print("=" * 72)
    print(f"Running visual condition: {args.condition}")
    print(f"  model      : {args.model}")
    print(f"  tasks      : {_task_ids_arg(task_ids)}")
    print(f"  episodes   : {args.episodes}")
    print(f"  output_dir : {condition_dir}")
    print(f"  baseline   : {baseline_eval_info or 'not found'}")
    print("=" * 72)

    return_code = _run_with_tee(
        cmd,
        cwd=script_dir,
        env=env,
        log_path=log_path,
    )
    _write_visual_outputs(
        output_root,
        args.condition,
        condition_dir,
        args.seed,
        baseline_eval_info,
    )

    if return_code != 0:
        print(f"ERROR: visual condition failed with exit code {return_code}")
        print(f"Partial outputs: {output_root}")
        print(f"Evaluation log: {log_path}")
        return return_code

    print("=" * 72)
    print("Visual ablation complete")
    print(f"Summary: {output_root / 'visual_summary.csv'}")
    print(f"Comparison: {output_root / 'visual_vs_baseline.csv'}")
    print(f"Outputs: {output_root}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
