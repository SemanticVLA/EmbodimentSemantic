"""Command-line front door for SO101 inventory, name probing, and agent graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_so101_config
from .dataset import SO101Dataset


def _tasks(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return available
    result = []
    for item in value.split(","):
        item = item.strip()
        if item.isdigit():
            index = int(item)
            if index >= len(available):
                raise ValueError(f"task index out of range: {index}")
            item = available[index]
        if item not in available:
            raise ValueError(f"unknown task: {item}")
        if item in result:
            raise ValueError(f"duplicate task selection: {item}")
        result.append(item)
    if not result:
        raise ValueError("no tasks selected")
    return result


def _episode_selection(value: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if value == "all":
        return available
    if ":" in value:
        fields = value.split(":")
        if len(fields) != 2 or not all(field.isdigit() for field in fields):
            raise ValueError("episodes must be all, a list, or a half-open range")
        start, stop = map(int, fields)
        requested = tuple(range(start, stop))
    else:
        fields = value.split(",")
        if not fields or not all(field.isdigit() for field in fields):
            raise ValueError("episodes must be all, a list, or a half-open range")
        requested = tuple(map(int, fields))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("episode selection must be non-empty and unique")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"requested episodes are unavailable: {missing}")
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tasks", default="all",
                        help="all, comma-separated exact task names, or sorted numeric indices")
    parser.add_argument("--episodes", default="0",
                        help="all, comma-separated indices, or a half-open range such as 0:5")
    parser.add_argument("--camera-mode", choices=("agent",), default="agent",
                        help="fixed external camera (the only supported SO101 view)")
    parser.add_argument("--output-stride", type=int, default=30,
                        help="save every Nth frame; native tracking always runs at stride 1")
    parser.add_argument("--objects-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new versioned output directory; existing paths are never overwritten")
    parser.add_argument("--stage", choices=("inventory", "probe", "run"), default="run")
    parser.add_argument("--probe-episode", type=int,
                        help="episode index for a text-only first-frame localization probe")
    parser.add_argument("--review-fps", type=float, default=2.0)
    parser.add_argument("--no-videos", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_stride < 1:
        raise ValueError("output-stride must be positive")
    dataset = SO101Dataset(args.dataset_root)
    config = load_so101_config(args.objects_config)
    available = sorted(dataset.task_paths)
    task_ids = _tasks(args.tasks, available)
    missing_config = set(task_ids) - set(config.tasks)
    if missing_config:
        raise ValueError(f"object configuration lacks dataset tasks: {sorted(missing_config)}")
    selections = {task: _episode_selection(args.episodes, dataset.episode_ids(task))
                  for task in task_ids}
    if args.stage == "inventory":
        from .pipeline import write_inventory
        value = write_inventory(dataset, args.output_dir)
    elif args.stage == "probe":
        if args.checkpoint is None:
            raise ValueError("probe requires --checkpoint")
        probe_episode = 0 if args.probe_episode is None else args.probe_episode
        if any(probe_episode not in selected for selected in selections.values()):
            raise ValueError("probe episode must be included in the selected episode list")
        from .probe import run_name_probe
        value = run_name_probe(
            dataset, config, args.checkpoint, args.output_dir, task_ids,
            episode=probe_episode,
        )
    else:
        if args.checkpoint is None:
            raise ValueError("run requires --checkpoint")
        from .pipeline import SO101AgentPipeline
        pipeline = SO101AgentPipeline(
            dataset,
            config,
            args.checkpoint,
            args.output_dir,
            output_stride=args.output_stride,
            review_fps=args.review_fps,
            write_videos=not args.no_videos,
        )
        value = pipeline.run(task_ids, selections)
    print(json.dumps({"output": str(args.output_dir.resolve()), "result": value}, indent=2))
    return 0


__all__ = ["build_parser", "main"]
