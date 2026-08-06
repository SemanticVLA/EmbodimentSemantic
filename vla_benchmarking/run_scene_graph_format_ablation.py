#!/usr/bin/env python3
"""Run the Stage A scene-graph format ablation matrix."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scene_graph_formats import DEFAULT_ABLATION_FORMATS, SUPPORTED_CONTEXT_FORMATS, normalize_context_format


DEFAULT_SMOLVLA_MODEL = "HuggingFaceVLA/smolvla_libero"
DEFAULT_TASKS = list(range(10))
DEFAULT_EPISODES = 4


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tasks(value: str) -> list[int]:
    cleaned = value.strip()
    if cleaned.startswith("["):
        parsed = json.loads(cleaned)
        return [int(item) for item in parsed]
    return [int(item) for item in _parse_csv(cleaned)]


def _task_ids_arg(task_ids: list[int]) -> str:
    return "[" + ",".join(str(task_id) for task_id in task_ids) + "]"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{_timestamp()}] {message}\n")


def _print_and_log(path: Path, message: str) -> None:
    print(message, flush=True)
    _log_line(path, message)


def _conda_env_name() -> str | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    return Path(prefix).name


def _preflight_check(requested_device: str, token_audit: bool) -> None:
    missing = [
        module_name
        for module_name in ("lerobot", "torch", "transformers")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        active_env = _conda_env_name() or "unknown"
        missing_text = ", ".join(missing)
        raise SystemExit(
            "ERROR: this Python environment is missing required packages: "
            f"{missing_text}.\n"
            f"Active conda env: {active_env}\n"
            f"Python: {sys.executable}\n\n"
            "Use the environment that has LeRobot installed:\n"
            "  conda activate vla_bench_py312\n"
            "  python run_scene_graph_format_ablation.py\n\n"
            "Or run it without changing shells:\n"
            "  conda run -n vla_bench_py312 python run_scene_graph_format_ablation.py"
        )

    if requested_device.strip().lower().startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: --device cuda was requested, but PyTorch reports CUDA is not available.\n"
                "The full ablation is large enough that silently running on CPU is usually a mistake.\n\n"
                "Fix CUDA in the active env, or explicitly run on CPU:\n"
                "  python run_scene_graph_format_ablation.py --device cpu"
            )

    if token_audit and importlib.util.find_spec("google.protobuf") is None:
        raise SystemExit(
            "ERROR: token audit is enabled, but protobuf is not installed in this environment.\n"
            "Without protobuf, prompt_audit.jsonl is still written but token counts are missing.\n\n"
            "Install it and restart:\n"
            "  python -m pip install protobuf\n\n"
            "Or run without token counts:\n"
            "  python run_scene_graph_format_ablation.py --no-token-audit"
        )


def _extract_success(info: dict[str, Any]) -> tuple[float | None, int | None, int | None]:
    overall = info.get("overall")
    if isinstance(overall, dict):
        pc_success = overall.get("pc_success")
        total = overall.get("n_episodes")
        if pc_success is not None and isinstance(total, int):
            count = round(float(pc_success) * total / 100.0)
            return float(pc_success), count, total
        if pc_success is not None:
            return float(pc_success), None, None

    successes = [
        row["success"]
        for row in _episode_rows_for_info("unknown", Path("."), info, None)
        if row["success"] != ""
    ]
    if successes:
        count = sum(str(item).lower() == "true" for item in successes)
        return 100.0 * count / len(successes), count, len(successes)
    return None, None, None


def _count_videos(output_dir: Path) -> int:
    videos_dir = output_dir / "videos"
    if not videos_dir.exists():
        return 0
    return sum(1 for _ in videos_dir.rglob("*.mp4"))


def _episode_rows_for_info(
    format_name: str,
    output_dir: Path,
    info: dict[str, Any],
    seed_base: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_task = info.get("per_task")
    if not isinstance(per_task, list):
        return rows

    for task_record in per_task:
        if not isinstance(task_record, dict):
            continue
        task_group = task_record.get("task_group", "")
        task_id = task_record.get("task_id", "")
        metrics = task_record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        successes = metrics.get("successes", [])
        sum_rewards = metrics.get("sum_rewards", [])
        max_rewards = metrics.get("max_rewards", [])
        video_paths = metrics.get("video_paths", [])
        predicted_video_paths = metrics.get("predicted_video_paths", [])
        episode_count = max(len(successes), len(sum_rewards), len(max_rewards))

        for episode_ix in range(episode_count):
            rows.append(
                {
                    "format": format_name,
                    "task_group": task_group,
                    "task_id": task_id,
                    "episode_ix": episode_ix,
                    "seed": "" if seed_base is None else seed_base + episode_ix,
                    "success": successes[episode_ix] if episode_ix < len(successes) else "",
                    "sum_reward": sum_rewards[episode_ix] if episode_ix < len(sum_rewards) else "",
                    "max_reward": max_rewards[episode_ix] if episode_ix < len(max_rewards) else "",
                    "video_path": video_paths[episode_ix] if episode_ix < len(video_paths) else "",
                    "predicted_video_path": (
                        predicted_video_paths[episode_ix]
                        if episode_ix < len(predicted_video_paths)
                        else ""
                    ),
                    "output_dir": output_dir.as_posix(),
                }
            )
    return rows


def _task_summary_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (str(row["format"]), str(row["task_group"]), str(row["task_id"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (format_name, task_group, task_id), items in sorted(grouped.items()):
        successes = [str(item["success"]).lower() == "true" for item in items if item["success"] != ""]
        sum_rewards = [float(item["sum_reward"]) for item in items if item["sum_reward"] != ""]
        max_rewards = [float(item["max_reward"]) for item in items if item["max_reward"] != ""]
        success_count = sum(successes)
        total = len(successes)
        rows.append(
            {
                "format": format_name,
                "task_group": task_group,
                "task_id": task_id,
                "episodes": total,
                "successes": success_count,
                "pc_success": "" if total == 0 else f"{100.0 * success_count / total:.4f}",
                "avg_sum_reward": "" if not sum_rewards else f"{sum(sum_rewards) / len(sum_rewards):.6f}",
                "avg_max_reward": "" if not max_rewards else f"{sum(max_rewards) / len(max_rewards):.6f}",
                "videos": sum(1 for item in items if item["video_path"]),
            }
        )
    return rows


def _audit_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _format_percent(done: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{max(0.0, min(100.0, 100.0 * done / total)):.1f}%"


def _progress_ticker(
    *,
    stop_event: threading.Event,
    progress_log: Path,
    format_name: str,
    format_index: int,
    total_formats: int,
    audit_path: Path,
    expected_prompt_calls: int,
) -> None:
    last_lines = -1
    last_print = 0.0
    interval_s = 60.0
    while not stop_event.wait(5.0):
        now = time.time()
        lines = _audit_line_count(audit_path)
        if lines == last_lines and now - last_print < interval_s:
            continue
        last_lines = lines
        last_print = now
        format_pct = _format_percent(lines, expected_prompt_calls)
        overall_done = (format_index - 1) + min(1.0, lines / expected_prompt_calls if expected_prompt_calls else 0.0)
        overall_pct = _format_percent(overall_done, total_formats)
        _print_and_log(
            progress_log,
            (
                f"PROGRESS format={format_name} [{format_index}/{total_formats}] "
                f"format_est={format_pct} overall_est={overall_pct} "
                f"prompt_rows={lines}/{expected_prompt_calls}"
            ),
        )


def _run_with_tee(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_fh:
        log_fh.write(f"[{_timestamp()}] COMMAND: {' '.join(cmd)}\n")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        write_lock = threading.Lock()

        def pipe_reader(stream, stream_name: str) -> None:
            for line in iter(stream.readline, ""):
                with write_lock:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_fh.write(f"[{stream_name}] {line}")
                    log_fh.flush()

        stdout_thread = threading.Thread(
            target=pipe_reader,
            args=(process.stdout, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pipe_reader,
            args=(process.stderr, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            raise

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        log_fh.write(f"[{_timestamp()}] EXIT_CODE: {return_code}\n")
        return return_code


def _read_manifest_seed(output_root: Path) -> int | None:
    manifest_path = output_root / "ablation_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    seed = manifest.get("seed")
    return int(seed) if seed is not None else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_root: Path, formats: list[str]) -> None:
    rows = []
    episode_rows: list[dict[str, Any]] = []
    seed_base = _read_manifest_seed(output_root)
    for format_name in formats:
        output_dir = output_root / format_name
        eval_info_path = output_dir / "eval_info.json"
        pc_success = None
        success_count = None
        total = None
        if eval_info_path.exists():
            with eval_info_path.open("r", encoding="utf-8") as fh:
                info = json.load(fh)
            pc_success, success_count, total = _extract_success(info)
            episode_rows.extend(_episode_rows_for_info(format_name, output_dir, info, seed_base))
        rows.append(
            {
                "format": format_name,
                "pc_success": "" if pc_success is None else f"{pc_success:.4f}",
                "successes": "" if success_count is None else str(success_count),
                "episodes": "" if total is None else str(total),
                "videos": str(_count_videos(output_dir)),
                "output_dir": output_dir.as_posix(),
                "eval_info": eval_info_path.as_posix() if eval_info_path.exists() else "",
                "prompt_audit": (output_dir / "prompt_audit.jsonl").as_posix()
                if (output_dir / "prompt_audit.jsonl").exists()
                else "",
            }
        )

    summary_json = output_root / "ablation_summary.json"
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    _write_csv(output_root / "ablation_summary.csv", rows, list(rows[0].keys()))
    _write_csv(
        output_root / "ablation_episodes.csv",
        episode_rows,
        [
            "format",
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
    _write_csv(
        output_root / "ablation_task_summary.csv",
        _task_summary_rows(episode_rows),
        [
            "format",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scene-graph format ablations through run_lerobot_eval_with_context.py."
    )
    parser.add_argument("--model", default=DEFAULT_SMOLVLA_MODEL)
    parser.add_argument("--tasks", default=_task_ids_arg(DEFAULT_TASKS))
    parser.add_argument("--formats", default=",".join(DEFAULT_ABLATION_FORMATS))
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("N_EPISODES", DEFAULT_EPISODES)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1000")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--token-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Rendered videos per task/format. Defaults to --episodes, so every evaluated episode gets one video.",
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
    _preflight_check(args.device, args.token_audit)
    script_dir = Path(__file__).resolve().parent
    task_ids = _parse_tasks(args.tasks)
    formats = [normalize_context_format(format_name) for format_name in _parse_csv(args.formats)]
    for format_name in formats:
        if format_name not in SUPPORTED_CONTEXT_FORMATS:
            raise SystemExit(
                f"{format_name!r} is not supported. Use one of: "
                f"{', '.join(SUPPORTED_CONTEXT_FORMATS)}"
            )

    run_id = args.run_id or datetime.now().strftime("sg_format_ablation_%Y_%m_%d_%H_%M_%S")
    output_root = Path(args.output_root) if args.output_root else script_dir / "ablation_outputs" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    progress_log = output_root / "progress.log"
    _print_and_log(progress_log, f"Run started: {run_id}")

    manifest = {
        "run_id": run_id,
        "model": args.model,
        "tasks": task_ids,
        "formats": formats,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "prompt_audit": args.prompt_audit,
        "token_audit": args.token_audit,
        "videos": args.videos,
        "max_videos": args.max_videos if args.max_videos is not None else args.episodes,
        "repo_commit": _git_commit(script_dir),
    }
    with (output_root / "ablation_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    _log_line(progress_log, f"Manifest written: {output_root / 'ablation_manifest.json'}")
    _log_line(progress_log, f"Formats: {', '.join(formats)}")
    _log_line(progress_log, f"Tasks: {task_ids}")
    _log_line(progress_log, f"Episodes per task/format: {args.episodes}")
    expected_prompt_calls = len(task_ids) * args.episodes
    _log_line(progress_log, f"Estimated prompt rows per format: {expected_prompt_calls}")

    task_arg = _task_ids_arg(task_ids)
    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["MODELS"] = args.model
    env_base["TASK_IDS"] = task_arg
    env_base["N_EPISODES"] = str(args.episodes)
    env_base["BATCH_SIZE"] = str(args.batch_size)
    env_base["SEED"] = str(args.seed)
    env_base["DEVICE"] = args.device
    env_base["PROMPT_AUDIT"] = "1" if args.prompt_audit else "0"
    env_base["TOKEN_AUDIT"] = "1" if args.token_audit else "0"
    env_base["MAX_EPISODES_RENDERED"] = (
        str(args.max_videos if args.max_videos is not None else args.episodes)
        if args.videos
        else "0"
    )
    env_base["RENDER_MODE"] = "rgb_array" if args.videos else "none"

    for index, format_name in enumerate(formats, start=1):
        output_dir = output_root / _safe_name(format_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        env = env_base.copy()
        env["CONTEXT_FORMAT"] = format_name
        env["CONTEXT_MODE"] = "standard" if format_name == "standard" else "scene_graph"

        cmd = [
            sys.executable,
            str(script_dir / "run_lerobot_eval_with_context.py"),
            "--eval.use_async_envs=false",
            f"--output_dir={output_dir.as_posix()}",
            *args.extra_arg,
        ]
        print("=" * 72)
        print(f"[{index}/{len(formats)}] Running {format_name}")
        print(f"  model      : {args.model}")
        print(f"  tasks      : {task_arg}")
        print(f"  episodes   : {args.episodes}")
        print(f"  progress   : {100.0 * (index - 1) / len(formats):.1f}% complete before this format")
        print(f"  output_dir : {output_dir}")
        print(f"  eval log   : {output_dir / 'eval_stdout_stderr.log'}")
        print("=" * 72)
        _print_and_log(
            progress_log,
            f"[{index}/{len(formats)}] START {format_name} -> {output_dir}",
        )
        stop_ticker = threading.Event()
        ticker_thread = threading.Thread(
            target=_progress_ticker,
            kwargs={
                "stop_event": stop_ticker,
                "progress_log": progress_log,
                "format_name": format_name,
                "format_index": index,
                "total_formats": len(formats),
                "audit_path": output_dir / "prompt_audit.jsonl",
                "expected_prompt_calls": expected_prompt_calls,
            },
            daemon=True,
        )
        ticker_thread.start()
        try:
            return_code = _run_with_tee(
                cmd,
                cwd=script_dir,
                env=env,
                log_path=output_dir / "eval_stdout_stderr.log",
            )
        finally:
            stop_ticker.set()
            ticker_thread.join(timeout=2)

        audit_rows = _audit_line_count(output_dir / "prompt_audit.jsonl")
        _print_and_log(
            progress_log,
            (
                f"PROGRESS format={format_name} [{index}/{len(formats)}] "
                f"format_est={_format_percent(audit_rows, expected_prompt_calls)} "
                f"overall_est={_format_percent(index, len(formats))} "
                f"prompt_rows={audit_rows}/{expected_prompt_calls}"
            ),
        )
        if return_code != 0:
            _write_summary(output_root, formats)
            _print_and_log(
                progress_log,
                f"[{index}/{len(formats)}] FAILED {format_name} exit_code={return_code}",
            )
            print(f"ERROR: format {format_name} failed with exit code {return_code}")
            print(f"Partial summary written under {output_root}")
            print(f"Format log: {output_dir / 'eval_stdout_stderr.log'}")
            return return_code
        _write_summary(output_root, formats)
        _print_and_log(
            progress_log,
            (
                f"[{index}/{len(formats)}] DONE {format_name}; "
                f"overall={100.0 * index / len(formats):.1f}%; summaries updated"
            ),
        )

    _write_summary(output_root, formats)
    _print_and_log(progress_log, "Run complete")
    print("=" * 72)
    print("Ablation complete")
    print(f"Summary: {output_root / 'ablation_summary.csv'}")
    print(f"Episodes: {output_root / 'ablation_episodes.csv'}")
    print(f"By task: {output_root / 'ablation_task_summary.csv'}")
    print(f"Outputs: {output_root}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
