#!/usr/bin/env python3
"""Print a compact status report for the latest ablation run."""

from __future__ import annotations

import csv
import json
from pathlib import Path


# This tool lives under ``vla_benchmarking/tools`` while historical ablation
# outputs remain owned by the evaluation root.  Derive the package root rather
# than silently looking for ``tools/ablation_outputs``.
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "ablation_outputs"


def _latest_run() -> Path | None:
    if not OUTPUTS.exists():
        return None
    runs = [path for path in OUTPUTS.iterdir() if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _tail(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    run = _latest_run()
    if run is None:
        print("No ablation_outputs runs found.")
        return 1

    manifest_path = run / "ablation_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    formats = manifest.get("formats", [])
    print(f"Run: {run}")
    if manifest:
        print(f"Model: {manifest.get('model')}")
        print(f"Tasks: {manifest.get('tasks')}")
        print(f"Episodes: {manifest.get('episodes')}")
        print(f"Device: {manifest.get('device')}")
        print(f"Formats: {len(formats)}")

    summary_rows = _csv_rows(run / "ablation_summary.csv")
    completed = {row["format"] for row in summary_rows if row.get("eval_info")}
    print(f"Completed formats: {len(completed)}/{len(formats) if formats else '?'}")
    if completed:
        print("Completed: " + ", ".join(sorted(completed)))

    for format_name in formats:
        if format_name not in completed and (run / format_name).exists():
            audit = run / format_name / "prompt_audit.jsonl"
            videos = list((run / format_name / "videos").rglob("*.mp4")) if (run / format_name / "videos").exists() else []
            print(f"Active/partial: {format_name}")
            print(f"  prompt audit rows: {_count_lines(audit)}")
            print(f"  video files: {len(videos)}")
            break

    progress_log = run / "progress.log"
    if progress_log.exists():
        print("\nprogress.log tail:")
        for line in _tail(progress_log):
            print(line)

    active_logs = [
        path for path in run.glob("*/eval_stdout_stderr.log")
        if path.exists()
    ]
    if active_logs:
        latest_log = max(active_logs, key=lambda path: path.stat().st_mtime)
        print(f"\nLatest eval log: {latest_log}")
        for line in _tail(latest_log):
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
