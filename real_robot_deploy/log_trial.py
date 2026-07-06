#!/usr/bin/env python3
"""Log a single real-robot trial result to results/trials.csv.

Run this immediately after each episode from run_semantic_episode.py.

Usage:
    # Successful trial
    python log_trial.py \\
        --trial-label pick_up_the_black --task "Pick up the black cube..." \\
        --context-mode scene_graph --grasp 1 --success 1 --ttc 14.3 \\
        --media-dir results/media/pick_up_the_black/scene_graph/trial_20260624_120000

    # Failed trial
    python log_trial.py \\
        --trial-label pick_up_the_black --context-mode standard \\
        --grasp 0 --success 0 --failure-mode grasp_fail

    # List recent trials
    python log_trial.py --show 20

    # Show summary table
    python log_trial.py --summary
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
TRIALS_CSV = RESULTS_DIR / "trials.csv"

CONTEXT_MODES = ["standard", "scene_graph"]
SCENE_GRAPH_FORMATS = ["", "triplet", "natural_language", "json"]
FAILURE_MODES = ["grasp_fail", "transport_fail", "placement_fail", "wrong_object", "no_motion", "none"]

CSV_HEADER = [
    "timestamp", "trial_label", "task", "context_mode", "scene_graph_format",
    "grasp_success", "task_success", "ttc_s", "failure_mode", "media_dir", "notes",
]


def ensure_csv():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TRIALS_CSV.exists():
        with open(TRIALS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"Created {TRIALS_CSV}")


def log_trial(args):
    ensure_csv()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trial_label": args.trial_label,
        "task": args.task or "",
        "context_mode": args.context_mode,
        "scene_graph_format": args.scene_graph_format or "",
        "grasp_success": int(args.grasp),
        "task_success": int(args.success),
        "ttc_s": args.ttc if args.success else "nan",
        "failure_mode": args.failure_mode if not args.success else "none",
        "media_dir": args.media_dir or "",
        "notes": args.notes or "",
    }
    with open(TRIALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writerow(row)

    status = "SUCCESS" if args.success else f"FAIL ({row['failure_mode']})"
    grasp = "grasp OK" if args.grasp else "no grasp"
    print(f"[logged] {args.trial_label}/{args.context_mode} — {status} | {grasp} | TTC={row['ttc_s']}s")
    _print_progress()


def _load_trials():
    if not TRIALS_CSV.exists():
        return []
    with open(TRIALS_CSV) as f:
        return list(csv.DictReader(f))


def _print_progress():
    rows = _load_trials()
    counts = defaultdict(lambda: defaultdict(int))
    successes = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["trial_label"]][r["context_mode"]] += 1
        successes[r["trial_label"]][r["context_mode"]] += int(r["task_success"])

    print("\n--- Trial Progress ---")
    print(f"{'Trial label':<24} {'Mode':<12} {'N':>4} {'SR':>6}")
    print("-" * 50)
    for label in sorted(counts):
        for mode in CONTEXT_MODES:
            n = counts[label][mode]
            if n == 0:
                continue
            sr = successes[label][mode] / n
            print(f"{label:<24} {mode:<12} {n:>4} {sr:>6.1%}")
    print(f"\nTotal trials logged: {len(rows)}")


def show_recent(n):
    rows = _load_trials()
    rows = rows[-n:]
    print(f"\n--- Last {len(rows)} trials ---")
    for r in rows:
        status = "OK" if r["task_success"] == "1" else "FAIL"
        print(f"  {r['timestamp']}  {r['trial_label']:<24} {r['context_mode']:<12} {status}  fm={r['failure_mode']}")


def show_summary():
    rows = _load_trials()
    if not rows:
        print("No trials logged yet.")
        return

    counts = defaultdict(lambda: defaultdict(int))
    successes = defaultdict(lambda: defaultdict(int))
    grasps = defaultdict(lambda: defaultdict(int))

    for r in rows:
        label, mode = r["trial_label"], r["context_mode"]
        counts[label][mode] += 1
        successes[label][mode] += int(r["task_success"])
        grasps[label][mode] += int(r["grasp_success"])

    # Wilson 95% CI
    def wilson_ci(s, n):
        if n == 0:
            return 0.0, 0.0
        z = 1.96
        p = s / n
        center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return max(0, center - half), min(1, center + half)

    print("\n" + "=" * 80)
    print("REAL ROBOT TRIAL SUMMARY")
    print("=" * 80)
    print(f"{'Trial label':<24} {'Mode':<12} {'N':>4} {'GSR':>6} {'SR':>6}  {'95% CI':<16}")
    print("-" * 80)

    for label in sorted(counts):
        for mode in CONTEXT_MODES:
            n = counts[label][mode]
            if n == 0:
                continue
            sr = successes[label][mode] / n
            gsr = grasps[label][mode] / n
            lo, hi = wilson_ci(successes[label][mode], n)
            ci_str = f"[{lo:.2f}, {hi:.2f}]"
            print(f"{label:<24} {mode:<12} {n:>4} {gsr:>6.1%} {sr:>6.1%}  {ci_str:<16}")
        print()

    print(f"Total trials: {len(rows)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial-label", help="Folder/group this trial belongs to (printed by run_semantic_episode.py)")
    p.add_argument("--task", default="", help="Task instruction that was run (for reference/notes)")
    p.add_argument("--context-mode", choices=CONTEXT_MODES)
    p.add_argument("--scene-graph-format", choices=SCENE_GRAPH_FORMATS, default="")
    p.add_argument("--grasp", type=int, choices=[0, 1])
    p.add_argument("--success", type=int, choices=[0, 1])
    p.add_argument("--ttc", type=float, default=float("nan"))
    p.add_argument("--failure-mode", choices=FAILURE_MODES, default="none")
    p.add_argument("--media-dir", default="",
                    help="Path to the snapshot/video/scene-graph-log folder for this trial "
                         "(printed by run_semantic_episode.py)")
    p.add_argument("--notes", default="")
    p.add_argument("--show", type=int, metavar="N", help="Show last N trials and exit")
    p.add_argument("--summary", action="store_true", help="Show summary table and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.summary:
        show_summary()
        return

    if args.show:
        show_recent(args.show)
        return

    required = ["trial_label", "context_mode", "grasp", "success"]
    missing = [r for r in required if getattr(args, r.replace("-", "_")) is None]
    if missing:
        print(f"ERROR: missing required arguments: {missing}")
        print("Run with --help for usage.")
        sys.exit(1)

    log_trial(args)


if __name__ == "__main__":
    main()
