#!/usr/bin/env python3
"""
Evaluate predicted CSVs against GT scene graphs stored in HDF5 files.

Usage examples:
  # single CSV
  python evaluate.py --csv output/gemini-3.1-pro/agentview/csv/task_agentview_v4.csv --input-dir data/libero_spatial_v5/

  # entire model output folder (recursively discovers nested CSVs)
  python evaluate.py --model-dir output/gemini-3.1-pro/ --input-dir data/libero_spatial_v5/ --verbose

  # compare two models side by side
  python evaluate.py --model-dir output/gemini-3.1-pro/ output/gpt-5.1/ --input-dir data/libero_spatial_v5/

  # unidirectional evaluation (only 4 canonical relations)
  python evaluate.py --model-dir output/gemini-3.1-pro/ --input-dir data/libero_spatial_v5/ --direction u
"""

import argparse
import csv as csv_mod
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vlm_bench.eval import (
    aggregate,
    csv_prediction_counts,
    evaluate_csv,
    evaluate_jsonl,
    jsonl_prediction_counts,
    print_aggregate_report,
    print_frame_report,
)
from vlm_bench.io_utils import rows_from_jsonl, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM scene graph predictions")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", nargs="+", help="One or more prediction CSV files")
    src.add_argument("--jsonl", nargs="+", help="One or more prediction JSONL log files (responses re-parsed)")
    src.add_argument("--model-dir", nargs="+", help="One or more model output folders (all CSVs inside evaluated)")
    src.add_argument("--model-dir-jsonl", nargs="+", help="One or more model output folders (all JSONL logs re-parsed)")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--direction", choices=["b", "u"], default="b",
                        help="b=bidirectional (8 relations), u=unidirectional (4 canonical relations)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Use only first N frame indices from config.yaml")
    parser.add_argument("--cameras", "--camera", dest="cameras", nargs="+", default=None,
                        help="Filter cameras to evaluate (e.g., agentview, eye_in_hand). Evaluates all in CSV if omitted.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-frame TP/FP/FN breakdown")
    parser.add_argument("--save-csv", default=None, help="Save per-frame results to this CSV path")
    parser.add_argument("--reparse-jsonl-to-csv", action="store_true",
                        help="When evaluating JSONL, regenerate parsed triplet CSVs from the raw JSONL responses")
    parser.add_argument("--validate-counts", action="store_true",
                        help="Print prediction-count reconciliation for each evaluated file")
    return parser.parse_args()

def collect_csvs(args) -> list[Path]:
    if args.csv:
        return [Path(p) for p in args.csv]
    if args.model_dir:
        paths = []
        for d in args.model_dir:
            paths.extend(p for p in sorted(Path(d).rglob("*.csv")) if p.parent.name != "json")
        return sorted(list(dict.fromkeys(paths)))
    return []


def collect_jsonls(args) -> list[Path]:
    if args.jsonl:
        return [Path(p) for p in args.jsonl]
    if args.model_dir_jsonl:
        paths = []
        for d in args.model_dir_jsonl:
            paths.extend(sorted(Path(d).rglob("*.jsonl")))
        return sorted(list(dict.fromkeys(paths)))
    return []


def parsed_csv_path_for_jsonl(jsonl_path: Path) -> Path:
    if jsonl_path.parent.name == "json":
        return jsonl_path.parent.parent / "csv" / f"{jsonl_path.stem}.csv"
    return jsonl_path.with_suffix(".csv")


def reparse_jsonl_to_csv(jsonl_path: Path) -> Path:
    out_path = parsed_csv_path_for_jsonl(jsonl_path)
    rows = rows_from_jsonl(jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(str(out_path), rows)
    return out_path


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cameras = args.cameras
    if "frame_step" in config["extraction"]:
        step = config["extraction"]["frame_step"]
        step = 1 if step == 0 else step
        frame_indices = list(range(0, config["extraction"].get("frame_max", 10000), step))
    else:
        frame_indices = config["extraction"].get("frame_indices", [0])
        
    if args.frames is not None:
        frame_indices = frame_indices[:args.frames]
    input_dir = Path(args.input_dir)

    csv_paths = collect_csvs(args)
    jsonl_paths = collect_jsonls(args)

    if not csv_paths and not jsonl_paths:
        print("No files found.")
        return

    if args.reparse_jsonl_to_csv:
        if not jsonl_paths:
            print("--reparse-jsonl-to-csv only applies to --jsonl or --model-dir-jsonl inputs.")
            return
        for jsonl_path in jsonl_paths:
            out_path = reparse_jsonl_to_csv(jsonl_path)
            print(f"  [reparse] {jsonl_path.name} -> {out_path}")

    all_results = []
    save_rows = []
    all_paths = [("csv", p) for p in csv_paths] + [("jsonl", p) for p in jsonl_paths]

    def _eval_one(kind_path):
        kind, path = kind_path
        if kind == "csv":
            return path, list(evaluate_csv(path, input_dir, cameras, frame_indices, direction=args.direction))
        else:
            return path, list(evaluate_jsonl(path, input_dir, cameras, frame_indices, direction=args.direction))

    n_workers = min(len(all_paths), os.cpu_count() or 4)
    ordered_results: dict[Path, list] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_one, kp): kp for kp in all_paths}
        for future in as_completed(futures):
            path, results = future.result()
            ordered_results[path] = results

    for _, path in all_paths:
        results = ordered_results[path]
        if not results:
            print(f"  [skip] {path.name} — no matching frames")
            continue

        if args.validate_counts:
            if path.suffix.lower() == ".jsonl":
                counts = jsonl_prediction_counts(path, cameras, frame_indices, direction=args.direction)
            else:
                counts = csv_prediction_counts(path, cameras, frame_indices, direction=args.direction)
            m_check = aggregate(results, direction=args.direction)
            print(
                f"  [counts] {path.name}: records={counts.records_total}, "
                f"selected={counts.records_selected}, frames={counts.scored_frames}, "
                f"parsed_pred={counts.parsed_triplets}, scored_pred={counts.scored_triplets}, "
                f"metric_pred={m_check.n_pred_triplets}, duplicates={counts.duplicate_records}, "
                f"dedup_drop={counts.dedup_triplets_dropped}, empty={counts.empty_records}, "
                f"bad_json={counts.bad_json_lines}"
            )

        for r in results:
            if args.verbose:
                print_frame_report(r, verbose=True)
            all_results.append(r)
            save_rows.append({
                "file": path.name,
                "task": r.task,
                "demo": r.demo,
                "frame": r.frame,
                "camera": r.camera,
                "direction": args.direction,
                "n_gt": len(r.gt),
                "n_pred": len(r.pred),
                "tp": len(r.tp),
                "fp": len(r.fp),
                "fn": len(r.fn),
                "precision": f"{r.precision:.4f}",
                "recall": f"{r.recall:.4f}",
                "f1": f"{r.f1:.4f}",
            })

        if len(all_paths) == 1 or args.verbose:
            m = aggregate(results, direction=args.direction)
            print_aggregate_report(m, label=path.name, direction=args.direction)

    if len(all_paths) > 1:
        m_all = aggregate(all_results, direction=args.direction)
        print_aggregate_report(m_all, label="ALL FILES COMBINED", direction=args.direction)

    if args.save_csv and save_rows:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(save_rows[0].keys()))
            writer.writeheader()
            writer.writerows(save_rows)
        print(f"\nPer-frame results saved to {out}")


if __name__ == "__main__":
    main()
