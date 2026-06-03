#!/usr/bin/env python3
"""
Evaluate JSONL model logs against GT scene graphs stored in HDF5 files.

Prediction CSVs may still be generated for inspection, but benchmark metrics are
computed from raw JSONL responses so empty parsed outputs and missing frames are
handled explicitly.

Usage examples:
  python evaluate.py --jsonl output/model/agentview/json/task_agentview_v4.jsonl --input-dir data/libero_spatial_v5/
  python evaluate.py --model-dir output/model/ --input-dir data/libero_spatial_v5/ --verbose
  python evaluate.py --model-dir output/model-a/ output/model-b/ --input-dir data/libero_spatial_v5/
"""

import argparse
import csv as csv_mod
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from vlm_bench.eval import (
    aggregate,
    evaluate_jsonl,
    jsonl_prediction_counts,
    print_aggregate_report,
    print_frame_report,
)
from vlm_bench.io_utils import rows_from_jsonl, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM scene graph JSONL predictions")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", nargs="+", help="One or more prediction JSONL log files")
    src.add_argument("--model-dir", nargs="+", help="One or more model output folders; JSONLs are discovered recursively")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDF5 files")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=None, help="Use only first N frame indices from config.yaml")
    parser.add_argument(
        "--cameras",
        "--camera",
        dest="cameras",
        nargs="+",
        default=None,
        help="Filter cameras to evaluate (e.g., agentview, eye_in_hand).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-frame TP/FP/FN breakdown")
    parser.add_argument("--save-csv", default=None, help="Save per-frame metric rows to this CSV path")
    parser.add_argument(
        "--reparse-jsonl-to-csv",
        action="store_true",
        help="Regenerate parsed triplet CSV artifacts from the raw JSONL responses",
    )
    parser.add_argument("--validate-counts", action="store_true", help="Print prediction-count reconciliation")
    return parser.parse_args()


def collect_jsonls(args) -> list[Path]:
    if args.jsonl:
        return [Path(p) for p in args.jsonl]
    paths = []
    for d in args.model_dir:
        paths.extend(sorted(Path(d).rglob("*.jsonl")))
    return sorted(list(dict.fromkeys(paths)))


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

    extraction = config["extraction"]
    if "frame_step" in extraction:
        step = extraction["frame_step"]
        step = 1 if step == 0 else step
        frame_indices = list(range(0, extraction.get("frame_max", 10000), step))
    else:
        frame_indices = extraction.get("frame_indices", [0])

    if args.frames is not None:
        frame_indices = frame_indices[: args.frames]

    input_dir = Path(args.input_dir)
    jsonl_paths = collect_jsonls(args)

    if not jsonl_paths:
        print("No JSONL files found.")
        return

    if args.reparse_jsonl_to_csv:
        for jsonl_path in jsonl_paths:
            out_path = reparse_jsonl_to_csv(jsonl_path)
            print(f"  [reparse] {jsonl_path.name} -> {out_path}")

    all_results = []
    save_rows = []

    def _eval_one(path: Path):
        return path, list(evaluate_jsonl(path, input_dir, args.cameras, frame_indices))

    n_workers = min(len(jsonl_paths), os.cpu_count() or 4)
    ordered_results: dict[Path, list] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_one, p): p for p in jsonl_paths}
        for future in as_completed(futures):
            path, results = future.result()
            ordered_results[path] = results

    for path in jsonl_paths:
        results = ordered_results[path]
        if not results:
            print(f"  [skip] {path.name} - no matching frames")
            continue

        if args.validate_counts:
            counts = jsonl_prediction_counts(path, args.cameras, frame_indices)
            m_check = aggregate(results)
            print(
                f"  [counts] {path.name}: records={counts.records_total}, "
                f"selected={counts.records_selected}, record_frames={counts.scored_frames}, "
                f"metric_frames={m_check.n_frames}, parsed_pred={counts.parsed_triplets}, "
                f"scored_pred={counts.scored_triplets}, metric_pred={m_check.n_pred_triplets}, "
                f"duplicates={counts.duplicate_records}, dedup_drop={counts.dedup_triplets_dropped}, "
                f"empty={counts.empty_records}, bad_json={counts.bad_json_lines}"
            )

        for r in results:
            if args.verbose:
                print_frame_report(r, verbose=True)
            all_results.append(r)
            save_rows.append(
                {
                    "file": path.name,
                    "task": r.task,
                    "demo": r.demo,
                    "frame": r.frame,
                    "camera": r.camera,
                    "n_gt": len(r.gt),
                    "n_pred": len(r.pred),
                    "tp": len(r.tp),
                    "fp": len(r.fp),
                    "fn": len(r.fn),
                    "precision": f"{r.precision:.4f}",
                    "recall": f"{r.recall:.4f}",
                    "f1": f"{r.f1:.4f}",
                }
            )

        if len(jsonl_paths) == 1 or args.verbose:
            print_aggregate_report(aggregate(results), label=path.name)

    if len(jsonl_paths) > 1:
        print_aggregate_report(aggregate(all_results), label="ALL FILES COMBINED")

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
