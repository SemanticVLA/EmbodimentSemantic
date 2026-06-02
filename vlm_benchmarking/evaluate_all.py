#!/usr/bin/env python3
"""
Evaluate all models in output/ and write a single paper_results.csv.

One row per model. Columns are grouped by camera (agentview_*, eye_in_hand_*).
Missing camera data is left blank so it's clear no output exists.
A `complete_{camera}` column flags whether the JSONL coverage matches what the
HDF5 ground-truth expects (same logic as runner.py's resume check).

Usage:
    python evaluate_all.py --input-dir data/libero_spatial_v5/
    python evaluate_all.py --input-dir data/libero_spatial_v5/ --output-dir output/ --direction b
"""

import argparse
import csv
import json
import math
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py

from vlm_bench.eval import evaluate_csv, evaluate_jsonl, aggregate
from vlm_bench.io_utils import image_key_for_camera, list_hdf5_files, rows_from_jsonl, write_csv
from vlm_bench.prompts import BIDIRECTIONAL_RELATIONS, UNIDIRECTIONAL_RELATIONS

CAMERAS = ["agentview", "eye_in_hand"]
SKIP_DIRS = {"archive"}

_RELATIONS_B = sorted(BIDIRECTIONAL_RELATIONS)
_RELATIONS_U = sorted(UNIDIRECTIONAL_RELATIONS)


def _relation_columns(direction: str) -> list[str]:
    return _RELATIONS_B if direction == "b" else _RELATIONS_U


def _metric_columns(direction: str) -> list[str]:
    rels = _relation_columns(direction)
    base = [
        "complete",
        "n_frames_expected", "n_frames_done",
        "n_frames",
        "macro_f1", "macro_precision", "macro_recall",
        "micro_f1", "micro_precision", "micro_recall",
        "mean_per_task_f1",
        "coverage", "hallucination_rate", "reversal_rate", "direction_consistency",
    ]
    per_rel = [f"{r}_f1" for r in rels]
    return base + per_rel


def _fieldnames(direction: str) -> list[str]:
    cols = _metric_columns(direction)
    fields = ["model"]
    for cam in CAMERAS:
        for col in cols:
            fields.append(f"{col}_{cam}")
    return fields


def _metrics_to_dict(m, direction: str, complete: bool, n_expected: int, n_done: int) -> dict:
    rels = _relation_columns(direction)
    dc = m.direction_consistency
    row = {
        "complete": "yes" if complete else "no",
        "n_frames_expected": n_expected,
        "n_frames_done": n_done,
        "n_frames": m.n_frames,
        "macro_f1": round(m.f1, 4),
        "macro_precision": round(m.precision, 4),
        "macro_recall": round(m.recall, 4),
        "micro_f1": m.micro_f1,
        "micro_precision": m.micro_precision,
        "micro_recall": m.micro_recall,
        "mean_per_task_f1": m.mean_per_task_f1,
        "coverage": m.coverage,
        "hallucination_rate": m.hallucination_rate,
        "reversal_rate": m.reversal_rate,
        "direction_consistency": "" if math.isnan(dc) else dc,
    }
    for rel in rels:
        row[f"{rel}_f1"] = m.per_relation.get(rel, {}).get("f1", "")
    return row


def _jsonl_done_pairs(jsonl_dir: Path) -> set[tuple[str, int]]:
    """Return set of (demo, frame) pairs present across all JSONL files in the directory."""
    done: set[tuple[str, int]] = set()
    if not jsonl_dir.is_dir():
        return done
    for jsonl_path in jsonl_dir.glob("*.jsonl"):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["demo"], int(rec["frame"])))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _completeness_for_jsonl_dir(
    jsonl_dir: Path,
    input_dir: Path,
    frame_indices: list[int],
    camera: str,
) -> tuple[int, int]:
    """Return (n_expected, n_done), counting missing task JSONLs as missing work."""
    import re
    frame_set = set(frame_indices)
    hdf5_by_stem = {
        hdf5_path.stem.replace("_demo", ""): hdf5_path
        for hdf5_path in list_hdf5_files(str(input_dir))
    }

    done_by_task: dict[str, set[tuple[str, int]]] = {}
    if jsonl_dir.is_dir():
        for jsonl_path in jsonl_dir.glob("*.jsonl"):
            task_name = re.sub(r"_(agentview|eye_in_hand)_v\d+$", "", jsonl_path.stem)
            task_done: set[tuple[str, int]] = set()
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        task_done.add((rec["demo"], int(rec["frame"])))
                    except (json.JSONDecodeError, KeyError):
                        continue
            done_by_task.setdefault(task_name, set()).update(task_done)

    n_expected = 0
    n_done = 0
    img_key = image_key_for_camera(camera)

    for task_name, hdf5_path in hdf5_by_stem.items():
        task_expected: set[tuple[str, int]] = set()
        with h5py.File(hdf5_path, "r") as f:
            data_group = f["data"] if "data" in f else f
            for demo_key in data_group.keys():
                try:
                    n_frames = len(data_group[f"{demo_key}/obs/{img_key}"])
                except KeyError:
                    continue
                for idx in frame_set:
                    if idx < n_frames:
                        task_expected.add((demo_key, idx))

        task_done = done_by_task.get(task_name, set())
        n_expected += len(task_expected)
        n_done += len(task_done & task_expected)

    return n_expected, n_done


def _expected_pairs(input_dir: Path, frame_indices: list[int]) -> set[tuple[str, int]]:
    """Return set of (demo, frame) pairs expected based on frame_indices and HDF5 contents."""
    frame_set = set(frame_indices)
    pairs: set[tuple[str, int]] = set()
    for hdf5_path in list_hdf5_files(str(input_dir)):
        with h5py.File(hdf5_path, "r") as f:
            data_group = f["data"] if "data" in f else f
            for demo_key in data_group.keys():
                try:
                    n_frames = len(data_group[f"{demo_key}/obs/agentview_rgb"])
                except KeyError:
                    continue
                for idx in frame_set:
                    if idx < n_frames:
                        pairs.add((demo_key, idx))
    return pairs


def _parsed_csv_path_for_jsonl(model_dir: Path, camera: str, jsonl_path: Path) -> Path:
    return model_dir / camera / "csv" / f"{jsonl_path.stem}.csv"


def _write_parsed_csvs(model_dir: Path, camera: str, jsonl_paths: list[Path]) -> None:
    for jsonl_path in jsonl_paths:
        out_path = _parsed_csv_path_for_jsonl(model_dir, camera, jsonl_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(str(out_path), rows_from_jsonl(jsonl_path))


def _eval_model_camera(
    model_dir: Path,
    camera: str,
    input_dir: Path,
    frame_indices: list[int],
    direction: str,
    source: str,
    write_parsed_csvs: bool,
) -> tuple[str, str, dict | None]:
    csv_dir = model_dir / camera / "csv"
    jsonl_dir = model_dir / camera / "json"

    csv_paths = sorted(csv_dir.glob("*.csv")) if csv_dir.is_dir() else []
    jsonl_paths = sorted(jsonl_dir.glob("*.jsonl")) if jsonl_dir.is_dir() else []

    if source == "csv" and not csv_paths:
        return model_dir.name, camera, None
    if source == "jsonl" and not jsonl_paths:
        return model_dir.name, camera, None

    n_expected, n_done = _completeness_for_jsonl_dir(jsonl_dir, input_dir, frame_indices, camera)
    complete = n_expected > 0 and n_done >= n_expected

    if source == "jsonl" and write_parsed_csvs:
        _write_parsed_csvs(model_dir, camera, jsonl_paths)

    results = []
    if source == "csv":
        for csv_path in csv_paths:
            results.extend(evaluate_csv(csv_path, input_dir, [camera], frame_indices, direction=direction))
    else:
        for jsonl_path in jsonl_paths:
            results.extend(evaluate_jsonl(jsonl_path, input_dir, [camera], frame_indices, direction=direction))

    if not results:
        return model_dir.name, camera, None

    m = aggregate(results, direction=direction)
    status = "COMPLETE" if complete else f"INCOMPLETE ({n_done}/{n_expected})"
    print(f"  [{model_dir.name} / {camera} / {source}] {status}  eval_frames={m.n_frames}  macro_f1={m.f1:.4f}")
    return model_dir.name, camera, _metrics_to_dict(m, direction, complete, n_expected, n_done)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models -> paper_results.csv")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDF5 ground-truth files")
    parser.add_argument("--output-dir", default="output", help="Root folder containing model subdirectories")
    parser.add_argument("--out", default="paper_results.csv", help="Output CSV path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--direction", choices=["b", "u"], default="b",
                        help="b=bidirectional (8 relations), u=unidirectional (4 canonical)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Use only first N frame indices from config")
    parser.add_argument("--source", choices=["jsonl", "csv"], default="jsonl",
                        help="jsonl=re-parse raw responses; csv=use existing parsed CSVs")
    parser.add_argument("--reparse-jsonl-to-csv", action="store_true",
                        help="With --source jsonl, also regenerate canonical CSV files from JSONL")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "frame_step" in config["extraction"]:
        step = config["extraction"]["frame_step"]
        step = 1 if step == 0 else step
        frame_indices = list(range(0, config["extraction"].get("frame_max", 10000), step))
    else:
        frame_indices = config["extraction"].get("frame_indices", [0])

    if args.frames is not None:
        frame_indices = frame_indices[: args.frames]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    model_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )

    if not model_dirs:
        print(f"No model directories found in {output_dir}")
        return

    print(f"Found {len(model_dirs)} model(s): {[d.name for d in model_dirs]}\n")

    step = frame_indices[1] if len(frame_indices) > 1 else 1
    print(f"Frame step={step}, source={args.source}, evaluating completeness per task against matching HDF5.\n")

    # Build work items: (model_dir, camera)
    work = [(md, cam) for md in model_dirs for cam in CAMERAS]

    # { model_name: { camera: metrics_dict | None } }
    results_map: dict[str, dict[str, dict | None]] = {md.name: {} for md in model_dirs}

    n_workers = min(len(work), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _eval_model_camera,
                md,
                cam,
                input_dir,
                frame_indices,
                args.direction,
                args.source,
                args.reparse_jsonl_to_csv,
            ): (md.name, cam)
            for md, cam in work
        }
        for future in as_completed(futures):
            model_name, camera, metrics = future.result()
            results_map[model_name][camera] = metrics

    # Assemble rows in sorted model order
    fieldnames = _fieldnames(args.direction)
    rows = []
    for md in model_dirs:
        row: dict = {"model": md.name}
        for fn in fieldnames:
            if fn != "model":
                row[fn] = ""
        for cam in CAMERAS:
            metrics = results_map[md.name].get(cam)
            if metrics is None:
                continue
            for col, val in metrics.items():
                row[f"{col}_{cam}"] = val
        rows.append(row)

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} model(s) -> {out_path}")

    # Print completeness summary
    print("\nCompleteness summary:")
    for row in rows:
        for cam in CAMERAS:
            c = row.get(f"complete_{cam}", "")
            if c:
                done = row.get(f"n_frames_done_{cam}", "")
                exp = row.get(f"n_frames_expected_{cam}", "")
                flag = "" if c == "yes" else " !"
                print(f"  {row['model']:<40} {cam:<12} {c}  ({done}/{exp}){flag}")


if __name__ == "__main__":
    main()
