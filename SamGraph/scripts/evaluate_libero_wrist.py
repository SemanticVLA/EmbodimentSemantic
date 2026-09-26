"""Evaluate demo-0 wrist JSONL with the repository's existing VLM scorer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import h5py
import yaml

from vlm_bench.eval import aggregate, evaluate_jsonl, jsonl_prediction_counts
from vlm_bench.io_utils import list_hdf5_files


CAMERA = "eye_in_hand"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_indices(config: Path) -> list[int]:
    extraction = yaml.safe_load(config.read_text(encoding="utf-8"))["extraction"]
    if "frame_step" in extraction:
        step = int(extraction["frame_step"]) or 1
        return list(range(0, int(extraction.get("frame_max", 10000)), step))
    return [int(value) for value in extraction.get("frame_indices", [0])]


def _demo0_views(source_root: Path, output_root: Path) -> list[Path]:
    """Create tiny HDF5 external-link views so the stock scorer sees only demo_0."""
    views = []
    for source in list_hdf5_files(str(source_root)):
        target = output_root / source.name
        with h5py.File(source, "r") as handle:
            if "data/demo_0/obs/robot0_eye_in_hand_scene_graph" not in handle:
                raise KeyError(f"missing wrist graph for demo_0 in {source}")
        with h5py.File(target, "w") as handle:
            data = handle.create_group("data")
            data["demo_0"] = h5py.ExternalLink(str(source.resolve()), "/data/demo_0")
        views.append(target)
    if len(views) != 10:
        raise ValueError(f"expected ten HDF5 task views, found {len(views)}")
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.json_dir.glob("*.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected ten wrist JSONL files, found {len(paths)}")
    indices = _frame_indices(args.config)
    scorer = Path(__file__).resolve().parents[3] / "vlm_benchmarking" / "vlm_bench" / "eval.py"
    ground_truth_paths = [Path(path) for path in list_hdf5_files(str(args.hdf5_root))]
    all_results = []
    count_rows = {}
    with tempfile.TemporaryDirectory(prefix="samgraph-wrist-ground-truth-") as temp:
        gt_view = Path(temp)
        _demo0_views(args.hdf5_root, gt_view)
        for path in paths:
            counts = jsonl_prediction_counts(path, [CAMERA], indices)
            count_rows[path.name] = asdict(counts)
            all_results.extend(evaluate_jsonl(path, gt_view, [CAMERA], indices))
    keys = {(row.task, row.demo, row.frame, row.camera) for row in all_results}
    if len(all_results) != 258 or len(keys) != 258:
        raise ValueError(f"existing scorer did not produce the expected 258 unique frames: {len(keys)}")
    if any(row.demo != "demo_0" or row.camera != CAMERA for row in all_results):
        raise ValueError("unexpected demo/camera in VLM scorer output")
    metrics = aggregate(all_results)
    value = {
        "schema": "samgraph.wrist_existing_vlm_evaluation.v1",
        "pipeline": "vlm_benchmarking.vlm_bench.eval.evaluate_jsonl + aggregate",
        "camera": CAMERA,
        "frames": len(all_results),
        "tasks": len({row.task for row in all_results}),
        "episodes": [0],
        "frame_indices": "config frame_step/frame_max, bounded by each HDF5 demo length",
        "bowl_policy": "existing FrameResult per-frame duplicate-bowl swap",
        "metrics": asdict(metrics),
        "prediction_counts": count_rows,
        "prediction_sha256": {path.name: _sha(path) for path in paths},
        "ground_truth_sha256": {path.name: _sha(path) for path in ground_truth_paths},
        "config_sha256": _sha(args.config),
        "scorer_source_sha256": _sha(scorer),
        "prediction_modified": False,
        "evaluation_only_hdf5_view": "external links restrict the existing scorer denominator to demo_0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({
        "frames": value["frames"],
        "tasks": value["tasks"],
        "macro_f1": metrics.f1,
        "micro_f1": metrics.micro_f1,
        "micro_precision": metrics.micro_precision,
        "micro_recall": metrics.micro_recall,
        "per_task_metrics": metrics.per_task_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
