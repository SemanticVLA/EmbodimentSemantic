"""Evaluate saved episode shards with the unchanged VLM bowl-invariant scorer.

Missing whole shards require --allow-partial and can never produce a final score.
Incomplete/malformed finalized shards fail closed; prediction errors are retained.
No masks, relations, or prediction identities are regenerated or tuned here.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SamGraph/src"))
sys.path.insert(0, str(REPO / "vlm_benchmarking"))
from score_existing_graphs import scoring_triplets


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_counts(frames):
    tp = sum(len(f.tp) for f in frames)
    fp = sum(len(f.fp) for f in frames)
    fn = sum(len(f.fn) for f in frames)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", default="0:50")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start, stop = map(int, args.episodes.split(":"))
    if not 0 <= start < stop <= 50:
        raise ValueError("episodes must be a half-open range within 0:50")
    episodes = list(range(start, stop))
    files = {e: args.prediction_root / f"full_{e}/SamGraph/artifacts/native_{e}/predictions.jsonl"
             for e in episodes}
    missing = [e for e, p in files.items() if not p.is_file()]
    if missing and not args.allow_partial:
        raise RuntimeError(f"No final score: missing episode shards {missing}")
    files = {e: p for e, p in files.items() if e not in missing}
    if not files:
        raise RuntimeError("No finalized prediction shards")

    code_files = [Path(__file__), REPO / "SamGraph/scripts/evaluate_libero_single_episode.py",
                  REPO / "SamGraph/src/samgraph_benchmark/metrics.py",
                  REPO / "SamGraph/src/samgraph_benchmark/ground_truth.py",
                  REPO / "vlm_benchmarking/vlm_bench/eval.py",
                  REPO / "vlm_benchmarking/vlm_bench/io_utils.py",
                  REPO / "vlm_benchmarking/vlm_bench/prompts.py"]
    code_hashes = {str(p.relative_to(REPO)): sha(p) for p in code_files}
    source_hashes = {str(e): {"path": str(p), "sha256": sha(p)} for e, p in files.items()}
    predicted = {}
    error_rows = []
    for episode, path in files.items():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["task"], row["demo"], row["frame"])
                if (row["demo"] != f"demo_{episode}" or type(row["frame"]) is not int
                        or row["frame"] < 0 or row["frame"] % 5 or key in predicted):
                    raise ValueError(f"Unexpected or duplicated frame: {key}")
                triplets = row["triplets"]
                if not isinstance(triplets, list) or any(
                    not isinstance(t, (list, tuple)) or len(t) != 3
                    or not all(isinstance(x, str) for x in t) for t in triplets
                ):
                    raise ValueError(f"Malformed triplets: {key}")
                predicted[key] = scoring_triplets(triplets, bowl_policy="vlm-swap")
                if row.get("error"):
                    error_rows.append({"key": key, "error": row["error"]})
    print(f"Frozen {len(files)} shards, {len(predicted)} prediction frames", flush=True)

    # Evaluation-only access: exclusively graph labels, never world coordinates,
    # boxes, segmentation, or RGB. Hash each exact HDF5 graph payload consumed.
    import h5py
    from samgraph_benchmark.ground_truth import canonical_triplet, hdf5_task_paths
    from samgraph_benchmark.metrics import ground_truth_triplet_set
    from vlm_bench.eval import FrameResult, aggregate
    from vlm_bench.io_utils import parse_triplets
    paths = hdf5_task_paths(args.hdf5_root)
    if len(paths) != 10:
        raise ValueError(f"Expected 10 task HDF5 files, found {len(paths)}")
    gt = {}
    expected_all = set()
    gt_hashes = {}
    for path in paths:
        task = path.stem.removesuffix("_demo")
        graph_hashes = {}
        with h5py.File(path, "r") as handle:
            for episode in episodes:
                demo = f"demo_{episode}"
                dataset = f"data/{demo}/obs/agentview_scene_graph"
                raw = handle[dataset][()]
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                graph_hashes[dataset] = hashlib.sha256(raw).hexdigest()
                values = json.loads(raw)
                for frame in range(0, len(values), 5):
                    key = (task, demo, frame)
                    expected_all.add(key)
                    if episode in files:
                        canonical = [canonical_triplet(t) for t in values[frame]]
                        if any(t is None for t in canonical):
                            raise ValueError(f"Malformed ground truth at {key}")
                        gt[key] = ground_truth_triplet_set(canonical)
        gt_hashes[path.name] = {"path": str(path), "file_size_bytes": path.stat().st_size,
                               "graph_dataset_sha256": graph_hashes}
    if set(predicted) != set(gt):
        raise ValueError(f"Finalized shard coverage mismatch: missing={len(set(gt)-set(predicted))}, "
                         f"extra={len(set(predicted)-set(gt))}")

    frames = []
    swapped_frames = 0
    for key in sorted(predicted):
        parsed = set(parse_triplets(json.dumps(sorted(predicted[key]))))
        if parsed != predicted[key]:
            raise ValueError(f"VLM parser parity failure at {key}")
        result = FrameResult(task=key[0], demo=key[1], frame=key[2], camera="agentview",
                             gt=gt[key], pred=parsed)
        swapped_frames += result.pred != parsed
        frames.append(result)
    vlm = aggregate(frames)
    counts = exact_counts(frames)
    if (vlm.n_tp, vlm.n_fp, vlm.n_fn, vlm.micro_f1) != (
        counts["tp"], counts["fp"], counts["fn"], round(counts["f1"], 4)
    ):
        raise ValueError("Independent ordered-triplet count parity failure")
    by_task = defaultdict(list)
    for frame in frames:
        by_task[frame.task].append(frame)
    per_task = {task: {**exact_counts(group), "frames": len(group),
                       "episodes": len({f.demo for f in group})}
                for task, group in sorted(by_task.items())}
    for episode, path in files.items():
        if sha(path) != source_hashes[str(episode)]["sha256"]:
            raise ValueError("Predictions changed during evaluation")
    if any(sha(REPO / p) != digest for p, digest in code_hashes.items()):
        raise ValueError("Scorer source changed during evaluation")
    value = {
        "schema": "samgraph.batch_graph_evaluation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial" if missing else "final",
        "camera": "agentview", "frame_stride": 5,
        "bowl_policy": "original VLM per-frame bowl swap; evaluation-only provisional aliases",
        "tasks": 10, "episode_indices_expected": episodes,
        "episode_indices_scored": sorted(files), "episode_indices_missing": missing,
        "task_episodes_expected": 10 * len(episodes), "task_episodes_scored": 10 * len(files),
        "frames_expected": len(expected_all), "frames_scored": len(frames),
        "missing_frames": len(expected_all - set(predicted)),
        "missing_frame_keys": sorted(expected_all - set(predicted)),
        "full_500_result_available": not missing and episodes == list(range(50)),
        "exact": counts, "per_task": per_task, "vlm_aggregate": asdict(vlm),
        "swapped_frames": swapped_frames, "prediction_error_rows": error_rows,
        "prediction_sources": source_hashes, "scorer_source_sha256": code_hashes,
        "ground_truth_sources": gt_hashes, "vlm_ordered_scorer_parity": True,
        "predictions_modified": False, "rules_tuned_by_this_scorer": False,
        "scope_note": "Metrics include every frame of available finalized shards; "
                      "unavailable shards are explicitly missing, never silently scored or dropped from coverage.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({k: value[k] for k in ("status", "task_episodes_scored", "task_episodes_expected",
                     "frames_scored", "frames_expected", "exact", "per_task")}, indent=2))


if __name__ == "__main__":
    main()
