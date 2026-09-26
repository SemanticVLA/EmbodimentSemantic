"""Score immutable saved triplets; never regenerate masks or tune rules."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SamGraph" / "src"))
sys.path.insert(0, str(REPO / "vlm_benchmarking"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoring_triplets(triplets, *, bowl_policy):
    """Normalize vocabulary, not mask identities; only for evaluation."""
    from samgraph_benchmark.metrics import triplet_set
    canonical = triplet_set(triplets)
    if bowl_policy == "strict":
        return canonical
    aliases = {"black_bowl_track_1": "akita_black_bowl_1",
               "black_bowl_track_2": "akita_black_bowl_2"}
    endpoints = {x for a, _, b in canonical for x in (a, b)}
    for source, target in aliases.items():
        if source in endpoints and target in endpoints:
            raise ValueError("provisional and named bowl IDs would collide")
    return {(aliases.get(a, a), relation, aliases.get(b, b))
            for a, relation, b in canonical}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--hdf5-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bowl-policy", choices=("vlm-swap", "strict"), default="vlm-swap",
                   help="default uses the actual VLM per-frame duplicate-bowl scorer")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hash = sha(args.predictions)
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    predicted = {(r["task"], r["demo"], r["frame"]): r["triplets"] for r in rows}
    if len(predicted) != len(rows) or len(rows) != 258 or len({r["task"] for r in rows}) != 10:
        raise ValueError("expected the accepted 258-row ten-task run")
    if any(r["demo"] != "demo_0" or r["frame"] % 5 or r.get("error") for r in rows):
        raise ValueError("unexpected demo/cadence or prediction error")
    code_paths = [Path(__file__), REPO / "SamGraph/src/samgraph_benchmark/metrics.py",
                  REPO / "SamGraph/src/samgraph_benchmark/ground_truth.py",
                  REPO / "vlm_benchmarking/vlm_bench/eval.py",
                  REPO / "vlm_benchmarking/vlm_bench/io_utils.py"]
    code_hashes = {str(path.relative_to(REPO)): sha(path) for path in code_paths}
    # Predictions are fixed before evaluation-only HDF5 access.
    from samgraph_benchmark.ground_truth import GroundTruthIndex, hdf5_task_paths, load_ground_truth
    from samgraph_benchmark.metrics import evaluate_predictions, ground_truth_triplet_set, triplet_set
    from vlm_bench.eval import FrameResult, aggregate
    from vlm_bench.io_utils import parse_triplets

    class ExactFrameResult(FrameResult):
        def __post_init__(self):
            # Same frozen ordered scorer as the prior SamGraph benchmark:
            # no GT-selected bowl permutation, no future relabeling.
            pass

    paths = hdf5_task_paths(args.hdf5_root)
    all_gt = load_ground_truth(paths)
    pairs = {(k[0], k[1]) for k in predicted}
    gt = GroundTruthIndex({k: v for k, v in all_gt.frames.items() if k[:2] in pairs and k[2] % 5 == 0})
    if set(gt.keys()) != set(predicted):
        raise ValueError("HDF5 sampled denominator differs from saved predictions")
    strict_report = evaluate_predictions(gt, predicted, frame_stride=5)
    frames = []
    swapped_frames = 0
    for key in sorted(predicted):
        canonical = scoring_triplets(predicted[key], bowl_policy=args.bowl_policy)
        parsed = set(parse_triplets(json.dumps(sorted(canonical))))
        if parsed != canonical:
            raise ValueError("VLM parser parity failure")
        frame_type = FrameResult if args.bowl_policy == "vlm-swap" else ExactFrameResult
        result = frame_type(task=key[0], demo=key[1], frame=key[2], camera="agentview",
                            gt=ground_truth_triplet_set(gt.frames[key]), pred=parsed)
        swapped_frames += result.pred != parsed
        frames.append(result)
    vlm = aggregate(frames)
    # Independently count the VLM-selected ordered triplets, not an alternative
    # bowl-matching implementation. The benchmark scorer itself owns the swap.
    scored = {(r.task, r.demo, r.frame): sorted(r.pred) for r in frames}
    report = evaluate_predictions(gt, scored, frame_stride=5)
    counts = report.exact
    if (vlm.n_tp, vlm.n_fp, vlm.n_fn, vlm.micro_f1) != (counts["tp"], counts["fp"], counts["fn"], counts["f1"]):
        raise ValueError("VLM ordered micro-F1 parity failure")
    for task, metrics in report.exact_per_task.items():
        if vlm.per_task_metrics[task]["micro_f1"] != metrics["f1"]:
            raise ValueError("per-task VLM parity failure")
    if sha(args.predictions) != source_hash or any(sha(REPO / path) != digest for path, digest in code_hashes.items()):
        raise ValueError("prediction/scorer changed during evaluation")
    value = {"schema": "samgraph.existing_graph_evaluation.v2", "frames": len(rows), "tasks": 10,
             "episodes_per_task": 1, "frame_stride": 5, "camera": "agentview",
             "exact": counts, "per_task": report.exact_per_task,
             "micro_f1_unrounded": 2*counts["tp"]/(2*counts["tp"]+counts["fp"]+counts["fn"]),
             "prediction_sha256": source_hash, "scorer_source_sha256": code_hashes,
             "ground_truth": {path.name: sha(path) for path in paths},
             "vlm_ordered_scorer_parity": True, "vlm_aggregate": asdict(vlm),
             "predictions_modified": False, "rules_tuned_by_this_scorer": False,
             "bowl_policy": args.bowl_policy, "swapped_frames": swapped_frames,
             "strict_raw_label_diagnostic": strict_report.exact,
             "identity_policy": ("evaluation-only provisional bowl aliases; original VLM per-frame bowl swap"
                                 if args.bowl_policy == "vlm-swap" else
                                 "static label aliases only; provisional IDs retained; no GT-selected bowl swap")}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({k:v for k,v in value.items() if k in {"frames","tasks","exact","per_task","micro_f1_unrounded"}}, indent=2))


if __name__ == "__main__":
    main()
