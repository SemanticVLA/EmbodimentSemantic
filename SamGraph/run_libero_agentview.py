"""Single entry point for automatic SAM masks, graphs, arrows and benchmark CSVs.

Requires the pinned SAM3.1 GPU environment for prediction. --predictions exports
an existing run on CPU without re-running SAM. Evaluation is intentionally absent.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--geometry-config", type=Path,
                        help="explicit mask-only relation profile; omitted keeps the existing default")
    parser.add_argument("--names-config", type=Path,
                        default=ROOT / "config" / "libero_spatial_object_prompts.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episodes", default="0",
                        help="episode IDs: 0, 0,1, 0:50 (exclusive end), or all")
    parser.add_argument("--predictions", type=Path,
                        help="export saved predictions only; no GPU inference")
    parser.add_argument("--allow-partial-export", action="store_true",
                        help="diagnostic export only; waive complete ten-task sampled coverage")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT.parent / "vlm_benchmarking" / "output")
    args = parser.parse_args()
    from samgraph_benchmark.frames import parse_episode_selection
    episodes = parse_episode_selection(args.episodes)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id):
        parser.error("run-id must be a simple, unique name")
    output = args.output_root / ("samgraph-" + args.run_id)
    if output.exists():
        raise FileExistsError(output)
    predictions = args.predictions
    if predictions is not None and args.geometry_config is not None:
        parser.error("--predictions exports unchanged saved triplets; geometry overrides require inference or a separate CPU replay")
    if predictions is None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for inference")
        from samgraph_benchmark.cli import main as predict_main
        relative = Path(args.run_id) / "predictions.jsonl"
        predict_args = ["predict", "--frames-root", str(args.frames_root),
                      "--checkpoint", str(args.checkpoint), "--mode", "automatic",
                      "--names-config", str(args.names_config),
                      "--episodes", args.episodes,
                      "--frame-stride", "5", "--tracking-stride", "1",
                      "--output", str(relative)]
        if args.geometry_config is not None:
            predict_args.extend(["--geometry-config", str(args.geometry_config)])
        predict_main(predict_args)
        predictions = ROOT / "artifacts" / relative
    from samgraph_benchmark.export import export_predictions
    manifest = export_predictions(predictions, args.frames_root, output,
                                  allow_partial=args.allow_partial_export, episodes=episodes)
    print(json.dumps({"output": str(output.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
