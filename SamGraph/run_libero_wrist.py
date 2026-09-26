"""Render-independent wrist extension: localize/track membership, filter frozen graphs, export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist-frames-root", type=Path, required=True)
    parser.add_argument("--agent-frames-root", type=Path, required=True)
    parser.add_argument("--agent-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--names-config", type=Path,
                        default=ROOT / "config" / "libero_wrist_object_prompts.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tasks", nargs="+",
                        help="Optional task subset; defaults to tasks present in the RGB archives.")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0],
                        help="Episode indices to process (default: 0).")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT.parent / "vlm_benchmarking" / "output")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--mask-checkpoint-root", type=Path,
        help="Optional contiguous per-task sampled-mask checkpoint to replay before native tracking resumes.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id):
        parser.error("run-id must be a simple, unique name")
    output = args.output_root / ("samgraph-wrist-" + args.run_id)
    if output.exists():
        raise FileExistsError(output)
    names_value = json.loads(args.names_config.read_text(encoding="utf-8"))
    if set(names_value) == {"defaults", "tasks", "overrides"}:
        defaults = names_value["defaults"]
        names = {task: {key: list(values) for key, values in defaults.items()}
                 for task in names_value["tasks"]}
        for task, classes in names_value["overrides"].items():
            for class_id, values in classes.items():
                names[task][class_id] = list(values)
    else:
        names = names_value
    from samgraph_wrist.pipeline import WristExtensionRunner
    from samgraph_wrist.export import export_wrist_predictions
    runner = WristExtensionRunner(args.checkpoint, names)
    try:
        run_manifest = runner.run(
            wrist_frames_root=args.wrist_frames_root,
            agent_frames_root=args.agent_frames_root,
            agent_predictions=args.agent_predictions,
            output=output / "artifacts",
            mask_checkpoint_root=args.mask_checkpoint_root,
            episodes=tuple(args.episodes),
            tasks=tuple(args.tasks) if args.tasks else None,
        )
    finally:
        runner.close()
    export_manifest = export_wrist_predictions(
        output / "artifacts", args.wrist_frames_root, output,
        write_videos=not args.no_videos,
    )
    print(json.dumps({"output": str(output.resolve()),
                      "run": run_manifest, "export": export_manifest}, indent=2))


if __name__ == "__main__":
    main()
