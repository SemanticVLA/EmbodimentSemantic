"""Regenerate wrist CSV/JSON, masks, arrows, graphs, and videos on CPU.

The command consumes saved artifact predictions and hash-verified NPZ masks.
It does not run SAM or recompute spatial relations.  Arrow endpoints are the
centroids of wrist-local masks, so agent-view pixel coordinates are never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SAMGRAPH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SAMGRAPH_ROOT / "src"))

from samgraph_wrist.export import export_wrist_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root", type=Path, required=True,
        help="Directory containing predictions.jsonl and its masks/ subtree.",
    )
    parser.add_argument(
        "--wrist-frames-root", type=Path, required=True,
        help="Synchronized 1024px wrist RGB ZIP root.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = export_wrist_predictions(
        args.prediction_root,
        args.wrist_frames_root,
        args.output,
        write_videos=not args.no_videos,
    )
    print(json.dumps({
        "output": str(args.output),
        "rows": manifest["rows"],
        "tasks": len(manifest["tasks"]),
        "camera": manifest["camera"],
        "arrow_coordinate_system": manifest["arrow_coordinate_system"],
    }, indent=2))


if __name__ == "__main__":
    main()
