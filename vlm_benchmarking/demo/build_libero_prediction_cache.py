from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vlm_bench.io_utils import parse_triplets


DEMO_ID = "demo_0"
CAMERAS = {"agentview", "eye_in_hand"}


def build(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Prediction output directory not found: {source_root}")
    if output_root.exists():
        shutil.rmtree(output_root)

    run_count = 0
    record_count = 0
    for source in sorted(source_root.glob("*/*/json/*.jsonl")):
        relative = source.relative_to(source_root)
        camera = relative.parts[1]
        if camera not in CAMERAS:
            continue
        destination = output_root / relative
        wrote_run = False
        lines = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("demo") != DEMO_ID:
                    continue
                triplets = list(dict.fromkeys(parse_triplets(str(record.get("response", "")))))
                compact = {
                    "task": record.get("task", ""),
                    "demo": DEMO_ID,
                    "frame": record.get("frame", 0),
                    "camera": record.get("camera", camera),
                    "model": record.get("model", relative.parts[0]),
                    "response": "\n".join(",".join(item) for item in triplets),
                }
                lines.append(json.dumps(compact, ensure_ascii=True, separators=(",", ":")))
                record_count += 1
                wrote_run = True
        if wrote_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n".join(lines) + "\n", encoding="ascii")
            run_count += 1
    total = sum(path.stat().st_size for path in output_root.rglob("*.jsonl"))
    print(
        f"Wrote {record_count} demo_0 prediction records across {run_count} runs "
        f"({total / 1024 / 1024:.2f} MiB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact LIBERO demo_0 predictions for the deployed demo."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
