from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


EPISODE = "episode_0"
ARTIFACT_FILES = (
    "bboxes/agent_view.jsonl",
    "bboxes/wrist.jsonl",
    "index/episodes.jsonl",
    "index/sampled_frames.jsonl",
    "metadata/frame_signals.jsonl",
    "proxy_graphs/gt/agent_view.jsonl",
    "proxy_graphs/gt/wrist.jsonl",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")


def build(dataset_root: Path, artifacts_root: Path, output_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    artifacts_root = artifacts_root.resolve()
    output_root = output_root.resolve()
    if not dataset_root.is_dir() or not artifacts_root.is_dir():
        raise FileNotFoundError("SO101 dataset and artifact directories must exist")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    output_dataset = output_root / "dataset"
    output_artifacts = output_root / "artifacts"
    output_predictions = output_root / "predictions"

    sampled_source = artifacts_root / "index/sampled_frames.jsonl"
    sampled_records = []
    for record in read_jsonl(sampled_source):
        if record.get("episode") != EPISODE:
            continue
        source_image = Path(str(record["image_path"])).resolve()
        relative_image = (
            Path(str(record["task"]))
            / "frames"
            / str(record["camera"])
            / EPISODE
            / source_image.name
        )
        destination = output_dataset / relative_image
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination)
        record = dict(record)
        record["image_path"] = relative_image.as_posix()
        sampled_records.append(record)
    write_jsonl(output_artifacts / "index/sampled_frames.jsonl", sampled_records)

    for relative in ARTIFACT_FILES:
        if relative == "index/sampled_frames.jsonl":
            continue
        source = artifacts_root / relative
        records = []
        for record in read_jsonl(source):
            if record.get("episode") != EPISODE:
                continue
            record = dict(record)
            if relative == "index/episodes.jsonl":
                record["videos"] = {
                    camera: {**value, "exists": False, "path": ""}
                    for camera, value in record.get("videos", {}).items()
                }
            records.append(record)
        write_jsonl(output_artifacts / relative, records)

    prediction_root = dataset_root / "gemini-generated graphs"
    if prediction_root.is_dir():
        for source in prediction_root.glob("*/json/*.jsonl"):
            records = [record for record in read_jsonl(source) if record.get("demo") == EPISODE]
            write_jsonl(output_predictions / source.relative_to(prediction_root), records)

    config = """schema_version: so101-proxy-v1
paths:
  so101_dataset: dataset
  libero_dataset: ../libero_demo_cache
  gemini_predictions: predictions
  artifacts: artifacts
demo:
  host: 0.0.0.0
  port: 7860
  default_camera: agent_view
  default_mode: gt
  default_subject: black_bowl
"""
    (output_root / "config.yaml").write_text(config, encoding="ascii")
    total = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    print(
        f"Wrote {len(sampled_records)} SO101 episode_0 frames and reduced artifacts "
        f"({total / 1024 / 1024:.2f} MiB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache SO101 episode_0 source frames, artifacts, and predictions."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.dataset, args.artifacts, args.output)


if __name__ == "__main__":
    main()
