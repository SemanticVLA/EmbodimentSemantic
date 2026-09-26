"""Export frozen native-SAM episode shards without reading evaluation labels.

The default table export needs only prediction JSONL and its run manifest. Optional
mask and arrow exports require the original per-frame mask NPZ and RGB ZIP caches.
No prediction, identity, or spatial relation is recomputed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SamGraph" / "src"))

from samgraph_benchmark.export import benchmark_triplets, EXPORT_LABELS
from samgraph_benchmark.frames import ordered_members, undo_agentview_rotation


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_paths(root: Path, episode: int) -> tuple[Path, Path]:
    base = root / f"full_{episode}" / "SamGraph" / "artifacts" / f"native_{episode}"
    return base / "predictions.jsonl", base / "predictions.manifest.json"


def load_frozen_rows(root: Path, episodes: range):
    rows = defaultdict(dict)
    sources = {}
    source_tasks = None
    source_contract = None
    for episode in episodes:
        prediction, manifest_path = episode_paths(root, episode)
        if not prediction.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"episode {episode} requires prediction and manifest: {prediction}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_names = manifest["input_archive_sha256"]
        expected_tasks = {name.rsplit("/demo_", 1)[0] for name in input_names}
        if any(name != f"{task}/demo_{episode}.zip"
               for name in input_names for task in [name.rsplit("/demo_", 1)[0]]):
            raise ValueError(f"episode {episode} input archive names differ")
        if (manifest.get("episode_selection") != str(episode)
                or manifest.get("camera") != "agentview"
                or manifest.get("mode") != "automatic"
                or manifest.get("frame_stride") != 5
                or len(expected_tasks) != 10):
            raise ValueError(f"episode {episode} has an incompatible manifest")
        if source_tasks is None:
            source_tasks = expected_tasks
        elif expected_tasks != source_tasks:
            raise ValueError(f"episode {episode} task set differs")
        contract = (manifest["source_tree"]["sha256"],
                    manifest["names_config_sha256"], manifest["geometry_rules_sha256"])
        if source_contract is None:
            source_contract = contract
        elif contract != source_contract:
            raise ValueError(f"episode {episode} source/name/geometry contract differs")
        seen = defaultdict(list)
        with prediction.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                task, demo, frame = row["task"], row["demo"], row["frame"]
                if (task not in expected_tasks or Path(task).name != task
                        or "/" in task or "\\" in task or demo != f"demo_{episode}"
                        or type(frame) is not int or frame < 0 or frame % 5
                        or row.get("error") or frame in rows[task].get(demo, {})):
                    raise ValueError(f"invalid or duplicate prediction: {task}/{demo}/{frame}")
                triplets = row.get("triplets")
                if (not isinstance(triplets, list) or any(
                    not isinstance(t, (list, tuple)) or len(t) != 3
                    or not all(isinstance(v, str) for v in t) for t in triplets
                )):
                    raise ValueError(f"invalid triplets: {task}/{demo}/{frame}")
                rows[task].setdefault(demo, {})[frame] = row
                seen[task].append(frame)
        for task in expected_tasks:
            frames = sorted(seen[task])
            if not frames or frames != list(range(0, frames[-1] + 1, 5)):
                raise ValueError(f"episode {episode} has missing sampled frames: {task}")
        sources[str(episode)] = {
            "prediction_sha256": digest(prediction),
            "manifest_sha256": digest(manifest_path),
            "input_archive_sha256": manifest["input_archive_sha256"],
            "source_tree_sha256": manifest["source_tree"]["sha256"],
            "names_config_sha256": manifest["names_config_sha256"],
            "geometry_rules_sha256": manifest["geometry_rules_sha256"],
        }
    return rows, sources


def mask_evidence(prediction: Path, row: dict, destination: Path | None):
    import numpy as np

    raw = Path(row["effective_masks_cache"])
    source = raw if raw.is_absolute() else prediction.parent / raw
    if os.name == "nt" and len(str(source.resolve())) >= 260:
        raise ValueError(f"Windows mask path exceeds MAX_PATH; use a shorter prediction root: {source}")
    if not source.is_file() or digest(source) != row["effective_masks_cache_sha256"]:
        raise ValueError(f"missing or changed mask archive: {source}")
    if destination is not None:
        if os.name == "nt" and len(str(destination.resolve())) >= 260:
            raise ValueError(f"Windows output mask path exceeds MAX_PATH; use a shorter --output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != row["effective_masks_cache_sha256"]:
            raise ValueError(f"copied mask archive changed: {destination}")
    instances = []
    with np.load(source, allow_pickle=False) as data:
        for key in sorted(data.files):
            y, x = np.nonzero(data[key])
            instances.append({"instance_id": EXPORT_LABELS.get(key, key),
                              "source_instance_id": key,
                              "center_xy": [float(x.mean()), float(y.mean())] if len(x) else None})
    return instances


def rgb_evidence(frames_root: Path, task: str, demo: str, frame: int, row: dict):
    import io
    import numpy as np
    from PIL import Image

    path = frames_root / task / f"{demo}.zip"
    with zipfile.ZipFile(path) as archive:
        members = ordered_members(archive)
        if frame >= len(members):
            raise ValueError(f"RGB frame absent: {task}/{demo}/{frame}")
        encoded = archive.read(members[frame])
    if hashlib.sha256(encoded).hexdigest() != row["source_sha256"]:
        raise ValueError(f"RGB source hash differs: {task}/{demo}/{frame}")
    with Image.open(io.BytesIO(encoded)) as image:
        return undo_agentview_rotation(np.asarray(image.convert("RGB"), dtype=np.uint8))


def export_batch(prediction_root: Path, output: Path, *, episodes: range,
                 include_masks: bool = False, render_arrows: bool = False,
                 frames_root: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    if render_arrows and frames_root is None:
        raise ValueError("--render-arrows requires --frames-root")
    rows, sources = load_frozen_rows(prediction_root, episodes)
    if len(rows) != 10:
        raise ValueError("expected ten Spatial tasks")
    output.mkdir(parents=True)
    camera = output / "agentview"
    for part in ("csv", "json", "graphs"):
        (camera / part).mkdir(parents=True, exist_ok=True)
    if include_masks:
        (camera / "masks").mkdir()
    if render_arrows:
        (camera / "arrows").mkdir()
    columns = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]
    counts = {}
    for task in sorted(rows):
        stem = task + "_agentview_v1"
        graph_root = camera / "graphs" / task
        graph_root.mkdir()
        with (camera / "csv" / f"{stem}.csv").open("x", newline="", encoding="utf-8") as csv_file, \
             (camera / "json" / f"{stem}.jsonl").open("x", encoding="utf-8") as jsonl:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            count = 0
            for episode in episodes:
                demo = f"demo_{episode}"
                frames = rows[task].get(demo)
                if frames is None:
                    raise ValueError(f"missing task/episode {task}/{demo}")
                graph_dir = graph_root / demo
                graph_dir.mkdir()
                for frame, row in sorted(frames.items()):
                    base = {"task": task, "demo": demo, "frame": frame, "camera": "agentview"}
                    triplets = benchmark_triplets(row["triplets"])
                    jsonl.write(json.dumps({**base, "response": json.dumps(triplets),
                        "source_triplets": row["triplets"],
                        "object_states": row.get("object_states", []),
                        "benchmark_label_aliases": EXPORT_LABELS,
                        "source_sha256": row["source_sha256"]}) + "\n")
                    for a, relation, b in triplets:
                        writer.writerow({**base, "objectA": a, "relation": relation, "objectB": b})
                    prediction, _ = episode_paths(prediction_root, episode)
                    mask_out = (camera / "masks" / task / demo / f"{frame:06d}.npz"
                                if include_masks else None)
                    instances = (mask_evidence(prediction, row, mask_out)
                                 if include_masks or render_arrows else [])
                    graph = {**base, "triplets": [
                        {"subject": a, "relation": relation, "object": b}
                        for a, relation, b in triplets],
                        "source_triplets": row["triplets"],
                        "object_states": row.get("object_states", []),
                        "instances": instances,
                        "mask_geometry_included": bool(include_masks or render_arrows),
                        "mask_archive": str(mask_out.relative_to(output)).replace("\\", "/") if mask_out else None,
                        "source_sha256": row["source_sha256"]}
                    (graph_dir / f"{frame:06d}.json").write_text(
                        json.dumps(graph, indent=2), encoding="utf-8")
                    if render_arrows:
                        from samgraph_core.geometric_graph import render_graph_overlay
                        from PIL import Image

                        rgb = rgb_evidence(frames_root, task, demo, frame, row)
                        arrow_dir = camera / "arrows" / task / demo
                        arrow_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(render_graph_overlay(rgb, graph)).save(
                            arrow_dir / f"{frame:06d}.png")
                    count += 1
            counts[task] = count
    manifest = {"schema": "samgraph.batch_vlm_export.v1", "camera": "agentview",
                "episodes": list(episodes), "tasks": counts, "frames": sum(counts.values()),
                "csv_columns": columns, "masks_included": include_masks,
                "arrows_rendered": render_arrows, "evaluation_performed": False,
                "prediction_sources": sources,
                "exporter_sha256": digest(Path(__file__)),
                "identity_policy": "causal IDs preserved; benchmark bowl aliases in exported triplets",
                "scope": "frozen predictions only; no relations or masks regenerated"}
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", default="0:50", help="half-open range, e.g. 0:50")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--render-arrows", action="store_true")
    parser.add_argument("--frames-root", type=Path)
    args = parser.parse_args()
    try:
        start, stop = map(int, args.episodes.split(":"))
        if not 0 <= start < stop <= 50:
            raise ValueError
    except ValueError:
        parser.error("--episodes must be a half-open range within 0:50")
    result = export_batch(args.prediction_root, args.output, episodes=range(start, stop),
                          include_masks=args.include_masks,
                          render_arrows=args.render_arrows, frames_root=args.frames_root)
    print(json.dumps({"output": str(args.output), "frames": result["frames"],
                      "tasks": len(result["tasks"]), "episodes": len(result["episodes"])}, indent=2))


if __name__ == "__main__":
    main()
