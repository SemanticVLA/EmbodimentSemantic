"""Prediction-only VLM benchmark export. Never reads evaluation labels."""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .frames import discover_archives, iter_zip_frames, ordered_members
from samgraph_core.geometric_graph import render_graph_overlay

# Static vocabulary translation only. Provisional track IDs deliberately remain
# provisional; this map cannot choose or swap the identities of identical bowls.
BENCHMARK_LABELS = {
    "black_bowl_1": "akita_black_bowl_1",
    "black_bowl_2": "akita_black_bowl_2",
    "bowl_1": "akita_black_bowl_1",
    "bowl_2": "akita_black_bowl_2",
    "white_ramekin_1": "glazed_rim_porcelain_ramekin_1",
}

# Arbitrary numbering for the benchmark's permutation-invariant bowl comparison.
# These are output vocabulary aliases, NOT assertions about manipulated identity.
BENCHMARK_TRACK_LABELS = {
    "black_bowl_track_1": "akita_black_bowl_1",
    "black_bowl_track_2": "akita_black_bowl_2",
}
EXPORT_LABELS = {**BENCHMARK_LABELS, **BENCHMARK_TRACK_LABELS}


def benchmark_triplets(triplets):
    endpoints = {x for a, _, b in triplets for x in (a, b)}
    canonical = [EXPORT_LABELS.get(x, x) for x in endpoints]
    if len(set(canonical)) != len(canonical):
        raise ValueError("distinct output identities collide under benchmark aliases")
    return [(EXPORT_LABELS.get(a, a), relation, EXPORT_LABELS.get(b, b))
            for a, relation, b in triplets]


def export_predictions(predictions: Path, frames_root: Path, output: Path, *,
                       allow_partial: bool = False, episodes=(0,)) -> dict:
    predictions = Path(predictions).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tasks = defaultdict(dict)
    for line in predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        task, frame = row["task"], (row["demo"], row["frame"])
        if Path(task).name != task or "/" in task or "\\" in task:
            raise ValueError("invalid task identifier")
        if frame in tasks[task]:
            raise ValueError("duplicate frame")
        if row.get("error"):
            raise ValueError(f"cannot export errored frame: {task}/{frame}")
        tasks[task][frame] = row
    if not allow_partial and len(tasks) != 10:
        raise ValueError("final Spatial export requires 10 tasks; use explicit partial export for diagnostics")
    # Validate exact frame coverage from RGB archive membership, never labels.
    # A total of 258 alone can hide a missing task/frame replaced by another.
    expected = defaultdict(set)
    archives = discover_archives(frames_root, episodes=episodes)
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            expected[archive_path.parent.name].update(
                (archive_path.stem, frame) for frame in range(0, len(ordered_members(archive)), 5))
    if not allow_partial and set(tasks) != set(expected):
        raise ValueError("prediction task set differs from RGB archives")
    for task, rows in tasks.items():
        if task not in expected or not set(rows).issubset(expected[task]):
            raise ValueError(f"unexpected sampled frames: {task}")
        if not allow_partial and set(rows) != expected[task]:
            raise ValueError(f"incomplete sampled frames: {task}")
    if os.name == "nt":
        camera = output / "agentview"
        paths = [output / "export_manifest.json"]
        for task, rows in tasks.items():
            stem = task + "_agentview_v1"
            paths.extend((camera / "csv" / f"{stem}.csv",
                          camera / "json" / f"{stem}.jsonl"))
            for demo, frame in rows:
                graph_dir = camera / "graphs" / task
                arrow_dir = camera / "arrows" / task
                if demo != "demo_0":
                    graph_dir = graph_dir / demo
                    arrow_dir = arrow_dir / demo
                paths.extend((graph_dir / f"{frame:06d}.json",
                              arrow_dir / f"{frame:06d}.png"))
        longest = max((path.resolve() for path in paths), key=lambda path: len(str(path)))
        if len(str(longest)) >= 260:
            raise ValueError(
                f"Windows output path is {len(str(longest))} characters; "
                "use a shorter --run-id or --output-root to stay below MAX_PATH"
            )
    output.mkdir(parents=True)
    camera = output / "agentview"
    for folder in ("csv", "json", "graphs", "arrows"):
        (camera / folder).mkdir(parents=True)
    counts = {}
    columns = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]
    for task, rows in sorted(tasks.items()):
        stem = task + "_agentview_v1"
        graph_dir, arrow_dir = camera / "graphs" / task, camera / "arrows" / task
        graph_dir.mkdir(); arrow_dir.mkdir()
        emitted = set()
        with (camera / "csv" / (stem + ".csv")).open("x", newline="", encoding="utf-8") as csv_file, \
             (camera / "json" / (stem + ".jsonl")).open("x", encoding="utf-8") as jsonl:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            task_frames = (frame for archive in archives if archive.parent.name == task
                           for frame in iter_zip_frames(archive))
            for frame in task_frames:
                key = (frame.demo, frame.frame)
                if key not in rows:
                    continue
                row = rows[key]
                if frame.source_sha256 != row["source_sha256"]:
                    raise ValueError("RGB source identity mismatch")
                base = {"task": task, "demo": row["demo"], "frame": frame.frame, "camera": "agentview"}
                triplets = row.get("triplets", [])
                if any(len(t) != 3 or not all(isinstance(x, str) for x in t) for t in triplets):
                    raise ValueError("malformed ordered triplet")
                source_triplets = triplets
                triplets = benchmark_triplets(triplets)
                # Empty responses are retained so the evaluator sees empty frames.
                jsonl.write(json.dumps({**base, "response": json.dumps(triplets),
                    "object_states": row.get("object_states", []),
                    "source_triplets": source_triplets,
                    "benchmark_label_aliases": EXPORT_LABELS,
                    "source_sha256": row["source_sha256"]}) + "\n")
                for subject, relation, object_ in triplets:
                    writer.writerow({**base, "objectA": subject, "relation": relation, "objectB": object_})
                mask_path = Path(row["effective_masks_cache"])
                if not mask_path.is_absolute():
                    mask_path = predictions.parent / mask_path
                if hashlib.sha256(mask_path.read_bytes()).hexdigest() != row["effective_masks_cache_sha256"]:
                    raise ValueError("mask archive identity mismatch")
                instances = []
                with np.load(mask_path, allow_pickle=False) as data:
                    for mask_key in data.files:
                        y, x = np.nonzero(data[mask_key])
                        instances.append({"instance_id": EXPORT_LABELS.get(mask_key, mask_key),
                            "source_instance_id": mask_key,
                            "center_xy": [float(x.mean()), float(y.mean())] if len(x) else None})
                graph = {**base, "instances": instances, "object_states": row.get("object_states", []),
                         "benchmark_label_aliases": EXPORT_LABELS,
                         "triplets": [{"subject": a, "relation": r, "object": b} for a, r, b in triplets]}
                episode_graph_dir = graph_dir if frame.demo == "demo_0" else graph_dir / frame.demo
                episode_arrow_dir = arrow_dir if frame.demo == "demo_0" else arrow_dir / frame.demo
                episode_graph_dir.mkdir(exist_ok=True)
                episode_arrow_dir.mkdir(exist_ok=True)
                (episode_graph_dir / f"{frame.frame:06d}.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
                Image.fromarray(render_graph_overlay(frame.rgb, graph)).save(episode_arrow_dir / f"{frame.frame:06d}.png")
                emitted.add(key)
        if emitted != rows.keys():
            raise ValueError("prediction frames absent from RGB archive")
        counts[task] = len(emitted)
    manifest = {"schema": "samgraph.vlm_export.v3", "tasks": counts,
                "static_label_aliases": BENCHMARK_LABELS,
                "provisional_benchmark_aliases": BENCHMARK_TRACK_LABELS,
                "jsonl_directory": "agentview/json",
                "episodes": "all" if episodes is None else list(episodes),
                "partial_export_allowed": allow_partial,
                "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "evaluation_performed": False,
                "identity_policy": "causal source IDs preserved in metadata; benchmark-only bowl aliases in exported triplets",
                "csv_columns": columns}
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
