"""VLM-compatible wrist export with current-mask and local-arrow diagnostics."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from samgraph_core.geometric_graph import render_graph_overlay
from samgraph_benchmark.export import EXPORT_LABELS, benchmark_triplets
from samgraph_benchmark.frames import discover_archives, iter_zip_frames


COLORS = (
    (255, 60, 220), (30, 225, 240), (255, 130, 30), (30, 240, 80),
    (255, 225, 30), (60, 145, 255), (240, 240, 240),
)

# LIBERO's VLM benchmark calls the wrist camera ``eye_in_hand``.  Keep the
# internal predictor name ``wrist`` separate from this on-disk schema value;
# ``wrist`` is the SO101 convention and is not recognized by LIBERO's scorer.
VLM_WRIST_CAMERA = "eye_in_hand"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    tasks: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["demo"]), int(row["frame"]))
        if key in tasks[str(row["task"])]:
            raise ValueError("duplicate wrist prediction row")
        tasks[str(row["task"])][key] = row
    return tasks


def _masks(prediction_root: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = (prediction_root / row["mask_cache"]).resolve()
    if not path.is_relative_to(prediction_root.resolve()):
        raise ValueError("wrist mask cache escapes prediction root")
    if _sha(path) != row["mask_cache_sha256"]:
        raise ValueError("wrist mask cache hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=bool) for key in data.files}


def _mask_overlay(rgb: np.ndarray, masks: dict[str, np.ndarray], objects: list[dict[str, Any]]) -> np.ndarray:
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    for index, state in enumerate(objects):
        key = state.get("wrist_mask_key")
        if not key or key not in masks:
            continue
        mask = masks[key]
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.72 + color * 0.28).astype(np.uint8)
        inside = mask.copy()
        inside[1:] &= mask[:-1]
        inside[:-1] &= mask[1:]
        inside[:, 1:] &= mask[:, :-1]
        inside[:, :-1] &= mask[:, 1:]
        overlay[mask & ~inside] = color.astype(np.uint8)
    return overlay


def _graph(row: dict[str, Any], masks: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[str]]:
    instances = []
    seen = set()
    for state in row["wrist_objects"]:
        target = state.get("agent_output_id")
        key = state.get("wrist_mask_key")
        if not target or target in seen or not key or key not in masks:
            continue
        y, x = np.nonzero(masks[key])
        if not len(x):
            continue
        instances.append({
            "instance_id": target,
            "benchmark_instance_id": EXPORT_LABELS.get(target, target),
            "wrist_track_id": state["wrist_track_id"],
            "center_xy": [float(x.mean()), float(y.mean())],
        })
        seen.add(target)
    triplets = [{"subject": a, "relation": relation, "object": b}
                for a, relation, b in row["triplets"]]
    endpoints = {value for a, _, b in row["triplets"] for value in (a, b)}
    omitted = sorted(endpoints - seen)
    return {
        "task": row["task"], "demo": row["demo"], "frame": row["frame"],
        "camera": VLM_WRIST_CAMERA, "instances": instances, "triplets": triplets,
        "arrow_coordinate_system": "wrist_current_mask_centroids",
        "arrow_omitted_unresolved_endpoints": omitted,
    }, omitted


def export_wrist_predictions(prediction_root: Path, wrist_frames_root: Path,
                             output: Path, *, write_videos: bool = True) -> dict[str, Any]:
    prediction_root = Path(prediction_root).resolve()
    predictions = prediction_root / "predictions.jsonl"
    wrist_frames_root = Path(wrist_frames_root).resolve()
    output = Path(output)
    tasks = _load_rows(predictions)
    row_pairs = {
        (task, demo)
        for task, rows in tasks.items()
        for demo, _ in rows
    }
    episodes = sorted({int(demo.split("_", 1)[1]) for _, demo in row_pairs})
    archives = {
        (path.parent.name, path.stem): path
        for path in discover_archives(wrist_frames_root, episodes=tuple(episodes))
        if path.parent.name in tasks
    }
    if row_pairs != set(archives):
        raise ValueError(
            f"wrist prediction/archive pairs differ: "
            f"missing={sorted(row_pairs - set(archives))} "
            f"extra={sorted(set(archives) - row_pairs)}"
        )
    row_count = sum(map(len, tasks.values()))
    if not tasks or row_count == 0:
        raise ValueError("wrist export requires at least one prediction row")
    camera = output / VLM_WRIST_CAMERA
    for folder in ("csv", "json", "graphs", "arrows", "masks", "videos"):
        (camera / folder).mkdir(parents=True, exist_ok=True)
    columns = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]
    frame_manifest = camera / "frame_manifest.jsonl"
    coverage = {}
    total_omitted = 0
    with frame_manifest.open("x", encoding="utf-8", newline="\n") as manifest_stream:
        for task_index, task in enumerate(sorted(tasks), 1):
            rows = tasks[task]
            stem = task + f"_{VLM_WRIST_CAMERA}_v1"
            graph_task_dir = camera / "graphs" / task
            arrow_task_dir = camera / "arrows" / task
            mask_task_dir = camera / "masks" / task
            graph_task_dir.mkdir()
            arrow_task_dir.mkdir()
            mask_task_dir.mkdir()
            task_coverage = {}
            with (camera / "csv" / f"{stem}.csv").open("x", newline="", encoding="utf-8") as csv_file, \
                 (camera / "json" / f"{stem}.jsonl").open("x", encoding="utf-8", newline="\n") as jsonl:
                writer = csv.DictWriter(csv_file, fieldnames=columns)
                writer.writeheader()
                for demo in sorted({demo for row_task, demo in row_pairs if row_task == task},
                                   key=lambda value: int(value.split("_", 1)[1])):
                    episode_rows = {key: row for key, row in rows.items() if key[0] == demo}
                    graph_dir = graph_task_dir / demo
                    arrow_dir = arrow_task_dir / demo
                    mask_dir = mask_task_dir / demo
                    graph_dir.mkdir()
                    arrow_dir.mkdir()
                    mask_dir.mkdir()
                    video_frames = []
                    emitted = set()
                    unresolved = 0
                    empty = 0
                    archive = archives[(task, demo)]
                    for frame in iter_zip_frames(archive, undo_rotation=False):
                        key = (frame.demo, frame.frame)
                        if key not in episode_rows:
                            continue
                        row = episode_rows[key]
                        if frame.source_sha256 != row["wrist_source_sha256"]:
                            raise ValueError(f"wrist RGB source mismatch: {task}/{demo}/{frame.frame}")
                        source_triplets = row["triplets"]
                        exported = benchmark_triplets(source_triplets)
                        base = {
                            "task": task,
                            "demo": frame.demo,
                            "frame": frame.frame,
                            "camera": VLM_WRIST_CAMERA,
                        }
                        jsonl.write(json.dumps({
                            **base, "response": json.dumps(exported),
                            "source_triplets": source_triplets,
                            "wrist_membership": row["wrist_membership"],
                            "wrist_objects": row["wrist_objects"],
                            "identity_evidence": row["identity_evidence"],
                            "benchmark_label_aliases": EXPORT_LABELS,
                            "wrist_source_sha256": row["wrist_source_sha256"],
                            "agent_source_sha256": row["agent_source_sha256"],
                        }, sort_keys=True) + "\n")
                        for subject, relation, object_ in exported:
                            writer.writerow({**base, "objectA": subject,
                                             "relation": relation, "objectB": object_})
                        masks = _masks(prediction_root, row)
                        mask_overlay = _mask_overlay(frame.rgb, masks, row["wrist_objects"])
                        graph, omitted = _graph(row, masks)
                        arrow_overlay = render_graph_overlay(frame.rgb, graph)
                        Image.fromarray(mask_overlay).save(mask_dir / f"{frame.frame:06d}.png")
                        Image.fromarray(arrow_overlay).save(arrow_dir / f"{frame.frame:06d}.png")
                        (graph_dir / f"{frame.frame:06d}.json").write_text(
                            json.dumps(graph, indent=2), encoding="utf-8")
                        frame_record = {
                            **base,
                            "wrist_source_sha256": row["wrist_source_sha256"],
                            "agent_source_sha256": row["agent_source_sha256"],
                            "membership_count": len(row["wrist_membership"]),
                            "triplet_count": len(row["triplets"]),
                            "empty_graph": not row["triplets"],
                            "identity_decision": row["identity_evidence"]["decision"],
                            "arrow_omitted_unresolved_endpoints": omitted,
                        }
                        manifest_stream.write(json.dumps(frame_record, sort_keys=True) + "\n")
                        empty += int(not row["triplets"])
                        unresolved += int(row["identity_evidence"]["decision"] == "unresolved_correspondence")
                        total_omitted += len(omitted)
                        emitted.add(key)
                        if write_videos:
                            video_frames.append(np.concatenate((mask_overlay, arrow_overlay), axis=1))
                    if emitted != set(episode_rows):
                        raise ValueError(f"wrist export omitted prediction frames: {task}/{demo}")
                    video_name = None
                    if write_videos:
                        import imageio.v2 as imageio
                        video_name = f"{task_index:02d}_{task}_{demo}.mp4"
                        with imageio.get_writer(
                            camera / "videos" / video_name, fps=2, codec="libx264", quality=8,
                            macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"],
                        ) as video:
                            for image in video_frames:
                                video.append_data(image)
                    task_coverage[demo] = {
                        "sampled_frames": len(emitted), "empty_graph_frames": empty,
                        "unresolved_identity_frames": unresolved, "video": video_name,
                    }
            coverage[task] = {
                "sampled_frames": sum(item["sampled_frames"] for item in task_coverage.values()),
                "empty_graph_frames": sum(item["empty_graph_frames"] for item in task_coverage.values()),
                "unresolved_identity_frames": sum(
                    item["unresolved_identity_frames"] for item in task_coverage.values()),
                "episodes": task_coverage,
            }
    manifest = {
        "schema": "samgraph.wrist_vlm_export.v1",
        "camera": VLM_WRIST_CAMERA, "tasks": coverage, "rows": row_count,
        "task_count": len(tasks), "episodes": episodes,
        "jsonl_directory": f"{VLM_WRIST_CAMERA}/json",
        "frame_manifest": f"{VLM_WRIST_CAMERA}/frame_manifest.jsonl",
        "predictions_sha256": _sha(predictions),
        "relations": "exact ordered subset of frozen agent-view triplets",
        "membership": "current wrist RGB masks only",
        "empty_frames_preserved": True,
        "arrow_coordinate_system": "wrist current-mask centroids",
        "arrow_unresolved_endpoint_omissions": total_omitted,
        "evaluation_performed": False,
        "csv_columns": columns,
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
