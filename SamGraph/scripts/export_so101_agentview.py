"""Build a provenance-checked SO101 agent-view release from immutable run shards.

This is an exporter, not a second perception pipeline. It never changes masks,
identities, or relations. Optional masks are hash-verified copies. Optional
arrow overlays are regenerated from the frozen graph and the original RGB.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "SamGraph" / "src"))

from samgraph_so101.dataset import SO101Dataset, sha256_file


CSV_COLUMNS = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_root(path: Path) -> Path:
    candidates = (path, path / "payload" / "output" / "run", path / "output" / "run")
    for candidate in candidates:
        if (candidate / "agent_view").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"SO101 run root not found below: {path}")


def _source_path(camera: Path, row: Mapping[str, Any], field: str) -> Path:
    raw = row.get(field)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"prediction row lacks {field}")
    path = Path(raw)
    return path if path.is_absolute() else camera / path


def _triplets_from_graph(graph: Mapping[str, Any]) -> list[list[str]]:
    result = []
    for item in graph.get("triplets", []):
        if not isinstance(item, dict):
            raise ValueError("graph triplet must be an object")
        result.append([item.get("subject"), item.get("relation"), item.get("object")])
    return result


class Shard:
    def __init__(self, source_id: str, path: Path):
        self.source_id = source_id
        self.root = _run_root(path)
        self.camera = self.root / "agent_view"
        self.predictions_path = next(
            (path for path in (
                self.camera / "predictions.jsonl",
                self.camera / "predictions.jsonl.partial",
            ) if path.is_file()), None,
        )
        self.frame_manifest_path = next(
            (path for path in (
                self.camera / "frame_manifest.jsonl",
                self.camera / "frame_manifest.jsonl.partial",
            ) if path.is_file()), None,
        )
        if self.predictions_path is None or self.frame_manifest_path is None:
            raise FileNotFoundError(f"prediction/frame manifest absent in shard {source_id}")
        self.rows = self._index(_jsonl(self.predictions_path), "prediction")
        self.frames = self._index(_jsonl(self.frame_manifest_path), "frame manifest")
        self.run_manifest = self.root / "run_manifest.json"
        self.failure = self.root / "failure.json"

    @staticmethod
    def _key(row: Mapping[str, Any]) -> tuple[str, int, int]:
        task = row.get("task")
        demo = row.get("demo")
        frame = row.get("frame")
        if (not isinstance(task, str) or not isinstance(demo, str)
                or not demo.startswith("episode_") or not demo[8:].isdigit()
                or type(frame) is not int or frame < 0):
            raise ValueError(f"invalid SO101 row key: {task}/{demo}/{frame}")
        if row.get("camera") != "agent_view":
            raise ValueError(f"unexpected camera in {task}/{demo}/{frame}")
        return task, int(demo[8:]), frame

    def _index(self, rows: Iterable[dict[str, Any]], label: str) -> dict[tuple[str, int, int], dict[str, Any]]:
        result = {}
        for row in rows:
            key = self._key(row)
            if key in result:
                raise ValueError(f"duplicate {label} row in {self.source_id}: {key}")
            result[key] = row
        return result

    def provenance(self) -> dict[str, Any]:
        result = {
            "source_id": self.source_id,
            "predictions_sha256": sha256_file(self.predictions_path),
            "frame_manifest_sha256": sha256_file(self.frame_manifest_path),
            "prediction_file_complete": self.predictions_path.name == "predictions.jsonl",
        }
        if self.run_manifest.is_file():
            result["run_manifest_sha256"] = sha256_file(self.run_manifest)
            manifest = _json(self.run_manifest)
            result["object_config_sha256"] = manifest.get("object_config_sha256")
            result["git"] = manifest.get("git")
            result["predictor"] = manifest.get("predictor")
        if self.failure.is_file():
            result["failure_sha256"] = sha256_file(self.failure)
        status = self.root / "status.json"
        if status.is_file():
            result["status_sha256"] = sha256_file(status)
            result["status"] = _json(status)
        identity_root = self.root / "_identity"
        if identity_root.is_dir():
            result["identity_files"] = {
                path.relative_to(identity_root).as_posix(): sha256_file(path)
                for path in sorted(identity_root.rglob("*.json"))
            }
        return result


def _selection(path: Path) -> tuple[int, list[dict[str, Any]]]:
    value = _json(path)
    if value.get("schema") != "samgraph.so101_release_selection.v1":
        raise ValueError("unsupported SO101 release selection schema")
    if value.get("camera") != "agent_view":
        raise ValueError("release selection must use agent_view")
    stride = value.get("output_stride")
    if type(stride) is not int or stride < 1:
        raise ValueError("release output_stride must be positive")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("release selection episodes must be non-empty")
    keys = []
    for item in episodes:
        if (not isinstance(item, dict) or not isinstance(item.get("task"), str)
                or type(item.get("episode")) is not int
                or not isinstance(item.get("source"), str)):
            raise ValueError(f"invalid release selection entry: {item}")
        keys.append((item["task"], item["episode"]))
    if len(keys) != len(set(keys)):
        raise ValueError("release selection contains duplicate task/episode entries")
    return stride, episodes


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError(f"missing or changed source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError(f"copied artifact hash differs: {destination}")


def export_release(*, dataset: SO101Dataset, selection_path: Path,
                   sources: Mapping[str, Path], output: Path,
                   include_masks: bool = False,
                   include_observed_masks: bool = False,
                   render_arrows: bool = False,
                   include_videos: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    stride, selected = _selection(selection_path)
    shards = {source_id: Shard(source_id, path) for source_id, path in sources.items()}
    missing_sources = sorted({item["source"] for item in selected} - set(shards))
    if missing_sources:
        raise ValueError(f"selection references missing shard IDs: {missing_sources}")

    output.mkdir(parents=True)
    # Match the repository's published VLM-result layout. The semantic camera
    # value inside every row remains the dataset's canonical ``agent_view``.
    camera_out = output / "agentview"
    for name in ("csv", "json", "graphs"):
        (camera_out / name).mkdir(parents=True)
    if include_masks:
        (camera_out / "masks").mkdir()
    if include_observed_masks:
        (camera_out / "observed_masks").mkdir()
    if render_arrows:
        (camera_out / "arrows").mkdir()
    if include_videos:
        (camera_out / "videos").mkdir()

    selected_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_task[item["task"]].append(item)
    all_predictions = camera_out / "predictions.jsonl"
    all_frames = camera_out / "frame_manifest.jsonl"
    coverage: dict[str, Counter] = defaultdict(Counter)
    selected_sources: list[dict[str, Any]] = []
    with all_predictions.open("x", encoding="utf-8", newline="\n") as prediction_stream, \
         all_frames.open("x", encoding="utf-8", newline="\n") as frame_stream:
        for task in sorted(selected_by_task):
            task_json = camera_out / "json" / f"{task}_agent_view_v1.jsonl"
            task_csv = camera_out / "csv" / f"{task}_agent_view_v1.csv"
            with task_json.open("x", encoding="utf-8", newline="\n") as json_stream, \
                 task_csv.open("x", encoding="utf-8", newline="") as csv_stream:
                writer = csv.DictWriter(csv_stream, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for item in sorted(selected_by_task[task], key=lambda row: row["episode"]):
                    episode = item["episode"]
                    source_id = item["source"]
                    shard = shards[source_id]
                    record = dataset.episode(task, episode)
                    expected_frames = list(range(0, record.length, stride))
                    keys = [(task, episode, frame) for frame in expected_frames]
                    missing_predictions = [key[2] for key in keys if key not in shard.rows]
                    missing_frames = [key[2] for key in keys if key not in shard.frames]
                    if missing_predictions or missing_frames:
                        raise ValueError(
                            f"incomplete selected episode {task}/episode_{episode} from {source_id}: "
                            f"prediction={missing_predictions}, manifest={missing_frames}"
                        )
                    selected_sources.append({
                        "task": task, "episode": episode, "source": source_id,
                        "sampled_frames": len(expected_frames),
                    })
                    decoded = None
                    if render_arrows:
                        decoded = {
                            frame.frame_index: frame
                            for frame in dataset.iter_frames(task, episode, "agent_view")
                            if frame.frame_index % stride == 0
                        }
                        if sorted(decoded) != expected_frames:
                            raise ValueError(f"RGB frame coverage differs: {task}/episode_{episode}")
                    for key in keys:
                        row = dict(shard.rows[key])
                        frame_row = dict(shard.frames[key])
                        if row.get("schema") != "samgraph.so101_agent_prediction.v1":
                            raise ValueError(f"prediction schema differs: {key}")
                        if row.get("tracking_stride") != 1 or row.get("output_stride") != stride:
                            raise ValueError(f"tracking/output cadence differs: {key}")
                        triplets = row.get("triplets")
                        if (not isinstance(triplets, list) or any(
                            not isinstance(value, list) or len(value) != 3
                            or any(not isinstance(part, str) for part in value)
                            for value in triplets
                        )):
                            raise ValueError(f"invalid triplets: {key}")
                        if (frame_row.get("source_sha256") != row.get("source_sha256")
                                or frame_row.get("triplet_count") != len(triplets)
                                or bool(frame_row.get("empty_prediction")) != (not triplets)):
                            raise ValueError(f"prediction/frame manifest mismatch: {key}")
                        graph_source = _source_path(shard.camera, row, "graph")
                        graph = _json(graph_source)
                        if (_triplets_from_graph(graph) != triplets
                                or graph.get("source_sha256") != row.get("source_sha256")
                                or (graph.get("task"), graph.get("demo"), graph.get("frame"))
                                != (task, f"episode_{episode}", key[2])):
                            raise ValueError(f"graph/prediction mismatch: {key}")
                        graph_destination = (
                            camera_out / "graphs" / task / f"episode_{episode}"
                            / f"{key[2]:06d}.json"
                        )
                        graph_destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(graph_source, graph_destination)

                        if include_masks:
                            mask_source = _source_path(shard.camera, row, "mask_cache")
                            mask_destination = (
                                camera_out / "masks" / task / f"episode_{episode}"
                                / f"{key[2]:06d}.npz"
                            )
                            _copy_verified(mask_source, mask_destination, row["mask_cache_sha256"])
                        if include_observed_masks:
                            observed_source = _source_path(shard.camera, row, "observed_mask_cache")
                            observed_destination = (
                                camera_out / "observed_masks" / task / f"episode_{episode}"
                                / f"{key[2]:06d}.npz"
                            )
                            _copy_verified(
                                observed_source, observed_destination,
                                row["observed_mask_cache_sha256"],
                            )
                        if render_arrows:
                            from samgraph_core.geometric_graph import render_graph_overlay
                            from PIL import Image

                            frame = decoded[key[2]]
                            if frame.source_sha256 != row.get("source_sha256"):
                                raise ValueError(f"RGB/prediction hash mismatch: {key}")
                            arrow = render_graph_overlay(frame.rgb, graph)
                            arrow_path = (
                                camera_out / "arrows" / task / f"episode_{episode}"
                                / f"{key[2]:06d}.png"
                            )
                            arrow_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(arrow).save(arrow_path)

                        row["release_source"] = source_id
                        row["graph"] = graph_destination.relative_to(camera_out).as_posix()
                        if include_masks:
                            row["mask_cache"] = (
                                camera_out / "masks" / task / f"episode_{episode}"
                                / f"{key[2]:06d}.npz"
                            ).relative_to(camera_out).as_posix()
                        if include_observed_masks:
                            row["observed_mask_cache"] = (
                                camera_out / "observed_masks" / task / f"episode_{episode}"
                                / f"{key[2]:06d}.npz"
                            ).relative_to(camera_out).as_posix()
                        encoded_row = json.dumps(row, sort_keys=True)
                        prediction_stream.write(encoded_row + "\n")
                        json_stream.write(encoded_row + "\n")
                        frame_row["release_source"] = source_id
                        frame_stream.write(json.dumps(frame_row, sort_keys=True) + "\n")
                        for object_a, relation, object_b in triplets:
                            writer.writerow({
                                "task": task, "demo": f"episode_{episode}",
                                "frame": key[2], "camera": "agent_view",
                                "objectA": object_a, "relation": relation,
                                "objectB": object_b,
                            })
                        coverage[task]["sampled_frames"] += 1
                        coverage[task]["empty_prediction_frames"] += int(not triplets)
                        coverage[task]["frames_with_unresolved_objects"] += int(any(
                            state.get("status") == "unresolved"
                            for state in row.get("object_states", [])
                        ))
                    coverage[task]["episodes"] += 1
                    if include_videos:
                        video = (
                            shard.camera / "videos"
                            / f"{task}__episode_{episode}.mp4"
                        )
                        if not video.is_file():
                            raise FileNotFoundError(f"review video absent: {video}")
                        shutil.copyfile(
                            video,
                            camera_out / "videos" / f"{task}__episode_{episode}.mp4",
                        )

    source_ids = sorted({item["source"] for item in selected})
    manifest = {
        "schema": "samgraph.so101_agent_release.v1",
        "status": "complete",
        "camera": "agent_view",
        "native_tracking_stride": 1,
        "output_stride": stride,
        "tasks": sorted(selected_by_task),
        "episodes": selected_sources,
        "coverage": {task: dict(counter) for task, counter in sorted(coverage.items())},
        "episode_count": len(selected),
        "sampled_frame_count": sum(row["sampled_frames"] for row in selected_sources),
        "csv_columns": CSV_COLUMNS,
        "empty_frames_preserved_in_jsonl": True,
        "masks_included": include_masks,
        "observed_masks_included": include_observed_masks,
        "arrows_regenerated": render_arrows,
        "videos_included": include_videos,
        "selection_sha256": sha256_file(selection_path),
        "predictions_sha256": sha256_file(all_predictions),
        "frame_manifest_sha256": sha256_file(all_frames),
        "sources": {source_id: shards[source_id].provenance() for source_id in source_ids},
        "exporter_sha256": sha256_file(Path(__file__)),
        "relations_recomputed": False,
        "masks_recomputed": False,
        "evaluation_performed": False,
        "f1_reported": False,
        "ground_truth_available": False,
    }
    _write_json(output / "export_manifest.json", manifest)
    return manifest


def _sources(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source must be SOURCE_ID=PATH")
        source_id, raw_path = value.split("=", 1)
        if not source_id or source_id in result:
            raise ValueError(f"invalid or duplicate source ID: {source_id}")
        result[source_id] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="repeat SOURCE_ID=PATH for every immutable run shard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--include-observed-masks", action="store_true")
    parser.add_argument("--render-arrows", action="store_true")
    parser.add_argument("--include-videos", action="store_true")
    args = parser.parse_args(argv)
    result = export_release(
        dataset=SO101Dataset(args.dataset_root),
        selection_path=args.selection,
        sources=_sources(args.source),
        output=args.output,
        include_masks=args.include_masks,
        include_observed_masks=args.include_observed_masks,
        render_arrows=args.render_arrows,
        include_videos=args.include_videos,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "episodes": result["episode_count"],
        "sampled_frames": result["sampled_frame_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
