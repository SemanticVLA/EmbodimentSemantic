"""Agent-view SO101 localization, native tracking, graphing, and export."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from samgraph_core.geometric_graph import render_graph_overlay

from .config import SO101Config
from .dataset import SO101Dataset, SO101Frame, sha256_file
from .render import render_mask_overlay
from .scene import (
    SO101AgentSceneController,
    SO101RolloverSceneTracker,
    install_so101_catalog,
)


CSV_COLUMNS = ["task", "demo", "frame", "camera", "objectA", "relation", "objectB"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json.tmp", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _save_masks(path: Path, masks: Mapping[str, np.ndarray]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **{
            str(key): np.ascontiguousarray(value, dtype=bool) for key, value in masks.items()
        })
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.as_posix(), sha256_file(path)


def _git_identity(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all")),
            "status": git("status", "--short", "--branch").splitlines(),
            "identity_source": "git_checkout",
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Source archives may omit .git. In that case, require the caller to
        # provide the immutable snapshot identity rather than inventing one.
        commit = os.environ.get("SAMGRAPH_SOURCE_COMMIT")
        branch = os.environ.get("SAMGRAPH_SOURCE_BRANCH")
        dirty = os.environ.get("SAMGRAPH_SOURCE_DIRTY")
        if not commit or not branch or dirty not in {"true", "false"}:
            raise RuntimeError(
                "source has no Git checkout and immutable snapshot identity is incomplete"
            )
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty == "true",
            "status": ["identity supplied by hash-verified immutable source snapshot"],
            "identity_source": "immutable_snapshot_manifest",
        }


def _scene_graph(frame: SO101Frame, scene: Mapping[str, Any], config: SO101Config) -> dict[str, Any]:
    return {
        "schema": "samgraph.so101_agent_graph.v1",
        "task": frame.task,
        "demo": f"episode_{frame.episode}",
        "frame": frame.frame_index,
        "camera": "agent_view",
        "trajectory_timestamp_s": frame.trajectory_timestamp_s,
        "video_timestamp_s": frame.video_timestamp_s,
        "relative_video_timestamp_s": frame.relative_video_timestamp_s,
        "timestamp_error_s": frame.timestamp_error_s,
        "source_sha256": frame.source_sha256,
        "instances": scene.get("instances", []),
        "object_states": scene.get("states", []),
        "triplets": [
            {"subject": row[0], "relation": row[1], "object": row[2]}
            for row in scene.get("triplets", [])
        ],
        "observed_triplets": [
            {"subject": row[0], "relation": row[1], "object": row[2]}
            for row in scene.get("observed_triplets", [])
        ],
        "camera_convention": config.camera_convention,
        "geometry_rules": scene.get("geometry_rules"),
        "geometry_rules_sha256": scene.get("geometry_rules_sha256"),
        "relation_revision": scene.get("relation_revision"),
        "tracking_session": scene.get("tracking_session"),
        "persistence_policy": scene.get("persistence_policy"),
        "relation_limitations": {
            "directions": "uncalibrated 2D image-plane labels, not 3D world relations",
            "support": "visual mask-overlap heuristic, not physical-contact evidence",
            "remembered_masks": "past-frame estimates are separately labelled and may be stale",
        },
    }


class SO101AgentPredictor:
    """One loaded SAM runtime shared across every selected episode and object."""

    def __init__(self, checkpoint: Path, config: SO101Config, task_ids: list[str], *,
                 runtime: Any | None = None, tracker: Any | None = None,
                 segmenter: Any | None = None):
        install_so101_catalog(config)
        if runtime is None:
            from samgraph_core import OfficialSam31Runtime
            runtime = OfficialSam31Runtime(checkpoint, text_detection_threshold=0.2)
        if tracker is None:
            tracker = SO101RolloverSceneTracker(runtime, camera_id="agent_view")
        if segmenter is None:
            from samgraph_core import LocalSam31Segmenter
            segmenter = LocalSam31Segmenter(runtime)
        self.runtime = runtime
        self.tracker = tracker
        self.segmenter = segmenter
        self.config = config
        self.controller = SO101AgentSceneController(
            segmenter,
            tracker,
            config=config,
            task_prompts=config.prompts_for_tasks(task_ids),
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "sam": dict(getattr(self.runtime, "model_identity", {})),
            "camera": "agent_view",
            "mode": "automatic_text_localization_native_video_tracking",
            "object_config_sha256": self.config.sha256,
            "manual_points_boxes_or_masks": False,
            "vlm_used": False,
            "native_session_rollover": {
                "enabled": isinstance(self.tracker, SO101RolloverSceneTracker),
                "policy": "previous_rgb_last_nonempty_native_mask",
                "causal": True,
                "preferred_rollover_after_frames": getattr(
                    self.tracker, "rollover_trigger_frames", None),
                "hard_native_session_frame_limit": getattr(
                    self.tracker, "max_native_session_frames", None),
            },
            "persistence_policy": dict(self.config.persistence),
            "initial_missing_object_policy": (
                self.controller.initial_missing_object_policy
            ),
        }

    def warmup(self) -> None:
        method = getattr(self.runtime, "warmup", None)
        if callable(method):
            method()

    def close_episode(self) -> None:
        self.controller.close_episode()

    def close(self) -> None:
        self.controller.close_episode()
        method = getattr(self.runtime, "close", None)
        if callable(method):
            method()

    def start(self, frame: SO101Frame, instruction: str) -> dict[str, Any]:
        return self.controller.start_episode(
            task=f"{frame.task}/episode_{frame.episode}",
            instruction=instruction,
            frame=frame.frame_index,
            rgb=frame.rgb,
        )

    def step(self, frame: SO101Frame) -> dict[str, Any]:
        return self.controller.step(frame=frame.frame_index, rgb=frame.rgb)


class SO101AgentPipeline:
    def __init__(self, dataset: SO101Dataset, config: SO101Config, checkpoint: Path,
                 output: Path, *, output_stride: int = 30, review_fps: float = 2.0,
                 write_videos: bool = True, predictor: SO101AgentPredictor | None = None):
        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        if review_fps <= 0:
            raise ValueError("review_fps must be positive")
        self.dataset = dataset
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.output = Path(output)
        self.output_stride = int(output_stride)
        self.review_fps = float(review_fps)
        self.write_videos = bool(write_videos)
        self.predictor = predictor

    def run(self, task_ids: list[str], episode_ids: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
        if self.output.exists():
            raise FileExistsError(self.output)
        unknown = set(task_ids) - set(self.config.tasks)
        if unknown:
            raise ValueError(f"object configuration lacks tasks: {sorted(unknown)}")
        self.output.mkdir(parents=True)
        camera_root = self.output / "agent_view"
        for name in ("masks", "observed_masks", "mask_overlays", "arrows", "graphs", "videos"):
            (camera_root / name).mkdir(parents=True)
        predictions_partial = camera_root / "predictions.jsonl.partial"
        frame_manifest_partial = camera_root / "frame_manifest.jsonl.partial"
        csv_root = camera_root / "csv"
        csv_root.mkdir()
        predictor = self.predictor or SO101AgentPredictor(
            self.checkpoint, self.config, task_ids,
        )
        own_predictor = self.predictor is None
        coverage: dict[str, Counter] = defaultdict(Counter)
        video_files: dict[str, dict[str, str]] = defaultdict(dict)
        input_videos: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        try:
            predictor.warmup()
            with predictions_partial.open("x", encoding="utf-8", newline="\n") as predictions, \
                 frame_manifest_partial.open("x", encoding="utf-8", newline="\n") as manifest:
                for task in task_ids:
                    csv_path = csv_root / f"{task}_agent_view_v1.csv"
                    with csv_path.open("x", encoding="utf-8", newline="") as csv_stream:
                        writer = csv.DictWriter(csv_stream, fieldnames=CSV_COLUMNS)
                        writer.writeheader()
                        for episode in episode_ids[task]:
                            record = self.dataset.episode(task, episode)
                            episode_name = f"episode_{episode}"
                            relative = Path(task) / episode_name
                            for name in ("masks", "observed_masks", "mask_overlays", "arrows", "graphs"):
                                (camera_root / name / relative).mkdir(parents=True)
                            review_frames = []
                            initial_complete = False
                            initial_unresolved_object_ids: list[str] = []
                            sampled = 0
                            episode_rollovers = 0
                            source_video: Path | None = None
                            try:
                                for frame in self.dataset.iter_frames(task, episode, "agent_view"):
                                    source_video = Path(frame.video_path)
                                    scene = (predictor.start(frame, record.instruction)
                                             if frame.frame_index == 0 else predictor.step(frame))
                                    episode_rollovers = int(
                                        (scene.get("tracking_session") or {}).get(
                                            "native_session_rollover_count", episode_rollovers)
                                    )
                                    if frame.frame_index == 0:
                                        states = scene.get("states", [])
                                        observed = {str(item.get("output_id")) for item in states
                                                    if item.get("status") == "observed"}
                                        initial_complete = observed == set(self.config.object_ids)
                                        initial_unresolved_object_ids = sorted(
                                            str(item.get("output_id")) for item in states
                                            if item.get("status") == "unresolved"
                                        )
                                    if frame.frame_index % self.output_stride:
                                        continue
                                    sampled += 1
                                    base = {
                                        "task": task,
                                        "demo": episode_name,
                                        "frame": frame.frame_index,
                                        "camera": "agent_view",
                                    }
                                    effective = scene.get("effective_masks", {})
                                    observed_masks = scene.get("masks", {})
                                    effective_path = camera_root / "masks" / relative / f"{frame.frame_index:06d}.npz"
                                    observed_path = camera_root / "observed_masks" / relative / f"{frame.frame_index:06d}.npz"
                                    _, effective_sha = _save_masks(effective_path, effective)
                                    _, observed_sha = _save_masks(observed_path, observed_masks)
                                    graph = _scene_graph(frame, scene, self.config)
                                    graph_path = camera_root / "graphs" / relative / f"{frame.frame_index:06d}.json"
                                    _atomic_json(graph_path, graph)
                                    mask_overlay = render_mask_overlay(frame.rgb, effective, scene.get("states", []))
                                    arrow_overlay = render_graph_overlay(frame.rgb, graph)
                                    Image.fromarray(mask_overlay).save(
                                        camera_root / "mask_overlays" / relative / f"{frame.frame_index:06d}.png"
                                    )
                                    Image.fromarray(arrow_overlay).save(
                                        camera_root / "arrows" / relative / f"{frame.frame_index:06d}.png"
                                    )
                                    if self.write_videos:
                                        review_frames.append(np.concatenate((mask_overlay, arrow_overlay), axis=1))
                                    triplets = [list(item) for item in scene.get("triplets", [])]
                                    row = {
                                        "schema": "samgraph.so101_agent_prediction.v1",
                                        **base,
                                        "trajectory_timestamp_s": frame.trajectory_timestamp_s,
                                        "video_timestamp_s": frame.video_timestamp_s,
                                        "relative_video_timestamp_s": frame.relative_video_timestamp_s,
                                        "timestamp_error_s": frame.timestamp_error_s,
                                        "source_sha256": frame.source_sha256,
                                        "triplets": triplets,
                                        "observed_triplets": scene.get("observed_triplets", []),
                                        "object_states": scene.get("states", []),
                                        "instances": scene.get("instances", []),
                                        "acquisition_diagnostics": scene.get("acquisition_diagnostics"),
                                        "mask_cache": effective_path.relative_to(camera_root).as_posix(),
                                        "mask_cache_sha256": effective_sha,
                                        "observed_mask_cache": observed_path.relative_to(camera_root).as_posix(),
                                        "observed_mask_cache_sha256": observed_sha,
                                        "graph": graph_path.relative_to(camera_root).as_posix(),
                                        "tracking_stride": 1,
                                        "output_stride": self.output_stride,
                                        "relation_revision": scene.get("relation_revision"),
                                        "geometry_rules_sha256": scene.get("geometry_rules_sha256"),
                                        "tracking_session": scene.get("tracking_session"),
                                        "persistence_policy": scene.get("persistence_policy"),
                                    }
                                    predictions.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                                    predictions.flush()
                                    unresolved = [item for item in row["object_states"]
                                                  if item.get("status") == "unresolved"]
                                    remembered = [item for item in row["object_states"]
                                                  if item.get("status") == "persisted"]
                                    frame_record = {
                                        **base,
                                        "trajectory_timestamp_s": frame.trajectory_timestamp_s,
                                        "video_timestamp_s": frame.video_timestamp_s,
                                        "timestamp_error_s": frame.timestamp_error_s,
                                        "source_sha256": frame.source_sha256,
                                        "triplet_count": len(triplets),
                                        "empty_prediction": not triplets,
                                        "observed_object_count": len(observed_masks),
                                        "remembered_object_ids": [item["output_id"] for item in remembered],
                                        "unresolved_object_ids": [item["output_id"] for item in unresolved],
                                        "initial_localization_complete": initial_complete,
                                        "initial_unresolved_object_ids": initial_unresolved_object_ids,
                                        "tracking_session": scene.get("tracking_session"),
                                    }
                                    manifest.write(json.dumps(_jsonable(frame_record), sort_keys=True) + "\n")
                                    manifest.flush()
                                    for subject, relation, object_ in triplets:
                                        writer.writerow({**base, "objectA": subject,
                                                         "relation": relation, "objectB": object_})
                                    coverage[task]["sampled_frames"] += 1
                                    coverage[task]["empty_prediction_frames"] += int(not triplets)
                                    coverage[task]["frames_with_remembered_masks"] += int(bool(remembered))
                                    coverage[task]["frames_with_unresolved_objects"] += int(bool(unresolved))
                                if sampled == 0:
                                    raise RuntimeError(f"episode produced no sampled output: {task}/{episode_name}")
                                coverage[task]["episodes"] += 1
                                coverage[task]["episodes_with_incomplete_initial_localization"] += int(
                                    not initial_complete
                                )
                                coverage[task]["native_session_rollovers"] += episode_rollovers
                                if source_video is not None:
                                    input_videos[task][episode_name] = {
                                        "path": str(source_video), "sha256": sha256_file(source_video),
                                    }
                                if self.write_videos:
                                    import imageio.v2 as imageio
                                    video_path = camera_root / "videos" / f"{task}__{episode_name}.mp4"
                                    with imageio.get_writer(
                                        video_path, fps=self.review_fps, codec="libx264", quality=8,
                                        macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"],
                                    ) as video:
                                        for image in review_frames:
                                            video.append_data(image)
                                    video_files[task][episode_name] = video_path.relative_to(self.output).as_posix()
                            finally:
                                predictor.close_episode()
            os.replace(predictions_partial, camera_root / "predictions.jsonl")
            os.replace(frame_manifest_partial, camera_root / "frame_manifest.jsonl")
            repo = Path(__file__).resolve().parents[3]
            result = {
                "schema": "samgraph.so101_agent_run.v1",
                "status": "complete",
                "dataset_inventory": self.dataset.inventory(),
                "selected_tasks": task_ids,
                "selected_episodes": {task: list(episode_ids[task]) for task in task_ids},
                "camera": "agent_view",
                "native_tracking_stride": 1,
                "native_fps": 30,
                "output_stride": self.output_stride,
                "output_fps": 30.0 / self.output_stride,
                "coverage": {task: dict(values) for task, values in sorted(coverage.items())},
                "episodes_with_incomplete_initial_localization": sum(
                    int(values.get("episodes_with_incomplete_initial_localization", 0))
                    for values in coverage.values()
                ),
                "incomplete_initial_localization_is_not_a_processing_failure": True,
                "review_videos": video_files,
                "input_videos": input_videos,
                "object_config": str(self.config.path),
                "object_config_sha256": self.config.sha256,
                "camera_convention": self.config.camera_convention,
                "geometry": self.config.geometry,
                "predictor": predictor.provenance,
                "git": _git_identity(repo),
                "predictions_sha256": sha256_file(camera_root / "predictions.jsonl"),
                "frame_manifest_sha256": sha256_file(camera_root / "frame_manifest.jsonl"),
                "evaluation_performed": False,
                "f1_reported": False,
                "ground_truth_available": False,
                "csv_columns": CSV_COLUMNS,
                "empty_frames_preserved_in_jsonl": True,
            }
            _atomic_json(self.output / "run_manifest.json", result)
            return result
        except Exception as exc:
            _atomic_json(self.output / "failure.json", {
                "schema": "samgraph.so101_agent_failure.v1",
                "error": f"{type(exc).__name__}: {exc}",
                "partial_predictions": predictions_partial.exists(),
                "partial_frame_manifest": frame_manifest_partial.exists(),
                "evaluation_performed": False,
            })
            raise
        finally:
            if own_predictor:
                predictor.close()


def write_inventory(dataset: SO101Dataset, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    value = dataset.inventory()
    _atomic_json(output / "dataset_inventory.json", value)
    return value


__all__ = [
    "CSV_COLUMNS", "SO101AgentPipeline", "SO101AgentPredictor", "write_inventory",
]
