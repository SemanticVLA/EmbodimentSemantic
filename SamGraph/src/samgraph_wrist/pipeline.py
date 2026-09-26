"""End-to-end RGB-only wrist membership prediction over frozen agent graphs."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from samgraph_core.geometry_profiles import samgraph_spatial_mask_geometry
from samgraph_benchmark.frames import discover_archives, iter_zip_frames

from .identity import CrossViewBowlResolver, filter_agent_triplets
from .wrist_scene import WristSceneController


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task"]), str(row.get("demo", "demo_0")), int(row["frame"]))
        if key in result:
            raise ValueError(f"duplicate frozen agent row: {key}")
        if row.get("error"):
            raise ValueError(f"errored frozen agent row: {key}")
        result[key] = row
    return result


def _load_masks(predictions: Path, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    relative = row.get("mask_cache") or row.get("effective_masks_cache")
    expected = row.get("mask_cache_sha256") or row.get("effective_masks_cache_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError("frozen agent row lacks a hash-checked mask archive")
    path = Path(relative)
    if not path.is_absolute():
        path = predictions.parent / path
    path = path.resolve()
    if _sha(path) != expected:
        raise ValueError(f"frozen agent mask hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.ascontiguousarray(data[key], dtype=bool) for key in data.files}


def _agent_outputs(states: list[Mapping[str, Any]]) -> dict[str, str]:
    return {str(state["track_id"]): str(state.get("output_id") or state["track_id"])
            for state in states}


def _unique_agent_output(states: list[Mapping[str, Any]], class_id: str) -> str | None:
    values = [str(state.get("output_id") or state["track_id"]) for state in states
              if state.get("class_id") == class_id]
    return values[0] if len(values) == 1 else None


def _checkpoint_class(track_id: str) -> str:
    if track_id.startswith("black_bowl_track_"):
        return "black_bowl"
    head, separator, tail = track_id.rpartition("_")
    return head if separator and tail.isdigit() else track_id


def _checkpoint_scene(path: Path, known_tracks: dict[str, str]) -> dict[str, Any]:
    """Recreate the observable scene contract from a persisted sampled mask.

    The checkpoint contains predictor outputs only. Absent previously seen tracks
    are explicitly out of view; they are never promoted to membership.
    """
    with np.load(path, allow_pickle=False) as data:
        masks = {key: np.ascontiguousarray(data[key], dtype=bool) for key in data.files}
    for track in masks:
        known_tracks.setdefault(track, _checkpoint_class(track))
    states = [{
        "track_id": track,
        "output_id": track,
        "class_id": class_id,
        "status": "observed" if track in masks else "out_of_view",
        "observation_state": "current_observation" if track in masks else "out_of_view",
        "rejection": None if track in masks else "absent_from_checkpoint_mask",
    } for track, class_id in sorted(known_tracks.items())]
    return {
        "masks": masks,
        "states": states,
        "acquisition_diagnostics": {
            "source": "hash_recorded_mask_checkpoint",
            "checkpoint_path": str(path),
        },
    }


class WristExtensionRunner:
    """Run wrist SAM while treating agent predictions as immutable relations."""

    def __init__(self, checkpoint: Path, names: Mapping[str, Any], *, runtime: Any | None = None,
                 tracker: Any | None = None, segmenter: Any | None = None):
        if runtime is None:
            from samgraph_core import OfficialSam31Runtime
            runtime = OfficialSam31Runtime(checkpoint, text_detection_threshold=0.2)
        if tracker is None:
            from samgraph_core import LocalSam31SceneTracker
            tracker = LocalSam31SceneTracker(runtime, camera_id="wrist")
        if segmenter is None:
            from samgraph_core import LocalSam31Segmenter
            segmenter = LocalSam31Segmenter(runtime)
        self.runtime = runtime
        self.tracker = tracker
        self.segmenter = segmenter
        self.names = dict(names)

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()

    def run(self, *, wrist_frames_root: Path, agent_frames_root: Path,
            agent_predictions: Path, output: Path,
            mask_checkpoint_root: Path | None = None,
            episodes: tuple[int, ...] = (0,),
            tasks: tuple[str, ...] | None = None) -> dict[str, Any]:
        wrist_frames_root = Path(wrist_frames_root).resolve()
        agent_frames_root = Path(agent_frames_root).resolve()
        agent_predictions = Path(agent_predictions).resolve()
        output = Path(output)
        checkpoint_root = (Path(mask_checkpoint_root).resolve()
                           if mask_checkpoint_root is not None else None)
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        masks_root = output / "masks"
        masks_root.mkdir()
        agent_rows = _rows(agent_predictions)
        if not episodes or len(set(episodes)) != len(episodes) or any(
                type(episode) is not int or episode < 0 for episode in episodes):
            raise ValueError("episodes must contain unique nonnegative integers")
        selected_tasks = set(tasks) if tasks is not None else None
        if selected_tasks is not None and (not selected_tasks or not selected_tasks <= set(self.names)):
            raise ValueError("requested tasks are empty or absent from wrist name configuration")
        wrist_archives = discover_archives(wrist_frames_root, episodes=episodes)
        if selected_tasks is not None:
            wrist_archives = [path for path in wrist_archives if path.parent.name in selected_tasks]
        archive_pairs = {(path.parent.name, path.stem) for path in wrist_archives}
        archive_tasks = {task for task, _ in archive_pairs}
        expected_tasks = selected_tasks if selected_tasks is not None else archive_tasks
        expected_pairs = {
            (task, f"demo_{episode}")
            for task in expected_tasks for episode in episodes
        }
        if archive_pairs != expected_pairs:
            raise ValueError(
                f"wrist archives differ from requested task/episode set: "
                f"missing={sorted(expected_pairs - archive_pairs)} "
                f"extra={sorted(archive_pairs - expected_pairs)}"
            )
        if not archive_tasks <= set(self.names):
            raise ValueError("wrist name configuration omits one or more input tasks")
        if any((key[0], key[1]) not in archive_pairs for key in agent_rows):
            raise ValueError("frozen agent rows include task/episode pairs outside this wrist shard")

        rows_out: list[dict[str, Any]] = []
        coverage: dict[str, Counter] = defaultdict(Counter)
        coverage_by_episode: dict[str, Counter] = defaultdict(Counter)
        archive_hashes = {}
        agent_archive_hashes = {}
        checkpoint_hashes: dict[str, str] = {}
        checkpoint_frames = 0
        geometry = samgraph_spatial_mask_geometry()
        for wrist_archive in wrist_archives:
            task = wrist_archive.parent.name
            demo = wrist_archive.stem
            agent_archive = agent_frames_root / task / f"{demo}.zip"
            if not agent_archive.is_file():
                raise FileNotFoundError(agent_archive)
            archive_key = f"{task}/{demo}.zip"
            archive_hashes[archive_key] = _sha(wrist_archive)
            agent_archive_hashes[archive_key] = _sha(agent_archive)
            controller = None
            native_started = False
            known_checkpoint_tracks: dict[str, str] = {}
            resolver = CrossViewBowlResolver()
            wrist_iter = iter_zip_frames(wrist_archive, undo_rotation=False)
            agent_iter = iter_zip_frames(agent_archive, undo_rotation=True)
            try:
                seen = 0
                for wrist_frame, agent_frame in zip(wrist_iter, agent_iter, strict=True):
                    if wrist_frame.frame != agent_frame.frame:
                        raise ValueError(f"camera frame mismatch: {task}")
                    checkpoint = None
                    if checkpoint_root is not None and wrist_frame.frame % 5 == 0:
                        checkpoint = checkpoint_root / task / demo / f"{wrist_frame.frame:06d}.npz"
                        if demo == "demo_0" and not checkpoint.is_file():
                            legacy_checkpoint = checkpoint_root / task / f"{wrist_frame.frame:06d}.npz"
                            if legacy_checkpoint.is_file():
                                checkpoint = legacy_checkpoint
                    if wrist_frame.frame % 5 and not native_started:
                        # A checkpointed prefix has no hidden SAM tracker state.
                        # Start native tracking exactly at its first missing
                        # sampled frame instead of recomputing the prefix.
                        continue
                    if checkpoint is not None and checkpoint.is_file() and not native_started:
                        scene = _checkpoint_scene(checkpoint, known_checkpoint_tracks)
                        relative = checkpoint.relative_to(checkpoint_root).as_posix()
                        checkpoint_hashes[relative] = _sha(checkpoint)
                        checkpoint_frames += 1
                    else:
                        if controller is None:
                            controller = WristSceneController(
                                self.segmenter, self.tracker, geometry_rules=geometry,
                                task_prompts={task: self.names[task]},
                            )
                        if not native_started:
                            scene = controller.start_episode(
                                task=f"{task}/{demo}", instruction=task.replace("_", " "),
                                frame=wrist_frame.frame, rgb=wrist_frame.rgb,
                            )
                            native_started = True
                        else:
                            scene = controller.step(frame=wrist_frame.frame, rgb=wrist_frame.rgb)
                    if wrist_frame.frame % 5:
                        continue
                    key = (task, demo, wrist_frame.frame)
                    if key not in agent_rows:
                        raise ValueError(f"frozen agent row missing for wrist frame: {key}")
                    agent_row = agent_rows[key]
                    if agent_row["source_sha256"] != agent_frame.source_sha256:
                        raise ValueError(f"frozen agent RGB identity mismatch: {key}")
                    agent_masks = _load_masks(agent_predictions, agent_row)
                    agent_states = list(agent_row.get("object_states", []))
                    wrist_masks = {name: np.asarray(mask, dtype=bool)
                                   for name, mask in scene.get("masks", {}).items()}
                    identity_evidence = resolver.update(
                        frame=wrist_frame.frame,
                        wrist_rgb=wrist_frame.rgb,
                        wrist_masks=wrist_masks,
                        wrist_states=scene["states"],
                        agent_rgb=agent_frame.rgb,
                        agent_masks=agent_masks,
                        agent_states=agent_states,
                    )
                    agent_outputs = _agent_outputs(agent_states)
                    members: set[str] = set()
                    objects = []
                    observed_bowls = [state for state in scene["states"]
                                      if state["class_id"] == "black_bowl"
                                      and state["observation_state"] == "current_observation"]
                    agent_bowls = [state for state in agent_states
                                   if state.get("class_id") == "black_bowl"]
                    collective_bowls = (not resolver.mapping and len(observed_bowls) == 2
                                        and len(agent_bowls) == 2)
                    if collective_bowls:
                        members.update(str(state.get("output_id") or state["track_id"])
                                       for state in agent_bowls)
                    for state in scene["states"]:
                        wrist_id = str(state["track_id"])
                        mask_key = str(state.get("output_id") or wrist_id)
                        eligible = state["observation_state"] == "current_observation"
                        agent_track = None
                        agent_output = None
                        correspondence = "not_current"
                        if eligible and state["class_id"] != "black_bowl":
                            agent_output = _unique_agent_output(agent_states, str(state["class_id"]))
                            correspondence = "unique_class" if agent_output else "missing_agent_class"
                        elif eligible and wrist_id in resolver.mapping:
                            agent_track = resolver.mapping[wrist_id]
                            agent_output = agent_outputs.get(agent_track)
                            correspondence = resolver.assignment_method or "causal_mapping"
                        elif eligible and collective_bowls:
                            correspondence = "collective_two_bowl_membership_identity_unresolved"
                        elif eligible:
                            correspondence = "unresolved_cross_view_bowl_identity"
                        if eligible and agent_output:
                            members.add(agent_output)
                        objects.append({
                            **state,
                            "wrist_track_id": wrist_id,
                            "wrist_mask_key": mask_key if mask_key in wrist_masks else None,
                            "agent_track_id": agent_track,
                            "agent_output_id": agent_output,
                            "correspondence": correspondence,
                            "included_in_membership": bool(agent_output in members) if agent_output else collective_bowls,
                        })
                        coverage[task][state["observation_state"]] += 1
                    source_triplets = list(agent_row.get("triplets", []))
                    triplets = filter_agent_triplets(source_triplets, members)
                    task_mask_dir = masks_root / task / demo
                    task_mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = task_mask_dir / f"{wrist_frame.frame:06d}.npz"
                    np.savez_compressed(mask_path, **wrist_masks)
                    row = {
                        "schema": "samgraph.wrist_prediction.v1",
                        "task": task, "demo": demo, "frame": wrist_frame.frame,
                        "camera": "wrist", "width": int(wrist_frame.rgb.shape[1]),
                        "height": int(wrist_frame.rgb.shape[0]),
                        "wrist_source_name": wrist_frame.source_name,
                        "wrist_source_sha256": wrist_frame.source_sha256,
                        "agent_source_sha256": agent_frame.source_sha256,
                        "agent_prediction_sha256": _sha(agent_predictions),
                        "agent_relation_revision": agent_row.get("relation_revision"),
                        "source_agent_triplets": source_triplets,
                        "triplets": triplets,
                        "wrist_membership": sorted(members),
                        "wrist_objects": objects,
                        "identity_evidence": identity_evidence,
                        "acquisition_diagnostics": scene.get("acquisition_diagnostics"),
                        "mask_cache": mask_path.relative_to(output).as_posix(),
                        "mask_cache_sha256": _sha(mask_path),
                        "selection_contract": "current RGB mask only; no remembered mask membership",
                        "relations_contract": "ordered subset of frozen agent triplets; no wrist relation inference",
                    }
                    rows_out.append(row)
                    coverage[task]["sampled_frames"] += 1
                    coverage[task]["empty_graph_frames"] += int(not triplets)
                    coverage[task]["unresolved_identity_frames"] += int(
                        identity_evidence["decision"] == "unresolved_correspondence")
                    episode_counts = coverage_by_episode[f"{task}/{demo}"]
                    episode_counts["sampled_frames"] += 1
                    episode_counts["empty_graph_frames"] += int(not triplets)
                    episode_counts["unresolved_identity_frames"] += int(
                        identity_evidence["decision"] == "unresolved_correspondence")
                    for state in scene["states"]:
                        episode_counts[state["observation_state"]] += 1
                    seen += 1
                expected_count = sum(1 for key in agent_rows if key[0] == task and key[1] == demo)
                if seen != expected_count:
                    raise ValueError(f"sampled frame coverage mismatch: {task}/{demo}")
            finally:
                if controller is not None and native_started:
                    controller.close_episode()

        if {(row["task"], row["demo"], row["frame"]) for row in rows_out} != set(agent_rows):
            raise ValueError("wrist rows do not exactly match frozen agent frame keys")
        predictions = output / "predictions.jsonl"
        with predictions.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows_out:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "schema": "samgraph.wrist_run.v1",
            "rows": len(rows_out), "tasks": len(archive_tasks),
            "task_ids": sorted(archive_tasks),
            "episodes": sorted(episodes),
            "frame_stride": 5, "tracking_stride": 1, "resolution": [1024, 1024],
            "predictions_sha256": _sha(predictions),
            "agent_predictions": str(agent_predictions),
            "agent_predictions_sha256": _sha(agent_predictions),
            "wrist_archive_sha256": archive_hashes,
            "agent_archive_sha256": agent_archive_hashes,
            "mask_checkpoint_root": str(checkpoint_root) if checkpoint_root else None,
            "mask_checkpoint_frames": checkpoint_frames,
            "mask_checkpoint_sha256": checkpoint_hashes,
            "coverage": {task: dict(counts) for task, counts in sorted(coverage.items())},
            "coverage_by_episode": {
                key: dict(counts) for key, counts in sorted(coverage_by_episode.items())
            },
            "prediction_inputs": "wrist RGB, synchronized agent RGB, frozen agent masks/states/triplets, task text",
            "forbidden_predictor_inputs": ["HDF5", "simulator geometry", "simulator boxes",
                                           "segmentation labels", "target triplets", "reference masks"],
            "occlusion_limitation": (
                "RGB masks can omit occluded objects that remain projected in notebook simulator boxes; "
                "membership and relation errors must be reported separately."
            ),
            "evaluation_performed": False,
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
