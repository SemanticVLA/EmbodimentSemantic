"""LeRobot v3 reader for the local SO101 real-robot task directories.

The adapter joins trajectory timestamps to encoded camera timestamps.  It does
not assume that equal camera frame indices imply synchronization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


CAMERA_KEYS = {
    "agent_view": "observation.images.agent_view",
    "wrist": "observation.images.wrist",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    episode: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    instruction: str
    camera_segments: Mapping[str, Mapping[str, int | float | str]]


@dataclass(frozen=True)
class SO101Frame:
    task: str
    episode: int
    camera: str
    frame_index: int
    trajectory_timestamp_s: float
    video_timestamp_s: float
    relative_video_timestamp_s: float
    timestamp_error_s: float
    rgb: np.ndarray
    source_sha256: str
    video_path: str


class SO101Dataset:
    """Read task metadata, trajectory timestamps, and native RGB video frames."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.task_paths = {
            path.name: path for path in sorted(self.root.iterdir())
            if path.is_dir() and (path / "meta" / "info.json").is_file()
        }
        if not self.task_paths:
            raise FileNotFoundError(f"no SO101 task directories under {self.root}")
        self._episodes: dict[str, dict[int, EpisodeRecord]] = {}
        self._timestamps: dict[tuple[str, int], list[float]] = {}

    @staticmethod
    def _pyarrow():
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("SO101 input requires pyarrow; install the 'so101' extra") from exc
        return pa, pc, pq

    def _episode_rows(self, task: str) -> dict[int, EpisodeRecord]:
        if task in self._episodes:
            return self._episodes[task]
        if task not in self.task_paths:
            raise KeyError(f"unknown SO101 task: {task}")
        pa, _, pq = self._pyarrow()
        files = sorted((self.task_paths[task] / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"episode metadata is missing for {task}")
        table = pa.concat_tables([pq.read_table(path) for path in files])
        rows: dict[int, EpisodeRecord] = {}
        for raw in table.to_pylist():
            episode = int(raw["episode_index"])
            if episode in rows:
                raise ValueError(f"duplicate episode metadata: {task}/{episode}")
            segments = {}
            for camera, key in CAMERA_KEYS.items():
                prefix = f"videos/{key}"
                required = [f"{prefix}/chunk_index", f"{prefix}/file_index",
                            f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"]
                if any(name not in raw or raw[name] is None for name in required):
                    continue
                chunk = int(raw[required[0]])
                file_index = int(raw[required[1]])
                path = (self.task_paths[task] / "videos" / key /
                        f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")
                segments[camera] = {
                    "chunk_index": chunk,
                    "file_index": file_index,
                    "from_timestamp": float(raw[required[2]]),
                    "to_timestamp": float(raw[required[3]]),
                    "path": str(path),
                }
            tasks = raw.get("tasks") or []
            rows[episode] = EpisodeRecord(
                task=task,
                episode=episode,
                length=int(raw["length"]),
                dataset_from_index=int(raw["dataset_from_index"]),
                dataset_to_index=int(raw["dataset_to_index"]),
                instruction=str(tasks[0]) if tasks else task.replace("-", " "),
                camera_segments=segments,
            )
        self._episodes[task] = rows
        return rows

    def episode(self, task: str, episode: int) -> EpisodeRecord:
        rows = self._episode_rows(task)
        if episode not in rows:
            raise FileNotFoundError(f"SO101 episode is unavailable: {task}/episode_{episode}")
        return rows[episode]

    def episode_ids(self, task: str) -> tuple[int, ...]:
        return tuple(sorted(self._episode_rows(task)))

    def trajectory_timestamps(self, task: str, episode: int) -> list[float]:
        key = (task, int(episode))
        if key in self._timestamps:
            return self._timestamps[key]
        pa, pc, pq = self._pyarrow()
        files = sorted((self.task_paths[task] / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"trajectory parquet is missing for {task}")
        selected = []
        for path in files:
            table = pq.read_table(path, columns=["timestamp", "frame_index", "episode_index"])
            table = table.filter(pc.equal(table["episode_index"], pa.scalar(int(episode))))
            if table.num_rows:
                selected.extend(table.select(["timestamp", "frame_index"]).to_pylist())
        selected.sort(key=lambda row: int(row["frame_index"]))
        record = self.episode(task, episode)
        frames = [int(row["frame_index"]) for row in selected]
        if frames != list(range(record.length)):
            raise ValueError(f"non-contiguous trajectory frames: {task}/episode_{episode}")
        timestamps = [float(row["timestamp"]) for row in selected]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"non-increasing trajectory timestamps: {task}/episode_{episode}")
        self._timestamps[key] = timestamps
        return timestamps

    def iter_frames(self, task: str, episode: int, camera: str = "agent_view") -> Iterator[SO101Frame]:
        if camera not in CAMERA_KEYS:
            raise ValueError(f"unsupported SO101 camera: {camera}")
        record = self.episode(task, episode)
        if camera not in record.camera_segments:
            raise FileNotFoundError(f"{camera} metadata missing: {task}/episode_{episode}")
        segment = record.camera_segments[camera]
        video_path = Path(str(segment["path"]))
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        timestamps = self.trajectory_timestamps(task, episode)
        try:
            import av
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("SO101 video input requires PyAV; install the 'so101' extra") from exc
        start = float(segment["from_timestamp"])
        stop = float(segment["to_timestamp"])
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else 30.0
            tolerance = max(1.0 / rate * 0.51, 1e-4)
            seek_time = max(0.0, start - 1.0)
            container.seek(int(seek_time / float(stream.time_base)), stream=stream, backward=True)
            emitted = 0
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                video_ts = float(decoded.pts * stream.time_base)
                if video_ts < start - tolerance:
                    continue
                if video_ts >= stop - tolerance and emitted >= record.length:
                    break
                if emitted >= record.length:
                    break
                trajectory_ts = timestamps[emitted]
                relative = video_ts - start
                error = relative - trajectory_ts
                if abs(error) > tolerance:
                    raise ValueError(
                        f"camera/trajectory timestamp mismatch {task}/episode_{episode}/"
                        f"{camera}/frame_{emitted}: {error:+.6f}s"
                    )
                rgb = np.ascontiguousarray(decoded.to_ndarray(format="rgb24"), dtype=np.uint8)
                yield SO101Frame(
                    task=task,
                    episode=episode,
                    camera=camera,
                    frame_index=emitted,
                    trajectory_timestamp_s=trajectory_ts,
                    video_timestamp_s=video_ts,
                    relative_video_timestamp_s=relative,
                    timestamp_error_s=error,
                    rgb=rgb,
                    source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    video_path=str(video_path),
                )
                emitted += 1
            if emitted != record.length:
                raise ValueError(
                    f"video segment frame count mismatch {task}/episode_{episode}/{camera}: "
                    f"expected {record.length}, decoded {emitted}"
                )

    def inventory(self) -> dict[str, Any]:
        tasks = {}
        total_available = 0
        total_frames = 0
        for task, path in self.task_paths.items():
            info = json.loads((path / "meta" / "info.json").read_text(encoding="utf-8"))
            episodes = self._episode_rows(task)
            available_ids = sorted(episodes)
            declared = int(info.get("total_episodes", 0))
            missing = sorted(set(range(declared)) - set(available_ids))
            cameras = {}
            for camera, key in CAMERA_KEYS.items():
                feature = (info.get("features") or {}).get(key)
                segments = sum(camera in record.camera_segments for record in episodes.values())
                files = sorted((path / "videos" / key).rglob("*.mp4"))
                cameras[camera] = {
                    "declared": feature,
                    "episode_segments": segments,
                    "video_files": len(files),
                }
            sync_mismatches = []
            for episode, record in episodes.items():
                if not all(name in record.camera_segments for name in CAMERA_KEYS):
                    sync_mismatches.append({"episode": episode, "reason": "camera_missing"})
                    continue
                agent, wrist = record.camera_segments["agent_view"], record.camera_segments["wrist"]
                comparable = (agent["file_index"], agent["from_timestamp"], agent["to_timestamp"])
                other = (wrist["file_index"], wrist["from_timestamp"], wrist["to_timestamp"])
                if comparable != other:
                    sync_mismatches.append({"episode": episode, "agent": comparable, "wrist": other})
            rows = sum(record.length for record in episodes.values())
            total_available += len(episodes)
            total_frames += rows
            tasks[task] = {
                "declared_episodes": declared,
                "available_episodes": len(episodes),
                "available_episode_ids": available_ids,
                "missing_declared_episode_ids": missing,
                "declared_frames": int(info.get("total_frames", 0)),
                "available_frames": rows,
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
                "cameras": cameras,
                "agent_wrist_metadata_sync_mismatches": sync_mismatches,
            }
        return {
            "schema": "samgraph.so101_inventory.v1",
            "dataset_root": str(self.root),
            "task_count": len(tasks),
            "available_episodes": total_available,
            "available_frames_per_camera": total_frames,
            "tasks": tasks,
        }


__all__ = [
    "CAMERA_KEYS", "EpisodeRecord", "SO101Dataset", "SO101Frame", "sha256_file",
]
