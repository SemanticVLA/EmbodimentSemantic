from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .artifacts import ArtifactStore
from .schemas import EpisodeRecord, SampledFrameRecord, read_jsonl


VIDEO_PREFIX = "observation.images."
CAMERAS = ("agent_view", "wrist")


def _read_parquet_rows(paths: Iterable[Path], columns: list[str]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read SO101 metadata: pip install pyarrow") from exc

    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected = [column for column in columns if column in available]
        if selected:
            rows.extend(parquet.read(columns=selected).to_pylist())
    return rows


def list_task_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").exists() and (path / "videos").is_dir()
    )


def _task_episode_rows(task_dir: Path) -> list[dict[str, Any]]:
    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for camera in (*CAMERAS, "agent_view_depth"):
        key = f"videos/{VIDEO_PREFIX}{camera}"
        columns.extend(
            [
                f"{key}/chunk_index",
                f"{key}/file_index",
                f"{key}/from_timestamp",
                f"{key}/to_timestamp",
            ]
        )
    rows = _read_parquet_rows(sorted((task_dir / "meta" / "episodes").rglob("*.parquet")), columns)
    return sorted(rows, key=lambda row: int(row.get("episode_index", -1)))


def index_dataset(dataset_root: str | Path) -> tuple[list[EpisodeRecord], list[SampledFrameRecord], dict[str, Any]]:
    root = Path(dataset_root).resolve()
    episodes: list[EpisodeRecord] = []
    sampled: list[SampledFrameRecord] = []
    task_reports: list[dict[str, Any]] = []

    for task_dir in list_task_dirs(root):
        info = json.loads((task_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        fps = float(info.get("fps", 30))
        camera_sizes = {
            camera: (
                int(info.get("features", {}).get(f"{VIDEO_PREFIX}{camera}", {}).get("shape", [480, 640])[1]),
                int(info.get("features", {}).get(f"{VIDEO_PREFIX}{camera}", {}).get("shape", [480, 640])[0]),
            )
            for camera in CAMERAS
        }
        episode_rows = _task_episode_rows(task_dir)
        metadata_indices = {int(row["episode_index"]) for row in episode_rows}
        declared_episodes = int(info.get("total_episodes", len(episode_rows)))
        missing = sorted(set(range(declared_episodes)) - metadata_indices)
        task_sample_count = 0

        for row in episode_rows:
            episode_index = int(row["episode_index"])
            episode = f"episode_{episode_index}"
            task_texts = [str(value).strip() for value in row.get("tasks", []) if str(value).strip()]
            task_text = task_texts[0] if task_texts else task_dir.name.replace("-", " ")
            videos: dict[str, dict[str, Any]] = {}
            for camera in (*CAMERAS, "agent_view_depth"):
                base = f"videos/{VIDEO_PREFIX}{camera}"
                chunk = int(row.get(f"{base}/chunk_index", 0))
                file_index = int(row.get(f"{base}/file_index", 0))
                path = task_dir / "videos" / f"{VIDEO_PREFIX}{camera}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                videos[camera] = {
                    "path": str(path.resolve()),
                    "chunk_index": chunk,
                    "file_index": file_index,
                    "from_timestamp": float(row.get(f"{base}/from_timestamp", 0.0)),
                    "to_timestamp": float(row.get(f"{base}/to_timestamp", 0.0)),
                    "exists": path.exists(),
                }

            episodes.append(
                EpisodeRecord(
                    task=task_dir.name,
                    task_text=task_text,
                    episode=episode,
                    episode_index=episode_index,
                    length=int(row["length"]),
                    fps=fps,
                    dataset_from_index=int(row.get("dataset_from_index", 0)),
                    dataset_to_index=int(row.get("dataset_to_index", 0)),
                    data_chunk_index=int(row.get("data/chunk_index", 0)),
                    data_file_index=int(row.get("data/file_index", 0)),
                    videos=videos,
                )
            )

            for camera in CAMERAS:
                frame_dir = task_dir / "frames" / camera / episode
                for image_path in sorted(frame_dir.glob("frame_*.jpg")):
                    frame = int(image_path.stem.rsplit("_", 1)[1])
                    width, height = camera_sizes[camera]
                    sampled.append(
                        SampledFrameRecord(
                            task=task_dir.name,
                            episode=episode,
                            episode_index=episode_index,
                            frame=frame,
                            timestamp=frame / fps,
                            camera=camera,
                            width=width,
                            height=height,
                            image_path=str(image_path.resolve()),
                        )
                    )
                    task_sample_count += 1

        task_reports.append(
            {
                "task": task_dir.name,
                "declared_episodes": declared_episodes,
                "usable_episodes": len(episode_rows),
                "missing_metadata_episode_indices": missing,
                "declared_frames": int(info.get("total_frames", 0)),
                "usable_frames": sum(int(row["length"]) for row in episode_rows),
                "sampled_jpegs": task_sample_count,
            }
        )

    report = {
        "dataset_root": str(root),
        "usable_episodes": len(episodes),
        "usable_episode_frames": sum(item.length for item in episodes),
        "sampled_jpegs": len(sampled),
        "sampled_per_camera": {
            camera: sum(1 for item in sampled if item.camera == camera) for camera in CAMERAS
        },
        "tasks": task_reports,
    }
    return episodes, sampled, report


def write_dataset_index(dataset_root: str | Path, artifacts: ArtifactStore) -> dict[str, Any]:
    episodes, sampled, report = index_dataset(dataset_root)
    artifacts.write_jsonl("index/episodes.jsonl", (item.to_dict() for item in episodes))
    artifacts.write_jsonl("index/sampled_frames.jsonl", (item.to_dict() for item in sampled))
    artifacts.write_json("reports/dataset_audit.json", report)
    return report


def load_episode_index(path: str | Path) -> list[EpisodeRecord]:
    return [EpisodeRecord.from_dict(value) for value in read_jsonl(path)]


def load_sampled_index(path: str | Path) -> list[SampledFrameRecord]:
    return [SampledFrameRecord.from_dict(value) for value in read_jsonl(path)]


def load_state_rows(dataset_root: str | Path, usable_episodes: Iterable[EpisodeRecord]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_task: dict[str, set[int]] = defaultdict(set)
    for episode in usable_episodes:
        by_task[episode.task].add(episode.episode_index)

    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    root = Path(dataset_root)
    columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index"]
    for task, active_indices in by_task.items():
        rows = _read_parquet_rows(sorted((root / task / "data").rglob("*.parquet")), columns)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            episode_index = int(row["episode_index"])
            if episode_index in active_indices:
                grouped[episode_index].append(row)
        for episode_index, episode_rows in grouped.items():
            result[(task, episode_index)] = sorted(episode_rows, key=lambda row: int(row["frame_index"]))
    return result
