"""
Runner for SO1001-format datasets (LeRobot folder structure with MP4 videos).

Dataset layout:
  <input_dir>/<task_name>/videos/observation.images.agent_view/chunk-000/file-000.mp4
  <input_dir>/<task_name>/videos/observation.images.wrist/chunk-000/file-000.mp4

Each camera MP4 is treated as one episode. No HDF5 ground truth - output only.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image
from tqdm import tqdm

from .io_utils import append_jsonl, parse_triplets, write_csv
from .models.base import BaseVLM
from .prompts import build_agentview_prompt, build_wrist_prompt


_VIDEO_KEY_PREFIX = "observation.images."
_CAMERA_ALIASES = {
    "agentview": "agent_view",
    "agent_view": "agent_view",
    "eye_in_hand": "wrist",
    "wrist": "wrist",
    "observation.images.agent_view": "agent_view",
    "observation.images.wrist": "wrist",
    "observation.images.agent_view_depth": "agent_view_depth",
}


@dataclass(frozen=True)
class _EpisodeClip:
    demo_key: str
    mp4_path: Path
    video_start_frame: int
    length: int
    fps: Optional[float]
    task_hint: Optional[str] = None


def _canonical_camera_name(camera: str) -> str:
    camera = camera.strip()
    alias = _CAMERA_ALIASES.get(camera, camera)
    if alias.startswith(_VIDEO_KEY_PREFIX):
        alias = alias[len(_VIDEO_KEY_PREFIX):]
    return alias


def _normalize_cameras(cameras: list[str]) -> list[str]:
    normalized = []
    for camera in cameras:
        canonical = _canonical_camera_name(camera)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _video_key_for_camera(camera: str) -> str:
    return f"{_VIDEO_KEY_PREFIX}{_canonical_camera_name(camera)}"


def _available_cameras(task_dir: Path) -> list[str]:
    videos_dir = task_dir / "videos"
    if not videos_dir.is_dir():
        return []
    cameras = []
    for video_dir in sorted(d for d in videos_dir.iterdir() if d.is_dir()):
        name = video_dir.name
        if name.startswith(_VIDEO_KEY_PREFIX):
            cameras.append(name[len(_VIDEO_KEY_PREFIX):])
    return cameras


def _list_episode_mp4s(task_dir: Path, camera: str) -> list[Path]:
    video_dir = task_dir / "videos" / _video_key_for_camera(camera)
    return sorted(video_dir.rglob("*.mp4")) if video_dir.is_dir() else []


def _mp4_for_video_indices(task_dir: Path, camera: str, chunk_index: int, file_index: int) -> Path:
    return (
        task_dir
        / "videos"
        / _video_key_for_camera(camera)
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def _list_task_dirs(input_dir: str, max_tasks: Optional[int] = None) -> list[Path]:
    root = Path(input_dir)
    tasks = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "videos").is_dir()
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def _read_info(task_dir: Path) -> dict:
    info_path = task_dir / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        with open(info_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_task_hint_with_pyarrow(tasks_path: Path) -> Optional[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None

    try:
        table = pq.read_table(tasks_path, columns=["task"])
        values = table.column("task").to_pylist()
    except Exception:
        return None

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _clean_task_hint_candidate(text: str) -> str:
    text = text.strip().strip("()[]{};:%")
    text = re.sub(r"^[^A-Za-z]+", "", text)
    if len(text) >= 3 and text[0].isupper() and text[1].isupper() and text[2].islower():
        text = text[1:]
    return re.sub(r"\s+", " ", text).strip()


def _read_task_hint_from_parquet_bytes(tasks_path: Path) -> Optional[str]:
    try:
        data = tasks_path.read_bytes()
    except OSError:
        return None

    reject = (
        "ARROW",
        "schema",
        "pandas",
        "parquet",
        "task_index",
        "field_name",
        "numpy_type",
        "creator",
        "columns",
    )
    candidates = []
    for raw in re.findall(rb"[ -~]{8,}", data):
        text = raw.decode("utf-8", errors="ignore")
        if any(token in text for token in reject):
            continue
        text = _clean_task_hint_candidate(text)
        if len(text.split()) >= 3:
            candidates.append(text)

    return max(candidates, key=len) if candidates else None


def _read_task_hint(task_dir: Path) -> Optional[str]:
    tasks_path = task_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    return (
        _read_task_hint_with_pyarrow(tasks_path)
        or _read_task_hint_from_parquet_bytes(tasks_path)
    )


def _read_episode_task_hint(task_dir: Path) -> Optional[str]:
    try:
        rows = _read_episode_rows(task_dir, ["tasks"])
    except RuntimeError:
        return None
    hints = []
    for row in rows:
        tasks = row.get("tasks") or []
        for task in tasks:
            if isinstance(task, str) and task.strip():
                hints.append(re.sub(r"\s+", " ", task).strip())
    return max(hints, key=len) if hints else None


def _task_name_from_dir(task_dir: Path) -> tuple[str, str]:
    name = task_dir.name
    hint = _read_episode_task_hint(task_dir) or _read_task_hint(task_dir) or name.replace("-", " ")
    return name, hint


def _extract_frame(mp4_path: Path, frame_idx: int) -> Optional[Image.Image]:
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _video_stats(mp4_path: Path) -> tuple[int, Optional[float]]:
    cap = cv2.VideoCapture(str(mp4_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    return count, fps if fps > 0 else None


def _episode_key_from_mp4(mp4_path: Path, chunks_size: int, fallback_idx: int) -> str:
    chunk_match = re.fullmatch(r"chunk-(\d+)", mp4_path.parent.name)
    file_match = re.fullmatch(r"file-(\d+)", mp4_path.stem)
    if chunk_match and file_match:
        episode_idx = int(chunk_match.group(1)) * chunks_size + int(file_match.group(1))
        return f"episode_{episode_idx}"
    return f"episode_{fallback_idx}"


def _episode_metadata_paths(task_dir: Path) -> list[Path]:
    episodes_dir = task_dir / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(episodes_dir.rglob("*.parquet"))


def _read_episode_rows(task_dir: Path, columns: list[str]) -> list[dict]:
    paths = _episode_metadata_paths(task_dir)
    if not paths:
        return []

    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SO1001 episode metadata is present, but pyarrow is not installed. "
            "Install pyarrow or run through an environment that includes it."
        ) from exc

    rows: list[dict] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        active_columns = [col for col in columns if col in schema_names]
        if not active_columns:
            continue
        table = parquet_file.read(columns=active_columns)
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: int(row.get("episode_index", 0)))
    return rows


def _load_episode_clips(task_dir: Path, camera: str) -> Optional[list[_EpisodeClip]]:
    paths = _episode_metadata_paths(task_dir)
    if not paths:
        return None

    video_key = _video_key_for_camera(camera)
    columns = [
        "episode_index",
        "length",
        "tasks",
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
    ]
    rows = _read_episode_rows(task_dir, columns)
    if not rows:
        return []

    stats_cache: dict[Path, tuple[int, Optional[float]]] = {}
    clips: list[_EpisodeClip] = []
    for row in rows:
        try:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            chunk_index = int(row[f"videos/{video_key}/chunk_index"])
            file_index = int(row[f"videos/{video_key}/file_index"])
            from_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        mp4_path = _mp4_for_video_indices(task_dir, camera, chunk_index, file_index)
        if not mp4_path.exists():
            continue

        if mp4_path not in stats_cache:
            stats_cache[mp4_path] = _video_stats(mp4_path)
        n_frames_total, fps = stats_cache[mp4_path]
        if n_frames_total <= 0:
            continue

        effective_fps = fps or 30.0
        video_start_frame = max(0, int(round(from_timestamp * effective_fps)))
        length = max(0, min(length, n_frames_total - video_start_frame))
        if length <= 0:
            continue

        task_hint = None
        tasks = row.get("tasks") or []
        if tasks:
            task_hint = re.sub(r"\s+", " ", str(tasks[0])).strip()

        clips.append(
            _EpisodeClip(
                demo_key=f"episode_{episode_index}",
                mp4_path=mp4_path,
                video_start_frame=video_start_frame,
                length=length,
                fps=fps,
                task_hint=task_hint,
            )
        )
    return clips


def _metadata_episode_indices(task_dir: Path) -> list[int]:
    rows = _read_episode_rows(task_dir, ["episode_index"])
    indices = []
    for row in rows:
        try:
            indices.append(int(row["episode_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(indices)


def _fallback_mp4_episode_clips(task_dir: Path, camera: str, chunks_size: int) -> list[_EpisodeClip]:
    clips = []
    for episode_i, mp4_path in enumerate(_list_episode_mp4s(task_dir, camera)):
        n_frames_total, fps = _video_stats(mp4_path)
        if n_frames_total <= 0:
            continue
        clips.append(
            _EpisodeClip(
                demo_key=_episode_key_from_mp4(mp4_path, chunks_size, episode_i),
                mp4_path=mp4_path,
                video_start_frame=0,
                length=n_frames_total,
                fps=fps,
            )
        )
    return clips


def _frame_indices_for_video(
    n_frames_total: int,
    frame_step: int,
    frame_max: Optional[int],
    max_frames: Optional[int],
) -> list[int]:
    stop = n_frames_total if frame_max is None else min(n_frames_total, frame_max)
    indices = list(range(0, stop, frame_step))
    return indices[:max_frames] if max_frames is not None else indices


def _prompt_for_camera(camera: str, objects: list[str], task_hint: str) -> str:
    camera = _canonical_camera_name(camera)
    if "wrist" in camera or camera == "eye_in_hand":
        return build_wrist_prompt(objects, task_hint)
    return build_agentview_prompt(objects, task_hint)


def _artifact_paths(out_root: Path, task_name: str, camera: str, prompt_version: str) -> tuple[Path, Path]:
    csv_dir = out_root / camera / "csv"
    json_dir = out_root / camera / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task_name}_{camera}_{prompt_version}"
    return csv_dir / f"{stem}.csv", json_dir / f"{stem}.jsonl"


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _save_input_frame(
    save_root: Path,
    model_name: str,
    task_name: str,
    camera: str,
    demo_key: str,
    frame_idx: int,
    image: Image.Image,
) -> None:
    frame_dir = (
        save_root
        / _safe_path_component(model_name)
        / _safe_path_component(task_name)
        / _safe_path_component(camera)
        / _safe_path_component(demo_key)
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    image.save(frame_dir / f"frame_{frame_idx:06d}.jpg", quality=95)


def _load_completed(jsonl_path: Path) -> set[tuple[str, int]]:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["demo"], int(rec["frame"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def _load_existing_csv_rows(csv_path: Path) -> list[dict]:
    import csv as csv_mod

    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv_mod.DictReader(f))


def run_so1001_experiment(
    model: BaseVLM,
    config: dict,
    input_dir: str,
    output_dir: str,
    max_tasks: Optional[int] = None,
    max_demos: Optional[int] = None,
    max_frames: Optional[int] = None,
    task_id: Optional[int] = None,
    cameras_to_run: Optional[list[str]] = None,
    save_frames_dir: Optional[str] = None,
    verbose: bool = False,
):
    so1001_cfg = config.get("so1001", {})
    extraction_cfg = config.get("extraction", {})
    prompt_version = so1001_cfg.get("prompt_version", extraction_cfg.get("prompt_version", "v1"))
    cameras = _normalize_cameras(cameras_to_run or so1001_cfg.get("cameras", ["agent_view", "wrist"]))
    objects = so1001_cfg.get("objects", config.get("objects", []))
    frame_step = int(so1001_cfg.get("frame_step", 30))
    frame_step = 1 if frame_step <= 0 else frame_step
    frame_max = so1001_cfg.get("frame_max")
    save_root = Path(save_frames_dir) if save_frames_dir else None

    task_dirs = _list_task_dirs(input_dir, max_tasks)
    if not task_dirs:
        raise FileNotFoundError(f"No SO1001 task directories found in {input_dir}")
    if task_id is not None:
        if task_id >= len(task_dirs):
            raise ValueError(f"--task-id {task_id} out of range (only {len(task_dirs)} tasks found)")
        task_dirs = [task_dirs[task_id]]

    out_root = Path(output_dir) / model.registry_name
    out_root.mkdir(parents=True, exist_ok=True)

    for task_i, task_dir in enumerate(tqdm(task_dirs, desc=model.registry_name), 1):
        task_name, task_hint = _task_name_from_dir(task_dir)
        task_info = _read_info(task_dir)
        chunks_size = int(task_info.get("chunks_size", 1000))
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Task {task_i}/{len(task_dirs)}: {task_name}")
        expected_episodes = task_info.get("total_episodes")
        if expected_episodes is not None:
            metadata_indices = _metadata_episode_indices(task_dir)
            if metadata_indices and len(metadata_indices) != int(expected_episodes):
                missing = sorted(set(range(metadata_indices[0], metadata_indices[-1] + 1)) - set(metadata_indices))
                missing_text = f"; missing episode indices={missing}" if missing else ""
                print(
                    f"  [warn] metadata has {len(metadata_indices)} episode row(s), "
                    f"info.json says {expected_episodes}{missing_text}",
                    flush=True,
                )
        if verbose:
            print(f"  task hint: {task_hint}")

        for camera in cameras:
            episode_clips = _load_episode_clips(task_dir, camera)
            if episode_clips is None:
                episode_clips = _fallback_mp4_episode_clips(task_dir, camera, chunks_size)
            if max_demos is not None:
                episode_clips = episode_clips[:max_demos]

            if not episode_clips:
                available = _available_cameras(task_dir)
                print(f"  [skip] no episode video found for camera '{camera}' in {task_dir.name}; available={available}")
                continue

            csv_path, jsonl_path = _artifact_paths(out_root, task_name, camera, prompt_version)
            completed = _load_completed(jsonl_path)
            if completed:
                print(f"  [resume] {len(completed)} (episode, frame) pairs already done, skipping.")

            rows = _load_existing_csv_rows(csv_path)

            for clip in episode_clips:
                active_indices = _frame_indices_for_video(clip.length, frame_step, frame_max, max_frames)
                pending_frames = [
                    idx for idx in active_indices
                    if (clip.demo_key, idx) not in completed
                ]

                fps_text = f"{clip.fps:.2f} fps" if clip.fps is not None else "unknown fps"
                seconds_text = f", ~{frame_step / clip.fps:.2f}s/sample" if clip.fps else ""
                print(
                    f"  [{camera}] {clip.demo_key}: {clip.length} episode frames, {fps_text}, "
                    f"frame_step={frame_step}{seconds_text}",
                    flush=True,
                )

                if not pending_frames:
                    print(f"  [skip] all frames already processed for {clip.demo_key} | {camera}")
                    continue

                if verbose:
                    print(f"    objects: {objects}")
                prompt = _prompt_for_camera(camera, objects, clip.task_hint or task_hint)
                bs = model.batch_size
                chunks = [pending_frames[i:i + bs] for i in range(0, len(pending_frames), bs)]

                for chunk in chunks:
                    pil_imgs = []
                    valid_indices = []
                    for idx in chunk:
                        img = _extract_frame(clip.mp4_path, clip.video_start_frame + idx)
                        if img is not None:
                            pil_imgs.append(img)
                            valid_indices.append(idx)
                            if save_root is not None:
                                _save_input_frame(
                                    save_root,
                                    model.registry_name,
                                    task_name,
                                    camera,
                                    clip.demo_key,
                                    idx,
                                    img,
                                )

                    if not pil_imgs:
                        continue

                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] {clip.demo_key} | {camera} | frames {valid_indices}",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    responses = model.query_batch(pil_imgs, prompt)
                    total_latency = time.perf_counter() - t0
                    per_latency = round(total_latency / len(responses), 4)
                    print(f"    -> {total_latency:.1f}s total ({per_latency:.1f}s/frame)", flush=True)

                    for frame_idx, pil_img, response in zip(valid_indices, pil_imgs, responses):
                        if not response:
                            continue
                        img_hash = hashlib.md5(pil_img.tobytes()).hexdigest()[:8]
                        append_jsonl(
                            str(jsonl_path),
                            {
                                "task": task_name,
                                "demo": clip.demo_key,
                                "frame": frame_idx,
                                "camera": camera,
                                "model": model.registry_name,
                                "input_hash": img_hash,
                                "response": response,
                                "latency_s": per_latency,
                            },
                        )

                        for a, rel, b in parse_triplets(response):
                            rows.append(
                                {
                                    "task": task_name,
                                    "demo": clip.demo_key,
                                    "frame": frame_idx,
                                    "camera": camera,
                                    "objectA": a,
                                    "relation": rel,
                                    "objectB": b,
                                }
                            )

            write_csv(str(csv_path), rows)
            if verbose:
                print(f"  {csv_path} - {len(rows)} triplets")
