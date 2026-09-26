import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
from tqdm import tqdm

from .io_utils import (
    append_jsonl,
    frame_to_pil,
    image_key_for_camera,
    list_hdf5_files,
    parse_triplets,
    task_name_from_path,
    write_csv,
)
from .models.base import BaseVLM
from .prompts import build_agentview_prompt, build_wrist_prompt


def _artifact_paths(out_root: Path, task_name: str, camera: str, prompt_version: str) -> tuple[Path, Path]:
    csv_dir = out_root / camera / "csv"
    json_dir = out_root / camera / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task_name}_{camera}_{prompt_version}"
    return csv_dir / f"{stem}.csv", json_dir / f"{stem}.jsonl"


def _load_completed(jsonl_path: Path) -> set[tuple[str, int]]:
    """Return set of (demo_key, frame_idx) already present in a JSONL log."""
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
                done.add((rec["demo"], rec["frame"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _load_existing_csv_rows(csv_path: Path) -> list[dict]:
    """Return rows already written to a CSV, or [] if it doesn't exist."""
    import csv as csv_mod

    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv_mod.DictReader(f))


def run_experiment(
    model: BaseVLM,
    config: dict,
    input_dir: str,
    output_dir: str,
    max_tasks: Optional[int] = None,
    max_demos: Optional[int] = None,
    max_frames: Optional[int] = None,
    task_id: Optional[int] = None,
    cameras_to_run: Optional[list[str]] = None,
    output_prompt_version: Optional[str] = None,
    verbose: bool = False,
):
    extraction = config["extraction"]
    cameras = cameras_to_run if cameras_to_run is not None else extraction.get("cameras", ["agentview"])
    if "frame_step" in extraction:
        step = extraction["frame_step"]
        step = 1 if step == 0 else step
        frame_indices = list(range(0, extraction.get("frame_max", 10000), step))
    else:
        frame_indices = extraction.get("frame_indices", [0])
    rotate180 = extraction.get("rotate_agentview_180", False)
    prompt_version = output_prompt_version if output_prompt_version is not None else extraction.get("prompt_version", "v1")
    objects = config["objects"]

    active_frame_indices = frame_indices[:max_frames] if max_frames is not None else frame_indices

    hdf5_files = list_hdf5_files(input_dir, max_tasks)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    if task_id is not None:
        if task_id >= len(hdf5_files):
            raise ValueError(f"--task-id {task_id} out of range (only {len(hdf5_files)} tasks found)")
        hdf5_files = [hdf5_files[task_id]]

    out_root = Path(output_dir) / model.registry_name
    out_root.mkdir(parents=True, exist_ok=True)

    for task_i, hdf5_path in enumerate(tqdm(hdf5_files, desc=model.registry_name), 1):
        task_name, task_hint = task_name_from_path(hdf5_path)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Task {task_i}/{len(hdf5_files)}: {task_name}")

        with h5py.File(hdf5_path, "r") as f:
            data_group = f["data"] if "data" in f else f
            if not isinstance(data_group, h5py.Group):
                raise TypeError(f"Expected HDF5 group at {hdf5_path}, got {type(data_group)}")

            file_objects = objects
            file_prompt_version = prompt_version
            file_cameras = cameras
            file_frame_indices = active_frame_indices
            file_prompts = {
                "agentview": build_agentview_prompt(file_objects, task_hint),
                "eye_in_hand": build_wrist_prompt(file_objects, task_hint),
            }

            demo_keys = sorted(data_group.keys())
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]

            for camera in file_cameras:
                img_key = image_key_for_camera(camera)
                csv_path, jsonl_path = _artifact_paths(out_root, task_name, camera, file_prompt_version)
                prompt = file_prompts.get(camera, file_prompts[file_cameras[0]])

                completed = _load_completed(jsonl_path)
                if completed:
                    print(f"  [resume] {len(completed)} (demo, frame) pairs already done, skipping.", flush=True)

                rows = _load_existing_csv_rows(csv_path)

                for demo_key in demo_keys:
                    demo = data_group[demo_key]
                    if not isinstance(demo, h5py.Group):
                        continue
                    obs = demo.get("obs")
                    frames = obs.get(img_key) if isinstance(obs, h5py.Group) else None
                    if not isinstance(frames, h5py.Dataset):
                        continue

                    should_rotate = rotate180 and camera == "agentview"
                    pending_frames = []
                    for idx in file_frame_indices:
                        if idx < len(frames) and (demo_key, idx) not in completed:
                            pending_frames.append((idx, frame_to_pil(frames[idx], rotate180=should_rotate)))

                    if not pending_frames:
                        continue

                    bs = model.batch_size
                    chunks = [pending_frames[i:i + bs] for i in range(0, len(pending_frames), bs)]
                    for chunk in chunks:
                        frame_indices_batch = [idx for idx, _ in chunk]
                        pil_imgs = [img for _, img in chunk]

                        print(
                            f"  [{datetime.now().strftime('%H:%M:%S')}] {demo_key} | {camera} | frames {frame_indices_batch}",
                            flush=True,
                        )
                        t0 = time.perf_counter()
                        responses = model.query_batch(pil_imgs, prompt)
                        total_latency = time.perf_counter() - t0
                        per_latency = round(total_latency / len(responses), 4)
                        print(f"    -> {total_latency:.1f}s total ({per_latency:.1f}s/frame)", flush=True)

                        for frame_idx, pil_img, response in zip(frame_indices_batch, pil_imgs, responses):
                            if not response:
                                continue
                            img_hash = hashlib.md5(pil_img.tobytes()).hexdigest()[:8]
                            append_jsonl(
                                str(jsonl_path),
                                {
                                    "task": task_name,
                                    "demo": demo_key,
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
                                        "demo": demo_key,
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
