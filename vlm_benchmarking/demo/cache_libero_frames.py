from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from .libero_backend import BundledFrameCache, CAMERA_INFO, Hdf5SceneGraphStore, SimFrameRenderer


def archive_relative_path(camera: str, task: str, demo: str) -> Path:
    return Path(camera) / task / f"{demo}.zip"


def archive_is_complete(path: Path, frame_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return (
                len(names) == frame_count
                and names[0] == BundledFrameCache.entry_name(0)
                and names[-1] == BundledFrameCache.entry_name(frame_count - 1)
                and archive.testzip() is None
            )
    except (OSError, zipfile.BadZipFile):
        return False


def write_archive(
    renderer: SimFrameRenderer,
    store: Hdf5SceneGraphStore,
    task: str,
    demo: str,
    output_path: Path,
    resolution: int,
    quality: int,
) -> None:
    demo_records = {record["id"]: record for record in store.list_demos(task)}
    frame_count = int(demo_records[demo]["frame_count"])
    hdf5_path = store.task_path(task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_path.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for frame in range(frame_count):
                image = renderer.render_state(
                    store.camera,
                    resolution,
                    hdf5_path,
                    store.state_for_frame(task, demo, frame),
                    store.fixed_body_transforms_for_frame(task, demo, frame),
                )
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False)
                entry = zipfile.ZipInfo(
                    BundledFrameCache.entry_name(frame),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_STORED
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, buffer.getvalue())
                if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == frame_count:
                    elapsed = time.monotonic() - started
                    print(
                        f"{store.camera}/{task}: {frame + 1}/{frame_count} frames ({elapsed:.1f}s)",
                        flush=True,
                    )
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    input_dir: Path,
    output_dir: Path,
    stores: dict[str, Hdf5SceneGraphStore],
    resolution: int,
    quality: int,
) -> dict[str, Any]:
    archives: dict[str, dict[str, Any]] = {}
    expected = 0
    cached = 0
    tasks = stores["agentview"].list_tasks()
    for camera, store in stores.items():
        for task_record in tasks:
            task = task_record["id"]
            for demo_record in store.list_demos(task):
                demo = demo_record["id"]
                frame_count = int(demo_record["frame_count"])
                expected += frame_count
                relative = archive_relative_path(camera, task, demo)
                path = output_dir / relative
                if not archive_is_complete(path, frame_count):
                    continue
                cached += frame_count
                archives[BundledFrameCache.key(camera, task, demo)] = {
                    "path": relative.as_posix(),
                    "frame_count": frame_count,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
    return {
        "version": 1,
        "resolution": resolution,
        "content_type": "image/jpeg",
        "extension": ".jpg",
        "quality": quality,
        "input": str(input_dir),
        "complete": cached == expected,
        "cached_frames": cached,
        "expected_frames": expected,
        "archives": archives,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the bundled 1024px LIBERO frame cache.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--camera", action="append", choices=sorted(CAMERA_INFO), default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cameras = args.camera or list(CAMERA_INFO)
    stores = {
        camera: Hdf5SceneGraphStore(
            args.input_dir,
            camera,
            rotate_agentview=True,
            semantic_cache_dir=None,
        )
        for camera in cameras
    }
    all_tasks = stores[cameras[0]].list_tasks()
    selected = set(args.task)
    tasks = [record["id"] for record in all_tasks if not selected or record["id"] in selected]
    missing_tasks = selected.difference(tasks)
    if missing_tasks:
        raise ValueError(f"Unknown LIBERO tasks: {sorted(missing_tasks)}")

    renderer = SimFrameRenderer(None, max_res=args.resolution, rotate_agentview=True)
    try:
        for task in tasks:
            for camera in cameras:
                store = stores[camera]
                demos = store.list_demos(task)
                for demo_record in demos:
                    demo = demo_record["id"]
                    frame_count = int(demo_record["frame_count"])
                    output_path = args.output_dir / archive_relative_path(camera, task, demo)
                    if not args.force and archive_is_complete(output_path, frame_count):
                        print(f"Skipping complete archive: {output_path}", flush=True)
                        continue
                    write_archive(
                        renderer,
                        store,
                        task,
                        demo,
                        output_path,
                        args.resolution,
                        args.quality,
                    )
    finally:
        renderer.close()

    manifest_stores = {
        camera: stores.get(camera)
        or Hdf5SceneGraphStore(args.input_dir, camera, rotate_agentview=True, semantic_cache_dir=None)
        for camera in CAMERA_INFO
    }
    manifest = build_manifest(
        args.input_dir,
        args.output_dir,
        manifest_stores,
        args.resolution,
        args.quality,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Cache manifest: {manifest['cached_frames']}/{manifest['expected_frames']} frames; "
        f"complete={manifest['complete']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
