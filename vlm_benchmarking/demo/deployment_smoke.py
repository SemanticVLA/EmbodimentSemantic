from __future__ import annotations

import argparse
import importlib
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from .libero_backend import (
    BundledFrameCache,
    CAMERA_INFO,
    Hdf5SceneGraphStore,
    PredictionStore,
    SimFrameRenderer,
    resolve_bddl_file,
)
from .so101_backend import DemoRepository
from .so101_proxy_demo.proxy.artifacts import ArtifactStore
from .so101_proxy_demo.proxy.config import load_config


DEMO_ROOT = Path(__file__).resolve().parent


def validate_runtime_imports(*, live_renderer: bool) -> None:
    modules = ["h5py", "numpy", "PIL", "yaml"]
    if live_renderer:
        modules.extend(
            (
                "bddl",
                "cloudpickle",
                "easydict",
                "future",
                "gym",
                "hydra",
                "matplotlib",
                "mujoco",
                "num2words",
                "cv2",
                "robosuite",
                "termcolor",
            )
        )
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise RuntimeError(f"Runtime dependency import failed: {module}") from exc


def libero_stores() -> dict[str, Hdf5SceneGraphStore]:
    cache_root = DEMO_ROOT / "libero_demo_cache"
    return {
        camera: Hdf5SceneGraphStore(cache_root, camera, semantic_cache_dir=None)
        for camera in CAMERA_INFO
    }


def validate_libero_metadata(stores: dict[str, Hdf5SceneGraphStore]) -> list[dict[str, str]]:
    store = stores["agentview"]
    tasks = store.list_tasks()
    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 LIBERO tasks, found {len(tasks)}")
    for task in tasks:
        demos = store.list_demos(task["id"])
        if [demo["id"] for demo in demos] != ["demo_0"]:
            raise RuntimeError(f"{task['id']} must contain only demo_0")

    predictions = PredictionStore(DEMO_ROOT / "libero_prediction_cache")
    for task in tasks:
        if not predictions.list_runs(task["id"], "agentview"):
            raise RuntimeError(f"LIBERO prediction cache is empty for task: {task['id']}")
    return tasks


def validate_live_libero(stores: dict[str, Hdf5SceneGraphStore], tasks: list[dict[str, str]]) -> None:
    from libero.libero import get_libero_path

    store = stores["agentview"]
    for task in tasks:
        resolve_bddl_file(store.task_path(task["id"]), get_libero_path)

    renderer = SimFrameRenderer(None, max_res=64, rotate_agentview=True)
    try:
        for task in tasks:
            task_id = task["id"]
            hdf5_path = store.task_path(task_id)
            state = store.state_for_frame(task_id, "demo_0", 0)
            transforms = store.fixed_body_transforms_for_frame(task_id, "demo_0", 0)
            try:
                image = renderer.render_state("agentview", 64, hdf5_path, state, transforms)
            except Exception as exc:
                raise RuntimeError(f"LIBERO render failed for task: {task_id}") from exc
            if image.size != (64, 64):
                raise RuntimeError(f"Unexpected LIBERO smoke-frame size for {task_id}: {image.size}")
            if not any(low != high for low, high in image.getextrema()):
                raise RuntimeError(f"LIBERO smoke frame is blank for task: {task_id}")
    finally:
        renderer.close()


def validate_bundled_libero(
    stores: dict[str, Hdf5SceneGraphStore],
    tasks: list[dict[str, str]],
    cache_dir: Path,
) -> None:
    cache = BundledFrameCache(cache_dir)
    if not cache.enabled:
        raise RuntimeError(f"Bundled LIBERO cache is missing or has no manifest: {cache_dir}")
    if cache.resolution != 1024:
        raise RuntimeError(f"Bundled LIBERO cache must be 1024px, found {cache.resolution}px")

    for camera, store in stores.items():
        for task in tasks:
            task_id = task["id"]
            for demo in store.list_demos(task_id):
                demo_id = demo["id"]
                frame_count = int(demo["frame_count"])
                record = cache.record(camera, task_id, demo_id)
                path = cache.archive_path(camera, task_id, demo_id)
                if record is None or int(record.get("frame_count", -1)) != frame_count:
                    raise RuntimeError(
                        f"Bundled frame count mismatch for {camera}/{task_id}/{demo_id}"
                    )
                if path is None or not path.is_file():
                    raise RuntimeError(f"Bundled frame archive is missing: {path}")
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    if len(names) != frame_count:
                        raise RuntimeError(f"Archive entry count mismatch for {path}")
                for frame in {0, frame_count - 1}:
                    data = cache.read(camera, task_id, demo_id, frame)
                    if data is None:
                        raise RuntimeError(f"Bundled frame is missing: {camera}/{task_id}/{frame}")
                    with Image.open(BytesIO(data)) as image:
                        image.load()
                        if image.size != (1024, 1024):
                            raise RuntimeError(f"Unexpected bundled frame size in {path}: {image.size}")
                        if not any(low != high for low, high in image.getextrema()):
                            raise RuntimeError(f"Bundled frame is blank: {camera}/{task_id}/{frame}")


def validate_so101() -> None:
    config = load_config(DEMO_ROOT / "so101_demo_cache" / "config.yaml")
    repository = DemoRepository(config, ArtifactStore(config["paths"]["artifacts"]))
    tasks = repository.tasks()
    if len(tasks) != 5:
        raise RuntimeError(f"Expected 5 SO101 tasks, found {len(tasks)}")

    for task in tasks:
        episodes = repository.episodes_for(task["id"])
        if [episode["id"] for episode in episodes] != ["episode_0"]:
            raise RuntimeError(f"{task['id']} must contain only episode_0")
        frames = repository.frames_for(task["id"], "episode_0", "agent_view")
        if not frames:
            raise RuntimeError(f"{task['id']} has no SO101 agent-view frames")
        sample = repository.sampled_frame(task["id"], "episode_0", frames[0], "agent_view")
        image_path = Path(sample.image_path)
        if not image_path.is_absolute():
            image_path = repository.dataset_root / image_path
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(f"Missing SO101 source frame: {image_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--bundled-cache-dir", type=Path, default=DEMO_ROOT / "libero_frame_cache")
    args = parser.parse_args(argv)

    validate_runtime_imports(live_renderer=not args.cached_only)
    stores = libero_stores()
    tasks = validate_libero_metadata(stores)
    if args.cached_only:
        validate_bundled_libero(stores, tasks, args.bundled_cache_dir)
    else:
        validate_live_libero(stores, tasks)
    validate_so101()
    mode = "bundled LIBERO frames" if args.cached_only else "live LIBERO rendering"
    print(f"Deployment smoke test passed: {mode} and SO101 assets loaded")


if __name__ == "__main__":
    main()
