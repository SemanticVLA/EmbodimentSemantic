from __future__ import annotations

from pathlib import Path

from .libero_backend import Hdf5SceneGraphStore, PredictionStore, SimFrameRenderer, resolve_bddl_file
from .so101_backend import DemoRepository
from .so101_proxy_demo.proxy.artifacts import ArtifactStore
from .so101_proxy_demo.proxy.config import load_config


DEMO_ROOT = Path(__file__).resolve().parent


def validate_libero() -> None:
    from libero.libero import get_libero_path

    cache_root = DEMO_ROOT / "libero_demo_cache"
    store = Hdf5SceneGraphStore(cache_root, "agentview", semantic_cache_dir=None)
    tasks = store.list_tasks()
    if len(tasks) != 10:
        raise RuntimeError(f"Expected 10 LIBERO tasks, found {len(tasks)}")

    for task in tasks:
        demos = store.list_demos(task["id"])
        if [demo["id"] for demo in demos] != ["demo_0"]:
            raise RuntimeError(f"{task['id']} must contain only demo_0")
        resolve_bddl_file(store.task_path(task["id"]), get_libero_path)

    task_id = tasks[0]["id"]
    hdf5_path = store.task_path(task_id)
    state = store.state_for_frame(task_id, "demo_0", 0)
    transforms = store.fixed_body_transforms_for_frame(task_id, "demo_0", 0)
    renderer = SimFrameRenderer(None, max_res=64, rotate_agentview=True)
    try:
        image = renderer.render_state("agentview", 64, hdf5_path, state, transforms)
    finally:
        renderer.close()

    if image.size != (64, 64):
        raise RuntimeError(f"Unexpected LIBERO smoke-frame size: {image.size}")
    if not any(low != high for low, high in image.getextrema()):
        raise RuntimeError("LIBERO smoke frame is blank")

    predictions = PredictionStore(DEMO_ROOT / "libero_prediction_cache")
    if not predictions.list_runs(task_id, "agentview"):
        raise RuntimeError("LIBERO prediction cache is empty")


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


def main() -> None:
    validate_libero()
    validate_so101()
    print("Deployment smoke test passed: LIBERO rendered and SO101 assets loaded")


if __name__ == "__main__":
    main()
