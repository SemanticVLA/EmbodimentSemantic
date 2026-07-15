from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import h5py
from PIL import Image

from .libero_backend import CAMERA_INFO, DEFAULT_SUBJECT, Hdf5SceneGraphStore
from .so101_proxy_demo.proxy.artifacts import ArtifactStore
from .so101_proxy_demo.proxy.config import load_config
from .so101_backend import DemoRepository


DEMO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = DEMO_ROOT / "data"
VLM_ROOT = DEMO_ROOT.parent
LIBERO_DEMOS = ("demo_0",)
SO101_EPISODE = "episode_0"
DEFAULT_SAMPLE_COUNT = 12
WEBP_QUALITY = 78
CAMERA_LABELS = {
    "agentview": "Agent view",
    "eye_in_hand": "Eye in hand",
    "agent_view": "Agent view",
    "wrist": "Wrist",
}
RELATION_LABELS = {
    "is_left_of": "left",
    "is_right_of": "right",
    "is_in_front_of": "front",
    "is_behind": "behind",
    "is_on_top_of": "top",
    "is_below_of": "below",
    "is_inside": "inside",
    "contains": "contains",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_frames(available: Iterable[int], maximum: int) -> list[int]:
    frames = sorted(set(int(frame) for frame in available))
    if maximum <= 0 or len(frames) <= maximum:
        return frames
    if maximum == 1:
        return [frames[0]]
    selected = {
        frames[round(index * (len(frames) - 1) / (maximum - 1))]
        for index in range(maximum)
    }
    return sorted(selected)


def _reset_generated_directory(name: str) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    target = (DATA_ROOT / name).resolve()
    root = DATA_ROOT.resolve()
    if target.parent != root:
        raise ValueError(f"Refusing to clear generated path outside {root}: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def _image_path(
    dataset: str,
    task: str,
    sequence: str,
    camera: str,
    frame: int,
) -> tuple[Path, str]:
    relative = (
        Path("data")
        / dataset
        / task
        / sequence
        / camera
        / f"frame_{frame:06d}.webp"
    )
    return DEMO_ROOT / relative, relative.as_posix()


def _save_webp(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        path,
        format="WEBP",
        quality=WEBP_QUALITY,
        method=6,
        optimize=True,
    )


def _build_libero(input_dir: Path, sample_count: int) -> dict[str, Any]:
    _reset_generated_directory("libero")
    stores = {
        camera: Hdf5SceneGraphStore(input_dir, camera, rotate_agentview=True)
        for camera in CAMERA_INFO
    }
    sequences: list[dict[str, Any]] = []
    tasks = stores["agentview"].list_tasks()

    for task_item in tasks:
        task = task_item["id"]
        hdf5_path = stores["agentview"].task_path(task)
        with h5py.File(hdf5_path, "r") as handle:
            for demo in LIBERO_DEMOS:
                if demo not in handle["data"]:
                    continue
                observations = handle[f"data/{demo}/obs"]
                for camera, store in stores.items():
                    rgb_key = CAMERA_INFO[camera]["rgb"]
                    rgb = observations[rgb_key]
                    frames = _sample_frames(range(int(rgb.shape[0])), sample_count)
                    output_frames: list[dict[str, Any]] = []
                    for frame in frames:
                        image = Image.fromarray(rgb[frame]).convert("RGB")
                        if camera == "agentview":
                            image = image.transpose(Image.Transpose.ROTATE_180)
                        width, height = (512, 512)
                        destination, image_url = _image_path(
                            "libero", task, demo, camera, frame
                        )
                        _save_webp(image, destination)
                        overlay = store.frame_payload(
                            task,
                            demo,
                            frame,
                            (width, height),
                            DEFAULT_SUBJECT,
                        )
                        output_frames.append(
                            {
                                "frame": frame,
                                "image": image_url,
                                "width": width,
                                "height": height,
                                "bboxes": overlay["bboxes"],
                                "relations": [
                                    {
                                        "subject": edge["subject"],
                                        "relation": edge["relation"],
                                        "label": edge["label"],
                                        "object": edge["object"],
                                        "start": edge["start"],
                                        "end": edge["end"],
                                    }
                                    for edge in overlay["edges"]
                                ],
                            }
                        )
                    sequences.append(
                        {
                            "id": f"{task}:{demo}:{camera}",
                            "task": task,
                            "task_label": task_item["name"],
                            "sequence": demo,
                            "sequence_label": demo.replace("_", " "),
                            "camera": camera,
                            "camera_label": CAMERA_LABELS[camera],
                            "frames": output_frames,
                        }
                    )

    return _bundle(
        "libero",
        "LIBERO ground truth",
        "agentview",
        sequences,
        sample_count,
        "demo_0",
    )


def _bbox_center(value: dict[str, Any]) -> list[float]:
    x1, y1, x2, y2 = value["bbox"]
    return [round((x1 + x2) / 2, 3), round((y1 + y2) / 2, 3)]


def _so101_relations(
    repository: DemoRepository,
    task: str,
    episode: str,
    frame: int,
    camera: str,
    bboxes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    proxy = repository.proxy.get((task, episode, frame, camera, "gt"))
    if proxy is None and camera == "wrist":
        proxy = repository.proxy.get((task, episode, frame, "agent_view", "gt"))
    if proxy is None:
        return []

    output = []
    for relation in proxy.relations:
        if (
            relation.subject != "black_bowl"
            or relation.subject not in bboxes
            or relation.object not in bboxes
        ):
            continue
        output.append(
            {
                "subject": relation.subject,
                "relation": relation.relation,
                "label": RELATION_LABELS.get(
                    relation.relation, relation.relation.replace("is_", "")
                ),
                "object": relation.object,
                "start": _bbox_center(bboxes[relation.subject]),
                "end": _bbox_center(bboxes[relation.object]),
            }
        )
    return output


def _complete_so101_camera(
    repository: DemoRepository,
    task: str,
    episode: str,
    camera: str,
) -> bool:
    frames = repository.frames_for(task, episode, camera)
    return bool(frames) and all(
        (task, episode, frame, camera) in repository.bboxes for frame in frames
    )


def _build_so101(
    config_path: str | Path | None,
    sample_count: int,
) -> dict[str, Any]:
    _reset_generated_directory("so101")
    config = load_config(config_path)
    repository = DemoRepository(
        config,
        ArtifactStore(config["paths"]["artifacts"]),
    )
    sequences: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []

    for task_item in repository.tasks():
        task = task_item["id"]
        episodes = {item["id"]: item for item in repository.episodes_for(task)}
        if SO101_EPISODE not in episodes:
            continue
        for camera in ("agent_view", "wrist"):
            if not _complete_so101_camera(repository, task, SO101_EPISODE, camera):
                omitted.append(
                    {
                        "task": task,
                        "sequence": SO101_EPISODE,
                        "camera": camera,
                        "reason": "bbox coverage incomplete",
                    }
                )
                continue
            available = repository.frames_for(task, SO101_EPISODE, camera)
            frames = _sample_frames(available, sample_count)
            output_frames: list[dict[str, Any]] = []
            for frame in frames:
                payload = repository.frame_payload(
                    task,
                    SO101_EPISODE,
                    frame,
                    camera,
                    "gt",
                )
                sample = repository.sampled_frame(
                    task,
                    SO101_EPISODE,
                    frame,
                    camera,
                )
                destination, image_url = _image_path(
                    "so101", task, SO101_EPISODE, camera, frame
                )
                with Image.open(sample.image_path) as source:
                    _save_webp(source, destination)
                bboxes = payload["bboxes"]
                output_frames.append(
                    {
                        "frame": frame,
                        "image": image_url,
                        "width": payload["width"],
                        "height": payload["height"],
                        "bboxes": [
                            {"object": name, "bbox": value["bbox"]}
                            for name, value in bboxes.items()
                        ],
                        "relations": _so101_relations(
                            repository,
                            task,
                            SO101_EPISODE,
                            frame,
                            camera,
                            bboxes,
                        ),
                    }
                )
            sequences.append(
                {
                    "id": f"{task}:{SO101_EPISODE}:{camera}",
                    "task": task,
                    "task_label": task_item["name"],
                    "sequence": SO101_EPISODE,
                    "sequence_label": "episode 0",
                    "camera": camera,
                    "camera_label": CAMERA_LABELS[camera],
                    "frames": output_frames,
                }
            )

    bundle = _bundle(
        "so101",
        "SO101 2D Proxy GT",
        "agent_view",
        sequences,
        sample_count,
        SO101_EPISODE,
    )
    bundle["omitted_sequences"] = omitted
    return bundle


def _bundle(
    dataset: str,
    graph_label: str,
    default_camera: str,
    sequences: list[dict[str, Any]],
    sample_count: int,
    default_sequence: str,
) -> dict[str, Any]:
    sequences.sort(key=lambda item: (item["camera"] != default_camera, item["task"]))
    default_task = next(
        (item["task"] for item in sequences if item["camera"] == default_camera),
        sequences[0]["task"] if sequences else "",
    )
    return {
        "schema_version": 2,
        "dataset": dataset,
        "graph_label": graph_label,
        "default_camera": default_camera,
        "default_task": default_task,
        "default_sequence": default_sequence,
        "sampling": {"method": "uniform", "maximum_frames_per_sequence": sample_count},
        "sequence_count": len(sequences),
        "frame_count": sum(len(item["frames"]) for item in sequences),
        "sequences": sequences,
    }


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build(
    config_path: str | Path | None = None,
    libero_input: str | Path | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, Any]:
    input_dir = Path(libero_input or VLM_ROOT / "data" / "libero_spatial_v5").resolve()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    libero = _build_libero(input_dir, sample_count)
    so101 = _build_so101(config_path, sample_count)
    if so101["omitted_sequences"]:
        omitted = ", ".join(
            f"{item['task']}/{item['camera']}"
            for item in so101["omitted_sequences"]
        )
        raise RuntimeError(
            "SO101 episode_0 bbox coverage is incomplete for: "
            f"{omitted}. Run wrist detection for episode_0 before rebuilding."
        )
    _write_json(DATA_ROOT / "libero.json", libero)
    _write_json(DATA_ROOT / "so101.json", so101)
    manifest = {
        "schema_version": 1,
        "description": "Compressed LIBERO demo_0 and SO101 episode_0 mini dataset for GitHub Pages",
        "image_format": "WebP",
        "image_quality": WEBP_QUALITY,
        "selection": {
            "libero_demos": list(LIBERO_DEMOS),
            "so101_episodes": [SO101_EPISODE],
            "maximum_frames_per_sequence": sample_count,
        },
        "bundles": {
            name: {
                "file": f"{name}.json",
                "sequence_count": bundle["sequence_count"],
                "frame_count": bundle["frame_count"],
            }
            for name, bundle in (("libero", libero), ("so101", so101))
        },
    }
    manifest["total_bytes"] = 0
    for _ in range(4):
        _write_json(DATA_ROOT / "manifest.json", manifest)
        total_bytes = _tree_size(DATA_ROOT)
        if manifest["total_bytes"] == total_bytes:
            break
        manifest["total_bytes"] = total_bytes
    print(
        f"Wrote {libero['sequence_count']} LIBERO sequences / "
        f"{libero['frame_count']} frames"
    )
    print(
        f"Wrote {so101['sequence_count']} SO101 sequences / "
        f"{so101['frame_count']} frames"
    )
    print(f"Compressed mini dataset: {manifest['total_bytes'] / (1024 * 1024):.2f} MiB")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the compressed LIBERO demo_0 and SO101 episode_0 dataset."
    )
    parser.add_argument("--so101-config", default=None)
    parser.add_argument("--libero-input", default=None)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Maximum uniformly sampled frames per task/camera sequence (0 keeps all).",
    )
    args = parser.parse_args()
    build(args.so101_config, args.libero_input, args.sample_count)


if __name__ == "__main__":
    main()
