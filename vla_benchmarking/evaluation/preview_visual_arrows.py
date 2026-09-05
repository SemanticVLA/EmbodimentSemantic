#!/usr/bin/env python3
"""Render the first policy frame with live scene-graph arrows for each task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support direct invocation from a clean checkout before optional simulator
# dependencies are imported.
if __package__ in {None, ""}:  # pragma: no cover - direct script smoke
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

os.environ.setdefault("MUJOCO_GL", "wgl")

from PIL import Image, ImageDraw
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

from vla_benchmarking.shared.config import (
    BENCHMARK_NAME,
    LEROBOT_CAMERA_KEYS,
    SCENE_GRAPH_SUBJECT_FILTER,
    SETTLE_STEPS_SWAP,
    TASK_REMOVE_CONFIG,
    TASK_SWAP_CONFIG,
)
from vla_benchmarking.evaluation.libero_live_semantic_context import LiveSemanticContextGenerator
from vla_benchmarking.evaluation.randomize_scenes import SceneRandomizerVecEnvWrapper, swap_objects
from vla_benchmarking.evaluation.run_lerobot_eval_with_context import (
    _camera_name_mapping,
    _patch_libero_env_bddl_selection,
    _patch_libero_env_camera_creation,
)
from vla_benchmarking.evaluation.visual_scene_graph import draw_scene_graph_arrows, drawable_relations


def _parse_tasks(value: str) -> list[int]:
    cleaned = value.strip()
    if cleaned.startswith("["):
        return [int(item) for item in json.loads(cleaned)]
    return [int(item.strip()) for item in cleaned.split(",") if item.strip()]


def _apply_task_swaps(inner_env, task_id: int) -> None:
    """Apply the configured distractor swaps with production protections."""
    protected = SceneRandomizerVecEnvWrapper._snapshot_protected(inner_env)
    configured = TASK_SWAP_CONFIG.get(task_id, [])
    results = []
    for obj_a, obj_b in configured:
        if not isinstance(obj_a, str) or not isinstance(obj_b, str):
            raise TypeError(
                "TASK_SWAP_CONFIG entries must be string object-name pairs; "
                f"got ({obj_a!r}, {obj_b!r})"
            )
        results.append(
            swap_objects(
                inner_env,
                obj_a,
                obj_b,
                settle_steps=SETTLE_STEPS_SWAP,
                verbose=False,
            )
        )

    expected_labels = [label for swap in configured for label in swap]
    applied_labels = [label for result in results for label in result.get("applied", [])]
    skipped = [item for result in results for item in result.get("skipped", [])]
    if skipped or sorted(applied_labels) != sorted(expected_labels):
        raise RuntimeError(
            f"task {task_id} layout was not fully applied: "
            f"expected={expected_labels}, applied={applied_labels}, skipped={skipped}"
        )

    SceneRandomizerVecEnvWrapper._restore_protected(inner_env, protected)
    inner_env.sim.forward()
    SceneRandomizerVecEnvWrapper._verify_protected(inner_env, protected)


def _policy_cameras(task_id: int) -> str:
    del task_id
    return ",".join(LEROBOT_CAMERA_KEYS)


def _make_contact_sheet(
    previews: list[tuple[int, str, int, Image.Image]],
    output_path: Path,
) -> None:
    columns = min(5, max(1, len(previews)))
    rows = (len(previews) + columns - 1) // columns
    image_width, image_height = previews[0][3].size
    title_height = 28
    sheet = Image.new(
        "RGB",
        (columns * image_width, rows * (image_height + title_height)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (task_id, camera, arrow_count, image) in enumerate(previews):
        row, column = divmod(index, columns)
        x = column * image_width
        y = row * (image_height + title_height)
        draw.text(
            (x + 6, y + 7),
            f"Task {task_id} | {camera} | {arrow_count} arrows",
            fill=(20, 20, 20),
        )
        sheet.paste(image, (x, y + title_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create arrow-overlaid first-frame previews for LIBERO Spatial tasks."
    )
    parser.add_argument("--tasks", default="[0,1,2,3,4,5,6,7,8,9]")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "visual_arrow_previews"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_ids = _parse_tasks(args.tasks)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if TASK_REMOVE_CONFIG:
        _patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)
    _patch_libero_env_camera_creation()

    task_suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    generator = LiveSemanticContextGenerator()
    generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
    previews: list[tuple[int, str, int, Image.Image]] = []
    audit_rows = []

    for task_id in task_ids:
        cameras = _policy_cameras(task_id)
        mapping = _camera_name_mapping(cameras)
        env = LiberoEnv(
            task_suite=task_suite,
            task_id=task_id,
            task_suite_name=BENCHMARK_NAME,
            camera_name=cameras,
            camera_name_mapping=mapping,
            init_states=True,
            episode_index=0,
            n_envs=1,
        )
        try:
            observation, _ = env.reset(seed=args.seed)
            _apply_task_swaps(env._env, task_id)
            env._env.sim.forward()
            raw_obs = env._env.env._get_observations(force_update=True)
            observation = env._format_raw_obs(raw_obs)

            camera = next(
                raw_camera.removesuffix("_image")
                for raw_camera, image_key in mapping.items()
                if image_key == "image"
            )
            raw_image = observation["pixels"]["image"]
            context = generator.observe_visual_graph(env, camera=camera)
            drawn, skipped = drawable_relations(context["bboxes"], context["relations"])
            overlaid = draw_scene_graph_arrows(
                raw_image,
                context["bboxes"],
                context["relations"],
            )

            raw_path = output_dir / f"task_{task_id:02d}_raw.png"
            arrow_path = output_dir / f"task_{task_id:02d}_visual_arrows.png"
            Image.fromarray(raw_image).save(raw_path)
            preview_image = Image.fromarray(overlaid)
            preview_image.save(arrow_path)
            previews.append((task_id, camera, len(drawn), preview_image))
            audit_rows.append(
                {
                    "task_id": task_id,
                    "camera": camera,
                    "image_slot": "pixels/image",
                    "raw_path": raw_path.as_posix(),
                    "arrow_path": arrow_path.as_posix(),
                    "bboxes": context["bboxes"],
                    "relations": [list(item) for item in context["relations"]],
                    "drawn_relations": [list(item) for item in drawn],
                    "skipped_relations_missing_bbox": [list(item) for item in skipped],
                }
            )
            print(
                f"Task {task_id}: {camera}, {len(drawn)} arrows -> {arrow_path}",
                flush=True,
            )
        finally:
            env.close()

    if previews:
        contact_sheet = output_dir / "visual_arrows_first_frames.png"
        _make_contact_sheet(previews, contact_sheet)
        print(f"Contact sheet: {contact_sheet}")

    with (output_dir / "preview_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit_rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
