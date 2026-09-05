#!/usr/bin/env python3
"""Render paired high-resolution visual-arrow ablation previews."""

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

from PIL import Image
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

from vla_benchmarking.shared.config import (
    BENCHMARK_NAME,
    SCENE_GRAPH_SUBJECT_FILTER,
    TASK_GOAL_OBJECT_CONFIG,
    TASK_REMOVE_CONFIG,
)
from vla_benchmarking.evaluation.libero_live_semantic_context import LiveSemanticContextGenerator
from vla_benchmarking.evaluation.preview_visual_arrows import _apply_task_swaps, _policy_cameras
from vla_benchmarking.evaluation.run_lerobot_eval_with_context import (
    _camera_name_mapping,
    _patch_libero_env_bddl_selection,
    _patch_libero_env_camera_creation,
)
from vla_benchmarking.evaluation.visual_scene_graph import (
    VISUAL_ARROWS_CONDITION,
    VISUAL_GOAL_ARROW_CONDITION,
    draw_scene_graph_arrows,
    drawable_relations,
    select_visual_relations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all bowl arrows and bowl-to-target arrow for one LIBERO task."
    )
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "visual_arrow_previews_highres"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if TASK_REMOVE_CONFIG:
        _patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)
    _patch_libero_env_camera_creation()

    cameras = _policy_cameras(args.task)
    mapping = _camera_name_mapping(cameras)
    task_suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()
    generator = LiveSemanticContextGenerator()
    generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
    goal_object = TASK_GOAL_OBJECT_CONFIG.get(args.task, "plate_1")

    env = LiberoEnv(
        task_suite=task_suite,
        task_id=args.task,
        task_suite_name=BENCHMARK_NAME,
        camera_name=cameras,
        camera_name_mapping=mapping,
        observation_width=args.resolution,
        observation_height=args.resolution,
        visualization_width=args.resolution,
        visualization_height=args.resolution,
        init_states=True,
        episode_index=0,
        n_envs=1,
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        _apply_task_swaps(env._env, args.task)
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
        bboxes = context["bboxes"]
        source_relations = context["relations"]

        all_relations = select_visual_relations(
            bboxes,
            source_relations,
            condition=VISUAL_ARROWS_CONDITION,
            subject=SCENE_GRAPH_SUBJECT_FILTER,
            goal_object=goal_object,
        )
        goal_relations = select_visual_relations(
            bboxes,
            source_relations,
            condition=VISUAL_GOAL_ARROW_CONDITION,
            subject=SCENE_GRAPH_SUBJECT_FILTER,
            goal_object=goal_object,
        )

        scale = max(1, round(args.resolution / 256))
        all_image = draw_scene_graph_arrows(
            raw_image,
            bboxes,
            all_relations,
            line_width=scale,
            head_length=8 * scale,
        )
        goal_image = draw_scene_graph_arrows(
            raw_image,
            bboxes,
            goal_relations,
            line_width=scale,
            head_length=8 * scale,
        )

        stem = f"task_{args.task:02d}_{args.resolution}"
        raw_path = output_dir / f"{stem}_raw.png"
        all_path = output_dir / f"{stem}_all_bowl_arrows.png"
        goal_path = output_dir / f"{stem}_bowl_to_target.png"
        Image.fromarray(raw_image).save(raw_path)
        Image.fromarray(all_image).save(all_path)
        Image.fromarray(goal_image).save(goal_path)

        all_drawn, all_skipped = drawable_relations(bboxes, all_relations)
        goal_drawn, goal_skipped = drawable_relations(bboxes, goal_relations)
        audit = {
            "task_id": args.task,
            "seed": args.seed,
            "resolution": args.resolution,
            "camera": camera,
            "subject": SCENE_GRAPH_SUBJECT_FILTER,
            "goal_object": goal_object,
            "raw_path": raw_path.as_posix(),
            "all_arrows_path": all_path.as_posix(),
            "goal_arrow_path": goal_path.as_posix(),
            "visible_bboxes": sorted(bboxes),
            "source_relations": [list(item) for item in source_relations],
            "all_drawn_relations": [list(item) for item in all_drawn],
            "all_skipped_relations_missing_bbox": [list(item) for item in all_skipped],
            "goal_drawn_relations": [list(item) for item in goal_drawn],
            "goal_skipped_relations_missing_bbox": [list(item) for item in goal_skipped],
        }
        audit_path = output_dir / f"{stem}_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        print(f"All bowl arrows: {all_path}")
        print(f"Bowl to target : {goal_path}")
        print(f"Drawn arrows   : all={len(all_drawn)}, target={len(goal_drawn)}")
        print(f"Audit          : {audit_path}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
