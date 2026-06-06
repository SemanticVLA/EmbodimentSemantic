"""Run lerobot-eval with live LIBERO semantic prompt augmentation."""

from __future__ import annotations

import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import lerobot.scripts.lerobot_eval as lerobot_eval
from lerobot.envs import libero as lerobot_libero

from bddl_utils import make_filtered_bddl
from config import (
    RANDOMIZE_SCENES, TASK_SWAP_CONFIG, SETTLE_STEPS_SWAP, SCENE_GRAPH_SUBJECT_FILTER,
    TASK_REMOVE_CONFIG, TASK_PROMPT_OVERRIDE, TASK_CAMERA_OVERRIDE,
)
from libero_live_semantic_context import LiveSemanticContextGenerator
from radomize_scenes import SceneRandomizerVecEnvWrapper, _resolve_envs

_FILTERED_BDDL_CACHE: dict[tuple[str, tuple[str, ...]], str] = {}


class TaskContextVecEnv:
    def __init__(self, env, live_generator, context_mode, debug=False, max_chars=4000):
        self.env = env
        self.live_generator = live_generator
        self.context_mode = context_mode
        self.debug = debug
        self.max_chars = max_chars
        self._debug_printed = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def call(self, name, *args, **kwargs):
        result = self.env.call(name, *args, **kwargs)

        if name != "task_description":
            return result

        sub_envs = _resolve_envs(self.env)

        # Apply per-task prompt override before semantic context suffix
        result = [
            TASK_PROMPT_OVERRIDE.get(sub_env.task_id, task)
            for task, sub_env in zip(result, sub_envs)
        ]

        if self.live_generator is None:
            return result

        suffixes = [
            self.live_generator.prompt_suffix(sub_env, self.context_mode)
            for sub_env in sub_envs
        ]

        final = [
            f"{task}{suffix}"
            for task, suffix in zip(result, suffixes)
        ]
        if self.debug and not self._debug_printed:
            self._debug_printed = True
            print("\n===== DEBUG_SEMANTIC_CONTEXT =====")
            print(final[0][:self.max_chars])
            print("===== END DEBUG_SEMANTIC_CONTEXT =====\n")

        return final


def _is_augmented_mode() -> bool:
    mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    valid_modes = {
        "standard",
        "scene_graph",
        "bounding_boxes",
        "scene_graph_bounding_boxes",
    }
    if mode not in valid_modes:
        raise SystemExit(
            "ERROR: CONTEXT_MODE must be one of: "
            "standard, scene_graph, bounding_boxes, scene_graph_bounding_boxes."
        )
    return mode != "standard"


def _has_cli_option(*names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in sys.argv[1:])


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true"}


def _append_default_lerobot_args() -> None:
    """Allow this wrapper to be run directly, not only via the matrix shell script."""
    script_dir = Path(__file__).resolve().parent

    mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    policy_path = os.environ.get("MODELS", "lerobot/pi0_libero_base")
    policy_tag = policy_path.split('/')[-1]
    seed = os.environ.get("SEED", "1000")
    out_put_dir = os.environ.get("OUTPUT_DIR", "lerobot_eval_outputs").strip()

    defaults = [
        (("--policy.path",), policy_path),
        (("--policy.device",), os.environ.get("DEVICE", "cuda")),
        (("--policy.n_action_steps",), 10),
        (("--env.type",), os.environ.get("ENV_TYPE", "libero")),
        (("--env.task",), os.environ.get("ENV_TASK", "libero_spatial")),
        (("--env.task_ids",), os.environ.get("TASK_IDS", "[0,1,2,3,4,5,6,7,8,9]")),
        (("--env.max_parallel_tasks",), os.environ.get("MAX_PARALLEL_TASKS", "1")),
        (("--eval.n_episodes",), os.environ.get("N_EPISODES", "1")),
        (("--eval.batch_size",), os.environ.get("BATCH_SIZE", "1")),
        (("--output_dir",), (script_dir / out_put_dir / mode / policy_tag / seed / time_str).as_posix()),
        (("--seed",), seed),
    ]
    # If the policy is smolvla, add a default rename_map to fix feature mismatch
    if policy_tag.lower() == "smolvla":
        rename_map = {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }
        defaults.append(
            (("--rename_map",), str(rename_map))
        )

    for names, value in defaults:
        if not _has_cli_option(*names):
            sys.argv.append(f"{names[0]}={value}")


def _normalize_task_ids(task_ids) -> list[int]:
    if task_ids is None:
        return []
    if isinstance(task_ids, (list, tuple, set)):
        return [int(task_id) for task_id in task_ids]
    return [int(task_ids)]


def _camera_name_mapping(camera_name: str) -> dict[str, str]:
    cam_list = [c.strip() for c in camera_name.split(",") if c.strip()]
    suffixes = [""] + [str(i + 2) for i in range(len(cam_list) - 1)]
    return {
        cam: f"image{suffix}"
        for cam, suffix in zip(cam_list, suffixes)
    }


def _normalize_lerobot_camera_name(camera_name: str | None) -> str | None:
    if camera_name is None:
        return None
    cam_list = [c.strip() for c in camera_name.split(",") if c.strip()]
    normalized = [
        cam if cam.endswith("_image") else f"{cam}_image"
        for cam in cam_list
    ]
    return ",".join(normalized)


def _task_groups_by_camera(env_instance) -> list[tuple[str | None, list[int]]]:
    task_ids = _normalize_task_ids(getattr(env_instance, "task_ids", None))
    if not task_ids:
        return [(_normalize_lerobot_camera_name(getattr(env_instance, "camera_name", None)), task_ids)]

    default_camera = _normalize_lerobot_camera_name(getattr(env_instance, "camera_name", None))
    grouped: dict[str | None, list[int]] = {}
    ordered_keys: list[str | None] = []

    for task_id in task_ids:
        camera_name = _normalize_lerobot_camera_name(
            TASK_CAMERA_OVERRIDE.get(task_id, default_camera)
        )
        if camera_name not in grouped:
            grouped[camera_name] = []
            ordered_keys.append(camera_name)
        grouped[camera_name].append(task_id)

    return [(camera_name, grouped[camera_name]) for camera_name in ordered_keys]


def _clone_env_instance_for_tasks(env_instance, task_ids: list[int], camera_name: str | None):
    cloned = copy.deepcopy(env_instance)
    cloned.task_ids = list(task_ids)
    cloned.camera_name = _normalize_lerobot_camera_name(getattr(cloned, "camera_name", None))

    if task_ids and task_ids[0] in TASK_CAMERA_OVERRIDE and camera_name:
        cloned.camera_name = camera_name
        cloned.camera_name_mapping = _camera_name_mapping(camera_name)

    return cloned


def _merge_env_results(results: list):
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    merged: dict = {}
    for result in results:
        if not isinstance(result, dict):
            raise TypeError(
                "Expected lerobot_eval.make_env to return a dict when splitting task-specific "
                "camera overrides across multiple task groups."
            )
        for suite_name, suite_map in result.items():
            if suite_name not in merged:
                merged[suite_name] = suite_map
                continue
            if not isinstance(merged[suite_name], dict) or not isinstance(suite_map, dict):
                raise TypeError(
                    f"Cannot merge make_env result for suite '{suite_name}': expected dict values."
                )
            overlap = set(merged[suite_name]) & set(suite_map)
            if overlap:
                raise ValueError(
                    f"Duplicate task ids while merging make_env results for suite '{suite_name}': "
                    f"{sorted(overlap)}"
                )
            merged[suite_name].update(suite_map)
    return merged


def _filtered_bddl_path(source_path: str, remove_objects: list[str]) -> str:
    key = (source_path, tuple(remove_objects))
    if key not in _FILTERED_BDDL_CACHE:
        _FILTERED_BDDL_CACHE[key] = make_filtered_bddl(source_path, remove_objects)
    return _FILTERED_BDDL_CACHE[key]


def _patch_libero_env_bddl_selection(remove_config: dict[int, list[str]]) -> None:
    """Patch LeRobot's LiberoEnv so task-specific BDDL filtering happens before first reset.

    LeRobot defers OffScreenRenderEnv construction until LiberoEnv.reset(), so changing
    self._task_bddl_file in LiberoEnv.__init__ is early enough for both SyncVectorEnv
    and AsyncVectorEnv code paths.
    """

    init_fn = lerobot_libero.LiberoEnv.__init__
    if getattr(init_fn, "_task_remove_config_patched", False):
        return

    original_init = init_fn

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        task_id = kwargs.get("task_id")
        if task_id is None and len(args) >= 2:
            task_id = args[1]
        if task_id is None:
            return

        remove_objects = remove_config.get(int(task_id))
        if not remove_objects:
            return

        self._task_bddl_file = _filtered_bddl_path(self._task_bddl_file, remove_objects)

    patched_init._task_remove_config_patched = True
    lerobot_libero.LiberoEnv.__init__ = patched_init


def _patch_libero_env_camera_creation() -> None:
    """Patch LeRobot's LiberoEnv so OffScreenRenderEnv is created with requested cameras.

    Upstream LiberoEnv stores self.camera_name, but _ensure_env() creates OffScreenRenderEnv
    without passing camera_names. That means raw_obs only contains LIBERO's default camera
    keys, and task-specific camera overrides fail in _format_raw_obs() with KeyError.
    """

    ensure_env_fn = lerobot_libero.LiberoEnv._ensure_env
    if getattr(ensure_env_fn, "_camera_override_patched", False):
        return

    def patched_ensure_env(self) -> None:
        if self._env is not None:
            return

        camera_names = [camera.removesuffix("_image") for camera in self.camera_name]
        env = lerobot_libero.OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_names=camera_names,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
        )
        env.reset()
        self._env = env

    patched_ensure_env._camera_override_patched = True
    lerobot_libero.LiberoEnv._ensure_env = patched_ensure_env


def _wrap_task_vec_envs(result, live_generator, context_mode, debug_semantic_context, debug_max_chars):
    if not isinstance(result, dict):
        return result

    for suite_map in result.values():
        if not isinstance(suite_map, dict):
            continue
        for task_id, vec_env in list(suite_map.items()):
            wrapped = vec_env
            if RANDOMIZE_SCENES:
                wrapped = SceneRandomizerVecEnvWrapper(
                    wrapped, task_id, TASK_SWAP_CONFIG, SETTLE_STEPS_SWAP
                )
            wrapped = TaskContextVecEnv(
                wrapped, live_generator, context_mode, debug_semantic_context, debug_max_chars
            )
            suite_map[task_id] = wrapped

    return result


def main() -> None:
    _append_default_lerobot_args()

    context_mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    use_live_context = _is_augmented_mode()
    debug_semantic_context = _env_flag("DEBUG_SEMANTIC_CONTEXT")
    debug_max_chars = int(os.environ.get("DEBUG_SEMANTIC_MAX_CHARS", "4000"))

    live_generator = None
    if use_live_context:
        live_generator = LiveSemanticContextGenerator()
        live_generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER

    if TASK_REMOVE_CONFIG:
        _patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)

    _patch_libero_env_camera_creation()

    # -------------------------
    # WRAP VECTOR ENVS ONLY
    # -------------------------

    original_make_env = lerobot_eval.make_env

    def make_env_patched(*args, **kwargs):
        args = list(args)
        env_instance = args[0]
        env_instance.render_mode = None
        task_groups = _task_groups_by_camera(env_instance)

        results = []
        for camera_name, group_task_ids in task_groups:
            local_args = list(args)
            local_args[0] = _clone_env_instance_for_tasks(env_instance, group_task_ids, camera_name)
            results.append(original_make_env(*local_args, **kwargs))

        result = _merge_env_results(results)

        return _wrap_task_vec_envs(
            result, live_generator, context_mode, debug_semantic_context, debug_max_chars
        )

    lerobot_eval.make_env = make_env_patched

    lerobot_eval.main()


if __name__ == "__main__":
    main()



