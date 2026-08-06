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
    TASK_GOAL_OBJECT_CONFIG, TASK_REMOVE_CONFIG, TASK_PROMPT_OVERRIDE, TASK_CAMERA_OVERRIDE,
)
from libero_live_semantic_context import LiveSemanticContextGenerator
from prompt_audit import PromptAuditLogger
from radomize_scenes import SceneRandomizerVecEnvWrapper, _resolve_envs
from scene_graph_formats import LEGACY_FORMAT, normalize_context_format
from visual_scene_graph import (
    DEFAULT_GOAL_OBJECT,
    SUPPORTED_VISUAL_CONDITIONS,
    VISUAL_GOAL_ARROW_CONDITION,
    VisualGraphVecEnvWrapper,
    VisualRelationAuditLogger,
    goal_arrow_prompt_hint,
    resolve_task_goal_object,
)

_FILTERED_BDDL_CACHE: dict[tuple[str, tuple[str, ...]], str] = {}


class TaskContextVecEnv:
    def __init__(
        self,
        env,
        live_generator,
        context_mode,
        context_format=LEGACY_FORMAT,
        debug=False,
        max_chars=4000,
        audit_logger=None,
        prompt_suffix="",
        prompt_suffix_by_task=None,
    ):
        self.env = env
        self.live_generator = live_generator
        self.context_mode = context_mode
        self.context_format = context_format
        self.debug = debug
        self.max_chars = max_chars
        self.audit_logger = audit_logger
        self.prompt_suffix = prompt_suffix.strip()
        self.prompt_suffix_by_task = prompt_suffix_by_task or {}
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
        task_texts = result
        final_result = []
        for task, sub_env in zip(result, sub_envs):
            task_id = getattr(sub_env, "task_id", None)
            suffix = self.prompt_suffix_by_task.get(task_id, self.prompt_suffix).strip()
            final_result.append(f"{task} {suffix}" if suffix else task)
        result = final_result

        if self.live_generator is None:
            if self.audit_logger is not None:
                for task, task_text, sub_env in zip(result, task_texts, sub_envs):
                    self.audit_logger.log(
                        prompt=task,
                        task_text=task_text,
                        context_mode=self.context_mode,
                        context_format=self.context_format,
                        task_id=getattr(sub_env, "task_id", None),
                        env_step=getattr(sub_env, "_elapsed_steps", None),
                        relations_generated=0,
                        relations_retained=0,
                    )
            return result

        final = []
        for task, sub_env in zip(result, sub_envs):
            prompt, relations, retained_count = self.live_generator.build_prompt(
                sub_env,
                task,
                self.context_mode,
                self.context_format,
            )
            final.append(prompt)
            if self.audit_logger is not None:
                self.audit_logger.log(
                    prompt=prompt,
                    task_text=task,
                    context_mode=self.context_mode,
                    context_format=self.context_format,
                    task_id=getattr(sub_env, "task_id", None),
                    env_step=getattr(sub_env, "_elapsed_steps", None),
                    relations_generated=len(relations),
                    relations_retained=retained_count,
                )

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


def _visual_condition() -> str | None:
    condition = os.environ.get("VISUAL_CONDITION", "").strip().lower()
    if not condition:
        return "visual_arrows" if _env_flag("VISUAL_ARROWS") else None
    if condition == "none":
        return None
    if condition not in SUPPORTED_VISUAL_CONDITIONS:
        options = ", ".join(["none", *sorted(SUPPORTED_VISUAL_CONDITIONS)])
        raise SystemExit(f"ERROR: VISUAL_CONDITION must be one of: {options}.")
    return condition


def _has_cli_option(*names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in sys.argv[1:])


def _cli_option_value(name: str) -> str | None:
    args = sys.argv[1:]
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "True"}


def _append_default_lerobot_args() -> None:
    """Allow this wrapper to be run directly, not only via the matrix shell script."""
    script_dir = Path(__file__).resolve().parent

    mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    policy_path = os.environ.get("MODELS", "lerobot/pi0_libero_base")
    policy_tag = policy_path.split('/')[-1]
    seed = os.environ.get("SEED", "1000")
    output_root = Path(os.environ.get("OUTPUT_DIR", "lerobot_eval_outputs").strip())
    if not output_root.is_absolute():
        output_root = script_dir / output_root

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
        (("--output_dir",), (output_root / mode / policy_tag / seed / time_str).as_posix()),
        (("--seed",), seed),
    ]
    rename_map = os.environ.get("RENAME_MAP")
    if rename_map:
        defaults.append((("--rename_map",), rename_map))

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


def _wrap_task_vec_envs(
    result,
    live_generator,
    prompt_live_generator,
    context_mode,
    context_format,
    debug_semantic_context,
    debug_max_chars,
    audit_logger,
    visual_condition,
    visual_audit_logger,
    visual_goal_objects,
    visual_prompt_suffix,
    visual_prompt_suffix_by_task,
):
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
            if visual_condition:
                wrapped = VisualGraphVecEnvWrapper(
                    wrapped,
                    live_generator,
                    condition=visual_condition,
                    goal_object=visual_goal_objects,
                    audit_logger=visual_audit_logger,
                )
            wrapped = TaskContextVecEnv(
                wrapped,
                prompt_live_generator,
                context_mode,
                context_format,
                debug_semantic_context,
                debug_max_chars,
                audit_logger,
                visual_prompt_suffix,
                visual_prompt_suffix_by_task,
            )
            suite_map[task_id] = wrapped

    return result


def _patch_max_episodes_rendered() -> None:
    value = os.environ.get("MAX_EPISODES_RENDERED")
    if not value:
        return
    try:
        max_episodes_rendered = int(value)
    except ValueError as exc:
        raise SystemExit("ERROR: MAX_EPISODES_RENDERED must be an integer.") from exc

    eval_policy_all_fn = getattr(lerobot_eval, "eval_policy_all", None)
    if eval_policy_all_fn is None or getattr(eval_policy_all_fn, "_max_videos_patched", False):
        return

    original_eval_policy_all = eval_policy_all_fn

    def patched_eval_policy_all(*args, **kwargs):
        kwargs["max_episodes_rendered"] = max_episodes_rendered
        return original_eval_policy_all(*args, **kwargs)

    patched_eval_policy_all._max_videos_patched = True
    lerobot_eval.eval_policy_all = patched_eval_policy_all


def main() -> None:
    _append_default_lerobot_args()

    context_mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    context_format = normalize_context_format(os.environ.get("CONTEXT_FORMAT", LEGACY_FORMAT))
    use_live_context = _is_augmented_mode()
    visual_condition = _visual_condition()
    visual_goal_override = os.environ.get("VISUAL_GOAL_OBJECT", "").strip()
    visual_goal_objects = (
        {task_id: visual_goal_override for task_id in TASK_GOAL_OBJECT_CONFIG}
        if visual_goal_override
        else dict(TASK_GOAL_OBJECT_CONFIG)
    )
    visual_prompt_suffix = ""
    visual_prompt_suffix_by_task = {}
    if visual_condition == VISUAL_GOAL_ARROW_CONDITION:
        explicit_hint = os.environ.get("VISUAL_PROMPT_HINT", "").strip()
        if explicit_hint:
            visual_prompt_suffix = explicit_hint
        else:
            visual_prompt_suffix_by_task = {
                task_id: goal_arrow_prompt_hint(
                    resolve_task_goal_object(task_id, visual_goal_objects)
                )
                for task_id in TASK_GOAL_OBJECT_CONFIG
            }
    debug_semantic_context = 1  # _env_flag("DEBUG_SEMANTIC_CONTEXT")
    debug_max_chars = int(os.environ.get("DEBUG_SEMANTIC_MAX_CHARS", "4000"))
    output_dir = _cli_option_value("--output_dir")
    audit_logger = PromptAuditLogger(
        output_dir,
        model_id=os.environ.get("MODELS", "lerobot/pi0_libero_base"),
    )
    visual_audit_logger = VisualRelationAuditLogger(
        output_dir,
        enabled=visual_condition is not None,
    )

    live_generator = None
    if use_live_context or visual_condition is not None:
        live_generator = LiveSemanticContextGenerator()
        live_generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER
    prompt_live_generator = live_generator if use_live_context else None

    if TASK_REMOVE_CONFIG:
        _patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)

    _patch_libero_env_camera_creation()
    _patch_max_episodes_rendered()

    # -------------------------
    # WRAP VECTOR ENVS ONLY
    # -------------------------

    original_make_env = lerobot_eval.make_env

    def make_env_patched(*args, **kwargs):
        args = list(args)
        env_instance = args[0]
        render_mode = os.environ.get("RENDER_MODE")
        if render_mode is not None:
            env_instance.render_mode = None if render_mode.lower() == "none" else render_mode
        else:
            env_instance.render_mode = None
        task_groups = _task_groups_by_camera(env_instance)

        results = []
        for camera_name, group_task_ids in task_groups:
            local_args = list(args)
            local_args[0] = _clone_env_instance_for_tasks(env_instance, group_task_ids, camera_name)
            results.append(original_make_env(*local_args, **kwargs))

        result = _merge_env_results(results)

        return _wrap_task_vec_envs(
            result,
            live_generator,
            prompt_live_generator,
            context_mode,
            context_format,
            debug_semantic_context,
            debug_max_chars,
            audit_logger,
            visual_condition,
            visual_audit_logger,
            visual_goal_objects,
            visual_prompt_suffix,
            visual_prompt_suffix_by_task,
        )

    lerobot_eval.make_env = make_env_patched

    try:
        lerobot_eval.main()
    finally:
        audit_logger.close()
        visual_audit_logger.close()


if __name__ == "__main__":
    main()
