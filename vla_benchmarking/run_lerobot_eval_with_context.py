"""Run lerobot-eval with live LIBERO semantic prompt augmentation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import lerobot.scripts.lerobot_eval as lerobot_eval
from lerobot.envs import libero as lerobot_libero

from bddl_utils import extract_joint_schema, make_filtered_bddl, project_init_states_by_joint_name
from config import (
    RANDOMIZE_SCENES, TASK_SWAP_CONFIG, SETTLE_STEPS_SWAP, SCENE_GRAPH_SUBJECT_FILTER,
    LEROBOT_CAMERA_KEYS,
    TASK_GOAL_OBJECT_CONFIG, TASK_REMOVE_CONFIG, TASK_PROMPT_OVERRIDE,
    task_randomization_dimensions,
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


class RandomizationAuditLogger:
    """Write one JSONL record for each realized randomization observation."""

    def __init__(self, output_dir):
        self._fh = None
        self._records: dict[tuple[int, int, int], dict] = {}
        if output_dir:
            path = Path(output_dir) / "randomization_audit.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8", buffering=1)

    def log(self, *, task_id, dimensions_enabled, dimensions_realized, details=None,
            env_index=0, reset_sequence=0):
        key = (int(task_id), int(env_index), int(reset_sequence))
        record = self._records.setdefault(key, {
            "task_id": int(task_id),
            "env_index": int(env_index),
            "reset_sequence": int(reset_sequence),
            "dimensions_enabled": {},
            "dimensions_realized": {},
            "details": {},
            "status": "pending",
        })
        record["dimensions_enabled"].update(dimensions_enabled or {})
        record["dimensions_realized"].update(dimensions_realized or {})
        if details:
            record["details"].update(details)
        if details and details.get("status"):
            record["status"] = details["status"]

    def update_prompt(self, *, task_id, env_index, reset_sequence, enabled, realized, details):
        self.log(
            task_id=task_id,
            env_index=env_index,
            reset_sequence=reset_sequence,
            dimensions_enabled=enabled,
            dimensions_realized=realized,
            details=details,
        )
        key = (int(task_id), int(env_index), int(reset_sequence))
        record = self._records[key]
        complete = (
            set(record["dimensions_enabled"]) == set(enabled)
            and set(record["dimensions_realized"]) == set(enabled)
            and all(
                bool(record["dimensions_realized"].get(name)) == bool(is_enabled)
                for name, is_enabled in enabled.items()
            )
        )
        record["status"] = "ok" if complete else "failed_prompt_realization"
        record["details"]["prompt_status"] = "ok" if complete else "failed"

    def close(self):
        pending = [key for key, record in self._records.items() if record.get("status") != "ok"]
        if pending:
            raise RuntimeError(f"randomization audit has unfinished reset records: {pending}")
        if self._fh is not None:
            for key in sorted(self._records):
                self._fh.write(json.dumps(self._records[key], ensure_ascii=False) + "\n")
            self._fh.close()
            self._fh = None


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
        randomization_audit_logger=None,
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
        self.randomization_audit_logger = randomization_audit_logger
        self._debug_printed = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _audit_randomization(self, prompts, canonical_task_texts, effective_task_texts, sub_envs):
        if self.randomization_audit_logger is None:
            return
        for prompt, canonical_task_text, effective_task_text, sub_env in zip(
            prompts, canonical_task_texts, effective_task_texts, sub_envs
        ):
            task_id = getattr(sub_env, "task_id", None)
            reset_sequence = int(getattr(sub_env, "_randomization_reset_sequence", 0))
            if reset_sequence <= 0:
                raise RuntimeError(f"task description observed before randomized reset for task {task_id}")
            enabled = (
                task_randomization_dimensions(task_id)
                if task_id is not None
                else {}
            )
            realized = {
                "prompt_variant": bool(
                    enabled.get("prompt_variant")
                    and TASK_PROMPT_OVERRIDE.get(task_id) == effective_task_text
                    and canonical_task_text != TASK_PROMPT_OVERRIDE.get(task_id)
                ),
            }
            self.randomization_audit_logger.update_prompt(
                task_id=task_id,
                env_index=getattr(sub_env, "_randomization_env_index", 0),
                reset_sequence=reset_sequence,
                enabled=enabled,
                realized=realized,
                details={
                    "effective_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                },
            )

    def call(self, name, *args, **kwargs):
        result = self.env.call(name, *args, **kwargs)

        if name != "task_description":
            return result

        sub_envs = _resolve_envs(self.env)

        # Apply per-task prompt override before semantic context suffix
        canonical_result = list(result)
        effective_task_texts = [
            TASK_PROMPT_OVERRIDE.get(sub_env.task_id, task)
            for task, sub_env in zip(result, sub_envs)
        ]
        final_result = []
        for task, sub_env in zip(effective_task_texts, sub_envs):
            task_id = getattr(sub_env, "task_id", None)
            suffix = self.prompt_suffix_by_task.get(task_id, self.prompt_suffix).strip()
            final_result.append(f"{task} {suffix}" if suffix else task)
        result = final_result

        if self.live_generator is None:
            self._audit_randomization(result, canonical_result, effective_task_texts, sub_envs)
            if self.audit_logger is not None:
                for task, task_text, sub_env in zip(result, effective_task_texts, sub_envs):
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

        self._audit_randomization(final, canonical_result, effective_task_texts, sub_envs)

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
        # Keep the historical 10-step override unless the sealed LoRA evaluator
        # explicitly asks the policy checkpoint to supply its own value.
        (("--policy.n_action_steps",), os.environ.get("N_ACTION_STEPS", "10")),
        (("--env.type",), os.environ.get("ENV_TYPE", "libero")),
        (("--env.task",), os.environ.get("ENV_TASK", "libero_spatial")),
        (("--env.task_ids",), os.environ.get("TASK_IDS", "[0,1,2,3,4,5,6,7,8,9]")),
        (("--env.camera_name",), ",".join(LEROBOT_CAMERA_KEYS)),
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
        if names == ("--policy.n_action_steps",) and str(value).strip().lower() == "checkpoint":
            continue
        if not _has_cli_option(*names):
            sys.argv.append(f"{names[0]}={value}")


def _camera_name_mapping(camera_name: str) -> dict[str, str]:
    """Map fixed LeRobot camera names to its ordered image observation keys."""
    cameras = [item.strip() for item in camera_name.split(",") if item.strip()]
    suffixes = [""] + [str(index + 2) for index in range(len(cameras) - 1)]
    return {
        camera: f"image{suffix}"
        for camera, suffix in zip(cameras, suffixes)
    }


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

        self._canonical_task_bddl_file = self._task_bddl_file
        self._task_randomization_removed_objects = tuple(remove_objects)
        self._task_bddl_file = _filtered_bddl_path(self._task_bddl_file, remove_objects)

    patched_init._task_remove_config_patched = True
    lerobot_libero.LiberoEnv.__init__ = patched_init


def _patch_libero_env_camera_creation() -> None:
    """Pass LeRobot's configured fixed camera pair to OffScreenRenderEnv.

    Upstream LiberoEnv stores self.camera_name, but _ensure_env() creates OffScreenRenderEnv
    without passing camera_names. Passing the configured names keeps the observation keys
    aligned with LeRobot's fixed training-camera contract.
    """

    ensure_env_fn = lerobot_libero.LiberoEnv._ensure_env
    if getattr(ensure_env_fn, "_camera_creation_patched", False):
        return

    def patched_ensure_env(self) -> None:
        if self._env is not None:
            return

        configured_cameras = self.camera_name
        if isinstance(configured_cameras, str):
            configured_cameras = [item.strip() for item in configured_cameras.split(",") if item.strip()]
        camera_names = [camera.removesuffix("_image") for camera in configured_cameras]
        canonical_path = getattr(self, "_canonical_task_bddl_file", None)
        projection_evidence = {
            "required": bool(canonical_path),
            "success": True,
            "projected": False,
            "removed_objects": list(getattr(self, "_task_randomization_removed_objects", ())),
        }
        canonical_env = None
        if canonical_path:
            canonical_env = lerobot_libero.OffScreenRenderEnv(
                bddl_file_name=canonical_path,
                camera_names=camera_names,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
            )
            canonical_env.reset()
        try:
            env = lerobot_libero.OffScreenRenderEnv(
                bddl_file_name=self._task_bddl_file,
                camera_names=camera_names,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
            )
            env.reset()
        except Exception:
            if canonical_env is not None:
                close = getattr(canonical_env, "close", None)
                if close is not None:
                    close()
            raise
        if canonical_env is not None:
            try:
                source_schema = extract_joint_schema(canonical_env.sim.model)
                target_schema = extract_joint_schema(env.sim.model)
                states = getattr(self, "_init_states", None)
                if states is not None:
                    projected = project_init_states_by_joint_name(
                        np.asarray(states), source_schema, target_schema
                    )
                    expected_width = 1 + target_schema.nq + target_schema.nv
                    if projected.shape[-1] != expected_width:
                        raise ValueError(
                            f"projected init-state width {projected.shape[-1]} does not match "
                            f"filtered schema width {expected_width}"
                        )
                    self._init_states = projected
                    projection_evidence["projected"] = True
                    projection_evidence["state_count"] = int(projected.shape[0]) if projected.ndim > 1 else 1
                projection_evidence.update({
                    "canonical_nq": source_schema.nq,
                    "filtered_nq": target_schema.nq,
                    "canonical_nv": source_schema.nv,
                    "filtered_nv": target_schema.nv,
                })
            except Exception as exc:
                projection_evidence.update({"success": False, "error": str(exc)})
                raise
            finally:
                close = getattr(canonical_env, "close", None)
                if close is not None:
                    close()
        self._env = env
        self._init_state_projection_evidence = projection_evidence

    patched_ensure_env._camera_creation_patched = True
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
    randomization_audit_logger,
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
                    wrapped,
                    task_id,
                    TASK_SWAP_CONFIG,
                    SETTLE_STEPS_SWAP,
                    audit_logger=randomization_audit_logger,
                    removal_config=TASK_REMOVE_CONFIG,
                )
            if visual_condition:
                wrapped = VisualGraphVecEnvWrapper(
                    wrapped,
                    live_generator,
                    condition=visual_condition,
                    goal_object=visual_goal_objects,
                    audit_logger=visual_audit_logger,
                    line_width=int(os.environ.get("VISUAL_ARROW_WIDTH", "1")),
                    head_length=int(os.environ.get("VISUAL_ARROW_HEAD_LENGTH", "8")),
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
                randomization_audit_logger,
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
    if visual_condition == VISUAL_GOAL_ARROW_CONDITION and not _env_flag(
        "DISABLE_VISUAL_PROMPT_HINT"
    ):
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
    randomization_audit_logger = RandomizationAuditLogger(output_dir)

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
        # All tasks intentionally share the training camera pair.  Keep one
        # make_env call so hybrid scene perturbations remain task-config driven.
        result = original_make_env(*args, **kwargs)

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
            randomization_audit_logger,
        )

    lerobot_eval.make_env = make_env_patched

    try:
        lerobot_eval.main()
    finally:
        audit_logger.close()
        visual_audit_logger.close()
        randomization_audit_logger.close()


if __name__ == "__main__":
    main()
