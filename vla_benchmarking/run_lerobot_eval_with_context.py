"""Run lerobot-eval with live LIBERO semantic prompt augmentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import lerobot.scripts.lerobot_eval as lerobot_eval

from libero_live_semantic_context import LiveSemanticContextGenerator


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

        if name != "task_description" or self.live_generator is None:
            return result

        suffixes = [
            self.live_generator.prompt_suffix(sub_env, self.context_mode)
            for sub_env in self.env.envs
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
    return os.environ.get(name, "").strip().lower() in {"1", "true", "True"}


# def _clip(text: str, limit: int) -> str:
#     if len(text) <= limit:
#         return text
#     return text[:limit] + "...<truncated>"


def _append_default_lerobot_args() -> None:
    """Allow this wrapper to be run directly, not only via the matrix shell script."""
    script_dir = Path(__file__).resolve().parent

    mode = os.environ.get("CONTEXT_MODE", "standard").strip().lower()
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    policy_path = os.environ.get("MODELS", "lerobot/pi0_libero_base")
    policy_tag = policy_path.split('/')[-1]

    defaults = [
        (("--policy.path",), policy_path),
        (("--policy.device",), os.environ.get("DEVICE", "cuda")),
        (("--policy.n_action_steps",), 10),
        (("--env.type",), os.environ.get("ENV_TYPE", "libero")),
        (("--env.task",), os.environ.get("ENV_TASK", "libero_spatial")),
        (("--env.task_ids",), os.environ.get("TASK_IDS", "[3]")),
        (("--env.max_parallel_tasks",), os.environ.get("MAX_PARALLEL_TASKS", "1")),
        (("--eval.n_episodes",), os.environ.get("N_EPISODES", "1")),
        (("--eval.batch_size",), os.environ.get("BATCH_SIZE", "1")),
        (("--output_dir",), str(script_dir / "lerobot_eval_outputs" / mode / policy_tag / time_str)),
        (("--seed",), os.environ.get("SEED", "1000")),
    ]

    for names, value in defaults:
        if not _has_cli_option(*names):
            sys.argv.append(f"{names[0]}={value}")


def main() -> None:
    _append_default_lerobot_args()

    context_mode = os.environ.get("CONTEXT_MODE").strip().lower()
    use_live_context = _is_augmented_mode()
    debug_semantic_context = 1  # _env_flag("DEBUG_SEMANTIC_CONTEXT")
    debug_max_chars = int(os.environ.get("DEBUG_SEMANTIC_MAX_CHARS", "4000"))

    live_generator = None
    if use_live_context:
        live_generator = LiveSemanticContextGenerator()

    # -------------------------
    # WRAP VECTOR ENVS ONLY
    # -------------------------

    def make_env_patched(*args, **kwargs):
        args = list(args)
        env_instance = args[0]
        env_instance.render_mode = None
        args[0] = env_instance

        result = lerobot_eval.make_env(*args, **kwargs)

        def wrap(obj):
            if hasattr(obj, "call") and hasattr(obj, "envs"):
                return TaskContextVecEnv(
                    obj,
                    live_generator,
                    context_mode,
                    debug_semantic_context,
                    debug_max_chars,
                )

            if isinstance(obj, dict):
                for k, v in obj.items():
                    obj[k] = wrap(v)

            return obj

        return wrap(result)

    lerobot_eval.make_env = make_env_patched

    lerobot_eval.main()


if __name__ == "__main__":
    main()
