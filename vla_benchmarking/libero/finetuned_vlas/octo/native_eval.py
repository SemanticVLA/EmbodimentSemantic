"""Checked-in native Octo/LIBERO evaluator.

This is the runtime bridge for the shared plan-aware evaluator.  It loads the
pinned Octo checkpoint through :class:`OctoPolicyAdapter`, constructs the
canonical LIBERO camera observation, and delegates action accounting to the
shared native rollout harness.  No fallback policy or caller-supplied action
statistics are accepted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adapter import OctoPolicyAdapter
from .config import COMMUNITY_EVAL_CONFIG, MATCHED_TRAIN_CONFIG
from .contracts import FRAME_ORIENTATION_LIBERO_CANONICAL, FRAME_ORIENTATION_STORED_RAW, rotate_stored_frame_180


def _plan_episodes(plan: Mapping[str, Any]) -> list[Any]:
    from vla_benchmarking.libero.evaluation.contracts import EvaluationCell, EvaluationCondition
    from vla_benchmarking.libero.evaluation.run_policy_eval import EpisodeSpec

    condition_payload = dict(plan["condition"])
    # v2 plans persist both the normalized visual input and the compatibility
    # boolean.  The normalized field is authoritative when rebuilding cells.
    condition_payload.pop("visual_arrow", None)
    condition = EvaluationCondition(**condition_payload)
    result = []
    for item in plan["schedule"]["cells"]:
        cell = EvaluationCell(
            cell_index=int(item["cell_index"]),
            task_id=int(item["task_id"]),
            episode_index=int(item["episode_index"]),
            seed=int(item["seed"]),
            init_state_index=int(item.get("init_state_index", item["episode_index"])),
            condition=condition,
        )
        result.append(EpisodeSpec(cell=cell, task_description=_task_description(cell.task_id, condition)))
    return result


def _task_description(task_id: int, condition: Any | None = None) -> str:
    from vla_benchmarking.libero.shared.config import TASK_PROMPT_OVERRIDE, TASK_NAMES

    # Prompt overrides are sealed randomized-condition metadata.  Vanilla
    # LIBERO must retain the canonical benchmark task name exactly.
    if condition is not None and getattr(condition, "suite_mode", "vanilla") == "sealed_randomized":
        return str(TASK_PROMPT_OVERRIDE.get(int(task_id), TASK_NAMES[int(task_id)]).replace("_", " "))
    return str(TASK_NAMES[int(task_id)].replace("_", " "))


def build_libero_env_factory(condition: Any | None = None) -> Any:
    """Create the lazy LIBERO environment factory used by native evaluation."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from lerobot.envs.libero import LiberoEnv
        from libero.libero import benchmark
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("native Octo evaluation requires the pinned LIBERO/LeRobot runtime") from exc
    from vla_benchmarking.libero.shared.config import BENCHMARK_NAME, TASK_REMOVE_CONFIG

    def _condition_value(name: str, default: Any) -> Any:
        if isinstance(condition, Mapping):
            return condition.get(name, default)
        return getattr(condition, name, default)

    suite_mode = str(_condition_value("suite_mode", "vanilla"))
    camera = str(_condition_value("camera", "agentview"))
    resolution = tuple(int(value) for value in _condition_value("resolution", (256, 256)))
    if camera != "agentview" or resolution != (256, 256):
        raise ValueError("native Octo LIBERO input is sealed to agentview at 256x256")
    if suite_mode == "sealed_randomized":
        # Keep native construction aligned with the shared sealed evaluator:
        # deterministic BDDL removal, camera naming, and terminal-reset
        # compensation are installed before any environment is instantiated.
        from vla_benchmarking.libero.evaluation.run_lerobot_eval_with_context import (
            _patch_libero_env_bddl_selection,
            _patch_libero_env_camera_creation,
            _patch_libero_env_terminal_reset_compensation,
        )

        _patch_libero_env_bddl_selection(TASK_REMOVE_CONFIG)
        _patch_libero_env_camera_creation()
        _patch_libero_env_terminal_reset_compensation()
    elif suite_mode != "vanilla":
        raise ValueError(f"unsupported native Octo suite_mode: {suite_mode}")

    task_suite = benchmark.get_benchmark_dict()[BENCHMARK_NAME]()

    def factory(episode: Any) -> Any:
        env = LiberoEnv(
                task_suite=task_suite,
                task_id=int(episode.cell.task_id),
                task_suite_name=BENCHMARK_NAME,
                camera_name=["agentview"],
                camera_name_mapping={"agentview": "image"},
                observation_width=resolution[0],
                observation_height=resolution[1],
                visualization_width=resolution[0],
                visualization_height=resolution[1],
                init_states=True,
                episode_index=int(episode.cell.init_state_index),
                n_envs=1,
            )
        return _LiberoObservationEnv(
            env,
            suite_mode=suite_mode,
            task_id=int(episode.cell.task_id),
            setup_audit={
                "suite_mode": suite_mode,
                "condition_camera": camera,
                "condition_resolution": list(resolution),
                "bddl_patch": suite_mode == "sealed_randomized",
                "task_swap_patch": suite_mode == "sealed_randomized",
                "frame_rotation_owner": "octo.native_eval._LiberoObservationEnv",
                "frame_rotation_count": 1,
            },
        )

    return factory


class _LiberoObservationEnv:
    """Adapt LeRobot's pixel dictionary to the shared Octo observation contract."""

    def __init__(
        self,
        env: Any,
        *,
        suite_mode: str = "vanilla",
        task_id: int | None = None,
        setup_audit: Mapping[str, Any] | None = None,
    ) -> None:
        self._env = env
        self._suite_mode = str(suite_mode)
        self._task_id = None if task_id is None else int(task_id)
        self.setup_audit = dict(setup_audit or {})

    def _apply_sealed_reset_layout(self) -> Mapping[str, Any]:
        """Apply swaps only after reset has selected the paired init state."""

        from vla_benchmarking.libero.evaluation.randomize_scenes import (
            SceneRandomizerVecEnvWrapper,
        )
        from vla_benchmarking.libero.shared.config import TASK_SWAP_CONFIG

        if self._task_id is None:
            raise RuntimeError("sealed Octo environment lacks a task id for scene randomization")
        inner_env = getattr(self._env, "_env", self._env)
        # LeRobot may expose one additional gym wrapper around the ControlEnv.
        control_env = inner_env if hasattr(inner_env, "sim") else getattr(inner_env, "env", inner_env)
        wrapper = SceneRandomizerVecEnvWrapper(
            None,
            self._task_id,
            TASK_SWAP_CONFIG,
            settle_steps=200,
            verbose=False,
        )
        protected = wrapper._snapshot_protected(control_env)
        results = wrapper._apply_swaps(control_env)
        expected = [label for pair in TASK_SWAP_CONFIG.get(self._task_id, []) for label in pair]
        applied = [label for item in results for label in item.get("applied", [])]
        skipped = [label for item in results for label in item.get("skipped", [])]
        if skipped or sorted(applied) != sorted(expected):
            raise RuntimeError(
                f"task {self._task_id} layout was not fully applied: "
                f"expected={expected}, applied={applied}, skipped={skipped}"
            )
        wrapper._restore_protected(control_env, protected)
        sim = getattr(control_env, "sim", None)
        if sim is not None and callable(getattr(sim, "forward", None)):
            sim.forward()
        wrapper._verify_protected(control_env, protected)
        return {
            "configured": [list(pair) for pair in TASK_SWAP_CONFIG.get(self._task_id, [])],
            "applied": applied,
            "skipped": skipped,
            "status": "environment_ok",
        }

    def _recapture_after_layout(self, fallback: Any) -> Any:
        """Force a fresh image after swaps, matching the shared vec wrapper."""

        del fallback
        # LeRobot's hierarchy is LiberoEnv (formatter) -> vector/gym wrapper
        # -> ControlEnv (raw observation getter).  Keep these responsibilities
        # separate; calling the formatter on the inner wrapper silently skips
        # the camera-key projection in some runtime versions.
        outer_env = self._env
        inner_env = getattr(outer_env, "_env", None)
        control_env = getattr(inner_env, "env", None) if inner_env is not None else None
        if control_env is None:
            control_env = inner_env
        getter = getattr(control_env, "_get_observations", None)
        if not callable(getter):
            raise RuntimeError("sealed Octo environment cannot recapture observation after task swaps")
        raw = getter(force_update=True)
        formatter = getattr(outer_env, "_format_raw_obs", None)
        if not callable(formatter):
            raise RuntimeError("sealed Octo LiberoEnv lacks _format_raw_obs for recapture")
        return formatter(raw)

    @staticmethod
    def _observation(value: Any) -> dict[str, Any]:
        if isinstance(value, tuple) and len(value) == 2:
            value = value[0]
        if not isinstance(value, Mapping):
            raise TypeError("LIBERO reset/step did not return an observation mapping")
        pixels = value.get("pixels", value)
        if not isinstance(pixels, Mapping):
            raise TypeError("LIBERO observation lacks a pixels mapping")
        image = pixels.get("image", pixels.get("image_primary"))
        if image is None:
            raise ValueError("LIBERO observation lacks the canonical agentview image")
        array = np.asarray(image)
        if array.shape == (1, 256, 256, 3):
            array = array[0]
        if array.shape != (256, 256, 3):
            raise ValueError(f"LIBERO agentview image must be 256x256 RGB, got {array.shape}")
        # LeRobot exposes the stored/raw camera orientation here.  This is the
        # sole live-evaluation rotation owner; downstream Octo validation only
        # accepts the resulting canonical frame and never flips it again.
        array = rotate_stored_frame_180(array)
        return {
            "image_primary": array,
            "timestep": 0,
            "frame_provenance": {
                "source_orientation": FRAME_ORIENTATION_STORED_RAW,
                "canonical_orientation": FRAME_ORIENTATION_LIBERO_CANONICAL,
                "rotation_owner": "octo.native_eval._LiberoObservationEnv",
                "rotation_count": 1,
            },
        }

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> Mapping[str, Any]:
        del episode_index
        if self._task_id is not None and int(task_id) != self._task_id:
            raise ValueError("native Octo episode task id disagrees with environment factory")
        self._timestep = 0
        reset_value = self._env.reset(seed=int(seed))
        observed = reset_value
        if self._suite_mode == "sealed_randomized":
            swap_evidence = self._apply_sealed_reset_layout()
            observed = self._recapture_after_layout(reset_value)
            self.setup_audit["swap_evidence"] = dict(swap_evidence)
            self.setup_audit["observation_recaptured"] = True
        return self._observation(observed)

    def step(self, action: np.ndarray) -> Any:
        result = self._env.step(np.asarray(action, dtype=np.float32))
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise TypeError("LIBERO step must return a 4- or 5-tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        else:
            observation, reward, terminated, info = result
            truncated = False
        current = self._observation(observation)
        current["timestep"] = int(getattr(self, "_timestep", 0)) + 1
        self._timestep = current["timestep"]
        return current, reward, terminated, truncated, info

    def check_success(self) -> Any:
        method = getattr(self._env, "check_success", None)
        return method() if callable(method) else False

    def close(self) -> None:
        self._env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("community_eval", "matched_train"), required=True)
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--policy-kind", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--io-sha256", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-tree-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--execute-horizon", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    return parser


def _validate_plan_provenance(
    plan: Mapping[str, Any], *, runtime_sha256: str, io_sha256: str,
    dataset_sha256: str, checkpoint_sha256: str | None,
    checkpoint_revision: str,
) -> None:
    """Check v2 receipt bindings before loading the native checkpoint."""

    if plan.get("schema") != "shared_evaluation_plan.v2":
        return
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise SystemExit("Octo v2 plan lacks provenance bindings")
    artifact = bindings.get("artifact")
    if not isinstance(artifact, Mapping):
        raise SystemExit("Octo v2 plan artifact binding is invalid")
    if str(artifact.get("checkpoint_revision", artifact.get("revision", ""))) != str(checkpoint_revision):
        raise SystemExit("Octo plan checkpoint revision disagrees with preflight")
    expected_hash = artifact.get("checkpoint_sha256", artifact.get("artifact_sha256", artifact.get("sha256")))
    if expected_hash is not None and str(expected_hash) != str(checkpoint_sha256):
        raise SystemExit("Octo plan checkpoint hash disagrees with preflight")
    for name, observed in (("runtime", runtime_sha256), ("io", io_sha256), ("dataset_manifest", dataset_sha256)):
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise SystemExit(f"Octo v2 plan {name} binding is invalid")
        expected = binding.get("sha256", binding.get("manifest_sha256", binding.get("id")))
        if expected is not None and str(expected) != str(observed):
            raise SystemExit(f"Octo plan {name} receipt disagrees with preflight")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action_horizon != 4 or args.execute_horizon != 4:
        raise SystemExit("Octo native evaluator is sealed to the four-action native horizon")
    from .preflight import run_preflight
    from .preflight import runtime_closure_paths
    from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt
    from vla_benchmarking.libero.evaluation.plan import validate_plan
    from vla_benchmarking.libero.evaluation.libero_policy_rollout import run_native_policy_eval

    evidence = run_preflight(
        mode=args.mode,
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
    )
    if str(evidence["checkpoint_revision"]) != str(args.checkpoint_revision):
        raise SystemExit("native Octo checkpoint revision disagrees with its preflight receipt")
    expected_dataset_digest = evidence.get("dataset_manifest_sha256") or evidence.get("dataset_statistics_sha256")
    if str(expected_dataset_digest) != str(args.dataset_manifest_sha256):
        raise SystemExit("native Octo dataset receipt digest disagrees with its preflight receipt")
    if str(evidence["runtime_sha256"]) != str(args.runtime_sha256):
        raise SystemExit("native Octo runtime provenance disagrees with its preflight receipt")
    if str(evidence["io_sha256"]) != str(args.io_sha256):
        raise SystemExit("native Octo IO provenance disagrees with its preflight receipt")
    runtime = derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)
    if str(runtime["sha256"]) != str(evidence["runtime_sha256"]):
        raise SystemExit("native Octo runtime closure changed after preflight")
    observed_checkpoint_sha256 = args.checkpoint_sha256 or args.checkpoint_tree_sha256
    if evidence.get("checkpoint_sha256") and str(evidence["checkpoint_sha256"]) != str(observed_checkpoint_sha256):
        raise SystemExit("native Octo checkpoint tree hash disagrees with its preflight receipt")
    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    config = COMMUNITY_EVAL_CONFIG if args.mode == "community_eval" else MATCHED_TRAIN_CONFIG
    if args.mode == "community_eval" and args.dataset_manifest is not None:
        raise SystemExit("community evaluation is bound to checkpoint-owned statistics, not a caller manifest")
    if args.mode == "matched_train" and not args.dataset_manifest:
        raise SystemExit("matched Octo evaluation requires the completed local dataset manifest")
    if plan["policy_kind"] != args.policy_kind or args.policy_kind != config.policy_kind:
        raise SystemExit("evaluation plan policy kind does not match the selected Octo mode")
    bindings = plan.get("bindings", {})
    runtime_binding = bindings.get("runtime", {}) if isinstance(bindings, Mapping) else {}
    io_binding = bindings.get("io", {}) if isinstance(bindings, Mapping) else {}
    dataset_binding = bindings.get("dataset_manifest", {}) if isinstance(bindings, Mapping) else {}
    _validate_plan_provenance(
        plan,
        runtime_sha256=args.runtime_sha256,
        io_sha256=args.io_sha256,
        dataset_sha256=args.dataset_manifest_sha256,
        checkpoint_sha256=observed_checkpoint_sha256,
        checkpoint_revision=args.checkpoint_revision,
    )
    adapter = OctoPolicyAdapter.from_pretrained(
        str(Path(args.checkpoint_path).expanduser().resolve()),
        seed=config.seed,
        mode=args.mode,
        checkpoint_revision=args.checkpoint_revision,
        artifact_id="octo_base15_spatial_no_arrow_matched_finetuned",
        runtime_id=str(runtime_binding.get("id")) if runtime_binding.get("id") else None,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        dataset_manifest_id=str(dataset_binding.get("id")) if dataset_binding.get("id") else None,
        runtime_sha256=args.runtime_sha256,
        io_id=str(io_binding.get("id")) if io_binding.get("id") else None,
        io_sha256=args.io_sha256,
        checkpoint_sha256=observed_checkpoint_sha256,
        checkpoint_root=str(Path(args.checkpoint_root).expanduser().resolve()),
        checkpoint_step=int(args.checkpoint_step),
    )
    records = run_native_policy_eval(
        adapter,
        plan,
        _plan_episodes(plan),
        env_factory=build_libero_env_factory(plan["condition"]),
        output_jsonl=args.output_jsonl,
    )
    print(json.dumps({"episodes": len(records), "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - runtime boundary
    raise SystemExit(main())
