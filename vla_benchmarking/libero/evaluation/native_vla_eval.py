"""Shared plan-consuming runtime for native VLA adapters.

The model-specific command builders historically pointed at the upstream
evaluators directly.  That made it possible for a printed plan to be ignored
by the actual rollout.  This module is the checked-in bridge used when those
builders receive ``--plan``: it loads the selected native adapter, binds the
plan's immutable receipts, and delegates all episode accounting to
``run_native_policy_eval``.

Optional model runtimes are imported only after argument and plan validation.
No fallback policy, action repetition, or caller-provided normalization is
accepted.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .libero_policy_rollout import ProductionLiberoEnvironmentFactory, run_native_policy_eval
from .plan import validate_libero_source_hashes, validate_plan
from .policy_adapter import (
    bind_adapter_provenance,
    derive_checkpoint_receipt,
    derive_dataset_manifest_receipt,
    derive_io_receipt,
    derive_runtime_receipt,
)
from .run_policy_eval import EpisodeSpec


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"symbol must use module:attribute syntax, got {spec!r}")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = module
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"configured symbol is not callable: {spec}")
    return value


def _task_description(task_id: int, condition: Any | None = None) -> str:
    from vla_benchmarking.libero.shared.config import TASK_NAMES, TASK_PROMPT_OVERRIDE

    suite_mode = getattr(condition, "suite_mode", None)
    if isinstance(condition, Mapping):
        suite_mode = condition.get("suite_mode")
    name = TASK_PROMPT_OVERRIDE.get(int(task_id), TASK_NAMES[int(task_id)]) if suite_mode == "sealed_randomized" else TASK_NAMES[int(task_id)]
    return str(name.replace("_", " "))


def plan_episodes(plan: Mapping[str, Any]) -> list[EpisodeSpec]:
    """Rebuild the exact schedule in a validated shared plan."""

    from .contracts import EvaluationCell, EvaluationCondition

    condition_payload = dict(plan["condition"])
    # ``visual_arrow`` is retained as a compatibility field in older plan
    # files but is not part of the dataclass constructor.
    condition_payload.pop("visual_arrow", None)
    condition = EvaluationCondition(**condition_payload)
    episodes: list[EpisodeSpec] = []
    for item in plan["schedule"]["cells"]:
        cell = EvaluationCell(
            cell_index=int(item["cell_index"]),
            task_id=int(item["task_id"]),
            episode_index=int(item["episode_index"]),
            seed=int(item["seed"]),
            init_state_index=int(item.get("init_state_index", item["episode_index"])),
            condition=condition,
        )
        episodes.append(EpisodeSpec(cell=cell, task_description=_task_description(cell.task_id, condition)))
    return episodes


def _quat_xyzw_to_rotvec(quaternion: Any) -> np.ndarray:
    """Convert LIBERO's xyzw quaternion to the 3-D axis-angle contract."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("robot0_eef_quat must be a finite xyzw quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("robot0_eef_quat has zero norm")
    x, y, z, w = q / norm
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * float(np.arccos(w))
    sine = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if sine <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.asarray((x, y, z), dtype=np.float64) / sine
    return np.asarray(axis * angle, dtype=np.float32)


def _canonical_observation(value: Any, *, original_openvla: bool = False) -> dict[str, Any]:
    """Normalize direct LIBERO pixels for original OpenVLA or OpenVLA-OFT."""

    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if not isinstance(value, Mapping):
        raise TypeError("LIBERO VLA environment must return an observation mapping")
    pixels = value.get("pixels", value)
    if not isinstance(pixels, Mapping):
        raise TypeError("LIBERO VLA observation pixels must be a mapping")

    def first(*keys: str) -> tuple[Any, bool]:
        for key in keys:
            if key in value:
                return value[key], key in {
                    "agentview_image", "agentview", "eye_in_hand_image", "robot0_eye_in_hand_image"
                }
            if key in pixels:
                return pixels[key], False
        return None, False

    def rgb(image: Any, name: str, *, rotate_raw: bool) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape != (256, 256, 3):
            raise ValueError(f"{name} must be 256x256 RGB, got {array.shape}")
        if rotate_raw:
            array = array[::-1, ::-1]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    primary, primary_raw = first(
        "observation.images.image", "image_primary", "image", "agentview_image", "agentview"
    )
    if primary is None:
        raise ValueError("LIBERO VLA observation must expose an agentview RGB image")

    if original_openvla:
        return {
            "agentview": rgb(primary, "agentview", rotate_raw=primary_raw),
            "frame_provenance": {
                "source_orientation": "libero_raw" if primary_raw else "libero_canonical",
                "canonical_orientation": "libero_canonical",
                "rotation_owner": "shared_evaluator" if primary_raw else "none",
            },
        }
    wrist, wrist_raw = first(
        "observation.images.image2", "image_wrist", "wrist_image", "eye_in_hand_image", "robot0_eye_in_hand_image"
    )
    if wrist is None:
        raise ValueError("LIBERO VLA observation must expose agentview and wrist RGB images")

    state_value, _ = first("observation.state", "state")
    if state_value is None:
        position, _ = first("robot0_eef_pos", "eef_pos")
        quaternion, _ = first("robot0_eef_quat", "eef_quat")
        gripper, _ = first("robot0_gripper_qpos", "gripper_qpos")
        if position is None or quaternion is None or gripper is None:
            raise ValueError("LIBERO VLA observation lacks the required 8-D proprioception")
        state_value = np.concatenate(
            [np.asarray(position, dtype=np.float32).reshape(-1),
             _quat_xyzw_to_rotvec(quaternion),
             np.asarray(gripper, dtype=np.float32).reshape(-1)]
        )
    state = np.asarray(state_value, dtype=np.float32).reshape(-1)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"LIBERO VLA observation.state must be finite shape (8,), got {state.shape}")
    return {
        "observation.images.image": rgb(primary, "agentview", rotate_raw=primary_raw),
        "observation.images.image2": rgb(wrist, "wrist", rotate_raw=wrist_raw),
        "observation.state": state,
        "frame_provenance": {
            "source_orientation": "libero_raw" if (primary_raw or wrist_raw) else "libero_canonical",
            "canonical_orientation": "libero_canonical",
            "rotation_owner": "shared_evaluator" if (primary_raw or wrist_raw) else "none",
        },
    }


class _VLAEnvironment:
    def __init__(self, environment: Any, *, original_openvla: bool = False) -> None:
        self._environment = environment
        self._original_openvla = bool(original_openvla)

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> Mapping[str, Any]:
        return _canonical_observation(
            self._environment.reset(seed=seed, task_id=task_id, episode_index=episode_index),
            original_openvla=self._original_openvla,
        )

    def step(self, action: np.ndarray) -> Any:
        result = self._environment.step(action)
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise TypeError("LIBERO VLA environment step must return a 4- or 5-tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return _canonical_observation(observation, original_openvla=self._original_openvla), reward, terminated, truncated, info
        observation, reward, done, info = result
        return _canonical_observation(observation, original_openvla=self._original_openvla), reward, done, info

    def check_success(self) -> Any:
        method = getattr(self._environment, "check_success", None)
        return method() if callable(method) else False

    def close(self) -> None:
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


def build_vla_environment_factory(plan: Mapping[str, Any]):
    original_openvla = str(plan.get("policy_kind")) == "openvla"
    base_factory = ProductionLiberoEnvironmentFactory(
        resolution=int(plan["condition"]["resolution"]),
        suite_mode=str(plan["condition"]["suite_mode"]),
        extra_camera_names=() if original_openvla else ("robot0_eye_in_hand",),
    )

    def factory(episode: EpisodeSpec) -> _VLAEnvironment:
        return _VLAEnvironment(base_factory(episode), original_openvla=original_openvla)

    return factory


def _runtime_closure_paths(model: str) -> tuple[Path, ...]:
    """Return the canonical source/runtime closure used by plans and eval."""

    repo_root = Path(__file__).resolve().parents[3]
    paths = [
        repo_root / "vla_benchmarking/libero/evaluation",
        repo_root / "vla_benchmarking/libero/finetuned_vlas/common",
        repo_root / "vla_benchmarking/libero/finetuned_vlas" / model,
    ]
    for module_name in ("lerobot", "experiments", "prismatic"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None:
            locations = list(spec.submodule_search_locations or ())
            if locations:
                paths.extend(Path(item).resolve() for item in locations)
            elif spec.origin:
                paths.append(Path(spec.origin).resolve())
    return tuple(dict.fromkeys(paths))


def _runtime_closure_for_adapter(model: str, adapter: Any, policy_config_factory: str | None) -> tuple[Path, ...]:
    """Return the canonical closure plus any explicitly selected factory."""

    paths = list(_runtime_closure_paths(model))
    if policy_config_factory:
        factory = _load_symbol(policy_config_factory)
        try:
            source = inspect.getsourcefile(factory)
        except (OSError, TypeError):
            source = None
        if source:
            paths.append(Path(source).resolve())
    return tuple(dict.fromkeys(paths))


def _artifact_for_loaded_checkpoint(model: str, checkpoint_path: str, checkpoint_revision: str) -> Any:
    """Build the loader artifact from model identity and external load evidence.

    The shared plan is an assertion to validate after loading, never a source
    of artifact labels.  ``checkpoint_revision`` must come from the preflight
    receipt/loader invocation and is independently checked as immutable.
    """

    if model == "pi05":
        from vla_benchmarking.libero.finetuned_vlas.pi05.contracts import PI05_ARTIFACT, PolicyArtifact

        return replace(PI05_ARTIFACT, revision=str(checkpoint_revision).lower(), checkpoint_path=checkpoint_path)
    if model == "openvla":
        from vla_benchmarking.libero.finetuned_vlas.openvla.contracts import OPENVLA_ARTIFACT, PolicyArtifact

        return replace(OPENVLA_ARTIFACT, revision=str(checkpoint_revision).lower(), checkpoint_path=checkpoint_path)
    from vla_benchmarking.libero.finetuned_vlas.openvla_oft.contracts import OPENVLA_ARTIFACT, PolicyArtifact

    return replace(OPENVLA_ARTIFACT, revision=str(checkpoint_revision).lower(), checkpoint_path=checkpoint_path)


def _load_adapter(
    model: str,
    checkpoint_path: str,
    checkpoint_revision: str,
    policy_config_factory: str | None,
    *,
    plan: Mapping[str, Any] | None = None,
):
    artifact = _artifact_for_loaded_checkpoint(model, checkpoint_path, checkpoint_revision)
    if model == "openvla":
        from vla_benchmarking.libero.finetuned_vlas.openvla.policy import OpenVLAAdapter

        return OpenVLAAdapter.load(artifact=artifact)
    if model == "openvla_oft":
        from vla_benchmarking.libero.finetuned_vlas.openvla_oft.policy import OpenVLAOFTAdapter

        return OpenVLAOFTAdapter.load(artifact=artifact)
    from vla_benchmarking.libero.finetuned_vlas.pi05.policy import Pi05Adapter

    if not policy_config_factory:
        # Pi05 checkpoints carry config.json and both processor pipelines.
        # The adapter loads those checkpoint-owned artifacts directly.
        return Pi05Adapter.load(artifact=artifact)

    config_bundle = _load_symbol(policy_config_factory)(
        checkpoint_path=checkpoint_path, artifact=artifact, plan=plan or {}
    )
    if not isinstance(config_bundle, Mapping):
        raise TypeError("Pi0.5 policy-config-factory must return a mapping")
    required = {"policy_config", "dataset_meta", "env_config"}
    if not required.issubset(config_bundle):
        raise ValueError(f"Pi0.5 policy-config-factory missing {sorted(required.difference(config_bundle))}")
    return Pi05Adapter.load(
        artifact=artifact,
        policy_config=config_bundle["policy_config"],
        dataset_meta=config_bundle["dataset_meta"],
        env_config=config_bundle["env_config"],
    )


def run_native_vla_eval(
    *, model: str, plan_path: str | Path, checkpoint_path: str, output_jsonl: str | Path | None = None,
    policy_config_factory: str | None = None, dataset_manifest_path: str | Path | None = None,
    checkpoint_revision: str | None = None,
) -> list[Any]:
    plan = validate_plan(json.loads(Path(plan_path).read_text(encoding="utf-8")))
    validate_libero_source_hashes(plan)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"native VLA evaluation requires a resolved local checkpoint path: {checkpoint}"
        )
    expected = {"pi05": "pi05", "openvla": "openvla", "openvla_oft": "openvla_oft"}
    if model not in expected:
        raise ValueError(f"unsupported native VLA model: {model}")
    if plan["policy_kind"] != expected[model]:
        raise ValueError("evaluation plan policy kind does not match native VLA model")
    if checkpoint_revision is None:
        raise ValueError("native VLA evaluation requires an independently supplied checkpoint revision")
    if dataset_manifest_path is None:
        raise ValueError("native VLA evaluation requires the source dataset manifest path")
    adapter = _load_adapter(
        model, str(checkpoint), str(checkpoint_revision), policy_config_factory, plan=plan,
    )
    metadata = adapter.metadata
    checkpoint_receipt = derive_checkpoint_receipt(
        checkpoint,
        artifact_id=metadata.artifact_id,
        checkpoint_revision=metadata.checkpoint_revision,
    )
    runtime_receipt = derive_runtime_receipt(
        closure_paths=_runtime_closure_for_adapter(model, adapter, policy_config_factory),
    )
    io_receipt = derive_io_receipt(metadata)
    dataset_receipt = derive_dataset_manifest_receipt(dataset_manifest_path)
    adapter = bind_adapter_provenance(
        adapter,
        artifact=checkpoint_receipt,
        runtime=runtime_receipt,
        io=io_receipt,
        dataset_manifest=dataset_receipt,
    )
    return run_native_policy_eval(
        adapter,
        plan,
        plan_episodes(plan),
        env_factory=build_vla_environment_factory(plan),
        output_jsonl=output_jsonl,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "openvla", "openvla_oft"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--policy-config-factory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = run_native_vla_eval(
        model=args.model,
        plan_path=args.plan,
        checkpoint_path=args.checkpoint_path,
        output_jsonl=args.output_jsonl,
        policy_config_factory=args.policy_config_factory,
        checkpoint_revision=args.checkpoint_revision,
        dataset_manifest_path=args.dataset_manifest,
    )
    print(json.dumps({"episodes": len(records), "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - runtime boundary
    raise SystemExit(main())


__all__ = ["build_vla_environment_factory", "plan_episodes", "run_native_vla_eval"]
