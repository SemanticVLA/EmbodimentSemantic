"""Direct image/state-only ArrowSmolVLA episode runner."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ..lerobot_integration import ArrowSmolVLAPolicy, load_pinned_smolvla
from .reference_protocol import ReferenceProtocol


NORMALIZER_NAME = "policy_preprocessor_step_5_normalizer_processor.safetensors"
EXPECTED_NORMALIZER_SHA256 = "4143bc95779eb16de8d77b29bae1e02e62ae67609ad223441bc8d107e334dd9c"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = "827ebb564e6b6e9371e7a30c2d953d6e9d6a0af9e793f99dc7af569b85ab3cab"
EXPECTED_MODEL_SHA256 = "5d7670368c1eec80eb7855454e77d69d4e177d693155246f6a68409646a85bcc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_observation(env: Any) -> Mapping[str, Any]:
    for owner in (getattr(env, "env", None), env):
        getter = getattr(owner, "_get_observations", None)
        if getter is None:
            continue
        try:
            value = getter(force_update=True)
        except TypeError:
            value = getter()
        if isinstance(value, Mapping):
            return value
    raise RuntimeError("LIBERO env did not expose a raw observation")


def _rgb(env: Any, resolution: int) -> np.ndarray:
    observation = _raw_observation(env)
    value = observation.get("agentview_image")
    if value is None:
        render = getattr(env, "render", None)
        if render is None:
            raise RuntimeError("agentview RGB is unavailable")
        value = render(camera_name="agentview", width=resolution, height=resolution, depth=False)
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, Mapping):
            value = value.get("agentview_image")
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.shape != (resolution, resolution, 3):
        raise ValueError(f"agentview RGB must be {(resolution, resolution, 3)}, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.array(image, copy=True, order="C")


def _axis_angle(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.isfinite(value).all():
        raise ValueError("eef quaternion must contain four finite values")
    try:
        from robosuite.utils.transform_utils import quat2axisangle

        result = np.asarray(quat2axisangle(value), dtype=np.float64)
        if result.size == 3 and np.isfinite(result).all():
            return result
    except Exception:
        pass
    # LIBERO/robosuite use wxyz.  This fallback is equivalent for the small
    # orientations used by the environment and keeps CPU contract tests pure.
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("eef quaternion has zero norm")
    w, x, y, z = value / norm
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), max(-1.0, min(1.0, w)))
    s = math.sqrt(x * x + y * y + z * z)
    return np.zeros(3, dtype=np.float64) if s <= 1e-9 else np.array([x, y, z], dtype=np.float64) / s * angle


def extract_state(env: Any) -> np.ndarray:
    observation = _raw_observation(env)
    def first(*names: str) -> Any:
        for name in names:
            if name in observation:
                return observation[name]
        raise KeyError(names[0])
    pos = np.asarray(first("robot0_eef_pos", "eef_pos"), dtype=np.float32).reshape(-1)
    quat = first("robot0_eef_quat", "eef_quat")
    grip = np.asarray(first("robot0_gripper_qpos", "gripper_qpos"), dtype=np.float32).reshape(-1)
    state = np.concatenate((pos, _axis_angle(quat).astype(np.float32), grip), axis=0)
    if state.size != 8 or not np.isfinite(state).all():
        raise ValueError(f"expected finite eight-dimensional eef state, got {state.shape}")
    return state


class Normalizer:
    def __init__(self, path: Path, *, expected_sha256: str = EXPECTED_NORMALIZER_SHA256) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.sha256 = sha256_file(path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise RuntimeError(f"normalizer SHA-256 mismatch: expected {expected_sha256}, got {self.sha256}")
        from safetensors.torch import load_file
        values = load_file(str(path), device="cpu")
        self.state_mean = values["observation.state.mean"].float()
        self.state_std = values["observation.state.std"].float().clamp_min(1e-8)
        self.action_mean = values["action.mean"].float()
        self.action_std = values["action.std"].float().clamp_min(1e-8)

    def state(self, value: np.ndarray, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return (tensor - self.state_mean.to(device)) / self.state_std.to(device)

    def action(self, value: torch.Tensor) -> np.ndarray:
        raw = value.float().detach().cpu() * self.action_std + self.action_mean
        return raw.numpy()


def resolve_normalizer(source: Path, base: Path) -> Normalizer:
    for directory in (source, base):
        candidate = directory / NORMALIZER_NAME
        if candidate.is_file():
            return Normalizer(candidate)
    raise FileNotFoundError(f"pinned normalizer is missing from source and base checkpoints: {source}, {base}")


class ArrowStudentRuntime:
    """Loads and validates one immutable student checkpoint."""

    def __init__(self, *, checkpoint: Path, source_checkpoint: Path, base_policy: Path, device: str = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for ArrowStudent evaluation")
        self.checkpoint = checkpoint.expanduser().resolve()
        self.source_checkpoint = source_checkpoint.expanduser().resolve()
        self.base_policy = base_policy.expanduser().resolve()
        manifest = self.checkpoint / "manifest.json"
        model_file = self.checkpoint / "model.pt"
        if sha256_file(manifest) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
            raise RuntimeError("student checkpoint manifest hash does not match the pinned Stage D artifact")
        if sha256_file(model_file) != EXPECTED_MODEL_SHA256:
            raise RuntimeError("student model.pt hash does not match the pinned Stage D artifact")
        source = load_pinned_smolvla(self.source_checkpoint, base_policy=self.base_policy, device=str(self.device))
        # Stage D was trained with the retained student in FP32; preserve
        # that dtype when reconstructing before loading its strict weights.
        self.policy = ArrowSmolVLAPolicy(source).float().to(self.device).eval()
        try:
            weights = torch.load(model_file, map_location=self.device, weights_only=True)
        except TypeError:  # pinned Legion torch may predate weights_only
            weights = torch.load(model_file, map_location=self.device)
        self.policy.load_state_dict(weights, strict=True)
        self.normalizer = resolve_normalizer(self.source_checkpoint, self.base_policy)
        config = getattr(source, "config", None)
        self.n_action_steps = int(getattr(config, "n_action_steps", 1))
        self.chunk_size = int(getattr(config, "chunk_size", 50))
        self.num_steps = int(getattr(config, "num_steps", 10))
        self.action_dim = int(self.policy.action_dim)
        self.max_action_dim = int(self.policy.max_action_dim)
        if self.n_action_steps <= 0 or self.chunk_size <= 0 or self.num_steps <= 0:
            raise RuntimeError("invalid pinned SmolVLA inference schedule")

    def _chunk(self, image: np.ndarray, state: np.ndarray, *, generator: torch.Generator | None = None) -> np.ndarray:
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(self.device)
        state_tensor = self.normalizer.state(state, self.device).unsqueeze(0)
        noise = torch.randn((1, self.chunk_size, self.max_action_dim), device=self.device, generator=generator)
        sample = noise
        with torch.inference_mode():
            for step in range(self.num_steps):
                timestep = torch.full((1,), 1.0 - (step / self.num_steps), device=self.device, dtype=sample.dtype)
                velocity = self.policy(image_tensor, state_tensor, sample, timestep).velocity
                if velocity.shape[-1] != self.action_dim or not torch.isfinite(velocity).all():
                    raise FloatingPointError("ArrowStudent emitted non-finite or incorrectly shaped flow velocity")
                sample = sample.clone()
                sample[..., : self.action_dim] += (-1.0 / self.num_steps) * velocity
        return self.normalizer.action(sample[..., : self.action_dim])[0]

    def preflight(self) -> dict[str, Any]:
        torch.manual_seed(1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1000)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        state = np.zeros(8, dtype=np.float32)
        actions = self._chunk(image, state)
        if actions.shape != (self.chunk_size, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError("student preflight produced invalid actions")
        return {
            "status": "passed",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "normalizer_sha256": self.normalizer.sha256,
            "device": str(self.device),
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "num_steps": self.num_steps,
            "n_action_steps": self.n_action_steps,
            "language_consumed": False,
            "tokenizer_accessed": False,
            "sample_finite": True,
        }


class ArrowStudentEpisodeRunner:
    def __init__(self, runtime: ArrowStudentRuntime, reference: ReferenceProtocol, *, resolution: int = 256, max_actions: int = 1200) -> None:
        self.runtime = runtime
        self.reference = reference
        self.resolution = int(resolution)
        self.max_actions = int(max_actions)

    def _arrow(self, env: Any, task_id: int, inputs: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        # Rebuild semantic bboxes on every chunk. They are privileged input
        # generation only; only the rendered pixels cross into the student.
        from ...run_arrow_pick_place_matrix import _default_arrow_inputs
        live = dict(_default_arrow_inputs(env, int(task_id), self.resolution))
        bboxes = live.get("bboxes", inputs.get("bboxes"))
        if not isinstance(bboxes, Mapping):
            raise RuntimeError("arrow input generation returned no bboxes")
        from ...run_arrow_pick_place_eval import render_exactly_one_arrow
        clean = _rgb(env, self.resolution)
        rendered, render_audit = render_exactly_one_arrow(
            clean, bboxes, subject=str(live.get("subject", inputs.get("subject"))),
            goal_object=str(live.get("goal_object", inputs.get("goal_object"))),
            line_width=1, head_length=16,
        )
        flipped = np.flip(rendered, axis=(0, 1)).copy()
        changed = np.any(flipped != np.flip(clean, axis=(0, 1)), axis=2)
        count = int(changed.sum())
        if count <= 0 or flipped.shape != (self.resolution, self.resolution, 3) or flipped.dtype != np.uint8:
            raise RuntimeError("arrow audit failed: no changed pixels or invalid transformed image")
        audit = {
            **render_audit,
            "camera": "agentview",
            "resolution": self.resolution,
            "arrow_color_rgb": [0, 166, 107],
            "line_width": 1,
            "head_length": 16,
            "resize_before_overlay": True,
            "flip": "180_degrees_both_axes",
            "changed_pixel_count": count,
            "clean_sha256": hashlib.sha256(np.flip(clean, axis=(0, 1)).tobytes()).hexdigest(),
            "arrow_sha256": hashlib.sha256(flipped.tobytes()).hexdigest(),
            "boxes_input_generation_only": {str(k): [float(x) for x in v] for k, v in bboxes.items()},
        }
        return flipped, audit

    def __call__(self, *, env: Any, task_id: int, seed: int, episode_index: int = 0, output_dir: str | Path, bboxes: Mapping[str, Any], subject: str = "akita_black_bowl_1", goal_object: str = "plate_1", dry_run: bool = False, evaluator: Callable[[Any], bool] | None = None, motion_started_callback: Callable[[], None] | None = None, **_: Any) -> Mapping[str, Any]:
        if dry_run:
            raise RuntimeError("ArrowStudent evaluator requires explicit execute-motion mode")
        reference_audit = self.reference.validate_environment(env, task_id=task_id, episode_index=episode_index, seed=seed)
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        rng = torch.Generator(device=getattr(self.runtime, "device", torch.device("cpu")))
        rng.manual_seed(int(seed))
        arrow_audits: list[dict[str, Any]] = []
        actions_sent = 0
        pending: list[np.ndarray] = []
        preview_path = out / "arrow_student_input.png"
        from PIL import Image
        while actions_sent < self.max_actions:
            if not pending:
                arrow, audit = self._arrow(env, int(task_id), {"bboxes": bboxes, "subject": subject, "goal_object": goal_object})
                audit["step"] = actions_sent
                arrow_audits.append(audit)
                if not preview_path.exists():
                    Image.fromarray(arrow).save(preview_path)
                state = extract_state(env)
                pending = [np.asarray(item, dtype=np.float32) for item in self.runtime._chunk(arrow, state, generator=rng)[: self.runtime.n_action_steps]]
            action = pending.pop(0)
            if action.shape != (self.runtime.action_dim,) or not np.isfinite(action).all():
                raise FloatingPointError("ArrowStudent action failed finite seven-dimensional contract")
            if motion_started_callback is not None and actions_sent == 0:
                motion_started_callback()
            result = env.step(action)
            actions_sent += 1
            done = bool(result[2]) if isinstance(result, tuple) and len(result) >= 3 else False
            if done:
                break
            if evaluator is not None and bool(evaluator(env)):
                break
        success = bool(evaluator(env)) if evaluator is not None else False
        return {
            "evaluator_success": success,
            "steps": actions_sent,
            "total_actions": actions_sent,
            "input_arrow_audit": arrow_audits[0] if arrow_audits else None,
            "arrow_refresh_audit": arrow_audits,
            "arrow_input_path": preview_path.as_posix(),
            "reference_protocol_audit": reference_audit,
            "language_consumed": False,
            "tokenizer_accessed": False,
            "observation_inputs": ["agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "action_dim": self.runtime.action_dim,
            "n_action_steps": self.runtime.n_action_steps,
        }


__all__ = ["ArrowStudentEpisodeRunner", "ArrowStudentRuntime", "Normalizer", "extract_state", "sha256_file"]
