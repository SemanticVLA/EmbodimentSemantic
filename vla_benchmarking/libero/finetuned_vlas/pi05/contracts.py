"""Model-neutral Pi0.5 contracts and sealed training constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PolicyArtifact:
    model_id: str
    revision: str
    checkpoint_path: str | None = None
    adapter_path: str | None = None
    provenance: str = "unverified"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyIOContract:
    camera_keys: tuple[str, str] = (
        "observation.images.image",
        "observation.images.image2",
    )
    image_size: tuple[int, int] = (256, 256)
    state_dim: int = 8
    action_dim: int = 7
    # Pi05 v044 natively predicts a 50-step chunk.  The shared evaluator may
    # stop consuming it early at an episode boundary, but must never slice it
    # before the adapter returns it or manufacture a shorter horizon.
    action_horizon: int = 50
    uses_language: bool = True
    uses_proprioception: bool = True

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["camera_keys"] = list(self.camera_keys)
        result["image_size"] = list(self.image_size)
        return result


@dataclass(frozen=True)
class TrainingSpec:
    dataset_id: str = "libero_spatial_no_arrows_v1"
    episodes: int = 500
    timesteps: int = 62250
    epochs: int = 15
    optimizer_updates: int = field(init=False)
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    seed: int = 1000
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    train_expert_only: bool = True
    learning_rate: float = 2.5e-5
    warmup_updates: int = 1000
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    checkpoint_frequency: int = 1946

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "optimizer_updates",
            optimizer_updates_for(self.timesteps, self.epochs, self.effective_batch_size),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps


def optimizer_updates_for(timesteps: int, epochs: int, effective_batch_size: int) -> int:
    """Return optimizer updates needed to expose all source samples."""
    if int(timesteps) <= 0 or int(epochs) <= 0 or int(effective_batch_size) <= 0:
        raise ValueError("timesteps, epochs, and effective_batch_size must be positive")
    return int(math.ceil(int(timesteps) * int(epochs) / int(effective_batch_size)))


def micro_steps_for(timesteps: int, epochs: int, effective_batch_size: int, accumulation_steps: int) -> int:
    """Return LeRobot loop steps when ``--steps`` counts micro-batches."""
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    return optimizer_updates_for(timesteps, epochs, effective_batch_size) * int(accumulation_steps)


PI05_ARTIFACT = PolicyArtifact(
    model_id="lerobot/pi05_libero_finetuned_v044",
    revision="8e174154ef5f6c60a8da12ae99c303d8963138c1",
    provenance="official-libero-finetuned-v044; Hub-main-resolved-2026-09-06; evaluate-without-additional-training",
)
PI05_IO = PolicyIOContract()
PI05_TRAINING = TrainingSpec()


def validate_observation(observation: Mapping[str, Any], contract: PolicyIOContract = PI05_IO) -> None:
    """Validate the exact no-arrow observation boundary before policy calls."""
    missing = [key for key in contract.camera_keys if key not in observation]
    if missing:
        raise ValueError(f"Pi0.5 observation is missing camera keys: {missing}")
    if "observation.state" not in observation:
        raise ValueError("Pi0.5 observation is missing observation.state")
    for key in contract.camera_keys:
        image = np.asarray(observation[key])
        if image.shape[-3:] not in {
            (contract.image_size[0], contract.image_size[1], 3),
            (3, contract.image_size[0], contract.image_size[1]),
        }:
            raise ValueError(f"{key} must be 256x256 RGB (HWC or CHW), got {image.shape}")
    state = np.asarray(observation["observation.state"])
    if state.shape != (contract.state_dim,):
        raise ValueError(f"observation.state must have shape ({contract.state_dim},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("observation.state contains non-finite values")


def validate_action_chunk(action: Any, contract: PolicyIOContract = PI05_IO) -> np.ndarray:
    """Validate one *native* Pi0.5 chunk.

    A single action is intentionally rejected.  Repeating it to manufacture a
    chunk hides a broken runtime contract and changes the policy semantics.
    """
    values = np.asarray(action, dtype=np.float32)
    expected = (contract.action_horizon, contract.action_dim)
    if values.shape != expected:
        raise ValueError(f"Pi0.5 action chunk must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Pi0.5 action chunk contains non-finite values")
    return values
