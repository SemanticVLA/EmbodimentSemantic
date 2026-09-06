"""OpenVLA-OFT contracts and matched-training constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import numpy as np


# This is an existing dataset/configuration name in the pinned OpenVLA-OFT
# fork (``prismatic.vla.datasets.rlds.oxe``).  Keep the generated TFDS builder,
# RLDSDataset lookup, and saved dataset-statistics key on the same identifier.
OPENVLA_DATASET_NAME = "libero_spatial_no_noops"
# The launcher must point at this exact upstream checkout.  Keeping the
# revision here makes command generation and shell launch validation agree.
OPENVLA_OFT_UPSTREAM_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
OPENVLA_FINETUNE_ENTRYPOINT = "vla-scripts/finetune.py"
OPENVLA_EVAL_ENTRYPOINT = "experiments/robot/libero/run_libero_eval.py"


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
    action_horizon: int = 8
    uses_language: bool = True
    uses_proprioception: bool = True

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["camera_keys"] = list(self.camera_keys)
        result["image_size"] = list(self.image_size)
        return result


@dataclass(frozen=True)
class TrainingSpec:
    dataset_id: str = OPENVLA_DATASET_NAME
    serialization: str = "rlds"
    episodes: int = 500
    timesteps: int = 62250
    epochs: int = 15
    optimizer_updates: int = field(init=False)
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 32
    seed: int = 1000
    precision: str = "bf16"
    lora_rank: int = 32
    action_loss: str = "l1_continuous"
    image_augmentation: bool = True
    center_crop_eval: bool = True
    learning_rate: float = 5e-4
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
    if int(timesteps) <= 0 or int(epochs) <= 0 or int(effective_batch_size) <= 0:
        raise ValueError("timesteps, epochs, and effective_batch_size must be positive")
    return int(math.ceil(int(timesteps) * int(epochs) / int(effective_batch_size)))


OPENVLA_ARTIFACT = PolicyArtifact(
    model_id="openvla/openvla-7b-finetuned-libero-spatial",
    revision="main",
    provenance="official-libero-spatial-finetuned; evaluate-without-additional-training",
)
OPENVLA_IO = PolicyIOContract()
OPENVLA_TRAINING = TrainingSpec()


def validate_observation(observation: Mapping[str, Any], contract: PolicyIOContract = OPENVLA_IO) -> None:
    missing = [key for key in contract.camera_keys if key not in observation]
    if missing:
        raise ValueError(f"OpenVLA-OFT observation is missing camera keys: {missing}")
    if "observation.state" not in observation:
        raise ValueError("OpenVLA-OFT observation is missing observation.state")
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


def validate_action_chunk(action: Any, contract: PolicyIOContract = OPENVLA_IO) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    expected = (contract.action_horizon, contract.action_dim)
    if values.shape != expected:
        raise ValueError(f"OpenVLA-OFT action chunk must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("OpenVLA-OFT action chunk contains non-finite values")
    return values
