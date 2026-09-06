"""Contracts for the original OpenVLA LIBERO checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np


OPENVLA_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
OPENVLA_UNNORM_KEY = "libero_spatial"
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"
OPENVLA_ACTION_DIM = 7
OPENVLA_ACTION_HORIZON = 1
OPENVLA_INPUT_RESOLUTION = 224
OPENVLA_CAMERA_KEY = "agentview"


@dataclass(frozen=True)
class PolicyArtifact:
    model_id: str = OPENVLA_MODEL_ID
    revision: str = "962318cec55ac10993ff0f5f43eda9a270b4c873"
    checkpoint_path: str | None = None
    provenance: str = "official-openvla-libero-spatial"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyIOContract:
    camera_keys: tuple[str, ...] = (OPENVLA_CAMERA_KEY,)
    image_size: tuple[int, int] = (OPENVLA_INPUT_RESOLUTION, OPENVLA_INPUT_RESOLUTION)
    state_dim: int | None = None
    action_dim: int = OPENVLA_ACTION_DIM
    action_horizon: int = OPENVLA_ACTION_HORIZON
    uses_language: bool = True
    uses_proprioception: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["camera_keys"] = list(self.camera_keys)
        result["image_size"] = list(self.image_size)
        return result


OPENVLA_ARTIFACT = PolicyArtifact()
OPENVLA_IO = PolicyIOContract()


def validate_observation(observation: Mapping[str, Any], contract: PolicyIOContract = OPENVLA_IO) -> None:
    missing = [key for key in contract.camera_keys if key not in observation]
    if missing:
        raise ValueError(f"original OpenVLA observation is missing camera keys: {missing}")
    image = np.asarray(observation[OPENVLA_CAMERA_KEY])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{OPENVLA_CAMERA_KEY} must be an HWC RGB image, got {image.shape}")
    if not np.isfinite(image.astype(np.float32, copy=False)).all():
        raise ValueError(f"{OPENVLA_CAMERA_KEY} contains non-finite values")


def validate_action(action: Any) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (OPENVLA_ACTION_DIM,):
        raise ValueError(f"original OpenVLA action must have shape (7,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("original OpenVLA action contains non-finite values")
    return values
