"""Original OpenVLA adapter using the official HF inference semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import (
    OPENVLA_ARTIFACT,
    OPENVLA_IO,
    OPENVLA_MODEL_ID,
    OPENVLA_PROMPT_TEMPLATE,
    OPENVLA_UNNORM_KEY,
    PolicyArtifact,
    PolicyIOContract,
    validate_action,
    validate_observation,
)

try:
    from vla_benchmarking.libero.evaluation.policy_adapter import PolicyMetadata, validate_policy_action
except ImportError:  # pragma: no cover
    PolicyMetadata = None  # type: ignore[assignment,misc]
    validate_policy_action = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OpenVLARuntime:
    processor: Any
    device: str = "cuda:0"
    dtype: Any | None = None
    center_crop: bool = True


def _center_crop_resize(image: Any, *, crop_scale: float = 0.9, size: int = 224) -> Any:
    """Match official ``crop_and_resize``: area .9, then resize to 224."""

    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = image.convert("RGB")
    width, height = image.size
    crop_width = max(1, int(round(width * float(crop_scale) ** 0.5)))
    crop_height = max(1, int(round(height * float(crop_scale) ** 0.5)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize(
        (int(size), int(size)), Image.Resampling.BILINEAR
    )


def _move_inputs(inputs: Any, *, device: str, dtype: Any | None) -> Any:
    if not hasattr(inputs, "to"):
        return inputs
    if dtype is not None:
        try:
            return inputs.to(device, dtype=dtype)
        except TypeError:
            pass
    return inputs.to(device)


class OpenVLAAdapter:
    """Adapter for ``openvla/openvla-7b-finetuned-libero-spatial``.

    Unlike OFT, original OpenVLA consumes one agentview image and language,
    emits one 7-D action, and performs the official LIBERO gripper conversion
    after ``predict_action(..., unnorm_key='libero_spatial')``.
    """

    def __init__(
        self,
        policy: Any | None = None,
        *,
        processor: Any | None = None,
        artifact: PolicyArtifact = OPENVLA_ARTIFACT,
        io_contract: PolicyIOContract = OPENVLA_IO,
        runtime: OpenVLARuntime | None = None,
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.artifact = artifact
        self.io_contract = io_contract
        self.runtime = runtime
        self._task_description = ""

    @property
    def metadata(self):
        if PolicyMetadata is None:  # pragma: no cover
            raise RuntimeError("shared evaluation policy contracts are unavailable")
        return PolicyMetadata(
            policy_kind="openvla",
            artifact_id=self.artifact.model_id,
            checkpoint_revision=self.artifact.revision,
            backend="openvla_hf_native",
            native_action_horizon=1,
            input_resolution=224,
            camera_keys=("agentview",),
            state_dim=None,
            extra={
                "uses_language": True,
                "uses_proprioception": False,
                "prompt_template": OPENVLA_PROMPT_TEMPLATE,
                "unnorm_key": OPENVLA_UNNORM_KEY,
                "center_crop_scale": 0.9,
                "native_output": "single_action",
            },
        )

    @classmethod
    def load(
        cls,
        artifact: PolicyArtifact = OPENVLA_ARTIFACT,
        *,
        loader: Callable[[PolicyArtifact], tuple[Any, Any]] | None = None,
        device: str = "cuda:0",
        center_crop: bool = True,
    ) -> "OpenVLAAdapter":
        if loader is not None:
            policy, processor = loader(artifact)
            return cls(policy, processor=processor, artifact=artifact)
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("original OpenVLA requires torch and transformers") from exc
        checkpoint = artifact.checkpoint_path or artifact.model_id
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        policy = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        return cls(
            policy,
            processor=processor,
            artifact=artifact,
            runtime=OpenVLARuntime(processor=processor, device=device, dtype=torch.bfloat16, center_crop=center_crop),
        )

    def reset(self, task_description: str, episode_seed: int) -> None:
        del episode_seed
        self._task_description = str(task_description)

    @staticmethod
    def _image(observation: Mapping[str, Any]) -> Any:
        for key in ("agentview", "observation.images.image", "full_image", "image"):
            if key in observation:
                return observation[key]
        raise ValueError("original OpenVLA requires one agentview image")

    @staticmethod
    def _libero_gripper_action(action: Any) -> np.ndarray:
        values = validate_action(action).copy()
        # Official OpenVLA LIBERO runner: [0,1] -> [-1,1], binarize, invert.
        values[-1] = np.sign(2.0 * values[-1] - 1.0) * -1.0
        return values

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        validate_observation({"agentview": self._image(observation)})
        if self.policy is None or self.processor is None:
            raise RuntimeError("original OpenVLA adapter has no loaded policy/processor")
        image = self._image(observation)
        center_crop = self.runtime.center_crop if self.runtime is not None else True
        image = _center_crop_resize(image) if center_crop else image
        task = self._task_description.lower()
        prompt = OPENVLA_PROMPT_TEMPLATE.format(task=task)
        inputs = self.processor(prompt, image)
        if self.runtime is not None:
            inputs = _move_inputs(inputs, device=self.runtime.device, dtype=self.runtime.dtype)
        raw = self.policy.predict_action(**inputs, unnorm_key=OPENVLA_UNNORM_KEY, do_sample=False)
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values = self._libero_gripper_action(raw)
        chunk = values.reshape(1, 7)
        if validate_policy_action is not None:
            return validate_policy_action(chunk, self.metadata, observation={"agentview": self._image(observation)})
        return chunk

    def predict(self, observation: Mapping[str, Any], *, task: str | None = None) -> np.ndarray:
        if task is not None:
            self._task_description = str(task)
        return self.act(observation)


__all__ = ["OpenVLAAdapter", "OpenVLARuntime", "_center_crop_resize"]
