"""Import-safe native Pi0.5 policy adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import PI05_ARTIFACT, PI05_IO, PolicyArtifact, PolicyIOContract, validate_action_chunk, validate_observation

try:
    from vla_benchmarking.libero.evaluation.policy_adapter import (
        PolicyMetadata,
        validate_arrow_free_observation,
        validate_policy_action,
    )
except ImportError:  # pragma: no cover - supports isolated package copying
    PolicyMetadata = None  # type: ignore[assignment,misc]
    validate_arrow_free_observation = None  # type: ignore[assignment]
    validate_policy_action = None  # type: ignore[assignment]


def _resolve_preprocessor_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build processor overrides, including the optional local Pi05 tokenizer.

    The Pi0.5 checkpoint's saved ``tokenizer_processor`` points at the gated
    ``google/paligemma-3b-pt-224`` Hub repository.  A repair job may resolve an
    immutable local snapshot first and expose it as ``PI05_TOKENIZER_PATH``.
    Passing it through LeRobot's processor override mechanism keeps the model
    checkpoint unchanged while making the dependency explicit and auditable.
    """

    resolved: dict[str, Any] = {
        str(key): value for key, value in (overrides or {}).items()
    }
    tokenizer_path = os.environ.get("PI05_TOKENIZER_PATH")
    if tokenizer_path:
        path = Path(tokenizer_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"PI05_TOKENIZER_PATH must be an existing local directory: {path}"
            )
        tokenizer_override = dict(resolved.get("tokenizer_processor", {}))
        tokenizer_override["tokenizer_name"] = str(path)
        resolved["tokenizer_processor"] = tokenizer_override
    return resolved


class Pi05Adapter:
    """Thin adapter around a loaded LeRobot Pi0.5 policy.

    ``policy`` and ``loader`` injection make contract tests possible without
    installing LeRobot or allocating GPU memory.  ``load()`` is the only path
    that imports the optional runtime.
    """

    def __init__(
        self,
        policy: Any | None = None,
        *,
        artifact: PolicyArtifact = PI05_ARTIFACT,
        io_contract: PolicyIOContract = PI05_IO,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        provenance: Mapping[str, Any] | Any | None = None,
        processors_required: bool = False,
    ) -> None:
        self.policy = policy
        self.artifact = artifact
        self.io_contract = io_contract
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.provenance = provenance
        self.processors_required = bool(processors_required)

    @property
    def metadata(self):
        if PolicyMetadata is None:  # pragma: no cover
            raise RuntimeError("shared evaluation policy contracts are unavailable")
        extra = {"uses_language": True, "train_expert_only": True}
        if self.provenance is not None:
            as_extra = getattr(self.provenance, "as_extra", None)
            values = as_extra() if callable(as_extra) else self.provenance
            if isinstance(values, Mapping):
                extra.update({str(key): value for key, value in values.items() if value is not None})
        return PolicyMetadata(
            policy_kind="pi05",
            artifact_id=self.artifact.model_id,
            checkpoint_revision=self.artifact.revision,
            backend="lerobot",
            native_action_horizon=self.io_contract.action_horizon,
            input_resolution=self.io_contract.image_size[0],
            camera_keys=self.io_contract.camera_keys,
            state_dim=self.io_contract.state_dim,
            extra=extra,
        )

    @classmethod
    def load(
        cls,
        artifact: PolicyArtifact = PI05_ARTIFACT,
        *,
        loader: Callable[[PolicyArtifact], Any] | None = None,
        factory: Callable[[Any, Any, Any], Any] | None = None,
        policy_config: Any | None = None,
        dataset_meta: Any | None = None,
        env_config: Any | None = None,
        ds_meta: Any | None = None,
        env_cfg: Any | None = None,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        preprocessor_overrides: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | Any | None = None,
    ) -> "Pi05Adapter":
        """Load using LeRobot's current ``make_policy(cfg, ds_meta, env_cfg)``.

        When no factory/config is supplied, load the checkpoint-owned
        ``PI05Config``, weights, preprocessor, and postprocessor directly from
        the selected artifact.  The explicit factory seam remains available
        for runtimes that need dataset metadata, while tests can inject a
        lightweight factory without importing LeRobot.
        """
        if loader is not None:
            return cls(
                loader(artifact), artifact=artifact, preprocessor=preprocessor,
                postprocessor=postprocessor, provenance=provenance,
            )
        if dataset_meta is not None and ds_meta is not None:
            raise ValueError("pass only one of dataset_meta or ds_meta")
        if env_config is not None and env_cfg is not None:
            raise ValueError("pass only one of env_config or env_cfg")
        dataset_meta = ds_meta if ds_meta is not None else dataset_meta
        env_config = env_cfg if env_cfg is not None else env_config
        preprocessor_overrides = _resolve_preprocessor_overrides(preprocessor_overrides)
        if policy_config is None and factory is None:
            try:
                from lerobot.policies import make_pre_post_processors
                from lerobot.policies.pi05 import PI05Policy
            except ImportError as exc:
                raise RuntimeError(
                    "Pi0.5 runtime requires the pinned LeRobot installation"
                ) from exc
            checkpoint = artifact.checkpoint_path or artifact.model_id
            revision = artifact.revision if len(str(artifact.revision)) in (40, 64) else None
            try:
                policy = PI05Policy.from_pretrained(checkpoint, revision=revision)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy.config,
                    pretrained_path=checkpoint,
                    preprocessor_overrides=preprocessor_overrides,
                )
            except Exception as exc:  # pragma: no cover - requires model files
                raise RuntimeError(
                    "Pi0.5 checkpoint-owned config/weights/processors could not be loaded"
                ) from exc
            return cls(
                policy,
                artifact=artifact,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                provenance=provenance,
                processors_required=True,
            )
        if policy_config is None:
            raise ValueError("policy_config is required when supplying a custom Pi0.5 factory")
        native_factory = factory is None
        if native_factory:
            try:
                from lerobot.policies.factory import make_policy  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Pi0.5 runtime requires the optional LeRobot installation; "
                    "install the pinned Pi0.5 environment before loading a policy"
                ) from exc
            factory = make_policy
        try:
            policy = factory(policy_config, dataset_meta, env_config)
        except Exception as exc:  # pragma: no cover - depends on LeRobot version
            raise RuntimeError(
                "Pi0.5 make_policy(cfg, ds_meta, env_cfg) failed; verify the pinned "
                "LeRobot PolicyConfig, dataset metadata, and environment config"
            ) from exc
        if native_factory and (preprocessor is None or postprocessor is None):
            try:
                from lerobot.policies import make_pre_post_processors  # type: ignore

                pretrained_path = getattr(policy_config, "pretrained_path", None)
                if pretrained_path is None:
                    pretrained_path = artifact.checkpoint_path or artifact.model_id
                stats = getattr(dataset_meta, "stats", None)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy_config,
                    pretrained_path=pretrained_path,
                    dataset_stats=stats,
                    preprocessor_overrides=preprocessor_overrides,
                )
            except Exception as exc:  # pragma: no cover - depends on model files
                raise RuntimeError(
                    "Pi0.5 pre/postprocessors could not be loaded; native inference "
                    "must use the checkpoint's processor configuration"
                ) from exc
        return cls(
            policy,
            artifact=artifact,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            provenance=provenance,
            processors_required=native_factory,
        )

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        validate_observation(observation, self.io_contract)

    def reset(self, task_description: str, episode_seed: int) -> None:
        if self.policy is None:
            raise RuntimeError("Pi0.5 adapter has no loaded policy")
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._task_description = str(task_description)
        self._episode_seed = int(episode_seed)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Shared-evaluator entry point, preserving the native 50-step chunk."""
        if validate_arrow_free_observation is not None:
            validate_arrow_free_observation(observation)
        raw = self.predict(observation, task=getattr(self, "_task_description", None))
        if validate_policy_action is None:  # pragma: no cover
            return raw
        # Pi0.5's checkpoint postprocessor unnormalizes actions back into the
        # LIBERO control space.  As with the simulator's own normalized action
        # boundary, clamp finite numerical overshoot to [-1, 1] before the
        # shared validator rejects it.  Shape, finiteness, and native horizon
        # are still fail-closed in ``predict`` and ``validate_policy_action``.
        metadata = self.metadata
        bounded = np.clip(
            np.asarray(raw, dtype=np.float32),
            np.asarray(metadata.action_low, dtype=np.float32)[None, :],
            np.asarray(metadata.action_high, dtype=np.float32)[None, :],
        )
        return validate_policy_action(bounded, metadata, observation=observation)

    def predict(self, observation: Mapping[str, Any], *, task: str | None = None) -> np.ndarray:
        self.validate_observation(observation)
        if self.policy is None:
            raise RuntimeError("Pi0.5 adapter has no loaded policy")
        payload = self._batched_torch_payload(observation, task=task)
        if self.preprocessor is not None:
            try:
                payload = self.preprocessor(payload)
            except Exception as exc:  # pragma: no cover - runtime processor
                raise RuntimeError("Pi0.5 observation preprocessor failed") from exc
        elif self.processors_required:
            raise RuntimeError(
                "native Pi0.5 inference requires the checkpoint preprocessor and postprocessor"
            )
        if hasattr(self.policy, "predict_action_chunk"):
            raw = self.policy.predict_action_chunk(payload)
        else:
            if hasattr(self.policy, "select_action"):
                raise RuntimeError(
                    "loaded Pi0.5 policy exposes only select_action (single-step); "
                    "a native predict_action_chunk method is required for horizon 50"
                )
            raise TypeError("loaded Pi0.5 policy exposes no predict_action_chunk interface")
        if self.postprocessor is not None:
            try:
                raw = self.postprocessor(raw)
            except Exception as exc:  # pragma: no cover - runtime processor
                raise RuntimeError("Pi0.5 action postprocessor failed") from exc
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        if isinstance(raw, Mapping) and "action" in raw:
            raw = raw["action"]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim == 3:
            if values.shape[0] != 1:
                raise ValueError(f"Pi0.5 action batch must have size 1, got {values.shape}")
            values = values[0]
        if values.ndim != 2 or values.shape[1] != self.io_contract.action_dim:
            raise ValueError(
                "Pi0.5 predict_action_chunk must return [batch, chunk, action_dim] "
                f"or [chunk, action_dim], got {values.shape}"
            )
        if values.shape[0] != self.io_contract.action_horizon:
            raise ValueError(
                f"Pi0.5 native chunk must preserve exactly {self.io_contract.action_horizon} steps; "
                f"got {values.shape[0]}"
            )
        return validate_action_chunk(values, self.io_contract)

    def _batched_torch_payload(self, observation: Mapping[str, Any], *, task: str | None) -> dict[str, Any]:
        """Convert raw LIBERO HWC observations to Pi05's batched torch boundary."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("Pi0.5 inference requires torch") from exc
        payload: dict[str, Any] = {}
        for key in self.io_contract.camera_keys:
            image = np.asarray(observation[key])
            tensor = torch.as_tensor(image)
            if tensor.ndim != 3:
                raise ValueError(f"{key} must be a 3-D image, got {tuple(tensor.shape)}")
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            elif tensor.shape[0] != 3:
                raise ValueError(f"{key} must be HWC or CHW RGB, got {tuple(tensor.shape)}")
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32) / 255.0
            else:
                tensor = tensor.to(dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            payload[key] = tensor.contiguous()
        state = torch.as_tensor(np.asarray(observation["observation.state"], dtype=np.float32))
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.io_contract.state_dim:
            raise ValueError(f"observation.state must be batched as [1, 8], got {tuple(state.shape)}")
        payload["observation.state"] = state.contiguous()
        description = task if task is not None else observation.get("task", "")
        payload["task"] = str(description)
        return payload

    def dry_run(self) -> dict[str, Any]:
        return {
            "model": self.artifact.as_dict(),
            "io": self.io_contract.as_dict(),
            "runtime": {"backend": "lerobot", "precision": "bf16", "device": "cuda"},
            "training": {
                "epochs": 15,
                "optimizer_updates": 29180,
                "microbatch_size": 1,
                "gradient_accumulation_steps": 32,
                "train_expert_only": True,
                "gradient_checkpointing": True,
            },
            "status": "configuration_only",
        }
