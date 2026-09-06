"""Import-safe native OpenVLA-OFT policy adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import OPENVLA_ARTIFACT, OPENVLA_IO, PolicyArtifact, PolicyIOContract, validate_action_chunk, validate_observation

try:
    from vla_benchmarking.libero.evaluation.policy_adapter import (
        PolicyProvenance,
        PolicyMetadata,
        validate_arrow_free_observation,
        validate_policy_action,
    )
except ImportError:  # pragma: no cover - supports isolated package copying
    PolicyProvenance = None  # type: ignore[assignment,misc]
    PolicyMetadata = None  # type: ignore[assignment,misc]
    validate_arrow_free_observation = None  # type: ignore[assignment]
    validate_policy_action = None  # type: ignore[assignment]


@dataclass(frozen=True)
class OpenVLAOFTNativeRuntime:
    """Explicit hooks around the upstream OpenVLA-OFT inference stack.

    The callback follows the upstream ``get_vla_action`` keyword contract.  It
    must return the already unnormalized ``[8, 7]`` chunk; the adapter applies
    OFT's LIBERO gripper postprocessing afterward.
    """

    config: Any
    processor: Any
    action_head: Any
    proprio_projector: Any
    get_vla_action: Callable[..., Any]
    noisy_action_projector: Any | None = None
    use_film: bool = False
    unnormalize_action: Callable[[Any], Any] | None = None
    invert_gripper: bool = True


class OpenVLAOFTAdapter:
    """Adapter for the native OpenVLA-OFT action-prediction API."""

    def __init__(
        self,
        policy: Any | None = None,
        *,
        artifact: PolicyArtifact = OPENVLA_ARTIFACT,
        io_contract: PolicyIOContract = OPENVLA_IO,
        native_runtime: OpenVLAOFTNativeRuntime | None = None,
        provenance: Any | None = None,
    ) -> None:
        self.policy = policy
        self.artifact = artifact
        self.io_contract = io_contract
        self.native_runtime = native_runtime
        self.provenance = provenance

    def _provenance_extra(self) -> dict[str, Any]:
        """Normalize optional v2 receipt provenance into metadata extras.

        Production callers may pass the shared ``PolicyProvenance`` object or
        a receipt mapping with ``runtime``, ``io``, and ``dataset_manifest``
        entries.  A flat mapping of already-normalized identity fields is also
        accepted for lightweight integrations.  Omitting provenance preserves
        the historical metadata contract exactly.
        """
        if self.provenance is None:
            return {}
        as_extra = getattr(self.provenance, "as_extra", None)
        if callable(as_extra):
            result = as_extra()
            if not isinstance(result, Mapping):
                raise TypeError("OpenVLA-OFT provenance as_extra() must return a mapping")
            return dict(result)
        if not isinstance(self.provenance, Mapping):
            raise TypeError("OpenVLA-OFT provenance must be PolicyProvenance or a mapping")
        receipt_keys = {"runtime", "io", "dataset_manifest"}
        if receipt_keys.issubset(self.provenance):
            if PolicyProvenance is None:  # pragma: no cover - isolated package copy
                raise RuntimeError("shared policy provenance contracts are unavailable")
            dataset_receipt = dict(self.provenance["dataset_manifest"])
            # OpenVLA preflight receipts name the content/file digests
            # explicitly; normalize the content digest to the shared receipt
            # field while preserving the original mapping for callers.
            if not any(
                dataset_receipt.get(key)
                for key in ("dataset_manifest_sha256", "manifest_sha256", "sha256", "receipt_sha256")
            ):
                for key in ("manifest_content_sha256", "manifest_file_sha256"):
                    if dataset_receipt.get(key):
                        dataset_receipt["manifest_sha256"] = dataset_receipt[key]
                        break
            resolved = PolicyProvenance.from_receipts(
                artifact=self.artifact.as_dict(),
                runtime=self.provenance["runtime"],
                io=self.provenance["io"],
                dataset_manifest=dataset_receipt,
            )
            return resolved.as_extra()
        aliases = {
            "runtime_id": "runtime_id",
            "runtime_sha256": "runtime_sha256",
            "runtime_contract_sha256": "runtime_sha256",
            "io_id": "io_id",
            "io_sha256": "io_sha256",
            "io_contract_sha256": "io_sha256",
            "dataset_manifest_id": "dataset_manifest_id",
            "dataset_manifest_sha256": "dataset_manifest_sha256",
            "manifest_sha256": "dataset_manifest_sha256",
            "manifest_content_sha256": "dataset_manifest_sha256",
        }
        return {
            target: self.provenance[source]
            for source, target in aliases.items()
            if self.provenance.get(source) is not None
        }

    @property
    def metadata(self):
        if PolicyMetadata is None:  # pragma: no cover
            raise RuntimeError("shared evaluation policy contracts are unavailable")
        metadata = PolicyMetadata(
            policy_kind="openvla_oft",
            artifact_id=self.artifact.model_id,
            checkpoint_revision=self.artifact.revision,
            backend="openvla_oft_native",
            native_action_horizon=self.io_contract.action_horizon,
            input_resolution=self.io_contract.image_size[0],
            camera_keys=self.io_contract.camera_keys,
            state_dim=self.io_contract.state_dim,
            extra={
                "uses_language": True,
                "lora_rank": 32,
                "action_loss": "l1_continuous",
                **self._provenance_extra(),
            },
        )
        return metadata

    @classmethod
    def load(
        cls,
        artifact: PolicyArtifact = OPENVLA_ARTIFACT,
        *,
        loader: Callable[[PolicyArtifact], Any] | None = None,
        config: Any | None = None,
        provenance: Any | None = None,
    ) -> "OpenVLAOFTAdapter":
        if loader is not None:
            return cls(loader(artifact), artifact=artifact, provenance=provenance)
        try:
            from experiments.robot.libero.run_libero_eval import GenerateConfig  # type: ignore
            from experiments.robot.openvla_utils import (  # type: ignore
                get_action_head,
                get_processor,
                get_proprio_projector,
                get_vla,
                get_vla_action,
            )
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenVLA-OFT runtime requires the pinned upstream repository and "
                "its LIBERO environment; install it before loading a policy"
            ) from exc
        if config is None:
            config = GenerateConfig(
                # A matched evaluation must load the exact local fine-tuned
                # checkpoint supplied by the caller.  ``artifact.model_id``
                # remains the immutable provenance label, while
                # ``checkpoint_path`` is the runtime load location.
                pretrained_checkpoint=artifact.checkpoint_path or artifact.model_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_images_in_input=2,
                use_proprio=True,
                load_in_8bit=False,
                load_in_4bit=False,
                center_crop=True,
                num_open_loop_steps=NUM_ACTIONS_CHUNK,
                unnorm_key="libero_spatial_no_noops",
            )
        try:
            policy = get_vla(config)
            processor = get_processor(config)
            action_head = get_action_head(config, llm_dim=policy.llm_dim)
            proprio_projector = get_proprio_projector(
                config, llm_dim=policy.llm_dim, proprio_dim=PROPRIO_DIM
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise RuntimeError("unable to load OpenVLA-OFT with its upstream runtime") from exc
        return cls(
            policy,
            artifact=artifact,
            native_runtime=OpenVLAOFTNativeRuntime(
                config=config,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                get_vla_action=get_vla_action,
                use_film=bool(getattr(config, "use_film", False)),
            ),
            provenance=provenance,
        )

    def validate_observation(self, observation: Mapping[str, Any]) -> None:
        validate_observation(observation, self.io_contract)

    def reset(self, task_description: str, episode_seed: int) -> None:
        if self.policy is None:
            raise RuntimeError("OpenVLA-OFT adapter has no loaded policy")
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._task_description = str(task_description)
        self._episode_seed = int(episode_seed)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Shared-evaluator entry point, preserving the native 8-step chunk."""
        if validate_arrow_free_observation is not None:
            validate_arrow_free_observation(observation)
        raw = self.predict(observation, task=getattr(self, "_task_description", None))
        if validate_policy_action is None:  # pragma: no cover
            return raw
        return validate_policy_action(raw, self.metadata, observation=observation)

    def predict(self, observation: Mapping[str, Any], *, task: str | None = None) -> np.ndarray:
        self.validate_observation(observation)
        if self.policy is None:
            raise RuntimeError("OpenVLA-OFT adapter has no loaded policy")
        payload = dict(observation)
        if task is not None:
            payload["task"] = task
        if self.native_runtime is not None:
            runtime = self.native_runtime
            raw = runtime.get_vla_action(
                cfg=runtime.config,
                vla=self.policy,
                processor=runtime.processor,
                obs={
                    "full_image": payload[self.io_contract.camera_keys[0]],
                    "wrist_image": payload[self.io_contract.camera_keys[1]],
                    "state": payload["observation.state"],
                    "task_description": task or payload.get("task", ""),
                },
                task_label=task or payload.get("task", ""),
                action_head=runtime.action_head,
                proprio_projector=runtime.proprio_projector,
                noisy_action_projector=runtime.noisy_action_projector,
                use_film=runtime.use_film,
            )
            if runtime.unnormalize_action is not None:
                raw = runtime.unnormalize_action(raw)
            if runtime.invert_gripper:
                raw = self._invert_gripper(raw)
        elif hasattr(self.policy, "predict_action"):
            raw = self.policy.predict_action(payload)
        else:
            raise RuntimeError(
                "OpenVLA-OFT requires OpenVLAOFTNativeRuntime hooks or a native "
                "predict_action method; get_action/callable fallbacks are disabled"
            )
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        raw = np.asarray(raw)
        if raw.shape == (1, self.io_contract.action_horizon, self.io_contract.action_dim):
            raw = raw[0]
        return validate_action_chunk(raw, self.io_contract)

    @staticmethod
    def _invert_gripper(action: Any) -> Any:
        """Normalize/binzarize and invert OFT's gripper convention for LIBERO."""
        if hasattr(action, "detach") and hasattr(action, "cpu"):
            action = action.detach().cpu().numpy()
        values = np.asarray(action, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError(f"OpenVLA-OFT native action must be [horizon,7], got {values.shape}")
        gripper = values[:, 6]
        # Upstream get_vla_action normally returns the unnormalized dataset
        # convention [0, 1].  Some injected runtimes return [-1, 1] already;
        # support that explicit post-unnormalization form without silently
        # applying a second affine transform.
        if np.all((gripper >= 0.0) & (gripper <= 1.0)):
            values[:, 6] = np.where(gripper >= 0.5, -1.0, 1.0)
        else:
            values[:, 6] = -gripper
        return values

    def dry_run(self) -> dict[str, Any]:
        return {
            "model": self.artifact.as_dict(),
            "io": self.io_contract.as_dict(),
            "runtime": {"backend": "openvla_oft_native", "precision": "bf16", "device": "cuda"},
            "training": {
                "epochs": 15,
                "optimizer_updates": 29180,
                "microbatch_size": 1,
                "gradient_accumulation_steps": 32,
                "effective_batch_size": 32,
                "lora_rank": 32,
                "action_loss": "l1_continuous",
                "serialization": "rlds",
            },
            "status": "configuration_only",
        }
