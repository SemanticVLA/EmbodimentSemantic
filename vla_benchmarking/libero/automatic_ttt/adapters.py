"""Explicit adapter boundary for the four LIBERO VLA families.

The repository already has model-specific evaluators.  This package does not
import them (their dependencies are heavyweight and model revisions differ);
instead, callers register a factory that implements this small contract.  A
missing registration is reported as a configuration error rather than
silently substituting another VLA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ContractError, TransitionRecord
from .fidelity import RuntimeAttestation


SUPPORTED_VLAS = ("openvla", "pi05", "smolvla", "ours")


class AdaptableVLA(Protocol):
    def begin_episode(self, task_description: str, episode_seed: int) -> None: ...
    def reset(self, task_description: str, episode_seed: int) -> None: ...
    def act(self, observation: Mapping[str, Any]) -> Sequence[float]: ...

    def reset_fast_state(self) -> None: ...
    def ingest_teacher_context(self, transitions: Sequence[TransitionRecord]) -> None: ...
    def adapt_fast_state(self, *, attestation: RuntimeAttestation) -> Any: ...

    def adapt_algorithmic_component(self) -> Any: ...
    def fast_state_digest(self) -> str: ...


@dataclass(frozen=True)
class AdapterMetadata:
    name: str
    checkpoint_id: str
    checkpoint_sha256: str | None = None
    action_dim: int = 7
    action_range: tuple[float, float] = (-1.0, 1.0)
    target_task_finetuned: bool = False
    image_preprocessing: Mapping[str, Any] | None = None
    state_layout: str | None = None
    orientation_representation: str | None = None
    instruction_format: str | None = None
    prompt_template: str | None = None
    action_chunk_horizon: int | None = None
    denoising_steps: int | None = None
    processor_revision: str | None = None
    processor_config_digest: str | None = None
    native_action_objective: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_VLAS:
            raise ContractError(f"unsupported VLA adapter {self.name!r}")
        if self.action_dim != 7 or self.action_range != (-1.0, 1.0):
            raise ContractError("LIBERO adapters must expose 7D normalized actions")
        if not self.checkpoint_id:
            raise ContractError("checkpoint_id is required")

    def exact_fields(self) -> dict[str, Any]:
        """Return processor/action semantics needed for paired comparisons."""
        return {
            "image_preprocessing": self.image_preprocessing,
            "state_layout": self.state_layout,
            "orientation_representation": self.orientation_representation,
            "instruction_format": self.instruction_format,
            "prompt_template": self.prompt_template,
            "action_chunk_horizon": self.action_chunk_horizon,
            "denoising_steps": self.denoising_steps,
            "processor_revision": self.processor_revision,
            "processor_config_digest": self.processor_config_digest,
            "native_action_objective": self.native_action_objective,
        }

    def require_exact_fields(self) -> None:
        missing = [name for name, value in self.exact_fields().items() if value in (None, "")]
        if missing:
            raise ContractError("exact adapter metadata has unresolved fields: " + ", ".join(missing))

    def compatibility_key(self) -> str:
        """Stable digest for checking baseline/adapted preprocessing parity."""
        payload = {
            "name": self.name,
            "action_dim": self.action_dim,
            "action_range": list(self.action_range),
            **self.exact_fields(),
        }
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError) as exc:
            raise ContractError("adapter metadata must be losslessly JSON serializable") from exc
        return hashlib.sha256(encoded).hexdigest()

    def assert_compatible(self, other: "AdapterMetadata") -> None:
        if self.compatibility_key() != other.compatibility_key():
            raise ContractError(
                f"baseline/adapted adapter metadata mismatch: {self.compatibility_key()} != {other.compatibility_key()}"
            )


@dataclass(frozen=True)
class AdapterRegistration:
    metadata: AdapterMetadata
    factory: Callable[..., AdaptableVLA]


class VLAAdapterRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, AdapterRegistration] = {}

    def register(self, metadata: AdapterMetadata, factory: Callable[..., AdaptableVLA]) -> None:
        if metadata.name in self._registrations:
            raise ContractError(f"duplicate VLA adapter registration: {metadata.name}")
        if not callable(factory):
            raise TypeError("adapter factory must be callable")
        self._registrations[metadata.name] = AdapterRegistration(metadata, factory)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def require(self, name: str) -> AdapterRegistration:
        canonical = name.lower().replace(".", "")
        aliases = {"openvla": "openvla", "pi05": "pi05", "smolvla": "smolvla", "ours": "ours", "arrow": "ours"}
        canonical = aliases.get(canonical, canonical)
        try:
            return self._registrations[canonical]
        except KeyError as exc:
            raise ContractError(
                f"no adapter registered for {name!r}; registered={self.names()}. "
                "Register the model-specific official processor/checkpoint explicitly."
            ) from exc

    def build(self, name: str, **kwargs: Any) -> AdaptableVLA:
        registration = self.require(name)
        return registration.factory(**kwargs)


class LegacyPolicyBridge:
    """Wrap an existing reset/act adapter without pretending it adapts."""

    def __init__(self, policy: Any, metadata: AdapterMetadata) -> None:
        if not callable(getattr(policy, "reset", None)) or not callable(getattr(policy, "act", None)):
            raise TypeError("legacy policy must expose reset() and act()")
        self.policy = policy
        self.metadata = metadata

    def reset(self, task_description: str, episode_seed: int) -> None:
        self.policy.reset(task_description, episode_seed)

    def begin_episode(self, task_description: str, episode_seed: int) -> None:
        """Canonical lifecycle name; legacy adapters map it to reset()."""
        self.reset(task_description, episode_seed)

    def act(self, observation: Mapping[str, Any]) -> Sequence[float]:
        return self.policy.act(observation)

    def reset_fast_state(self) -> None:
        method = getattr(self.policy, "reset_fast_state", None)
        if not callable(method):
            raise ContractError("policy has no RoboTTT fast-state reset; use it only for frozen plumbing checks")
        method()

    def ingest_teacher_context(self, transitions: Sequence[TransitionRecord]) -> None:
        method = getattr(self.policy, "ingest_teacher_context", None)
        if not callable(method):
            raise ContractError("policy has no teacher-context ingestion hook")
        method(transitions)

    def adapt_fast_state(self, *, attestation: RuntimeAttestation) -> Any:
        if not isinstance(attestation, RuntimeAttestation):
            raise ContractError("exact fast-state adaptation requires a verifier-generated RuntimeAttestation")
        method = getattr(self.policy, "adapt_fast_state", None)
        if not callable(method):
            raise ContractError("legacy policy cannot perform RoboTTT adaptation")
        try:
            return method(attestation=attestation)
        except TypeError as exc:
            raise ContractError(
                "legacy policy adaptation hook must accept attestation=; boolean exactness flags are unsupported"
            ) from exc

    def adapt_algorithmic_component(self) -> Any:
        method = getattr(self.policy, "adapt_algorithmic_component", None)
        if not callable(method):
            raise ContractError("legacy policy has no explicitly labelled algorithmic-component adaptation hook")
        return method()

    def fast_state_digest(self) -> str:
        method = getattr(self.policy, "fast_state_digest", None)
        if not callable(method):
            raise ContractError("policy has no fast-state digest hook")
        return str(method())


__all__ = ["AdaptableVLA", "AdapterMetadata", "AdapterRegistration", "LegacyPolicyBridge", "SUPPORTED_VLAS", "VLAAdapterRegistry"]
