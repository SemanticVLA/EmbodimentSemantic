"""Dependency-tolerant LIBERO environment boundary for the Arrow policy suite.

The suite's runtime only needs ``observe``, ``step``, ``snapshot`` and
``restore``.  This module adapts an already-created LIBERO environment to that
small contract without importing MuJoCo or LIBERO at module import time.  The
raw environment remains owned by the caller; this adapter never creates,
resets, or closes it implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ContractError, Environment, clip_action, digest, state8
from .splits import ResetIdentity


class AdapterUnavailableError(RuntimeError):
    """Raised when an optional production integration is not installed."""


class RawLiberoEnvironment(Protocol):
    def step(self, action: Sequence[float]) -> Any: ...


def _unwrap_observation(value: Any) -> Mapping[str, Any]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        value = value[0]
    if not isinstance(value, Mapping):
        raise ContractError("LIBERO observation must be a mapping")
    return value


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def canonicalize_observation(
    value: Any,
    *,
    instruction: str | None = None,
    require_images: bool = False,
) -> dict[str, Any]:
    """Project a raw or canonical LIBERO observation to the suite contract.

    Raw evaluator observations are normalized through the existing native
    evaluator helper when available.  A direct canonical mapping is accepted
    without NumPy, which keeps fake environments and login-node preflight
    usable.  Unknown fields are deliberately dropped so simulator/controller
    sidecars cannot enter a VLA proposal.
    """

    source = _unwrap_observation(value)
    nested = source.get("pixels", source)
    if not isinstance(nested, Mapping):
        nested = source

    agentview = _first(source, "agentview", "observation.images.image", "image_primary", "image", "agentview_image")
    if agentview is None:
        agentview = _first(nested, "agentview", "observation.images.image", "image_primary", "image", "agentview_image")
    wrist = _first(source, "wrist", "observation.images.image2", "image_wrist", "wrist_image", "eye_in_hand_image", "robot0_eye_in_hand_image")
    if wrist is None:
        wrist = _first(nested, "wrist", "observation.images.image2", "image_wrist", "wrist_image", "eye_in_hand_image", "robot0_eye_in_hand_image")
    raw_state = _first(source, "state", "observation.state")
    if raw_state is None:
        raw_state = _first(nested, "state", "observation.state")

    # Use the stable evaluator canonicalizer only when the direct projection is
    # incomplete.  Importing it here is intentional: NumPy and the native VLA
    # stack are optional for this package.
    if raw_state is None or (require_images and (agentview is None or wrist is None)):
        try:
            from vla_benchmarking.libero.evaluation.native_vla_eval import _canonical_observation
        except (ImportError, ModuleNotFoundError) as exc:
            raise AdapterUnavailableError(
                "native LIBERO observation canonicalizer is unavailable; "
                "provide canonical agentview/wrist/state fields or install the evaluator dependencies"
            ) from exc
        try:
            normalized = _canonical_observation(source, original_openvla=False)
        except Exception as exc:
            raise ContractError("LIBERO observation could not be canonicalized") from exc
        agentview = normalized.get("observation.images.image", agentview)
        wrist = normalized.get("observation.images.image2", wrist)
        raw_state = normalized.get("observation.state", raw_state)

    if raw_state is None:
        raise ContractError("LIBERO observation lacks canonical eight-value state")
    state = state8({"state": raw_state})
    if require_images and (agentview is None or wrist is None):
        raise ContractError("LIBERO VLA observation must expose agentview and wrist images")

    result: dict[str, Any] = {"state": state}
    if agentview is not None:
        result["agentview"] = agentview
    if wrist is not None:
        result["wrist"] = wrist
    if instruction is None:
        candidate = source.get("instruction", source.get("task"))
        if isinstance(candidate, str) and candidate.strip():
            instruction = candidate
    if instruction is not None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ContractError("instruction must be non-empty text")
        result["instruction"] = instruction
    return result


def _read_raw_observation(environment: Any) -> Mapping[str, Any]:
    """Read the authoritative observation without touching simulator state."""

    try:
        from vla_benchmarking.libero.evaluation.observation import read_raw_observation
        value = read_raw_observation(environment, required=False)
    except (ImportError, ModuleNotFoundError):
        value = None
    if value is not None:
        return _unwrap_observation(value)
    for name in ("observe", "get_observation", "observation"):
        member = getattr(environment, name, None)
        if callable(member):
            return _unwrap_observation(member())
        if isinstance(member, Mapping):
            return _unwrap_observation(member)
    raise ContractError("LIBERO environment exposes no raw observation reader")


def _observation_from_step(result: Any) -> Mapping[str, Any] | None:
    if isinstance(result, tuple) and len(result) in (4, 5) and isinstance(result[0], Mapping):
        return result[0]
    if isinstance(result, Mapping):
        for key in ("observation", "obs", "next_observation"):
            if isinstance(result.get(key), Mapping):
                return result[key]
    return None


@dataclass(frozen=True)
class LiberoSnapshot:
    """Adapter snapshot retaining the raw environment payload and observation."""

    payload: Any
    observation: Mapping[str, Any]


class LiberoEnvironmentAdapter(Environment):
    """Adapt one live LIBERO environment to the Arrow suite Environment API."""

    def __init__(
        self,
        environment: Any,
        *,
        instruction: str | None = None,
        require_images: bool = False,
        snapshot_hook: Callable[[], Any] | None = None,
        restore_hook: Callable[[Any], None] | None = None,
        reset_identity: ResetIdentity | None = None,
    ) -> None:
        if environment is None or not callable(getattr(environment, "step", None)):
            raise TypeError("LIBERO environment must expose step(action)")
        self.raw_environment = environment
        self.instruction = instruction
        self.require_images = bool(require_images)
        self._snapshot_hook = snapshot_hook
        self._restore_hook = restore_hook
        self._reset_identity: ResetIdentity | None = reset_identity
        self._observation = self._canonicalize(_read_raw_observation(environment))

    @classmethod
    def from_live_libero(
        cls,
        environment: Any,
        *,
        instruction: str | None = None,
        require_images: bool = True,
        strict_snapshot: bool = True,
        reset_identity: ResetIdentity | None = None,
    ) -> "LiberoEnvironmentAdapter":
        """Wrap a production ``OffScreenRenderEnv`` with verified rollback.

        The heavy LIBERO/MuJoCo imports remain lazy.  Construction validates
        the live simulator state interface immediately and fails closed before
        the host can issue an action if full observation replay is unavailable.
        """
        from .libero_state import make_offscreen_snapshot_hooks
        snapshot_hook, restore_hook = make_offscreen_snapshot_hooks(
            environment, strict=strict_snapshot,
        )
        return cls(
            environment,
            instruction=instruction,
            require_images=require_images,
            snapshot_hook=snapshot_hook,
            restore_hook=restore_hook,
            reset_identity=reset_identity,
        )

    @classmethod
    def from_reset_identity(
        cls,
        environment: Any,
        identity: ResetIdentity,
        *,
        instruction: str | None = None,
        require_images: bool = False,
        snapshot_hook: Callable[[], Any] | None = None,
        restore_hook: Callable[[Any], None] | None = None,
        reset: bool = True,
    ) -> "LiberoEnvironmentAdapter":
        """Bind one live environment to a predeclared reset identity.

        The raw LIBERO builder remains owned by the caller.  When ``reset`` is
        true, the adapter invokes the raw ``reset`` hook exactly once and
        verifies the resulting student observation digest against the identity.
        Existing evaluators that have already reset the environment can pass
        ``reset=False`` and receive the same digest check without a second
        reset.  This makes reset identity part of the live runtime contract,
        rather than merely a report field.
        """
        if not isinstance(identity, ResetIdentity):
            raise TypeError("identity must be an arrow_policy_suite ResetIdentity")
        adapter = cls(
            environment,
            instruction=instruction,
            require_images=require_images,
            snapshot_hook=snapshot_hook,
            restore_hook=restore_hook,
            reset_identity=identity,
        )
        if reset:
            reset_hook = getattr(environment, "reset", None)
            if not callable(reset_hook):
                raise AdapterUnavailableError("LIBERO environment has no reset() hook")
            # The canonical evaluator already selected the init state before
            # this adapter is constructed.  Preserve the supplied reset
            # identity as an audit contract and forward only a standard seed
            # when the raw environment accepts it.
            try:
                value = reset_hook(seed=int(identity.seed))
            except TypeError:
                value = reset_hook()
            adapter._observation = adapter._canonicalize(value)
        digest_value = digest(adapter._observation)
        if digest_value != identity.observation_sha256:
            raise ContractError(
                "live LIBERO reset observation does not match ResetIdentity "
                f"({digest_value} != {identity.observation_sha256})"
            )
        return adapter

    @property
    def reset_identity(self) -> ResetIdentity | None:
        return self._reset_identity

    def bind_reset_identity(self, identity: ResetIdentity) -> None:
        """Attach an already-verified identity to this live adapter."""
        if not isinstance(identity, ResetIdentity):
            raise TypeError("identity must be an arrow_policy_suite ResetIdentity")
        current_digest = digest(self._observation)
        if current_digest != identity.observation_sha256:
            raise ContractError("current live observation does not match ResetIdentity")
        self._reset_identity = identity

    def _canonicalize(self, value: Any) -> dict[str, Any]:
        return canonicalize_observation(value, instruction=self.instruction, require_images=self.require_images)

    def observe(self) -> Mapping[str, Any]:
        self._observation = self._canonicalize(_read_raw_observation(self.raw_environment))
        return dict(self._observation)

    def step(self, action: Sequence[float]) -> Any:
        normalized = clip_action(action)
        result = self.raw_environment.step(normalized)
        candidate = _observation_from_step(result)
        if candidate is None:
            candidate = _read_raw_observation(self.raw_environment)
        self._observation = self._canonicalize(candidate)
        return result

    def reset(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        reset = getattr(self.raw_environment, "reset", None)
        if not callable(reset):
            raise AdapterUnavailableError("LIBERO environment has no reset() hook")
        value = reset(*args, **kwargs)
        self._observation = self._canonicalize(value)
        if self._reset_identity is not None:
            if digest(self._observation) != self._reset_identity.observation_sha256:
                raise ContractError("LIBERO reset observation does not match bound ResetIdentity")
        return dict(self._observation)

    def snapshot(self) -> LiberoSnapshot:
        hook = self._snapshot_hook or getattr(self.raw_environment, "snapshot", None)
        if not callable(hook):
            raise AdapterUnavailableError(
                "LIBERO snapshot is unavailable; inject snapshot_hook for interruptible rollouts"
            )
        return LiberoSnapshot(hook(), dict(self._observation))

    def restore(self, snapshot: LiberoSnapshot | Any) -> None:
        payload = snapshot.payload if isinstance(snapshot, LiberoSnapshot) else snapshot
        hook = self._restore_hook or getattr(self.raw_environment, "restore", None)
        if not callable(hook):
            raise AdapterUnavailableError(
                "LIBERO restore is unavailable; inject restore_hook for interruptible rollouts"
            )
        hook(payload)
        self._observation = self._canonicalize(_read_raw_observation(self.raw_environment))

    def check_success(self) -> Any:
        """Forward LIBERO's explicit post-step evaluator when available."""
        method = getattr(self.raw_environment, "check_success", None)
        return method() if callable(method) else False

    def close(self) -> None:
        close = getattr(self.raw_environment, "close", None)
        if callable(close):
            close()


__all__ = [
    "AdapterUnavailableError", "LiberoEnvironmentAdapter", "LiberoSnapshot",
    "RawLiberoEnvironment", "canonicalize_observation",
]
