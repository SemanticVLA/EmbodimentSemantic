"""Fail-closed runtime contracts for automatic TTT experiments.

The package intentionally does not know how to construct a LIBERO simulator,
VLA, or Arrow controller.  A deployment supplies a :class:`RuntimeFactory`
with those operations.  This module owns only lifecycle and provenance: it
creates one environment, resets it once, exposes no reset/close operation to
the experiment callback, and closes it once even when setup fails.

The two state identifiers in :class:`EnvironmentIdentity` are deliberately
different.  ``observation_hash`` hashes the student-visible observation;
``simulator_state_digest`` is supplied by the runtime from its simulator state
serializer.  A replay key is retained separately so an experiment can replay
the same initial state without treating an observation as simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

from .contracts import ContractError, EpisodeSpec, _json_safe


class RuntimeHostError(RuntimeError):
    """Base class for runtime lifecycle and contract failures."""


class RuntimeUnavailableError(RuntimeHostError):
    """Raised when no concrete simulator/runtime factory was supplied."""


class RuntimeContractError(RuntimeHostError, ContractError):
    """Raised when a factory or operation violates a typed boundary."""


class OperationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    ERROR = "error"


ResultT = TypeVar("ResultT")


def _sha256_payload(value: Any) -> str:
    try:
        payload = _json_safe(value)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, ContractError) as exc:
        raise RuntimeContractError("value cannot be hashed as canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def observation_hash(observation: Mapping[str, Any]) -> str:
    """Hash only the student-visible observation payload."""

    if not isinstance(observation, Mapping):
        raise RuntimeContractError("reset observation must be a mapping")
    return _sha256_payload(observation)


def _validate_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RuntimeContractError(f"{name} must be a SHA-256 hexadecimal digest")
    return value.lower()


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Immutable identity for one reset initial state."""

    environment_id: str
    task_id: int
    seed: int
    observation_hash: str
    simulator_state_digest: str
    replay_key: str
    observation_hash_source: str = "student_observation"
    simulator_state_digest_source: str = "simulator_state_serializer"

    def __post_init__(self) -> None:
        if not self.environment_id or not isinstance(self.environment_id, str):
            raise RuntimeContractError("environment_id is required")
        if isinstance(self.task_id, bool) or self.task_id < 0:
            raise RuntimeContractError("task_id must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RuntimeContractError("seed must be an integer")
        _validate_digest("observation_hash", self.observation_hash)
        _validate_digest("simulator_state_digest", self.simulator_state_digest)
        if self.observation_hash.lower() == self.simulator_state_digest.lower():
            raise RuntimeContractError("simulator state digest must be distinct from observation hash")
        if not self.replay_key or not isinstance(self.replay_key, str):
            raise RuntimeContractError("replay_key is required")
        if self.observation_hash_source == self.simulator_state_digest_source:
            raise RuntimeContractError("observation and simulator identity sources must be distinct")
        if self.observation_hash_source != "student_observation":
            raise RuntimeContractError("observation_hash_source must identify the student observation")
        if self.simulator_state_digest_source != "simulator_state_serializer":
            raise RuntimeContractError("simulator_state_digest_source must identify simulator serialization")


@runtime_checkable
class RuntimeFactory(Protocol):
    """Typed factory boundary supplied by a concrete LIBERO deployment.

    ``create_environment`` must not reset or close the returned object.  The
    host is the sole owner of those lifecycle calls.  Implementations should
    make ``simulator_state_digest`` independent of the student observation.
    """

    factory_id: str
    factory_version: str

    def create_environment(self, episode: EpisodeSpec) -> Any: ...

    def reset_environment(self, environment: Any, episode: EpisodeSpec) -> Mapping[str, Any]: ...

    def observe_environment(self, environment: Any) -> Mapping[str, Any]: ...

    def step_environment(self, environment: Any, action: Sequence[float]) -> Mapping[str, Any]: ...

    def simulator_state_digest(self, environment: Any) -> str: ...

    def replay_key(self, environment: Any, episode: EpisodeSpec) -> str: ...

    def close_environment(self, environment: Any) -> None: ...


class EnvironmentView:
    """Student operation view; lifecycle methods are intentionally absent."""

    __slots__ = ("__host", "__lease")

    def __init__(self, host: "RuntimeHost", lease: "EnvironmentLease") -> None:
        self.__host = host
        self.__lease = lease

    @property
    def identity(self) -> EnvironmentIdentity:
        return self.__lease.identity

    def observe(self) -> Mapping[str, Any]:
        return self.__host._observe(self.__lease)

    def step(self, action: Sequence[float]) -> Mapping[str, Any]:
        return self.__host._step(self.__lease, action)


@dataclass(frozen=True)
class OperationResult(Generic[ResultT]):
    """Closed result algebra returned by every host operation.

    Arbitrary mappings are not accepted by :meth:`RuntimeHost.execute`; this
    prevents a controller manifest's ``success`` key from being mistaken for
    a typed evaluator verdict.
    """

    operation_id: str
    status: OperationStatus
    value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    environment_identity: EnvironmentIdentity | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise RuntimeContractError("operation_id is required")
        try:
            status = self.status if isinstance(self.status, OperationStatus) else OperationStatus(self.status)
        except ValueError as exc:
            raise RuntimeContractError("unknown operation status") from exc
        object.__setattr__(self, "status", status)
        if status is OperationStatus.SUCCESS and self.error_type is not None:
            raise RuntimeContractError("successful operation cannot carry an error")
        if status is not OperationStatus.SUCCESS and not self.error_type:
            raise RuntimeContractError("non-success operation requires error_type")
        if status is OperationStatus.SUCCESS and self.value is None:
            raise RuntimeContractError("successful operation requires a typed value")
        if not isinstance(self.metadata, Mapping):
            raise RuntimeContractError("operation metadata must be a mapping")
        _json_safe(self.metadata)

    @classmethod
    def success(cls, operation_id: str, value: Any, *, environment_identity: EnvironmentIdentity | None = None, metadata: Mapping[str, Any] | None = None) -> "OperationResult":
        if isinstance(value, Mapping):
            raise RuntimeContractError("operation success value must be a typed object, not a mapping")
        return cls(operation_id, OperationStatus.SUCCESS, value, environment_identity=environment_identity, metadata=metadata or {})

    @classmethod
    def failure(cls, operation_id: str, status: OperationStatus, error_type: str, error_message: str, *, environment_identity: EnvironmentIdentity | None = None, metadata: Mapping[str, Any] | None = None) -> "OperationResult":
        if status is OperationStatus.SUCCESS:
            raise RuntimeContractError("failure factory cannot create success")
        return cls(operation_id, status, None, error_type, error_message, environment_identity, metadata or {})


@dataclass(frozen=True)
class RuntimePreflightReceipt:
    factory_id: str
    factory_version: str
    episode_id: str
    identity: EnvironmentIdentity
    reset_calls: int
    close_calls: int
    status: str = "passed"

    def __post_init__(self) -> None:
        if not self.factory_id or not self.factory_version or not self.episode_id:
            raise RuntimeContractError("preflight receipt identity is incomplete")
        if self.reset_calls != 1 or self.close_calls != 1:
            raise RuntimeContractError("preflight must perform exactly one reset and one close")
        if self.status != "passed":
            raise RuntimeContractError("only passed preflight receipts may be issued")


class EnvironmentLease:
    """Host-owned lifecycle lease; only ``RuntimeHost`` may close it."""

    __slots__ = ("_host", "_environment", "identity", "initial_observation", "_closed", "_view")

    def __init__(self, host: "RuntimeHost", environment: Any, identity: EnvironmentIdentity, initial_observation: Mapping[str, Any]) -> None:
        self._host = host
        self._environment = environment
        self.identity = identity
        self.initial_observation = dict(initial_observation)
        self._closed = False
        self._view = EnvironmentView(host, self)

    @property
    def view(self) -> EnvironmentView:
        if self._closed:
            raise RuntimeHostError("environment lease is closed")
        return self._view

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            raise RuntimeHostError("environment close attempted more than once")
        self._host._close(self)


class RuntimeHost:
    """Lifecycle owner and typed operation boundary for one runtime factory."""

    def __init__(self, factory: RuntimeFactory | None) -> None:
        if factory is None:
            raise RuntimeUnavailableError("a concrete RuntimeFactory is required; no runtime loader is bundled")
        if not isinstance(factory, RuntimeFactory):
            raise RuntimeContractError("factory must implement the RuntimeFactory protocol")
        required = ("create_environment", "reset_environment", "observe_environment", "step_environment", "simulator_state_digest", "replay_key", "close_environment")
        missing = [name for name in required if not callable(getattr(factory, name, None))]
        if missing:
            raise RuntimeContractError(f"runtime factory is missing required methods: {missing}")
        if not isinstance(getattr(factory, "factory_id", None), str) or not factory.factory_id:
            raise RuntimeContractError("runtime factory must expose non-empty factory_id")
        if not isinstance(getattr(factory, "factory_version", None), str) or not factory.factory_version:
            raise RuntimeContractError("runtime factory must expose non-empty factory_version")
        self.factory = factory
        self._active: set[int] = set()
        self._reset_counts: dict[int, int] = {}
        self._close_counts: dict[int, int] = {}

    def open(self, episode: EpisodeSpec) -> EnvironmentLease:
        environment = self.factory.create_environment(episode)
        token = id(environment)
        if token in self._active:
            raise RuntimeContractError("factory returned an environment already owned by this host")
        self._active.add(token)
        self._reset_counts[token] = 0
        self._close_counts[token] = 0
        try:
            initial = self.factory.reset_environment(environment, episode)
            self._reset_counts[token] += 1
            if self._reset_counts[token] != 1:
                raise RuntimeContractError("host must reset each environment exactly once")
            if not isinstance(initial, Mapping):
                raise RuntimeContractError("reset_environment must return a mapping observation")
            # The object returned by reset is not accepted as proof of the
            # actual live state.  Read the environment once through the
            # authoritative observation path and require byte-for-byte
            # canonical equality before any policy action can run.
            first_observation = self.factory.observe_environment(environment)
            if not isinstance(first_observation, Mapping):
                raise RuntimeContractError("observe_environment must return a mapping observation")
            if observation_hash(first_observation) != observation_hash(initial):
                raise RuntimeContractError(
                    "post-reset first observation does not match reset observation; "
                    "scored execution is blocked before the first policy action"
                )
            identity = EnvironmentIdentity(
                environment_id=f"{self.factory.factory_id}:{episode.episode_id}",
                task_id=episode.task_id,
                seed=episode.seed,
                observation_hash=observation_hash(first_observation),
                simulator_state_digest=_validate_digest("simulator_state_digest", self.factory.simulator_state_digest(environment)),
                replay_key=self.factory.replay_key(environment, episode),
            )
            return EnvironmentLease(self, environment, identity, first_observation)
        except Exception:
            self._close_raw(environment, token)
            raise

    def preflight(self, episode: EpisodeSpec) -> RuntimePreflightReceipt:
        lease = self.open(episode)
        try:
            identity = lease.identity
        finally:
            if not lease.closed:
                lease.close()
        return RuntimePreflightReceipt(
            self.factory.factory_id,
            self.factory.factory_version,
            episode.episode_id,
            identity,
            self._reset_counts[id(lease._environment)],
            self._close_counts[id(lease._environment)],
        )

    def execute(self, lease: EnvironmentLease, operation_id: str, operation: Callable[[EnvironmentView], OperationResult]) -> OperationResult:
        if not isinstance(lease, EnvironmentLease) or lease._host is not self:
            raise RuntimeContractError("lease belongs to another runtime host")
        if lease.closed:
            raise RuntimeHostError("cannot execute on a closed environment lease")
        try:
            result = operation(lease.view)
        except Exception as exc:
            return OperationResult.failure(operation_id, OperationStatus.ERROR, type(exc).__name__, str(exc), environment_identity=lease.identity)
        if not isinstance(result, OperationResult):
            raise RuntimeContractError("operation must return OperationResult; arbitrary success mappings are rejected")
        if result.operation_id != operation_id:
            raise RuntimeContractError("operation result ID does not match requested operation")
        if result.environment_identity != lease.identity:
            raise RuntimeContractError("operation result must carry the exact lease environment identity")
        return result

    def _observe(self, lease: EnvironmentLease) -> Mapping[str, Any]:
        self._assert_active(lease)
        value = self.factory.observe_environment(lease._environment)
        if not isinstance(value, Mapping):
            raise RuntimeContractError("observe_environment must return a mapping")
        return value

    def _step(self, lease: EnvironmentLease, action: Sequence[float]) -> Mapping[str, Any]:
        self._assert_active(lease)
        value = self.factory.step_environment(lease._environment, action)
        if not isinstance(value, Mapping):
            raise RuntimeContractError("step_environment must return a mapping")
        return value

    def _assert_active(self, lease: EnvironmentLease) -> None:
        if lease.closed or id(lease._environment) not in self._active:
            raise RuntimeHostError("environment lease is not active")

    def _close(self, lease: EnvironmentLease) -> None:
        self._assert_active(lease)
        try:
            self._close_raw(lease._environment, id(lease._environment))
        finally:
            # A failed close is still one close *attempt*.  Marking the lease
            # closed prevents a caller from invoking a potentially unsafe
            # second close while the original exception is being handled.
            lease._closed = True

    def _close_raw(self, environment: Any, token: int) -> None:
        if self._close_counts.get(token, 0) != 0:
            raise RuntimeHostError("environment close attempted more than once")
        self._close_counts[token] = 1
        try:
            self.factory.close_environment(environment)
        finally:
            self._active.discard(token)


class UnavailableRuntimeFactory:
    """Explicit fail-closed placeholder used when deployment wiring is absent."""

    factory_id = "unavailable"
    factory_version = "0"

    def __init__(self, reason: str = "no concrete LIBERO runtime factory configured") -> None:
        self.reason = reason

    def __getattribute__(self, name: str) -> Any:
        if name in {"factory_id", "factory_version", "reason", "__class__", "__dict__", "__getattribute__"}:
            return object.__getattribute__(self, name)
        raise RuntimeUnavailableError(object.__getattribute__(self, "reason"))


__all__ = [
    "EnvironmentIdentity", "EnvironmentLease", "EnvironmentView", "OperationResult", "OperationStatus",
    "RuntimeContractError", "RuntimeFactory", "RuntimeHost", "RuntimeHostError", "RuntimePreflightReceipt",
    "RuntimeUnavailableError", "UnavailableRuntimeFactory", "observation_hash",
]
