"""Dependency-light contracts for the seven-policy Arrow benchmark.

This module intentionally knows nothing about LIBERO, MuJoCo, NumPy, or a
particular policy implementation.  Values returned by an environment are
kept opaque so benchmark adapters can preserve their native metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Protocol, TypeAlias

ACTION_DIM = 7
ACTION_MIN = -1.0
ACTION_MAX = 1.0
CANONICAL_OBSERVATION_SCHEMA = "libero_rgb_state8_instruction_v1"
POLICY_IDS = (
    "arrow_together", "arrow_on_call", "arrow_apprentice", "arrow_editor",
    "arrow_minimal", "arrow_fast", "arrow_trace",
)
STUDENT_OBSERVATION_FIELDS = frozenset(
    {"agentview", "wrist", "state", "observation.state", "instruction"}
)


class ContractError(ValueError):
    """Raised when a benchmark boundary contract is violated."""


_PRIVILEGED_KEY_FRAGMENTS = (
    "bbox", "bounding_box", "segmentation", "contact", "simulator", "mujoco",
    "sim_state", "privileged", "ground_truth", "gt_pose", "object_pose", "object_world",
    "object_state", "teacher_action", "controller_action",
)
_OUTCOME_KEY_NAMES = frozenset({
    "terminal", "terminated", "truncated", "done", "success", "task_success",
    "is_success", "reward", "return", "score",
})


def _is_privileged_key(key_text: str) -> bool:
    """Reject privileged channels and explicit evaluator outcome fields.

    Geometry-controller diagnostics are not policy metadata and are removed
    by the teacher bridge before this contract boundary. Keeping the generic
    boundary strict avoids accidentally allowing adjacent fields such as
    ``contact_mode`` or future contact diagnostics.
    Outcome fields are rejected both by exact name and by explicit
    ``terminal_*``/``success_*``/``reward_*`` naming.
    """
    normalized = str(key_text).lower()
    if normalized in _OUTCOME_KEY_NAMES:
        return True
    if any(fragment in normalized for fragment in _PRIVILEGED_KEY_FRAGMENTS):
        return True
    return any(
        normalized.startswith(f"{prefix}_") or normalized.endswith(f"_{prefix}")
        for prefix in ("terminal", "success", "reward")
    )


def _safe(value: Any) -> Any:
    """Convert metadata to strict JSON-compatible primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite metadata value")
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _safe(tolist())
    raise ContractError(f"metadata value {type(value).__name__} is not JSON safe")


def assert_student_observation(value: Mapping[str, Any], *, path: str = "observation") -> None:
    """Reject simulator/controller side channels recursively."""
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be a mapping")
    for key, item in value.items():
        key_text = str(key).lower()
        if _is_privileged_key(key_text):
            raise ContractError(f"privileged field {path}.{key} is not allowed")
        if isinstance(item, Mapping):
            assert_student_observation(item, path=f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                if isinstance(nested, Mapping):
                    assert_student_observation(nested, path=f"{path}.{key}[{index}]")


def validate_student_observation(
    observation: Mapping[str, Any], *, require_complete: bool = False,
    schema: str = CANONICAL_OBSERVATION_SCHEMA,
) -> None:
    """Validate the canonical student observation without importing NumPy."""
    if schema != CANONICAL_OBSERVATION_SCHEMA:
        raise ContractError(f"unsupported observation schema {schema!r}")
    assert_student_observation(observation)
    unknown = sorted(set(observation) - STUDENT_OBSERVATION_FIELDS)
    if unknown:
        raise ContractError(f"unknown/non-student observation fields: {unknown}")
    if require_complete:
        missing = sorted({"agentview", "wrist", "instruction"} - set(observation))
        if "state" not in observation and "observation.state" not in observation:
            missing.append("state")
        if missing:
            raise ContractError(f"canonical observation missing fields: {missing}")
    if "state" in observation or "observation.state" in observation:
        state8(observation)


def digest(value: Any) -> str:
    """Stable SHA-256 digest for JSON-compatible provenance values."""
    import hashlib
    import json
    payload = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    # Metadata/provenance are not student observations, but they still cross
    # the policy boundary.  Reject simulator/controller side channels at the
    # boundary rather than relying on callers to filter them later.
    assert_student_observation(value, path=name)
    # Shallow copying preserves raw nested values while preventing callers
    # from mutating the contract's top-level metadata after construction.
    return dict(value)


def _freeze(value: Any) -> Any:
    """Recursively freeze common containers for policy-facing views."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    # NumPy is optional.  When an ndarray is supplied, make a detached
    # read-only copy instead of leaking a mutable caller-owned buffer.
    if type(value).__module__.startswith("numpy") and callable(getattr(value, "copy", None)):
        frozen = value.copy()
        setflags = getattr(frozen, "setflags", None)
        if callable(setflags):
            setflags(write=False)
        return frozen
    return value


def validate_action(action: Sequence[float], *, action_dim: int = ACTION_DIM) -> tuple[float, ...]:
    """Validate and normalize one normalized action vector.

    The default LIBERO contract is exactly seven finite values in [-1, 1].
    ``tolist``-based arrays remain supported without importing an array
    library.
    """

    if isinstance(action, (str, bytes)):
        raise ContractError("action must be a numeric sequence")
    try:
        values = tuple(float(item) for item in action)
    except (TypeError, ValueError) as exc:
        raise ContractError("action must be a numeric sequence") from exc
    if len(values) != action_dim:
        raise ContractError(f"expected action dimension {action_dim}, got {len(values)}")
    if any(not math.isfinite(value) for value in values):
        raise ContractError("action contains NaN or infinity")
    if any(value < ACTION_MIN or value > ACTION_MAX for value in values):
        raise ContractError("action is outside normalized range [-1, 1]")
    return values


def clip_action(action: Sequence[float], *, action_dim: int = ACTION_DIM) -> tuple[float, ...]:
    raw = tuple(float(item) for item in action)
    if any(not math.isfinite(value) for value in raw):
        raise ContractError("cannot clip non-finite action")
    if len(raw) != action_dim:
        raise ContractError(f"expected action dimension {action_dim}, got {len(raw)}")
    return validate_action(tuple(max(ACTION_MIN, min(ACTION_MAX, value)) for value in raw), action_dim=action_dim)


def state8(observation: Mapping[str, Any]) -> tuple[float, ...]:
    value = observation.get("state", observation.get("observation.state"))
    if value is None:
        raise ContractError("observation lacks canonical eight-value state")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError("canonical state must contain eight finite values") from exc
    if len(values) != 8 or any(not math.isfinite(item) for item in values):
        raise ContractError("canonical state must contain eight finite values")
    return values


@dataclass(frozen=True)
class ObservationFrame:
    """An observation delivered to a policy, with provenance retained."""

    observation: Mapping[str, Any]
    timestep: int = 0
    episode_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    raw_observation: Mapping[str, Any] = field(init=False, repr=False)
    student_observation: Mapping[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise ContractError("observation must be a mapping")
        if self.timestep < 0:
            raise ContractError("observation timestep must be non-negative")
        raw_observation = dict(self.observation)
        assert_student_observation(raw_observation)
        # Keep arbitrary non-privileged runtime fields available only through
        # an explicit audit handle.  Policies receive the canonical allowlist.
        student = {key: raw_observation[key] for key in STUDENT_OBSERVATION_FIELDS if key in raw_observation}
        validate_student_observation(student)
        frozen_student = _freeze(student)
        object.__setattr__(self, "raw_observation", _freeze(raw_observation))
        object.__setattr__(self, "observation", frozen_student)
        object.__setattr__(self, "student_observation", frozen_student)
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))

    @property
    def step(self) -> int:
        """Legacy alias used by dataset adapters."""
        return self.timestep

    @property
    def digest(self) -> str:
        return digest(self.observation)

    def __getitem__(self, key: str) -> Any:
        return self.observation[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.observation.get(key, default)


@dataclass(frozen=True, init=False)
class ActionProposal:
    """A policy action awaiting execution by the environment owner."""

    action: tuple[float, ...]
    policy_id: str = "policy"
    timestep: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    interruptible: bool = False
    observation_digest: str | None = None

    def __init__(
        self,
        action: Sequence[float],
        policy_id: str = "policy",
        timestep: int | str = 0,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        interruptible: bool = False,
        observation_digest: str | None = None,
    ) -> None:
        # Older suite callers passed the observation digest as the third
        # positional argument.  Keep that spelling source-compatible while
        # making timestep the canonical new field.
        if isinstance(timestep, str):
            if observation_digest is not None and observation_digest != timestep:
                raise ContractError("conflicting proposal observation digests")
            observation_digest = timestep
            timestep = 0
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise ContractError("proposal timestep must be a non-negative integer")
        values = validate_action(action)
        if not policy_id:
            raise ContractError("policy_id is required")
        if timestep < 0:
            raise ContractError("proposal timestep must be non-negative")
        object.__setattr__(self, "action", values)
        object.__setattr__(self, "policy_id", str(policy_id))
        object.__setattr__(self, "timestep", timestep)
        object.__setattr__(self, "metadata", _mapping(metadata, "metadata"))
        object.__setattr__(self, "provenance", _mapping(provenance, "provenance"))
        object.__setattr__(self, "interruptible", bool(interruptible))
        object.__setattr__(self, "observation_digest", observation_digest)

    @property
    def frame_digest(self) -> str | None:
        """Compatibility alias used by the transactional runtime."""
        return self.observation_digest


@dataclass(frozen=True)
class EpisodeSnapshot(Mapping[str, Any]):
    """Opaque post-step environment state.

    ``values`` is never interpreted by this package.  Consumers may retain
    image arrays, tensors, simulator-free fake values, or adapter metadata.
    """

    values: Mapping[str, Any]
    timestep: int = 0
    terminated: bool = False
    truncated: bool = False
    reward: Any = None
    info: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise ContractError("snapshot values must be a mapping")
        if self.timestep < 0:
            raise ContractError("snapshot timestep must be non-negative")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ContractError("snapshot terminal flags must be booleans")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.values))


@dataclass(frozen=True, init=False)
class PolicyDecision:
    """Committed result of executing a proposal."""

    proposal: ActionProposal
    snapshot: EpisodeSnapshot | Any
    raw_result: Any = None
    success: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    teacher_used: bool = False
    teacher_groups: tuple[str, ...] = ()
    policy_id: str = "policy"
    observation_digest: str | None = None

    def __init__(self, first: Any, second: Any = None, third: Any = None,
                 fourth: Any = None, fifth: Any = None, sixth: Any = None,
                 **kwargs: Any) -> None:
        """Accept both the current transaction form and the original policy form.

        Current form: ``PolicyDecision(proposal, snapshot, raw_result, success,
        metadata, provenance)``.  Original form: ``PolicyDecision(action,
        policy_id, observation_digest, teacher_used, teacher_groups, metadata)``.
        Keeping this bridge lets old policy adapters and new runtime adapters be
        tested together while the package is migrated.
        """
        teacher_used = bool(kwargs.pop("teacher_used", False))
        teacher_groups = kwargs.pop("teacher_groups", ())
        keyword_metadata = kwargs.pop("metadata", None)
        keyword_provenance = kwargs.pop("provenance", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected PolicyDecision arguments: {unknown}")
        if isinstance(first, ActionProposal):
            proposal = first
            snapshot = second
            raw_result = third
            success = fourth
            metadata = keyword_metadata if keyword_metadata is not None else (fifth if fifth is not None else {})
            provenance = keyword_provenance if keyword_provenance is not None else (sixth if sixth is not None else {})
            if not isinstance(snapshot, EpisodeSnapshot):
                raise ContractError("decision snapshot must be an EpisodeSnapshot")
            if success is not None and not isinstance(success, bool):
                raise ContractError("decision success must be a boolean or None")
            policy_id = proposal.policy_id
            observation_digest = proposal.observation_digest
        else:
            # Legacy policy decision.  The snapshot is intentionally opaque;
            # it is populated when the coordinator commits a real step.
            proposal = ActionProposal(
                first,
                policy_id=str(second or "policy"),
                observation_digest=str(third) if third is not None else None,
            )
            snapshot = None
            raw_result = None
            success = None
            metadata = keyword_metadata if keyword_metadata is not None else (sixth if sixth is not None else {})
            provenance = keyword_provenance if keyword_provenance is not None else {}
            teacher_used = bool(fourth) if fourth is not None else teacher_used
            teacher_groups = fifth if fifth is not None else teacher_groups
            policy_id = proposal.policy_id
            observation_digest = proposal.observation_digest
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "raw_result", raw_result)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "metadata", _mapping(metadata, "metadata"))
        object.__setattr__(self, "provenance", _mapping(provenance, "provenance"))
        object.__setattr__(self, "teacher_used", bool(teacher_used))
        object.__setattr__(self, "teacher_groups", tuple(teacher_groups or ()))
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "observation_digest", observation_digest)

    @property
    def action(self) -> tuple[float, ...]:
        return self.proposal.action


EnvironmentResult: TypeAlias = Any


@dataclass(frozen=True)
class GeometryAnchors:
    source: tuple[float, float, float]
    destination: tuple[float, float, float]
    frame_name: str
    calibration_revision: str
    provider: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.source) != 3 or len(self.destination) != 3:
            raise ContractError("geometry anchors must be 3D")
        if not self.frame_name or not self.calibration_revision or not self.provider:
            raise ContractError("geometry provenance is incomplete")
        text = str(self.provider).lower() + " " + str(self.provenance).lower()
        if any(token in text for token in ("ground_truth", "sim_ground_truth", "oracle_pose", "object_pose_gt")):
            raise ContractError("ground-truth geometry is not allowed for Trace")
        _safe(self.provenance)


class Environment(Protocol):
    def observe(self) -> Mapping[str, Any]: ...
    def step(self, action: Sequence[float]) -> Any: ...


class VLA(Protocol):
    def propose(self, frame: ObservationFrame) -> ActionProposal: ...


class Teacher(Protocol):
    def propose(self, frame: ObservationFrame) -> ActionProposal | None: ...


ResidualFn = Callable[[ObservationFrame, tuple[float, ...]], Sequence[float]]


@dataclass(frozen=True)
class StepRecord:
    frame: ObservationFrame
    base: ActionProposal
    teacher: ActionProposal | None
    decision: PolicyDecision
    next_frame: ObservationFrame
    result: Any = None
    success: bool = False
    terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ObservationFrame) or not isinstance(self.next_frame, ObservationFrame):
            raise ContractError("step record requires observation frames")
        if not isinstance(self.base, ActionProposal):
            raise ContractError("step record base must be an ActionProposal")
        if self.teacher is not None and not isinstance(self.teacher, ActionProposal):
            raise ContractError("step record teacher must be an ActionProposal or None")
        if not isinstance(self.decision, PolicyDecision):
            raise ContractError("step record decision must be a PolicyDecision")
        for name, proposal in (("base", self.base), ("teacher", self.teacher)):
            if proposal is not None and proposal.observation_digest is not None and proposal.observation_digest != self.frame.digest:
                raise ContractError(f"stale {name} proposal in step record")
        decision_digest = self.decision.observation_digest
        if decision_digest is not None and decision_digest != self.frame.digest:
            raise ContractError("stale policy decision in step record")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))

__all__ = [
    "ACTION_DIM", "ACTION_MAX", "ACTION_MIN", "CANONICAL_OBSERVATION_SCHEMA", "POLICY_IDS",
    "ActionProposal", "ContractError", "EpisodeSnapshot", "Environment", "EnvironmentResult",
    "GeometryAnchors", "ObservationFrame", "PolicyDecision", "ResidualFn", "StepRecord", "Teacher", "VLA",
    "assert_student_observation", "clip_action", "digest", "state8", "validate_action",
    "validate_student_observation",
]
