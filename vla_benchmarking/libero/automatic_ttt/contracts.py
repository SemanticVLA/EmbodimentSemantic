"""Dependency-light contracts for automatic LIBERO test-time training.

The package deliberately does not import a VLA, MuJoCo, or the Arrow controller.
Those systems are supplied through the protocols below.  Keeping the boundary
small makes it possible to test collection and provenance without a simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TypeAlias


ACTION_DIM = 7
ACTION_MIN = -1.0
ACTION_MAX = 1.0
SCHEMA_VERSION = "automatic-ttt-transition-v1"


class ContractError(ValueError):
    """Raised when a collection or training-boundary contract is violated."""


class Actor(str, Enum):
    VLA = "vla"
    TEACHER = "arrow_grasp_controller"


class SourceState(str, Enum):
    SOURCE_UNHELD = "source_unheld"
    SOURCE_HELD = "source_held"
    WRONG_OBJECT_HELD = "wrong_object_held"
    UNKNOWN = "unknown"
    STALE = "stale"
    UNSAFE = "unsafe"
    TERMINAL = "terminal"


class EpisodeStatus(str, Enum):
    VLA_SUCCESS = "vla_success"
    TEACHER_SUCCESS = "teacher_success"
    TEACHER_FAILED = "teacher_failed"
    ABORTED = "aborted"


def _json_safe(value: Any) -> Any:
    """Convert common array/scalar values to strict JSON-compatible values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite float cannot be serialized")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # numpy and torch tensors expose tolist without requiring either dependency.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    raise ContractError(f"value of type {type(value).__name__} is not JSON-safe")


_PRIVILEGED_KEY_FRAGMENTS = (
    "bbox",
    "bounding_box",
    "segmentation",
    "contact",
    "simulator",
    "privileged",
    "object_pose_gt",
    "ground_truth",
    "gt_pose",
)


def assert_student_observation(value: Mapping[str, Any], *, path: str = "observation") -> None:
    """Reject simulator/controller side channels from student observations."""

    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be a mapping")
    for key, item in value.items():
        key_text = str(key).lower()
        if any(fragment in key_text for fragment in _PRIVILEGED_KEY_FRAGMENTS):
            raise ContractError(f"privileged field {path}.{key} is not allowed")
        if isinstance(item, Mapping):
            assert_student_observation(item, path=f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                if isinstance(nested, Mapping):
                    assert_student_observation(nested, path=f"{path}.{key}[{index}]")


def validate_action(action: Sequence[float], *, action_dim: int = ACTION_DIM) -> tuple[float, ...]:
    """Validate normalized LIBERO OSC action vectors."""

    if isinstance(action, (str, bytes)):
        raise ContractError("action must be a numeric sequence")
    try:
        values = tuple(float(item) for item in action)
    except (TypeError, ValueError) as exc:
        raise ContractError("action must be a numeric sequence") from exc
    if len(values) != action_dim:
        raise ContractError(f"expected action dimension {action_dim}, got {len(values)}")
    if not all(math.isfinite(item) for item in values):
        raise ContractError("action contains NaN or infinity")
    if not all(ACTION_MIN <= item <= ACTION_MAX for item in values):
        raise ContractError("action is outside normalized range [-1, 1]")
    return values


@dataclass(frozen=True)
class ActionChunk:
    """A selected VLA denoising/action chunk, retained as one unit in traces."""

    actions: tuple[tuple[float, ...], ...]
    chunk_id: str
    start_index: int = 0
    denoising_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actions or not self.chunk_id:
            raise ContractError("action chunk requires actions and chunk_id")
        if self.start_index < 0:
            raise ContractError("action chunk start_index must be non-negative")
        for action in self.actions:
            validate_action(action)
        _json_safe(self.denoising_metadata)

    @property
    def horizon(self) -> int:
        return len(self.actions)


def normalize_action_chunk(action: Sequence[float] | ActionChunk, *, chunk_id: str | None = None) -> ActionChunk:
    """Normalize a legacy 7D action or an explicit sequence of actions."""

    if isinstance(action, ActionChunk):
        return action
    if isinstance(action, (str, bytes)):
        raise ContractError("action chunk must be numeric")
    try:
        values = tuple(action)
    except TypeError as exc:
        raise ContractError("action chunk must be a sequence") from exc
    try:
        return ActionChunk((validate_action(values),), chunk_id or "single-step")
    except ContractError:
        if not values:
            raise ContractError("action chunk cannot be empty")
        actions = tuple(validate_action(item) for item in values)
        return ActionChunk(actions, chunk_id or "chunk-0")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    task_id: int
    seed: int
    task_description: str
    policy_id: str
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.episode_id or not self.policy_id or not self.task_description:
            raise ContractError("episode_id, policy_id, and task_description are required")
        if self.split not in {"train", "validation", "test"}:
            raise ContractError("split must be train, validation, or test")


@dataclass(frozen=True)
class TransitionRecord:
    """One executed state transition; no teacher-only data is stored in obs."""

    episode_id: str
    timestep: int
    actor: Actor
    observation: Mapping[str, Any]
    action: tuple[float, ...]
    next_observation: Mapping[str, Any]
    done: bool
    success: bool
    training_eligible: bool
    source_state: SourceState | None = None
    action_chunk_index: int = 0
    action_chunk_id: str = "single-step"
    action_chunk_horizon: int = 1
    denoising_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            actor = self.actor if isinstance(self.actor, Actor) else Actor(self.actor)
        except ValueError as exc:
            raise ContractError(f"unknown actor {self.actor!r}") from exc
        object.__setattr__(self, "actor", actor)
        if self.source_state is not None and not isinstance(self.source_state, SourceState):
            try:
                object.__setattr__(self, "source_state", SourceState(self.source_state))
            except ValueError as exc:
                raise ContractError(f"unknown source state {self.source_state!r}") from exc
        if self.timestep < 0 or self.action_chunk_index < 0:
            raise ContractError("timestep and action_chunk_index must be non-negative")
        if not self.action_chunk_id or self.action_chunk_horizon <= 0 or self.action_chunk_index >= self.action_chunk_horizon:
            raise ContractError("invalid action chunk identity/index/horizon")
        _json_safe(self.denoising_metadata)
        validate_action(self.action)
        assert_student_observation(self.observation, path="observation")
        assert_student_observation(self.next_observation, path="next_observation")
        if self.actor is Actor.TEACHER and not self.training_eligible:
            raise ContractError("teacher transitions must be training eligible")

    def to_json(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class TeacherRecoveryRequest:
    episode: EpisodeSpec
    source_state: SourceState
    observation: Mapping[str, Any]
    vla_history: tuple[TransitionRecord, ...]
    remaining_budget: int

    def __post_init__(self) -> None:
        if self.source_state not in {SourceState.SOURCE_UNHELD, SourceState.SOURCE_HELD}:
            raise ContractError(
                "teacher takeover requires a fresh, classified source state: source_unheld or source_held"
            )
        assert_student_observation(self.observation)
        if self.remaining_budget <= 0:
            raise ContractError("teacher takeover requires a positive remaining budget")
        if any(record.episode_id != self.episode.episode_id for record in self.vla_history):
            raise ContractError("VLA history contains another episode")


@dataclass(frozen=True)
class TeacherRecoveryResult:
    transitions: tuple[TransitionRecord, ...]
    success: bool
    status: EpisodeStatus
    teacher_id: str = "arrow_grasp_controller"
    teacher_privilege: str = "simulator_bbox_and_contact_state"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.teacher_id or not self.teacher_privilege:
            raise ContractError("teacher identity and privilege provenance are required")
        if not isinstance(self.success, bool):
            raise ContractError("teacher recovery success must be a boolean")
        if not isinstance(self.status, EpisodeStatus):
            try:
                object.__setattr__(self, "status", EpisodeStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ContractError("teacher recovery status is invalid") from exc
        if any(record.actor is not Actor.TEACHER for record in self.transitions):
            raise ContractError("teacher recovery result may contain only teacher transitions")
        if any(not record.training_eligible for record in self.transitions):
            raise ContractError("all teacher transitions must be training eligible")
        _json_safe(self.metadata)


class LiveEnvironment(Protocol):
    """Minimal live environment surface used by the coordinator and teacher."""

    def observe(self) -> Mapping[str, Any]: ...

    def step(self, action: Sequence[float]) -> Any: ...


ObservationFn: TypeAlias = Callable[[LiveEnvironment], Mapping[str, Any]]
SuccessFn: TypeAlias = Callable[[LiveEnvironment, Any], bool]
TerminalFn: TypeAlias = Callable[[LiveEnvironment, Any], bool]
SourceStateFn: TypeAlias = Callable[[LiveEnvironment, Mapping[str, Any]], SourceState]
VLAActionFn: TypeAlias = Callable[[Mapping[str, Any], int], Sequence[float] | ActionChunk]


class RecoveryTeacher(Protocol):
    teacher_id: str

    def recover(self, environment: LiveEnvironment, request: TeacherRecoveryRequest) -> TeacherRecoveryResult: ...


def serialize_records(records: Iterable[TransitionRecord]) -> list[dict[str, Any]]:
    return [record.to_json() for record in records]


__all__ = [
    "ACTION_DIM", "ActionChunk", "Actor", "ContractError", "EpisodeSpec", "EpisodeStatus",
    "LiveEnvironment", "RecoveryTeacher", "SCHEMA_VERSION", "SourceState",
    "TeacherRecoveryRequest", "TeacherRecoveryResult", "TerminalFn", "TransitionRecord",
    "VLAActionFn", "ObservationFn", "SourceStateFn", "SuccessFn", "assert_student_observation",
    "normalize_action_chunk", "serialize_records", "validate_action",
]
