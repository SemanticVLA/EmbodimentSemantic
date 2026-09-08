"""Safe teacher boundary for the existing Arrow grasp controller."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    ACTION_DIM,
    Actor,
    ContractError,
    EpisodeStatus,
    LiveEnvironment,
    SourceState,
    TeacherRecoveryRequest,
    TeacherRecoveryResult,
    TransitionRecord,
    assert_student_observation,
    validate_action,
    _json_safe,
)
from .demonstrations import validate_and_build_demonstration


class TakeoverEnvironmentView:
    """A live-environment view that cannot reset or close the episode.

    The Arrow implementation may be supplied as ``recover_fn`` without making
    it a lifecycle owner.  Any attempt to call reset/close through this view is
    rejected before it reaches the LIBERO environment.
    """

    _BLOCKED = frozenset({
        "reset", "close", "reset_model", "hard_reset", "reload_model",
        "initialize_episode", "set_state", "set_qpos", "set_qvel",
    })

    def __init__(
        self,
        environment: LiveEnvironment,
        *,
        expected_identity: int | None = None,
        episode_id: str | None = None,
        start_timestep: int = 0,
        source_state: SourceState | None = None,
    ) -> None:
        self._environment = environment
        self._expected_identity = expected_identity if expected_identity is not None else id(environment)
        self._step_count = 0
        self._closed = False
        if start_timestep < 0:
            raise ContractError("start_timestep must be non-negative")
        self._episode_id = episode_id
        self._start_timestep = start_timestep
        self._source_state = source_state
        self._executed_transitions: list[TransitionRecord] = []
        self._capture_error: BaseException | None = None

    @property
    def environment_identity(self) -> int:
        return self._expected_identity

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def executed_transitions(self) -> tuple[TransitionRecord, ...]:
        """Atomic records captured from every successful ``step`` call."""

        return tuple(self._executed_transitions)

    def observe(self) -> Mapping[str, Any]:
        self.assert_usable()
        observation = self._environment.observe()
        if not isinstance(observation, Mapping):
            raise ContractError("teacher environment observe() must return a mapping")
        assert_student_observation(observation)
        return observation

    def step(self, action: Sequence[float]) -> Any:
        return self.step_chunk(action)

    def step_chunk(
        self,
        action: Sequence[float],
        *,
        chunk_id: str | None = None,
        chunk_index: int | None = None,
        chunk_horizon: int = 1,
        denoising_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        self.assert_usable()
        before = self.observe()
        action_tuple = validate_action(action)
        result = self._environment.step(action_tuple)
        self._step_count += 1
        try:
            after = self.observe()
        except BaseException as exc:
            # The environment has moved but its transition cannot be safely
            # serialized.  Keep a sticky error so a caller cannot continue
            # and accidentally claim a complete teacher trace.
            self._capture_error = exc
            raise ContractError("teacher post-step observation could not be captured") from exc
        if self._episode_id is not None:
            self._executed_transitions.append(
                TransitionRecord(
                    episode_id=self._episode_id,
                    timestep=self._start_timestep + self._step_count - 1,
                    actor=Actor.TEACHER,
                    observation=before,
                    action=action_tuple,
                    next_observation=after,
                    done=_step_done(result),
                    success=_step_success(result),
                    training_eligible=True,
                    source_state=self._source_state,
                    action_chunk_index=(0 if chunk_index is None else chunk_index),
                    action_chunk_id=chunk_id or f"teacher-step-{self._step_count - 1}",
                    action_chunk_horizon=chunk_horizon,
                    denoising_metadata=dict(denoising_metadata or {}),
                )
            )
        return result

    def assert_usable(self) -> None:
        if self._closed:
            raise ContractError("teacher environment view is closed")
        if id(self._environment) != self._expected_identity:
            raise ContractError("teacher environment identity changed")

    def assert_integrity(self) -> None:
        self.assert_usable()
        if self._capture_error is not None:
            raise ContractError("teacher transition capture failed") from self._capture_error
        if self._episode_id is not None and len(self._executed_transitions) != self._step_count:
            raise ContractError(
                f"captured {len(self._executed_transitions)} teacher transitions for {self._step_count} steps"
            )
        if self._episode_id is not None:
            expected = list(range(self._start_timestep, self._start_timestep + self._step_count))
            actual = [record.timestep for record in self._executed_transitions]
            if actual != expected:
                raise ContractError(f"teacher transition timesteps are not contiguous: {actual}")

    def __getattr__(self, name: str) -> Any:
        if name in self._BLOCKED or name.lower().endswith(("reset", "close")):
            raise ContractError(f"teacher is not allowed to call environment.{name}()")
        # Deliberately do not forward arbitrary attributes: this prevents
        # accidental access to simulator state and keeps the contract explicit.
        raise AttributeError(
            f"TakeoverEnvironmentView exposes only observe(), step(), and environment_identity; {name!r} is unavailable"
        )


class PrivilegedTakeoverEnvironmentView(TakeoverEnvironmentView):
    """Controller-only proxy exposing simulator/render hooks safely.

    The Arrow teacher legitimately needs RGB-D capture, calibration and
    simulator geometry.  Those methods are forwarded here, but lifecycle
    operations remain blocked and the proxy is never passed to the VLA or
    serialized into a student observation.  This makes the privileged boundary
    explicit instead of accidentally bypassing the takeover guard.
    """

    _MUTATING_PREFIXES = (
        "set", "load", "restore", "reset", "reload", "init", "initialize",
        "apply", "write", "update", "mutate", "teleport", "seed",
    )
    _MUTATING_NAMES = frozenset({
        "state", "qpos", "qvel", "model_state", "sim_state", "initial_state",
    })

    def __getattr__(self, name: str) -> Any:
        lowered = name.lower()
        if (
            name in self._BLOCKED
            or lowered.endswith(("reset", "close"))
            or lowered in self._MUTATING_NAMES
            or lowered.startswith(self._MUTATING_PREFIXES)
            or any(token in lowered for token in ("set_state", "load_state", "restore_state", "reset_state"))
        ):
            raise ContractError(f"teacher is not allowed to call environment.{name}()")
        return getattr(self._environment, name)


RecoveryFn = Callable[[TakeoverEnvironmentView, TeacherRecoveryRequest], Any]


def _runtime_boolean(value: Any, *, field: str) -> bool:
    """Normalize Python and NumPy booleans at the simulator boundary."""

    if type(value) is bool:
        return value
    value_type = type(value)
    if value_type.__name__ == "bool_" and value_type.__module__.startswith("numpy"):
        return bool(value)
    raise ContractError(f"environment step {field} field must be boolean")


def _step_done(result: Any) -> bool:
    if isinstance(result, Mapping):
        done = result.get("done", result.get("terminated", False))
        truncated = result.get("truncated", False)
        return _runtime_boolean(done, field="done/terminated") or _runtime_boolean(
            truncated, field="truncated"
        )
    if isinstance(result, tuple) and len(result) >= 4:
        terminated = result[2]
        truncated = result[3] if len(result) >= 5 else False
        return _runtime_boolean(terminated, field="terminated") or _runtime_boolean(
            truncated, field="truncated"
        )
    return False


def _step_success(result: Any) -> bool:
    if isinstance(result, Mapping):
        success = result.get("success", False)
        return _runtime_boolean(success, field="success")
    if isinstance(result, tuple) and len(result) == 5:
        info = result[4] if isinstance(result[4], Mapping) else {}
        success = info.get("success", False)
        return _runtime_boolean(success, field="info.success")
    if isinstance(result, tuple) and len(result) == 4:
        info = result[3] if isinstance(result[3], Mapping) else {}
        success = info.get("success", False)
        return _runtime_boolean(success, field="info.success")
    return False


def _same_execution(left: TransitionRecord, right: TransitionRecord) -> bool:
    """Compare the immutable execution fields, tolerating outcome annotations."""

    return (
        left.episode_id == right.episode_id
        and left.timestep == right.timestep
        and left.actor is right.actor is Actor.TEACHER
        and left.action == right.action
        and _json_safe(left.observation) == _json_safe(right.observation)
        and _json_safe(left.next_observation) == _json_safe(right.next_observation)
        and left.training_eligible == right.training_eligible
    )


def validate_teacher_transitions(
    environment: TakeoverEnvironmentView,
    transitions: Sequence[TransitionRecord],
) -> tuple[TransitionRecord, ...]:
    """Require returned rows to be exactly the rows captured from ``step``."""

    environment.assert_integrity()
    returned = tuple(transitions)
    if len(returned) != environment.step_count:
        raise ContractError(
            f"teacher returned {len(returned)} transitions after {environment.step_count} executed steps"
        )
    for captured, row in zip(environment.executed_transitions, returned):
        if not _same_execution(captured, row):
            raise ContractError("teacher returned transitions that do not match executed environment steps")
    # The captured rows are authoritative.  Returning them prevents a caller
    # from accidentally training on fabricated annotations.
    return environment.executed_transitions


class ArrowGraspControllerTeacher:
    """Adapter around a caller-provided Arrow controller recovery function.

    ``recover_fn`` is normally a thin bridge to ``arrow_grasp_controller``.
    It must execute actions on the supplied live view and return either a
    ``TeacherRecoveryResult`` or a mapping with ``transitions`` and ``success``.
    Controller diagnostics may be returned under ``metadata``; they are kept
    out of observations and therefore cannot leak into student training input.
    """

    def __init__(
        self,
        recover_fn: RecoveryFn,
        *,
        teacher_id: str = "arrow_grasp_controller",
        teacher_privilege: str = "simulator_bbox_and_contact_state",
        requires_privileged_environment: bool = False,
    ) -> None:
        if not callable(recover_fn):
            raise TypeError("recover_fn must be callable")
        self.recover_fn = recover_fn
        self.teacher_id = teacher_id
        self.teacher_privilege = teacher_privilege
        self.requires_privileged_environment = bool(requires_privileged_environment)

    def recover(self, environment: TakeoverEnvironmentView, request: TeacherRecoveryRequest) -> TeacherRecoveryResult:
        return self._recover(environment, request, collection_mode="same_episode_takeover")

    def recover_from_reset(self, environment: TakeoverEnvironmentView, request: TeacherRecoveryRequest) -> TeacherRecoveryResult:
        """Execute Arrow immediately from a freshly reset environment.

        This is deliberately a separate entry point from ``recover``.  It
        keeps the original same-episode takeover contract intact while
        allowing the pilot to collect ordinary successful Arrow rollouts.
        ``request.vla_history`` must be empty and the view must start at
        timestep zero, so no VLA action can be silently mixed into the data.
        """
        if request.vla_history:
            raise ContractError("fresh Arrow demonstration cannot include VLA history")
        return self._recover(environment, request, collection_mode="fresh_arrow")

    def _recover(
        self,
        environment: TakeoverEnvironmentView,
        request: TeacherRecoveryRequest,
        *,
        collection_mode: str,
    ) -> TeacherRecoveryResult:
        if not isinstance(environment, TakeoverEnvironmentView):
            raise TypeError("Arrow teacher requires TakeoverEnvironmentView")
        environment.assert_usable()
        raw = self.recover_fn(environment, request)
        environment.assert_usable()
        result = self._coerce_result(raw, request)
        if self.requires_privileged_environment and not isinstance(environment, PrivilegedTakeoverEnvironmentView):
            raise ContractError("privileged Arrow teacher requires PrivilegedTakeoverEnvironmentView")
        captured = validate_teacher_transitions(environment, result.transitions)
        result = replace(result, transitions=captured)
        if result.teacher_id != self.teacher_id or result.teacher_privilege != self.teacher_privilege:
            result = replace(result, teacher_id=self.teacher_id, teacher_privilege=self.teacher_privilege)
        if result.success:
            teacher_metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
            evaluator_success = teacher_metadata.get("evaluator_success", result.success)
            if not isinstance(evaluator_success, bool):
                raise ContractError("teacher metadata evaluator_success must be a boolean")
            validated = validate_and_build_demonstration(
                tuple(request.vla_history) + tuple(captured),
                task_id=request.episode.task_id,
                seed=request.episode.seed,
                environment_identity=str(environment.environment_identity),
                teacher_success=True,
                evaluator_success=evaluator_success,
                source_controller=self.teacher_id,
                provenance=dict(teacher_metadata),
                collection_mode=collection_mode,
            )
            result = replace(
                result,
                metadata={**dict(teacher_metadata), "demonstration_receipt": validated.receipt.to_json()},
            )
        return result

    def _coerce_result(self, raw: Any, request: TeacherRecoveryRequest) -> TeacherRecoveryResult:
        if isinstance(raw, TeacherRecoveryResult):
            return raw
        if not isinstance(raw, Mapping):
            raise ContractError("Arrow recovery function must return TeacherRecoveryResult or mapping")
        raw_transitions = raw.get("transitions", ())
        if not isinstance(raw_transitions, (list, tuple)):
            raise ContractError("teacher transitions must be a list or tuple")
        transitions: list[TransitionRecord] = []
        for index, item in enumerate(raw_transitions):
            if isinstance(item, TransitionRecord):
                record = item
            elif isinstance(item, Mapping):
                if "observation" not in item or "action" not in item or "next_observation" not in item:
                    raise ContractError("teacher transition requires observation, action, and next_observation")
                record = TransitionRecord(
                    episode_id=request.episode.episode_id,
                    timestep=int(item.get("timestep", len(request.vla_history) + index)),
                    actor=Actor.TEACHER,
                    observation=item["observation"],
                    action=validate_action(item["action"]),
                    next_observation=item["next_observation"],
                    done=_strict_mapping_bool(item, "done"),
                    success=_strict_mapping_bool(item, "success"),
                    training_eligible=True,
                    source_state=request.source_state,
                    action_chunk_index=int(item.get("action_chunk_index", 0)),
                    action_chunk_id=str(item.get("action_chunk_id", f"teacher-step-{index}")),
                    action_chunk_horizon=int(item.get("action_chunk_horizon", 1)),
                    denoising_metadata=item.get("denoising_metadata", {}),
                )
            else:
                raise ContractError("teacher transition must be a mapping or TransitionRecord")
            if record.actor is not Actor.TEACHER or not record.training_eligible:
                raise ContractError("teacher transition actor/training_eligible contract violated")
            transitions.append(record)
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ContractError("teacher metadata must be a mapping")
        _json_safe(metadata)
        status_value = raw.get("status")
        success = _strict_mapping_bool(raw, "success")
        if status_value is None:
            status = EpisodeStatus.TEACHER_SUCCESS if success else EpisodeStatus.TEACHER_FAILED
        else:
            try:
                status = EpisodeStatus(status_value)
            except ValueError as exc:
                raise ContractError(f"unknown teacher status {status_value!r}") from exc
        return TeacherRecoveryResult(
            transitions=tuple(transitions),
            success=success,
            status=status,
            teacher_id=self.teacher_id,
            teacher_privilege=self.teacher_privilege,
            metadata=dict(metadata),
        )


def _strict_mapping_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        raise ContractError(f"teacher field {key!r} must be boolean")
    return value


__all__ = [
    "ArrowGraspControllerTeacher", "TakeoverEnvironmentView", "PrivilegedTakeoverEnvironmentView",
    "RecoveryFn", "validate_teacher_transitions",
]
