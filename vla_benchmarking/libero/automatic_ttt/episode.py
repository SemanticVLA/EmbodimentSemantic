"""Single-environment VLA attempt -> Arrow teacher takeover lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import (
    Actor,
    ActionChunk,
    ContractError,
    EpisodeSpec,
    EpisodeStatus,
    LiveEnvironment,
    RecoveryTeacher,
    SourceState,
    SourceStateFn,
    TeacherRecoveryRequest,
    TransitionRecord,
    VLAActionFn,
    normalize_action_chunk,
)
from .recording import JSONLTransitionWriter
from .teacher import PrivilegedTakeoverEnvironmentView, TakeoverEnvironmentView, validate_teacher_transitions
from .dataset import validate_student_observation_schema
from .dataset import CANONICAL_OBSERVATION_SCHEMA
from .demonstrations import validate_and_build_demonstration


@dataclass(frozen=True)
class RolloutResult:
    episode_id: str
    status: EpisodeStatus
    success: bool
    vla_steps: int
    teacher_steps: int
    source_state: SourceState | None
    transitions: tuple[TransitionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EpisodeCoordinator:
    """Own an already-reset environment for the complete experiment episode.

    The coordinator intentionally never calls ``reset`` or ``close``.  The
    caller creates the LIBERO environment once, passes it here, and closes it
    after the returned artifact has been flushed.  This is what preserves the
    failed VLA state for a real teacher takeover.
    """

    def __init__(
        self,
        environment: LiveEnvironment,
        episode: EpisodeSpec,
        *,
        observation_fn: Any | None = None,
        step_fn: Any | None = None,
        writer: JSONLTransitionWriter | None = None,
        observation_schema: str | None = None,
    ) -> None:
        self.environment = environment
        self.episode = episode
        self.observation_fn = observation_fn or (lambda env: env.observe())
        self.step_fn = step_fn or (lambda env, action: env.step(action))
        self.writer = writer
        self.observation_schema = observation_schema
        if writer is not None and observation_schema != CANONICAL_OBSERVATION_SCHEMA:
            raise ContractError(
                "recording requires observation_schema='libero_rgb_state8_instruction_v1'; "
                "raw simulator observations cannot be written as training data"
            )
        self._environment_identity = id(environment)
        self._transitions: list[TransitionRecord] = []

    @property
    def transitions(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._transitions)

    def _observe(self) -> Mapping[str, Any]:
        observation = self.observation_fn(self.environment)
        if not isinstance(observation, Mapping):
            raise ContractError("observation_fn must return a mapping")
        if self.observation_schema is not None:
            validate_student_observation_schema(
                observation, require_complete=True, schema=self.observation_schema
            )
        # TransitionRecord performs the recursive privileged-field check.
        return observation

    def _append(self, record: TransitionRecord) -> None:
        if record.episode_id != self.episode.episode_id:
            raise ContractError("transition belongs to another episode")
        if self._transitions and record.timestep != self._transitions[-1].timestep + 1:
            raise ContractError("transition timesteps must be contiguous")
        self._transitions.append(record)
        if self.writer is not None:
            self.writer.write(record)

    def run_vla_then_teacher(
        self,
        vla_action: VLAActionFn,
        teacher: RecoveryTeacher,
        *,
        success_fn: Any,
        terminal_fn: Any,
        source_state_fn: SourceStateFn,
        vla_step_budget: int,
        teacher_step_budget: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> RolloutResult:
        if vla_step_budget <= 0 or teacher_step_budget <= 0:
            raise ContractError("both VLA and teacher budgets must be positive")
        observation = self._observe()
        vla_steps = 0
        success = False
        terminal = False

        while vla_steps < vla_step_budget:
            chunk = normalize_action_chunk(vla_action(observation, vla_steps), chunk_id=f"vla-chunk-{vla_steps}")
            for chunk_index, action in enumerate(chunk.actions):
                if vla_steps >= vla_step_budget:
                    break
                step_result = self.step_fn(self.environment, action)
                next_observation = self._observe()
                success = success_fn(self.environment, step_result)
                terminal = terminal_fn(self.environment, step_result)
                if not isinstance(success, bool) or not isinstance(terminal, bool):
                    raise ContractError("success_fn and terminal_fn must return booleans")
                record = TransitionRecord(
                    episode_id=self.episode.episode_id,
                    timestep=len(self._transitions),
                    actor=Actor.VLA,
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    done=success or terminal,
                    success=success,
                    training_eligible=False,
                    action_chunk_index=chunk_index,
                    action_chunk_id=chunk.chunk_id,
                    action_chunk_horizon=chunk.horizon,
                    denoising_metadata=chunk.denoising_metadata,
                )
                self._append(record)
                vla_steps += 1
                observation = next_observation
                if success or terminal:
                    break
            if success or terminal:
                break

        if success:
            result = RolloutResult(
                self.episode.episode_id, EpisodeStatus.VLA_SUCCESS, True,
                vla_steps, 0, None, self.transitions, metadata or {},
            )
            self._finalize_writer(result)
            return result
        if terminal:
            result = RolloutResult(
                self.episode.episode_id, EpisodeStatus.ABORTED, False,
                vla_steps, 0, SourceState.TERMINAL, self.transitions, metadata or {},
            )
            self._finalize_writer(result)
            return result

        source_state = source_state_fn(self.environment, observation)
        if source_state in {SourceState.UNSAFE, SourceState.TERMINAL}:
            result = RolloutResult(
                self.episode.episode_id, EpisodeStatus.ABORTED, False,
                vla_steps, 0, source_state, self.transitions, metadata or {},
            )
            self._finalize_writer(result)
            return result

        request = TeacherRecoveryRequest(
            episode=self.episode,
            source_state=source_state,
            observation=observation,
            vla_history=tuple(self._transitions),
            remaining_budget=teacher_step_budget,
        )
        view_type = PrivilegedTakeoverEnvironmentView if getattr(teacher, "requires_privileged_environment", False) else TakeoverEnvironmentView
        view = view_type(
            self.environment,
            expected_identity=self._environment_identity,
            episode_id=self.episode.episode_id,
            start_timestep=len(self._transitions),
            source_state=source_state,
        )
        teacher_result = teacher.recover(view, request)
        captured_transitions = validate_teacher_transitions(view, teacher_result.transitions)
        if id(self.environment) != self._environment_identity:
            raise ContractError("environment identity changed during teacher takeover")
        if view.step_count > teacher_step_budget or len(captured_transitions) > teacher_step_budget:
            raise ContractError("teacher exceeded the configured takeover budget")
        if teacher_result.success and not captured_transitions:
            raise ContractError("teacher cannot report success without executed correction transitions")

        validated_demo = None
        if teacher_result.success:
            teacher_metadata = teacher_result.metadata if isinstance(teacher_result.metadata, Mapping) else {}
            evaluator_success = teacher_metadata.get("evaluator_success", teacher_result.success)
            validated_demo = validate_and_build_demonstration(
                tuple(self._transitions) + tuple(captured_transitions),
                task_id=self.episode.task_id,
                seed=self.episode.seed,
                environment_identity=str(self._environment_identity),
                teacher_success=True,
                evaluator_success=bool(evaluator_success),
                source_controller=str(getattr(teacher_result, "teacher_id", "arrow_grasp_controller")),
                provenance=dict(teacher_metadata),
            )

        previous_timestep = len(self._transitions) - 1
        for offset, record in enumerate(captured_transitions):
            if record.episode_id != self.episode.episode_id:
                raise ContractError("teacher returned a transition for another episode")
            expected_timestep = previous_timestep + offset + 1
            if record.timestep != expected_timestep:
                raise ContractError(
                    f"teacher timestep {record.timestep} does not continue VLA timestep {previous_timestep}"
                )
            self._append(record)

        status = EpisodeStatus.TEACHER_SUCCESS if teacher_result.success else EpisodeStatus.TEACHER_FAILED
        result = RolloutResult(
            self.episode.episode_id,
            status,
            teacher_result.success,
            vla_steps,
            len(captured_transitions),
            source_state,
            self.transitions,
            {
                **(metadata or {}),
                "teacher": teacher_result.metadata,
                "demonstration_receipt": validated_demo.receipt.to_json() if validated_demo else None,
            },
        )
        self._finalize_writer(result)
        return result

    def _finalize_writer(self, result: RolloutResult) -> None:
        if self.writer is not None:
            self.writer.finalize(status=result.status.value, metadata=result.metadata)


__all__ = ["EpisodeCoordinator", "RolloutResult"]
