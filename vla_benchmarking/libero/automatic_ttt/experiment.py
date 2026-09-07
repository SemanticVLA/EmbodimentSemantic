"""Reproducible experiment orchestration for automatic LIBERO TTT.

This module intentionally owns *experiment bookkeeping*, not simulator/model
construction.  Callers inject the existing VLA adapter, a live LIBERO
environment factory, and the Arrow recovery service.  That keeps the package
self-contained while preventing the two existing command-line runners from
resetting/closing the same environment at takeover.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .contracts import (
    Actor,
    ContractError,
    EpisodeSpec,
    EpisodeStatus,
    LiveEnvironment,
    SourceState,
    TeacherRecoveryRequest,
    TransitionRecord,
    VLAActionFn,
    normalize_action_chunk,
    validate_action,
)
from .teacher import PrivilegedTakeoverEnvironmentView, TakeoverEnvironmentView, validate_teacher_transitions
from .dataset import (
    CANONICAL_OBSERVATION_SCHEMA,
    TTTDataset,
    episode_from_executed_records,
    validate_student_observation_schema,
)
from .demonstrations import validate_and_build_demonstration


class Policy(Protocol):
    def reset(self, task_description: str, episode_seed: int) -> None: ...
    def act(self, observation: Mapping[str, Any]) -> Sequence[float]: ...


class Updater(Protocol):
    def adapt(self, dataset: TTTDataset, *, episode_id: str) -> Any: ...


class EnvironmentFactory(Protocol):
    def __call__(self, spec: EpisodeSpec) -> "EnvironmentHandle | LiveEnvironment": ...


@dataclass
class EnvironmentHandle:
    """Canonical factory result: one selected reset state and one close owner."""

    environment: LiveEnvironment
    initial_observation: Mapping[str, Any]
    initial_state_hash: str
    environment_identity: str
    runtime_identity_verifier: Callable[[LiveEnvironment, Mapping[str, Any]], bool] | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.initial_observation, Mapping):
            raise ContractError("EnvironmentHandle.initial_observation must be a mapping")
        if not self.initial_state_hash or not self.environment_identity:
            raise ContractError("EnvironmentHandle requires state hash and identity")

    def validate_initial_state(self) -> None:
        observed_hash = initial_state_hash(self.initial_observation)
        if observed_hash != self.initial_state_hash:
            raise ContractError(
                f"factory initial-state hash mismatch: expected {self.initial_state_hash}, got {observed_hash}"
            )

    def verify_live_initial_state(self) -> None:
        """Verify the live post-reset state before policy inference begins."""

        if self.runtime_identity_verifier is None:
            raise ContractError(
                "EnvironmentHandle requires runtime_identity_verifier; "
                "declared initial observations alone cannot prove live simulator identity"
            )
        verified = self.runtime_identity_verifier(self.environment, self.initial_observation)
        if not isinstance(verified, bool) or not verified:
            raise ContractError("live environment state does not match the selected reset state")

    def close_once(self) -> None:
        if self._closed:
            raise ContractError("EnvironmentHandle.close_once() called more than once")
        self._closed = True
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class SplitManifest:
    """Pre-registered support/query split; hashes make accidental leakage loud."""

    support_seeds: tuple[int, ...]
    query_seeds: tuple[int, ...]
    support_initial_state_hashes: tuple[str, ...] = ()
    query_initial_state_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.support_seeds or not self.query_seeds:
            raise ContractError("support and query splits must both be non-empty")
        if set(self.support_seeds) & set(self.query_seeds):
            raise ContractError("support/query seeds overlap")
        if set(self.support_initial_state_hashes) & set(self.query_initial_state_hashes):
            raise ContractError("support/query initial states overlap")

    def to_json(self) -> dict[str, Any]:
        return {
            "support_seeds": list(self.support_seeds),
            "query_seeds": list(self.query_seeds),
            "support_initial_state_hashes": list(self.support_initial_state_hashes),
            "query_initial_state_hashes": list(self.query_initial_state_hashes),
        }


@dataclass(frozen=True)
class EpisodeRunConfig:
    """Fixed controls for one paired arm comparison.

    The declarative multi-VLA configuration lives in :mod:`.config` and is
    intentionally named ``ExperimentConfig``.  Keeping this lower-level
    single-task runtime config distinct prevents package consumers from
    accidentally passing the wrong object to the CLI backends.
    """

    task_id: int = 5
    task_description: str = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    vla_budget: int = 220
    teacher_budget: int = 1200
    combined_budget: int = 1420
    split: SplitManifest = field(
        default_factory=lambda: SplitManifest(
            support_seeds=tuple(range(1000, 1010)),
            query_seeds=tuple(range(2000, 2020)),
        )
    )

    def __post_init__(self) -> None:
        if self.vla_budget <= 0 or self.teacher_budget <= 0:
            raise ContractError("budgets must be positive")
        if self.combined_budget < self.vla_budget + self.teacher_budget:
            raise ContractError("combined budget must cover VLA plus teacher budgets")


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    status: EpisodeStatus
    vla_steps: int
    teacher_steps: int
    teacher_used: bool
    teacher_success: bool
    adaptation_applied: bool
    transitions: tuple[TransitionRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def vla_success(self) -> bool:
        return self.status is EpisodeStatus.VLA_SUCCESS

    @property
    def total_success(self) -> bool:
        return self.status in {EpisodeStatus.VLA_SUCCESS, EpisodeStatus.TEACHER_SUCCESS}


def initial_state_hash(observation: Mapping[str, Any]) -> str:
    """Stable hash of clean reset observation (never teacher sidecars)."""

    if not isinstance(observation, Mapping):
        raise ContractError("initial observation must be a mapping")
    # ``default=str`` is unsafe for split identity: ndarray/tensor values with
    # different dtype/shape (and unrelated objects sharing a repr) can hash to
    # the same bytes.  Use explicit type/shape/data tags and fail closed for
    # unsupported objects rather than silently collapsing states.
    canonical = _lossless_json_value(observation)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lossless_json_value(value: Any) -> Any:
    """Encode observation values without lossy ``str`` fallbacks.

    Arrays and tensors include their concrete type, dtype, shape and values;
    bytes include their exact contents.  This function intentionally supports
    only values that can be represented unambiguously in the split manifest.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("initial observation contains a non-finite float")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__type__": "bytes", "data_hex": bytes(value).hex()}
    if isinstance(value, Mapping):
        # Encoding keys as values preserves distinctions such as 1 vs "1".
        items = [[_lossless_json_value(key), _lossless_json_value(item)] for key, item in value.items()]
        return {"__type__": "mapping", "items": items}
    if isinstance(value, list):
        return {"__type__": "list", "items": [_lossless_json_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_lossless_json_value(item) for item in value]}

    # numpy arrays and torch tensors both expose shape, dtype, and tolist.
    # Detach/cpu is used only when available; importing either dependency here
    # would make bookkeeping unusable in lightweight test environments.
    tolist = getattr(value, "tolist", None)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if callable(tolist) and shape is not None and dtype is not None:
        array_value = value
        detach = getattr(array_value, "detach", None)
        if callable(detach):
            array_value = detach()
        cpu = getattr(array_value, "cpu", None)
        if callable(cpu):
            array_value = cpu()
        return {
            "__type__": f"array:{type(value).__module__}.{type(value).__qualname__}",
            "dtype": str(dtype),
            "shape": [_lossless_json_value(int(dimension)) for dimension in tuple(shape)],
            "data": _lossless_json_value(array_value.tolist()),
        }

    # numpy scalar values have dtype/item but no shape.  Preserve dtype so
    # int8(1) and int64(1) cannot be conflated.
    item = getattr(value, "item", None)
    if callable(item) and dtype is not None:
        return {
            "__type__": f"scalar:{type(value).__module__}.{type(value).__qualname__}",
            "dtype": str(dtype),
            "data": _lossless_json_value(item()),
        }
    raise ContractError(
        f"unsupported initial observation value {type(value).__module__}.{type(value).__qualname__}; "
        "provide a JSON-safe scalar, mapping, sequence, bytes, ndarray, or tensor"
    )


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * ((p * (1 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _step_result(result: Any) -> tuple[Mapping[str, Any], bool, bool, bool]:
    """Normalize Gym(nasium)-style or project-style step results."""

    def strict_flag(value: Any, name: str) -> bool:
        if not isinstance(value, bool):
            raise ContractError(f"environment step {name} field must be boolean")
        return value

    def info_success(info: Any) -> bool:
        if info is None:
            return False
        if not isinstance(info, Mapping):
            raise ContractError("environment step info must be a mapping")
        return strict_flag(info.get("success", False), "success")

    if isinstance(result, tuple):
        if len(result) == 5:
            obs, _reward, terminated, truncated, info = result
            return obs, strict_flag(terminated, "terminated"), strict_flag(truncated, "truncated"), info_success(info)
        if len(result) == 4:
            obs, _reward, done, info = result
            return obs, strict_flag(done, "done"), False, info_success(info)
    if isinstance(result, Mapping):
        obs = result.get("observation", result.get("obs"))
        terminated = result.get("terminated", result.get("done", False))
        truncated = result.get("truncated", False)
        success = result.get("success", False)
        return obs, strict_flag(terminated, "terminated"), strict_flag(truncated, "truncated"), strict_flag(success, "success")
    raise ContractError("environment.step must return a mapping or Gym-style tuple")


def run_episode(
    spec: EpisodeSpec,
    *,
    environment_factory: EnvironmentFactory,
    policy: Policy,
    action_fn: VLAActionFn | None = None,
    teacher: Any | None = None,
    updater: Updater | None = None,
    source_state_fn: Callable[[LiveEnvironment, Mapping[str, Any]], SourceState] | None = None,
    success_fn: Callable[[LiveEnvironment, Any], bool] | None = None,
    vla_budget: int = 220,
    teacher_budget: int = 1200,
    observation_schema: str | None = None,
) -> EpisodeOutcome:
    """Run VLA and (only after failure) Arrow takeover on one live env.

    The environment is created, reset and closed by this function exactly
    once.  The teacher receives the same object and is prohibited by contract
    from calling reset/close.  Only executed actions are recorded by the
    injected environment recorder/teacher, so unexecuted policy proposals can
    never become training labels.
    """

    if vla_budget <= 0 or teacher_budget <= 0:
        raise ContractError("budgets must be positive")
    if updater is not None and observation_schema is None:
        raise ContractError(
            "updater-enabled runs must declare the canonical student observation schema; "
            "raw simulator observations cannot reach training"
        )
    if updater is not None and observation_schema != CANONICAL_OBSERVATION_SCHEMA:
        raise ContractError(
            "updater requires observation_schema='libero_rgb_state8_instruction_v1'; "
            "raw simulator observations cannot enter training"
        )
    factory_result = environment_factory(spec)
    handle = factory_result if isinstance(factory_result, EnvironmentHandle) else None
    env = handle.environment if handle is not None else factory_result
    if handle is not None:
        try:
            handle.validate_initial_state()
        except BaseException:
            handle.close_once()
            raise
    transitions: list[TransitionRecord] = []
    vla_steps = 0
    teacher_steps = 0
    teacher_used = False
    teacher_success = False
    adaptation_applied = False
    validated_demo = None
    status = EpisodeStatus.ABORTED
    vla_terminal = False
    try:
        if handle is not None:
            # The factory already selected and hashed the reset state.  A
            # second reset would invalidate paired initial-state identity.
            observation = handle.initial_observation
            handle.verify_live_initial_state()
        else:
            # Legacy fallback is intentionally retained for old callers, but
            # it is not the canonical experiment path.
            reset = getattr(env, "reset", None)
            if not callable(reset):
                raise ContractError("legacy environment_factory must return an environment exposing reset()")
            reset_result = reset()
            if isinstance(reset_result, tuple) and reset_result and isinstance(reset_result[0], Mapping):
                observation = reset_result[0]
            elif isinstance(reset_result, Mapping):
                observation = reset_result
            else:
                observation = env.observe()
        if observation_schema is not None:
            validate_student_observation_schema(observation, require_complete=True, schema=observation_schema)
        policy.reset(spec.task_description, spec.seed)
        if teacher is not None and source_state_fn is None:
            raise ContractError("source_state_fn is mandatory whenever teacher takeover is enabled")

        chunk_counter = 0
        while vla_steps < vla_budget:
            chunk = normalize_action_chunk(
                action_fn(observation, chunk_counter) if action_fn else policy.act(observation),
                chunk_id=f"vla-chunk-{chunk_counter}",
            )
            chunk_counter += 1
            for chunk_index, action in enumerate(chunk.actions):
                if vla_steps >= vla_budget:
                    break
                result = env.step(action)
                next_observation, terminated, truncated, step_success = _step_result(result)
                if observation_schema is not None:
                    validate_student_observation_schema(
                        next_observation, require_complete=True, schema=observation_schema
                    )
                success_override = False
                if success_fn is not None:
                    success_override = success_fn(env, result)
                    if not isinstance(success_override, bool):
                        raise ContractError("success_fn must return a boolean")
                canonical_success = step_success or success_override
                transitions.append(
                    TransitionRecord(
                        episode_id=spec.episode_id,
                        timestep=len(transitions),
                        actor=Actor.VLA,
                        observation=observation,
                        action=action,
                        next_observation=next_observation,
                        done=terminated or truncated or canonical_success,
                        success=canonical_success,
                        training_eligible=False,
                        action_chunk_index=chunk_index,
                        action_chunk_id=chunk.chunk_id,
                        action_chunk_horizon=chunk.horizon,
                        denoising_metadata=chunk.denoising_metadata,
                    )
                )
                vla_steps += 1
                observation = next_observation
                if canonical_success:
                    status = EpisodeStatus.VLA_SUCCESS
                    break
                if terminated or truncated:
                    status = EpisodeStatus.ABORTED
                    vla_terminal = True
                    break
            if status is EpisodeStatus.VLA_SUCCESS or vla_terminal:
                break
        if status is EpisodeStatus.ABORTED and not vla_terminal:
            status = EpisodeStatus.ABORTED

        if status is not EpisodeStatus.VLA_SUCCESS and not vla_terminal and teacher is not None:
            state = source_state_fn(env, observation)
            if state not in {SourceState.SOURCE_UNHELD, SourceState.SOURCE_HELD}:
                return EpisodeOutcome(spec.episode_id, status, vla_steps, 0, False, False, False, tuple(transitions), {"takeover_rejected": state.value})
            teacher_used = True
            request = TeacherRecoveryRequest(spec, state, observation, tuple(transitions), teacher_budget)
            # Expose only the non-owning takeover view.  This prevents a
            # controller adapter from resetting/closing the live episode or
            # reaching simulator-only state through arbitrary attributes.
            view_type = PrivilegedTakeoverEnvironmentView if getattr(teacher, "requires_privileged_environment", False) else TakeoverEnvironmentView
            view = view_type(
                env,
                expected_identity=id(env),
                episode_id=spec.episode_id,
                start_timestep=len(transitions),
                source_state=state,
            )
            recovery = teacher.recover(view, request)
            teacher_transitions = validate_teacher_transitions(view, recovery.transitions)
            if observation_schema is not None:
                for record in teacher_transitions:
                    validate_student_observation_schema(record.observation, require_complete=True, schema=observation_schema)
                    validate_student_observation_schema(record.next_observation, require_complete=True, schema=observation_schema)
            if view.step_count > teacher_budget or len(teacher_transitions) > teacher_budget:
                raise ContractError("teacher exceeded the configured takeover budget")
            if recovery.success and not teacher_transitions:
                raise ContractError("teacher cannot report success without executed correction transitions")
            for offset, record in enumerate(teacher_transitions):
                if record.episode_id != spec.episode_id or record.actor is not Actor.TEACHER or not record.training_eligible:
                    raise ContractError("teacher returned an invalid transition")
                expected = len(transitions) + offset
                if record.timestep != expected:
                    raise ContractError(f"teacher timestep {record.timestep} does not continue VLA timestep {expected - 1}")
            teacher_steps = len(teacher_transitions)
            transitions.extend(teacher_transitions)
            teacher_success = bool(recovery.success)
            status = EpisodeStatus.TEACHER_SUCCESS if teacher_success else EpisodeStatus.TEACHER_FAILED
            if teacher_success:
                # A successful Arrow label is not enough to make a training
                # demonstration valid.  Validate the complete interleaved
                # trace at the live-environment boundary before any updater
                # can consume it.  Arrow's bridge emits an explicit evaluator
                # verdict; generic teachers must opt into the same field.
                evaluator_success = recovery.metadata.get("evaluator_success") if isinstance(recovery.metadata, Mapping) else None
                if evaluator_success is None:
                    evaluator_success = recovery.success
                elif not isinstance(evaluator_success, bool):
                    raise ContractError("recovery.metadata.evaluator_success must be a boolean")
                validated_demo = validate_and_build_demonstration(
                    tuple(transitions),
                    task_id=spec.task_id,
                    seed=spec.seed,
                    environment_identity=str(getattr(handle, "environment_identity", id(env))),
                    teacher_success=True,
                    evaluator_success=evaluator_success,
                    source_controller=str(getattr(recovery, "teacher_id", "arrow_grasp_controller")),
                    provenance=(dict(recovery.metadata) if isinstance(recovery.metadata, Mapping) else {}),
                )
            else:
                validated_demo = None
            if teacher_success and updater is not None:
                training_episode = episode_from_executed_records(
                    tuple(transitions),
                    task_id=spec.task_id,
                    seed=spec.seed,
                    episode_id=spec.episode_id,
                    outcome=status.value,
                    environment_identity=str(getattr(handle, "environment_identity", id(env))),
                    observation_schema=observation_schema,
                )
                # Keep the validated receipt attached to the episode lineage;
                # the raw transitions remain the sole source of model inputs.
                training_episode = replace(
                    training_episode,
                    provenance={
                        **dict(training_episode.provenance),
                        "demonstration_receipt": validated_demo.receipt.to_json() if validated_demo else None,
                    },
                )
                training_dataset = TTTDataset([training_episode])
                updater.adapt(training_dataset, episode_id=spec.episode_id)
                adaptation_applied = True
        outcome_metadata: dict[str, Any] = {}
        if validated_demo is not None:
            outcome_metadata["demonstration_receipt"] = validated_demo.receipt.to_json()
        return EpisodeOutcome(
            spec.episode_id,
            status,
            vla_steps,
            teacher_steps,
            teacher_used,
            teacher_success,
            adaptation_applied,
            tuple(transitions),
            outcome_metadata,
        )
    finally:
        if handle is not None:
            handle.close_once()
        else:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def summarize(outcomes: Iterable[EpisodeOutcome]) -> dict[str, Any]:
    rows = list(outcomes)
    n = len(rows)
    if not n:
        return {"trials": 0}
    counts = {
        "vla_success": sum(row.vla_success for row in rows),
        "teacher_used": sum(row.teacher_used for row in rows),
        "teacher_success": sum(row.teacher_success for row in rows),
        "total_success": sum(row.total_success for row in rows),
    }
    result: dict[str, Any] = {"trials": n, **counts}
    for key, value in counts.items():
        result[f"{key}_rate"] = value / n
        result[f"{key}_wilson95"] = wilson_interval(value, n)
    return result


def collect_from_config(*, config: Any, args: Any | None = None) -> dict[str, Any]:
    """CLI entry point that records the launch contract without fake data.

    A real collection run must inject factories for each VLA, LIBERO's live
    environment and the Arrow controller bridge.  Keeping those imports out
    of this package makes preflight usable on CPU/login nodes and prevents a
    command from claiming improvement when no episodes were collected.
    """

    return {
        "status": "BLOCKED_NEEDS_FACTORIES",
        "config_digest": config.digest(),
        "vlas": list(config.canonical_dict()["vla_names"]),
        "tasks": list(config.task_ids),
        "episodes_per_task": config.episodes_per_task,
        "student_step_budget": config.student_step_budget,
        "teacher_step_budget": config.teacher_step_budget,
        "message": "Register environment_factory, VLA adapters, source-state classifier, and ArrowGraspControllerTeacher before collection; no episodes were launched.",
    }


__all__ = [
    "EnvironmentFactory", "EnvironmentHandle", "EpisodeOutcome", "EpisodeRunConfig", "Policy", "SplitManifest",
    "Updater", "initial_state_hash", "run_episode", "summarize", "wilson_interval",
]
