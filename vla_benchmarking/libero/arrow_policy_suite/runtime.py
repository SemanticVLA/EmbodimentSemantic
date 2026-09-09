"""Transactional, simulator-free runtime for policy episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import threading
from collections import deque
import math
import time
from typing import Any, Protocol

from .contracts import ActionProposal, ContractError, EpisodeSnapshot, ObservationFrame, PolicyDecision
try:  # The richer suite contracts add StepRecord; keep this runtime importable during assembly.
    from .contracts import StepRecord  # type: ignore
except ImportError:  # pragma: no cover - compatibility while workers assemble package
    StepRecord = Any  # type: ignore[misc,assignment]
from .controller import InterruptibleTeacher, Policy, policy_identifier


class Environment(Protocol):
    def observe(self) -> Mapping[str, Any]: ...
    def step(self, action: Sequence[float]) -> Any: ...


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    kind: str
    timestep: int
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


class EventTracker:
    """Deterministic in-memory event log that retains raw payloads."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def record(self, kind: str, *, timestep: int = 0, payload: Any = None,
               metadata: Mapping[str, Any] | None = None,
               provenance: Mapping[str, Any] | None = None) -> RuntimeEvent:
        if not kind:
            raise ContractError("event kind is required")
        event = RuntimeEvent(len(self._events), kind, timestep, payload,
                             dict(metadata or {}), dict(provenance or {}))
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        return self.events


class ProgressTracker:
    """Minimal deterministic trigger tracker used by On-Call policies."""

    def __init__(self, *, window: int = 20) -> None:
        if window <= 0:
            raise ContractError("progress window must be positive")
        self.window = int(window)
        self._updates = 0
        self._last_phase_error: float | None = None
        self._errors: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._updates = 0
        self._last_phase_error = None
        self._errors.clear()

    def should_trigger(self, teacher: Any) -> bool:
        metadata = getattr(teacher, "metadata", {})
        if not isinstance(metadata, Mapping):
            return False
        # Safety/gripper hazards are allowed to interrupt immediately; normal
        # phase correction must first establish a window of error history.
        if any(bool(metadata.get(key, False)) for key in (
            "safety", "safety_violation", "force_takeover", "gripper_conflict",
        )):
            return True
        raw_error = metadata.get("phase_error")
        if raw_error is None:
            return False
        try:
            error = float(raw_error)
        except (TypeError, ValueError) as exc:
            raise ContractError("teacher phase_error must be numeric") from exc
        if not self._errors or len(self._errors) < self.window:
            return False
        # A takeover is warranted only when the latest error improved by less
        # than ten percent relative to the start of the completed window.
        baseline = self._errors[0]
        if abs(baseline) <= 1e-12:
            improvement = 1.0 if abs(error) <= 1e-12 else 0.0
        else:
            improvement = (baseline - error) / abs(baseline)
        # Treat the declared 10% boundary exactly, rather than allowing
        # binary floating-point noise (e.g. 1.0 -> 0.9) to trigger takeover.
        return improvement < 0.10 and not math.isclose(improvement, 0.10, abs_tol=1e-12)

    @property
    def updates(self) -> int:
        return self._updates

    def progress_improvement(self, teacher: Any | None = None) -> float | None:
        """Return fractional phase-error improvement over the active window."""
        if len(self._errors) < self.window:
            return None
        raw = getattr(teacher, "metadata", {}).get("phase_error") if teacher is not None else None
        error = float(raw) if raw is not None else self._errors[-1]
        baseline = self._errors[0]
        if abs(baseline) <= 1e-12:
            return 1.0 if abs(error) <= 1e-12 else 0.0
        return (baseline - error) / abs(baseline)

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "window": self.window,
            "updates": self._updates,
            "last_phase_error": self._last_phase_error,
            "errors": tuple(self._errors),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("window", self.window)) != self.window:
            raise ContractError("cannot restore ProgressTracker with a different window")
        self._updates = int(state.get("updates", 0))
        raw = state.get("last_phase_error")
        self._last_phase_error = None if raw is None else float(raw)
        self._errors.clear()
        self._errors.extend(float(value) for value in state.get("errors", ()))

    def update(self, _frame: ObservationFrame, teacher: Any) -> None:
        self._updates += 1
        metadata = getattr(teacher, "metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("phase_error") is not None:
            self._last_phase_error = float(metadata["phase_error"])
            self._errors.append(self._last_phase_error)


@dataclass(frozen=True)
class EpisodeStats:
    success: bool
    terminal: bool
    steps: int
    # ``None`` keeps three-argument construction source-compatible with old
    # artifacts; native runs always populate the explicit cost counters.
    scored_steps: int | None = None
    cloned_steps: int = 0
    teacher_proposals: int = 0
    teacher_steps: int = 0
    branch_steps: int = 0
    latency_seconds: float = 0.0
    policy_latency_seconds: float = 0.0
    branch_latency_seconds: float = 0.0


@dataclass(frozen=True)
class EpisodeRecord:
    frame: ObservationFrame
    base: Any
    teacher: Any
    decision: Any
    next_frame: ObservationFrame
    success: bool = False
    terminal: bool = False
    raw_result: Any = None


@dataclass(frozen=True)
class EpisodeResult:
    records: tuple[EpisodeRecord, ...]
    stats: EpisodeStats
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _frame_digest(frame: Any) -> str | None:
    return getattr(frame, "digest", None)


def make_proposal(action: Sequence[float], policy_id: str, frame: Any = None, *,
                  timestep: int | None = None, observation_digest: str | None = None,
                  **metadata: Any) -> ActionProposal:
    """Build a proposal tied to the exact observation a policy saw."""
    digest = observation_digest if observation_digest is not None else _frame_digest(frame)
    step = getattr(frame, "timestep", 0) if timestep is None else timestep
    try:
        return ActionProposal(action, policy_id=policy_id, timestep=step,
                              metadata=metadata, observation_digest=digest)
    except TypeError as exc:
        # Compatibility with the earliest lightweight contract, which did not
        # yet expose observation_digest as a keyword.
        if "observation_digest" not in str(exc):
            raise
        return ActionProposal(action, policy_id=policy_id, timestep=step, metadata=metadata)


@dataclass(frozen=True)
class PendingStep:
    proposal: ActionProposal
    raw_result: Any
    observation: ObservationFrame
    environment_state: Any = None
    rollback_supported: bool = False


def _parse_result(raw: Any) -> tuple[Mapping[str, Any] | None, Any, bool, bool, Any]:
    if isinstance(raw, tuple) and len(raw) == 5 and isinstance(raw[0], Mapping):
        return raw[0], raw[1], bool(raw[2]), bool(raw[3]), raw[4]
    if isinstance(raw, tuple) and len(raw) == 4 and isinstance(raw[0], Mapping):
        return raw[0], raw[1], bool(raw[2]), False, raw[3]
    if isinstance(raw, Mapping):
        return (
            raw.get("observation") if isinstance(raw.get("observation"), Mapping) else None,
            raw.get("reward"),
            bool(raw.get("terminated", raw.get("done", raw.get("terminal", False)))),
            bool(raw.get("truncated", False)),
            raw,
        )
    return None, None, False, False, None


class TransactionalCoordinator:
    """The sole owner of an environment's propose/step/commit transaction."""

    def __init__(self, environment: Environment, vla: Any | None = None, teacher: Any | None = None,
                 *, episode_id: str | None = None, success_fn: Any | None = None,
                 terminal_fn: Any | None = None,
                 event_tracker: EventTracker | None = None,
                 metadata: Mapping[str, Any] | None = None,
                 provenance: Mapping[str, Any] | None = None) -> None:
        if not hasattr(environment, "step"):
            raise ContractError("environment must provide step(action)")
        self._environment = environment
        self.vla = vla
        self.teacher = teacher
        self._success_fn_explicit = success_fn is not None
        self._terminal_fn_explicit = terminal_fn is not None
        self.success_fn = success_fn or (lambda value: bool(value.get("success", False)) if isinstance(value, Mapping) else False)
        self.terminal_fn = terminal_fn or (lambda value: bool(value.get("terminal", False)) if isinstance(value, Mapping) else False)
        self.episode_id = episode_id
        self.event_tracker = event_tracker or EventTracker()
        self.metadata = dict(metadata or {})
        self.provenance = dict(provenance or {})
        self._lock = threading.RLock()
        self._current: ObservationFrame | None = None
        self._pending: PendingStep | None = None
        self._closed = False
        self._timestep = 0

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def current(self) -> ObservationFrame | None:
        return self._current

    @property
    def pending(self) -> PendingStep | None:
        return self._pending

    def observe(self) -> ObservationFrame:
        with self._lock:
            self._ensure_open()
            frame = self._read_observation()
            self._current = frame
            return frame

    def propose(self, policy: Policy, observation: ObservationFrame | None = None) -> ActionProposal:
        with self._lock:
            self._ensure_open()
            if self._pending is not None:
                raise ContractError("commit the pending step before proposing another action")
            frame = observation or self._current or self.observe()
            method = getattr(policy, "propose", None) or getattr(policy, "act", None)
            if not callable(method):
                raise ContractError("policy must provide propose(frame) or act(frame)")
            result = method(frame)
            if isinstance(result, ActionProposal):
                proposal = result
            else:
                proposal = ActionProposal(result, policy_id=policy_identifier(policy), timestep=self._timestep)
            if proposal.timestep != self._timestep:
                raise ContractError(f"proposal timestep {proposal.timestep} does not match {self._timestep}")
            self._validate_proposal_frame(proposal, frame)
            # Validation occurs before any environment side effect.
            from .contracts import validate_action
            validate_action(proposal.action)
            self.event_tracker.record("proposal", timestep=self._timestep, payload=proposal,
                                      metadata=proposal.metadata, provenance=proposal.provenance)
            return proposal

    def step(self, proposal: ActionProposal | Policy) -> PendingStep:
        with self._lock:
            self._ensure_open()
            if self._pending is not None:
                raise ContractError("a step is already pending; commit it first")
            if not isinstance(proposal, ActionProposal):
                proposal = self.propose(proposal)
            before = self._current or self.observe()
            if proposal.timestep != self._timestep:
                raise ContractError(f"proposal timestep {proposal.timestep} does not match {self._timestep}")
            self._validate_proposal_frame(proposal, before)
            # Revalidate immediately before the side effect.
            from .contracts import validate_action
            validate_action(proposal.action)
            environment_state, rollback_supported = self._capture_environment_state()
            try:
                raw = self._environment.step(proposal.action)
            except BaseException:
                if not self._rollback_environment(environment_state, rollback_supported):
                    self._closed = True
                raise
            pending = PendingStep(proposal, raw, before, environment_state, rollback_supported)
            # Keep the result pending until commit; failed observation capture
            # cannot silently advance the episode.  The observation embedded
            # in a Gym-style result is intentionally parsed again at commit so
            # the raw result remains untouched.
            self._pending = pending
            return pending

    def commit(self, pending: PendingStep | None = None, *, success: bool | None = None) -> PolicyDecision:
        with self._lock:
            self._ensure_open()
            pending = pending or self._pending
            if pending is None or pending is not self._pending:
                raise ContractError("commit requires the coordinator's pending step")
            returned, reward, terminated, truncated, info = _parse_result(pending.raw_result)
            previous_current = self._current
            previous_timestep = self._timestep
            try:
                if hasattr(self._environment, "observe"):
                    values = self._environment.observe()
                elif returned is not None:
                    values = returned
                else:
                    raise ContractError("environment must provide observe() or return an observation from step()")
                if not isinstance(values, Mapping):
                    raise ContractError("environment observation must be a mapping")
                snapshot = EpisodeSnapshot(values, timestep=self._timestep + 1,
                                           terminated=terminated, truncated=truncated,
                                           reward=reward, info=info,
                                           metadata=self.metadata, provenance=self.provenance)
                if success is None and isinstance(info, Mapping) and "success" in info:
                    success = info["success"]
                if success is not None and not isinstance(success, bool):
                    raise ContractError("success must be a boolean or None")
                decision = PolicyDecision(pending.proposal, snapshot, pending.raw_result,
                                          success, self.metadata, self.provenance)
                next_frame = ObservationFrame(values, timestep=self._timestep + 1,
                                              episode_id=self.episode_id,
                                              metadata=self.metadata, provenance=self.provenance)
                self._current = next_frame
                self._timestep += 1
                self._pending = None
                self.event_tracker.record("commit", timestep=self._timestep, payload=decision,
                                          metadata=self.metadata, provenance=self.provenance)
                return decision
            except BaseException:
                restored = self._rollback_environment(pending.environment_state, pending.rollback_supported)
                self._current = previous_current
                self._timestep = previous_timestep
                self._pending = None
                if not restored:
                    self._closed = True
                raise

    def interrupt(self, teacher: InterruptibleTeacher, *, reason: str,
                  context: Mapping[str, Any] | None = None) -> ActionProposal | None:
        with self._lock:
            self._ensure_open()
            if self._pending is not None:
                raise ContractError("cannot interrupt with an uncommitted step")
            frame = self._current or self.observe()
            method = getattr(teacher, "interrupt", None) or getattr(teacher, "takeover", None)
            if not callable(method):
                raise ContractError("teacher must provide interrupt(...) or takeover(...)")
            proposal = method(frame, reason=reason, context=context or {})
            if proposal is None:
                self.event_tracker.record("teacher_interrupt", timestep=self._timestep,
                                          payload=None, metadata={"reason": reason})
                return None
            if not isinstance(proposal, ActionProposal):
                proposal = ActionProposal(proposal, policy_id=getattr(teacher, "teacher_id", "teacher"),
                                           timestep=self._timestep, interruptible=True)
            if proposal.timestep != self._timestep:
                raise ContractError("teacher proposal timestep does not match coordinator")
            self.event_tracker.record("teacher_interrupt", timestep=self._timestep,
                                      payload=proposal, metadata={"reason": reason})
            return proposal

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def run(self, policy: Any, *, max_steps: int) -> EpisodeResult:
        """Run the complete paired-policy loop with pre-step validation.

        The environment is stepped exactly once per committed record.  Policy,
        VLA, and teacher objects only propose/commit; they never receive the
        environment itself.
        """
        if max_steps <= 0:
            raise ContractError("max_steps must be positive")
        if self.vla is None:
            raise ContractError("run() requires a VLA policy supplied at construction")
        with self._lock:
            self._ensure_open()
            if self._pending is not None:
                raise ContractError("cannot run while a manual step is pending; commit it first")
            for component in (self.vla, self.teacher, policy):
                reset = getattr(component, "reset", None)
                if callable(reset):
                    reset()
            frame = self._current or self.observe()
            records: list[EpisodeRecord] = []
            saw_success = False
            saw_terminal = False
            run_started = time.perf_counter()
            policy_latency = 0.0
            teacher_proposals = 0
            teacher_steps = 0
            cloned_steps = 0
            branch_latency = 0.0
            for _ in range(max_steps):
                base = self._call_propose(self.vla, frame)
                teacher = self._call_propose(self.teacher, frame) if self.teacher is not None else None
                teacher_proposals += int(teacher is not None)
                self._validate_proposal_frame(base, frame)
                if teacher is not None:
                    self._validate_proposal_frame(teacher, frame)
                decide = getattr(policy, "decide", None)
                if not callable(decide):
                    raise ContractError("suite policy must provide decide(frame, base, teacher)")
                policy_started = time.perf_counter()
                decision = decide(frame, base, teacher)
                policy_latency += time.perf_counter() - policy_started
                self._validate_decision_frame(decision, frame)
                action = getattr(decision, "action", None)
                if action is None:
                    raise ContractError("policy decision must expose action")
                from .contracts import validate_action
                validate_action(action)
                environment_state, rollback_supported = self._capture_environment_state()
                component_states = self._capture_component_states((self.vla, self.teacher, policy))
                try:
                    raw = self._environment.step(action)
                    returned, _, terminated_from_env, truncated_from_env, info = _parse_result(raw)
                    if hasattr(self._environment, "observe"):
                        values = self._environment.observe()
                    elif returned is not None:
                        values = returned
                    else:
                        raise ContractError("environment must provide observe() or return an observation from step()")
                    if not isinstance(values, Mapping):
                        raise ContractError("environment observation must be a mapping")
                    next_frame = ObservationFrame(values, timestep=self._timestep + 1,
                                                  episode_id=self.episode_id,
                                                  metadata=self.metadata, provenance=self.provenance)
                    success = self.success_fn(raw)
                    if not self._success_fn_explicit and isinstance(info, Mapping) and "success" in info:
                        success = bool(info["success"])
                    terminal = bool(terminated_from_env or truncated_from_env or self.terminal_fn(raw))
                    if not isinstance(success, bool) or not isinstance(terminal, bool):
                        raise ContractError("success_fn and terminal_fn must return booleans")
                    try:
                        record = StepRecord(frame, base, teacher, decision, next_frame,
                                           result=raw, success=success, terminal=terminal)
                    except TypeError:
                        try:
                            record = StepRecord(frame, base, teacher, decision, next_frame,
                                               success=success, terminal=terminal)
                        except TypeError:
                            record = EpisodeRecord(frame, base, teacher, decision, next_frame, success, terminal, raw)
                    for component in (self.vla, self.teacher, policy):
                        commit = getattr(component, "commit", None)
                        if callable(commit):
                            commit(record)
                except BaseException:
                    components_restored = self._restore_component_states(component_states)
                    restored = self._rollback_environment(environment_state, rollback_supported)
                    if not restored or not components_restored:
                        self._closed = True
                    raise
                records.append(record)
                decision_metadata = getattr(decision, "metadata", {})
                used_teacher = bool(
                    getattr(decision, "teacher_used", False)
                    or (isinstance(decision_metadata, Mapping) and decision_metadata.get("teacher_used", False))
                )
                teacher_steps += int(used_teacher)
                branch_steps = int(
                    decision_metadata.get("branch_steps", 0)
                    if isinstance(decision_metadata, Mapping) else 0
                )
                cloned_steps += branch_steps
                branch_latency += float(
                    decision_metadata.get("branch_latency_seconds", 0.0)
                    if isinstance(decision_metadata, Mapping) else 0.0
                )
                self.event_tracker.record("commit", timestep=self._timestep + 1,
                                          payload=record, metadata=self.metadata,
                                          provenance=self.provenance)
                frame = next_frame
                self._current = frame
                self._timestep += 1
                saw_success = saw_success or success
                saw_terminal = saw_terminal or terminal
                # A terminal environment must never be stepped again.  A
                # success is also absorbing for LIBERO pick-and-place; stop at
                # the first verified success so the 280/1200 metrics can be
                # derived from this single causal rollout.
                if success or terminal:
                    break
            elapsed = time.perf_counter() - run_started
            stats = EpisodeStats(
                saw_success,
                saw_terminal,
                len(records),
                scored_steps=len(records),
                cloned_steps=cloned_steps,
                teacher_proposals=teacher_proposals,
                teacher_steps=teacher_steps,
                branch_steps=cloned_steps,
                latency_seconds=elapsed,
                policy_latency_seconds=policy_latency,
                branch_latency_seconds=branch_latency,
            )
            result_metadata = dict(self.metadata)
            result_metadata.setdefault("cost", {})
            prior_cost = dict(result_metadata["cost"] or {}) if isinstance(result_metadata["cost"], Mapping) else {}
            result_metadata["cost"] = {
                **prior_cost,
                "scored_steps": len(records),
                "cloned_steps": cloned_steps,
                "teacher_proposals": teacher_proposals,
                "teacher_steps": teacher_steps,
                "branch_steps": cloned_steps,
                "latency_seconds": elapsed,
                "policy_latency_seconds": policy_latency,
                "branch_latency_seconds": branch_latency,
            }
            return EpisodeResult(tuple(records), stats, result_metadata)

    def _capture_environment_state(self) -> tuple[Any, bool]:
        snapshot = getattr(self._environment, "snapshot", None)
        restore = getattr(self._environment, "restore", None)
        if snapshot is None and restore is None:
            return None, False
        if not callable(snapshot) or not callable(restore):
            raise ContractError("environment snapshot/restore must be provided as a pair")
        return snapshot(), True

    def _rollback_environment(self, state: Any, supported: bool) -> bool:
        if not supported:
            return False
        try:
            self._environment.restore(state)
            return True
        except BaseException:
            return False

    @staticmethod
    def _capture_component_states(components: Sequence[Any]) -> tuple[tuple[Any, Any, Any] | None, ...]:
        states: list[tuple[Any, Any, Any] | None] = []
        for component in components:
            snapshot = getattr(component, "snapshot_state", None)
            restore = getattr(component, "restore_state", None)
            states.append((component, snapshot(), restore) if callable(snapshot) and callable(restore) else None)
        return tuple(states)

    @staticmethod
    def _restore_component_states(states: Sequence[tuple[Any, Any, Any] | None]) -> bool:
        restored = True
        for entry in states:
            if entry is not None:
                _, state, restore = entry
                try:
                    restore(state)
                except BaseException:
                    restored = False
        return restored

    @staticmethod
    def _call_propose(component: Any, frame: ObservationFrame) -> Any:
        if component is None:
            return None
        method = getattr(component, "propose", None)
        if not callable(method):
            method = getattr(component, "act", None)
        if not callable(method):
            raise ContractError("policy component must provide propose(frame)")
        return method(frame)

    @staticmethod
    def _validate_proposal_frame(proposal: Any, frame: ObservationFrame) -> None:
        if proposal is None:
            return
        proposal_digest = getattr(proposal, "observation_digest", None)
        if proposal_digest is None:
            proposal_digest = getattr(proposal, "frame_digest", None)
        frame_digest = getattr(frame, "digest", None)
        if proposal_digest is not None and frame_digest is not None and proposal_digest != frame_digest:
            raise ContractError("stale proposal observation digest")

    @staticmethod
    def _validate_decision_frame(decision: Any, frame: ObservationFrame) -> None:
        decision_digest = getattr(decision, "observation_digest", None)
        if decision_digest is None:
            proposal = getattr(decision, "proposal", None)
            decision_digest = getattr(proposal, "observation_digest", None)
        if decision_digest is not None and decision_digest != frame.digest:
            raise ContractError("stale policy decision observation digest")

    def _read_observation(self) -> ObservationFrame:
        if not hasattr(self._environment, "observe"):
            raise ContractError("environment must provide observe()")
        values = self._environment.observe()
        if not isinstance(values, Mapping):
            raise ContractError("environment observation must be a mapping")
        return ObservationFrame(values, timestep=self._timestep,
                                episode_id=self.episode_id,
                                metadata=self.metadata, provenance=self.provenance)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractError("coordinator is closed")


Coordinator = TransactionalCoordinator

__all__ = ["Coordinator", "Environment", "EpisodeRecord", "EpisodeResult", "EpisodeStats",
           "EventTracker", "PendingStep", "ProgressTracker", "RuntimeEvent",
           "TransactionalCoordinator", "make_proposal"]
