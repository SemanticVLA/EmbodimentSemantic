"""Shared native host for one-reset Arrow/SmolVLA experiments.

``NativeHost`` is the sole owner of ``environment.step``.  VLA and Arrow are
only proposal producers: both receive the same immutable observation frame,
and all mutable proposal state is captured before evaluation so a failed
transaction can restore the exact pre-action state.  This is intentionally a
small dependency-free seam; LIBERO, Torch and perception are injected by the
launcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from dataclasses import fields, is_dataclass
import inspect
import random
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .contracts import ActionProposal, ContractError, ObservationFrame, digest, validate_action
from .interruptible_arrow import ArrowPerceptionUnavailable
from .splits import ResetIdentity


def _hooks(component: Any, name: str, *, required: bool = False):
    if getattr(component, "rollback_complete", None) is False:
        raise ContractError(f"{name} has incomplete mutable rollback state")
    preferred = (getattr(component, "snapshot_state", None), getattr(component, "restore_state", None))
    fallback = (getattr(component, "snapshot", None), getattr(component, "restore", None))
    if callable(preferred[0]) != callable(preferred[1]):
        raise ContractError(f"{name} exposes only one snapshot_state/restore_state hook")
    if callable(preferred[0]):
        return preferred
    if callable(fallback[0]) != callable(fallback[1]):
        raise ContractError(f"{name} exposes only one snapshot/restore hook")
    if callable(fallback[0]):
        return fallback
    if required:
        raise ContractError(f"{name} requires paired snapshot/restore hooks")
    return None


def _capture_rng() -> Mapping[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    try:
        import torch
        state["torch"] = torch.random.get_rng_state().clone()
        state["torch_cuda"] = tuple(v.clone() for v in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        state["torch"] = None
        state["torch_cuda"] = None
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    try:
        import numpy as np
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass
    try:
        import torch
        if state.get("torch") is not None:
            torch.random.set_rng_state(state["torch"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (ImportError, RuntimeError):
        pass


def _copy_state(hook: Any, name: str) -> Any:
    try:
        return copy.deepcopy(hook())
    except Exception as exc:
        raise ContractError(
            f"{name} snapshot failed: {type(exc).__name__}: {exc}"
        ) from exc


def _restore_state(pair: Any, state: Any, name: str) -> None:
    if pair is None:
        return
    try:
        pair[1](state)
    except Exception as exc:
        raise ContractError(
            f"{name} restore failed: {type(exc).__name__}: {exc}"
        ) from exc


def _equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None or type(left) is not type(right):
        return False
    # Snapshot payloads may contain NumPy arrays (MuJoCo qpos/qvel, RGB
    # caches) or Torch tensors.  Their ``==`` operators return elementwise
    # values and cannot be used as a Python boolean.
    if hasattr(left, "shape") and hasattr(right, "shape"):
        try:
            import numpy as np
            return bool(np.array_equal(left, right))
        except (ImportError, TypeError, ValueError):
            pass
        try:
            equal = getattr(left, "equal", None)
            if callable(equal):
                return bool(equal(right))
        except Exception:
            pass
    if is_dataclass(left) and is_dataclass(right):
        return all(_equal(getattr(left, item.name), getattr(right, item.name)) for item in fields(left))
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (set, frozenset)):
        return left == right
    try:
        result = left == right
        if isinstance(result, bool):
            return result
        item = getattr(result, "item", None)
        if callable(item):
            value = item()
            if isinstance(value, bool):
                return value
    except Exception:
        pass
    return False


@dataclass(frozen=True)
class TeacherStatus:
    available: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeStep:
    frame: ObservationFrame
    base: ActionProposal
    teacher: ActionProposal | None
    action: tuple[float, ...]
    result: Any
    next_frame: ObservationFrame
    teacher_status: TeacherStatus
    executed_by: str
    proposal_state_unchanged: bool
    success: bool = False
    terminal: bool = False


@dataclass(frozen=True)
class NativeHostSnapshot:
    environment_state: Any
    vla_state: Any
    teacher_state: Any
    policy_state: Any
    rng_state: Mapping[str, Any]
    timestep: int
    reset_identity: ResetIdentity | None
    current_frame: ObservationFrame | None


class NativeHost:
    """Run a one-reset rollout with exactly one environment owner."""

    def __init__(
        self,
        environment: Any,
        vla: Any,
        teacher: Any | None = None,
        *,
        policy: Any | None = None,
        reset_identity: ResetIdentity | None = None,
        graph_context_fn: Callable[[ObservationFrame], Mapping[str, Any] | None] | None = None,
        action_selector: Callable[..., Sequence[float]] | None = None,
        success_fn: Callable[[Any], bool] | None = None,
        terminal_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        if environment is None or not callable(getattr(environment, "observe", None)) or not callable(getattr(environment, "step", None)):
            raise TypeError("NativeHost environment must expose observe() and step(action)")
        if vla is None or not callable(getattr(vla, "propose", None)):
            raise TypeError("NativeHost VLA must expose propose(frame)")
        self.environment = environment
        self.vla = vla
        self.teacher = teacher
        self.policy = policy
        self.reset_identity = reset_identity
        self.graph_context_fn = graph_context_fn
        self.action_selector = action_selector
        self.success_fn = success_fn
        self.terminal_fn = terminal_fn
        self.timestep = 0
        self.current_frame: ObservationFrame | None = None
        self.last_teacher_status = TeacherStatus(teacher is not None, None if teacher is not None else "teacher_not_configured")
        self._env_hooks = _hooks(environment, "environment", required=True)
        self._vla_hooks = _hooks(vla, "VLA", required=True)
        self._teacher_hooks = _hooks(teacher, "teacher", required=False) if teacher is not None else None
        self._policy_hooks = _hooks(policy, "policy", required=False) if policy is not None else None

    def _metadata(self) -> dict[str, Any]:
        identity = self.reset_identity
        return {
            "reset_identity": None if identity is None else identity.digest,
            "teacher_available": self.last_teacher_status.available,
            "teacher_unavailable_reason": self.last_teacher_status.reason,
        }

    def _make_frame(self) -> ObservationFrame:
        raw = self.environment.observe()
        if not isinstance(raw, Mapping):
            raise ContractError("native environment observe() must return a mapping")
        frame = ObservationFrame(raw, timestep=self.timestep, episode_id=(self.reset_identity.episode_id if self.reset_identity else None), metadata=self._metadata())
        if self.graph_context_fn is not None:
            context = self.graph_context_fn(frame)
            if context is not None:
                if not isinstance(context, Mapping):
                    raise ContractError("graph_context_fn must return a mapping or None")
                metadata = dict(frame.metadata)
                metadata["graph_context"] = dict(context)
                frame = ObservationFrame(raw, timestep=frame.timestep, episode_id=frame.episode_id, metadata=metadata)
        self.current_frame = frame
        return frame

    def observe_frame(self) -> ObservationFrame:
        """Read the current authoritative observation as one policy frame."""
        return self._make_frame()

    def reset(self, *, reset_environment: bool = True, **kwargs: Any) -> ObservationFrame:
        if reset_environment:
            reset = getattr(self.environment, "reset", None)
            if not callable(reset):
                raise ContractError("NativeHost environment has no reset() hook")
            reset(**kwargs)
        for component in (self.vla, self.teacher, self.policy):
            reset = getattr(component, "reset", None) if component is not None else None
            if callable(reset):
                reset()
        self.last_teacher_status = TeacherStatus(
            self.teacher is not None,
            None if self.teacher is not None else "teacher_not_configured",
        )
        self.timestep = 0
        frame = self._make_frame()
        if self.reset_identity is not None and digest(frame.observation) != self.reset_identity.observation_sha256:
            raise ContractError("post-reset frame does not match ResetIdentity")
        return frame

    def _snapshot(self) -> NativeHostSnapshot:
        return NativeHostSnapshot(
            _copy_state(self._env_hooks[0], "environment"),
            None if self._vla_hooks is None else _copy_state(self._vla_hooks[0], "VLA"),
            None if self._teacher_hooks is None else _copy_state(self._teacher_hooks[0], "teacher"),
            None if self._policy_hooks is None else _copy_state(self._policy_hooks[0], "policy"),
            _capture_rng(), self.timestep, self.reset_identity, self.current_frame,
        )

    def _restore(self, snapshot: NativeHostSnapshot) -> None:
        _restore_state(self._env_hooks, snapshot.environment_state, "environment")
        _restore_state(self._vla_hooks, snapshot.vla_state, "VLA")
        _restore_state(self._teacher_hooks, snapshot.teacher_state, "teacher")
        _restore_state(self._policy_hooks, snapshot.policy_state, "policy")
        _restore_rng(snapshot.rng_state)
        self.timestep = snapshot.timestep
        self.reset_identity = snapshot.reset_identity
        self.current_frame = snapshot.current_frame

    def snapshot_state(self) -> NativeHostSnapshot:
        return self._snapshot()

    def restore_state(self, snapshot: NativeHostSnapshot) -> None:
        if not isinstance(snapshot, NativeHostSnapshot):
            raise ContractError("NativeHost restore requires NativeHostSnapshot")
        self._restore(snapshot)

    snapshot = snapshot_state
    restore = restore_state

    def _select_action(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> tuple[tuple[float, ...], str]:
        if self.action_selector is None:
            return base.action, "vla"
        selector = self.action_selector
        try:
            parameters = inspect.signature(selector).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and len(parameters) >= 3:
            action = selector(base, teacher, frame)
        else:
            action = selector(base, teacher)
        normalized = validate_action(action)
        if normalized == base.action:
            return normalized, "vla"
        if teacher is not None and normalized == teacher.action:
            return normalized, "arrow"
        return normalized, "hybrid"

    @staticmethod
    def _flags(result: Any, success_fn: Callable[[Any], bool] | None, terminal_fn: Callable[[Any], bool] | None) -> tuple[bool, bool]:
        info: Mapping[str, Any] = {}
        if isinstance(result, tuple):
            # LIBERO/robosuite: obs, reward, done, info; Gymnasium adds
            # terminated/truncated before info.
            if len(result) == 4 and isinstance(result[3], Mapping):
                info = result[3]
            elif len(result) >= 5 and isinstance(result[4], Mapping):
                info = result[4]
        elif isinstance(result, Mapping):
            info = result
        success = bool(success_fn(result)) if success_fn is not None else bool(
            info.get("success", info.get("task_success", info.get("is_success", False)))
        )
        if terminal_fn is not None:
            terminal = bool(terminal_fn(result))
        elif isinstance(result, tuple) and len(result) == 4:
            terminal = bool(result[2])
        elif isinstance(result, tuple) and len(result) >= 5:
            terminal = bool(result[2]) or bool(result[3])
        elif isinstance(result, Mapping):
            terminal = bool(result.get("terminal", result.get("done", result.get("terminated", False)))) or bool(result.get("truncated", False))
        else:
            terminal = False
        return success, terminal

    def step(self) -> NativeStep:
        transaction = self._snapshot()
        try:
            frame = self._make_frame()
            base = self.vla.propose(frame)
            if not isinstance(base, ActionProposal) or base.observation_digest != frame.digest:
                raise ContractError("VLA proposal must match the current frame")
            teacher: ActionProposal | None = None
            if self.teacher is None:
                self.last_teacher_status = TeacherStatus(False, "teacher_not_configured")
            else:
                try:
                    teacher = self.teacher.propose(frame)
                except ArrowPerceptionUnavailable as exc:
                    self.last_teacher_status = TeacherStatus(False, str(exc) or "perception_unavailable")
                    teacher = None
                if teacher is None:
                    reason = getattr(self.teacher, "last_unavailable_reason", None) or "perception_unavailable"
                    availability = getattr(self.teacher, "last_availability", None)
                    if availability is not None:
                        reason = availability.reason or reason
                    self.last_teacher_status = TeacherStatus(False, reason)
                else:
                    if not isinstance(teacher, ActionProposal) or teacher.observation_digest != frame.digest:
                        raise ContractError("Arrow proposal must match the current frame")
                    self.last_teacher_status = TeacherStatus(True)
            after_proposals = _copy_state(self._env_hooks[0], "environment")
            proposal_unchanged = _equal(transaction.environment_state, after_proposals)
            if not proposal_unchanged:
                raise ContractError("proposal producers advanced environment state")
            action, executed_by = self._select_action(frame, base, teacher)
            result = self.environment.step(action)
            self.timestep += 1
            next_frame = self._make_frame()
            success, terminal = self._flags(result, self.success_fn, self.terminal_fn)
            if not success and self.success_fn is None:
                check_success = getattr(self.environment, "check_success", None)
                if callable(check_success):
                    checked = check_success()
                    success = bool(checked.get("success", checked.get("task_success", checked.get("is_success", False)))) if isinstance(checked, Mapping) else bool(checked)
            record = SimpleNamespace(frame=frame, base=base, teacher=teacher, next_frame=next_frame, result=result, success=success, terminal=terminal, decision=SimpleNamespace(action=action))
            if executed_by == "vla":
                commit = getattr(self.vla, "commit", None)
                if callable(commit):
                    commit(record)
            else:
                invalidate = getattr(self.vla, "invalidate_pending", None) or getattr(self.vla, "invalidate_queue", None)
                if callable(invalidate):
                    invalidate(reason=f"{executed_by}_action")
            if teacher is not None and executed_by == "vla":
                invalidate_teacher = (
                    getattr(self.teacher, "invalidate_pending", None)
                    or getattr(self.teacher, "invalidate_queue", None)
                    or getattr(self.teacher, "interrupt", None)
                )
                if callable(invalidate_teacher):
                    try:
                        invalidate_teacher(reason="vla_action")
                    except TypeError:
                        invalidate_teacher()
            if teacher is not None and executed_by in {"arrow", "hybrid"}:
                commit = getattr(self.teacher, "commit", None)
                if callable(commit):
                    commit(record)
            if self.policy is not None:
                commit = getattr(self.policy, "commit", None)
                if callable(commit):
                    commit(record)
            # Never allow a proposal to mutate the environment.  The one
            # actual step above is the sole environment transition.
            return NativeStep(frame, base, teacher, action, result, next_frame, self.last_teacher_status, executed_by, proposal_unchanged, success, terminal)
        except BaseException:
            self._restore(transaction)
            raise

    def run(self, *, max_steps: int = 3, reset_environment: bool = False, **reset_kwargs: Any) -> tuple[NativeStep, ...]:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if reset_environment or self.current_frame is None:
            self.reset(reset_environment=True, **reset_kwargs) if reset_environment else self.reset(reset_environment=False)
        records: list[NativeStep] = []
        for _ in range(int(max_steps)):
            record = self.step()
            records.append(record)
            if record.success or record.terminal:
                break
        return tuple(records)

    def close(self) -> None:
        """Close all runtime-owned components exactly once."""
        seen: set[int] = set()
        for component in (self.teacher, self.vla, self.policy, self.environment):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()


__all__ = ["NativeHost", "NativeHostSnapshot", "NativeStep", "TeacherStatus"]
