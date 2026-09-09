"""Fail-closed rollback hooks for a live LIBERO/robosuite environment.

The policy-suite host cannot safely branch or retry an ``OffScreenRenderEnv``
unless restoring the simulator also restores wrapper bookkeeping and the
authoritative observation cache.  ``OffScreenRenderState`` captures the
MuJoCo flat state, known episode/cache fields, observable caches, RNG streams,
and any explicitly stateful controller/task hooks.  Unknown mutable state is
reported as unsupported instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Any, Callable, Mapping

from .contracts import ContractError, digest


class LiberoRollbackUnavailable(ContractError):
    """Raised when a live environment cannot provide complete rollback."""


_KNOWN_SCALARS = frozenset({
    "_elapsed_steps", "elapsed_steps", "timestep", "_timestep", "step_count",
    "_step_count", "_time", "time", "done", "_done", "terminated",
    "_terminated", "truncated", "_truncated", "_episode_done", "_reset_count",
    "reward", "_reward", "last_reward", "_last_reward", "_arrow_motion_began",
    "_grasp_controller_action_count", "_grasp_controller_gripper_open",
})
_KNOWN_CACHES = frozenset({
    "_observation", "observation", "_obs", "obs", "_last_obs", "_last_observation",
    "last_observation", "_cached_obs", "_observation_cache", "_last_info", "last_info",
    "_arrow_phase_audit", "_arrow_motion_trace", "_arrow_failure_snapshots",
    "_arrow_micro_correction_audit", "_grasp_controller_observation_hover",
    "_grasp_controller_action_budget",
})
_RNG_NAMES = frozenset({"rng", "np_random", "_np_random", "random_state", "_random_state"})
_INFRASTRUCTURE_FIELDS = frozenset({
    "sim", "env", "_env", "unwrapped", "_unwrapped", "robots", "robot",
    "model", "data", "observables", "_observables", "action_dim", "_action_dim",
    "control_freq", "_control_freq", "camera_names", "_camera_names", "camera_heights",
    "camera_widths", "camera_depths", "controller_configs", "_controller_configs",
})


def _pair(value: Any) -> tuple[Callable[[], Any], Callable[[Any], None]] | None:
    for names in (("snapshot_state", "restore_state"), ("snapshot", "restore")):
        left, right = (getattr(value, name, None) for name in names)
        if callable(left) != callable(right):
            raise LiberoRollbackUnavailable(f"{type(value).__name__} exposes only one {names[0]}/{names[1]} hook")
        if callable(left):
            return left, right
    return None


def _capture_rng() -> dict[str, Any]:
    result: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np
        result["numpy"] = np.random.get_state()
    except ImportError:
        result["numpy"] = None
    try:
        import torch
        result["torch"] = torch.random.get_rng_state().clone()
        result["torch_cuda"] = tuple(value.clone() for value in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        result["torch"] = result["torch_cuda"] = None
    return result


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


def _safe_copy(value: Any, label: str) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise LiberoRollbackUnavailable(f"cannot snapshot {label}") from exc


def _restore_attrs(owner: Any, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        try:
            setattr(owner, name, _safe_copy(value, name))
        except Exception as exc:
            raise LiberoRollbackUnavailable(f"cannot restore environment field {name}") from exc


def _read_observation(environment: Any) -> Any:
    try:
        from vla_benchmarking.libero.evaluation.observation import read_raw_observation
        return read_raw_observation(environment, required=True)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        for name in ("observe", "get_observation"):
            method = getattr(environment, name, None)
            if callable(method):
                return method()
    raise LiberoRollbackUnavailable("live LIBERO environment has no authoritative observation reader")


def _sim_state(sim: Any) -> tuple[str, Any]:
    getter = getattr(sim, "get_state", None)
    setter = getattr(sim, "set_state", None)
    if callable(getter) != callable(setter):
        raise LiberoRollbackUnavailable("sim exposes only one get_state/set_state hook")
    if callable(getter):
        return "get_state", _safe_copy(getter(), "sim.get_state()")
    data = getattr(sim, "data", None)
    if data is None or not hasattr(data, "qpos") or not hasattr(data, "qvel"):
        raise LiberoRollbackUnavailable("MuJoCo sim lacks get_state/set_state and qpos/qvel fallback")
    values: dict[str, Any] = {}
    for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "time"):
        if hasattr(data, name):
            values[name] = _safe_copy(getattr(data, name), f"sim.data.{name}")
    if not {"qpos", "qvel"}.issubset(values):
        raise LiberoRollbackUnavailable("MuJoCo fallback state lacks qpos/qvel")
    return "data_arrays", values


def _restore_sim(sim: Any, mode: str, state: Any) -> None:
    if mode == "get_state":
        sim.set_state(_safe_copy(state, "sim state"))
        forward = getattr(sim, "forward", None)
        if callable(forward):
            forward()
        return
    data = getattr(sim, "data", None)
    if data is None:
        raise LiberoRollbackUnavailable("sim data disappeared during restore")
    for name, value in state.items():
        target = getattr(data, name, None)
        if hasattr(target, "__setitem__") and not isinstance(value, (int, float)):
            target[...] = value
        else:
            setattr(data, name, value)
    forward = getattr(sim, "forward", None)
    if callable(forward):
        forward()


@dataclass(frozen=True)
class OffScreenRenderSnapshot:
    sim_mode: str
    sim_state: Any
    wrapper_fields: Mapping[str, Any]
    observable_fields: Mapping[str, Mapping[str, Any]]
    component_states: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    observation_digest: str


class OffScreenRenderState:
    """Snapshot/restore provider for a live ``OffScreenRenderEnv``.

    ``strict=True`` rejects a controller/task object that advertises paired
    state hooks but fails to snapshot, and rejects unsupported simulator state.
    The provider validates the restored canonical raw observation digest before
    returning, so a superficially matching qpos/qvel restore cannot pass.
    """

    def __init__(self, environment: Any, *, strict: bool = True, observation_fn: Callable[[Any], Any] | None = None) -> None:
        self.environment = environment
        self.strict = bool(strict)
        self.observation_fn = observation_fn or _read_observation
        self.sim = getattr(environment, "sim", None)
        if self.sim is None:
            raise LiberoRollbackUnavailable("OffScreenRenderEnv has no sim handle")
        # Validate the simulator contract at construction, before a host runs.
        _sim_state(self.sim)

    def _wrapper_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        unsupported: list[str] = []
        for name in _KNOWN_SCALARS | _KNOWN_CACHES:
            if hasattr(self.environment, name):
                value = getattr(self.environment, name)
                try:
                    fields[name] = _safe_copy(value, f"environment.{name}")
                except LiberoRollbackUnavailable:
                    unsupported.append(name)
        # Runtime-added Arrow/controller bookkeeping is stateful and must not
        # be silently omitted.  Capture every field in those namespaces.
        for name in vars(self.environment):
            if (name.startswith("_arrow_") or name.startswith("_grasp_controller_")) and name not in fields:
                try:
                    fields[name] = _safe_copy(getattr(self.environment, name), f"environment.{name}")
                except LiberoRollbackUnavailable:
                    unsupported.append(name)
        # Catch mutable episode bookkeeping that the wrapper has introduced
        # under a new name.  Large simulator ownership references are handled
        # by ``sim_state`` or explicit component hooks and are not copied here.
        for name, value in vars(self.environment).items():
            if name in fields or name in _INFRASTRUCTURE_FIELDS:
                continue
            if name.startswith("__"):
                continue
            mutable = isinstance(value, (dict, list, set, bytearray)) or hasattr(value, "shape")
            if mutable or isinstance(value, (bool, int, float, str, type(None))):
                unsupported.append(name)
        if unsupported and self.strict:
            raise LiberoRollbackUnavailable(f"unsupported mutable environment fields: {unsupported}")
        return fields

    def _observable_fields(self) -> dict[str, dict[str, Any]]:
        observables = getattr(self.environment, "observables", None)
        if not isinstance(observables, Mapping):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for key, observable in observables.items():
            values: dict[str, Any] = {}
            for name in ("_current_value", "_cache", "_active", "_enabled", "_time_since_update", "_last_update"):
                if hasattr(observable, name):
                    values[name] = _safe_copy(getattr(observable, name), f"observable.{key}.{name}")
            if values:
                result[str(key)] = values
        return result

    def _component_states(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for name in ("controller", "task", "_controller", "_task"):
            component = getattr(self.environment, name, None)
            if component is None:
                continue
            hooks = _pair(component)
            if hooks is None:
                # These handles are often immutable configuration, while the
                # actual mutable controller phase lives in ArrowAdapter.  Do
                # not guess a deep-copy protocol for arbitrary simulator
                # objects; strict mode reports the missing hook.
                if self.strict and hasattr(component, "__dict__") and vars(component):
                    raise LiberoRollbackUnavailable(f"{name} lacks paired snapshot/restore hooks")
                continue
            states[name] = _safe_copy(hooks[0](), f"{name} state")
        return states

    def snapshot(self) -> OffScreenRenderSnapshot:
        mode, state = _sim_state(self.sim)
        observation = self.observation_fn(self.environment)
        return OffScreenRenderSnapshot(
            mode, state, self._wrapper_fields(), self._observable_fields(),
            self._component_states(), _capture_rng(), digest(observation),
        )

    def restore(self, snapshot: OffScreenRenderSnapshot) -> None:
        if not isinstance(snapshot, OffScreenRenderSnapshot):
            raise LiberoRollbackUnavailable("restore requires OffScreenRenderSnapshot")
        _restore_sim(self.sim, snapshot.sim_mode, snapshot.sim_state)
        _restore_attrs(self.environment, snapshot.wrapper_fields)
        observables = getattr(self.environment, "observables", None)
        if isinstance(observables, Mapping):
            for key, values in snapshot.observable_fields.items():
                observable = observables.get(key)
                if observable is not None:
                    _restore_attrs(observable, values)
        for name, state in snapshot.component_states.items():
            component = getattr(self.environment, name, None)
            hooks = _pair(component)
            if hooks is None:
                raise LiberoRollbackUnavailable(f"{name} restore hook disappeared")
            hooks[1](_safe_copy(state, f"{name} state"))
        _restore_rng(snapshot.rng_state)
        # Force wrappers/observables to rebuild from restored simulator data.
        refresh = getattr(self.environment, "_get_observations", None)
        if callable(refresh):
            refresh(force_update=True)
        current = self.observation_fn(self.environment)
        current_digest = digest(current)
        if current_digest != snapshot.observation_digest:
            raise LiberoRollbackUnavailable(
                "restored LIBERO observation digest differs from snapshot "
                f"({current_digest} != {snapshot.observation_digest})"
            )


def make_offscreen_snapshot_hooks(environment: Any, *, strict: bool = True, observation_fn: Callable[[Any], Any] | None = None) -> tuple[Callable[[], OffScreenRenderSnapshot], Callable[[OffScreenRenderSnapshot], None]]:
    provider = OffScreenRenderState(environment, strict=strict, observation_fn=observation_fn)
    return provider.snapshot, provider.restore


__all__ = ["LiberoRollbackUnavailable", "OffScreenRenderSnapshot", "OffScreenRenderState", "make_offscreen_snapshot_hooks"]
