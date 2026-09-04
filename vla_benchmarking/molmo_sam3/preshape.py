"""Bounded, robot-only fixed-opening preshape before RGB-D perception."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .. import run_arrow_pick_place_eval as episode
except ImportError:  # pragma: no cover - direct script use
    import run_arrow_pick_place_eval as episode


TARGET_OPENING_M = 0.040
OPENING_MIN_M = 0.035
OPENING_MAX_M = 0.045
OPENING_STABILITY_M = 0.00025
SETTLE_STEPS = 5
MAX_ACTIONS = 160
MAX_POSE_DRIFT_M = 0.020


class PreshapeError(RuntimeError):
    """Fail-closed preshape error with a stable category and audit payload."""

    def __init__(self, category: str, audit: Mapping[str, Any] | None = None):
        self.category = str(category)
        self.audit = dict(audit or {})
        super().__init__(f"preshape {self.category}")


def _failure(audit: dict[str, Any], reason: str, exc: BaseException | None = None) -> dict[str, Any]:
    audit["status"] = "failed"
    audit["failure_reason"] = str(reason)
    if exc is not None:
        audit["error_type"] = type(exc).__name__
        audit["error"] = str(exc)
    return audit


def _opening(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("measured usable pad opening must be finite and positive")
    return result


def _pose_from_observation(observation: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    proprio = episode._proprioception(observation)
    position = np.asarray(proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
    quaternion = np.asarray(proprio.get("eef_quat"), dtype=np.float64).reshape(-1)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("preshape requires finite EEF position and quaternion proprioception")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-9:
        raise ValueError("preshape EEF pose is non-finite")
    return position, quaternion / np.linalg.norm(quaternion)


def perform_preshape(
    env: Any,
    *,
    measure_opening_fn: Callable[[Any], float],
    output_dir: str | Path,
    motion_started_callback: Callable[[], None] | None = None,
    motion_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Close an already-open Panda gripper toward a measured 40 mm opening.

    Only signed full-magnitude close pulses and zero-gripper pose holds are
    dispatched.  The helper never opens the gripper or reads simulator/object
    state; all decisions use the supplied opening probe and robot proprioception.
    """
    audit: dict[str, Any] = {
        "status": "pending", "target_opening_m": TARGET_OPENING_M,
        "acceptance_band_m": [OPENING_MIN_M, OPENING_MAX_M],
        "stability_tolerance_m": OPENING_STABILITY_M,
        "max_actions": MAX_ACTIONS, "actions": [], "opening_readings_m": [],
    }

    def persist() -> None:
        if isinstance(getattr(env, "_molmo_sam3_action_budget", None), episode._ActionBudget):
            setattr(env, "_molmo_sam3_action_count", int(env._molmo_sam3_action_budget.used))
        try:
            root = Path(output_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            path = root / "preshape_audit.json"
            path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            audit["audit_path"] = path.as_posix()
        except Exception as exc:
            audit["audit_persist_error"] = str(exc)

    def fail(category: str, exc: BaseException | None = None) -> None:
        _failure(audit, category, exc)
        persist()
        raise PreshapeError(category, audit)

    if not callable(measure_opening_fn):
        fail("missing_measurement")
    budget = getattr(env, "_molmo_sam3_action_budget", None)
    if not isinstance(budget, episode._ActionBudget):
        fail("unsupported_budget")
    if budget.remaining <= 0:
        audit["budget_used"] = int(budget.used)
        fail("timeout")
    try:
        initial_observation = episode._raw_observation(env)
        initial_position, held_quaternion = _pose_from_observation(initial_observation)
        opening = _opening(measure_opening_fn(env))
    except BaseException as exc:
        fail("missing_measurement", exc)
    audit["initial_eef_pos_m"] = initial_position.tolist()
    audit["initial_eef_quat_xyzw"] = held_quaternion.tolist()
    audit["opening_readings_m"].append(opening)
    if opening < OPENING_MIN_M:
        audit["budget_used"] = int(budget.used)
        fail("below_band")

    settings = dict(motion_settings or {})
    scale = settings.get("osc_position_scale_m")
    target = episode._pose_waypoint(initial_position, held_quaternion)
    proprio = episode._proprioception(initial_observation)
    last_readings = [opening]
    motion_notified = False
    action_count = 0

    def dispatch(command: str, gripper: float) -> bool:
        nonlocal proprio, action_count, motion_notified
        if action_count >= MAX_ACTIONS or budget.remaining <= 0:
            raise PreshapeError("timeout", audit)
        action = episode.normalized_action_for_waypoint(
            proprio, target, gripper=gripper, held_rotation=held_quaternion,
            osc_position_scale_m=scale,
        )
        if not np.isfinite(action).all() or action.shape != (7,):
            raise ValueError("preshape generated invalid normalized action")
        if not motion_notified:
            if motion_started_callback is not None:
                motion_started_callback()
            motion_notified = True
        # Consume before dispatch so a failed env.step is still charged as an
        # attempted pulse/hold in the shared global budget.
        if not budget.consume():
            raise RuntimeError("preshape shared action budget exhausted")
        action_count += 1
        entry: dict[str, Any] = {"step": action_count, "command": command, "gripper": float(gripper), "action": np.asarray(action, dtype=float).tolist()}
        entry_recorded = False
        try:
            observation, done, info = episode._step_once(env, action)
            new_proprio = episode._proprioception(observation)
            position = np.asarray(new_proprio.get("eef_pos"), dtype=np.float64).reshape(-1)
            quaternion = np.asarray(new_proprio.get("eef_quat"), dtype=np.float64).reshape(-1)
            if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise ValueError("preshape action returned invalid EEF pose")
            drift = float(np.linalg.norm(position - initial_position))
            entry["eef_pos_m"] = position.tolist()
            entry["eef_drift_m"] = drift
            if drift > MAX_POSE_DRIFT_M:
                raise PreshapeError("pose_drift", audit)
            orientation_error = float(episode._rotation_error_rad(quaternion, held_quaternion))
            entry["orientation_error_rad"] = orientation_error
            if orientation_error > float(settings.get("orientation_tolerance_rad", 0.12)):
                raise PreshapeError("pose_drift", audit)
            proprio = new_proprio
            measured = _opening(measure_opening_fn(env))
            entry["opening_m"] = measured
            audit["opening_readings_m"].append(measured)
            last_readings.append(measured)
            del last_readings[:-3]
            audit["actions"].append(entry)
            entry_recorded = True
            if measured < OPENING_MIN_M:
                raise PreshapeError("below_band", audit)
            return True
        except BaseException as exc:
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)
            if not entry_recorded:
                audit["actions"].append(entry)
            raise

    try:
        while action_count < MAX_ACTIONS:
            if opening < OPENING_MIN_M:
                fail("below_band")
            stable = (
                len(last_readings) >= 3
                and all(OPENING_MIN_M <= value <= OPENING_MAX_M for value in last_readings[-3:])
                and max(last_readings[-3:]) - min(last_readings[-3:]) <= OPENING_STABILITY_M
            )
            if stable:
                audit["status"] = "completed"
                audit["final_opening_m"] = float(last_readings[-1])
                audit["settled_readings_m"] = [float(value) for value in last_readings[-3:]]
                audit["actions_sent"] = action_count
                audit["budget_used"] = int(budget.used)
                break
            settled = (
                len(last_readings) >= 3
                and max(last_readings[-3:]) - min(last_readings[-3:]) <= OPENING_STABILITY_M
            )
            if opening > OPENING_MAX_M and (len(last_readings) < 3 or settled):
                dispatch("close", 1.0)
            # Every decision is followed by zero-gripper settling actions.
            # This also handles an in-band but noisy probe without opening.
            for _ in range(SETTLE_STEPS):
                dispatch("hold", 0.0)
                opening = float(last_readings[-1])
                if opening < OPENING_MIN_M:
                    fail("below_band")
        else:
            fail("timeout")
    except PreshapeError as exc:
        audit["actions_sent"] = action_count
        audit["budget_used"] = int(budget.used)
        if not audit.get("failure_reason"):
            _failure(audit, exc.category)
        persist()
        raise PreshapeError(exc.category, audit) from exc
    except BaseException as exc:
        audit["actions_sent"] = action_count
        audit["budget_used"] = int(budget.used)
        _failure(audit, "motion_failed", exc)
        persist()
        raise PreshapeError("motion_failed", audit) from exc
    finally:
        audit.setdefault("actions_sent", action_count)
        audit.setdefault("budget_used", int(budget.used))
        persist()
    return audit


__all__ = ["perform_preshape"]
