"""Robot-only settling at the original observation-hover pose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .. import run_arrow_pick_place_eval as episode
except ImportError:  # pragma: no cover - direct script use
    import run_arrow_pick_place_eval as episode


MAX_SETTLING_ACTIONS = 160
SHARED_ACTION_LIMIT = 1200
SETTLING_POSITION_TOLERANCE_M = 0.0005
SETTLING_ORIENTATION_INCREMENT_RAD = 0.01
HOVER_POSITION_TOLERANCE_M = 0.020
HOVER_ORIENTATION_TOLERANCE_RAD = 0.12
CONSECUTIVE_SAMPLES = 5


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _proprio(observation: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    values = episode._proprioception(observation)
    position = np.asarray(values.get("eef_pos"), dtype=np.float64).reshape(-1)
    quaternion = np.asarray(values.get("eef_quat"), dtype=np.float64).reshape(-1)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise RuntimeError("opening probe requires finite EEF position and quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise RuntimeError("opening probe received a degenerate EEF quaternion")
    return position, quaternion / norm


def _telemetry(env: Any, motion_settings: Mapping[str, Any]) -> dict[str, Any]:
    try:
        telemetry_value = episode._experimental_controller_telemetry(env)
    except Exception as exc:  # diagnostics cannot mask the probe
        telemetry_value = {"status": "unavailable", "reason": str(exc)}
    owners: list[Any] = [env, getattr(env, "env", None)]
    for owner in tuple(owners):
        if owner is None:
            continue
        robots = getattr(owner, "robots", None)
        if isinstance(robots, (list, tuple)):
            owners.extend(robots[:2])
        robot = getattr(owner, "robot", None)
        if robot is not None:
            owners.append(robot)
    for owner in tuple(owners):
        controller = getattr(owner, "controller", None) if owner is not None else None
        if controller is not None:
            owners.append(controller)
    eef_name = next((getattr(owner, name, None) for owner in owners for name in ("eef_name", "grip_site_name") if getattr(owner, name, None) is not None), None)
    controller_scale = None
    use_delta = None
    controller_bounds: dict[str, Any] = {}
    for owner in owners:
        if owner is None:
            continue
        for name in ("action_scale", "output_scale", "output_scales", "osc_position_scale_m", "position_scale"):
            candidate = getattr(owner, name, None)
            if candidate is not None:
                controller_scale = candidate
                break
        if use_delta is None and hasattr(owner, "use_delta"):
            use_delta = bool(getattr(owner, "use_delta"))
        for name in ("input_min", "input_max", "output_min", "output_max"):
            candidate = getattr(owner, name, None)
            if candidate is not None:
                controller_bounds[name] = candidate
    controller = dict(telemetry_value) if isinstance(telemetry_value, Mapping) else {"status": "unavailable"}
    controller["output_scale"] = controller_scale
    controller["use_delta"] = use_delta
    controller["bounds"] = controller_bounds
    return {
        "eef_name": None if eef_name is None else str(eef_name),
        "output_scale_m": motion_settings.get("osc_position_scale_m"),
        "controller": controller,
    }


def _settle_to_original_hover(
    env: Any,
    *,
    hover_audit: Mapping[str, Any],
    initial_hover_observation: Mapping[str, Any] | None,
    output_dir: Path,
    motion_settings: Mapping[str, Any],
    motion_started_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Command the original hover pose until five fresh robot samples settle."""
    budget = getattr(env, "_molmo_sam3_action_budget", None)
    if not isinstance(budget, episode._ActionBudget):
        raise RuntimeError("opening settling requires shared _ActionBudget")
    target = np.asarray(hover_audit.get("hover_world_m"), dtype=np.float64).reshape(-1)
    region_q90 = float(hover_audit.get("region_q90_world_z_m"))
    if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(region_q90):
        raise RuntimeError("opening settling lacks the original hover target or observed q90")
    _prehover_position, initial_quaternion = _proprio(initial_hover_observation)
    target_waypoint = episode._pose_waypoint(target, initial_quaternion)
    settle_observation = episode._raw_observation(env)
    current_position, current_quaternion = _proprio(settle_observation)
    proprio = episode._proprioception(settle_observation)
    actions: list[dict[str, Any]] = []
    consecutive = 0
    previous_position = current_position
    previous_quaternion = current_quaternion
    motion_notified = False
    audit: dict[str, Any] | None = None
    try:
        for index in range(MAX_SETTLING_ACTIONS):
            if budget.remaining <= 0:
                raise RuntimeError("opening settling exhausted shared action budget")
            action = episode.normalized_action_for_waypoint(
                proprio, target_waypoint, gripper=0.0,
                held_rotation=initial_quaternion,
                osc_position_scale_m=motion_settings.get("osc_position_scale_m"),
            )
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("opening settling generated invalid normalized action")
            if not motion_notified:
                if motion_started_callback is not None:
                    motion_started_callback()
                motion_notified = True
            budget.consume()
            try:
                returned_observation, _done, _info = episode._step_once(env, action)
            except BaseException as exc:
                actions.append({"step": index + 1, "action_sent": True, "action": action.tolist(), "error_type": type(exc).__name__, "error": str(exc), "shared_budget_used": int(budget.used)})
                raise
            returned_position, returned_quaternion = _proprio(returned_observation)
            fresh_observation = episode._raw_observation(env)
            fresh_position, fresh_quaternion = _proprio(fresh_observation)
            displacement = float(np.linalg.norm(fresh_position - target))
            hover_orientation_error = float(episode._rotation_error_rad(fresh_quaternion, initial_quaternion))
            orientation_increment = float(episode._rotation_error_rad(fresh_quaternion, previous_quaternion))
            sample_displacement = float(np.linalg.norm(fresh_position - previous_position))
            height_margin = float(fresh_position[2] - region_q90)
            entry = {
                "step": index + 1, "action_sent": True, "gripper": 0.0,
                "action": action.tolist(), "returned_eef_pos_m": returned_position.tolist(),
                "returned_eef_quat_xyzw": returned_quaternion.tolist(),
                "fresh_eef_pos_m": fresh_position.tolist(), "fresh_eef_quat_xyzw": fresh_quaternion.tolist(),
                "displacement_from_nominal_hover_m": displacement,
                "displacement_from_previous_sample_m": sample_displacement,
                "orientation_increment_rad": orientation_increment,
                "orientation_error_from_nominal_rad": hover_orientation_error,
                "actual_height_margin_m": height_margin,
                "shared_budget_used": int(budget.used),
            }
            if displacement > HOVER_POSITION_TOLERANCE_M or hover_orientation_error > HOVER_ORIENTATION_TOLERANCE_RAD or height_margin < 0.080:
                entry["within_sample_bounds"] = False
                entry["safety_failure"] = "hover_envelope"
                actions.append(entry)
                raise RuntimeError("opening settling left the original hover safety envelope")
            within = sample_displacement <= SETTLING_POSITION_TOLERANCE_M and orientation_increment <= SETTLING_ORIENTATION_INCREMENT_RAD
            consecutive = consecutive + 1 if within else 0
            telemetry = _telemetry(env, motion_settings)
            controller_telemetry = telemetry.get("controller", {})
            entry.update({
                "within_sample_bounds": within, "controller_telemetry": telemetry,
                "controller_ee_pos_m": controller_telemetry.get("controller_ee_pos_m") if isinstance(controller_telemetry, Mapping) else None,
                "controller_goal_pos_m": controller_telemetry.get("controller_goal_pos_m") if isinstance(controller_telemetry, Mapping) else None,
                "controller_output_scale": controller_telemetry.get("output_scale") if isinstance(controller_telemetry, Mapping) else None,
                "eef_name": telemetry.get("eef_name"),
            })
            actions.append(entry)
            proprio = episode._proprioception(fresh_observation)
            previous_position = fresh_position
            previous_quaternion = fresh_quaternion
            if consecutive >= CONSECUTIVE_SAMPLES:
                audit = _json_safe({
                    "status": "completed", "target_hover_world_m": target.tolist(),
                    "region_q90_world_z_m": region_q90, "actions": actions,
                    "actions_sent": len(actions), "shared_budget_used": int(budget.used),
                    "consecutive_settled_samples": consecutive,
                })
                setattr(env, "_molmo_opening_settling_audit", audit)
                return audit
        raise RuntimeError("opening settling exceeded 160 actions")
    except BaseException as exc:
        audit = _json_safe({
            "status": "failed", "target_hover_world_m": target.tolist(),
            "region_q90_world_z_m": region_q90, "actions": actions,
            "actions_sent": len(actions), "shared_budget_used": int(budget.used),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        setattr(env, "_molmo_opening_settling_audit", audit)
        raise
    finally:
        setattr(env, "_molmo_sam3_action_count", int(budget.used))
        persisted = audit if audit is not None else getattr(env, "_molmo_opening_settling_audit", None)
        if isinstance(persisted, Mapping):
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "opening_settling_audit.json").write_text(
                    json.dumps(_json_safe(dict(persisted)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            except Exception:
                pass


__all__ = [
    "CONSECUTIVE_SAMPLES", "HOVER_ORIENTATION_TOLERANCE_RAD", "HOVER_POSITION_TOLERANCE_M",
    "MAX_SETTLING_ACTIONS", "SETTLING_ORIENTATION_INCREMENT_RAD", "SETTLING_POSITION_TOLERANCE_M", "SHARED_ACTION_LIMIT",
    "_proprio", "_settle_to_original_hover", "_telemetry",
]
