#!/usr/bin/env python3
"""Model-free paired opening/preshape diagnostic for the T4 vanilla cell."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from . import run_arrow_pick_place_eval as episode
    from . import run_arrow_pick_place_matrix as matrix
    from . import run_molmo_sam3_canary as canary
    from .molmo_sam3 import preshape
except ImportError:  # pragma: no cover - direct script use
    import run_arrow_pick_place_eval as episode
    import run_arrow_pick_place_matrix as matrix
    import run_molmo_sam3_canary as canary
    from molmo_sam3 import preshape

try:
    from .molmo_sam3.settling import (
        CONSECUTIVE_SAMPLES, HOVER_ORIENTATION_TOLERANCE_RAD, HOVER_POSITION_TOLERANCE_M,
        MAX_SETTLING_ACTIONS, SETTLING_ORIENTATION_INCREMENT_RAD, SETTLING_POSITION_TOLERANCE_M, SHARED_ACTION_LIMIT,
        _proprio, _settle_to_original_hover, _telemetry,
    )
except ImportError:  # pragma: no cover - direct script use
    from molmo_sam3.settling import (
        CONSECUTIVE_SAMPLES, HOVER_ORIENTATION_TOLERANCE_RAD, HOVER_POSITION_TOLERANCE_M,
        MAX_SETTLING_ACTIONS, SETTLING_ORIENTATION_INCREMENT_RAD, SETTLING_POSITION_TOLERANCE_M, SHARED_ACTION_LIMIT,
        _proprio, _settle_to_original_hover, _telemetry,
    )

def run_case(
    *,
    mode: str,
    seed: int,
    output_dir: str | Path,
    env_builder: Callable[[int, int, int], Any],
    capture_fn: Callable[..., Any],
    arrow_input_builder: Callable[[Any, int, int], Mapping[str, Any]],
    open_fn: Callable[..., Mapping[str, Any]],
    hover_fn: Callable[..., Mapping[str, Any]],
    probe_fn: Callable[[Any], tuple[Any, Any, Mapping[str, Any]]],
    preshape_fn: Callable[..., Mapping[str, Any]],
    motion_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"control", "treatment"}:
        raise ValueError("mode must be control or treatment")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{mode}__seed{int(seed)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = env_builder(4, int(seed), 256)
    result: dict[str, Any] = {"mode": mode, "task_id": 4, "seed": int(seed), "suite_mode": "vanilla", "status": "failed"}
    try:
        budget = episode._ActionBudget(SHARED_ACTION_LIMIT)
        setattr(env, "_molmo_sam3_action_budget", budget)
        setattr(env, "_molmo_sam3_action_count", 0)
        initial = capture_fn(env, resolution=256, camera_name="agentview")
        arrow_inputs = dict(arrow_input_builder(env, 4, 256))
        rendered, _ = episode.render_exactly_one_arrow(
            initial.rgb, arrow_inputs["bboxes"], subject=arrow_inputs.get("subject", "bowl"),
            goal_object=arrow_inputs.get("goal_object", "plate"), anchor_policy="bbox_center",
        )
        source_uv, destination_uv = episode.decode_arrow_pixels(initial.rgb, rendered)
        open_audit = open_fn(env, output_dir=case_dir / "gripper_open", motion_settings=motion_settings or {})
        # Probe after the established open preflight, before hover and before
        # every preshape measurement.  It is robot-only and model-free.
        _calibration, _transform, probe = probe_fn(env)
        geometry_audit = probe.get("gripper_geometry", {}) if isinstance(probe, Mapping) else {}
        opening_measurements: list[float] = []
        controller_snapshots: list[Mapping[str, Any]] = []

        def measure_opening(measure_env: Any) -> float:
            _measured_calibration, _measured_transform, measured_probe = probe_fn(measure_env)
            measured_geometry = measured_probe.get("gripper_geometry", {}) if isinstance(measured_probe, Mapping) else {}
            value = float(measured_geometry.get("measured_opening_m"))
            opening_measurements.append(value)
            controller_snapshots.append(_telemetry(measure_env, motion_settings or {}))
            return value

        hover_observation = episode._raw_observation(env)
        hover_audit = hover_fn(
            env, initial, source_uv, output_dir=case_dir / "observation_hover",
            motion_settings={"observation_profile": "hover20mm", **dict(motion_settings or {})},
        )
        result.update({
            "arrow_identity": {"source_uv": list(source_uv), "destination_uv": None if destination_uv is None else list(destination_uv)},
            "gripper_open": open_audit, "observation_hover": hover_audit,
            "opening_before_preshape_m": float(geometry_audit.get("measured_opening_m")) if geometry_audit.get("measured_opening_m") is not None else None,
            "controller_telemetry_snapshots": controller_snapshots,
        })
        if mode == "treatment":
            result["settling"] = _settle_to_original_hover(
                env, hover_audit=hover_audit, initial_hover_observation=hover_observation,
                output_dir=case_dir / "opening_settling", motion_settings=motion_settings or {},
            )
        preshape_audit = preshape_fn(
            env, measure_opening_fn=measure_opening, output_dir=case_dir / "preshape",
            motion_settings=motion_settings or {},
        )
        result["preshape"] = dict(preshape_audit)
        final_opening = preshape_audit.get("final_opening_m")
        if preshape_audit.get("status") != "completed" or final_opening is None or not np.isfinite(float(final_opening)) or not (0.035 <= float(final_opening) <= 0.045):
            raise RuntimeError("preshape did not complete within the fixed 35-45 mm opening band")
        result["opening_measurements_m"] = opening_measurements
        result["total_actions"] = int(budget.used)
        result["candidate_actions"] = 0
        result["evaluator_calls"] = 0
        result["diagnostic_kind"] = "model_free_opening_preshape"
        result["status"] = "completed"
        return result
    except Exception as exc:
        settling_audit = getattr(env, "_molmo_opening_settling_audit", None)
        if isinstance(settling_audit, Mapping):
            result.setdefault("settling", dict(settling_audit))
        if isinstance(getattr(exc, "audit", None), Mapping):
            result.setdefault("preshape", dict(exc.audit))
        result.update({"error_type": type(exc).__name__, "error": str(exc), "diagnostic_kind": "model_free_opening_preshape"})
        budget = getattr(env, "_molmo_sam3_action_budget", None)
        result["total_actions"] = int(budget.used) if isinstance(budget, episode._ActionBudget) else None
        return result
    finally:
        try:
            env.close()
        except Exception as exc:
            result.setdefault("close_error", str(exc))


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(canary._json_safe(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _diagnostic_outcomes(cases: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    treatment_cases = [case for case in cases if case.get("mode") == "treatment"]
    control_cases = [case for case in cases if case.get("mode") == "control"]
    control_failures = [
        case.get("preshape", {}).get("failure_reason")
        for case in control_cases
        if isinstance(case.get("preshape"), Mapping)
    ]
    return {
        "treatment_passed": bool(len(treatment_cases) == 2 and all(case.get("status") == "completed" for case in treatment_cases)),
        "control_reproduced": bool(len(control_cases) == 2 and control_failures == ["pose_drift", "pose_drift"]),
        "control_completed": bool(len(control_cases) == 2 and all(case.get("status") == "completed" for case in control_cases)),
    }


def run_opening_probe(*, output_dir: str | Path, label: str = "molmo_opening_probe") -> dict[str, Any]:
    """Run control/treatment pairs in fixed order, preserving each case."""
    root = Path(output_dir).expanduser().resolve()
    summary: dict[str, Any] = {
        "schema": "molmo_opening_probe.v1", "label": str(label),
        "diagnostic_kind": "model_free_opening_preshape", "task_id": 4,
        "suite_mode": "vanilla", "seeds": [1000, 1001],
        "models_requested": False, "candidate_execution": False, "evaluator_execution": False,
        "cases": [],
    }

    def build(task_id: int, seed: int, resolution: int) -> Any:
        return episode.build_libero_env(
            task_id, seed, resolution, suite_mode="vanilla",
            controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
        )

    settings = {"observation_profile": "hover20mm", "motion_diagnostics": True}
    for mode in ("control", "treatment"):
        for seed in (1000, 1001):
            case = run_case(
                mode=mode, seed=seed, output_dir=root, env_builder=build,
                capture_fn=episode.capture_agentview, arrow_input_builder=matrix._default_arrow_inputs,
                open_fn=canary._perform_gripper_open, hover_fn=canary._perform_observation_hover,
                probe_fn=canary.probe_robot_calibration, preshape_fn=preshape.perform_preshape,
                motion_settings=settings,
            )
            summary["cases"].append(case)
            _atomic_write(root / f"{label}__summary.json", summary)
    summary["durable_case_count"] = len(summary["cases"])
    summary["completed_case_count"] = sum(case.get("status") == "completed" for case in summary["cases"])
    summary.update(_diagnostic_outcomes(summary["cases"]))
    _atomic_write(root / f"{label}__summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="molmo_opening_probe")
    args = parser.parse_args(argv)
    summary = run_opening_probe(output_dir=args.output_dir, label=args.label)
    print(json.dumps({"summary_path": (Path(args.output_dir).resolve() / f"{args.label}__summary.json").as_posix(), "completed_case_count": summary.get("completed_case_count", 0)}, sort_keys=True))
    return 0 if summary.get("durable_case_count") == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
