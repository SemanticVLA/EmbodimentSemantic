"""Run a bounded, standalone RoboCasa PandaOmron runtime motion probe.

The probe is deliberately separate from scored evaluation cells.  It performs
one reset, a no-motion calibration/introspection pass, and exactly ten copies
of the canonical 7-D action through :class:`RoboCasaControllerEnv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..environment.runtime import create_robocasa_env
from ..evaluation.live import (
    RoboCasaControllerEnv,
    _current_eef_b0,
    _diagnostic_safe,
    _runtime_class,
    _source_hashes,
    _write_diagnostic_json,
    begin_motion_diagnostics,
    collect_pre_motion_diagnostics,
    finalize_motion_diagnostics,
)


CANONICAL_ACTION: tuple[float, ...] = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0)
PROBE_STEPS = 10
MIN_POSITIVE_Z_DISPLACEMENT_M = 0.005


def _json_default(value: Any) -> Any:
    return _diagnostic_safe(value)


def run_runtime_probe(
    *,
    task_name: str,
    seed: int,
    output: Path,
    execute_motion: bool = False,
) -> dict[str, Any]:
    """Execute the bounded probe and always write a JSON evidence record."""
    output = Path(output).expanduser().resolve()
    evidence: dict[str, Any] = {
        "schema": "robocasa_runtime_motion_probe.v1",
        "task": str(task_name),
        "seed": int(seed),
        "required_steps": PROBE_STEPS,
        "canonical_action": list(CANONICAL_ACTION),
        "min_positive_z_displacement_m": MIN_POSITIVE_Z_DISPLACEMENT_M,
        "execute_motion_requested": bool(execute_motion),
        "success": False,
        "exceptions": [],
        "source_hashes": _source_hashes(),
    }
    raw_env: Any = None
    wrapped: RoboCasaControllerEnv | None = None
    initial: Mapping[str, Any] | None = None
    try:
        if not execute_motion:
            evidence["exceptions"].append({
                "stage": "guard", "type": "MotionNotAuthorized",
                "error": "--execute-motion is required for the standalone motion probe",
            })
            return evidence
        raw_env = create_robocasa_env(task_name, seed=seed)
        evidence["raw_environment_class"] = _runtime_class(raw_env)
        wrapped = RoboCasaControllerEnv(raw_env)
        wrapped.reset()
        initial = begin_motion_diagnostics(wrapped)
        preflight = collect_pre_motion_diagnostics(
            wrapped, task_name=task_name, seed=seed
        )
        evidence["pre_motion_diagnostics"] = preflight
        evidence["calibration_passed"] = bool(preflight.get("calibration_passed"))
        evidence["calibration_record"] = preflight.get("calibration")
        controller_contract = (
            preflight.get("controller", {}).get("runtime_controller", {})
            if isinstance(preflight.get("controller"), Mapping)
            else {}
        )
        evidence["controller_contract_matches"] = bool(
            controller_contract.get("contract_matches")
        )
        if not evidence["calibration_passed"] or not evidence["controller_contract_matches"]:
            evidence["exceptions"].append({
                "stage": "runtime_contract_gate",
                "type": "CalibrationOrControllerContractFailed",
                "error": "read-only calibration or instantiated controller contract did not pass",
            })
            return evidence

        for index in range(PROBE_STEPS):
            try:
                wrapped.step(CANONICAL_ACTION)
                evidence.setdefault("step_results", []).append({"index": index, "status": "sent"})
            except Exception as exc:
                evidence.setdefault("step_results", []).append({
                    "index": index, "status": "error", "type": type(exc).__name__, "error": str(exc)
                })
                evidence["exceptions"].append({
                    "stage": "step", "index": index, "type": type(exc).__name__, "error": str(exc)
                })
                break
    except Exception as exc:
        evidence["exceptions"].append({
            "stage": "runtime", "type": type(exc).__name__, "error": str(exc)
        })
    finally:
        if wrapped is not None:
            try:
                motion = finalize_motion_diagnostics(wrapped, initial, outcome="probe")
                evidence["motion_diagnostics"] = motion
                evidence["actions"] = motion.get("actions", [])
                evidence["action_count"] = motion.get("action_count", 0)
                evidence["arm_command_count"] = motion.get("arm_command_count", 0)
                evidence["gripper_only_command_count"] = motion.get("gripper_only_command_count", 0)
                evidence["eef_trajectory_b0_m"] = motion.get("eef_trajectory_b0_m", [])
                evidence["eef_displacement_m"] = motion.get("eef_displacement_m")
                evidence["max_eef_displacement_m"] = motion.get("max_eef_displacement_m")
                trajectory = np.asarray(motion.get("eef_trajectory_b0_m", []), dtype=np.float64)
                if trajectory.ndim == 2 and trajectory.shape[1] == 3 and len(trajectory) >= 2 and np.isfinite(trajectory).all():
                    evidence["z_delta_m"] = float(trajectory[-1, 2] - trajectory[0, 2])
                    evidence["z_sign"] = "positive" if evidence["z_delta_m"] > 0.0 else "non_positive"
                else:
                    evidence["z_delta_m"] = None
                    evidence["z_sign"] = "unknown"
                packed = [item.get("packed_action", []) for item in motion.get("actions", [])]
                base_values = [float(value) for action in packed for value in action[7:11]] if packed else []
                torso_values = [float(action[10]) for action in packed if len(action) > 10]
                evidence["base_torso_zero"] = bool(
                    packed and all(abs(value) == 0.0 for value in base_values + torso_values)
                )
                evidence["exactly_ten_commands"] = bool(motion.get("action_count") == PROBE_STEPS)
                evidence["all_steps_sent"] = bool(
                    wrapped.steps == PROBE_STEPS
                    and len(motion.get("actions", [])) == PROBE_STEPS
                    and all(item.get("status") == "sent" for item in motion.get("actions", []))
                )
                evidence["all_actions_canonical"] = bool(
                    len(motion.get("actions", [])) == PROBE_STEPS
                    and all(item.get("controller_action") == list(CANONICAL_ACTION) for item in motion.get("actions", []))
                )
                z_delta = evidence.get("z_delta_m")
                b0 = evidence.get("pre_motion_diagnostics", {}).get("b0", {})
                b0_matrix = np.asarray(b0.get("base_from_world"), dtype=np.float64)
                valid_b0 = b0_matrix.shape == (4, 4) and np.isfinite(b0_matrix).all()
                base_comparison = b0.get("base_center_comparison", {})
                evidence["valid_b0_source"] = bool(
                    valid_b0
                    and motion.get("initial_eef_b0_m") is not None
                    and base_comparison.get("available")
                    and base_comparison.get("matches")
                )
                evidence["motion_gate_passed"] = bool(
                    evidence["exactly_ten_commands"]
                    and evidence["all_steps_sent"]
                    and evidence["all_actions_canonical"]
                    and evidence["base_torso_zero"]
                    and not evidence["exceptions"]
                    and evidence["valid_b0_source"]
                    and evidence.get("controller_contract_matches", False)
                    and isinstance(z_delta, float)
                    and np.isfinite(z_delta)
                    and z_delta >= MIN_POSITIVE_Z_DISPLACEMENT_M
                    and z_delta > 0.0
                )
            except Exception as exc:
                evidence["exceptions"].append({
                    "stage": "motion_diagnostics", "type": type(exc).__name__, "error": str(exc)
                })
        close = getattr(raw_env, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                evidence["exceptions"].append({
                    "stage": "close", "type": type(exc).__name__, "error": str(exc)
                })
        evidence["success"] = bool(
            evidence.get("calibration_passed")
            and evidence.get("motion_gate_passed")
            and not evidence.get("exceptions")
        )
        evidence["exit_code"] = 0 if evidence["success"] else 2
        try:
            wrote_evidence = bool(_write_diagnostic_json(output, evidence))
        except Exception as exc:
            wrote_evidence = False
            write_error = f"{type(exc).__name__}: {exc}"
        if not wrote_evidence:
            # There is no trustworthy evidence artifact, so success is
            # impossible even when the motion gate itself passed.
            evidence["exceptions"].append({
                "stage": "write", "type": "DiagnosticsWriteFailed",
                "error": locals().get("write_error", f"could not write probe evidence to {output}"),
            })
            evidence["success"] = False
            evidence["exit_code"] = 2
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-motion", action="store_true")
    args = parser.parse_args(argv)
    evidence = run_runtime_probe(
        task_name=args.task, seed=args.seed, output=args.output,
        execute_motion=args.execute_motion,
    )
    return int(evidence.get("exit_code", 2))


__all__ = [
    "CANONICAL_ACTION", "MIN_POSITIVE_Z_DISPLACEMENT_M", "PROBE_STEPS",
    "main", "run_runtime_probe",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
