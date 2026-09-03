#!/usr/bin/env python3
"""Run the bounded, perception-only T4 robot-exclusion clearance probe.

The canary is allowed to perform its normal observation capture/hover.  The
candidate seam is intercepted, evaluated twice with the *same* RGB-D/Molmo
inputs, and replaced with an empty result so no candidate motion or evaluator
can run.  This file deliberately does not alter the canary runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .molmo_sam3 import grasp_candidates as geometry
    from . import run_molmo_sam3_canary as canary
except ImportError:  # pragma: no cover - direct script use
    from molmo_sam3 import grasp_candidates as geometry
    import run_molmo_sam3_canary as canary


class _StopAfterProbe(RuntimeError):
    """Internal control flow after the first durable matrix cell."""


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _safe(vars(value))
    return str(value)


def _hash_bytes(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _point_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _safe(value)
    return {
        "u": float(getattr(value, "u", getattr(value, "x", value[0] if hasattr(value, "__getitem__") else 0.0))),
        "v": float(getattr(value, "v", getattr(value, "y", value[1] if hasattr(value, "__getitem__") else 0.0))),
        "confidence": float(getattr(value, "confidence", 1.0)),
        "label": str(getattr(value, "label", "rim")),
    }


def _result_record(result: Any) -> dict[str, Any]:
    candidates = tuple(getattr(result, "candidates", ()))
    rejected = tuple(getattr(result, "rejected", ()))
    return {
        "policy": str(getattr(result, "policy", "unknown")),
        "candidate_count": len(candidates),
        "admitted_candidate_ids": [str(item.candidate_id) for item in candidates],
        # Candidate clearance is the completed swept-sample minimum, rather
        # than a first-hit value.  Keep the full audit for investigation.
        "actual_clearances_m": [float(item.clearance_m) for item in candidates],
        "rejection_count": len(rejected),
        "rejections": [
            {
                "seed_index": int(item.seed_index), "yaw_deg": float(item.yaw_deg),
                "insertion_depth_m": float(item.insertion_depth_m),
                "reason": str(item.reason), "details": _safe(item.details),
            }
            for item in rejected
        ],
        "audit": _safe(getattr(result, "audit", {})),
    }


def run_geometry_passes(
    seam_kwargs: Mapping[str, Any],
    *,
    output_dir: str | Path,
    label: str = "molmo_geometry_probe",
    generate_fn: Callable[..., Any] = geometry.generate_grasp_candidates,
    execution_sha: str | None = None,
    task_id: int = 4,
    seed: int = 1000,
    suite_mode: str = "vanilla",
) -> dict[str, Any]:
    """Run baseline and reduced-admission geometry on one immutable seam.

    ``seam_kwargs`` is the complete argument set from the canary's normal
    ``generate_grasp_candidates`` call.  The function never captures or steps
    an environment and is therefore also useful for focused unit tests.
    """
    required = {"rgb", "metric_depth_m", "sam_mask", "calibration", "robot_calibration", "molmo_points", "policy"}
    missing = sorted(required - set(seam_kwargs))
    if missing:
        raise ValueError(f"geometry seam is missing required inputs: {', '.join(missing)}")
    policy = seam_kwargs["policy"]
    if not isinstance(policy, geometry.CandidatePolicy):
        raise TypeError("geometry seam policy must be CandidatePolicy")
    base_policy = replace(policy, obstruction_clearance_m=0.006, robot_exclusion_clearance_m=0.006)
    reduced_policy = replace(policy, obstruction_clearance_m=0.005, robot_exclusion_clearance_m=0.006)
    baseline = generate_fn(**{**dict(seam_kwargs), "policy": base_policy})
    reduced = generate_fn(**{**dict(seam_kwargs), "policy": reduced_policy})
    rgb = np.asarray(seam_kwargs["rgb"])
    depth = np.asarray(seam_kwargs["metric_depth_m"])
    mask = np.asarray(seam_kwargs["sam_mask"])
    calibration = seam_kwargs["calibration"]
    input_hash = _hash_bytes(rgb, depth, mask, calibration.intrinsic, calibration.world_from_camera)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{label}__inputs.npz"
    np.savez_compressed(npz_path, rgb=rgb, metric_depth_m=depth, sam_mask=mask,
                        intrinsic=np.asarray(calibration.intrinsic),
                        world_from_camera=np.asarray(calibration.world_from_camera))
    base_policy_payload = _safe(asdict(policy))
    base_robot_payload = _safe(asdict(seam_kwargs["robot_calibration"]) if hasattr(seam_kwargs["robot_calibration"], "__dataclass_fields__") else seam_kwargs["robot_calibration"])
    robot_calibration_sha256 = hashlib.sha256(json.dumps(base_robot_payload, sort_keys=True).encode()).hexdigest()
    baseline_record = _result_record(baseline)
    baseline_record["policy_config"] = _safe(asdict(base_policy))
    reduced_record = _result_record(reduced)
    reduced_record["policy_config"] = _safe(asdict(reduced_policy))
    provenance = {
        "schema": "molmo_geometry_probe.v1",
        "label": str(label), "task_id": int(task_id), "seed": int(seed), "suite_mode": str(suite_mode),
        "variant": "molmo_dense_agentview", "region_backend": "rgbd",
        "observation_profile": "hover20mm", "camera_name": "agentview",
        "input_hash": input_hash,
        "execution_sha": execution_sha,
        "robot_calibration_sha256": robot_calibration_sha256,
        "camera_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "calibration": _safe(asdict(calibration) if hasattr(calibration, "__dataclass_fields__") else calibration),
        "robot_calibration": base_robot_payload,
        "molmo_points": [_point_safe(item) for item in seam_kwargs["molmo_points"]],
        "source_policy": base_policy_payload,
        "passes": {"baseline_6mm_robot_6mm": baseline_record, "admission_5mm_robot_6mm": reduced_record},
        "npz": {"path": npz_path.name, "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest()},
        "candidate_execution": {"candidate_result_replaced_with_empty": True, "candidate_actions": 0, "evaluator_calls": 0},
    }
    json_path = root / f"{label}__geometry.json"
    payload = json.dumps(_safe(provenance), indent=2, sort_keys=True) + "\n"
    provenance["json_payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    json_path.write_text(json.dumps(_safe(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="molmo_geometry_probe")
    args = parser.parse_args(argv)
    captured: dict[str, Any] = {}
    requested_cli = [
        "--variant", "molmo_dense_agentview", "--output-dir", str(args.output_dir),
        "--label", args.label, "--region-backend", "rgbd", "--observation-profile", "hover20mm",
        "--episodes-per-task", "1", "--task-ids", "4", "--suite-modes", "vanilla",
        "--seed-base", "1000", "--molmopoint-prompt-id", "rim_clearance",
    ]
    captured["requested_cli"] = requested_cli
    captured["execution_sha"] = canary._execution_provenance(require_clean=False).get("execution_sha")
    captured["canary_cli_constraint"] = "main requires task IDs 4,6,9 and suite modes vanilla,sealed_randomized"
    callback_state = {"reached": False, "record": None}
    guard_counts = {"run_episode": 0, "evaluator": 0}

    original_generate = geometry.generate_grasp_candidates
    original_run_episode: Any = None
    original_run_matrix: Any = None
    try:
        try:
            from . import run_arrow_pick_place_eval as episode
        except ImportError:  # pragma: no cover
            import run_arrow_pick_place_eval as episode
        original_run_episode = episode.run_episode

        def guard(*_args: Any, **_kwargs: Any) -> Any:
            guard_counts["run_episode"] += 1
            raise RuntimeError("geometry probe must not execute episode motion")

        def guard_evaluator(*_args: Any, **_kwargs: Any) -> Any:
            guard_counts["evaluator"] += 1
            raise RuntimeError("geometry probe must not execute evaluator")

        def intercept(**kwargs: Any) -> Any:
            if "policy" in captured:
                raise RuntimeError("geometry probe received more than one perception seam")
            captured.update(kwargs)
            baseline_and_reduced = run_geometry_passes(
                kwargs, output_dir=args.output_dir, label=args.label, generate_fn=original_generate,
                execution_sha=captured["execution_sha"], task_id=4, seed=1000, suite_mode="vanilla",
            )
            captured["probe"] = baseline_and_reduced
            # Empty CandidateSet is deliberate: the normal canary then closes
            # the cell without invoking run_episode or an evaluator.
            return geometry.GraspCandidateResult((), (), (), str(kwargs["policy"].name), {"probe_replaced": True})

        geometry.generate_grasp_candidates = intercept
        episode.run_episode = guard
        try:
            from . import run_arrow_pick_place_matrix as matrix
        except ImportError:  # pragma: no cover
            import run_arrow_pick_place_matrix as matrix
        original_run_matrix = matrix.run_matrix

        def guarded_matrix(*matrix_args: Any, **matrix_kwargs: Any) -> Any:
            matrix_kwargs["evaluator"] = guard_evaluator
            metadata = matrix_kwargs.get("experiment_metadata")
            diagnostic_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            diagnostic_metadata["diagnostic_kind"] = "perception_geometry_probe"
            matrix_kwargs["experiment_metadata"] = diagnostic_metadata
            return original_run_matrix(*matrix_args, **matrix_kwargs)

        matrix.run_matrix = guarded_matrix

        def stop_after_first(_record: Mapping[str, Any]) -> None:
            callback_state["reached"] = True
            callback_state["record"] = dict(_record)
            raise _StopAfterProbe()

        # Main intentionally validates the frozen 4,6,9/two-suite screen.
        # Reach its first durable cell with one accepted invocation; the exact
        # narrow request remains recorded above as the diagnostic intent.
        accepted_cli = [
            "--variant", "molmo_dense_agentview", "--output-dir", str(args.output_dir),
            "--label", args.label, "--region-backend", "rgbd", "--observation-profile", "hover20mm",
            "--episodes-per-task", "1", "--task-ids", "4,6,9", "--suite-modes", "vanilla,sealed_randomized",
            "--seed-base", "1000", "--molmopoint-prompt-id", "rim_clearance",
        ]
        status = canary.main(accepted_cli, cell_completed_callback=stop_after_first)
    except _StopAfterProbe:
        status = 2
    finally:
        geometry.generate_grasp_candidates = original_generate
        if original_run_episode is not None:
            episode.run_episode = original_run_episode
        if original_run_matrix is not None:
            matrix.run_matrix = original_run_matrix
    if not captured.get("probe") or not callback_state["reached"]:
        raise RuntimeError("canary completed without a geometry seam capture")
    callback_record = callback_state["record"]
    if not isinstance(callback_record, Mapping) or (
        int(callback_record.get("task_id", -1)) != 4
        or int(callback_record.get("seed", -1)) != 1000
        or str(callback_record.get("suite_mode", "")) != "vanilla"
        or str(callback_record.get("status", "")) != "completed"
        or callback_record.get("evaluator_result") is not None
    ):
        raise RuntimeError("geometry probe callback did not report a completed evaluator-null T4 vanilla seed-1000 cell")
    if guard_counts != {"run_episode": 0, "evaluator": 0}:
        raise RuntimeError(f"geometry probe execution guard was invoked: {guard_counts}")
    if status not in (0, 2):
        raise RuntimeError(f"unexpected canary status {status}")
    # The canary catches the deliberate callback stop and returns 2 while
    # writing its partial manifest.  Any other return-2 path lacks this proof.
    if status == 2 and not callback_state["reached"]:
        raise RuntimeError("canary returned operation error before controlled stop")
    probe_json = Path(args.output_dir).expanduser().resolve() / f"{args.label}__geometry.json"
    captured["probe"]["requested_cli"] = requested_cli
    captured["probe"]["canary_cli_constraint"] = captured.get("canary_cli_constraint")
    captured["probe"]["guard_counts"] = dict(guard_counts)
    captured["probe"]["controlled_stop"] = True
    captured["probe"]["first_cell"] = _safe(callback_record)
    final_payload = dict(captured["probe"])
    final_payload["json_payload_sha256"] = hashlib.sha256(
        (json.dumps(_safe({key: value for key, value in final_payload.items() if key != "json_payload_sha256"}), indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    probe_json.write_text(json.dumps(_safe(final_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "probe": final_payload}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
