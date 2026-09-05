"""Construct one settled LIBERO scene and record a no-command Panda frame probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script smoke
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from vla_benchmarking.arrow_grasp_controller.configs import load_controller_config
from vla_benchmarking.evaluation.run_arrow_pick_place_eval import (
    build_libero_env,
    controller_variant_from_config,
)
from vla_benchmarking.tools.sanity_checks.probe_panda_grip_site_frame import (
    probe_grip_site_frame,
)


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def run_probe(
    *,
    task_id: int,
    seed: int,
    resolution: int,
    suite_mode: str,
    controller_config: Path,
    calibration_input: Path,
) -> dict[str, Any]:
    expanded = load_controller_config(controller_config)
    variant = controller_variant_from_config(expanded, suite_mode=suite_mode)
    env = build_libero_env(
        task_id,
        seed,
        resolution,
        suite_mode=suite_mode,
        controller_variant=variant,
    )
    try:
        result = probe_grip_site_frame(
            env,
            calibration_input=calibration_input,
            angular_tolerance_deg=1.0,
        )
        result["execution"] = {
            "task_id": int(task_id),
            "seed": int(seed),
            "resolution": int(resolution),
            "suite_mode": suite_mode,
            "controller_name": variant.name,
            "controller_config_hash": expanded["config_hash"],
            "commanded_robot_motion": False,
            "physics_settling_performed": True,
            "controller_inference": False,
            "evaluator_queried": False,
        }
        result["settle_diagnostics"] = getattr(env, "_arrow_settle_diagnostics", None)
        return result
    finally:
        _close(env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--suite-mode",
        choices=("vanilla", "sealed_randomized"),
        required=True,
    )
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--calibration-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    result = run_probe(
        task_id=args.task_id,
        seed=args.seed,
        resolution=args.resolution,
        suite_mode=args.suite_mode,
        controller_config=args.controller_config,
        calibration_input=args.calibration_input,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve().as_posix())
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
