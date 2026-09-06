"""RoboCasa-owned boundary for the unchanged arrow motion engine.

The policy implementation remains the existing canonical arrow controller.  We
adapt only the camera label expected by that controller's LIBERO-era contract;
the underlying pixels, calibration numbers, waypoints, phases, and action
logic are unchanged.  Keeping this shim here means the RoboCasa evaluation
package has one local controller entrypoint and no edits are required in the
LIBERO suite.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def run_episode(
    *,
    env: Any,
    capture: Any,
    arrow_rgb: Any,
    bboxes: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    seed: int,
    source: str,
    destination: str,
    resolution: int,
    evaluator: Callable[[Any], bool],
) -> dict[str, Any]:
    """Run one RoboCasa cell through the unchanged canonical arrow engine."""

    try:
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as canonical
    except ImportError as exc:  # pragma: no cover - live runtime boundary
        raise RuntimeError(
            "the existing arrow grasp controller runtime is unavailable; "
            "install the controller dependencies before executing RoboCasa"
        ) from exc

    # The controller's public capture contract predates RoboCasa and names its
    # camera ``agentview``.  The actual RGB-D/calibration values remain the
    # RoboCasa camera; this is only an internal compatibility label, recorded
    # separately by the RoboCasa result row.
    calibration = replace(capture.calibration, camera_name=canonical.CAMERA_NAME)
    controller_capture = replace(capture, calibration=calibration)
    return canonical.run_episode(
        env=env,
        task_id=0,
        seed=int(seed),
        output_dir=output_dir,
        arrow_rgb=arrow_rgb,
        bboxes=bboxes,
        dry_run=False,
        resolution=int(resolution),
        goal_object=destination,
        subject=source,
        capture=controller_capture,
        allow_unvalidated_profile=True,
        evaluator=evaluator,
    )


__all__ = ["run_episode"]
