"""Capture an immutable live LIBERO camera calibration for Arrow Trace.

This utility is intentionally a compute-node artifact step.  It constructs the
same production environment used by the native policy host, captures one
synchronized RGB-D frame, and writes only camera geometry and provenance.  No
image, depth map, action, or simulator state is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from .contracts import ContractError
from .rgbd_geometry import _hash_payload, _validate_calibration


SCHEMA = "arrow_policy_suite.trace_calibration.v1"
_SHA256 = re.compile(r"^[0-9a-f]{40}$")


def build_calibration_artifact(
    calibration: Any,
    *,
    revision: str,
    expected_commit: str,
    task_id: int,
    seed: int,
    init_state_index: int,
    resolution: int,
) -> dict[str, Any]:
    """Project a runtime calibration to the minimal frozen Trace contract."""

    if not revision.strip() or revision.lower() in {"unknown", "latest", "unresolved"}:
        raise ContractError("Trace calibration revision must be explicit")
    if not _SHA256.fullmatch(expected_commit):
        raise ContractError("expected_commit must be a full lowercase Git SHA")
    try:
        frame_name = str(calibration.world_frame)
        camera_name = str(calibration.camera_name)
        intrinsics, world_from_camera = _validate_calibration(
            calibration.intrinsic, calibration.world_from_camera
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError("runtime capture did not expose a valid camera calibration") from exc
    calibration_hash = _hash_payload({
        "intrinsics": intrinsics.tolist(),
        "world_from_camera": world_from_camera.tolist(),
        "frame": frame_name,
        "revision": revision,
    })
    return {
        "schema": SCHEMA,
        "create_only": True,
        "revision": revision,
        "frame_name": frame_name,
        "intrinsics": intrinsics.tolist(),
        "world_from_camera": world_from_camera.tolist(),
        "calibration_hash": calibration_hash,
        "provenance": {
            "source_kind": "live_libero_rgbd_capture",
            "source_commit": expected_commit,
            "camera_name": camera_name,
            "task_id": int(task_id),
            "seed": int(seed),
            "init_state_index": int(init_state_index),
            "resolution": [int(resolution), int(resolution)],
            "retained_fields": ["intrinsics", "world_from_camera"],
            "omitted": ["rgb", "depth", "actions", "simulator_state"],
        },
    }


def write_create_only(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path).expanduser().resolve()
    if target.is_symlink():
        raise ContractError("Trace calibration destination must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with target.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite Trace calibration artifact: {target}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-state-index", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--suite-mode", choices=("vanilla", "sealed_randomized"), default="vanilla")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != args.expected_commit:
        raise ContractError(
            f"Trace calibration checkout mismatch: expected {args.expected_commit}, got {actual_commit}"
        )
    from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import (
        build_libero_env,
        capture_agentview,
    )

    environment = build_libero_env(
        args.task_id,
        args.seed,
        args.resolution,
        suite_mode=args.suite_mode,
        init_state_index=args.init_state_index,
    )
    try:
        capture = capture_agentview(environment, resolution=args.resolution, camera_name="agentview")
        payload = build_calibration_artifact(
            capture.calibration,
            revision=args.revision,
            expected_commit=args.expected_commit,
            task_id=args.task_id,
            seed=args.seed,
            init_state_index=args.init_state_index,
            resolution=args.resolution,
        )
        content_sha256 = write_create_only(args.output, payload)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    print(json.dumps({
        "status": "COMPLETED",
        "schema": SCHEMA,
        "output": str(Path(args.output).expanduser().resolve()),
        "content_sha256": content_sha256,
        "calibration_hash": payload["calibration_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "build_calibration_artifact", "write_create_only", "main"]
