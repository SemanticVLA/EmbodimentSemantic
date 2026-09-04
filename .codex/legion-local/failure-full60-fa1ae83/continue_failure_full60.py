#!/usr/bin/env python3
"""Continue the completed failure-screen winner without replaying its 12-cell prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_EXECUTION_SHA = "fa1ae83b4560f831cc1173994f58d0ad9c3c3639"
EXPECTED_PREFIX_LABEL = "molmo_failure_screen_fa1ae83"
EXPECTED_ARM = "failure_opening40_retreat80"
EXPECTED_VARIANT = "molmo_dense_agentview"
EXPECTED_PROMPT = "rim_clearance"
EXPECTED_MOTION = "release20_retreat80mm"
EXPECTED_OPENING = "preshape40mm"
EXPECTED_OBSERVATION = "parked"
EXPECTED_MODEL = "allenai/MolmoPoint-8B"
EXPECTED_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
EXPECTED_MODEL_CONFIG_SHA = "f243da038d3394788357a644d42157aeb5c2b64a7620e14888e62b4ba1af487b"
TERMINAL = {"completed", "failed", "interrupted"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path) -> None:
    actual_root = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()
    if Path(actual_root).resolve() != repo_root:
        raise RuntimeError("REPO_ROOT is not the checkout root")
    actual = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip().lower()
    if actual != EXPECTED_EXECUTION_SHA:
        raise RuntimeError(f"release SHA mismatch: {actual} != {EXPECTED_EXECUTION_SHA}")
    if subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.strip():
        raise RuntimeError("release checkout is dirty")


def _load_release(repo_root: Path):
    package_root = repo_root / "vla_benchmarking"
    sys.path.insert(0, str(package_root))
    import run_molmo_sam3_canary as canary  # type: ignore
    from run_v9d_molmo_campaign import StopRules, metrics, write_json  # type: ignore
    from molmo_sam3.molmopoint import MolmoPointRuntime  # type: ignore
    return canary, MolmoPointRuntime, StopRules, metrics, write_json


def _records(arm_root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, Any]] = {}
    for suite in ("vanilla", "sealed_randomized"):
        status = _json(arm_root / suite / "arrow_pick_place_matrix_status.json")
        cells = status.get("cells")
        if not isinstance(cells, list) or len(cells) != 30:
            raise RuntimeError(f"{suite} prefix must contain the complete 30-cell inventory")
        terminal = [cell for cell in cells if isinstance(cell, Mapping) and cell.get("status") in TERMINAL]
        if len(terminal) != 6 or len(cells) - len(terminal) != 24:
            raise RuntimeError(f"{suite} prefix must contain six terminal and 24 planned cells")
        if any(any(isinstance(a, Mapping) and a.get("status") == "running" for a in cell.get("attempts", [])) for cell in terminal):
            raise RuntimeError(f"{suite} prefix contains a running attempt")
        records.extend(terminal)
        statuses[suite] = status
    return records, statuses


def _validate_prefix(prefix_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = _json(prefix_root / "campaign.json")
    if report.get("label") != EXPECTED_PREFIX_LABEL or report.get("status") != "completed":
        raise RuntimeError("failure-screen prefix is not the expected completed campaign")
    matches = [item for item in report.get("screen", []) if (item.get("arm") or {}).get("name") == EXPECTED_ARM]
    if len(matches) != 1:
        raise RuntimeError("prefix does not contain exactly one winning arm record")
    screen = matches[0]
    metrics = screen.get("metrics") or {}
    if screen.get("phase") != "prefix" or screen.get("returncode") != 0 or screen.get("fatal") or screen.get("stop_reason") not in (None, ""):
        raise RuntimeError("winning prefix arm is not cleanly completed")
    if (metrics.get("planned"), metrics.get("terminal_cells"), metrics.get("successes")) != (12, 12, 8):
        raise RuntimeError("winning prefix must be the recorded 8/12 result")
    arm = screen.get("arm") or {}
    expected_arm = {"variant": EXPECTED_VARIANT, "prompt": EXPECTED_PROMPT, "motion_profile": EXPECTED_MOTION, "opening_profile": EXPECTED_OPENING}
    for key, expected in expected_arm.items():
        if arm.get(key) != expected:
            raise RuntimeError(f"prefix arm {key}={arm.get(key)!r}, expected {expected!r}")
    arm_root = prefix_root / EXPECTED_ARM
    summary = _json(arm_root / "molmo_sam3_canary_summary.json")
    if summary.get("experiment_id") != f"{EXPECTED_PREFIX_LABEL}__{EXPECTED_ARM}":
        raise RuntimeError("prefix experiment identity mismatch")
    if (summary.get("execution_provenance") or {}).get("execution_sha") != EXPECTED_EXECUTION_SHA:
        raise RuntimeError("prefix execution SHA mismatch")
    if summary.get("region_backend") != "rgbd" or summary.get("sam3_used") is not False:
        raise RuntimeError("prefix is not the no-SAM RGB-D path")
    model = summary.get("model_provenance") or {}
    if (model.get("model_id"), model.get("model_revision"), model.get("config_sha256")) != (EXPECTED_MODEL, EXPECTED_MODEL_REVISION, EXPECTED_MODEL_CONFIG_SHA):
        raise RuntimeError("prefix Molmo provenance mismatch")
    records, statuses = _records(arm_root)
    identities = [(statuses[s].get("protocol") or {}).get("experimental_identity") for s in statuses]
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise RuntimeError("prefix scientific identity is missing")
        for key, expected in {"variant": EXPECTED_VARIANT, "region_backend": "rgbd", "molmopoint_prompt_id": EXPECTED_PROMPT, "motion_profile": EXPECTED_MOTION, "observation_profile": EXPECTED_OBSERVATION, "opening_profile": EXPECTED_OPENING, "sam3_used": False}.items():
            if identity.get(key) != expected:
                raise RuntimeError(f"prefix identity {key} mismatch")
    hashes = {identity.get("scientific_identity_hash") for identity in identities}
    if len(hashes) != 1 or None in hashes or summary.get("scientific_identity_hash") not in hashes:
        raise RuntimeError("prefix scientific identities disagree")
    return report, summary, records


def _runner_args(destination: Path) -> list[str]:
    return [
        "--region-backend", "rgbd", "--variant", EXPECTED_VARIANT,
        "--output-dir", str(destination), "--label", f"{EXPECTED_PREFIX_LABEL}__{EXPECTED_ARM}",
        "--phase", "full60", "--episodes-per-task", "10", "--task-ids", "4,6,9", "--seed-base", "1000",
        "--suite-modes", "vanilla,sealed_randomized", "--molmopoint-revision", EXPECTED_MODEL_REVISION,
        "--molmopoint-prompt-id", EXPECTED_PROMPT, "--motion-profile", EXPECTED_MOTION,
        "--observation-profile", EXPECTED_OBSERVATION, "--opening-profile", EXPECTED_OPENING,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    prefix_root = args.prefix_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not repo_root.is_dir() or not prefix_root.is_dir() or output_root.exists():
        raise RuntimeError("repo/prefix must exist and output must be new")
    _git(repo_root)
    report, summary, records = _validate_prefix(prefix_root)
    canary, MolmoPointRuntime, StopRules, metric_fn, write_json = _load_release(repo_root)
    source_arm = prefix_root / EXPECTED_ARM
    destination_arm = output_root / EXPECTED_ARM
    output_root.mkdir(parents=True)
    shutil.copytree(source_arm, destination_arm, symlinks=False)
    runtime = MolmoPointRuntime()
    rules = StopRules()
    for record in records:
        rules(record)
    manifest_path = output_root / "continuation_manifest.json"
    state: dict[str, Any] = {
        "schema": "molmo_failure_full60_continuation.v1", "operator_label": args.operator_label,
        "execution_sha": EXPECTED_EXECUTION_SHA, "prefix_campaign_label": report.get("label"),
        "prefix_arm": EXPECTED_ARM, "prefix_successes": 8, "prefix_planned": 12,
        "prefix_scientific_identity_hash": summary.get("scientific_identity_hash"),
        "prefix_root": str(prefix_root), "arms": {}, "status": "running", "started_unix": time.time(),
    }
    write_json(manifest_path, state)
    arm_state: dict[str, Any] = {"status": "running", "source_arm": str(source_arm), "destination_arm": str(destination_arm), "continuation_argv": _runner_args(destination_arm), "new_terminal_records": []}
    state["arms"][EXPECTED_ARM] = arm_state
    write_json(manifest_path, state)

    def on_cell(record: Mapping[str, Any]) -> None:
        rules(record)
        arm_state["new_terminal_records"].append({k: record.get(k) for k in ("suite_mode", "task_id", "seed", "episode_index", "status")})
        write_json(manifest_path, state)

    try:
        rc = canary.main(_runner_args(destination_arm), molmo_runtime=runtime, cell_completed_callback=on_cell)
        arm_state["returncode"] = int(rc)
        arm_state["metrics"] = metric_fn(destination_arm, 60)
        arm_state["stop_reason"] = rules.reason
        arm_state["fatal"] = rules.fatal
        arm_state["status"] = "completed" if rc == 0 and arm_state["metrics"]["terminal_cells"] == 60 else "failed"
        state["status"] = arm_state["status"]
        state["finished_unix"] = time.time()
        write_json(manifest_path, state)
        return 0 if state["status"] == "completed" else 2
    except Exception as exc:
        state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "finished_unix": time.time()})
        write_json(manifest_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
