#!/usr/bin/env python3
"""One GPU, persistent MolmoPoint, SAM-free v9d grasp ablations."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from . import run_molmo_sam3_canary as canary
    from .molmo_sam3.molmopoint import MolmoPointRuntime
except ImportError:
    import run_molmo_sam3_canary as canary
    from molmo_sam3.molmopoint import MolmoPointRuntime


@dataclass(frozen=True)
class Arm:
    name: str
    variant: str
    prompt: str = "rim_downward_approach"


ARMS = (
    Arm("dense_agentview", "molmo_dense_agentview"),
    Arm("geometry_agentview", "rgbd_geometry_agentview"),
    Arm("local_agentview", "molmo_local_agentview"),
    Arm("dense_agentview_contact", "molmo_dense_agentview", "rim_contact"),
    Arm("dense_agentview_clearance", "molmo_dense_agentview", "rim_clearance"),
    Arm("dense_wrist", "molmo_dense_wrist"),
)
TERMINAL = {"completed", "failed", "interrupted"}
CONTRACT_ERRORS = (
    "evaluator called before retreat", "evaluator use before retreat",
    "coordinate is outside the original image", "source capture camera does not match",
    "action budget exceeded", "rgb and metric depth are not aligned",
    "evaluator failure",
)


class ArmStop(RuntimeError):
    pass


class RepairGate:
    """Use the first four planned cells to verify the repaired motion path."""

    expected = {(task, seed) for task in (4, 6) for seed in (1000, 1001)}

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], dict[str, Any]] = {}
        self.status = "pending"
        self.reason: str | None = None

    def observe(self, record: Mapping[str, Any]) -> None:
        if record.get("suite_mode") != "vanilla":
            return
        key = (int(record["task_id"]), int(record["seed"]))
        if key not in self.expected:
            return
        audit = record.get("audit") or {}
        manifest = audit.get("canary_manifest") or {}
        results = [result for attempt in manifest.get("attempts", []) for result in attempt.get("results", [])]
        reached = set()
        for result in results:
            reached.update(result.get("motion_phases_reached") or [])
            for phase in (result.get("audit") or {}).get("phases", []):
                if phase.get("status") in {"reached", "dwell", "stop"}:
                    reached.add(phase.get("phase"))
        self.cells[key] = {
            "task_id": key[0], "seed": key[1],
            "hover_completed": (audit.get("observation_hover") or {}).get("status") == "completed",
            "completed_contact_phase": bool(reached & {"close", "lift"}),
            "motion_phases_reached": sorted(p for p in reached if p),
            "status": record.get("status"),
        }
        if set(self.cells) != self.expected:
            return
        hover_ok = all(cell["hover_completed"] for cell in self.cells.values())
        contact_ok = all(any(cell["task_id"] == task and cell["completed_contact_phase"] for cell in self.cells.values()) for task in (4, 6))
        self.status = "passed" if hover_ok and contact_ok else "failed"
        if self.status == "failed":
            self.reason = "repair replay requires four completed hovers and a completed close/lift on both T4 and T6"
            raise ArmStop(self.reason)

    def canonical(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "cells": [self.cells[key] for key in sorted(self.cells)],
                "purpose": "motion integration check; success is not required and evaluator results do not select grasps"}


class StopRules:
    def __init__(self) -> None:
        self.failures: dict[tuple[str, str, str], set[tuple[str, int, int]]] = {}
        self.reason: str | None = None
        self.fatal = False

    def __call__(self, record: Mapping[str, Any]) -> None:
        # Full audit catches contract errors wrapped by the bounded retry runner.
        text = json.dumps(record.get("audit", {}), default=str).lower()
        error = str(record.get("error") or "")
        if any(marker in text + error.lower() for marker in CONTRACT_ERRORS):
            self.fatal = True
            self.reason = "coordinate/evaluator/action contract violation"
            raise ArmStop(self.reason)
        operational = [record] if record.get("status") == "failed" else []
        episode = (record.get("audit") or {}).get("canary_manifest") or {}
        for attempt in episode.get("attempts", []):
            for result in attempt.get("results", []):
                if result.get("error_type") in {"ImportError", "ModuleNotFoundError", "NameError", "TypeError", "AttributeError", "FileNotFoundError"} or "calibration failed closed" in str(result.get("error", "")).lower():
                    operational.append(result)
        # Empty closes, timeouts, no candidates and retention failures remain
        # grasp outcomes. Wrapped programming/calibration failures still pause.
        for failure in operational:
            key = (str(failure.get("stage", "run_episode")), str(failure.get("error_type")), str(failure.get("error")))
            cells = self.failures.setdefault(key, set())
            cells.add((str(record.get("suite_mode")), int(record["task_id"]), int(record["seed"])))
            if len(cells) >= 2:
                self.reason = f"same operational failure on two cells: {key[0]}: {key[1]}: {key[2]}"
                raise ArmStop(self.reason)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics(root: Path, planned: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for suite in ("vanilla", "sealed_randomized"):
        status = root / suite / "arrow_pick_place_matrix_status.json"
        if status.is_file():
            records.extend(json.loads(status.read_text(encoding="utf-8"))["cells"])
    terminal = [r for r in records if r.get("status") in TERMINAL]
    successes = sum(r.get("status") == "completed" and r.get("evaluator_result") is True for r in terminal)
    retained, retry_successes, actions, perception_s = 0, 0, 0, 0.0
    failure_counts: Counter[str] = Counter()
    for record in terminal:
        audit = record.get("audit") or {}
        episode = audit.get("canary_manifest") or {}
        attempts = episode.get("attempts") or []
        final = episode.get("final_result") or {}
        retained += int(any(result.get("grasp_retained") is True for attempt in attempts for result in attempt.get("results", [])))
        retry_successes += int(len(attempts) > 1 and final.get("evaluator_success") is True)
        for attempt in attempts:
            perception_s += float(attempt.get("candidate_diagnostics", {}).get("perception_latency_s", 0.0))
        # Matrix phases record the last attempt; the experimental audit stores
        # a shared count when provided. Never infer total action counts from time.
        count = audit.get("total_actions", final.get("total_actions"))
        if count is None:
            count = (final.get("audit") or {}).get("experimental_total_actions")
        if count is not None:
            actions += int(count)
        if record.get("evaluator_result") is not True:
            category = record.get("failure_class") or final.get("status") or (attempts[-1].get("status") if attempts else None) or record.get("status")
            failure_counts[str(category)] += 1
    return {
        "planned": planned, "terminal_cells": len(terminal), "successes": successes,
        "successes_per_planned": successes / planned, "retained_lifts": retained,
        "successful_retries": retry_successes, "reported_actions": actions,
        "perception_latency_s": perception_s, "failure_categories": dict(failure_counts),
        "per_suite": {suite: {"successes": sum(r.get("status") == "completed" and r.get("evaluator_result") is True for r in terminal if r.get("suite_mode") == suite), "planned": planned // 2} for suite in ("vanilla", "sealed_randomized")},
    }


def rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    m = item["metrics"]
    return (-m["successes_per_planned"], -m["retained_lifts"], m["reported_actions"], m["perception_latency_s"], item["arm"]["name"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="v9d_molmo_rgbd")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--repair-gate", action="store_true", help="require the first T4/T6 cells to clear hover and contact before continuing the campaign")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "campaign.json"
    if report_path.exists():
        raise RuntimeError("campaign output already exists; use a new run identity")
    report: dict[str, Any] = {
        "schema": "v9d_molmo_rgbd_campaign.v1", "label": args.label,
        "baseline_commit": canary.BASELINE_COMMIT, "sam3_used": False,
        "region_backend": "v9d_rgbd_region", "arms": [asdict(a) for a in ARMS],
        "screen_planned_per_arm": 12, "screen": [], "finalists": [],
        "status": "running", "started_unix": time.time(),
        "comparison": "exploratory grasp pipeline comparison; geometry control shares motion and hover",
        "historical_baselines": {"1914768": {"successes": 15, "planned": 60}, "1917528": {"successes": 14, "planned": 60}},
        "historical_comparison_note": "Historical commits differ; compare matching task/seed/suite cells. No baseline rerun or default promotion.",
        "repair_gate": {"status": "pending" if args.repair_gate else "disabled"},
    }
    write_json(report_path, report)
    runtime = MolmoPointRuntime()
    repair_gate = RepairGate() if args.repair_gate else None

    def run_arm(arm: Arm, phase: str) -> dict[str, Any]:
        rules = StopRules()
        def on_cell(record: Mapping[str, Any]) -> None:
            rules(record)
            if repair_gate is not None and arm.name == ARMS[0].name and phase == "prefix":
                try:
                    repair_gate.observe(record)
                finally:
                    report["repair_gate"] = repair_gate.canonical()
                    write_json(report_path, report)
        started = time.monotonic()
        print(f"CAMPAIGN arm={arm.name} phase={phase} prompt={arm.prompt} starting", flush=True)
        rc = canary.main([
            "--region-backend", "rgbd", "--variant", arm.variant,
            "--output-dir", str(root / arm.name), "--label", f"{args.label}__{arm.name}",
            "--phase", phase, "--molmopoint-revision", canary.MOLMOPOINT_MODEL_REVISION,
            "--molmopoint-prompt-id", arm.prompt,
        ], molmo_runtime=runtime, cell_completed_callback=on_cell)
        result = {"arm": asdict(arm), "phase": phase, "returncode": rc,
                  "stop_reason": rules.reason, "fatal": rules.fatal,
                  "elapsed_s": time.monotonic() - started,
                  "metrics": metrics(root / arm.name, 12 if phase == "prefix" else 60)}
        print("CAMPAIGN " + json.dumps(result, sort_keys=True), flush=True)
        return result

    for arm in ARMS:
        result = run_arm(arm, "prefix")
        report["screen"].append(result)
        write_json(report_path, report)
        if result["fatal"]:
            report["status"] = "stopped_contract_violation"
            write_json(report_path, report)
            return 2
        if repair_gate is not None and arm.name == ARMS[0].name and repair_gate.status != "passed":
            report["repair_gate"] = repair_gate.canonical()
            report["status"] = "stopped_repair_gate"
            report["finished_unix"] = time.time()
            write_json(report_path, report)
            return 2
    eligible = sorted((r for r in report["screen"] if r["returncode"] == 0 and r["metrics"]["terminal_cells"] == 12), key=rank_key)
    if not args.screen_only:
        for selected in eligible[:2]:
            result = run_arm(Arm(**selected["arm"]), "full60")
            report["finalists"].append(result)
            write_json(report_path, report)
            if result["fatal"]:
                report["status"] = "stopped_contract_violation"
                write_json(report_path, report)
                return 2
    incomplete_finalists = any(r["returncode"] != 0 or r["metrics"]["terminal_cells"] != 60 for r in report["finalists"])
    report["status"] = ("no_executable_arm" if not eligible else "incomplete_finalists" if incomplete_finalists else "completed")
    report["finished_unix"] = time.time()
    write_json(report_path, report)
    return 0 if eligible and not incomplete_finalists else 2


if __name__ == "__main__":
    raise SystemExit(main())
