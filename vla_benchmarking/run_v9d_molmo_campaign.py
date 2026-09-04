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
    motion_profile: str = "baseline"
    opening_profile: str = "full_open"


ARMS = (
    Arm("dense_agentview", "molmo_dense_agentview"),
    Arm("geometry_agentview", "rgbd_geometry_agentview"),
    Arm("local_agentview", "molmo_local_agentview"),
    Arm("dense_agentview_contact", "molmo_dense_agentview", "rim_contact"),
    Arm("dense_agentview_clearance", "molmo_dense_agentview", "rim_clearance"),
    Arm("dense_wrist", "molmo_dense_wrist"),
)
FAILURE_ARMS = (
    Arm("failure_retreat80", "molmo_dense_agentview", "rim_clearance", "release20_retreat80mm"),
    Arm("failure_release40", "molmo_dense_agentview", "rim_clearance", "release_plus40mm"),
    Arm("failure_opening40_retreat80", "molmo_dense_agentview", "rim_clearance", "release20_retreat80mm", "preshape40mm"),
)
MOTION_PROBE_ARMS = (
    Arm("placement_control", "molmo_dense_agentview"),
    Arm("placement_burst5mm", "molmo_dense_agentview", motion_profile="placement_micro5mm"),
)
OPENING_PROBE_ARMS = (
    Arm("opening40mm", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "preshape40mm"),
    Arm("opening_control", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "full_open"),
)
SETTLED_OPENING_PROBE_ARMS = (
    Arm("settled_opening40mm", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "preshape40mm_settled"),
    Arm("settled_opening_control", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "full_open_settled"),
)
PARKED_OPENING_PROBE_ARMS = (
    Arm("parked_opening40mm", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "preshape40mm"),
    Arm("parked_opening_control", "molmo_dense_agentview", "rim_clearance", "release_plus20mm", "full_open"),
)
# Fixed, failure-stratified screen identities.  These are task/seed keys only;
# the matrix planner still materializes the complete 30-cell inventory for
# each suite and derives the canonical episode/init-state index.
FAILURE_STRATIFIED_SCREEN_IDENTITIES = {
    "vanilla": (
        (4, 1000), (6, 1005), (9, 1004),
        (9, 1001), (4, 1002), (6, 1000),
    ),
    "sealed_randomized": (
        (4, 1000), (9, 1003), (9, 1007),
        (4, 1006), (4, 1002), (9, 1000),
    ),
}
TERMINAL = {"completed", "failed", "interrupted"}
CONTRACT_ERRORS = (
    "evaluator called before retreat", "evaluator use before retreat",
    "coordinate is outside the original image", "source capture camera does not match",
    "action budget exceeded", "rgb and metric depth are not aligned",
    "evaluator failure",
)
PRESHAPE_FAILURE_CATEGORIES = ("below_band", "timeout", "pose_drift", "missing_measurement", "motion_failed", "unsupported_budget", "opening_guard", "opening_settling_failed")


def failure_stratified_screen_identities(suite_mode: str) -> tuple[tuple[int, int], ...]:
    """Return the fixed 6-cell screen sample for one canonical suite."""
    try:
        selected = FAILURE_STRATIFIED_SCREEN_IDENTITIES[str(suite_mode)]
    except KeyError as exc:
        raise ValueError(
            "suite_mode must be one of vanilla, sealed_randomized"
        ) from exc
    return tuple((int(task_id), int(seed)) for task_id, seed in selected)


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
        # Preshape failures carry a stable helper category; avoid keying stop
        # accounting on measured numeric details that differ across cells.
        audit_text = json.dumps(record.get("audit", {}), default=str).lower() + " " + error.lower()
        for category in PRESHAPE_FAILURE_CATEGORIES:
            if f"preshape_failed:{category}" in audit_text or f"preshape {category}" in audit_text:
                operational.append({"stage": "preshape", "error_type": "PreshapeError", "error": category})
                break
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


def _valid_action_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
        if count < 0 or float(value) != float(count):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return count


def _reported_action_count(record: Mapping[str, Any]) -> int | None:
    """Select one per-cell count, preferring final audit evidence."""
    audit = record.get("audit")
    audit = audit if isinstance(audit, Mapping) else None
    episode = audit.get("canary_manifest") if audit else None
    episode = episode if isinstance(episode, Mapping) else None
    final = episode.get("final_result") if episode else None
    final = final if isinstance(final, Mapping) else None
    final_audit = final.get("audit") if final else None
    final_audit = final_audit if isinstance(final_audit, Mapping) else None
    partial = record.get("partial_audit")
    partial = partial if isinstance(partial, Mapping) else None

    # The canary wrapper's top-level total is the live shared-budget count,
    # including when a retry's nested final result is stale or absent.
    for source in (audit, record, partial):
        if source is not None:
            count = _valid_action_count(source.get("total_actions"))
            if count is not None:
                return count
    # Nested final evidence is a fallback for older records without a wrapper
    # total, including exact zero values.
    for source in (final_audit, final):
        if source is not None:
            for key in ("total_actions", "experimental_total_actions"):
                count = _valid_action_count(source.get(key))
                if count is not None:
                    return count
    # Finally use an explicitly valid shared experimental budget snapshot.
    for source in (final_audit, final, audit, partial, record):
        if source is None:
            continue
        budget = source.get("experimental_action_budget")
        if isinstance(budget, Mapping):
            count = _valid_action_count(budget.get("used"))
            limit = _valid_action_count(budget.get("limit"))
            if count is not None and limit is not None and count <= limit:
                return count
    return None


def metrics(root: Path, planned: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for suite in ("vanilla", "sealed_randomized"):
        status = root / suite / "arrow_pick_place_matrix_status.json"
        if status.is_file():
            records.extend(json.loads(status.read_text(encoding="utf-8"))["cells"])
    terminal = [r for r in records if r.get("status") in TERMINAL]
    successes = sum(r.get("status") == "completed" and r.get("evaluator_result") is True for r in terminal)
    retained, retry_successes, actions, perception_s = 0, 0, 0, 0.0
    unknown_action_count_cells = 0
    retention_sources: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    for record in terminal:
        audit = record.get("audit") or {}
        episode = audit.get("canary_manifest") or {}
        attempts = episode.get("attempts") or []
        final = episode.get("final_result") or {}
        # A placement timeout can occur after a completed lift retention gate.
        # Count that cell's retained lift from explicit lift evidence even when
        # the final attempt result correctly reports grasp_retained=False for
        # the failed placement. Missing, rejected, or incomplete evidence
        # remains fail-closed. This is proprioception-only heuristic evidence,
        # not physical object-retention proof.
        retained_for_cell = False
        source_for_cell = "no_explicit_retention_evidence"
        for attempt in attempts:
            for result in attempt.get("results", []):
                if result.get("grasp_retained") is True:
                    retained_for_cell = True
                    source_for_cell = "result_grasp_retained"
                    break
                for phase in result.get("attempt_phases", []) or []:
                    if not isinstance(phase, Mapping) or phase.get("phase") != "lift":
                        continue
                    if phase.get("status") not in {"reached", "dwell", "stop"}:
                        continue
                    gate = phase.get("retention_gate")
                    if isinstance(gate, Mapping) and gate.get("retained") is True and gate.get("status") == "passed":
                        retained_for_cell = True
                        source_for_cell = "completed_lift_retention_gate"
                        break
                if retained_for_cell:
                    break
            if retained_for_cell:
                break
        retained += int(retained_for_cell)
        retention_sources[source_for_cell] += 1
        retry_successes += int(len(attempts) > 1 and final.get("evaluator_success") is True)
        for attempt in attempts:
            perception_s += float(attempt.get("candidate_diagnostics", {}).get("perception_latency_s", 0.0))
        # Matrix phases record the last attempt; the experimental audit stores
        # a shared count when provided. Never infer total action counts from time.
        count = _reported_action_count(record)
        if count is None:
            unknown_action_count_cells += 1
        else:
            actions += count
        if record.get("evaluator_result") is not True:
            category = record.get("failure_class") or final.get("status") or (attempts[-1].get("status") if attempts else None) or record.get("status")
            failure_counts[str(category)] += 1
    return {
        "planned": planned, "terminal_cells": len(terminal), "successes": successes,
        "successes_per_planned": successes / planned, "retained_lifts": retained,
        "retention_metric_source": {
            "description": "explicit result flag or completed lift retention gate; proprioception-only heuristic, not physical object-retention proof",
            "cells_by_source": dict(retention_sources),
        },
        "successful_retries": retry_successes, "reported_actions": actions,
        "reported_action_count_cells": len(terminal) - unknown_action_count_cells,
        "unknown_action_count_cells": unknown_action_count_cells,
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
    parser.add_argument("--failure-stratified-screen", action="store_true",
                        help="execute the fixed six-cell-per-suite failure-stratified screen")
    parser.add_argument("--arms", help="comma-separated subset of existing screen arms; fresh run identity required")
    parser.add_argument("--observation-profile", choices=("baseline", "hover20mm", "parked"), default=None)
    parser.add_argument("--motion-profile", choices=canary.MOTION_PROFILE_NAMES, default=None)
    parser.add_argument("--motion-probe", action="store_true", help="paired 12-cell placement-control/bounded-correction diagnostic; never extends to 60")
    parser.add_argument("--opening-probe", action="store_true", help="paired 12-cell full-open versus measured 40mm preshape diagnostic")
    parser.add_argument("--settled-opening-probe", action="store_true", help="paired settled-hover full-open versus measured 40mm opening diagnostic")
    parser.add_argument("--parked-opening-probe", action="store_true", help="paired no-hover full-open versus measured 40mm opening diagnostic")
    parser.add_argument("--repair-gate", action="store_true", help="require the first T4/T6 cells to clear hover and contact before continuing the campaign")
    args = parser.parse_args(argv)
    if args.motion_probe and args.repair_gate:
        parser.error("--motion-probe cannot enable the obsolete global repair gate")
    if (args.opening_probe or args.settled_opening_probe or args.parked_opening_probe) and (args.motion_probe or args.repair_gate or args.arms is not None or args.screen_only):
        parser.error("--opening-probe fixes its paired arms and screen-only execution")
    if sum(bool(value) for value in (args.opening_probe, args.settled_opening_probe, args.parked_opening_probe)) > 1:
        parser.error("opening probe modes are mutually exclusive")
    if (args.opening_probe or args.settled_opening_probe) and (args.observation_profile not in (None, "hover20mm") or args.motion_profile not in (None, "release_plus20mm")):
        parser.error("--opening-probe requires --observation-profile hover20mm and --motion-profile release_plus20mm")
    if args.parked_opening_probe and (args.observation_profile not in (None, "parked") or args.motion_profile not in (None, "release_plus20mm")):
        parser.error("--parked-opening-probe requires --observation-profile parked and --motion-profile release_plus20mm")
    if args.failure_stratified_screen:
        if args.motion_profile not in (None, "baseline"):
            parser.error("--failure-stratified-screen uses fixed per-arm motion profiles")
        if args.observation_profile not in (None, "parked"):
            parser.error("--failure-stratified-screen requires the parked observation procedure")
    observation_profile = (
        "parked" if (args.parked_opening_probe or args.failure_stratified_screen)
        else "hover20mm" if (args.opening_probe or args.settled_opening_probe)
        else (args.observation_profile or "baseline")
    )
    motion_profile = "release_plus20mm" if (args.opening_probe or args.settled_opening_probe or args.parked_opening_probe) else (args.motion_profile or "baseline")
    if args.motion_probe and (args.arms is not None or args.observation_profile not in (None, "baseline") or args.motion_profile not in (None, "baseline")):
        parser.error("--motion-probe fixes its paired arms and baseline observation/profile selection")
    if args.arms is not None and args.repair_gate:
        parser.error("--arms cannot enable the obsolete global repair gate")
    arms = PARKED_OPENING_PROBE_ARMS if args.parked_opening_probe else SETTLED_OPENING_PROBE_ARMS if args.settled_opening_probe else OPENING_PROBE_ARMS if args.opening_probe else MOTION_PROBE_ARMS if args.motion_probe else FAILURE_ARMS if args.failure_stratified_screen else ARMS
    if args.arms is not None:
        names = args.arms.split(",")
        available = {arm.name: arm for arm in (*ARMS, *FAILURE_ARMS)}
        if len(names) != len(set(names)) or any(name not in available for name in names):
            parser.error("--arms must contain unique existing screen arm names")
        arms = tuple(available[name] for name in names)
    if not args.motion_probe and not args.opening_probe and not args.settled_opening_probe and not args.parked_opening_probe and not args.failure_stratified_screen:
        arms = tuple(Arm(arm.name, arm.variant, arm.prompt, motion_profile, arm.opening_profile) for arm in arms)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "campaign.json"
    if report_path.exists():
        raise RuntimeError("campaign output already exists; use a new run identity")
    report: dict[str, Any] = {
        "schema": "v9d_molmo_rgbd_campaign.v1", "label": args.label,
        "baseline_commit": canary.BASELINE_COMMIT, "sam3_used": False,
        "region_backend": "v9d_rgbd_region", "arms": [asdict(a) for a in arms],
        "mode": "parked_opening_probe" if args.parked_opening_probe else "settled_opening_probe" if args.settled_opening_probe else "opening_probe" if args.opening_probe else "placement_motion_probe" if args.motion_probe else "candidate_screen",
        "motion_diagnostics": bool(args.motion_probe),
        "observation_profile": observation_profile,
        "motion_profile": "paired_probe" if args.motion_probe else motion_profile,
        "opening_probe": bool(args.opening_probe or args.settled_opening_probe or args.parked_opening_probe),
        "settled_opening_probe": bool(args.settled_opening_probe),
        "parked_opening_probe": bool(args.parked_opening_probe),
        "opening_profile": "parked_paired_probe" if args.parked_opening_probe else "settled_paired_probe" if args.settled_opening_probe else "paired_probe" if args.opening_probe else "full_open",
        "opening_profile_params": (
            {
                profile: canary.resolve_opening_profile(profile, region_backend="rgbd", camera_name=canary.AGENTVIEW)
                for profile in (("full_open", "preshape40mm") if args.parked_opening_probe else ("full_open_settled", "preshape40mm_settled") if args.settled_opening_probe else ("full_open", "preshape40mm"))
            }
            if args.opening_probe or args.settled_opening_probe or args.parked_opening_probe
            else canary.resolve_opening_profile("full_open", region_backend="rgbd", camera_name=canary.AGENTVIEW)
        ),
        "screen_planned_per_arm": 12, "screen": [], "finalists": [],
        "screen_selection": (
            {suite: [list(pair) for pair in failure_stratified_screen_identities(suite)]
             for suite in ("vanilla", "sealed_randomized")}
            if args.failure_stratified_screen else None
        ),
        "status": "running", "started_unix": time.time(),
        "comparison": "exploratory grasp pipeline comparison; geometry control shares motion and hover",
        "historical_baselines": {"1914768": {"successes": 15, "planned": 60}, "1917528": {"successes": 14, "planned": 60}},
        "historical_comparison_note": "Historical commits differ; compare matching task/seed/suite cells. No baseline rerun or default promotion.",
        "repair_gate": {"status": "pending" if args.repair_gate else "disabled"},
    }
    if args.motion_probe:
        report["comparison"] = "paired exploratory placement-response probe; same dense-agentview perception, prompt and 12 canary cells; only bounded placement correction differs"
        report["motion_probe_contract"] = {
            "planned_cells": 24, "phases": ["preplace", "descend_place"],
            "max_added_target_bias_m": 0.005, "burst_steps": 8,
            "max_rounds_per_phase": 1, "max_correction_actions_per_cell": 16,
            "phase_step_limit": 160, "cell_action_limit": 1200,
            "convergence_target": "original_waypoint", "finalist_extension": False,
            "interpretation": "motion response and available robot load diagnostics; no response without load evidence is inconclusive; unchanged evaluator determines task success",
        }
    if args.opening_probe:
        report["comparison"] = "paired exploratory opening-response probe; same dense-agentview perception, rim-clearance prompt, hover and release profiles; only measured preshape opening differs"
        report["opening_probe_contract"] = {
            "planned_cells": 24, "arm_order": [arm.name for arm in OPENING_PROBE_ARMS],
            "control_profile": "full_open", "treatment_profile": "preshape40mm",
            "observation_profile": "hover20mm", "motion_profile": "release_plus20mm",
            "screen_only": True, "finalist_extension": False,
            "interpretation": "whole opening-policy comparison; shaping occurs before each fresh RGB-D proposal capture and candidate geometry uses fresh measured calibration",
        }
    if args.settled_opening_probe:
        report["comparison"] = "paired exploratory settled-opening probe; same dense-agentview perception, rim-clearance prompt, hover and release profiles; only bounded hover settling and measured opening differ"
        report["settled_opening_probe_contract"] = {
            "planned_cells": 24, "arm_order": [arm.name for arm in SETTLED_OPENING_PROBE_ARMS],
            "control_profile": "full_open_settled", "treatment_profile": "preshape40mm_settled",
            "observation_profile": "hover20mm", "motion_profile": "release_plus20mm",
            "screen_only": True, "finalist_extension": False,
            "retry_protocol": "new_agentview_arrow_hover_settle_fresh_capture",
            "interpretation": "settling is robot-only revalidation at the original hover before each fresh RGB-D proposal; optional 40mm shaping is measured and bounded",
        }
    if args.parked_opening_probe:
        report["comparison"] = "paired exploratory parked-opening probe; same dense-agentview perception, rim-clearance prompt, parked observation policy and release profile; the paired arms differ only in measured opening"
        report["parked_opening_probe_contract"] = {
            "planned_cells": 24, "arm_order": [arm.name for arm in PARKED_OPENING_PROBE_ARMS],
            "control_profile": "full_open", "treatment_profile": "preshape40mm",
            "observation_profile": "parked", "motion_profile": "release_plus20mm",
            "screen_only": True, "finalist_extension": False,
            "retry_protocol": "recovered_open_retreat_optional_preshape_fresh_agentview_current_arrow_calibration",
            "interpretation": "direct guarded canary remains at the post-open parked pose; no observation hover or settling controller is executed",
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
        canary_args = [
            "--region-backend", "rgbd", "--variant", arm.variant,
            "--output-dir", str(root / arm.name), "--label", f"{args.label}__{arm.name}",
            "--phase", phase, "--molmopoint-revision", canary.MOLMOPOINT_MODEL_REVISION,
            "--molmopoint-prompt-id", arm.prompt,
        ]
        if args.motion_probe:
            canary_args.extend(["--motion-profile", arm.motion_profile, "--motion-diagnostics"])
        elif arm.motion_profile != "baseline":
            canary_args.extend(["--motion-profile", arm.motion_profile])
        if args.opening_probe or args.settled_opening_probe or args.parked_opening_probe or arm.opening_profile != "full_open":
            canary_args.extend(["--opening-profile", arm.opening_profile])
        if observation_profile != "baseline":
            canary_args.extend(["--observation-profile", observation_profile])
        canary_kwargs = {
            "molmo_runtime": runtime,
            "cell_completed_callback": on_cell,
        }
        if args.failure_stratified_screen and phase == "prefix":
            canary_kwargs["execution_cell_identities_by_suite"] = {
                suite: failure_stratified_screen_identities(suite)
                for suite in ("vanilla", "sealed_randomized")
            }
        rc = canary.main(canary_args, **canary_kwargs)
        result = {"arm": asdict(arm), "phase": phase, "returncode": rc,
                  "stop_reason": rules.reason, "fatal": rules.fatal,
                  "elapsed_s": time.monotonic() - started,
                  "metrics": metrics(root / arm.name, 12 if phase == "prefix" else 60)}
        print("CAMPAIGN " + json.dumps(result, sort_keys=True), flush=True)
        return result

    for arm in arms:
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
    if args.motion_probe or args.opening_probe or args.settled_opening_probe or args.parked_opening_probe:
        complete = all(r["returncode"] == 0 and r["metrics"]["terminal_cells"] == 12 for r in report["screen"])
        if args.parked_opening_probe:
            report["status"] = "parked_opening_probe_completed" if complete else "parked_opening_probe_stopped"
        elif args.settled_opening_probe:
            report["status"] = "settled_opening_probe_completed" if complete else "settled_opening_probe_stopped"
        elif args.opening_probe:
            report["status"] = "opening_probe_completed" if complete else "opening_probe_stopped"
        else:
            report["status"] = "motion_probe_completed" if complete else "motion_probe_stopped"
        report["finished_unix"] = time.time()
        write_json(report_path, report)
        return 0 if complete else 2
    eligible = sorted((r for r in report["screen"] if r["returncode"] == 0 and r["metrics"]["terminal_cells"] == 12 and r["metrics"]["successes"] > 0), key=rank_key)
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
    report["status"] = ("no_successful_arm" if not eligible else "incomplete_finalists" if incomplete_finalists else "completed")
    report["finished_unix"] = time.time()
    write_json(report_path, report)
    return 0 if eligible and not incomplete_finalists else 2


if __name__ == "__main__":
    raise SystemExit(main())
