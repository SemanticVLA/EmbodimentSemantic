import json

import pytest

import run_v9d_molmo_campaign as campaign


def _repair_record(task, seed, *, hover=True, reached=()):
    return {"status": "completed", "suite_mode": "vanilla", "task_id": task, "seed": seed,
            "evaluator_result": False, "audit": {
                "observation_hover": {"status": "completed" if hover else "failed"},
                "canary_manifest": {"attempts": [{"results": [{"motion_phases_reached": list(reached)}]}]},
            }}


def test_repair_gate_needs_completed_hover_and_contact_for_each_task():
    gate = campaign.RepairGate()
    gate.observe(_repair_record(4, 1000, reached=("close",)))
    gate.observe(_repair_record(4, 1001))
    gate.observe(_repair_record(6, 1000, reached=("lift",)))
    gate.observe(_repair_record(6, 1001))
    assert gate.status == "passed"  # No evaluator success was supplied.
    assert len(gate.canonical()["cells"]) == 4


@pytest.mark.parametrize("bad_hover", [False, True])
def test_repair_gate_fails_closed_on_missing_contact_or_hover(bad_hover):
    gate = campaign.RepairGate()
    gate.observe(_repair_record(4, 1000, reached=("close",)))
    gate.observe(_repair_record(4, 1001))
    gate.observe(_repair_record(6, 1000, hover=not bad_hover,
                                reached=("close",) if bad_hover else ("pregrasp",)))
    with pytest.raises(campaign.ArmStop, match="repair replay"):
        gate.observe(_repair_record(6, 1001))
    assert gate.status == "failed"


def test_repair_gate_stops_before_any_other_arm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "MolmoPointRuntime", object)

    def fake_main(argv, *, molmo_runtime, cell_completed_callback):
        calls.append(argv)
        try:
            for task in (4, 6):
                for seed in (1000, 1001):
                    cell_completed_callback(_repair_record(task, seed))
        except campaign.ArmStop:
            return 2
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    assert campaign.main(["--output-dir", str(tmp_path), "--repair-gate"]) == 2
    assert len(calls) == 1
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert report["status"] == "stopped_repair_gate"
    assert report["repair_gate"]["status"] == "failed"
    assert report["finalists"] == []


def test_repair_gate_pass_continues_same_prefix_without_replay(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "MolmoPointRuntime", object)

    def fake_main(argv, *, molmo_runtime, cell_completed_callback):
        args = dict(zip(argv[::2], argv[1::2]))
        calls.append(args)
        for task in (4, 6):
            for seed in (1000, 1001):
                cell_completed_callback(_repair_record(task, seed, reached=("close",)))
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    # No fake matrix files: screen is incomplete, but all six arms execute.
    assert campaign.main(["--output-dir", str(tmp_path), "--repair-gate", "--screen-only"]) == 2
    assert len(calls) == 6
    assert len({call["--output-dir"] for call in calls}) == 6
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert report["repair_gate"]["status"] == "passed"


def test_missing_and_failed_cells_stay_in_planned_denominator(tmp_path):
    campaign.write_json(tmp_path / "vanilla" / "arrow_pick_place_matrix_status.json", {"cells": [
        {"status": "completed", "suite_mode": "vanilla", "evaluator_result": True},
        {"status": "failed", "suite_mode": "vanilla", "evaluator_result": None},
        {"status": "planned", "suite_mode": "vanilla"},
    ]})
    result = campaign.metrics(tmp_path, 12)
    assert result["successes"] == 1
    assert result["terminal_cells"] == 2
    assert result["successes_per_planned"] == 1 / 12


def test_metrics_reconstructs_retained_lift_after_placement_failure(tmp_path):
    def cell(seed, result):
        return {
            "status": "failed", "suite_mode": "vanilla", "task_id": 4,
            "seed": seed, "evaluator_result": False,
            "audit": {"canary_manifest": {"attempts": [{"results": [result]}]}},
        }

    passed_after_placement_failure = {
        "grasp_retained": False,
        "attempt_phases": [{
            "phase": "lift", "status": "reached",
            "retention_gate": {"enabled": True, "retained": True, "status": "passed"},
        }],
    }
    rejected_gate = {
        "grasp_retained": False,
        "attempt_phases": [{
            "phase": "lift", "status": "reached",
            "retention_gate": {"enabled": True, "retained": False, "status": "rejected"},
        }],
    }
    missing_gate = {"grasp_retained": False, "attempt_phases": [{"phase": "lift", "status": "reached"}]}
    campaign.write_json(tmp_path / "vanilla" / "arrow_pick_place_matrix_status.json", {"cells": [
        cell(1000, passed_after_placement_failure), cell(1001, rejected_gate), cell(1002, missing_gate),
    ]})
    result = campaign.metrics(tmp_path, 12)
    assert result["retained_lifts"] == 1
    assert result["retention_metric_source"]["cells_by_source"] == {
        "completed_lift_retention_gate": 1,
        "no_explicit_retention_evidence": 2,
    }


def test_operational_stop_requires_two_distinct_cells():
    rules = campaign.StopRules()
    record = dict(status="failed", stage="build_env", error_type="ImportError", error="missing dependency", suite_mode="vanilla", task_id=4, seed=1000)
    rules(record)
    rules(record)
    with pytest.raises(campaign.ArmStop, match="two cells"):
        rules({**record, "seed": 1001})
    assert not rules.fatal


def test_wrapped_contract_violation_stops_immediately():
    rules = campaign.StopRules()
    with pytest.raises(campaign.ArmStop):
        rules({"status": "completed", "audit": {"error": "evaluator called before retreat completed"}})
    assert rules.fatal


def test_wrapped_programming_failure_pauses_but_grasp_timeout_does_not():
    def record(seed, kind, message):
        return {"status": "completed", "suite_mode": "vanilla", "task_id": 4, "seed": seed,
                "audit": {"canary_manifest": {"attempts": [{"results": [{"error_type": kind, "error": message}]}]}}}
    rules = campaign.StopRules()
    rules(record(1000, "RuntimeError", "close timeout"))
    rules(record(1001, "RuntimeError", "close timeout"))
    rules(record(1000, "NameError", "bad_symbol is undefined"))
    with pytest.raises(campaign.ArmStop):
        rules(record(1001, "NameError", "bad_symbol is undefined"))


@pytest.mark.parametrize("finalist_failure", [False, True])
def test_all_arms_share_runtime_and_best_two_extend(tmp_path, monkeypatch, finalist_failure):
    calls = []
    sentinel = object()
    monkeypatch.setattr(campaign, "MolmoPointRuntime", lambda: sentinel)

    def fake_main(argv, *, molmo_runtime, cell_completed_callback):
        args = dict(zip(argv[::2], argv[1::2]))
        calls.append(args)
        assert molmo_runtime is sentinel
        assert args["--region-backend"] == "rgbd"
        assert not any("sam3" in arg for arg in argv)
        if finalist_failure and args["--phase"] == "full60":
            return 2
        from pathlib import Path
        output = Path(args["--output-dir"])
        n = 6 if args["--phase"] == "prefix" else 30
        for suite in ("vanilla", "sealed_randomized"):
            campaign.write_json(output / suite / "arrow_pick_place_matrix_status.json", {"cells": [
                {"status": "completed", "evaluator_result": output.name in {"dense_agentview", "geometry_agentview"},
                 "suite_mode": suite, "task_id": 4, "seed": 1000 + i} for i in range(n)
            ]})
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    assert campaign.main(["--output-dir", str(tmp_path)]) == (2 if finalist_failure else 0)
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert len(report["screen"]) == 6
    assert len(report["finalists"]) == 2
    assert report["finalists"][0]["arm"]["name"] == "dense_agentview"
    assert len(calls) == 8
    assert {c["--molmopoint-prompt-id"] for c in calls} == set(campaign.canary.MOLMOPOINT_PROMPT_IDS)
    assert report["sam3_used"] is False
    assert report["status"] == ("incomplete_finalists" if finalist_failure else "completed")


def test_zero_success_screen_arms_are_not_extended(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "MolmoPointRuntime", object)

    def fake_main(argv, *, molmo_runtime, cell_completed_callback):
        args = dict(zip(argv[::2], argv[1::2]))
        calls.append(args)
        output = __import__("pathlib").Path(args["--output-dir"])
        count = 6 if args["--phase"] == "prefix" else 30
        for suite in ("vanilla", "sealed_randomized"):
            campaign.write_json(output / suite / "arrow_pick_place_matrix_status.json", {"cells": [
                {"status": "completed", "evaluator_result": False, "suite_mode": suite, "task_id": 4, "seed": 1000 + i}
                for i in range(count)
            ]})
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    assert campaign.main(["--output-dir", str(tmp_path)]) == 2
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert len(calls) == len(campaign.ARMS)
    assert all(screen["metrics"]["terminal_cells"] == 12 for screen in report["screen"])
    assert report["finalists"] == []
    assert report["status"] == "no_successful_arm"


@pytest.mark.parametrize("success", [False, True])
def test_motion_probe_has_two_matched_prefixes_one_runtime_and_no_extension(tmp_path, monkeypatch, success):
    sentinel = object()
    calls = []
    monkeypatch.setattr(campaign, "MolmoPointRuntime", lambda: sentinel)

    def fake_main(argv, *, molmo_runtime, cell_completed_callback):
        assert molmo_runtime is sentinel
        assert "--motion-diagnostics" in argv
        pairs = [item for item in argv if item != "--motion-diagnostics"]
        args = dict(zip(pairs[::2], pairs[1::2]))
        calls.append(args)
        assert args["--phase"] == "prefix"
        assert args["--region-backend"] == "rgbd"
        assert args["--variant"] == "molmo_dense_agentview"
        output = __import__("pathlib").Path(args["--output-dir"])
        for suite in ("vanilla", "sealed_randomized"):
            campaign.write_json(output / suite / "arrow_pick_place_matrix_status.json", {"cells": [
                {"status": "completed", "evaluator_result": success, "suite_mode": suite, "task_id": task, "seed": seed}
                for task in (4, 6, 9) for seed in (1000, 1001)
            ]})
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    assert campaign.main(["--output-dir", str(tmp_path), "--motion-probe"]) == 0
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert len(calls) == 2
    assert [c["--motion-profile"] for c in calls] == ["baseline", "placement_micro5mm"]
    assert len({c["--molmopoint-prompt-id"] for c in calls}) == 1
    assert sum(r["metrics"]["planned"] for r in report["screen"]) == 24
    assert all(r["metrics"]["terminal_cells"] == 12 for r in report["screen"])
    assert report["finalists"] == []
    assert report["status"] == "motion_probe_completed"


def test_motion_probe_rejects_global_repair_gate(tmp_path):
    with pytest.raises(SystemExit):
        campaign.main(["--output-dir", str(tmp_path), "--motion-probe", "--repair-gate"])


@pytest.mark.parametrize("extra", [
    ["--arms", "unknown"], ["--arms", ""],
    ["--arms", "dense_agentview,dense_agentview"],
    ["--motion-probe", "--arms", "dense_agentview"],
    ["--motion-probe", "--observation-profile", "hover20mm"],
    ["--motion-probe", "--motion-profile", "placement_micro5mm"],
    ["--repair-gate", "--arms", "dense_agentview"],
])
def test_selected_screen_rejects_invalid_or_confounded_arguments(tmp_path, extra):
    with pytest.raises(SystemExit):
        campaign.main(["--output-dir", str(tmp_path), *extra])
    assert not (tmp_path / "campaign.json").exists()


@pytest.mark.parametrize("motion_profile", ["baseline", "release_plus20mm", "release20_visual_xy"])
def test_selected_clearance_screen_records_hover_profile_without_extension(tmp_path, monkeypatch, motion_profile):
    calls = []
    monkeypatch.setattr(campaign, "MolmoPointRuntime", object)

    def fake_main(argv, **kwargs):
        args = dict(zip(argv[::2], argv[1::2]))
        calls.append(args)
        output = __import__("pathlib").Path(args["--output-dir"])
        for suite in ("vanilla", "sealed_randomized"):
            campaign.write_json(output / suite / "arrow_pick_place_matrix_status.json", {"cells": [
                {"status": "completed", "evaluator_result": True, "suite_mode": suite,
                 "task_id": task, "seed": seed}
                for task in (4, 6, 9) for seed in (1000, 1001)
            ]})
        return 0

    monkeypatch.setattr(campaign.canary, "main", fake_main)
    assert campaign.main(["--output-dir", str(tmp_path), "--arms", "dense_agentview_clearance",
                          "--observation-profile", "hover20mm", "--motion-profile", motion_profile,
                          "--screen-only"]) == 0
    assert len(calls) == 1
    assert calls[0]["--molmopoint-prompt-id"] == "rim_clearance"
    assert calls[0]["--observation-profile"] == "hover20mm"
    assert calls[0]["--phase"] == "prefix"
    if motion_profile == "baseline":
        assert "--motion-profile" not in calls[0]
    else:
        assert calls[0]["--motion-profile"] == motion_profile
    report = json.loads((tmp_path / "campaign.json").read_text())
    assert report["observation_profile"] == "hover20mm"
    assert report["motion_profile"] == motion_profile
    assert report["arms"][0]["motion_profile"] == motion_profile
    assert [arm["name"] for arm in report["arms"]] == ["dense_agentview_clearance"]
    assert report["screen"][0]["metrics"]["planned"] == 12
    assert report["finalists"] == []
