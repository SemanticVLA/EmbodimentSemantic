import json

import pytest

import run_v9d_molmo_campaign as campaign


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
                {"status": "completed", "evaluator_result": output.name == "dense_agentview",
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
