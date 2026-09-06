from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_benchmarking.robocasa.evaluation import runner


def test_controller_identity_records_canonical_and_adapter_provenance() -> None:
    identity = runner._robocasa_controller_identity()

    assert identity["schema"] == "robocasa_controller_identity.v2"
    assert identity["canonical_controller"]["name"]
    assert identity["canonical_controller"]["config_hash"]
    assert identity["canonical_controller"]["config_hash"] == (
        identity["canonical_controller"]["policy_lock_canonical_config_sha256"]
    )
    if identity["robocasa_adapter"]["module"] is not None:
        assert identity["robocasa_adapter"]["module"].endswith(
            "robocasa.arrow_grasp_controller.controller.runner"
        )
        assert identity["robocasa_adapter"]["source_sha256"]
    assert identity["robocasa_adapter"]["frame_contract"] == "robocasa_base_frame_v1"


def test_runner_has_no_libero_import_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "vla_benchmarking.libero" not in source


def test_controller_identity_has_preflight_safe_failure_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = runner.importlib.import_module

    def fail_live_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("vla_benchmarking.robocasa.arrow_grasp_controller"):
            raise ModuleNotFoundError("robocasa runtime unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(runner.importlib, "import_module", fail_live_import)
    identity = runner._robocasa_controller_identity()

    assert identity["availability"] == "partial"
    assert identity["canonical_controller"]["config_hash"] is None
    assert identity["errors"][0]["stage"] == "robocasa_adapter"


@pytest.mark.parametrize(
    ("status", "error", "error_type", "expected"),
    [
        (
            "runtime_backend_unavailable",
            "RoboCasaLiveError: role bbox has no visible area after projection",
            "RoboCasaLiveError",
            "geometry_contract_failure",
        ),
        (
            "controller_failure",
            "workspace point lies outside controller bounds",
            "ValueError",
            "geometry_contract_failure",
        ),
        (
            "runtime_backend_unavailable",
            "No module named 'robocasa'",
            "ModuleNotFoundError",
            "dependency_missing",
        ),
        (
            "controller_failure",
            "phase pregrasp exceeded timeout",
            "ControllerMotionTimeout",
            "controller_failure",
        ),
    ],
)
def test_failure_accounting_separates_geometry_from_environment(
    status: str,
    error: str,
    error_type: str,
    expected: str,
) -> None:
    assert runner._failure_category(
        status, error, error_type=error_type
    ) == expected


def test_canonical_phase_timeout_is_separate_from_task_horizon() -> None:
    live = {
        "horizon": 300,
        "audit": {"controller_variant": {"phase_timeout_steps": 160}},
    }

    assert live["horizon"] != runner._canonical_phase_timeout(live)
    assert runner._canonical_phase_timeout(live) == 160


def test_run_records_identity_and_preserves_all_preflight_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    identity = {
        "schema": "robocasa_controller_identity.v2",
        "availability": "available",
        "canonical_controller": {"name": "canonical", "config_hash": "abc"},
        "robocasa_adapter": {"frame_contract": "base"},
        "errors": [],
    }
    monkeypatch.setattr(runner, "_robocasa_controller_identity", lambda: identity)
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)

    assert runner.run(
        output_dir=tmp_path,
        mode="preflight",
        task_names=["PickPlaceCounterToMicrowave"],
        episodes_per_task=2,
        seed_base=1000,
    ) == 0

    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {1000, 1001}
    assert all(row["metadata"]["controller_identity"] == identity for row in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["expected"] == 2
    assert summary["planned"] == 2
    assert summary["complete"] == 0


def test_geometry_failure_is_completed_failure_not_infrastructure(
    tmp_path,
) -> None:
    row = runner.TerminalRow(
        task="example",
        episode_index=0,
        seed=1000,
        split="target",
        mode="full",
        terminal=True,
        success=False,
        failure_category="geometry_contract_failure",
        metadata={"experiment_identity": "test"},
    )

    runner._write_outputs(
        tmp_path,
        [row],
        mode="full",
        experiment_identity="test",
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["failure_categories"] == {"geometry_contract_failure": 1}
    assert summary["evaluation_status"] == "completed_with_failures"
