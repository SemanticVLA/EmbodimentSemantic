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
    assert identity["robocasa_adapter"]["frame_contract"] == "robocasa_frozen_b0_proprio_current_base_actions_v2"


def test_semantic_source_fingerprint_changes_identity_and_rejects_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "live.py"
    source.write_text("ADAPTER_REVISION = 1\n", encoding="utf-8")
    fingerprint = runner._robocasa_semantic_source_fingerprint
    monkeypatch.setattr(
        runner,
        "_robocasa_semantic_source_fingerprint",
        lambda: fingerprint(tmp_path),
    )
    identity_a = runner._robocasa_controller_identity()
    experiment_a = runner._experiment_identity(
        tasks=runner.PICK_PLACE_TASKS[:1], episodes_per_task=1,
        seed_base=1000, split="target", mode="full",
        controller_identity=identity_a,
    )
    source.write_text("ADAPTER_REVISION = 2\n", encoding="utf-8")
    identity_b = runner._robocasa_controller_identity()
    experiment_b = runner._experiment_identity(
        tasks=runner.PICK_PLACE_TASKS[:1], episodes_per_task=1,
        seed_base=1000, split="target", mode="full",
        controller_identity=identity_b,
    )

    fingerprint_a = identity_a["robocasa_adapter"]["semantic_source_fingerprint"]
    fingerprint_b = identity_b["robocasa_adapter"]["semantic_source_fingerprint"]
    assert fingerprint_a["files"] == fingerprint_b["files"] == ["live.py"]
    assert fingerprint_a["sha256"] != fingerprint_b["sha256"]
    assert experiment_a != experiment_b

    row = runner.TerminalRow(
        task=runner.PICK_PLACE_TASKS[0].name, episode_index=0, seed=1000,
        split="target", mode="full", terminal=True, success=False,
        failure_category="task_failure",
        metadata={"experiment_identity": experiment_a, "controller_identity": identity_a},
    )
    runner._write_outputs(tmp_path / "results", [row], mode="full", experiment_identity=experiment_a)
    results_path = tmp_path / "results" / "results.jsonl"
    preserved = results_path.read_bytes()
    with pytest.raises(ValueError, match="different experiment identity"):
        runner._read_existing(results_path, experiment_identity=experiment_b)
    assert results_path.read_bytes() == preserved


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
    expected_identity = {**identity, "grasp_profile": runner._grasp_profile_identity("canonical_rim")}
    assert all(row["metadata"]["controller_identity"] == expected_identity for row in rows)
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


def test_interrupted_matrix_preserves_rows_and_resumes_missing_cells(monkeypatch, tmp_path):
    calls = []
    interrupted = True

    def cell(**kwargs):
        nonlocal interrupted
        name = kwargs["task"].name
        calls.append(name)
        if len(calls) == 3 and interrupted:
            interrupted = False
            raise KeyboardInterrupt("simulated interruption")
        return runner.TerminalRow(
            task=name, episode_index=kwargs["episode_index"], seed=kwargs["seed"],
            split="target", mode=kwargs["mode"], terminal=True, success=False,
            failure_category="task_failure",
            metadata={"experiment_identity": kwargs["experiment_identity"]},
        )

    monkeypatch.setattr(runner, "_run_live_cell", cell)
    with pytest.raises(KeyboardInterrupt):
        runner.run(output_dir=tmp_path, mode="full", execute_motion=True)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["expected"] == summary["planned"] == 21
    assert summary["terminal"] == 2
    assert summary["evaluation_status"] == "incomplete"
    first_two = calls[:2]
    calls.clear()
    assert runner.run(output_dir=tmp_path, mode="full", execute_motion=True) == 0
    assert len(calls) == 19
    assert not set(first_two).intersection(calls)
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len({(row["task"], row["seed"]) for row in rows}) == 21
    assert all(row["seed"] == 1000 for row in rows)
    assert json.loads((tmp_path / "summary.json").read_text())["terminal"] == 21


@pytest.mark.parametrize("kwargs", [
    {"episodes_per_task": 0}, {"seed_base": -1}, {"split": "train"},
    {"task_names": ["CheesyBread", "CheesyBread"]},
])
def test_live_matrix_rejects_invalid_contract_before_execution(monkeypatch, tmp_path, kwargs):
    monkeypatch.setattr(runner, "_run_live_cell", lambda **kw: pytest.fail("must not execute"))
    with pytest.raises(ValueError):
        runner.run(output_dir=tmp_path, mode="full", execute_motion=True, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_names": ["PickPlaceCounterToCabinet"]}, "all 21"),
        ({"episodes_per_task": 2}, "one episode"),
        ({"seed_base": 1001}, "seed_base=1000"),
    ],
)
def test_full_mode_requires_frozen_canonical_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    kwargs: dict[str, object], message: str,
) -> None:
    monkeypatch.setattr(runner, "_robocasa_controller_identity", lambda: {"canonical_controller": {}})
    monkeypatch.setattr(runner, "_run_live_cell", lambda **_kw: pytest.fail("full guard must run before cells"))
    with pytest.raises(ValueError, match=message):
        runner.run(output_dir=tmp_path, mode="full", execute_motion=True, **kwargs)


def test_full_mode_requires_motion_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_robocasa_controller_identity", lambda: {"canonical_controller": {}})
    with pytest.raises(ValueError, match="execute_motion=True"):
        runner.run(output_dir=tmp_path, mode="full")
    assert not (tmp_path / "results.jsonl").exists()
