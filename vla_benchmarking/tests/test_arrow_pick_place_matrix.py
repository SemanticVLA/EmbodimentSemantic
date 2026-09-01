"""Non-motion contract tests for the arrow pick/place matrix launcher."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def matrix():
    try:
        return importlib.import_module("run_arrow_pick_place_matrix")
    except ModuleNotFoundError as exc:
        if exc.name in {"run_arrow_pick_place_matrix", "robosuite", "libero", "lerobot"}:
            pytest.skip(f"optional dependency unavailable: {exc.name}")
        raise


def test_default_plan_has_100_cells_unique_seeds_and_paths(matrix, tmp_path: Path):
    cells = matrix.plan_cells(output_root=tmp_path)
    assert len(cells) == 100
    assert [cell["task_id"] for cell in cells[:10]] == [0] * 10
    assert [cell["seed"] for cell in cells[:10]] == list(range(1000, 1010))
    assert len({cell["output_dir"] for cell in cells}) == 100
    assert all(Path(cell["output_dir"]).name == f"episode_{cell['episode_index']}_seed_{cell['seed']}" for cell in cells)
    assert all(cell["resolution"] == 256 for cell in cells)
    assert sum(cell["profile_validated"] for cell in cells) == 1
    assert len({(cell["task_id"], cell["init_state_index"]) for cell in cells}) == 100


def test_matrix_requires_explicit_motion_or_dry_run(matrix):
    with pytest.raises(ValueError, match="explicit --execute-motion"):
        matrix.validate_motion_authorization(
            matrix.plan_cells(task_ids=[0], episodes_per_task=1),
            execute_motion=False,
            dry_run=False,
            allow_unvalidated_profile=False,
        )
    with pytest.raises(ValueError, match="allow-unvalidated-profile"):
        matrix.validate_motion_authorization(
            matrix.plan_cells(task_ids=[0], episodes_per_task=2),
            execute_motion=True,
            dry_run=False,
            allow_unvalidated_profile=False,
        )
    matrix.validate_motion_authorization(
        matrix.plan_cells(task_ids=[0], episodes_per_task=2),
        execute_motion=True,
        dry_run=False,
        allow_unvalidated_profile=True,
    )


def test_cli_requires_mode_and_parses_safety_flags(matrix):
    with pytest.raises(SystemExit):
        matrix.parse_args([])
    dry = matrix.parse_args(["--dry-run"])
    assert dry.dry_run is True and dry.execute_motion is False
    execute = matrix.parse_args(["--execute-motion", "--allow-unvalidated-profile", "--resume"])
    assert execute.execute_motion is True and execute.dry_run is False
    assert execute.allow_unvalidated_profile is True and execute.resume is True
    assert matrix.parse_args(["--execute-motion", "--allow-unvalidated-profile", "--continue-on-motion-failure"]).continue_on_motion_failure is True


def test_matrix_writes_complete_plan_manifest_before_build_and_continues_failures(
    matrix, tmp_path: Path
):
    observed = {"plan_cells": None, "built": [], "closed": []}

    class Env:
        def __init__(self, task_id, seed):
            self.task_id, self.seed = task_id, seed

        def close(self):
            observed["closed"].append((self.task_id, self.seed))

    def build_env(task_id, seed, resolution):
        manifest = tmp_path / matrix.MANIFEST_JSON_FILENAME
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        observed["plan_cells"] = len(plan["cells"])
        observed["built"].append((task_id, seed, resolution))
        return Env(task_id, seed)

    def run_episode(**kwargs):
        if kwargs["task_id"] == 1:
            raise RuntimeError("synthetic episode failure")
        return {
            "audit_path": (Path(kwargs["output_dir"]) / "arrow_pick_place_audit.json").as_posix(),
            "frames": ["clean.png"],
            "phase_frames": ["phase.png"],
            "phases": [{"phase": "pregrasp"}],
            "capture_contract": {"valid": True},
            "endpoint_depths_m": {"source_tail": 1.0, "destination_head": 1.0},
            "deprojected_visual_endpoint_world_points_m": {},
            "control_targets_world_m": {},
            "waypoints_world_m": [],
            "evaluator_success": None,
            "motion_executed": False,
        }

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0, 1],
        episodes_per_task=2,
        dry_run=True,
        env_builder=build_env,
        episode_runner=run_episode,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert observed["plan_cells"] == 4
    assert observed["built"] == [(0, 1000, 256), (0, 1001, 256), (1, 1000, 256), (1, 1001, 256)]
    assert len(observed["closed"]) == 4
    assert summary["total_cells"] == 4
    assert summary["completed_cells"] == 2
    assert summary["failed_cells_count"] == 2
    assert summary["failure_by_stage"] == {"run_episode": 2}
    assert summary["failure_by_class"] == {"controller_failure": 2}
    assert summary["planned_success_rate"] is None
    assert summary["evaluable_success_rate"] is None
    assert summary["per_task"]["1"]["failed"] == 2
    lines = (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    records = [json.loads(line) for line in lines]
    assert len(records) == 4
    assert all("diagnostics" in record and "duration_s" in record and "motion_began" in record for record in records)
    assert all("attempts" in record and record["attempts"] for record in records)
    assert (tmp_path / matrix.STATUS_FILENAME).is_file()
    assert (tmp_path / matrix.MANIFEST_JSON_FILENAME).is_file()
    assert summary["contract_hash"]


def test_dry_run_allows_unvalidated_matrix_and_preserves_null_evaluator(matrix, tmp_path: Path):
    closed = []

    class Env:
        def close(self):
            closed.append(True)

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=2,
        dry_run=True,
        env_builder=lambda task_id, seed, resolution: Env(),
        episode_runner=lambda **kwargs: {
            "audit_path": None,
            "frames": [],
            "phase_frames": [],
            "phases": [],
            "evaluator_success": True,
            "motion_executed": False,
        },
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert summary["total_cells"] == 2
    assert summary["success_rate"] is None
    assert summary["planned_success_rate"] is None
    assert summary["conservative_success_rate"] is None
    assert summary["evaluable_success_rate"] is None
    assert summary["per_task"]["0"]["planned_success_rate"] is None
    assert summary["per_task"]["0"]["conservative_success_rate"] is None
    assert len(closed) == 2
    terminal = [
        json.loads(line)
        for line in (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
        if json.loads(line)["status"] == "completed"
    ]
    assert all(record["evaluator_result"] is None for record in terminal)


def test_resume_skips_completed_cells_without_overwriting_plan(matrix, tmp_path: Path):
    calls = []

    class Env:
        def close(self):
            pass

    def build_env(task_id, seed, resolution):
        calls.append((task_id, seed))
        return Env()

    episode = lambda **kwargs: {"audit_path": None, "evaluator_success": None, "motion_executed": False}
    first = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        dry_run=True,
        env_builder=build_env,
        episode_runner=episode,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert first["total_cells"] == 1 and calls == [(0, 1000)]
    resumed = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        dry_run=True,
        resume=True,
        env_builder=lambda *args: (_ for _ in ()).throw(AssertionError("completed cell rerun")),
        episode_runner=episode,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert resumed["completed_cells"] == 1
    assert len(json.loads((tmp_path / matrix.STATUS_FILENAME).read_text())["cells"][0]["attempts"]) == 1
    manifest_records = [
        json.loads(line)
        for line in (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_records) == 1
    assert len(manifest_records[0]["attempts"]) == 1


def test_close_keyboard_interrupt_is_persisted_before_reraise(matrix, tmp_path: Path):
    class Env:
        def close(self):
            raise KeyboardInterrupt("synthetic close interrupt")

    with pytest.raises(KeyboardInterrupt, match="synthetic close interrupt"):
        matrix.run_matrix(
            output_root=tmp_path,
            task_ids=[0],
            episodes_per_task=1,
            dry_run=True,
            env_builder=lambda task_id, seed, resolution: Env(),
            episode_runner=lambda **kwargs: {
                "evaluator_success": None,
                "motion_executed": False,
            },
            arrow_input_builder=lambda env, task_id, resolution: {},
        )
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    cell = status["cells"][0]
    assert cell["status"] == "interrupted"
    assert cell["failure_class"] == "interrupted"
    assert cell["attempts"][-1]["status"] == "interrupted"


def test_non_resume_run_refuses_existing_matrix_outputs(matrix, tmp_path: Path):
    kwargs = {
        "output_root": tmp_path,
        "task_ids": [0],
        "episodes_per_task": 1,
        "dry_run": True,
        "env_builder": lambda task_id, seed, resolution: object(),
        "episode_runner": lambda **unused: {"evaluator_success": None, "motion_executed": False},
        "arrow_input_builder": lambda env, task_id, resolution: {},
    }
    matrix.run_matrix(**kwargs)
    with pytest.raises(FileExistsError, match="output already exists"):
        matrix.run_matrix(**kwargs)


def test_resume_repairs_terminal_status_missing_from_jsonl(matrix, tmp_path: Path):
    class Env:
        def close(self):
            pass

    kwargs = {
        "output_root": tmp_path,
        "task_ids": [0],
        "episodes_per_task": 1,
        "dry_run": True,
        "env_builder": lambda task_id, seed, resolution: Env(),
        "episode_runner": lambda **unused: {"evaluator_success": None, "motion_executed": False},
        "arrow_input_builder": lambda env, task_id, resolution: {},
    }
    matrix.run_matrix(**kwargs)
    manifest_path = tmp_path / matrix.MANIFEST_JSONL_FILENAME
    manifest_path.write_text("", encoding="utf-8")
    matrix.run_matrix(**{**kwargs, "resume": True, "env_builder": lambda *args: (_ for _ in ()).throw(
        AssertionError("completed cell rerun")
    )})
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "completed"


def test_false_evaluator_is_classified_without_changing_execution_failure_count(matrix, tmp_path: Path):
    class Env:
        def close(self):
            pass

    evaluator_seen = []

    def fake_episode(**kwargs):
        evaluator_seen.append(kwargs.get("evaluator"))
        return {
            "audit_path": None,
            "evaluator_success": False,
            "motion_executed": True,
            "phases": [{"phase": "retreat", "status": "reached"}],
        }

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        dry_run=False,
        execute_motion=True,
        env_builder=lambda task_id, seed, resolution: Env(),
        episode_runner=fake_episode,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert summary["completed_cells"] == 1
    assert summary["failed_cells_count"] == 0
    assert summary["failure_by_class"] == {"task_failure": 1}
    assert summary["success_rate"] == 0.0
    assert summary["evaluable_success_rate"] == 0.0
    assert len(evaluator_seen) == 1 and callable(evaluator_seen[0])


def test_interrupt_after_motion_is_persisted_and_resume_refuses_retry(matrix, tmp_path: Path):
    class Env:
        def close(self):
            pass

    def interrupted_episode(**kwargs):
        kwargs["motion_started_callback"]()
        raise KeyboardInterrupt("synthetic interrupt after motion")

    with pytest.raises(KeyboardInterrupt):
        matrix.run_matrix(
            output_root=tmp_path,
            task_ids=[0],
            episodes_per_task=1,
            dry_run=False,
            execute_motion=True,
            env_builder=lambda task_id, seed, resolution: Env(),
            episode_runner=interrupted_episode,
            arrow_input_builder=lambda env, task_id, resolution: {},
        )
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    cell = status["cells"][0]
    assert cell["status"] == "interrupted"
    assert cell["motion_began"] is True
    assert cell["attempts"][-1]["status"] == "interrupted"
    with pytest.raises(RuntimeError, match="motion-started"):
        matrix.run_matrix(
            output_root=tmp_path,
            task_ids=[0],
            episodes_per_task=1,
            dry_run=False,
            execute_motion=True,
            resume=True,
            env_builder=lambda *args: (_ for _ in ()).throw(AssertionError("must not build")),
            episode_runner=interrupted_episode,
            arrow_input_builder=lambda env, task_id, resolution: {},
        )


def test_timeout_after_motion_requires_explicit_retry(matrix, tmp_path: Path):
    class Env:
        def close(self):
            pass

    def timeout_episode(**kwargs):
        kwargs["motion_started_callback"]()
        raise TimeoutError("synthetic controller timeout")

    with pytest.raises(TimeoutError):
        matrix.run_matrix(
            output_root=tmp_path,
            task_ids=[0],
            episodes_per_task=1,
            dry_run=False,
            execute_motion=True,
            env_builder=lambda task_id, seed, resolution: Env(),
            episode_runner=timeout_episode,
            arrow_input_builder=lambda env, task_id, resolution: {},
        )
    with pytest.raises(RuntimeError, match="motion-started"):
        matrix.run_matrix(
            output_root=tmp_path,
            task_ids=[0],
            episodes_per_task=1,
            dry_run=False,
            execute_motion=True,
            resume=True,
            env_builder=lambda *args: (_ for _ in ()).throw(AssertionError("must not build")),
            episode_runner=timeout_episode,
            arrow_input_builder=lambda env, task_id, resolution: {},
        )


def test_continue_on_motion_failure_runs_later_independent_cells(matrix, tmp_path: Path):
    closed = []
    calls = []

    class Env:
        def close(self):
            closed.append(True)

    def episode_runner(**kwargs):
        calls.append(kwargs["task_id"])
        kwargs["motion_started_callback"]()
        if kwargs["task_id"] == 0:
            raise TimeoutError("phase descend_place exceeded 80 steps")
        return {
            "audit_path": None,
            "frames": [],
            "phase_frames": [],
            "phases": [{"phase": "retreat", "status": "stop"}],
            "evaluator_success": True,
            "motion_executed": True,
        }

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0, 1],
        episodes_per_task=1,
        dry_run=False,
        execute_motion=True,
        allow_unvalidated_profile=True,
        continue_on_motion_failure=True,
        env_builder=lambda *args: Env(),
        episode_runner=episode_runner,
        arrow_input_builder=lambda *args: {},
    )
    assert calls == [0, 1]
    assert len(closed) == 2
    assert summary["failed_cells_count"] == 1
    assert summary["completed_cells"] == 1
    assert summary["failure_by_class"] == {"controller_failure": 1}


def test_settle_diagnostics_are_retained_for_environment_failures(matrix, tmp_path: Path):
    class Env:
        _arrow_settle_diagnostics = {
            "steps": 500,
            "final_max_velocity_m_s": 0.12,
            "settled": False,
        }

        def close(self):
            pass

    def episode(**kwargs):
        raise RuntimeError("refusing motion: LIBERO physics was not confirmed settled")

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[5],
        episodes_per_task=1,
        dry_run=False,
        execute_motion=True,
        allow_unvalidated_profile=True,
        env_builder=lambda task_id, seed, resolution: Env(),
        episode_runner=episode,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    cell = status["cells"][0]
    assert cell["failure_class"] == "environment_failure"
    assert cell["settle_diagnostics"]["settled"] is False
    assert summary["diagnostic_aggregates"]["settle"] == {
        "recorded": 1,
        "settled": 0,
        "unsettled": 1,
        "max_final_velocity_m_s": 0.12,
    }


def test_timeout_phase_is_included_in_phase_aggregates(matrix):
    records = [
        {
            "status": "failed",
            "audit": None,
            "phases": [
                {"phase": "preplace", "status": "reached"},
                {"phase": "descend_place", "status": "timeout"},
            ],
        }
    ]
    aggregates = matrix._phase_aggregates(records)
    assert aggregates["preplace"] == {
        "count": 1,
        "statuses": {"reached": 1},
        "failed": 0,
    }
    assert aggregates["descend_place"] == {
        "count": 1,
        "statuses": {"timeout": 1},
        "failed": 1,
    }
