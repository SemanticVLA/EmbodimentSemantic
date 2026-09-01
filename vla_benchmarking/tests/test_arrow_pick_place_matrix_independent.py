"""Adversarial bookkeeping tests for the arrow pick/place matrix.

These tests do not construct LIBERO or call simulator motion.  They exercise
the matrix through its injected environment/input/episode seams and check the
contracts that are easy to get wrong while the happy-path tests remain green.
"""

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


class _Env:
    def __init__(self, task_id: int, seed: int):
        self.task_id = task_id
        self.seed = seed
        self.closed = False

    def close(self):
        self.closed = True


def _audit(success, *, motion=False):
    return {
        "audit_path": None,
        "frames": [],
        "phase_frames": [],
        "phases": [],
        "evaluator_success": success,
        "motion_executed": motion,
    }


def _run(matrix, root, *, task_ids=(0,), episodes=1, dry_run=False,
         execute_motion=True, allow_unvalidated_profile=True, **overrides):
    return matrix.run_matrix(
        output_root=root,
        task_ids=list(task_ids),
        episodes_per_task=episodes,
        dry_run=dry_run,
        execute_motion=execute_motion,
        allow_unvalidated_profile=allow_unvalidated_profile,
        env_builder=overrides.pop("env_builder", lambda task, seed, resolution: _Env(task, seed)),
        episode_runner=overrides.pop("episode_runner", lambda **kwargs: _audit(None, motion=not dry_run)),
        arrow_input_builder=overrides.pop("arrow_input_builder", lambda env, task, resolution: {}),
        **overrides,
    )


def test_real_run_wires_evaluator_callback_to_episode_runner(matrix, tmp_path: Path):
    """A motion run must provide the episode evaluator instead of silently recording null."""
    seen = {}

    def episode_runner(**kwargs):
        seen.update(kwargs)
        return _audit(True, motion=True)

    summary = _run(
        matrix,
        tmp_path,
        dry_run=False,
        execute_motion=True,
        allow_unvalidated_profile=False,
        episode_runner=episode_runner,
    )
    assert callable(seen.get("evaluator")), "real matrix run did not wire evaluator callback"
    assert summary["evaluated_cells"] == 1
    assert summary["success_rate"] == 1.0


def test_failure_classes_distinguish_environment_input_and_evaluator_failures(matrix, tmp_path: Path):
    def build_env(task_id, seed, resolution):
        if task_id == 0:
            raise RuntimeError("simulator unavailable")
        return _Env(task_id, seed)

    def build_inputs(env, task_id, resolution):
        if task_id == 1:
            raise ValueError("arrow depth missing")
        return {}

    def episode_runner(**kwargs):
        raise RuntimeError("evaluator check failed")

    summary = _run(
        matrix,
        tmp_path,
        task_ids=(0, 1, 2),
        dry_run=False,
        execute_motion=True,
        env_builder=build_env,
        arrow_input_builder=build_inputs,
        episode_runner=episode_runner,
    )
    assert summary["failed_cells_count"] == 3
    records = [
        json.loads(line)
        for line in (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert {record["task_id"]: record["failure_class"] for record in records} == {
        0: "environment_failure",
        1: "input_failure",
        2: "evaluator_failure",
    }


def test_summary_exposes_conservative_and_evaluable_denominators(matrix, tmp_path: Path):
    """Success rates must make both excluded-null and all-cell denominators explicit."""
    outcomes = iter((True, False, None, "raise"))

    def episode_runner(**kwargs):
        outcome = next(outcomes)
        if outcome == "raise":
            raise RuntimeError("controller timeout")
        return _audit(outcome, motion=True)

    summary = _run(
        matrix,
        tmp_path,
        episodes=4,
        dry_run=False,
        execute_motion=True,
        episode_runner=episode_runner,
    )
    assert summary["evaluated_cells"] == 2
    assert summary["successes"] == 1
    assert summary["evaluable_denominator"] == 2
    assert summary["conservative_denominator"] == 4
    assert summary["evaluable_success_rate"] == pytest.approx(0.5)
    assert summary["conservative_success_rate"] == pytest.approx(0.25)


def test_resume_preserves_existing_jsonl_and_appends_retry_record(matrix, tmp_path: Path):
    attempts = {1000: 0, 1001: 0}

    def episode_runner(**kwargs):
        seed = kwargs["seed"]
        attempts[seed] += 1
        if seed == 1001 and attempts[seed] == 1:
            raise RuntimeError("transient controller timeout")
        return _audit(True, motion=False)

    first = _run(
        matrix,
        tmp_path,
        episodes=2,
        dry_run=True,
        execute_motion=False,
        episode_runner=episode_runner,
    )
    assert first["failed_cells_count"] == 1
    manifest_path = tmp_path / matrix.MANIFEST_JSONL_FILENAME
    original_lines = manifest_path.read_text(encoding="utf-8").splitlines()

    resumed = _run(
        matrix,
        tmp_path,
        episodes=2,
        dry_run=True,
        execute_motion=False,
        resume=True,
        episode_runner=episode_runner,
    )
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert lines[: len(original_lines)] == original_lines
    assert len(lines) == len(original_lines) + 1
    assert json.loads(lines[-1])["seed"] == 1001
    assert resumed["completed_cells"] == 2
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    retried = next(record for record in status["cells"] if record["seed"] == 1001)
    assert len(retried["attempts"]) == 2


def test_interruption_leaves_resumeable_status_and_missing_cell_is_retried(matrix, tmp_path: Path):
    interrupted = {"raised": False}

    def interrupting_episode(**kwargs):
        if kwargs["seed"] == 1001 and not interrupted["raised"]:
            interrupted["raised"] = True
            kwargs["env"]._arrow_motion_began = True
            raise KeyboardInterrupt()
        return _audit(None, motion=False)

    with pytest.raises(KeyboardInterrupt):
        _run(matrix, tmp_path, episodes=2, dry_run=True, execute_motion=False,
             episode_runner=interrupting_episode)
    status_path = tmp_path / matrix.STATUS_FILENAME
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert [record["status"] for record in status["cells"]] == ["completed", "interrupted"]
    interrupted_record = status["cells"][1]
    assert interrupted_record["failure_class"] == "interrupted"
    assert interrupted_record["motion_began"] is True

    with pytest.raises(RuntimeError, match="refusing automatic resume"):
        _run(matrix, tmp_path, episodes=2, dry_run=True, execute_motion=False,
             resume=True, episode_runner=interrupting_episode)

    resumed = _run(
        matrix, tmp_path, episodes=2, dry_run=True, execute_motion=False,
        resume=True, retry_motion_began=True, episode_runner=interrupting_episode,
    )
    assert resumed["completed_cells"] == 2
    assert interrupted["raised"] is True
    final_status = json.loads(status_path.read_text(encoding="utf-8"))
    retried = final_status["cells"][1]
    assert retried["status"] == "completed"
    assert retried["attempt_output_dir"].endswith("attempt_02")
    assert [attempt["status"] for attempt in retried["attempts"]] == ["interrupted", "completed"]


def test_init_state_metadata_survives_status_and_manifest_records(matrix, tmp_path: Path):
    _run(matrix, tmp_path, episodes=3, dry_run=True, execute_motion=False)
    manifest = [
        json.loads(line)
        for line in (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    assert [record["init_state_index_candidate"] for record in manifest] == [0, 1, 2]
    assert [record["init_state_index_candidate"] for record in status["cells"]] == [0, 1, 2]
    policy = status["protocol"]["init_state_policy"]
    assert policy["index"] == "env_reported_selected_index"
    assert policy["fallback"] == "none; never infer from episode_index"


def test_dry_run_forces_evaluator_result_to_null(matrix, tmp_path: Path):
    summary = _run(
        matrix,
        tmp_path,
        dry_run=True,
        execute_motion=False,
        episode_runner=lambda **kwargs: _audit(True, motion=False),
    )
    record = json.loads(
        (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["evaluator_result"] is None
    assert record["audit"]["evaluator_success"] is None
    assert summary["evaluated_cells"] == 0
    assert summary["success_rate"] is None
