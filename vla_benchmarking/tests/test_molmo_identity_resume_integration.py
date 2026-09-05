"""Independent matrix-resume tests for scientific identity preservation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vla_benchmarking.evaluation import run_arrow_pick_place_matrix as matrix


TASKS = (4, 6, 9)
SUITES = ("vanilla", "sealed_randomized")


def _identity_metadata() -> dict:
    return {
        "scientific_identity_hash": "h0",
        "scientific_identity_payload": {
            "schema": "canonical_grasp_controller_identity.v1",
            "execution_sha": "a" * 40,
            "controller_config_digest": "b" * 64,
            "model": {
                "molmopoint_model": "model",
                "molmopoint_revision": "c" * 40,
                "molmopoint_prompt_id": "rim_clearance",
            },
            "variant": {"camera_name": "agentview"},
            "motion": {"profile": "baseline", "params": {"release_height_offset_m": 0.0}},
            "observation": {"profile": "baseline", "params": {"phase_tolerance_m": 0.015}},
            "task_seed": {"task_ids": list(TASKS), "seed_base": 1000},
        },
        "region_backend": "rgbd",
        "backend": "rgbd_region",
    }


class _Env:
    def close(self) -> None:
        pass


def _episode_for(*, calls: list[tuple[int, int]], interrupt_last_prefix: bool = False):
    def episode(**kwargs):
        task, seed = int(kwargs["task_id"]), int(kwargs["seed"])
        calls.append((task, seed))
        if interrupt_last_prefix and (task, seed) == (9, 1001):
            raise KeyboardInterrupt("prefix interrupted after durable prior cells")
        if (task, seed) == (6, 1000):
            raise RuntimeError("synthetic controller failure")
        # Exercise both evaluator outcomes while keeping all terminal records
        # eligible for extension.  The matrix itself owns status persistence.
        return {
            "evaluator_success": bool((task, seed) == (4, 1000)),
            "motion_executed": True,
            "audit_path": None,
        }

    return episode


def _matrix_kwargs(root: Path, *, suite: str, episodes: int, metadata, calls: list[tuple[int, int]]):
    return {
        "output_root": root / suite,
        "task_ids": TASKS,
        "episodes_per_task": episodes,
        "seed_base": 1000,
        "suite_mode": suite,
        "dry_run": False,
        "execute_motion": True,
        "allow_unvalidated_profile": True,
        "continue_on_motion_failure": True,
        "env_builder": lambda *_args, **_kwargs: _Env(),
        "episode_runner": _episode_for(calls=calls),
        "arrow_input_builder": lambda *_args, **_kwargs: {},
        "experiment_metadata": metadata,
    }


def test_prefix_to_full_resume_preserves_terminal_outcomes_and_runs_only_new_seeds(tmp_path):
    for suite in SUITES:
        calls: list[tuple[int, int]] = []
        prefix_kwargs = _matrix_kwargs(
            tmp_path, suite=suite, episodes=2, metadata=_identity_metadata(), calls=calls,
        )
        prefix_kwargs["episode_runner"] = _episode_for(calls=calls, interrupt_last_prefix=True)
        with pytest.raises(KeyboardInterrupt, match="prefix interrupted"):
            matrix.run_matrix(**prefix_kwargs)

        status_path = tmp_path / suite / matrix.STATUS_FILENAME
        prefix_status = json.loads(status_path.read_text(encoding="utf-8"))
        prefix_by_key = {
            (int(item["task_id"]), int(item["seed"])): item
            for item in prefix_status["cells"]
        }
        assert prefix_by_key[(4, 1000)]["status"] == "completed"
        assert prefix_by_key[(4, 1000)]["evaluator_result"] is True
        assert prefix_by_key[(4, 1001)]["status"] == "completed"
        assert prefix_by_key[(4, 1001)]["evaluator_result"] is False
        assert prefix_by_key[(6, 1000)]["status"] == "failed"
        assert prefix_by_key[(9, 1001)]["status"] == "interrupted"

        resume_calls: list[tuple[int, int]] = []
        full_kwargs = _matrix_kwargs(
            tmp_path, suite=suite, episodes=10, metadata=_identity_metadata(), calls=resume_calls,
        )
        summary = matrix.run_matrix(
            **{**full_kwargs, "resume": True, "resume_terminal": True},
        )
        assert summary["total_cells"] == 30
        assert set(resume_calls) == {
            (task, seed) for task in TASKS for seed in range(1002, 1010)
        }
        assert len(resume_calls) == 24
        full_status = json.loads(status_path.read_text(encoding="utf-8"))
        full_by_key = {
            (int(item["task_id"]), int(item["seed"])): item
            for item in full_status["cells"]
        }
        assert full_by_key[(4, 1000)]["evaluator_result"] is True
        assert full_by_key[(4, 1001)]["evaluator_result"] is False
        assert full_by_key[(6, 1000)]["status"] == "failed"
        assert full_by_key[(9, 1001)]["status"] == "interrupted"


@pytest.mark.parametrize("path, value", [
    (("scientific_identity_payload", "execution_sha"), "f" * 40),
    (("scientific_identity_payload", "model", "molmopoint_prompt_id"), "other_prompt"),
    (("scientific_identity_payload", "model", "molmopoint_revision"), "d" * 40),
    (("scientific_identity_payload", "controller_config_digest"), "e" * 64),
    (("scientific_identity_payload", "variant", "camera_name"), "robot0_eye_in_hand"),
    (("scientific_identity_payload", "motion", "profile"), "release_plus20mm"),
    (("scientific_identity_payload", "observation", "profile"), "hover20mm"),
    (("scientific_identity_payload", "task_seed", "seed_base"), 2000),
])
def test_identity_change_is_rejected_before_resume_output_mutation(tmp_path, path, value):
    base = _identity_metadata()
    calls: list[tuple[int, int]] = []
    matrix.run_matrix(**_matrix_kwargs(
        tmp_path, suite="vanilla", episodes=2, metadata=base, calls=calls,
    ))
    output_root = tmp_path / "vanilla"
    tracked = tuple(output_root / name for name in (
        matrix.MANIFEST_JSON_FILENAME,
        matrix.MANIFEST_JSONL_FILENAME,
        matrix.STATUS_FILENAME,
        matrix.SUMMARY_FILENAME,
    ))
    before = {path: path.read_bytes() for path in tracked}
    changed = deepcopy(base)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="resume contract hash"):
        matrix.run_matrix(**{
            **_matrix_kwargs(tmp_path, suite="vanilla", episodes=10, metadata=changed, calls=[]),
            "resume": True,
            "resume_terminal": True,
        })
    assert {path: path.read_bytes() for path in tracked} == before


def test_legacy_resume_without_identity_is_rejected_before_mutation(tmp_path):
    calls: list[tuple[int, int]] = []
    matrix.run_matrix(**_matrix_kwargs(
        tmp_path, suite="vanilla", episodes=2, metadata=None, calls=calls,
    ))
    output_root = tmp_path / "vanilla"
    status_path = output_root / matrix.STATUS_FILENAME
    before = status_path.read_bytes()
    with pytest.raises(ValueError, match="resume contract hash"):
        matrix.run_matrix(**{
            **_matrix_kwargs(tmp_path, suite="vanilla", episodes=10, metadata=_identity_metadata(), calls=[]),
            "resume": True,
            "resume_terminal": True,
        })
    assert status_path.read_bytes() == before
