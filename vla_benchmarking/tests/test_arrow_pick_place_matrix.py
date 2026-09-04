"""Non-motion contract tests for the arrow pick/place matrix launcher."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert {cell["suite_mode"] for cell in cells} == {"vanilla"}
    assert {cell["controller_variant"] for cell in cells} == {matrix.DEFAULT_CONTROLLER_VARIANT}
    assert all(
        Path(cell["output_dir"]).parts[-4:-2]
        == ("vanilla", matrix.DEFAULT_CONTROLLER_VARIANT)
        for cell in cells
    )
    randomized = matrix.plan_cells(
        task_ids=[0], episodes_per_task=1, output_root=tmp_path,
        suite_mode="sealed_randomized", controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
    )[0]
    assert randomized["profile_validated"] is False
    assert Path(randomized["output_dir"]).parts[-4:-2] == (
        "sealed_randomized", matrix.DEFAULT_CONTROLLER_VARIANT
    )


def test_early_runtime_diagnostics_preserve_opening_audits_and_exact_budget(matrix):
    env = SimpleNamespace(
        _molmo_sam3_opening_preshape={"status": "failed", "failure_reason": "timeout", "budget_used": 61},
        _molmo_opening_settling_audit={"status": "failed", "shared_budget_used": 61},
        _molmo_sam3_observation_hover={"status": "completed"},
        _molmo_sam3_gripper_open={"status": "completed"},
        _molmo_sam3_action_budget=matrix._episode_module._ActionBudget(1200, used=61),
    )
    observed = matrix._early_runtime_diagnostics(env)
    assert observed["opening_preshape"]["failure_reason"] == "timeout"
    assert observed["opening_settling"]["shared_budget_used"] == 61
    assert observed["observation_hover"]["status"] == "completed"
    assert observed["gripper_open"]["status"] == "completed"
    assert observed["total_actions"] == 61
    assert observed["experimental_action_budget"] == {"used": 61, "limit": 1200, "remaining": 1139}


def test_early_runtime_diagnostics_does_not_add_experimental_fields_to_baseline(matrix):
    assert matrix._early_runtime_diagnostics(SimpleNamespace()) == {}


@pytest.mark.parametrize("fail_after_sample", [False, True])
def test_settling_numpy_telemetry_survives_matrix_json(matrix, tmp_path, monkeypatch, fail_after_sample):
    import numpy as np
    from molmo_sam3 import settling

    observation = {"eef_pos": np.array([0.1, 0.2, 0.5]), "eef_quat": np.array([0., 0., 0., 1.])}
    env = SimpleNamespace(_molmo_sam3_action_budget=matrix._episode_module._ActionBudget(1200))
    env._get_observations = lambda **_kwargs: observation
    steps = []

    def step(action):
        steps.append(action)
        if fail_after_sample and len(steps) == 2:
            observation["eef_pos"] = np.array([0.13, 0.2, 0.5])
        return observation, 0., False, {}

    env.step = step
    monkeypatch.setattr(settling.episode, "normalized_action_for_waypoint", lambda *_args, **_kwargs: np.zeros(7))
    monkeypatch.setattr(settling, "_telemetry", lambda *_args: {"controller": {"output_scale": np.array([0.05, 0.05, 0.05])}})
    kwargs = dict(hover_audit={"hover_world_m": [0.1, 0.2, 0.5], "region_q90_world_z_m": 0.4}, initial_hover_observation=dict(observation), output_dir=tmp_path, motion_settings={})
    if fail_after_sample:
        with pytest.raises(RuntimeError, match="safety envelope"):
            settling._settle_to_original_hover(env, **kwargs)
    else:
        settling._settle_to_original_hover(env, **kwargs)
    record = {"audit": {"opening_settling": {"status": "older"}}}
    matrix._attach_early_runtime_diagnostics(record, matrix._early_runtime_diagnostics(env))
    restored = json.loads(json.dumps(record))
    assert restored["audit"]["opening_settling"]["actions"][0]["controller_telemetry"]["controller"]["output_scale"] == [0.05, 0.05, 0.05]
    assert restored["audit"]["total_actions"] == len(steps)


def test_condition_labels_are_validated(matrix):
    with pytest.raises(ValueError, match="suite_mode must be one of"):
        matrix.plan_cells(suite_mode="random")
    with pytest.raises(ValueError, match="controller_variant"):
        matrix.plan_cells(controller_variant="../unsafe")


def test_source_hash_inventory_covers_condition_and_control_implementations(matrix):
    hashes = matrix._source_file_hashes()
    required = {
        "run_arrow_pick_place_matrix.py",
        "run_arrow_pick_place_eval.py",
        "arrow_controller.py",
        "config.py",
        "bddl_utils.py",
        "radomize_scenes.py",
        "preview_visual_arrows.py",
        "legion/run_arrow_pick_place_dual_matrix.sbatch",
    }
    assert required <= set(hashes)
    assert all(hashes[path] for path in required)
    assert hashes == matrix._source_file_hashes()


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
    condition = matrix.parse_args([
        "--dry-run", "--suite-mode", "sealed_randomized",
        "--controller-variant", matrix.DEFAULT_CONTROLLER_VARIANT,
    ])
    assert condition.suite_mode == "sealed_randomized"
    assert condition.controller_variant == matrix.DEFAULT_CONTROLLER_VARIANT


def test_condition_metadata_is_forwarded_only_when_seam_accepts_it(matrix, tmp_path: Path):
    seen = {}

    class Env:
        def close(self):
            pass

    def build_env(task_id, seed, resolution, suite_mode):
        seen["builder"] = (task_id, seed, resolution, suite_mode)
        return Env()

    def build_inputs(env, task_id, resolution, suite_mode, controller_variant):
        seen["inputs"] = (suite_mode, controller_variant)
        return {}

    def episode_runner(**kwargs):
        seen["episode"] = (kwargs["suite_mode"], kwargs["controller_variant"])
        return {"evaluator_success": None, "motion_executed": False}

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0], episodes_per_task=1, dry_run=True,
        suite_mode="sealed_randomized", controller_variant=matrix.DEFAULT_CONTROLLER_VARIANT,
        env_builder=build_env, arrow_input_builder=build_inputs,
        episode_runner=episode_runner,
    )
    assert seen["builder"] == (0, 1000, 256, "sealed_randomized")
    assert seen["inputs"] == ("sealed_randomized", matrix.DEFAULT_CONTROLLER_VARIANT)
    assert seen["episode"][0] == "sealed_randomized"
    assert seen["episode"][1].name == matrix.DEFAULT_CONTROLLER_VARIANT
    assert summary["suite_mode"] == "sealed_randomized"
    assert summary["controller_variant"] == matrix.DEFAULT_CONTROLLER_VARIANT
    assert summary["protocol"]["condition_label"] == f"sealed_randomized__{matrix.DEFAULT_CONTROLLER_VARIANT}"
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["suite_mode"] == "sealed_randomized"
    assert status["controller_variant"] == matrix.DEFAULT_CONTROLLER_VARIANT
    record = status["cells"][0]
    assert record["suite_mode"] == "sealed_randomized"
    assert record["controller_variant"] == matrix.DEFAULT_CONTROLLER_VARIANT


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


def test_continue_on_motion_failure_aborts_when_close_fails(matrix, tmp_path: Path):
    calls = []

    class Env:
        def close(self):
            raise RuntimeError("close failed")

    def episode_runner(**kwargs):
        calls.append(kwargs["task_id"])
        kwargs["motion_started_callback"]()
        raise TimeoutError("phase descend_place exceeded 80 steps")

    with pytest.raises(RuntimeError, match="close failed"):
        matrix.run_matrix(
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
    assert calls == [0]
    status = json.loads((tmp_path / matrix.STATUS_FILENAME).read_text(encoding="utf-8"))
    cell = status["cells"][0]
    assert cell["close_succeeded"] is False
    assert cell["failure_class"] == "environment_failure"
    assert cell["close_error"] == "close failed"


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


def test_early_capture_and_depth_policy_diagnostics_survive_episode_failure(
    matrix, tmp_path: Path
):
    class Env:
        _arrow_capture_contract = {
            "valid": True,
            "camera_name": "agentview",
            "rgb_shape": [256, 256, 3],
        }
        _arrow_input_context = {
            "camera": "agentview",
            "subject": "akita_black_bowl_1",
            "goal_object": "plate_1",
            "bboxes": {
                "akita_black_bowl_1": [10.0, 20.0, 30.0, 40.0],
                "plate_1": [50.0, 60.0, 70.0, 80.0],
            },
            "relations": [["akita_black_bowl_1", "goal", "plate_1"]],
        }
        _arrow_input_arrow_audit = {
            "relation_count": 1,
            "relation": ["akita_black_bowl_1", "goal", "plate_1"],
            "input_generation_only": True,
        }
        _arrow_endpoints_uv = {
            "source_tail": [20.0, 30.0],
            "destination_head": [60.0, 70.0],
        }
        _arrow_decode_audit = {
            "source": "decode_arrow",
            "success": True,
            "changed_pixel_count": 123,
        }
        _arrow_depth_sanitization_policy = {
            "status": "rejected",
            "rejection_reason": "invalid metric depth at arrow endpoint",
        }
        _arrow_endpoint_depths_m = {
            "source_tail": 0.8,
            "destination_head": 0.9,
        }
        _arrow_endpoint_depth_statistics = {
            "source_tail": {"statistic": "median", "valid_count": 9},
            "destination_head": {"statistic": "median", "valid_count": 9},
        }
        _arrow_deprojected_visual_endpoint_world_points_m = {
            "source_tail": [0.1, 0.2, 0.3],
            "destination_head": [0.4, 0.5, 0.6],
        }
        _arrow_control_targets_world_m = {
            "source_grasp": [0.1, 0.2, 0.33],
            "destination_release": [0.4, 0.5, 0.63],
        }
        _arrow_workspace_validation = {
            "status": "rejected",
            "reason": "synthetic early workspace failure",
        }
        _arrow_waypoints_world_m = {
            "waypoint_0": [0.1, 0.2, 0.4],
            "waypoint_1": [0.1, 0.2, 0.3],
        }
        _arrow_grasp_retry_audit = [
            {"attempt": 1, "status": "completed_no_contact", "z_offset_m": -0.012}
        ]
        _arrow_phase_audit = [
            {"phase": "preplace", "status": "reached", "steps": 4},
            {"phase": "descend_place", "status": "timeout", "steps": 80},
        ]

        def close(self):
            pass

    def episode_runner(**kwargs):
        raise RuntimeError(
            "refusing motion: normalized depth sanitization policy rejected capture"
        )

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        dry_run=False,
        execute_motion=True,
        allow_unvalidated_profile=True,
        env_builder=lambda task_id, seed, resolution: Env(),
        episode_runner=episode_runner,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    assert summary["failure_by_class"] == {"input_failure": 1}
    record = json.loads(
        (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    expected_capture = Env._arrow_capture_contract
    expected_policy = Env._arrow_depth_sanitization_policy
    expected_early_visual = {
        "input_context": Env._arrow_input_context,
        "input_arrow_audit": Env._arrow_input_arrow_audit,
        "arrow_endpoints_uv": Env._arrow_endpoints_uv,
        "arrow_decode_audit": Env._arrow_decode_audit,
    }
    for field, expected in expected_early_visual.items():
        assert record[field] == expected
        assert record["diagnostics"][field] == expected
        assert record["partial_audit"][field] == expected
    assert record["capture_contract"] == expected_capture
    assert record["depth_sanitization_policy"] == expected_policy
    assert record["diagnostics"]["capture_contract"] == expected_capture
    assert record["diagnostics"]["depth_sanitization_policy"] == expected_policy
    assert record["audit"] is None
    assert record["partial_audit"]["capture_contract"] == expected_capture
    assert record["partial_audit"]["depth_sanitization_policy"] == expected_policy
    for field in (
        "endpoint_depths_m",
        "endpoint_depth_statistics",
        "deprojected_visual_endpoint_world_points_m",
        "control_targets_world_m",
        "workspace_validation",
        "waypoints_world_m",
    ):
        expected = getattr(Env, f"_arrow_{field}")
        assert record[field] == expected
        assert record["diagnostics"][field] == expected
        assert record["partial_audit"][field] == expected
    expected_grasp_retries = Env._arrow_grasp_retry_audit
    assert record["grasp_retries"] == expected_grasp_retries
    assert record["diagnostics"]["grasp_retries"] == expected_grasp_retries
    assert record["partial_audit"]["grasp_retries"] == expected_grasp_retries
    assert summary["phase_aggregates"]["descend_place"] == {
        "count": 1,
        "statuses": {"timeout": 1},
        "failed": 1,
    }
    assert summary["diagnostic_aggregates"]["capture"] == {"valid": 1, "missing": 0}
    assert summary["diagnostic_aggregates"]["parser"] == {"success": 1, "failure": 0}
    assert summary["diagnostic_aggregates"]["depth"] == {"recorded": 0, "failure": 1}


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


def test_external_controller_config_is_resolved_once_and_recorded(matrix, tmp_path: Path):
    config = Path(matrix.__file__).resolve().parent / "controller_configs" / "v9d_rgbd_region_grasp_search.json"
    observed = {}

    class Env:
        def close(self):
            pass

    def external_env_builder(task_id, seed, resolution, *, suite_mode, controller_variant):
        observed["environment_controller_variant"] = controller_variant
        return Env()

    def episode_runner(**kwargs):
        observed.update(kwargs)
        variant = kwargs["controller_variant"]
        return {
            "evaluator_success": None,
            "controller_variant": variant.provenance(),
            "capture_contract": {"valid": True},
            "phases": [],
            "grasp_search": [],
            "micro_corrections": [],
        }

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        controller_config=config,
        dry_run=True,
        env_builder=external_env_builder,
        episode_runner=episode_runner,
        arrow_input_builder=lambda env, task_id, resolution: {},
    )

    assert "controller_config" not in observed
    assert not isinstance(observed["controller_variant"], str)
    assert observed["environment_controller_variant"] is observed["controller_variant"]
    config_provenance = summary["controller_config"]
    assert config_provenance["config_hash"]
    assert summary["protocol"]["controller_config"] == config_provenance
    record = json.loads(
        (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["controller_config_hash"] == config_provenance["config_hash"]
    assert record["controller_config_path"] == config_provenance["path"]
    assert record["controller_config"] == config_provenance


def test_invalid_external_controller_config_fails_before_environment_build(
    matrix, tmp_path: Path
):
    invalid = tmp_path / "invalid-controller.json"
    invalid.write_text(json.dumps({"name": "v9", "unknown_policy": 1}), encoding="utf-8")
    env_calls = []

    with pytest.raises(ValueError, match="unknown controller config keys"):
        matrix.run_matrix(
            output_root=tmp_path / "outputs",
            task_ids=[0],
            episodes_per_task=1,
            controller_config=invalid,
            dry_run=True,
            env_builder=lambda *args: env_calls.append(args),
            episode_runner=lambda **kwargs: {},
            arrow_input_builder=lambda *args: {},
        )

    assert env_calls == []
    assert not (tmp_path / "outputs").exists()


def test_external_runtime_provenance_survives_controller_failure(matrix, tmp_path: Path):
    config = Path(matrix.__file__).resolve().parent / "controller_configs" / "v9d_rgbd_region_grasp_search.json"

    class Env:
        _arrow_capture_contract = {"valid": True}
        _arrow_phase_audit = [{"phase": "descend", "status": "timeout", "steps": 160}]
        _arrow_grasp_search_audit = []
        _arrow_micro_correction_audit = []

        def close(self):
            pass

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        controller_config=config,
        dry_run=True,
        env_builder=lambda task_id, seed, resolution: Env(),
        episode_runner=lambda **kwargs: (_ for _ in ()).throw(TimeoutError("controller timeout")),
        arrow_input_builder=lambda env, task_id, resolution: {},
    )
    record = json.loads(
        (tmp_path / matrix.MANIFEST_JSONL_FILENAME).read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["status"] == "failed"
    assert record["controller_config_observed"]["config_hash"] == summary["controller_config"]["runtime_hash"]
    assert record["controller_config_observed"]["canonical"] == summary["controller_config"]["canonical"]
