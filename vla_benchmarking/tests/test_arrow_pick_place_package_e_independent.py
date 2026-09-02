"""Independent Package-E checks for suite realization and bounded recovery.

All runtime checks use injected fakes.  No LIBERO environment is constructed
and no simulator action is sent.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def matrix():
    return importlib.import_module("run_arrow_pick_place_matrix")


@pytest.fixture(scope="module")
def episode():
    return importlib.import_module("run_arrow_pick_place_eval")


def test_sealed_suite_mode_is_realized_at_each_injected_boundary(matrix, tmp_path: Path):
    seen = {}

    class Env:
        def close(self):
            pass

    def build_env(task_id, seed, resolution, *, suite_mode):
        seen["build"] = (task_id, seed, resolution, suite_mode)
        return Env()

    def build_inputs(env, task_id, resolution, *, suite_mode, controller_variant):
        seen["inputs"] = (suite_mode, controller_variant)
        return {}

    def run_episode(**kwargs):
        seen["episode"] = (kwargs["suite_mode"], kwargs["controller_variant"])
        return {"evaluator_success": None, "motion_executed": False}

    summary = matrix.run_matrix(
        output_root=tmp_path,
        task_ids=[0],
        episodes_per_task=1,
        suite_mode="sealed_randomized",
        controller_variant="rgbd_arrow_v2",
        dry_run=True,
        env_builder=build_env,
        arrow_input_builder=build_inputs,
        episode_runner=run_episode,
    )
    assert seen["build"] == (0, 1000, 256, "sealed_randomized")
    assert seen["inputs"] == ("sealed_randomized", "rgbd_arrow_v2")
    assert seen["episode"] == ("sealed_randomized", "rgbd_arrow_v2")
    assert summary["condition_label"] == "sealed_randomized__rgbd_arrow_v2"
    assert summary["protocol"]["suite_contract"]["sealed_randomized"]


def test_pure_controller_has_no_new_runtime_api_or_variant_dependency(episode):
    controller = importlib.import_module("arrow_controller")
    path = Path(controller.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    forbidden = {
        "libero", "mujoco", "robosuite", "env", "environment", "sim", "model",
        "bboxes", "bbox", "evaluator", "success", "suite_mode", "controller_variant",
        "recovery", "recovery_callback", "body_xpos", "geom_xpos", "object_pose",
    }
    executable_refs = set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            executable_refs.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            executable_refs.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0].lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0].lower())
    assert not executable_refs.intersection(forbidden)
    assert imports <= {"__future__", "dataclasses", "typing", "numpy"}

    # The new suite/variant/recovery APIs stay in the runner.  The controller
    # function signatures remain explicit geometry-only seams.
    for name in ("decode_arrow", "deproject_endpoint", "build_bowl_waypoints", "normalized_osc_action"):
        parameters = inspect.signature(getattr(controller, name)).parameters
        assert not set(parameters).intersection(forbidden)


def test_stall_detection_is_bounded_and_records_partial_phase(episode):
    class StationaryEnv:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            }, 0.0, False, {}

    env = StationaryEnv()
    with pytest.raises(TimeoutError, match="pregrasp stalled for 3 steps"):
        episode._run_motion(
            env,
            np.asarray([[0.3, 0.0, 0.3]] * 6),
            {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            },
            phase_timeout_steps=20,
            gripper_dwell_steps=5,
            stop_after_phase="retreat",
            dry_run=False,
            stall_window_steps=3,
            stall_delta_m=1e-8,
        )
    assert len(env.actions) == 3
    assert env._arrow_phase_audit[-1]["phase"] == "pregrasp"
    assert env._arrow_phase_audit[-1]["status"] == "stall"
    assert env._arrow_phase_audit[-1]["steps"] == 3


def test_recovery_attempts_and_steps_have_hard_upper_bounds(episode, monkeypatch, tmp_path: Path):
    class Env:
        _arrow_settle_diagnostics = {"settled": True}

        def __init__(self):
            self.actions = []
            self._arrow_phase_audit = []

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            }, 0.0, False, {}

    env = Env()
    calibration = episode.CameraCalibration(
        "agentview", 256, 256,
        [[10.0, 0.0, 128.0], [0.0, -10.0, 128.0], [0.0, 0.0, 1.0]],
        np.eye(4).tolist(),
    )
    capture = episode.CapturedRGBD(
        np.zeros((256, 256, 3), dtype=np.uint8),
        np.ones((256, 256), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        calibration,
        {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
    )

    class Arrow:
        source_xy = (80.0, 80.0)
        target_xy = (160.0, 160.0)

    monkeypatch.setattr(episode, "decode_arrow", lambda **kwargs: Arrow())
    monkeypatch.setattr(episode, "decode_arrow_diagnostics", None)
    monkeypatch.setattr(episode, "deproject_endpoint", lambda *args: np.asarray([0.0, 0.0, 0.5]))
    monkeypatch.setattr(episode, "build_bowl_waypoints", lambda *args: np.zeros((6, 3)))

    def stalled_motion(*args, **kwargs):
        args[0]._arrow_phase_audit[:] = [{"phase": "close", "status": "stall", "steps": 2}]
        raise TimeoutError("phase close stalled for 2 steps")

    monkeypatch.setattr(episode, "_run_motion", stalled_motion)
    recovery_calls = []

    def fake_recovery(*args, **kwargs):
        recovery_calls.append(kwargs["phase"])
        return {
            "phase": kwargs["phase"],
            "eef_pos_m": [0.0, 0.0, 0.0],
            "eef_quat": [0.0, 0.0, 0.0, 1.0],
            "gripper_qpos": [],
        }

    monkeypatch.setattr(episode, "recover_grasp_or_release", fake_recovery)
    variant = episode.ControllerVariantConfig(
        suite_mode="vanilla", stall_window_steps=2, recovery_attempts=3, recovery_steps=2
    )
    with pytest.raises(TimeoutError, match="phase close stalled"):
        episode.run_episode(
            env=env,
            task_id=0,
            seed=1000,
            output_dir=tmp_path,
            arrow_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            capture=capture,
            dry_run=False,
            controller_variant=variant,
            recovery_callback=lambda phase, proposal: False,
        )
    assert recovery_calls == ["close", "close", "close"]
    # A false approval must never send recovery actions.  The proposals are
    # still retained for diagnostics and the original timeout remains raised.
    assert env.actions == []


def test_dual_wrapper_runs_vanilla_then_randomized_sequentially_and_combines_results():
    wrapper = Path("vla_benchmarking/legion/run_arrow_pick_place_dual_matrix.sbatch")
    text = wrapper.read_text(encoding="utf-8")
    vanilla = text.index('run_suite vanilla "$VANILLA_ROOT" 0')
    randomized = text.index('run_suite sealed_randomized "$RANDOMIZED_ROOT" 1')
    assert vanilla < randomized
    assert 'export ARROW_SUITE_MODE="$suite" LIBERO_SUITE_MODE="$suite"' in text
    assert '"$PYTHON" "$MATRIX_LAUNCHER" "${matrix_args[@]}"' in text
    assert "--suite-mode \"$suite\"" in text
    assert "--controller-variant \"$CONTROLLER_VARIANT\"" in text
    assert "arrow_pick_place_dual_matrix_summary.json" in text
    assert "vanilla/arrow_pick_place_matrix_summary.json" in text
    assert "sealed_randomized/arrow_pick_place_matrix_summary.json" in text
    assert "planned_success_rate" in text and "evaluable_success_rate" in text
    assert '"planned_success_rate": successes / (completed + failed) if evaluated else None' in text
    assert "set -Eeuo pipefail" in text
    # Neither suite invocation is backgrounded; a single allocation owns the
    # two runs and the second starts only after the first returns.
    assert 'run_suite vanilla "$VANILLA_ROOT" 0 &' not in text
    assert 'run_suite sealed_randomized "$RANDOMIZED_ROOT" 1 &' not in text


@pytest.mark.parametrize(
    "wrapper_name",
    (
        "run_arrow_pick_place_matrix.sbatch",
        "run_arrow_pick_place_dual_matrix.sbatch",
    ),
)
def test_launcher_job_context_has_no_blank_duplicate_controller_config_fields(wrapper_name):
    """Resolved config fields must not be shadowed by an earlier blank write."""
    wrapper = Path("vla_benchmarking/legion") / wrapper_name
    text = wrapper.read_text(encoding="utf-8")
    context_start = text.index('cat > "$RUN_ROOT/job_context.env" <<EOF')
    context_end = text.index("\nEOF", context_start)
    initial_context = text[context_start:context_end]
    # The semantic fields are appended only after config validation.  A blank
    # assignment in the initial heredoc would make naive env-file readers see
    # the wrong value.
    assert "controller_config_hash=" not in initial_context
    assert "controller_config_canonical_path=" not in initial_context
    append_start = text.index("printf 'controller_config_hash=%s\\n'", context_end)
    append_end = text.index('} >> "$RUN_ROOT/job_context.env"', append_start)
    appended = text[append_start:append_end]
    assert appended.count("controller_config_hash=") == 1
    assert appended.count("controller_config_canonical_path=") == 1
