"""Focused lifecycle tests for the shared per-run Molmo runtime."""

from __future__ import annotations

import pytest

from vla_benchmarking.robocasa.evaluation import runner


def _identity() -> dict[str, object]:
    return {
        "schema": "test",
        "canonical_controller": {"name": "test", "config_hash": "test"},
        "robocasa_adapter": {},
        "errors": [],
    }


def _patch_two_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "selected_tasks", lambda _names=None: runner.PICK_PLACE_TASKS[:2])
    monkeypatch.setattr(runner, "_robocasa_controller_identity", _identity)
    monkeypatch.setattr(runner, "_experiment_identity", lambda **_kwargs: "runtime-test")
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: object())


def test_motion_run_builds_once_reuses_runtime_and_closes_once(monkeypatch, tmp_path) -> None:
    _patch_two_tasks(monkeypatch)
    runtimes: list[object] = []
    closed: list[object] = []

    class Runtime:
        def close(self) -> None:
            closed.append(self)

    def build() -> Runtime:
        runtime = Runtime()
        runtimes.append(runtime)
        return runtime

    seen: list[object] = []
    monkeypatch.setattr(
        "vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner.build_local_molmo_runtime",
        build,
    )

    def cell(**kwargs):
        seen.append(kwargs["molmo_runtime"])
        task = kwargs["task"]
        return runner.TerminalRow(
            task=task.name, episode_index=kwargs["episode_index"], seed=kwargs["seed"],
            split="target", mode=kwargs["mode"], terminal=True, success=False,
            failure_category="task_failure", metadata={"experiment_identity": "runtime-test"},
        )

    monkeypatch.setattr(runner, "_run_live_cell", cell)
    assert runner.run(output_dir=tmp_path, mode="smoke", execute_motion=True) == 0
    assert len(runtimes) == 1
    assert seen == [runtimes[0], runtimes[0]]
    assert closed == [runtimes[0]]


def test_runtime_closes_once_when_cell_interrupts(monkeypatch, tmp_path) -> None:
    _patch_two_tasks(monkeypatch)
    built: list[object] = []
    closed: list[object] = []

    class Runtime:
        def close(self) -> None:
            closed.append(self)

    monkeypatch.setattr(
        "vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner.build_local_molmo_runtime",
        lambda: built.append(Runtime()) or built[-1],
    )
    calls = 0

    def interrupting_cell(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("test interruption")
        return runner.TerminalRow(
            task="first", episode_index=0, seed=1000, split="target", mode="smoke",
            terminal=True, success=False, failure_category="task_failure",
            metadata={"experiment_identity": "runtime-test"},
        )

    monkeypatch.setattr(runner, "_run_live_cell", interrupting_cell)
    with pytest.raises(KeyboardInterrupt):
        runner.run(output_dir=tmp_path, mode="smoke", execute_motion=True)
    assert len(built) == 1
    assert closed == built


def test_preflight_and_no_motion_do_not_build_runtime(monkeypatch, tmp_path) -> None:
    _patch_two_tasks(monkeypatch)
    builds: list[object] = []
    monkeypatch.setattr(
        "vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner.build_local_molmo_runtime",
        lambda: builds.append(object()),
    )
    def preflight_cell(**kwargs):
        assert kwargs["molmo_runtime"] is None
        return runner.TerminalRow(
            task=kwargs["task"].name, episode_index=kwargs["episode_index"], seed=kwargs["seed"],
            split="target", mode=kwargs["mode"], terminal=True, success=False,
            failure_category=None, metadata={"experiment_identity": "runtime-test"},
        )

    monkeypatch.setattr(runner, "_run_live_cell", preflight_cell)
    assert runner.run(output_dir=tmp_path / "preflight", mode="preflight") == 0
    assert runner.run(output_dir=tmp_path / "no-motion", mode="smoke", execute_motion=False) == 0
    assert builds == []


def test_all_skipped_resume_does_not_build_or_close_runtime(monkeypatch, tmp_path) -> None:
    _patch_two_tasks(monkeypatch)
    runtime = object()
    builds: list[object] = []
    monkeypatch.setattr(
        "vla_benchmarking.robocasa.arrow_grasp_controller.controller.runner.build_local_molmo_runtime",
        lambda: builds.append(runtime) or runtime,
    )

    def completed_cell(**kwargs):
        return runner.TerminalRow(
            task=kwargs["task"].name, episode_index=kwargs["episode_index"], seed=kwargs["seed"],
            split="target", mode=kwargs["mode"], terminal=True, success=False,
            failure_category="task_failure", metadata={"experiment_identity": "runtime-test"},
        )

    monkeypatch.setattr(runner, "_run_live_cell", completed_cell)
    assert runner.run(output_dir=tmp_path, mode="smoke", execute_motion=True) == 0
    builds.clear()
    monkeypatch.setattr(runner, "_run_live_cell", lambda **_kwargs: pytest.fail("all cells should be skipped"))
    assert runner.run(output_dir=tmp_path, mode="smoke", execute_motion=True) == 0
    assert builds == []
