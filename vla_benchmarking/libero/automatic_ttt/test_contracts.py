from __future__ import annotations

import json

import pytest

from .contracts import (
    Actor,
    ContractError,
    EpisodeSpec,
    EpisodeStatus,
    SourceState,
    TeacherRecoveryResult,
    TransitionRecord,
)
from .episode import EpisodeCoordinator
from .recording import EpisodeManifest, ExperimentProvenance, JSONLTransitionWriter
from .teacher import TakeoverEnvironmentView


class FakeEnvironment:
    def __init__(self) -> None:
        self.step_calls = 0
        self.identity_at_step: list[int] = []

    def observe(self):
        return {
            "agentview": [[[self.step_calls, 0, 0]]],
            "wrist": [[[0, self.step_calls, 0]]],
            "state": [self.step_calls] + [0] * 7,
            "instruction": "pick up the object",
        }

    def step(self, action):
        self.identity_at_step.append(id(self))
        self.step_calls += 1
        # The second action is the teacher correction in this fixture; expose
        # the evaluator's terminal/success signal so the demonstration gate
        # validates the same evidence a real Arrow bridge must provide.
        return {"done": self.step_calls >= 2, "success": self.step_calls >= 2}

    def reset(self):  # pragma: no cover - should never be called
        raise AssertionError("reset was called")

    def close(self):  # pragma: no cover - should never be called
        raise AssertionError("close was called")


def _spec() -> EpisodeSpec:
    return EpisodeSpec("episode-0", 0, 1000, "pick up the object", "openvla", "train")


def _provenance() -> ExperimentProvenance:
    return ExperimentProvenance(
        "robottt-kvb", "arxiv:2607.15275", "unresolved", None, "openvla", None,
        "arrow_grasp_controller", "canonical", None, "libero-spatial", None, "working-tree",
        "algorithmic_port_blocked",
    )


def test_teacher_view_blocks_reset_and_close():
    env = FakeEnvironment()
    view = TakeoverEnvironmentView(env)
    with pytest.raises(ContractError):
        view.reset
    with pytest.raises(ContractError):
        view.close
    view.step([0.0] * 7)
    assert view.environment_identity == id(env)
    assert env.identity_at_step == [id(env)]


def test_coordinator_preserves_env_and_records_teacher_correction(tmp_path):
    env = FakeEnvironment()
    spec = _spec()
    manifest = EpisodeManifest(spec.episode_id, spec.task_id, spec.seed, spec.split, spec.task_description,
                               spec.policy_id, _provenance(), ("state", "rgb"))
    writer = JSONLTransitionWriter(tmp_path / "episode.jsonl", manifest)

    class Teacher:
        teacher_id = "arrow_grasp_controller"

        def recover(self, view, request):
            assert view.environment_identity == id(env)
            before = view.observe()
            view.step([0.0] * 7)
            after = view.observe()
            correction = TransitionRecord(
                request.episode.episode_id, len(request.vla_history), Actor.TEACHER,
                before, (0.0,) * 7, after, True, True, True, request.source_state,
            )
            return TeacherRecoveryResult((correction,), True, EpisodeStatus.TEACHER_SUCCESS)

    coordinator = EpisodeCoordinator(env, spec, writer=writer, observation_schema="libero_rgb_state8_instruction_v1")
    result = coordinator.run_vla_then_teacher(
        lambda _observation, _step: [0.0] * 7,
        Teacher(),
        success_fn=lambda _env, _result: False,
        terminal_fn=lambda _env, _result: False,
        source_state_fn=lambda _env, _observation: SourceState.SOURCE_UNHELD,
        vla_step_budget=1,
        teacher_step_budget=4,
    )
    assert result.status is EpisodeStatus.TEACHER_SUCCESS
    assert result.vla_steps == 1 and result.teacher_steps == 1
    assert env.step_calls == 2
    assert writer.path.exists()
    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["training_eligible"] is True


def test_privileged_observation_is_rejected():
    with pytest.raises(ContractError, match="privileged"):
        TransitionRecord("e", 0, Actor.VLA, {"rgb": [], "simulator_bbox": []}, (0.0,) * 7, {}, False, False, False)


def test_teacher_result_rejects_non_boolean_success():
    with pytest.raises(ContractError, match="success must be a boolean"):
        TeacherRecoveryResult((), "false", EpisodeStatus.TEACHER_FAILED)


def test_unsafe_source_does_not_invoke_teacher():
    env = FakeEnvironment()
    called = False

    class Teacher:
        teacher_id = "arrow_grasp_controller"

        def recover(self, _view, _request):
            nonlocal called
            called = True
            raise AssertionError("unsafe state was handed to teacher")

    result = EpisodeCoordinator(env, _spec()).run_vla_then_teacher(
        lambda _observation, _step: [0.0] * 7,
        Teacher(),
        success_fn=lambda _env, _result: False,
        terminal_fn=lambda _env, _result: False,
        source_state_fn=lambda _env, _observation: SourceState.UNSAFE,
        vla_step_budget=1,
        teacher_step_budget=4,
    )
    assert result.status is EpisodeStatus.ABORTED
    assert not called


def test_writer_preserves_incomplete_partial_by_default(tmp_path):
    target = tmp_path / "episode.jsonl"
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text("incomplete", encoding="utf-8")
    spec = _spec()
    manifest = EpisodeManifest(spec.episode_id, spec.task_id, spec.seed, spec.split, spec.task_description,
                               spec.policy_id, _provenance(), ("state",))
    with pytest.raises(FileExistsError):
        JSONLTransitionWriter(target, manifest)
    assert partial.read_text(encoding="utf-8") == "incomplete"
