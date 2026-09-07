from __future__ import annotations

import pytest

from .contracts import ContractError, EpisodeSpec, EpisodeStatus, SourceState, TeacherRecoveryResult
from .episode import EpisodeCoordinator
from .teacher import PrivilegedTakeoverEnvironmentView, TakeoverEnvironmentView


class Env:
    def __init__(self):
        self.t = 0

    def observe(self):
        return {"state": [self.t]}

    def step(self, _action):
        self.t += 1
        return {"done": False}


def _spec():
    return EpisodeSpec("integrity-0", 0, 1, "task", "policy")


def _run(teacher):
    return EpisodeCoordinator(Env(), _spec()).run_vla_then_teacher(
        lambda _observation, _step: [0.0] * 7,
        teacher,
        success_fn=lambda _env, _result: False,
        terminal_fn=lambda _env, _result: False,
        source_state_fn=lambda _env, _observation: SourceState.SOURCE_UNHELD,
        vla_step_budget=1,
        teacher_step_budget=3,
    )


def test_omitted_teacher_rows_are_rejected():
    class Teacher:
        teacher_id = "arrow_grasp_controller"

        def recover(self, view, _request):
            view.step([0.0] * 7)
            return TeacherRecoveryResult((), False, EpisodeStatus.TEACHER_FAILED)

    with pytest.raises(ContractError, match="returned 0 transitions"):
        _run(Teacher())


def test_fabricated_teacher_action_is_rejected():
    class Teacher:
        teacher_id = "arrow_grasp_controller"

        def recover(self, view, request):
            before = view.observe()
            view.step([0.0] * 7)
            after = view.observe()
            from .contracts import Actor, TransitionRecord
            fabricated = TransitionRecord(
                request.episode.episode_id, len(request.vla_history), Actor.TEACHER,
                before, (1.0,) * 7, after, False, False, True, request.source_state,
            )
            return TeacherRecoveryResult((fabricated,), False, EpisodeStatus.TEACHER_FAILED)

    with pytest.raises(ContractError, match="do not match"):
        _run(Teacher())


def test_ordinary_teacher_steps_are_independent_one_step_chunks():
    env = Env()
    view = TakeoverEnvironmentView(
        env, episode_id="episode", start_timestep=0, source_state=SourceState.SOURCE_UNHELD,
    )
    view.step([0.0] * 7)
    view.step([0.0] * 7)
    assert [row.action_chunk_index for row in view.executed_transitions] == [0, 0]
    assert [row.action_chunk_horizon for row in view.executed_transitions] == [1, 1]


def test_privileged_view_blocks_state_lifecycle_mutators():
    env = Env()
    env.set_init_state = lambda: None
    env.load_state = lambda: None
    env.restore_state = lambda: None
    view = PrivilegedTakeoverEnvironmentView(env)
    for name in ("set_init_state", "load_state", "restore_state"):
        with pytest.raises(ContractError, match="not allowed"):
            getattr(view, name)
