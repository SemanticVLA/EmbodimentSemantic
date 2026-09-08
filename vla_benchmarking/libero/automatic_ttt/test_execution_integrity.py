from __future__ import annotations

import numpy as np
import pytest

from . import live_collection
from .contracts import ContractError, EpisodeSpec, EpisodeStatus, SourceState, TeacherRecoveryResult
from .episode import EpisodeCoordinator
from .live_collection import CanonicalLiveEnvironment
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


def test_live_view_normalizes_numpy_boolean_step_fields():
    class NumpyBoolEnv(Env):
        def step(self, _action):
            self.t += 1
            return self.observe(), 0.0, np.asarray(False), {"success": np.asarray(True)}

    view = TakeoverEnvironmentView(
        NumpyBoolEnv(), episode_id="episode", start_timestep=0,
        source_state=SourceState.SOURCE_UNHELD,
    )
    view.step([0.0] * 7)
    assert view.executed_transitions[0].done is False
    assert view.executed_transitions[0].success is True


def test_privileged_controller_gets_raw_proprioception_but_records_only_canonical_observations(monkeypatch):
    class RawEnv:
        def __init__(self):
            self.steps = 0

        def step(self, _action):
            self.steps += 1
            return {
                "image": f"image-{self.steps}",
                "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
                "secret_object_state": np.asarray([9.0]),
            }, 0.0, False, {"success": False}

    monkeypatch.setattr(
        live_collection,
        "canonical_student_observation",
        lambda value, *, instruction: {
            "agentview": value["image"],
            "wrist": value["image"],
            "state": [0.0] * 8,
            "instruction": instruction,
        },
    )
    raw = RawEnv()
    live = CanonicalLiveEnvironment(
        raw,
        initial_observation={"image": "image-0"},
        instruction="pick up the object",
    )
    view = PrivilegedTakeoverEnvironmentView(
        live,
        episode_id="episode",
        source_state=SourceState.SOURCE_UNHELD,
    )

    returned_observation, *_ = view.step([0.0] * 7)

    assert raw.steps == 1
    assert "robot0_eef_pos" in returned_observation
    assert "secret_object_state" in returned_observation
    transition = view.executed_transitions[0]
    assert set(transition.observation) == {"agentview", "wrist", "state", "instruction"}
    assert set(transition.next_observation) == {"agentview", "wrist", "state", "instruction"}


def test_privileged_controller_defers_libero_done_until_retreat_trace_finishes():
    class SuccessPredicateEnv(Env):
        def step(self, _action):
            self.t += 1
            return self.observe(), 0.0, self.t == 1, {"success": self.t == 1}

    view = PrivilegedTakeoverEnvironmentView(
        SuccessPredicateEnv(), episode_id="episode", source_state=SourceState.SOURCE_UNHELD,
    )
    view.step([0.0] * 7)
    view.step([0.0] * 7)
    assert [row.done for row in view.executed_transitions] == [False, False]
    assert [row.success for row in view.executed_transitions] == [True, False]

    view.finalize_controller_trace()

    assert [row.done for row in view.executed_transitions] == [False, True]
    with pytest.raises(ContractError, match="cannot step after"):
        view.step([0.0] * 7)
