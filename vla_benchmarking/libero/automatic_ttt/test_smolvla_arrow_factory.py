from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import live_collection
from . import smolvla_arrow_factory as runtime
from .live_collection import CanonicalLiveEnvironment
from .teacher import ArrowGraspControllerTeacher
from .teacher import TakeoverEnvironmentView
from .contracts import Actor, EpisodeSpec, SourceState, TeacherRecoveryRequest, TransitionRecord


class _RawEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.close_count = 0

    @staticmethod
    def observation(step: int) -> dict[str, object]:
        image = np.full((256, 256, 3), step, dtype=np.uint8)
        # An open gripper is classified as source_unheld; the Arrow teacher
        # must still take over the same object after the VLA step.
        return {
            "image": image,
            "image_wrist": image.copy(),
            "state": np.asarray((0, 0, 0, 0, 0, 0, 0.02, 0.02), dtype=np.float32),
        }

    def reset(self, *, seed: int, task_id: int, episode_index: int) -> dict[str, object]:
        assert (seed, task_id, episode_index) == (3000, 0, 0)
        self.reset_count += 1
        return self.observation(0)

    def step(self, _action: object):
        self.step_count += 1
        success = self.step_count == 2
        return self.observation(self.step_count), 0.0, False, {"success": success}

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.parametrize("value, expected", [(False, False), (True, True), (np.bool_(False), False), (np.bool_(True), True)])
def test_direct_environment_normalizes_boolean_success_scalars(value, expected) -> None:
    raw = SimpleNamespace(check_success=lambda: value)
    environment = runtime._DirectLiberoEnvironment(
        raw,
        episode=EpisodeSpec("episode", 9, 3000, "task", "smolvla"),
    )

    result = environment.check_success()

    assert result is expected
    assert type(result) is bool


def test_direct_environment_rejects_integer_success_scalar() -> None:
    raw = SimpleNamespace(check_success=lambda: 0)
    environment = runtime._DirectLiberoEnvironment(
        raw,
        episode=EpisodeSpec("episode", 9, 3000, "task", "smolvla"),
    )

    with pytest.raises(RuntimeError, match=r"unsupported value.*builtins\.int"):
        environment.check_success()


def test_arrow_session_normalizes_numpy_boolean_mapping(tmp_path) -> None:
    session = runtime._ArrowSession(molmo=object(), resolution=8, output_root=tmp_path)
    environment = SimpleNamespace(check_success=lambda: {"task_success": np.bool_(True)})

    result = session.evaluate(environment)

    assert result is True
    assert type(result) is bool


def test_factory_preserves_live_identity_and_blocks_takeover_reset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw = _RawEnvironment()
    identities: dict[str, object] = {}
    monkeypatch.setattr(
        live_collection,
        "canonical_student_observation",
        lambda value, *, instruction: {
            "agentview": value["image"], "wrist": value["image_wrist"],
            "state": value["state"], "instruction": instruction,
        },
    )

    def make(_episode):
        identities["raw"] = raw
        return raw

    def reset(environment, episode):
        return environment.reset(seed=episode.seed, task_id=episode.task_id, episode_index=0)

    def close(environment):
        environment.close()

    monkeypatch.setattr(runtime, "_build_live_environment_factory", lambda **_: (make, reset, close))
    monkeypatch.setattr(runtime, "_build_smolvla_action", lambda *_args, **_kwargs: lambda _obs, _step: (0.0,) * 7)

    def teacher_factory(_episode, _output):
        def recover(view, _request):
            identities["takeover_raw"] = view._environment.raw_environment
            with pytest.raises(Exception):
                view.reset()
            view.step((0.0,) * 7)
            return {
                "transitions": list(view.executed_transitions),
                "success": True,
                "metadata": {"evaluator_success": True, "evaluator_phase": "post_retreat", "evaluator_receipt_id": "test-receipt"},
            }

        return ArrowGraspControllerTeacher(recover)

    def prepare(environment):
        identities["prepared"] = environment

    teacher_factory.prepare_for_environment = prepare  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime, "_build_arrow_teacher_factory", lambda **_: (teacher_factory, type("S", (), {"provenance": {}})()))

    def fake_collect(**kwargs):
        assert kwargs["vla_step_budget"] == 1
        assert kwargs["teacher_step_budget"] == 5
        episode = EpisodeSpec("test-episode", 0, 3000, "pick up the bowl", "smolvla")
        environment = kwargs["environment_factory"](episode)
        reset_observation = kwargs["reset_environment"](environment, episode)
        live = CanonicalLiveEnvironment(environment, initial_observation=reset_observation, instruction=episode.task_description)
        # This is the exact boundary used by EpisodeCoordinator: the student
        # gets the live facade, and Arrow receives a takeover view over it.
        before = live.observe()
        vla_action = kwargs["vla_action"](before, 0)
        live.step(vla_action)
        after = live.observe()
        vla_record = TransitionRecord(
            episode_id=episode.episode_id, timestep=0, actor=Actor.VLA,
            observation=before, action=tuple(vla_action), next_observation=after,
            done=False, success=False, training_eligible=False,
        )
        teacher = kwargs["teacher_factory"](episode, tmp_path / "arrow")
        view = TakeoverEnvironmentView(live, expected_identity=id(live), episode_id=episode.episode_id,
                                       start_timestep=1, source_state=SourceState.SOURCE_UNHELD)
        assert view._environment.raw_environment is environment
        with pytest.raises(Exception):
            view.reset()
        teacher.recover(
            view,
            TeacherRecoveryRequest(
                episode=episode, source_state=SourceState.SOURCE_UNHELD,
                observation=live.observe(), vla_history=(vla_record,), remaining_budget=5,
            ),
        )
        kwargs["close_environment"](environment)
        return SimpleNamespace(
            task_id=0, accepted_count=1, attempted_count=1,
            accepted_path=tmp_path / "accepted_episodes.jsonl",
            failed_path=None,
            manifest_path=tmp_path / "collection_manifest.json",
            manifest_sha256="a" * 64,
        )

    monkeypatch.setattr(runtime, "collect_task_corrections", fake_collect)

    factory = runtime.build_collection_factory(base_policy="unused", vla_step_budget=1, arrow_step_budget=5)
    result = factory(
        task_id=0,
        task_description="pick up the bowl",
        output_root=tmp_path,
        accepted_target=1,
        adaptation_seed_start=3000,
        controller_config_hash="a" * 64,
    )

    assert result["status"] == "COLLECTION_COMPLETE"
    assert result["failed_path"] is None
    assert identities["prepared"] is raw
    assert identities["takeover_raw"] is raw
    assert raw.reset_count == 1
    assert raw.step_count == 2
    assert raw.close_count == 1


def test_fresh_factory_never_loads_or_calls_smolvla(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("SmolVLA must not be constructed for fresh Arrow collection")

    monkeypatch.setattr(runtime, "_build_smolvla_action", forbidden)
    monkeypatch.setattr(
        runtime,
        "_build_live_environment_factory",
        lambda **_: (lambda _episode: object(), lambda _env, _episode: {}, lambda _env: None),
    )
    teacher_factory = lambda _episode, _output: object()
    teacher_factory.prepare_for_environment = lambda _env: None  # type: ignore[attr-defined]
    session = SimpleNamespace(provenance={})
    monkeypatch.setattr(
        runtime,
        "_build_arrow_teacher_factory",
        lambda **_: (teacher_factory, session),
    )

    def fake_fresh(**kwargs):
        assert kwargs["policy_id"] == "smolvla"
        assert kwargs["accepted_target"] == 1
        return SimpleNamespace(
            task_id=0, accepted_count=1, attempted_count=1,
            accepted_path=tmp_path / "accepted.jsonl", failed_path=None,
            manifest_path=tmp_path / "collection.json", manifest_sha256="a" * 64,
        )

    monkeypatch.setattr(runtime, "collect_fresh_arrow_demonstrations", fake_fresh)
    factory = runtime.build_collection_factory(
        collection_mode="fresh_arrow", arrow_step_budget=1200
    )
    result = factory(
        task_id=0, task_description="pick up the bowl", output_root=tmp_path,
        accepted_target=1, adaptation_seed_start=3000, max_attempts=2,
        controller_config_hash="a" * 64,
    )
    assert result["status"] == "COLLECTION_COMPLETE"
    assert result["failed_path"] is None


def test_source_classifier_fails_closed_on_missing_or_malformed_state() -> None:
    with pytest.raises(Exception, match="canonical finite 8-D state"):
        runtime.classify_source_state(object(), {})
    with pytest.raises(Exception, match="canonical finite 8-D state"):
        runtime.classify_source_state(object(), {"state": [0.0] * 7})


def test_source_classifier_discards_closed_or_ambiguous_gripper() -> None:
    def state(left: float, right: float) -> dict[str, object]:
        return {"state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, left, right]}

    assert runtime.classify_source_state(object(), state(0.02, 0.02)) is SourceState.SOURCE_UNHELD
    assert runtime.classify_source_state(object(), state(0.0, 0.0)) is SourceState.UNSAFE
    assert runtime.classify_source_state(object(), state(0.015, 0.02)) is SourceState.UNSAFE


def test_arrow_session_forwards_current_rendered_arrow_to_motion_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode

    arrow_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    session = runtime._ArrowSession(
        molmo=object(), resolution=8, output_root=tmp_path,
        transform=np.eye(3), opening_m=0.04, task_id=0, seed=3000,
        pending_arrow=(arrow_rgb, (1.0, 2.0), (3.0, 4.0)),
    )
    rendered, source, destination = session.refresh_arrow(object())
    assert rendered is arrow_rgb
    assert source == (1.0, 2.0)
    assert destination == (3.0, 4.0)

    seen = {}

    def fake_run_episode(**kwargs):
        seen.update(kwargs)
        return {"status": "task_failure"}

    monkeypatch.setattr(episode, "run_episode", fake_run_episode)
    session.episode_runner(
        env=object(),
        context=SimpleNamespace(
            output_dir=tmp_path, candidate=object(), agentview_capture=object()
        ),
        evaluator=None,
    )
    assert seen["arrow_rgb"] is arrow_rgb


def test_arrow_session_normalizes_exhausted_grasp_search_as_failed_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode

    session = runtime._ArrowSession(
        molmo=object(), resolution=8, output_root=tmp_path,
        transform=np.eye(3), opening_m=0.04, task_id=9, seed=3040,
        current_arrow_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    def exhausted_search(**_kwargs):
        raise episode._GraspSearchRequested("empty_gripper_likely")

    monkeypatch.setattr(episode, "run_episode", exhausted_search)

    result = session.episode_runner(
        env=object(),
        context=SimpleNamespace(
            output_dir=tmp_path, candidate=object(), agentview_capture=object()
        ),
        evaluator=None,
    )

    assert result == {
        "status": "empty_gripper_likely",
        "controller_failure": "empty_gripper_likely",
        "grasp_retained": False,
        "evaluator_called": False,
        "evaluator_success": False,
        "retreat_complete": False,
    }


def test_arrow_session_cleanup_releases_all_attempt_references(tmp_path):
    """A completed attempt cannot retain env/images/calibration into retry 2."""
    worker = SimpleNamespace(robot_calibration=object())
    session = runtime._ArrowSession(
        molmo=object(), resolution=8, output_root=tmp_path,
        worker=worker, transform=np.eye(3), opening_m=0.04,
        pending_capture=object(), pending_arrow=(np.zeros((8, 8, 3)), (1, 2), None),
        current_arrow_rgb=np.ones((8, 8, 3)), current_view=object(), task_id=0, seed=3000,
        action_budget=object(),
    )

    session.cleanup_attempt()

    assert session.worker is None
    assert worker.robot_calibration is None
    assert session.transform is None
    assert session.opening_m is None
    assert session.pending_capture is None
    assert session.pending_arrow is None
    assert session.current_arrow_rgb is None
    assert session.current_view is None
    assert session.action_budget is None
    assert session.task_id is None
    assert session.seed is None
    # Exception/retry cleanup is deliberately idempotent.
    session.cleanup_attempt()
