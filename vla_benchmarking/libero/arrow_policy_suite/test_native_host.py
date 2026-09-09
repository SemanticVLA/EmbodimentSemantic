from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arrow_policy_suite.contracts import ActionProposal, ObservationFrame, digest
from arrow_policy_suite.interruptible_arrow import ArrowPerceptionUnavailable, InterruptibleArrow
from arrow_policy_suite.libero_adapter import LiberoEnvironmentAdapter
from arrow_policy_suite.native_host import NativeHost, _equal, _proposal_equal, _proposal_mismatch
from arrow_policy_suite.libero_state import OffScreenRenderSnapshot
from arrow_policy_suite.native_arrow_teacher import PerFrameArrowTeacher
from arrow_policy_suite.policies import OnCallPolicy
from arrow_policy_suite.smolvla_adapter import SmolVLAAdapter
from arrow_policy_suite.splits import ResetIdentity


class Env:
    def __init__(self):
        self.value = 0.0
        self.steps = 0

    def observe(self):
        return {"state": [self.value] + [0.0] * 7}

    def step(self, action):
        self.value += float(action[0])
        self.steps += 1
        return {"terminal": False, "success": False}

    def snapshot_state(self):
        return self.value, self.steps

    def restore_state(self, state):
        self.value, self.steps = state


class StatefulArrow:
    def __init__(self):
        self.calls = 0
        self.commits = 0

    def propose(self, frame, _context=None):
        self.calls += 1
        return {"action": (0.4,) + (0.0,) * 6}

    def commit(self, _record):
        self.commits += 1

    def snapshot_state(self):
        return self.calls, self.commits

    def restore_state(self, state):
        self.calls, self.commits = state


def _vla(calls):
    def inference(_observation, _step):
        calls.append(1)
        return [[0.1] + [0.0] * 6, [0.2] + [0.0] * 6]

    inference.__arrow_stateless__ = True
    return SmolVLAAdapter(inference=inference)


def test_native_host_same_frame_one_step_owner_and_queue_invalidation():
    env = Env()
    calls = []
    vla = _vla(calls)
    arrow = InterruptibleArrow(StatefulArrow(), perception=lambda frame: frame.metadata["graph_context"])
    host = NativeHost(
        env, vla, arrow,
        graph_context_fn=lambda _frame: {"triplet": "hand-source-destination"},
        action_selector=lambda _base, teacher: teacher.action,
    )
    records = host.run(max_steps=3)
    assert len(records) == 3
    assert env.steps == 3
    assert all(record.frame is record.frame for record in records)
    assert all(record.executed_by == "arrow" for record in records)
    assert all(record.teacher_status.available for record in records)
    # External actions invalidate the two-row VLA chunk, forcing inference on
    # every new frame rather than applying an action for a stale state.
    assert len(calls) == 3
    assert records[0].frame.metadata["graph_context"]["triplet"] == "hand-source-destination"


def test_native_host_none_teacher_is_not_a_fault():
    env = Env()

    class NoArrow:
        def propose(self, _frame):
            return None

        def snapshot_state(self):
            return 0

        def restore_state(self, _state):
            return None

    vla = _vla([])
    host = NativeHost(env, vla, NoArrow())
    record = host.step()
    assert record.teacher is None
    assert not record.teacher_status.available
    assert record.teacher_status.reason == "perception_unavailable"
    assert env.steps == 1


def test_native_host_invalidates_interruptible_teacher_when_base_wins_two_steps():
    env = Env()
    vla = _vla([])
    teacher = PerFrameArrowTeacher(
        lambda _frame: {"waypoints": [[0.01, 0.0, 0.0]] * 6},
        gripper_dwell_steps=1,
    )
    # Together/base arbitration intentionally ignores a valid teacher action;
    # the host must interrupt that pending proposal before timestep two.
    host = NativeHost(env, vla, teacher, action_selector=lambda base, _teacher: base.action)
    records = host.run(max_steps=2)
    assert len(records) == 2
    assert env.steps == 2
    assert all(record.executed_by == "vla" for record in records)


def test_on_call_intervenes_from_per_frame_phase_error_without_success_labels():
    """A stalled Arrow phase must cause a real takeover in one rollout."""

    class FlatEnv(Env):
        def step(self, _action):
            self.steps += 1
            # Keep the EEF fixed so the teacher's frame-derived phase error is
            # deliberately stalled.  No evaluator success signal is supplied.
            return {"terminal": False}

    class FlatVLA:
        policy_id = "vla"

        def propose(self, frame):
            return ActionProposal((0.0,) * 7, "vla", frame.timestep, observation_digest=frame.digest)

        def snapshot_state(self):
            return 0

        def restore_state(self, _state):
            return None

    env = FlatEnv()
    teacher = PerFrameArrowTeacher(
        lambda _frame: {
            "candidate_id": "stalling-candidate",
            "waypoints": [[0.20, 0.0, 0.0]] * 6,
            "provenance": {"source": "synthetic-rgbd"},
        },
        gripper_dwell_steps=1,
    )
    policy = OnCallPolicy()

    def select(base, arrow, frame):
        decision = policy.decide(frame, base, arrow)
        return decision.action

    host = NativeHost(env, FlatVLA(), teacher, policy=policy, action_selector=select)
    records = host.run(max_steps=45)
    assert len(records) == 45
    assert all(record.teacher is not None for record in records)
    assert all(isinstance(record.teacher.metadata.get("phase_error"), float) for record in records)
    executed = [record.executed_by for record in records]
    assert executed[:20] == ["vla"] * 20
    assert executed[20] == "arrow"
    assert any(value == "arrow" for value in executed[20:])


def test_snapshot_structural_equality_handles_mujoco_arrays():
    left = OffScreenRenderSnapshot("get_state", {"qpos": np.array([1.0, 2.0])},
                                   {}, {}, {}, {}, "digest")
    right = OffScreenRenderSnapshot("get_state", {"qpos": np.array([1.0, 2.0])},
                                    {}, {}, {}, {}, "digest")
    changed = OffScreenRenderSnapshot("get_state", {"qpos": np.array([1.0, 3.0])},
                                      {}, {}, {}, {}, "digest")
    assert _equal(left, right)
    assert not _equal(left, changed)


def test_proposal_equality_ignores_only_rng_but_full_equality_does_not():
    left = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 2.0])}, {}, {}, {}, {"draw": 1}, "digest"
    )
    rng_changed = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 2.0])}, {}, {}, {}, {"draw": 2}, "digest"
    )
    sim_changed = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 3.0])}, {}, {}, {}, {"draw": 2}, "digest"
    )

    # Proposal purity ignores the explicitly rollback-only RNG field, while
    # the complete comparator still treats it as a snapshot difference.
    assert _proposal_equal(left, rng_changed)
    assert _proposal_mismatch(left, rng_changed) is None
    assert not _equal(left, rng_changed)
    assert not _proposal_equal(left, sim_changed)


def test_proposal_mismatch_reports_exact_sim_and_wrapper_paths_without_values():
    left = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 2.0])}, {"steps": 1}, {}, {}, {}, "digest"
    )
    sim_changed = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 3.0])}, {"steps": 1}, {}, {}, {}, "digest"
    )
    wrapper_changed = OffScreenRenderSnapshot(
        "get_state", {"qpos": np.array([1.0, 2.0])}, {"steps": 2}, {}, {}, {}, "digest"
    )

    sim_mismatch = _proposal_mismatch(left, sim_changed, path="environment")
    wrapper_mismatch = _proposal_mismatch(left, wrapper_changed, path="environment")
    assert sim_mismatch is not None
    assert sim_mismatch[0] == "environment.sim_state.qpos"
    assert sim_mismatch[1:] == ("numpy.ndarray", "numpy.ndarray")
    assert wrapper_mismatch is not None
    assert wrapper_mismatch[0] == "environment.wrapper_fields.steps"
    assert wrapper_mismatch[1:] == ("int", "int")


def test_render_cache_projection_detects_state_mutation_but_ignores_image_redraw():
    left = OffScreenRenderSnapshot(
        "get_state", {}, {}, {}, {}, {}, "digest",
        rendered_wrapper_fields={
            "_last_obs": {
                "state": np.array([0.0, 1.0]),
                "agentview": np.array([1, 2], dtype=np.uint8),
            },
        },
        proposal_wrapper_fields={"_last_obs": {"state": np.array([0.0, 1.0])}},
    )
    image_redrawn = OffScreenRenderSnapshot(
        "get_state", {}, {}, {}, {}, {}, "digest",
        rendered_wrapper_fields={
            "_last_obs": {
                "state": np.array([0.0, 1.0]),
                "agentview": np.array([9, 8], dtype=np.uint8),
            },
        },
        proposal_wrapper_fields={"_last_obs": {"state": np.array([0.0, 1.0])}},
    )
    state_mutated = OffScreenRenderSnapshot(
        "get_state", {}, {}, {}, {}, {}, "digest",
        rendered_wrapper_fields=image_redrawn.rendered_wrapper_fields,
        proposal_wrapper_fields={"_last_obs": {"state": np.array([0.0, 2.0])}},
    )
    assert _proposal_equal(left, image_redrawn)
    assert not _proposal_equal(left, state_mutated)
    mismatch = _proposal_mismatch(left, state_mutated)
    assert mismatch is not None
    assert mismatch[0] == "$.proposal_wrapper_fields._last_obs.state"


def test_proposal_mismatch_honors_purity_excluded_dataclass_field():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Snapshot:
        mutable: tuple[int, ...]
        rng: tuple[int, ...] = field(metadata={"proposal_purity": False})

    left = Snapshot((1,), (2,))
    right = Snapshot((1,), (3,))
    assert _proposal_equal(left, right)
    assert _proposal_mismatch(left, right) is None


def test_pinned_mj_sim_state_compares_value_fields_not_object_identity():
    class MjSimState:
        __module__ = "robosuite.utils.binding_utils"

        def __init__(self, time, qpos, qvel):
            self.time = time
            self.qpos = qpos
            self.qvel = qvel

    left_state = MjSimState(0.25, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    equal_state = MjSimState(0.25, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    changed_state = MjSimState(0.25, np.array([1.0, 3.0]), np.array([0.1, 0.2]))

    assert left_state is not equal_state
    assert _equal(left_state, equal_state)
    assert not _equal(left_state, changed_state)
    assert _proposal_equal(left_state, equal_state)
    assert not _proposal_equal(left_state, changed_state)
    mismatch = _proposal_mismatch(left_state, changed_state, path="environment.payload.sim_state")
    assert mismatch == (
        "environment.payload.sim_state.qpos",
        "numpy.ndarray",
        "numpy.ndarray",
    )


def test_native_host_accepts_rng_only_environment_snapshot_change_during_proposal():
    import random

    class RngSnapshotEnv(Env):
        def snapshot_state(self):
            return OffScreenRenderSnapshot(
                "get_state",
                {"value": self.value},
                {"steps": self.steps},
                {},
                {},
                {"python": random.getstate()},
                "digest",
            )

        def restore_state(self, state):
            self.value = float(state.sim_state["value"])
            self.steps = int(state.wrapper_fields["steps"])

    class RngConsumingVLA:
        def propose(self, frame):
            random.random()
            return ActionProposal(
                (0.0,) * 7, "vla", frame.timestep, observation_digest=frame.digest
            )

        def snapshot_state(self):
            return 0

        def restore_state(self, _state):
            return None

    state = random.getstate()
    try:
        env = RngSnapshotEnv()
        host = NativeHost(env, RngConsumingVLA())
        record = host.step()
        assert record.proposal_state_unchanged
        assert env.steps == 1
    finally:
        random.setstate(state)


def test_interruptible_arrow_distinguishes_unavailable_from_fault():
    controller = StatefulArrow()
    unavailable = InterruptibleArrow(controller, perception=lambda _frame: None)
    frame = ObservationFrame({"state": [0.0] * 8})
    assert unavailable.propose(frame) is None
    assert unavailable.last_availability.available is False

    def fault(_frame):
        raise ValueError("real controller fault")

    broken = InterruptibleArrow(controller, perception=fault)
    with pytest.raises(ValueError, match="real controller fault"):
        broken.propose(frame)


def test_native_host_rolls_back_environment_and_components_on_commit_fault():
    env = Env()
    calls = []
    vla = _vla(calls)

    class FaultArrow(StatefulArrow):
        def commit(self, _record):
            self.commits += 1
            raise RuntimeError("commit fault")

    arrow = InterruptibleArrow(FaultArrow())
    host = NativeHost(env, vla, arrow, action_selector=lambda _base, teacher: teacher.action)
    with pytest.raises(RuntimeError, match="commit fault"):
        host.step()
    assert env.steps == 0
    assert env.value == 0.0
    assert host.timestep == 0
    # The injected inference's external telemetry list is intentionally not
    # part of the adapter state; the queued/model state itself is restored.
    assert calls == [1]


def test_native_host_rejects_producer_that_mutates_environment_before_step():
    env = Env()

    class MutatingVLA:
        def propose(self, frame):
            env.value += 0.5
            return ActionProposal((0.0,) * 7, policy_id="vla", timestep=frame.timestep, observation_digest=frame.digest)

        def snapshot_state(self):
            return 0

        def restore_state(self, _state):
            return None

    with pytest.raises(Exception, match=r"advanced environment state at environment\[0\]"):
        NativeHost(env, MutatingVLA()).step()
    assert env.value == 0.0
    assert env.steps == 0


def test_native_host_preserves_original_failure_when_rollback_also_fails():
    class RestoreFailEnv(Env):
        def restore_state(self, _state):
            raise RuntimeError("restore failure")

    class ProposalFailVLA:
        def snapshot_state(self):
            return None

        def restore_state(self, _state):
            return None

        def propose(self, _frame):
            raise ValueError("original proposal failure")

    with pytest.raises(ValueError, match="original proposal failure") as error:
        NativeHost(RestoreFailEnv(), ProposalFailVLA()).step()
    assert any("rollback failed" in note and "restore failure" in note for note in error.value.__notes__)


def test_live_adapter_binds_reset_identity_and_checks_digest():
    class Raw:
        def __init__(self):
            self.value = -1.0

        def observe(self):
            return {"state": [self.value] + [0.0] * 7}

        def reset(self, **_kwargs):
            self.value = 0.0
            return self.observe()

        def step(self, action):
            self.value += action[0]
            return {"observation": self.observe()}

    expected = {"state": [0.0] + [0.0] * 7}
    identity = ResetIdentity(
        task_id=0, episode_id="task0-test", seed=7, reset_index=1,
        observation_sha256=digest(expected), environment_fingerprint="fake-env",
        replay_key="reset-7",
    )
    adapter = LiberoEnvironmentAdapter.from_reset_identity(Raw(), identity)
    assert adapter.reset_identity == identity
    assert digest(adapter.observe()) == identity.observation_sha256
