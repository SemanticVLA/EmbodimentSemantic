from __future__ import annotations

import json

import pytest

from .collection import AttemptIdentity, TrainingSourceWriter, load_training_source
from .contracts import ActionProposal, ObservationFrame, PolicyDecision, StepRecord
from .branching import BranchRunner
from .native_host import NativeHost
from .native_factory import action_selector_for
from .native_executor import _native_identity
from .policies import MinimalPolicy, OnCallPolicy


def _record(*, executed: bool = True, success: bool = True, terminal: bool = True) -> StepRecord:
    frame = ObservationFrame({"state": [0.0] * 8}, timestep=0)
    next_frame = ObservationFrame({"state": [0.1] * 8}, timestep=1)
    base = ActionProposal((0.0,) * 7, "vla", 0, observation_digest=frame.digest)
    teacher = ActionProposal((0.2,) + (0.0,) * 6, "arrow", 0, observation_digest=frame.digest)
    decision = PolicyDecision(
        (0.2 if executed else 0.0,) + (0.0,) * 6,
        "arrow_on_call",
        frame.digest,
        executed,
        ("translation",) if executed else (),
        {"teacher_used": executed},
    )
    return StepRecord(frame, base, teacher, decision, next_frame,
                      {"success": success, "terminal": terminal, "sim_state": "must_not_be_logged"}, success, terminal)


def test_training_source_is_append_only_loadable_and_hash_verified(tmp_path):
    path = tmp_path / "training-source.jsonl"
    writer = TrainingSourceWriter(path)
    identity = AttemptIdentity(3, "reset-1", "episode-1", reset_identity={"simulator_state_sha256": "secret"})
    row = writer.append(_record(), identity=identity, source_hashes={"config_sha256": "a" * 64})
    assert row["eligible"] is True
    loaded = load_training_source(path, eligible_only=True)
    assert loaded[0]["observation"] == {"state": [0.0] * 8}
    assert "sim_state" not in json.dumps(loaded[0])
    assert loaded[0]["identity"]["reset_identity_sha256"]
    with path.open("ab") as handle:
        handle.write(b'{"schema":"wrong"}\n')
    with pytest.raises(Exception, match="unsupported training source schema"):
        load_training_source(path)
    with pytest.raises(Exception, match="explicit task, reset, and episode identity"):
        TrainingSourceWriter(tmp_path / "missing-identity.jsonl").append(_record())


def test_minimal_runtime_without_real_runner_fails_closed():
    frame = ObservationFrame({"state": [0.0] * 8})
    base = ActionProposal((0.0,) * 7, "vla", 0, observation_digest=frame.digest)
    teacher = ActionProposal((0.2,) * 7, "arrow", 0, observation_digest=frame.digest)
    with pytest.raises(Exception, match="real branch runner"):
        MinimalPolicy().decide(frame, base, teacher)


class _HostEnv:
    def __init__(self):
        self.value = 0.0
        self.steps = 0

    def observe(self):
        return {"state": [self.value] + [0.0] * 7}

    def step(self, action):
        self.value += float(action[0])
        self.steps += 1
        return {"success": self.steps >= 1, "terminal": self.steps >= 1}

    def snapshot_state(self):
        return self.value, self.steps

    def restore_state(self, state):
        self.value, self.steps = state


class _HostProposal:
    def __init__(self, policy_id, value):
        self.policy_id = policy_id
        self.value = value

    def propose(self, frame):
        return ActionProposal((self.value,) + (0.0,) * 6, self.policy_id,
                              frame.timestep, observation_digest=frame.digest)

    def snapshot_state(self):
        return ()

    def restore_state(self, _state):
        return None


class _BranchEnv:
    def __init__(self):
        self.value = 0.0

    def snapshot(self):
        return self.value

    def restore(self, state):
        self.value = state

    def step(self, action):
        self.value += float(action[0])
        return {"reward": 1.0 if self.value else 0.0, "done": False}


def test_native_on_call_decision_survives_and_training_source_serializes_it(tmp_path):
    policy = OnCallPolicy()
    policy._takeover = True  # deterministic fixture for the teacher-owned path
    host = NativeHost(
        _HostEnv(), _HostProposal("vla", 0.0), _HostProposal("arrow", 0.2),
        policy=policy,
        action_selector=lambda base, teacher, frame: policy.decide(frame, base, teacher),
    )
    record = host.step()
    assert isinstance(record.decision, PolicyDecision)
    assert record.decision.policy_id == "arrow_on_call"
    assert record.decision.action == record.action == record.teacher.action
    writer = TrainingSourceWriter(tmp_path / "on-call.jsonl")
    row = writer.append(record, identity=AttemptIdentity(0, "reset", "episode"))
    assert row["eligible"] is True
    assert row["decision"]["policy_id"] == "arrow_on_call"
    assert row["executed_action"] == list(record.action)


def test_native_minimal_decision_retains_branch_label_and_cost(tmp_path):
    runner = BranchRunner(
        _BranchEnv(),
        action_selector=lambda mask, _index, _state: ((1.0 if mask == 1 else 0.0),) + (0.0,) * 6,
        outcome_fn=lambda mask, _actions, _raws: {"sufficient": mask == 1, "progress": float(mask == 1)},
    )
    policy = MinimalPolicy(branch_runner=runner)
    host = NativeHost(
        _HostEnv(), _HostProposal("vla", 0.0), _HostProposal("arrow", 1.0),
        policy=policy,
        action_selector=lambda base, teacher, frame: policy.decide(frame, base, teacher),
    )
    record = host.step()
    assert isinstance(record.decision, PolicyDecision)
    assert record.decision.policy_id == "arrow_minimal"
    assert record.decision.action == record.action
    assert record.decision.metadata["branch_selected_mask"] == 1
    assert record.decision.metadata["branch_cloned_steps"] == 160
    path = tmp_path / "minimal-training.jsonl"
    writer = TrainingSourceWriter(path)
    row = writer.append(record, identity=AttemptIdentity(2, "reset-19", "episode-2", seed=123,
                                                         reset_identity={"init_state_index": 19}))
    assert row["eligible"] is True
    assert row["source"] == "minimal_branch"
    assert row["minimal_branch"]["selected_mask"] == 1
    from .learning import load_persisted_training_view
    view = load_persisted_training_view(path, variant="arrow_minimal_learned")
    assert len(view.rows) == 1
    assert view.rows[0].source == "minimal_branch"
    assert view.rows[0].label_source == "minimal_branch_runtime"
    assert view.rows[0].label_mask is not None
    assert view.rows[0].label_action == record.action


def test_native_factory_policy_selector_preserves_decision_controls_stay_raw():
    frame = ObservationFrame({"state": [0.0] * 8})
    base = ActionProposal((0.0,) * 7, "vla", 0, observation_digest=frame.digest)
    teacher = ActionProposal((0.2,) * 7, "arrow", 0, observation_digest=frame.digest)
    policy = OnCallPolicy()
    selected = action_selector_for("arrow_on_call", policy)(base, teacher, frame)
    assert isinstance(selected, PolicyDecision)
    assert selected.policy_id == "arrow_on_call"
    assert selected.action == base.action
    assert tuple(action_selector_for("frozen_base")(base, teacher)) == base.action


def test_legion_identity_is_explicit_in_manifest_row_contract_and_never_synthetic(tmp_path):
    class LegionHost:
        task_id = 7
        seed = 123
        init_state_index = 19
        reset_identity = None

    identity = _native_identity(LegionHost(), operation="collect")
    assert identity == {
        "task_id": 7,
        "seed": 123,
        "init_state_index": 19,
        "reset_id": "init_state_index:19",
        "episode_id": "task-7-seed-123-init-19",
    }
    assert "unknown" not in json.dumps(identity)
    assert "identity_seal" not in json.dumps(identity)
    row = TrainingSourceWriter(tmp_path / "identity-training-source.jsonl").append(
        _record(), identity=identity
    )
    assert row["task_id"] == 7 and row["seed"] == 123 and row["init_state_index"] == 19
    assert row["reset_id"] == "init_state_index:19"
    with pytest.raises(Exception, match="explicit host identity"):
        _native_identity(type("Missing", (), {"task_id": 7, "seed": 123})(), operation="evaluate")


def test_training_source_preserves_failed_rows_but_loads_only_successful_complete_rows(tmp_path):
    path = tmp_path / "mixed.jsonl"
    writer = TrainingSourceWriter(path)
    success = writer.append(
        _record(), identity=AttemptIdentity(4, "reset-1", "episode-success", seed=8),
        episode_success=True, episode_complete=True,
    )
    failed = writer.append(
        _record(success=False, terminal=False),
        identity=AttemptIdentity(4, "reset-2", "episode-failed", seed=8),
        episode_success=False, episode_complete=False,
    )
    assert success["eligible"] is True
    assert failed["eligible"] is False
    assert "episode_not_successful" in failed["eligibility_reasons"]
    assert "episode_not_complete" in failed["eligibility_reasons"]
    from .learning import load_persisted_training_view
    view = load_persisted_training_view(path, variant="arrow_editor")
    assert [row.episode_id for row in view.rows] == ["episode-success"]
    assert view.rejected["ineligible_source_row"] == 1
