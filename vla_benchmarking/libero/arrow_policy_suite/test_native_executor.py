from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arrow_policy_suite.config import ProtocolSeal, StudyConfig
from arrow_policy_suite.contracts import ActionProposal, ObservationFrame
from arrow_policy_suite.native_executor import execute_native, production_preflight
from arrow_policy_suite.native_factory import NativeHostSpec, action_selector_for, build_native_host, build_policy
from arrow_policy_suite.fast import FastPolicy
from arrow_policy_suite.splits import ResetIdentity, build_split_manifest


def _config() -> StudyConfig:
    tasks = (0,)
    return StudyConfig(
        task_ids=tasks,
        test_reset_ids={0: tuple(f"test-{i}" for i in range(10))},
        validation_reset_ids={0: tuple(f"validation-{i}" for i in range(10))},
        model_revision="model-1", controller_revision="controller-1",
        calibration_revision="calibration-1", environment_revision="env-1",
        trace_geometry_provider_revision="graph-1", processor_revision="processor-1",
        action_encoding="osc7-v1", camera_names=("agentview",),
        image_resolution=(256, 256), depth_units="meters", renderer_revision="renderer-1",
        metric_revision="success-1200-v1",
    )


def _identities(split: str, count: int, start: int) -> list[ResetIdentity]:
    return [ResetIdentity(
        task_id=0, episode_id=f"{split}-{i}", seed=i, reset_index=1,
        observation_sha256="a" * 64, environment_fingerprint="env-1",
        simulator_state_sha256=f"{start + i:064x}",
    ) for i in range(count)]


def _seal() -> ProtocolSeal:
    config = _config()
    return ProtocolSeal(
        config,
        build_split_manifest("collection", _identities("collection", 50, 0)),
        build_split_manifest("validation", _identities("validation", 10, 100)),
        build_split_manifest("test", _identities("test", 10, 200)),
    )


class _Env:
    def __init__(self):
        self.value = 0.0
        self.steps = 0

    def observe(self):
        return {"state": [self.value] + [0.0] * 7}

    def step(self, action):
        self.value += float(action[0])
        self.steps += 1
        return {"success": self.steps >= 2, "terminal": self.steps >= 2}

    def snapshot_state(self):
        return self.value, self.steps

    def restore_state(self, value):
        self.value, self.steps = value


class _VLA:
    n_action_steps = 1

    def propose(self, frame: ObservationFrame):
        return ActionProposal((0.1,) + (0.0,) * 6, "smolvla", frame.timestep, observation_digest=frame.digest)

    def snapshot_state(self):
        return 0

    def restore_state(self, value):
        assert value == 0


class _Arrow:
    def propose(self, frame: ObservationFrame):
        return ActionProposal((0.2,) + (0.0,) * 6, "arrow", frame.timestep, observation_digest=frame.digest)

    def commit(self, _record):
        return None

    def snapshot_state(self):
        return 0

    def restore_state(self, value):
        assert value == 0

    def interrupt(self):
        return None


def _factory(**_kwargs):
    return build_native_host(NativeHostSpec(_Env(), _VLA(), _Arrow(), policy_id="teacher_only"))


def test_protocol_seal_does_not_require_training_digest():
    seal = _seal()
    assert seal.protocol_sha256 and len(seal.protocol_sha256) == 64
    assert ProtocolSeal.from_dict(seal.to_dict()).protocol_sha256 == seal.protocol_sha256


def test_native_executor_runs_explicit_factory_and_writes_create_only_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    seal = _seal()
    receipt = execute_native(
        _factory, config=seal.config, protocol_seal=seal, operation="canary",
        policy_id="teacher_only", run_dir=tmp_path / "run", output=tmp_path / "receipt.json",
        max_steps=3,
    )
    assert receipt.status == "COMPLETED"
    assert receipt.steps == 2
    assert json.loads((tmp_path / "run" / "run_manifest.json").read_text())["git_revision"]
    with pytest.raises(Exception):
        execute_native(_factory, config=seal.config, operation="canary", policy_id="teacher_only", run_dir=tmp_path / "run", max_steps=1)


def test_production_preflight_requires_graph_context_for_fast(tmp_path):
    host = build_native_host(NativeHostSpec(_Env(), _VLA(), _Arrow(), policy_id="teacher_only"))
    with pytest.raises(Exception, match="graph-context"):
        production_preflight(host, _config(), policy_id="arrow_fast")


def test_fast_preflight_requires_teacher_free_host_and_support_receipt():
    class Corrector:
        def correction(self, _payload):
            return (0.0,) * 7

    host = build_native_host(NativeHostSpec(
        _Env(), _VLA(), None, policy_id="arrow_fast",
        policy=FastPolicy(Corrector()), graph_context_fn=lambda _frame: {"triplet": "a"},
    ))
    with pytest.raises(Exception, match="lifecycle receipt"):
        production_preflight(host, _config(), policy_id="arrow_fast")
    host.fast_lifecycle_receipt = type("Receipt", (), {
        "support_attempts": 1, "support_steps": 1,
        "support_complete": True, "restored_t0": True,
        "fast_slot_capacity": 448, "scored_teacher_calls": 0, "teacher_detached": True,
        "teacher_destroyed": True,
        "non_fast_changed": 0,
        "slow_manifest_sha256_before": "a" * 64,
        "slow_manifest_sha256_after": "a" * 64,
        "vla_manifest_sha256_before": "b" * 64,
        "vla_manifest_sha256_after": "b" * 64,
    })()
    ready = production_preflight(host, _config(), policy_id="arrow_fast")
    assert ready["fast_lifecycle"]["support_complete"] is True


def test_fast_preflight_rejects_invalid_lifecycle_evidence():
    class Corrector:
        def correction(self, _payload):
            return (0.0,) * 7

    host = build_native_host(NativeHostSpec(
        _Env(), _VLA(), None, policy_id="arrow_fast",
        policy=FastPolicy(Corrector()), graph_context_fn=lambda _frame: {"triplet": "a"},
    ))
    values = dict(
        support_attempts=1, support_steps=1, support_complete=True, restored_t0=True,
        fast_slot_capacity=448, scored_teacher_calls=0, teacher_destroyed=True,
        teacher_detached=False, non_fast_changed=0,
        slow_manifest_sha256_before="a" * 64, slow_manifest_sha256_after="a" * 64,
        vla_manifest_sha256_before="b" * 64, vla_manifest_sha256_after="b" * 64,
    )
    for field, value in (("support_steps", 2), ("fast_slot_capacity", 447), ("non_fast_changed", 1)):
        invalid = dict(values)
        invalid[field] = value
        host.fast_lifecycle_receipt = SimpleNamespace(**invalid)
        with pytest.raises(Exception):
            production_preflight(host, _config(), policy_id="arrow_fast")
    invalid = dict(values)
    invalid["vla_manifest_sha256_after"] = "c" * 64
    host.fast_lifecycle_receipt = SimpleNamespace(**invalid)
    with pytest.raises(Exception, match="manifest"):
        production_preflight(host, _config(), policy_id="arrow_fast")


def test_controls_have_explicit_selection_semantics():
    base = ActionProposal((0.1,) + (0.0,) * 6, "vla", 0)
    teacher = ActionProposal((0.2,) + (0.0,) * 6, "arrow", 0)
    assert tuple(action_selector_for("frozen_base")(base, teacher)) == base.action
    assert tuple(action_selector_for("teacher_only")(base, teacher)) == teacher.action
    assert tuple(action_selector_for("teacher_only")(base, None)) == base.action


def test_minimal_row_aliases_are_explicit():
    runtime = build_policy("arrow_minimal_runtime")
    learned = build_policy("arrow_minimal_learned", learned_fn=lambda _frame, action: action)
    assert runtime.policy_id == "arrow_minimal" and runtime.variant == "runtime_oracle"
    assert learned.policy_id == "arrow_minimal" and learned.variant == "learned"


def test_learned_rows_fail_closed_without_runner_hooks():
    for policy_id in ("arrow_apprentice", "arrow_editor", "arrow_minimal_learned"):
        with pytest.raises(Exception, match="requires|hook"):
            build_policy(policy_id)
