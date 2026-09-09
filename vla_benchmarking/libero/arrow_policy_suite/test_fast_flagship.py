from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import types

import pytest

# The parent package is assembled concurrently by the native-spine worker.  A
# minimal package shell keeps this focused test independent of unrelated
# optional suite modules while still exercising the real owned modules.
if "arrow_policy_suite" not in sys.modules:
    _package = types.ModuleType("arrow_policy_suite")
    _package.__path__ = [str(Path(__file__).parent)]
    sys.modules["arrow_policy_suite"] = _package

from arrow_policy_suite.contracts import ActionProposal, ContractError, ObservationFrame
from arrow_policy_suite.fast import FastOneObservationSession, FastPolicy
from arrow_policy_suite.fast_training import (
    FAST_PARAMETER_COUNT,
    FastSupportExample,
    GraphFastModel,
    GraphRouter,
    GraphSlowEncoder,
    train_fast_support,
)
from arrow_policy_suite.graph_context import GraphContextPacket, make_graph_context, validate_context_mapping


def _frame(step: int = 0) -> ObservationFrame:
    return ObservationFrame(
        {"state": [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1], "instruction": "pick"},
        timestep=step,
        episode_id="ep-1",
    )


def _context(frame: ObservationFrame) -> GraphContextPacket:
    return make_graph_context(
        frame,
        {"subject": "hand", "relation": "toward", "object": "source"},
        {"polyline": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1]], "units": "m"},
        graph_revision="triplets-v3",
        arrow_revision="arrow-rgbd-v2",
        phase="approach_source",
    )


def _proposal(frame: ObservationFrame, value: float, policy_id: str) -> ActionProposal:
    return ActionProposal(
        (value,) + (0.0,) * 6,
        policy_id,
        timestep=frame.timestep,
        observation_digest=frame.digest,
    )


def test_graph_context_is_digest_bound_and_not_a_vla_observation_sidecar():
    frame = _frame()
    packet = _context(frame)
    assert packet.packet_digest == packet.packet_digest
    packet.assert_frame(frame)
    validate_context_mapping(packet.to_mapping(), frame)
    with pytest.raises(ContractError, match="stale"):
        packet.assert_frame(_frame(1))
    assert "graph_context" not in frame.observation
    assert packet.observation_digest == frame.digest


def test_native_slow_encoder_and_router_require_explicit_frozen_artifacts():
    with pytest.raises(ContractError, match="frozen revisioned artifact"):
        GraphSlowEncoder(native_mode=True)
    with pytest.raises(ContractError, match="frozen revisioned artifact"):
        GraphRouter(native_mode=True)
    model = GraphFastModel(native_mode=True)
    manifest = model.slow_manifest()
    assert manifest["encoder"]["revision"] == "deterministic-digest-v1"
    assert manifest["router"]["revision"] == "deterministic-phase-router-v1"
    assert len(model.slow_manifest_sha256()) == 64
    with pytest.raises(ContractError, match="digest-bound"):
        model.correction({"graph_features": [0.0] * 32})


class _FakeVLA:
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, frame: ObservationFrame) -> ActionProposal:
        self.calls += 1
        return _proposal(frame, 0.0, "vla")

    def manifest(self):
        return {"checkpoint": "smolvla-test", "revision": "frozen"}


class _FakeTeacher:
    def __init__(self) -> None:
        self.calls = 0
        self.detached = False
        self.closed = False

    def propose(self, frame: ObservationFrame) -> ActionProposal:
        self.calls += 1
        return _proposal(frame, 0.5, "arrow")

    def detach(self) -> None:
        self.detached = True

    def close(self) -> None:
        self.closed = True


def test_one_observation_session_has_one_attempt_and_teacher_free_query():
    model = GraphFastModel(native_mode=True, vla_manifest_sha256=sha256(b"vla").hexdigest())
    vla = _FakeVLA()
    teacher = _FakeTeacher()
    state = {"reset": "task0", "step": 0}

    def snapshot():
        return dict(state)

    def restore(value):
        state.clear()
        state.update(value)

    session = FastOneObservationSession(
        model,
        snapshot_fn=snapshot,
        restore_fn=restore,
        graph_context_fn=_context,
        vla=vla,
        teacher=teacher,
        vla_identity_fn=lambda _component: sha256(b"vla").hexdigest(),
    )
    result = session.run(
        [_frame(0)], [_frame(0)], support_success=True,
        source_manifest_sha256=sha256(b"support-manifest").hexdigest(),
    )
    receipt = result.receipt
    assert receipt.support_attempts == 1
    assert receipt.support_steps == 1
    assert receipt.support_vla_calls == 1
    assert receipt.support_teacher_calls == 1
    assert receipt.scored_steps == 1
    assert receipt.scored_teacher_calls == 0
    assert teacher.calls == 1
    assert teacher.detached and teacher.closed
    assert receipt.restored_t0 is True
    assert receipt.non_fast_changed == 0
    assert receipt.fast_slot_capacity == FAST_PARAMETER_COUNT == 448
    assert receipt.vla_manifest_sha256_before == receipt.vla_manifest_sha256_after
    assert receipt.slow_manifest_sha256_before == receipt.slow_manifest_sha256_after
    assert result.decisions[0].metadata["teacher_calls"] == 0


def test_fast_policy_rejects_accidental_teacher_at_scored_boundary():
    model = GraphFastModel()
    frame = _frame()
    base = _proposal(frame, 0.0, "vla")
    policy = FastPolicy(model, graph_fn=lambda _frame: {"graph_features": [0.0] * 32})
    with pytest.raises(ContractError, match="teacher-free"):
        policy.decide(frame, base, _proposal(frame, 0.2, "arrow"))


def test_failed_support_is_consumed_without_retry():
    class BadTeacher(_FakeTeacher):
        def propose(self, frame):
            self.calls += 1
            raise RuntimeError("teacher failed")

    model = GraphFastModel(native_mode=True)
    vla = _FakeVLA()
    teacher = BadTeacher()
    state = {"step": 0}
    session = FastOneObservationSession(
        model,
        snapshot_fn=lambda: dict(state),
        restore_fn=lambda value: state.update(value),
        graph_context_fn=_context,
        vla=vla,
        teacher=teacher,
        vla_identity_fn=lambda _component: sha256(b"vla").hexdigest(),
    )
    result = session.run(
        [_frame(0)], [], support_success=True,
        source_manifest_sha256=sha256(b"support-manifest").hexdigest(),
    )
    assert result.receipt.support_attempts == 1
    assert result.receipt.support_complete is False
    assert result.receipt.support_teacher_calls == 0
    assert model.last_receipt is not None and model.last_receipt.fallback is True
    with pytest.raises(ContractError, match="one-shot"):
        session.run([], [], support_success=True, source_manifest_sha256=sha256(b"x").hexdigest())
