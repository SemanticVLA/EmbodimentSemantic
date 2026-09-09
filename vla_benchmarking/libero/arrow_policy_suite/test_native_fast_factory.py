from __future__ import annotations

from hashlib import sha256
import copy
from pathlib import Path
import random
import sys
import types

import pytest

if "arrow_policy_suite" not in sys.modules:
    _package = types.ModuleType("arrow_policy_suite")
    _package.__path__ = [str(Path(__file__).parent)]
    sys.modules["arrow_policy_suite"] = _package

from arrow_policy_suite.contracts import ActionProposal, ContractError, ObservationFrame
from arrow_policy_suite.graph_context import make_graph_context
from arrow_policy_suite.native_fast_factory import build_fast_native_host
from arrow_policy_suite.native_arrow_teacher import PerFrameArrowTeacher


def _configure(monkeypatch):
    monkeypatch.setenv("ARROW_SUITE_FAST_ARTIFACT_KIND", "deterministic")
    monkeypatch.setenv("ARROW_SUITE_FAST_ENCODER_REVISION", "encoder-v1")
    monkeypatch.setenv("ARROW_SUITE_FAST_ENCODER_SHA256", "1" * 64)
    monkeypatch.setenv("ARROW_SUITE_FAST_ROUTER_REVISION", "router-v1")
    monkeypatch.setenv("ARROW_SUITE_FAST_ROUTER_SHA256", "2" * 64)
    monkeypatch.setenv("ARROW_SUITE_FAST_SOURCE_MANIFEST_SHA256", "3" * 64)
    monkeypatch.setenv("ARROW_SUITE_FAST_VLA_MANIFEST_SHA256", "4" * 64)


def _context(frame: ObservationFrame):
    return make_graph_context(
        frame,
        {"subject": "hand", "relation": "toward", "object": "block"},
        {"polyline": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.1]], "provider": "arrow_rgbd", "units": "m"},
        graph_revision="graph-v1", arrow_revision="graph-v1", phase="approach_source",
    ).to_mapping()


class _Env:
    def __init__(self):
        self.value = 0.0
        self.steps = 0

    def observe(self):
        return {"state": [self.value] + [0.0] * 7, "instruction": "pick"}

    def step(self, action):
        self.value += float(action[0])
        self.steps += 1
        return {"success": True, "terminal": True}

    def snapshot_state(self):
        return self.value, self.steps

    def restore_state(self, value):
        self.value, self.steps = value


class _VLA:
    n_action_steps = 1

    def propose(self, frame):
        return ActionProposal((0.0,) * 7, "smolvla", frame.timestep, observation_digest=frame.digest)

    def snapshot_state(self):
        return "frozen"

    def restore_state(self, value):
        assert value == "frozen"

    def reset(self):
        return None


class _Teacher:
    def __init__(self):
        self.calls = 0
        self.detached = False
        self.closed = False
        self.close_calls = 0

    def propose(self, frame):
        self.calls += 1
        return ActionProposal((0.5,) + (0.0,) * 6, "arrow", frame.timestep, observation_digest=frame.digest)

    def snapshot_state(self):
        return "teacher"

    def restore_state(self, value):
        assert value == "teacher"

    def detach(self):
        self.detached = True

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_native_fast_factory_adapts_once_then_returns_teacher_free_host(monkeypatch):
    _configure(monkeypatch)
    env, vla, teacher = _Env(), _VLA(), _Teacher()
    host, bundle = build_fast_native_host(env, vla, teacher, graph_context_fn=_context)
    assert host.teacher is None
    assert teacher.calls == 1
    assert teacher.detached and teacher.closed
    assert teacher.close_calls == 1
    assert bundle.receipt.support_attempts == 1
    assert bundle.receipt.support_steps == 1
    assert bundle.receipt.support_teacher_calls == 1
    assert bundle.receipt.scored_teacher_calls == 0
    assert bundle.receipt.teacher_destroyed is True
    assert bundle.receipt.restored_t0 is True
    assert bundle.corrector.metadata.adapted is True
    records = host.run(max_steps=1, reset_environment=False)
    assert len(records) == 1
    assert records[0].teacher is None
    assert teacher.calls == 1
    assert bundle.receipt.non_fast_changed == 0
    assert bundle.receipt.fast_slot_capacity == 448


def test_native_fast_factory_fails_without_explicit_artifact_identity(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.delenv("ARROW_SUITE_FAST_ENCODER_SHA256")
    with pytest.raises(ContractError, match="ENCODER_SHA256"):
        build_fast_native_host(_Env(), _VLA(), _Teacher(), graph_context_fn=_context)


def test_fast_closes_per_frame_teacher_and_runs_cleanup_once(monkeypatch):
    _configure(monkeypatch)
    cleanup_calls = []
    teacher = PerFrameArrowTeacher(
        lambda _frame: {"waypoints": [[0.5, 0.0, 0.5]] * 6},
        gripper_dwell_steps=1,
        cleanup_attempt=lambda: cleanup_calls.append("cleanup"),
    )
    _host, bundle = build_fast_native_host(_Env(), _VLA(), teacher, graph_context_fn=_context)
    assert bundle.receipt.teacher_destroyed is True
    assert cleanup_calls == ["cleanup"]
    teacher.destroy()
    assert cleanup_calls == ["cleanup"]


def test_native_fast_support_restores_process_rng_states(monkeypatch):
    _configure(monkeypatch)
    numpy = pytest.importorskip("numpy")
    try:
        import torch
    except ImportError:
        torch = None

    class RngVLA(_VLA):
        def propose(self, frame):
            random.random()
            numpy.random.random()
            if torch is not None:
                torch.rand(3)
                if torch.cuda.is_available():
                    torch.rand(3, device="cuda")
            return super().propose(frame)

    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(numpy.random.get_state())
    torch_before = torch.random.get_rng_state().clone() if torch is not None else None
    cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all()) if torch is not None and torch.cuda.is_available() else None
    try:
        _host, bundle = build_fast_native_host(_Env(), RngVLA(), _Teacher(), graph_context_fn=_context)
        assert bundle.receipt.restored_t0 is True
        assert random.getstate() == python_before
        numpy_after = numpy.random.get_state()
        assert numpy_after[0] == numpy_before[0]
        assert numpy.array_equal(numpy_after[1], numpy_before[1])
        assert numpy_after[2:] == numpy_before[2:]
        if torch is not None:
            assert torch.equal(torch.random.get_rng_state(), torch_before)
            if cuda_before is not None:
                assert all(torch.equal(after, before) for after, before in zip(torch.cuda.get_rng_state_all(), cuda_before))
    finally:
        random.setstate(python_before)
        numpy.random.set_state(numpy_before)
        if torch is not None:
            torch.random.set_rng_state(torch_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state_all(cuda_before)
