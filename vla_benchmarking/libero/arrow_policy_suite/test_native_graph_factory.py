from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

if "arrow_policy_suite" not in sys.modules:
    _package = types.ModuleType("arrow_policy_suite")
    _package.__path__ = [str(Path(__file__).parent)]
    sys.modules["arrow_policy_suite"] = _package

from arrow_policy_suite.contracts import ContractError, ObservationFrame
from arrow_policy_suite.graph_context import validate_context_mapping
from arrow_policy_suite.native_graph_factory import build_graph_context, build_packet


def _frame():
    return ObservationFrame(
        {"state": [0.0] * 8, "instruction": "pick the block"},
        timestep=0,
        episode_id="canary-0",
    )


def _triplet():
    return {"subject": "hand", "relation": "toward", "object": "red block"}


def _arrow():
    return {
        "polyline": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.2]],
        "provider": "arrow_rgbd",
        "coordinate_frame": "world",
        "units": "m",
    }


def test_native_graph_factory_returns_sealed_digest_bound_mapping(monkeypatch):
    monkeypatch.setenv("ARROW_SUITE_GRAPH_CONTEXT_REVISION", "graph-seal-v1")
    packet = build_packet(_frame(), triplet=_triplet(), arrow_geometry=_arrow(), task_text="pick the block")
    assert packet.observation_digest == _frame().digest
    assert packet.graph_revision == packet.arrow_revision == "graph-seal-v1"
    mapping = build_graph_context(_frame(), triplet=_triplet(), arrow_geometry=_arrow(), task_text="pick the block")
    validate_context_mapping(mapping, _frame())
    assert mapping["packet_digest"] == packet.packet_digest
    assert "agentview" not in mapping
    assert "observation" not in mapping


def test_native_graph_factory_fails_closed_when_inputs_are_missing(monkeypatch):
    monkeypatch.setenv("ARROW_SUITE_GRAPH_CONTEXT_REVISION", "graph-seal-v1")
    monkeypatch.delenv("ARROW_SUITE_TEXT_GRAPH_TRIPLET", raising=False)
    monkeypatch.delenv("ARROW_SUITE_GRAPH_TRIPLET", raising=False)
    monkeypatch.delenv("ARROW_SUITE_VISUAL_ARROW", raising=False)
    monkeypatch.delenv("ARROW_SUITE_ARROW_METADATA", raising=False)
    with pytest.raises(ContractError, match="text graph triplet"):
        build_graph_context(_frame())
    with pytest.raises(ContractError, match="visual-arrow"):
        build_graph_context(_frame(), triplet=_triplet())


def test_native_graph_factory_rejects_privileged_sidecars_and_unsealed_revision(monkeypatch):
    monkeypatch.setenv("ARROW_SUITE_GRAPH_CONTEXT_REVISION", "graph-seal-v1")
    with pytest.raises(ContractError, match="privileged"):
        build_packet(
            _frame(), triplet=_triplet(),
            arrow_geometry={**_arrow(), "simulator_state": {"qpos": [0.0]}},
        )
    monkeypatch.setenv("ARROW_SUITE_GRAPH_CONTEXT_REVISION", "latest")
    with pytest.raises(ContractError, match="sealed"):
        build_packet(_frame(), triplet=_triplet(), arrow_geometry=_arrow())


def test_native_graph_factory_rejects_revision_mismatch(monkeypatch):
    monkeypatch.setenv("ARROW_SUITE_GRAPH_CONTEXT_REVISION", "graph-seal-v1")
    with pytest.raises(ContractError, match="revisions"):
        build_packet(
            _frame(), triplet=_triplet(), arrow_geometry=_arrow(),
            graph_revision="graph-other-v1",
        )
