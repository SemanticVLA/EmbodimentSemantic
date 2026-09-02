"""Integrity and failure-path tests for the observation-only episode trace."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arrow_episode_trace import ArrowEpisodeTrace, TraceError, validate_trace


def _observation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.ones((4, 5), dtype=np.float32),
        np.array([0.1, -0.2], dtype=np.float64),
    )


def test_trace_persists_atomic_snapshot_refs_and_flushes_monotonic_steps(tmp_path: Path):
    trace = ArrowEpisodeTrace(tmp_path / "episode", episode_id="ep-1", metadata={"camera": "agentview"})
    rgb, depth, proprio = _observation()
    first = trace.record_observation(rgb, depth, proprio, phase="pregrasp")
    trace.append_step({"kind": "phase", "phase": "grasp", "status": "reached"})
    result = trace.finalize(summary={"operator_note": "test"})

    assert first["path"] == "snapshots/step_000000.npz"
    assert (trace.root / first["path"]).is_file()
    assert result["step_count"] == 2
    lines = [json.loads(line) for line in trace.steps_path.read_text(encoding="utf-8").splitlines()]
    assert [line["seq"] for line in lines] == [0, 1]
    with np.load(trace.root / first["path"]) as snapshot:
        np.testing.assert_array_equal(snapshot["rgb"], rgb)
        np.testing.assert_array_equal(snapshot["depth"], depth)
        np.testing.assert_array_equal(snapshot["proprio"], proprio)
    assert validate_trace(trace.root)["valid"] is True


def test_trace_is_write_once_and_writer_rejects_duplicate_or_skipped_seq(tmp_path: Path):
    root = tmp_path / "episode"
    trace = ArrowEpisodeTrace(root)
    trace.append_step({"kind": "heartbeat"}, seq=0)
    with pytest.raises(TraceError, match="non-monotonic"):
        trace.append_step({"kind": "duplicate"}, seq=0)
    with pytest.raises(TraceError, match="non-monotonic"):
        trace.append_step({"kind": "skipped"}, seq=2)
    trace.finalize()
    with pytest.raises(TraceError, match="already exists"):
        ArrowEpisodeTrace(root)


def test_validate_trace_detects_tampering_and_duplicate_seq(tmp_path: Path):
    trace = ArrowEpisodeTrace(tmp_path / "episode")
    trace.append_step({"kind": "heartbeat"})
    trace.finalize()
    trace.steps_path.open("a", encoding="utf-8").write('{"kind":"tampered","seq":0}\n')
    with pytest.raises(TraceError, match="hash mismatch"):
        validate_trace(trace.root)

    # Rebuild a trace, then rewrite the manifest hash only in the test fixture
    # to exercise the independent duplicate-sequence validator.
    second = ArrowEpisodeTrace(tmp_path / "episode-duplicate")
    second.append_step({"kind": "heartbeat"})
    second.finalize()
    with second.steps_path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"duplicate","seq":0}\n')
    manifest_path = second.root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib
    manifest["artifacts"]["steps.jsonl"] = hashlib.sha256(second.steps_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(TraceError, match="strictly monotonic"):
        validate_trace(second.root)


def test_context_manager_emits_failure_bundle_on_interruption(tmp_path: Path):
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        with ArrowEpisodeTrace(root, episode_id="interrupted") as trace:
            trace.append_step({"kind": "heartbeat"})
            raise KeyboardInterrupt()
    failure = json.loads((root / "failure_bundle.json").read_text(encoding="utf-8"))
    assert failure["interrupted"] is True
    assert failure["type"] == "KeyboardInterrupt"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    assert validate_trace(root)["valid"] is True


def test_observation_callback_never_returns_action_or_accepts_unaligned_depth(tmp_path: Path):
    trace = ArrowEpisodeTrace(tmp_path / "episode")
    rgb, depth, proprio = _observation()
    assert "action" not in trace.observe(rgb, depth, proprio)
    with pytest.raises(TraceError, match="aligned"):
        trace.observe(rgb, np.ones((3, 5)), proprio)
    trace.finalize()
