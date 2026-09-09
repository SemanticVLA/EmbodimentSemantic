from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arrow_policy_suite.contracts import ContractError, digest
from arrow_policy_suite.residual_job import build_parser, main


def _source_row(*, eligible: bool = True) -> dict:
    observation = {"state": [0.0] * 8}
    action = [0.2] + [0.0] * 6
    row = {
        "schema": "arrow_policy_suite.training_source.v1",
        "task_id": 3,
        "reset_id": "reset-1",
        "episode_id": "episode-1",
        "timestep": 0,
        "observation": observation,
        "observation_digest": digest(observation),
        "base_proposal": {"action": [0.0] * 7},
        "teacher_proposal": {"action": action},
        "decision": {"action": action, "policy_id": "arrow_on_call", "teacher_used": True},
        "executed_action": action,
        "outcome": {"success": eligible, "terminal": True},
        "eligible": eligible,
        "hashes": {"observation_sha256": digest(observation)},
    }
    row["transition_sha256"] = hashlib.sha256(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    return row


def _write_source(path: Path, *, eligible: bool = True) -> None:
    path.write_text(json.dumps(_source_row(eligible=eligible), sort_keys=True, separators=(",", ":")) + "\n")


def _args(source: Path, output: Path) -> list[str]:
    return [
        "--variant", "editor",
        "--training-source", str(source),
        "--output-checkpoint", str(output),
        "--base-vla-sha256", "a" * 64,
        "--task-id", "3",
        "--reset-id", "reset-1",
        "--episode-id", "episode-1",
        "--seed", "17",
        "--epochs", "1",
        "--batch-size", "1",
        "--output-mode", "residual",
    ]


def test_parser_requires_exact_variant_choice():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--variant", "minimal", "--training-source", "x", "--output", "y", "--base-vla-sha256", "a" * 64])


def test_editor_job_emits_machine_receipt_and_immutable_checkpoint(tmp_path, capsys):
    pytest.importorskip("torch")
    source = tmp_path / "training-source.jsonl"
    output = tmp_path / "editor.pt"
    _write_source(source)
    assert main(_args(source, output)) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "COMPLETED"
    assert receipt["variant"] == "editor"
    assert receipt["receipt"]["config"]["seed"] == 17
    assert output.is_file() and Path(str(output) + ".json").is_file()
    assert main(_args(source, output)) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED"
    assert "overwrite" in blocked["error"] or "immutable" in blocked["error"]


def test_minimal_job_refuses_incompatible_on_call_rows(tmp_path, capsys):
    pytest.importorskip("torch")
    source = tmp_path / "training-source.jsonl"
    output = tmp_path / "minimal.pt"
    _write_source(source)
    args = _args(source, output)
    args[1] = "minimal-learned"
    assert main(args) == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "BLOCKED"
    assert "Minimal" in receipt["error"] or "minimal" in receipt["error"]
    assert not output.exists()
