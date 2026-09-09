from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from arrow_policy_suite.apprentice_training import save_adapter_checkpoint, verify_adapter_reload
from arrow_policy_suite.contracts import ActionProposal, ObservationFrame, PolicyDecision, StepRecord
from arrow_policy_suite.learning import intervention_rows, manifest_for_rows, write_dataset_view
from arrow_policy_suite.native_learned_training import load_native_residual, train_minimal_learned
from arrow_policy_suite.native_learned_training import build_native_learned_policy


def _record(*, success: bool, step: int = 0) -> StepRecord:
    frame = ObservationFrame({"state": [0.0] * 8}, timestep=step, episode_id="ep-fail",
                             metadata={"episode_id": "ep-fail", "task_id": 3})
    base = ActionProposal((0.0,) * 7, "smolvla", timestep=step, observation_digest=frame.digest)
    teacher = ActionProposal((0.25,) + (0.0,) * 6, "arrow", timestep=step, observation_digest=frame.digest)
    decision = PolicyDecision(teacher.action, "arrow_on_call", frame.digest, True, ("translation",),
                              {"teacher_used": True})
    return StepRecord(frame, base, teacher, decision, frame, {"success": success}, success, True)


def test_shared_filter_keeps_failed_executed_teacher_transition_and_hashes_view(tmp_path):
    rows = intervention_rows([_record(success=False)])
    assert len(rows) == 1
    assert rows[0].success_episode is False
    assert rows[0].outcome["success"] is False
    manifest = manifest_for_rows(rows, parent_artifact="on-call-archive-sha")
    view = write_dataset_view(tmp_path / "rows.jsonl", rows, manifest)
    assert view.rows == 1 and view.manifest.source_sha256 == manifest.content_sha256
    with pytest.raises(Exception, match="overwrite|immutable"):
        write_dataset_view(tmp_path / "rows.jsonl", rows, manifest)


def test_adapter_save_reload_is_immutable_and_base_bound(tmp_path):
    path = tmp_path / "adapter.bin"
    manifest = save_adapter_checkpoint(path, {"adapter.weight": [1.0, 2.0]}, base_vla_sha256="a" * 64)
    assert manifest["adapter_only"] is True
    assert verify_adapter_reload(path, base_vla_sha256="a" * 64)["reloaded"] is True
    with pytest.raises(Exception, match="base"):
        verify_adapter_reload(path, base_vla_sha256="b" * 64)


def test_native_minimal_checkpoint_is_teacher_free_and_reloadable(tmp_path):
    pytest.importorskip("torch")
    from arrow_policy_suite.learning import InterventionRow
    rows = (InterventionRow("ep", 0, 0, {"state": [0.0] * 8}, (0.0,) * 7,
                            (0.2,) + (0.0,) * 6, False, source="minimal_branch",
                            label_action=(0.1,) + (0.0,) * 6,
                            label_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                            label_source="minimal_branch_runtime"),)
    manifest = manifest_for_rows(rows, parent_artifact="on-call-archive-sha")
    receipt = train_minimal_learned(rows, output_path=tmp_path / "minimal.pt",
                                    base_vla_sha256="c" * 64, manifest=manifest)
    assert receipt.cost["failure_rows"] == 1 and receipt.optimizer_steps == 1
    model = load_native_residual(tmp_path / "minimal.pt", base_vla_sha256="c" * 64)
    _, corrected = model.predict_correction([0.0] * 32, (0.0,) * 7)
    assert len(corrected) == 7


def test_native_policy_factory_fails_closed_without_learned_artifact(tmp_path):
    with pytest.raises(Exception, match="missing"):
        build_native_learned_policy("arrow_editor", checkpoint_path=tmp_path / "missing.pt",
                                    base_vla_sha256="a" * 64)
    with pytest.raises(Exception, match="missing"):
        build_native_learned_policy("arrow_minimal_learned", checkpoint_path=tmp_path / "missing.pt",
                                    base_vla_sha256="a" * 64)


def test_native_policy_factory_loads_editor_and_minimal_artifacts(tmp_path):
    pytest.importorskip("torch")
    from arrow_policy_suite.learning import InterventionRow
    rows = (InterventionRow("ep", 0, 0, {"state": [0.0] * 8}, (0.0,) * 7,
                            (0.2,) + (0.0,) * 6, False, source="minimal_branch",
                            label_action=(0.1,) + (0.0,) * 6,
                            label_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                            label_source="minimal_branch_runtime"),)
    manifest = manifest_for_rows(rows, parent_artifact="on-call-archive-sha")
    editor_path = tmp_path / "editor.pt"
    train_minimal_learned(rows, output_path=editor_path, base_vla_sha256="d" * 64, manifest=manifest)
    editor = build_native_learned_policy("arrow_minimal_learned", checkpoint_path=editor_path,
                                         base_vla_sha256="d" * 64)
    frame = ObservationFrame({"state": [0.0] * 8}, episode_id="eval", timestep=0)
    base = ActionProposal((0.0,) * 7, "smolvla", timestep=0, observation_digest=frame.digest)
    decision = editor.decide(frame, base, None)
    assert len(decision.action) == 7 and decision.metadata["teacher_free"] is True


def test_apprentice_factory_requires_explicit_native_adapter_loader(tmp_path):
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, {"adapter.weight": [1.0]}, base_vla_sha256="e" * 64)
    with pytest.raises(Exception, match="action loader"):
        build_native_learned_policy("arrow_apprentice", checkpoint_path=path, base_vla_sha256="e" * 64)
    policy = build_native_learned_policy(
        "arrow_apprentice", checkpoint_path=path, base_vla_sha256="e" * 64,
        adapter_action_fn_factory=lambda _path, _base_hash: lambda _frame, base: base,
    )
    assert policy.policy_id == "arrow_apprentice"
