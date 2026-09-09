import json
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

import pytest

from . import smolvla_peft_resume as resume


def test_resume_stage_plan_reuses_completed_stages_and_only_publishes():
    plan = resume.stage_plan()
    assert plan.collection == "VALIDATED_REUSED"
    assert plan.training == "VALIDATED_REUSED"
    assert plan.adapted_evaluation == "VALIDATED_REUSED"
    assert plan.artifact_publication == "PENDING_ATOMIC_PUBLICATION"


def test_integer_epoch_equivalent_does_not_use_rounded_source_value():
    assert resume.derive_training_steps(129) == 81
    value = resume.derive_epoch_equivalent(81, 129)
    assert repr(value) == "5.023255813953488"
    assert value == 81 * 8 / 129


def test_corrupt_source_status_rejects_before_any_ml_stage(tmp_path, monkeypatch):
    source = tmp_path / "archive" / "task_0" / "run"
    source.mkdir(parents=True)
    status = source.parent / "archive_status.env"
    status.write_text(
        "status=VERIFIED\njob_id=1925947\ntask_id=0\ntask_scope=task_0\n"
        "vla=smolvla\nbatch_size=8\narrow_demos=1\nworkload_exit_code=0\n",
        encoding="utf-8",
    )
    called = []

    def forbidden(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("a rejected resume must not enter an ML stage")

    monkeypatch.setattr(resume, "load_arrow_collection_manifest", forbidden)
    with pytest.raises(resume.ResumeError, match="preserved failed"):
        resume.validate_resume_source(
            source,
            expected_controller_config_hash="a" * 64,
            expected_base_policy_revision="base",
        )
    assert called == []


def test_archive_inventory_rejects_parseable_bit_flip(tmp_path):
    archive = tmp_path / "archive" / "task_0"
    source = archive / "run"
    checkpoint = source / "training" / "checkpoints" / "81" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    payload = checkpoint / "adapter_model.safetensors"
    payload.write_bytes(b"valid adapter bytes")
    other = source / "training_metadata" / "runtime_evidence.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"valid runtime bytes")
    status = archive / "archive_status.env"
    status.write_text("status=PRESERVED_FAILURE\n", encoding="utf-8")
    inventory = archive / "inventory.sha256"
    entries = [
        f"{resume.sha256_file(path)}  {path.resolve()}\n"
        for path in sorted((payload, other), key=lambda path: str(path))
    ]
    inventory.write_bytes("".join(entries).encode("ascii"))
    (archive / "tree_sha256").write_bytes((resume.sha256_file(inventory) + "\n").encode("ascii"))
    resume._validate_archive_integrity(archive)
    payload.write_bytes(b"valid adapter byteS")
    with pytest.raises(resume.ResumeError, match="digest mismatch"):
        resume._validate_archive_integrity(archive)


def test_archive_allows_only_expected_last_checkpoint_symlink(tmp_path):
    archive = tmp_path / "archive" / "task_0"
    checkpoints = archive / "run" / "training" / "checkpoints"
    checkpoint = checkpoints / "000081"
    checkpoint.mkdir(parents=True)
    payload = checkpoint / "adapter_model.safetensors"
    payload.write_bytes(b"valid adapter bytes")
    status = archive / "archive_status.env"
    status.write_bytes(b"status=PRESERVED_FAILURE\n")
    inventory = archive / "inventory.sha256"
    inventory.write_bytes(f"{resume.sha256_file(payload)}  {payload.resolve()}\n".encode("ascii"))
    (archive / "tree_sha256").write_bytes((resume.sha256_file(inventory) + "\n").encode("ascii"))
    link = checkpoints / "last"
    try:
        link.symlink_to("000081", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    resume._validate_archive_integrity(
        archive, allowed_symlink=link, allowed_target="000081", validated_checkpoint=checkpoint
    )
    link.unlink()
    link.symlink_to("000080", target_is_directory=True)
    with pytest.raises(resume.ResumeError, match="symlink target"):
        resume._validate_archive_integrity(
            archive, allowed_symlink=link, allowed_target="000081", validated_checkpoint=checkpoint
        )
    link.unlink()
    link.symlink_to("000081", target_is_directory=True)
    extra = checkpoints / "unexpected"
    extra.symlink_to("000081", target_is_directory=True)
    with pytest.raises(resume.ResumeError, match="unexpected symlink"):
        resume._validate_archive_integrity(
            archive, allowed_symlink=link, allowed_target="000081", validated_checkpoint=checkpoint
        )


def test_resume_module_has_no_collector_trainer_or_evaluator_execution_path():
    text = Path(resume.__file__).read_text(encoding="utf-8")
    assert "collect_smolvla_arrow_corrections" not in text
    assert "run_lerobot_train" not in text
    assert "run_lerobot_eval_with_context" not in text
    assert "save_peft_adapter(" in text
    assert 'publication_recovery_receipt.json' in text
    assert 'git_commit=source.training_commit' in text


def test_source_status_is_task_archive_adjacent():
    text = Path(resume.__file__).read_text(encoding="utf-8")
    assert 'source.parent / "archive_status.env"' in text
    assert 'source.parent.parent / "archive_status.env"' not in text


def test_publication_records_training_commit_and_recovery_commit_separately(tmp_path, monkeypatch):
    source_root = tmp_path / "archive" / "task_0" / "run"
    source_root.mkdir(parents=True)
    source = resume.ResumeSource(
        source_run_root=source_root,
        archive_status=source_root.parent / "archive_status.env",
        collection_manifest=source_root / "collection.json",
        dataset_root=source_root / "dataset",
        dataset_manifest=source_root / "dataset.json",
        checkpoint=source_root / "checkpoint",
        checkpoint_step=81,
        runtime_evidence=source_root / "runtime.json",
        training_plan=source_root / "plan.json",
        adapted_eval=source_root / "eval.json",
        adapted_audit=source_root / "audit.jsonl",
        eval_reset_contract=source_root / "reset.json",
        expected_inventory=source_root / "inventory.json",
        collection_success_count=1,
        dataset_frames=129,
        controller_config_hash="a" * 64,
        source_collection_manifest_sha256="b" * 64,
        training_commit=resume.TRAINING_COMMIT,
        training_runtime_versions={"python": "3.12.0", "torch": "2.7.0"},
    )
    monkeypatch.setattr(resume, "_stats", lambda _: {"successes": 1, "episodes": 10, "success_rate": 0.1, "seeds": resume.SEALED_SEEDS, "path": str(source.adapted_eval)})
    monkeypatch.setattr(resume, "sha256_file", lambda _: "b" * 64)
    captured = {}

    @dataclass(frozen=True)
    class FakeManifest:
        artifact_path: str

    @dataclass(frozen=True)
    class FakeLoadedManifest:
        artifact_path: str = "/tmp/published/smolvla/task_0/peft_adapter/recovery"
        task_id: int = 0
        checkpoint_step: int = 81
        collection_success_count: int = 1
        train_counts: dict = None
        artifact_tree_sha256: str = "d" * 64

        def __post_init__(self):
            if self.train_counts is None:
                object.__setattr__(self, "train_counts", {"dataset_frames": 129, "epoch_equivalent": 81 * 8 / 129})

    def fake_save(*args, **kwargs):
        captured.update(kwargs)
        return FakeManifest("/tmp/published/smolvla/task_0/peft_adapter/recovery")

    monkeypatch.setattr(resume, "save_peft_adapter", fake_save)
    monkeypatch.setattr(resume, "load_peft_manifest", lambda _: FakeLoadedManifest())
    monkeypatch.setattr(resume, "tree_sha256", lambda _: "e" * 64)
    monkeypatch.setattr(resume, "_versions", lambda: {"python": "publication"})
    artifact = resume.publish_resume(
        source,
        output_run_root=tmp_path / "new" / "run",
        output_archive_root=tmp_path / "new" / "archive",
        current_commit="c" * 40,
        base_policy=tmp_path / "base",
        base_policy_revision="base-revision",
    )
    assert str(artifact).endswith("recovery")
    assert captured["git_commit"] == resume.TRAINING_COMMIT
    assert captured["runtime_versions"] == source.training_runtime_versions
    assert captured["runtime_evidence"] == str(source.runtime_evidence)
    assert isinstance(captured["runtime_evidence"], str)
    adapter_pointer = json.loads((tmp_path / "new" / "run" / "ADAPTER_ARTIFACT_PATH.json").read_text())
    assert adapter_pointer == str(Path("/tmp/published/smolvla/task_0/peft_adapter/recovery").resolve())
    receipt = json.loads((tmp_path / "new" / "run" / "publication_recovery_receipt.json").read_text())
    summary = json.loads((tmp_path / "new" / "run" / "experiment_summary.json").read_text())
    assert receipt["training_commit"] == resume.TRAINING_COMMIT
    assert receipt["publication_commit"] == "c" * 40
    assert receipt["publication_runtime_versions"] == {"python": "publication"}
    assert set(receipt["reused_stage_hashes"]) == {
        "collection_manifest_sha256", "checkpoint_adapter_model_sha256",
        "checkpoint_tree_sha256", "runtime_evidence_sha256",
        "adapted_eval_info_sha256", "adapted_randomization_audit_sha256",
        "eval_reset_contract_sha256",
    }
    assert receipt["final_artifact"]["manifest_sha256"] == "b" * 64
    assert receipt["final_artifact"]["tree_sha256"] == "d" * 64
    assert summary["training_commit"] == resume.TRAINING_COMMIT
    assert summary["publication_commit"] == "c" * 40


def test_publication_rejects_output_roots_nested_under_source(tmp_path, monkeypatch):
    source_root = tmp_path / "archive" / "task_0" / "run"
    source_root.mkdir(parents=True)
    source = resume.ResumeSource(
        source_run_root=source_root,
        archive_status=source_root.parent / "archive_status.env",
        collection_manifest=source_root / "collection.json",
        dataset_root=source_root / "dataset",
        dataset_manifest=source_root / "dataset.json",
        checkpoint=source_root / "checkpoint",
        checkpoint_step=81,
        runtime_evidence=source_root / "runtime.json",
        training_plan=source_root / "plan.json",
        adapted_eval=source_root / "eval.json",
        adapted_audit=source_root / "audit.jsonl",
        eval_reset_contract=source_root / "reset.json",
        expected_inventory=source_root / "inventory.json",
        collection_success_count=1,
        dataset_frames=129,
        controller_config_hash="a" * 64,
        source_collection_manifest_sha256="b" * 64,
        training_commit=resume.TRAINING_COMMIT,
    )
    monkeypatch.setattr(resume, "save_peft_adapter", lambda *_args, **_kwargs: pytest.fail("publication must be rejected before save"))
    cases = (
        (source_root / "new", tmp_path / "separate-archive"),
        (tmp_path / "separate-run", source_root.parent / "new-archive"),
        (source_root.parent, tmp_path / "separate-archive-2"),
    )
    for output_run_root, output_archive_root in cases:
        with pytest.raises(resume.ResumeError, match="disjoint"):
            resume.publish_resume(
                source,
                output_run_root=output_run_root,
                output_archive_root=output_archive_root,
                current_commit="c" * 40,
                base_policy=tmp_path / "base",
                base_policy_revision="base-revision",
            )


def test_runner_resume_branch_precedes_all_fresh_ml_stages():
    runner = Path(__file__).parent / "legion" / "run_smolvla_peft_arrow_task.sbatch"
    text = runner.read_text(encoding="utf-8")
    branch = text.index('if [[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" ]]; then')
    baseline = text.index("BASELINE_EVAL_FILE=", branch)
    assert branch < baseline
    assert "--source-run-root \"$PEFT_RESUME_SOURCE_RUN_ROOT\"" in text
    assert "publication-only resume" in text


def test_runner_resume_setup_executes_with_empty_run_root(tmp_path):
    """Execute the runner's setup block: only TMP_ROOT may receive cache files."""
    runner = Path(__file__).parent / "legion" / "run_smolvla_peft_arrow_task.sbatch"
    text = runner.read_text(encoding="utf-8")
    start = text.index('mkdir -p "$RUN_ROOT" "$TMP_ROOT"')
    end = text.index("LIBERO_CONFIG_PATH=", start)
    setup = text[start:end]
    script = (
        "set -Eeuo pipefail\n"
        'RUN_ROOT="/tmp/codex-resume-run-$$"\n'
        'TMP_ROOT="/tmp/codex-resume-tmp-$$"\n'
        "trap 'rm -rf -- \"$RUN_ROOT\" \"$TMP_ROOT\"' EXIT\n"
        + setup + "\n"
        + (
        'test -d "$TMP_ROOT/xdg"\n'
        '! test -e "$RUN_ROOT/dataset"\n'
        '! test -e "$RUN_ROOT/eval_baseline"\n'
        '! test -e "$RUN_ROOT/training"\n'
        '! test -e "$RUN_ROOT/training_metadata"\n'
        '! test -e "$RUN_ROOT/eval_adapted"\n'
        )
    )
    env = os.environ.copy()
    env["PEFT_RESUME_SOURCE_RUN_ROOT"] = "/preserved/task_0/run"
    result = subprocess.run(
        [r"C:\Program Files\Git\bin\bash.exe", "-c", script],
        env=env,
        cwd=runner.parents[4],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_resume_wrapper_requires_explicit_source_and_forces_one_shot_task0():
    wrapper = Path(__file__).parent / "legion" / "run_smolvla_peft_arrow_resume.sbatch"
    text = wrapper.read_text(encoding="utf-8")
    assert '[[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" ]]' in text
    assert "export PEFT_TASK_ID=0 PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1" in text
    assert "run_smolvla_peft_arrow_task.sbatch" in text
