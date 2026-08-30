from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "submit_legion_action_visual_lora_pilot.sh"


def test_action_visual_launcher_is_separate_and_sealed():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "existing graph pilot" in text  # documentation of the separation
    assert "action_visual_lora_v1" in text
    assert "no_arrow_treatment" in text
    assert "--partition=gpu_a40" in text
    assert "--exclude=compute-4-13" in text
    assert "--expected-commit" in text
    assert "LIBERO_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'" in text
    assert "BASE_POLICY_REVISION='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'" in text
    assert "29190" in text and "1946" in text
    assert "78" in text and "156" in text and "1585152" in text
    assert "TRAINING_RUNTIME_EVIDENCE" in text
    assert "smoke_training_runtime.json" in text
    assert "full_training_runtime.json" in text
    assert "all_losses_finite" in text and "all_grad_norms_finite" in text
    assert "connector LoRA-B" in text and "late-vision LoRA-B" in text
    assert "input_provenance.json" in text
    assert "inventory.sha256" in text and "tree_sha256" in text
    assert "action_visual_lora_no_arrow_pair_manifest.json" in text
    assert "action_only_lora_v1_no_arrows" in text
    assert "action_visual_lora_v1_no_arrows" in text
    assert "--episodes 10 --batch-size 1 --device cuda" in text
    assert "VISUAL_CONDITION=none" in text and "VISUAL_ARROWS=0" in text
    assert "mkdir \"$LAUNCH_LOCK\"" in text
    assert text.count("status --porcelain --untracked-files=all") >= 3
    assert "setup_archive_tree_sha256" in text and "train_archive_tree_sha256" in text and "eval_archive_tree_sha256" in text
    assert "archive inventory is incomplete" in text


def test_action_visual_launcher_queues_all_three_jobs_with_afterok_dependencies():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'setup_raw="$(sbatch --parsable "$JOB_DIR/setup.sbatch")"' in text
    assert 'sbatch --parsable --dependency="afterok:$setup_job_id" "$JOB_DIR/train.sbatch"' in text
    assert 'sbatch --parsable --dependency="afterok:$train_job_id" "$rendered_eval"' in text
    assert "train_dependency=\"afterok:$setup_job_id\"" in text
    assert "eval_dependency=\"afterok:$train_job_id\"" in text
    assert "bash \"\\$REPO/vla_benchmarking/launch_lora_treatment.sh\" full" in text


def test_eval_stage_binds_resolved_libero_and_archives_preflight_failures():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "LIBERO_DIR='${LIBERO_DIR}'" in text
    assert 'git -C "\\$LIBERO_DIR" rev-parse HEAD' in text
    assert 'git -C "\\$LIBERO_DIR" status --porcelain --untracked-files=no' in text
    assert 'export LIBERO_DIR="\\$LIBERO_DIR"' in text
    eval_start = text.index("# Matched clean/no-arrow evaluation")
    eval_end = text.index("\nEOF", eval_start)
    eval_text = text[eval_start:eval_end]
    assert "copy_tree \"\\$EVIDENCE\" \"\\$ARCHIVE_DIR/evidence\"" in eval_text
    assert "if [[ \\$arc -eq 0 && -d \"\\$EVAL_ROOT\" ]]; then" in eval_text
    assert eval_text.index("ARCHIVE_DIR/evidence") < eval_text.index("ARCHIVE_DIR/results")
    assert "seal_tree \"\\$ARCHIVE_DIR\" build" in eval_text
    assert "seal_tree \"\\$ARCHIVE_DIR\" verify" in eval_text
    assert "TRAINING_PROFILE=no_arrow_treatment" in text
    assert "--action-only-checkpoint" in text and "--action-visual-checkpoint" in text
    assert "--action-only-training-manifest" in text and "--action-visual-training-manifest" in text
    assert "bash \"\\$REPO/vla_benchmarking/launch_lora_treatment.sh\" full" in text


def test_generated_stage_traps_before_fail_closed_preflight_checks():
    text = SCRIPT.read_text(encoding="utf-8")
    setup_start = text.index('cat > "$JOB_DIR/setup.sbatch"')
    train_start = text.index('cat > "$JOB_DIR/train.sbatch"')
    eval_start = text.index('cat > "$JOB_DIR/eval.sbatch"')
    sections = (
        text[setup_start:train_start],
        text[train_start:eval_start],
        text[eval_start:text.index("\nEOF", eval_start)],
    )
    train_text = sections[1]
    assert 'if [[ \\$arc -eq 0 && -d "\\$TRAIN_ROOT" ]]; then' in train_text
    assert train_text.index("ARCHIVE_DIR/evidence") < train_text.index("ARCHIVE_DIR/run")
    assert train_text.index("ARCHIVE_DIR/run") < train_text.index('seal_tree "\\$ARCHIVE_DIR" build')
    for section in sections:
        trap = section.index("trap finish EXIT")
        for initialized in ("STATE_FILE=", "ARCHIVE_DIR=", "EVIDENCE="):
            assert section.index(initialized) < trap
        for preflight in (
            "repository commit drift",
            "repository is dirty",
            "LIBERO checkout is not pinned",
            "LIBERO checkout is dirty",
        ):
            if preflight in section:
                assert trap < section.index(preflight)


@pytest.mark.skipif(os.name == "nt", reason="fake sbatch requires POSIX bash")
def test_fake_login_submission_does_not_run_compute_jobs(tmp_path):
    expected = "a" * 40
    fake_repo = tmp_path / "repo"
    fake_vla = fake_repo / "vla_benchmarking"
    fake_vla.mkdir(parents=True)
    (fake_repo / ".git").mkdir()
    probe = fake_vla / "submit.sh"
    source = SCRIPT.read_text(encoding="utf-8")
    probe.write_text(source, encoding="utf-8")
    runtime = tmp_path / "runtime"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive"
    data = scratch / "lora_datasets"
    libero_data = scratch / "libero_spatial_v5"
    libero = scratch / "LIBERO" / ".git"
    base = scratch / "base"
    baseline = tmp_path / "baseline" / "pretrained_model"
    baseline.mkdir(parents=True)
    for directory in (runtime, scratch, archive, data, libero_data, base):
        directory.mkdir(parents=True, exist_ok=True)
    libero.mkdir(parents=True)
    (tmp_path / "libero_config.yaml").write_text("config\n", encoding="utf-8")
    manifest = tmp_path / "historical.json"
    manifest.write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "sbatch.log"
    (fake_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        f"if [[ \"$*\" == *'rev-parse HEAD'* ]]; then printf '%s\\n' '{expected}'; fi\n",
        encoding="utf-8",
    )
    (fake_bin / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        "n=$(wc -l < '" + str(log) + "')\n"
        "printf '70%s\\n' \"$n\"\n",
        encoding="utf-8",
    )
    for path in (probe, fake_bin / "git", fake_bin / "sbatch"):
        path.chmod(0o700)
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env["PATH"] = os.pathsep.join((str(fake_bin), env.get("PATH", "")))
    result = subprocess.run(
        [
            shutil.which("bash"), str(probe),
            "--expected-commit", expected,
            "--action-only-checkpoint", str(baseline),
            "--action-only-training-manifest", str(manifest),
            "--data-root", str(data), "--libero-data-dir", str(libero_data),
            "--libero-dir", str(libero.parent), "--base-policy", str(base),
            "--libero-config", str(tmp_path / "libero_config.yaml"),
            "--runtime-root", str(runtime), "--scratch-root", str(scratch),
            "--archive-root", str(archive), "--label", "fake_action_visual",
        ],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert calls[0].startswith("--parsable ") and calls[0].endswith("/setup.sbatch")
    assert "--dependency=afterok:701" in calls[1]
    assert "--dependency=afterok:702" in calls[2]
    state = runtime / "action_visual_lora_pilot" / "fake_action_visual" / "state.env"
    state_text = state.read_text(encoding="utf-8")
    assert "setup_job_id=701" in state_text
    assert "train_job_id=702" in state_text
    assert "eval_job_id=703" in state_text
    assert "policy_id=action_visual_lora_v1" in state_text
    assert "training_profile=no_arrow_treatment" in state_text


def test_action_visual_launcher_bash_syntax_is_valid():
    if os.name == "nt":
        pytest.skip("bash path translation is unavailable on native Windows")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    script_arg = str(SCRIPT).replace("\\", "/") if os.name == "nt" else str(SCRIPT)
    result = subprocess.run([bash, "-n", script_arg], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
