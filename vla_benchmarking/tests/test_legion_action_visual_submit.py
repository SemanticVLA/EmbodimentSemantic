from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys

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
    assert "historical_action_only_lora_v1_no_arrows" in text
    assert "action_visual_lora_v1_no_arrows" in text
    assert "legacy_action_only_evidence_v1" in text
    assert "legacy_action_only_evidence.py\" build" in text
    assert "legacy_action_only_evidence.py\" validate" in text
    assert "--action-only-legacy-evidence-bundle" in text
    assert "--training-data-root" in text
    assert "--episodes 10 --batch-size 1 --device cuda" in text
    assert "VISUAL_CONDITION=none" in text and "VISUAL_ARROWS=0" in text
    assert "mkdir \"$LAUNCH_LOCK\"" in text
    assert text.count("status --porcelain --untracked-files=all") >= 3
    assert "setup_archive_tree_sha256" in text and "train_archive_tree_sha256" in text and "eval_archive_tree_sha256" in text
    assert "archive inventory is incomplete" in text
    assert 'local raw="$1"; local id="${raw%%;*}"' in text


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
    eval_start = text.index("# Matched clean/no-arrow evaluation")
    eval_end = text.index("\nEOF", eval_start)
    eval_text = text[eval_start:eval_end]
    assert "LIBERO_DIR='${LIBERO_DIR}'" in eval_text
    assert 'git -C "\\$LIBERO_DIR" rev-parse HEAD' in eval_text
    assert 'git -C "\\$LIBERO_DIR" status --porcelain --untracked-files=no' in eval_text
    assert 'export LIBERO_DIR="\\$LIBERO_DIR"' in eval_text
    assert "DATA_ROOT='${DATA_ROOT}'" in eval_text
    assert "BASE_POLICY_REVISION='${BASE_POLICY_REVISION}'" in eval_text
    libero_assignment = eval_text.index("LIBERO_DIR='${LIBERO_DIR}'")
    first_libero_reference = eval_text.index('\\$LIBERO_DIR', libero_assignment + len("LIBERO_DIR='${LIBERO_DIR}'"))
    assert libero_assignment < first_libero_reference
    assert "copy_tree \"\\$EVIDENCE\" \"\\$ARCHIVE_DIR/evidence\"" in eval_text
    assert "if [[ \\$arc -eq 0 && -d \"\\$EVAL_ROOT\" ]]; then" in eval_text
    assert eval_text.index("ARCHIVE_DIR/evidence") < eval_text.index("ARCHIVE_DIR/results")
    assert "seal_tree \"\\$ARCHIVE_DIR\" build" in eval_text
    assert "seal_tree \"\\$ARCHIVE_DIR\" verify" in eval_text
    assert "historical_action_only_lora_v1_no_arrows" in eval_text
    assert "historical action-only retrospective contract" in eval_text
    assert "legacy evidence bundle is missing" in eval_text
    assert "TRAINING_PROFILE=no_arrow_treatment" in text
    assert "--action-only-checkpoint" in text and "--action-visual-checkpoint" in text
    assert "--action-only-training-manifest" in text and "--action-visual-training-manifest" in text
    assert "--action-only-legacy-evidence-bundle \"\\$LEGACY_EVIDENCE_BUNDLE\"" in eval_text
    assert "--training-data-root \"\\$DATA_ROOT\"" in eval_text
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


def test_setup_archive_hash_uses_verified_tree_value():
    text = SCRIPT.read_text(encoding="utf-8")
    setup_start = text.index("# Candidate policy validation")
    setup_end = text.index("\nEOF", setup_start)
    setup_text = text[setup_start:setup_end]
    assert "local rc=\\$? arc=0 tree=''" in setup_text
    assert 'if [[ \\$arc -eq 0 ]]; then tree="\\$(tr -d \'[:space:]\' < "\\$ARCHIVE_DIR/tree_sha256")"; fi' in setup_text
    assert "setup_archive_tree_sha256=%s" in setup_text


def test_generated_jobs_let_the_training_launcher_resolve_policy_regex():
    text = SCRIPT.read_text(encoding="utf-8")
    setup_start = text.index('cat > "$JOB_DIR/setup.sbatch"')
    train_start = text.index('cat > "$JOB_DIR/train.sbatch"')
    eval_start = text.index('cat > "$JOB_DIR/eval.sbatch"')
    setup_text = text[setup_start:train_start]
    train_text = text[train_start:eval_start]
    assert "FINETUNING_POLICY_ID=action_visual_lora_v1" in setup_text
    assert "FINETUNING_POLICY_ID=action_visual_lora_v1" in train_text
    assert "POLICY_TARGET_REGEX=" not in setup_text
    assert "POLICY_TARGET_REGEX=" not in train_text


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
        f"if [[ \"$*\" == *'rev-parse HEAD'* ]]; then if [[ \"${{FORCE_SETUP_FAIL:-}}\" == 1 ]]; then printf '%s\\n' '{'b' * 40}'; else printf '%s\\n' '{expected}'; fi; fi\n",
        encoding="utf-8",
    )
    (fake_bin / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        "n=$(wc -l < '" + str(log) + "')\n"
        "printf '70%s;fakecluster\\n' \"$n\"\n",
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

    setup_source = runtime / "operator" / "jobs" / "fake_action_visual" / "setup.sbatch"
    syntax = subprocess.run(
        [shutil.which("bash"), "-n", str(setup_source)],
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    rendered_setup = tmp_path / "setup.rendered.sbatch"
    rendered_setup.write_text(
        setup_source.read_text(encoding="utf-8").replace(
            "/home/hjaber/.conda/envs/embodiment-smolvla-py312/bin/python", sys.executable
        ),
        encoding="utf-8",
    )
    failed_setup = subprocess.run(
        [shutil.which("bash"), str(rendered_setup)],
        env={**env, "SLURM_JOB_ID": "701", "FORCE_SETUP_FAIL": "1"},
        capture_output=True,
        text=True,
    )
    assert failed_setup.returncode == 90, failed_setup.stderr
    archive = archive / "setup" / "fake_action_visual_701"
    inventory = archive / "inventory.sha256"
    tree = archive / "tree_sha256"
    assert inventory.is_file() and tree.is_file()
    assert tree.read_text(encoding="utf-8").strip() == hashlib.sha256(inventory.read_bytes()).hexdigest()
    state_text = state.read_text(encoding="utf-8")
    assert "setup_archive_status=VERIFIED" in state_text
    assert "setup_status=FAILED" in state_text


def test_action_visual_launcher_bash_syntax_is_valid():
    if os.name == "nt":
        pytest.skip("bash path translation is unavailable on native Windows")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    script_arg = str(SCRIPT).replace("\\", "/") if os.name == "nt" else str(SCRIPT)
    result = subprocess.run([bash, "-n", script_arg], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
