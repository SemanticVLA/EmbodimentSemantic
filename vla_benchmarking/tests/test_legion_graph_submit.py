from pathlib import Path
import os
import json
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "submit_legion_graph_pilot.sh"


def test_graph_pilot_submitter_has_sealed_chain_and_no_arrow_graph_run():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--expected-commit" in text
    assert re.search(r"EXPECTED_REPO_COMMIT=''", text)
    assert "EXPECTED_REPO_COMMIT='2f86a5a2b0bd13bc48b829192e481e982dbd5ee2'" not in text
    assert "EXPECTED_REPO_COMMIT='${EXPECTED_REPO_COMMIT}'" in text
    assert "ACTION\" == setup" in text
    assert "ACTION\" == launch" in text
    assert "LIBERO_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'" in text
    assert "BASE_POLICY_REVISION='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'" in text
    assert "SCRATCH_ROOT='/mnt/beegfs/hjaber/EmbodimentSemantic_runtime'" in text
    assert "--partition=gpu_a40" in text
    assert "--exclude=compute-4-13" in text
    assert "--dependency=afterok:$setup_job_id" in text
    assert "--dependency=afterok:$train_job_id" in text
    assert "setup-only" not in text.lower() or "setup/smoke" in text
    assert "sacct -j \"$setup_job_id\"" in text
    assert "scontrol show job -o \"$setup_job_id\"" in text
    assert "JobState=COMPLETED" in text
    assert "ExitCode=0:0" in text
    assert "setup_state_source='scontrol'" in text
    assert "setup_state_source='durable_state'" in text
    assert "git -C \"$REPO\" merge-base --is-ancestor \"$EXPECTED_REPO_COMMIT\" \"$CURRENT_REPO_COMMIT\"" in text
    assert '"$(state_value setup_status)" == OK' in text
    assert '"$setup_archive_status" == VERIFIED' in text
    assert "-graph96-v2" in text
    assert "-graph96\"" not in text
    assert 'input_bundle_status="$(state_value input_bundle_status)"' in text
    assert "TRAINING_PROFILE=graph_treatment" in text
    assert "TRAINING_MODE=full" in text
    assert "BATCH_SIZE=32 SEED=1000 PEFT_R=16" in text
    assert "29190/pretrained_model" in text
    assert "--mode convert-graph-pair" in text
    assert "--mode verify-graph" in text
    assert "--mode preflight --data-dir" in text
    assert "--mode preflight-graph --data-dir" in text
    assert text.index("--mode verify --data-dir") < text.index("--mode convert-graph-pair")
    assert text.index("--mode convert-graph-pair") < text.index("--mode verify-graph")
    setup_start = text.index('for p in "\\$LIBERO_DATA_DIR"')
    setup_end = text.index("module purge; module load", setup_start)
    setup_prerequisites = text[setup_start:setup_end]
    assert "\\$DATA_ROOT/graph_treatment" not in setup_prerequisites
    assert "\\$DATA_ROOT/arrow_graph_treatment" not in setup_prerequisites
    assert "\\$DATA_ROOT/sealed_lora_graph_pair_manifest.json" not in setup_prerequisites
    assert "graph_artifact_count" in text
    assert "partial graph dataset/pair artifacts found" in text
    assert "--prepare-graph-policy" in text
    assert "--seeds 1000 --episodes 10 --batch-size 1" in text
    assert "--videos --max-videos 1" in text
    assert "unset EPOCHS STEPS SAVE_FREQ UPDATES_PER_EPOCH" in text
    assert "arrow_graph_treatment" in text  # required only as sealed pair evidence
    assert "TRAINING_PROFILE=arrow_graph_treatment" not in text
    assert "run_lora_graph_pair_eval.py" in text
    assert "trap finish EXIT" in text
    assert "BUNDLE_PARENT=\"\\$ARCHIVE_ROOT/input_bundles\"" in text
    assert "input_bundle_tree_sha256" in text
    assert "input_bundle_status=VERIFIED" in text
    assert "quota -P" not in text
    assert "HOME filesystem free-space precheck failed" in text
    assert "hdf5_source_manifest.json" in text
    assert "datasets/graph_treatment" in text
    assert "datasets/arrow_graph_treatment" in text
    assert "graph_policy/tokenizer_provenance.json" in text
    assert "inventory.sha256" in text and "tree_sha256" in text
    assert "seal_tree \"\\$INPUT_BUNDLE_PATH\" verify" in text
    assert "bundle_metadata.json" in text and "repo_commit" in text
    assert "archive_status=VERIFIED" in text
    assert "archive_tree_sha256" in text
    assert "if [[ \\$workload_rc -eq 0 && \\$archive_rc -ne 0 ]]; then workload_rc=90" in text
    assert "input_bundle_status=VERIFIED" in text
    assert "setup_archive_status" in text and "train_archive_status" in text and "eval_archive_status" in text
    assert "operator job template was tampered with after setup" in text
    assert "operator job template drifted from the sealed HOME bundle" in text
    assert "rendered_eval=\"$job_dir/eval.${train_job_id}.sbatch\"" in text
    assert "launch_lock=\"$state_file.launch.lock\"" in text
    assert "mkdir \"$launch_lock\"" in text
    assert "sed -i" not in text
    assert "eval_rendered_sha256" in text
    assert text.count("archive inventory is incomplete") >= 2
    assert "already been claimed" in text
    assert "SUBMIT_FAILED" in text
    assert 'setup_archive_status="$(state_value setup_archive_status)"' in text
    assert 'setup_archive_tree_sha256="$(state_value setup_archive_tree_sha256)"' in text
    assert "copy_required \"\\$RUNTIME/operator/jobs/\\$LABEL/train.sbatch\"" in text
    assert "copy_required \"\\$RUNTIME/operator/jobs/\\$LABEL/eval.sbatch\"" in text


def test_generated_setup_embeds_operator_job_dir_for_bundle_staging():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "STATE_FILE='${state_file}'; JOB_DIR='${job_dir}'" in text
    assert 'copy_required "\\$JOB_DIR/setup.sbatch"' in text
    assert 'copy_required "\\$JOB_DIR/train.sbatch"' in text
    assert 'copy_required "\\$JOB_DIR/eval.sbatch"' in text
    assert 'copy_required "\\$job_dir/' not in text


def test_setup_does_not_submit_training_or_evaluation():
    text = SCRIPT.read_text(encoding="utf-8")
    submit_section = text.index("# Materialize and submit only the requested stage.")
    setup_branch = text.index('if [[ "$ACTION" == setup ]]; then', submit_section)
    launch_branch = text.index("else\n", setup_branch)
    setup_text = text[setup_branch:launch_branch]
    launch_text = text[launch_branch:]
    assert 'sbatch --parsable "$job_dir/setup.sbatch"' in setup_text
    assert "--dependency=afterok:$setup_job_id" not in setup_text
    assert "--dependency=afterok:$setup_job_id" in launch_text
    assert "--dependency=afterok:$train_job_id" in launch_text


def test_setup_disables_unrelated_pytest_plugin_autoload():
    text = SCRIPT.read_text(encoding="utf-8")
    assert (
        'env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "\\$PYTHON" -m pytest -q '
        '"\\$REPO/vla_benchmarking/tests/test_terminal_reset_compensation.py"'
    ) in text


def test_eval_archive_seal_rejects_files_added_after_inventory_build():
    text = SCRIPT.read_text(encoding="utf-8")
    eval_start = text.index("# Exactly two no-arrow cells")
    eval_end = text.index("EOF\nfi", eval_start)
    eval_job = text[eval_start:eval_end]
    assert "listed = set()" in eval_job
    assert "actual = {p.relative_to(root).as_posix()" in eval_job
    assert "if actual != listed: raise SystemExit(\"archive inventory is incomplete\")" in eval_job


def test_setup_requires_live_checkpoint_and_real_libero_reset_smokes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'local raw="$1" id; id="${raw%%;*}"' in text
    assert 'git -C "\\$REPO" archive --format=tar "\\$EXPECTED_REPO_COMMIT" vla_benchmarking' in text
    assert 'SMOKE_CHECKPOINT="\\$EVIDENCE/smoke/checkpoints/000002/pretrained_model"' in text
    assert '--policy.path="\\$SMOKE_CHECKPOINT"' in text
    assert "--env.task_ids='[0,2]'" in text
    assert "--env.episode_length=2" in text
    assert 'TOKENIZER_MODEL="\\$SMOKE_CHECKPOINT/tokenizer"' in text
    assert 'scope": "actual_checkpoint_load_preprocess_cuda_inference_timeout_and_randomization"' in text
    assert 'scope": "actual_pinned_libero_terminal_reset_semantics_not_task_success"' in text
    assert 'env._env.check_success = lambda: True' in text
    assert 'dummy_action = np.asarray(lerobot_libero.get_libero_dummy_action(), dtype=np.float32)' in text
    assert 'env.step(dummy_action)' in text
    assert 'if int(env.init_state_id) != counter_before:' in text
    assert 'if set(by_task) != {0, 2}:' in text
    assert 'setup_archive_path="$ARCHIVE_ROOT/setup/${label}_${setup_job_id}"' in text
    assert 'raise SystemExit("setup archive inventory is incomplete")' in text
    assert 'upstream_step_source = inspect.getsource(lerobot_libero.LiberoEnv.step)' in text
    assert '"upstream_libero_env_step_sha256"' in text
    verify_call = 'hdf5_to_lerobot_dataset.py" --mode verify --data-dir'
    sentinel_gate = "historical pair verification did not produce its sentinel"
    assert text.index(verify_call) < text.index(sentinel_gate)

@pytest.mark.skipif(os.name == "nt", reason="fake Legion submission requires a POSIX shell")
def test_setup_with_fake_sbatch_is_single_submission_and_ambient_safe(tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive"
    (repo / ".git").mkdir(parents=True)
    (repo / "vla_benchmarking").mkdir()
    runtime.mkdir()
    scratch.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "sbatch.log"
    expected = "2f86a5a2b0bd13bc48b829192e481e982dbd5ee2"
    for old, new in (
        ("RUNTIME='/home/hjaber/EmbodimentSemantic_runtime'", f"RUNTIME='{runtime}'"),
        ("SCRATCH_ROOT='/mnt/beegfs/hjaber/EmbodimentSemantic_runtime'", f"SCRATCH_ROOT='{scratch}'"),
        ("ARCHIVE_ROOT='/home/hjaber/EmbodimentSemantic_archive'", f"ARCHIVE_ROOT='{archive}'"),
    ):
        source = source.replace(old, new)
    probe = repo / "vla_benchmarking" / "submit.sh"
    probe.write_text(source, encoding="utf-8")
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        f"if [[ \"$*\" == *'rev-parse HEAD'* ]]; then printf '%s\\n' '{expected}'; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> '{log}'\n"
        "count=$(wc -l < '" + str(log) + "')\n"
        "printf '90%s\\n' \"$count\"\n",
        encoding="utf-8",
    )
    for executable in (fake_git, fake_sbatch, probe):
        executable.chmod(0o700)
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    env["PATH"] = os.pathsep.join((str(fake_bin), env.get("PATH", "")))
    env.update({"EPOCHS": "1", "STEPS": "2", "SAVE_FREQ": "3", "UPDATES_PER_EPOCH": "4", "TRAINING_PROFILE": "arrow_graph_treatment"})
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(probe), "setup", "--expected-commit", expected],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("--parsable ") and calls[0].endswith("/setup.sbatch")
    generated = next((runtime / "operator" / "jobs").glob("*"))
    train = (generated / "train.sbatch").read_text(encoding="utf-8")
    assert "TRAINING_PROFILE=graph_treatment" in train
    assert "unset EPOCHS STEPS SAVE_FREQ UPDATES_PER_EPOCH" in train
    assert "TRAINING_PROFILE=arrow_graph_treatment" not in train


@pytest.mark.skipif(os.name == "nt", reason="tamper test requires a POSIX shell")
def test_launch_rejects_tampered_job_template_before_any_sbatch(tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive"
    (repo / ".git").mkdir(parents=True)
    (repo / "vla_benchmarking").mkdir()
    runtime.mkdir(); scratch.mkdir(); archive.mkdir()
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    expected = "2f86a5a2b0bd13bc48b829192e481e982dbd5ee2"
    for old, new in (
        ("RUNTIME='/home/hjaber/EmbodimentSemantic_runtime'", f"RUNTIME='{runtime}'"),
        ("SCRATCH_ROOT='/mnt/beegfs/hjaber/EmbodimentSemantic_runtime'", f"SCRATCH_ROOT='{scratch}'"),
        ("ARCHIVE_ROOT='/home/hjaber/EmbodimentSemantic_archive'", f"ARCHIVE_ROOT='{archive}'"),
    ):
        source = source.replace(old, new)
    probe = repo / "vla_benchmarking" / "submit.sh"; probe.write_text(source, encoding="utf-8")
    label = "legion_graph_treatment_lora_full_s1000_v1_tamper"
    state_dir = runtime / "graph_pilot" / label; state_dir.mkdir(parents=True)
    job_dir = runtime / "operator" / "jobs" / label; job_dir.mkdir(parents=True)
    bundle = archive / "input_bundles" / "bundle"; (bundle / "operator").mkdir(parents=True)
    files = {"setup.sbatch": b"setup template\n", "train.sbatch": b"train template\n", "eval.sbatch": b"eval __TRAIN_JOB_ID__ template\n"}
    import hashlib
    for name, content in files.items():
        (job_dir / name).write_bytes(content)
        (bundle / "operator" / name).write_bytes(content)
    (job_dir / "train.sbatch").write_bytes(b"tampered train template\n")
    (bundle / "tree_sha256").write_text("a" * 64 + "\n", encoding="utf-8")
    (bundle / "bundle_metadata.json").write_text(json.dumps({"repo_commit": expected}) + "\n", encoding="utf-8")
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    state = state_dir / "state.env"
    state.write_text(
        "\n".join((
            f"label={label}", f"expected_repo_commit={expected}", "setup_job_id=9001",
            "train_job_id=", "eval_job_id=", "setup_status=OK", "train_status=PENDING", "eval_status=PENDING",
            "launch_status=PENDING", "launch_nonce=", f"setup_template_sha256={hashes['setup.sbatch']}",
            f"train_template_sha256={hashes['train.sbatch']}", f"eval_template_sha256={hashes['eval.sbatch']}",
            "eval_rendered_sha256=", f"input_bundle_path={bundle}", f"input_bundle_tree_sha256={'a' * 64}",
            "input_bundle_status=VERIFIED", "setup_archive_status=VERIFIED", f"setup_archive_tree_sha256={'b' * 64}",
        )) + "\n",
        encoding="utf-8",
    )
    (fake_bin / "git").write_text(
        "#!/usr/bin/env bash\n"
        f"if [[ \"$*\" == *'rev-parse HEAD'* ]]; then printf '%s\\n' '{expected}'; fi\n",
        encoding="utf-8",
    )
    (fake_bin / "sacct").write_text("#!/usr/bin/env bash\nprintf 'COMPLETED\\n'\n", encoding="utf-8")
    sbatch_log = tmp_path / "sbatch.log"
    (fake_bin / "sbatch").write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> '{sbatch_log}'\nprintf '9901\\n'\n", encoding="utf-8")
    for path in (probe, fake_bin / "git", fake_bin / "sacct", fake_bin / "sbatch"):
        path.chmod(0o700)
    env = os.environ.copy(); env.pop("SLURM_JOB_ID", None); env["PATH"] = os.pathsep.join((str(fake_bin), env.get("PATH", "")))
    result = subprocess.run([shutil.which("bash"), str(probe), "launch", "--expected-commit", expected, "--state-file", str(state)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "tampered" in result.stderr
    assert not sbatch_log.exists()


def test_graph_pilot_submitter_is_bash_syntax_valid():
    bash = shutil.which("bash")
    if os.name == "nt":
        for candidate in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ):
            if candidate.is_file():
                bash = str(candidate)
                break
    if bash is None:
        pytest.skip("bash is unavailable on this host")
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
