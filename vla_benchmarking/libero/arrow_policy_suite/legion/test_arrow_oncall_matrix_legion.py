"""Contract tests for the Arrow OnCall Legion array wrapper.

These tests are intentionally static: submitting a SLURM job is an operator
action and must not happen during a local unit-test run.
"""

from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SBATCH = ROOT / "run_arrow_oncall_matrix.sbatch"
SUBMIT = ROOT / "submit_arrow_oncall_matrix.ps1"


def test_sbatch_is_a_single_bounded_array() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-9%2" in text
    assert text.count("#SBATCH --array=") == 1
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --partition=gpu_a40" in text
    assert "#SBATCH --time=1-00:00:00" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "-le 9" in text


def test_sbatch_seals_the_schedule_and_calls_both_matrix_commands() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for expected in (
        "export ARROW_ONCALL_EPISODES=10",
        "export ARROW_ONCALL_EPISODE_START=0",
        "export ARROW_ONCALL_SEED_BASE=1000",
        "export ARROW_ONCALL_INIT_STATE_BASE=0",
        "export ARROW_ONCALL_HORIZON=280",
        "export ARROW_ONCALL_POLICY=arrow_on_call",
        "export ARROW_ONCALL_VIDEO_EPISODE=0",
        "export ARROW_ONCALL_ARCHIVE_STATUS=\"$ARCHIVE_ROOT/matrix_status.json\"",
        "-m vla_benchmarking.libero.arrow_policy_suite.oncall_matrix run-task",
        '--config "$ARROW_ONCALL_CONFIG"',
        '--output-root "$RUN_ROOT"',
        '--task-id "$TASK_ID"',
        '--factory vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host',
        '--checkpoint "$ARROW_ONCALL_CHECKPOINT"',
        '--controller "$ARROW_ONCALL_CONTROLLER"',
        "-m vla_benchmarking.libero.arrow_policy_suite.oncall_matrix finalize",
        '--archive-root "$ARCHIVE_ROOT"',
    ):
        assert expected in text
    assert "task_rc=$?" in text and "final_rc=$?" in text
    assert "if [[ \"$task_rc\" -ne 0 ]]; then exit \"$task_rc\"; fi" in text
    assert text.count("--factory vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host") >= 3
    assert text.count('--checkpoint "$ARROW_ONCALL_CHECKPOINT"') >= 3
    assert text.count('--controller "$ARROW_ONCALL_CONTROLLER"') >= 3
    assert "automatic_ttt" not in text


def test_sbatch_has_compute_node_video_preflight_without_running_an_episode() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    for expected in (
        "torch.cuda.is_available()",
        "native_legion_factory",
        "callable(build_host)",
        "codec='libx264'",
        "pixelformat='yuv420p'",
        "imageio.mimread",
    ):
        assert expected in text
    assert "ONCALL_COMPUTE_PREFLIGHT_VERIFIED" in text


def test_sbatch_has_failure_safe_per_task_archival_and_provenance() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "trap archive_on_exit EXIT" in text
    assert "PRESERVED_FAILURE" in text
    assert "inventory.sha256" in text
    assert "experiment_evidence=false" in text
    assert "actual_commit" in text
    assert "TASK_ARCHIVE_ROOT" in text
    assert "task_rc=125" in text
    assert "ARROW_SUITE_*|ARROW_CONTROLLER_*) unset" in text
    for name in (
        "ARROW_MOLMOPOINT_MODEL",
        "ARROW_MOLMOPOINT_REVISION",
        "ARROW_GRIPPER_CLOSED_THRESHOLD",
        "ARROW_GRIPPER_OPEN_THRESHOLD",
        "SMOLVLA_BASE_POLICY",
    ):
        assert name in text
    assert "export ARROW_SUITE_SUITE_MODE=vanilla" in text
    assert "export ARROW_SUITE_RESOLUTION=256" in text
    assert "export ARROW_SUITE_TASK_ID" not in text
    assert "export ARROW_SUITE_SEED" not in text
    assert "export ARROW_SUITE_INIT_STATE_INDEX" not in text
    assert "arrow_policy_suite.oncall_worker_terminal.v1" in text
    assert '"task_status": sys.argv[3]' in text
    assert '"task_exit_code": int(sys.argv[4])' in text
    assert '"workload_exit_code": int(sys.argv[5])' in text
    assert '"task_id": int(sys.argv[2])' in text
    assert 'path.open("x"' not in text
    assert 'path.open("w"' in text
    assert 'os.fsync(handle.fileno())' in text
    assert 'mktemp "$status_root/.${TASK_NAME}.terminal.XXXXXX"' in text
    assert 'ln -- "$temp_path" "$status_path"' in text
    assert '> "$status_path"' not in text
    assert 'worker status schema mismatch' in text
    assert 'finalize_log="$RUN_ROOT/finalize_logs/$TASK_NAME.log"' in text
    assert '>> "$finalize_log"' in text
    assert 'cp -p -- "$finalize_log"' in text
    assert '|| true' in text
    assert 'archive_status=PRESERVED_FAILURE' in text
    assert '"$archive_rc" -eq 0' in text
    assert '-s "$TASK_ARCHIVE_ROOT/task_summary.json"' in text
    assert '-s "$TASK_ARCHIVE_ROOT/worker_status/$TASK_NAME.json"' in text
    for expected in (
        "! -L \"$ARROW_ONCALL_CHECKPOINT\"",
        "pinned SmolVLM snapshot",
        "pinned MolmoPoint snapshot",
        "MOLMO_REVISION='188130f961c8e0888a34e11121a1423c461a01ba'",
        "SMOLVLM_REVISION='7b375e1b73b11138ff12fe22c8f2822d8fe03467'",
        "MODEL_CACHE_ROOT=\"$RUN_ROOT/model_cache/$TASK_NAME/hub\"",
        "archive seal is missing",
        "task_summary.json",
    ):
        assert expected in text


def test_submit_front_door_requires_explicit_immutable_inputs() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    for name in (
        "ExpectedCommit",
        "RemoteRepoRoot",
        "Config",
        "Checkpoint",
        "Controller",
        "RuntimePython",
        "HfCache",
        "RunRoot",
        "ArchiveRoot",
    ):
        assert re.search(rf"\[Parameter\(Mandatory = \$true\)\].*\[string\]\${name}\b", text)
    assert "--array=0-9%2" in text
    assert "ARROW_*|SMOLVLA_BASE_POLICY) unset" in text
    assert "ARROW_ONCALL_" in text
    assert "--export=ALL" in text
    assert "bash -n" in text
    assert "automatic_ttt" not in text


def test_bash_syntax_when_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        return
    # Git-for-Windows' bash cannot consume a native drive-letter path.  Only
    # run this check when a POSIX path is available (or when wslpath can make
    # one); the launcher itself is tested statically on Windows.
    script_path = str(SBATCH)
    if re.match(r"^[A-Za-z]:[\\/]", script_path):
        wslpath = shutil.which("wslpath")
        if wslpath is None:
            return
        converted = subprocess.run([wslpath, "-a", script_path], capture_output=True, text=True)
        if converted.returncode != 0:
            return
        script_path = converted.stdout.strip()
    result = subprocess.run([bash, "-n", script_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_powershell_parses_when_available() -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        return
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", f"& {str(SUBMIT)!r} -?"],
        capture_output=True,
        text=True,
    )
    # -? prints help and exits successfully on supported PowerShell versions.
    assert result.returncode == 0, result.stderr
