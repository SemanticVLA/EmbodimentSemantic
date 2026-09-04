from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


_BASH = r"C:\Program Files\Git\bin\bash.exe" if os.path.isfile(r"C:\Program Files\Git\bin\bash.exe") else shutil.which("bash")
_SBATCH = Path(__file__).parents[1] / "legion" / "v9d_molmo_campaign.sbatch"


def _source_slice(start_marker: str, end_marker: str) -> str:
    source = _SBATCH.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _run_bash(script: str, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if _BASH is None:
        pytest.skip("bash is required for launcher seam validation")
    return subprocess.run(
        [_BASH, "--noprofile", "--norc", "-uc", script],
        cwd=cwd, env=env, text=True, capture_output=True,
    )


def test_handoff_selector_context_and_dispatch_are_model_free(tmp_path):
    selector = _source_slice(
        'CAMPAIGN_SCRIPT="$REPO_ROOT/vla_benchmarking/run_v9d_molmo_campaign.py"',
        'REQUIREMENTS_FILE=',
    )
    context = _source_slice(
        'opening_probe="${V9D_MOLMO_OPENING_PROBE:-0}"',
        "\nEOF",
    ) + "\nEOF"
    dispatch = _source_slice(
        'if [[ "${V9D_MOLMO_GEOMETRY_PROBE:-0}" == 1 ||',
        '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"',
    ) + '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"'
    env = {
        "PATH": os.environ.get("PATH", ""), "REPO_ROOT": "/release",
        "V9D_MOLMO_HANDOFF_PROBE": "1", "PYTHON": "/bin/echo",
        "OUTPUT_ROOT": "/tmp/output", "CANARY_LABEL": "handoff",
        "RUN_ROOT": ".", "SLURM_JOB_ID": "1", "SLURM_JOB_NAME": "job",
        "CANARY_EXPECTED_COMMIT": "a" * 40, "actual_commit": "a" * 40,
        "CAMPAIGN_SCRIPT": "/release/vla_benchmarking/run_molmo_opening_probe.py",
        "campaign_sha256": "b" * 64, "REQUIREMENTS_FILE": "/release/req",
        "requirements_sha256": "c" * 64, "BASE_PYTHON": "/python",
        "RUNTIME_ROOT": "/tmp/runtime", "ARCHIVE_ROOT": "/tmp/archive",
        "HF_HOME": "/tmp/hf", "setup_started_epoch": "1",
        "setup_finished_epoch": "2",
    }
    selector_result = _run_bash(
        'set -Eeuo pipefail; ' + selector + 'printf "selected=%s\\n" "$CAMPAIGN_SCRIPT"',
        cwd=tmp_path, env=env,
    )
    assert selector_result.returncode == 0, selector_result.stderr
    assert "selected=/release/vla_benchmarking/run_molmo_opening_probe.py" in selector_result.stdout

    context_result = _run_bash(
        "set -Eeuo pipefail; " + context + '\necho context; cat "$RUN_ROOT/job_context.env"',
        cwd=tmp_path, env=env,
    )
    assert context_result.returncode == 0, context_result.stderr
    rendered = context_result.stdout
    assert "model_load_count_expected=0" in rendered
    assert "handoff_probe=1" in rendered
    assert "observation_profile=hover20mm" in rendered
    assert "motion_profile=not_applicable_robot_only_diagnostic" in rendered
    assert "if [[" not in rendered

    dispatch_result = _run_bash(
        "set -Eeuo pipefail; " + dispatch,
        cwd=tmp_path, env=env,
    )
    assert dispatch_result.returncode == 0, dispatch_result.stderr
    assert "/release/vla_benchmarking/run_molmo_opening_probe.py --output-dir /tmp/output --label handoff" in dispatch_result.stdout


def test_settled_opening_selector_context_and_dispatch_forces_paired_profiles(tmp_path):
    selector = _source_slice(
        'CAMPAIGN_SCRIPT="$REPO_ROOT/vla_benchmarking/run_v9d_molmo_campaign.py"',
        'REQUIREMENTS_FILE=',
    )
    context = _source_slice(
        'opening_probe="${V9D_MOLMO_OPENING_PROBE:-0}"',
        "\nEOF",
    ) + "\nEOF"
    dispatch = _source_slice(
        'if [[ "${V9D_MOLMO_GEOMETRY_PROBE:-0}" == 1 ||',
        '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"',
    ) + '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"'
    env = {
        "PATH": os.environ.get("PATH", ""), "REPO_ROOT": "/release",
        "V9D_MOLMO_SETTLED_OPENING_PROBE": "1", "PYTHON": "/bin/echo",
        "settled_opening_probe": "1", "parked_opening_probe": "0", "opening_probe": "0",
        "OUTPUT_ROOT": "/tmp/output", "CANARY_LABEL": "settled",
        "RUN_ROOT": ".", "SLURM_JOB_ID": "1", "SLURM_JOB_NAME": "job",
        "CANARY_EXPECTED_COMMIT": "a" * 40, "actual_commit": "a" * 40,
        "CAMPAIGN_SCRIPT": "/release/vla_benchmarking/run_v9d_molmo_campaign.py",
        "campaign_sha256": "b" * 64, "REQUIREMENTS_FILE": "/release/req",
        "requirements_sha256": "c" * 64, "BASE_PYTHON": "/python",
        "RUNTIME_ROOT": "/tmp/runtime", "ARCHIVE_ROOT": "/tmp/archive",
        "HF_HOME": "/tmp/hf", "setup_started_epoch": "1",
        "setup_finished_epoch": "2",
    }
    selector_result = _run_bash(
        'set -Eeuo pipefail; ' + selector + 'printf "selected=%s\\n" "$CAMPAIGN_SCRIPT"',
        cwd=tmp_path, env=env,
    )
    assert selector_result.returncode == 0, selector_result.stderr
    assert "selected=/release/vla_benchmarking/run_v9d_molmo_campaign.py" in selector_result.stdout

    context_result = _run_bash(
        "set -Eeuo pipefail; " + context + '\necho context; cat "$RUN_ROOT/job_context.env"',
        cwd=tmp_path, env=env,
    )
    assert context_result.returncode == 0, context_result.stderr
    rendered = context_result.stdout
    assert "model_load_count_expected=1" in rendered
    assert "settled_opening_probe=1" in rendered
    assert "observation_profile=hover20mm" in rendered
    assert "motion_profile=release_plus20mm" in rendered
    assert "if [[" not in rendered

    dispatch_result = _run_bash(
        "set -Eeuo pipefail; die(){ printf '%s\\n' \"$*\" >&2; exit 2; }; " + dispatch,
        cwd=tmp_path, env=env,
    )
    assert dispatch_result.returncode == 0, dispatch_result.stderr
    assert "/release/vla_benchmarking/run_v9d_molmo_campaign.py --output-dir /tmp/output --label settled --settled-opening-probe --observation-profile hover20mm --motion-profile release_plus20mm" in dispatch_result.stdout


def test_parked_opening_selector_context_and_dispatch_forces_no_hover(tmp_path):
    selector = _source_slice(
        'CAMPAIGN_SCRIPT="$REPO_ROOT/vla_benchmarking/run_v9d_molmo_campaign.py"',
        'REQUIREMENTS_FILE=',
    )
    context = _source_slice(
        'opening_probe="${V9D_MOLMO_OPENING_PROBE:-0}"',
        "\nEOF",
    ) + "\nEOF"
    dispatch = _source_slice(
        'if [[ "${V9D_MOLMO_GEOMETRY_PROBE:-0}" == 1 ||',
        '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"',
    ) + '"$PYTHON" "$CAMPAIGN_SCRIPT" "${campaign_args[@]}"'
    env = {
        "PATH": os.environ.get("PATH", ""), "REPO_ROOT": "/release",
        "V9D_MOLMO_PARKED_OPENING_PROBE": "1", "PYTHON": "/bin/echo",
        "parked_opening_probe": "1", "opening_probe": "0", "settled_opening_probe": "0",
        "OUTPUT_ROOT": "/tmp/output", "CANARY_LABEL": "parked", "RUN_ROOT": ".",
        "SLURM_JOB_ID": "1", "SLURM_JOB_NAME": "job", "CANARY_EXPECTED_COMMIT": "a" * 40,
        "actual_commit": "a" * 40, "CAMPAIGN_SCRIPT": "/release/vla_benchmarking/run_v9d_molmo_campaign.py",
        "campaign_sha256": "b" * 64, "REQUIREMENTS_FILE": "/release/req", "requirements_sha256": "c" * 64,
        "BASE_PYTHON": "/python", "RUNTIME_ROOT": "/tmp/runtime", "ARCHIVE_ROOT": "/tmp/archive",
        "HF_HOME": "/tmp/hf", "setup_started_epoch": "1", "setup_finished_epoch": "2",
    }
    selector_result = _run_bash(
        'set -Eeuo pipefail; ' + selector + 'printf "selected=%s\\n" "$CAMPAIGN_SCRIPT"',
        cwd=tmp_path, env=env,
    )
    assert selector_result.returncode == 0, selector_result.stderr
    assert "selected=/release/vla_benchmarking/run_v9d_molmo_campaign.py" in selector_result.stdout
    context_result = _run_bash(
        "set -Eeuo pipefail; " + context + '\necho context; cat "$RUN_ROOT/job_context.env"',
        cwd=tmp_path, env=env,
    )
    assert context_result.returncode == 0, context_result.stderr
    rendered = context_result.stdout
    assert "model_load_count_expected=1" in rendered
    assert "parked_opening_probe=1" in rendered
    assert "observation_profile=parked" in rendered
    assert "motion_profile=release_plus20mm" in rendered
    assert "if [[" not in rendered
    dispatch_result = _run_bash(
        "set -Eeuo pipefail; die(){ printf '%s\\n' \"$*\" >&2; exit 2; }; " + dispatch,
        cwd=tmp_path, env=env,
    )
    assert dispatch_result.returncode == 0, dispatch_result.stderr
    assert "/release/vla_benchmarking/run_v9d_molmo_campaign.py --output-dir /tmp/output --label parked --parked-opening-probe --observation-profile parked --motion-profile release_plus20mm" in dispatch_result.stdout


@pytest.mark.parametrize(
    "incompatible",
    ("V9D_MOLMO_OPENING_PROBE", "V9D_MOLMO_SETTLED_OPENING_PROBE", "V9D_MOLMO_HANDOFF_PROBE",
     "V9D_MOLMO_MOTION_PROBE", "V9D_MOLMO_REPAIR_GATE", "V9D_MOLMO_GEOMETRY_PROBE",
     "V9D_MOLMO_SCREEN_ONLY", "V9D_MOLMO_ARMS"),
)
def test_parked_opening_rejects_incompatible_modes_before_setup(tmp_path, incompatible):
    validation = _source_slice(
        'case "${V9D_MOLMO_HANDOFF_PROBE:-0}" in',
        "command -v realpath",
    )
    env = {"PATH": os.environ.get("PATH", ""), "V9D_MOLMO_PARKED_OPENING_PROBE": "1", incompatible: "1"}
    result = _run_bash(
        "set -Eeuo pipefail; die(){ printf '%s\\n' \"$*\" >&2; exit 2; }; "
        + validation + "; echo SETUP_REACHED", cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert "SETUP_REACHED" not in result.stdout


@pytest.mark.parametrize(
    "incompatible",
    ("V9D_MOLMO_OPENING_PROBE", "V9D_MOLMO_MOTION_PROBE", "V9D_MOLMO_REPAIR_GATE",
     "V9D_MOLMO_GEOMETRY_PROBE", "V9D_MOLMO_SCREEN_ONLY", "V9D_MOLMO_ARMS"),
)
def test_handoff_rejects_incompatible_modes_before_setup(tmp_path, incompatible):
    validation = _source_slice(
        'case "${V9D_MOLMO_HANDOFF_PROBE:-0}" in',
        "command -v realpath",
    )
    env = {"PATH": os.environ.get("PATH", ""), "V9D_MOLMO_HANDOFF_PROBE": "1", incompatible: "1"}
    result = _run_bash(
        "set -Eeuo pipefail; die(){ printf '%s\\n' \"$*\" >&2; exit 2; }; "
        + validation + "; echo SETUP_REACHED",
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert "SETUP_REACHED" not in result.stdout


@pytest.mark.parametrize(
    "incompatible",
    ("V9D_MOLMO_OPENING_PROBE", "V9D_MOLMO_HANDOFF_PROBE", "V9D_MOLMO_MOTION_PROBE",
     "V9D_MOLMO_REPAIR_GATE", "V9D_MOLMO_GEOMETRY_PROBE", "V9D_MOLMO_SCREEN_ONLY",
     "V9D_MOLMO_ARMS"),
)
def test_settled_opening_rejects_incompatible_modes_before_setup(tmp_path, incompatible):
    validation = _source_slice(
        'case "${V9D_MOLMO_HANDOFF_PROBE:-0}" in',
        "command -v realpath",
    )
    env = {"PATH": os.environ.get("PATH", ""), "V9D_MOLMO_SETTLED_OPENING_PROBE": "1", incompatible: "1"}
    result = _run_bash(
        "set -Eeuo pipefail; die(){ printf '%s\\n' \"$*\" >&2; exit 2; }; "
        + validation + "; echo SETUP_REACHED",
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 2
    assert "SETUP_REACHED" not in result.stdout


@pytest.mark.parametrize(
    ("handoff", "expected_model_count", "expected_obs", "expected_motion"),
    (("0", "1", "baseline", "baseline"), ("1", "0", "hover20mm", "not_applicable_robot_only_diagnostic")),
)
def test_job_context_defaults_and_handoff_profile_are_distinct(
    tmp_path, handoff, expected_model_count, expected_obs, expected_motion,
):
    context = _source_slice(
        'opening_probe="${V9D_MOLMO_OPENING_PROBE:-0}"',
        "\nEOF",
    ) + "\nEOF"
    env = {
        "PATH": os.environ.get("PATH", ""), "V9D_MOLMO_HANDOFF_PROBE": handoff,
        "RUN_ROOT": ".", "SLURM_JOB_ID": "1", "SLURM_JOB_NAME": "job",
        "CANARY_LABEL": "label", "CANARY_EXPECTED_COMMIT": "a" * 40,
        "actual_commit": "a" * 40, "REPO_ROOT": "/release",
        "CAMPAIGN_SCRIPT": "/release/campaign.py", "campaign_sha256": "b" * 64,
        "REQUIREMENTS_FILE": "/release/req", "requirements_sha256": "c" * 64,
        "BASE_PYTHON": "/python", "PYTHON": "/python", "RUNTIME_ROOT": "/tmp/runtime",
        "OUTPUT_ROOT": "/tmp/output", "ARCHIVE_ROOT": "/tmp/archive", "HF_HOME": "/tmp/hf",
        "setup_started_epoch": "1", "setup_finished_epoch": "2",
    }
    result = _run_bash(
        "set -Eeuo pipefail; " + context + '\necho context; cat "$RUN_ROOT/job_context.env"',
        cwd=tmp_path, env=env,
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout
    assert f"model_load_count_expected={expected_model_count}" in rendered
    assert f"observation_profile={expected_obs}" in rendered
    assert f"motion_profile={expected_motion}" in rendered
