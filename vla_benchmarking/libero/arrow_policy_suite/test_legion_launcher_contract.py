from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parent
SUBMIT = (ROOT / "legion" / "submit_arrow_policy_suite_canary.ps1").read_text(encoding="utf-8")
SBATCH = (ROOT / "legion" / "run_arrow_policy_suite_canary.sbatch").read_text(encoding="utf-8")


def test_submit_uses_immutable_remote_repo_release_and_linux_paths():
    assert "$RemoteRepoRoot" in SUBMIT
    assert "EmbodimentSemantic_runtime/releases/" in SUBMIT
    assert 'cd "$ARROW_SUITE_REPO_ROOT"' in SUBMIT
    assert "RemoteConfig must be a safe absolute Linux path" in SUBMIT
    assert "RunRoot and ArchiveRoot must be safe absolute Linux paths" in SUBMIT
    assert "[IO.Path]::IsPathRooted($RunRoot)" not in SUBMIT


def test_submit_rejects_shell_injection_before_interpolation():
    for value in ("RemoteRepoRoot", "RemoteConfig", "RunRoot", "ArchiveRoot", "Checkpoint", "Controller", "GraphContextRevision"):
        assert value in SUBMIT
    assert "contains unsafe shell characters" in SUBMIT
    assert "must be a safe absolute Linux path" in SUBMIT


def test_submit_builds_remote_command_from_joined_lines():
    """The PowerShell launcher must interpolate values before sending SSH text."""
    assert "$remoteLines = @(" in SUBMIT
    assert '$remote = ($remoteLines -join "`n") + "`n"' in SUBMIT
    assert "$remote = @'" not in SUBMIT
    assert "' + $repoExport + @'" not in SUBMIT
    assert '"export ARROW_SUITE_EXPECTED_COMMIT=\'$ExpectedCommit\'"' in SUBMIT


def test_sbatch_resolves_array_log_name_and_has_engineering_smoke_marker():
    assert 'SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID' in SBATCH
    assert "ARROW_SUITE_ENGINEERING_SMOKE" in SBATCH
    assert '"experiment_evidence": False' in SBATCH
    assert "arrow_policy_suite.engineering_smoke.v1" in SBATCH
