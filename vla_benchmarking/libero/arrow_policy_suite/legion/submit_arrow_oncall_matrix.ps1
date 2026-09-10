param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedCommit,
    [Parameter(Mandatory = $true)][string]$RemoteRepoRoot,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Controller,
    [Parameter(Mandatory = $true)][Alias('Python')][string]$RuntimePython,
    [Parameter(Mandatory = $true)][Alias('MolmoCache')][string]$HfCache,
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [Parameter(Mandatory = $true)][string]$ArchiveRoot,
    [ValidateSet('gpu_a40','gpu_v100')][string]$Partition = 'gpu_a40',
    [ValidatePattern('^[0-9]+-[0-9]{2}:[0-5][0-9]:[0-5][0-9]$')][string]$TimeLimit = '1-00:00:00'
)

$ErrorActionPreference = 'Stop'

function Assert-SafeRemotePath([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or
        $Value -notmatch '^/[A-Za-z0-9_./-]+$' -or
        $Value -match '(^|/)\.\.(/|$)' -or $Value -eq '/') {
        throw "$Name must be a non-root absolute Linux path without traversal or shell metacharacters."
    }
}

Assert-SafeRemotePath 'RemoteRepoRoot' $RemoteRepoRoot
Assert-SafeRemotePath 'Config' $Config
Assert-SafeRemotePath 'Checkpoint' $Checkpoint
Assert-SafeRemotePath 'Controller' $Controller
Assert-SafeRemotePath 'RuntimePython' $RuntimePython
Assert-SafeRemotePath 'HfCache' $HfCache
Assert-SafeRemotePath 'RunRoot' $RunRoot
Assert-SafeRemotePath 'ArchiveRoot' $ArchiveRoot
if ($RunRoot -eq $ArchiveRoot -or $RunRoot.StartsWith("$ArchiveRoot/", [StringComparison]::Ordinal) -or
    $ArchiveRoot.StartsWith("$RunRoot/", [StringComparison]::Ordinal)) {
    throw 'RunRoot and ArchiveRoot must be disjoint.'
}

$timeParts = $TimeLimit -split '[-:]'
$timeSeconds = ([int64]$timeParts[0] * 86400) + ([int64]$timeParts[1] * 3600) +
    ([int64]$timeParts[2] * 60) + [int64]$timeParts[3]
if ($timeSeconds -lt 1 -or $timeSeconds -gt (7 * 86400)) { throw 'TimeLimit must be between 00:00:01 and 7-00:00:00.' }

$job = Join-Path $PSScriptRoot 'run_arrow_oncall_matrix.sbatch'
if (-not (Test-Path -LiteralPath $job -PathType Leaf)) { throw "Missing launcher: $job" }

# Values are restricted above before interpolation into this short remote
# command.  The release, model, controller, and cache are all explicit; no
# moving branch or ambient login-shell default is allowed into the job.
$remoteLines = @(
    'set -euo pipefail'
    # sbatch --export=ALL is used for the sealed input paths, but ambient
    # policy controls must never leak into this matrix.  Remove every prior
    # Any inherited ARROW_* policy control (including ARROW_ONCALL_*) and the
    # legacy SmolVLA policy selector must be removed before the sealed matrix
    # allow-list is exported below.
    'while IFS= read -r name; do case "$name" in ARROW_*|SMOLVLA_BASE_POLICY) unset "$name" ;; esac; done < <(compgen -v)'
    "export ARROW_ONCALL_REPO_ROOT='$RemoteRepoRoot'"
    "export ARROW_ONCALL_EXPECTED_COMMIT='$ExpectedCommit'"
    "export ARROW_ONCALL_CONFIG='$Config'"
    "export ARROW_ONCALL_CHECKPOINT='$Checkpoint'"
    "export ARROW_ONCALL_CONTROLLER='$Controller'"
    "export ARROW_ONCALL_RUNTIME_PYTHON='$RuntimePython'"
    "export ARROW_ONCALL_HF_CACHE='$HfCache'"
    "export ARROW_ONCALL_RUN_ROOT='$RunRoot'"
    "export ARROW_ONCALL_ARCHIVE_ROOT='$ArchiveRoot'"
    "cd '$RemoteRepoRoot'"
    'test "$(git rev-parse HEAD)" = "$ARROW_ONCALL_EXPECTED_COMMIT"'
    'test -z "$(git status --porcelain --untracked-files=all)"'
    'test -f "$ARROW_ONCALL_CONFIG"'
    'test -e "$ARROW_ONCALL_CHECKPOINT" && test ! -L "$ARROW_ONCALL_CHECKPOINT"'
    'test -f "$ARROW_ONCALL_CONTROLLER"'
    'test -x "$ARROW_ONCALL_RUNTIME_PYTHON"'
    'test -d "$ARROW_ONCALL_HF_CACHE"'
    'unset ARROW_ONCALL_SINGLE_TASK_ID'
    'test ! -e "$ARROW_ONCALL_RUN_ROOT"'
    'test ! -e "$ARROW_ONCALL_ARCHIVE_ROOT"'
    "test -f 'vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_oncall_matrix.sbatch'"
    "bash -n 'vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_oncall_matrix.sbatch'"
    "sbatch_args=(--parsable --partition='$Partition' --time='$TimeLimit' --export=ALL)"
    'array_mode=two_lane'
    'job_id="$(sbatch "${sbatch_args[@]}" vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_oncall_matrix.sbatch)" || { rc=$?; printf "sbatch failed (rc=%s)\n" "$rc" >&2; exit "$rc"; }'
    '[[ "$job_id" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]] || { printf "sbatch returned an invalid job id: %s\n" "$job_id" >&2; exit 1; }'
    'printf "job_id=%s mode=%s\n" "$job_id" "$array_mode"'
)
$remote = ($remoteLines -join "`n") + "`n"

& "$PSScriptRoot\..\..\..\..\.codex\legion-local\Invoke-Legion.ps1" -RemoteCommand $remote
