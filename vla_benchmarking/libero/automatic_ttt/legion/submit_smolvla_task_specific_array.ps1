[CmdletBinding()]
param(
    [ValidateSet('preflight', 'dry-run', 'submit')]
    [string]$Mode = 'preflight',
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedCommit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ControllerConfigHash,
    [string]$RemoteRepoRoot = '',
    [string]$RemoteRunRoot = '',
    [string]$RemoteArchiveRoot = '',
    [string]$CanaryRunRoot = '',
    [string]$CanaryArchiveRoot = '',
    [ValidatePattern('^[A-Za-z0-9_.-]{1,72}$')]
    [string]$Label = 'smolvla_task_specific_arrow_peft',
    [ValidatePattern('^[A-Za-z0-9_.-]{1,96}$')]
    [string]$JobName = 'smolvla_task_specific',
    [switch]$ConfirmExpensiveRun
)

$ErrorActionPreference = 'Stop'
$repo = if ($RemoteRepoRoot) { $RemoteRepoRoot } else { "/home/hjaber/EmbodimentSemantic_releases/$ExpectedCommit" }
$runRoot = if ($RemoteRunRoot) { $RemoteRunRoot } else { "/mnt/beegfs/hjaber/EmbodimentSemantic_runtime/peft_arrow/$Label" }
$archiveRoot = if ($RemoteArchiveRoot) { $RemoteArchiveRoot } else { "/home/hjaber/EmbodimentSemantic_archive/peft_arrow/$Label" }
$canaryRunRoot = if ($CanaryRunRoot) { $CanaryRunRoot } else { "${runRoot}_canary" }
$canaryArchiveRoot = if ($CanaryArchiveRoot) { $CanaryArchiveRoot } else { "${archiveRoot}_canary" }
$runnerRelative = 'vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_all_tasks.sbatch'
$canaryRelative = 'vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_arrow_collector_canary.sbatch'
$invokeLegion = Join-Path $PSScriptRoot '../../../../.codex/legion-local/Invoke-Legion.ps1'

function Assert-RemotePath {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value) -or -not $Value.StartsWith('/') -or
        $Value -eq '/' -or $Value.Contains("'") -or $Value.Contains('"') -or
        $Value -match '(^|/)\.\.?(/|$)' -or $Value -match '[\r\n,\s=]') {
        throw "$Name must be an absolute, shell-safe non-root path: $Value"
    }
}
foreach ($item in @(
    @('RemoteRepoRoot', $repo), @('RemoteRunRoot', $runRoot),
    @('RemoteArchiveRoot', $archiveRoot), @('CanaryRunRoot', $canaryRunRoot),
    @('CanaryArchiveRoot', $canaryArchiveRoot)
)) { Assert-RemotePath $item[0] $item[1] }
if ($repo -notmatch "/EmbodimentSemantic_releases/$ExpectedCommit$") {
    throw "RemoteRepoRoot must end in exact immutable release $ExpectedCommit"
}
if ($runRoot -eq $archiveRoot -or $runRoot.StartsWith("$archiveRoot/") -or $archiveRoot.StartsWith("$runRoot/")) {
    throw 'RemoteRunRoot and RemoteArchiveRoot must be disjoint.'
}
if ($canaryRunRoot.StartsWith("$repo/") -or $canaryArchiveRoot.StartsWith("$repo/") -or
    $canaryRunRoot.StartsWith("$archiveRoot/") -or $canaryArchiveRoot.StartsWith("$runRoot/") -or
    $runRoot.StartsWith("$canaryRunRoot/") -or $archiveRoot.StartsWith("$canaryArchiveRoot/") -or
    $canaryRunRoot.StartsWith("$runRoot/") -or $canaryArchiveRoot.StartsWith("$archiveRoot/")) {
    throw 'Canary roots must be independent from training/archive roots.'
}
if (-not (Test-Path -LiteralPath $invokeLegion -PathType Leaf)) {
    throw "Legion SSH wrapper is missing: $invokeLegion"
}
if ($Mode -eq 'submit' -and -not $ConfirmExpensiveRun) {
    throw 'submit launches one canary plus one sequential ten-task GPU job; use -ConfirmExpensiveRun.'
}

function Quote-Bash([string]$Value) { return "'" + ($Value -replace "'", "'\'''") + "'" }
$repoQ = Quote-Bash $repo
$runQ = Quote-Bash $runRoot
$archiveQ = Quote-Bash $archiveRoot
$canaryRunQ = Quote-Bash $canaryRunRoot
$canaryArchiveQ = Quote-Bash $canaryArchiveRoot
$hashQ = Quote-Bash $ControllerConfigHash
$expectedQ = Quote-Bash $ExpectedCommit
$runnerQ = Quote-Bash $runnerRelative
$canaryQ = Quote-Bash $canaryRelative

# The canary is submitted first. One dependent job then runs all ten tasks
# sequentially on a single GPU allocation and saves one adapter per task.
$remoteScript = @'
set -Eeuo pipefail
repo=__REPO__
run_root=__RUN_ROOT__
archive_root=__ARCHIVE_ROOT__
canary_run_root=__CANARY_RUN_ROOT__
canary_archive_root=__CANARY_ARCHIVE_ROOT__
expected_commit=__EXPECTED_COMMIT__
controller_hash=__CONTROLLER_HASH__
runner_rel=__RUNNER_REL__
canary_rel=__CANARY_REL__
test -d "$repo/.git"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
test -f "$repo/$runner_rel"
test -f "$repo/$canary_rel"
test -f "$repo/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json"
printf 'preflight=PASS\nexecution=single_sequential_job\ntasks=0-9\nexpected_commit=%s\ncontroller_config_hash=%s\ncollection_mode=fresh_arrow\narrow_demos=50\nrequested_epochs=5\noptimizer_steps=derived_from_dataset\ncanary_run_root=%s\ncanary_archive_root=%s\n' "$expected_commit" "$controller_hash" "$canary_run_root" "$canary_archive_root"
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  printf 'task=%s run_root=%s/task_%s/run dataset_root=%s/task_%s/run/dataset training_root=%s/task_%s/run/training archive_root=%s/task_%s\n' \
    "$task_id" "$run_root" "$task_id" "$run_root" "$task_id" "$run_root" "$task_id" "$archive_root" "$task_id"
done
if [[ '__MODE__' == 'submit' ]]; then
  export REPO_ROOT="$repo" PEFT_EXPECTED_COMMIT="$expected_commit"
  export PEFT_CANARY_RUN_ROOT="$canary_run_root" PEFT_CANARY_ARCHIVE_ROOT="$canary_archive_root"
  export PEFT_CANARY_CONTROLLER_HASH="$controller_hash"
  canary_id="$(sbatch --parsable --export=ALL --job-name=__JOB_NAME___canary --partition=gpu_a40 --exclude=compute-4-13 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=0-04:00:00 --output="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.out" --error="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.err" "$repo/$canary_rel")"
  [[ "$canary_id" =~ ^[0-9]+$ ]] || { printf 'invalid canary job id: %s\n' "$canary_id" >&2; exit 2; }
  export REPO_ROOT="$repo" PEFT_EXPECTED_CONTROLLER_HASH="$controller_hash"
  export PEFT_ALL_TASK_RUN_ROOT="$run_root" PEFT_ALL_TASK_ARCHIVE_ROOT="$archive_root" PEFT_ALL_TASK_LABEL=__LABEL__
  all_task_id="$(sbatch --parsable --dependency=afterok:"$canary_id" --job-name=__JOB_NAME__ --partition=gpu_a40_ext --exclude=compute-4-13 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=5-00:00:00 --output="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.out" --error="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.err" --export=ALL "$repo/$runner_rel")"
  [[ "$all_task_id" =~ ^[0-9]+$ ]] || { printf 'invalid all-task job id: %s\n' "$all_task_id" >&2; exit 2; }
  printf 'collector_canary_job=%s\nall_task_job=%s\ndependency=afterok:%s\n' "$canary_id" "$all_task_id" "$canary_id"
fi
'@
$remoteScript = $remoteScript.Replace('__REPO__', $repoQ)
$remoteScript = $remoteScript.Replace('__RUN_ROOT__', $runQ)
$remoteScript = $remoteScript.Replace('__ARCHIVE_ROOT__', $archiveQ)
$remoteScript = $remoteScript.Replace('__CANARY_RUN_ROOT__', $canaryRunQ)
$remoteScript = $remoteScript.Replace('__CANARY_ARCHIVE_ROOT__', $canaryArchiveQ)
$remoteScript = $remoteScript.Replace('__EXPECTED_COMMIT__', $expectedQ)
$remoteScript = $remoteScript.Replace('__CONTROLLER_HASH__', $hashQ)
$remoteScript = $remoteScript.Replace('__RUNNER_REL__', $runnerQ)
$remoteScript = $remoteScript.Replace('__CANARY_REL__', $canaryQ)
$remoteScript = $remoteScript.Replace('__LABEL__', (Quote-Bash $Label))
$remoteScript = $remoteScript.Replace('__JOB_NAME__', (Quote-Bash $JobName))
$remoteScript = $remoteScript.Replace('__MODE__', $Mode)
$remoteScript = $remoteScript.Replace(([char]13).ToString() + ([char]10).ToString(), ([char]10).ToString())

Write-Host "Legion SmolVLA live collector-canary + task-specific array (mode=$Mode)"
& $invokeLegion -RemoteCommand $remoteScript
exit $LASTEXITCODE
