[CmdletBinding()]
param(
    [ValidateSet('preflight', 'dry-run', 'submit', 'resume-submit')]
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
    [string]$ResumeSourceRunRoot = '',
    [ValidateRange(0, 2147483647)]
    [int]$ResumeSourceJobId = 0,
    [string]$ResumeTrainingCommit = '',
    [ValidateRange(1, 1000000)]
    [int]$RequestedEpochs = 5,
    [ValidateRange(0, 9)]
    [int]$EndTaskId = 9,
    [string]$ResumeRunnerRelative = 'vla_benchmarking/libero/automatic_ttt/legion/run_smolvla_peft_arrow_resume.sbatch',
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
if ($ResumeSourceRunRoot) { Assert-RemotePath 'ResumeSourceRunRoot' $ResumeSourceRunRoot }
if ([IO.Path]::IsPathRooted($ResumeRunnerRelative) -or $ResumeRunnerRelative -match '(^|/)\.\.?(/|$)' -or $ResumeRunnerRelative -match '[\r\n\s''"]') {
    throw 'ResumeRunnerRelative must be a relative shell-safe path.'
}
if ($Mode -eq 'resume-submit' -and -not $ResumeSourceRunRoot) {
    throw 'resume-submit requires -ResumeSourceRunRoot pointing at the preserved failed task-0 run.'
}
if ($Mode -eq 'resume-submit' -and ($ResumeSourceJobId -le 0 -or $ResumeTrainingCommit -notmatch '^[0-9a-f]{40}$')) {
    throw 'resume-submit requires -ResumeSourceJobId and -ResumeTrainingCommit from the preserved task-0 run.'
}
if ($repo -notmatch "/EmbodimentSemantic_releases/$ExpectedCommit$") {
    throw "RemoteRepoRoot must end in exact immutable release $ExpectedCommit"
}
if ($runRoot -eq $archiveRoot -or $runRoot.StartsWith("$archiveRoot/") -or $archiveRoot.StartsWith("$runRoot/")) {
    throw 'RemoteRunRoot and RemoteArchiveRoot must be disjoint.'
}
if ($ResumeSourceRunRoot -and (
        $ResumeSourceRunRoot -eq $runRoot -or
        $ResumeSourceRunRoot.StartsWith("$runRoot/") -or
        $runRoot.StartsWith("$ResumeSourceRunRoot/") -or
        $ResumeSourceRunRoot -eq $archiveRoot -or
        $ResumeSourceRunRoot.StartsWith("$archiveRoot/") -or
        $archiveRoot.StartsWith("$ResumeSourceRunRoot/"))) {
    throw 'ResumeSourceRunRoot must be disjoint from RemoteRunRoot and RemoteArchiveRoot.'
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
$resumeSourceQ = if ($ResumeSourceRunRoot) { Quote-Bash $ResumeSourceRunRoot } else { "''" }
$resumeRunnerQ = Quote-Bash $ResumeRunnerRelative
$resumeSourceJobIdQ = Quote-Bash ([string]$ResumeSourceJobId)
$resumeTrainingCommitQ = Quote-Bash $ResumeTrainingCommit
$requestedEpochsQ = Quote-Bash ([string]$RequestedEpochs)
$endTaskIdQ = Quote-Bash ([string]$EndTaskId)

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
resume_source=__RESUME_SOURCE__
resume_runner_rel=__RESUME_RUNNER_REL__
resume_source_job_id=__RESUME_SOURCE_JOB_ID__
resume_training_commit=__RESUME_TRAINING_COMMIT__
requested_epochs=__REQUESTED_EPOCHS__
end_task_id=__END_TASK_ID__
test -d "$repo/.git"
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
test -f "$repo/$runner_rel"
test -f "$repo/$canary_rel"
test -f "$repo/vla_benchmarking/libero/arrow_grasp_controller/configs/canonical_molmo_rgbd_grasp.json"
printf 'preflight=PASS\nexecution=single_sequential_job\ntasks=0-%s\nexpected_commit=%s\ncontroller_config_hash=%s\ncollection_mode=fresh_arrow\narrow_demos=1\nskip_baseline=1\nadapted_eval_episodes=10\nrequested_epochs=%s\noptimizer_steps=derived_from_dataset\ncanary_run_root=%s\ncanary_archive_root=%s\n' "$end_task_id" "$expected_commit" "$controller_hash" "$requested_epochs" "$canary_run_root" "$canary_archive_root"
for task_id in $(seq 0 "$end_task_id"); do
  printf 'task=%s run_root=%s/task_%s/run dataset_root=%s/task_%s/run/dataset training_root=%s/task_%s/run/training archive_root=%s/task_%s\n' \
    "$task_id" "$run_root" "$task_id" "$run_root" "$task_id" "$run_root" "$task_id" "$archive_root" "$task_id"
done
if [[ '__MODE__' == 'submit' ]]; then
  export REPO_ROOT="$repo" PEFT_EXPECTED_COMMIT="$expected_commit" PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1
  export PEFT_REQUESTED_EPOCHS="$requested_epochs" PEFT_END_TASK_ID="$end_task_id"
  export PEFT_CANARY_RUN_ROOT="$canary_run_root" PEFT_CANARY_ARCHIVE_ROOT="$canary_archive_root"
  export PEFT_CANARY_CONTROLLER_HASH="$controller_hash"
  canary_id="$(sbatch --parsable --export=ALL --job-name=__JOB_NAME___canary --partition=gpu_a100 --exclude=compute-4-13 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=0-04:00:00 --output="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.out" --error="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.err" "$repo/$canary_rel")"
  [[ "$canary_id" =~ ^[0-9]+$ ]] || { printf 'invalid canary job id: %s\n' "$canary_id" >&2; exit 2; }
  export REPO_ROOT="$repo" PEFT_EXPECTED_CONTROLLER_HASH="$controller_hash" PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1
  export PEFT_ALL_TASK_RUN_ROOT="$run_root" PEFT_ALL_TASK_ARCHIVE_ROOT="$archive_root" PEFT_ALL_TASK_LABEL=__LABEL__
  all_task_id="$(sbatch --parsable --dependency=afterok:"$canary_id" --job-name=__JOB_NAME__ --partition=gpu_a40_ext --exclude=compute-4-13 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=5-00:00:00 --output="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.out" --error="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.err" --export=ALL "$repo/$runner_rel")"
  [[ "$all_task_id" =~ ^[0-9]+$ ]] || { printf 'invalid all-task job id: %s\n' "$all_task_id" >&2; exit 2; }
  printf 'collector_canary_job=%s\nall_task_job=%s\ndependency=afterok:%s\n' "$canary_id" "$all_task_id" "$canary_id"
elif [[ '__MODE__' == 'resume-submit' ]]; then
  test -n "$resume_source"
  test -d "$resume_source"
  resume_source="$(realpath -m -- "$resume_source")"
  normalized_run_root="$(realpath -m -- "$run_root")"
  normalized_archive_root="$(realpath -m -- "$archive_root")"
  case "$resume_source" in
    "$normalized_run_root"|"$normalized_run_root"/*|"$normalized_archive_root"|"$normalized_archive_root"/*)
      printf 'resume source overlaps new run/archive roots\n' >&2
      exit 2
      ;;
  esac
  case "$normalized_run_root" in
    "$resume_source"/*) printf 'new run root is inside resume source\n' >&2; exit 2 ;;
  esac
  case "$normalized_archive_root" in
    "$resume_source"/*) printf 'new archive root is inside resume source\n' >&2; exit 2 ;;
  esac
  export REPO_ROOT="$repo" PEFT_EXPECTED_COMMIT="$expected_commit" PEFT_EXPECTED_CONTROLLER_HASH="$controller_hash" PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1
  export PEFT_ALL_TASK_RUN_ROOT="$run_root" PEFT_ALL_TASK_ARCHIVE_ROOT="$archive_root" PEFT_ALL_TASK_LABEL=__LABEL__
  export PEFT_START_TASK_ID=0 PEFT_END_TASK_ID="$end_task_id" PEFT_REQUESTED_EPOCHS="$requested_epochs"
  export PEFT_RESUME_SOURCE_RUN_ROOT="$resume_source" PEFT_RESUME_SOURCE_JOB_ID="$resume_source_job_id"
  export PEFT_RESUME_TRAINING_COMMIT="$resume_training_commit" PEFT_RESUME_RUNNER_RELATIVE="$resume_runner_rel"
  all_task_id="$(sbatch --parsable --job-name=__JOB_NAME___resume --partition=gpu_a40_ext --exclude=compute-4-13 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=5-00:00:00 --output="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.out" --error="$HOME/EmbodimentSemantic_runtime/operator/logs/%x_%j.err" --export=ALL "$repo/$runner_rel")"
  [[ "$all_task_id" =~ ^[0-9]+$ ]] || { printf 'invalid resume all-task job id: %s\n' "$all_task_id" >&2; exit 2; }
  printf 'collector_canary_job=REUSED\nall_task_job=%s\ndependency=none\nresume_source=%s\n' "$all_task_id" "$resume_source"
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
$remoteScript = $remoteScript.Replace('__RESUME_SOURCE__', $resumeSourceQ)
$remoteScript = $remoteScript.Replace('__RESUME_RUNNER_REL__', $resumeRunnerQ)
$remoteScript = $remoteScript.Replace('__RESUME_SOURCE_JOB_ID__', $resumeSourceJobIdQ)
$remoteScript = $remoteScript.Replace('__RESUME_TRAINING_COMMIT__', $resumeTrainingCommitQ)
$remoteScript = $remoteScript.Replace('__REQUESTED_EPOCHS__', $requestedEpochsQ)
$remoteScript = $remoteScript.Replace('__END_TASK_ID__', $endTaskIdQ)
$remoteScript = $remoteScript.Replace('__LABEL__', (Quote-Bash $Label))
$remoteScript = $remoteScript.Replace('__JOB_NAME__', (Quote-Bash $JobName))
$remoteScript = $remoteScript.Replace('__MODE__', $Mode)
$remoteScript = $remoteScript.Replace(([char]13).ToString() + ([char]10).ToString(), ([char]10).ToString())

Write-Host "Legion SmolVLA live collector-canary + task-specific array (mode=$Mode)"
& $invokeLegion -RemoteCommand $remoteScript
exit $LASTEXITCODE
