param(
    [Parameter(Mandatory = $true)][string]$ExpectedCommit,
    # RemoteRepoRoot must point at a clean immutable staged release, not the
    # shared HOME checkout that may contain user assets/untracked files.
    [string]$RemoteRepoRoot,
    [Alias('Config')][string]$RemoteConfig,
    [string]$LocalConfig,
    [string]$Factory = 'vla_benchmarking.libero.arrow_policy_suite.native_legion_factory:build_host',
    [ValidateSet('frozen_base','teacher_only','arrow_together','arrow_on_call','arrow_apprentice','arrow_editor','arrow_minimal','arrow_minimal_runtime','arrow_minimal_learned','arrow_fast','arrow_trace')][string]$Policy = 'frozen_base',
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [Parameter(Mandatory = $true)][string]$ArchiveRoot,
    [ValidateSet('canary','collect','evaluate','engineering_smoke')][string]$Operation = 'canary',
    [ValidateRange(1,1200)][int]$Steps = 3,
    [string]$Checkpoint,
    [string]$Controller,
    [string]$GraphContextRevision,
    [ValidateSet('rgbd','simulator_assisted_rgbd')][string]$TraceGeometryVariant,
    [switch]$EngineeringSmoke
)

$ErrorActionPreference = 'Stop'
if ($ExpectedCommit -notmatch '^[0-9a-f]{40}$') { throw 'ExpectedCommit must be a full lowercase commit SHA.' }
if ($EngineeringSmoke) { $Operation = 'engineering_smoke' }
$hasRemoteRepoRoot = -not [string]::IsNullOrWhiteSpace($RemoteRepoRoot)
if ($hasRemoteRepoRoot -and ($RemoteRepoRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteRepoRoot -match '(^|/)\.\.(/|$)')) { throw 'RemoteRepoRoot must be a safe absolute Linux path.' }
if (-not $EngineeringSmoke -and [string]::IsNullOrWhiteSpace($RemoteConfig)) { throw 'RemoteConfig is required unless -EngineeringSmoke is selected.' }
if (-not $EngineeringSmoke -and ($RemoteConfig -notmatch '^/[A-Za-z0-9_./-]+$' -or $RemoteConfig -match '(^|/)\.\.(/|$)')) { throw 'RemoteConfig must be a safe absolute Linux path.' }
if (-not [string]::IsNullOrWhiteSpace($LocalConfig)) {
    if ([IO.Path]::IsPathRooted($LocalConfig) -eq $false -or -not (Test-Path -LiteralPath $LocalConfig -PathType Leaf)) { throw 'LocalConfig must be an existing Windows absolute file.' }
}
if (-not $EngineeringSmoke -and [string]::IsNullOrWhiteSpace($Factory)) { throw 'Factory is required unless -EngineeringSmoke is selected.' }
if (-not $EngineeringSmoke -and $Factory -notmatch '^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$') { throw 'Factory must be module:callable.' }
if ($Controller -and ($Controller -notmatch '^/[A-Za-z0-9_./-]+$' -or $Controller -match '(^|/)\.\.(/|$)')) { throw 'Controller must be a safe absolute Linux path.' }
if ($RunRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $RunRoot -match '(^|/)\.\.(/|$)' -or $ArchiveRoot -notmatch '^/[A-Za-z0-9_./-]+$' -or $ArchiveRoot -match '(^|/)\.\.(/|$)') { throw 'RunRoot and ArchiveRoot must be safe absolute Linux paths.' }
if ($Checkpoint -and ($Checkpoint -notmatch '^/[A-Za-z0-9_./-]+$' -or $Checkpoint -match '(^|/)\.\.(/|$)')) { throw 'Checkpoint must be a safe absolute Linux path.' }
if ($GraphContextRevision -and $GraphContextRevision -notmatch '^[A-Za-z0-9_.:/-]+$') { throw 'GraphContextRevision contains unsafe shell characters.' }
$job = Join-Path $PSScriptRoot 'run_arrow_policy_suite_canary.sbatch'
if (-not (Test-Path -LiteralPath $job -PathType Leaf)) { throw "Missing launcher: $job" }

$repoExport = if ($hasRemoteRepoRoot) {
    "export ARROW_SUITE_REPO_ROOT='$RemoteRepoRoot'"
} else {
    'export ARROW_SUITE_REPO_ROOT="$HOME/EmbodimentSemantic_runtime/releases/' + $ExpectedCommit + '"'
}
$remoteLines = @(
    'set -euo pipefail'
    $repoExport
    'cd "$ARROW_SUITE_REPO_ROOT"'
    'test -f vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_canary.sbatch'
    'bash -n vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_canary.sbatch'
    "export ARROW_SUITE_EXPECTED_COMMIT='$ExpectedCommit'"
    "export ARROW_SUITE_RUN_ROOT='$RunRoot'"
    "export ARROW_SUITE_ARCHIVE_ROOT='$ArchiveRoot'"
    "export ARROW_SUITE_OPERATION='$Operation'"
    "export ARROW_SUITE_STEPS='$Steps'"
)
if (-not $EngineeringSmoke) {
    $remoteLines += "export ARROW_SUITE_CONFIG='$RemoteConfig'"
    $remoteLines += "export ARROW_SUITE_FACTORY='$Factory'"
    $remoteLines += "export ARROW_SUITE_POLICY='$Policy'"
} else {
    $remoteLines += 'export ARROW_SUITE_ENGINEERING_SMOKE=1'
}
if ($Checkpoint) { $remoteLines += "export ARROW_SUITE_CHECKPOINT='$Checkpoint'" }
if ($Controller) { $remoteLines += "export ARROW_SUITE_CONTROLLER='$Controller'" }
if ($GraphContextRevision) { $remoteLines += "export ARROW_SUITE_GRAPH_CONTEXT_REVISION='$GraphContextRevision'" }
if ($TraceGeometryVariant) { $remoteLines += "export ARROW_SUITE_TRACE_GEOMETRY_VARIANT='$TraceGeometryVariant'" }
$remoteLines += 'sbatch --parsable vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_canary.sbatch'
$remote = ($remoteLines -join "`n") + "`n"

& "$PSScriptRoot\..\..\..\..\.codex\legion-local\Invoke-Legion.ps1" -RemoteCommand $remote
