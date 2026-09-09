param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedCommit,
    [string]$RemoteRepoRoot,
    [Parameter(Mandatory = $true)][ValidateSet('editor','minimal_learned','apprentice')][string]$TrainKind,
    [Parameter(Mandatory = $true)][string]$SourceArtifact,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$SourceSha256,
    [Parameter(Mandatory = $true)][string]$BaseCheckpoint,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BaseSha256,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][string]$ArchiveRoot,
    [string[]]$TaskId = @(),
    [string[]]$ResetId = @(),
    [string[]]$EpisodeId = @(),
    [ValidateRange(0,2147483647)][Nullable[int]]$Seed,
    [ValidateRange(0,2147483647)][Nullable[int]]$InitStateIndex,
    [ValidateRange(1,1000)][int]$Epochs = 1,
    [ValidateRange(1,4096)][int]$BatchSize = 32,
    [ValidateSet('cpu','cuda','mps')][string]$Device = 'cuda',
    [ValidatePattern('^[0-9]+-[0-9]{2}:[0-5][0-9]:[0-5][0-9]$')][string]$TimeLimit = '0-02:00:00',
    [ValidateSet('gpu_a40','gpu_v100')][string]$Partition = 'gpu_a40',
    [ValidatePattern('^[0-9]+$')][string]$AfterOkJobId,
    [string]$ModelRevision,
    [string]$ProcessorRevision
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RemoteRepoRoot)) {
    $RemoteRepoRoot = "/home/hjaber/EmbodimentSemantic_runtime/releases/$ExpectedCommit"
}

function Assert-SafeLinuxPath([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or
        $Value -notmatch '^/[A-Za-z0-9_./-]+$' -or
        $Value -match '(^|/)\.\.(/|$)' -or $Value -eq '/') {
        throw "$Name must be a non-root safe absolute Linux path."
    }
}
function Assert-SafeToken([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch '^[A-Za-z0-9_.:/-]+$') {
        throw "$Name contains unsafe shell characters."
    }
}

Assert-SafeLinuxPath 'RemoteRepoRoot' $RemoteRepoRoot
foreach ($path in @(
    @{ Name = 'SourceArtifact'; Value = $SourceArtifact },
    @{ Name = 'BaseCheckpoint'; Value = $BaseCheckpoint },
    @{ Name = 'OutputRoot'; Value = $OutputRoot },
    @{ Name = 'ArchiveRoot'; Value = $ArchiveRoot }
)) { Assert-SafeLinuxPath $path.Name $path.Value }
if ($OutputRoot -eq $ArchiveRoot) { throw 'OutputRoot and ArchiveRoot must be distinct.' }
foreach ($artifact in @($SourceArtifact, $BaseCheckpoint)) {
    if ($artifact.StartsWith("$OutputRoot/", [StringComparison]::Ordinal) -or
        $artifact.StartsWith("$ArchiveRoot/", [StringComparison]::Ordinal) -or
        $OutputRoot.StartsWith("$artifact/", [StringComparison]::Ordinal) -or
        $ArchiveRoot.StartsWith("$artifact/", [StringComparison]::Ordinal)) {
        throw 'SourceArtifact and BaseCheckpoint must not overlap output/archive roots.'
    }
}
$time = [regex]::Match($TimeLimit, '^([0-9]+)-([0-9]{2}):([0-5][0-9]):([0-5][0-9])$')
if (-not $time.Success) { throw 'TimeLimit must use D-HH:MM:SS.' }
$timeSeconds = ([int64]$time.Groups[1].Value * 86400) + ([int64]$time.Groups[2].Value * 3600) + ([int64]$time.Groups[3].Value * 60) + [int64]$time.Groups[4].Value
if ($timeSeconds -lt 1 -or $timeSeconds -gt (7 * 86400)) { throw 'TimeLimit must be between 00:00:01 and 7-00:00:00.' }
foreach ($value in @($TaskId)) { Assert-SafeToken 'TaskId' $value }
foreach ($value in @($ResetId)) { Assert-SafeToken 'ResetId' $value }
foreach ($value in @($EpisodeId)) { Assert-SafeToken 'EpisodeId' $value }
if ($ModelRevision) { Assert-SafeToken 'ModelRevision' $ModelRevision }
if ($ProcessorRevision) { Assert-SafeToken 'ProcessorRevision' $ProcessorRevision }
if ($TrainKind -eq 'apprentice') {
    if ([string]::IsNullOrWhiteSpace($ModelRevision) -or [string]::IsNullOrWhiteSpace($ProcessorRevision)) {
        throw 'ModelRevision and ProcessorRevision are required for Apprentice training.'
    }
}

$job = Join-Path $PSScriptRoot 'run_arrow_policy_suite_training.sbatch'
if (-not (Test-Path -LiteralPath $job -PathType Leaf)) { throw "Missing launcher: $job" }
$repoExport = "export ARROW_TRAIN_REPO_ROOT='$RemoteRepoRoot'"
$remoteLines = @(
    'set -euo pipefail',
    $repoExport,
    'cd "$ARROW_TRAIN_REPO_ROOT"',
    'test -f vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_training.sbatch',
    'bash -n vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_training.sbatch',
    "export ARROW_TRAIN_EXPECTED_COMMIT='$ExpectedCommit'",
    "export ARROW_TRAIN_KIND='$TrainKind'",
    "export ARROW_TRAIN_SOURCE_ARTIFACT='$SourceArtifact'",
    "export ARROW_TRAIN_SOURCE_SHA256='$SourceSha256'",
    "export ARROW_TRAIN_BASE_CHECKPOINT='$BaseCheckpoint'",
    "export ARROW_TRAIN_BASE_SHA256='$BaseSha256'",
    "export ARROW_TRAIN_OUTPUT_ROOT='$OutputRoot'",
    "export ARROW_TRAIN_ARCHIVE_ROOT='$ArchiveRoot'",
    "export ARROW_TRAIN_TASK_IDS='$(@($TaskId) -join ':')'",
    "export ARROW_TRAIN_RESET_IDS='$(@($ResetId) -join ':')'",
    "export ARROW_TRAIN_EPISODE_IDS='$(@($EpisodeId) -join ':')'",
    "export ARROW_TRAIN_EPOCHS='$Epochs'",
    "export ARROW_TRAIN_BATCH_SIZE='$BatchSize'",
    "export ARROW_TRAIN_DEVICE='$Device'"
)
if ($null -ne $Seed) { $remoteLines += "export ARROW_TRAIN_SEED='$Seed'" } else { $remoteLines += 'unset ARROW_TRAIN_SEED' }
if ($null -ne $InitStateIndex) { $remoteLines += "export ARROW_TRAIN_INIT_STATE_INDEX='$InitStateIndex'" } else { $remoteLines += 'unset ARROW_TRAIN_INIT_STATE_INDEX' }
if ($ModelRevision) { $remoteLines += "export ARROW_TRAIN_MODEL_REVISION='$ModelRevision'" } else { $remoteLines += 'unset ARROW_TRAIN_MODEL_REVISION' }
if ($ProcessorRevision) { $remoteLines += "export ARROW_TRAIN_PROCESSOR_REVISION='$ProcessorRevision'" } else { $remoteLines += 'unset ARROW_TRAIN_PROCESSOR_REVISION' }
$sbatch = "sbatch --parsable --partition='$Partition' --export=ALL --time='$TimeLimit'"
if ($AfterOkJobId) { $sbatch += " --dependency=afterok:$AfterOkJobId" }
$remoteLines += "$sbatch vla_benchmarking/libero/arrow_policy_suite/legion/run_arrow_policy_suite_training.sbatch"
$remote = ($remoteLines -join "`n") + "`n"

& "$PSScriptRoot\..\..\..\..\.codex\legion-local\Invoke-Legion.ps1" -RemoteCommand $remote
