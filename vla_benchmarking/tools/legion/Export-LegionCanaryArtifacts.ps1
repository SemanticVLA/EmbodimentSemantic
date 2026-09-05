[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^/home/hjaber/EmbodimentSemantic_archive/[A-Za-z0-9._/-]+$')]
    [string]$RemoteArchive,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')]
    [string]$Variant,

    [switch]$VideosOnly,

    [string]$DesktopRoot = [Environment]::GetFolderPath('Desktop'),

    [string]$LegionHelperRoot = ''
)

$ErrorActionPreference = 'Stop'
$gateway = 'cagliero_thesis_students@mp4.polito.it'
$legion = 'hjaber@hpc-legionlogin.polito.it'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
if ([string]::IsNullOrWhiteSpace($LegionHelperRoot)) {
    $LegionHelperRoot = Join-Path $repositoryRoot '.codex\legion-local'
}
$helperRoot = [IO.Path]::GetFullPath($LegionHelperRoot)
$invokeLegion = Join-Path $helperRoot 'Invoke-Legion.ps1'
$askpass = Join-Path $helperRoot 'legion-askpass.cmd'
if (-not (Test-Path -LiteralPath $invokeLegion -PathType Leaf)) {
    throw "Missing Legion command helper: $invokeLegion"
}
if (-not (Test-Path -LiteralPath $askpass -PathType Leaf)) {
    throw "Missing Legion askpass helper: $askpass"
}

if ([string]::IsNullOrWhiteSpace($DesktopRoot)) { throw 'DesktopRoot is required.' }
$desktop = [IO.Path]::GetFullPath($DesktopRoot)
$canaryRoot = Join-Path $desktop 'arrow_canaries'
$runRoot = Join-Path $canaryRoot $RunId
if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing to overwrite existing canary export: $runRoot"
}
New-Item -ItemType Directory -Path $canaryRoot -Force | Out-Null
New-Item -ItemType Directory -Path $runRoot -ErrorAction Stop | Out-Null

function Invoke-ScopedSsh {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Operation,
        [Parameter(Mandatory = $true)][ref]$ExitCode
    )
    $savedEnvironment = @{}
    foreach ($name in @('SSH_ASKPASS', 'SSH_ASKPASS_REQUIRE', 'DISPLAY')) {
        $savedEnvironment[$name] = if (Test-Path -LiteralPath "Env:$name") { (Get-Item -LiteralPath "Env:$name").Value } else { $null }
    }
    try {
        # Reuse the repository's credential-vault askpass path; credentials are
        # never placed in command arguments, files, or exported environment.
        $env:SSH_ASKPASS = $askpass
        $env:SSH_ASKPASS_REQUIRE = 'force'
        $env:DISPLAY = 'codex-legion'
        & $Operation
        $ExitCode.Value = $LASTEXITCODE
    }
    finally {
        foreach ($name in $savedEnvironment.Keys) {
            if ($null -eq $savedEnvironment[$name]) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
            else { Set-Item -LiteralPath "Env:$name" -Value $savedEnvironment[$name] }
        }
    }
}

$findClause = if ($VideosOnly) {
    "find . -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' \)"
} else {
    "find . -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -path '*/phase_frames/*.png' -o -path '*/phase_frames/*.jpg' -o -path '*/frames/*.png' -o -path '*/frames/*.jpg' -o -iname '*.jsonl' -o -iname '*manifest*.json' -o -iname '*summary*.json' -o -name 'job_context.env' -o -name 'archive_status.env' \)"
}
$remoteCommand = @'
set -Eeuo pipefail
root='__REMOTE_ARCHIVE__'
test -d "$root"
cd -- "$root"
__FIND_CLAUSE__ -print | sort | while IFS= read -r path; do
  rel="${path#./}"
  test "$rel" != "$path" || test -n "$rel"
  sha=$(sha256sum -- "$path" | awk '{print $1}')
  bytes=$(stat -c %s -- "$path")
  printf '%s\t%s\t%s\n' "$sha" "$bytes" "$rel"
done
'@.Replace('__REMOTE_ARCHIVE__', $RemoteArchive).Replace('__FIND_CLAUSE__', $findClause)

$queryExit = 1
$remoteLines = @(& $invokeLegion -RemoteCommand $remoteCommand)
$queryExit = $LASTEXITCODE
if ($queryExit -ne 0) { throw 'Remote archive manifest query failed.' }
$remoteLines = @($remoteLines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if ($remoteLines.Count -eq 0) { throw 'Remote archive contains no canary artifacts.' }

function Get-Label([string]$relative, [string]$pattern, [string]$fallback) {
    $match = [regex]::Match($relative, $pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($match.Success) { return $match.Groups[1].Value }
    return $fallback
}

$artifacts = [System.Collections.Generic.List[object]]::new()
$seenDestinations = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$index = 0
foreach ($line in $remoteLines) {
    $parts = $line -split "`t", 3
    if ($parts.Count -ne 3 -or $parts[0] -notmatch '^[0-9a-f]{64}$' -or $parts[1] -notmatch '^[0-9]+$') {
        throw 'Remote manifest contains an invalid hash/size record.'
    }
    $remoteHash = $parts[0].ToLowerInvariant()
    $remoteBytes = [int64]$parts[1]
    $relative = $parts[2]
    if ($relative -notmatch '^[A-Za-z0-9._/-]+$' -or $relative.StartsWith('/') -or $relative -match '(^|/)\.\.?(/|$)') {
        throw "Remote manifest contains an unsafe relative path: $relative"
    }
    $suite = Get-Label $relative '(?:^|/)(vanilla|sealed_randomized)(?:/|$)' 'unknown_suite'
    $taskNumber = Get-Label $relative '(?:^|/)(?:task|task_id)[_-]([0-9]+)(?:/|$)' 'unknown'
$episodeNumber = Get-Label $relative '(?:^|/)(?:episode|episode_id)[_-]([0-9]+)(?:_|/|$)' 'unknown'
    $task = if ($taskNumber -eq 'unknown') { 'task_unknown' } else { "task_$taskNumber" }
    $episode = if ($episodeNumber -eq 'unknown') { 'episode_unknown' } else { "episode_$episodeNumber" }
    $kind = if ($relative -match '(?i)\.(mp4|avi|mov)$') { 'video' } elseif ($relative -match '(?i)(^|/)(phase_frames|frames)/.*\.(png|jpg)$') { 'frame' } else { 'manifest' }
    $destinationDirectory = Join-Path $runRoot (Join-Path $Variant (Join-Path $suite (Join-Path $task $episode)))
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    $leaf = [IO.Path]::GetFileName($relative)
    # Attempts and phase segments commonly reuse names such as motion.mp4 or
    # 00_pregrasp.png.  Preserve every artifact by making the destination
    # collision-proof with the already verified remote digest prefix.
    $destination = Join-Path $destinationDirectory ("{0}__{1}{2}" -f [IO.Path]::GetFileNameWithoutExtension($leaf), $remoteHash.Substring(0, 12), [IO.Path]::GetExtension($leaf))
    if (-not $seenDestinations.Add($destination)) { throw "Duplicate export destination: $destination" }
    $partial = "$destination.partial"
    $remoteSource = "${legion}:$RemoteArchive/$relative"
    $copyExit = 1
    Invoke-ScopedSsh {
        & scp -q -o 'BatchMode=no' -o 'NumberOfPasswordPrompts=1' -o 'StrictHostKeyChecking=accept-new' -J $gateway $remoteSource $partial
    } ([ref]$copyExit)
    if ($copyExit -ne 0) { throw "Artifact download failed for $relative." }
    if (-not (Test-Path -LiteralPath $partial -PathType Leaf)) { throw "Downloaded artifact is missing: $relative" }
    $localItem = Get-Item -LiteralPath $partial
    $localHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($localItem.Length -ne $remoteBytes -or $localHash -ne $remoteHash) {
        throw "Artifact verification failed for $relative."
    }
    Move-Item -LiteralPath $partial -Destination $destination -ErrorAction Stop
    $artifacts.Add([ordered]@{
        remote_relative_path = $relative
        local_relative_path = $destination.Substring($runRoot.Length).TrimStart('\', '/')
        kind = $kind
        variant = $Variant
        suite = $suite
        task = $task
        episode = $episode
        bytes = $remoteBytes
        sha256 = $remoteHash
        verified = $true
    })
    $index++
}

$manifest = [ordered]@{
    schema_version = 'arrow_canary_export.v1'
    run_id = $RunId
    variant = $Variant
    remote_archive = $RemoteArchive
    local_root = $runRoot
    artifact_count = $artifacts.Count
    total_bytes = [int64](($artifacts | Measure-Object -Property bytes -Sum).Sum)
    hashes_verified = ($artifacts.Count -gt 0 -and @($artifacts | Where-Object { -not $_.verified }).Count -eq 0)
    artifacts = @($artifacts)
}
$manifestPath = Join-Path $runRoot 'export_manifest.json'
[IO.File]::WriteAllText($manifestPath, (($manifest | ConvertTo-Json -Depth 10) + "`n"), [Text.UTF8Encoding]::new($false))
Write-Host "Exported $($artifacts.Count) verified canary artifacts to $runRoot"
