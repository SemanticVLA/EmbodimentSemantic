from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / ".codex" / "legion-local" / "Export-LegionCanaryArtifacts.ps1"


def test_canary_export_uses_scoped_legion_wrapper_and_verifies_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Invoke-Legion.ps1" in text
    assert "legion-askpass.cmd" in text
    assert "StrictHostKeyChecking=accept-new" in text
    assert "scp -q" in text
    assert "Get-FileHash" in text
    assert "hashes_verified" in text
    assert "export_manifest.json" in text
    assert "artifact_count" in text
    assert "total_bytes" in text
    assert "remote_relative_path" in text
    assert "Move-Item" in text
    assert ".partial" in text
    assert "Refusing to overwrite existing canary export" in text


def test_canary_export_places_variant_suite_task_episode_and_rejects_unsafe_paths():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Join-Path $runRoot (Join-Path $Variant" in text
    assert "unknown_suite" in text
    assert "task_unknown" in text
    assert "episode_unknown" in text
    assert "StartsWith('/')" in text
    assert "(^|/)\\.\\.?(/|$)" in text
    assert "RemoteArchive" in text
    assert "EmbodimentSemantic_archive" in text


def test_canary_export_powershell_syntax_if_available():
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        return
    result = subprocess.run(
        [pwsh, "-NoProfile", "-Command", f"$ErrorActionPreference='Stop'; [scriptblock]::Create((Get-Content -Raw -LiteralPath '{SCRIPT}')) | Out-Null"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
