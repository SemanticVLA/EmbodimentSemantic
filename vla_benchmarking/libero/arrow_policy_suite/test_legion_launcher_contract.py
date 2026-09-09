from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parents[2]))
SUBMIT = (ROOT / "legion" / "submit_arrow_policy_suite_canary.ps1").read_text(encoding="utf-8")
SBATCH = (ROOT / "legion" / "run_arrow_policy_suite_canary.sbatch").read_text(encoding="utf-8")
TRAIN_SUBMIT = (ROOT / "legion" / "submit_arrow_policy_suite_training.ps1").read_text(encoding="utf-8")
TRAIN_SBATCH = (ROOT / "legion" / "run_arrow_policy_suite_training.sbatch").read_text(encoding="utf-8")


def test_training_front_door_has_sealed_kind_and_artifact_contract():
    assert "ValidateSet('editor','minimal_learned','apprentice')" in TRAIN_SUBMIT
    for parameter in ("ExpectedCommit", "SourceArtifact", "SourceSha256", "BaseCheckpoint", "BaseSha256", "OutputRoot", "ArchiveRoot"):
        assert f"${parameter}" in TRAIN_SUBMIT
    for env_name in (
        "ARROW_TRAIN_EXPECTED_COMMIT", "ARROW_TRAIN_KIND", "ARROW_TRAIN_SOURCE_ARTIFACT",
        "ARROW_TRAIN_SOURCE_SHA256", "ARROW_TRAIN_BASE_CHECKPOINT", "ARROW_TRAIN_BASE_SHA256",
        "ARROW_TRAIN_OUTPUT_ROOT", "ARROW_TRAIN_ARCHIVE_ROOT",
    ):
        assert env_name in TRAIN_SUBMIT
        assert env_name in TRAIN_SBATCH
    assert "artifact_sha256" in TRAIN_SBATCH
    assert "json.dumps(entries, separators=(',', ':'))" in TRAIN_SBATCH
    assert "output root already exists or is a symlink" in TRAIN_SBATCH
    assert "archive root already exists or is a symlink" in TRAIN_SBATCH


def test_training_base_hash_matches_canonical_snapshot_exclusions(tmp_path):
    """The launcher must use the Apprentice/base snapshot hash convention."""
    from vla_benchmarking.libero.arrow_policy_suite.apprentice_training import _base_tree_sha256

    base = tmp_path / "base"
    (base / ".cache").mkdir(parents=True)
    (base / "weights.bin").write_bytes(b"frozen")
    (base / ".cache" / "volatile.bin").write_bytes(b"ignore-me")
    (base / "base_snapshot_manifest.json").write_text('{"inventory": "producer"}', encoding="utf-8")
    canonical = _base_tree_sha256(base)
    (base / ".cache" / "volatile.bin").write_bytes(b"changed-but-excluded")
    (base / "base_snapshot_manifest.json").write_text('{"inventory": "changed"}', encoding="utf-8")
    assert _base_tree_sha256(base) == canonical
    assert "base_snapshot_manifest.json" in TRAIN_SBATCH
    assert "'.cache' in relative.parts" in TRAIN_SBATCH
    assert "len(encoded).to_bytes(8, 'big')" in TRAIN_SBATCH


def test_training_front_door_forwards_filters_and_dispatches_exact_modules():
    for parameter, env_name in (
        ("TaskId", "ARROW_TRAIN_TASK_IDS"),
        ("ResetId", "ARROW_TRAIN_RESET_IDS"),
        ("EpisodeId", "ARROW_TRAIN_EPISODE_IDS"),
    ):
        assert f"${parameter}" in TRAIN_SUBMIT
        assert env_name in TRAIN_SUBMIT
        assert env_name in TRAIN_SBATCH
    assert "--task-id" in TRAIN_SBATCH
    assert "--reset-id" in TRAIN_SBATCH
    assert "--episode-id" in TRAIN_SBATCH
    assert "args+=(--success-only)" in TRAIN_SBATCH
    assert "vla_benchmarking.libero.arrow_policy_suite.residual_job" in TRAIN_SBATCH
    assert "vla_benchmarking.libero.arrow_policy_suite.apprentice_job" in TRAIN_SBATCH
    assert "verify_adapter_reload" in TRAIN_SBATCH
    assert "load_native_residual" in TRAIN_SBATCH
    assert "--training-source" in TRAIN_SBATCH
    assert "--transitions" in TRAIN_SBATCH
    assert "--base-vla-sha256" in TRAIN_SBATCH
    assert "--base-sha256" in TRAIN_SBATCH


def test_training_front_door_archives_receipts_checkpoints_and_terminal_markers():
    assert "STARTED" in TRAIN_SBATCH
    assert "COMPLETED" in TRAIN_SBATCH
    assert "FAILED" in TRAIN_SBATCH
    assert "terminal_status" in TRAIN_SBATCH
    assert "training.log" in TRAIN_SBATCH
    assert "artifact_receipt.json" in TRAIN_SBATCH
    assert "training_receipt.json" not in TRAIN_SBATCH
    assert "runtime_artifact_path" in TRAIN_SBATCH
    assert "archive_artifact_path" in TRAIN_SBATCH
    assert "learned_artifact_runtime_path" in TRAIN_SBATCH
    assert "learned_artifact_archive_path" in TRAIN_SBATCH
    assert "execution_receipt.json" in TRAIN_SBATCH
    assert '"experiment_evidence": False' in TRAIN_SBATCH
    assert 'cp -a -- "$RUN_ROOT/." "$ARCHIVE_ROOT/run/"' in TRAIN_SBATCH


def test_training_artifact_receipt_runtime_contract(tmp_path):
    """Execute the launcher's embedded receipt writer for both artifact kinds."""
    marker = '"$PYTHON" - "$artifact_receipt" <<\'PY\'\n'
    start = TRAIN_SBATCH.index(marker) + len(marker)
    body = TRAIN_SBATCH[start:TRAIN_SBATCH.index("\nPY\n", start)]

    def run_receipt(kind: str, runtime: Path, *, sidecar: Path | None = None, manifest: Path | None = None):
        run_root = tmp_path / kind / "scratch"
        archive_root = tmp_path / kind / "archive"
        run_root.mkdir(parents=True, exist_ok=True)
        receipt = run_root / "artifact_receipt.json"
        env = dict(os.environ)
        env.update({
            "ARROW_TRAIN_RUNTIME_KIND": "residual" if sidecar else "apprentice",
            "ARROW_TRAIN_RUNTIME_ARTIFACT": str(runtime),
            "ARROW_TRAIN_RUNTIME_SIDECAR": str(sidecar) if sidecar else "",
            "ARROW_TRAIN_RUNTIME_MANIFEST": str(manifest) if manifest else "",
            "ARROW_TRAIN_OUTPUT_ROOT": str(run_root),
            "ARROW_TRAIN_ARCHIVE_ROOT": str(archive_root),
            "ARROW_TRAIN_KIND": kind,
            "ARROW_TRAIN_SOURCE_SHA256": "1" * 64,
            "ARROW_TRAIN_BASE_SHA256": "2" * 64,
        })
        result = subprocess.run(
            [sys.executable, "-", str(receipt)], input=body, text=True,
            capture_output=True, env=env, check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(receipt.read_text(encoding="utf-8"))

    residual = tmp_path / "editor" / "scratch" / "checkpoint.pt"
    residual.parent.mkdir(parents=True)
    residual.write_bytes(b"checkpoint")
    residual_sidecar = Path(str(residual) + ".json")
    residual_sidecar.write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest()}), encoding="utf-8")
    residual_receipt = run_receipt("editor", residual, sidecar=residual_sidecar)
    assert residual_receipt["runtime_artifact_path"].endswith("checkpoint.pt")
    assert residual_receipt["learned_artifact_runtime_path"].endswith("checkpoint.pt")
    assert residual_receipt["archive_artifact_path"].replace("\\", "/").endswith("/run/checkpoint.pt")
    assert residual_receipt["sidecar"]["path"].endswith("checkpoint.pt.json")
    assert residual_receipt["sidecar"]["archive_path"].replace("\\", "/").endswith("/run/checkpoint.pt.json")
    assert residual_receipt["manifest"] is None
    assert residual_receipt["experiment_evidence"] is False

    bundle = tmp_path / "apprentice" / "scratch" / "training" / "bundle"
    bundle.mkdir(parents=True)
    payload = b"adapter"
    (bundle / "adapter_model.safetensors").write_bytes(payload)
    inventory = {"adapter_model.safetensors": hashlib.sha256(payload).hexdigest()}
    inventory_hash = hashlib.sha256((json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    manifest = bundle / "apprentice_manifest.json"
    manifest.write_text(json.dumps({"checkpoint_inventory": inventory, "checkpoint_sha256": inventory_hash}), encoding="utf-8")
    apprentice_receipt = run_receipt("apprentice", bundle, manifest=manifest)
    assert apprentice_receipt["runtime_artifact_path"].replace("\\", "/").endswith("training/bundle")
    assert apprentice_receipt["archive_artifact_path"].replace("\\", "/").endswith("/run/training/bundle")
    assert apprentice_receipt["manifest"]["path"].endswith("apprentice_manifest.json")
    assert apprentice_receipt["manifest"]["archive_path"].replace("\\", "/").endswith("/run/training/bundle/apprentice_manifest.json")
    assert apprentice_receipt["sidecar"] is None
    assert apprentice_receipt["experiment_evidence"] is False


def test_training_submit_validates_time_limit_and_optional_dependency():
    assert "TimeLimit" in TRAIN_SUBMIT
    assert "TimeLimit must be between 00:00:01 and 7-00:00:00." in TRAIN_SUBMIT
    assert "AfterOkJobId" in TRAIN_SUBMIT
    assert "--dependency=afterok:$AfterOkJobId" in TRAIN_SUBMIT
    assert "--time='$TimeLimit'" in TRAIN_SUBMIT


def test_training_launchers_parse_with_native_shells():
    bash = shutil.which("bash")
    if bash:
        batch_path = (ROOT / "legion" / "run_arrow_policy_suite_training.sbatch").relative_to(Path.cwd())
        result = subprocess.run([bash, "-n", batch_path.as_posix()], cwd=Path.cwd(), check=False)
        assert result.returncode == 0
    pwsh = shutil.which("pwsh")
    if pwsh:
        command = (
            "$null = [System.Management.Automation.Language.Parser]::ParseFile("
            f"'{str(ROOT / 'legion' / 'submit_arrow_policy_suite_training.ps1').replace(chr(39), chr(39) + chr(39))}', [ref]$null, [ref]$null)"
        )
        result = subprocess.run([pwsh, "-NoProfile", "-Command", command], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def test_submit_uses_immutable_remote_repo_release_and_linux_paths():
    assert "$RemoteRepoRoot" in SUBMIT
    assert "EmbodimentSemantic_runtime/releases/" in SUBMIT
    assert 'cd "$ARROW_SUITE_REPO_ROOT"' in SUBMIT
    assert "RemoteConfig must be a safe absolute Linux path" in SUBMIT
    assert "RunRoot and ArchiveRoot must be safe absolute Linux paths" in SUBMIT
    assert "[IO.Path]::IsPathRooted($RunRoot)" not in SUBMIT


def test_submit_rejects_shell_injection_before_interpolation():
    for value in ("RemoteRepoRoot", "RemoteConfig", "RunRoot", "ArchiveRoot", "Checkpoint", "Controller", "RuntimePython", "HfCache", "GraphContextRevision"):
        assert value in SUBMIT
    assert "contains unsafe shell characters" in SUBMIT
    assert "must be a safe absolute Linux path" in SUBMIT


def test_submit_requires_explicit_canonical_runtime_for_teacher_policies():
    assert "teacherPolicies" in SUBMIT
    assert "RuntimePython and HfCache are required for teacher-dependent policies" in SUBMIT
    assert "frozen_base and arrow_trace do not select Molmo implicitly" in SUBMIT
    assert "export ARROW_SUITE_PYTHON='$RuntimePython'" in SUBMIT
    assert "export ARROW_SUITE_HF_CACHE='$HfCache'" in SUBMIT


def test_submit_and_batch_forward_repeatable_teacher_free_learned_artifacts():
    assert "LearnedArtifact" in SUBMIT
    assert "ARROW_SUITE_LEARNED_ARTIFACTS" in SUBMIT
    assert "LearnedArtifact is only valid for Apprentice, Editor, and Minimal-Learned policies." in SUBMIT
    assert "$teacherPolicies = @('teacher_only','arrow_together','arrow_on_call','arrow_minimal','arrow_minimal_runtime','arrow_fast')" in SUBMIT
    assert "ARROW_SUITE_LEARNED_ARTIFACTS" in SBATCH
    assert "validate_immutable_learned_file 'learned artifact sidecar'" in SBATCH
    assert 'args+=(--input "$learned_artifact")' in SBATCH
    assert "teacher_only|arrow_together|arrow_on_call|arrow_minimal|arrow_minimal_runtime|arrow_fast" in SBATCH


def test_submit_forwards_explicit_paired_reset_identity_without_defaults():
    for parameter, env_name in (
        ("TaskId", "ARROW_SUITE_TASK_ID"),
        ("Seed", "ARROW_SUITE_SEED"),
        ("InitStateIndex", "ARROW_SUITE_INIT_STATE_INDEX"),
    ):
        assert f"${parameter}" in SUBMIT
        assert env_name in SUBMIT
        assert env_name in SBATCH
    assert "[Nullable[int]]$TaskId" in SUBMIT
    assert "[Nullable[int]]$Seed" in SUBMIT
    assert "[Nullable[int]]$InitStateIndex" in SUBMIT
    assert "unset $envName" in SUBMIT
    assert "must be a non-negative integer" in SBATCH


def test_submit_forwards_bounded_explicit_slurm_time_limit():
    assert "[ValidatePattern('^[0-9]+-[0-9]{2}:[0-5][0-9]:[0-5][0-9]$')]" in SUBMIT
    assert "[string]$TimeLimit = '0-00:30:00'" in SUBMIT
    assert "TimeLimit must be between 00:00:01 and 7-00:00:00." in SUBMIT
    assert "sbatch --parsable --partition='$Partition' --export=ALL --time='$TimeLimit'" in SUBMIT


def test_submit_and_batch_require_explicit_collection_evaluation_horizons():
    assert "Steps is required explicitly for $Operation" in SUBMIT
    assert "Evaluate Steps must be one of the predeclared 280 or 1200 horizons." in SUBMIT
    assert "ARROW_SUITE_STEPS is required explicitly for collect" in SBATCH
    assert "ARROW_SUITE_STEPS is required explicitly for evaluate" in SBATCH
    assert "evaluate accepts only the predeclared 280 or 1200 step horizons" in SBATCH
    assert 'steps="${ARROW_SUITE_STEPS:-3}"' in SBATCH


def test_submit_reset_identity_loop_generates_exact_exports_and_unsets_at_runtime():
    """Execute the launcher's reset mapping, rather than checking source text only."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        return
    path = str(ROOT / "legion" / "submit_arrow_policy_suite_canary.ps1").replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
$source = Get-Content -Raw -LiteralPath '{path}'
$start = $source.IndexOf('$resetEnvNames = @{{')
$end = $source.IndexOf('$remoteLines += \"sbatch', $start)
if ($start -lt 0 -or $end -lt 0) {{ throw 'reset export loop not found' }}
$fragment = $source.Substring($start, $end - $start)
$resetValues = @(
    @{{ Name = 'TaskId'; Value = 7 }},
    @{{ Name = 'Seed'; Value = 11 }},
    @{{ Name = 'InitStateIndex'; Value = 2 }}
)
$remoteLines = @()
Invoke-Expression $fragment
$withValues = $remoteLines
$resetValues = @(
    @{{ Name = 'TaskId'; Value = $null }},
    @{{ Name = 'Seed'; Value = $null }},
    @{{ Name = 'InitStateIndex'; Value = $null }}
)
$remoteLines = @()
Invoke-Expression $fragment
$withoutValues = $remoteLines
Write-Output (($withValues -join '|') + "`n" + ($withoutValues -join '|'))
"""
    result = subprocess.run(
        [pwsh, "-NoProfile", "-Command", command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "export ARROW_SUITE_TASK_ID='7'|export ARROW_SUITE_SEED='11'|export ARROW_SUITE_INIT_STATE_INDEX='2'",
        "unset ARROW_SUITE_TASK_ID|unset ARROW_SUITE_SEED|unset ARROW_SUITE_INIT_STATE_INDEX",
    ]


def test_apprentice_bundle_and_residual_checkpoint_contracts_are_distinct():
    assert "validate_immutable_apprentice_bundle" in SBATCH
    assert "apprentice_manifest.json" in SBATCH
    assert "Apprentice bundle payload is empty" in SBATCH
    assert '[[ "$ARROW_SUITE_POLICY" == arrow_apprentice ]]' in SBATCH
    assert "learned artifact sidecar" in SBATCH
    assert "requires exactly one learned artifact" in SBATCH


def test_submit_builds_remote_command_from_joined_lines():
    """The PowerShell launcher must interpolate values before sending SSH text."""
    assert "$remoteLines = @(" in SUBMIT
    assert '$remote = ($remoteLines -join "`n") + "`n"' in SUBMIT
    assert "$remote = @'" not in SUBMIT
    assert "' + $repoExport + @'" not in SUBMIT
    assert '"export ARROW_SUITE_EXPECTED_COMMIT=\'$ExpectedCommit\'"' in SUBMIT
    assert "sbatch --parsable --partition='$Partition' --export=ALL" in SUBMIT


def test_sbatch_resolves_array_log_name_and_has_engineering_smoke_marker():
    assert 'SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID' in SBATCH
    assert "ARROW_SUITE_ENGINEERING_SMOKE" in SBATCH
    assert '"experiment_evidence": False' in SBATCH
    assert "arrow_policy_suite.engineering_smoke.v1" in SBATCH


def test_sbatch_uses_canonical_controller_hash_not_raw_file_hash():
    assert "from vla_benchmarking.libero.arrow_grasp_controller.configs import load_controller_config" in SBATCH
    assert 'print(load_controller_config(sys.argv[1])["config_hash"])' in SBATCH
    assert 'sha256sum -- "$ARROW_SUITE_CONTROLLER"' not in SBATCH


def test_submit_forwards_and_validates_fast_and_trace_contract_inputs():
    for name in (
        "GraphFactory",
        "GraphTripletArtifact",
        "VisualArrowArtifact",
        "FastArtifactKind",
        "FastEncoderRevision",
        "FastEncoderSha256",
        "FastRouterRevision",
        "FastRouterSha256",
        "FastSourceManifestSha256",
        "FastVlaManifestSha256",
        "TraceRouteArtifact",
        "TraceCalibrationArtifact",
    ):
        assert f"${name}" in SUBMIT
    for env_name in (
        "ARROW_SUITE_GRAPH_FACTORY",
        "ARROW_SUITE_TEXT_GRAPH_TRIPLET",
        "ARROW_SUITE_VISUAL_ARROW",
        "ARROW_SUITE_FAST_ARTIFACT_KIND",
        "ARROW_SUITE_FAST_ENCODER_REVISION",
        "ARROW_SUITE_FAST_ENCODER_SHA256",
        "ARROW_SUITE_FAST_ROUTER_REVISION",
        "ARROW_SUITE_FAST_ROUTER_SHA256",
        "ARROW_SUITE_FAST_SOURCE_MANIFEST_SHA256",
        "ARROW_SUITE_FAST_VLA_MANIFEST_SHA256",
        "ARROW_SUITE_TRACE_ROUTE_ARTIFACT",
        "ARROW_SUITE_TRACE_CALIBRATION_ARTIFACT",
    ):
        assert env_name in SUBMIT
        assert env_name in SBATCH
    assert "GraphFactory is required for Fast and Trace policies." in SUBMIT
    assert "FastArtifactKind must be" not in SUBMIT  # kind is validated as a token, runtime owns support
    assert "required for arrow_fast" in SUBMIT
    assert "required for arrow_trace" in SUBMIT


def test_batch_anchors_python_execution_in_verified_repository():
    assert 'cd -- "$REPO_ROOT" || die' in SBATCH
    assert 'export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in SBATCH


def test_batch_validates_explicit_runtime_and_pinned_offline_molmo_cache():
    assert "ARROW_SUITE_PYTHON must be a safe absolute Linux path" in SBATCH
    assert 'ARROW_SUITE_PYTHON_RESOLVED="$(realpath -e -- "$ARROW_SUITE_PYTHON")"' in SBATCH
    assert 'ARROW_SUITE_PYTHON target is not a safe executable regular path' in SBATCH
    assert 'PYTHON="${ARROW_SUITE_PYTHON:-' in SBATCH
    assert 'PYTHON="${ARROW_SUITE_PYTHON_RESOLVED:-' not in SBATCH
    assert 'explicit Arrow teacher runtime must retain its virtual-environment context' in SBATCH
    assert "ARROW_SUITE_HF_CACHE must be a safe absolute Linux path" in SBATCH
    assert "teacher-dependent policies require explicit ARROW_SUITE_PYTHON and ARROW_SUITE_HF_CACHE" in SBATCH
    assert "models--allenai--MolmoPoint-8B/snapshots/$MOLMO_REVISION" in SBATCH
    assert "models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/$SMOLVLM_REVISION" in SBATCH
    assert "for cache_layout in hub transformers" in SBATCH
    assert '"$smolvlm_snapshot/config.json"' in SBATCH
    assert '"$smolvlm_snapshot/processor_config.json"' in SBATCH
    assert '"$molmo_snapshot/processing_molmo2.py"' in SBATCH
    assert '"$molmo_snapshot/video_processing_molmo2.py"' in SBATCH
    assert 'legacy_cache="$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/cache/huggingface"' in SBATCH
    assert "MODEL_CACHE_ROOT=\"$RUN_ROOT/model_cache/hub\"" in SBATCH
    assert 'ln -s -- "$MOLMO_MODEL_DIR"' in SBATCH
    assert 'ln -s -- "$SMOLVLM_MODEL_DIR"' in SBATCH
    assert 'export HF_HUB_CACHE="$MODEL_CACHE_ROOT"' in SBATCH
    assert 'export TRANSFORMERS_CACHE="$MODEL_CACHE_ROOT"' in SBATCH
    assert "MOLMO_REVISION='188130f961c8e0888a34e11121a1423c461a01ba'" in SBATCH
    assert "SMOLVLM_REVISION='7b375e1b73b11138ff12fe22c8f2822d8fe03467'" in SBATCH
    assert 'export HF_MODULES_CACHE="$HF_CACHE/modules"' in SBATCH
    assert 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1' in SBATCH
    assert 'explicit Arrow teacher runtime requires {package}=={wanted}' in SBATCH
    assert '"transformers": "4.57.1"' in SBATCH


def test_sbatch_verifies_pure_receipt_and_archived_native_manifest():
    assert "ARROW_SUITE_OUTPUT must be a safe absolute Linux path" in SBATCH
    assert "ARROW_SUITE_OUTPUT must remain inside the run root" in SBATCH
    assert "ARROW_SUITE_OUTPUT already exists" in SBATCH
    assert 'tee "$RUN_ROOT/cli_stdout.log"' in SBATCH
    assert '"$PYTHON" - "$output" <<\'PY\'' in SBATCH
    assert "native execution receipt is missing or empty" in SBATCH
    assert '"$RUN_ROOT/COMPLETED"' in SBATCH
    assert '"$RUN_ROOT/native_run/run_manifest.json"' in SBATCH
    assert '"$ARCHIVE_ROOT/run/native_run/run_manifest.json"' in SBATCH
    assert '"$ARCHIVE_ROOT/run/COMPLETED"' in SBATCH


def test_collect_launcher_operation_uses_native_execution_contract():
    """The Legion collect operation must reach the executable handoff parser.

    The sbatch launcher invokes ``cli collect`` with the same --config,
    --factory, --execute, and run-directory arguments as canary/evaluate.
    Keep this as a parser-level regression test: a separate receipt-only
    ``collect CONFIG`` command cannot execute the training-source spine.
    """
    from vla_benchmarking.libero.arrow_policy_suite.cli import _cmd_handoff, build_parser

    args = build_parser().parse_args([
        "collect",
        "--config", "/tmp/study.json",
        "--factory", "example.factory:build",
        "--execute",
        "--policy", "arrow_on_call",
        "--run-dir", "/tmp/run",
        "--output", "/tmp/receipt.json",
        "--steps", "3",
    ])
    assert args.func is _cmd_handoff
    assert args.execute is True
