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
