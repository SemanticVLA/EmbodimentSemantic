from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "finetuned_vlas"
    / "smolvla"
    / "no_arrows"
    / "legion"
    / "run_training.sbatch"
)


def test_no_arrow_training_launcher_is_present_and_bash_valid() -> None:
    assert LAUNCHER.is_file()
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER.relative_to(ROOT)).replace("\\", "/")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_arrow_training_launcher_is_immutable_and_scoped() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "NO_ARROW_TRAINING_EXPECTED_COMMIT" in source
    assert "NO_ARROW_TRAINING_LABEL" in source
    assert "NO_ARROW_TRAINING_SCOPE" in source
    assert "smoke) STEPS=2; SAVE_FREQ=2" in source
    assert "full) STEPS=29190; SAVE_FREQ=1946" in source
    assert 'bash "$PREPARE_DATA_SCRIPT" no_arrow_treatment' in source
    assert 'bash "$PIPELINE_SCRIPT" "$SCOPE" --profile no-arrow' in source
    assert '--python "$PYTHON"' in source
    assert "BASE_POLICY_REVISION" in source
    assert "LIBERO_COMMIT" in source
    assert "#SBATCH --exclude=compute-4-13" in source
    assert "#SBATCH --time=0-23:59:00" in source
    assert 'sbatch "$PREPARE_DATA_SCRIPT"' not in source
    assert 'sbatch "$PIPELINE_SCRIPT"' not in source


def test_no_arrow_training_launcher_reuses_only_an_explicit_verified_pair() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'DATA_MODE="${NO_ARROW_TRAINING_DATA_MODE:-prepare}"' in source
    assert "prepare|reuse_verified" in source
    assert "reuse_verified is permitted only for an explicit smoke run" in source
    assert "NO_ARROW_TRAINING_VERIFIED_DATA_ROOT" in source
    assert "sealed_lora_pair_manifest.json" in source
    assert "sealed_lora_pair_verified.json" in source
    assert '--mode verify --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"' in source
    assert '--mode preflight --data-dir "$HDF5_ROOT" --output-root "$DATA_ROOT"' in source
    assert 'DATA_LOCK="$DATA_CACHE/.sealed-no-arrow-${BASE_POLICY_REVISION}-no_arrow_treatment.lock"' in source
    assert 'exec 8>>"$DATA_LOCK"' in source
    assert "flock -x 8" in source
    assert "flock -u 8" in source


def test_no_arrow_training_launcher_canonicalizes_and_isolates_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "canonical_external_root()" in source
    assert 'realpath -m -- "$value"' in source
    assert '[[ "$canonical" != / ]]' in source
    assert '[[ ! -L "$value" ]]' in source
    assert '"$REPO_ROOT"|"$REPO_ROOT"/*' in source
    assert "assert_disjoint_root NO_ARROW_HDF5_ROOT" in source
    assert "assert_disjoint_root NO_ARROW_DATA_ROOT" in source
    assert 'assert_disjoint_root "$data_name" "$data_path" RUN_ROOT' in source
    assert 'assert_disjoint_root "$data_name" "$data_path" ARCHIVE_ROOT' in source
    assert '[[ ! -e "$RUN_ROOT" ]]' in source
    assert '[[ ! -e "$ARCHIVE_ROOT" ]]' in source
    assert 'case "$RUN_ROOT" in "$REPO_ROOT"' in source
    assert 'case "$ARCHIVE_ROOT" in "$REPO_ROOT"' in source
    assert "archive_on_exit" in source
    assert "inventory.sha256" in source


def test_no_arrow_training_launcher_uses_bounded_job_temp_and_cleans_only_it() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'TMP_ROOT="/tmp/smvla-noarrow-${SLURM_JOB_ID}"' in source
    assert 'export TMPDIR="$TMP_ROOT"' in source
    assert '[[ "$TMP_ROOT" == "/tmp/smvla-noarrow-${SLURM_JOB_ID}" ]]' in source
    assert '[[ ! -e "$TMP_ROOT" ]]' in source
    assert 'rm -rf -- "$TMP_ROOT"' in source
    assert "cleanup_short_tmp" in source
    assert 'TMPDIR="$RUN_ROOT/tmp"' not in source
    # The fixed prefix plus a normal numeric Slurm job id stays well below the
    # Linux AF_UNIX pathname limit even before Python appends socket names.
    assert len("/tmp/smvla-noarrow-" + "9" * 10) < 50
