from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "finetuned_vlas"
    / "smolvla"
    / "target_arrow_only"
    / "legion"
    / "run_training.sbatch"
)


def test_target_training_canonicalizes_and_isolates_data_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "canonical_external_root()" in source
    assert 'realpath -m -- "$value"' in source
    assert '[[ "$canonical" != / ]]' in source
    assert '[[ ! -L "$value" ]]' in source
    assert '"$REPO_ROOT"|"$REPO_ROOT"/*' in source
    assert "assert_disjoint_root TARGET_ARROW_HDF5_ROOT" in source
    assert "assert_disjoint_root TARGET_ARROW_DATA_ROOT" in source
    assert "TARGET_ARROW_DATA_CACHE" in source
    assert 'assert_disjoint_root "$data_name" "$data_path" RUN_ROOT' in source
    assert 'assert_disjoint_root "$data_name" "$data_path" ARCHIVE_ROOT' in source


def test_target_training_serializes_shared_dataset_preparation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'DATA_LOCK="$DATA_CACHE/.sealed-target-arrow-${BASE_POLICY_REVISION}-target_arrow_treatment.lock"' in source
    assert 'exec 8>>"$DATA_LOCK"' in source
    assert "flock -x 8" in source
    assert "flock -u 8" in source
    assert "target_arrow_treatment" in source
    assert "BASE_POLICY_REVISION" in source


def test_target_training_keeps_scope_and_absolute_helper_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "smoke) STEPS=2; SAVE_FREQ=2" in source
    assert "full) STEPS=29190; SAVE_FREQ=1946" in source
    assert 'bash "$PREPARE_DATA_SCRIPT" target_arrow_treatment' in source
    assert 'bash "$PIPELINE_SCRIPT" "$SCOPE" --profile target-arrow' in source
    assert '--python "$PYTHON"' in source
    assert 'sbatch "$PREPARE_DATA_SCRIPT"' not in source
    assert 'sbatch "$PIPELINE_SCRIPT"' not in source


def test_target_training_requires_fresh_external_run_and_archive_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '[[ ! -e "$RUN_ROOT" ]]' in source
    assert '[[ ! -e "$ARCHIVE_ROOT" ]]' in source
    assert 'case "$RUN_ROOT" in "$REPO_ROOT"' in source
    assert 'case "$ARCHIVE_ROOT" in "$REPO_ROOT"' in source
    assert 'case "$ARCHIVE_ROOT" in "$RUN_ROOT"' in source
    assert 'case "$RUN_ROOT" in "$ARCHIVE_ROOT"' in source
    assert 'TARGET_ARROW_RUN_ROOT' in source
    assert 'TARGET_ARROW_ARCHIVE_ROOT' in source
