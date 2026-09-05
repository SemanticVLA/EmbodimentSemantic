from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
WORKFLOW_ROOT = ROOT / "finetuned_vlas" / "smolvla" / "workflows"


def test_only_two_smolvla_profiles_are_operator_reachable() -> None:
    pipeline = (WORKFLOW_ROOT / "run_smolvla_pipeline.sh").read_text(encoding="utf-8")
    assert "no_arrow_treatment" in pipeline
    assert "target_arrow_treatment" in pipeline
    assert "graph_treatment" not in pipeline
    assert "action_visual_lora" not in pipeline


def test_target_profile_uses_single_goal_arrow_pair_contract() -> None:
    launch = (WORKFLOW_ROOT / "launch_lora_treatment.sh").read_text(encoding="utf-8")
    prepare = (WORKFLOW_ROOT / "prepare_lambda_data.sh").read_text(encoding="utf-8")
    evaluate = (WORKFLOW_ROOT / "run_lora_no_arrow_pair_eval.py").read_text(encoding="utf-8")
    assert "target_arrow_treatment" in launch
    assert "sealed_lora_control_target_arrow_treatment" in launch
    assert "convert-target-arrow-pair" in prepare
    assert "visual_goal_arrow" in evaluate
    assert "DISABLE_VISUAL_PROMPT_HINT" in evaluate


def test_training_constants_are_shared_and_sealed() -> None:
    for name in ("launch_lora_treatment.sh", "train_lora.sh"):
        source = (WORKFLOW_ROOT / name).read_text(encoding="utf-8")
        assert "29190" in source
        assert "1946" in source
        assert "BATCH_SIZE" in source
        assert "SEED" in source
        assert "PEFT_R" in source


def test_workflow_shell_scripts_parse() -> None:
    for path in WORKFLOW_ROOT.glob("*.sh"):
        # Bash on the Windows test host receives POSIX-relative paths through
        # the compatibility layer; native drive-letter paths are mangled.
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(["bash", "-n", relative_path], capture_output=True, text=True, cwd=REPO_ROOT)
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_target_arrow_legion_training_launcher_is_immutable_and_scoped() -> None:
    launcher_path = (
        ROOT
        / "finetuned_vlas"
        / "smolvla"
        / "target_arrow_only"
        / "legion"
        / "run_training.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    assert "TARGET_ARROW_TRAINING_EXPECTED_COMMIT" in launcher
    assert "TARGET_ARROW_TRAINING_LABEL" in launcher
    assert "TARGET_ARROW_TRAINING_SCOPE" in launcher
    assert "smoke) STEPS=2; SAVE_FREQ=2" in launcher
    assert "full) STEPS=29190; SAVE_FREQ=1946" in launcher
    assert 'bash "$PREPARE_DATA_SCRIPT" target_arrow_treatment' in launcher
    assert 'bash "$PIPELINE_SCRIPT" "$SCOPE" --profile target-arrow' in launcher
    assert "--python \"$PYTHON\"" in launcher
    assert "TARGET_ARROW_BASE_POLICY" in launcher
    assert "TARGET_ARROW_LIBERO_DIR" in launcher
    assert "TARGET_ARROW_HDF5_ROOT" in launcher
    assert "ARCHIVE_ROOT" in launcher
    assert "#SBATCH --exclude=compute-4-13" in launcher
    assert "#SBATCH --time=0-23:59:00" in launcher
    assert "TARGET_ARROW_TRAINING_EXPECTED_COMMIT" in launcher
    assert 'sbatch "$PREPARE_DATA_SCRIPT"' not in launcher
    assert 'sbatch "$PIPELINE_SCRIPT"' not in launcher


def test_target_arrow_legion_launcher_has_no_repo_output_fallback() -> None:
    launcher_path = (
        ROOT
        / "finetuned_vlas"
        / "smolvla"
        / "target_arrow_only"
        / "legion"
        / "run_training.sbatch"
    )
    launcher = launcher_path.read_text(encoding="utf-8")
    assert 'REPO_ROOT="$REPO_ROOT/vla_benchmarking/libero"' not in launcher
    assert 'case "$RUN_ROOT" in "$REPO_ROOT"' in launcher
    assert 'case "$ARCHIVE_ROOT" in "$REPO_ROOT"' in launcher
    assert '[[ ! -e "$RUN_ROOT" ]]' in launcher
    assert '[[ ! -e "$ARCHIVE_ROOT" ]]' in launcher
