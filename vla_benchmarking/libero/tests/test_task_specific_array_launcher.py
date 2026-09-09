"""Static contracts for the Legion canary plus one sequential all-task job."""

from pathlib import Path
import subprocess


LEGION = Path(__file__).parents[1] / "automatic_ttt" / "legion"
LAUNCHER = LEGION / "submit_smolvla_task_specific_array.ps1"
CANARY = LEGION / "run_smolvla_arrow_collector_canary.sbatch"
ALL_TASKS = LEGION / "run_smolvla_peft_arrow_all_tasks.sbatch"


def test_array_launcher_is_task_specific_and_five_epoch_budget() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--array=" not in source
    assert "single_sequential_job" in source
    all_tasks = ALL_TASKS.read_text(encoding="utf-8")
    assert 'for task_id in $(seq "$FIRST_TASK_ID" 9); do' in all_tasks
    assert 'export PEFT_TASK_ID="$task_id"' in all_tasks
    assert "unset PEFT_TASK_IDS PEFT_ARROW_COLLECTION_MANIFEST" in all_tasks
    assert "arrow_demos=1" in source
    assert "collection_mode=fresh_arrow" in source
    assert "requested_epochs=5" in source
    assert "optimizer_steps=derived_from_dataset" in source
    assert "PEFT_STEPS=20000" not in source
    assert "PEFT_EPOCHS" not in source
    assert "run_smolvla_peft_arrow_all_tasks.sbatch" in source
    assert "run_smolvla_peft_arrow_task.sbatch" in all_tasks
    assert "run_smolvla_arrow_collector_canary.sbatch" in source


def test_array_launcher_has_immutable_canary_gate_and_separate_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'git -C "$repo" rev-parse HEAD' in source
    assert 'git -C "$repo" status --porcelain --untracked-files=all' in source
    assert "afterok:" in source
    assert "sbatch --parsable" in source
    assert "PEFT_CANARY_CONTROLLER_HASH" in source
    assert "PEFT_EXPECTED_CONTROLLER_HASH" in source
    assert "PEFT_EXPECTED_CONTROLLER_HASH" in source
    assert "canary_run_root" in source
    assert "canary_archive_root" in source
    assert "RemoteCollectionRoot" not in source
    assert "ARRAY_COLLECTION_ROOT" not in source
    assert "if ($Mode -eq 'submit' -and -not $ConfirmExpensiveRun)" in source


def test_canary_validates_collection_training_reload_and_adapted_eval() -> None:
    source = CANARY.read_text(encoding="utf-8")
    assert "--accepted-target 1" in source
    assert "COLLECTOR_FACTORY=vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory:collect" in source
    assert '--factory "$COLLECTOR_FACTORY"' in source
    assert "load_arrow_collection_manifest" in source
    assert "expected_successes=1" in source
    assert 'value.get("collection_mode") != "fresh_arrow"' in source
    assert 'raw.get("starts_from_reset") is not True' in source
    assert 'raw.get("vla_called") is not False' in source
    assert "collector_canary_marker.json" in source
    assert "--steps=2 --save_freq=2 --eval_freq=0 --batch_size=8" in source
    assert 'v.get("updates_observed") != 2' in source
    assert '"optimizer_updates":2' in source
    assert "validate_eval_info" in source
    assert "validate_randomization_audit" in source
    assert "adapted sealed reset identity matches skip-mode reservation" in source
    assert "N_ACTION_STEPS=checkpoint LIBERO_EPISODE_LENGTH=280 EVAL_RESOLUTION=256" in source
    assert "EVAL_CAMERAS=agentview_image,robot0_eye_in_hand_image" in source
    assert '"adapted_eval_seeds":[1000]' in source
    assert "inventory.sha256" in source
    assert "tree_sha256" in source
    assert "status=VERIFIED" in source
    assert "_task_description" in source
    assert "TASK_DESCRIPTION='pick up" not in source
    assert source.index("trap archive_on_exit EXIT") < source.index('COLLECTION_ROOT="$RUN_ROOT/collection"')


def test_shell_scripts_parse() -> None:
    for script in (CANARY, ALL_TASKS, LEGION / "run_smolvla_peft_arrow_task.sbatch"):
        relative_script = script.relative_to(Path.cwd()).as_posix()
        result = subprocess.run(
            ["bash", "-n", relative_script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
