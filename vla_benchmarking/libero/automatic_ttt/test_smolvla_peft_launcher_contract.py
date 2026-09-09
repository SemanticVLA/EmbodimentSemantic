import json
from contextlib import redirect_stdout
from io import StringIO
import re
import sys
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).parent / "legion" / "run_smolvla_peft_arrow_task.sbatch"
CANARY = Path(__file__).parent / "legion" / "run_smolvla_arrow_collector_canary.sbatch"
EVALUATOR = Path(__file__).parents[1] / "evaluation" / "run_lerobot_eval_with_context.py"


def test_launcher_is_scalar_task_specific_and_manifest_driven():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "PEFT_TASK_ID=\"${PEFT_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}\"" in text
    assert "PEFT_TASK_IDS" not in text
    assert "TASK_IDS_JSON=\"[$TASK_ID]\"" in text
    assert "PEFT_ARROW_COLLECTION_MANIFEST" in text
    assert "collect_smolvla_arrow_corrections" in text
    assert '--accepted-target "$ARROW_DEMOS"' in text
    assert "--adaptation-seed-start 3000" in text
    assert '--max-attempts "$MAX_ATTEMPTS"' in text
    assert "MAX_ATTEMPTS=500" in text
    assert "--factory vla_benchmarking.libero.automatic_ttt.smolvla_arrow_factory:collect" in text
    assert "COLLECTION_ROOT=\"$RUN_ROOT/collection\"" in text
    assert 'if [[ -z "$COLLECTION_MANIFEST" ]]; then' in text
    assert 'COLLECTION_MANIFEST="$COLLECTION_ROOT/collection_manifest.json"' in text
    assert "arrow_grasp_controller_fresh_demonstration" in text
    assert "--collection-mode fresh_arrow" in text
    assert "hdf5_to_lerobot_dataset" not in text
    assert "convert-pair" not in text
    assert '--steps="$STEPS"' in text
    assert "--peft.method_type=LORA" in text
    assert '--peft.full_training_modules="[]"' in text
    for unsupported in ("--peft.lora_alpha", "--peft.lora_dropout", "--peft.bias", "--peft.init_lora_weights", "--peft.use_rslora", "--peft.fan_in_fan_out", "--peft.modules_to_save"):
        assert unsupported not in text
    assert '--policy.optimizer_lr="$PEAK_LR"' in text
    assert '--policy.optimizer_weight_decay="$WEIGHT_DECAY"' in text
    assert '--policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS"' in text
    assert "LIBERO_CONFIG_PATH" in text
    assert "SMOLVLA_BASE_POLICY" in text
    assert "ARROW_CONTROLLER_CONFIG" in text
    assert "PEFT_EXPECTED_CONTROLLER_HASH" in text
    assert '"$CONTROLLER_CONFIG_HASH" == "$EXPECTED_CONTROLLER_HASH"' in text
    assert "save_peft_adapter(" in text
    assert text.count("save_peft_adapter(") == 1
    assert "module load miniforge/24.3.0-0" in text


def test_launcher_supports_one_shot_skip_baseline_without_removing_baseline_path():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'PEFT_ARROW_DEMOS="${PEFT_ARROW_DEMOS:-50}"' in text
    assert 'PEFT_SKIP_BASELINE="${PEFT_SKIP_BASELINE:-0}"' in text
    assert 'if [[ "$PEFT_SKIP_BASELINE" == 0 ]]; then' in text
    assert 'if [[ "$PEFT_SKIP_BASELINE" == 1 ]]; then' in text
    assert "probe_libero_eval_resets" in text
    assert "eval_reset_contract validate-audit" in text
    assert 'baseline_stage.v1' in text
    assert '"evaluation_mode":"adapted_only"' in text
    assert '"mode":"adapted_only"' in text
    assert '"status":"COMPLETED"' in text
    assert '"comparison_available":baseline is not None' in text
    assert '"improvement_claim_supported":False' in text
    assert '"success_delta":(adapted["success_rate"]-baseline["success_rate"] if baseline is not None else None)' in text
    assert '"task_id": int(os.environ["TASK_ID"])' in text
    assert '"base_policy_revision": os.environ["BASE_POLICY_REVISION"]' in text
    assert '"eval_seeds_reserved": list(range(1000, 1010))' in text


def test_embedded_launcher_python_heredocs_compile():
    """Catch shell-heredoc indentation errors before submitting a GPU job."""
    text = LAUNCHER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, flags=re.DOTALL)
    assert blocks, "launcher must contain embedded Python validation blocks"
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER}:heredoc-{index}", "exec")


def test_runtime_evidence_validation_maps_optimizer_lr_to_base_lr_and_rejects_missing(tmp_path, monkeypatch):
    """The runtime optimizer field is ``lr`` while the sealed contract uses ``base_lr``."""
    text = LAUNCHER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, flags=re.DOTALL)
    block = next(
        block for block in blocks
        if 'value.get("optimizer_hyperparameters", {})' in block
    )
    settings = {
        "STEPS": "3",
        "PEAK_LR": "5e-5",
        "WEIGHT_DECAY": "1e-5",
        "OPTIMIZER_EPS": "1e-8",
        "GRAD_CLIP": "10.0",
        "SCHEDULER_NAME": "cosine_decay_with_warmup",
        "SCHEDULER_WARMUP_STEPS": "0",
        "SCHEDULER_DECAY_STEPS": "3",
        "SCHEDULER_DECAY_LR": "2.5e-6",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    expected = {
        "updates": 3,
        "base_lr": 5e-5,
        "weight_decay": 1e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "grad_clip_norm": 10.0,
        "scheduler": "cosine_decay_with_warmup",
        "warmup_steps": 0,
        "decay_steps": 3,
        "decay_lr": 2.5e-6,
    }
    runtime = tmp_path / "runtime_evidence.json"
    runtime.write_text(json.dumps({
        "attestation_status": "VERIFIED",
        "updates_observed": 3,
        "optimizer_class": "torch.optim.adamw.AdamW",
        "scheduler_class": "torch.optim.lr_scheduler.LambdaLR",
        "expected_contract": expected,
        "optimizer_hyperparameters": {
            "lr": 5e-5, "weight_decay": 1e-5, "eps": 1e-8,
            "grad_clip_norm": 10.0, "betas": [0.9, 0.95],
        },
        "optimizer_scheduler_objects_checked": True,
        "lr_milestones_consistent": True,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "min_learning_rate": 2.5e-6,
        "max_learning_rate": 5e-5,
        "last_learning_rate": 2.5e-6,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(LAUNCHER), str(runtime)])

    exec(compile(block, f"{LAUNCHER}:runtime-validation", "exec"), {})

    missing_lr = json.loads(runtime.read_text(encoding="utf-8"))
    del missing_lr["optimizer_hyperparameters"]["lr"]
    runtime.write_text(json.dumps(missing_lr) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="runtime optimizer evidence mismatch"):
        exec(compile(block, f"{LAUNCHER}:runtime-validation-missing-lr", "exec"), {})


def test_all_task_wrapper_seals_one_demo_and_skips_baseline():
    text = (LAUNCHER.parent / "run_smolvla_peft_arrow_all_tasks.sbatch").read_text(encoding="utf-8")
    assert 'export PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1' in text
    assert '${PEFT_ARROW_DEMOS:-}" != 1' in text
    assert '${PEFT_SKIP_BASELINE:-}" != 1' in text
    assert 'VALIDATION_PYTHON="${PEFT_PYTHON:-$HOME/EmbodimentSemantic_runtime/EmbodimentSemantic/grasp_controller/venv-py312-eacdc54a9663db0a/bin/python}"' in text
    assert '[[ -x "$VALIDATION_PYTHON" ]]' in text
    assert 'PEFT_START_TASK_ID="${PEFT_START_TASK_ID:-0}"' in text
    assert '(( PEFT_START_TASK_ID <= 9 ))' in text
    assert '(( PEFT_START_TASK_ID == 0 ))' not in text
    assert 'FIRST_TASK_ID="$PEFT_START_TASK_ID"' in text
    assert 'for task_id in $(seq "$FIRST_TASK_ID" 9); do' in text
    assert 'bash "$RUNNER"' in text


def test_all_task_wrapper_resumes_task_zero_then_starts_fresh_at_task_one():
    text = (LAUNCHER.parent / "run_smolvla_peft_arrow_all_tasks.sbatch").read_text(encoding="utf-8")
    resume_branch = text.index('if [[ -n "${PEFT_RESUME_SOURCE_RUN_ROOT:-}" ]]; then')
    task_loop = text.index('for task_id in $(seq "$FIRST_TASK_ID" 9); do')
    resume_section = text[resume_branch:task_loop]

    assert 'PEFT_RESUME_SOURCE_RUN_ROOT is permitted only when PEFT_START_TASK_ID=0' in text
    assert 'bash "$RESUME_RUNNER"' in resume_section
    assert '[[ -f "$PEFT_RUN_ROOT/COMPLETED" ]]' in resume_section
    assert 'publication_recovery_receipt.json' in resume_section
    assert 'expected an object' in resume_section
    assert '"$VALIDATION_PYTHON" - "$RECOVERY_RECEIPT" <<\'PY\'' in resume_section
    assert 'python "$RECOVERY_RECEIPT" <<\'PY\'' not in resume_section
    assert 'does not point to an existing absolute artifact' in resume_section
    assert 'unset PEFT_RESUME_SOURCE_RUN_ROOT PEFT_RESUME_RUNNER' in resume_section
    assert 'PEFT_START_TASK_ID=1' in resume_section
    assert task_loop > resume_branch


def test_resume_submit_mode_reuses_canary_and_passes_source_to_single_job():
    submitter = (LAUNCHER.parent / "submit_smolvla_task_specific_array.ps1").read_text(encoding="utf-8")
    assert "'resume-submit'" in submitter
    assert '$ResumeSourceRunRoot' in submitter
    assert "resume-submit requires -ResumeSourceRunRoot" in submitter
    assert 'ResumeSourceRunRoot must be disjoint from RemoteRunRoot and RemoteArchiveRoot.' in submitter
    resume_branch = submitter.index("elif [[ '__MODE__' == 'resume-submit' ]]; then")
    resume_section = submitter[resume_branch:]
    assert 'collector_canary_job=REUSED' in resume_section
    assert 'PEFT_START_TASK_ID=0' in resume_section
    assert 'PEFT_RESUME_SOURCE_RUN_ROOT="$resume_source"' in resume_section
    assert '--partition=gpu_a40_ext' in resume_section
    assert '--dependency=' not in resume_section
    assert 'resume_source="$(realpath -m -- "$resume_source")"' in resume_section
    assert 'resume source overlaps new run/archive roots' in resume_section


def test_canary_is_adapted_only_and_uses_shared_reset_contract():
    text = CANARY.read_text(encoding="utf-8")
    assert 'export PEFT_ARROW_DEMOS=1 PEFT_SKIP_BASELINE=1' in text
    assert "probe_libero_eval_resets" in text
    assert "OUTPUT_DIR=\"$out\"" in text
    assert '"adapted_eval_seeds":[1000]' in text
    assert '"paired_eval_seeds"' not in text


def test_launcher_uses_matching_sealed_evaluation_seeds():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "EVAL_EPISODES=10" in text
    assert "SEED=1000" in text
    assert "list(range(1000,1010))" in text
    assert "training_scope\":\"task_specific\"" in text
    assert 'print(int(sys.argv[1]) // 30)' in text
    assert 'SCHEDULER_DECAY_STEPS="$STEPS"' in text
    assert "SCHEDULER_DECAY_LR=2.5e-6" in text
    assert '--save_freq="$SAVE_FREQ"' in text
    assert 'print(min(2000, int(sys.argv[1])))' in text
    assert 'print(repr(int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3])))' in text
    assert 'format(int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3]), ".8f")' not in text
    assert 'task_id = int(sys.argv[3])' in text
    assert 'eval_info, output, task_id = map(pathlib.Path, sys.argv[1:])' not in text
    assert "REQUESTED_EPOCHS=5" in text
    assert "math.ceil(epochs*frames/batch)" in text
    assert text.count("PEFT_PAIRED_EVAL=1") == 2
    assert "reset identities" in text
    assert "runtime_evidence=os.environ[\"TRAIN_META_ROOT\"]" in text
    assert "EVAL_CAMERAS='agentview_image,robot0_eye_in_hand_image'" in text
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    assert 'os.environ.get("EVAL_CAMERAS"' in evaluator
    assert 'os.environ.get("LIBERO_EPISODE_LENGTH")' in evaluator
    assert 'os.environ.get("EVAL_RESOLUTION")' in evaluator
    assert '(("--env.episode_length",), episode_length)' in evaluator
    assert '(("--env.observation_height",), resolution)' in evaluator
    assert '(("--env.observation_width",), resolution)' in evaluator


def test_epoch_equivalent_heredoc_preserves_full_round_trip_precision():
    """The artifact validator must receive the exact integer-derived float."""
    text = LAUNCHER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, flags=re.DOTALL)
    block = next(
        block for block in blocks
        if "print(repr(int(sys.argv[1])*int(sys.argv[2])/int(sys.argv[3])))" in block
    )
    output = StringIO()
    old_argv = sys.argv
    try:
        sys.argv = [str(LAUNCHER), "54", "8", "86"]
        with redirect_stdout(output):
            exec(compile(block, f"{LAUNCHER}:epoch-equivalent", "exec"), {})
    finally:
        sys.argv = old_argv
    value = output.getvalue().strip()
    assert value == "5.023255813953488"
    assert float(value) == 54 * 8 / 86


def test_live_run_order_is_baseline_then_collection_then_training_then_eval():
    text = LAUNCHER.read_text(encoding="utf-8")
    baseline = text.index("baseline evaluation task")
    collection = text.index("collect_smolvla_arrow_corrections")
    training = text.index("training task")
    adapted = text.index("adapted evaluation task")
    assert baseline < collection < training < adapted


def test_launchers_switch_from_arrow_cache_to_pinned_smolvlm_cache_before_training():
    text = LAUNCHER.read_text(encoding="utf-8")
    baseline = text.index("baseline evaluation task")
    live_branch = text.index('if [[ -z "$COLLECTION_MANIFEST" ]]; then')
    arrow_switch = text.index("use_arrow_cache", live_branch)
    collector = text.index("collect_smolvla_arrow_corrections", live_branch)
    smol_switch_after_collection = text.index("use_smolvla_cache", collector)
    audit = text.index('"$PYTHON" "$AUDITOR" --generate-expected')
    assert text.index("use_smolvla_cache") < baseline < live_branch
    assert live_branch < arrow_switch < collector < smol_switch_after_collection < audit
    assert "models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots" in text
    assert 'HF_HUB_CACHE="$SMOLVLA_HF_CACHE/hub"' in text

    canary = CANARY.read_text(encoding="utf-8")
    switch = canary.index('SMOLVLA_HF_CACHE="${SMOLVLA_HF_CACHE:-$HOME/.cache/huggingface}"')
    audit = canary.index('"$PYTHON" "$AUDITOR" --generate-expected')
    assert switch < audit
    assert "models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots" in canary
    assert "--peft.method_type=LORA" in canary
    assert "--peft.full_training_modules='[]'" in canary


def test_canary_uses_short_tmpdir_for_torch_multiprocessing_sockets():
    canary = CANARY.read_text(encoding="utf-8")
    assert 'TMP_ROOT="/tmp/peft-canary-${SLURM_JOB_ID}"' in canary
    assert 'export TMPDIR="$TMP_ROOT"' in canary
    assert 'TMPDIR="$RUN_ROOT/tmp"' not in canary
