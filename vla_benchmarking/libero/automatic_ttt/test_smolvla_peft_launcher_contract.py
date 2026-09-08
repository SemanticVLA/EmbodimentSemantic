from pathlib import Path


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
