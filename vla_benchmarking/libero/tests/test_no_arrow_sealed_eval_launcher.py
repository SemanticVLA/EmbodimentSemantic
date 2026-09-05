from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "finetuned_vlas"
    / "smolvla"
    / "no_arrows"
    / "legion"
    / "run_sealed_randomized_eval.sbatch"
)


def test_no_arrow_sealed_launcher_is_isolated_and_hash_locked():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--partition=gpu_a40" in text
    assert "NO_ARROW_EVAL_SCOPE must be smoke or full" in text
    assert "EPISODES=1" in text and "EPISODES=50" in text
    assert 'TASK_IDS_CSV="0,4"' in text
    assert '--task-ids "$TASK_IDS_CSV"' in text
    assert "NO_ARROW_EVAL_EXPECTED_COMMIT" in text
    assert 'shared HOME checkout is forbidden' in text
    assert "80b3c23fc3987530d57766ab45ed33db918f08983739139c1ff0397184cc7092" in text
    assert "6382c39c1df2e4ccfc43946b36aa014abb73e1767366f009999415981b01efb1" in text
    assert "89a570fb1d07e93ec16adde1e78f18b0f0f7c0148da7896a7b5ba257797658f8" in text
    assert "95e376aff504265bea2bb53e63cc221fb42d7baa01dd6c3810317de85875c391" in text
    assert 'sha256sum -- "$ADAPTER_DIR/adapter_config.json"' in text
    assert 'sha256sum -- "$ADAPTER_DIR/train_config.json"' in text
    assert "--protocol \"$SCOPE\"" in text
    assert "--episodes \"$EPISODES\"" in text
    assert "--no-videos" in text
    assert "visual_condition=none" in text
    assert "archive_on_exit" in text and "inventory.sha256" in text


def test_no_arrow_sealed_launcher_uses_organized_entrypoint_only():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "finetuned_vlas/smolvla/workflows/run_no_arrow_sealed_eval.py" in text
    assert "run_lora_no_arrow_pair_eval.py" not in text
    assert "run_lerobot_eval_with_context.py" not in text
