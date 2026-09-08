from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


pytest.importorskip("lerobot")
from vla_benchmarking.libero.evaluation import run_lerobot_eval_with_context as runtime


def _paired_env() -> SimpleNamespace:
    return SimpleNamespace(
        task_id=0,
        _randomization_env_index=2,
        _randomization_reset_sequence=7,
        _init_state_id=3,
        episode_length=280,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        observation_height=256,
        observation_width=256,
    )


def test_paired_mode_requires_explicit_valid_flag(monkeypatch):
    monkeypatch.delenv("PEFT_PAIRED_EVAL", raising=False)
    assert runtime._paired_evaluation_enabled() is False

    monkeypatch.setenv("PEFT_PAIRED_EVAL", "1")
    assert runtime._paired_evaluation_enabled() is True
    monkeypatch.setenv("PEFT_PAIRED_EVAL", "false")
    assert runtime._paired_evaluation_enabled() is False

    monkeypatch.setenv("PEFT_PAIRED_EVAL", "maybe")
    with pytest.raises(SystemExit, match="PEFT_PAIRED_EVAL"):
        runtime._paired_evaluation_enabled()


def test_paired_reset_identity_uses_pre_reset_init_state_evidence():
    identity = runtime._paired_reset_identity(
        sub_env=_paired_env(),
        task_id=0,
        env_index=2,
        reset_sequence=7,
        reset_details={
            "init_state": {
                "selected_index": 3,
                "selected_row_sha256": "a" * 64,
            }
        },
    )

    assert identity == {
        "task_id": 0,
        "env_index": 2,
        "reset_sequence": 7,
        "selected_init_state_index": 3,
        "init_state_sha256": "a" * 64,
        "horizon": 280,
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        "resolution": {"height": 256, "width": 256},
    }


def test_paired_reset_identity_fails_without_selected_state():
    with pytest.raises(RuntimeError, match="pre-reset init-state evidence"):
        runtime._paired_reset_identity(
            sub_env=_paired_env(),
            task_id=0,
            env_index=2,
            reset_sequence=7,
            reset_details={},
        )


def test_standard_paired_audit_persists_reset_identity(tmp_path):
    logger = runtime.RandomizationAuditLogger(tmp_path)
    logger.log(
        task_id=0,
        env_index=2,
        reset_sequence=7,
        dimensions_enabled={"scene_layout": False, "object_removal": True, "prompt_variant": True},
        dimensions_realized={"scene_layout": False, "object_removal": True, "prompt_variant": False},
        details={
            "init_state": {"selected_index": 3, "selected_row_sha256": "b" * 64},
            "status": "environment_ok",
        },
    )
    wrapper = runtime.TaskContextVecEnv(
        env=SimpleNamespace(),
        live_generator=None,
        context_mode="standard",
        randomization_audit_logger=logger,
        suite_mode="sealed_randomized",
        paired_reset_mode=True,
    )
    wrapper._audit_randomization(
        prompts=[runtime.TASK_PROMPT_OVERRIDE[0]],
        canonical_task_texts=["canonical"],
        effective_task_texts=[runtime.TASK_PROMPT_OVERRIDE[0]],
        sub_envs=[_paired_env()],
    )
    logger.close()

    records = [json.loads(line) for line in (tmp_path / "randomization_audit.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["details"]["reset_identity"] == {
        "task_id": 0,
        "env_index": 2,
        "reset_sequence": 7,
        "selected_init_state_index": 3,
        "init_state_sha256": "b" * 64,
        "horizon": 280,
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        "resolution": {"height": 256, "width": 256},
    }


def test_paired_vector_autoreset_is_disabled():
    try:
        from gymnasium.vector.vector_env import AutoresetMode
    except ImportError:
        pytest.skip("gymnasium is unavailable")
    env = SimpleNamespace(autoreset_mode=AutoresetMode.NEXT_STEP)
    runtime._disable_vector_autoreset(env)
    assert env.autoreset_mode == AutoresetMode.DISABLED
    assert env._paired_explicit_reset_mode is True
