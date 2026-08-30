from __future__ import annotations

import io

import pytest

from prompt_audit import PromptAuditLogger


class _FakeTokenizer:
    model_max_length = 48
    truncation_side = "right"

    def __call__(self, prompt, *, add_special_tokens, truncation, max_length=None):
        assert add_special_tokens is True
        if truncation:
            return {"input_ids": [0] * max_length}
        return {"input_ids": [0] * 100}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "truncated text without the task instruction"


def _logger(tmp_path=None) -> PromptAuditLogger:
    # Avoid filesystem fixtures here: the Windows test host may not expose its
    # global temporary directory, while the audit logic itself is stream-based.
    logger = PromptAuditLogger(None, model_id="smolvla_libero", enabled=False, token_audit=True)
    logger.enabled = True
    logger._fh = io.StringIO()
    return logger


def _log(logger: PromptAuditLogger) -> None:
    logger.log(
        prompt="Task: Move the target bowl. Spatial relations: The target black bowl is left of the plate.",
        task_text="Move the target bowl.",
        context_mode="scene_graph",
        context_format="target_natural_v1",
        task_id=0,
        env_step=0,
        relations_generated=1,
        relations_retained=1,
        relations=[("akita_black_bowl_1", "is_left_of", "plate_1")],
        reset_sequence=1,
    )


def test_graph_prompt_audit_fails_closed_on_truncation_and_task_loss(monkeypatch):
    monkeypatch.setenv("TRAINING_PROFILE", "graph_treatment")
    logger = _logger()
    logger._tokenizer = _FakeTokenizer()
    try:
        with pytest.raises(RuntimeError, match="prompt was truncated"):
            _log(logger)
    finally:
        logger.close()


def test_graph_prompt_audit_fails_closed_when_tokenizer_is_unavailable(monkeypatch):
    monkeypatch.setenv("TRAINING_PROFILE", "arrow_graph_treatment")
    logger = _logger()
    logger._tokenizer_error = "offline tokenizer unavailable"
    try:
        with pytest.raises(RuntimeError, match="offline tokenizer unavailable"):
            _log(logger)
    finally:
        logger.close()


def test_graph_profile_uses_96_tokens_but_historical_smolvla_defaults_to_48(monkeypatch):
    monkeypatch.setenv("TRAINING_PROFILE", "graph_treatment")
    graph_logger = PromptAuditLogger(None, model_id="smolvla_libero", enabled=False)
    assert graph_logger.max_length_override == 96

    monkeypatch.setenv("TRAINING_PROFILE", "treatment")
    historical_logger = PromptAuditLogger(None, model_id="smolvla_libero", enabled=False)
    assert historical_logger.max_length_override == 48
