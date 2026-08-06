"""JSONL prompt audit logging for VLA context ablations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _flatten_input_ids(input_ids: Any) -> list[int]:
    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


class PromptAuditLogger:
    """Append prompt metadata, and optional tokenizer stats, to JSONL."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | None,
        *,
        model_id: str | None,
        enabled: bool | None = None,
        token_audit: bool | None = None,
    ):
        self.enabled = _bool_env("PROMPT_AUDIT", False) if enabled is None else enabled
        self.token_audit = _bool_env("TOKEN_AUDIT", False) if token_audit is None else token_audit
        self.model_id = model_id
        self.tokenizer_model_id = self._resolve_tokenizer_model_id(model_id)
        self.max_length_override = self._resolve_max_length_override(model_id)
        self._tokenizer = None
        self._tokenizer_error: str | None = None
        self._warned_tokenizer_error = False
        self._fh = None

        if not self.enabled:
            self.path = None
            return

        if output_dir is None:
            output_dir = "."
        path = Path(output_dir) / "prompt_audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _resolve_tokenizer_model_id(model_id: str | None) -> str | None:
        override = os.environ.get("TOKENIZER_MODEL") or os.environ.get("TOKENIZER_MODEL_ID")
        if override:
            return override
        if model_id and "smolvla" in model_id.lower():
            return "HuggingFaceTB/SmolVLM2-500M-Instruct"
        return model_id

    @staticmethod
    def _resolve_max_length_override(model_id: str | None) -> int | None:
        value = os.environ.get("TOKENIZER_MAX_LENGTH")
        if value:
            return int(value)
        if model_id and "smolvla" in model_id.lower():
            return 48
        return None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _load_tokenizer(self):
        if self._tokenizer is not None or self._tokenizer_error is not None:
            return self._tokenizer
        if not self.tokenizer_model_id:
            self._tokenizer_error = "tokenizer model is not set"
            return None

        token = os.environ.get("HF_TOKEN") or None
        try:
            from transformers import AutoProcessor, AutoTokenizer

            try:
                processor = AutoProcessor.from_pretrained(
                    self.tokenizer_model_id,
                    token=token,
                    trust_remote_code=True,
                )
                self._tokenizer = getattr(processor, "tokenizer", None)
            except Exception:
                self._tokenizer = None

            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_model_id,
                    token=token,
                    trust_remote_code=True,
                )
        except Exception as exc:
            self._tokenizer_error = f"{type(exc).__name__}: {exc}"
            return None

        return self._tokenizer

    def _token_stats(self, prompt: str) -> dict[str, Any]:
        if not self.token_audit:
            return {
                "token_audit_enabled": False,
                "token_count_before_truncation": None,
                "token_count_after_truncation": None,
                "truncation_occurred": None,
                "decoded_truncated_prompt": None,
                "tokenizer_error": None,
            }

        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            return {
                "token_audit_enabled": True,
                "token_count_before_truncation": None,
                "token_count_after_truncation": None,
                "truncation_occurred": None,
                "decoded_truncated_prompt": None,
                "tokenizer_error": self._tokenizer_error,
            }

        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
        before_ids = _flatten_input_ids(encoded.get("input_ids"))
        max_len = self.max_length_override or getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 1_000_000:
            max_len = None

        if max_len is None:
            after_ids = before_ids
        else:
            truncated = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
            )
            after_ids = _flatten_input_ids(truncated.get("input_ids"))

        try:
            decoded = tokenizer.decode(after_ids, skip_special_tokens=False)
        except Exception:
            decoded = None

        return {
            "token_audit_enabled": True,
            "tokenizer_model_id": self.tokenizer_model_id,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_model_max_length": max_len,
            "tokenizer_truncation_side": getattr(tokenizer, "truncation_side", None),
            "token_count_before_truncation": len(before_ids),
            "token_count_after_truncation": len(after_ids),
            "truncation_occurred": len(after_ids) < len(before_ids),
            "decoded_truncated_prompt": decoded,
            "tokenizer_error": None,
        }

    def log(
        self,
        *,
        prompt: str,
        task_text: str,
        context_mode: str,
        context_format: str,
        task_id: int | None,
        env_step: int | None,
        relations_generated: int,
        relations_retained: int,
    ) -> None:
        if not self.enabled or self._fh is None:
            return

        token_stats = self._token_stats(prompt)
        decoded = token_stats.get("decoded_truncated_prompt")
        record = {
            "format_name": context_format,
            "context_mode": context_mode,
            "task_id": task_id,
            "episode_id": None,
            "env_step": env_step,
            "raw_prompt": prompt,
            "task_text": task_text,
            "character_count": len(prompt),
            "number_of_relations_generated": relations_generated,
            "number_of_relations_retained": relations_retained,
            "task_instruction_present_after_truncation": (
                None if decoded is None else task_text in decoded
            ),
        }
        record.update(token_stats)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
