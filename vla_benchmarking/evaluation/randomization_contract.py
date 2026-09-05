"""Stable serialization of the configured LIBERO randomization contract."""

from __future__ import annotations

from typing import Any

from vla_benchmarking.shared.config import (
    TASK_PROMPT_OVERRIDE,
    TASK_REMOVE_CONFIG,
    TASK_SWAP_CONFIG,
)


def randomization_config_payload() -> dict[str, Any]:
    """Return the canonical JSON-ready randomization configuration."""
    return {
        "remove": {
            str(task_id): list(values)
            for task_id, values in sorted(TASK_REMOVE_CONFIG.items())
        },
        "layout": {
            str(task_id): [list(operation) for operation in values]
            for task_id, values in sorted(TASK_SWAP_CONFIG.items())
        },
        "prompt": {
            str(task_id): value
            for task_id, value in sorted(TASK_PROMPT_OVERRIDE.items())
        },
    }


__all__ = ["randomization_config_payload"]
