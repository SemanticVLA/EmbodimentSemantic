from __future__ import annotations

from vla_benchmarking.evaluation.randomization_contract import (
    randomization_config_payload,
)
from vla_benchmarking.shared.config import (
    TASK_PROMPT_OVERRIDE,
    TASK_REMOVE_CONFIG,
    TASK_SWAP_CONFIG,
)


def test_randomization_config_payload_is_complete_and_json_ready() -> None:
    payload = randomization_config_payload()
    assert payload == {
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


def test_randomization_config_payload_returns_fresh_containers() -> None:
    first = randomization_config_payload()
    second = randomization_config_payload()
    first["remove"].clear()
    assert second["remove"]
