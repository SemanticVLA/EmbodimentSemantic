import pytest

from samgraph_benchmark.cli import _scope_task_prompts


def test_full_names_config_can_be_used_for_scoped_task_input():
    names = {
        "target_task": {"black_bowl": ["black bowl"]},
        "other_task": {"plate": ["plate"]},
    }

    assert _scope_task_prompts(names, {"target_task"}) == {
        "target_task": names["target_task"]
    }


def test_names_config_still_rejects_missing_input_task():
    with pytest.raises(ValueError, match=r"missing input tasks: \['target_task'\]"):
        _scope_task_prompts({"other_task": {"plate": ["plate"]}}, {"target_task"})
