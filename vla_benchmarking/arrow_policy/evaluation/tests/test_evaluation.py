from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vla_benchmarking.arrow_policy.evaluation.reference_protocol import load_reference_protocol
from vla_benchmarking.arrow_policy.evaluation.runner import ArrowStudentEpisodeRunner, extract_state


def _manifest_pair(root: Path) -> None:
    protocol = {"suite_mode": "sealed_randomized", "camera": "agentview", "task_ids": list(range(10)), "episodes_per_task": 10, "seed_base": 1000, "resolution": 256}
    cells = []
    terminal = []
    for task in range(10):
        for episode in range(10):
            seed = 1000 + episode
            cell = {"cell_index": task * 10 + episode, "task_id": task, "episode_index": episode, "seed": seed, "init_state_index": seed, "resolution": 256, "suite_mode": "sealed_randomized"}
            cells.append(cell)
            terminal.append({**cell, "status": "completed", "init_state_diagnostics": {"selected_index": seed}, "audit": {"environment_audit": {"suite_mode": "sealed_randomized", "requested_removals": [], "applied_removals": [], "requested_swaps": [], "applied_swaps": [], "randomization_dimensions": {}, "state_hash_sha256_pre_settle": f"pre-{task}-{episode}", "state_hash_sha256": f"post-{task}-{episode}"}}})
    root.mkdir(parents=True, exist_ok=True)
    (root / "arrow_pick_place_matrix_manifest.json").write_text(json.dumps({"schema_version": "test", "contract_hash": "test", "protocol": protocol, "cells": cells}), encoding="utf-8")
    (root / "arrow_pick_place_matrix_manifest.jsonl").write_text("\n".join(json.dumps(item) for item in terminal) + "\n", encoding="utf-8")


def test_reference_protocol_binds_all_cells(tmp_path: Path) -> None:
    _manifest_pair(tmp_path)
    protocol = load_reference_protocol(tmp_path, require_expected_hash=False)
    assert len(protocol.cells) == 100
    assert protocol.for_cell(6, 0, 1000)["init_state_index"] == 1000


def test_extract_state_is_eight_dimensional() -> None:
    class Env:
        def _get_observations(self, force_update=True):
            return {"robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [1, 0, 0, 0], "robot0_gripper_qpos": [0.1, 0.2]}

    result = extract_state(Env())
    assert result.shape == (8,)
    assert np.isfinite(result).all()


def test_episode_runner_emits_no_language_and_actions(monkeypatch, tmp_path: Path) -> None:
    class Env:
        def __init__(self):
            self.steps = 0
            self._arrow_environment_audit = {"suite_mode": "sealed_randomized", "requested_removals": [], "applied_removals": [], "requested_swaps": [], "applied_swaps": [], "randomization_dimensions": {}, "state_hash_sha256_pre_settle": "pre", "state_hash_sha256": "post"}
            self._arrow_init_state_diagnostics = {"selected_index": 0}

        def _get_observations(self, force_update=True):
            return {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8), "robot0_eef_pos": [0, 0, 0], "robot0_eef_quat": [1, 0, 0, 0], "robot0_gripper_qpos": [0, 0]}

        def step(self, action):
            assert action.shape == (7,)
            self.steps += 1
            return self._get_observations(), 0.0, self.steps >= 1, {}

    class Reference:
        def validate_environment(self, env, **kwargs):
            return {"status": "matched"}

    class Runtime:
        device = torch.device("cpu")
        action_dim = 7
        n_action_steps = 1

        def _chunk(self, image, state, generator=None):
            return np.zeros((2, 7), dtype=np.float32)

    monkeypatch.setattr("vla_benchmarking.run_arrow_pick_place_matrix._default_arrow_inputs", lambda env, task_id, resolution: {"bboxes": {"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [40, 40, 60, 60]}, "subject": "akita_black_bowl_1", "goal_object": "plate_1"})
    monkeypatch.setattr("vla_benchmarking.run_arrow_pick_place_eval.render_exactly_one_arrow", lambda clean, bboxes, **kwargs: (clean.copy() + np.uint8(1), {"relation_count": 1}))
    runner = ArrowStudentEpisodeRunner(Runtime(), Reference())
    result = runner(env=Env(), task_id=0, episode_index=0, seed=1000, output_dir=tmp_path, bboxes={})
    assert result["steps"] == 1
    assert result["language_consumed"] is False
    assert result["tokenizer_accessed"] is False
