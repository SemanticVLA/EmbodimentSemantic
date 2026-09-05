from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.evaluation.reference_protocol import ReferenceProtocol, load_reference_protocol
from vla_benchmarking.arrow_finetuned_vla.language_free_arrow_student.evaluation.runner import ArrowStudentEpisodeRunner, extract_state


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


def test_task6_archived_missing_audit_uses_schedule_only(tmp_path: Path) -> None:
    _manifest_pair(tmp_path)
    loaded = load_reference_protocol(tmp_path, require_expected_hash=False)
    cells = [dict(cell) for cell in loaded.cells]
    target = next(cell for cell in cells if cell["task_id"] == 6 and cell["episode_index"] == 0)
    target["state_hash_evidence"] = "unavailable_due_archived_input_failure"
    target["requested_removals"] = []
    target["applied_removals"] = []
    target["requested_swaps"] = []
    target["applied_swaps"] = []
    loaded = ReferenceProtocol(root=loaded.root, manifest_sha256=loaded.manifest_sha256, manifest=loaded.manifest, cells=cells)

    class Env:
        _arrow_init_state_diagnostics = {"selected_index": 1000}
        _arrow_environment_audit = {
            "suite_mode": "sealed_randomized",
            "canonical_bddl_file": "/tmp/current_canonical_task.bddl",
            "prompt_provenance": "not_applicable_direct_runner",
            "requested_removals": ["ramekin_1"],
            "applied_removals": ["ramekin_1"],
            "requested_swaps": [],
            "applied_swaps": [],
            "randomization_dimensions": {"object_removal": True},
        }

    result = loaded.validate_environment(Env(), task_id=6, episode_index=0, seed=1000)
    assert result["status"] == "matched"
    assert result["checks"]["archived_state_hash_evidence"] is True


def test_archived_missing_hash_set_includes_non_task6_cells(tmp_path: Path) -> None:
    _manifest_pair(tmp_path)
    missing = {(4, 1, 1001), (4, 5, 1005), (6, 0, 1000), (9, 1, 1001), (9, 2, 1002), (9, 8, 1008)}
    terminal_path = tmp_path / "arrow_pick_place_matrix_manifest.jsonl"
    records = []
    for line in terminal_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if (record["task_id"], record["episode_index"], record["seed"]) in missing:
            del record["audit"]["environment_audit"]["state_hash_sha256_pre_settle"]
            del record["audit"]["environment_audit"]["state_hash_sha256"]
        records.append(record)
    terminal_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    loaded = load_reference_protocol(tmp_path, require_expected_hash=False)
    target = next(cell for cell in loaded.cells if cell["task_id"] == 4 and cell["episode_index"] == 1)
    assert target["state_hash_evidence"] == "unavailable_due_archived_input_failure"


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

    monkeypatch.setattr("vla_benchmarking.evaluation.run_arrow_pick_place_matrix._default_arrow_inputs", lambda env, task_id, resolution: {"bboxes": {"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [40, 40, 60, 60]}, "subject": "akita_black_bowl_1", "goal_object": "plate_1"})
    monkeypatch.setattr("vla_benchmarking.evaluation.run_arrow_pick_place_eval.render_exactly_one_arrow", lambda clean, bboxes, **kwargs: (clean.copy() + np.uint8(1), {"relation_count": 1}))
    runner = ArrowStudentEpisodeRunner(Runtime(), Reference())
    result = runner(env=Env(), task_id=0, episode_index=0, seed=1000, output_dir=tmp_path, bboxes={})
    assert result["steps"] == 1
    assert result["language_consumed"] is False
    assert result["tokenizer_accessed"] is False
