"""Contract checks for the SO101 frozen-shard release exporter."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_so101_agentview import export_release  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture_run(root: Path, task: str) -> Path:
    camera = root / "agent_view"
    prediction_rows = []
    frame_rows = []
    for frame in (0, 30, 60):
        relative = Path(task) / "episode_0" / f"{frame:06d}"
        mask = camera / "masks" / relative.with_suffix(".npz")
        observed = camera / "observed_masks" / relative.with_suffix(".npz")
        mask.parent.mkdir(parents=True, exist_ok=True)
        observed.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask, black_bowl=np.ones((2, 2), dtype=bool))
        np.savez_compressed(observed, black_bowl=np.ones((2, 2), dtype=bool))
        graph = camera / "graphs" / relative.with_suffix(".json")
        source_sha = f"{frame:064x}"
        triplets = [] if frame == 30 else [["black_bowl", "is_left_of", "black_stove"]]
        _json(graph, {
            "task": task, "demo": "episode_0", "frame": frame,
            "camera": "agent_view", "source_sha256": source_sha,
            "triplets": [
                {"subject": subject, "relation": relation, "object": object_}
                for subject, relation, object_ in triplets
            ],
        })
        row = {
            "schema": "samgraph.so101_agent_prediction.v1",
            "task": task, "demo": "episode_0", "frame": frame,
            "camera": "agent_view", "source_sha256": source_sha,
            "tracking_stride": 1, "output_stride": 30,
            "triplets": triplets, "object_states": [],
            "graph": graph.relative_to(camera).as_posix(),
            "mask_cache": mask.relative_to(camera).as_posix(),
            "mask_cache_sha256": _sha(mask),
            "observed_mask_cache": observed.relative_to(camera).as_posix(),
            "observed_mask_cache_sha256": _sha(observed),
        }
        prediction_rows.append(row)
        frame_rows.append({
            "task": task, "demo": "episode_0", "frame": frame,
            "camera": "agent_view", "source_sha256": source_sha,
            "triplet_count": len(triplets), "empty_prediction": not triplets,
        })
    camera.mkdir(parents=True, exist_ok=True)
    (camera / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    (camera / "frame_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frame_rows), encoding="utf-8"
    )
    _json(root / "run_manifest.json", {
        "status": "complete", "object_config_sha256": "config",
        "git": {"commit": "abc"}, "predictor": {"checkpoint_sha256": "checkpoint"},
    })
    return root


class Dataset:
    def __init__(self, task: str):
        self.task = task

    def episode(self, task: str, episode: int):
        assert task == self.task and episode == 0
        return SimpleNamespace(length=61)


def test_release_export_preserves_empty_frames_and_exact_csv_schema(tmp_path):
    task = "real-task"
    run = _fixture_run(tmp_path / "run", task)
    selection = tmp_path / "selection.json"
    _json(selection, {
        "schema": "samgraph.so101_release_selection.v1",
        "camera": "agent_view", "output_stride": 30,
        "episodes": [{"task": task, "episode": 0, "source": "run-a"}],
    })
    output = tmp_path / "release"
    result = export_release(
        dataset=Dataset(task), selection_path=selection,
        sources={"run-a": run}, output=output,
        include_masks=True, include_observed_masks=True,
    )

    assert result["episode_count"] == 1
    assert result["sampled_frame_count"] == 3
    assert result["evaluation_performed"] is False
    assert result["f1_reported"] is False
    rows = [json.loads(line) for line in (
        output / "agentview" / "predictions.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert [row["frame"] for row in rows] == [0, 30, 60]
    assert rows[1]["triplets"] == []
    with (output / "agentview" / "csv" / f"{task}_agent_view_v1.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "task", "demo", "frame", "camera", "objectA", "relation", "objectB",
        ]
        assert len(list(reader)) == 2
    assert (output / "agentview" / "masks" / task / "episode_0" / "000030.npz").is_file()
    assert (output / "agentview" / "graphs" / task / "episode_0" / "000030.json").is_file()


def test_release_export_rejects_incomplete_episode(tmp_path):
    task = "real-task"
    run = _fixture_run(tmp_path / "run", task)
    lines = (run / "agent_view" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "agent_view" / "predictions.jsonl").write_text(
        "\n".join(lines[:2]) + "\n", encoding="utf-8"
    )
    selection = tmp_path / "selection.json"
    _json(selection, {
        "schema": "samgraph.so101_release_selection.v1",
        "camera": "agent_view", "output_stride": 30,
        "episodes": [{"task": task, "episode": 0, "source": "run-a"}],
    })
    import pytest

    with pytest.raises(ValueError, match="incomplete selected episode"):
        export_release(
            dataset=Dataset(task), selection_path=selection,
            sources={"run-a": run}, output=tmp_path / "release",
        )
