"""Contract checks for the frozen-shard, evaluation-free table exporter."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_libero_agentview.py"
spec = importlib.util.spec_from_file_location("export_libero_agentview", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path: Path, *, missing_frame=False):
    root = tmp_path / "shards"
    artifact = root / "full_0" / "SamGraph" / "artifacts" / "native_0"
    artifact.mkdir(parents=True)
    tasks = [f"task_{i}" for i in range(10)]
    manifest = {
        "episode_selection": "0", "camera": "agentview", "mode": "automatic",
        "frame_stride": 5, "source_tree": {"sha256": "a" * 64},
        "names_config_sha256": "b" * 64,
        "geometry_rules_sha256": "c" * 64,
        "input_archive_sha256": {f"{task}/demo_0.zip": "d" * 64 for task in tasks},
    }
    (artifact / "predictions.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (artifact / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for task in tasks:
            for frame in (0,) if missing_frame and task == tasks[0] else (0, 5):
                stream.write(json.dumps({
                    "task": task, "demo": "demo_0", "frame": frame, "error": None,
                    "triplets": [["black_bowl_track_1", "is_left_of", "plate_1"]],
                    "object_states": [], "source_sha256": "e" * 64,
                }) + "\n")
    return root


def test_table_export_writes_ten_vlm_csvs_without_masks_or_labels(tmp_path):
    root = source(tmp_path)
    output = tmp_path / "output"
    result = module.export_batch(root, output, episodes=range(0, 1))
    assert result["frames"] == 20
    assert result["evaluation_performed"] is False
    assert result["masks_included"] is False
    assert len(list((output / "agentview" / "csv").glob("*.csv"))) == 10
    assert len(list((output / "agentview" / "graphs").rglob("*.json"))) == 20
    with (output / "agentview" / "csv" / "task_0_agentview_v1.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [int(row["frame"]) for row in records] == [0, 5]
    assert all(row["objectA"] == "akita_black_bowl_1" for row in records)
    graph = json.loads((output / "agentview" / "graphs" / "task_0" /
                        "demo_0" / "000000.json").read_text())
    assert graph["mask_geometry_included"] is False
    assert graph["instances"] == []


def test_table_export_rejects_missing_sampled_frame(tmp_path):
    root = source(tmp_path, missing_frame=True)
    # A single frame can be a complete short video; make a real interior gap.
    prediction, _ = module.episode_paths(root, 0)
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    for row in rows:
        if row["task"] == "task_0" and row["frame"] == 0:
            row["frame"] = 10
    prediction.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="missing sampled frames"):
        module.export_batch(root, tmp_path / "output", episodes=range(0, 1))
    assert not (tmp_path / "output").exists()
