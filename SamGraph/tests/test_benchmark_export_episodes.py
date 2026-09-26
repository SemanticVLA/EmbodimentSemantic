import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from samgraph_benchmark.export import export_predictions, BENCHMARK_LABELS
from samgraph_benchmark.frames import parse_episode_selection


def test_export_keeps_same_frame_number_in_distinct_episodes(tmp_path):
    frames = tmp_path / "frames"
    (frames / "task").mkdir(parents=True)
    rows = []
    for demo in ("demo_0", "demo_1"):
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(image, format="PNG")
        rgb = image.getvalue()
        with zipfile.ZipFile(frames / "task" / (demo + ".zip"), "w") as archive:
            archive.writestr("000000.png", rgb)
        masks = tmp_path / (demo + ".npz")
        np.savez_compressed(masks, bowl=np.ones((8, 8), dtype=bool))
        rows.append(dict(task="task", demo=demo, frame=0, triplets=[],
                         source_sha256=hashlib.sha256(rgb).hexdigest(),
                         effective_masks_cache=str(masks),
                         effective_masks_cache_sha256=hashlib.sha256(masks.read_bytes()).hexdigest()))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(map(json.dumps, rows)))
    out = tmp_path / "out"
    report = export_predictions(predictions, frames, out, episodes=(0, 1), allow_partial=True)
    assert report["tasks"] == {"task": 2}
    assert (out / "agentview/graphs/task/000000.json").exists()
    assert (out / "agentview/graphs/task/demo_1/000000.json").exists()
    records = [json.loads(line) for line in
               (out / "agentview/json/task_agentview_v1.jsonl").read_text().splitlines()]
    assert [r["demo"] for r in records] == ["demo_0", "demo_1"]
    with pytest.raises(ValueError, match="unexpected sampled frames"):
        export_predictions(predictions, frames, tmp_path / "wrong", episodes=(0,), allow_partial=True)


def test_episode_selection_contract():
    assert parse_episode_selection("0:50") == tuple(range(50))
    assert parse_episode_selection("all") is None
    assert parse_episode_selection("0,2") == (0, 2)


def test_export_vocabulary_matches_evaluation_without_resolving_tracks():
    from samgraph_benchmark.ground_truth import LABEL_ALIASES
    assert BENCHMARK_LABELS == LABEL_ALIASES
    assert "black_bowl_track_1" not in BENCHMARK_LABELS


def test_provisional_aliases_are_export_only_and_reject_collisions():
    from samgraph_benchmark.export import benchmark_triplets
    source = [["black_bowl_track_2", "is_left_of", "plate_1"]]
    assert benchmark_triplets(source) == [("akita_black_bowl_2", "is_left_of", "plate_1")]
    assert source[0][0] == "black_bowl_track_2"
    with pytest.raises(ValueError, match="collide"):
        benchmark_triplets(source + [["black_bowl_2", "is_right_of", "plate_1"]])
