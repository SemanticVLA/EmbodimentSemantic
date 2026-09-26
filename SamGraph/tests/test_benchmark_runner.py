import io
import zipfile

import numpy as np
from PIL import Image

from samgraph_benchmark.artifacts import ArtifactStore
from samgraph_benchmark.cli import _prediction_frame_stride
from samgraph_benchmark.runner import PredictionRunner, read_predictions_jsonl
from samgraph_benchmark.predictor import SamGraphSamPredictor


def test_two_episodes_keep_distinct_masks_and_reset_predictor(tmp_path):
    task = tmp_path / "frames" / "task"
    task.mkdir(parents=True)
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    for demo in (0, 1):
        with zipfile.ZipFile(task / f"demo_{demo}.zip", "w") as archive:
            archive.writestr("000000.png", stream.getvalue())
    class Predictor:
        def __init__(self):
            self.starts = []
            self.ends = 0
        def on_episode_start(self, task, demo):
            self.starts.append((task, demo))
        def on_archive_end(self):
            self.ends += 1
        def __call__(self, *_):
            return {"triplets": [], "masks": {"plate_1": np.eye(4, dtype=bool)}}
    predictor = Predictor()
    runner = PredictionRunner(predictor, mask_cache_dir=tmp_path / "masks", episodes=(0, 1), no_clobber=True)
    rows = runner.run_root(tmp_path / "frames")
    assert [r["demo"] for r in rows] == ["demo_0", "demo_1"]
    assert rows[0]["mask_cache"] != rows[1]["mask_cache"]
    assert predictor.starts == [("task", "demo_0"), ("task", "demo_1")]
    assert predictor.ends == 2


def test_missing_requested_episode_is_not_silently_skipped(tmp_path):
    import pytest
    from samgraph_benchmark.frames import discover_archives
    task = tmp_path / "task"
    task.mkdir()
    (task / "demo_0.zip").touch()
    with pytest.raises(FileNotFoundError, match="missing requested episodes"):
        discover_archives(tmp_path, episodes=(0, 1))


def test_runner_preserves_failed_frame_as_empty_prediction(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())

    def fail(_rgb, _task, _frame):
        raise RuntimeError("GPU unavailable")

    rows = PredictionRunner(fail).run_archive(archive)
    assert len(rows) == 1
    assert rows[0]["triplets"] == []
    assert "GPU unavailable" in rows[0]["error"]


def test_runner_streams_mask_cache_as_compressed_frame_artifact(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())

    cache = tmp_path / "artifacts" / "masks"
    runner = PredictionRunner(lambda *_: {"triplets": [], "masks": {"plate_1": np.eye(4, dtype=bool)}}, mask_cache_dir=cache)
    rows = runner.run_archive(archive)
    assert (cache / "task" / "000000.npz").is_file()
    with np.load(cache / "task" / "000000.npz") as loaded:
        assert loaded["plate_1"].shape == (4, 4)


def test_runner_persists_partial_inventory_and_tracking_diagnostics(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())

    output = {
        "triplets": [],
        "inventory": {
            "allow_partial_inventory": True,
            "coverage": {"class_fraction": 0.5, "instance_fraction": 0.4},
            "completeness": {"plate": {"observed_count": 1, "expected_count": 1}},
            "missing_class_ids": ["cookies"],
        },
        "tracking_pair": {
            "requested": ["black_bowl_1", "plate_1"],
            "effective": ["plate_1", "wooden_cabinet_1"],
            "fallback": True,
            "diagnostic_only": True,
        },
    }
    row = PredictionRunner(lambda *_: output).run_archive(archive)[0]
    assert row["allow_partial_inventory"] is True
    assert row["inventory_coverage"]["observed_instance_count"] == 0
    assert row["inventory_coverage"]["class_fraction"] == 0.0
    assert row["missing_class_ids"] == ["plate"]
    assert row["initial_inventory_coverage"]["class_fraction"] == 0.5
    assert row["initial_missing_class_ids"] == ["cookies"]
    assert row["tracking_pair"]["requested"] == ["black_bowl_1", "plate_1"]
    assert row["tracking_pair"]["effective"] == ["plate_1", "wooden_cabinet_1"]


def test_runner_inventory_uses_finalized_masks_not_complete_initial_inventory(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())

    classes = [
        "black_bowl", "plate", "cookies", "flat_stove",
        "wooden_cabinet", "white_ramekin", "drawer",
    ]
    output = {
        "triplets": [],
        "masks": {
            f"{class_id}_1": np.eye(4, dtype=bool)
            for class_id in classes
            if class_id != "cookies"
        },
        "inventory": {
            "coverage": {
                "expected_class_count": 7,
                "observed_class_count": 7,
                "class_fraction": 1.0,
                "expected_instance_count": 7,
                "observed_instance_count": 7,
                "instance_fraction": 1.0,
            },
            "completeness": {
                class_id: {"expected_count": 1, "observed_count": 1, "complete": True, "missing": False}
                for class_id in classes
            },
            "missing_class_ids": [],
        },
    }
    row = PredictionRunner(lambda *_: output, mask_cache_dir=tmp_path / "masks").run_archive(archive)[0]

    coverage = row["inventory_coverage"]
    assert coverage["expected_class_count"] == 7
    assert coverage["observed_class_count"] == 6
    assert coverage["observed_instance_count"] == 6
    assert coverage["class_fraction"] == 6 / 7
    assert coverage["instance_fraction"] == 6 / 7
    assert row["missing_class_ids"] == ["cookies"]
    assert row["incomplete_class_ids"] == ["cookies"]
    assert row["missing_instance_ids"] == ["cookies_1"]
    assert row["initial_inventory_coverage"]["class_fraction"] == 1.0


def test_runner_unknown_inventory_does_not_fabricate_expected_counts(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())

    output = {"triplets": [], "masks": {"object_1": np.eye(4, dtype=bool)}}
    row = PredictionRunner(lambda *_: output).run_archive(archive)[0]
    assert row["inventory_coverage"]["observed_instance_count"] == 1
    assert row["inventory_coverage"]["expected_instance_count"] is None
    assert row["inventory_coverage"]["instance_fraction"] is None
    assert row["missing_class_ids"] is None


def test_artifacts_cannot_escape_store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    try:
        store.path("..\\outside.json")
    except ValueError:
        pass
    else:
        raise AssertionError("artifact path traversal was accepted")


def test_runner_calls_archive_lifecycle_hooks(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", stream.getvalue())
        z.writestr("000001.png", stream.getvalue())

    class Predictor:
        def __init__(self): self.events = []
        def on_archive_start(self, task): self.events.append(("start", task))
        def on_archive_end(self): self.events.append(("end",))
        def __call__(self, _rgb, _task, frame): self.events.append(("frame", frame)); return []

    predictor = Predictor()
    PredictionRunner(predictor).run_archive(archive)
    assert predictor.events == [("start", "task"), ("frame", 0), ("frame", 1), ("end",)]


def test_runner_processes_zero_based_stride_without_renumbering_frames(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    stream = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(stream, format="PNG")
    with zipfile.ZipFile(archive, "w") as z:
        for index in range(11):
            z.writestr(f"{index:06d}.png", stream.getvalue())

    seen = []
    rows = PredictionRunner(
        lambda _rgb, _task, frame: seen.append(frame) or [], frame_stride=5
    ).run_archive(archive)
    assert seen == [0, 5, 10]
    assert [row["frame"] for row in rows] == [0, 5, 10]
    assert all(row["frame_stride"] == 5 for row in rows)


def test_runner_rejects_nonpositive_stride():
    try:
        PredictionRunner(lambda *_: [], frame_stride=0)
    except ValueError as error:
        assert "positive integer" in str(error)
    else:
        raise AssertionError("zero frame stride was accepted")


def test_jsonl_reader_unions_duplicate_frame_records_like_vlm_evaluator(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"task":"task","demo":"demo_0","frame":0,"triplets":[["a","is_left_of","b"]]}\n'
        '{"task":"task","demo":"demo_0","frame":0,"triplets":[["c","is_right_of","d"]]}\n',
        encoding="utf-8",
    )
    assert set(map(tuple, read_predictions_jsonl(path)[("task", "demo_0", 0)])) == {
        ("a", "is_left_of", "b"), ("c", "is_right_of", "d")
    }


def test_score_stride_is_inferred_and_mixed_prediction_cadence_is_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"task":"task","demo":"demo_0","frame":0,"frame_stride":5}\n'
        '{"task":"task","demo":"demo_0","frame":5,"frame_stride":5}\n',
        encoding="utf-8",
    )
    assert _prediction_frame_stride(path) == 5
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"task":"task","demo":"demo_0","frame":6,"frame_stride":1}\n')
    try:
        _prediction_frame_stride(path)
    except ValueError as error:
        assert "mixed frame strides" in str(error)
    else:
        raise AssertionError("mixed frame-stride prediction artifact was accepted")


def test_sam_predictor_uses_resolver_graph_without_gt():
    class Runtime:
        model_identity = {"source_revision": "test"}
        def close(self): pass
    class Resolver:
        def __init__(self): self.calls = []; self.closed = 0
        def close(self): self.closed += 1
        def resolve(self, **kwargs):
            self.calls.append(kwargs)
            return {"graph": {"triplets": [
                {"subject": "black_bowl_1", "relation": "is_left_of", "object": "plate_1",
                 "current_geometry_valid": True},
                {"subject": "plate_1", "relation": "unknown", "object": "cookies_1",
                 "current_geometry_valid": False},
            ], "instances": []}, "provenance": {}}
    resolver = Resolver()
    predictor = SamGraphSamPredictor("unused", runtime=Runtime(), tracker=object(), segmenter=object(), resolver=resolver)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    value = predictor(image, "pick_up_the_black_bowl_on_the_plate", 0)
    assert value["triplets"] == [["black_bowl_1", "is_left_of", "plate_1"]]
    call = resolver.calls[0]
    assert call["selected_pair"] == ("black_bowl_1", "plate_1")
    assert call["state_sequence"] == 0
    assert call["rgb_sha256"]
