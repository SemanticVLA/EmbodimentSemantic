import json
import hashlib
from pathlib import Path
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from samgraph_benchmark.artifacts import ArtifactStore
from samgraph_benchmark import cli
from samgraph_benchmark.cli import (
    _artifact_output,
    _input_archive_hashes,
    _input_archive_label,
    _mask_cache_for_output,
    _manifest_for_output,
    _source_tree_identity,
)


def test_nested_prediction_output_gets_adjacent_cache_inside_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    output = _artifact_output(store, Path("artifacts/subdir/run.jsonl"), "predictions.jsonl")
    cache = _mask_cache_for_output(store, output)
    assert output == store.root / "subdir" / "run.jsonl"
    assert cache == store.root / "subdir" / "run_masks"
    assert store.root in cache.parents


def test_input_manifest_hashes_are_relative_and_anonymous(tmp_path):
    root = tmp_path / "agentview"
    archive = root / "task_a" / "demo_0.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"fixture")
    hashes = _input_archive_hashes(root, [archive])
    assert list(hashes) == ["task_a/demo_0.zip"]
    assert all(not Path(key).is_absolute() for key in hashes)
    assert _input_archive_label(root, archive) == "task_a/demo_0.zip"


def test_input_manifest_keeps_lexical_label_for_in_root_symlink_archive(tmp_path):
    root = tmp_path / "agentview"
    target = tmp_path / "outside" / "demo_target.zip"
    archive = root / "task_a" / "demo_0.zip"
    target.parent.mkdir(parents=True)
    archive.parent.mkdir(parents=True)
    payload = b"symlink-target-bytes"
    target.write_bytes(payload)
    try:
        archive.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    hashes = _input_archive_hashes(root, [archive])
    assert list(hashes) == ["task_a/demo_0.zip"]
    assert hashes["task_a/demo_0.zip"] == hashlib.sha256(payload).hexdigest()
    assert _input_archive_label(root, archive) == "task_a/demo_0.zip"


def test_input_manifest_rejects_lexical_archive_outside_frames_root(tmp_path):
    root = tmp_path / "agentview"
    archive = tmp_path / "outside" / "demo_0.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes frames root"):
        _input_archive_label(root, archive)
    with pytest.raises(ValueError, match="escapes frames root"):
        _input_archive_hashes(root, [archive])


def test_source_identity_is_deterministic_and_portable():
    identity = _source_tree_identity()
    assert len(identity["sha256"]) == 64
    assert identity["files"]
    assert all(not Path(label).is_absolute() for label in identity["files"])
    assert all("samgraph_benchmark/" in label or "samgraph_core/" in label for label in identity["files"])
    assert identity == _source_tree_identity()


def test_distinct_prediction_outputs_get_distinct_manifests(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    manifests = []
    for name, digest in (("run_a.jsonl", "a" * 64), ("run_b.jsonl", "b" * 64)):
        output = _artifact_output(store, Path("artifacts/runs") / name, "predictions.jsonl")
        manifest = _manifest_for_output(store, output)
        store.write_json(manifest.relative_to(store.root), {
            "predictions": str(output.relative_to(store.root)),
            "source_tree": {"sha256": digest, "files": ["samgraph_benchmark/cli.py"]},
        })
        manifests.append(manifest)
    assert manifests[0] != manifests[1]
    assert (manifests[0].read_text(encoding="utf-8") != manifests[1].read_text(encoding="utf-8"))
    assert "run_a" in manifests[0].read_text(encoding="utf-8")
    assert "run_b" in manifests[1].read_text(encoding="utf-8")


def _frame_root(tmp_path: Path) -> Path:
    root = tmp_path / "frames" / "task_a"
    root.mkdir(parents=True)
    image = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(image, format="PNG")
    with zipfile.ZipFile(root / "demo_0.zip", "w") as archive:
        archive.writestr("000000.png", image.getvalue())
    return root.parent


def test_main_predict_writes_distinct_adjacent_manifests(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    frames_root = _frame_root(tmp_path)
    monkeypatch.setattr(cli, "ArtifactStore", lambda: ArtifactStore(artifact_root))
    instances = iter((1, 2))

    class FakePredictor:
        def __init__(self, _checkpoint, *, geometry_rules=None):
            self.provenance = {"fake_predictor_instance": next(instances), "geometry_rules": geometry_rules}
        def warmup(self): pass
        def close(self): pass

    class FakeRunner:
        def __init__(self, _predictor, **_kwargs): pass
        def run_root_to_jsonl(self, _frames_root, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{\"task\": \"task_a\", \"demo\": \"demo_0\", \"frame\": 0, \"triplets\": []}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "SamGraphSamPredictor", FakePredictor)
    monkeypatch.setattr("samgraph_benchmark.runner.PredictionRunner", FakeRunner)
    monkeypatch.setattr(cli, "_source_tree_identity", lambda: {"sha256": "a" * 64, "files": ["samgraph_benchmark/cli.py"]})

    cli.main(["predict", "--frames-root", str(frames_root), "--checkpoint", "fake.pt", "--output", "artifacts/run_a.jsonl"])
    cli.main(["predict", "--frames-root", str(frames_root), "--checkpoint", "fake.pt", "--output", "artifacts/run_b.jsonl"])

    first = artifact_root / "run_a.manifest.json"
    second = artifact_root / "run_b.manifest.json"
    assert first.is_file() and second.is_file()
    first_value, second_value = json.loads(first.read_text()), json.loads(second.read_text())
    assert first_value["predictions"] == "run_a.jsonl"
    assert second_value["predictions"] == "run_b.jsonl"
    assert first_value["predictor"]["fake_predictor_instance"] == 1
    assert second_value["predictor"]["fake_predictor_instance"] == 2
    assert not (artifact_root / "prediction_manifest.json").exists()


def test_main_predict_refuses_manifest_when_source_changes(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    frames_root = _frame_root(tmp_path)
    monkeypatch.setattr(cli, "ArtifactStore", lambda: ArtifactStore(artifact_root))

    class FakePredictor:
        provenance = {"fake": True}
        def __init__(self, _checkpoint, *, geometry_rules=None): pass
        def warmup(self): pass
        def close(self): pass

    class FakeRunner:
        def __init__(self, _predictor, **_kwargs): pass
        def run_root_to_jsonl(self, _frames_root, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli, "SamGraphSamPredictor", FakePredictor)
    monkeypatch.setattr("samgraph_benchmark.runner.PredictionRunner", FakeRunner)
    identities = iter((
        {"sha256": "a" * 64, "files": ["samgraph_benchmark/cli.py"]},
        {"sha256": "b" * 64, "files": ["samgraph_benchmark/cli.py"]},
    ))
    monkeypatch.setattr(cli, "_source_tree_identity", lambda: next(identities))

    with pytest.raises(RuntimeError, match="source tree changed"):
        cli.main(["predict", "--frames-root", str(frames_root), "--checkpoint", "fake.pt", "--output", "artifacts/mutated.jsonl"])
    assert not (artifact_root / "mutated.manifest.json").exists()
