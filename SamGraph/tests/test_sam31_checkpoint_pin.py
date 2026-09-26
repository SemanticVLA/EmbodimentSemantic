import hashlib
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SAMGRAPH_SRC = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
if SAMGRAPH_SRC not in sys.path:
    sys.path.insert(0, SAMGRAPH_SRC)

import samgraph_core.local_sam31 as local_sam31


def _point_runtime(tmp_path, monkeypatch, *, fail_prompt=False):
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"qualified fixture")
    monkeypatch.setattr(
        local_sam31,
        "SAM31_CHECKPOINT_SHA256",
        hashlib.sha256(b"qualified fixture").hexdigest(),
    )

    class Predictor:
        def __init__(self):
            self.requests = []

        def handle_request(self, request):
            self.requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "point-session"}
            if request["type"] == "add_prompt":
                if fail_prompt:
                    raise RuntimeError("synthetic prompt failure")
                return {
                    "outputs": {
                        "out_binary_masks": np.array([[[1, 0], [1, 1]]], dtype=bool),
                        "out_obj_ids": np.array([1], dtype=np.int64),
                        "out_probs": np.array([0.75], dtype=np.float32),
                    }
                }
            if request["type"] == "close_session":
                return {}
            raise AssertionError(request)

    predictor = Predictor()
    runtime = local_sam31.OfficialSam31Runtime(
        checkpoint,
        predictor_factory=lambda **kwargs: predictor,
    )
    encoded = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(encoded, format="PNG")
    return runtime, predictor, encoded.getvalue()


def test_runtime_rejects_checkpoint_with_wrong_hash_before_loading(tmp_path):
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"not the qualified checkpoint")
    factory_calls = []
    runtime = local_sam31.OfficialSam31Runtime(
        checkpoint, predictor_factory=lambda **kwargs: factory_calls.append(kwargs)
    )

    with pytest.raises(RuntimeError, match="SHA-256 does not match"):
        runtime.warmup()

    assert factory_calls == []


def test_runtime_records_checkpoint_digest_without_rehashing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"qualified fixture")
    expected = hashlib.sha256(b"qualified fixture").hexdigest()
    monkeypatch.setattr(local_sam31, "SAM31_CHECKPOINT_SHA256", expected)
    runtime = local_sam31.OfficialSam31Runtime(checkpoint, predictor_factory=lambda **kwargs: object())

    runtime.warmup()
    first = runtime.model_identity["checkpoint_sha256"]
    checkpoint.write_bytes(b"mutated after warmup")
    second = runtime.model_identity["checkpoint_sha256"]

    assert first == second == expected


def test_runtime_rechecks_checkpoint_after_earlier_provenance_read(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"qualified fixture")
    expected = hashlib.sha256(b"qualified fixture").hexdigest()
    monkeypatch.setattr(local_sam31, "SAM31_CHECKPOINT_SHA256", expected)
    runtime = local_sam31.OfficialSam31Runtime(checkpoint, predictor_factory=lambda **kwargs: object())
    assert runtime.model_identity["checkpoint_sha256"] == expected
    checkpoint.write_bytes(b"changed before model load")

    with pytest.raises(RuntimeError, match="SHA-256 does not match"):
        runtime.warmup()


def test_injected_predictor_factory_is_explicitly_unverified(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"qualified fixture")
    monkeypatch.setattr(
        local_sam31,
        "SAM31_CHECKPOINT_SHA256",
        hashlib.sha256(b"qualified fixture").hexdigest(),
    )
    runtime = local_sam31.OfficialSam31Runtime(
        checkpoint, predictor_factory=lambda **kwargs: object()
    )

    runtime.warmup()
    identity = runtime.model_identity
    assert identity["source_revision"] is None
    assert identity["source_verification"] == "unverified_injected_predictor_factory"
    assert identity["predictor_factory_injected"] is True


def test_source_verification_rejects_wrong_git_commit(tmp_path, monkeypatch):
    package_root = tmp_path / "sam3"
    package_root.mkdir()

    def fake_git(*arguments, cwd):
        if arguments[-1] == "--show-toplevel":
            return str(tmp_path)
        if arguments[-1] == "HEAD":
            return "wrong-commit"
        raise AssertionError(arguments)

    monkeypatch.setattr(local_sam31, "_git_output", fake_git)
    with pytest.raises(RuntimeError, match="does not match the qualified pin"):
        local_sam31._sam31_git_identity(package_root)


def test_source_verification_rejects_dirty_package(tmp_path, monkeypatch):
    package_root = tmp_path / "sam3"
    package_root.mkdir()

    def fake_git(*arguments, cwd):
        if arguments[-1] == "--show-toplevel":
            return str(tmp_path)
        if arguments[-1] == "HEAD":
            return local_sam31.SAM31_SOURCE_REVISION
        if "status" in arguments:
            return " M sam3/model_builder.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(local_sam31, "_git_output", fake_git)
    with pytest.raises(RuntimeError, match="source package is dirty"):
        local_sam31._sam31_git_identity(package_root)


def test_source_verification_rejects_model_builder_outside_package(tmp_path, monkeypatch):
    package_root = tmp_path / "sam3"
    package_root.mkdir()
    outside_builder = tmp_path / "other" / "model_builder.py"
    outside_builder.parent.mkdir()
    outside_builder.touch()
    package_spec = SimpleNamespace(submodule_search_locations=[str(package_root)])
    builder_spec = SimpleNamespace(origin=str(outside_builder))

    monkeypatch.setattr(
        local_sam31.importlib.util,
        "find_spec",
        lambda name: package_spec if name == "sam3" else builder_spec,
    )
    with pytest.raises(RuntimeError, match="outside the single sam3 package root"):
        local_sam31._sam31_package_spec()


def test_source_verification_rejects_mixed_loaded_module_origins(tmp_path, monkeypatch):
    package_root = tmp_path / "sam3"
    package_root.mkdir()
    outside_module = tmp_path / "other" / "tracking.py"
    outside_module.parent.mkdir()
    outside_module.touch()
    monkeypatch.setitem(sys.modules, "sam3.mixed_origin", SimpleNamespace(__file__=str(outside_module)))

    with pytest.raises(RuntimeError, match="mixed origins"):
        local_sam31._verify_loaded_sam31_modules(package_root)


def test_source_verification_records_pinned_clean_tree(tmp_path, monkeypatch):
    package_root = tmp_path / "sam3"
    package_root.mkdir()
    builder = package_root / "model_builder.py"
    builder.touch()
    package_spec = SimpleNamespace(submodule_search_locations=[str(package_root)])
    builder_spec = SimpleNamespace(origin=str(builder))
    identity = {
        "source_revision": local_sam31.SAM31_SOURCE_REVISION,
        "source_path": "sam3",
        "source_git_root": "git_worktree",
        "source_git_clean": True,
        "source_verification": "verified_git_checkout",
    }
    monkeypatch.setattr(
        local_sam31.importlib.util,
        "find_spec",
        lambda name: package_spec if name == "sam3" else builder_spec,
    )
    monkeypatch.setattr(local_sam31, "_sam31_git_identity", lambda path: dict(identity))

    result = local_sam31._verify_sam31_source()
    assert result["source_revision"] == local_sam31.SAM31_SOURCE_REVISION
    assert result["source_path"] == "sam3"
    assert result["source_git_root"] == "git_worktree"
    assert result["source_git_clean"] is True
    assert result["source_model_builder_path"] == "sam3/model_builder.py"


def test_segment_point_uses_exact_positive_prompt_and_closes_session(tmp_path, monkeypatch):
    runtime, predictor, encoded = _point_runtime(tmp_path, monkeypatch)
    result = runtime.segment_point(point_xy_rel=(0.25, 0.75), encoded_rgb=encoded)

    assert len(result) == 1
    mask, score, object_id = result[0]
    assert mask.shape == (2, 2)
    assert score == pytest.approx(0.75)
    assert object_id == 1
    assert predictor.requests[0]["type"] == "start_session"
    prompt = predictor.requests[1]
    assert prompt == {
        "type": "add_prompt",
        "session_id": "point-session",
        "frame_index": 0,
        "obj_id": 1,
        "points": [[0.25, 0.75]],
        "point_labels": [1],
        "clear_old_points": True,
        "rel_coordinates": True,
        "output_prob_thresh": 0.0,
    }
    assert predictor.requests[-1] == {
        "type": "close_session",
        "session_id": "point-session",
        "run_gc_collect": False,
    }


def test_segment_point_closes_session_when_prompt_fails(tmp_path, monkeypatch):
    runtime, predictor, encoded = _point_runtime(tmp_path, monkeypatch, fail_prompt=True)
    with pytest.raises(RuntimeError, match="synthetic prompt failure"):
        runtime.segment_point(point_xy_rel=(0.1, 0.2), encoded_rgb=encoded)
    assert predictor.requests[-1]["type"] == "close_session"


@pytest.mark.parametrize("point", [(-0.01, 0.5), (0.5, 1.0), (float("nan"), 0.2), (0.2, float("inf")), (0.5,)])
def test_segment_point_rejects_bad_coordinates(tmp_path, monkeypatch, point):
    runtime, predictor, encoded = _point_runtime(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="point_xy_rel"):
        runtime.segment_point(point_xy_rel=point, encoded_rgb=encoded)
    assert predictor.requests == []


def test_local_segmenter_forwards_point_prompt():
    class Runtime:
        model_identity = {}

        def segment_point(self, **kwargs):
            return [("mask", 1.0, 1)]

    segmenter = local_sam31.LocalSam31Segmenter(Runtime())
    assert segmenter.segment_point(point_xy_rel=(0.1, 0.2), encoded_rgb=b"rgb") == [("mask", 1.0, 1)]
