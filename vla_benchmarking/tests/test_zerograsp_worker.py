from __future__ import annotations

import numpy as np
import pytest
import sys
from types import SimpleNamespace

from zerograsp_contracts import encode_array
from zerograsp_worker import (
    _decode_request,
    _install_official_ofe_single_scene_compat,
    _normalise_output,
    _set_seed,
)


def _raw_output(**overrides):
    output = {
        "grasps": np.eye(4, dtype=np.float64)[None],
        "scores": [1.0],
        "translation_reference": ["grasp_center"],
        "translation_frame": ["camera_graspnet"],
        "rotation_frame": ["camera_graspnet"],
    }
    output.update(overrides)
    return output


def _request() -> dict[str, object]:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)
    source = np.zeros((2, 3), dtype=bool)
    source[0, 0] = True
    destination = np.zeros((2, 3), dtype=bool)
    destination[1, 2] = True
    return {
        "type": "infer",
        "request_id": 0,
        "clean_rgb": encode_array(rgb),
        "depth_m": encode_array(depth),
        "source_mask": encode_array(source),
        "destination_mask": encode_array(destination),
        "K_model": encode_array(np.eye(3, dtype=np.float32)),
        "source_px_model": (0.0, 0.0),
        "destination_px_model": (2.0, 1.0),
        "request_hash": "request-hash",
        "fixed_seed": 0,
    }


def test_decode_accepts_protocol_infer_metadata():
    decoded = _decode_request(_request())
    assert decoded["clean_rgb"].shape == (2, 3, 3)


def test_decode_rejects_task_id_injection():
    request = _request()
    request["task_id"] = "task-9"
    with pytest.raises(ValueError, match="unsupported fields.*task_id"):
        _decode_request(request)


def test_decode_rejects_non_infer_request_type():
    request = _request()
    request["type"] = "handshake"
    with pytest.raises(ValueError, match="type must be 'infer'"):
        _decode_request(request)


@pytest.mark.parametrize("missing", ("translation_reference", "translation_frame", "rotation_frame"))
def test_normalise_requires_explicit_camera_graspnet_frame_tags(missing):
    raw = _raw_output(collision_free=[True])
    raw.pop(missing)
    with pytest.raises(ValueError, match=missing):
        _normalise_output(raw)


def test_normalise_emits_exact_frame_tags_and_preserves_collision_metadata():
    normalised = _normalise_output(_raw_output(collision_free=[True]))
    assert normalised["translation_reference"] == ["grasp_center"]
    assert normalised["translation_frame"] == ["camera_graspnet"]
    assert normalised["rotation_frame"] == ["camera_graspnet"]
    assert normalised["collision_free"] == [True]


@pytest.mark.parametrize("field,value", (("translation_reference", [None]), ("translation_frame", ["camera"]), ("rotation_frame", ["world"])))
def test_normalise_rejects_null_or_wrong_frame_tags(field, value):
    with pytest.raises(ValueError, match=field):
        _normalise_output(_raw_output(collision_free=[True], **{field: value}))


@pytest.mark.parametrize("source_frame", (None, "world"))
def test_normalise_requires_explicit_camera_reconstruction_frame(source_frame):
    reconstruction = {
        "dimensions_m": [0.08, 0.06, 0.04],
        "confidence": 1.0,
        "centroid_camera_m": [0.0, 0.0, 1.0],
        "bounds_camera_m": [-0.04, -0.03, 0.98, 0.04, 0.03, 1.02],
    }
    if source_frame is not None:
        reconstruction["source_frame"] = source_frame
    with pytest.raises(ValueError, match="source_frame"):
        _normalise_output(_raw_output(collision_free=[True], reconstruction=reconstruction))


def test_set_seed_resets_torch_cpu_and_cuda_rngs(monkeypatch):
    calls = []
    fake_cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: calls.append(("cuda", seed))),
        backends=SimpleNamespace(cudnn=fake_cudnn),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    status = _set_seed(73)
    assert calls == [("cpu", 73), ("cuda", 73)]
    assert status["torch"] and status["cuda"] and status["cudnn_deterministic"]
    assert fake_cudnn.deterministic is True and fake_cudnn.benchmark is False


def test_official_ofe_compat_supplies_single_scene_object_boundaries():
    calls = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class BatchIds:
        def __init__(self, values):
            self.values = values

        def numel(self):
            return len(self.values)

        def min(self):
            return Scalar(min(self.values))

        def max(self):
            return Scalar(max(self.values))

        def new_zeros(self, shape):
            return [0] * shape[0]

        def new_full(self, shape, value):
            return [value] * shape[0]

    class OFE:
        def forward(self, pts, mask, depth_map, K, batch_id, batch_start_id, batch_end_id, grid_size):
            calls.append((batch_start_id, batch_end_id))
            return "features"

    model = SimpleNamespace(ofe=OFE())
    assert _install_official_ofe_single_scene_compat(model) == "single_scene_boundaries_v1"
    result = model.ofe.forward(
        None,
        np.zeros((3, 2, 2)),
        np.zeros((3, 2, 2)),
        None,
        BatchIds([0, 1, 2]),
        10.0,
    )
    assert result == "features"
    assert calls == [([0, 0, 0], [3, 3, 3])]


def test_official_ofe_compat_rejects_unexpected_signature():
    class OFE:
        def forward(self, pts, mask):
            return None

    with pytest.raises(RuntimeError, match="unsupported official OFE"):
        _install_official_ofe_single_scene_compat(SimpleNamespace(ofe=OFE()))
