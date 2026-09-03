from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from molmo_sam3 import (
    DEFAULT_CHECKPOINT_SHA256,
    SAM3_SOURCE_COMMIT,
    Sam3Request,
    Sam3Runtime,
    Sam3RuntimeConfig,
    Sam3RuntimeError,
    compute_file_sha256,
    validate_rgb_image,
)


def _config(tmp_path: Path, *, checkpoint_sha256: str | None = None) -> Sam3RuntimeConfig:
    source = tmp_path / "sam3-source"
    (source / "sam3").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    return Sam3RuntimeConfig(
        sam3_source_dir=source,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256 or hashlib.sha256(b"fixture").hexdigest(),
    )


def _fake_runtime(tmp_path: Path) -> tuple[Sam3Runtime, dict[str, int]]:
    config = _config(tmp_path)
    calls = {"model": 0, "processor": 0, "image": 0}

    class Processor:
        def __init__(self, model, *, device, confidence_threshold):
            assert confidence_threshold == 0.0
            calls["processor"] += 1

        def set_image(self, image):
            calls["image"] += 1
            return {"state": True}

        def set_text_prompt(self, *, prompt, state):
            assert prompt == "bowl"
            assert state == {"state": True}
            return {
                "masks": np.array([[[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
                "scores": np.array([0.9], dtype=np.float32),
                "boxes": np.array([[1.0, 0.0, 3.0, 2.0]], dtype=np.float32),
            }

    def model_factory(**kwargs):
        calls["model"] += 1
        assert kwargs["checkpoint_path"].endswith("checkpoint.pt")
        return object()

    return Sam3Runtime(config, model_factory=model_factory, processor_factory=Processor), calls


def test_lazy_load_and_original_size_output(tmp_path: Path):
    runtime, calls = _fake_runtime(tmp_path)
    assert not runtime.loaded
    result = runtime.predict(Sam3Request(np.zeros((2, 3, 3), dtype=np.uint8)))
    assert runtime.loaded
    assert calls == {"model": 1, "processor": 1, "image": 1}
    assert result.image_height == 2 and result.image_width == 3
    assert result.detections[0].mask.shape == (2, 3)
    assert result.detections[0].mask.dtype == np.bool_
    runtime.predict(Sam3Request(np.zeros((2, 3, 3), dtype=np.uint8)))
    assert calls == {"model": 1, "processor": 1, "image": 2}


@pytest.mark.parametrize("rgb", [np.zeros((3, 3), dtype=np.uint8), np.zeros((2, 3, 4), dtype=np.uint8), np.zeros((2, 3, 3), dtype=np.float32)])
def test_rgb_contract_fails_closed(rgb):
    with pytest.raises((TypeError, ValueError)):
        validate_rgb_image(rgb)
    with pytest.raises((TypeError, ValueError)):
        Sam3Request(rgb)


def test_checkpoint_digest_mismatch_fails_before_factory(tmp_path: Path):
    config = _config(tmp_path, checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256)
    runtime = Sam3Runtime(config, model_factory=lambda **_: pytest.fail("model must not load"), processor_factory=lambda *_args, **_kwargs: None)
    with pytest.raises(Sam3RuntimeError, match="SHA-256"):
        runtime.predict(Sam3Request(np.zeros((2, 2, 3), dtype=np.uint8)))


def test_omnis_source_is_rejected(tmp_path: Path):
    config = _config(tmp_path)
    source = tmp_path / "omnis" / "sam3-source"
    (source / "sam3").mkdir(parents=True)
    object.__setattr__(config, "sam3_source_dir", source)
    runtime = Sam3Runtime(config, model_factory=lambda **_: pytest.fail("model must not load"), processor_factory=lambda *_args, **_kwargs: None)
    with pytest.raises(Sam3RuntimeError, match="Omnis"):
        runtime.predict(Sam3Request(np.zeros((2, 2, 3), dtype=np.uint8)))


def test_non_original_size_masks_are_rejected(tmp_path: Path):
    config = _config(tmp_path)

    class Processor:
        def __init__(self, *_args, **_kwargs): pass
        def set_image(self, _image): return None
        def set_text_prompt(self, **_kwargs):
            return {"masks": np.ones((1, 1, 1), dtype=np.float32), "scores": [0.9], "boxes": [[0, 0, 1, 1]]}

    runtime = Sam3Runtime(config, model_factory=lambda **_: object(), processor_factory=Processor)
    with pytest.raises(Sam3RuntimeError, match="original image resolution"):
        runtime.predict(Sam3Request(np.zeros((2, 2, 3), dtype=np.uint8)))


def test_digest_helper_is_bounded_and_exact(tmp_path: Path):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    assert compute_file_sha256(path) == hashlib.sha256(b"abc").hexdigest()
    assert SAM3_SOURCE_COMMIT == "96914d2425f90a64f45ca977c2b5165418099543"
