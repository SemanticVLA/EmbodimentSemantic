from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest

from grasp_controller.molmopoint import (
    DEFAULT_PROMPT_ID,
    MOLMOPOINT_MODEL_ID,
    MOLMOPOINT_MODEL_REVISION,
    MolmoPointRequest,
    MolmoPointRuntime,
    MolmoPointRuntimeConfig,
    MolmoPointRuntimeError,
    build_mask_highlight,
)


class _Tensor:
    def __init__(self, value, *, shape=None):
        self.value = value
        self.shape = shape or getattr(value, "shape", None)
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(device)
        return self

    def __getitem__(self, item):
        return self.value[item]


class _Torch:
    bfloat16 = "BF16"

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def autocast(**_kwargs):
        return contextlib.nullcontext()


def _runtime(*, raw_points=None, config=None, capture=None):
    calls = {"model": [], "processor": [], "generate": []}
    raw_points = [[4, 0, 10.5, 5.0]] if raw_points is None else raw_points

    class Processor:
        def __init__(self, model_id, **kwargs):
            calls["processor"].append((model_id, kwargs))

        def apply_chat_template(self, messages, **kwargs):
            if capture is not None:
                capture["messages"] = messages
                capture["kwargs"] = kwargs
            return {
                "input_ids": _Tensor(np.zeros((1, 3), dtype=np.int64), shape=(1, 3)),
                "attention_mask": _Tensor(np.ones((1, 3), dtype=np.int64), shape=(1, 3)),
                "metadata": {"token_pooling": "pool", "subpatch_mapping": "map", "image_sizes": "sizes"},
            }

        def post_process_image_text_to_text(self, tokens, **kwargs):
            calls["post"] = (tokens, kwargs)
            return ["<point>"]

    class Model:
        def eval(self):
            calls["eval"] = True

        def build_logit_processor_from_inputs(self, inputs):
            calls["logit_inputs"] = inputs
            return "logits"

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return np.array([[101, 102, 103, 104, 105]], dtype=np.int64)

        def extract_image_points(self, text, pooling, mapping, sizes):
            calls["extract"] = (text, pooling, mapping, sizes)
            return raw_points

    def model_factory(model_id, **kwargs):
        calls["model"].append((model_id, kwargs))
        return Model()

    runtime = MolmoPointRuntime(
        config or MolmoPointRuntimeConfig(device="cpu", device_map=None),
        model_factory=model_factory,
        processor_factory=Processor,
        torch_factory=lambda: _Torch(),
    )
    return runtime, calls


def test_lazy_load_uses_official_contract_and_original_pixels():
    runtime, calls = _runtime()
    assert not runtime.loaded
    result = runtime.predict(MolmoPointRequest(np.zeros((8, 12, 3), dtype=np.uint8)))
    assert runtime.loaded
    assert result.points[0].uv == (10.5, 5.0)
    assert calls["model"][0][0] == MOLMOPOINT_MODEL_ID
    assert calls["model"][0][1]["revision"] == MOLMOPOINT_MODEL_REVISION
    assert calls["model"][0][1]["dtype"] == "BF16"
    assert calls["processor"][0][1]["trust_remote_code"] is True
    assert calls["extract"] == ("<point>", "pool", "map", "sizes")
    assert calls["post"][1]["skip_special_tokens"] is False
    assert calls["post"][1]["clean_up_tokenization_spaces"] is False
    assert calls["generate"][0]["logits_processor"] == "logits"
    assert calls["generate"][0]["max_new_tokens"] == 200
    assert calls["generate"][0]["do_sample"] is False
    assert calls["generate"][0]["num_beams"] == 1
    assert result.provenance["generated_text"] == "<point>"
    assert result.provenance["decoded_point_count"] == 1


def test_mask_highlight_is_original_size_and_sent_to_processor():
    capture = {}
    runtime, _calls = _runtime(capture=capture, raw_points=[[0, 0, 2.0, 1.0]])
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.bool_)
    mask[1, 2] = True
    result = runtime.predict(MolmoPointRequest(image, mask=mask))
    assert result.highlighted is True
    sent = capture["messages"][0]["content"][1]["image"]
    assert sent.size == (5, 4)
    assert np.asarray(sent).shape == image.shape
    assert np.asarray(sent)[1, 2, 0] > 0


@pytest.mark.parametrize(
    "raw_points, match",
    [([[0, 1, 1, 1]], "image_num"), ([[0, 0, -1, 1]], "outside"), ([[True, 0, 1, 1]], "object_id"), ([[0, 0, np.nan, 1]], "finite")],
)
def test_point_frame_and_ids_fail_closed(raw_points, match):
    runtime, _calls = _runtime(raw_points=raw_points)
    with pytest.raises(MolmoPointRuntimeError, match=match):
        runtime.predict(MolmoPointRequest(np.zeros((4, 4, 3), dtype=np.uint8)))


def test_config_rejects_non_bfloat16_and_provenance_is_hashed():
    with pytest.raises(ValueError, match="bfloat16"):
        MolmoPointRuntimeConfig(dtype="float16")
    provenance = MolmoPointRuntimeConfig().provenance()
    assert provenance["model_revision"] == MOLMOPOINT_MODEL_REVISION
    assert len(provenance["config_sha256"]) == 64


def test_default_prompt_requests_executable_contact_alternatives():
    config = MolmoPointRuntimeConfig()
    assert DEFAULT_PROMPT_ID == "rim_clearance"
    assert "exposed parts" in config.prompt
    assert "robot fingers" in config.prompt
    assert "nearby objects" in config.prompt
    assert config.provenance()["prompt_id"] == DEFAULT_PROMPT_ID


def test_request_rejects_wrong_mask_frame():
    with pytest.raises(ValueError, match="mask"):
        MolmoPointRequest(np.zeros((3, 4, 3), dtype=np.uint8), mask=np.ones((4, 3), dtype=np.bool_))
    with pytest.raises(TypeError, match="boolean"):
        build_mask_highlight(np.zeros((3, 4, 3), dtype=np.uint8), np.ones((3, 4), dtype=np.uint8))
