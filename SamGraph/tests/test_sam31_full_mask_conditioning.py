from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
import threading
import types

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import the focused module without executing the broad package initializer;
# that initializer imports optional SciPy-backed geometry dependencies which
# are unrelated to this native tracker contract.
if "samgraph_core" not in sys.modules:
    package = types.ModuleType("samgraph_core")
    package.__path__ = [str(ROOT / "src" / "samgraph_core")]
    sys.modules["samgraph_core"] = package

from samgraph_core.sam31_live_scene import (  # noqa: E402
    LocalSam31SceneTracker,
    _mask_foreground_point,
)


def _hollow_mask(size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[2:-2, 2:-2] = True
    mask[4:-4, 4:-4] = False
    return mask


class _NativeTracker:
    def __init__(self, singletons: dict[int, dict], *, fail: bool = False):
        self.singletons = singletons
        self.fail = fail
        self.add_calls = []
        self.preflight_calls = []

    def add_new_masks(self, state, frame_idx, obj_ids, masks, *,
                      add_mask_to_memory, reconditioning):
        if self.fail:
            raise RuntimeError("synthetic native mask failure")
        assert frame_idx >= 0
        assert add_mask_to_memory is False
        assert reconditioning is True
        # Match the real tracker boundary: the outer scene has no such map.
        assert all(object_id in state["obj_id_to_idx"] for object_id in obj_ids)
        self.add_calls.append((state, frame_idx, obj_ids, masks))
        # Pinned SAM's production singleton state stores the replacement as
        # [N,C,H,W], not directly as H,W.
        for object_id, mask in zip(obj_ids, masks):
            assert state is self.singletons[object_id]
            position = state["obj_id_to_idx"][object_id]
            state["mask_inputs_per_obj"][position] = {frame_idx: mask[None, None]}
            state["point_inputs_per_obj"][position] = {}

    def propagate_in_video_preflight(self, state, *, run_mem_encoder):
        assert run_mem_encoder is True
        self.preflight_calls.append(state)
        assert "obj_id_to_idx" in state
        state["maskmem_features"] = [object()]


class _Model:
    def __init__(self, singletons: dict[int, dict], *, fail: bool = False):
        self.singletons = singletons
        for object_id, native in singletons.items():
            mapping = native.setdefault("obj_id_to_idx", {})
            mapping.setdefault(object_id, len(mapping))
            native.setdefault("device", "cpu")
            native.setdefault("mask_inputs_per_obj", {})
            native.setdefault("point_inputs_per_obj", {})[mapping[object_id]] = {0: [1]}
        self.tracker = _NativeTracker(self.singletons, fail=fail)
        self._cache_calls = []

    def _get_sam2_inference_states_by_obj_ids(self, state, object_ids):
        return [self.singletons[object_id] for object_id in object_ids]

    def _cache_frame_outputs(self, state, frame_idx, outputs):
        self._cache_calls.append((state, frame_idx, outputs))


def _tracker(model: _Model) -> LocalSam31SceneTracker:
    tracker = LocalSam31SceneTracker.__new__(LocalSam31SceneTracker)
    tracker._lock = threading.RLock()
    tracker._runtime = None
    tracker._sessions = {}
    tracker._scenes = {}
    tracker._dispatch_patched = False
    return tracker


def test_foreground_point_is_inside_hollow_mask():
    point = _mask_foreground_point(_hollow_mask())
    x = int(point[0] * 16)
    y = int(point[1] * 16)
    assert _hollow_mask()[y, x]


def test_full_mask_contract_replaces_point_and_encodes_memory():
    import torch

    seed = _hollow_mask()
    singleton = {}
    model = _Model({1: singleton})
    tracker = _tracker(model)
    session = types.SimpleNamespace(last_mask=seed, snapshot={})
    state = {
        "orig_height": 16, "orig_width": 16, "device": "cpu",
        "previous_stages_out": [None],
        "sam2_inference_states": [{"obj_id_to_idx": {1: 0}}],
    }
    digests = tracker._condition_full_mask(
        types.SimpleNamespace(), model, state, [session], [1])
    assert model.tracker.add_calls
    assert model.tracker.add_calls[0][3].dtype == torch.bool
    assert np.array_equal(model.tracker.add_calls[0][3].cpu().numpy()[0], seed)
    assert singleton["point_inputs_per_obj"] == {0: {}}
    assert singleton["maskmem_features"]
    assert state["previous_stages_out"][0] == "_THIS_FRAME_HAS_OUTPUTS_"
    assert len(model._cache_calls) == 1
    assert list(model._cache_calls[0][2]) == [1]
    assert tuple(model._cache_calls[0][2][1].shape) == (1, 16, 16)
    assert digests[1]


def test_full_mask_outer_cache_preserves_two_objects_and_batch_shape():
    import torch

    first = _hollow_mask()
    second = np.zeros_like(first)
    second[1:4, 1:4] = True
    shared_native = {}
    model = _Model({1: shared_native, 2: shared_native})
    tracker = _tracker(model)
    state = {
        "orig_height": 16, "orig_width": 16, "device": "cpu",
        "previous_stages_out": [None],
        "sam2_inference_states": [{"obj_id_to_idx": {1: 0, 2: 1}}],
    }
    tracker._condition_full_mask(
        types.SimpleNamespace(), model, state,
        [types.SimpleNamespace(last_mask=first), types.SimpleNamespace(last_mask=second)], [1, 2])
    assert list(model._cache_calls[0][2]) == [1, 2]
    assert len(model.tracker.add_calls) == 1
    assert model.tracker.add_calls[0][0] is shared_native
    assert tuple(model.tracker.add_calls[0][3].shape) == (2, 16, 16)
    assert all(tuple(value.shape) == (1, 16, 16) and value.dtype == torch.bool
               for value in model._cache_calls[0][2].values())


def test_full_mask_failure_is_fail_closed():
    seed = _hollow_mask()
    singleton = {}
    model = _Model({1: singleton}, fail=True)
    tracker = _tracker(model)
    state = {
        "orig_height": 16, "orig_width": 16, "device": "cpu",
        "previous_stages_out": [None],
        "sam2_inference_states": [{"obj_id_to_idx": {1: 0}}],
    }
    try:
        tracker._condition_full_mask(
            types.SimpleNamespace(), model, state,
            [types.SimpleNamespace(last_mask=seed)], [1])
    except RuntimeError as exc:
        assert "synthetic native mask failure" in str(exc)
    else:
        raise AssertionError("partial native conditioning was accepted")
    assert not model._cache_calls
    assert state["previous_stages_out"] == [None]


def test_current_frame_insertion_preserves_other_cached_object():
    import torch

    first = _hollow_mask()
    second = np.zeros_like(first)
    second[1:4, 1:4] = True
    shared_native = {}
    model = _Model({1: shared_native, 2: shared_native})
    tracker = _tracker(model)
    shared_native["point_inputs_per_obj"][0] = {}
    first_cache = torch.as_tensor(first).unsqueeze(0)
    state = {
        "orig_height": 16, "orig_width": 16, "device": "cpu",
        "previous_stages_out": [None, None, None],
        "cached_frame_outputs": {2: {1: first_cache}},
        "sam2_inference_states": [{"obj_id_to_idx": {1: 0, 2: 1}}],
    }
    tracker._condition_full_mask(
        types.SimpleNamespace(), model, state,
        [types.SimpleNamespace(last_mask=second)], [2], frame_index=2,
    )
    _, index, cached = model._cache_calls[0]
    assert index == 2
    assert list(cached) == [1, 2]
    assert torch.equal(cached[1], first_cache)
    assert state["previous_stages_out"][2] == "_THIS_FRAME_HAS_OUTPUTS_"
    assert 2 in shared_native["mask_inputs_per_obj"][1]
