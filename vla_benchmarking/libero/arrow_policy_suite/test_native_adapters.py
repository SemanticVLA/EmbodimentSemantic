from __future__ import annotations

from types import SimpleNamespace
from collections import deque
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite.arrow_adapter import ArrowAdapter
from arrow_policy_suite.contracts import ActionProposal, ObservationFrame
from arrow_policy_suite.libero_adapter import LiberoEnvironmentAdapter, canonicalize_observation
from arrow_policy_suite.smolvla_adapter import SmolVLAAdapter


def _frame(step: int = 0) -> ObservationFrame:
    return ObservationFrame({"state": [0.0] * 8, "instruction": "pick"}, timestep=step)


def test_canonical_projection_drops_raw_sidecars_without_optional_runtime():
    value = {
        "state": [0.0] * 8,
        "agentview": [[[0, 0, 0]]],
        "wrist": [[[0, 0, 0]]],
        "object_pose_gt": [1, 2, 3],
        "instruction": "pick",
    }
    projected = canonicalize_observation(value)
    assert set(projected) == {"state", "agentview", "wrist", "instruction"}


class _Environment:
    def __init__(self) -> None:
        self.value = 0.0

    def observe(self):
        return {"state": [self.value] + [0.0] * 7}

    def step(self, action):
        self.value += action[0]
        return {"observation": self.observe(), "success": False}

    def snapshot(self):
        return self.value

    def restore(self, value):
        self.value = value


def test_libero_adapter_steps_once_and_round_trips_snapshot():
    raw = _Environment()
    environment = LiberoEnvironmentAdapter(raw)
    snapshot = environment.snapshot()
    environment.step([0.25] + [0.0] * 6)
    assert environment.observe()["state"][0] == pytest.approx(0.25)
    environment.restore(snapshot)
    assert environment.observe()["state"][0] == pytest.approx(0.0)


def test_smolvla_adapter_consumes_chunk_before_inference_again():
    calls = []
    adapter = SmolVLAAdapter(inference=lambda observation, step: (calls.append(step) or [[0.1] * 7, [0.2] * 7]))
    first = adapter.propose(_frame())
    adapter.commit(SimpleNamespace(base=first))
    second = adapter.propose(_frame(1))
    assert first.action == (0.1,) * 7
    assert second.action == (0.2,) * 7
    assert calls == [0]


def test_smolvla_snapshot_state_preserves_inflight_and_queue():
    adapter = SmolVLAAdapter(inference=lambda observation, step: [[0.1] * 7, [0.2] * 7])
    frame = _frame()
    proposal = adapter.propose(frame)
    snapshot = adapter.snapshot_state()
    adapter.reset()
    adapter.restore_state(snapshot)
    assert adapter.propose(frame) == proposal


def test_native_lerobot_queue_and_single_step_setting_round_trip():
    class NativePolicy:
        def __init__(self):
            self._queues = {"action": deque([[0.1] * 7])}
            self.config = SimpleNamespace(n_action_steps=50)

        def propose(self, _payload, _step):
            return list(self._queues["action"])

    policy = NativePolicy()
    adapter = SmolVLAAdapter(policy=policy, force_single_action_step=True)
    assert policy.config.n_action_steps == 1
    snapshot = adapter.snapshot()
    expected = random.random()
    policy._queues["action"].append([0.9] * 7)
    policy.config.n_action_steps = 50
    random.random()
    adapter.restore(snapshot)
    assert list(policy._queues["action"]) == [[0.1] * 7]
    assert policy.config.n_action_steps == 1
    assert random.random() == expected
    assert adapter.rollback_complete


def test_native_smolvla_converts_readonly_hwc_images_to_batched_chw_tensors():
    captured = {}

    class NativePolicy:
        def select_action(self, batch):
            captured["batch"] = batch
            return torch.zeros((1, 7), dtype=torch.float32)

    def preprocessor(payload):
        captured["payload"] = payload
        return payload

    adapter = SmolVLAAdapter(
        policy=NativePolicy(),
        preprocessor=preprocessor,
        postprocessor=lambda value: value,
    )
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    frame = ObservationFrame(
        {
            "agentview": image,
            "wrist": image.copy(),
            "state": np.zeros(8, dtype=np.float32),
            "instruction": "pick up the bowl",
        },
        timestep=0,
    )

    proposal = adapter.propose(frame)
    adapter.commit(SimpleNamespace(base=proposal))

    payload = captured["payload"]
    for key in ("observation.images.image", "observation.images.image2"):
        value = payload[key]
        assert tuple(value.shape) == (1, 3, 256, 256)
        assert value.dtype == torch.float32
        assert value.is_contiguous()
        assert value.isfinite().all()
        assert float(value.min()) == pytest.approx(128.0 / 255.0)
    assert tuple(payload["observation.state"].shape) == (1, 8)
    assert payload["task"] == "pick up the bowl"


def test_native_smolvla_does_not_rescale_already_prepared_bchw_images():
    captured = {}

    class NativePolicy:
        def select_action(self, batch):
            captured["batch"] = batch
            return torch.zeros((1, 7), dtype=torch.float32)

    def preprocessor(payload):
        captured["payload"] = payload
        return payload

    image = torch.full((1, 3, 256, 256), 0.5, dtype=torch.float32)
    adapter = SmolVLAAdapter(
        policy=NativePolicy(),
        preprocessor=preprocessor,
        postprocessor=lambda value: value,
    )
    frame = ObservationFrame(
        {"agentview": image, "wrist": image, "state": torch.zeros(8), "instruction": "pick"},
        timestep=0,
    )
    proposal = adapter.propose(frame)
    adapter.commit(SimpleNamespace(base=proposal))

    assert torch.allclose(captured["payload"]["observation.images.image"], image)
    assert tuple(captured["payload"]["observation.state"].shape) == (1, 8)


def test_processor_pipeline_state_dictionary_is_rollback_covered():
    class Processor:
        def __init__(self):
            self.counter = 3

        def __call__(self, value):
            self.counter += 1
            return value

    inference = lambda _observation, _step: [0.0] * 7
    inference.__arrow_stateless__ = True
    pre = Processor()
    post = Processor()
    adapter = SmolVLAAdapter(inference=inference, preprocessor=pre, postprocessor=post)
    assert adapter.rollback_complete
    snapshot = adapter.snapshot()
    pre.counter = 99
    post.counter = 101
    adapter.restore(snapshot)
    assert pre.counter == 3 and post.counter == 3


class _Controller:
    def __init__(self) -> None:
        self.commits = 0

    def reset(self):
        return None

    def propose(self, frame):
        return {"action": [0.0] * 7}

    def commit(self, record):
        self.commits += 1

    def interrupt(self):
        return None

    def snapshot(self):
        return self.commits

    def restore(self, value):
        self.commits = value


def test_arrow_adapter_requires_matching_commit_and_exposes_interrupt_hook():
    controller = _Controller()
    adapter = ArrowAdapter(controller, require_interrupt_hook=True)
    adapter.reset()
    proposal = adapter.propose(_frame())
    adapter.commit(SimpleNamespace(teacher=proposal))
    assert controller.commits == 1
    interrupted = adapter.interrupt(_frame(1), reason="test")
    assert interrupted is not None and interrupted.interruptible


def test_arrow_snapshot_state_alias_preserves_pending_proposal():
    controller = _Controller()
    adapter = ArrowAdapter(controller)
    adapter.reset()
    frame = _frame()
    proposal = adapter.propose(frame)
    snapshot = adapter.snapshot_state()
    adapter.restore_state(snapshot)
    assert adapter.propose(frame) == proposal


def test_adapters_prefer_state_only_hooks_for_rollback():
    class StateController(_Controller):
        snapshot = None
        restore = None

        def snapshot_state(self):
            return self.commits

        def restore_state(self, value):
            self.commits = value

    controller = StateController()
    arrow = ArrowAdapter(controller)
    arrow.reset()
    frame = _frame()
    arrow.propose(frame)
    state = arrow.snapshot_state()
    controller.commits = 7
    arrow.restore_state(state)
    assert controller.commits == 0

    class StatePolicy:
        def __init__(self):
            self.value = 0

        def __call__(self, _observation, _step):
            return [0.0] * 7

        def snapshot_state(self):
            return self.value

        def restore_state(self, value):
            self.value = value

    policy = StatePolicy()
    smolvla = SmolVLAAdapter(policy=policy)
    smolvla.propose(frame)
    state = smolvla.snapshot_state()
    policy.value = 9
    smolvla.restore_state(state)
    assert policy.value == 0


def test_arrow_adapter_rejects_whole_episode_recovery_only_controller():
    class Legacy:
        def recover(self, *_args):
            return None

    with pytest.raises(TypeError, match="propose"):
        ArrowAdapter(Legacy())


def test_arrow_adapter_requires_explicit_stateless_callable_declaration():
    class CallableController:
        def __call__(self, _frame):
            return [0.0] * 7

        def commit(self, _record):
            return None

    controller = CallableController()
    adapter = ArrowAdapter(controller)
    assert adapter.rollback_complete is False
    controller.__arrow_stateless__ = True
    assert adapter.rollback_complete is True
