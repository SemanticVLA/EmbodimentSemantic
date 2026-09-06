from __future__ import annotations

import numpy as np

from vla_benchmarking.libero.evaluation.native_vla_eval import _canonical_observation
from vla_benchmarking.libero.evaluation.registry import get_policy_capabilities
from vla_benchmarking.libero.finetuned_vlas.openvla import OpenVLAAdapter


def test_original_openvla_adapter_uses_official_prompt_crop_and_action_semantics() -> None:
    calls: dict[str, object] = {}

    class Processor:
        def __call__(self, prompt, image):
            calls["prompt"] = prompt
            calls["size"] = image.size
            return {}

    class Model:
        def predict_action(self, **kwargs):
            calls["kwargs"] = kwargs
            return np.asarray([0, 0, 0, 0, 0, 0, 0.8], dtype=np.float32)

    adapter = OpenVLAAdapter(Model(), processor=Processor())
    adapter.reset("Pick Up The Bowl", 0)
    action = adapter.act({"agentview": np.zeros((256, 256, 3), dtype=np.uint8)})
    assert action.shape == (1, 7)
    assert action[0, -1] == -1.0
    assert calls["prompt"] == "In: What action should the robot take to pick up the bowl?\nOut:"
    assert calls["size"] == (224, 224)
    assert calls["kwargs"] == {"unnorm_key": "libero_spatial", "do_sample": False}


def test_original_openvla_environment_contract_has_only_agentview() -> None:
    observation = _canonical_observation(
        {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.zeros(2),
        },
        original_openvla=True,
    )
    assert sorted(observation) == ["agentview", "frame_provenance"]


def test_original_openvla_is_a_distinct_native_policy_kind() -> None:
    capabilities = get_policy_capabilities("openvla")
    assert capabilities.backend.name == "native_policy"
    assert capabilities.visual_inputs == ("none",)
