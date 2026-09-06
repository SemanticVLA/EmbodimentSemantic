from __future__ import annotations

import numpy as np

from vla_benchmarking.libero.evaluation.registry import get_policy_capabilities
from vla_benchmarking.libero.finetuned_vlas.openvla_oft import OpenVLAOFTAdapter
from vla_benchmarking.libero.finetuned_vlas.pi05 import Pi05Adapter


def test_native_adapter_metadata_policy_kinds_are_registered() -> None:
    """Adapter metadata must be consumable by the shared policy registry."""

    pi = Pi05Adapter(
        type("FakePi", (), {"select_action": lambda _self, _payload: np.zeros((10, 7), dtype=np.float32)})()
    )
    oft = OpenVLAOFTAdapter(
        type("FakeOFT", (), {"predict_action": lambda _self, _payload: np.zeros((8, 7), dtype=np.float32)})()
    )

    for adapter in (pi, oft):
        capabilities = get_policy_capabilities(adapter.metadata.policy_kind)
        assert capabilities.backend.name
