from __future__ import annotations

import pytest

from vla_benchmarking.evaluation.observation import read_raw_observation


def test_read_raw_observation_prefers_wrapped_owner_and_forces_update() -> None:
    calls: list[tuple[str, bool]] = []

    class Inner:
        def _get_observations(self, force_update: bool = False):
            calls.append(("inner", force_update))
            return {"source": "inner"}

    class Outer:
        env = Inner()

        def _get_observations(self, force_update: bool = False):
            calls.append(("outer", force_update))
            return {"source": "outer"}

    assert read_raw_observation(Outer(), required=True) == {"source": "inner"}
    assert calls == [("inner", True)]


def test_read_raw_observation_required_contract() -> None:
    assert read_raw_observation(object(), required=False) is None
    with pytest.raises(RuntimeError, match="raw observation"):
        read_raw_observation(object(), required=True)
