from __future__ import annotations

import pytest

from .contracts import ContractError
from .experiment import initial_state_hash


np = pytest.importorskip("numpy")


def test_initial_state_hash_preserves_array_dtype_and_shape():
    int8 = {"state": np.asarray([1], dtype=np.int8)}
    int64 = {"state": np.asarray([1], dtype=np.int64)}
    row = {"state": np.asarray([[1]], dtype=np.int8)}
    assert initial_state_hash(int8) != initial_state_hash(int64)
    assert initial_state_hash(int8) != initial_state_hash(row)


def test_initial_state_hash_rejects_nonfinite_values():
    with pytest.raises(ContractError, match="non-finite"):
        initial_state_hash({"state": np.asarray([float("nan")], dtype=np.float32)})

