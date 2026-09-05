"""Shared raw-observation access for LIBERO evaluation adapters."""

from __future__ import annotations

from typing import Any, Mapping


def read_raw_observation(
    env: Any,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    """Read one current observation without consulting simulator object poses."""
    for owner in (getattr(env, "env", None), env):
        getter = getattr(owner, "_get_observations", None)
        if getter is None:
            continue
        try:
            value = getter(force_update=True)
        except TypeError:
            value = getter()
        if isinstance(value, Mapping):
            return value
    if required:
        raise RuntimeError("LIBERO env did not expose a raw observation")
    return None


__all__ = ["read_raw_observation"]
