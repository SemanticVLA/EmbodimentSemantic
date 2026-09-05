"""External, reproducible loading for the sole canonical grasp policy.

Historical controller documents remain recoverable in Git and experiment
archives, but this loader exposes only the promoted canonical policy. Any
other file or policy name fails closed before environment construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_PACKAGE_DIR = Path(__file__).resolve().parent
ACTIVE_CONTROLLER_CONFIG_FILENAME = "canonical_molmo_rgbd_grasp.json"
ACTIVE_CONTROLLER_CONFIG_PATH = _PACKAGE_DIR / ACTIVE_CONTROLLER_CONFIG_FILENAME
ACTIVE_POLICY_LOCK_FILENAME = "active_policy.lock.json"
ACTIVE_POLICY_LOCK_PATH = _PACKAGE_DIR / ACTIVE_POLICY_LOCK_FILENAME
ACTIVE_CONTROLLER_NAME = "libero_spatial_akita_bowl_agentview_canonical"


class ControllerConfigError(ValueError):
    """Raised for invalid or unsafe external controller configurations."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControllerConfigError("configuration contains a non-finite number")
        return value
    raise ControllerConfigError(f"configuration contains unsupported value {type(value).__name__}")


def canonical_controller_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible canonical expansion with deterministic keys."""
    if not isinstance(config, Mapping):
        raise ControllerConfigError("controller config root must be a JSON object")
    return _canonical(config)


def controller_config_hash(config: Mapping[str, Any]) -> str:
    # Provenance fields are intentionally excluded so moving a config file or
    # reading an already-loaded expansion cannot change the policy identity.
    semantic = {
        key: value for key, value in config.items()
        if key not in {"config_source", "config_hash"}
    }
    payload = json.dumps(canonical_controller_config(semantic), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_reference(reference: str) -> Path:
    candidate = Path(reference).expanduser()
    # A few historical evaluators constructed this path relative to the
    # evaluator module (``evaluation/controller_configs/...``).  The config is
    # now owned by arrow_grasp_controller; map that one known legacy spelling
    # to the sole active file without creating a second executable JSON.
    if (
        candidate.name == ACTIVE_CONTROLLER_CONFIG_FILENAME
        and candidate.parent.name == "controller_configs"
        and candidate.parent.parent.name == "evaluation"
    ):
        return ACTIVE_CONTROLLER_CONFIG_PATH.resolve()
    choices: list[Path] = []
    if candidate.is_absolute():
        choices.append(candidate)
    else:
        choices.extend((Path.cwd() / candidate, _PACKAGE_DIR / candidate))
        if candidate.suffix == "":
            choices.append(_PACKAGE_DIR / f"{candidate.name}.json")
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise ControllerConfigError(f"controller config not found: {reference!r}")


def load_controller_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the sole checked-in policy; arbitrary external JSON is rejected."""
    root = ACTIVE_CONTROLLER_CONFIG_PATH.resolve() if path is None else _resolve_reference(str(path))
    if root != ACTIVE_CONTROLLER_CONFIG_PATH.resolve():
        raise ControllerConfigError(
            f"only {ACTIVE_CONTROLLER_CONFIG_FILENAME} is executable"
        )
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControllerConfigError(f"invalid controller config JSON: {root}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ControllerConfigError(f"controller config root must be an object: {root}")
    if "extends" in data:
        raise ControllerConfigError("the canonical policy must be self-contained")
    expanded = canonical_controller_config(data)
    if expanded.get("name") != ACTIVE_CONTROLLER_NAME:
        raise ControllerConfigError(
            f"canonical policy name mismatch: {expanded.get('name')!r}"
        )
    expanded["config_source"] = root.as_posix()
    # Source path is provenance, but not a semantic configuration value.  Keep
    # it out of the hash so copying a config does not change controller policy.
    expanded["config_hash"] = controller_config_hash(expanded)
    return expanded


__all__ = [
    "ACTIVE_CONTROLLER_CONFIG_FILENAME",
    "ACTIVE_CONTROLLER_CONFIG_PATH",
    "ACTIVE_POLICY_LOCK_FILENAME",
    "ACTIVE_POLICY_LOCK_PATH",
    "ACTIVE_CONTROLLER_NAME",
    "ControllerConfigError",
    "canonical_controller_config",
    "controller_config_hash",
    "load_controller_config",
]
