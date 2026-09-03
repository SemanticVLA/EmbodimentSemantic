"""External, reproducible controller configuration loading.

The checked-in arrow runtime has one active policy: v9d.  Historical arrow
and ZeroGrasp documents live below ``retired/`` and are deliberately rejected
by this loader.  Fine-tuned VLA configurations are outside this package and
are unaffected.  Expansion is deliberately small: ``extends`` may name a
relative JSON file (or a bundled config stem), and mapping values are
recursively merged while lists are replaced as a unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_PACKAGE_DIR = Path(__file__).resolve().parent
ACTIVE_CONTROLLER_CONFIG_FILENAME = "v9d_rgbd_region_grasp_search.json"
ACTIVE_CONTROLLER_CONFIG_PATH = _PACKAGE_DIR / ACTIVE_CONTROLLER_CONFIG_FILENAME
ACTIVE_CONTROLLER_NAME = "libero_spatial_akita_bowl_agentview_v9d_rgbd_region_grasp_search"
_RETIRED_DIRNAME = "retired"
RETIRED_CONTROLLER_NAMES = frozenset(
    {
        "libero_spatial_akita_bowl_agentview_v9_patient_control",
        "libero_spatial_akita_bowl_agentview_v9a_bounded_grasp_search",
        "libero_spatial_akita_bowl_agentview_v9b_residual_micro_correction",
        "libero_spatial_akita_bowl_agentview_v9c_combined",
        "libero_spatial_akita_bowl_agentview_v9e_rgbd_region_with_micro_correction",
        "libero_spatial_akita_bowl_agentview_v9f_rgbd_region_height_sweep",
        "libero_spatial_akita_bowl_agentview_v9g_post_lift_retention",
        "libero_spatial_akita_bowl_agentview_v9h_rgbd_source_approach",
        "libero_spatial_akita_bowl_agentview_v9i_support_plane_placement",
        "libero_spatial_akita_bowl_agentview_v9j_combined_evidence_repair",
        "libero_spatial_akita_bowl_agentview_v10_zg_grasp_only",
        "libero_spatial_akita_bowl_agentview_v10_zg_grasp_recon_place",
    }
)


class ControllerConfigError(ValueError):
    """Raised for invalid or unsafe external controller configurations."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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


def _resolve_reference(reference: str, parent: Path) -> Path:
    candidate = Path(reference).expanduser()
    choices = []
    if candidate.is_absolute():
        choices.append(candidate)
    else:
        choices.extend((parent / candidate, _PACKAGE_DIR / candidate))
        if candidate.suffix == "":
            choices.extend((parent / f"{candidate.name}.json", _PACKAGE_DIR / f"{candidate.name}.json"))
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise ControllerConfigError(f"controller config not found: {reference!r} (from {parent})")


def _reject_retired_config(path: Path) -> None:
    """Fail closed for archived experimental policies before runtime use."""
    try:
        path.relative_to(_PACKAGE_DIR / _RETIRED_DIRNAME)
    except ValueError:
        return
    raise ControllerConfigError(
        f"controller config is retired and cannot be executed: {path.name}"
    )


def load_controller_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and fully expand one controller JSON document.

    Cycles are rejected using canonical resolved paths.  The returned mapping
    contains no ``extends`` key, making it safe to hash and audit directly.
    """
    root = ACTIVE_CONTROLLER_CONFIG_PATH if path is None else _resolve_reference(str(path), Path.cwd())
    root = root.resolve()
    _reject_retired_config(root)
    active: list[Path] = []

    def load_one(current: Path) -> dict[str, Any]:
        current = current.resolve()
        _reject_retired_config(current)
        if current in active:
            chain = " -> ".join(str(item) for item in (*active, current))
            raise ControllerConfigError(f"controller config extends cycle: {chain}")
        try:
            data = json.loads(current.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ControllerConfigError(f"invalid controller config JSON: {current}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ControllerConfigError(f"controller config root must be an object: {current}")
        active.append(current)
        try:
            parent_refs = data.get("extends", [])
            if isinstance(parent_refs, str):
                parent_refs = [parent_refs]
            if not isinstance(parent_refs, list) or not all(isinstance(item, str) for item in parent_refs):
                raise ControllerConfigError(f"extends must be a string or list of strings: {current}")
            merged: dict[str, Any] = {}
            for ref in parent_refs:
                merged = _deep_merge(merged, load_one(_resolve_reference(ref, current.parent)))
            own = {key: value for key, value in data.items() if key != "extends"}
            merged = _deep_merge(merged, own)
            return canonical_controller_config(merged)
        finally:
            active.pop()

    expanded = load_one(root)
    if expanded.get("name") in RETIRED_CONTROLLER_NAMES:
        raise ControllerConfigError(
            f"controller policy is retired and cannot be executed: {expanded['name']}"
        )
    expanded["config_source"] = root.as_posix()
    # Source path is provenance, but not a semantic configuration value.  Keep
    # it out of the hash so copying a config does not change controller policy.
    expanded["config_hash"] = controller_config_hash(expanded)
    return expanded


__all__ = [
    "ACTIVE_CONTROLLER_CONFIG_FILENAME",
    "ACTIVE_CONTROLLER_CONFIG_PATH",
    "ACTIVE_CONTROLLER_NAME",
    "ControllerConfigError",
    "RETIRED_CONTROLLER_NAMES",
    "canonical_controller_config",
    "controller_config_hash",
    "load_controller_config",
]
