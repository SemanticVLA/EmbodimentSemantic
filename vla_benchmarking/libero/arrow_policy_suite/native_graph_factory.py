"""Fail-closed native graph-context factory for Fast and Trace.

``native_legion_factory`` loads this module through
``ARROW_SUITE_GRAPH_FACTORY=arrow_policy_suite.native_graph_factory:build_graph_context``.
The callback returns a plain mapping because ``NativeHost`` deliberately
round-trips callback metadata through ``dict(...)``.  The mapping is the
serialized form of :class:`GraphContextPacket` and retains its packet digest.

The factory does not inspect simulator state and never adds graph context to
the VLA payload.  The selected text triplet and visual-arrow metadata must be
provided explicitly through environment variables (or direct keyword
arguments in tests/launchers), together with a sealed graph-context revision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, ObservationFrame, _safe
from .graph_context import GraphContextPacket, make_graph_context


_TRIPLET_ENV_NAMES = ("ARROW_SUITE_TEXT_GRAPH_TRIPLET", "ARROW_SUITE_GRAPH_TRIPLET")
_ARROW_ENV_NAMES = ("ARROW_SUITE_VISUAL_ARROW", "ARROW_SUITE_ARROW_METADATA")
_FORBIDDEN_KEYS = (
    "ground_truth", "sim_ground_truth", "oracle_pose", "object_pose_gt",
    "simulator_state", "mujoco_state", "joint_qpos", "teacher_action",
    "executed_action", "action_chunk", "controller_state",
)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None or not str(value).strip() else str(value).strip()


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _env(name)
        if value is not None:
            return value
    return None


def _parse_json(value: str, *, label: str) -> Any:
    source = value
    if value.startswith("@"):
        path = Path(value[1:])
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContractError(f"{label} JSON file is unreadable: {path}") from exc
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} must be valid JSON or @path") from exc
    try:
        return _safe(parsed)
    except ContractError as exc:
        raise ContractError(f"{label} contains non-serializable values") from exc


def _reject_privileged(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in _FORBIDDEN_KEYS):
                raise ContractError(f"privileged graph field {path}.{key} is not allowed")
            _reject_privileged(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_privileged(item, path=f"{path}[{index}]")


def _validate_triplet(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not value:
            raise ContractError("text graph triplet cannot be empty")
        required = {"subject", "relation", "object"}
        if not required.issubset(value):
            raise ContractError("text graph triplet requires subject, relation, and object")
        for key in required:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ContractError(f"text graph triplet {key} must be non-empty text")
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ContractError("text graph triplet entries must be non-empty text")
        return {"subject": value[0], "relation": value[1], "object": value[2]}
    raise ContractError("text graph triplet must be an object or three-item text list")


def _validate_arrow(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError("visual-arrow metadata must be a non-empty object")
    geometry_keys = {"polyline", "points", "source", "destination", "waypoints", "path"}
    if not geometry_keys.intersection(value):
        raise ContractError("visual-arrow metadata requires explicit geometry")
    if "provider" not in value and "provenance" not in value and "source_kind" not in value:
        raise ContractError("visual-arrow metadata requires provider/provenance")
    return dict(value)


def _sealed_revision(explicit: str | None) -> str:
    revision = explicit or _env("ARROW_SUITE_GRAPH_CONTEXT_REVISION")
    if revision is None or revision.lower() in {"", "latest", "unknown", "unresolved"}:
        raise ContractError("ARROW_SUITE_GRAPH_CONTEXT_REVISION must be sealed explicitly")
    return revision


def build_packet(
    frame: ObservationFrame,
    *,
    triplet: Any | None = None,
    arrow_geometry: Mapping[str, Any] | None = None,
    graph_revision: str | None = None,
    arrow_revision: str | None = None,
    phase: str | None = None,
    task_text: str | None = None,
) -> GraphContextPacket:
    """Construct one sealed packet from explicit graph and Arrow inputs."""

    if not isinstance(frame, ObservationFrame):
        raise ContractError("graph factory requires an ObservationFrame")
    raw_triplet = triplet
    if raw_triplet is None:
        encoded = _first_env(_TRIPLET_ENV_NAMES)
        if encoded is None:
            raise ContractError("explicit text graph triplet is required")
        raw_triplet = _parse_json(encoded, label="text graph triplet")
    normalized_triplet = _validate_triplet(raw_triplet)
    if task_text is None:
        task_text = _env("ARROW_SUITE_TASK_TEXT")
    if task_text is not None:
        if not isinstance(task_text, str) or not task_text.strip():
            raise ContractError("task text must be non-empty text")
        normalized_triplet = {**normalized_triplet, "task_text": task_text.strip()}
    _reject_privileged(normalized_triplet, path="triplet")

    raw_arrow = arrow_geometry
    if raw_arrow is None:
        encoded = _first_env(_ARROW_ENV_NAMES)
        if encoded is None:
            raise ContractError("explicit visual-arrow metadata is required")
        raw_arrow = _parse_json(encoded, label="visual-arrow metadata")
    normalized_arrow = _validate_arrow(raw_arrow)
    _reject_privileged(normalized_arrow, path="arrow_geometry")

    sealed = _sealed_revision(None)
    graph_rev = graph_revision or _env("ARROW_SUITE_TEXT_GRAPH_REVISION") or sealed
    arrow_rev = arrow_revision or _env("ARROW_SUITE_ARROW_RENDER_REVISION") or sealed
    if graph_rev != sealed or arrow_rev != sealed:
        raise ContractError("graph and Arrow revisions must match sealed graph-context revision")
    resolved_phase = phase or _env("ARROW_SUITE_GRAPH_PHASE") or "unknown"
    return make_graph_context(
        frame, normalized_triplet, normalized_arrow,
        graph_revision=graph_rev, arrow_revision=arrow_rev, phase=resolved_phase,
    )


def build_graph_context(frame: ObservationFrame, **kwargs: Any) -> Mapping[str, Any]:
    """Native-legion callback returning the packet's host metadata mapping."""

    return build_packet(frame, **kwargs).to_mapping()


__all__ = ["build_packet", "build_graph_context"]
