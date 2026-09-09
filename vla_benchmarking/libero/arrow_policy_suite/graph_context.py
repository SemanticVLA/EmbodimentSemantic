"""Digest-bound context passed to the graph-local Fast corrector.

The context packet is deliberately *outside* the VLA observation.  It is a
small, immutable view joining the selected text-graph relation, the Arrow
rendering of that relation, and the authoritative observation digest.  A Fast
artifact may use this packet to correct a VLA proposal, while the VLA itself
continues to receive exactly its checkpoint-owned observation payload.

Keeping the packet as a first-class, hashed object is important for the
one-shot experiment: support and scored queries can only be paired when they
refer to the same graph/arrow revision and observation frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .contracts import ContractError, ObservationFrame, _safe, digest, state8


GRAPH_CONTEXT_SCHEMA = "arrow_policy_suite.graph_context.v1"


def _finite_tuple(value: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be a numeric sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a numeric sequence") from exc
    if any(not math.isfinite(item) for item in result):
        raise ContractError(f"{name} contains a non-finite value")
    return result


def _canonical(value: Any) -> Any:
    """Return JSON-safe, deterministic data or fail closed."""

    try:
        return _safe(value)
    except ContractError:
        # Contexts should never silently stringify arbitrary simulator objects:
        # doing so would make the digest unverifiable across processes.
        raise


@dataclass(frozen=True)
class GraphContextPacket:
    """Immutable graph/arrow/observation context for one proposal.

    ``observation`` is a compact student-side view.  Images are intentionally
    not copied into the packet; ``observation_digest`` binds it to the full
    canonical frame and ``state`` gives the optional low-dimensional signal
    used by deterministic encoders.  No packet field is passed to the VLA.
    """

    triplet: Any
    arrow_geometry: Any
    observation_digest: str
    observation_state: tuple[float, ...] | None
    graph_revision: str
    arrow_revision: str
    observation_timestep: int = 0
    episode_id: str | None = None
    phase: str = "unknown"
    schema: str = GRAPH_CONTEXT_SCHEMA
    packet_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema != GRAPH_CONTEXT_SCHEMA:
            raise ContractError(f"unsupported graph context schema {self.schema!r}")
        for name, value in (
            ("observation_digest", self.observation_digest),
            ("graph_revision", self.graph_revision),
            ("arrow_revision", self.arrow_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} is required")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ContractError("context phase is required")
        if isinstance(self.observation_timestep, bool) or self.observation_timestep < 0:
            raise ContractError("observation_timestep must be non-negative")
        if self.observation_state is not None:
            state = _finite_tuple(self.observation_state, name="observation_state")
            object.__setattr__(self, "observation_state", state)
        # Validate once at construction so later digest computation cannot
        # encounter an unserialisable native object.
        payload = self.payload()
        calculated = digest(payload)
        if self.packet_digest and self.packet_digest != calculated:
            raise ContractError("graph context packet digest does not match contents")
        object.__setattr__(self, "packet_digest", calculated)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "triplet": _canonical(self.triplet),
            "arrow_geometry": _canonical(self.arrow_geometry),
            "observation_digest": self.observation_digest,
            "observation_state": list(self.observation_state) if self.observation_state is not None else None,
            "observation_timestep": self.observation_timestep,
            "episode_id": self.episode_id,
            "graph_revision": self.graph_revision,
            "arrow_revision": self.arrow_revision,
            "phase": self.phase,
        }

    def for_role(self, role: str) -> dict[str, Any]:
        """Return a role-tagged payload for a frozen slow encoder."""

        if role not in {"hand_to_source", "hand_to_destination"}:
            raise ContractError(f"unknown graph context role {role!r}")
        return {"role": role, "packet": self.payload(), "packet_digest": self.packet_digest}

    def to_mapping(self) -> dict[str, Any]:
        """Return the native-host metadata form.

        ``NativeHost`` intentionally keeps metadata as a plain mapping when it
        rebuilds an ``ObservationFrame``.  The packet digest is therefore
        included explicitly so FastPolicy can revalidate the mapping at the
        scored boundary.
        """

        return {**self.payload(), "packet_digest": self.packet_digest}

    def assert_frame(self, frame: ObservationFrame) -> None:
        if frame.digest != self.observation_digest or frame.timestep != self.observation_timestep:
            raise ContractError("graph context is stale for the supplied observation frame")
        if self.episode_id is not None and frame.episode_id != self.episode_id:
            raise ContractError("graph context belongs to a different episode")


def make_graph_context(
    frame: ObservationFrame,
    triplet: Any,
    arrow_geometry: Any,
    *,
    graph_revision: str,
    arrow_revision: str,
    phase: str = "unknown",
) -> GraphContextPacket:
    """Build a packet bound to the exact frame supplied by the environment."""

    state = None
    try:
        state = state8(frame.observation)
    except ContractError:
        # Images-only observations are valid for a VLA.  The packet remains
        # bound by the full observation digest and simply omits state features.
        state = None
    return GraphContextPacket(
        triplet=triplet,
        arrow_geometry=arrow_geometry,
        observation_digest=frame.digest,
        observation_state=state,
        observation_timestep=frame.timestep,
        episode_id=frame.episode_id,
        graph_revision=graph_revision,
        arrow_revision=arrow_revision,
        phase=phase,
    )


def packet_digest(packet: GraphContextPacket) -> str:
    """Explicit helper for callers recording context provenance."""

    if not isinstance(packet, GraphContextPacket):
        raise ContractError("packet_digest requires a GraphContextPacket")
    return packet.packet_digest


def validate_context_mapping(value: Mapping[str, Any], frame: ObservationFrame | None = None) -> Mapping[str, Any]:
    """Validate a packet mapping after a native host metadata round-trip."""

    if not isinstance(value, Mapping):
        raise ContractError("graph context must be a GraphContextPacket or mapping")
    required = {"schema", "triplet", "arrow_geometry", "observation_digest", "graph_revision",
                "arrow_revision", "observation_timestep", "phase", "packet_digest"}
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"graph context mapping missing fields: {missing}")
    payload = {key: value[key] for key in value if key != "packet_digest"}
    if digest(payload) != value["packet_digest"]:
        raise ContractError("graph context mapping digest does not match contents")
    if frame is not None:
        if value["observation_digest"] != frame.digest or value["observation_timestep"] != frame.timestep:
            raise ContractError("graph context is stale for the supplied observation frame")
        if value.get("episode_id") is not None and value.get("episode_id") != frame.episode_id:
            raise ContractError("graph context belongs to a different episode")
    return value


__all__ = [
    "GRAPH_CONTEXT_SCHEMA",
    "GraphContextPacket",
    "make_graph_context",
    "packet_digest",
    "validate_context_mapping",
]
