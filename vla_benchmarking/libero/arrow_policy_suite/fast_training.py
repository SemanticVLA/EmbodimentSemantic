"""Graph Fast slow-encoder/router and exact fast-weight update contract.

The runtime ``fast.py`` module owns the existing policy adapter.  This module
adds a dependency-free training/support-update contract around the same graph
roles and parameterization: two role-specific 7x32 matrices, exactly 448 fast
parameters.  A native torch encoder can be injected, but no native dependency
is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ACTION_DIM, ContractError, _safe, clip_action
from .fast import FastCorrector, FastMetadata
from .graph_context import GraphContextPacket, validate_context_mapping


FAST_TRAINING_SCHEMA = "arrow_policy_suite.fast_training.v1"
FAST_FEATURE_DIM = 32
FAST_ROLES = ("hand_to_source", "hand_to_destination")
FAST_PARAMETER_COUNT = len(FAST_ROLES) * ACTION_DIM * FAST_FEATURE_DIM


def _manifest_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenEncoderArtifact:
    """Versioned frozen slow encoder body.

    The callable is supplied by the native host (for example a checkpointed
    graph encoder).  The revision and hash are mandatory provenance rather
    than cosmetic labels: native mode refuses an unversioned encoder.
    """

    encode_fn: GraphEncoderFn
    revision: str
    artifact_sha256: str
    kind: str = "learned_artifact"
    output_dim: int = FAST_FEATURE_DIM

    def __post_init__(self) -> None:
        if not callable(self.encode_fn) or not self.revision.strip():
            raise ContractError("frozen encoder requires callable body and revision")
        if (len(self.artifact_sha256) != 64 or self.artifact_sha256 != self.artifact_sha256.lower()
                or any(c not in "0123456789abcdef" for c in self.artifact_sha256)):
            raise ContractError("frozen encoder artifact_sha256 must be lowercase SHA-256")
        if self.output_dim != FAST_FEATURE_DIM or self.kind not in {"learned_artifact", "deterministic"}:
            raise ContractError("unsupported frozen encoder artifact")

    def manifest(self) -> dict[str, Any]:
        return {
            "name": "graph_slow_encoder",
            "implementation": self.kind,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "output_dim": self.output_dim,
        }


@dataclass(frozen=True)
class FrozenRouterArtifact:
    """Versioned frozen phase router body."""

    route_fn: GraphRouterFn
    revision: str
    artifact_sha256: str
    kind: str = "learned_artifact"

    def __post_init__(self) -> None:
        if not callable(self.route_fn) or not self.revision.strip():
            raise ContractError("frozen router requires callable body and revision")
        if (len(self.artifact_sha256) != 64 or self.artifact_sha256 != self.artifact_sha256.lower()
                or any(c not in "0123456789abcdef" for c in self.artifact_sha256)):
            raise ContractError("frozen router artifact_sha256 must be lowercase SHA-256")
        if self.kind not in {"learned_artifact", "deterministic"}:
            raise ContractError("unsupported frozen router artifact")

    def manifest(self) -> dict[str, Any]:
        return {
            "name": "graph_router",
            "implementation": self.kind,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
        }


def fast_parameter_count() -> int:
    """Return the immutable fast-state size (2 roles x 7 actions x 32 features)."""

    return FAST_PARAMETER_COUNT


def _finite_vector(value: Sequence[float], *, name: str, width: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must be a numeric sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a numeric sequence") from exc
    if width is not None and len(result) != width:
        raise ContractError(f"{name} must have width {width}, got {len(result)}")
    if any(not math.isfinite(item) for item in result):
        raise ContractError(f"{name} must contain finite values")
    return result


class GraphEncoderFn(Protocol):
    def __call__(self, graph: Any, role: str) -> Sequence[float]: ...


class GraphRouterFn(Protocol):
    def __call__(self, graph: Any, role: str) -> float: ...


def _digest_features(graph: Any, role: str) -> tuple[float, ...]:
    """Deterministic, dependency-free frozen encoder for canaries.

    Native learned encoders should be supplied as ``FrozenEncoderArtifact``.
    This path is intentionally explicit and revisioned; it is not the old
    shape-repeat fallback.  Numeric values in a graph packet are retained in
    stable order and SHA-256 bytes provide a fixed-width completion signal.
    """

    if isinstance(graph, GraphContextPacket):
        payload: Any = graph.for_role(role)
    else:
        payload = {"role": role, "graph": _safe(graph)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    values: list[float] = []
    counter = 0
    while len(values) < FAST_FEATURE_DIM:
        block = hashlib.sha256(encoded + counter.to_bytes(4, "big")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in block)
        counter += 1
    return tuple(values[:FAST_FEATURE_DIM])


def _deterministic_encoder_artifact() -> FrozenEncoderArtifact:
    revision = "deterministic-digest-v1"
    artifact_sha = hashlib.sha256(revision.encode("utf-8")).hexdigest()
    return FrozenEncoderArtifact(_digest_features, revision, artifact_sha, kind="deterministic")


def _deterministic_router(graph: Any, role: str) -> float:
    phase = str(getattr(graph, "phase", "") or (graph.get("phase", "") if isinstance(graph, Mapping) else "")).lower()
    if role == "hand_to_source":
        return 1.0 if phase in {"", "unknown", "approach_source", "grasp", "source"} else 0.0
    return 1.0 if phase in {"transfer", "approach_destination", "release", "destination"} else 0.0


def _deterministic_router_artifact() -> FrozenRouterArtifact:
    revision = "deterministic-phase-router-v1"
    artifact_sha = hashlib.sha256(revision.encode("utf-8")).hexdigest()
    return FrozenRouterArtifact(_deterministic_router, revision, artifact_sha, kind="deterministic")


def make_deterministic_encoder_artifact(*, revision: str, artifact_sha256: str) -> FrozenEncoderArtifact:
    """Construct the explicitly selected deterministic slow encoder artifact."""

    return FrozenEncoderArtifact(_digest_features, revision, artifact_sha256, kind="deterministic")


def make_deterministic_router_artifact(*, revision: str, artifact_sha256: str) -> FrozenRouterArtifact:
    """Construct the explicitly selected deterministic phase-router artifact."""

    return FrozenRouterArtifact(_deterministic_router, revision, artifact_sha256, kind="deterministic")


class GraphSlowEncoder:
    """A fixed-width slow graph representation with an injectable native body."""

    def __init__(self, *, input_dim: int = FAST_FEATURE_DIM, output_dim: int = FAST_FEATURE_DIM,
                 encode_fn: GraphEncoderFn | None = None,
                 artifact: FrozenEncoderArtifact | None = None,
                 native_mode: bool = False) -> None:
        if input_dim <= 0 or output_dim != FAST_FEATURE_DIM:
            raise ContractError("GraphSlowEncoder output_dim must be exactly 32")
        if encode_fn is not None and artifact is not None:
            raise ContractError("provide either encode_fn or artifact, not both")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.native_mode = bool(native_mode)
        self.artifact = artifact
        self.encode_fn = encode_fn or (artifact.encode_fn if artifact is not None else None)
        if self.native_mode and artifact is None:
            raise ContractError("native GraphSlowEncoder requires a frozen revisioned artifact")

    def encode(self, graph: Any, role: str) -> tuple[float, ...]:
        if role not in FAST_ROLES:
            raise ContractError(f"unknown fast graph role {role!r}")
        if self.encode_fn is not None:
            return _finite_vector(self.encode_fn(graph, role), name=f"encoded[{role}]", width=self.output_dim)
        if self.native_mode:
            raise ContractError("native GraphSlowEncoder has no frozen encoding body")
        value = graph
        if isinstance(graph, Mapping):
            if role in graph:
                value = graph[role]
            elif "graph_features" in graph:
                value = graph["graph_features"]
            elif "features" in graph:
                value = graph["features"]
            else:
                raise ContractError("graph mapping requires role, graph_features, or features")
        raw = _finite_vector(value, name="graph_features", width=self.input_dim)
        if self.input_dim == self.output_dim:
            return raw
        # Keep compatibility for the dependency-light legacy tests.  Native
        # mode fails above and can never use this shape-only behavior.
        return tuple(raw[index % self.input_dim] for index in range(self.output_dim))

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "name": "graph_slow_encoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "implementation": "injected" if self.encode_fn is not None else "shape_safe_repeat",
            "native_mode": self.native_mode,
        }
        if self.artifact is not None:
            manifest.update(self.artifact.manifest())
        return manifest


class GraphRouter:
    """Role gate contract separating graph encoding from fast-state routing."""

    def __init__(self, route_fn: GraphRouterFn | None = None,
                 artifact: FrozenRouterArtifact | None = None,
                 native_mode: bool = False) -> None:
        if route_fn is not None and artifact is not None:
            raise ContractError("provide either route_fn or artifact, not both")
        self.native_mode = bool(native_mode)
        self.artifact = artifact
        self.route_fn = route_fn or (artifact.route_fn if artifact is not None else None)
        if self.native_mode and artifact is None:
            raise ContractError("native GraphRouter requires a frozen revisioned artifact")

    def gate(self, graph: Any, role: str) -> float:
        if role not in FAST_ROLES:
            raise ContractError(f"unknown fast graph role {role!r}")
        if self.route_fn is None:
            if self.native_mode:
                raise ContractError("native GraphRouter has no frozen routing body")
            if isinstance(graph, Mapping) and isinstance(graph.get("router"), Mapping):
                value = graph["router"].get(role, 0.0)
            else:
                value = 1.0 if role == FAST_ROLES[0] else 0.0
        else:
            value = self.route_fn(graph, role)
        try:
            gate = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"router gate for {role} must be numeric") from exc
        if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
            raise ContractError(f"router gate for {role} must lie in [0, 1]")
        return gate

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "name": "graph_router",
            "implementation": "injected" if self.route_fn is not None else "role_default",
            "native_mode": self.native_mode,
        }
        if self.artifact is not None:
            manifest.update(self.artifact.manifest())
        return manifest


@dataclass(frozen=True)
class FastSupportExample:
    graph: Any
    base_action: tuple[float, ...]
    teacher_action: tuple[float, ...]
    episode_id: str = "support"
    timestep: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_action", clip_action(self.base_action))
        object.__setattr__(self, "teacher_action", clip_action(self.teacher_action))
        if not self.episode_id or self.timestep < 0:
            raise ContractError("Fast support examples require episode id and non-negative timestep")

    @property
    def target_residual(self) -> tuple[float, ...]:
        return tuple(self.teacher_action[i] - self.base_action[i] for i in range(ACTION_DIM))

    @property
    def context_digest(self) -> str | None:
        return self.graph.packet_digest if isinstance(self.graph, GraphContextPacket) else None


@dataclass(frozen=True)
class FastTrainingConfig:
    update_rate: float = 0.5
    seed: int = 1000
    require_success: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.update_rate) <= 1.0 or self.seed < 0:
            raise ContractError("Fast update_rate must lie in (0, 1] and seed must be non-negative")


@dataclass(frozen=True)
class FastTrainingReceipt:
    parameter_count: int
    support_steps: int
    adapted: bool
    fallback: bool
    encoder: Mapping[str, Any]
    router: Mapping[str, Any]
    source_manifest_sha256: str | None
    support_sha256: str
    fast_slot_capacity: int = FAST_PARAMETER_COUNT
    changed_fast_slots: int = 0
    non_fast_changed: int = 0
    slow_manifest_sha256_before: str | None = None
    slow_manifest_sha256_after: str | None = None
    vla_manifest_sha256: str | None = None
    context_digests: tuple[str, ...] = ()


class GraphFastModel:
    """Two 7x32 role matrices updated by one chronological support trace."""

    def __init__(self, *, encoder: GraphSlowEncoder | None = None, router: GraphRouter | None = None,
                 graph_fn: Callable[[Any], Any] | None = None,
                 native_mode: bool = False,
                 vla_manifest_sha256: str | None = None) -> None:
        if native_mode and encoder is None:
            encoder = GraphSlowEncoder(artifact=_deterministic_encoder_artifact(), native_mode=True)
        if native_mode and router is None:
            router = GraphRouter(artifact=_deterministic_router_artifact(), native_mode=True)
        self.encoder = encoder or GraphSlowEncoder()
        self.router = router or GraphRouter()
        if native_mode and (not self.encoder.native_mode or not self.router.native_mode):
            raise ContractError("native GraphFastModel requires native frozen encoder and router")
        self.native_mode = bool(native_mode)
        self.vla_manifest_sha256 = vla_manifest_sha256
        self.graph_fn = graph_fn
        self.weights: dict[str, list[list[float]]] = {
            role: [[0.0] * FAST_FEATURE_DIM for _ in range(ACTION_DIM)] for role in FAST_ROLES
        }
        self.source_manifest_sha256: str | None = None
        self.last_receipt: FastTrainingReceipt | None = None
        self.metadata = FastMetadata()
        self._fit_consumed = False
        self._slow_manifest_before: str | None = None

    def parameter_count(self) -> int:
        return sum(len(row) for matrix in self.weights.values() for row in matrix)

    def _feature(self, graph: Any, role: str) -> tuple[float, ...]:
        if self.native_mode:
            if isinstance(graph, GraphContextPacket):
                pass
            elif isinstance(graph, Mapping) and "packet_digest" in graph:
                validate_context_mapping(graph)
            else:
                raise ContractError("native Fast correction requires a digest-bound GraphContextPacket")
        return self.encoder.encode(graph, role)

    def slow_manifest(self) -> dict[str, Any]:
        return {"encoder": self.encoder.manifest(), "router": self.router.manifest()}

    def slow_manifest_sha256(self) -> str:
        return _manifest_digest(self.slow_manifest())

    def correction(self, graph: Any) -> tuple[float, ...]:
        # ``graph_fn`` is a convenience for direct FastPolicy use.  The policy
        # factory normally applies its own graph_fn before calling correction;
        # only an ObservationFrame-like object is converted here, preventing
        # fitted graph payloads from being transformed twice.
        if self.graph_fn is not None and hasattr(graph, "observation"):
            graph = self.graph_fn(graph)
        output = [0.0] * ACTION_DIM
        for role in FAST_ROLES:
            feature = self._feature(graph, role)
            gate = self.router.gate(graph, role)
            for action_index in range(ACTION_DIM):
                output[action_index] += gate * sum(
                    self.weights[role][action_index][column] * feature[column]
                    for column in range(FAST_FEATURE_DIM)
                )
        return tuple(output)

    def reset(self) -> None:
        for matrix in self.weights.values():
            for row in matrix:
                for index in range(FAST_FEATURE_DIM):
                    row[index] = 0.0
        self.source_manifest_sha256 = None
        self.last_receipt = None
        self.metadata = FastMetadata()

    def fit_support(
        self,
        support: Sequence[FastSupportExample],
        *,
        success: bool,
        config: FastTrainingConfig | None = None,
        source_manifest_sha256: str | None = None,
    ) -> FastTrainingReceipt:
        resolved = config or FastTrainingConfig()
        if self._fit_consumed:
            raise ContractError("Fast support update is one-shot; create a new model for a retry")
        if self.parameter_count() != FAST_PARAMETER_COUNT:
            raise ContractError("fast model must contain exactly 448 parameters")
        slow_before = self.slow_manifest_sha256()
        if self.native_mode and source_manifest_sha256 is None:
            raise ContractError("native Fast support requires source manifest hash")
        support_rows = tuple(support)
        if support_rows:
            episode_ids = {row.episode_id for row in support_rows}
            if len(episode_ids) != 1:
                raise ContractError("Fast support must contain exactly one chronological episode")
            timesteps = tuple(row.timestep for row in support_rows)
            if any(left >= right for left, right in zip(timesteps, timesteps[1:])):
                raise ContractError("Fast support timesteps must be strictly chronological with no retry")
        self._fit_consumed = True
        self.reset()
        if not success or not support_rows:
            receipt = _receipt(self, support_rows, adapted=False, fallback=True, source_manifest_sha256=source_manifest_sha256,
                               slow_before=slow_before, vla_manifest_sha256=self.vla_manifest_sha256)
            self.last_receipt = receipt
            self.metadata = FastMetadata(parameter_count=FAST_PARAMETER_COUNT,
                                         support_steps=len(support_rows), adapted=False, fallback=True)
            return receipt
        for example in support_rows:
            prediction = self.correction(example.graph)
            error = tuple(example.target_residual[index] - prediction[index] for index in range(ACTION_DIM))
            active: list[tuple[str, tuple[float, ...], float]] = []
            for role in FAST_ROLES:
                feature = self._feature(example.graph, role)
                gate = self.router.gate(example.graph, role)
                active.append((role, feature, gate))
            denom = sum(gate * gate * sum(item * item for item in feature) for _, feature, gate in active) + 1e-6
            for role, feature, gate in active:
                scale = resolved.update_rate * gate / denom
                matrix = self.weights[role]
                for action_index in range(ACTION_DIM):
                    for column in range(FAST_FEATURE_DIM):
                        matrix[action_index][column] += scale * error[action_index] * feature[column]
        self.source_manifest_sha256 = source_manifest_sha256
        slow_after = self.slow_manifest_sha256()
        if slow_before != slow_after:
            raise ContractError("slow encoder/router manifest changed during fast update")
        receipt = _receipt(self, support_rows, adapted=True, fallback=False, source_manifest_sha256=source_manifest_sha256,
                           slow_before=slow_before, vla_manifest_sha256=self.vla_manifest_sha256)
        self.last_receipt = receipt
        self.metadata = FastMetadata(parameter_count=FAST_PARAMETER_COUNT,
                                     support_steps=len(support_rows), adapted=True, fallback=False)
        return receipt


def _receipt(model: GraphFastModel, support: Sequence[FastSupportExample], *, adapted: bool,
             fallback: bool, source_manifest_sha256: str | None,
             slow_before: str | None = None,
             vla_manifest_sha256: str | None = None) -> FastTrainingReceipt:
    payload = []
    for row in support:
        graph_payload = row.graph.payload() if isinstance(row.graph, GraphContextPacket) else row.graph
        payload.append({
            "episode_id": row.episode_id, "timestep": row.timestep,
            "graph": graph_payload, "base_action": list(row.base_action),
            "teacher_action": list(row.teacher_action),
        })
    encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    changed = sum(
        1 for matrix in model.weights.values() for row in matrix for value in row if value != 0.0
    )
    context_digests = tuple(
        row.graph.packet_digest for row in support if isinstance(row.graph, GraphContextPacket)
    )
    return FastTrainingReceipt(
        parameter_count=model.parameter_count(), support_steps=len(support), adapted=adapted,
        fallback=fallback, encoder=model.encoder.manifest(), router=model.router.manifest(),
        source_manifest_sha256=source_manifest_sha256, support_sha256=hashlib.sha256(encoded).hexdigest(),
        fast_slot_capacity=FAST_PARAMETER_COUNT, changed_fast_slots=changed, non_fast_changed=0,
        slow_manifest_sha256_before=slow_before, slow_manifest_sha256_after=model.slow_manifest_sha256(),
        vla_manifest_sha256=vla_manifest_sha256, context_digests=context_digests,
    )


def train_fast_support(
    model: GraphFastModel,
    support: Sequence[FastSupportExample],
    *,
    success: bool,
    config: FastTrainingConfig | None = None,
    source_manifest_sha256: str | None = None,
) -> FastTrainingReceipt:
    """Explicit support-update entry point; no training starts at import time."""

    return model.fit_support(
        support, success=success, config=config, source_manifest_sha256=source_manifest_sha256
    )


SlowGraphEncoder = GraphSlowEncoder
GraphFastTrainer = GraphFastModel


__all__ = [
    "FAST_FEATURE_DIM", "FAST_PARAMETER_COUNT", "FAST_ROLES", "FAST_TRAINING_SCHEMA",
    "FastCorrector", "FastSupportExample", "FastTrainingConfig", "FastTrainingReceipt", "FrozenEncoderArtifact",
    "FrozenRouterArtifact", "GraphFastModel", "GraphFastTrainer", "GraphRouter", "GraphSlowEncoder",
    "SlowGraphEncoder", "fast_parameter_count", "make_deterministic_encoder_artifact",
    "make_deterministic_router_artifact", "train_fast_support",
]
