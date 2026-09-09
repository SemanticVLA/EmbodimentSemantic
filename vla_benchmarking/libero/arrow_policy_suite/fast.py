"""Graph-local disposable fast correction (the saved Arrow Fast plan)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ActionProposal, ContractError, ObservationFrame, PolicyDecision, StepRecord, clip_action, digest
from .graph_context import GraphContextPacket, validate_context_mapping


FEATURE_DIM = 32
FAST_CONNECTIONS = ("hand_to_source", "hand_to_destination")


def _outer_add(matrix: list[list[float]], error: Sequence[float], feature: Sequence[float], scale: float) -> None:
    for row in range(7):
        for col in range(FEATURE_DIM):
            matrix[row][col] += scale * float(error[row]) * float(feature[col])


@dataclass(frozen=True)
class FastMetadata:
    parameter_count: int = 448
    support_steps: int = 0
    adapted: bool = False
    fallback: bool = False


class FastCorrector(Protocol):
    """Shared runtime surface for fitted disposable Fast artifacts."""

    metadata: FastMetadata
    last_receipt: Any

    def correction(self, payload: Any) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class FastLifecycleReceipt:
    """Auditable accounting for one disposable Fast adaptation attempt."""

    t0_identity_sha256: str
    restored_t0: bool
    support_attempts: int
    support_steps: int
    support_complete: bool
    support_vla_calls: int
    support_teacher_calls: int
    support_context_calls: int
    scored_steps: int
    scored_vla_calls: int
    scored_context_calls: int
    scored_teacher_calls: int
    teacher_detached: bool
    teacher_destroyed: bool
    fast_slot_capacity: int
    changed_fast_slots: int
    non_fast_changed: int
    slow_manifest_sha256_before: str | None
    slow_manifest_sha256_after: str | None
    vla_manifest_sha256_before: str | None
    vla_manifest_sha256_after: str | None
    support_error: str | None = None


@dataclass(frozen=True)
class FastLifecycleResult:
    receipt: FastLifecycleReceipt
    decisions: tuple[PolicyDecision, ...] = ()


def _validate_proposal(value: Any, frame: ObservationFrame, *, name: str) -> ActionProposal:
    if not isinstance(value, ActionProposal):
        raise ContractError(f"{name} must return an ActionProposal")
    if value.timestep != frame.timestep or value.observation_digest != frame.digest:
        raise ContractError(f"{name} proposal is stale for this frame")
    return value


def _component_hash(component: Any, identity_fn: Callable[[Any], str] | None) -> str | None:
    if identity_fn is not None:
        value = identity_fn(component)
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError("identity_fn must return a SHA-256 string")
        return value
    manifest = getattr(component, "manifest", None)
    if callable(manifest):
        value = manifest()
        return digest(value)
    return None


def _state_digest(value: Any) -> str:
    """Hash JSON-safe snapshots, with an opaque-state fallback.

    NativeHost snapshots contain RNG tensors and dataclass instances that are
    intentionally opaque to the policy contracts.  Pickle is only used for
    the local before/after identity comparison; it is never sent to a model
    or persisted as an experiment artifact.
    """

    try:
        return digest(value)
    except ContractError:
        try:
            return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            raise ContractError("Fast t0 snapshot is not hashable") from exc


def _close_teacher(teacher: Any) -> tuple[bool, bool]:
    """Detach/destroy exactly once; tolerate a teacher exposing either hook."""

    detached = False
    destroyed = False
    detach = getattr(teacher, "detach", None)
    if callable(detach):
        detach()
        detached = True
    close = getattr(teacher, "destroy", None)
    if not callable(close):
        close = getattr(teacher, "close", None)
    if callable(close):
        close()
        destroyed = True
    return detached, destroyed


class FastOneObservationSession:
    """Run one support attempt and then a teacher-free scored query.

    The host supplies a transactional snapshot/restore covering the
    environment and VLA.  Frames are captured from one reset by the host; this
    class never retries a failed support proposal and never calls the teacher
    after the one support attempt.  It is intentionally VLA-agnostic: only the
    ``propose``/``snapshot``/``restore`` protocol is required.
    """

    def __init__(
        self,
        corrector: Any,
        *,
        snapshot_fn: Callable[[], Any],
        restore_fn: Callable[[Any], None],
        graph_context_fn: Callable[[ObservationFrame], Any],
        vla: Any,
        teacher: Any,
        vla_identity_fn: Callable[[Any], str] | None = None,
        native_mode: bool = True,
    ) -> None:
        if not callable(snapshot_fn) or not callable(restore_fn):
            raise ContractError("Fast session requires transactional snapshot/restore hooks")
        if not callable(graph_context_fn):
            raise ContractError("Fast session requires a graph context provider")
        if not callable(getattr(vla, "propose", None)) or not callable(getattr(teacher, "propose", None)):
            raise ContractError("Fast session requires VLA and teacher propose hooks")
        if native_mode and vla_identity_fn is None and not callable(getattr(vla, "manifest", None)):
            raise ContractError("native Fast session requires a VLA identity hash")
        self.corrector = corrector
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.graph_context_fn = graph_context_fn
        self.vla = vla
        self.teacher = teacher
        self.vla_identity_fn = vla_identity_fn
        self.native_mode = bool(native_mode)
        self._attempt_consumed = False

    def run(
        self,
        support_frames: Sequence[ObservationFrame],
        scored_frames: Sequence[ObservationFrame],
        *,
        support_success: bool,
        source_manifest_sha256: str | None = None,
        training_config: Any | None = None,
    ) -> FastLifecycleResult:
        if self._attempt_consumed:
            raise ContractError("Fast support attempt is one-shot; create a new session for retry")
        self._attempt_consumed = True
        initial = self.snapshot_fn()
        initial_digest = _state_digest(initial)
        vla_before = _component_hash(self.vla, self.vla_identity_fn)
        slow_before = getattr(self.corrector, "slow_manifest_sha256", lambda: None)()
        support_rows: list[Any] = []
        context_calls = vla_calls = teacher_calls = 0
        support_vla_calls = support_teacher_calls = support_context_calls = 0
        scored_vla_calls = scored_context_calls = 0
        support_complete = True
        support_error: str | None = None
        try:
            from .fast_training import FastSupportExample, train_fast_support

            for index, frame in enumerate(tuple(support_frames)):
                context = self.graph_context_fn(frame)
                context_calls += 1
                support_context_calls += 1
                if isinstance(context, GraphContextPacket):
                    context.assert_frame(frame)
                elif isinstance(context, Mapping) and "packet_digest" in context:
                    validate_context_mapping(context, frame)
                base = _validate_proposal(self.vla.propose(frame), frame, name="VLA")
                vla_calls += 1
                support_vla_calls += 1
                teacher = _validate_proposal(self.teacher.propose(frame), frame, name="teacher")
                teacher_calls += 1
                support_teacher_calls += 1
                support_rows.append(
                    FastSupportExample(context, base.action, teacher.action,
                                       episode_id=frame.episode_id or "support", timestep=frame.timestep)
                )
        except BaseException as exc:
            support_complete = False
            support_error = f"{type(exc).__name__}: {exc}"
        # The one failed/incomplete attempt is still consumed.  There is no
        # support retry, and fit_support records an explicit fallback.
        fit_error: BaseException | None = None
        restore_error: BaseException | None = None
        try:
            from .fast_training import train_fast_support
            train_fast_support(
                self.corrector, support_rows, success=bool(support_success and support_complete),
                config=training_config, source_manifest_sha256=source_manifest_sha256,
            )
        except BaseException as exc:
            fit_error = exc
        finally:
            try:
                self.restore_fn(initial)
            except BaseException as exc:
                restore_error = exc
        restored = False
        if restore_error is None:
            try:
                restored = _state_digest(self.snapshot_fn()) == initial_digest
            except BaseException as exc:
                restore_error = exc
        detached, destroyed = _close_teacher(self.teacher)
        if fit_error is not None:
            raise fit_error
        if restore_error is not None:
            raise ContractError("Fast session could not restore exact t0 state") from restore_error
        decisions: list[PolicyDecision] = []
        policy = FastPolicy(self.corrector, graph_fn=lambda frame: self.graph_context_fn(frame))
        for frame in tuple(scored_frames):
            context = self.graph_context_fn(frame)
            context_calls += 1
            scored_context_calls += 1
            if isinstance(context, GraphContextPacket):
                context.assert_frame(frame)
            elif isinstance(context, Mapping) and "packet_digest" in context:
                validate_context_mapping(context, frame)
            base = _validate_proposal(self.vla.propose(frame), frame, name="VLA")
            vla_calls += 1
            scored_vla_calls += 1
            decisions.append(policy.decide(frame, base, None))
        vla_after = _component_hash(self.vla, self.vla_identity_fn)
        slow_after = getattr(self.corrector, "slow_manifest_sha256", lambda: None)()
        metadata = getattr(self.corrector, "metadata", None)
        receipt = FastLifecycleReceipt(
            t0_identity_sha256=initial_digest, restored_t0=restored,
            support_attempts=1, support_steps=len(support_rows), support_complete=support_complete,
            support_vla_calls=support_vla_calls, support_teacher_calls=support_teacher_calls,
            support_context_calls=support_context_calls, scored_steps=len(decisions),
            scored_vla_calls=scored_vla_calls, scored_context_calls=scored_context_calls,
            scored_teacher_calls=0,
            teacher_detached=detached, teacher_destroyed=destroyed,
            fast_slot_capacity=int(getattr(metadata, "parameter_count", 448)),
            changed_fast_slots=int(getattr(getattr(self.corrector, "last_receipt", None), "changed_fast_slots", 0)),
            non_fast_changed=int(getattr(getattr(self.corrector, "last_receipt", None), "non_fast_changed", 0)),
            slow_manifest_sha256_before=slow_before, slow_manifest_sha256_after=slow_after,
            vla_manifest_sha256_before=vla_before, vla_manifest_sha256_after=vla_after,
            support_error=support_error,
        )
        return FastLifecycleResult(receipt, tuple(decisions))


class GraphFastCorrector:
    """Two role-specific 7x32 matrices, updated by one chronological demo."""

    def __init__(self, feature_fn: Callable[[ObservationFrame, str], Sequence[float]],
                 router_fn: Callable[[ObservationFrame, str], float] | None = None) -> None:
        self.feature_fn = feature_fn
        self.router_fn = router_fn or (lambda _frame, role: 1.0 if role == "hand_to_source" else 0.0)
        self.weights = {role: [[0.0] * FEATURE_DIM for _ in range(7)] for role in FAST_CONNECTIONS}
        self.metadata = FastMetadata()
        self.last_receipt: FastMetadata | None = None

    def reset(self) -> None:
        for matrix in self.weights.values():
            for row in matrix:
                for col in range(FEATURE_DIM):
                    row[col] = 0.0
        self.metadata = FastMetadata()
        self.last_receipt = None

    def _feature(self, frame: ObservationFrame, role: str) -> tuple[float, ...]:
        feature = tuple(float(v) for v in self.feature_fn(frame, role))
        if len(feature) != FEATURE_DIM:
            raise ContractError(f"{role} feature must have dimension {FEATURE_DIM}")
        return feature

    def correction(self, frame: ObservationFrame) -> tuple[float, ...]:
        output = [0.0] * 7
        for role, matrix in self.weights.items():
            feature = self._feature(frame, role)
            scale = float(self.router_fn(frame, role))
            for row in range(7):
                output[row] += scale * sum(matrix[row][col] * feature[col] for col in range(FEATURE_DIM))
        return tuple(output)

    def fit(self, support: Sequence[tuple[ObservationFrame, Sequence[float], Sequence[float]]], *, success: bool) -> FastMetadata:
        self.reset()
        if not success or not support:
            self.metadata = FastMetadata(support_steps=len(support), fallback=True)
            self.last_receipt = self.metadata
            return self.metadata
        for frame, base_action, teacher_action in support:
            base = tuple(float(v) for v in base_action)
            target = tuple(float(v) for v in teacher_action)
            correction = self.correction(frame)
            error = tuple(target[i] - base[i] - correction[i] for i in range(7))
            active: list[tuple[str, tuple[float, ...], float]] = []
            for role in FAST_CONNECTIONS:
                feature = self._feature(frame, role)
                gate = float(self.router_fn(frame, role))
                active.append((role, feature, gate))
            denom = sum(g * g * sum(x * x for x in feature) for _, feature, g in active) + 1e-6
            for role, feature, gate in active:
                _outer_add(self.weights[role], error, feature, 0.5 * gate / denom)
        self.metadata = FastMetadata(support_steps=len(support), adapted=True)
        self.last_receipt = self.metadata
        return self.metadata


class FastPolicy:
    policy_id = "arrow_fast"

    def __init__(self, corrector: FastCorrector, *, graph_fn: Callable[[ObservationFrame], object] | None = None) -> None:
        self.corrector = corrector
        self.graph_fn = graph_fn

    def reset(self) -> None:
        # Fast state is intentionally supplied by the caller before evaluation;
        # reset here only clears episode-local bookkeeping, not fitted weights.
        return None

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        if teacher is not None:
            # ``arrow_fast`` is a scored, teacher-free policy.  Rejecting an
            # accidentally supplied proposal makes a teacher call visible at
            # the boundary instead of silently turning this into On-Call.
            raise ContractError("arrow_fast scored decisions must be teacher-free; a teacher proposal was supplied")
        # ``GraphFastCorrector`` consumes the frame directly.  The slow/fast
        # training facade in ``fast_training.py`` consumes a graph payload;
        # accepting an explicit graph_fn keeps both implementations on the
        # same runtime policy boundary without leaking graph construction into
        # the coordinator.
        payload = self.graph_fn(frame) if self.graph_fn is not None else frame.metadata.get("graph_context", frame)
        if isinstance(payload, GraphContextPacket):
            payload.assert_frame(frame)
        elif isinstance(payload, Mapping) and "packet_digest" in payload:
            validate_context_mapping(payload, frame)
        correction = self.corrector.correction(payload)
        action = clip_action(tuple(base.action[i] + correction[i] for i in range(7)))
        context_digest = payload.packet_digest if isinstance(payload, GraphContextPacket) else None
        return PolicyDecision(action, self.policy_id, frame.digest,
                              metadata={"fast_parameter_count": 448,
                                        "fast_adapted": self.corrector.metadata.adapted,
                                        "fast_fallback": self.corrector.metadata.fallback,
                                        "fast_receipt": getattr(self.corrector, "last_receipt", None) is not None,
                                        "graph_context_digest": context_digest,
                                        "teacher_calls": 0})

    def commit(self, record: StepRecord) -> None:
        return None
