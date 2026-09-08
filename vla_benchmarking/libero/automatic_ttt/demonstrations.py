"""Strict conversion of executed Arrow demonstrations and takeover traces.

This module does not invoke the controller.  It validates records produced by
the live environment view and creates an immutable receipt suitable for a
dataset manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    Actor,
    ContractError,
    SourceState,
    TransitionRecord,
    _json_safe,
    assert_student_observation,
    validate_action,
)


class DemonstrationValidationError(ContractError):
    """A trace was rejected and must not reach a training dataset."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__("invalid Arrow demonstration: " + "; ".join(self.reasons))


def _canonical(value: Any) -> Any:
    """Strictly canonicalize observations for exact boundary/hash checks."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _canonical({name: getattr(value, name) for name in value.__dataclass_fields__})
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DemonstrationValidationError(("observation contains a non-finite float",))
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "hex": bytes(value).hex()}
    tolist = getattr(value, "tolist", None)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if callable(tolist) and shape is not None and dtype is not None:
        array_value = value
        detach = getattr(array_value, "detach", None)
        if callable(detach):
            array_value = detach()
        cpu = getattr(array_value, "cpu", None)
        if callable(cpu):
            array_value = cpu()
        return {
            "type": f"array:{type(value).__module__}.{type(value).__qualname__}",
            "dtype": str(dtype),
            "shape": [int(dimension) for dimension in tuple(shape)],
            "data": _canonical(array_value.tolist()),
        }
    raise DemonstrationValidationError((f"unsupported observation value {type(value).__name__}",))


def _same_observation(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical(left) == _canonical(right)


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DemonstrationRecord:
    episode_id: str
    task_id: int | str
    seed: int
    environment_identity: str
    transitions: tuple[TransitionRecord, ...]
    teacher_success: bool
    evaluator_success: bool
    source_controller: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.environment_identity or not self.source_controller:
            raise DemonstrationValidationError(("demonstration identity is incomplete",))
        _json_safe(self.provenance)


@dataclass(frozen=True)
class DemonstrationReceipt:
    episode_id: str
    task_id: int | str
    seed: int
    environment_identity: str
    source_controller: str
    vla_transition_count: int
    teacher_transition_count: int
    accepted_correction_chunks: tuple[Mapping[str, Any], ...]
    transitions_sha256: str
    observations_sha256: str
    teacher_success: bool
    evaluator_success: bool
    rejected_reasons: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    collection_mode: str = "same_episode_takeover"

    def __post_init__(self) -> None:
        if self.rejected_reasons:
            raise DemonstrationValidationError(self.rejected_reasons)
        if self.teacher_transition_count <= 0:
            raise DemonstrationValidationError(("demonstration requires teacher transitions",))
        if self.collection_mode not in {"same_episode_takeover", "fresh_arrow"}:
            raise DemonstrationValidationError(("unknown demonstration collection mode",))
        if self.collection_mode == "same_episode_takeover" and self.vla_transition_count <= 0:
            raise DemonstrationValidationError(("takeover demonstration requires VLA transitions",))
        if self.collection_mode == "fresh_arrow" and self.vla_transition_count != 0:
            raise DemonstrationValidationError(("fresh Arrow demonstration cannot contain VLA transitions",))
        if len(self.transitions_sha256) != 64 or len(self.observations_sha256) != 64:
            raise DemonstrationValidationError(("demonstration hashes must be SHA-256",))
        _json_safe(self.provenance)

    def to_json(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass(frozen=True)
class ValidatedDemonstration:
    record: DemonstrationRecord
    receipt: DemonstrationReceipt

    def __iter__(self):
        yield self.record
        yield self.receipt


def validate_and_build_demonstration(
    records: Sequence[TransitionRecord],
    *,
    task_id: int | str,
    seed: int,
    environment_identity: str,
    teacher_success: bool,
    evaluator_success: bool,
    source_controller: str = "arrow_grasp_controller",
    provenance: Mapping[str, Any] | None = None,
    collection_mode: str = "same_episode_takeover",
) -> ValidatedDemonstration:
    """Validate one executed trace for takeover or fresh Arrow collection."""

    reasons: list[str] = []
    resolved_provenance = dict(provenance or {})
    rows = tuple(records)
    if not rows:
        reasons.append("trace is empty")
    episode_id = rows[0].episode_id if rows else ""
    if rows and any(row.episode_id != episode_id for row in rows):
        reasons.append("transitions belong to multiple episodes")
    if rows and [row.timestep for row in rows] != list(range(len(rows))):
        reasons.append("timesteps are not contiguous from zero")
    if not environment_identity:
        reasons.append("environment_identity is required")
    if not source_controller:
        reasons.append("source_controller is required")
    if not isinstance(evaluator_success, bool) or not isinstance(teacher_success, bool):
        reasons.append("teacher_success and evaluator_success must be boolean")

    if collection_mode not in {"same_episode_takeover", "fresh_arrow"}:
        reasons.append("collection_mode must be same_episode_takeover or fresh_arrow")
    first_teacher = next((index for index, row in enumerate(rows) if row.actor is Actor.TEACHER), None)
    if first_teacher is None:
        reasons.append("trace has no teacher correction")
        first_teacher = len(rows)
    if collection_mode == "same_episode_takeover":
        if first_teacher == 0:
            reasons.append("trace must begin with at least one VLA transition")
        if first_teacher == len(rows) and rows:
            reasons.append("trace has no teacher correction suffix")
        if any(row.actor is Actor.VLA for row in rows[first_teacher:]):
            reasons.append("VLA transitions cannot follow teacher takeover")
    else:
        if first_teacher != 0:
            reasons.append("fresh Arrow trace must contain only Arrow transitions")
        if any(row.actor is not Actor.TEACHER for row in rows):
            reasons.append("fresh Arrow trace must contain only teacher transitions")
        if any(row.actor is Actor.TEACHER and not row.training_eligible for row in rows):
            reasons.append("fresh Arrow transitions must be training eligible")

    teacher_rows = rows[first_teacher:]
    for index, row in enumerate(rows):
        row_metadata = getattr(row, "metadata", {})
        if not isinstance(row_metadata, Mapping):
            row_metadata = {}
        try:
            assert_student_observation(row.observation)
            assert_student_observation(row.next_observation)
            validate_action(row.action)
        except (ContractError, ValueError) as exc:
            reasons.append(f"timestep {index}: invalid student transition: {exc}")
        if index and not _same_observation(rows[index - 1].next_observation, row.observation):
            reasons.append(f"timestep {index}: observation boundary does not match prior next_observation")
        if row.done and index != len(rows) - 1:
            reasons.append(f"timestep {index}: transition follows a terminal transition")
        if row_metadata.get("captured") is False or row_metadata.get("capture_source") == "synthetic":
            reasons.append(f"timestep {index}: transition is marked non-captured/fabricated")
        for field_name, expected in (("task_id", task_id), ("seed", seed), ("environment_identity", environment_identity)):
            if field_name in row_metadata and row_metadata[field_name] != expected:
                reasons.append(f"timestep {index}: {field_name} identity mismatch")
    for index, row in enumerate(teacher_rows):
        if row.actor is not Actor.TEACHER:
            reasons.append(f"timestep {first_teacher + index}: non-teacher row in correction suffix")
        if not row.training_eligible:
            reasons.append(f"timestep {first_teacher + index}: teacher row is not training eligible")
        if row.source_state not in {SourceState.SOURCE_UNHELD, SourceState.SOURCE_HELD}:
            reasons.append(f"timestep {first_teacher + index}: invalid teacher source state")
    if collection_mode == "same_episode_takeover" and first_teacher and teacher_rows and not _same_observation(rows[first_teacher - 1].next_observation, teacher_rows[0].observation):
        reasons.append("first teacher observation does not equal final VLA next_observation")
    if teacher_success:
        if not teacher_rows:
            reasons.append("successful teacher outcome requires executed teacher transitions")
        elif not teacher_rows[-1].success and resolved_provenance.get("evaluator_phase") != "post_retreat":
            reasons.append("successful teacher outcome requires final teacher transition success")
        if evaluator_success is not True:
            reasons.append("successful teacher outcome requires evaluator_success=True")
    elif evaluator_success is True and (not teacher_rows or not teacher_rows[-1].success) and resolved_provenance.get("evaluator_phase") != "post_retreat":
        # The Arrow evaluator is queried after retreat, while TransitionRecord
        # success is an annotation of an individual executed env.step.  A
        # positive evaluator verdict cannot rewrite that final row; retain the
        # trace only as a failed/cost record and force the caller to resolve the
        # discrepancy explicitly.
        reasons.append(
            "evaluator_success=True disagrees with final executed transition success=False; "
            "no fabricated transition success is allowed"
        )

    chunks: list[Mapping[str, Any]] = []
    for chunk_id in dict.fromkeys(row.action_chunk_id for row in teacher_rows):
        chunk_rows = [row for row in teacher_rows if row.action_chunk_id == chunk_id]
        horizon = chunk_rows[0].action_chunk_horizon
        indices = sorted(row.action_chunk_index for row in chunk_rows)
        if any(row.action_chunk_horizon != horizon for row in chunk_rows) or indices != list(range(horizon)):
            reasons.append(f"teacher action chunk {chunk_id!r} is incomplete or reordered")
        chunks.append({"chunk_id": chunk_id, "horizon": horizon, "executed": len(chunk_rows)})

    if reasons:
        raise DemonstrationValidationError(tuple(dict.fromkeys(reasons)))
    record = DemonstrationRecord(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        environment_identity=environment_identity,
        transitions=rows,
        teacher_success=teacher_success,
        evaluator_success=evaluator_success,
        source_controller=source_controller,
        provenance=resolved_provenance,
    )
    receipt = DemonstrationReceipt(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        environment_identity=environment_identity,
        source_controller=source_controller,
        vla_transition_count=first_teacher,
        teacher_transition_count=len(teacher_rows),
        accepted_correction_chunks=tuple(chunks),
        transitions_sha256=_digest(rows),
        observations_sha256=_digest([(row.observation, row.next_observation) for row in rows]),
        teacher_success=teacher_success,
        evaluator_success=evaluator_success,
        provenance=resolved_provenance,
        collection_mode=collection_mode,
    )
    return ValidatedDemonstration(record, receipt)


__all__ = [
    "DemonstrationRecord", "DemonstrationReceipt", "DemonstrationValidationError",
    "ValidatedDemonstration", "validate_and_build_demonstration",
]
