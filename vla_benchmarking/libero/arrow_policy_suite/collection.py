"""Dependency-light collection orchestration for the Arrow policy suite.

This module owns bookkeeping around an injected episode runner.  It does not
construct LIBERO, Arrow, LeRobot, or a trainer.  Every attempt is written to a
master log (including rejected attempts); only the explicit eligible view is
used to build learning/Trace derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .benchmark import EvaluationRow
from .config import StudyConfig
from .contracts import ContractError, StepRecord, _safe, state8
from .learning import state_only_routes
from .runtime import EpisodeResult


COLLECTION_SCHEMA = "arrow_policy_suite.collection.v1"
MASTER_LOG_SCHEMA = "arrow_policy_suite.master_log.v1"
TRACE_SCHEMA = "arrow_policy_suite.trace_view.v1"
COLLECTION_ARCHIVE_SCHEMA = "arrow_policy_suite.on_call_archive.v1"
TRAINING_SOURCE_SCHEMA = "arrow_policy_suite.training_source.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _call_hook(hook: Callable[[Mapping[str, Any]], Any] | None, payload: Mapping[str, Any]) -> None:
    if hook is not None:
        hook(payload)


def _proposal_payload(proposal: Any) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "policy_id": str(getattr(proposal, "policy_id", "unknown")),
        "action": list(getattr(proposal, "action", ())),
        "timestep": int(getattr(proposal, "timestep", getattr(proposal, "step", 0))),
        "observation_digest": getattr(proposal, "observation_digest", None),
        "metadata": dict(getattr(proposal, "metadata", {}) or {}),
        "provenance": dict(getattr(proposal, "provenance", {}) or {}),
    }


def _decision_payload(decision: Any) -> dict[str, Any]:
    metadata = dict(getattr(decision, "metadata", {}) or {})
    action = getattr(decision, "action", None)
    if action is None:
        action = getattr(getattr(decision, "proposal", None), "action", ())
    return {
        "policy_id": str(getattr(decision, "policy_id", getattr(getattr(decision, "proposal", None), "policy_id", "unknown"))),
        "action": list(action),
        "observation_digest": getattr(decision, "observation_digest", getattr(getattr(decision, "proposal", None), "observation_digest", None)),
        "teacher_used": bool(getattr(decision, "teacher_used", False) or metadata.get("teacher_used", False)),
        "teacher_groups": list(getattr(decision, "teacher_groups", metadata.get("teacher_groups", ())) or ()),
        "metadata": metadata,
        "provenance": dict(getattr(decision, "provenance", {}) or {}),
    }


def _outcome_payload(record: Any) -> dict[str, Any]:
    """Keep outcomes useful to trainers without serializing simulator internals."""
    raw = getattr(record, "result", getattr(record, "raw_result", None))
    values: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        for key in ("reward", "return", "score", "success", "task_success", "is_success",
                    "terminal", "terminated", "truncated", "done"):
            if key in raw:
                values[key] = raw[key]
    elif isinstance(raw, tuple):
        if len(raw) >= 2 and isinstance(raw[1], (int, float)):
            values["reward"] = raw[1]
        if len(raw) >= 3:
            values["terminal"] = bool(raw[2])
        if len(raw) >= 4:
            values["truncated"] = bool(raw[3])
    values.update({"success": bool(getattr(record, "success", False)),
                   "terminal": bool(getattr(record, "terminal", False))})
    return _safe(values)


@dataclass(frozen=True)
class AttemptIdentity:
    """Stable identity of one reset/episode used for collection and pairing."""

    task_id: int | str
    reset_id: str
    episode_id: str = ""
    seed: int | None = None
    reset_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_id in (None, "") or not self.reset_id:
            raise ContractError("attempt identity requires task_id and reset_id")
        if self.episode_id == "":
            object.__setattr__(self, "episode_id", f"task-{self.task_id}-reset-{self.reset_id}")
        if not self.episode_id:
            raise ContractError("attempt identity requires episode_id")
        if self.seed is not None:
            try:
                object.__setattr__(self, "seed", int(self.seed))
            except (TypeError, ValueError) as exc:
                raise ContractError("attempt seed must be an integer") from exc
        _safe(self.reset_identity)

    @property
    def key(self) -> tuple[str, str, str]:
        return str(self.task_id), self.reset_id, self.episode_id

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "reset_id": self.reset_id,
            "episode_id": self.episode_id,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.reset_identity:
            payload["reset_identity"] = dict(self.reset_identity)
        return payload


@dataclass(frozen=True)
class CollectionAttempt:
    """One attempted On-Call rollout and its eligibility decision."""

    identity: AttemptIdentity
    result: EpisodeResult
    eligible: bool
    eligibility_reasons: tuple[str, ...] = ()
    source: str = "on_call"

    def __post_init__(self) -> None:
        if self.source != "on_call":
            raise ContractError("collection attempts must be sourced from On-Call")
        if self.eligible and self.eligibility_reasons:
            raise ContractError("eligible attempts cannot carry rejection reasons")
        if not self.eligible and not self.eligibility_reasons:
            raise ContractError("ineligible attempts must explain why")

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return self.result.records

    def summary(self) -> dict[str, Any]:
        stats = self.result.stats
        records = tuple(self.result.records)
        return {
            **self.identity.as_dict(),
            "source": self.source,
            "eligible": self.eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "steps": stats.steps,
            "success": bool(stats.success),
            "terminal": bool(stats.terminal),
            "teacher_proposals": int(getattr(stats, "teacher_proposals", sum(record.teacher is not None for record in records))),
            "teacher_steps": int(getattr(stats, "teacher_steps", sum(
                bool(getattr(record.decision, "teacher_used", False) or
                     getattr(record.decision, "metadata", {}).get("teacher_used", False))
                for record in records))),
            "branch_steps": int(getattr(stats, "branch_steps", sum(
                int(getattr(record.decision, "metadata", {}).get("branch_steps", 0)) for record in records))),
            "metadata": dict(getattr(stats, "metadata", getattr(self.result, "metadata", {}))),
        }


@dataclass(frozen=True)
class CollectionManifest:
    """Immutable receipt for a complete fixed-size collection."""

    schema: str
    config_sha256: str
    master_log_sha256: str
    attempted: int
    eligible: int
    attempts_per_task: int
    task_counts: Mapping[str, Mapping[str, int]]
    eligible_episode_ids: tuple[str, ...]
    discarded_reasons: Mapping[str, int]
    parent_artifact: str = ""
    training_source_sha256: str = ""

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config_sha256": self.config_sha256,
            "master_log_sha256": self.master_log_sha256,
            "attempted": self.attempted,
            "eligible": self.eligible,
            "attempts_per_task": self.attempts_per_task,
            "task_counts": {str(k): dict(v) for k, v in self.task_counts.items()},
            "eligible_episode_ids": list(self.eligible_episode_ids),
            "discarded_reasons": dict(self.discarded_reasons),
            "parent_artifact": self.parent_artifact,
            "training_source_sha256": self.training_source_sha256,
        }

    @property
    def manifest_sha256(self) -> str:
        return _digest(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class CollectionArchive:
    """Immutable archive receipt retaining every fixed On-Call attempt.

    ``attempts`` is deliberately a complete serialized view, not only the
    successful subset.  Failed, truncated, and perception-unavailable
    attempts therefore remain available for the outcome-aware Apprentice and
    Editor views.
    """

    manifest: CollectionManifest
    attempts: tuple[Mapping[str, Any], ...]
    schema: str = COLLECTION_ARCHIVE_SCHEMA
    archive_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != COLLECTION_ARCHIVE_SCHEMA:
            raise ContractError("unsupported On-Call collection archive schema")
        if len(self.attempts) != self.manifest.attempted:
            raise ContractError("archive attempt count does not match collection manifest")
        for attempt in self.attempts:
            _safe(attempt)
        calculated = _digest({"schema": self.schema, "manifest": self.manifest.as_dict(),
                              "attempts": list(self.attempts)})
        if self.archive_sha256 and self.archive_sha256 != calculated:
            raise ContractError("collection archive digest does not match contents")
        object.__setattr__(self, "archive_sha256", calculated)

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "manifest": self.manifest.as_dict(),
                "attempts": list(self.attempts), "archive_sha256": self.archive_sha256}


def training_source_payload(
    record: Any,
    *,
    identity: AttemptIdentity | Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    eligible: bool | None = None,
    episode_success: bool | None = None,
    episode_complete: bool | None = None,
) -> dict[str, Any]:
    """Return one loadable, policy-facing training transition.

    This is intentionally narrower than the master log.  ``ObservationFrame``
    already exposes the canonical student observation, so simulator state and
    raw environment diagnostics never cross this artifact boundary.
    """
    frame = getattr(record, "frame", None)
    next_frame = getattr(record, "next_frame", None)
    if frame is None or next_frame is None or not hasattr(frame, "observation"):
        raise ContractError("training source records require before/after observation frames")
    observation = _safe(frame.observation)
    next_observation = _safe(next_frame.observation)
    # The frame contract is the final privilege boundary.  This also makes
    # malformed hand-built records fail before they are persisted.
    from .contracts import assert_student_observation
    assert_student_observation(observation, path="training_source.observation")
    assert_student_observation(next_observation, path="training_source.next_observation")
    decision = _decision_payload(getattr(record, "decision", None))
    teacher = _proposal_payload(getattr(record, "teacher", None))
    teacher_used = bool(decision["teacher_used"])
    decision_metadata = decision["metadata"]
    executed_by = str(getattr(record, "executed_by", ""))
    if not executed_by:
        executed_by = "arrow" if teacher_used else "vla"
    resolved_episode_success = bool(getattr(record, "success", False)) if episode_success is None else bool(episode_success)
    resolved_episode_complete = bool(getattr(record, "terminal", False)) if episode_complete is None else bool(episode_complete)
    minimal_branch: dict[str, Any] | None = None
    if decision["policy_id"] == "arrow_minimal" and decision_metadata.get("variant") == "runtime_oracle":
        selected_mask = decision_metadata.get("branch_selected_mask")
        try:
            valid_mask = not isinstance(selected_mask, bool) and 0 <= int(selected_mask) <= 7
        except (TypeError, ValueError):
            valid_mask = False
        branch_fallback = bool(decision_metadata.get("branch_fallback", True))
        evaluated = int(decision_metadata.get("branch_masks_evaluated", 0) or 0)
        reused = bool(decision_metadata.get("branch_reuse", False))
        # Initial decisions must carry the complete eight-mask/160-step
        # sandbox receipt. Subsequent real burst steps may reuse that receipt,
        # but never synthesize labels for a full-teacher fallback.
        real_label = valid_mask and not branch_fallback and (evaluated == 8 or reused)
        if real_label:
            minimal_branch = {
                "selected_hybrid_action": list(getattr(record, "action", decision["action"])),
                "selected_mask": int(selected_mask),
                "label_source": "minimal_branch_runtime",
                "branch_steps": int(decision_metadata.get("branch_steps", 0) or 0),
                "branch_cloned_steps": int(decision_metadata.get("branch_cloned_steps", 0) or 0),
                "branch_masks_evaluated": evaluated,
                "branch_reuse": reused,
            }
        if eligible is None:
            eligible = bool(real_label)
    if eligible is None:
        eligible = executed_by in {"arrow", "hybrid"} and teacher is not None and teacher_used and decision["policy_id"] == "arrow_on_call"
    eligibility_reasons: list[str] = []
    if not resolved_episode_success:
        eligibility_reasons.append("episode_not_successful")
    if not resolved_episode_complete:
        eligibility_reasons.append("episode_not_complete")
    if decision["policy_id"] == "arrow_minimal" and decision_metadata.get("variant") == "runtime_oracle" and minimal_branch is None:
        eligibility_reasons.append("no_real_minimal_branch_label")
    if decision["policy_id"] == "arrow_on_call":
        if teacher is None:
            eligibility_reasons.append("no_teacher_proposal")
        if not teacher_used:
            eligibility_reasons.append("teacher_action_not_executed")
    # Episode-level outcome is authoritative. A caller may add a stricter
    # eligibility filter, but cannot mark a failed/incomplete episode eligible.
    eligible = bool(eligible) and not eligibility_reasons
    if isinstance(identity, AttemptIdentity):
        identity_payload = identity.as_dict()
        task_id, reset_id, episode_id = identity.task_id, identity.reset_id, identity.episode_id
        seed = identity.seed
        init_state_index = None
        reset_digest = _digest(identity.reset_identity) if identity.reset_identity else None
    else:
        identity_payload = dict(identity or {})
        task_id = identity_payload.get("task_id", frame.metadata.get("task_id"))
        reset_value_id = identity_payload.get("reset_id", frame.metadata.get("reset_id"))
        reset_id = str(reset_value_id) if reset_value_id not in (None, "") else ""
        episode_id = str(identity_payload.get("episode_id", frame.episode_id or frame.metadata.get("episode_id", "")))
        seed = identity_payload.get("seed", frame.metadata.get("seed"))
        init_state_index = identity_payload.get("init_state_index", frame.metadata.get("init_state_index"))
        reset_value = identity_payload.get("reset_identity")
        reset_digest = identity_payload.get("reset_identity_sha256")
        if reset_digest is None:
            reset_digest = _digest(reset_value) if reset_value is not None else frame.metadata.get("reset_identity")
    if task_id in (None, "") or not reset_id or not episode_id:
        raise ContractError("training source records require explicit task, reset, and episode identity")
    resolved_hashes = dict(source_hashes or {})
    for hash_name in ("config_sha256", "model_sha256", "controller_sha256", "checkpoint_sha256",
                      "protocol_seal_sha256"):
        resolved_hashes.setdefault(hash_name, frame.metadata.get(hash_name))
    payload: dict[str, Any] = {
        "schema": TRAINING_SOURCE_SCHEMA,
        "task_id": task_id,
        "seed": seed,
        "init_state_index": init_state_index,
        "reset_id": reset_id,
        "episode_id": episode_id,
        "timestep": int(getattr(frame, "timestep", getattr(frame, "step", 0))),
        "identity": {"task_id": task_id, "seed": seed, "init_state_index": init_state_index,
                     "reset_id": reset_id, "episode_id": episode_id,
                     "reset_identity_sha256": reset_digest},
        "observation": observation,
        "observation_digest": frame.digest,
        "next_observation": next_observation,
        "next_observation_digest": next_frame.digest,
        "base_proposal": _proposal_payload(getattr(record, "base", None)),
        "teacher_proposal": teacher,
        "decision": decision,
        "executed_action": list(getattr(record, "action", decision["action"])),
        "executed_by": executed_by,
        "outcome": _outcome_payload(record),
        "episode_success": resolved_episode_success,
        "episode_complete": resolved_episode_complete,
        "eligible": bool(eligible),
        "eligibility_reasons": eligibility_reasons,
        "source_hashes": resolved_hashes,
    }
    payload["outcome"].update({
        "success": resolved_episode_success,
        "terminal": resolved_episode_complete,
        "episode_success": resolved_episode_success,
        "episode_complete": resolved_episode_complete,
    })
    if decision["policy_id"] == "arrow_minimal" and decision_metadata.get("variant") == "runtime_oracle":
        payload["source"] = "minimal_branch" if minimal_branch is not None else "arrow_minimal_runtime"
    if minimal_branch is not None:
        payload["minimal_branch"] = minimal_branch
    payload["hashes"] = {
        "observation_sha256": frame.digest,
        "next_observation_sha256": next_frame.digest,
        "base_proposal_sha256": _digest(payload["base_proposal"]),
        "teacher_proposal_sha256": _digest(payload["teacher_proposal"]) if teacher is not None else None,
        "decision_sha256": _digest(decision),
        "outcome_sha256": _digest(payload["outcome"]),
    }
    payload["transition_sha256"] = _digest(payload)
    return payload


class TrainingSourceWriter:
    """Append-only JSONL writer for downstream learned-artifact builders."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def append(self, record: Any, *, identity: AttemptIdentity | Mapping[str, Any] | None = None,
               source_hashes: Mapping[str, str] | None = None, eligible: bool | None = None,
               episode_success: bool | None = None, episode_complete: bool | None = None) -> dict[str, Any]:
        payload = training_source_payload(
            record, identity=identity, source_hashes=source_hashes, eligible=eligible,
            episode_success=episode_success, episode_complete=episode_complete,
        )
        with self.path.open("ab") as handle:
            handle.write(_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        self._count += 1
        return payload

    def append_attempt(self, attempt: CollectionAttempt, *, source_hashes: Mapping[str, str] | None = None) -> int:
        episode_success = bool(getattr(attempt.result.stats, "success", False))
        episode_complete = bool(getattr(attempt.result.stats, "terminal", False))
        for record in attempt.records:
            self.append(record, identity=attempt.identity, source_hashes=source_hashes,
                        episode_success=episode_success, episode_complete=episode_complete)
        return len(attempt.records)

    @property
    def count(self) -> int:
        return self._count

    def sha256(self) -> str:
        digest = hashlib.sha256()
        if not self.path.exists():
            return digest.hexdigest()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def load_training_source(path: str | Path, *, eligible_only: bool = False) -> tuple[Mapping[str, Any], ...]:
    """Load and verify the append-only transition artifact."""
    target = Path(path)
    if not target.is_file():
        raise ContractError(f"training source artifact does not exist: {target}")
    rows: list[Mapping[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid training source JSON at line {line_number}") from exc
            if row.get("schema") != TRAINING_SOURCE_SCHEMA:
                raise ContractError(f"unsupported training source schema at line {line_number}")
            supplied = row.get("transition_sha256")
            unsigned = dict(row)
            unsigned.pop("transition_sha256", None)
            if supplied != _digest(unsigned):
                raise ContractError(f"training source digest mismatch at line {line_number}")
            from .contracts import assert_student_observation
            assert_student_observation(row.get("observation", {}), path=f"training_source[{line_number}].observation")
            assert_student_observation(row.get("next_observation", {}), path=f"training_source[{line_number}].next_observation")
            if eligible_only and not bool(row.get("eligible", False)):
                continue
            rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class TraceEpisode:
    task_id: int | str
    reset_id: str
    episode_id: str
    states: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        # Deliberately no action, image, or simulator-pose fields.
        return {
            "task_id": self.task_id,
            "reset_id": self.reset_id,
            "episode_id": self.episode_id,
            "states": [list(state) for state in self.states],
        }


@dataclass(frozen=True)
class TraceView:
    schema: str
    parent_manifest_sha256: str
    routes: tuple[TraceEpisode, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.parent_manifest_sha256):
            raise ContractError("Trace view requires the SHA-256 of its source collection manifest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "routes": [route.as_dict() for route in self.routes],
        }


def default_on_call_eligibility(result: EpisodeResult) -> tuple[bool, tuple[str, ...]]:
    """Return the conservative training eligibility decision for an attempt."""

    reasons: list[str] = []
    if not bool(result.stats.success):
        reasons.append("episode_not_successful")
    if not bool(result.stats.terminal):
        reasons.append("episode_not_complete")
    if not result.records:
        reasons.append("episode_has_no_steps")
    if not any(
        bool(getattr(record, "teacher", None) is not None)
        and bool(getattr(getattr(record, "decision", None), "teacher_used", False)
                 or getattr(getattr(record, "decision", None), "metadata", {}).get("teacher_used", False))
        for record in result.records
    ):
        reasons.append("no_executed_teacher_transition")
    source_families = {_record_policy_family(record) for record in result.records}
    if "arrow_on_call" not in source_families:
        reasons.append("source_policy_is_not_arrow_on_call")
    return not reasons, tuple(reasons)


def _record_policy_family(record: Any) -> str:
    decision = getattr(record, "decision", None)
    value = getattr(decision, "policy_id", None)
    if value is None:
        value = getattr(getattr(decision, "proposal", None), "policy_id", None)
    if value is None:
        value = getattr(getattr(record, "policy", None), "policy_id", None)
    return str(value) if value is not None else ""


def _coerce_result(value: Any) -> EpisodeResult:
    if isinstance(value, EpisodeResult):
        return value
    if isinstance(value, CollectionAttempt):
        return value.result
    raise ContractError("episode runner must return EpisodeResult")


def _record_payload(record: StepRecord) -> dict[str, Any]:
    """Serialize the complete master transition without dropping modalities."""

    teacher = None
    if getattr(record, "teacher", None) is not None:
        proposal = record.teacher
        teacher = {
            "action": list(proposal.action),
            "producer": getattr(proposal, "producer", getattr(proposal, "policy_id", "teacher")),
            "confidence": getattr(proposal, "confidence", 1.0),
            "valid": getattr(proposal, "valid", True),
            "metadata": dict(getattr(proposal, "metadata", {})),
        }
    decision = record.decision
    decision_metadata = dict(getattr(decision, "metadata", {}))
    decision_action = getattr(decision, "action", None)
    if decision_action is None:
        decision_action = getattr(getattr(decision, "proposal", None), "action", ())
    return {
        "step": record.frame.step,
        "observation": record.frame.observation,
        "observation_digest": record.frame.digest,
        "base_action": list(record.base.action),
        "base_metadata": dict(getattr(record.base, "metadata", {})),
        "teacher": teacher,
        "decision": {
            "action": list(decision_action),
            "policy_id": getattr(decision, "policy_id", getattr(getattr(decision, "proposal", None), "policy_id", "unknown")),
            "teacher_used": bool(getattr(decision, "teacher_used", False) or decision_metadata.get("teacher_used", False)),
            "teacher_groups": list(getattr(decision, "teacher_groups", decision_metadata.get("teacher_groups", ()))),
            "metadata": decision_metadata,
            "provenance": dict(getattr(decision, "provenance", {}) or {}),
        },
        "executed_by": str(getattr(record, "executed_by", "")),
        "executed_action": list(getattr(record, "action", decision_action)),
        "next_observation": record.next_frame.observation,
        "next_observation_digest": record.next_frame.digest,
        "result": getattr(record, "result", getattr(record, "raw_result", None)),
        "success": bool(getattr(record, "success", False)),
        "terminal": bool(getattr(record, "terminal", False)),
    }


def _attempt_payload(attempt: CollectionAttempt) -> dict[str, Any]:
    return {
        "schema": MASTER_LOG_SCHEMA,
        "attempt": attempt.summary(),
        "records": [_record_payload(record) for record in attempt.records],
    }


class MasterLogWriter:
    """Append-only JSONL writer retaining every On-Call attempt."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def append(self, attempt: CollectionAttempt) -> None:
        payload = _canonical(_attempt_payload(attempt))
        with self.path.open("ab") as handle:
            handle.write(payload)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload)
    try:
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd != -1:
                os.close(fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}") from exc
    return target


def collect_on_call(
    task_ids: Sequence[int | str],
    runner: Callable[[int | str, int, AttemptIdentity], EpisodeResult | CollectionAttempt],
    *,
    config: StudyConfig | None = None,
    reset_ids: Mapping[int | str, Sequence[str]] | None = None,
    master_log: str | Path | MasterLogWriter | None = None,
    training_source: str | Path | TrainingSourceWriter | None = None,
    source_hashes: Mapping[str, str] | None = None,
    manifest_path: str | Path | None = None,
    parent_artifact: str = "",
    eligibility: Callable[[EpisodeResult], tuple[bool, Sequence[str]]] = default_on_call_eligibility,
    lineage_hook: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[tuple[CollectionAttempt, ...], CollectionManifest]:
    """Collect exactly 50 injected On-Call attempts per task.

    The callback is the only execution boundary.  A caller that supplies a
    real runner owns environment creation and can persist richer artifacts;
    this function itself performs no expensive work on import or preflight.
    """

    if not task_ids:
        raise ContractError("collection requires at least one task")
    if len(set(map(str, task_ids))) != len(task_ids):
        raise ContractError("collection task_ids must be unique")
    attempts_per_task = config.on_call_attempts_per_task if config is not None else 50
    if attempts_per_task != 50:
        raise ContractError("On-Call collection requires exactly 50 attempts per task")
    if config is not None:
        config.validate()
    reset_ids = reset_ids or {}
    reserved: set[str] = set()
    if config is not None:
        for task in task_ids:
            reserved.update(str(reset) for reset in config.test_reset_ids.get(task, ()))
            reserved.update(str(reset) for reset in config.validation_reset_ids.get(task, ()))
    writer = master_log if isinstance(master_log, MasterLogWriter) else MasterLogWriter(master_log) if master_log else None
    source_writer = (training_source if isinstance(training_source, TrainingSourceWriter)
                     else TrainingSourceWriter(training_source) if training_source else None)
    collected: list[CollectionAttempt] = []
    seen: set[tuple[str, str, str]] = set()
    discarded: dict[str, int] = {}
    task_counts: dict[str, dict[str, int]] = {}
    for task_id in task_ids:
        supplied = tuple(str(value) for value in reset_ids.get(task_id, ()))
        if supplied and len(supplied) != attempts_per_task:
            raise ContractError(f"task {task_id} must provide exactly 50 reset identities")
        identities = supplied or tuple(f"task-{task_id}-attempt-{index}" for index in range(attempts_per_task))
        if len(set(identities)) != attempts_per_task:
            raise ContractError(f"task {task_id} has duplicate reset identities")
        if reserved.intersection(identities):
            raise ContractError(f"task {task_id} collection overlaps a reserved evaluation reset identity")
        task_counts[str(task_id)] = {"attempted": 0, "eligible": 0}
        for index, reset_id in enumerate(identities):
            identity = AttemptIdentity(task_id, reset_id, f"task-{task_id}-attempt-{index}")
            result = _coerce_result(runner(task_id, index, identity))
            eligibility_result = eligibility(result)
            if isinstance(eligibility_result, tuple) and len(eligibility_result) == 2:
                accepted, reasons = eligibility_result
            else:
                accepted, reasons = bool(eligibility_result), ()
            if not accepted and not reasons:
                reasons = ("rejected_by_eligibility_filter",)
            # A caller-supplied filter may add business rules, but cannot
            # relabel a non-On-Call rollout as eligible.
            source_ok = "arrow_on_call" in {_record_policy_family(record) for record in result.records}
            if not source_ok:
                accepted = False
                reasons = tuple(dict.fromkeys((*tuple(str(reason) for reason in reasons),
                                               "source_policy_is_not_arrow_on_call")))
            attempt = CollectionAttempt(identity, result, bool(accepted), tuple(str(reason) for reason in reasons))
            if attempt.identity.key in seen:
                raise ContractError(f"duplicate collection identity {attempt.identity.key}")
            seen.add(attempt.identity.key)
            if writer is not None:
                writer.append(attempt)
            if source_writer is not None:
                source_writer.append_attempt(attempt, source_hashes=source_hashes)
            collected.append(attempt)
            task_counts[str(task_id)]["attempted"] += 1
            if attempt.eligible:
                task_counts[str(task_id)]["eligible"] += 1
            else:
                for reason in attempt.eligibility_reasons:
                    discarded[reason] = discarded.get(reason, 0) + 1
    config_manifest = config.manifest() if config is not None else {"schema": "unconfigured"}
    log_digest = writer.sha256() if writer is not None else _digest([_attempt_payload(attempt) for attempt in collected])
    training_source_digest = source_writer.sha256() if source_writer is not None else ""
    manifest = CollectionManifest(
        COLLECTION_SCHEMA,
        _digest(config_manifest),
        log_digest,
        len(collected),
        sum(int(item.eligible) for item in collected),
        attempts_per_task,
        task_counts,
        tuple(item.identity.episode_id for item in collected if item.eligible),
        discarded,
        parent_artifact,
        training_source_digest,
    )
    if manifest_path is not None:
        _write_immutable(manifest_path, manifest.as_dict())
    _call_hook(lineage_hook, manifest.as_dict())
    return tuple(collected), manifest


def derive_trace_view(
    attempts: Iterable[CollectionAttempt],
    *,
    parent_manifest_sha256: str = "",
    eligible_only: bool = True,
    manifest_path: str | Path | None = None,
    lineage_hook: Callable[[Mapping[str, Any]], Any] | None = None,
) -> TraceView:
    """Build the state-only Trace view from master records."""

    if not re.fullmatch(r"[0-9a-f]{64}", parent_manifest_sha256):
        raise ContractError("Trace view requires parent_manifest_sha256 from the collection manifest")
    views: list[TraceEpisode] = []
    for attempt in attempts:
        if eligible_only and not attempt.eligible:
            continue
        # state_only_routes performs the structural grouping; state8 performs
        # the canonical eight-value validation for every retained frame.
        grouped = state_only_routes(attempt.records)
        states = tuple(state8(record.frame.observation) for record in attempt.records)
        if len(grouped) != 1 or len(states) != len(grouped[0]):
            raise ContractError("Trace state route grouping is inconsistent")
        views.append(TraceEpisode(attempt.identity.task_id, attempt.identity.reset_id,
                                  attempt.identity.episode_id, states))
    result = TraceView(TRACE_SCHEMA, parent_manifest_sha256, tuple(views))
    if manifest_path is not None:
        _write_immutable(manifest_path, result.as_dict())
    _call_hook(lineage_hook, result.as_dict())
    return result


def write_master_log(path: str | Path, attempts: Iterable[CollectionAttempt]) -> Path:
    """Write all supplied attempts to a new append-only master log."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite master log: {target}")
    writer = MasterLogWriter(target)
    for attempt in attempts:
        writer.append(attempt)
    return target


def write_training_source_artifact(path: str | Path, attempts: Sequence[CollectionAttempt], *,
                                   source_hashes: Mapping[str, str] | None = None) -> TrainingSourceWriter:
    """Append all records from supplied attempts without overwriting ``path``."""
    writer = TrainingSourceWriter(path)
    for attempt in attempts:
        writer.append_attempt(attempt, source_hashes=source_hashes)
    return writer


def make_collection_archive(attempts: Sequence[CollectionAttempt], manifest: CollectionManifest) -> CollectionArchive:
    """Build a complete archive from the exact attempts used for a manifest."""
    values = tuple(_attempt_payload(attempt) for attempt in attempts)
    if len(values) != manifest.attempted:
        raise ContractError("archive attempts must cover every fixed collection attempt")
    return CollectionArchive(manifest, values)


def write_collection_archive(path: str | Path, attempts: Sequence[CollectionAttempt], manifest: CollectionManifest) -> CollectionArchive:
    """Persist the complete On-Call archive once and return its digest receipt."""
    archive = make_collection_archive(attempts, manifest)
    _write_immutable(path, archive.as_dict())
    return archive


# Naming aliases retained for launchers that spell out the fixed collection
# protocol in their call sites.
collect_on_call_attempts = collect_on_call
build_trace_view = derive_trace_view


__all__ = [
    "AttemptIdentity", "CollectionAttempt", "CollectionManifest", "CollectionArchive", "MasterLogWriter",
    "TrainingSourceWriter", "TRAINING_SOURCE_SCHEMA", "training_source_payload", "load_training_source",
    "TraceEpisode", "TraceView", "COLLECTION_SCHEMA", "MASTER_LOG_SCHEMA", "TRACE_SCHEMA", "COLLECTION_ARCHIVE_SCHEMA",
    "collect_on_call", "collect_on_call_attempts", "default_on_call_eligibility",
    "derive_trace_view", "build_trace_view", "write_master_log", "write_training_source_artifact",
    "make_collection_archive", "write_collection_archive",
]
