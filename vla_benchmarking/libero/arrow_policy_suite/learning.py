"""Offline transition views shared by Apprentice and Editor.

On-Call keeps every attempt, including failures.  A transition is eligible
whenever Arrow actually supplied the executed action; episode outcome is
retained on every row rather than used as a success-only filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import ContractError, StepRecord, _safe, clip_action, state8

TRANSITION_SCHEMA = "arrow_policy_suite.interventions.v2"
DATASET_VIEW_SCHEMA = "arrow_policy_suite.dataset_view.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _record_policy_family(record: Any) -> str:
    decision = getattr(record, "decision", None)
    value = getattr(decision, "policy_id", None)
    if value is None:
        value = getattr(getattr(decision, "proposal", None), "policy_id", None)
    if value is None:
        value = getattr(getattr(record, "policy", None), "policy_id", None)
    return str(value) if value is not None else ""


def _teacher_was_executed(record: Any) -> bool:
    decision = getattr(record, "decision", None)
    metadata = getattr(decision, "metadata", {}) or {}
    return bool(getattr(decision, "teacher_used", False) or metadata.get("teacher_used", False))


def transition_eligibility(record: StepRecord) -> tuple[bool, tuple[str, ...]]:
    """Shared Apprentice/Editor rule: every executed Arrow On-Call transition."""
    reasons: list[str] = []
    if getattr(record, "teacher", None) is None:
        reasons.append("no_teacher_proposal")
    if not _teacher_was_executed(record):
        reasons.append("teacher_action_not_executed")
    if _record_policy_family(record) != "arrow_on_call":
        reasons.append("source_policy_is_not_arrow_on_call")
    return not reasons, tuple(reasons)


@dataclass(frozen=True)
class InterventionRow:
    episode_id: str
    task_id: int | str
    timestep: int
    observation: Mapping[str, Any]
    base_action: tuple[float, ...]
    teacher_action: tuple[float, ...]
    success_episode: bool
    source: str = "on_call_teacher"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    outcome: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _safe(self.observation)
        object.__setattr__(self, "base_action", clip_action(self.base_action))
        object.__setattr__(self, "teacher_action", clip_action(self.teacher_action))
        if self.timestep < 0 or not self.episode_id:
            raise ContractError("invalid intervention row")
        if self.source != "on_call_teacher":
            raise ContractError("intervention rows must come from Arrow On-Call")
        _safe(self.metadata)
        _safe(self.outcome)

    @property
    def residual(self) -> tuple[float, ...]:
        return tuple(self.teacher_action[i] - self.base_action[i] for i in range(7))

    @property
    def executed_teacher(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return _safe(self.__dict__)


@dataclass(frozen=True)
class DatasetManifest:
    schema: str
    rows: int
    episode_ids: tuple[str, ...]
    parent_artifact: str
    content_sha256: str
    filter_name: str
    source_sha256: str = ""
    outcome_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema not in {TRANSITION_SCHEMA, "arrow_policy_suite.interventions.v1"}:
            raise ContractError(f"unsupported dataset manifest schema: {self.schema}")
        if self.rows < 0 or not self.parent_artifact or not self.filter_name:
            raise ContractError("dataset manifest requires non-negative rows, parent, and filter")
        if not self.content_sha256:
            raise ContractError("dataset manifest requires a source content hash")
        if not self.source_sha256:
            object.__setattr__(self, "source_sha256", self.content_sha256)
        _safe(self.outcome_counts)

    @property
    def manifest_sha256(self) -> str:
        payload = {"schema": self.schema, "rows": self.rows, "episode_ids": self.episode_ids,
                   "parent_artifact": self.parent_artifact, "content_sha256": self.content_sha256,
                   "filter_name": self.filter_name, "source_sha256": self.source_sha256,
                   "outcome_counts": self.outcome_counts}
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**_safe(self.__dict__), "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class DatasetView:
    """A deterministic, create-only materialized view over intervention rows."""

    path: str
    manifest: DatasetManifest
    rows: int
    view_sha256: str


def intervention_rows(
    records: Iterable[StepRecord],
    *,
    require_success: bool = False,
    task_ids: Sequence[int | str] | None = None,
    episode_ids: Sequence[str] | None = None,
) -> tuple[InterventionRow, ...]:
    """Extract every executed Arrow On-Call transition.

    ``require_success`` is an explicit compatibility switch for analyses that
    want successful episodes only.  The training default is outcome-aware.
    """
    grouped: dict[str, list[StepRecord]] = {}
    for record in records:
        episode_id = record.frame.metadata.get("episode_id") or getattr(record.frame, "episode_id", None)
        if episode_id in (None, ""):
            raise ContractError("intervention records require frame.metadata['episode_id']")
        grouped.setdefault(str(episode_id), []).append(record)
    rows: list[InterventionRow] = []
    allowed_tasks = {str(value) for value in task_ids} if task_ids is not None else None
    allowed_episodes = {str(value) for value in episode_ids} if episode_ids is not None else None
    for episode_id, episode_records in grouped.items():
        episode_records = sorted(episode_records, key=lambda record: record.frame.step)
        if allowed_episodes is not None and episode_id not in allowed_episodes:
            continue
        task_id = episode_records[0].frame.metadata.get("task_id", "unknown") if episode_records else "unknown"
        if allowed_tasks is not None and str(task_id) not in allowed_tasks:
            continue
        success = bool(episode_records and (episode_records[-1].success or getattr(episode_records[-1].decision, "success", False)))
        if require_success and not success:
            continue
        terminal = bool(episode_records and episode_records[-1].terminal)
        outcome = {"success": success, "terminal": terminal, "steps": len(episode_records), "episode_id": episode_id}
        for record in episode_records:
            accepted, _ = transition_eligibility(record)
            if not accepted:
                continue
            decision_metadata = dict(getattr(record.decision, "metadata", {}) or {})
            rows.append(InterventionRow(
                episode_id=episode_id, task_id=task_id, timestep=record.frame.step,
                observation=record.frame.observation, base_action=record.base.action,
                teacher_action=record.teacher.action, success_episode=success,
                metadata={"observation_digest": record.frame.digest,
                          "policy": _record_policy_family(record),
                          "source_policy_family": "arrow_on_call",
                          "teacher_groups": list(getattr(record.decision, "teacher_groups", ())),
                          "decision_metadata": decision_metadata,
                          "cost": dict(getattr(record, "metadata", {}).get("cost", {}))},
                outcome=outcome,
            ))
    return tuple(sorted(rows, key=lambda row: (str(row.episode_id), row.timestep)))


eligible_intervention_rows = intervention_rows
shared_transition_filter = transition_eligibility


def state_only_routes(records: Iterable[StepRecord]) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], ...]:
    grouped: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    for record in records:
        key_value = record.frame.metadata.get("episode_id") or getattr(record.frame, "episode_id", None)
        if key_value in (None, ""):
            raise ContractError("Trace records require frame.metadata['episode_id']")
        grouped.setdefault(str(key_value), []).append(("state", state8(record.frame.observation)))
    return tuple(tuple(values) for _, values in sorted(grouped.items()))


def manifest_for_rows(rows: Sequence[InterventionRow], *, parent_artifact: str,
                      filter_name: str = "executed_teacher_transitions") -> DatasetManifest:
    if not parent_artifact:
        raise ContractError("learning manifest requires a parent master artifact")
    ordered = tuple(sorted(rows, key=lambda row: (str(row.episode_id), row.timestep)))
    for row in ordered:
        if row.source != "on_call_teacher" or row.metadata.get("source_policy_family", "arrow_on_call") != "arrow_on_call":
            raise ContractError("learning rows must remain executed On-Call teacher transitions")
    content_hash = hashlib.sha256(_canonical([row.as_dict() for row in ordered])).hexdigest()
    counts = {"success": sum(bool(row.success_episode) for row in ordered),
              "failure": sum(not bool(row.success_episode) for row in ordered)}
    return DatasetManifest(TRANSITION_SCHEMA, len(ordered), tuple(sorted({row.episode_id for row in ordered})),
                           parent_artifact, content_hash, filter_name, content_hash, counts)


def _create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
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
        raise ContractError(f"refusing to overwrite immutable dataset view: {path}") from exc


def write_dataset_view(path: str | Path, rows: Sequence[InterventionRow], manifest: DatasetManifest) -> DatasetView:
    ordered = tuple(sorted(rows, key=lambda row: (str(row.episode_id), row.timestep)))
    actual = manifest_for_rows(ordered, parent_artifact=manifest.parent_artifact, filter_name=manifest.filter_name)
    if actual.content_sha256 != manifest.content_sha256 or actual.rows != manifest.rows:
        raise ContractError("dataset rows do not match supplied source manifest")
    payload = [_safe({"schema": DATASET_VIEW_SCHEMA, "manifest": manifest.as_dict()})]
    payload.extend(_safe({"row": row.as_dict()}) for row in ordered)
    data = b"".join(_canonical(item) for item in payload)
    _create_only(Path(path), data)
    return DatasetView(str(Path(path)), manifest, len(ordered), hashlib.sha256(data).hexdigest())


def write_manifest(path: str | Path, manifest: DatasetManifest,
                   *, lineage_hook: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
    _create_only(Path(path), _canonical(manifest.as_dict()))
    if lineage_hook is not None:
        lineage_hook(manifest.as_dict())


def fit_with_callback(rows: Sequence[InterventionRow], fitter: Callable[[Sequence[InterventionRow]], Any], *, policy: str,
                      manifest: DatasetManifest | None = None,
                      lineage_hook: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    if not rows:
        raise ContractError(f"{policy} has no eligible intervention rows")
    if policy in {"arrow_apprentice", "arrow_editor"}:
        if manifest is None or manifest.rows != len(rows):
            raise ContractError(f"{policy} rows do not match the source dataset manifest")
        for row in rows:
            if row.source != "on_call_teacher" or row.metadata.get("source_policy_family", "arrow_on_call") != "arrow_on_call":
                raise ContractError(f"{policy} rows must remain executed On-Call teacher transitions")
    result = fitter(tuple(rows))
    cost = {"offline_rows": len(rows), "offline_episodes": len({r.episode_id for r in rows}),
            "success_rows": sum(bool(r.success_episode) for r in rows),
            "failure_rows": sum(not bool(r.success_episode) for r in rows)}
    payload: dict[str, Any] = {"policy": policy, "rows": len(rows), "episodes": len({r.episode_id for r in rows}),
                               "cost": cost, "trainer_result": result}
    if manifest is not None:
        payload["manifest"] = manifest.as_dict()
    if lineage_hook is not None:
        lineage_hook(payload)
    return payload


__all__ = [
    "TRANSITION_SCHEMA", "DATASET_VIEW_SCHEMA", "InterventionRow", "DatasetManifest", "DatasetView",
    "transition_eligibility", "shared_transition_filter", "intervention_rows", "eligible_intervention_rows",
    "state_only_routes", "manifest_for_rows", "write_dataset_view", "write_manifest", "fit_with_callback",
]
