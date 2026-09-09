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

from .contracts import ContractError, StepRecord, _safe, clip_action, digest as observation_digest, state8

TRANSITION_SCHEMA = "arrow_policy_suite.interventions.v2"
DATASET_VIEW_SCHEMA = "arrow_policy_suite.dataset_view.v1"
PERSISTED_TRAINING_SCHEMA = "arrow_policy_suite.training_transitions.v1"
MINIMAL_BRANCH_SCHEMA = "arrow_policy_suite.minimal_branch_labels.v1"
_SHA256 = 64


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
    reset_id: str = ""
    executed_action: tuple[float, ...] | None = None
    observation_sha256: str = ""
    transition_sha256: str = ""
    label_action: tuple[float, ...] | None = None
    label_mask: tuple[float, ...] | None = None
    label_source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Mapping):
            raise ContractError("intervention observation must be a mapping")
        _safe(self.observation)
        object.__setattr__(self, "base_action", clip_action(self.base_action))
        object.__setattr__(self, "teacher_action", clip_action(self.teacher_action))
        if self.executed_action is not None:
            object.__setattr__(self, "executed_action", clip_action(self.executed_action))
        if self.label_action is not None:
            object.__setattr__(self, "label_action", clip_action(self.label_action))
            if not self.label_source:
                raise ContractError("label_action requires a non-empty label_source")
        if self.label_mask is not None:
            mask = tuple(float(value) for value in self.label_mask)
            if len(mask) != 7 or any(value not in {0.0, 1.0} for value in mask):
                raise ContractError("label_mask must contain exactly seven binary values")
            object.__setattr__(self, "label_mask", mask)
        if self.timestep < 0 or not self.episode_id:
            raise ContractError("invalid intervention row")
        if self.reset_id is not None and not isinstance(self.reset_id, str):
            raise ContractError("reset_id must be a string when supplied")
        if self.source not in {"on_call_teacher", "minimal_branch"}:
            raise ContractError("intervention rows must come from On-Call or Minimal runtime")
        for value, name in ((self.observation_sha256, "observation_sha256"),
                            (self.transition_sha256, "transition_sha256")):
            if value and (len(value) != _SHA256 or value.lower() != value or any(c not in "0123456789abcdef" for c in value)):
                raise ContractError(f"{name} must be a lowercase SHA-256 when supplied")
        _safe(self.metadata)
        _safe(self.outcome)

    @property
    def residual(self) -> tuple[float, ...]:
        return tuple(self.teacher_action[i] - self.base_action[i] for i in range(7))

    @property
    def executed_teacher(self) -> bool:
        return True

    @property
    def target_action(self) -> tuple[float, ...]:
        """The label consumed by a residual learner, never an online teacher."""
        return self.label_action if self.label_action is not None else self.teacher_action

    @property
    def target_kind(self) -> str:
        return "minimal_branch_residual" if self.label_action is not None else "teacher_residual"

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
        if self.schema not in {TRANSITION_SCHEMA, "arrow_policy_suite.interventions.v1", PERSISTED_TRAINING_SCHEMA}:
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
                outcome=outcome, reset_id=str(record.frame.metadata.get("reset_id", "")),
                executed_action=tuple(getattr(record.decision, "action", record.teacher.action)),
                observation_sha256=record.frame.digest,
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
        if row.source == "on_call_teacher" and row.metadata.get("source_policy_family", "arrow_on_call") != "arrow_on_call":
            raise ContractError("learning rows must remain executed On-Call teacher transitions")
        if row.source == "minimal_branch" and not row.label_action:
            raise ContractError("Minimal training rows require an explicit branch target action")
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
    if policy == "arrow_minimal_learned":
        if manifest is None or manifest.rows != len(rows):
            raise ContractError("arrow_minimal_learned rows do not match the source dataset manifest")
        if any(row.source != "minimal_branch" or row.label_action is None for row in rows):
            raise ContractError("Minimal-Learned requires persisted branch action labels")
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


@dataclass(frozen=True)
class PersistedTrainingView:
    """Strict, deterministic view over an immutable runtime artifact."""

    path: str
    rows: tuple[InterventionRow, ...]
    manifest: DatasetManifest
    source_sha256: str
    filter_name: str
    rejected: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.manifest.rows != len(self.rows):
            raise ContractError("persisted training manifest row count mismatch")
        if len(self.source_sha256) != _SHA256 or self.source_sha256.lower() != self.source_sha256 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ContractError("persisted training source hash must be a lowercase SHA-256")
        if not self.filter_name:
            raise ContractError("persisted training filter_name is required")
        _safe(self.rejected)


def _hash_field(value: Any, *, name: str, required: bool = False) -> str:
    if value in (None, ""):
        if required:
            raise ContractError(f"persisted transition requires {name}")
        return ""
    result = str(value)
    if len(result) != _SHA256 or result.lower() != result or any(c not in "0123456789abcdef" for c in result):
        raise ContractError(f"persisted transition {name} must be a lowercase SHA-256")
    return result


def _action_field(value: Any, *, name: str, required: bool = True) -> tuple[float, ...] | None:
    if value is None:
        if required:
            raise ContractError(f"persisted transition requires {name}")
        return None
    try:
        return clip_action(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"persisted transition {name} must be a seven-value action") from exc


def _minimal_mask(value: Any) -> tuple[float, ...]:
    if isinstance(value, bool):
        raise ContractError("Minimal branch mask cannot be boolean")
    if isinstance(value, int):
        if value < 0 or value > 7:
            raise ContractError("Minimal branch mask integer must be in [0, 7]")
        return tuple(
            float(bool(value & 1)) for _ in range(3)
        ) + tuple(float(bool(value & 2)) for _ in range(3)) + (float(bool(value & 4)),)
    if isinstance(value, Mapping):
        groups = {str(key): bool(item) for key, item in value.items()}
        return tuple(float(groups.get(group, False)) for group in ("translation", "translation", "translation", "rotation", "rotation", "rotation", "gripper"))
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError("Minimal branch mask must be an integer, mapping, or binary vector") from exc
    if len(values) == 3:
        values = (values[0], values[0], values[0], values[1], values[1], values[1], values[2])
    if len(values) != 7 or any(item not in {0.0, 1.0} for item in values):
        raise ContractError("Minimal branch mask must expand to seven binary values")
    return values


def _source_and_context(raw: Mapping[str, Any], attempt: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any]]:
    source = raw.get("source", raw.get("source_policy", ""))
    if not source and attempt is not None:
        source = attempt.get("source", "")
    return str(source), attempt or {}


def _row_from_persisted(raw: Mapping[str, Any], *, attempt: Mapping[str, Any] | None,
                        variant: str, source_path: str) -> InterventionRow:
    if not isinstance(raw, Mapping):
        raise ContractError("persisted transition row must be an object")
    schema = str(raw.get("schema", ""))
    source, context = _source_and_context(raw, attempt)
    task_id = raw.get("task_id", context.get("task_id"))
    reset_id = raw.get("reset_id", context.get("reset_id"))
    episode_id = raw.get("episode_id", context.get("episode_id"))
    if task_id in (None, "") or reset_id in (None, "") or episode_id in (None, ""):
        raise ContractError("persisted transition requires task_id, reset_id, and episode_id")
    timestep = raw.get("timestep", raw.get("step"))
    if isinstance(timestep, bool) or timestep is None:
        raise ContractError("persisted transition requires a non-negative timestep")
    try:
        timestep = int(timestep)
    except (TypeError, ValueError) as exc:
        raise ContractError("persisted transition timestep must be an integer") from exc
    if timestep < 0:
        raise ContractError("persisted transition timestep must be non-negative")
    observation = raw.get("observation")
    if not isinstance(observation, Mapping):
        raise ContractError("persisted transition requires an observation mapping")
    state8(observation)
    hashes = raw.get("hashes", {})
    if hashes and not isinstance(hashes, Mapping):
        raise ContractError("persisted transition hashes must be an object")
    obs_hash = _hash_field(raw.get("observation_digest", raw.get("observation_sha256", raw.get("observation_hash", hashes.get("observation_sha256", hashes.get("observation_hash", hashes.get("observation")))))), name="observation_digest", required=True)
    if observation_digest(observation) != obs_hash:
        raise ContractError("persisted observation_digest does not match observation bytes")
    transition_hash = _hash_field(raw.get("transition_sha256", raw.get("transition_hash", hashes.get("transition_sha256", hashes.get("transition_hash", hashes.get("transition"))))), name="transition_sha256")
    base_value = raw.get("base_action", raw.get("base_proposal"))
    if isinstance(base_value, Mapping):
        base_value = base_value.get("action")
    base_action = _action_field(base_value, name="base_action")
    teacher = raw.get("teacher", raw.get("teacher_proposal"))
    if teacher is not None and not isinstance(teacher, Mapping):
        raise ContractError("persisted teacher must be an object")
    teacher_action = _action_field(raw.get("teacher_action", teacher.get("action") if teacher else None), name="teacher_action", required=variant == "arrow_editor")
    decision = raw.get("decision", {})
    if decision and not isinstance(decision, Mapping):
        raise ContractError("persisted decision must be an object")
    executed_value = raw.get("executed_action", raw.get("executed", decision.get("action") if decision else None))
    if isinstance(executed_value, Mapping):
        executed_value = executed_value.get("action")
    executed_action = _action_field(executed_value, name="executed_action")
    outcome = raw.get("outcome", {})
    if not isinstance(outcome, Mapping):
        raise ContractError("persisted outcome must be an object")
    success = bool(outcome.get("success", raw.get("success", False)))
    terminal = bool(outcome.get("terminal", raw.get("terminal", False)))
    if schema == "arrow_policy_suite.training_source.v1" and not source:
        source = "on_call" if str(decision.get("policy_id", "")) == "arrow_on_call" else ""
    if schema == "arrow_policy_suite.training_source.v1":
        supplied_transition = raw.get("transition_sha256")
        unsigned = dict(raw)
        unsigned.pop("transition_sha256", None)
        if supplied_transition != hashlib.sha256(_canonical(unsigned)).hexdigest():
            raise ContractError("persisted training source transition hash mismatch")
    if variant == "arrow_editor":
        if source not in {"on_call", "on_call_teacher", "arrow_on_call"}:
            raise ContractError("Editor training accepts only persisted Arrow On-Call transitions")
        if teacher_action is None or executed_action != teacher_action:
            raise ContractError("Editor transition must persist the executed teacher action")
        if decision and not bool(decision.get("teacher_used", False)):
            raise ContractError("Editor transition must mark teacher_used")
        normalized_source = "on_call_teacher"
        label_action = None
        label_mask = None
        label_source = "executed_on_call_teacher"
    elif variant == "arrow_minimal_learned":
        if not source.startswith("minimal") and source not in {"arrow_minimal", "arrow_minimal_runtime"}:
            raise ContractError("Minimal-Learned accepts only persisted Minimal runtime transitions")
        branch = raw.get("minimal_branch", raw.get("branch", raw.get("labels", raw)))
        if not isinstance(branch, Mapping):
            raise ContractError("Minimal transition requires a branch-label object")
        selected = branch.get("selected_hybrid_action", branch.get("target_action", branch.get("hybrid_action", branch.get("selected_action", raw.get("selected_hybrid_action")))))
        label_action = _action_field(selected, name="selected_hybrid_action")
        mask_value = branch.get("selected_mask", branch.get("selected_mask_bits", branch.get("ownership_mask", branch.get("mask", raw.get("selected_mask", raw.get("selected_mask_bits", raw.get("mask")))))))
        if mask_value is None:
            raise ContractError("Minimal transition requires the selected branch mask")
        label_mask = _minimal_mask(mask_value)
        label_source = str(branch.get("label_source", "minimal_branch_runtime"))
        if not label_source or label_source in {"implicit", "all_ones", "permissive"}:
            raise ContractError("Minimal branch label_source must identify a real persisted branch")
        normalized_source = "minimal_branch"
        if teacher_action is None:
            teacher_action = executed_action
    else:
        raise ContractError(f"unsupported persisted training variant: {variant}")
    metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {}
    metadata.update({"source_policy_family": "arrow_on_call" if normalized_source == "on_call_teacher" else "arrow_minimal_runtime",
                     "persisted_source": source, "source_path": source_path,
                     "observation_digest": obs_hash})
    if isinstance(raw.get("source_hashes"), Mapping):
        metadata["source_hashes"] = dict(raw["source_hashes"])
    if label_mask is not None:
        metadata["minimal_label_mask"] = list(label_mask)
    return InterventionRow(
        episode_id=str(episode_id), task_id=task_id, timestep=timestep,
        observation=observation, base_action=base_action, teacher_action=teacher_action or executed_action,
        success_episode=success, source=normalized_source, metadata=metadata,
        outcome={**dict(outcome), "success": success, "terminal": terminal}, reset_id=str(reset_id),
        executed_action=executed_action, observation_sha256=obs_hash, transition_sha256=transition_hash,
        label_action=label_action, label_mask=label_mask, label_source=label_source,
    )


def _persisted_records(root: Any, *, source_path: str) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any] | None], ...]:
    """Normalize archive, master-log, dataset-view, and minimal-label artifacts."""
    entries: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    values = root if isinstance(root, list) else [root]
    for item in values:
        if not isinstance(item, Mapping):
            raise ContractError("persisted training artifact entries must be objects")
        schema = str(item.get("schema", ""))
        if schema in {"arrow_policy_suite.on_call_archive.v1"}:
            attempts = item.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise ContractError("On-Call archive requires a non-empty attempts list")
            archive_hash = item.get("archive_sha256")
            if archive_hash:
                expected_archive = hashlib.sha256(_canonical({"schema": schema, "manifest": item.get("manifest", {}),
                                                              "attempts": attempts})).hexdigest()
                if archive_hash != expected_archive:
                    raise ContractError("On-Call archive digest does not match contents")
            for attempt_payload in attempts:
                if not isinstance(attempt_payload, Mapping):
                    raise ContractError("On-Call archive attempt must be an object")
                attempt = attempt_payload.get("attempt", attempt_payload)
                records = attempt_payload.get("records")
                if not isinstance(attempt, Mapping) or not isinstance(records, list):
                    raise ContractError("On-Call archive attempt requires attempt and records")
                entries.extend((record, attempt) for record in records)
        elif schema in {"arrow_policy_suite.master_log.v1", PERSISTED_TRAINING_SCHEMA, MINIMAL_BRANCH_SCHEMA, DATASET_VIEW_SCHEMA, ""} or "minimal" in schema or "training" in schema:
            if schema == DATASET_VIEW_SCHEMA and "manifest" in item and "row" not in item and "rows" not in item:
                # ``write_dataset_view`` emits a JSONL header followed by row
                # objects.  The header is provenance, not a transition.
                continue
            if "row" in item:
                row = item.get("row")
                if not isinstance(row, Mapping):
                    raise ContractError("dataset view row must be an object")
                entries.append((row, None))
            elif isinstance(item.get("records"), list):
                attempt = item.get("attempt", item.get("context", {}))
                if not isinstance(attempt, Mapping):
                    raise ContractError("master-log attempt context must be an object")
                entries.extend((record, attempt) for record in item["records"])
            elif isinstance(item.get("rows", item.get("transitions")), list):
                entries.extend((record, None) for record in item.get("rows", item.get("transitions")))
            elif schema == "arrow_policy_suite.training_source.v1":
                entries.append((item, None))
            elif schema:
                raise ContractError(f"persisted artifact schema {schema!r} has no records")
            else:
                entries.append((item, None))
        else:
            raise ContractError(f"unsupported persisted training artifact schema: {schema}")
    if not entries:
        raise ContractError("persisted training artifact contains no transitions")
    return tuple(entries)


def load_persisted_training_view(path: str | Path, *, variant: str,
                                 task_ids: Sequence[int | str] | None = None,
                                 reset_ids: Sequence[str] | None = None,
    episode_ids: Sequence[str] | None = None,
    require_success: bool = False,
    eligible_only: bool = True,
    parent_artifact: str | None = None,
                                 filter_name: str | None = None) -> PersistedTrainingView:
    """Load real persisted On-Call/Minimal rows with fixed, auditable filtering."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ContractError(f"persisted training artifact is missing or is a symlink: {target}")
    data = target.read_bytes()
    source_sha = hashlib.sha256(data).hexdigest()
    try:
        root = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            root = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot parse persisted training artifact: {target}") from exc
    pairs = _persisted_records(root, source_path=str(target))
    allowed_tasks = None if task_ids is None else {str(value) for value in task_ids}
    allowed_resets = None if reset_ids is None else {str(value) for value in reset_ids}
    allowed_episodes = None if episode_ids is None else {str(value) for value in episode_ids}
    rejected: dict[str, int] = {}
    rows: list[InterventionRow] = []
    for raw, attempt in pairs:
        if eligible_only and str(raw.get("schema", "")) == "arrow_policy_suite.training_source.v1" and not bool(raw.get("eligible", False)):
            rejected["ineligible_source_row"] = rejected.get("ineligible_source_row", 0) + 1
            continue
        row = _row_from_persisted(raw, attempt=attempt, variant=variant, source_path=str(target))
        reason = None
        if allowed_tasks is not None and str(row.task_id) not in allowed_tasks:
            reason = "task_filter"
        elif allowed_resets is not None and row.reset_id not in allowed_resets:
            reason = "reset_filter"
        elif allowed_episodes is not None and row.episode_id not in allowed_episodes:
            reason = "episode_filter"
        elif require_success and not row.success_episode:
            reason = "success_filter"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.task_id), row.reset_id, row.episode_id, row.timestep))
    if not rows:
        raise ContractError("persisted training filters selected zero transitions")
    identities = [(row.episode_id, row.timestep) for row in rows]
    if len(identities) != len(set(identities)):
        raise ContractError("persisted training artifact contains duplicate episode/timestep rows")
    resolved_filter = filter_name or f"persisted:{variant}:task={sorted(allowed_tasks) if allowed_tasks is not None else '*'}:reset={sorted(allowed_resets) if allowed_resets is not None else '*'}:episode={sorted(allowed_episodes) if allowed_episodes is not None else '*'}:success={require_success}:eligible={eligible_only}"
    manifest = DatasetManifest(PERSISTED_TRAINING_SCHEMA, len(rows), tuple(sorted({row.episode_id for row in rows})),
                               parent_artifact or source_sha, hashlib.sha256(_canonical([row.as_dict() for row in rows])).hexdigest(),
                               resolved_filter, source_sha,
                               {"success": sum(int(row.success_episode) for row in rows), "failure": sum(int(not row.success_episode) for row in rows),
                                "rejected": sum(rejected.values())})
    return PersistedTrainingView(str(target), tuple(rows), manifest, source_sha, resolved_filter, rejected)


load_persisted_training_rows = load_persisted_training_view
load_training_transitions = load_persisted_training_view
load_persisted_transition_rows = load_persisted_training_view
load_training_artifact = load_persisted_training_view


__all__ = [
    "TRANSITION_SCHEMA", "DATASET_VIEW_SCHEMA", "PERSISTED_TRAINING_SCHEMA", "MINIMAL_BRANCH_SCHEMA",
    "InterventionRow", "DatasetManifest", "DatasetView", "PersistedTrainingView",
    "transition_eligibility", "shared_transition_filter", "intervention_rows", "eligible_intervention_rows",
    "state_only_routes", "manifest_for_rows", "write_dataset_view", "write_manifest", "fit_with_callback",
    "load_persisted_training_view", "load_persisted_training_rows", "load_training_transitions",
    "load_persisted_transition_rows", "load_training_artifact",
]
