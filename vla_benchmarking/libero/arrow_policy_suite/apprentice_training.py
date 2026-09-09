"""Native SmolVLA/LoRA Apprentice orchestration contracts.

The production exporter and trainer are intentionally injected.  Importing this
module never imports LeRobot/PEFT and never starts a dataset conversion or
training job.  The contract makes the successful On-Call teacher rows,
SmolVLA identity, LoRA settings, and dataset lineage explicit at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .contracts import (
    ContractError,
    ObservationFrame,
    _safe,
    digest,
    validate_action,
    validate_student_observation,
)
from .learning import DatasetManifest, InterventionRow


APPRENTICE_SCHEMA = "arrow_policy_suite.apprentice.v1"
DEFAULT_SMOLVLA_MODEL = "HuggingFaceVLA/smolvla_libero"
APPRENTICE_DATASET_SCHEMA = APPRENTICE_SCHEMA + ".lerobot_dataset"
APPRENTICE_ADAPTER_SCHEMA = APPRENTICE_SCHEMA + ".adapter"
DEFAULT_SMOLVLA_PEFT_TARGET_REGEX = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)
_SHA256 = frozenset("0123456789abcdef")
_IMAGE_KEYS = {
    "agentview": ("agentview", "observation.images.image"),
    "wrist": ("wrist", "observation.images.image2"),
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _create_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite immutable Apprentice artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in _SHA256 for char in normalized):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _decode_persisted(value: Any) -> Any:
    """Decode the JSON-safe ndarray/bytes envelope used by On-Call archives."""
    if isinstance(value, list):
        return [_decode_persisted(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    tag_key = "__type__" if "__type__" in value else "__automatic_ttt_type__" if "__automatic_ttt_type__" in value else None
    if tag_key is None:
        return {str(key): _decode_persisted(item) for key, item in value.items()}
    kind = value.get(tag_key)
    if kind == "tuple":
        return tuple(_decode_persisted(item) for item in value.get("items", ()))
    if kind == "mapping":
        return {
            _decode_persisted(item[0]): _decode_persisted(item[1])
            for item in value.get("items", ())
        }
    if kind == "bytes":
        import base64
        return base64.b64decode(str(value.get("data", "")).encode("ascii"))
    if kind == "ndarray":
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ContractError("numpy is required to decode persisted image observations") from exc
        import base64
        raw = base64.b64decode(str(value.get("data", "")).encode("ascii"))
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy().reshape(tuple(value["shape"]))
    raise ContractError(f"unknown persisted value type {kind!r}")


def _image_array(value: Any, *, name: str) -> Any:
    """Return a native image array and reject digest-only placeholders."""
    if isinstance(value, Mapping) and any(key in value for key in ("sha256", "digest", "image_sha256")):
        raise ContractError(f"{name} contains only a digest; raw image payload is required")
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ContractError("numpy is required for Apprentice image validation") from exc
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ContractError(f"{name} is not a valid image payload") from exc
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4) or min(array.shape[:2]) <= 0:
        raise ContractError(f"{name} must be an HxWxC image array")
    if not np.issubdtype(array.dtype, np.number):
        raise ContractError(f"{name} must contain numeric pixels")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains non-finite pixels")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _state_payload(observation: Mapping[str, Any]) -> tuple[float, ...]:
    value = observation.get("observation.state", observation.get("state"))
    if value is None:
        raise ContractError("Apprentice transition requires an eight-value state")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ContractError("Apprentice transition state is not numeric") from exc
    if len(values) != 8 or any(not math.isfinite(item) for item in values):
        raise ContractError("Apprentice transition requires an eight-value finite state")
    return values


def _strict_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise ContractError("Apprentice transition observation must be a mapping")
    decoded = _decode_persisted(observation)
    if not isinstance(decoded, Mapping):
        raise ContractError("Apprentice transition observation must be a mapping")
    decoded = dict(decoded)
    # LeRobot names cameras with feature paths; normalize those aliases to the
    # Arrow policy's canonical student-observation names before validation.
    if "agentview" not in decoded and "observation.images.image" in decoded:
        decoded["agentview"] = decoded["observation.images.image"]
    if "wrist" not in decoded and "observation.images.image2" in decoded:
        decoded["wrist"] = decoded["observation.images.image2"]
    if "instruction" not in decoded and decoded.get("task") not in (None, ""):
        decoded["instruction"] = decoded["task"]
    decoded.pop("task", None)
    decoded.pop("observation.images.image", None)
    decoded.pop("observation.images.image2", None)
    # The canonical policy boundary rejects privileged fields and unknown
    # modalities.  Require the two cameras, not their hashes or metadata.
    validate_student_observation(decoded, require_complete=True)
    normalized = dict(decoded)
    for canonical, aliases in _IMAGE_KEYS.items():
        key = next((candidate for candidate in aliases if candidate in normalized), None)
        if key is None:
            raise ContractError(f"Apprentice transition requires raw {canonical} image")
        normalized[canonical] = _image_array(normalized[key], name=canonical)
    normalized["state"] = _state_payload(normalized)
    return normalized


def _transition_row_from_mapping(raw: Mapping[str, Any], *, episode_id: str | None = None,
                                 task_id: int | str | None = None,
                                 success_episode: bool | None = None) -> InterventionRow:
    """Normalize one persisted teacher transition without accepting digest-only rows."""
    payload = _decode_persisted(raw)
    if not isinstance(payload, Mapping):
        raise ContractError("persisted Apprentice transition must be a mapping")
    observation = payload.get("observation")
    if not isinstance(observation, Mapping):
        raise ContractError("persisted Apprentice transition is missing raw observation")
    observation = _strict_observation(observation)
    teacher = payload.get("teacher_action")
    if teacher is None:
        teacher_record = payload.get("teacher")
        if isinstance(teacher_record, Mapping):
            teacher = teacher_record.get("action")
    if teacher is None:
        teacher = payload.get("action") if payload.get("source") in {"teacher_correction", "on_call_teacher"} else None
    if teacher is None:
        raise ContractError("persisted Apprentice transition is missing executed teacher action")
    try:
        teacher_action = validate_action(teacher)
    except (TypeError, ValueError) as exc:
        raise ContractError("persisted teacher action is invalid") from exc
    resolved_episode = payload.get("episode_id", episode_id)
    if resolved_episode in (None, ""):
        raise ContractError("persisted Apprentice transition is missing episode_id")
    resolved_task = payload.get("task_id", task_id)
    if resolved_task in (None, ""):
        task = payload.get("task") or observation.get("instruction")
        resolved_task = task
    if resolved_task in (None, ""):
        raise ContractError("persisted Apprentice transition is missing task")
    timestep = payload.get("timestep", payload.get("step"))
    if timestep is None:
        raise ContractError("persisted Apprentice transition is missing timestep")
    try:
        timestep = int(timestep)
    except (TypeError, ValueError) as exc:
        raise ContractError("persisted Apprentice timestep is invalid") from exc
    base = payload.get("base_action", tuple(0.0 for _ in range(7)))
    base_action = validate_action(base)
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {}
    metadata.update({"source_policy_family": "arrow_on_call", "task": str(resolved_task)})
    return InterventionRow(
        episode_id=str(resolved_episode), task_id=resolved_task, timestep=timestep,
        observation=observation, base_action=base_action, teacher_action=teacher_action,
        success_episode=bool(payload.get("success_episode", success_episode if success_episode is not None else payload.get("success", False))),
        metadata=metadata, outcome=dict(payload.get("outcome", {})) if isinstance(payload.get("outcome", {}), Mapping) else {},
    )


def _iter_persisted_rows(payload: Any) -> Iterable[InterventionRow]:
    """Yield strict rows from the legacy persisted Apprentice views."""
    payload = _decode_persisted(payload)
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_persisted_rows(item)
        return
    if not isinstance(payload, Mapping):
        raise ContractError("persisted Apprentice source must contain mappings")
    schema = str(payload.get("schema", ""))
    if schema.endswith("dataset_view.v1") or "row" in payload:
        row = payload.get("row", payload)
        if isinstance(row, Mapping):
            if "row" in payload:
                yield _transition_row_from_mapping(row)
            return
    if "attempts" in payload:
        attempts = payload.get("attempts")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise ContractError("On-Call archive attempts must be a sequence")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise ContractError("On-Call archive attempt must be a mapping")
            summary = attempt.get("attempt", attempt.get("summary", {}))
            records = attempt.get("records", ())
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                raise ContractError("On-Call archive records must be a sequence")
            for record in records:
                if not isinstance(record, Mapping):
                    raise ContractError("On-Call archive record must be a mapping")
                teacher = record.get("teacher")
                if not isinstance(teacher, Mapping):
                    continue
                decision = record.get("decision", {})
                if not isinstance(decision, Mapping) or not bool(decision.get("teacher_used", False)):
                    continue
                row = dict(record)
                row["teacher_action"] = teacher.get("action")
                if isinstance(summary, Mapping):
                    row.setdefault("episode_id", summary.get("episode_id"))
                    row.setdefault("task_id", summary.get("task_id"))
                    row.setdefault("success_episode", summary.get("success", False))
                yield _transition_row_from_mapping(row)
        return
    if schema.endswith("master_log.v1") or "records" in payload:
        summary = payload.get("attempt", payload.get("summary", {}))
        records = payload.get("records", ())
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ContractError("master-log records must be a sequence")
        for record in records:
            if not isinstance(record, Mapping):
                raise ContractError("master-log record must be a mapping")
            teacher = record.get("teacher")
            decision = record.get("decision", {})
            if not isinstance(teacher, Mapping) or not isinstance(decision, Mapping) or not decision.get("teacher_used", False):
                continue
            row = dict(record)
            row["teacher_action"] = teacher.get("action")
            if isinstance(summary, Mapping):
                row.setdefault("episode_id", summary.get("episode_id"))
                row.setdefault("task_id", summary.get("task_id"))
                row.setdefault("success_episode", summary.get("success", False))
            yield _transition_row_from_mapping(row)
        return
    if "transitions" in payload:
        transitions = payload.get("transitions")
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
            raise ContractError("episode transitions must be a sequence")
        for transition in transitions:
            if not isinstance(transition, Mapping):
                raise ContractError("episode transition must be a mapping")
            source = transition.get("source")
            if source not in {"teacher_correction", "on_call_teacher"}:
                continue
            row = dict(transition)
            row.setdefault("teacher_action", transition.get("teacher_action"))
            row.setdefault("success_episode", payload.get("outcome") in {"teacher_success", "success"})
            row.setdefault("task_id", payload.get("task_id"))
            yield _transition_row_from_mapping(row)
        return
    if "teacher_action" in payload or payload.get("source") in {"teacher_correction", "on_call_teacher"}:
        yield _transition_row_from_mapping(payload)
        return
    raise ContractError("persisted source contains no executed teacher transitions")


def _training_source_row(raw: Mapping[str, Any], *, line_number: int) -> InterventionRow:
    """Normalize one canonical ``training_source.v1`` line.

    The training source is an append-only boundary owned by collection.py. It
    is deliberately stricter than the compatibility readers above: every line
    must be an eligible, executed Arrow On-Call teacher transition, and both
    proposal actions must be present. Digest-only observations are never
    converted into synthetic examples.
    """
    if not isinstance(raw, Mapping) or raw.get("schema") != "arrow_policy_suite.training_source.v1":
        raise ContractError("unsupported training source schema")
    supplied = _verify_training_source_digest(raw)
    if raw.get("eligible") is not True:
        raise ContractError("training source row is ineligible for Apprentice")

    identity = raw.get("identity", {})
    if not isinstance(identity, Mapping):
        raise ContractError("training source identity must be an object")
    task_id = raw.get("task_id", identity.get("task_id"))
    episode_id = raw.get("episode_id", identity.get("episode_id"))
    reset_id = raw.get("reset_id", identity.get("reset_id", ""))
    if task_id in (None, "") or episode_id in (None, "") or reset_id in (None, ""):
        raise ContractError("training source row requires task, reset, and episode identity")
    timestep = raw.get("timestep")
    if isinstance(timestep, bool) or timestep is None:
        raise ContractError("training source row requires a non-negative timestep")
    try:
        timestep = int(timestep)
    except (TypeError, ValueError) as exc:
        raise ContractError("training source timestep must be an integer") from exc
    if timestep < 0:
        raise ContractError("training source timestep must be non-negative")

    observation = raw.get("observation")
    if not isinstance(observation, Mapping):
        raise ContractError("training source row requires a raw observation")
    observation = _strict_observation(observation)
    observation_digest = raw.get("observation_digest")
    if not isinstance(observation_digest, str) or len(observation_digest) != 64:
        raise ContractError("training source row requires observation_digest")
    if digest(observation) != observation_digest:
        raise ContractError("training source observation digest does not match image/state/task bytes")

    base = raw.get("base_proposal")
    teacher = raw.get("teacher_proposal")
    if not isinstance(base, Mapping) or base.get("action") is None:
        raise ContractError("training source row requires base_proposal.action")
    if not isinstance(teacher, Mapping) or teacher.get("action") is None:
        raise ContractError("training source row requires teacher_proposal.action")
    decision = raw.get("decision")
    if not isinstance(decision, Mapping):
        raise ContractError("training source row requires a decision object")
    if str(decision.get("policy_id", "")) != "arrow_on_call" or not bool(decision.get("teacher_used", False)):
        raise ContractError("training source row is not an executed Arrow On-Call teacher transition")
    if str(raw.get("executed_by", "")) not in {"arrow", "hybrid"}:
        raise ContractError("training source row does not record an executed teacher action")
    try:
        base_action = validate_action(base["action"])
        teacher_action = validate_action(teacher["action"])
        executed_action = validate_action(raw.get("executed_action"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("training source proposal/action is invalid") from exc
    if executed_action != teacher_action:
        raise ContractError("training source executed action does not match teacher proposal")

    outcome = raw.get("outcome", {})
    if not isinstance(outcome, Mapping):
        raise ContractError("training source outcome must be an object")
    success = bool(outcome.get("success", outcome.get("task_success", outcome.get("is_success", False))))
    metadata = {
        "source_policy_family": "arrow_on_call",
        "persisted_source": "arrow_policy_suite.training_source.v1",
        "source_line": line_number,
    }
    if isinstance(raw.get("source_hashes"), Mapping):
        metadata["source_hashes"] = dict(raw["source_hashes"])
    return InterventionRow(
        episode_id=str(episode_id), task_id=task_id, timestep=timestep,
        observation=observation, base_action=base_action, teacher_action=teacher_action,
        success_episode=success, source="on_call_teacher", metadata=metadata,
        outcome=dict(outcome), reset_id=str(reset_id), executed_action=executed_action,
        observation_sha256=observation_digest,
        transition_sha256=str(supplied),
    )


def _verify_training_source_digest(raw: Mapping[str, Any]) -> str:
    """Verify the append-only envelope before deciding whether to skip it."""
    supplied = raw.get("transition_sha256")
    unsigned = dict(raw)
    unsigned.pop("transition_sha256", None)
    if not isinstance(supplied, str) or supplied != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ContractError("training source transition digest mismatch")
    return supplied


def _filter_values(values: Sequence[Any] | None) -> set[str] | None:
    if values is None:
        return None
    normalized: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                normalized.add(item)
    if not normalized:
        raise ContractError("Apprentice transition filters cannot be empty")
    return normalized


def _load_json_values(text: str) -> tuple[list[Any], bool]:
    """Read either one JSON value or JSONL, retaining the line-oriented form."""
    try:
        return [json.loads(text)], False
    except json.JSONDecodeError:
        values: list[Any] = []
        for index, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid Apprentice JSON at line {index}") from exc
        return values, True


def load_apprentice_transitions(
    path: str | Path,
    *,
    task_ids: Sequence[int | str] | None = None,
    reset_ids: Sequence[str] | None = None,
    episode_ids: Sequence[str] | None = None,
) -> tuple[InterventionRow, ...]:
    """Load complete On-Call transitions with raw images/state/task/actions.

    Canonical ``arrow_policy_suite.training_source.v1`` JSONL is the strict
    production input. Legacy persisted views remain readable for existing
    artifacts, but digest-only rows and missing camera payloads are rejected.
    """
    source = Path(path)
    if not source.is_file():
        raise ContractError(f"Apprentice transition source is missing: {source}")
    text = source.read_text(encoding="utf-8")
    values, line_oriented = _load_json_values(text)
    flattened = values[0] if len(values) == 1 and isinstance(values[0], list) and not line_oriented else values
    candidates = flattened if isinstance(flattened, list) else [flattened]
    unsupported_training_schemas = [
        value.get("schema") for value in candidates
        if isinstance(value, Mapping)
        and str(value.get("schema", "")).startswith("arrow_policy_suite.training_source.")
        and value.get("schema") != "arrow_policy_suite.training_source.v1"
    ]
    if unsupported_training_schemas:
        raise ContractError(f"unsupported training source schema: {unsupported_training_schemas[0]}")
    is_training_source = any(isinstance(value, Mapping) and value.get("schema") == "arrow_policy_suite.training_source.v1"
                             for value in candidates)
    rows: list[InterventionRow] = []
    if is_training_source:
        if any(not isinstance(value, Mapping) or value.get("schema") != "arrow_policy_suite.training_source.v1"
               for value in candidates):
            raise ContractError("training_source.v1 input cannot be mixed with another persisted schema")
        for index, value in enumerate(candidates, start=1):
            try:
                _verify_training_source_digest(value)
                # Collection writes mixed rollout steps to one source.  Only
                # eligible, executed Arrow On-Call rows are consumed; valid
                # ineligible/VLA rows are intentionally ignored.
                if value.get("eligible") is not True:
                    continue
                decision = value.get("decision")
                if isinstance(decision, Mapping):
                    on_call = (
                        str(decision.get("policy_id", "")) == "arrow_on_call"
                        and bool(decision.get("teacher_used", False))
                        and str(value.get("executed_by", "")) in {"arrow", "hybrid"}
                    )
                    if not on_call:
                        continue
                rows.append(_training_source_row(value, line_number=index))
            except ContractError as exc:
                raise ContractError(f"invalid training source row at line {index}: {exc}") from exc
    else:
        for value in candidates:
            rows.extend(_iter_persisted_rows(value))
    if not rows:
        raise ContractError("Apprentice transition source contains no eligible executed teacher rows")
    allowed_tasks = _filter_values(task_ids)
    allowed_resets = _filter_values(reset_ids)
    allowed_episodes = _filter_values(episode_ids)
    rows = [
        row for row in rows
        if (allowed_tasks is None or str(row.task_id) in allowed_tasks)
        and (allowed_resets is None or str(row.reset_id) in allowed_resets)
        and (allowed_episodes is None or str(row.episode_id) in allowed_episodes)
    ]
    if not rows:
        raise ContractError("Apprentice transition filters selected no eligible executed teacher rows")
    rows.sort(key=lambda row: (str(row.episode_id), row.timestep))
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row.episode_id, row.timestep)
        if key in seen:
            raise ContractError(f"duplicate Apprentice transition {key!r}")
        seen.add(key)
    return tuple(rows)


# Names used by launchers and older orchestration prototypes.
load_teacher_transitions = load_apprentice_transitions
load_on_call_teacher_transitions = load_apprentice_transitions


@dataclass(frozen=True)
class SmolVLALoRAConfig:
    """Pinned action-side PEFT values for the native SmolVLA adapter."""

    model_id: str = DEFAULT_SMOLVLA_MODEL
    model_revision: str = "unresolved"
    processor_revision: str = "unresolved"
    rank: int = 16
    alpha: int = 8
    dropout: float = 0.0
    bias: str = "none"
    target_modules: str = DEFAULT_SMOLVLA_PEFT_TARGET_REGEX
    modules_to_save: tuple[str, ...] = ()
    base_vla_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.model_id or self.rank <= 0 or self.alpha <= 0:
            raise ContractError("SmolVLA model identity and positive LoRA rank/alpha are required")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ContractError("LoRA bias must be none, all, or lora_only")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ContractError("LoRA dropout must lie in [0, 1)")
        if not self.target_modules:
            raise ContractError("native SmolVLA LoRA requires explicit target modules")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": APPRENTICE_SCHEMA,
            "provider": "huggingface/lerobot/peft",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": float(self.dropout),
            "bias": self.bias,
            "target_modules": self.target_modules,
            "modules_to_save": list(self.modules_to_save),
            "base_vla_sha256": self.base_vla_sha256,
            "base_hash_convention": "tree_sha256_len_prefixed_v1_excluding_cache_and_base_snapshot_manifest",
        }


@dataclass(frozen=True)
class ApprenticeTrainingConfig:
    """Native training values; execution is owned by the injected trainer."""

    seed: int = 1000
    effective_batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_fraction_denominator: int = 30
    max_grad_norm: float = 10.0
    mixed_precision: str = "no"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.effective_batch_size <= 0 or self.epochs <= 0:
            raise ContractError("invalid SmolVLA training seed, batch size, or epoch count")
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0:
            raise ContractError("SmolVLA learning_rate must be positive and weight_decay non-negative")
        if self.warmup_fraction_denominator <= 0 or float(self.max_grad_norm) <= 0.0:
            raise ContractError("invalid scheduler or gradient clipping configuration")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ContractError("mixed_precision must be no, fp16, or bf16")

    def derived_steps(self, frames: int) -> int:
        if frames <= 0:
            raise ContractError("exported dataset must contain at least one frame")
        return max(1, (self.epochs * frames + self.effective_batch_size - 1) // self.effective_batch_size)

    def manifest(self, *, frames: int) -> dict[str, Any]:
        steps = self.derived_steps(frames)
        return {
            "seed": self.seed,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "warmup_steps": steps // self.warmup_fraction_denominator,
            "optimizer_steps": steps,
            "max_grad_norm": float(self.max_grad_norm),
            "mixed_precision": self.mixed_precision,
        }


@dataclass(frozen=True)
class ApprenticeExportRequest:
    """Rows and provenance passed to a native LeRobot exporter."""

    rows: tuple[InterventionRow, ...]
    parent_manifest: DatasetManifest
    destination: str
    variant: str = "apprentice_teacher_actions"
    state_key: str = "observation.state"
    action_key: str = "action"

    def __post_init__(self) -> None:
        if not self.rows:
            raise ContractError("Apprentice export requires at least one executed teacher row")
        if not self.destination or self.variant != "apprentice_teacher_actions":
            raise ContractError("Apprentice export destination/variant is invalid")
        if self.parent_manifest.rows != len(self.rows):
            raise ContractError("parent manifest row count does not match export rows")
        if not self.parent_manifest.content_sha256:
            raise ContractError("parent manifest content hash is required")
        seen: set[tuple[str, int]] = set()
        for row in self.rows:
            if row.source != "on_call_teacher" or row.metadata.get("source_policy_family", "arrow_on_call") != "arrow_on_call":
                raise ContractError("Apprentice rows must be executed On-Call teacher rows")
            identity = (row.episode_id, row.timestep)
            if identity in seen:
                raise ContractError(f"duplicate source row {identity!r}")
            seen.add(identity)

    @property
    def episodes(self) -> tuple[str, ...]:
        return tuple(sorted({row.episode_id for row in self.rows}))

    def native_rows(self) -> tuple[dict[str, Any], ...]:
        """Return only observation + teacher action to the native BC exporter."""
        return tuple(
            {
                "episode_id": row.episode_id,
                "task_id": row.task_id,
                "timestep": row.timestep,
                "observation": row.observation,
                self.action_key: list(row.teacher_action),
            }
            for row in self.rows
        )

    def lineage(self) -> dict[str, Any]:
        payload = {
            "schema": APPRENTICE_SCHEMA,
            "parent_manifest_sha256": self.parent_manifest.content_sha256,
            "parent_artifact": self.parent_manifest.parent_artifact,
            "rows": len(self.rows),
            "episodes": list(self.episodes),
            "destination": self.destination,
            "variant": self.variant,
            "state_key": self.state_key,
            "action_key": self.action_key,
            "teacher_source": "executed_on_call_teacher",
            "outcomes": {"success_rows": sum(bool(row.success_episode) for row in self.rows),
                          "failure_rows": sum(not bool(row.success_episode) for row in self.rows)},
            "base_actions_exported": False,
        }
        encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**payload, "request_sha256": hashlib.sha256(encoded).hexdigest()}


@dataclass(frozen=True)
class ApprenticeExportReceipt:
    request: ApprenticeExportRequest
    dataset_path: str
    dataset_manifest_sha256: str
    frames: int
    native_result: Any

    def __post_init__(self) -> None:
        if not self.dataset_path or self.frames <= 0 or not self.dataset_manifest_sha256:
            raise ContractError("native export must return path, positive frame count, and dataset hash")

    def lineage(self) -> dict[str, Any]:
        return {
            **self.request.lineage(),
            "dataset_path": str(Path(self.dataset_path)),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "frames": self.frames,
        }


class NativeSmolVLAExporter(Protocol):
    def __call__(self, request: ApprenticeExportRequest) -> Mapping[str, Any]: ...


class NativeSmolVLATrainer(Protocol):
    def __call__(self, receipt: ApprenticeExportReceipt, *, lora: SmolVLALoRAConfig,
                 training: ApprenticeTrainingConfig) -> Any: ...


def make_export_request(
    rows: Sequence[InterventionRow],
    parent_manifest: DatasetManifest,
    *,
    destination: str,
) -> ApprenticeExportRequest:
    """Build a fail-closed request; does not write data or import native code."""

    return ApprenticeExportRequest(tuple(rows), parent_manifest, destination)


def export_apprentice_dataset(
    request: ApprenticeExportRequest,
    exporter: NativeSmolVLAExporter,
) -> ApprenticeExportReceipt:
    """Invoke the caller's native exporter and check returned lineage fields."""

    result = exporter(request)
    if not isinstance(result, Mapping):
        raise ContractError("native exporter must return a mapping")
    parent_hash = result.get("parent_manifest_sha256")
    if parent_hash is not None and str(parent_hash) != request.parent_manifest.content_sha256:
        raise ContractError("native exporter returned the wrong parent manifest hash")
    try:
        return ApprenticeExportReceipt(
            request=request,
            dataset_path=str(result["dataset_path"]),
            dataset_manifest_sha256=str(result["dataset_manifest_sha256"]),
            frames=int(result["frames"]),
            native_result=result.get("native_result"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("native exporter result lacks required dataset identity fields") from exc


def export_apprentice_dataset_native(
    request: ApprenticeExportRequest,
    *,
    dataset_factory: Callable[..., Any] | None = None,
    fps: int = 10,
    robot_type: str = "libero",
) -> ApprenticeExportReceipt:
    """Export rows through the installed LeRobot dataset API.

    Imports are intentionally lazy.  A caller can inject ``dataset_factory``
    for a pinned LeRobot version or tests; otherwise the function resolves
    ``LeRobotDataset`` at call time and refuses to guess a remote repository.
    """
    if fps <= 0 or not robot_type:
        raise ContractError("native dataset export requires positive fps and robot_type")
    if Path(request.destination).exists():
        raise ContractError(f"refusing to overwrite existing Apprentice dataset: {request.destination}")
    factory = dataset_factory
    if factory is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        except ImportError as exc:
            raise ContractError("LeRobot is required for native Apprentice export") from exc
        factory = LeRobotDataset
    destination = Path(request.destination)
    strict_rows = []
    for row in request.rows:
        observation = _strict_observation(row.observation)
        task = row.metadata.get("task", row.task_id)
        if task in (None, ""):
            raise ContractError("Apprentice transition requires a non-empty task")
        strict_rows.append((row, observation, str(task)))
    # LeRobot versions differ in their feature declaration.  Prefer the
    # modern create() API, while keeping the factory injectable for a pinned
    # installation.
    first_agentview = strict_rows[0][1]["agentview"]
    first_wrist = strict_rows[0][1]["wrist"]
    features = {
        request.action_key: {"dtype": "float32", "shape": (7,)},
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "observation.images.image": {
            "dtype": "image", "shape": tuple(int(x) for x in first_agentview.shape),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "image", "shape": tuple(int(x) for x in first_wrist.shape),
            "names": ["height", "width", "channel"],
        },
    }
    try:
        dataset = factory.create(repo_id=f"local/{destination.name}", root=destination,
                                 fps=fps, robot_type=robot_type, features=features)
    except TypeError:
        dataset = factory.create(repo_id=f"local/{destination.name}", root=destination, fps=fps, features=features)
    current_episode: str | None = None
    episode_count = 0
    for row, observation, task in strict_rows:
        if current_episode is not None and row.episode_id != current_episode:
            save_episode = getattr(dataset, "save_episode", None)
            if not callable(save_episode):
                raise ContractError("LeRobot dataset object lacks save_episode")
            save_episode()
            episode_count += 1
        current_episode = row.episode_id
        frame = {
            request.action_key: list(row.teacher_action),
            "task": task,
            "observation.images.image": observation["agentview"],
            "observation.images.image2": observation["wrist"],
            "observation.state": list(_state_payload(observation)),
        }
        add_frame = getattr(dataset, "add_frame", None)
        if not callable(add_frame):
            raise ContractError("LeRobot dataset object lacks add_frame")
        add_frame(frame)
    save_episode = getattr(dataset, "save_episode", None)
    if callable(save_episode):
        save_episode()
        episode_count += 1
    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    manifest_path = destination / "meta" / "info.json"
    lineage_path = destination / "meta" / "apprentice_lineage.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_payload = {
        "schema": APPRENTICE_DATASET_SCHEMA,
        "request": request.lineage(),
        "parent_manifest_sha256": request.parent_manifest.content_sha256,
        "frames": len(strict_rows),
        "episodes": episode_count,
        "camera_features": ["observation.images.image", "observation.images.image2"],
        "state_feature": "observation.state",
        "action_feature": request.action_key,
        "task_feature": "task",
        "teacher_actions_only": True,
        "base_actions_exported": False,
    }
    _create_immutable_json(lineage_path, lineage_payload)
    dataset_hash = hashlib.sha256()
    if manifest_path.exists():
        dataset_hash.update(manifest_path.read_bytes())
    else:
        dataset_hash.update(_canonical(lineage_payload))
    dataset_hash.update(lineage_path.read_bytes())
    result = {"dataset_path": str(destination), "dataset_manifest_sha256": dataset_hash.hexdigest(),
              "frames": len(request.rows), "parent_manifest_sha256": request.parent_manifest.content_sha256,
              "native_result": {"provider": "lerobot", "fps": fps, "robot_type": robot_type}}
    return export_apprentice_dataset(request, lambda _request: result)


def train_apprentice(
    receipt: ApprenticeExportReceipt,
    trainer: NativeSmolVLATrainer,
    *,
    lora: SmolVLALoRAConfig | None = None,
    training: ApprenticeTrainingConfig | None = None,
) -> dict[str, Any]:
    """Invoke native SmolVLA training only through an explicit injected hook."""

    resolved_lora = lora or SmolVLALoRAConfig()
    resolved_training = training or ApprenticeTrainingConfig()
    native_result = trainer(receipt, lora=resolved_lora, training=resolved_training)
    return {
        "schema": APPRENTICE_SCHEMA,
        "model": resolved_lora.manifest(),
        "training": resolved_training.manifest(frames=receipt.frames),
        "dataset": receipt.lineage(),
        "native_result": native_result,
    }


@dataclass(frozen=True)
class ApprenticeTrainingJob:
    """Resolved deterministic LeRobot/PEFT job command and provenance."""

    command: tuple[str, ...]
    environment: Mapping[str, str]
    output_dir: str
    manifest: Mapping[str, Any]


def build_apprentice_training_job(
    receipt: ApprenticeExportReceipt,
    *,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    lora: SmolVLALoRAConfig | None = None,
    training: ApprenticeTrainingConfig | None = None,
    python_executable: str | Path | None = None,
    trainer_module: str = "vla_benchmarking.libero.finetuned_vlas.smolvla.workflows.run_lerobot_train",
    device: str = "cpu",
) -> ApprenticeTrainingJob:
    """Build a real, deterministic LeRobot PEFT command without executing it.

    The command always starts from the explicitly supplied frozen base policy;
    the only trainable parameters are the PEFT adapter selected by ``peft.r``.
    Output directories are create-only so a prior checkpoint cannot be silently
    replaced.
    """
    resolved_lora = lora or SmolVLALoRAConfig()
    resolved_training = training or ApprenticeTrainingConfig()
    base = Path(base_checkpoint).expanduser()
    target = Path(output_dir).expanduser()
    if target.exists():
        raise ContractError(f"refusing to overwrite existing Apprentice training output: {target}")
    if not str(base_checkpoint).strip():
        raise ContractError("frozen base checkpoint is required")
    if resolved_lora.base_vla_sha256:
        expected_base_hash = _require_sha256(resolved_lora.base_vla_sha256, name="base_vla_sha256")
    else:
        raise ContractError("frozen base requires an explicit base_vla_sha256")
    if resolved_lora.model_revision in {"", "unresolved"} or resolved_lora.processor_revision in {"", "unresolved"}:
        raise ContractError("SmolVLA model and processor revisions must be pinned for a training job")
    if base.exists() and not base.is_dir():
        raise ContractError("frozen base checkpoint must be a snapshot directory")
    if base.is_dir():
        _validate_base_snapshot_manifest(base)
        actual_base_hash = _base_tree_sha256(base)
        if actual_base_hash != expected_base_hash:
            raise ContractError("frozen base checkpoint tree hash does not match base_vla_sha256")
    if device not in {"cpu", "cuda", "mps"}:
        raise ContractError("Apprentice training device must be cpu, cuda, or mps")
    executable = str(python_executable or sys.executable)
    steps = resolved_training.derived_steps(receipt.frames)
    dataset_root = str(Path(receipt.dataset_path).expanduser().resolve())
    target_root = str(target.resolve())
    command = (
        executable, "-m", trainer_module,
        "--policy.push_to_hub=false",
        f"--policy.path={str(base_checkpoint)}",
        f"--peft.r={resolved_lora.rank}",
        f"--dataset.repo_id=local/{Path(receipt.dataset_path).name}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={target_root}",
        f"--steps={steps}",
        f"--save_freq={steps}",
        "--eval_freq=0",
        f"--batch_size={resolved_training.effective_batch_size}",
        f"--policy.device={device}",
        f"--seed={resolved_training.seed}",
    )
    environment = {
        "PYTHONHASHSEED": str(resolved_training.seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
    }
    manifest = {
        "schema": APPRENTICE_SCHEMA + ".training_job",
        "status": "configured",
        "base_checkpoint": str(base_checkpoint),
        "base_vla_sha256": expected_base_hash,
        "base_hash_convention": "tree_sha256_len_prefixed_v1_excluding_cache_and_base_snapshot_manifest",
        "base_frozen": True,
        "adapter_only": True,
        "dataset": receipt.lineage(),
        "model": resolved_lora.manifest(),
        "training": resolved_training.manifest(frames=receipt.frames),
        "command": list(command),
        "environment": dict(environment),
        "device": device,
    }
    return ApprenticeTrainingJob(command, environment, target_root, manifest)


def run_apprentice_training_job(
    job: ApprenticeTrainingJob,
    *,
    runner: Callable[..., Mapping[str, Any]] | None = None,
    cwd: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute a resolved job once and emit an immutable training manifest."""
    output = Path(job.output_dir)
    if output.exists():
        raise ContractError(f"refusing to overwrite existing Apprentice training output: {output}")
    if runner is None:
        env = os.environ.copy()
        env.update(job.environment)
        completed = subprocess.run(job.command, cwd=cwd, env=env, capture_output=True,
                                   text=True, timeout=timeout, check=False)
        if completed.returncode != 0:
            raise ContractError(f"Apprentice training failed with exit code {completed.returncode}: {completed.stderr[-2000:]}")
        result: Mapping[str, Any] = {
            "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
        }
    else:
        result = runner(job.command, cwd=cwd, env=dict(job.environment), timeout=timeout)
        if not isinstance(result, Mapping):
            raise ContractError("Apprentice training runner must return a mapping")
    if output.exists():
        if not output.is_dir():
            raise ContractError(f"Apprentice training output is not a directory: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    explicit_adapter = result.get("adapter_checkpoint")
    if explicit_adapter:
        adapter_candidates = [Path(str(explicit_adapter)).expanduser()]
    else:
        adapter_candidates = sorted(output.rglob("adapter_model.safetensors"))
    if len(adapter_candidates) != 1:
        raise ContractError(
            "Apprentice training did not produce exactly one adapter_model.safetensors checkpoint"
        )
    adapter_path = adapter_candidates[0]
    adapter_bundle = adapter_path.parent if adapter_path.name == "adapter_model.safetensors" else adapter_path
    adapter_manifest_path = adapter_bundle / "apprentice_manifest.json" if adapter_bundle.is_dir() else Path(str(adapter_bundle) + ".json")
    manifest = {
        **dict(job.manifest), "status": "completed", "native_result": dict(result),
        "adapter_checkpoint": str(adapter_path), "adapter_manifest_path": str(adapter_manifest_path),
    }
    # If the trainer placed the adapter directly in output/, the training
    # receipt is part of the bundle inventory and must exist before that
    # inventory is sealed.  Nested LeRobot outputs can seal first because the
    # receipt is outside the adapter directory.
    training_manifest_path = output / "apprentice_training_manifest.json"
    if training_manifest_path.parent.resolve() == adapter_bundle.resolve():
        _create_immutable_json(training_manifest_path, manifest)
    adapter_manifest = write_adapter_manifest(
        adapter_bundle,
        base_vla_sha256=str(job.manifest["base_vla_sha256"]),
        model=job.manifest.get("model", {}), dataset=job.manifest.get("dataset", {}),
        training=job.manifest.get("training", {}),
        base_checkpoint=job.manifest.get("base_checkpoint"),
        processor_provenance={
            "processor_revision": dict(job.manifest.get("model", {})).get("processor_revision", ""),
            "selection_order": ["adapter", "base"],
            "base_checkpoint": str(job.manifest.get("base_checkpoint", "")),
        },
    )
    if not training_manifest_path.exists():
        _create_immutable_json(training_manifest_path, {**manifest, "adapter_manifest": adapter_manifest})
    manifest["adapter_manifest"] = adapter_manifest
    return manifest


def _artifact_inventory(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    if not path.is_dir():
        raise ContractError(f"adapter artifact is missing: {path}")
    inventory: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.relative_to(path).as_posix() != "apprentice_manifest.json":
            inventory[str(item.relative_to(path)).replace("\\", "/")] = hashlib.sha256(item.read_bytes()).hexdigest()
    if not inventory:
        raise ContractError(f"adapter artifact directory is empty: {path}")
    return inventory


def _inventory_hash(inventory: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical(dict(sorted(inventory.items())))).hexdigest()


def _normalized_peft_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(sorted(str(item) for item in value))
    raise ContractError("PEFT config value must be a string or sequence")


def _same_model_reference(actual: str, expected: Sequence[str]) -> bool:
    actual_text = str(actual)
    for candidate in expected:
        candidate_text = str(candidate)
        if actual_text == candidate_text:
            return True
        try:
            if Path(actual_text).exists() and Path(candidate_text).exists():
                if Path(actual_text).expanduser().resolve() == Path(candidate_text).expanduser().resolve():
                    return True
        except OSError:
            pass
    return False


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _processor_inventory(target: Path, expected_revision: str) -> dict[str, Any]:
    files: list[Path] = []
    preprocessor: list[Path] = []
    postprocessor: list[Path] = []
    for item in sorted(target.rglob("*")):
        if not item.is_file() or item.name == "apprentice_manifest.json":
            continue
        lowered = item.name.lower()
        if "processor" not in lowered:
            continue
        files.append(item)
        if "postprocessor" in lowered:
            postprocessor.append(item)
        elif "preprocessor" in lowered:
            preprocessor.append(item)
    if not preprocessor or not postprocessor:
        raise ContractError("adapter bundle must contain checkpoint-owned preprocessor and postprocessor files")
    revision_hits: list[str] = []
    for item in files:
        try:
            raw = item.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ContractError(f"processor provenance file is unreadable: {item.name}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        explicit_values: list[str] = []
        def collect_revision_fields(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    if str(key) in {"revision", "processor_revision", "_commit_hash"} and isinstance(nested, str):
                        explicit_values.append(nested)
                    collect_revision_fields(nested)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for nested in value:
                    collect_revision_fields(nested)
        collect_revision_fields(payload)
        if any(value != expected_revision for value in explicit_values):
            raise ContractError(f"processor revision field contradicts sealed provenance: {item.name}")
        if explicit_values:
            revision_hits.append(str(item.relative_to(target)).replace("\\", "/"))
    return {
        "preprocessor": {
            str(item.relative_to(target)).replace("\\", "/"): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in preprocessor
        },
        "postprocessor": {
            str(item.relative_to(target)).replace("\\", "/"): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in postprocessor
        },
        "revision": expected_revision,
        "revision_binding": "sealed_training_job_input",
        "revision_evidence": sorted(revision_hits),
    }


def _adapter_weight_keys(path: Path) -> tuple[str, ...]:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional native dependency
            raise ContractError("safetensors is required to audit adapter weights") from exc
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return tuple(sorted(str(key) for key in handle.keys()))
        except Exception as exc:
            raise ContractError("adapter safetensors weights are unreadable") from exc
    try:
        import torch  # type: ignore
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, TypeError) as exc:
        raise ContractError("adapter state weights are unreadable") from exc
    if not isinstance(state, Mapping):
        raise ContractError("adapter state weights must contain a mapping")
    return tuple(sorted(str(key) for key in state))


def _audit_adapter_bundle(
    target: Path,
    *,
    model: Mapping[str, Any],
    base_checkpoint: str | Path | None,
    processor_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the produced PEFT files, not only the requested job settings."""
    if not target.is_dir():
        raise ContractError("PEFT adapter audit requires a directory bundle")
    config_path = target / "adapter_config.json"
    if not config_path.is_file():
        raise ContractError("adapter bundle is missing adapter_config.json")
    config = _read_json_mapping(config_path, label="adapter_config.json")
    expected_model = dict(model)
    expected_revision = str(expected_model.get("processor_revision", processor_provenance.get("processor_revision", "")))
    model_revision = str(expected_model.get("model_revision", ""))
    if not model_revision or model_revision == "unresolved" or not expected_revision or expected_revision == "unresolved":
        raise ContractError("adapter audit requires pinned model and processor revisions")
    if str(processor_provenance.get("processor_revision", expected_revision)) != expected_revision:
        raise ContractError("processor provenance revision does not match the requested revision")
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        raise ContractError("adapter config is not a LoRA PEFT config")
    expected_targets = _normalized_peft_value(expected_model.get("target_modules"))
    actual_targets = _normalized_peft_value(config.get("target_modules"))
    expected_modules_to_save = _normalized_peft_value(expected_model.get("modules_to_save"))
    actual_modules_to_save = _normalized_peft_value(config.get("modules_to_save"))
    try:
        checks = (
            ("r", int(config.get("r", -1)), int(expected_model.get("rank", -2))),
            ("lora_alpha", int(config.get("lora_alpha", -1)), int(expected_model.get("alpha", -2))),
            ("lora_dropout", float(config.get("lora_dropout", -1.0)), float(expected_model.get("dropout", -2.0))),
            ("bias", str(config.get("bias", "")), str(expected_model.get("bias", ""))),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("adapter config contains invalid LoRA settings") from exc
    for name, actual, expected in checks:
        if actual != expected:
            raise ContractError(f"adapter config {name} does not match requested LoRA settings")
    if actual_targets != expected_targets:
        raise ContractError("adapter config target_modules do not match requested SmolVLA targets")
    if actual_modules_to_save != expected_modules_to_save:
        raise ContractError("adapter config modules_to_save do not match requested settings")
    actual_base = config.get("base_model_name_or_path")
    expected_bases = [str(value) for value in (base_checkpoint, expected_model.get("model_id")) if value not in (None, "")]
    if not isinstance(actual_base, str) or not _same_model_reference(actual_base, expected_bases):
        raise ContractError("adapter config base model does not match the frozen training base")
    config_revision = config.get("revision", config.get("model_revision", config.get("_commit_hash")))
    if config_revision is not None and str(config_revision) != model_revision:
        raise ContractError("adapter config model revision does not match the requested revision")
    weight_path = target / "adapter_model.safetensors"
    if not weight_path.is_file():
        weight_path = target / "adapter_model.bin"
    if not weight_path.is_file():
        raise ContractError("adapter bundle is missing adapter weights")
    weight_keys = _adapter_weight_keys(weight_path)
    if not weight_keys or any(not any(marker in key.lower() for marker in ("lora_a", "lora_b", "lora_embedding", "modules_to_save")) for key in weight_keys):
        raise ContractError("adapter weights contain full-base or non-LoRA parameters")
    processor = _processor_inventory(target, expected_revision)
    return {
        "schema": APPRENTICE_SCHEMA + ".adapter_audit",
        "adapter_config": {
            "path": "adapter_config.json",
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "weights": {
            "path": str(weight_path.relative_to(target)).replace("\\", "/"),
            "sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
            "keys": list(weight_keys),
            "key_count": len(weight_keys),
        },
        "peft": {
            "r": int(config["r"]), "lora_alpha": int(config["lora_alpha"]),
            "lora_dropout": float(config["lora_dropout"]), "bias": str(config["bias"]),
            "target_modules": list(actual_targets), "modules_to_save": list(actual_modules_to_save),
            "base_model_name_or_path": str(actual_base), "model_revision": model_revision,
        },
        "no_full_base_weights": True,
        "processor": processor,
    }


def _tree_sha256(root: Path, *, exclude: Callable[[Path], bool] | None = None) -> str:
    """Canonical project tree hash: length-prefixed relative path + bytes."""
    if not root.is_dir():
        raise ContractError(f"base checkpoint tree is not a directory: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root)
        if exclude is not None and exclude(relative):
            continue
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _base_tree_sha256(root: Path) -> str:
    """Hash base snapshots exactly as prepare_base_snapshot.sh does."""
    return _tree_sha256(
        root,
        exclude=lambda relative: relative.name == "base_snapshot_manifest.json" or ".cache" in relative.parts,
    )


def _validate_base_snapshot_manifest(root: Path) -> None:
    """Validate the optional producer manifest without hashing that manifest."""
    manifest_path = root / "base_snapshot_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("frozen base snapshot manifest is unreadable") from exc
    files = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise ContractError("frozen base snapshot manifest has no file inventory")
    actual_names = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "base_snapshot_manifest.json" and ".cache" not in item.parts
    }
    if actual_names != {str(name) for name in files}:
        raise ContractError("frozen base snapshot file inventory differs from its manifest")
    for name, expected in files.items():
        candidate = root / str(name)
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != str(expected).lower():
            raise ContractError(f"frozen base snapshot file hash mismatch: {name}")


def write_adapter_manifest(
    checkpoint: str | Path,
    *,
    base_vla_sha256: str,
    model: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    base_checkpoint: str | Path | None = None,
    processor_provenance: Mapping[str, Any] | None = None,
    adapter_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a LeRobot adapter directory/file without copying or overwriting it."""
    base_hash = _require_sha256(base_vla_sha256, name="base_vla_sha256")
    target = Path(checkpoint).expanduser()
    inventory = _artifact_inventory(target)
    resolved_model = dict(model or {})
    resolved_processor = dict(processor_provenance or {"selection_order": ["adapter", "base"]})
    if target.is_dir():
        resolved_audit = dict(adapter_audit or _audit_adapter_bundle(
            target, model=resolved_model, base_checkpoint=base_checkpoint,
            processor_provenance=resolved_processor,
        ))
    else:
        resolved_audit = None
    manifest = {
        "schema": APPRENTICE_ADAPTER_SCHEMA,
        "checkpoint_path": str(target),
        "checkpoint_inventory": inventory,
        "checkpoint_sha256": _inventory_hash(inventory),
        "inventory_sha256": _inventory_hash(inventory),
        "adapter_sha256": hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else inventory.get("adapter_model.safetensors", ""),
        "base_vla_sha256": base_hash,
        "base_hash_convention": "tree_sha256_len_prefixed_v1_excluding_cache_and_base_snapshot_manifest",
        "base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else "",
        "adapter_only": True,
        "base_frozen": True,
        "model": resolved_model,
        "dataset": dict(dataset or {}),
        "training": dict(training or {}),
        "processor_provenance": resolved_processor,
    }
    if resolved_audit is not None:
        manifest["adapter_audit"] = resolved_audit
    # Directory checkpoints are the canonical CLI/SLURM bundle.  Keep the
    # historical sidecar only for single-file checkpoints used by lightweight
    # tests and older learned-policy artifacts.
    manifest_path = target / "apprentice_manifest.json" if target.is_dir() else Path(str(target) + ".json")
    _create_immutable_json(manifest_path, manifest)
    return manifest


def save_adapter_checkpoint(
    path: str | Path,
    adapter_state: Mapping[str, Any],
    *,
    base_vla_sha256: str,
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save adapter-only state once and emit a hashable verification record."""
    base_vla_sha256 = _require_sha256(base_vla_sha256, name="base_vla_sha256")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ContractError("adapter_state must be a non-empty mapping")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # torch.save is used only when tensors are present; JSON keeps the module
    # useful in lightweight environments and in contract tests.
    try:
        import torch  # type: ignore
        buffer = io.BytesIO()
        torch.save(dict(adapter_state), buffer)
        payload = buffer.getvalue()
        serialization = "torch"
    except (ImportError, RuntimeError, TypeError):
        payload = (_canonical(adapter_state))
        serialization = "json"
    if target.exists():
        raise ContractError(f"refusing to overwrite immutable adapter checkpoint: {target}")
    target.write_bytes(payload)
    adapter_hash = hashlib.sha256(payload).hexdigest()
    manifest = {"schema": APPRENTICE_ADAPTER_SCHEMA, "checkpoint_path": str(target),
                "adapter_sha256": adapter_hash, "base_vla_sha256": base_vla_sha256,
                "adapter_only": True, "base_frozen": True, "serialization": serialization,
                "keys": sorted(str(key) for key in adapter_state), "training": dict(training or {})}
    manifest_path = Path(str(target) + ".json")
    if manifest_path.exists():
        raise ContractError(f"refusing to overwrite immutable adapter manifest: {manifest_path}")
    manifest_path.write_bytes(_canonical(manifest))
    return manifest


def verify_adapter_reload(path: str | Path, *, base_vla_sha256: str,
                          loader: Callable[[Path], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Reload an adapter and verify bytes, frozen-base identity, and keys."""
    target = Path(path).expanduser()
    base_vla_sha256 = _require_sha256(base_vla_sha256, name="base_vla_sha256")
    manifest_path = target / "apprentice_manifest.json" if target.is_dir() else Path(str(target) + ".json")
    if target.is_dir() and not manifest_path.is_file():
        # Read legacy directory sidecars only to provide a controlled upgrade
        # path; all newly produced bundles use the in-directory contract.
        legacy = Path(str(target) + ".json")
        if legacy.is_file():
            manifest_path = legacy
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = None if target.is_dir() else target.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("adapter checkpoint or manifest is unreadable") from exc
    if manifest.get("base_vla_sha256") != base_vla_sha256 or manifest.get("adapter_only") is not True or manifest.get("base_frozen") is not True:
        raise ContractError("adapter reload failed frozen-base/adapter-only verification")
    if target.is_dir():
        inventory = _artifact_inventory(target)
        expected_inventory_hash = manifest.get("inventory_sha256", manifest.get("checkpoint_sha256"))
        if inventory != manifest.get("checkpoint_inventory") or _inventory_hash(inventory) != expected_inventory_hash:
            raise ContractError("adapter checkpoint inventory changed")
        if not isinstance(manifest.get("processor_provenance"), Mapping):
            raise ContractError("adapter manifest is missing processor provenance")
        recorded_audit = manifest.get("adapter_audit")
        if not isinstance(recorded_audit, Mapping):
            raise ContractError("adapter manifest is missing PEFT adapter audit")
        recorded_model = manifest.get("model")
        if not isinstance(recorded_model, Mapping):
            raise ContractError("adapter manifest is missing model provenance")
        if recorded_model.get("base_vla_sha256") not in (None, "", base_vla_sha256):
            raise ContractError("adapter model provenance base hash differs from manifest base hash")
        actual_audit = _audit_adapter_bundle(
            target,
            model=manifest.get("model", {}) if isinstance(manifest.get("model"), Mapping) else {},
            base_checkpoint=manifest.get("base_checkpoint") or None,
            processor_provenance=manifest["processor_provenance"],
        )
        if dict(recorded_audit) != actual_audit:
            raise ContractError("adapter PEFT audit changed after training")
        if loader is None:
            raise ContractError("directory adapter reload requires an explicit native loader")
        state = loader(target)
        if not isinstance(state, Mapping):
            raise ContractError("native adapter loader must return a mapping for directory verification")
        return {**manifest, "reloaded": True}
    if payload is None or hashlib.sha256(payload).hexdigest() != manifest.get("adapter_sha256"):
        raise ContractError("adapter checkpoint hash changed")
    if loader is None:
        if manifest.get("serialization") == "torch":
            try:
                import torch  # type: ignore
                state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            except (ImportError, RuntimeError, TypeError) as exc:
                raise ContractError("torch adapter reload unavailable") from exc
        else:
            state = json.loads(payload.decode("utf-8"))
    else:
        state = loader(target)
    if not isinstance(state, Mapping) or sorted(str(key) for key in state) != list(manifest.get("keys", ())):
        raise ContractError("reloaded adapter keys differ from saved adapter")
    return {**manifest, "reloaded": True}


def _call_adapter_loader(loader: Callable[..., Any], path: Path, base_hash: str) -> Any:
    try:
        signature = inspect.signature(loader)
        parameters = tuple(signature.parameters.values())
        accepts_two = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters) or len(parameters) >= 2
    except (TypeError, ValueError):
        accepts_two = False
    return loader(path, base_hash) if accepts_two else loader(path)


def _runtime_payload(frame: ObservationFrame) -> dict[str, Any]:
    raw_observation = dict(frame.observation)
    if "instruction" not in raw_observation and frame.metadata.get("task") not in (None, ""):
        raw_observation["instruction"] = frame.metadata["task"]
    observation = _strict_observation(raw_observation)
    instruction = observation.get("instruction", frame.metadata.get("task", ""))
    if instruction in (None, ""):
        raise ContractError("Apprentice runtime frame is missing task/instruction")
    return {
        "observation.images.image": observation["agentview"],
        "observation.images.image2": observation["wrist"],
        "observation.state": list(_state_payload(observation)),
        "task": str(instruction),
    }


def _coerce_runtime_action(value: Any) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    # SmolVLA may return a one-item action chunk or a batch dimension.
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return validate_action(value)


def load_native_smolvla_adapter(
    checkpoint_path: str | Path,
    *,
    base_checkpoint: str | Path,
    base_vla_sha256: str,
    device: str = "cpu",
) -> Mapping[str, Any]:
    """Load a local LeRobot SmolVLA base plus PEFT adapter lazily.

    This follows LeRobot 0.5.2's native PEFT path: ``PeftConfig`` identifies
    the adapter, ``SmolVLAPolicy.from_pretrained(..., local_files_only=True)``
    loads the frozen base, and ``PeftModel.from_pretrained(...,
    is_trainable=False)`` attaches inference-only LoRA weights.  Processors
    are read from the adapter checkpoint first, with the frozen base as an
    explicit compatibility fallback.
    """
    adapter = Path(checkpoint_path).expanduser().resolve()
    base = Path(base_checkpoint).expanduser().resolve()
    expected_hash = _require_sha256(base_vla_sha256, name="base_vla_sha256")
    if not base.is_dir():
        raise ContractError("native SmolVLA loader requires a local frozen base snapshot directory")
    _validate_base_snapshot_manifest(base)
    if _base_tree_sha256(base) != expected_hash:
        raise ContractError("native SmolVLA base snapshot hash mismatch")
    if device not in {"cpu", "cuda", "mps"}:
        raise ContractError("native SmolVLA loader device must be cpu, cuda, or mps")
    if not adapter.is_dir():
        raise ContractError("native SmolVLA loader requires an adapter directory")
    verify_adapter_reload(adapter, base_vla_sha256=expected_hash, loader=lambda _path: {})
    try:
        from peft import PeftConfig, PeftModel  # type: ignore
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional native runtime
        raise ContractError("native SmolVLA loader requires LeRobot SmolVLA and PEFT") from exc
    try:
        try:
            peft_config = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
        except TypeError:
            peft_config = PeftConfig.from_pretrained(str(adapter))
        base_loader = getattr(SmolVLAPolicy, "from_pretrained", None)
        if not callable(base_loader):
            raise TypeError("SmolVLAPolicy.from_pretrained is unavailable")
        try:
            policy = base_loader(pretrained_name_or_path=str(base), local_files_only=True)
        except TypeError:
            policy = base_loader(str(base), local_files_only=True)
        policy_config = getattr(policy, "config", None)
        try:
            policy = PeftModel.from_pretrained(
                policy, str(adapter), config=peft_config, is_trainable=False, local_files_only=True
            )
        except TypeError:
            policy = PeftModel.from_pretrained(policy, str(adapter), config=peft_config, is_trainable=False)
        to = getattr(policy, "to", None)
        if callable(to):
            to(device)
        eval_method = getattr(policy, "eval", None)
        if callable(eval_method):
            eval_method()
        from lerobot.policies import make_pre_post_processors  # type: ignore
        processor_error: Exception | None = None
        for processor_root in (adapter, base):
            try:
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy_config,
                    pretrained_path=str(processor_root),
                )
                if callable(preprocessor) and callable(postprocessor):
                    break
            except Exception as exc:  # pragma: no cover - native version dependent
                processor_error = exc
        else:
            raise ContractError("native SmolVLA processor pipelines are missing from adapter and base") from processor_error
        if not callable(getattr(policy, "select_action", None)):
            raise ContractError("native PEFT SmolVLA policy exposes no select_action")
        return {"policy": policy, "preprocessor": preprocessor, "postprocessor": postprocessor,
                "base_checkpoint": str(base), "adapter_checkpoint": str(adapter),
                "peft_config": peft_config}
    except ContractError:
        raise
    except Exception as exc:  # pragma: no cover - native version dependent
        raise ContractError("native SmolVLA PEFT adapter could not be loaded locally") from exc


def load_apprentice_runtime_adapter(
    checkpoint_path: str | Path,
    *,
    base_vla_sha256: str,
    adapter_loader: Callable[..., Any] | None = None,
    base_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> Callable[[ObservationFrame, Sequence[float]], tuple[float, ...]]:
    """Load a verified PEFT adapter as ``(frame, frozen_base_action) -> action``.

    The suite-owned native loader is used when ``adapter_loader`` is omitted;
    callers may inject a test/dialect loader explicitly.  A loader may return
    that callable directly, an object with ``select_action`` plus
    ``preprocessor``/``postprocessor``, or a mapping with ``action_fn``.  No
    teacher, zero-action, or frozen fallback is ever used.
    """
    path = Path(checkpoint_path).expanduser()
    base_hash = _require_sha256(base_vla_sha256, name="base_vla_sha256")
    if not path.exists():
        raise ContractError(f"Apprentice adapter checkpoint is missing: {path}")
    # Verify bytes/tree before invoking any native code.  Directory adapters
    # need an injected loader only for the type check, then are loaded once
    # below; the loader itself is never treated as a source of identity.
    if path.is_dir():
        verify_adapter_reload(path, base_vla_sha256=base_hash, loader=lambda _path: {})
    else:
        verify_adapter_reload(path, base_vla_sha256=base_hash)
    if adapter_loader is None:
        if base_checkpoint is None:
            raise ContractError("native Apprentice runtime requires base_checkpoint when adapter_loader is omitted")
        loaded = load_native_smolvla_adapter(
            path, base_checkpoint=base_checkpoint, base_vla_sha256=base_hash, device=device
        )
    else:
        loaded = _call_adapter_loader(adapter_loader, path, base_hash)
    action_fn: Any = None
    if callable(loaded):
        action_fn = loaded
    elif isinstance(loaded, Mapping) and callable(loaded.get("action_fn")):
        action_fn = loaded["action_fn"]
    else:
        policy = loaded.get("policy", loaded) if isinstance(loaded, Mapping) else getattr(loaded, "policy", loaded)
        preprocessor = loaded.get("preprocessor") if isinstance(loaded, Mapping) else getattr(loaded, "preprocessor", None)
        postprocessor = loaded.get("postprocessor") if isinstance(loaded, Mapping) else getattr(loaded, "postprocessor", None)
        select_action = getattr(policy, "select_action", None)
        if callable(select_action) and callable(preprocessor) and callable(postprocessor):
            def native_action(frame: ObservationFrame, _base: Sequence[float]) -> tuple[float, ...]:
                raw = select_action(preprocessor(_runtime_payload(frame)))
                return _coerce_runtime_action(postprocessor(raw))
            action_fn = native_action
    if not callable(action_fn):
        raise ContractError("native adapter_loader did not return a compatible action callable")

    def teacher_free_action(frame: ObservationFrame, base: Sequence[float]) -> tuple[float, ...]:
        if not isinstance(frame, ObservationFrame):
            raise ContractError("Apprentice runtime expects an ObservationFrame")
        # Validate the frozen base input even though native SmolVLA may not use
        # it; it remains part of the policy contract and audit trace.
        validate_action(base)
        return _coerce_runtime_action(action_fn(frame, tuple(float(v) for v in base)))

    return teacher_free_action


def load_apprentice_policy(
    checkpoint_path: str | Path,
    *,
    base_vla_sha256: str,
    adapter_loader: Callable[..., Any] | None = None,
    base_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> Any:
    """Return an :class:`ApprenticePolicy` backed by the verified callable."""
    from .policies import ApprenticePolicy
    return ApprenticePolicy(action_fn=load_apprentice_runtime_adapter(
        checkpoint_path, base_vla_sha256=base_vla_sha256, adapter_loader=adapter_loader,
        base_checkpoint=base_checkpoint, device=device,
    ))


def invoke_native_trainer(command: Sequence[str], *, cwd: str | Path | None = None,
                          timeout: float | None = None) -> dict[str, Any]:
    """Invoke a pinned LeRobot trainer command without importing it eagerly."""
    if not command:
        raise ContractError("native trainer command cannot be empty")
    completed = subprocess.run(tuple(str(item) for item in command), cwd=cwd,
                               capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise ContractError(f"native trainer failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


# Short aliases keep integration launchers readable without changing the
# explicit manifest-oriented class names above.
NativeSmolVLAConfig = SmolVLALoRAConfig
ApprenticeDatasetRequest = ApprenticeExportRequest
export_teacher_rows = export_apprentice_dataset


__all__ = [
    "APPRENTICE_SCHEMA", "APPRENTICE_DATASET_SCHEMA", "APPRENTICE_ADAPTER_SCHEMA", "DEFAULT_SMOLVLA_MODEL",
    "DEFAULT_SMOLVLA_PEFT_TARGET_REGEX",
    "ApprenticeDatasetRequest", "ApprenticeExportReceipt",
    "ApprenticeExportRequest", "ApprenticeTrainingConfig", "NativeSmolVLAConfig", "NativeSmolVLAExporter",
    "NativeSmolVLATrainer", "SmolVLALoRAConfig", "ApprenticeTrainingJob",
    "export_apprentice_dataset", "export_teacher_rows", "make_export_request", "train_apprentice",
    "export_apprentice_dataset_native", "build_apprentice_training_job", "run_apprentice_training_job",
    "load_apprentice_transitions", "load_teacher_transitions", "load_on_call_teacher_transitions",
    "save_adapter_checkpoint", "write_adapter_manifest", "verify_adapter_reload",
    "load_native_smolvla_adapter", "load_apprentice_runtime_adapter", "load_apprentice_policy",
    "invoke_native_trainer",
]
