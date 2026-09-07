"""Leakage-resistant records for automatic DAgger-style LIBERO TTT."""

from __future__ import annotations

import hashlib
import json
import base64
import math
import os
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import Actor as ExecutedActor
from .contracts import TransitionRecord as ExecutedTransition
from .contracts import assert_student_observation
from .contracts import validate_action


_TYPE_TAG = "__automatic_ttt_type__"
CANONICAL_OBSERVATION_SCHEMA = "libero_rgb_state8_instruction_v1"


class ObservationSchemaError(ValueError):
    """Raised when raw simulator fields enter the student training stream."""


def _shape(value: Any) -> tuple[int, ...] | None:
    declared = getattr(value, "shape", None)
    if declared is not None:
        try:
            return tuple(int(dim) for dim in declared)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shapes = [_shape(item) for item in value]
        if any(item is None for item in child_shapes):
            return (len(value),)
        first = child_shapes[0]
        if any(item != first for item in child_shapes):
            return None
        return (len(value),) + (first or ())
    return None


def validate_student_observation_schema(
    observation: Mapping[str, Any],
    *,
    require_complete: bool = False,
    schema: str = CANONICAL_OBSERVATION_SCHEMA,
) -> None:
    """Validate the only student observation projection used by this package.

    Allowed fields are ``agentview`` and ``wrist`` RGB images, ``state`` with
    exactly eight values (EEF position, axis-angle orientation, gripper), and
    ``instruction`` text.  Teacher/controller sidecars belong in provenance,
    never in this mapping.  ``require_complete=True`` is mandatory at the
    executed-records -> dataset boundary.
    """
    if schema != CANONICAL_OBSERVATION_SCHEMA:
        raise ObservationSchemaError(f"unsupported student observation schema {schema!r}")
    if not isinstance(observation, Mapping):
        raise ObservationSchemaError("student observation must be a mapping")
    allowed = {"agentview", "wrist", "state", "instruction"}
    unknown = sorted(set(observation) - allowed)
    if unknown:
        raise ObservationSchemaError(f"unknown/non-student observation fields: {unknown}")
    try:
        assert_student_observation(observation)
    except Exception as exc:
        raise ObservationSchemaError(str(exc)) from exc
    required = allowed if require_complete else set()
    missing = sorted(required - set(observation))
    if missing:
        raise ObservationSchemaError(f"canonical student observation missing fields: {missing}")
    for name in ("agentview", "wrist"):
        if name not in observation:
            continue
        shape = _shape(observation[name])
        if shape is None or len(shape) != 3 or shape[-1] != 3:
            raise ObservationSchemaError(f"{name} must be an HxWx3 RGB observation; got shape={shape}")
    if "state" in observation:
        shape = _shape(observation["state"])
        if shape != (8,):
            raise ObservationSchemaError(f"state must contain exactly 8 values; got shape={shape}")
    if "instruction" in observation and not isinstance(observation["instruction"], str):
        raise ObservationSchemaError("instruction must be text")


def _encode(value: Any) -> Any:
    """Encode values losslessly into strict JSON-compatible primitives.

    In particular, observations commonly contain HxWxC uint8 arrays.  Using
    ``str(array)`` silently truncates those arrays and makes the resulting
    dataset unusable, so arrays are stored as dtype/shape/raw-byte records.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be serialized")
        return value
    if isinstance(value, bytes):
        return {_TYPE_TAG: "bytes", "data": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value) and not isinstance(value, type):
        return _encode({item.name: getattr(value, item.name) for item in fields(value)})
    if isinstance(value, Mapping):
        encoded_items = [[_encode(key), _encode(item)] for key, item in value.items()]
        encoded_items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return {_TYPE_TAG: "mapping", "items": encoded_items}
    if isinstance(value, tuple):
        return {_TYPE_TAG: "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]

    # NumPy is optional.  Duck-typing keeps dataset bookkeeping usable on a
    # login node while still preserving ndarray dtype/shape/data when present.
    module_name = type(value).__module__
    if module_name.startswith("numpy") and hasattr(value, "dtype") and hasattr(value, "shape"):
        dtype = value.dtype
        if getattr(dtype, "hasobject", False):
            raise ValueError("object-dtype arrays are not supported in dataset observations")
        if dtype.kind in "fc":
            import numpy as np
            if not np.isfinite(value).all():
                raise ValueError("non-finite array values cannot be serialized")
        raw = value.tobytes(order="C")
        return {
            _TYPE_TAG: "ndarray",
            "dtype": dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if module_name.startswith("torch") and hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
        return _encode(array)
    # NumPy scalar values do not have a shape but must not be stringified.
    if module_name.startswith("numpy") and hasattr(value, "item"):
        return _encode(value.item())
    raise TypeError(f"unsupported dataset value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict) or _TYPE_TAG not in value:
        if isinstance(value, dict):
            return {key: _decode(item) for key, item in value.items()}
        return value
    kind = value[_TYPE_TAG]
    if kind == "bytes":
        return base64.b64decode(value["data"].encode("ascii"))
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "mapping":
        return {_decode(key): _decode(item) for key, item in value["items"]}
    if kind == "ndarray":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("reading ndarray observations requires optional numpy") from exc
        raw = base64.b64decode(value["data"].encode("ascii"))
        array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy()
        return array.reshape(tuple(value["shape"]), order="C")
    raise ValueError(f"unknown encoded dataset type {kind!r}")


def _json_payload(value: Any) -> str:
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TransitionRecord:
    episode_id: str
    timestep: int
    observation: Mapping[str, Any]
    robot_action: tuple[float, ...] | None
    teacher_action: tuple[float, ...] | None
    # Robot chunks are retained as TTT context, but receive zero action-loss
    # weight.  Teacher correction chunks receive both context and action loss.
    context_loss_mask: float
    action_loss_mask: float
    source: str  # "robot", "teacher_correction", or "rejected"
    task_id: int | str
    seed: int
    teacher_privilege: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    action_chunk_id: str = "single-step"
    action_chunk_index: int = 0
    action_chunk_horizon: int = 1
    denoising_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in {"robot", "teacher_correction", "rejected"}:
            raise ValueError(f"invalid source={self.source!r}")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if (
            not self.action_chunk_id
            or self.action_chunk_index < 0
            or self.action_chunk_horizon <= 0
            or self.action_chunk_index >= self.action_chunk_horizon
        ):
            raise ValueError("invalid action chunk identity/index/horizon")
        _encode(self.denoising_metadata)
        validate_student_observation_schema(self.observation)
        if self.robot_action is not None:
            validate_action(self.robot_action)
        if self.teacher_action is not None:
            validate_action(self.teacher_action)
        if len(self.teacher_action or ()) and self.robot_action is not None:
            if len(self.teacher_action) != len(self.robot_action):
                raise ValueError("robot and teacher action dimensions differ")
        if not 0.0 <= self.context_loss_mask <= 1.0 or not 0.0 <= self.action_loss_mask <= 1.0:
            raise ValueError("loss masks must be in [0, 1]")
        if self.source == "robot" and self.action_loss_mask != 0.0:
            raise ValueError("robot context must have action_loss_mask=0")
        if self.source == "teacher_correction" and self.action_loss_mask != 1.0:
            raise ValueError("teacher corrections must have action_loss_mask=1")
        if self.source == "robot" and self.robot_action is None:
            raise ValueError("robot context requires the executed robot action")
        if self.source == "teacher_correction" and self.teacher_action is None:
            raise ValueError("teacher correction requires the executed teacher action")

    @property
    def supervised_action(self) -> tuple[float, ...] | None:
        return self.teacher_action if self.action_loss_mask else None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    task_id: int | str
    seed: int
    outcome: str
    transitions: tuple[TransitionRecord, ...]
    teacher_used: bool
    environment_identity: str
    source_split: str = "train"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        expected = list(range(len(self.transitions)))
        actual = [t.timestep for t in self.transitions]
        if actual != expected:
            raise ValueError(f"episode {self.episode_id} timesteps are not contiguous: {actual}")
        if any(t.episode_id != self.episode_id for t in self.transitions):
            raise ValueError("transition episode_id mismatch")
        if any(t.task_id != self.task_id or t.seed != self.seed for t in self.transitions):
            raise ValueError("transition task/seed mismatch")

    def content_hash(self) -> str:
        blob = _json_payload({
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "outcome": self.outcome,
            "transitions": self.transitions,
            "teacher_used": self.teacher_used,
            "environment_identity": self.environment_identity,
            "source_split": self.source_split,
            "provenance": self.provenance,
        }).encode()
        return hashlib.sha256(blob).hexdigest()


class TTTDataset:
    """Append-only dataset with episode-level split checks."""

    def __init__(self, episodes: Iterable[EpisodeRecord] = ()) -> None:
        self.episodes = list(episodes)
        for episode in self.episodes:
            episode.validate()

    def add(self, episode: EpisodeRecord) -> None:
        episode.validate()
        if any(old.episode_id == episode.episode_id for old in self.episodes):
            raise ValueError(f"duplicate episode_id={episode.episode_id}")
        self.episodes.append(episode)

    @staticmethod
    def _split_identity(episode: EpisodeRecord) -> tuple[Any, ...]:
        provenance = episode.provenance
        return (
            episode.task_id,
            episode.seed,
            provenance.get("init_state_hash"),
            provenance.get("randomization_identity"),
            episode.environment_identity,
        )

    def validate_disjoint(self, other: "TTTDataset") -> None:
        left = {episode.episode_id for episode in self.episodes}
        overlap = left.intersection(episode.episode_id for episode in other.episodes)
        if overlap:
            raise ValueError(f"episode leakage across splits: {sorted(overlap)}")
        left_identity = {self._split_identity(episode) for episode in self.episodes}
        right_identity = {self._split_identity(episode) for episode in other.episodes}
        identity_overlap = left_identity.intersection(right_identity)
        if identity_overlap:
            raise ValueError(f"task/seed/initial-state/environment leakage across splits: {sorted(identity_overlap)}")

    def validate_against_split(self, split_manifest: Mapping[str, Any]) -> None:
        """Reject evaluation/query episodes before they reach an optimizer."""
        evaluation_ids = set(split_manifest.get("eval_episode_ids", split_manifest.get("query_episode_ids", ())))
        if not evaluation_ids:
            evaluation_ids = set(split_manifest.get("eval_ids", ()))
        registered_ids: set[Any] = set()
        for key in ("train_episode_ids", "support_episode_ids", "train_ids", "support_ids"):
            values = split_manifest.get(key, ())
            if values:
                registered_ids.update(values)
        if not registered_ids:
            raise ValueError("split manifest must explicitly list train/support episode IDs")
        leaked = sorted(episode.episode_id for episode in self.episodes if episode.episode_id in evaluation_ids)
        if leaked:
            raise ValueError(f"evaluation/query episodes cannot be used for training: {leaked}")
        for episode in self.episodes:
            if episode.source_split not in {"train", "support"}:
                raise ValueError(f"non-training split {episode.source_split!r} cannot reach optimizer")
            if episode.episode_id not in registered_ids:
                raise ValueError(f"unregistered training episode ID cannot reach optimizer: {episode.episode_id}")
        # When the manifest supplies per-episode identities, bind all three
        # fields that determine a reset state.  Missing or mismatched values
        # must fail closed rather than silently pairing another episode.
        identities = split_manifest.get("episode_identities", split_manifest.get("identities", {}))
        if identities:
            if not isinstance(identities, Mapping):
                raise ValueError("split episode_identities must be a mapping")
            for episode in self.episodes:
                expected = identities.get(episode.episode_id)
                if not isinstance(expected, Mapping):
                    raise ValueError(f"missing identity for registered episode {episode.episode_id}")
                if expected.get("task_id") != episode.task_id or expected.get("seed") != episode.seed:
                    raise ValueError(f"task/seed mismatch for registered episode {episode.episode_id}")
                expected_hash = expected.get("init_state_hash", expected.get("initial_state_hash"))
                actual_hash = episode.provenance.get("init_state_hash", episode.provenance.get("initial_state_hash"))
                if expected_hash is None or actual_hash != expected_hash:
                    raise ValueError(f"initial-state hash mismatch for registered episode {episode.episode_id}")

    def split(self, name: str) -> "TTTDataset":
        return TTTDataset(episode for episode in self.episodes if episode.source_split == name)

    def records(self, include_rejected: bool = False) -> list[TransitionRecord]:
        rows = [row for episode in self.episodes for row in episode.transitions]
        return rows if include_rejected else [row for row in rows if row.source != "rejected"]

    def action_training_records(self, *, allowed_splits: Sequence[str] = ("train", "support")) -> list[TransitionRecord]:
        return [
            row for episode in self.episodes if episode.source_split in allowed_splits
            for row in episode.transitions
            if row.source != "rejected" and row.action_loss_mask > 0 and row.teacher_action is not None
        ]

    def context_records(self, *, allowed_splits: Sequence[str] = ("train", "support")) -> list[TransitionRecord]:
        return [
            row for episode in self.episodes if episode.source_split in allowed_splits
            for row in episode.transitions if row.source != "rejected" and (row.context_loss_mask > 0 or row.source == "robot")
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "episode_count": len(self.episodes),
            "transition_count": len(self.records()),
            "action_supervision_count": len(self.action_training_records()),
            "episode_hashes": {episode.episode_id: episode.content_hash() for episode in self.episodes},
            "splits": {split: len(self.split(split).episodes) for split in {e.source_split for e in self.episodes}},
        }

    def write_jsonl(self, path: str | Path, *, overwrite: bool = False) -> None:
        target = Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite immutable dataset artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".partial", delete=False
            ) as handle:
                temporary = handle.name
                for episode in self.episodes:
                    handle.write(_json_payload(episode) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "TTTDataset":
        episodes = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = _decode(json.loads(line))
            transitions = tuple(TransitionRecord(**row) for row in raw.pop("transitions"))
            episodes.append(EpisodeRecord(transitions=transitions, **raw))
        return cls(episodes)


def episode_from_executed_records(
    records: Sequence[ExecutedTransition],
    *,
    task_id: int | str,
    seed: int,
    episode_id: str | None = None,
    source_split: str = "train",
    outcome: str = "teacher_success",
    environment_identity: str = "unknown",
    provenance: Mapping[str, Any] | None = None,
    observation_schema: str | None = None,
) -> EpisodeRecord:
    """Convert the lifecycle recorder's atomic transitions to TTT masks.

    VLA actions are retained as context with zero action-loss weight.  Arrow
    actions are the only supervised targets.  This conversion is the sole
    place where actor labels become the RoboTTT/DAgger-Distillation masks;
    unexecuted proposals never enter the dataset.
    """

    if not records:
        raise ValueError("cannot create a training episode without transitions")
    if observation_schema is None:
        raise ObservationSchemaError(
            "observation_schema must be supplied by the VLA adapter; raw simulator observations cannot enter training"
        )
    if observation_schema != CANONICAL_OBSERVATION_SCHEMA:
        raise ObservationSchemaError(f"unsupported observation_schema={observation_schema!r}")
    for executed in records:
        validate_student_observation_schema(executed.observation, require_complete=True, schema=observation_schema)
    resolved_id = episode_id or records[0].episode_id
    rows: list[TransitionRecord] = []
    for row in records:
        if row.episode_id != resolved_id:
            raise ValueError("executed records belong to multiple episodes")
        if row.actor is ExecutedActor.VLA:
            rows.append(
                TransitionRecord(
                    episode_id=resolved_id,
                    timestep=row.timestep,
                    observation=row.observation,
                    robot_action=tuple(row.action),
                    teacher_action=None,
                    context_loss_mask=1.0,
                    action_loss_mask=0.0,
                    source="robot",
                    task_id=task_id,
                    seed=seed,
                    metadata={"done": row.done, "success": row.success},
                    action_chunk_id=row.action_chunk_id,
                    action_chunk_index=row.action_chunk_index,
                    action_chunk_horizon=row.action_chunk_horizon,
                    denoising_metadata=row.denoising_metadata,
                )
            )
        elif row.actor is ExecutedActor.TEACHER:
            rows.append(
                TransitionRecord(
                    episode_id=resolved_id,
                    timestep=row.timestep,
                    observation=row.observation,
                    robot_action=None,
                    teacher_action=tuple(row.action),
                    context_loss_mask=1.0,
                    action_loss_mask=1.0,
                    source="teacher_correction",
                    task_id=task_id,
                    seed=seed,
                    teacher_privilege="simulator_bbox_and_contact_state",
                    metadata={"done": row.done, "success": row.success},
                    action_chunk_id=row.action_chunk_id,
                    action_chunk_index=row.action_chunk_index,
                    action_chunk_horizon=row.action_chunk_horizon,
                    denoising_metadata=row.denoising_metadata,
                )
            )
        else:  # pragma: no cover - Actor is an enum and currently has two values
            raise ValueError(f"unsupported actor {row.actor!r}")
    return EpisodeRecord(
        episode_id=resolved_id,
        task_id=task_id,
        seed=seed,
        outcome=outcome,
        transitions=tuple(rows),
        teacher_used=any(row.actor is ExecutedActor.TEACHER for row in records),
        environment_identity=environment_identity,
        source_split=source_split,
        provenance=dict(provenance or {}),
    )


__all__ = [
    "EpisodeRecord", "TTTDataset", "TransitionRecord", "episode_from_executed_records",
]
