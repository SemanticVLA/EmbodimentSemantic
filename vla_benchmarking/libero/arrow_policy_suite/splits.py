"""Frozen reset-identity split manifests for Arrow policy experiments.

Splits are keyed by reset identity, never by row position or a mutable output
path.  A manifest can be created once and then only read/verified; attempting
to mutate an existing manifest fails closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ContractError, _safe


class SplitError(ContractError):
    """Raised when reset identities or split manifests are unsafe."""


def _sha256_json(value: Any) -> str:
    blob = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ResetIdentity:
    """Immutable identity for one reset initial state.

    ``simulator_state_sha256`` and ``replay_key`` are alternatives: at least
    one must be supplied.  The student observation hash is deliberately kept
    separate from simulator identity.
    """

    task_id: int
    episode_id: str
    seed: int
    reset_index: int
    observation_sha256: str
    environment_fingerprint: str
    simulator_state_sha256: str | None = None
    replay_key: str | None = None
    simulator_replay_key: str | None = None
    query_index: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or self.task_id < 0:
            raise SplitError("task_id must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise SplitError("seed must be a non-negative integer")
        if self.reset_index != 1:
            raise SplitError("a scored reset identity must have reset_index=1")
        if not self.episode_id or not self.environment_fingerprint:
            raise SplitError("episode_id and environment_fingerprint are required")
        for name, value in (("observation_sha256", self.observation_sha256),
                            ("simulator_state_sha256", self.simulator_state_sha256)):
            if value is None:
                continue
            if (not isinstance(value, str) or len(value) != 64 or value != value.lower()
                    or any(c not in "0123456789abcdef" for c in value)):
                raise SplitError(f"{name} must be a lowercase SHA-256 digest")
        if self.simulator_state_sha256 is None and not self.replay_key and not self.simulator_replay_key:
            raise SplitError("reset identity requires simulator_state_sha256 or replay_key")
        if self.replay_key is None and self.simulator_replay_key is not None:
            object.__setattr__(self, "replay_key", self.simulator_replay_key)
        if self.query_index is not None and (isinstance(self.query_index, bool) or self.query_index < 0):
            raise SplitError("query_index must be a non-negative integer")

    @property
    def key(self) -> tuple[int, int, str]:
        # Query index is the authoritative stable key when present; episode
        # IDs remain the fallback for older manifests.
        return self.task_id, self.seed, str(self.query_index if self.query_index is not None else self.episode_id)

    @property
    def collision_key(self) -> tuple[str, int, str]:
        """Authoritative reset identity used for split collision checks.

        Metadata such as seed, query index, and episode label can be renamed
        without changing the reset.  A simulator-state digest therefore takes
        precedence; replay keys are the fallback when no state digest exists.
        """
        source = self.simulator_state_sha256 or self.replay_key or self.simulator_replay_key
        if not source:  # guarded by __post_init__, retained for defensive typing
            raise SplitError("reset identity has no authoritative collision source")
        return self.environment_fingerprint, self.task_id, str(source)

    @property
    def digest(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True)
class SplitManifest:
    split: str
    identities: tuple[ResetIdentity, ...]
    schema: str = "arrow_policy_suite.reset_split.v1"
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if self.split not in {"train", "collection", "validation", "test"}:
            raise SplitError("split must be train, collection, validation, or test")
        if self.schema != "arrow_policy_suite.reset_split.v1":
            raise SplitError("unsupported split manifest schema")
        identities = tuple(self.identities)
        if any(not isinstance(item, ResetIdentity) for item in identities):
            raise SplitError("split identities must be ResetIdentity values")
        object.__setattr__(self, "identities", identities)
        keys = [item.collision_key for item in self.identities]
        if len(keys) != len(set(keys)):
            raise SplitError(f"duplicate reset identity in {self.split} split")
        payload = self.payload()
        computed = _sha256_json(payload)
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise SplitError("split manifest digest does not match its contents")
        object.__setattr__(self, "manifest_sha256", computed)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "identities": [asdict(item) for item in self.identities],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}

    @property
    def reset_ids(self) -> tuple[str, ...]:
        """Compatibility view for callers that refer to episode reset IDs."""
        return tuple(item.episode_id for item in self.identities)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SplitManifest":
        if value.get("schema") != "arrow_policy_suite.reset_split.v1":
            raise SplitError("unsupported split manifest schema")
        identities = tuple(ResetIdentity(**item) for item in value.get("identities", ()))
        return cls(str(value["split"]), identities, str(value["schema"]), str(value.get("manifest_sha256", "")))


def build_split_manifest(split: str, identities: Iterable[ResetIdentity | Mapping[str, Any]]) -> SplitManifest:
    normalized = tuple(item if isinstance(item, ResetIdentity) else ResetIdentity(**item) for item in identities)
    return SplitManifest(split, normalized)


def validate_split_manifests(
    manifests: Sequence[SplitManifest], *, task_ids: Sequence[int] | None = None,
    required_per_task: int | None = None,
) -> None:
    """Reject collisions across train/validation/test reset identities."""
    seen: dict[tuple[str, int, str], str] = {}
    for manifest in manifests:
        for identity in manifest.identities:
            prior = seen.get(identity.collision_key)
            if prior is not None and prior != manifest.split:
                raise SplitError(
                    f"reset collision identity {identity.collision_key!r} appears in both {prior} and {manifest.split}"
                )
            seen[identity.collision_key] = manifest.split
    if task_ids is not None:
        expected_tasks = tuple(int(task) for task in task_ids)
        if len(set(expected_tasks)) != len(expected_tasks):
            raise SplitError("task_ids must be unique for split validation")
        if required_per_task is None or required_per_task <= 0:
            raise SplitError("required_per_task must be positive when task_ids are supplied")
        by_split_task: dict[tuple[str, int], int] = {}
        for manifest in manifests:
            for identity in manifest.identities:
                by_split_task[(manifest.split, identity.task_id)] = by_split_task.get((manifest.split, identity.task_id), 0) + 1
        for split in ("test", "validation"):
            for task in expected_tasks:
                observed = by_split_task.get((split, task), 0)
                if observed != required_per_task:
                    raise SplitError(
                        f"{split} task {task} requires exactly {required_per_task} frozen reset identities; observed {observed}"
                    )


def assert_disjoint(*manifests: SplitManifest) -> None:
    """Alias for the explicit cross-split collision check."""
    validate_split_manifests(manifests)


def validate_complete_split_manifests(
    manifests: Sequence[SplitManifest], task_ids: Sequence[int], *, required_per_task: int = 10,
) -> None:
    """Validate complete test/validation coverage and cross-split disjointness."""
    validate_split_manifests(manifests, task_ids=task_ids, required_per_task=required_per_task)


def validate_study_split_manifests(
    collection: SplitManifest, validation: SplitManifest, test: SplitManifest,
    task_ids: Sequence[int], *, collection_per_task: int = 50,
    validation_per_task: int = 10, test_per_task: int = 10,
) -> None:
    """Validate the complete collection/validation/test identity grid."""
    if collection.split not in {"collection", "train"}:
        raise SplitError("collection manifest must use split='collection' or legacy split='train'")
    if validation.split != "validation" or test.split != "test":
        raise SplitError("validation/test manifests have incorrect split names")
    manifests = (collection, validation, test)
    validate_split_manifests(manifests)
    expected_tasks = tuple(int(task) for task in task_ids)
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise SplitError("task_ids must be non-empty and unique")
    counts = {
        "collection": collection_per_task,
        "validation": validation_per_task,
        "test": test_per_task,
    }
    for manifest, label in ((collection, "collection"), (validation, "validation"), (test, "test")):
        per_task: dict[int, int] = {}
        for identity in manifest.identities:
            per_task[identity.task_id] = per_task.get(identity.task_id, 0) + 1
        for task in expected_tasks:
            observed = per_task.get(task, 0)
            if observed != counts[label]:
                raise SplitError(f"{label} task {task} requires exactly {counts[label]} identities; observed {observed}")
        extras = set(per_task) - set(expected_tasks)
        if extras:
            raise SplitError(f"{label} manifest contains tasks outside task_ids: {sorted(extras)}")


def split_set_sha256(manifests: Sequence[SplitManifest]) -> str:
    """Return a stable digest independent of input ordering."""
    entries = sorted((manifest.split, manifest.manifest_sha256) for manifest in manifests)
    return _sha256_json(entries)


def write_split_manifest(path: str | os.PathLike[str], manifest: SplitManifest) -> str:
    """Create a manifest exactly once and return its SHA-256 digest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise SplitError(f"refusing to mutate existing split manifest: {target}") from exc
    return hashlib.sha256(data).hexdigest()


def read_split_manifest(path: str | os.PathLike[str]) -> SplitManifest:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"cannot read split manifest {target}") from exc
    if not isinstance(value, Mapping):
        raise SplitError("split manifest must be a JSON object")
    return SplitManifest.from_dict(value)


__all__ = [
    "ResetIdentity", "SplitManifest", "SplitError", "build_split_manifest", "validate_split_manifests",
    "assert_disjoint", "validate_complete_split_manifests", "validate_study_split_manifests", "split_set_sha256",
    "write_split_manifest", "read_split_manifest",
]
