"""Durable, bounded-memory cache for accepted Arrow episodes.

The collector writes one canonical JSON object per accepted episode before it
starts the next attempt.  The index contains only scalar episode metadata and
paths; it never contains transition payloads.  A contract file binds a cache
to the exact collection configuration so a stale cache cannot silently be
reused for another task, controller, or evaluation split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .contracts import ContractError, _json_safe


CACHE_SCHEMA = "automatic_ttt.accepted_episode_cache.v2"


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_atomic_json(path: Path, payload: Any) -> str:
    """Write one canonical JSON object with file and directory durability."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable cache artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return hashlib.sha256(encoded).hexdigest()
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(encoded).hexdigest()


def _replace_atomic_json(path: Path, payload: Any) -> None:
    """Replace mutable cache state atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_single_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {path}") from exc
    if len(lines) != 1:
        raise ContractError(f"{label} must contain exactly one JSON line: {path}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


@dataclass(frozen=True)
class CacheRef:
    """Scalar metadata for one cached row; no transition data is retained."""

    episode_id: str
    seed: int
    path: str
    task_id: int
    source_kind: str
    reset_identity: Mapping[str, Any] | None = None
    evaluator_receipt: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "path": self.path,
            "task_id": self.task_id,
            "source_kind": self.source_kind,
        }
        if self.reset_identity is not None:
            value["reset_identity"] = dict(self.reset_identity)
        if self.evaluator_receipt is not None:
            value["evaluator_receipt"] = dict(self.evaluator_receipt)
        return value


class DurableEpisodeCache:
    """Bounded-memory append-only episode cache with resumable state."""

    def __init__(self, root: str | Path, *, contract: Mapping[str, Any], target: int) -> None:
        self.root = Path(root).resolve() / ".accepted_episode_cache"
        # Keep the established ``.accepted_episode_cache/seed-*.json`` layout
        # so existing artifact browsers and tests remain compatible.
        self.episodes_root = self.root
        self.contract = dict(_json_safe(contract))
        self.target = int(target)
        if self.target <= 0:
            raise ContractError("cache target must be positive")
        canonical = _canonical_json(self.contract).encode("utf-8")
        self.contract_sha256 = hashlib.sha256(canonical).hexdigest()
        self.contract_path = self.root / "collection_contract.json"
        self.index_path = self.root / "collection_index.json"
        self._state: dict[str, Any] = {}
        self._refs: list[CacheRef] = []
        self._open()

    @property
    def accepted_count(self) -> int:
        return len(self._refs)

    @property
    def attempted_count(self) -> int:
        return int(self._state.get("attempted_count", 0))

    @property
    def next_attempt_index(self) -> int:
        return int(self._state.get("next_attempt_index", 0))

    @property
    def failure_categories(self) -> dict[str, int]:
        value = self._state.get("discarded_failure_categories", {})
        return {str(key): int(count) for key, count in dict(value).items()}

    @property
    def refs(self) -> tuple[CacheRef, ...]:
        return tuple(self._refs)

    def _open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        if self.contract_path.exists():
            stored = _read_single_json(self.contract_path, "collection cache contract")
            stored_hash = stored.get("contract_sha256")
            if stored_hash != self.contract_sha256 or stored.get("contract") != self.contract:
                raise ContractError(
                    "accepted episode cache contract mismatch; refusing to reuse trajectories"
                )
        else:
            # A legacy cache without a binding is unsafe to reuse.  Refuse it
            # explicitly instead of silently mixing data from another run.
            legacy = [path for path in self.root.glob("seed-*.json") if path.is_file()]
            if legacy:
                raise ContractError("accepted episode cache is unbound legacy data; refusing reuse")
            _write_atomic_json(self.contract_path, {
                "schema": CACHE_SCHEMA,
                "contract": self.contract,
                "contract_sha256": self.contract_sha256,
            })
        if self.index_path.exists():
            state = _read_single_json(self.index_path, "collection cache index")
            if state.get("schema") != CACHE_SCHEMA or state.get("contract_sha256") != self.contract_sha256:
                raise ContractError("accepted episode cache index contract mismatch")
            self._state = dict(state)
        else:
            self._state = {
                "schema": CACHE_SCHEMA,
                "contract_sha256": self.contract_sha256,
                "target": self.target,
                "attempted_count": 0,
                "next_attempt_index": 0,
                "discarded_failure_categories": {},
                "accepted": [],
            }
            _replace_atomic_json(self.index_path, self._state)
        if int(self._state.get("target", self.target)) != self.target:
            raise ContractError("accepted episode cache target mismatch")
        self._load_refs()

    def _load_refs(self) -> None:
        entries = self._state.get("accepted", [])
        if not isinstance(entries, list):
            raise ContractError("accepted episode cache index accepted field must be a list")
        refs: list[CacheRef] = []
        seen_ids: set[str] = set()
        seen_seeds: set[int] = set()
        for entry in entries:
            ref = self._ref_from_entry(entry)
            if ref.episode_id in seen_ids or ref.seed in seen_seeds:
                raise ContractError("accepted episode cache contains duplicate identity")
            self._validate_cached_row(ref)
            refs.append(ref)
            seen_ids.add(ref.episode_id)
            seen_seeds.add(ref.seed)
        # Recover an episode whose file was durable but whose mutable index
        # update was interrupted.  Each candidate is parsed one at a time.
        indexed_paths = {ref.path for ref in refs}
        for path in sorted(self.episodes_root.glob("seed-*.json")):
            relative = path.relative_to(self.root).as_posix()
            if relative in indexed_paths:
                continue
            row = _read_single_json(path, "cached accepted episode")
            ref = self._ref_from_row(row, relative)
            self._validate_cached_row(ref, row=row)
            if ref.episode_id in seen_ids or ref.seed in seen_seeds:
                raise ContractError("accepted episode cache contains duplicate orphan identity")
            refs.append(ref)
            seen_ids.add(ref.episode_id)
            seen_seeds.add(ref.seed)
        refs.sort(key=lambda ref: ref.seed)
        self._refs = refs
        if len(refs) > self.target:
            raise ContractError("accepted episode cache contains more successes than target")
        if len(refs) != len(entries):
            self._state["accepted"] = [ref.as_dict() for ref in refs]
            self._save_state()

    def _ref_from_entry(self, entry: Any) -> CacheRef:
        if not isinstance(entry, Mapping):
            raise ContractError("accepted episode cache index entry must be an object")
        try:
            episode_id = str(entry["episode_id"])
            seed = int(entry["seed"])
            task_id = int(entry["task_id"])
            path = str(entry["path"])
            source_kind = str(entry["source_kind"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("accepted episode cache index entry is malformed") from exc
        if not episode_id or not source_kind or not path:
            raise ContractError("accepted episode cache index entry has empty identity")
        reset_identity = entry.get("reset_identity")
        receipt = entry.get("evaluator_receipt")
        if reset_identity is not None and not isinstance(reset_identity, Mapping):
            raise ContractError("accepted episode cache reset identity is malformed")
        if receipt is not None and not isinstance(receipt, Mapping):
            raise ContractError("accepted episode cache evaluator receipt is malformed")
        return CacheRef(episode_id, seed, path, task_id, source_kind,
                        dict(reset_identity) if reset_identity is not None else None,
                        dict(receipt) if receipt is not None else None)

    def _ref_from_row(self, row: Mapping[str, Any], relative: str) -> CacheRef:
        return self._ref_from_entry({
            "episode_id": row.get("episode_id"), "seed": row.get("seed"),
            "task_id": row.get("task_id"), "path": relative,
            "source_kind": row.get("source_kind"),
            "reset_identity": row.get("reset_identity"),
            "evaluator_receipt": row.get("evaluator_receipt"),
        })

    def _validate_cached_row(self, ref: CacheRef, *, row: Mapping[str, Any] | None = None) -> None:
        path = (self.root / ref.path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ContractError("accepted episode cache path escapes cache root") from exc
        if not path.is_file():
            raise ContractError(f"accepted episode cache file is missing: {path}")
        if row is None:
            row = _read_single_json(path, "cached accepted episode")
        if row.get("episode_id") != ref.episode_id or int(row.get("seed", -1)) != ref.seed:
            raise ContractError("accepted episode cache row identity does not match index")
        if int(row.get("task_id", -1)) != ref.task_id or row.get("source_kind") != ref.source_kind:
            raise ContractError("accepted episode cache row provenance does not match index")

    def _save_state(self) -> None:
        _replace_atomic_json(self.index_path, self._state)

    def begin_attempt(self, attempt_index: int) -> int:
        if attempt_index < self.next_attempt_index:
            raise ContractError("collection attempt index moved backwards")
        self._state["attempted_count"] = max(self.attempted_count, int(attempt_index) + 1)
        self._state["next_attempt_index"] = int(attempt_index) + 1
        self._save_state()
        return self.attempted_count

    def record_failure(self, category: str) -> None:
        categories = self.failure_categories
        categories[str(category)] = categories.get(str(category), 0) + 1
        self._state["discarded_failure_categories"] = categories
        self._save_state()

    def add_success(self, row: Mapping[str, Any]) -> CacheRef:
        row = dict(row)
        try:
            seed = int(row["seed"])
            episode_id = str(row["episode_id"])
            task_id = int(row["task_id"])
            source_kind = str(row["source_kind"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("accepted cache row identity is malformed") from exc
        if any(ref.seed == seed for ref in self._refs) or any(ref.episode_id == episode_id for ref in self._refs):
            raise ContractError("accepted episode cache success identity is duplicated")
        target = self.episodes_root / f"seed-{seed}.json"
        relative = target.relative_to(self.root).as_posix()
        _write_atomic_json(target, row)
        ref = self._ref_from_row(row, relative)
        self._validate_cached_row(ref, row=row)
        self._refs = [*self._refs, ref]
        self._refs.sort(key=lambda item: item.seed)
        self._state["accepted"] = [item.as_dict() for item in self._refs]
        self._save_state()
        return ref

    def iter_rows(self, refs: Iterable[CacheRef] | None = None) -> Iterator[Mapping[str, Any]]:
        for ref in self._refs if refs is None else refs:
            row = _read_single_json(self.root / ref.path, "cached accepted episode")
            self._validate_cached_row(ref, row=row)
            yield row


def collection_contract_hash(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


__all__ = ["CACHE_SCHEMA", "CacheRef", "DurableEpisodeCache", "collection_contract_hash"]
