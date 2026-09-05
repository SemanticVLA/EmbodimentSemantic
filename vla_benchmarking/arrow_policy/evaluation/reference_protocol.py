"""Immutable binding to the archived 87-percent sealed 100-cell campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REFERENCE_MANIFEST = "arrow_pick_place_matrix_manifest.json"
REFERENCE_TERMINAL_MANIFEST = "arrow_pick_place_matrix_manifest.jsonl"
EXPECTED_REFERENCE_SHA256 = "eb0788ce41e10215f4ee730d53d52360cd1731a6e6a91999af385babd127b369"
EXPECTED_REFERENCE_CONTRACT = "3b3d0ae33b81452214e6f61b0352e84285e2e49ff3ef39169052be1f80083982"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_mapping(value: Any, key: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return candidate
        for child in value.values():
            found = _find_mapping(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping(child, key)
            if found is not None:
                return found
    return None


def _find_environment_audit(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    audit = record.get("audit")
    if isinstance(audit, Mapping):
        found = _find_mapping(audit, "environment_audit")
        if found is not None:
            return found
    return _find_mapping(record, "environment_audit")


def _cell_key(cell: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(cell["task_id"]), int(cell["episode_index"]), int(cell["seed"])


def _path_identity(value: Any) -> dict[str, Any]:
    """Keep path-independent BDDL identity when the archived path is gone."""
    if not value:
        return {"path": None, "basename": None, "sha256": None}
    path = Path(str(value))
    return {
        "path": str(value),
        "basename": path.name,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _compact_cell(cell: Mapping[str, Any], terminal: Mapping[str, Any] | None) -> dict[str, Any]:
    environment = _find_environment_audit(terminal or {}) or {}
    init = (terminal or {}).get("init_state_diagnostics")
    if not isinstance(init, Mapping):
        init = {}
    return {
        "cell_index": int(cell["cell_index"]),
        "task_id": int(cell["task_id"]),
        "episode_index": int(cell["episode_index"]),
        "seed": int(cell["seed"]),
        "init_state_index": int((terminal or {}).get("init_state_index", cell.get("init_state_index", -1))),
        "resolution": int(cell["resolution"]),
        "suite_mode": str(cell["suite_mode"]),
        "canonical_bddl": _path_identity(environment.get("canonical_bddl_file")),
        "applied_bddl": _path_identity(environment.get("applied_bddl_file")),
        "requested_removals": list(environment.get("requested_removals", [])),
        "applied_removals": list(environment.get("applied_removals", [])),
        "requested_swaps": list(environment.get("requested_swaps", [])),
        "applied_swaps": list(environment.get("applied_swaps", [])),
        "randomization_dimensions": dict(environment.get("randomization_dimensions", {})),
        "state_hash_sha256_pre_settle": environment.get("state_hash_sha256_pre_settle"),
        "state_hash_sha256": environment.get("state_hash_sha256"),
        "init_state_diagnostics": dict(init),
        "state_hash_evidence": "available" if environment.get("state_hash_sha256") and environment.get("state_hash_sha256_pre_settle") else "unavailable_due_archived_input_failure",
    }


class ReferenceProtocol:
    """Validated, serializable identity of the prior sealed matrix."""

    def __init__(self, *, root: Path, manifest_sha256: str, manifest: Mapping[str, Any], cells: list[dict[str, Any]]) -> None:
        self.root = root
        self.manifest_sha256 = manifest_sha256
        self.manifest = dict(manifest)
        self.cells = cells
        self.by_identity = {_cell_key(cell): cell for cell in cells}

    @property
    def contract_hash(self) -> str:
        return str(self.manifest.get("contract_hash", ""))

    def for_cell(self, task_id: int, episode_index: int, seed: int) -> Mapping[str, Any]:
        try:
            return self.by_identity[(int(task_id), int(episode_index), int(seed))]
        except KeyError as exc:
            raise RuntimeError(f"cell is absent from immutable reference protocol: {(task_id, episode_index, seed)}") from exc

    def validate_environment(self, env: Any, *, task_id: int, episode_index: int, seed: int) -> dict[str, Any]:
        """Compare the freshly built scene with the archived cell before motion."""
        expected = self.for_cell(task_id, episode_index, seed)
        observed_env = getattr(env, "_arrow_environment_audit", None)
        # The archived task6/e0 failure was recorded before the old runner
        # published environment_audit.  The immutable manifest still binds
        # this cell's task/episode/seed schedule; absence is retained as
        # evidence rather than treated as a reason to skip the new policy.
        if not isinstance(observed_env, Mapping):
            return {
                "status": "matched_schedule_only",
                "reference_manifest_sha256": self.manifest_sha256,
                "reference_contract_hash": self.contract_hash,
                "cell": dict(expected),
                "checks": {"task_id": True, "episode_index": True, "seed": True},
                "environment_audit": "unavailable_due_archived_input_failure",
            }
        observed_init = getattr(env, "_arrow_init_state_diagnostics", None)
        if not isinstance(observed_init, Mapping):
            raise RuntimeError("LIBERO environment did not publish init-state diagnostics")
        checks = {
            "task_id": int(task_id) == int(expected["task_id"]),
            "episode_index": int(episode_index) == int(expected["episode_index"]),
            "seed": int(seed) == int(expected["seed"]),
            "resolution": int(expected["resolution"]) == 256,
            "suite_mode": observed_env.get("suite_mode") == expected["suite_mode"],
            "init_state_index": int(observed_init.get("selected_index", -1)) == int(expected["init_state_index"]),
            "requested_removals": list(observed_env.get("requested_removals", [])) == list(expected["requested_removals"]),
            "applied_removals": list(observed_env.get("applied_removals", [])) == list(expected["applied_removals"]),
            "requested_swaps": list(observed_env.get("requested_swaps", [])) == list(expected["requested_swaps"]),
            "applied_swaps": list(observed_env.get("applied_swaps", [])) == list(expected["applied_swaps"]),
            "randomization_dimensions": dict(observed_env.get("randomization_dimensions", {})) == dict(expected["randomization_dimensions"]),
        }
        if expected.get("state_hash_evidence") == "unavailable_due_archived_input_failure":
            # task6/e0 failed before the archived environment audit was
            # persisted. Bind its deterministic schedule, suite, resolution,
            # and selected init state; do not compare absent archived fields.
            checks = {
                "task_id": int(task_id) == int(expected["task_id"]),
                "episode_index": int(episode_index) == int(expected["episode_index"]),
                "seed": int(seed) == int(expected["seed"]),
                "resolution": int(expected["resolution"]) == 256,
                "suite_mode": observed_env.get("suite_mode") == "sealed_randomized",
                "init_state_index": int(observed_init.get("selected_index", -1)) == int(expected["init_state_index"]),
                "current_sealed_randomization_contract": (
                    observed_env.get("prompt_provenance") in {None, "not_applicable_direct_runner"}
                    and isinstance(observed_env.get("requested_removals", []), list)
                    and isinstance(observed_env.get("applied_removals", []), list)
                    and isinstance(observed_env.get("requested_swaps", []), list)
                    and isinstance(observed_env.get("applied_swaps", []), list)
                ),
                "archived_state_hash_evidence": True,
            }
        if expected.get("state_hash_evidence") != "unavailable_due_archived_input_failure":
            expected_bddl = expected.get("canonical_bddl", {})
            observed_bddl = _path_identity(observed_env.get("canonical_bddl_file"))
            checks["canonical_bddl_basename"] = observed_bddl.get("basename") == expected_bddl.get("basename")
            if expected_bddl.get("sha256"):
                checks["canonical_bddl_sha256"] = observed_bddl.get("sha256") == expected_bddl.get("sha256")
        for name in ("state_hash_sha256_pre_settle", "state_hash_sha256"):
            reference_hash = expected.get(name)
            observed_hash = observed_env.get(name)
            # The archived task6/episode0 failure occurred before the old
            # runner published environment audit; preserve that limitation
            # instead of inventing a hash.  Deterministic schedule/BDDL and
            # init-state checks above still bind this cell.
            checks[name] = (expected.get("state_hash_evidence") == "unavailable_due_archived_input_failure") or (bool(reference_hash) and observed_hash == reference_hash)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"reference protocol mismatch before motion: {failed}")
        return {
            "status": "matched",
            "reference_manifest_sha256": self.manifest_sha256,
            "reference_contract_hash": self.contract_hash,
            "cell": dict(expected),
            "checks": checks,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "arrow_student_reference_protocol.v1",
            "source_root": self.root.as_posix(),
            "source_manifest": (self.root / REFERENCE_MANIFEST).as_posix(),
            "source_manifest_sha256": self.manifest_sha256,
            "source_contract_hash": self.contract_hash,
            "source_schema_version": self.manifest.get("schema_version"),
            "protocol": self.manifest.get("protocol"),
            "cells": self.cells,
        }


def load_reference_protocol(root: str | Path, *, require_expected_hash: bool = True) -> ReferenceProtocol:
    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / REFERENCE_MANIFEST
    terminal_path = root_path / REFERENCE_TERMINAL_MANIFEST
    if not manifest_path.is_file() or not terminal_path.is_file():
        raise FileNotFoundError(f"reference manifest pair is incomplete under {root_path}")
    actual_hash = _sha256(manifest_path)
    if require_expected_hash and actual_hash != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"reference manifest SHA-256 mismatch: expected {EXPECTED_REFERENCE_SHA256}, got {actual_hash}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("reference manifest must be a JSON object")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("reference manifest has no protocol")
    if manifest.get("contract_hash") != EXPECTED_REFERENCE_CONTRACT and require_expected_hash:
        raise RuntimeError("reference contract hash does not match archived 87-percent campaign")
    if protocol.get("suite_mode") != "sealed_randomized" or protocol.get("camera") != "agentview":
        raise RuntimeError("reference protocol is not the sealed agentview condition")
    if protocol.get("task_ids") != list(range(10)) or int(protocol.get("episodes_per_task", -1)) != 10 or int(protocol.get("seed_base", -1)) != 1000 or int(protocol.get("resolution", -1)) != 256:
        raise RuntimeError("reference protocol is not the exact 10x10 seed schedule")
    planned = manifest.get("cells")
    if not isinstance(planned, list) or len(planned) != 100:
        raise RuntimeError("reference manifest must contain exactly 100 planned cells")
    expected_schedule = [(task, episode, 1000 + episode) for task in range(10) for episode in range(10)]
    if [_cell_key(cell) for cell in planned] != expected_schedule:
        raise RuntimeError("reference manifest cell order or task/episode/seed schedule changed")
    terminal: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for line in terminal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, Mapping) and all(key in record for key in ("task_id", "episode_index", "seed")):
            terminal[_cell_key(record)] = record
    if len(terminal) < 100:
        raise RuntimeError(f"reference terminal manifest has only {len(terminal)} cells")
    cells = [_compact_cell(cell, terminal.get(_cell_key(cell))) for cell in planned]
    missing_hash_cells = [
        _cell_key(cell) for cell in cells
        if cell["state_hash_evidence"] == "unavailable_due_archived_input_failure"
    ]
    if missing_hash_cells not in ([], [(6, 0, 1000)]):
        raise RuntimeError(f"unexpected missing reference state-hash evidence: {missing_hash_cells}")
    return ReferenceProtocol(root=root_path, manifest_sha256=actual_hash, manifest=manifest, cells=cells)


__all__ = ["ReferenceProtocol", "load_reference_protocol"]
