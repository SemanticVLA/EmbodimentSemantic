"""Deterministic, policy-independent evaluation plans.

The two production evaluators keep their native execution loops, but they must
agree on the cells they claim to evaluate.  This module is intentionally
dependency-light: it only serializes the policy/condition contract, the
row-major task/episode/seed schedule, and the configured randomization payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    DEFAULT_RESOLUTION,
    DEFAULT_SEED_BASE,
    DEFAULT_TASK_IDS,
    EvaluationCell,
    EvaluationCondition,
    build_task_seed_matrix,
)
from .randomization_contract import randomization_config_payload
from .registry import validate_policy_condition


LEGACY_PLAN_SCHEMA = "shared_evaluation_plan.v1"
PLAN_SCHEMA = LEGACY_PLAN_SCHEMA
PLAN_SCHEMA_V2 = "shared_evaluation_plan.v2"
SUPPORTED_PLAN_SCHEMAS = (LEGACY_PLAN_SCHEMA, PLAN_SCHEMA_V2)
_V2_POLICY_KINDS = frozenset({
    "pi05", "openvla", "openvla_oft", "octo_community_multisuite_190k",
    "octo_base15_spatial_no_arrow_matched",
})
_V2_BINDING_KEYS = frozenset({"artifact", "runtime", "io", "dataset_manifest", "adapter_kind"})
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.IGNORECASE)
_IMMUTABLE_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
SCHEDULE_SCHEMA = "shared_evaluation_schedule.v1"


def _validate_v2_bindings(bindings: Mapping[str, Any]) -> None:
    """Validate the receipt identities embedded in a v2 plan.

    The function is shared by plan construction and loading so a plan cannot
    be emitted with weaker provenance checks than the evaluator applies later.
    """

    missing = sorted(_V2_BINDING_KEYS.difference(bindings))
    if missing:
        raise ValueError("v2 evaluation plans require bindings for: " + ", ".join(missing))
    if not str(bindings["adapter_kind"]).strip():
        raise ValueError("v2 adapter_kind binding must be non-empty")
    for binding_name in ("artifact", "runtime", "io", "dataset_manifest"):
        binding = bindings[binding_name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"v2 {binding_name} binding must be an object")
        if not any(str(binding.get(key, "")).strip() for key in ("id", "sha256", "checkpoint_sha256", "manifest_sha256", "revision")):
            raise ValueError(f"v2 {binding_name} binding has no identity")
    artifact = bindings["artifact"]
    artifact_id = artifact.get("artifact_id", artifact.get("id"))
    if not str(artifact_id or "").strip():
        raise ValueError("v2 artifact binding requires an artifact id")
    revision = artifact.get("checkpoint_revision", artifact.get("revision"))
    if not revision or not _IMMUTABLE_REVISION.fullmatch(str(revision)):
        raise ValueError("v2 artifact binding requires an immutable 40- or 64-character checkpoint SHA")
    checkpoint_sha256 = artifact.get("checkpoint_sha256", artifact.get("artifact_sha256"))
    if not checkpoint_sha256 or not _IMMUTABLE_SHA256.fullmatch(str(checkpoint_sha256)):
        raise ValueError("v2 artifact binding requires a canonical checkpoint_sha256")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_source_hashes(repo_root: str | Path | None = None) -> dict[str, str | None]:
    """Hash the source files that define the shared evaluation contract.

    Policy-specific source hashes remain in each native manifest.  These four
    hashes are deliberately identical for matched controller/VLA plans and
    make it possible to tell whether the common contract changed.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    paths = (
        "vla_benchmarking/libero/evaluation/contracts.py",
        "vla_benchmarking/libero/evaluation/plan.py",
        "vla_benchmarking/libero/evaluation/policy_adapter.py",
        "vla_benchmarking/libero/evaluation/registry.py",
        "vla_benchmarking/libero/evaluation/randomization_contract.py",
        "vla_benchmarking/libero/finetuned_vlas/common/manifest.py",
        "vla_benchmarking/libero/finetuned_vlas/common/source_contract.py",
        "vla_benchmarking/libero/shared/config.py",
    )
    hashes: dict[str, str | None] = {relative: _sha256_file(root / relative) for relative in paths}
    # LIBERO task definitions are supplied by a separately maintained nested
    # checkout on Legion.  Bind that exact checkout when a launcher provides
    # it, so plans cannot silently drift away from the evaluated BDDL files.
    if os.environ.get("LIBERO_SOURCE_ROOT"):
        hashes.update(libero_source_hashes())
    return hashes


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"source tree is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def libero_source_hashes() -> dict[str, str]:
    """Return and validate the nested LIBERO checkout provenance."""

    source_root = Path(os.environ["LIBERO_SOURCE_ROOT"]).expanduser().resolve()
    bddl_root = Path(os.environ["LIBERO_BDDL_SOURCE"]).expanduser().resolve()
    if not source_root.is_dir() or not bddl_root.is_dir():
        raise ValueError("LIBERO source/BDDL roots are missing")
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"nested LIBERO checkout is dirty: {source_root}")
    expected = os.environ.get("LIBERO_SOURCE_COMMIT", "").strip()
    if expected and commit != expected:
        raise ValueError(f"nested LIBERO commit mismatch: {commit} != {expected}")
    return {
        "libero_source_commit": commit,
        "libero_bddl_tree_sha256": _tree_sha256(bddl_root),
    }


def validate_libero_source_hashes(plan: Mapping[str, Any]) -> None:
    """Fail closed if the runtime LIBERO task definitions differ from plan."""

    expected = plan.get("source_hashes", {})
    if not isinstance(expected, Mapping) or "libero_source_commit" not in expected:
        return
    observed = libero_source_hashes()
    for key in ("libero_source_commit", "libero_bddl_tree_sha256"):
        if str(expected.get(key, "")) != str(observed.get(key, "")):
            raise ValueError(f"LIBERO provenance mismatch for {key}")


def _schedule_cells(cells: Sequence[EvaluationCell]) -> list[dict[str, int]]:
    return [
        {
            "cell_index": int(cell.cell_index),
            "task_id": int(cell.task_id),
            "episode_index": int(cell.episode_index),
            "seed": int(cell.seed),
            "init_state_index": int(cell.init_state_index),
        }
        for cell in cells
    ]


def schedule_hash(cells: Sequence[EvaluationCell] | Sequence[Mapping[str, Any]]) -> str:
    """Hash only row-major cell identity, independent of policy or inputs."""

    normalized: list[dict[str, int]] = []
    for position, cell in enumerate(cells):
        if isinstance(cell, EvaluationCell):
            normalized.append(
                {
                    "cell_index": int(cell.cell_index),
                    "task_id": int(cell.task_id),
                    "episode_index": int(cell.episode_index),
                    "seed": int(cell.seed),
                    "init_state_index": int(cell.init_state_index),
                }
            )
            continue
        normalized.append(
            {
                "cell_index": int(cell.get("cell_index", position)),
                "task_id": int(cell["task_id"]),
                "episode_index": int(cell["episode_index"]),
                "seed": int(cell["seed"]),
                "init_state_index": int(cell.get("init_state_index", cell.get("episode_index", 0))),
            }
        )
    return canonical_sha256({"schema": SCHEDULE_SCHEMA, "cells": normalized})


def _prompt_applicability(policy_kind: str, condition: EvaluationCondition) -> str:
    capabilities = validate_policy_condition(policy_kind, condition)
    if hasattr(capabilities, "prompt_applicability"):
        value = capabilities.prompt_applicability.get(condition.suite_mode, "not_applicable")
        return str(value)
    return "not_applicable"


def build_evaluation_plan(
    *,
    policy_kind: str,
    suite_mode: str,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed_base: int = DEFAULT_SEED_BASE,
    camera: str = "agentview",
    resolution: int = DEFAULT_RESOLUTION,
    text_context: str = "none",
    text_format: str | None = None,
    visual_input: str = "none",
    visual_arrow: bool | None = None,
    randomization: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str | None] | None = None,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single serialized plan consumed by either native evaluator.

    New VLA plans are v2 and must bind the exact artifact, runtime, I/O,
    dataset-manifest identity, and adapter kind.  Historical policies retain
    v1 output and intentionally do not require those bindings.
    """

    condition = EvaluationCondition(
        suite_mode=suite_mode,
        camera=camera,
        resolution=resolution,
        text_context=text_context,
        text_format=text_format,
        visual_input=visual_input,
        visual_arrow=visual_arrow,
    )
    capabilities = validate_policy_condition(policy_kind, condition)
    cells = build_task_seed_matrix(
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        seed_base=seed_base,
        suite_mode=condition.suite_mode,
        resolution=condition.resolution,
        camera=condition.camera,
        text_context=condition.text_context,
        text_format=condition.text_format,
        visual_input=condition.visual_input,
    )
    payload = dict(randomization if randomization is not None else randomization_config_payload())
    schedule = _schedule_cells(cells)
    plan_schema = PLAN_SCHEMA_V2 if str(policy_kind) in _V2_POLICY_KINDS else PLAN_SCHEMA
    normalized_bindings = dict(bindings or {})
    if plan_schema == PLAN_SCHEMA_V2:
        _validate_v2_bindings(normalized_bindings)
    plan = {
        # Keep historical SmolVLA/controller manifests byte-compatible. New
        # native VLA policies opt into v2 without invalidating old results.
        "schema": plan_schema,
        "policy_kind": str(policy_kind),
        "condition": condition.as_dict(),
        "text_contract": capabilities.text_contract,
        "prompt_applicability": _prompt_applicability(policy_kind, condition),
        "randomization": {
            "payload": payload,
            "sha256": canonical_sha256(payload),
            "enabled": condition.suite_mode == "sealed_randomized",
        },
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "cells": schedule,
            "sha256": schedule_hash(cells),
        },
        "source_hashes": dict(sorted((source_hashes or shared_source_hashes()).items())),
    }
    if plan_schema == PLAN_SCHEMA_V2:
        plan["bindings"] = normalized_bindings
    plan["sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized plan and its internal hashes before execution."""

    if plan.get("schema") not in SUPPORTED_PLAN_SCHEMAS:
        raise ValueError(f"unsupported evaluation plan schema: {plan.get('schema')!r}")
    schedule = plan.get("schedule")
    if not isinstance(schedule, Mapping) or schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("evaluation plan schedule schema is invalid")
    cells = schedule.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("evaluation plan must contain non-empty schedule cells")
    if schedule.get("sha256") != schedule_hash(cells):
        raise ValueError("evaluation plan schedule hash is invalid")
    randomization = plan.get("randomization")
    if not isinstance(randomization, Mapping) or randomization.get("sha256") != canonical_sha256(randomization.get("payload")):
        raise ValueError("evaluation plan randomization hash is invalid")
    if plan.get("schema") == PLAN_SCHEMA_V2:
        bindings = plan.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError("v2 evaluation plan bindings are missing")
        _validate_v2_bindings(bindings)
    expected = dict(plan)
    observed_hash = expected.pop("sha256", None)
    if observed_hash != canonical_sha256(expected):
        raise ValueError("evaluation plan hash is invalid")
    return dict(plan)


def validate_native_schedule(
    plan: Mapping[str, Any], native_cells: Sequence[Mapping[str, Any]]
) -> None:
    """Fail closed if a native runner's planned task/seed schedule drifts."""

    checked = validate_plan(plan)
    expected = checked["schedule"]["cells"]
    actual = [
        {
            "cell_index": int(cell.get("cell_index", position)),
            "task_id": int(cell["task_id"]),
            "episode_index": int(cell["episode_index"]),
            "seed": int(cell["seed"]),
            "init_state_index": int(cell.get("init_state_index", cell.get("episode_index", 0))),
        }
        for position, cell in enumerate(native_cells)
    ]
    if actual != expected:
        raise ValueError("native planned cell schedule differs from shared evaluation plan")


__all__ = [
    "PLAN_SCHEMA",
    "LEGACY_PLAN_SCHEMA",
    "PLAN_SCHEMA_V2",
    "SCHEDULE_SCHEMA",
    "SUPPORTED_PLAN_SCHEMAS",
    "canonical_json",
    "canonical_sha256",
    "shared_source_hashes",
    "schedule_hash",
    "build_evaluation_plan",
    "validate_plan",
    "validate_native_schedule",
]
