"""Completion-manifest and schedule derivation helpers for Octo runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import EFFECTIVE_BATCH, OctoConfig

COMPLETION_SCHEMA = "octo_dataset_completion.v1"


def compute_optimizer_updates(*, transition_count: int, epochs: int, effective_batch: int = EFFECTIVE_BATCH) -> int:
    """Derive optimizer updates from verified transitions, never a schedule constant."""

    if int(transition_count) <= 0 or int(epochs) <= 0 or int(effective_batch) <= 0:
        raise ValueError("transition count, epochs, and effective batch must be positive")
    return int(math.ceil(int(transition_count) * int(epochs) / int(effective_batch)))


def _validate_fingerprint(fingerprint: Any) -> dict[str, Any]:
    if not isinstance(fingerprint, dict):
        raise ValueError("completion manifest fingerprint must be an object")
    required = {"sha256", "episode_count", "step_count", "action_count", "image_bytes", "action_bytes"}
    if not required <= set(fingerprint):
        raise ValueError(f"completion manifest fingerprint lacks {sorted(required - set(fingerprint))}")
    digest = str(fingerprint["sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError("completion manifest fingerprint sha256 is invalid")
    for key in required - {"sha256"}:
        if int(fingerprint[key]) < 0:
            raise ValueError(f"completion manifest fingerprint count {key} is negative")
    return dict(fingerprint)


def build_completion_manifest(
    *,
    config: OctoConfig,
    fingerprint: dict[str, Any],
    source_revision: str,
    completed: bool = True,
) -> dict[str, Any]:
    """Build a manifest binding source counts and the byte fingerprint."""

    checked = _validate_fingerprint(fingerprint)
    if int(checked["episode_count"]) != config.episodes:
        raise ValueError("fingerprint episode count does not match selected Octo mode")
    if int(checked["step_count"]) != config.timesteps or int(checked["action_count"]) != config.timesteps:
        raise ValueError("fingerprint transition count does not match selected Octo mode")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source_revision is required for Octo provenance")
    manifest = {
        "schema": COMPLETION_SCHEMA,
        "complete": bool(completed),
        "arrow_free": True,
        "mode": config.mode,
        "policy_kind": config.policy_kind,
        "dataset_name": config.dataset_name,
        "source_revision": source_revision,
        "episodes": config.episodes,
        "transitions": config.timesteps,
        "fingerprint": checked,
    }
    if config.mode == "matched_train":
        manifest["epochs"] = config.epochs
        manifest["effective_batch"] = EFFECTIVE_BATCH
        manifest["optimizer_updates"] = compute_optimizer_updates(
            transition_count=int(checked["step_count"]),
            epochs=config.epochs,
        )
    return manifest


def write_completion_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write one completion manifest without silently changing its contents."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    target.write_text(encoded, encoding="utf-8")
    return target


def load_and_validate_completion_manifest(path: str | Path, *, config: OctoConfig) -> dict[str, Any]:
    """Fail closed unless the completion manifest is complete and mode-bound."""

    target = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Octo completion manifest is unreadable: {target}") from exc
    if manifest.get("schema") != COMPLETION_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Octo dataset completion manifest is not complete or has an unsupported schema")
    if manifest.get("arrow_free") is not True:
        raise ValueError("Octo dataset completion manifest is not arrow-free")
    if manifest.get("mode") != config.mode or manifest.get("policy_kind") != config.policy_kind:
        raise ValueError("Octo completion manifest mode/policy provenance does not match selected mode")
    if int(manifest.get("episodes", -1)) != config.episodes or int(manifest.get("transitions", -1)) != config.timesteps:
        raise ValueError("Octo completion manifest counts do not match selected mode")
    fingerprint = _validate_fingerprint(manifest.get("fingerprint"))
    if int(fingerprint["episode_count"]) != config.episodes or int(fingerprint["step_count"]) != config.timesteps:
        raise ValueError("Octo completion fingerprint counts are inconsistent")
    if config.mode == "matched_train":
        derived = compute_optimizer_updates(
            transition_count=int(fingerprint["step_count"]), epochs=config.epochs
        )
        if int(manifest.get("optimizer_updates", -1)) != derived:
            raise ValueError("Octo optimizer updates are not derived from the verified manifest")
    return manifest


def completion_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Digest a normalized completion manifest for run provenance."""

    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
