"""Versioned provenance manifest shared by all VLA policy artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .source_contract import DatasetSourceContract, hash_json

MANIFEST_SCHEMA = "vla_policy_manifest.v1"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.IGNORECASE)


def _digest(name: str, value: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PolicyManifest:
    policy_kind: str
    artifact_id: str
    backend: str
    model_revision: str
    checkpoint_sha256: str
    dataset: DatasetSourceContract
    preprocessing_sha256: str
    training_sha256: str
    evaluation_sha256: str
    action_horizon: int
    action_dim: int = 7
    camera_keys: tuple[str, ...] = ("agentview",)
    state_dim: int | None = None
    arrow_absent: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.policy_kind).strip() or not str(self.artifact_id).strip():
            raise ValueError("policy_kind and artifact_id must be non-empty")
        if not str(self.backend).strip() or not str(self.model_revision).strip():
            raise ValueError("backend and model_revision must be non-empty")
        _digest("checkpoint_sha256", self.checkpoint_sha256)
        for name in ("preprocessing_sha256", "training_sha256", "evaluation_sha256"):
            _digest(name, getattr(self, name))
        if int(self.action_horizon) <= 0 or int(self.action_dim) != 7:
            raise ValueError("action_horizon must be positive and action_dim must be 7")
        if not self.camera_keys or any(not str(item).strip() for item in self.camera_keys):
            raise ValueError("camera_keys must contain non-empty names")
        if self.state_dim is not None and int(self.state_dim) < 0:
            raise ValueError("state_dim must be non-negative or None")
        if self.arrow_absent is not True:
            raise ValueError("matched no-arrow policy manifests must set arrow_absent=true")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "policy_kind": self.policy_kind,
            "artifact_id": self.artifact_id,
            "backend": self.backend,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset": self.dataset.as_dict(),
            "dataset_sha256": self.dataset.sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "training_sha256": self.training_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "action_horizon": int(self.action_horizon),
            "action_dim": int(self.action_dim),
            "camera_keys": list(self.camera_keys),
            "state_dim": self.state_dim,
            "arrow_absent": True,
            "extra": dict(self.extra),
        }

    @property
    def sha256(self) -> str:
        return hash_json(self.as_dict())


def build_policy_manifest(**kwargs: Any) -> dict[str, Any]:
    """Build and validate a serializable manifest from keyword fields."""

    manifest = PolicyManifest(**kwargs)
    result = manifest.as_dict()
    result["manifest_sha256"] = hash_json(result)
    return result


def build_plan_bindings(
    *,
    adapter_kind: str,
    artifact: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    io_receipt: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build validated v2 plan bindings from immutable receipts.

    Receipts may use provider-specific names, but each identity is normalized
    to a stable ``id``/``sha256`` field before it enters the hashed plan.
    Mutable checkpoint labels (``main``, ``latest``, etc.) are rejected.
    """

    if not str(adapter_kind).strip():
        raise ValueError("adapter_kind must be non-empty")
    artifact_id = artifact.get("artifact_id", artifact.get("id", artifact.get("model_id")))
    revision = artifact.get("checkpoint_revision", artifact.get("revision"))
    if not str(artifact_id).strip() or not str(revision).strip():
        raise ValueError("artifact receipt requires artifact id and checkpoint revision")
    if not _IMMUTABLE_REVISION.fullmatch(str(revision)):
        raise ValueError("artifact checkpoint revision must be an immutable 40- or 64-character SHA")
    checkpoint_sha256 = artifact.get("checkpoint_sha256", artifact.get("artifact_sha256"))
    if not checkpoint_sha256 or len(str(checkpoint_sha256)) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in str(checkpoint_sha256)
    ):
        raise ValueError("artifact receipt requires canonical checkpoint_sha256")

    def normalize(name: str, receipt: Mapping[str, Any], aliases: tuple[str, ...]) -> dict[str, Any]:
        identity = next((receipt[key] for key in aliases if receipt.get(key) is not None), None)
        if identity is None or not str(identity).strip():
            raise ValueError(f"{name} receipt lacks an immutable identity")
        normalized = dict(receipt)
        normalized.setdefault("id", str(identity))
        if name == "dataset_manifest":
            normalized.setdefault("manifest_sha256", str(identity))
        else:
            normalized.setdefault("sha256", str(identity))
        return normalized

    return {
        "adapter_kind": str(adapter_kind),
        "artifact": {
            **dict(artifact),
            "id": str(artifact_id),
            "revision": str(revision),
            "checkpoint_sha256": str(checkpoint_sha256).lower(),
        },
        "runtime": normalize("runtime", runtime_receipt, ("sha256", "runtime_sha256", "receipt_sha256", "id")),
        "io": normalize("io", io_receipt, ("sha256", "io_sha256", "receipt_sha256", "id")),
        "dataset_manifest": normalize(
            "dataset_manifest", dataset_manifest,
            ("manifest_sha256", "dataset_manifest_sha256", "manifest_file_sha256", "content_sha256", "sha256", "id"),
        ),
    }


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported policy manifest schema: {value.get('schema')!r}")
    if value.get("arrow_absent") is not True:
        raise ValueError("policy manifest is not arrow-free")
    dataset = value.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("policy manifest dataset contract is missing")
    if dataset.get("schema") != "vla_dataset_source_contract.v1":
        raise ValueError("policy manifest dataset schema is unsupported")
    contract = DatasetSourceContract(
        source_id=str(dataset["source_id"]), source_revision=str(dataset["source_revision"]),
        episode_count=int(dataset["episode_count"]), timestep_count=int(dataset["timestep_count"]),
        schema_sha256=str(dataset["schema_sha256"]), source_sha256=str(dataset["source_sha256"]),
        split_sha256=str(dataset["split_sha256"]), arrow_absent=bool(dataset.get("arrow_absent")),
        preprocessing_sha256=dataset.get("preprocessing_sha256"), extra=dataset.get("extra", {}),
    )
    if value.get("dataset_sha256") != contract.sha256:
        raise ValueError("policy manifest dataset hash is invalid")
    payload = dict(value)
    observed = payload.pop("manifest_sha256", None)
    if observed != hash_json(payload):
        raise ValueError("policy manifest hash is invalid")
    PolicyManifest(
        policy_kind=str(value["policy_kind"]), artifact_id=str(value["artifact_id"]),
        backend=str(value["backend"]), model_revision=str(value["model_revision"]),
        checkpoint_sha256=str(value["checkpoint_sha256"]), dataset=contract,
        preprocessing_sha256=str(value["preprocessing_sha256"]),
        training_sha256=str(value["training_sha256"]), evaluation_sha256=str(value["evaluation_sha256"]),
        action_horizon=int(value["action_horizon"]), action_dim=int(value.get("action_dim", 7)),
        camera_keys=tuple(value.get("camera_keys", ("agentview",))), state_dim=value.get("state_dim"),
        arrow_absent=True, extra=value.get("extra", {}),
    )
    return dict(value)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    checked = validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("policy manifest must contain a JSON object")
    return validate_manifest(value)


__all__ = [
    "MANIFEST_SCHEMA", "PolicyManifest", "build_plan_bindings", "build_policy_manifest",
    "load_manifest", "validate_manifest", "write_manifest",
]
