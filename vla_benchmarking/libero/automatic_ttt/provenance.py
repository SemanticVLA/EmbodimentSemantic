"""Fail-closed zero-shot provenance for automatic test-time training.

The automatic-TTT experiment can only make a zero-shot claim when the
checkpoint bytes, model configuration, processor, source revision, and
training-data manifests have a stable identity.  A model-side flag such as
``is_zero_shot=True`` is not evidence.  This module therefore separates an
immutable *declaration* (``ZeroShotEvidence``) from an opaque capability
(``ZeroShotAuditAttestation``) which is issued only after the verifier has
read and hashed an audit artifact.

The audit artifact is deliberately concrete.  It must contain one verified
exclusion record for every target task and tie each record to one of the
declared dataset-manifest digests.  Unknown, contaminated, boolean-only, or
task-mismatched declarations are rejected.

This module does not verify any real VLA checkpoint by itself.  A caller must
provide paths and digests for the actual run artifacts; no placeholder is
accepted by the verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
AUDIT_SCHEMA = "robottt.zero_shot_audit.v1"
ZERO_SHOT_EXPOSURES = frozenset({"not_exposed", "zero_shot", "unseen"})
_ATTESTATION_TOKEN = object()
_ISSUED_ATTESTATIONS: "weakref.WeakSet[ZeroShotAuditAttestation]" = weakref.WeakSet()


class ZeroShotProvenanceError(ValueError):
    """Raised when zero-shot provenance cannot be verified."""


class ZeroShotAuditAttestation:
    """Opaque verifier-issued proof of a scoped zero-shot audit.

    The constructor is intentionally not a public assertion API.  It requires
    a private module token and is only called by
    :func:`verify_zero_shot_evidence`.  The object is also tracked by weak
    identity so a deserialized dictionary cannot be substituted for proof.
    """

    __slots__ = ("_payload", "_token", "_evidence_id", "__weakref__")

    def __init__(self, payload: Mapping[str, Any], token: object, evidence: "ZeroShotEvidence") -> None:
        if token is not _ATTESTATION_TOKEN:
            raise TypeError(
                "ZeroShotAuditAttestation is verifier-issued; call verify_zero_shot_evidence()"
            )
        self._payload = _freeze_json(payload)
        self._token = token
        self._evidence_id = id(evidence)

    def _as_mapping(self) -> Mapping[str, Any]:
        """Internal read-only payload access for package integrations."""

        return self._payload

    @property
    def claim_scope(self) -> tuple[str, ...]:
        return tuple(self._payload["claim_scope"])

    @property
    def evidence_id(self) -> int:
        """Identity of the immutable evidence object that was checked."""

        return self._evidence_id


@dataclass(frozen=True)
class ZeroShotEvidence:
    """Immutable byte/configuration/data declaration for one zero-shot claim.

    ``target_task_ids`` and ``claim_scope`` are normalized to canonical task
    strings.  ``claim_scope`` must name exactly the same tasks as
    ``target_task_ids``; broad labels such as ``all_tasks`` are not accepted.
    The audit file is checked later by :func:`verify_zero_shot_evidence`,
    because checking its bytes is an external side effect.
    """

    model_bytes_sha256: str
    config_bytes_sha256: str
    processor_bytes_sha256: str
    source_repo: str
    source_revision: str
    dataset_manifest_digests: Mapping[str, str]
    declared_target_task_exposure: str
    target_task_ids: tuple[str | int, ...]
    audit_evidence_path: str
    audit_evidence_sha256: str
    claim_scope: tuple[str | int, ...]
    # Optional model identifier is descriptive only; the digest remains the
    # identity used by the verifier.
    model_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_manifest_digests", MappingProxyType(dict(self.dataset_manifest_digests)))
        object.__setattr__(self, "target_task_ids", _normalize_task_sequence(self.target_task_ids, "target_task_ids"))
        object.__setattr__(self, "claim_scope", _normalize_task_sequence(self.claim_scope, "claim_scope"))

        for field_name in (
            "model_bytes_sha256",
            "config_bytes_sha256",
            "processor_bytes_sha256",
            "audit_evidence_sha256",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ZeroShotProvenanceError(f"{field_name} must be a 64-character SHA-256 byte digest")

        for field_name in ("source_repo", "source_revision", "audit_evidence_path"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
                raise ZeroShotProvenanceError(f"{field_name} must be a non-empty string")
            if value.strip().lower() in {"unknown", "latest", "unverified", "none", "null"}:
                raise ZeroShotProvenanceError(f"{field_name} cannot use an unresolved placeholder")

        if isinstance(self.declared_target_task_exposure, bool) or not isinstance(
            self.declared_target_task_exposure, str
        ):
            raise ZeroShotProvenanceError("declared_target_task_exposure must be an explicit string, not a boolean")
        exposure = self.declared_target_task_exposure.strip().lower()
        if exposure not in ZERO_SHOT_EXPOSURES:
            raise ZeroShotProvenanceError(
                "declared_target_task_exposure is not an auditable zero-shot value: " + repr(exposure)
            )
        object.__setattr__(self, "declared_target_task_exposure", exposure)

        if not self.dataset_manifest_digests:
            raise ZeroShotProvenanceError("dataset_manifest_digests must be non-empty")
        for manifest_name, digest in self.dataset_manifest_digests.items():
            if isinstance(manifest_name, bool) or not isinstance(manifest_name, str) or not manifest_name.strip():
                raise ZeroShotProvenanceError("dataset manifest names must be non-empty strings")
            if isinstance(digest, bool) or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ZeroShotProvenanceError(f"dataset manifest digest for {manifest_name!r} is not SHA-256")

        if not self.target_task_ids:
            raise ZeroShotProvenanceError("target_task_ids must be non-empty")
        if set(self.target_task_ids) != set(self.claim_scope):
            raise ZeroShotProvenanceError("claim_scope must match target_task_ids exactly")

    @property
    def model_sha256(self) -> str:
        """Compatibility alias for integrations that call this a model digest."""

        return self.model_bytes_sha256

    @property
    def config_sha256(self) -> str:
        return self.config_bytes_sha256

    @property
    def processor_sha256(self) -> str:
        return self.processor_bytes_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_bytes_sha256": self.model_bytes_sha256,
            "config_bytes_sha256": self.config_bytes_sha256,
            "processor_bytes_sha256": self.processor_bytes_sha256,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "dataset_manifest_digests": dict(self.dataset_manifest_digests),
            "declared_target_task_exposure": self.declared_target_task_exposure,
            "target_task_ids": list(self.target_task_ids),
            "audit_evidence_path": self.audit_evidence_path,
            "audit_evidence_sha256": self.audit_evidence_sha256,
            "claim_scope": list(self.claim_scope),
            "model_id": self.model_id,
        }


def verify_zero_shot_evidence(
    evidence: ZeroShotEvidence,
    *,
    target_task_ids: Sequence[str | int],
) -> ZeroShotAuditAttestation:
    """Verify an audit artifact and issue an opaque zero-shot attestation.

    The verifier requires exact target-task identity.  It reads only local
    byte-addressable files: URI-only audit evidence is rejected because its
    bytes cannot be independently checked here.  The caller cannot unlock the
    claim by supplying a boolean; the artifact must contain concrete,
    digest-linked exclusion records.
    """

    if not isinstance(evidence, ZeroShotEvidence):
        raise ZeroShotProvenanceError("evidence must be a ZeroShotEvidence instance")
    requested = _normalize_task_sequence(target_task_ids, "target_task_ids")
    if set(requested) != set(evidence.target_task_ids):
        raise ZeroShotProvenanceError(
            f"target-task mismatch: requested={list(requested)!r}, evidence={list(evidence.target_task_ids)!r}"
        )
    path_text = evidence.audit_evidence_path.strip()
    if "://" in path_text:
        raise ZeroShotProvenanceError("audit evidence URI has no locally verifiable bytes")
    path = Path(path_text)
    if not path.is_file():
        raise ZeroShotProvenanceError(f"audit evidence file does not exist: {path_text}")
    actual_digest = _sha256_file(path)
    if actual_digest.lower() != evidence.audit_evidence_sha256.lower():
        raise ZeroShotProvenanceError(
            "audit evidence digest mismatch: "
            f"declared={evidence.audit_evidence_sha256}, actual={actual_digest}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ZeroShotProvenanceError(f"audit evidence is not valid UTF-8 JSON: {exc}") from exc
    _verify_audit_payload(payload, evidence, requested)

    attestation_payload = {
        "schema": AUDIT_SCHEMA,
        "claim_scope": list(evidence.claim_scope),
        "target_task_ids": list(evidence.target_task_ids),
        "audit_evidence_sha256": evidence.audit_evidence_sha256.lower(),
        "model_bytes_sha256": evidence.model_bytes_sha256.lower(),
        "config_bytes_sha256": evidence.config_bytes_sha256.lower(),
        "processor_bytes_sha256": evidence.processor_bytes_sha256.lower(),
        "source_repo": evidence.source_repo,
        "source_revision": evidence.source_revision,
        "verification": "byte_hash_and_task_exclusion_records",
    }
    attestation = ZeroShotAuditAttestation(attestation_payload, _ATTESTATION_TOKEN, evidence)
    _ISSUED_ATTESTATIONS.add(attestation)
    return attestation


def validate_zero_shot_attestation(
    attestation: ZeroShotAuditAttestation | Mapping[str, Any] | None,
    *,
    evidence: ZeroShotEvidence | None = None,
    target_task_ids: Sequence[str | int] | None = None,
) -> None:
    """Require a live verifier-issued attestation at a model integration boundary."""

    if not isinstance(attestation, ZeroShotAuditAttestation) or attestation not in _ISSUED_ATTESTATIONS:
        raise ZeroShotProvenanceError("zero-shot proof must be an opaque verifier-issued attestation")
    if evidence is not None and attestation.evidence_id != id(evidence):
        raise ZeroShotProvenanceError("zero-shot attestation belongs to a different evidence object")
    if target_task_ids is not None:
        expected = set(_normalize_task_sequence(target_task_ids, "target_task_ids"))
        if set(attestation.claim_scope) != expected:
            raise ZeroShotProvenanceError("zero-shot attestation scope does not match target tasks")


def _verify_audit_payload(
    payload: Any,
    evidence: ZeroShotEvidence,
    requested: tuple[str, ...],
) -> None:
    if not isinstance(payload, Mapping):
        raise ZeroShotProvenanceError("audit evidence root must be a JSON object")
    _reject_boolean_assertions(payload)
    if payload.get("schema", payload.get("schema_version")) != AUDIT_SCHEMA:
        raise ZeroShotProvenanceError(f"audit evidence must declare schema {AUDIT_SCHEMA!r}")
    if payload.get("source_repo") != evidence.source_repo or payload.get("source_revision") != evidence.source_revision:
        raise ZeroShotProvenanceError("audit source repository/revision does not match evidence")
    if _normalize_task_sequence(payload.get("target_task_ids", ()), "audit target_task_ids") != requested:
        raise ZeroShotProvenanceError("audit target_task_ids do not match the requested task scope")
    audit_scope = _normalize_task_sequence(payload.get("claim_scope", ()), "audit claim_scope")
    if set(audit_scope) != set(requested):
        raise ZeroShotProvenanceError("audit claim_scope does not match target tasks")
    if payload.get("declared_target_task_exposure") != evidence.declared_target_task_exposure:
        raise ZeroShotProvenanceError("audit target-task exposure does not match evidence")

    records = payload.get("exclusion_records", payload.get("audit_records", payload.get("records")))
    if not isinstance(records, list) or not records:
        raise ZeroShotProvenanceError("audit evidence requires non-empty exclusion_records")
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ZeroShotProvenanceError(f"audit exclusion record {index} is not an object")
        task = _normalize_task_id(record.get("task_id"), f"record[{index}].task_id")
        if task not in requested:
            raise ZeroShotProvenanceError(f"audit record task {task!r} is outside requested scope")
        if task in seen:
            raise ZeroShotProvenanceError(f"duplicate audit exclusion record for {task}")
        seen.add(task)
        if record.get("status") != "verified":
            raise ZeroShotProvenanceError(f"audit record {task} is not verifier-verified")
        if record.get("exposure") != evidence.declared_target_task_exposure:
            raise ZeroShotProvenanceError(f"audit record {task} has an incompatible exposure declaration")
        manifest_name = record.get("dataset_manifest")
        if not isinstance(manifest_name, str) or manifest_name not in evidence.dataset_manifest_digests:
            raise ZeroShotProvenanceError(f"audit record {task} is not tied to a declared dataset manifest")
        if record.get("dataset_manifest_sha256") != evidence.dataset_manifest_digests[manifest_name]:
            raise ZeroShotProvenanceError(f"audit record {task} has a dataset manifest digest mismatch")
        # A status alone is a self-authored assertion.  Require a concrete
        # audit method and source location/identifier for every task record.
        method = record.get("method")
        source = record.get("source")
        if not isinstance(method, str) or not method.strip() or not isinstance(source, str) or not source.strip():
            raise ZeroShotProvenanceError(f"audit record {task} lacks concrete method/source evidence")
    if seen != set(requested):
        missing = sorted(set(requested) - seen)
        raise ZeroShotProvenanceError(f"audit evidence omits target tasks: {missing}")


def _normalize_task_sequence(value: Sequence[str | int], field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ZeroShotProvenanceError(f"{field_name} must be a sequence of task IDs, not a string")
    try:
        normalized = tuple(_normalize_task_id(item, field_name) for item in value)
    except TypeError as exc:
        raise ZeroShotProvenanceError(f"{field_name} must be a sequence of task IDs") from exc
    if len(normalized) != len(set(normalized)):
        raise ZeroShotProvenanceError(f"{field_name} contains duplicate task IDs")
    return normalized


def _normalize_task_id(value: Any, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ZeroShotProvenanceError(f"{field_name} contains an invalid task ID")
    text = str(value).strip()
    if not text:
        raise ZeroShotProvenanceError(f"{field_name} contains an empty task ID")
    if text.isdigit():
        return f"task{int(text)}"
    return text.lower()


def _reject_boolean_assertions(value: Any, *, path: str = "audit") -> None:
    if isinstance(value, bool):
        raise ZeroShotProvenanceError(f"{path} contains a self-authored boolean assertion")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_boolean_assertions(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_boolean_assertions(item, path=f"{path}[{index}]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ZeroShotProvenanceError(f"could not hash audit evidence: {exc}") from exc
    return digest.hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


__all__ = [
    "AUDIT_SCHEMA",
    "SHA256_RE",
    "ZERO_SHOT_EXPOSURES",
    "ZeroShotAuditAttestation",
    "ZeroShotEvidence",
    "ZeroShotProvenanceError",
    "validate_zero_shot_attestation",
    "verify_zero_shot_evidence",
]
