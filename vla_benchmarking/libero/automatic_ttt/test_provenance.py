from __future__ import annotations

import hashlib
import json

import pytest

from .provenance import (
    AUDIT_SCHEMA,
    ZeroShotAuditAttestation,
    ZeroShotEvidence,
    ZeroShotProvenanceError,
    validate_zero_shot_attestation,
    verify_zero_shot_evidence,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evidence(tmp_path, *, tasks=(0,), exposure="not_exposed", audit_payload=None, audit_digest=None):
    manifest_bytes = b"training-manifest"
    manifest_digest = _digest(manifest_bytes)
    payload = audit_payload or {
        "schema": AUDIT_SCHEMA,
        "source_repo": "https://example.invalid/vla",
        "source_revision": "a" * 40,
        "target_task_ids": [f"task{task}" for task in tasks],
        "claim_scope": [f"task{task}" for task in tasks],
        "declared_target_task_exposure": exposure,
        "exclusion_records": [
            {
                "task_id": f"task{task}",
                "status": "verified",
                "exposure": exposure,
                "dataset_manifest": "pretraining",
                "dataset_manifest_sha256": manifest_digest,
                "method": "manifest_task_exclusion_query",
                "source": "audit-db://run-17/task-exclusion",
            }
            for task in tasks
        ],
    }
    audit_path = tmp_path / "zero_shot_audit.json"
    audit_bytes = json.dumps(payload, sort_keys=True).encode()
    audit_path.write_bytes(audit_bytes)
    evidence = ZeroShotEvidence(
        model_bytes_sha256=_digest(b"model-bytes"),
        config_bytes_sha256=_digest(b"config-bytes"),
        processor_bytes_sha256=_digest(b"processor-bytes"),
        source_repo="https://example.invalid/vla",
        source_revision="a" * 40,
        dataset_manifest_digests={"pretraining": manifest_digest},
        declared_target_task_exposure=exposure,
        target_task_ids=tuple(tasks),
        audit_evidence_path=str(audit_path),
        audit_evidence_sha256=audit_digest or _digest(audit_bytes),
        claim_scope=tuple(tasks),
        model_id="test-model",
    )
    return evidence


def test_verified_evidence_issues_opaque_attestation(tmp_path):
    evidence = _evidence(tmp_path)

    attestation = verify_zero_shot_evidence(evidence, target_task_ids=[0])

    assert isinstance(attestation, ZeroShotAuditAttestation)
    validate_zero_shot_attestation(attestation, evidence=evidence, target_task_ids=["task0"])
    with pytest.raises(ZeroShotProvenanceError, match="opaque"):
        validate_zero_shot_attestation({"zero_shot": True}, evidence=evidence)


def test_self_authored_boolean_audit_is_rejected(tmp_path):
    evidence = _evidence(
        tmp_path,
        audit_payload={
            "schema": AUDIT_SCHEMA,
            "source_repo": "https://example.invalid/vla",
            "source_revision": "a" * 40,
            "target_task_ids": ["task0"],
            "claim_scope": ["task0"],
            "declared_target_task_exposure": "not_exposed",
            "zero_shot": True,
        },
    )

    with pytest.raises(ZeroShotProvenanceError, match="boolean"):
        verify_zero_shot_evidence(evidence, target_task_ids=[0])


def test_audit_digest_mismatch_is_rejected(tmp_path):
    evidence = _evidence(tmp_path, audit_digest=_digest(b"different-bytes"))

    with pytest.raises(ZeroShotProvenanceError, match="digest mismatch"):
        verify_zero_shot_evidence(evidence, target_task_ids=[0])


def test_unknown_and_contaminated_exposure_are_rejected(tmp_path):
    for exposure in ("unknown", "contaminated", "exposed"):
        with pytest.raises(ZeroShotProvenanceError, match="not an auditable"):
            _evidence(tmp_path, exposure=exposure)


def test_task_mismatch_is_rejected_before_claim(tmp_path):
    evidence = _evidence(tmp_path, tasks=(0,))

    with pytest.raises(ZeroShotProvenanceError, match="target-task mismatch"):
        verify_zero_shot_evidence(evidence, target_task_ids=[1])


def test_uri_only_audit_evidence_is_rejected(tmp_path):
    evidence = _evidence(tmp_path)
    evidence = ZeroShotEvidence(
        **{**evidence.to_dict(), "audit_evidence_path": "https://example.invalid/audit.json"}
    )

    with pytest.raises(ZeroShotProvenanceError, match="URI"):
        verify_zero_shot_evidence(evidence, target_task_ids=[0])


def test_dataset_manifest_digest_mismatch_is_rejected(tmp_path):
    evidence = _evidence(tmp_path)
    payload = json.loads((tmp_path / "zero_shot_audit.json").read_text())
    payload["exclusion_records"][0]["dataset_manifest_sha256"] = _digest(b"wrong-manifest")
    audit_bytes = json.dumps(payload, sort_keys=True).encode()
    (tmp_path / "zero_shot_audit.json").write_bytes(audit_bytes)
    evidence = ZeroShotEvidence(**{**evidence.to_dict(), "audit_evidence_sha256": _digest(audit_bytes)})

    with pytest.raises(ZeroShotProvenanceError, match="manifest digest mismatch"):
        verify_zero_shot_evidence(evidence, target_task_ids=[0])


def test_evidence_is_immutable(tmp_path):
    evidence = _evidence(tmp_path)
    with pytest.raises(TypeError):
        evidence.dataset_manifest_digests["another"] = "0" * 64

