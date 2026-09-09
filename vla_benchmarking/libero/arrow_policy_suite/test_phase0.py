from __future__ import annotations

import json

import pytest

from arrow_policy_suite.artifacts import (
    ArtifactError, ArtifactRef, LineageManifest, read_lineage_manifest,
    verify_artifact, write_artifact, write_lineage_manifest,
)
from arrow_policy_suite.contracts import ContractError, ObservationFrame, validate_student_observation
from arrow_policy_suite.splits import (
    ResetIdentity,
    SplitError,
    build_split_manifest,
    read_split_manifest,
    validate_split_manifests,
    write_split_manifest,
)


def _identity(name: str) -> ResetIdentity:
    return ResetIdentity(
        task_id=0,
        episode_id=name,
        seed=17,
        reset_index=1,
        observation_sha256="a" * 64,
        simulator_state_sha256="b" * 64,
        environment_fingerprint="libero-test-v1",
    )


def test_split_collision_is_rejected_across_manifests():
    train = build_split_manifest("train", [_identity("episode-1")])
    test = build_split_manifest("test", [_identity("episode-1")])
    with pytest.raises(SplitError, match="both"):
        validate_split_manifests((train, test))


def test_split_manifest_is_create_only_and_digest_checked(tmp_path):
    manifest = build_split_manifest("validation", [_identity("episode-2")])
    path = tmp_path / "validation.json"
    write_split_manifest(path, manifest)
    assert read_split_manifest(path).manifest_sha256 == manifest.manifest_sha256
    with pytest.raises(SplitError, match="refusing to mutate"):
        write_split_manifest(path, manifest)
    payload = json.loads(path.read_text())
    payload["identities"][0]["episode_id"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(SplitError, match="digest"):
        read_split_manifest(path)


def test_privileged_observation_fields_are_rejected_recursively():
    with pytest.raises(ContractError, match="privileged"):
        validate_student_observation({"state": [0.0] * 8, "object_pose_gt": [0.0] * 3})
    with pytest.raises(ContractError, match="privileged"):
        ObservationFrame({"state": [0.0] * 8, "nested": {"simulator_state": 1}}, 0)


def test_observation_frame_exposes_only_immutable_canonical_student_view():
    frame = ObservationFrame(
        {"state": [0.0] * 8, "instruction": "pick", "action": [0.5] * 7, "debug": "sidecar"},
        0,
    )
    assert set(frame.observation) == {"state", "instruction"}
    assert set(frame.student_observation) == {"state", "instruction"}
    assert set(frame.raw_observation) == {"state", "instruction", "action", "debug"}
    with pytest.raises(TypeError):
        frame.student_observation["instruction"] = "overwrite"  # type: ignore[index]
    with pytest.raises(TypeError):
        frame.student_observation["state"][0] = 1.0  # type: ignore[index]


def test_metadata_and_provenance_privileged_fields_fail_closed():
    with pytest.raises(ContractError, match="privileged"):
        ObservationFrame({"state": [0.0] * 8}, metadata={"nested": {"object_pose_gt": [0.0] * 3}})
    with pytest.raises(ContractError, match="privileged"):
        ObservationFrame({"state": [0.0] * 8}, provenance={"simulator_state_digest": "x"})


def test_student_validation_is_an_allowlist():
    with pytest.raises(ContractError, match="unknown/non-student"):
        validate_student_observation({"state": [0.0] * 8, "debug_sidecar": True})


def test_artifacts_are_immutable_and_verifiable(tmp_path):
    path = tmp_path / "raw.bin"
    ref = write_artifact(path, b"payload", kind="test", lineage=("c" * 64,))
    assert verify_artifact(ref).sha256 == ref.sha256
    with pytest.raises(ArtifactError, match="overwrite"):
        write_artifact(path, b"different", kind="test")
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="mismatch"):
        verify_artifact(ref)


def test_lineage_manifest_is_content_addressed_and_create_only(tmp_path):
    ref = write_artifact(tmp_path / "parent.bin", b"parent", kind="dataset")
    manifest = LineageManifest((ref,))
    manifest_path = tmp_path / "lineage.json"
    write_lineage_manifest(manifest_path, manifest)
    assert read_lineage_manifest(manifest_path).manifest_sha256 == manifest.manifest_sha256
    with pytest.raises(ArtifactError, match="overwrite"):
        write_lineage_manifest(manifest_path, manifest)
