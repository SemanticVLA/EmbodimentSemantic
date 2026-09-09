from __future__ import annotations

from dataclasses import replace

import pytest

from arrow_policy_suite.config import FrozenStudyManifest, StudyConfig
from arrow_policy_suite.contracts import ContractError
from arrow_policy_suite.splits import (
    ResetIdentity,
    SplitError,
    build_split_manifest,
    split_set_sha256,
    validate_complete_split_manifests,
)


def _config(**overrides):
    tasks = (0, 1)
    payload = {
        "task_ids": tasks,
        "test_reset_ids": {task: tuple(f"test-{task}-{index}" for index in range(10)) for task in tasks},
        "validation_reset_ids": {task: tuple(f"validation-{task}-{index}" for index in range(10)) for task in tasks},
        "trace_geometry_provider_revision": "trace-calibration-1",
        "model_revision": "model-1",
        "controller_revision": "controller-1",
        "calibration_revision": "calibration-1",
        "environment_revision": "environment-1",
        "processor_revision": "processor-1",
        "action_encoding": "normalized_osc7_v1",
        "n_action_steps": 1,
        "camera_names": ("agentview", "robot0_eye_in_hand"),
        "image_resolution": (256, 256),
        "depth_units": "meters",
        "renderer_revision": "renderer-1",
        "metric_revision": "success_at_1200_v1",
    }
    payload.update(overrides)
    return StudyConfig(**payload)


def test_config_requires_exactly_ten_test_and_validation_ids_per_task():
    config = _config(test_reset_ids={0: tuple(f"x-{index}" for index in range(9)), 1: tuple(f"x-{index}" for index in range(10))})
    with pytest.raises(ContractError, match="exactly ten frozen test"):
        config.validate()


def test_config_seal_round_trips_and_detects_mutation():
    manifest = _config().manifest()
    StudyConfig.verify_manifest(manifest)
    restored = StudyConfig.from_manifest(manifest)
    assert restored.config_sha256() == manifest["config_sha256"]
    tampered = dict(manifest)
    tampered["model_revision"] = "model-2"
    with pytest.raises(ContractError, match="config_sha256"):
        StudyConfig.verify_manifest(tampered)


def test_config_rejects_unresolved_revisions_and_global_reset_overlap():
    with pytest.raises(ContractError, match="model_revision"):
        _config(model_revision="unresolved").validate()
    overlapping = _config(validation_reset_ids={0: tuple(f"test-0-{index}" for index in range(10)), 1: tuple(f"validation-1-{index}" for index in range(10))})
    with pytest.raises(ContractError, match="reuses|overlap"):
        overlapping.validate()


def test_config_freezes_policy_interface_pins_in_digest():
    config = _config()
    config.validate()
    assert config.n_action_steps == 1
    assert config.image_resolution == (256, 256)
    assert config.config_sha256() != _config(metric_revision="success_at_280_v1").config_sha256()
    with pytest.raises(ContractError, match="processor_revision"):
        _config(processor_revision="unresolved").validate()
    with pytest.raises(ContractError, match="action_encoding"):
        _config(action_encoding="unknown").validate()
    with pytest.raises(ContractError, match="n_action_steps"):
        _config(n_action_steps=2).validate()


def _identity(task: int, name: str, query: int) -> ResetIdentity:
    return ResetIdentity(
        task_id=task,
        episode_id=name,
        seed=17,
        reset_index=1,
        query_index=query,
        observation_sha256="a" * 64,
        simulator_state_sha256=f"{query:064x}",
        environment_fingerprint="environment-1",
    )


def test_complete_split_validation_requires_both_sides_and_stable_set_digest():
    test = build_split_manifest("test", [_identity(0, f"test-{index}", index) for index in range(10)])
    validation = build_split_manifest("validation", [_identity(0, f"validation-{index}", index + 10) for index in range(10)])
    validate_complete_split_manifests((test, validation), (0,))
    assert split_set_sha256((validation, test)) == split_set_sha256((test, validation))
    with pytest.raises(SplitError, match="exactly 10"):
        validate_complete_split_manifests((test,), (0,))


def test_simulator_state_digest_is_authoritative_collision_identity():
    common = {
        "task_id": 0,
        "reset_index": 1,
        "observation_sha256": "a" * 64,
        "simulator_state_sha256": "b" * 64,
        "environment_fingerprint": "env-1",
    }
    first = ResetIdentity(episode_id="test-renamed", seed=17, query_index=1, **common)
    renamed = ResetIdentity(episode_id="validation-renamed", seed=29, query_index=999, **common)
    assert first.key != renamed.key
    assert first.collision_key == renamed.collision_key
    test = build_split_manifest("test", [first])
    validation = build_split_manifest("validation", [renamed])
    with pytest.raises(SplitError, match="reset collision"):
        from arrow_policy_suite.splits import validate_split_manifests
        validate_split_manifests((test, validation))

    with pytest.raises(SplitError, match="duplicate reset identity"):
        build_split_manifest("test", [first, renamed])


def test_frozen_study_manifest_binds_counts_lineage_and_round_trips(tmp_path):
    config = _config()
    collection = build_split_manifest("collection", [_identity(0, f"collection-{index}", index) for index in range(50)] + [
        _identity(1, f"collection-1-{index}", 100 + index) for index in range(50)
    ])
    validation = build_split_manifest("validation", [_identity(0, f"validation-0-{index}", 200 + index) for index in range(10)] + [
        _identity(1, f"validation-1-{index}", 300 + index) for index in range(10)
    ])
    test = build_split_manifest("test", [_identity(0, f"test-0-{index}", 400 + index) for index in range(10)] + [
        _identity(1, f"test-1-{index}", 500 + index) for index in range(10)
    ])
    seal = FrozenStudyManifest(config, collection, validation, test, ("c" * 64, "d" * 64))
    FrozenStudyManifest.verify_manifest(seal.to_dict())
    restored = FrozenStudyManifest.from_dict(seal.to_dict())
    assert restored.composite_sha256 == seal.composite_sha256
    ref = seal.write_artifact(tmp_path / "study.json")
    assert ref.kind == "frozen-study-manifest"
    assert FrozenStudyManifest.read_artifact(tmp_path / "study.json").composite_sha256 == seal.composite_sha256
    with pytest.raises(ContractError, match="composite digest"):
        FrozenStudyManifest.from_dict({**seal.to_dict(), "composite_sha256": "e" * 64})


def test_frozen_study_manifest_reconciles_environment_and_reset_maps():
    config = _config()
    collection = build_split_manifest("collection", [_identity(0, f"collection-{index}", index) for index in range(50)] + [
        _identity(1, f"collection-1-{index}", 100 + index) for index in range(50)
    ])
    validation = build_split_manifest("validation", [_identity(0, f"validation-0-{index}", 200 + index) for index in range(10)] + [
        _identity(1, f"validation-1-{index}", 300 + index) for index in range(10)
    ])
    test = build_split_manifest("test", [_identity(0, f"test-0-{index}", 400 + index) for index in range(10)] + [
        _identity(1, f"test-1-{index}", 500 + index) for index in range(10)
    ])
    wrong_environment = build_split_manifest(
        "collection", [replace(collection.identities[0], environment_fingerprint="other-environment")] + list(collection.identities[1:])
    )
    with pytest.raises(ContractError, match="environment_revision"):
        FrozenStudyManifest(config, wrong_environment, validation, test, ("c" * 64,))
    wrong_reset = build_split_manifest(
        "validation", [replace(validation.identities[0], episode_id="not-in-study-config")] + list(validation.identities[1:])
    )
    with pytest.raises(ContractError, match="StudyConfig maps"):
        FrozenStudyManifest(config, collection, wrong_reset, test, ("c" * 64,))
