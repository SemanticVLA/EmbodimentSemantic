from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from zerograsp_contracts import (  # noqa: E402
    GRASPNET_CAMERA_FRAME,
    GRASPNET_TRANSLATION_REFERENCE,
    GraspCandidate,
    ZeroGraspConfig,
    ZeroGraspObservation,
    hash_array,
    serialize_audit,
)


def _observation(fy: float = 100.0) -> ZeroGraspObservation:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    K = np.array([[100.0, 0, 16.0], [0, fy, 16.0], [0, 0, 1.0]])
    return ZeroGraspObservation(rgb, rgb.copy(), depth, K, np.eye(4), (8.0, 8.0), (24.0, 24.0))


def test_observation_contract_rejects_simulator_fields_and_bad_units():
    with pytest.raises(TypeError):
        ZeroGraspObservation(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3), np.eye(4), (1, 1), (2, 2), task_id=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ZeroGraspObservation(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.uint16), np.eye(3), np.eye(4), (1, 1), (2, 2))
    with pytest.raises(ValueError):
        ZeroGraspObservation(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3), np.eye(4), (1, 1), (2, 2), eef_quaternion_right_hand_xyzw=[0, 0, 0, 2])


def test_config_range_and_frame_contract_are_explicit():
    config = ZeroGraspConfig.from_mapping({"grasp_width_range_m": [0.01, 0.09], "approach_axis_local": [1, 0, 0]}).validate()
    assert config.grasp_width_range_m == (0.01, 0.09)
    with pytest.raises(ValueError):
        ZeroGraspConfig.from_mapping({"unknown": 1})


def test_observation_exposes_only_separate_proprioception_fields():
    observation = _observation()
    assert not hasattr(observation, "eef_pose")
    assert hasattr(observation, "eef_position_world_m")
    assert hasattr(observation, "eef_quaternion_right_hand_xyzw")


def test_candidate_rejects_non_source_and_bad_transform():
    candidate = GraspCandidate(np.eye(4), score=0.5, width_m=0.04, translation_reference=GRASPNET_TRANSLATION_REFERENCE, translation_frame=GRASPNET_CAMERA_FRAME, rotation_frame=GRASPNET_CAMERA_FRAME)
    assert candidate.source_role == "source"
    with pytest.raises(ValueError):
        GraspCandidate(np.eye(4), source_role="destination", translation_reference=GRASPNET_TRANSLATION_REFERENCE, translation_frame=GRASPNET_CAMERA_FRAME, rotation_frame=GRASPNET_CAMERA_FRAME)


def test_candidate_requires_explicit_camera_graspnet_frame_tags():
    kwargs = {"translation_reference": GRASPNET_TRANSLATION_REFERENCE, "translation_frame": GRASPNET_CAMERA_FRAME, "rotation_frame": GRASPNET_CAMERA_FRAME}
    for field, value in (("translation_reference", None), ("translation_frame", None), ("rotation_frame", "world")):
        invalid = dict(kwargs)
        invalid[field] = value
        with pytest.raises(ValueError, match=field):
            GraspCandidate(np.eye(4), **invalid)


def test_array_hash_and_audit_are_stable_and_json_safe():
    value = np.arange(4, dtype=np.float32)
    assert hash_array(value) == hash_array(value.copy())
    audit = serialize_audit({"array": value, "n": np.int64(2)})
    assert audit["array"]["shape"] == [4]
    assert audit["n"] == 2
