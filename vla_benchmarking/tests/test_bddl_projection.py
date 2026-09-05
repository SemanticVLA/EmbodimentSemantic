import numpy as np
import pytest

from vla_benchmarking.evaluation.bddl_utils import (
    JointSchema,
    JointSlice,
    extract_joint_schema,
    project_flattened_state,
)


class _JointIdNameModel:
    njnt = 1
    nq = 1
    nv = 1
    jnt_type = [3]
    jnt_qposadr = [0]
    jnt_dofadr = [0]

    def joint_id2name(self, joint_id):
        return "fallback_joint"


def test_extract_schema_supports_joint_id2name_model_api():
    schema = extract_joint_schema(_JointIdNameModel())
    assert schema.names == ("fallback_joint",)


def _source_schema():
    return JointSchema(
        joints=(
            JointSlice("root", 3, 0, 1, 0, 1),
            JointSlice("removed", 0, 1, 7, 1, 6),
            JointSlice("downstream", 3, 8, 1, 7, 1),
        ),
        nq=9,
        nv=8,
    )


def _target_schema():
    return JointSchema(
        joints=(
            JointSlice("root", 3, 0, 1, 0, 1),
            JointSlice("downstream", 3, 1, 1, 1, 1),
        ),
        nq=2,
        nv=2,
    )


def test_projection_preserves_downstream_named_joint_after_middle_free_joint_removal():
    source = np.arange(2 * 18, dtype=np.float32).reshape(2, 18)
    projected = project_flattened_state(source, _source_schema(), _target_schema())
    assert projected.shape == (2, 5)
    np.testing.assert_array_equal(projected[:, 0], source[:, 0])  # time
    np.testing.assert_array_equal(projected[:, 1], source[:, 1])  # root qpos
    np.testing.assert_array_equal(projected[:, 2], source[:, 9])  # downstream qpos
    np.testing.assert_array_equal(projected[:, 3], source[:, 10])  # root qvel
    np.testing.assert_array_equal(projected[:, 4], source[:, 17])  # downstream qvel


def test_projection_rejects_state_width_mismatch():
    with pytest.raises(ValueError, match="state width"):
        project_flattened_state(np.zeros((1, 17)), _source_schema(), _target_schema())


def test_projection_rejects_missing_retained_joint():
    target = JointSchema(
        joints=(JointSlice("missing", 3, 0, 1, 0, 1),),
        nq=1,
        nv=1,
    )
    with pytest.raises(ValueError, match="missing"):
        project_flattened_state(np.zeros((1, 18)), _source_schema(), target)


def test_projection_rejects_joint_type_width_mismatch():
    target = JointSchema(
        joints=(JointSlice("root", 2, 0, 1, 0, 1),),
        nq=1,
        nv=1,
    )
    with pytest.raises(ValueError, match="type/width mismatch"):
        project_flattened_state(np.zeros((1, 18)), _source_schema(), target)
