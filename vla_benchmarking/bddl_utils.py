"""BDDL file rewriting utilities for object-removal randomization."""

from __future__ import annotations

import atexit
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

_tmp_files: list[str] = []


_JOINT_WIDTHS = {
    0: (7, 6),  # free
    1: (4, 3),  # ball
    2: (1, 1),  # slide
    3: (1, 1),  # hinge
}


@dataclass(frozen=True)
class JointSlice:
    """Named MuJoCo joint slices in flattened qpos/qvel state space."""

    name: str
    joint_type: int
    qpos_start: int
    qpos_width: int
    qvel_start: int
    qvel_width: int


@dataclass(frozen=True)
class JointSchema:
    """Pure, serializable joint schema used for canonical-state projection."""

    joints: tuple[JointSlice, ...]
    nq: int
    nv: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)


def _model_joint_name(model: Any, joint_id: int) -> str:
    if hasattr(model, "joint_id2name"):
        name = model.joint_id2name(joint_id)
        if isinstance(name, bytes):
            name = name.decode()
        if name:
            return str(name)
    joint = model.joint(joint_id) if hasattr(model, "joint") else None
    name = getattr(joint, "name", None)
    if name is None and hasattr(model, "id2name"):
        name = model.id2name(joint_id, "joint")
    if isinstance(name, bytes):
        name = name.decode()
    if not name:
        raise ValueError(f"MuJoCo joint {joint_id} has no name")
    return str(name)


def extract_joint_schema(model: Any) -> JointSchema:
    """Extract a named joint schema from a MuJoCo model, failing closed."""
    joints: list[JointSlice] = []
    seen: set[str] = set()
    for joint_id in range(int(model.njnt)):
        name = _model_joint_name(model, joint_id)
        if name in seen:
            raise ValueError(f"duplicate MuJoCo joint name: {name}")
        seen.add(name)
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in _JOINT_WIDTHS:
            raise ValueError(f"unsupported MuJoCo joint type {joint_type} for {name}")
        qpos_width, qvel_width = _JOINT_WIDTHS[joint_type]
        joints.append(JointSlice(
            name=name,
            joint_type=joint_type,
            qpos_start=int(model.jnt_qposadr[joint_id]),
            qpos_width=qpos_width,
            qvel_start=int(model.jnt_dofadr[joint_id]),
            qvel_width=qvel_width,
        ))
    schema = JointSchema(joints=tuple(joints), nq=int(model.nq), nv=int(model.nv))
    _validate_joint_schema(schema)
    return schema


def _validate_joint_schema(schema: JointSchema) -> None:
    seen: set[str] = set()
    qpos_ranges: set[int] = set()
    qvel_ranges: set[int] = set()
    for joint in schema.joints:
        if joint.name in seen:
            raise ValueError(f"duplicate joint in schema: {joint.name}")
        seen.add(joint.name)
        expected = _JOINT_WIDTHS.get(joint.joint_type)
        if expected != (joint.qpos_width, joint.qvel_width):
            raise ValueError(f"joint type/width mismatch for {joint.name}")
        qpos = set(range(joint.qpos_start, joint.qpos_start + joint.qpos_width))
        qvel = set(range(joint.qvel_start, joint.qvel_start + joint.qvel_width))
        if min(qpos, default=0) < 0 or min(qvel, default=0) < 0:
            raise ValueError(f"negative joint slice for {joint.name}")
        if qpos & qpos_ranges or qvel & qvel_ranges:
            raise ValueError(f"overlapping joint slices for {joint.name}")
        qpos_ranges |= qpos
        qvel_ranges |= qvel
    if qpos_ranges and max(qpos_ranges) >= schema.nq:
        raise ValueError("joint qpos slice exceeds model nq")
    if qvel_ranges and max(qvel_ranges) >= schema.nv:
        raise ValueError("joint qvel slice exceeds model nv")


def project_flattened_state(
    states: np.ndarray,
    source_schema: JointSchema,
    target_schema: JointSchema,
) -> np.ndarray:
    """Project ``[..., time, qpos, qvel]`` states by retained joint name.

    The source may contain removed joints, but every target joint must occur once
    in the source with identical type and qpos/qvel widths. State length and all
    schema invariants are checked before any projection is performed.
    """
    _validate_joint_schema(source_schema)
    _validate_joint_schema(target_schema)
    array = np.asarray(states)
    expected_source_width = 1 + source_schema.nq + source_schema.nv
    if array.ndim < 1 or array.shape[-1] != expected_source_width:
        raise ValueError(
            f"canonical state width {array.shape[-1] if array.ndim else None} "
            f"does not match schema width {expected_source_width}"
        )
    source_by_name = {joint.name: joint for joint in source_schema.joints}
    target_by_name = {joint.name: joint for joint in target_schema.joints}
    if len(target_by_name) != len(target_schema.joints):
        raise ValueError("duplicate target joint names")
    output = np.zeros(array.shape[:-1] + (1 + target_schema.nq + target_schema.nv,), dtype=array.dtype)
    output[..., 0] = array[..., 0]
    for target in target_schema.joints:
        source = source_by_name.get(target.name)
        if source is None:
            raise ValueError(f"target retained joint missing from canonical schema: {target.name}")
        if (source.joint_type, source.qpos_width, source.qvel_width) != (
            target.joint_type, target.qpos_width, target.qvel_width
        ):
            raise ValueError(f"joint type/width mismatch for retained joint: {target.name}")
        output[..., 1 + target.qpos_start:1 + target.qpos_start + target.qpos_width] = array[
            ..., 1 + source.qpos_start:1 + source.qpos_start + source.qpos_width
        ]
        output[..., 1 + target_schema.nq + target.qvel_start:1 + target_schema.nq + target.qvel_start + target.qvel_width] = array[
            ..., 1 + source_schema.nq + source.qvel_start:1 + source_schema.nq + source.qvel_start + source.qvel_width
        ]
    return output


# Descriptive alias for callers that work specifically with LeRobot init states.
project_init_states_by_joint_name = project_flattened_state


@atexit.register
def _cleanup_tmp_files() -> None:
    for path in _tmp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def check_removal_safety(bddl_text: str, obj_name: str) -> None:
    """Raise ValueError if removing obj_name would orphan akita_black_bowl_1's placement.

    This catches the task 3 / task 5 trap: bowl_1 starts On a distractor object,
    so removing that object leaves bowl_1 with no placement sampler and it spawns
    at the origin.
    """
    pattern = re.compile(
        r"\(\s*On\s+akita_black_bowl_1\s+" + re.escape(obj_name) + r"\s*\)",
        re.IGNORECASE,
    )
    if pattern.search(bddl_text):
        raise ValueError(
            f"Cannot remove '{obj_name}': akita_black_bowl_1 starts On {obj_name} "
            f"in (:init). Removing it would leave bowl_1 with no placement sampler "
            f"and it would spawn at the origin. "
            f"Known unsafe combinations: task 3 + 'cookies_1', task 5 + 'glazed_rim_porcelain_ramekin_1'."
        )


def remove_object_from_bddl(bddl_text: str, obj_name: str) -> str:
    """Return bddl_text with all references to obj_name stripped.

    Edits only (:objects ...) and (:init ...). Does not touch (:regions),
    (:obj_of_interest), (:goal), or (:fixtures) — distractors never appear there
    in libero_spatial tasks.

    Raises ValueError if obj_name appears in (:goal ...) as a safety guard.
    """
    # Guard: obj_name must not be a goal object
    goal_match = re.search(r"\(:goal(.*?)\)\s*\)", bddl_text, re.DOTALL)
    if goal_match and re.search(r"\b" + re.escape(obj_name) + r"\b", goal_match.group(1)):
        raise ValueError(
            f"'{obj_name}' appears in (:goal ...) — removing it would make the task unreachable."
        )

    lines = bddl_text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()

        # (:objects ...) — remove the object token; drop line if it becomes empty or type-only
        # Format is: "  name1 name2 - type" — obj_name appears as a space-separated token
        # before the " - type" part.
        if re.search(r"\b" + re.escape(obj_name) + r"\b", stripped) and " - " in stripped and not stripped.startswith("("):
            new_line = re.sub(r"\b" + re.escape(obj_name) + r"\b\s*", "", line)
            # If only whitespace and "- type" remain (no object names left), drop the line
            remainder = new_line.strip()
            if re.fullmatch(r"-\s*\w+", remainder):
                continue
            out.append(new_line.rstrip())
            continue

        # (:init ...) — drop any predicate line that mentions obj_name
        # Matches lines like: (On obj_name ...) or (On something obj_name)
        if re.search(r"\b" + re.escape(obj_name) + r"\b", stripped) and stripped.startswith("("):
            continue

        out.append(line)

    return "\n".join(out)


def make_filtered_bddl(source_path: str, remove_objects: list[str]) -> str:
    """Write a filtered copy of source_path with remove_objects stripped out.

    Performs safety checks before rewriting. Returns the path to the temp file.
    The temp file is registered for deletion on process exit.
    """
    with open(source_path) as f:
        bddl = f.read()

    for obj in remove_objects:
        check_removal_safety(bddl, obj)
        bddl = remove_object_from_bddl(bddl, obj)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False)
    tmp.write(bddl)
    tmp.close()
    _tmp_files.append(tmp.name)
    return tmp.name
