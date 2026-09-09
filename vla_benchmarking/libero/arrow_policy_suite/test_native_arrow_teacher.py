from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arrow_policy_suite.contracts import ObservationFrame
from arrow_policy_suite.native_arrow_teacher import PerFrameArrowTeacher


def _frame(step: int = 0, x: float = 0.0, rotvec=(0.0, 0.0, 0.0)) -> ObservationFrame:
    # Canonical LIBERO state convention: EEF xyz, axis-angle rotvec, gripper.
    return ObservationFrame({"state": [x, 0.0, 0.5, *rotvec, 0.0, 0.0]}, timestep=step)


def test_phase_teacher_proposes_one_action_and_advances_only_after_commit():
    calls = []

    def perceive(frame):
        calls.append(frame.timestep)
        waypoints = [[0.01, 0.0, 0.5], [0.02, 0.0, 0.5], [0.03, 0.0, 0.5],
                     [0.04, 0.0, 0.5], [0.05, 0.0, 0.5], [0.06, 0.0, 0.5]]
        return {"candidate_id": "c0", "waypoints": waypoints, "provenance": {"arrow": "synthetic"}}

    teacher = PerFrameArrowTeacher(perceive, gripper_dwell_steps=1)
    first = teacher.propose(_frame())
    assert first is not None
    assert first.metadata["phase"] == "pregrasp"
    assert calls == [0]
    # A second proposal in the same frame is idempotent and does not re-run
    # perception or advance the phase.
    assert teacher.propose(_frame()) == first
    assert first.metadata["phase"] == "pregrasp"
    teacher.commit(SimpleNamespace(base=first, teacher=first, next_frame=_frame(1, 0.01)))
    second = teacher.propose(_frame(1, 0.01))
    assert second is not None and second.metadata["phase"] == "descend"
    assert calls == [0]


def test_phase_teacher_snapshot_restores_plan_and_pending_action():
    teacher = PerFrameArrowTeacher(lambda _frame: {"waypoints": [[0.0, 0.0, 0.5]] * 6})
    frame = _frame()
    proposal = teacher.propose(frame)
    state = teacher.snapshot_state()
    teacher.reset()
    teacher.restore_state(state)
    assert teacher.propose(frame) == proposal


def test_canonical_rotvec_is_converted_to_quaternion_for_osc():
    teacher = PerFrameArrowTeacher(lambda _frame: {"waypoints": [[0.01, 0.0, 0.5]] * 6})
    proposal = teacher.propose(_frame(rotvec=(0.0, 0.0, 1.5707963267948966)))
    assert proposal is not None
    assert all(abs(float(value)) <= 1.0 for value in proposal.action)
