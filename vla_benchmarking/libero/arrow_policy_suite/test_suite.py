from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrow_policy_suite import (  # noqa: E402
    ActionProposal,
    GeometryAnchors,
    GraphFastCorrector,
    MinimalPolicy,
    ObservationFrame,
    TogetherPolicy,
    TransactionalCoordinator,
    TracePolicy,
    extract_state_route,
    warp_route,
)
from arrow_policy_suite.runtime import make_proposal  # noqa: E402


class FakeEnvironment:
    def __init__(self) -> None:
        self.position = 0.0
        self.steps = 0

    def observe(self):
        return {"state": [self.position, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1]}

    def step(self, action):
        self.position += float(action[0])
        self.steps += 1
        return {"success": self.position >= 1.0, "terminal": self.position >= 1.0}

    def snapshot(self):
        return self.position, self.steps

    def restore(self, snapshot):
        self.position, self.steps = snapshot


class FakeVLA:
    def reset(self):
        return None

    def propose(self, frame):
        return make_proposal((1.0, 0, 0, 0, 0, 0, -1.0), "vla", frame)

    def commit(self, record):
        return None


class FakeTeacher:
    def reset(self):
        return None

    def propose(self, frame):
        return make_proposal((0.0, 0, 0, 0, 0, 0, 1.0), "arrow", frame, phase_error=1.0)

    def commit(self, record):
        return None


def test_transactional_coordinator_steps_once_and_preserves_gripper_owner():
    env = FakeEnvironment()
    result = TransactionalCoordinator(
        env, FakeVLA(), FakeTeacher(),
        success_fn=lambda value: bool(value["success"]),
        terminal_fn=lambda value: bool(value["terminal"]),
    ).run(TogetherPolicy(), max_steps=2)
    assert env.steps == 2
    assert result.stats.success is True
    assert result.records[0].decision.action[6] == pytest.approx(1.0)
    assert result.records[0].base.observation_digest == result.records[0].frame.digest


def test_minimal_has_all_eight_masks_and_learned_is_teacher_free():
    assert len(MinimalPolicy.masks()) == 8
    policy = MinimalPolicy(variant="learned", residual_fn=lambda _frame, _base: (0,) * 7)
    frame = ObservationFrame({"state": [0.0] * 8}, 0)
    base = make_proposal((0,) * 7, "vla", frame)
    teacher = make_proposal((1,) * 7, "arrow", frame)
    decision = policy.decide(frame, base, teacher)
    assert decision.teacher_used is False
    assert decision.metadata["teacher_free"] is True


def test_fast_updates_exactly_448_parameters():
    frame = ObservationFrame({"state": [0.0] * 8}, 0)
    corrector = GraphFastCorrector(lambda _frame, _role: [1.0] + [0.0] * 31,
                                   lambda _frame, role: 1.0 if role == "hand_to_source" else 0.0)
    corrector.fit([(frame, (0,) * 7, (0.5,) + (0,) * 6)], success=True)
    assert corrector.metadata.parameter_count == 448
    assert any(value != 0.0 for row in corrector.weights["hand_to_source"] for value in row)


def test_trace_is_state_only_and_warp_uses_rgbd_provenance():
    states = [
        [0.0, 0.0, 0.2, 0, 0, 0, 0.1, 0.1],
        [0.2, 0.0, 0.3, 0, 0, 0, 0.0, 0.0],
        [0.5, 0.0, 0.3, 0, 0, 0, 0.0, 0.0],
        [0.8, 0.0, 0.2, 0, 0, 0, 0.1, 0.1],
    ]
    route = extract_state_route(states, source_anchor=(0, 0, 0), destination_anchor=(1, 0, 0), samples=8)
    anchors = GeometryAnchors((2, 0, 0), (2, 1, 0), "world", "calib-1", "arrow_rgbd",
                              {"arrow_origin": "sim_bbox_then_rgbd_deprojection"})
    warped = warp_route(route, anchors)
    assert warped.provenance["warp_provider"] == "arrow_rgbd"
    assert {point.event for point in route.points} & {"close", "reopen"}
    with pytest.raises(Exception):
        GeometryAnchors((0, 0, 0), (1, 0, 0), "world", "calib-1", "sim_ground_truth")
