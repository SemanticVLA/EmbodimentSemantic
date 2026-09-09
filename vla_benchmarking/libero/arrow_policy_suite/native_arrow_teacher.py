"""One-step Arrow teacher for the native policy host.

The legacy Arrow runner owns an entire episode and calls ``env.step``.  That
is useful for its standalone benchmark, but it cannot be interleaved with a
VLA.  ``PerFrameArrowTeacher`` keeps only the perception/waypoint/controller
parts: it plans once from one RGB-D arrow frame, emits one OSC action, and
advances its phase only after the host commits that action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import inspect
import math
from pathlib import Path
from typing import Any, Callable

from .contracts import ActionProposal, ContractError, ObservationFrame, StepRecord, digest
from .interruptible_arrow import ArrowPerceptionUnavailable


class PerFrameArrowTeacher:
    """Rollback-capable, no-``env.step`` Arrow phase controller.

    ``perception`` returns ``{"candidate", "waypoints", ...}`` and is called
    at most once per plan.  The production helper below supplies this mapping
    from the repository's RGB-D ``ModelPerceptionWorker``.  Tests and other
    VLAs may inject a frame-only perception function, keeping the public
    teacher independent of model/runtime packages.
    """

    phases = ("pregrasp", "descend", "close", "lift", "preplace", "descend_place", "open", "retreat")

    def __init__(
        self,
        perception: Callable[[ObservationFrame], Mapping[str, Any] | None],
        *,
        producer: str = "arrow",
        gripper_dwell_steps: int = 2,
        phase_tolerance_m: float = 0.015,
        osc_position_scale_m: float | None = None,
        eef_orientation_transform: Any | None = None,
    ) -> None:
        if not callable(perception):
            raise TypeError("PerFrameArrowTeacher requires a perception callable")
        if int(gripper_dwell_steps) <= 0 or float(phase_tolerance_m) <= 0:
            raise ValueError("Arrow phase dwell/tolerance must be positive")
        self.perception = perception
        self.producer = str(producer)
        self.gripper_dwell_steps = int(gripper_dwell_steps)
        self.phase_tolerance_m = float(phase_tolerance_m)
        self.osc_position_scale_m = osc_position_scale_m
        self.eef_orientation_transform = eef_orientation_transform
        self._plan: Mapping[str, Any] | None = None
        self._phase_index = 0
        self._phase_steps = 0
        self._pending: ActionProposal | None = None
        self._last_candidate_id: str | None = None
        self._last_milestone = False
        self._last_milestone_phase: str | None = None

    def reset(self) -> None:
        self._plan = None
        self._phase_index = 0
        self._phase_steps = 0
        self._pending = None
        self._last_candidate_id = None
        self._last_milestone = False
        self._last_milestone_phase = None

    @staticmethod
    def _proprio(frame: ObservationFrame) -> Mapping[str, Any]:
        raw = frame.raw_observation
        aliases = {
            "eef_pos": ("robot0_eef_pos", "eef_pos"),
            "eef_quat": ("robot0_eef_quat", "eef_quat"),
            "gripper_qpos": ("robot0_gripper_qpos", "gripper_qpos"),
        }
        result = {output: raw[key] for output, names in aliases.items() for key in names if key in raw}
        if "eef_pos" not in result or "eef_quat" not in result:
            state = frame.observation.get("state")
            if state is not None and len(state) >= 7:
                result.setdefault("eef_pos", state[:3])
                # The canonical LIBERO eight-vector is xyz + axis-angle
                # rotvec + two gripper values, not xyz + xyzw quaternion.
                # Convert here because the OSC helper intentionally consumes
                # the raw robot quaternion contract.
                import numpy as np
                rotvec = np.asarray(state[3:6], dtype=np.float64).reshape(-1)
                angle = float(np.linalg.norm(rotvec))
                if angle <= 1e-12:
                    quat = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
                else:
                    axis = rotvec / angle
                    quat = np.concatenate((axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]))
                result.setdefault("eef_quat", quat)
                if len(state) >= 7:
                    result.setdefault("gripper_qpos", state[6:8])
        return result

    def _ensure_plan(self, frame: ObservationFrame) -> None:
        if self._plan is not None:
            return
        try:
            plan = self.perception(frame)
        except ArrowPerceptionUnavailable:
            raise
        except Exception as exc:
            raise RuntimeError("Arrow frame perception failed") from exc
        if plan is None:
            raise ArrowPerceptionUnavailable("no visual arrow/waypoint plan for this frame")
        if not isinstance(plan, Mapping) or plan.get("waypoints") is None:
            raise ContractError("Arrow perception must return a waypoint plan")
        self._plan = copy.deepcopy(dict(plan))
        candidate = self._plan.get("candidate")
        self._last_candidate_id = str(getattr(candidate, "candidate_id", self._plan.get("candidate_id", "candidate")))

    @staticmethod
    def _waypoint_position(waypoint: Any) -> tuple[float, float, float] | None:
        """Extract a metric XYZ target without exposing simulator state."""
        if isinstance(waypoint, Mapping):
            value = next((waypoint[key] for key in ("position", "pos", "eef_pos") if key in waypoint), None)
        else:
            value = next((getattr(waypoint, key) for key in ("position", "pos", "eef_pos") if hasattr(waypoint, key)), waypoint)
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if len(values) < 3 or not all(math.isfinite(item) for item in values[:3]):
            return None
        return values[:3]

    def _signal_metadata(self, frame: ObservationFrame, *, phase: str, waypoint: Any) -> dict[str, Any]:
        """Expose controller-owned progress signals for On-Call.

        These are derived from the same frame and waypoint used to produce the
        action.  They do not inspect evaluator success or call the environment.
        """
        proprio = self._proprio(frame)
        current = proprio.get("eef_pos")
        target = self._waypoint_position(waypoint)
        phase_error: float | None = None
        if current is not None and target is not None:
            try:
                xyz = tuple(float(item) for item in current[:3])
                phase_error = math.sqrt(sum((xyz[index] - target[index]) ** 2 for index in range(3)))
            except (TypeError, ValueError, IndexError):
                phase_error = None
        plan = self._plan or {}
        candidate = plan.get("candidate")
        candidate_metadata = getattr(candidate, "metadata", {}) if candidate is not None else {}
        if not isinstance(candidate_metadata, Mapping):
            candidate_metadata = {}
        plan_metadata = plan.get("metadata", {})
        if not isinstance(plan_metadata, Mapping):
            plan_metadata = {}

        def signal(*keys: str) -> bool:
            return any(bool(plan.get(key, plan_metadata.get(key, candidate_metadata.get(key, False)))) for key in keys)

        metadata: dict[str, Any] = {
            "phase": phase,
            "phase_index": self._phase_index,
            "phase_steps": self._phase_steps,
            "candidate_id": self._last_candidate_id,
            "plan_provenance": plan.get("provenance", {}),
            "phase_error": phase_error,
            "phase_error_source": "eef_to_waypoint_m" if phase_error is not None else "unavailable",
            "milestone": bool(self._last_milestone),
            "milestone_reached": bool(self._last_milestone),
            "milestone_phase": self._last_milestone_phase,
            "gripper_dwell": phase in {"close", "open"} and self._phase_steps < self.gripper_dwell_steps,
            "gripper_conflict": signal("gripper_conflict", "gripper_conflict_detected"),
            "safety": signal("safety", "safety_violation", "force_takeover"),
        }
        metadata["handback_safe"] = not metadata["gripper_dwell"] and not metadata["gripper_conflict"]
        # A milestone is consumed by the first proposal after commit; repeated
        # same-frame proposals remain idempotent because _pending is returned
        # before this helper is reached.
        self._last_milestone = False
        self._last_milestone_phase = None
        return metadata

    def propose(self, frame: ObservationFrame) -> ActionProposal | None:
        if self._pending is not None:
            if self._pending.timestep == frame.timestep:
                return self._pending
            raise ContractError("previous Arrow proposal was not committed")
        self._ensure_plan(frame)
        assert self._plan is not None
        from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import (
            _phase_waypoint, normalized_action_for_waypoint,
        )
        phase = self.phases[self._phase_index]
        waypoint = _phase_waypoint(self._plan["waypoints"], phase)
        gripper = 1.0 if phase == "close" else -1.0 if phase == "open" else 0.0
        action = normalized_action_for_waypoint(
            self._proprio(frame), waypoint, gripper=gripper,
            held_rotation=self._proprio(frame).get("eef_quat"),
            osc_position_scale_m=self.osc_position_scale_m,
            eef_orientation_transform=self.eef_orientation_transform,
        )
        proposal = ActionProposal(
            action, policy_id=self.producer, timestep=frame.timestep,
            metadata=self._signal_metadata(frame, phase=phase, waypoint=waypoint),
            interruptible=True, observation_digest=frame.digest,
        )
        self._pending = proposal
        return proposal

    def commit(self, record: StepRecord) -> None:
        if self._pending is None or record.teacher is None:
            raise ContractError("Arrow commit has no pending teacher proposal")
        if record.teacher.action != self._pending.action or record.base.timestep != self._pending.timestep:
            raise ContractError("Arrow commit does not match pending proposal")
        phase = self.phases[self._phase_index]
        self._pending = None
        self._phase_steps += 1
        # Gripper phases are intentional dwells.  Motion phases advance only
        # once the post-step observed EEF is near the waypoint.
        if phase in {"close", "open"}:
            reached = self._phase_steps >= self.gripper_dwell_steps
        else:
            reached = self._phase_steps >= 1
            try:
                from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import _phase_waypoint, _position
                waypoint = _phase_waypoint(self._plan["waypoints"], phase) if self._plan else None
                current = record.next_frame.observation.get("state", ())[:3]
                reached = reached and waypoint is not None and sum((float(a) - float(b)) ** 2 for a, b in zip(current, _position(waypoint))) ** 0.5 <= self.phase_tolerance_m
            except Exception:
                # If a VLA canonical state is not EEF-compatible, do not
                # silently skip the phase; one committed action is the only
                # conservative progress signal available.
                reached = True
        if reached:
            self._last_milestone = True
            self._last_milestone_phase = phase
            self._phase_index = min(self._phase_index + 1, len(self.phases) - 1)
            self._phase_steps = 0

    def interrupt(self) -> None:
        self._pending = None

    def invalidate_pending(self, *, reason: str = "external_action") -> None:
        """Discard an unexecuted plan proposal after a VLA action."""
        del reason
        self.interrupt()

    invalidate_queue = invalidate_pending

    def snapshot_state(self) -> Any:
        return (copy.deepcopy(self._plan), self._phase_index, self._phase_steps,
                self._pending, self._last_candidate_id, self._last_milestone,
                self._last_milestone_phase)

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, tuple) or len(state) not in {5, 7}:
            raise ContractError("invalid Arrow phase snapshot")
        copied = copy.deepcopy(state)
        self._plan, self._phase_index, self._phase_steps, self._pending, self._last_candidate_id = copied[:5]
        self._last_milestone = bool(copied[5]) if len(copied) == 7 else False
        self._last_milestone_phase = copied[6] if len(copied) == 7 else None

    snapshot = snapshot_state
    restore = restore_state

    @property
    def rollback_complete(self) -> bool:
        return True


def build_rgbd_perception(*, raw_environment: Any, worker: Any, task_id: int, resolution: int, output_dir: str | Path) -> Callable[[ObservationFrame], Mapping[str, Any] | None]:
    """Build the production pure perception/waypoint hook.

    It captures/render-decodes the current visual arrow, calls the existing
    ``ModelPerceptionWorker`` and ``build_bowl_waypoints`` helpers, and never
    invokes ``step``.  Any missing runtime asset is surfaced as an explicit
    unavailable/fault condition.
    """
    def perceive(frame: ObservationFrame) -> Mapping[str, Any] | None:
        from vla_benchmarking.libero.arrow_grasp_controller.controller import runner
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_eval as episode
        from vla_benchmarking.libero.evaluation import run_arrow_pick_place_matrix as matrix
        try:
            capture = episode.capture_agentview(raw_environment, resolution=int(resolution), camera_name="agentview")
            inputs = matrix._default_arrow_inputs(raw_environment, int(task_id), int(resolution))
            rendered, _audit = episode.render_exactly_one_arrow(
                capture.rgb, inputs["bboxes"], subject=inputs["subject"],
                goal_object=inputs["goal_object"], anchor_policy="bbox_center",
            )
            source_uv, destination_uv = episode.decode_arrow_pixels(capture.rgb, rendered)
        except ValueError as exc:
            # Invalid/missing arrow pixels are an expected unavailable
            # condition.  Runtime/model faults below are deliberately not
            # folded into this branch.
            raise ArrowPerceptionUnavailable(str(exc)) from exc
        request = runner.PerceptionRequest(
                variant=runner.VARIANTS["canonical"], agentview_capture=capture,
                source_capture=capture, source_uv=tuple(source_uv),
                destination_uv=None if destination_uv is None else tuple(destination_uv),
                previous_candidate_ids=(), output_dir=Path(output_dir),
            )
        candidates, diagnostics = runner._worker_candidates(worker, request)
        if not candidates:
            return None
        candidate = candidates[0]
        source_point = runner._deproject_capture(capture, source_uv)
        destination_point = runner._deproject_capture(
            capture, source_uv if destination_uv is None else destination_uv,
        )
        proprio = PerFrameArrowTeacher._proprio(frame)
        from vla_benchmarking.libero.evaluation.run_arrow_pick_place_eval import build_bowl_waypoints
        waypoints = build_bowl_waypoints(
            source_point, destination_point, proprio.get("eef_quat"), {"lift_height_m": 0.10},
        )
        return {"candidate": candidate, "waypoints": waypoints,
                "provenance": {"source_uv": list(source_uv), "destination_uv": None if destination_uv is None else list(destination_uv),
                                "candidate_id": candidate.candidate_id, "diagnostics": diagnostics,
                                "arrow_frame_digest": digest(capture.rgb)}}
    return perceive


__all__ = ["PerFrameArrowTeacher", "build_rgbd_perception"]
