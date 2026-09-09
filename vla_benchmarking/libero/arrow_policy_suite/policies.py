"""The seven concrete policy families.

Each class is an adapter around a frozen VLA proposal and an optional Arrow
proposal.  Model training is intentionally injected as a callback in
``learning.py`` so importing this package never launches an expensive job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import inspect
from typing import Any, Callable, Mapping, Sequence

from .contracts import ActionProposal, ObservationFrame, PolicyDecision, ResidualFn, StepRecord, clip_action
from .runtime import ProgressTracker


class _BasePolicy:
    policy_id = ""

    def reset(self) -> None:
        pass

    def commit(self, record: StepRecord) -> None:
        pass


# A learned policy hook receives the authoritative frame and the frozen VLA
# action.  It returns the complete seven-dimensional action.  Keeping this
# hook at the policy boundary makes the teacher-free path usable with a native
# SmolVLA/PEFT adapter without importing that training stack here.
LearnedActionFn = Callable[[ObservationFrame, tuple[float, ...]], Sequence[float]]


def _policy_decision(
    frame: ObservationFrame,
    action: Sequence[float],
    policy_id: str,
    *,
    teacher_used: bool = False,
    teacher_groups: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PolicyDecision:
    """Build a decision across the lightweight and transactional contracts.

    During package assembly the coordinator contract may be supplied by a
    native adapter.  The historical suite contract stores action/digest
    directly, while the transactional contract stores an ActionProposal and a
    post-step snapshot.  Keeping this compatibility seam local avoids making
    policy semantics depend on either runtime implementation.
    """
    values = clip_action(action)
    payload = dict(metadata or {})
    if "proposal" not in inspect.signature(PolicyDecision).parameters:
        return PolicyDecision(values, policy_id, frame.digest, teacher_used, teacher_groups, payload)
    from .contracts import EpisodeSnapshot
    snapshot = EpisodeSnapshot(frame.observation, timestep=getattr(frame, "timestep", getattr(frame, "step", 0)))
    # Keep the same-frame identity on the decision.  This is what lets the
    # transactional coordinator reject a stale correction before stepping.
    proposal = ActionProposal(
        values,
        policy_id=policy_id,
        timestep=getattr(frame, "timestep", getattr(frame, "step", 0)),
        metadata=payload,
        observation_digest=getattr(frame, "digest", None),
    )
    return PolicyDecision(
        proposal,
        snapshot,
        metadata=payload,
        provenance={"policy_id": policy_id},
        teacher_used=teacher_used,
        teacher_groups=teacher_groups,
    )


def _proposal_metadata(proposal: Any) -> Mapping[str, Any] | None:
    """Return a small immutable log view instead of leaking proposal objects."""
    if proposal is None:
        return None
    return {
        "policy_id": str(getattr(proposal, "policy_id", "unknown")),
        "action": tuple(float(value) for value in getattr(proposal, "action", ())),
        "timestep": int(getattr(proposal, "timestep", 0)),
        "observation_digest": getattr(proposal, "observation_digest", None),
        "metadata": dict(getattr(proposal, "metadata", {}) or {}),
    }


def _teacher_logged_unavailable(teacher: Any) -> tuple[bool, str]:
    """Recognise an Arrow log miss without treating it as a live action.

    Native Arrow adapters use one of the explicit status spellings below.  The
    conservative default is false: an arbitrary metadata field must never
    silently discard a valid teacher proposal.
    """
    if teacher is None:
        return True, "teacher_missing"
    metadata = getattr(teacher, "metadata", {})
    if not isinstance(metadata, Mapping):
        return False, ""
    if bool(metadata.get("logged_unavailable", False)) or bool(metadata.get("teacher_logged_unavailable", False)):
        return True, "teacher_logged_unavailable"
    status = str(metadata.get("availability", metadata.get("status", ""))).lower().strip()
    if status in {"logged_unavailable", "unavailable", "missing", "not_logged"}:
        return True, f"teacher_{status}"
    return False, ""


class TogetherPolicy(_BasePolicy):
    policy_id = "arrow_together"

    def __init__(self, pose_weight: float = 0.5) -> None:
        self.pose_weight = float(pose_weight)

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        unavailable, reason = _teacher_logged_unavailable(teacher)
        if teacher is None:
            frame_reason = frame.metadata.get("teacher_unavailable_reason") if isinstance(frame.metadata, Mapping) else None
            if frame_reason:
                reason = f"teacher_{frame_reason}"
        if unavailable:
            # A missing Arrow log is a data-availability event, not permission
            # to invent a partial action.  Execute the complete frozen-VLA
            # action and preserve both same-frame candidates for audit.
            return _policy_decision(
                frame, base.action, self.policy_id,
                metadata={
                    "teacher_available": False,
                    "fallback": True,
                    "fallback_reason": reason,
                    "gripper_owner": "vla_fallback",
                    "base_proposal": _proposal_metadata(base),
                    "teacher_proposal": _proposal_metadata(teacher),
                },
            )
        w = max(0.0, min(1.0, self.pose_weight))
        action = tuple((1.0 - w) * base.action[i] + w * teacher.action[i] for i in range(6)) + (teacher.action[6],)
        return _policy_decision(frame, action, self.policy_id, teacher_used=True, teacher_groups=("gripper",),
                                metadata={"pose_weight": w, "gripper_owner": "arrow"})


class OnCallPolicy(_BasePolicy):
    policy_id = "arrow_on_call"

    def __init__(self, *, tracker: ProgressTracker | None = None) -> None:
        self.tracker = tracker or ProgressTracker(window=20)
        self._takeover = False
        self._teacher_steps = 0
        self._decision_index = 0
        self._handback_notice: dict[str, Any] | None = None
        self._handback_cooldown = False

    def reset(self) -> None:
        self.tracker.reset()
        self._takeover = False
        self._teacher_steps = 0
        self._decision_index = 0
        self._handback_notice = None
        self._handback_cooldown = False

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        self._decision_index += 1
        # ``commit`` runs after env.step.  A notice therefore belongs on the
        # next decision, making the handback ordering observable without
        # mutating an already-recorded PolicyDecision.
        notice = self._handback_notice
        self._handback_notice = None
        common = {
            "proposal_order": ("base", "teacher", "policy"),
            "decision_index": self._decision_index,
            "teacher_available": teacher is not None,
            "takeover_active": self._takeover,
            "milestone": False,
            "gripper_dwell": False,
            "gripper_conflict": False,
        }
        if notice:
            common.update(notice)
        if teacher is None:
            common.update({"phase": "base", "takeover": False, "teacher_used": False})
            return _policy_decision(frame, base.action, self.policy_id, metadata=common)
        teacher_metadata = teacher.metadata if isinstance(teacher.metadata, Mapping) else {}
        milestone = bool(
            teacher_metadata.get("milestone_complete", False)
            or teacher_metadata.get("milestone_reached", False)
            or teacher_metadata.get("milestone", False) is True
        )
        dwell = bool(teacher_metadata.get("gripper_dwell", False))
        conflict = bool(teacher_metadata.get("gripper_conflict", False))
        common.update({
            "milestone": milestone,
            "gripper_dwell": dwell,
            "gripper_conflict": conflict,
            "progress_window": self.tracker.window,
            "phase_error_improvement": self.tracker.progress_improvement(teacher),
            "handback_eligible": bool(
                self._takeover
                and self._teacher_steps + 1 >= self.tracker.window
                and milestone and not dwell and not conflict
            ),
        })
        takeover_started = False
        if not self._takeover and not self._handback_cooldown and self.tracker.should_trigger(teacher):
            self._takeover = True
            self._teacher_steps = 0
            takeover_started = True
            common["takeover_reason"] = "safety_or_gripper_conflict" if (
                conflict or bool(teacher_metadata.get("safety", False))
            ) else "progress_stalled"
        self._handback_cooldown = False
        if self._takeover:
            common.update({
                "phase": "takeover",
                "takeover": True,
                "takeover_started": takeover_started,
                "teacher_used": True,
                "teacher_step": self._teacher_steps,
                "teacher_steps_before_commit": self._teacher_steps,
                "handback": False,
            })
            return _policy_decision(frame, teacher.action, self.policy_id, teacher_used=True,
                                    teacher_groups=("translation", "rotation", "gripper"), metadata=common)
        common.update({"phase": "base", "takeover": False, "takeover_started": False,
                       "teacher_used": False, "handback": bool(notice)})
        return _policy_decision(frame, base.action, self.policy_id, metadata=common)

    def commit(self, record: StepRecord) -> None:
        # Progress is updated before evaluating handback so the current
        # teacher proposal is included in the transition that may release the
        # takeover.  No evaluator success or simulator state is consulted.
        self.tracker.update(record.frame, record.teacher)
        if self._takeover:
            self._teacher_steps += 1
            teacher_metadata = record.teacher.metadata if record.teacher is not None else {}
            milestone = bool(
                teacher_metadata.get("milestone_complete", False)
                or teacher_metadata.get("milestone_reached", False)
                or teacher_metadata.get("milestone", False) is True
            )
            safe = not bool(teacher_metadata.get("gripper_dwell", False))
            if "handback_safe" in teacher_metadata:
                safe = safe and bool(teacher_metadata["handback_safe"])
            if bool(teacher_metadata.get("gripper_conflict", False)):
                safe = False
            if self._teacher_steps >= self.tracker.window and milestone and safe:
                self._takeover = False
                completed_steps = self._teacher_steps
                self._teacher_steps = 0
                self._handback_notice = {
                    "handback": True,
                    "handback_reason": "milestone_complete",
                    "handback_after_teacher_steps": completed_steps,
                }
                self._handback_cooldown = True

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "tracker": self.tracker.snapshot_state(),
            "takeover": self._takeover,
            "teacher_steps": self._teacher_steps,
            "decision_index": self._decision_index,
            "handback_notice": dict(self._handback_notice or {}),
            "handback_cooldown": self._handback_cooldown,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.tracker.restore_state(state["tracker"])
        self._takeover = bool(state["takeover"])
        self._teacher_steps = int(state["teacher_steps"])
        self._decision_index = int(state["decision_index"])
        self._handback_notice = dict(state["handback_notice"]) or None
        self._handback_cooldown = bool(state["handback_cooldown"])


class ResidualPolicy(_BasePolicy):
    """Frozen-VLA wrapper used by Editor and Minimal-Learned."""

    def __init__(self, policy_id: str, residual_fn: ResidualFn | None = None, *, source: str = "external_residual") -> None:
        self.policy_id = policy_id
        self.residual_fn = residual_fn
        self.source = source

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        residual = self.residual_fn(frame, base.action) if self.residual_fn is not None else (0.0,) * 7
        residual = tuple(float(v) for v in residual)
        if len(residual) != 7:
            raise ValueError("residual function must return seven values")
        action = clip_action(tuple(base.action[i] + residual[i] for i in range(7)))
        return _policy_decision(frame, action, self.policy_id,
                                metadata={"teacher_free": True, "teacher_ignored": teacher is not None,
                                          "residual_source": self.source})


class EditorPolicy(ResidualPolicy):
    """Named Arrow Editor wrapper; the VLA remains frozen."""

    def __init__(self, residual_fn: ResidualFn | None = None) -> None:
        super().__init__("arrow_editor", residual_fn, source="teacher_minus_base_residual")


class ApprenticePolicy(_BasePolicy):
    policy_id = "arrow_apprentice"

    def __init__(self, action_fn: LearnedActionFn | None = None) -> None:
        self.action_fn = action_fn

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        action = base.action if self.action_fn is None else clip_action(self.action_fn(frame, base.action))
        return _policy_decision(frame, action, self.policy_id,
                                metadata={"teacher_free": True, "teacher_ignored": teacher is not None,
                                          "adaptation": "native_smolvla_lora",
                                          "action_source": "frozen_vla" if self.action_fn is None else "learned_hook"})


@dataclass(frozen=True)
class BranchOutcome:
    sufficient: bool
    progress: float
    phase_advanced: bool = False
    steps: int = 20
    metadata: Mapping[str, Any] = field(default_factory=dict)


BranchEvaluator = Callable[[ObservationFrame, ActionProposal, ActionProposal, tuple[str, ...], int], BranchOutcome]


class MinimalPolicy(_BasePolicy):
    policy_id = "arrow_minimal"
    GROUPS = ("translation", "rotation", "gripper")

    def __init__(self, *, variant: str = "runtime_oracle", branch_evaluator: BranchEvaluator | None = None,
                 residual_fn: ResidualFn | None = None, learned_fn: LearnedActionFn | None = None,
                 action_fn: LearnedActionFn | None = None, max_decisions: int = 60,
                 branch_runner: Any | None = None) -> None:
        if variant not in {"runtime_oracle", "learned"}:
            raise ValueError("Minimal variant must be runtime_oracle or learned")
        if max_decisions < 0:
            raise ValueError("Minimal max_decisions must be non-negative")
        self.variant = variant
        self.branch_evaluator = branch_evaluator
        self.residual_fn = residual_fn
        if learned_fn is not None and action_fn is not None:
            raise ValueError("provide only one of learned_fn or action_fn")
        self.learned_fn = learned_fn or action_fn
        if variant == "runtime_oracle" and (self.learned_fn is not None or residual_fn is not None):
            raise ValueError("runtime_oracle Minimal cannot use a teacher-free learned hook")
        self.max_decisions = max_decisions
        self.decisions = 0
        self.branch_runner = branch_runner
        self._active_mask: tuple[str, ...] | None = None
        self._burst_remaining = 0
        self._last_branch_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self.decisions = 0
        self._active_mask = None
        self._burst_remaining = 0
        self._last_branch_metadata = {}

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "decisions": self.decisions,
            "active_mask": self._active_mask,
            "burst_remaining": self._burst_remaining,
            "last_branch_metadata": dict(self._last_branch_metadata),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.decisions = int(state["decisions"])
        active = state.get("active_mask")
        self._active_mask = tuple(active) if active is not None else None
        self._burst_remaining = int(state.get("burst_remaining", 0))
        self._last_branch_metadata = dict(state.get("last_branch_metadata", {}))

    @staticmethod
    def masks() -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(group for group, active in zip(MinimalPolicy.GROUPS, bits) if active)
                     for bits in product((False, True), repeat=3))

    @classmethod
    def mask_from_bits(cls, mask: int) -> tuple[str, ...]:
        if mask < 0 or mask >= 8:
            raise ValueError("Minimal mask must be in [0, 7]")
        return tuple(group for index, group in enumerate(cls.GROUPS) if mask & (1 << index))

    def _apply_mask(self, base: Sequence[float], teacher: Sequence[float], mask: Sequence[str]) -> tuple[float, ...]:
        output = list(base)
        if "translation" in mask:
            output[0:3] = teacher[0:3]
        if "rotation" in mask:
            output[3:6] = teacher[3:6]
        if "gripper" in mask:
            output[6] = teacher[6]
        return clip_action(output)

    def decide(self, frame: ObservationFrame, base: ActionProposal, teacher: ActionProposal | None) -> PolicyDecision:
        if self.variant == "learned":
            if self.learned_fn is not None and self.residual_fn is not None:
                raise ValueError("learned Minimal cannot combine learned_fn and residual_fn")
            if self.learned_fn is not None:
                action = clip_action(self.learned_fn(frame, base.action))
                source = "learned_action_hook"
            elif self.residual_fn is None:
                action = base.action
                source = "frozen_vla_fallback"
            else:
                residual = tuple(float(v) for v in self.residual_fn(frame, base.action))
                if len(residual) != 7:
                    raise ValueError("residual function must return seven values")
                action = clip_action(tuple(base.action[i] + residual[i] for i in range(7)))
                source = "learned_residual_hook"
            return _policy_decision(
                frame, action, self.policy_id,
                metadata={"variant": "learned", "teacher_free": True,
                          "teacher_ignored": teacher is not None, "action_source": source},
            )
        if teacher is None:
            return _policy_decision(frame, base.action, self.policy_id,
                                    metadata={"variant": self.variant, "teacher_available": False})
        # A selected mask is a real 20-action burst.  The coordinator still
        # owns each step; this policy only reuses the selected ownership mask.
        if self._active_mask is not None and self._burst_remaining > 0:
            mask, outcome = self._active_mask, None
            branch_metadata = dict(self._last_branch_metadata)
            # The branch was cloned once at burst start.  Reusing the chosen
            # ownership mask must not charge the same 160 cloned transitions
            # to each of the twenty scored actions.
            branch_metadata.update({
                "branch_steps": 0,
                "branch_masks_evaluated": 0,
                "branch_reuse": True,
            })
        elif self.decisions >= self.max_decisions:
            mask = self.GROUPS
            outcome = None
            branch_metadata = {"branch_steps": 0, "branch_masks_evaluated": 0, "branch_fallback": True}
        elif self.branch_runner is not None:
            results = tuple(self.branch_runner.run_all())
            if len(results) != 8:
                raise ValueError("Minimal-Runtime must evaluate exactly eight masks")
            candidates = [result for result in results if bool(result.metadata.get("sufficient", False))]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda result: (
                        len(self.mask_from_bits(result.mask)),
                        -float(result.metadata.get("progress", result.score)),
                        result.mask,
                    ),
                )
                mask = self.mask_from_bits(chosen.mask)
                outcome = None
            else:
                mask, outcome = self.GROUPS, None
                chosen = None
            cloned_steps = sum(int(result.metadata.get("cloned_steps", len(result.actions))) for result in results)
            branch_metadata = {
                "branch_steps": cloned_steps,
                "branch_masks_evaluated": len(results),
                "branch_fallback": chosen is None,
                "branch_selected_mask": chosen.mask if chosen is not None else 7,
                "branch_latency_seconds": sum(float(result.metadata.get("branch_latency_seconds", 0.0)) for result in results),
            }
            self._active_mask = tuple(mask)
            self._burst_remaining = 20
            self._last_branch_metadata = dict(branch_metadata)
        elif self.decisions >= self.max_decisions or self.branch_evaluator is None:
            mask = self.GROUPS
            outcome = None
            branch_metadata = {"branch_steps": 0, "branch_masks_evaluated": 0, "branch_fallback": True}
        else:
            outcomes: list[tuple[tuple[str, ...], BranchOutcome]] = []
            for mask in self.masks():
                outcome = self.branch_evaluator(frame, base, teacher, mask, 20)
                if outcome.sufficient:
                    outcomes.append((mask, outcome))
            if outcomes:
                mask, outcome = min(outcomes, key=lambda pair: (len(pair[0]), -pair[1].progress, pair[0]))
            else:
                mask, outcome = self.GROUPS, None
            branch_metadata = {
                "branch_steps": 20,
                "branch_masks_evaluated": 8,
                "branch_fallback": not bool(outcomes),
                "branch_selected_mask": self.GROUPS,
            }
        self.decisions += 1
        action = self._apply_mask(base.action, teacher.action, mask)
        metadata = {"variant": "runtime_oracle", "branch_steps": 20 if self.branch_runner is None else int(branch_metadata.get("branch_steps", 0)),
                    "mask": mask, "branch_sufficient": outcome.sufficient if outcome else False,
                    **branch_metadata}
        return _policy_decision(frame, action, self.policy_id, teacher_used=True,
                                teacher_groups=mask, metadata=metadata)

    def commit(self, record: StepRecord) -> None:
        if self._active_mask is not None and self._burst_remaining > 0:
            self._burst_remaining -= 1
            if self._burst_remaining == 0:
                self._active_mask = None
                self._last_branch_metadata = {}


def make_policy(
    policy_id: str,
    *,
    hooks: Mapping[str, Callable[..., _BasePolicy]] | None = None,
    **kwargs: Any,
) -> _BasePolicy:
    """Construct one policy with explicit dependency-injection hooks.

    ``hooks`` is intentionally opt-in and keyed by the public policy id.  It
    is useful for native SmolVLA/Arrow adapters that need a richer constructor
    than this dependency-light package can provide.  A hook receives the same
    keyword arguments as the built-in constructor and must return a policy
    object; all built-in behavior remains unchanged when no hook is supplied.
    """
    if hooks is not None:
        if not isinstance(hooks, Mapping):
            raise ValueError("policy hooks must be a mapping")
        hook = hooks.get(policy_id)
        if hook is not None:
            if not callable(hook):
                raise ValueError(f"policy hook for {policy_id!r} is not callable")
            return hook(**kwargs)
    constructors = {
        "arrow_together": TogetherPolicy,
        "arrow_on_call": OnCallPolicy,
        "arrow_apprentice": ApprenticePolicy,
        "arrow_editor": EditorPolicy,
        "arrow_minimal": MinimalPolicy,
    }
    if policy_id == "arrow_fast":
        from .fast import FastPolicy
        try:
            return FastPolicy(kwargs["corrector"], graph_fn=kwargs.get("graph_fn"))
        except KeyError as exc:
            raise ValueError("arrow_fast requires a FastCorrector-compatible corrector") from exc
    if policy_id == "arrow_trace":
        from .trace import TracePolicy
        try:
            return TracePolicy(kwargs["routes"], kwargs["geometry_provider"], kwargs.get("waypoint_action"),
                               lookahead=kwargs.get("lookahead", 2))
        except KeyError as exc:
            raise ValueError("arrow_trace requires routes and geometry_provider") from exc
    try:
        return constructors[policy_id](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown policy {policy_id!r}") from exc
