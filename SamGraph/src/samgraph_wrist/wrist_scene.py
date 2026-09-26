"""Wrist-camera acquisition on top of the native causal SAM3 scene tracker.

The fixed-camera controller retains past masks as an explicitly labelled
estimate.  A moving wrist camera cannot use those pixels as current membership,
so this adapter emits current masks only and classifies lost tracks separately.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from samgraph_core.automatic_scene import (
    AUTOMATIC_CATALOG,
    AutomaticSceneController,
    MaskProposal,
    ObjectMemory,
    _appearance,
    _center,
    _overlap_fraction,
)


class WristSceneController(AutomaticSceneController):
    """Native SAM tracking with moving-camera visibility semantics."""

    def __init__(self, segmenter: Any, tracker: Any, *, geometry_rules: Any,
                 task_prompts: Mapping[str, Mapping[str, list[str]]] | None = None):
        super().__init__(
            segmenter,
            tracker,
            geometry_rules=geometry_rules,
            task_prompts=task_prompts,
            camera_id="wrist",
        )
        self._centers: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def close_episode(self) -> None:
        super().close_episode()
        self._centers.clear()

    def _remember(self, item: ObjectMemory, proposal: MaskProposal, rgb: np.ndarray,
                  frame: int, method: str) -> None:
        AutomaticSceneController._remember(item, proposal, rgb, frame, method)
        history = self._centers[item.track_id]
        center = _center(proposal.mask).astype(np.float64)
        if not history or history[-1][0] != int(frame):
            history.append((int(frame), center))
        else:
            history[-1] = (int(frame), center)
        if len(history) > 32:
            del history[:-32]

    def _reacquisition_cost(self, item: ObjectMemory, proposal: MaskProposal,
                            rgb: np.ndarray) -> float | None:
        """Use causal appearance plus short-horizon motion, never static pixels."""
        if item.last_mask is None:
            return None
        descriptor = (item.descriptor if item.descriptor is not None
                      else item.persistence_descriptor)
        appearance = (float(np.abs(_appearance(rgb, proposal.mask) - descriptor).sum())
                      if descriptor is not None else 0.0)
        history = self._centers.get(item.track_id, [])
        spatial = 0.0
        if history:
            predicted = history[-1][1]
            if len(history) >= 2:
                f0, p0 = history[-2]
                f1, p1 = history[-1]
                if f1 > f0:
                    predicted = p1 + (p1 - p0) * ((self._last_frame - f1) / (f1 - f0))
            spatial = float(np.linalg.norm(_center(proposal.mask) - predicted))
            spatial /= max(float(max(rgb.shape[:2])), 1.0)
        # Identical bowls have deliberately weak appearance evidence.  Give
        # causal motion continuity enough weight to keep two established
        # wrist tracks apart without assigning them by left/right ordering.
        return 0.65 * appearance + spatial

    def _quality(self, item: ObjectMemory, mask: np.ndarray, rgb: np.ndarray,
                 evidence: Mapping[str, Any]) -> str | None:
        """Validate moving-camera masks against the last observation.

        The agent-view implementation compares every mask with a fixed,
        largest-seen persistence anchor.  That is appropriate for a fixed
        camera, but a wrist camera legitimately changes an object's scale and
        appearance.  This override remains causal: it compares only with the
        immediately preceding accepted RGB observation and relaxes further
        after a real observation gap.
        """
        value = np.asarray(mask, dtype=bool)
        if not value.any():
            return "no_visible_mask"
        if evidence.get("object_score_logits_max") is not None:
            if float(evidence["object_score_logits_max"]) <= 0.0:
                return "native_presence_nonpositive"
        if item.last_mask is not None:
            gap = max(int(self._last_frame) - int(item.last_observed_frame or 0), 1)
            ratio = int(value.sum()) / max(int(item.last_mask.sum()), 1)
            if gap == 1:
                if ratio < 0.08 or ratio > 12.0:
                    return "wrist_temporal_area_jump"
            elif ratio < 0.02 or ratio > 40.0:
                return "wrist_reacquisition_area_jump"
            descriptor = _appearance(rgb, value)
            reference = (item.descriptor if item.descriptor is not None
                         else item.persistence_descriptor)
            limit = 0.95 if gap == 1 else 1.25
            if reference is not None and float(np.abs(descriptor - reference).sum()) > limit:
                return "wrist_temporal_appearance_jump"
        for other in self.objects.values():
            if other is item or other.status != "observed" or other.last_observed_frame != self._last_frame:
                continue
            if other.last_mask is not None and _overlap_fraction(value, other.last_mask) >= 0.90:
                return f"duplicate_of:{other.track_id}"
        return None

    def _accept_reacquisition(self, item: ObjectMemory, proposal: MaskProposal,
                              rgb: np.ndarray, frame: int, encoded: bytes) -> None:
        if item.native_member is not None:
            self._remove_native(item)
        self._remember(item, proposal, rgb, frame, "wrist_text_reacquisition")
        member = self._member_name(item)
        if self._members:
            self.tracker.add_current_object(
                scene_member=self._members[0], env_id=member,
                frame_ts=float(frame), jpeg=encoded, seed_mask=proposal.mask,
                query=proposal.prompt,
            )
            item.native_member = member
            self._members.append(member)
        else:
            self._bind(frame, encoded, [item])

    def _recover_one(self, rgb: np.ndarray, frame: int, encoded: bytes) -> None:
        """Rediscover every currently missing class on each sampled frame.

        The parent controller rotates through one missing class at a time.
        With six classes that can leave a wrist object absent for roughly 30
        frames.  The wrist extension emits every fifth frame, so perform one
        batched, current-frame text sweep at those exact frames and seed every
        unambiguous result back into native video tracking.
        """
        if frame % 5:
            return
        missing_by_class = {
            concept.name: [item for item in self.objects.values()
                           if item.class_id == concept.name and item.status != "observed"]
            for concept in AUTOMATIC_CATALOG
        }
        active_classes = tuple(name for name, items in missing_by_class.items() if items)
        if not active_classes:
            self._latest_acquisition = {
                "wrist_periodic_discovery": {
                    "frame": int(frame), "classes": [], "decision": "all_tracks_current",
                }
            }
            return
        proposals_by_class = self.acquirer.acquire(
            rgb, classes=active_classes, multiscale=True,
        )
        diagnostics = dict(self.acquirer.diagnostics)
        decisions: dict[str, Any] = {}

        for class_id in active_classes:
            missing = missing_by_class[class_id]
            proposals = []
            for proposal in proposals_by_class[class_id]:
                if any(other.status == "observed" and other.last_mask is not None
                       and _overlap_fraction(proposal.mask, other.last_mask) >= 0.90
                       for other in self.objects.values()):
                    continue
                proposals.append(proposal)
            accepted: list[tuple[ObjectMemory, MaskProposal]] = []

            if class_id != "black_bowl":
                item = missing[0]
                ranked = sorted(
                    ((self._reacquisition_cost(item, proposal, rgb), index, proposal)
                     for index, proposal in enumerate(proposals)
                     if self._quality(item, proposal.mask, rgb, {}) is None),
                    key=lambda row: (float("inf") if row[0] is None else row[0], row[1]),
                )
                if ranked:
                    accepted.append((item, ranked[0][2]))
                decisions[class_id] = {
                    "missing_tracks": [item.track_id for item in missing],
                    "usable_proposals": len(ranked),
                    "decision": "accepted" if accepted else "no_usable_current_proposal",
                }
            elif len(missing) == 1:
                item = missing[0]
                ranked = sorted(
                    ((self._reacquisition_cost(item, proposal, rgb), index, proposal)
                     for index, proposal in enumerate(proposals)
                     if self._quality(item, proposal.mask, rgb, {}) is None),
                    key=lambda row: (float("inf") if row[0] is None else row[0], row[1]),
                )
                if ranked and self._owns_reacquisition(item, ranked[0][2], rgb):
                    accepted.append((item, ranked[0][2]))
                decisions[class_id] = {
                    "missing_tracks": [item.track_id], "usable_proposals": len(ranked),
                    "decision": "accepted" if accepted else "unresolved_single_bowl_identity",
                }
            else:
                # Two lost identical bowls need a joint temporal assignment.
                # Never assign a lone proposal to an arbitrary numbered track.
                import itertools
                candidates: list[tuple[float, tuple[int, ...]]] = []
                if len(missing) == 2 and len(proposals) >= 2:
                    for indices in itertools.permutations(range(len(proposals)), 2):
                        costs = [self._reacquisition_cost(item, proposals[index], rgb)
                                 for item, index in zip(missing, indices)]
                        if any(cost is None for cost in costs):
                            continue
                        if any(self._quality(item, proposals[index].mask, rgb, {}) is not None
                               for item, index in zip(missing, indices)):
                            continue
                        candidates.append((float(sum(costs)), indices))
                candidates.sort(key=lambda row: (row[0], row[1]))
                margin = (candidates[1][0] - candidates[0][0]
                          if len(candidates) > 1 else None)
                if candidates and (margin is None or margin >= 0.03):
                    accepted.extend((item, proposals[index])
                                    for item, index in zip(missing, candidates[0][1]))
                decisions[class_id] = {
                    "missing_tracks": [item.track_id for item in missing],
                    "usable_proposals": len(proposals),
                    "joint_assignment_margin": margin,
                    "required_margin": 0.03,
                    "decision": "accepted_joint_temporal_assignment" if accepted
                                else "unresolved_identical_bowl_assignment",
                }

            for item, proposal in accepted:
                self._accept_reacquisition(item, proposal, rgb, frame, encoded)

        diagnostics["wrist_periodic_discovery"] = {
            "frame": int(frame), "classes": list(active_classes), "decisions": decisions,
            "contract": "current-frame RGB text discovery; past-only identity matching",
        }
        self._latest_acquisition = diagnostics

    def _name_moving_bowl(self, frame: int) -> None:
        # Wrist-local numbering must never become a cross-view identity claim.
        return None

    @staticmethod
    def _touches_border(mask: np.ndarray, margin: int = 3) -> bool:
        if mask is None or not np.asarray(mask, dtype=bool).any():
            return False
        value = np.asarray(mask, dtype=bool)
        return bool(value[:margin].any() or value[-margin:].any()
                    or value[:, :margin].any() or value[:, -margin:].any())

    def _lost_state(self, item: ObjectMemory, frame: int) -> tuple[str, dict[str, Any]]:
        if item.last_observed_frame is None or item.last_mask is None:
            return "unresolved", {"reason": "never_observed"}
        history = self._centers.get(item.track_id, [])
        predicted = history[-1][1].copy() if history else _center(item.last_mask)
        velocity = np.zeros(2, dtype=np.float64)
        if len(history) >= 2:
            f0, p0 = history[-2]
            f1, p1 = history[-1]
            if f1 > f0:
                velocity = (p1 - p0) / (f1 - f0)
                predicted = p1 + velocity * (frame - f1)
        height, width = self._last_rgb_shape or item.last_mask.shape
        outside = not (-0.02 * width <= predicted[0] <= 1.02 * width
                       and -0.02 * height <= predicted[1] <= 1.02 * height)
        border_exit = self._touches_border(item.last_mask) and float(np.linalg.norm(velocity)) > 0.5
        state = "out_of_view" if outside or border_exit else "occluded_remembered"
        return state, {
            "reason": "causal_border_or_motion_exit" if state == "out_of_view"
                      else "lost_inside_image_without_exit_evidence",
            "last_observed_frame": item.last_observed_frame,
            "predicted_center_xy": [float(predicted[0]), float(predicted[1])],
            "velocity_xy_per_frame": [float(velocity[0]), float(velocity[1])],
            "last_mask_touched_border": self._touches_border(item.last_mask),
        }

    def _emit(self, frame: int) -> dict[str, Any]:
        output = super()._emit(frame)
        by_track = {item.track_id: item for item in self.objects.values()}
        for state in output["states"]:
            item = by_track[state["track_id"]]
            tracker_status = state["status"]
            if tracker_status == "observed":
                observation_state = "current_observation"
                visibility_evidence = {"reason": "nonempty_native_current_mask"}
            else:
                observation_state, visibility_evidence = self._lost_state(item, frame)
            state.update({
                "tracker_status": tracker_status,
                "status": observation_state,
                "observation_state": observation_state,
                "visibility_evidence": visibility_evidence,
                "membership_eligible": observation_state == "current_observation",
                "persistence_method": None,
                "geometry_source_frame": (frame if observation_state == "current_observation"
                                          else None),
            })
        # Never expose remembered wrist pixels as current membership or geometry.
        output["effective_masks"] = dict(output["masks"])
        output["triplets"] = []
        output["observed_triplets"] = []
        output["relation_revision"] = "frozen_agent_triplet_filter_only"
        return output
