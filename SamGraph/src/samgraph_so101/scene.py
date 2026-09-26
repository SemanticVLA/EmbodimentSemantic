"""SO101 specialization of SamGraph's native automatic scene controller."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from samgraph_core import automatic_scene as automatic
from samgraph_core import LocalSam31SceneTracker

from .config import SO101Config


SO101_SINGLE_CLASS_RETRY_MIN_SCORE = 0.60


class _SO101SingleClassScoreGate:
    """Apply the SO101 fallback confidence floor to every isolated text pass."""

    def __init__(self, acquirer: Any):
        self._acquirer = acquirer
        self.last_rejected_candidates: dict[str, list[dict[str, Any]]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._acquirer, name)

    def acquire(self, rgb: np.ndarray, *, classes: tuple[str, ...] | None = None,
                multiscale: bool = True):
        results = self._acquirer.acquire(rgb, classes=classes, multiscale=multiscale)
        self.last_rejected_candidates = {}
        if classes is None or len(classes) != 1:
            return results
        class_id = classes[0]
        accepted = []
        rejected = []
        for proposal in results.get(class_id, []):
            if (proposal.score is None
                    or proposal.score < SO101_SINGLE_CLASS_RETRY_MIN_SCORE):
                rejected.append({
                    "prompt": proposal.prompt,
                    "sam_score": proposal.score,
                    "area": proposal.area,
                    "reason": "below_so101_single_class_retry_min_sam_score",
                    "min_sam_score": SO101_SINGLE_CLASS_RETRY_MIN_SCORE,
                })
            else:
                accepted.append(proposal)
        if rejected:
            results[class_id] = accepted
            self.last_rejected_candidates[class_id] = rejected
            self._acquirer.diagnostics.setdefault(class_id, {})[
                "so101_single_class_retry_score_gate"
            ] = {
                "min_sam_score": SO101_SINGLE_CLASS_RETRY_MIN_SCORE,
                "rejected_candidates": rejected,
            }
        return results


def install_so101_catalog(config: SO101Config) -> tuple[automatic.ObjectClass, ...]:
    """Install a process-local catalog while leaving SamGraph source unchanged.

    ``automatic_scene`` was originally built around a module-level LIBERO
    inventory.  SO101 runs in a dedicated process, so replacing that inventory
    before constructing any controller reuses localization, native tracking,
    reacquisition, persistence, and graph construction without importing the
    simulator-specific object list.
    """
    catalog = tuple(
        automatic.ObjectClass(item.object_id, item.role, 1, item.descriptions)
        for item in config.objects
    )
    automatic.AUTOMATIC_CATALOG = catalog
    automatic.CATALOG_BY_NAME = {item.name: item for item in catalog}
    return catalog


class SO101RolloverSceneTracker(LocalSam31SceneTracker):
    """Keep long real-robot episodes causal across SAM's 256-frame limit.

    The pinned SAM3.1 live adapter deliberately bounds one native session to
    256 RGB frames.  SO101 episodes are longer, so this dataset adapter starts
    a fresh native session on the *previously processed* RGB and conditions it
    with each canonical track's last non-empty native mask.  The next RGB is
    then appended normally.  No future frame, text relocalization, optical
    flow, manually supplied prompt, or cross-object identity matching is used.

    This class intentionally lives in the SO101 adapter instead of changing
    the shared LIBERO-qualified tracker.
    """

    def __init__(self, runtime: Any, *, max_native_session_frames: int = 256, **kwargs: Any):
        if not 2 <= int(max_native_session_frames) <= 256:
            raise ValueError("SO101 native session limit must be in [2, 256]")
        super().__init__(runtime, **kwargs)
        self.max_native_session_frames = int(max_native_session_frames)
        # Prefer a clean, fully visible handoff before the hard ceiling while
        # retaining the ceiling as a bounded fallback.
        self.rollover_trigger_frames = min(240, self.max_native_session_frames)
        self._so101_frames_in_session = 0
        self._so101_total_frames = 0
        self._so101_rollover_count = 0
        self._so101_latest_jpeg: bytes | None = None
        self._so101_latest_frame_ts: float | None = None
        self._so101_last_nonempty: dict[str, np.ndarray] = {}
        self._so101_queries: dict[str, str] = {}
        self._so101_last_rollover: dict[str, Any] | None = None

    def _reset_rollover_state(self) -> None:
        self._so101_frames_in_session = 0
        self._so101_total_frames = 0
        self._so101_rollover_count = 0
        self._so101_latest_jpeg = None
        self._so101_latest_frame_ts = None
        self._so101_last_nonempty.clear()
        self._so101_queries.clear()
        self._so101_last_rollover = None

    def _remember_scene_masks(self, members: list[str]) -> None:
        for member in members:
            session = self._sessions.get((member, self.camera_id))
            if session is None:
                continue
            self._so101_queries[member] = str(session.query)
            mask = np.ascontiguousarray(np.asarray(session.last_mask, dtype=bool))
            if mask.ndim == 2 and mask.any():
                self._so101_last_nonempty[member] = np.array(mask, copy=True)

    def _annotate_scene(self, members: list[str]) -> None:
        for member in members:
            session = self._sessions.get((member, self.camera_id))
            if session is None:
                continue
            session.snapshot.update({
                "native_session_generation": self._so101_rollover_count,
                "native_session_rollover_count": self._so101_rollover_count,
                "native_session_frames": self._so101_frames_in_session,
                "native_session_frame_limit": self.max_native_session_frames,
                "native_session_rollover_policy": "previous_rgb_last_nonempty_native_mask",
                "native_session_rollover": dict(self._so101_last_rollover)
                if self._so101_last_rollover is not None else None,
            })

    def bind_scene(self, members: list[str], *, seed_conditioning: str = "centroid_point") -> None:
        super().bind_scene(members, seed_conditioning=seed_conditioning)
        first = self._sessions[(members[0], self.camera_id)]
        self._so101_frames_in_session = 1
        self._so101_total_frames = 1
        self._so101_rollover_count = 0
        self._so101_latest_jpeg = bytes(first.previous_rgb)
        self._so101_latest_frame_ts = float(first.frame_ts)
        self._so101_last_nonempty.clear()
        self._so101_queries.clear()
        self._so101_last_rollover = None
        self._remember_scene_masks(list(members))
        self._annotate_scene(list(members))

    def _rollover_scene(self, scene_member: str) -> None:
        scene = self._scenes[str(scene_member)]
        members = list(scene.members)
        self._remember_scene_masks(members)
        if self._so101_latest_jpeg is None or self._so101_latest_frame_ts is None:
            raise RuntimeError("SO101 native rollover lacks the previous causal RGB frame")
        missing = [member for member in members if member not in self._so101_last_nonempty]
        if missing:
            raise RuntimeError(
                "SO101 native rollover lacks a prior non-empty mask for: " + ", ".join(missing)
            )
        previous_jpeg = self._so101_latest_jpeg
        previous_ts = self._so101_latest_frame_ts
        masks = {member: np.array(self._so101_last_nonempty[member], copy=True)
                 for member in members}
        queries = {member: self._so101_queries[member] for member in members}
        previous_session_id = scene.session_id
        previous_generation_frames = self._so101_frames_in_session

        # Use the qualified native stop/start/bind operations. Canonical member
        # names remain identical, so the controller's object identities do too.
        super().stop()
        started: list[str] = []
        try:
            for member in members:
                super().start(
                    env_id=member,
                    camera_id=self.camera_id,
                    query=queries[member],
                    frame_ts=previous_ts,
                    jpeg=previous_jpeg,
                    seed_mask=masks[member],
                )
                started.append(member)
            super().bind_scene(members, seed_conditioning="full_mask")
        except Exception:
            for member in started:
                try:
                    super().stop(member, camera_id=self.camera_id)
                except Exception:
                    pass
            raise

        self._so101_rollover_count += 1
        self._so101_frames_in_session = 1
        new_scene = self._scenes[members[0]]
        self._so101_last_rollover = {
            "generation": self._so101_rollover_count,
            "at_previous_frame_ts": previous_ts,
            "seed_source": "last_nonempty_native_mask",
            "seed_rgb_sha256": hashlib.sha256(previous_jpeg).hexdigest(),
            "members": list(members),
            "previous_native_session_id": previous_session_id,
            "new_native_session_id": new_scene.session_id,
            "previous_generation_frames": previous_generation_frames,
            "causal": True,
        }
        self._remember_scene_masks(members)
        self._annotate_scene(members)

    def _has_clean_rollover_seeds(self, scene_member: str) -> bool:
        scene = self._scenes[str(scene_member)]
        masks: list[np.ndarray] = []
        for member in scene.members:
            session = self._sessions.get((member, self.camera_id))
            if session is None:
                return False
            mask = np.asarray(session.last_mask, dtype=bool)
            if mask.ndim != 2 or not mask.any():
                return False
            if any(mask.shape != other.shape or np.logical_and(mask, other).any()
                   for other in masks):
                return False
            masks.append(mask)
        return True

    def submit_frame(self, *, env_id: str, camera_id: str,
                     frame_ts: float, jpeg: bytes) -> bool:
        scene = self._scenes[str(env_id)]
        digest = hashlib.sha256(jpeg).hexdigest()
        duplicate = float(frame_ts) == scene.frame_ts and digest == scene.digest
        if not duplicate:
            clean_handoff = (
                self._so101_frames_in_session >= self.rollover_trigger_frames
                and self._has_clean_rollover_seeds(str(env_id))
            )
            hard_handoff = self._so101_frames_in_session >= self.max_native_session_frames
            if clean_handoff or hard_handoff:
                self._rollover_scene(str(env_id))
        result = super().submit_frame(
            env_id=env_id, camera_id=camera_id, frame_ts=frame_ts, jpeg=jpeg,
        )
        scene = self._scenes[str(env_id)]
        if not duplicate:
            self._so101_frames_in_session += 1
            self._so101_total_frames += 1
            self._so101_latest_jpeg = bytes(jpeg)
            self._so101_latest_frame_ts = float(frame_ts)
            self._remember_scene_masks(list(scene.members))
        self._annotate_scene(list(scene.members))
        return result

    def add_current_object(self, *, scene_member: str, env_id: str,
                           frame_ts: float, jpeg: bytes, seed_mask: np.ndarray,
                           query: str) -> None:
        super().add_current_object(
            scene_member=scene_member, env_id=env_id, frame_ts=frame_ts,
            jpeg=jpeg, seed_mask=seed_mask, query=query,
        )
        self._so101_queries[str(env_id)] = str(query)
        self._so101_last_nonempty[str(env_id)] = np.array(seed_mask, dtype=bool, copy=True)
        scene = self._scenes[str(scene_member)]
        self._annotate_scene(list(scene.members))

    def remove_object(self, env_id: str, *, camera_id: str) -> None:
        member = str(env_id)
        super().remove_object(member, camera_id=camera_id)
        self._so101_last_nonempty.pop(member, None)
        self._so101_queries.pop(member, None)

    def rollover_provenance(self) -> dict[str, Any]:
        return {
            "policy": "previous_rgb_last_nonempty_native_mask",
            "causal": True,
            "native_session_frame_limit": self.max_native_session_frames,
            "preferred_rollover_after_frames": self.rollover_trigger_frames,
            "native_session_generation": self._so101_rollover_count,
            "native_session_rollover_count": self._so101_rollover_count,
            "frames_in_native_session": self._so101_frames_in_session,
            "total_episode_frames_processed": self._so101_total_frames,
            "latest_frame_ts": self._so101_latest_frame_ts,
            "last_rollover": dict(self._so101_last_rollover)
            if self._so101_last_rollover is not None else None,
        }

    def stop(self, env_id: str | None = None, camera_id: str | None = None):
        result = super().stop(env_id, camera_id)
        if not self._scenes:
            self._reset_rollover_state()
        return result


class SO101AgentSceneController(automatic.AutomaticSceneController):
    """Canonicalize the single SO101 instance of each configured class."""

    def __init__(self, segmenter: Any, tracker: Any, *, config: SO101Config,
                 task_prompts: Mapping[str, Mapping[str, list[str]]]):
        self.so101_config = config
        self.manipulated_object_ids = frozenset(
            str(item) for item in config.persistence["manipulated_object_ids"]
        )
        self.max_remembered_native_frames = int(
            config.persistence["max_remembered_native_frames"]
        )
        super().__init__(
            segmenter,
            tracker,
            geometry_rules=config.geometry,
            task_prompts=task_prompts,
            camera_id="agent_view",
        )
        self.acquirer = _SO101SingleClassScoreGate(self.acquirer)

    def start_episode(self, *, task: str, instruction: str, frame: int,
                      rgb):
        super().start_episode(task=task, instruction=instruction, frame=frame, rgb=rgb)
        for item in self.objects.values():
            item.semantic_id = item.class_id

        # The shared acquirer resolves all configured text classes jointly.
        # On real SO101 frames, a proposal for one class can suppress a valid
        # proposal for another class before it is seeded into SAM's native
        # tracker. Retry only still-missing classes through the same SAM text
        # acquirer in isolation; reject a retry if it duplicates an already
        # seeded object under SamGraph's existing >=90% overlap rule.
        joint_diagnostics = self._latest_acquisition
        fallback_records: list[dict[str, Any]] = []
        encoded = automatic._encode_rgb(np.asarray(rgb, dtype=np.uint8))
        missing_items = [item for item in self.objects.values() if item.last_mask is None]
        for item in missing_items:
            proposals_by_class = self.acquirer.acquire(
                rgb, classes=(item.class_id,), multiscale=True,
            )
            proposals = proposals_by_class.get(item.class_id, [])
            acquisition_diagnostics = self.acquirer.diagnostics.get(item.class_id, {})
            rejections: list[dict[str, Any]] = list(
                self.acquirer.last_rejected_candidates.get(item.class_id, [])
            )
            selected = None
            for proposal in proposals:
                if (proposal.score is None
                        or proposal.score < SO101_SINGLE_CLASS_RETRY_MIN_SCORE):
                    rejections.append({
                        "prompt": proposal.prompt,
                        "sam_score": proposal.score,
                        "area": proposal.area,
                        "reason": "below_so101_single_class_retry_min_sam_score",
                        "min_sam_score": SO101_SINGLE_CLASS_RETRY_MIN_SCORE,
                    })
                    continue
                conflicts = []
                for other in self.objects.values():
                    if other is item or other.last_mask is None:
                        continue
                    overlap = automatic._overlap_fraction(proposal.mask, other.last_mask)
                    if overlap >= 0.90:
                        conflicts.append({
                            "object_id": other.semantic_id or other.track_id,
                            "overlap_fraction_of_smaller_mask": float(overlap),
                        })
                if conflicts:
                    rejections.append({
                        "prompt": proposal.prompt,
                        "sam_score": proposal.score,
                        "area": proposal.area,
                        "reason": "duplicate_of_existing_seed",
                        "conflicts": conflicts,
                    })
                    continue
                selected = proposal
                break

            if selected is None:
                item.rejection = ("single_class_retry_candidates_conflict_with_existing_seeds"
                                  if rejections else "no_single_class_retry_candidate")
                fallback_records.append({
                    "object_id": item.class_id,
                    "status": "unresolved",
                    "candidate_count": len(proposals) + len(rejections),
                    "acquisition": acquisition_diagnostics,
                    "rejected_candidates": rejections,
                    "reason": item.rejection,
                })
                continue

            self._remember(
                item, selected, np.asarray(rgb, dtype=np.uint8), frame,
                "text_initialization_single_class_retry",
            )
            member = self._member_name(item)
            if self._members:
                self.tracker.add_current_object(
                    scene_member=self._members[0], env_id=member,
                    frame_ts=float(frame), jpeg=encoded, seed_mask=selected.mask,
                    query=selected.prompt,
                )
                item.native_member = member
                self._members.append(member)
            else:
                self._bind(frame, encoded, [item])
            fallback_records.append({
                "object_id": item.class_id,
                "status": "observed",
                "candidate_count": len(proposals) + len(rejections),
                "selected": {
                    "prompt": selected.prompt,
                    "sam_score": selected.score,
                    "prompt_votes": selected.prompt_votes,
                    "area": selected.area,
                    "mask_sha256": hashlib.sha256(selected.mask.tobytes()).hexdigest(),
                },
                "acquisition": acquisition_diagnostics,
                "rejected_candidates": rejections,
            })

        if missing_items:
            self._latest_acquisition = {
                "policy": "joint_text_acquisition_then_single_class_retry_for_missing_objects",
                "joint_acquisition": joint_diagnostics,
                "single_class_missing_object_fallbacks": fallback_records,
                "single_class_retry_min_sam_score": SO101_SINGLE_CLASS_RETRY_MIN_SCORE,
                "manual_points_boxes_or_masks": False,
            }
        return self._emit(frame)

    @property
    def initial_missing_object_policy(self) -> str:
        return (
            "joint_acquisition_then_single_class_text_retry_with_duplicate_seed_rejection;"
            f"min_sam_score={SO101_SINGLE_CLASS_RETRY_MIN_SCORE:.2f}"
        )

    def _name_moving_bowl(self, frame: int) -> None:
        # There is one configured black bowl, so its canonical identity is not
        # inferred from position or motion.
        return None

    def _temporal_reference(self, item: automatic.ObjectMemory):
        """Use the latest accepted bowl observation for native continuity."""
        if item.class_id in self.manipulated_object_ids and item.last_mask is not None:
            return item.last_mask, item.descriptor
        return super()._temporal_reference(item)

    def _reacquisition_reference(self, item: automatic.ObjectMemory):
        """Validate a reacquired bowl against its clean pre-occlusion anchor."""
        if item.class_id in self.manipulated_object_ids:
            reference = (
                item.persistence_mask
                if item.persistence_mask is not None else item.last_mask
            )
            if reference is None:
                raise ValueError("reacquisition reference requires a prior mask")
            descriptor = (
                item.persistence_descriptor
                if item.persistence_descriptor is not None else item.descriptor
            )
            return reference, descriptor
        return super()._reacquisition_reference(item)

    def _effective_mask(self, item: automatic.ObjectMemory, frame: int):
        if item.class_id not in self.manipulated_object_ids or item.status != "persisted":
            return super()._effective_mask(item, frame)
        if item.last_mask is None or item.last_observed_frame is None:
            return None, None, "unresolved", None
        age = int(frame) - int(item.last_observed_frame)
        if age <= self.max_remembered_native_frames:
            return (
                item.last_mask,
                item.last_observed_frame,
                "persisted",
                "short_horizon_latest_observation",
            )
        return None, None, "unresolved", "expired_manipulated_object_memory"

    def _emit(self, frame: int) -> dict[str, Any]:
        result = super()._emit(frame)
        snapshots: dict[str, dict[str, Any]] = {}
        for item in self.objects.values():
            if item.native_member is None:
                continue
            try:
                snapshots[item.native_member] = self.tracker.snapshot(
                    item.native_member, camera_id=self.camera_id)
            except (KeyError, RuntimeError):
                continue
        for state in result["states"]:
            snapshot = snapshots.get(state.get("native_member"))
            if snapshot is None:
                continue
            state["native_tracking"] = {
                key: snapshot.get(key) for key in (
                    "status", "mask_present", "source_frame_ts", "native_session_id",
                    "native_frame_index", "native_session_generation",
                    "native_session_rollover_count", "native_session_frames",
                    "native_session_frame_limit", "native_session_rollover_policy",
                    "native_session_rollover", "tracking_method", "visibility_reason",
                )
            }
        provenance = getattr(self.tracker, "rollover_provenance", None)
        result["tracking_session"] = provenance() if callable(provenance) else None
        result["persistence_policy"] = {
            **dict(self.so101_config.persistence),
            "native_tracking_cadence": 1,
        }
        return result


__all__ = [
    "SO101AgentSceneController", "SO101RolloverSceneTracker", "install_so101_catalog",
]
