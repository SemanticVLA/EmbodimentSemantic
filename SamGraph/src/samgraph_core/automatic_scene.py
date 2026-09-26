"""Causal, inventory-guided SAM scene acquisition and object memory.

Only RGB, task text, and the pinned SAM runtime enter this module. Reference
masks and benchmark labels are deliberately absent from its interface.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .geometric_graph import (
    build_directed_relations, geometric_relation_revision,
    geometric_relation_rules, geometric_relation_rules_sha256, graph_instance,
    mask_centroid,
)


@dataclass(frozen=True)
class ObjectClass:
    name: str
    role: str
    count: int
    prompts: tuple[str, ...]


AUTOMATIC_CATALOG = (
    ObjectClass("black_bowl", "bowl", 2, (
        "ceramic bowl with patterned interior", "black and white patterned bowl",
    )),
    ObjectClass("cookies", "support", 1, (
        "box of cookies", "cookie package", "rectangular food package", "cracker box",
    )),
    ObjectClass("plate", "support", 1, ("white plate with red rings", "white dinner plate")),
    ObjectClass("white_ramekin", "support", 1, (
        "ribbed silver cup", "ribbed metal container", "silver cylindrical cup",
    )),
    # A visible burner is an accepted spatial proxy for the larger stove.
    ObjectClass("flat_stove", "support", 1, (
        "burner", "stove burner", "spiral burner", "circular grate",
    )),
    ObjectClass("wooden_cabinet", "cabinet", 1, (
        "wooden cabinet", "black wooden cabinet",
        "wooden furniture", "open wooden drawer",
    )),
)
CATALOG_BY_NAME = {item.name: item for item in AUTOMATIC_CATALOG}


@dataclass(frozen=True)
class MaskProposal:
    class_id: str
    mask: np.ndarray
    prompt: str
    score: float | None
    prompt_votes: int = 1

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(left.sum()) + int(right.sum()) - intersection
    return intersection / union if union else 0.0


def _overlap_fraction(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    return intersection / min(int(left.sum()), int(right.sum()))


def _mask_from_png(png: bytes, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(io.BytesIO(png))
    channel = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
    mask = np.ascontiguousarray(np.asarray(channel) > 0, dtype=bool)
    if mask.shape != shape or not mask.any():
        raise ValueError("SAM proposal has empty or incorrectly sized mask")
    return mask


def _encode_rgb(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()


def _usable_counts(raw: Mapping[str, list[MaskProposal]]) -> dict[str, int]:
    """Count selectable identities, not duplicate/wrong-class raw detections."""
    accepted: dict[str, list[MaskProposal]] = {name: [] for name in raw}
    for proposal in sorted(
        (p for proposals in raw.values() for p in proposals),
        key=lambda p: (-(p.score if p.score is not None else -1), -p.prompt_votes,
                       -p.area, p.class_id),
    ):
        if len(accepted[proposal.class_id]) >= CATALOG_BY_NAME[proposal.class_id].count:
            continue
        if any(_overlap_fraction(proposal.mask, old.mask) >= 0.90
               for values in accepted.values() for old in values):
            continue
        accepted[proposal.class_id].append(proposal)
    all_selected = [p for values in accepted.values() for p in values]
    counts = {name: 0 for name in raw}
    for proposal in all_selected:
        overlap = np.zeros_like(proposal.mask)
        for other in all_selected:
            if other is not proposal:
                overlap |= other.mask
        if int(np.logical_and(proposal.mask, overlap).sum()) <= proposal.area * 0.01:
            counts[proposal.class_id] += 1
    return counts


class AutomaticMaskAcquirer:
    """Full-frame, shared-vocabulary acquisition; never accepts pixel hints."""

    def __init__(self, segmenter: Any, *, agreement_iou: float = 0.70,
                 capture_candidates: bool = False,
                 prompts_by_class: Mapping[str, list[str]] | None = None):
        self.segmenter = segmenter
        self.agreement_iou = float(agreement_iou)
        self.diagnostics: dict[str, Any] = {}
        self.capture_candidates = bool(capture_candidates)
        self.candidate_masks: dict[str, np.ndarray] = {}
        self.candidate_evidence: list[dict[str, Any]] = []
        self.set_prompts(prompts_by_class)

    def set_prompts(self, prompts_by_class: Mapping[str, list[str]] | None) -> None:
        if prompts_by_class is None:
            self.catalog = CATALOG_BY_NAME
            return
        if set(prompts_by_class) != set(CATALOG_BY_NAME):
            raise ValueError("name configuration must contain exactly the six object classes")
        self.catalog = {}
        for name, concept in CATALOG_BY_NAME.items():
            prompts = prompts_by_class[name]
            if not isinstance(prompts, (list, tuple)) or not prompts or any(
                    not isinstance(p, str) or not p.strip() for p in prompts):
                raise ValueError(f"invalid description list: {name}")
            self.catalog[name] = ObjectClass(name, concept.role, concept.count, tuple(prompts))

    def _record_candidate(self, proposal: MaskProposal, *,
                          window_xywh: tuple[int, int, int, int]) -> None:
        """Retain model outputs before selection for CPU analysis, never inputs."""
        if not self.capture_candidates:
            return
        key = hashlib.sha256(proposal.mask.tobytes()).hexdigest()
        self.candidate_masks.setdefault(key, proposal.mask)
        self.candidate_evidence.append({
            "mask_key": key, "class_id": proposal.class_id,
            "prompt": proposal.prompt, "score": proposal.score,
            "area": proposal.area, "window_xywh": list(window_xywh),
        })

    def acquire(
        self, rgb: np.ndarray, *, classes: tuple[str, ...] | None = None,
        multiscale: bool = True,
    ) -> dict[str, list[MaskProposal]]:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("automatic acquisition requires HxWx3 RGB")
        self.candidate_masks = {}
        self.candidate_evidence = []
        encoded = _encode_rgb(image)
        selected = classes or tuple(CATALOG_BY_NAME)
        raw: dict[str, list[MaskProposal]] = {}
        diagnostics: dict[str, Any] = {}
        segment_many = getattr(self.segmenter, "segment_many", None)
        segment = (getattr(self.segmenter, "segment_candidates", None)
                   or getattr(self.segmenter, "segment", None))
        if not callable(segment_many) and not callable(segment):
            raise TypeError("segmenter must provide segment_many or segment_candidates")
        all_prompts = tuple(dict.fromkeys(
            prompt for class_id in selected for prompt in self.catalog[class_id].prompts
        ))
        responses = (segment_many(prompts=all_prompts, jpeg=encoded)
                     if callable(segment_many) else None)
        for class_id in selected:
            concept = self.catalog[class_id]
            candidates: list[MaskProposal] = []
            prompt_counts: dict[str, int] = {}
            for prompt in concept.prompts:
                response = (responses[prompt] if responses is not None
                            else segment(prompt=prompt, jpeg=encoded))
                prompt_counts[prompt] = len(response.masks)
                for item in response.masks:
                    mask = _mask_from_png(item.png, image.shape[:2])
                    score_value = getattr(item, "score", None)
                    score = None if score_value is None else float(score_value)
                    proposal = MaskProposal(class_id, mask, prompt, score)
                    candidates.append(proposal)
                    self._record_candidate(proposal, window_xywh=(
                        0, 0, image.shape[1], image.shape[0],
                    ))
            # A single effective name is sufficient. Synonyms consolidate
            # duplicate outputs. Keep the strongest scored representative;
            # choosing by area first discards confidence evidence and can make
            # a wrong-object cluster outrank the true object's strong proposal.
            # never union a printed part into a whole-object mask.
            clusters: list[list[MaskProposal]] = []
            for proposal in sorted(candidates, key=lambda p: (-p.area, p.prompt)):
                match = next((cluster for cluster in clusters
                              if any(_iou(proposal.mask, member.mask) >= self.agreement_iou
                                     for member in cluster)), None)
                if match is None:
                    clusters.append([proposal])
                else:
                    match.append(proposal)
            retained: list[MaskProposal] = []
            cluster_evidence = []
            for cluster in clusters:
                votes = len({item.prompt for item in cluster})
                cluster_evidence.append({
                    "prompt_votes": votes, "member_count": len(cluster),
                    "areas": sorted({item.area for item in cluster}),
                    "scores": sorted({item.score for item in cluster
                                      if item.score is not None}),
                })
                representative = max(
                    cluster, key=lambda p: (p.score if p.score is not None else -1, p.area),
                )
                retained.append(MaskProposal(
                    class_id, representative.mask, representative.prompt,
                    representative.score, votes,
                ))
            retained.sort(key=lambda p: (
                -p.prompt_votes, -(p.score if p.score is not None else -1),
                -p.area, hashlib.sha256(p.mask.tobytes()).hexdigest(),
            ))
            raw[class_id] = retained
            diagnostics[class_id] = {
                "per_prompt_counts": prompt_counts,
                "raw_candidate_count": len(candidates),
                "clusters": cluster_evidence,
                "agreed_candidate_count": len(retained),
            }

        # SAM text grounding can miss a partly occluded object when it is
        # small in the full image. Search deterministic overlapping tiles,
        # proceeding to the finer scale only for still-missing classes.
        # These windows are identical for every task and contain no manual
        # object coordinates or reference-mask information.
        usable_counts = _usable_counts(raw)
        missing = tuple(name for name in selected
                        if usable_counts[name] < CATALOG_BY_NAME[name].count)
        if multiscale and missing and min(image.shape[:2]) >= 128:
            height, width = image.shape[:2]
            windows: list[tuple[int, int, int, int, int]] = []
            for scale, positions in ((0.65, 2), (0.38, 4)):
                tile_height = min(height, max(128, round(height * scale)))
                tile_width = min(width, max(128, round(width * scale)))
                ys = tuple(round(i * (height - tile_height) / (positions - 1))
                           for i in range(positions))
                xs = tuple(round(i * (width - tile_width) / (positions - 1))
                           for i in range(positions))
                windows.extend((x, y, tile_width, tile_height, positions)
                               for y in ys for x in xs)
            fine_missing = missing
            fine_started = False
            for x, y, tile_width, tile_height, positions in windows:
                if positions == 4 and not fine_started:
                    fine_started = True
                    usable_counts = _usable_counts(raw)
                    fine_missing = tuple(name for name in missing
                                         if usable_counts[name] < CATALOG_BY_NAME[name].count)
                active_names = missing if positions == 2 else fine_missing
                if not active_names:
                    break
                tile_prompts = tuple(dict.fromkeys(
                    prompt for name in active_names
                    for prompt in self.catalog[name].prompts
                ))
                tile = image[y:y + tile_height, x:x + tile_width]
                tile_encoded = _encode_rgb(tile)
                tile_responses = (segment_many(prompts=tile_prompts, jpeg=tile_encoded)
                                  if callable(segment_many) else None)
                for name in active_names:
                    diagnostics[name]["tile_window_count"] = (
                        diagnostics[name].get("tile_window_count", 0) + 1
                    )
                    tile_candidates: list[MaskProposal] = []
                    for prompt in self.catalog[name].prompts:
                        response = (tile_responses[prompt] if tile_responses is not None
                                    else segment(prompt=prompt, jpeg=tile_encoded))
                        for item in response.masks:
                            local = _mask_from_png(item.png, tile.shape[:2])
                            mask = np.zeros(image.shape[:2], dtype=bool)
                            mask[y:y + tile_height, x:x + tile_width] = local
                            score_value = getattr(item, "score", None)
                            proposal = MaskProposal(
                                name, np.ascontiguousarray(mask), prompt,
                                None if score_value is None else float(score_value),
                            )
                            tile_candidates.append(proposal)
                            self._record_candidate(proposal, window_xywh=(
                                x, y, tile_width, tile_height,
                            ))
                    clusters: list[list[MaskProposal]] = []
                    for proposal in sorted(tile_candidates, key=lambda p: (-p.area, p.prompt)):
                        match = next((cluster for cluster in clusters
                                      if any(_iou(proposal.mask, member.mask)
                                             >= self.agreement_iou for member in cluster)), None)
                        if match is None:
                            clusters.append([proposal])
                        else:
                            match.append(proposal)
                    for cluster in clusters:
                        votes = len({item.prompt for item in cluster})
                        representative = max(
                            cluster, key=lambda p: (p.score if p.score is not None else -1, p.area),
                        )
                        # A small cup-like fragment of a full patterned bowl
                        # is not a second object, even if several synonymous
                        # prompts agree on that fragment.
                        if name == "white_ramekin" and any(
                            bowl.area >= 2 * representative.area
                            and _overlap_fraction(representative.mask, bowl.mask) >= 0.90
                            for bowl in raw.get("black_bowl", [])
                        ):
                            continue
                        raw[name].append(MaskProposal(
                            name, representative.mask, representative.prompt,
                            representative.score, votes,
                        ))
                        diagnostics[name].setdefault("tile_candidates", []).append({
                            "window_xywh": [x, y, tile_width, tile_height],
                            "prompt_votes": votes, "area": representative.area,
                            "prompt": representative.prompt,
                        })
            for name in missing:
                raw[name].sort(key=lambda p: (
                    -p.prompt_votes, -(p.score if p.score is not None else -1),
                    -p.area, hashlib.sha256(p.mask.tobytes()).hexdigest(),
                ))
                diagnostics[name]["agreed_candidate_count"] = len(raw[name])

        # Resolve entire-scene duplicate identities before assigning slots.
        ranked = sorted(
            (p for proposals in raw.values() for p in proposals),
            key=lambda p: (-(p.score if p.score is not None else -1), -p.prompt_votes,
                           -p.area, p.class_id),
        )
        accepted: dict[str, list[MaskProposal]] = {name: [] for name in selected}
        for proposal in ranked:
            if len(accepted[proposal.class_id]) >= CATALOG_BY_NAME[proposal.class_id].count:
                continue
            if any(_overlap_fraction(proposal.mask, old.mask) >= 0.90
                   for values in accepted.values() for old in values):
                continue
            accepted[proposal.class_id].append(proposal)
        # Native full-mask binding rejects overlapping seeds. Discard only a
        # tiny boundary disagreement; a material collision stays unresolved.
        # Evaluate every collision against the same selection. Mutating
        # accepted while inspecting it makes rejection depend on class order.
        selected_proposals = tuple(p for values in accepted.values() for p in values)
        cleaned: dict[str, list[MaskProposal]] = {}
        for name, proposals in accepted.items():
            clean: list[MaskProposal] = []
            for proposal in proposals:
                overlap = np.zeros(image.shape[:2], dtype=bool)
                for other in selected_proposals:
                    if other is not proposal:
                        overlap |= other.mask
                lost = int(np.logical_and(proposal.mask, overlap).sum())
                if lost > proposal.area * 0.01:
                    continue
                mask = np.logical_and(proposal.mask, np.logical_not(overlap))
                if mask.any():
                    clean.append(MaskProposal(
                        name, np.ascontiguousarray(mask), proposal.prompt,
                        proposal.score, proposal.prompt_votes,
                    ))
            cleaned[name] = clean
            diagnostics[name]["selected_count"] = len(clean)
            diagnostics[name]["selected_areas"] = [item.area for item in clean]
            diagnostics[name]["prompt_list"] = list(self.catalog[name].prompts)
            diagnostics[name]["selected_masks"] = [
                {"prompt": item.prompt, "score": item.score,
                 "prompt_votes": item.prompt_votes, "area": item.area,
                 "mask_sha256": hashlib.sha256(item.mask.tobytes()).hexdigest()}
                for item in clean
            ]
        self.diagnostics = diagnostics
        return cleaned


def _appearance(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if not len(pixels):
        return np.zeros(24, dtype=np.float32)
    descriptor = np.concatenate([
        np.histogram(pixels[:, channel], bins=8, range=(0, 256))[0]
        for channel in range(3)
    ]).astype(np.float32)
    return descriptor / max(float(descriptor.sum()), 1.0)


def _center(mask: np.ndarray) -> np.ndarray:
    return mask_centroid(mask)


@dataclass
class ObjectMemory:
    track_id: str
    class_id: str
    role: str
    semantic_id: str | None
    native_member: str | None = None
    last_mask: np.ndarray | None = None
    last_observed_frame: int | None = None
    persistence_mask: np.ndarray | None = None
    persistence_frame: int | None = None
    persistence_max_area: int = 0
    persistence_descriptor: np.ndarray | None = None
    descriptor: np.ndarray | None = None
    status: str = "unresolved"
    method: str | None = None
    prompt: str | None = None
    rejection: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    initial_center: np.ndarray | None = None
    initial_size: float | None = None
    motion_streak: int = 0
    identity_assigned_frame: int | None = None


class AutomaticSceneController:
    """One causal episode, including objects whose current mask disappears."""

    def __init__(self, segmenter: Any, tracker: Any, *, geometry_rules: Any,
                 task_prompts: Mapping[str, Mapping[str, list[str]]] | None = None,
                 camera_id: str = "agentview"):
        self.acquirer = AutomaticMaskAcquirer(segmenter)
        self.task_prompts = task_prompts
        self.tracker = tracker
        self.camera_id = str(camera_id)
        self.geometry_rules = geometry_rules
        self.objects: dict[str, ObjectMemory] = {}
        self._members: list[str] = []
        self._episode = ""
        self._instruction = ""
        self._last_frame = -1
        self._last_rgb_shape: tuple[int, int] | None = None
        self._last_attempt: dict[str, int] = {}
        self._next_class = 0
        self._latest_acquisition: dict[str, Any] | None = None

    def close_episode(self) -> None:
        for member in list(self._members):
            self.tracker.stop(member, camera_id=self.camera_id)
        self._members.clear()
        self.objects.clear()
        self._last_frame = -1

    def _member_name(self, item: ObjectMemory) -> str:
        return f"automatic:{self._episode}:{item.track_id}"

    def _bind(self, frame: int, encoded: bytes, items: list[ObjectMemory]) -> None:
        new_members = []
        try:
            for item in items:
                if item.last_mask is None:
                    continue
                member = self._member_name(item)
                self.tracker.start(
                    env_id=member, camera_id=self.camera_id, query=item.prompt or item.class_id,
                    frame_ts=float(frame), jpeg=encoded, seed_mask=item.last_mask,
                )
                new_members.append(member)
                item.native_member = member
            if new_members:
                self.tracker.bind_scene(new_members, seed_conditioning="full_mask")
                self._members.extend(new_members)
        except Exception:
            for member in new_members:
                self.tracker.stop(member, camera_id=self.camera_id)
            for item in items:
                item.native_member = None
            raise

    def start_episode(self, *, task: str, instruction: str, frame: int,
                      rgb: np.ndarray) -> dict[str, Any]:
        self.close_episode()
        self._episode = str(task)
        if self.task_prompts is not None:
            task_name = str(task).split("/")[0]
            if task_name not in self.task_prompts:
                raise ValueError(f"no hardcoded names configured for task: {task_name}")
            self.acquirer.set_prompts(self.task_prompts[task_name])
        self._instruction = str(instruction)
        self._last_frame = int(frame)
        self._last_rgb_shape = rgb.shape[:2]
        self._last_attempt = {concept.name: int(frame) for concept in AUTOMATIC_CATALOG}
        self._next_class = 0
        acquired = self.acquirer.acquire(rgb)
        self._latest_acquisition = self.acquirer.diagnostics
        # Identical bowls retain temporary track IDs until observed motion
        # identifies the manipulated bowl. Localization never depends on
        # resolving the instruction's spatial reference at frame zero.
        for concept in AUTOMATIC_CATALOG:
            proposals = acquired[concept.name]
            for slot in range(1, concept.count + 1):
                track_id = f"{concept.name}_track_{slot}"
                semantic_id = f"{concept.name}_{slot}" if concept.name != "black_bowl" else None
                item = ObjectMemory(track_id, concept.name, concept.role, semantic_id)
                proposal_index = slot - 1
                if proposal_index < len(proposals):
                    self._remember(item, proposals[proposal_index], rgb, frame, "text_initialization")
                self.objects[track_id] = item
        seeds = [item for item in self.objects.values() if item.last_mask is not None]
        if seeds:
            self._bind(frame, _encode_rgb(rgb), seeds)
        return self._emit(frame)

    @staticmethod
    def _remember(item: ObjectMemory, proposal: MaskProposal, rgb: np.ndarray,
                  frame: int, method: str) -> None:
        item.last_mask = np.ascontiguousarray(proposal.mask, dtype=bool)
        # Keep a distinct, past-only shape anchor. Gradual occlusion must not
        # erode this memory through a sequence of individually small changes.
        # This anchor is an estimate only when emitted during lost observation;
        # it never conditions or replaces the native current-frame tracker.
        area = int(item.last_mask.sum())
        if item.persistence_mask is None or area >= 0.90 * item.persistence_max_area:
            item.persistence_mask = item.last_mask.copy()
            item.persistence_frame = frame
            item.persistence_max_area = max(item.persistence_max_area, area)
            item.persistence_descriptor = _appearance(rgb, item.last_mask)
        if item.initial_center is None:
            item.initial_center = _center(item.last_mask).copy()
            ys, xs = np.nonzero(item.last_mask)
            item.initial_size = max(float(np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1)), 1.0)
        item.last_observed_frame = frame
        item.descriptor = _appearance(rgb, proposal.mask)
        item.status = "observed"
        item.method = method
        item.prompt = proposal.prompt
        item.rejection = None
        item.history.append({
            "frame": frame, "method": method, "prompt": proposal.prompt,
            "mask_sha256": hashlib.sha256(item.last_mask.tobytes()).hexdigest(),
            "prompt_votes": proposal.prompt_votes,
            "sam_score": proposal.score,
        })

    def _quality(self, item: ObjectMemory, mask: np.ndarray, rgb: np.ndarray,
                 evidence: Mapping[str, Any], *, reacquisition: bool = False) -> str | None:
        if not mask.any():
            return "no_visible_mask"
        if evidence.get("object_score_logits_max") is not None:
            if float(evidence["object_score_logits_max"]) <= 0.0:
                return "native_presence_nonpositive"
        if item.last_mask is not None:
            reference, reference_descriptor = (
                self._reacquisition_reference(item)
                if reacquisition else self._temporal_reference(item)
            )
            ratio = int(mask.sum()) / max(int(reference.sum()), 1)
            if ratio < 0.25 or ratio > 4.0:
                return "area_jump"
            descriptor = _appearance(rgb, mask)
            if reference_descriptor is not None and float(np.abs(descriptor - reference_descriptor).sum()) > 0.50:
                return "appearance_jump"
        for other in self.objects.values():
            if other is item or other.status != "observed" or other.last_observed_frame != self._last_frame:
                continue
            if other.last_mask is not None and _overlap_fraction(mask, other.last_mask) >= 0.90:
                return f"duplicate_of:{other.track_id}"
        return None

    def _temporal_reference(self, item: ObjectMemory) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the default stationary-object continuity reference.

        Dataset adapters may override this narrow policy seam for explicitly
        manipulated objects. The default preserves the qualified LIBERO
        maximum-area anchor exactly.
        """
        reference = item.persistence_mask if item.persistence_mask is not None else item.last_mask
        if reference is None:
            raise ValueError("temporal reference requires a prior mask")
        descriptor = (item.persistence_descriptor if item.persistence_descriptor is not None
                      else item.descriptor)
        return reference, descriptor

    def _reacquisition_reference(
        self, item: ObjectMemory,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the quality reference used for text reacquisition.

        This separate adapter seam lets a real-world dataset retain a clean
        pre-occlusion appearance anchor without changing native tracking or the
        default LIBERO policy.
        """
        return self._temporal_reference(item)

    def _remove_native(self, item: ObjectMemory) -> None:
        if item.native_member is None:
            return
        self.tracker.remove_object(item.native_member, camera_id=self.camera_id)
        self._members.remove(item.native_member)
        item.native_member = None

    def _reacquisition_cost(self, item: ObjectMemory, proposal: MaskProposal,
                            rgb: np.ndarray) -> float | None:
        if item.last_mask is None:
            return None
        reference, descriptor = self._temporal_reference(item)
        displacement = np.linalg.norm(_center(proposal.mask) - _center(reference))
        displacement /= max(float(max(rgb.shape[:2])), 1.0)
        appearance = (float(np.abs(_appearance(rgb, proposal.mask) - descriptor).sum())
                      if descriptor is not None else 0.0)
        return float(displacement + appearance)

    def _owns_reacquisition(self, item: ObjectMemory, proposal: MaskProposal,
                            rgb: np.ndarray) -> bool:
        """Require a proposal to prefer this track over identical-object memory.

        A single visible bowl is not evidence that it belongs to whichever lost
        track happens to be visited first. Compare *all* same-class memories,
        including occluded tracks, before accepting a text reacquisition. This
        uses only causal runtime masks and RGB, never benchmark identities.
        """
        own = self._reacquisition_cost(item, proposal, rgb)
        for other in self.objects.values():
            if other is item or other.class_id != item.class_id:
                continue
            competing = self._reacquisition_cost(other, proposal, rgb)
            if competing is not None and (own is None or competing - own < 0.10):
                return False
        return True

    def _recover_one(self, rgb: np.ndarray, frame: int, encoded: bytes) -> None:
        concepts = list(AUTOMATIC_CATALOG)
        for offset in range(len(concepts)):
            index = (self._next_class + offset) % len(concepts)
            concept = concepts[index]
            missing = [item for item in self.objects.values()
                       if item.class_id == concept.name and item.status != "observed"]
            if not missing or frame - self._last_attempt.get(concept.name, -5) < 5:
                continue
            self._next_class = (index + 1) % len(concepts)
            self._last_attempt[concept.name] = frame
            proposals = self.acquirer.acquire(
                rgb, classes=(concept.name,), multiscale=False,
            )[concept.name]
            self._latest_acquisition = self.acquirer.diagnostics
            used: set[int] = set()
            for item in missing:
                options: list[tuple[float, int, MaskProposal]] = []
                for proposal_index, proposal in enumerate(proposals):
                    if proposal_index in used:
                        continue
                    if any(other.status == "observed"
                           and other.last_mask is not None
                           and _overlap_fraction(proposal.mask, other.last_mask) >= 0.90
                           for other in self.objects.values() if other is not item):
                        continue
                    cost = self._reacquisition_cost(item, proposal, rgb)
                    if cost is None:
                        cost = float(proposal_index)
                    options.append((cost, proposal_index, proposal))
                options.sort(key=lambda row: (row[0], row[1]))
                if not options or (len(options) > 1 and options[1][0] - options[0][0] < 0.10):
                    continue
                _, proposal_index, proposal = options[0]
                if not self._owns_reacquisition(item, proposal, rgb):
                    item.rejection = "ambiguous_reacquisition_identity"
                    continue
                if self._quality(
                    item, proposal.mask, rgb, {}, reacquisition=True,
                ) is not None:
                    continue
                if item.native_member is not None:
                    self._remove_native(item)
                self._remember(item, proposal, rgb, frame, "text_reacquisition")
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
                used.add(proposal_index)
            return

    def _name_moving_bowl(self, frame: int) -> None:
        """Name the moving bowl using accepted observations through this frame.

        Three consecutive observations must show one bowl moving relative to
        its initial size while the other stays put. Until then both masks are
        emitted under stable temporary track IDs. Earlier outputs are untouched.
        """
        bowls = [item for item in self.objects.values() if item.class_id == "black_bowl"]
        if len(bowls) != 2 or any(item.semantic_id is not None for item in bowls):
            return
        if any(item.initial_center is None for item in bowls):
            for item in bowls:
                item.motion_streak = 0
            return
        for index, item in enumerate(bowls):
            other = bowls[1 - index]
            # Motion must be observed now. The other bowl may be occluded:
            # user-authorized stationary memory is sufficient to keep its
            # identity, but never to invent motion of the picked-up bowl.
            other_mask = (other.persistence_mask if other.status == "persisted"
                          else other.last_mask)
            moving = False
            if (item.status == "observed" and item.last_observed_frame == frame
                    and other.status in {"observed", "persisted"}
                    and item.last_mask is not None and other_mask is not None):
                displacement = float(np.linalg.norm(_center(item.last_mask) - item.initial_center)) / item.initial_size
                other_displacement = float(np.linalg.norm(_center(other_mask) - other.initial_center)) / other.initial_size
                moving = displacement >= 0.35 and other_displacement <= 0.15
            item.motion_streak = item.motion_streak + 1 if moving else 0
            if item.motion_streak >= 3:
                item.semantic_id = "black_bowl_1"
                bowls[1 - index].semantic_id = "black_bowl_2"
                for bowl in bowls:
                    bowl.identity_assigned_frame = frame
                return

    def step(self, *, frame: int, rgb: np.ndarray) -> dict[str, Any]:
        if self._last_frame < 0 or frame != self._last_frame + 1:
            raise ValueError("automatic scene requires consecutive, increasing raw frames")
        if rgb.shape[:2] != self._last_rgb_shape:
            raise ValueError("automatic scene RGB dimensions changed")
        self._last_frame = int(frame)
        self._latest_acquisition = None
        encoded = _encode_rgb(rgb)
        if self._members:
            self.tracker.submit_frame(
                env_id=self._members[0], camera_id=self.camera_id,
                frame_ts=float(frame), jpeg=encoded,
            )
        for item in self.objects.values():
            if item.native_member is None:
                item.status = "persisted" if item.last_mask is not None else "unresolved"
                continue
            snapshot = self.tracker.snapshot(item.native_member, camera_id=self.camera_id)
            mask = self.tracker.latest_mask(item.native_member, camera_id=self.camera_id)
            mask = (np.asarray(mask, dtype=bool) if mask is not None
                    else np.zeros(rgb.shape[:2], dtype=bool))
            rejection = self._quality(item, mask, rgb, snapshot.get("native_evidence") or {})
            if rejection is None:
                proposal = MaskProposal(item.class_id, mask, item.prompt or item.class_id, None)
                self._remember(item, proposal, rgb, frame, "native_tracking")
            else:
                item.status = "persisted" if item.last_mask is not None else "unresolved"
                item.rejection = rejection
                # A wrong current mask may already have entered native memory.
                # Remove that native generation before any new proposal binds.
                if rejection != "no_visible_mask":
                    self._remove_native(item)
        self._recover_one(rgb, frame, encoded)
        self._name_moving_bowl(frame)
        return self._emit(frame)

    def _emit(self, frame: int) -> dict[str, Any]:
        observed: dict[str, np.ndarray] = {}
        effective: dict[str, np.ndarray] = {}
        instances: list[dict[str, Any]] = []
        for item in self.objects.values():
            if item.last_mask is None:
                continue
            output_id = item.semantic_id or item.track_id
            emitted_mask, geometry_frame, public_status, persistence_method = (
                self._effective_mask(item, frame)
            )
            if emitted_mask is None:
                continue
            template = {"instance_id": output_id, "class_id": item.class_id,
                        "role": item.role}
            instance = graph_instance(
                template, emitted_mask, track_state=public_status,
                tracking_evidence=item.method,
            )
            instance.update({
                "current_geometry_valid": public_status == "observed",
                "last_observed_frame": item.last_observed_frame,
                "geometry_source_frame": geometry_frame,
                "persistence_method": persistence_method,
            })
            instances.append(instance)
            effective[output_id] = emitted_mask
            if item.status == "observed":
                observed[output_id] = item.last_mask

        def graph(masks: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
            visible = [item for item in instances if item["instance_id"] in masks]
            if len(visible) < 2:
                return []
            try:
                return build_directed_relations(
                    visible, masks, suite="spatial", instruction=self._instruction,
                    shape=self._last_rgb_shape, geometry_rules=self.geometry_rules,
                    allow_partial_visibility=True,
                )[1]
            except ValueError:
                # Incomplete or contradictory geometry is explicit in the
                # object states; do not create arbitrary relations.
                return []

        states = [{
            "track_id": item.track_id, "class_id": item.class_id,
            "semantic_id": item.semantic_id,
            "status": self._effective_mask(item, frame)[2],
            "output_id": item.semantic_id or item.track_id,
            "identity_assigned_frame": item.identity_assigned_frame,
            "identity_method": ("causal_motion" if item.identity_assigned_frame is not None
                                else "provisional" if item.semantic_id is None else "class_name"),
            "last_observed_frame": item.last_observed_frame,
            "geometry_source_frame": self._effective_mask(item, frame)[1],
            "persistence_method": self._effective_mask(item, frame)[3],
            "persistence_mask_sha256": (hashlib.sha256(item.persistence_mask.tobytes()).hexdigest()
                                        if item.persistence_mask is not None else None),
            "age_frames": (frame - item.last_observed_frame
                           if item.last_observed_frame is not None else None),
            "native_member": item.native_member, "method": item.method,
            "last_text_acquisition": next((dict(event) for event in reversed(item.history)
                                            if event["method"] in {"text_initialization", "text_reacquisition"}), None),
            "prompt": item.prompt, "rejection": item.rejection,
            "mask_scope": ("burner_proxy" if item.class_id == "flat_stove"
                           and item.last_mask is not None else "whole_object"),
            "mask_sha256": (hashlib.sha256(item.last_mask.tobytes()).hexdigest()
                            if item.last_mask is not None else None),
        } for item in self.objects.values()]
        return {
            "triplets": [[row["subject"], row["relation"], row["object"]]
                         for row in graph(effective)],
            "observed_triplets": [[row["subject"], row["relation"], row["object"]]
                                  for row in graph(observed)],
            "masks": observed, "effective_masks": effective,
            "states": states, "instances": instances,
            "acquisition_diagnostics": self._latest_acquisition,
            "inventory": {
                "coverage": {
                    "expected_class_count": len(AUTOMATIC_CATALOG),
                    "expected_instance_count": sum(item.count for item in AUTOMATIC_CATALOG),
                },
                "completeness": {
                    concept.name: {"expected_count": concept.count}
                    for concept in AUTOMATIC_CATALOG
                },
                "allow_partial_inventory": True,
            },
            "geometry_rules": geometric_relation_rules(self.geometry_rules),
            "geometry_rules_sha256": geometric_relation_rules_sha256(self.geometry_rules),
            "relation_revision": geometric_relation_revision(self.geometry_rules),
        }

    def _effective_mask(
        self, item: ObjectMemory, frame: int,
    ) -> tuple[np.ndarray | None, int | None, str, str | None]:
        """Select geometry exposed to renderers and graph consumers.

        The default retains the existing stationary past-shape estimate. A
        dataset adapter can instead expire stale geometry while leaving native
        tracking and reacquisition unchanged.
        """
        if item.last_mask is None:
            return None, None, "unresolved", None
        if item.status == "persisted" and item.persistence_mask is not None:
            return (
                item.persistence_mask,
                item.persistence_frame,
                "persisted",
                "stationary_past_shape_estimate",
            )
        return item.last_mask, item.last_observed_frame, item.status, None
