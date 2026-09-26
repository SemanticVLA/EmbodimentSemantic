"""Submitted-frame LIBERO endpoint resolution using masks, a graph brain, and tracking.

The service deliberately knows nothing about a simulator.  The first RGB frame
creates a mask scene and selects source/destination once.  Later submitted RGB
frames advance every visual entity track, so graphs and arrow endpoints use
fresh mask centroids rather than copied anchor geometry. An explicit display
pair can bypass graph-brain selection for perception-only operation.
"""

from __future__ import annotations

import hashlib
import base64
import io
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .geometric_graph import (
    GEOMETRIC_GRAPH_SCHEMA,
    GEOMETRIC_RELATION_RULES_REVISION,
    GeometricRelationConfig,
    build_directed_relations,
    geometric_relation_rules,
    geometric_relation_revision,
    geometric_relation_rules_sha256,
    graph_instance,
    render_graph_overlay,
)
from .mask_scene import MaskSceneInitializer, decode_mask, encode_mask
from .contracts import GraphBrainError, GraphTriplet


class LiveResolutionError(ValueError):
    """A submitted RGB frame could not yield two current observed masks."""


def _encode_rgb_png(rgb: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        output, format="PNG", optimize=False
    )
    return base64.b64encode(output.getvalue()).decode("ascii")


@dataclass(slots=True)
class _Episode:
    episode_id: str
    instruction: str
    suite: str
    source_id: str
    destination_id: str
    source_query: str
    destination_query: str
    source_session: str
    destination_session: str
    camera_id: str
    selection: Any
    last_sequence: int
    initial_scene: dict[str, Any]
    entity_sessions: dict[str, str]
    tracking_source_id: str = ""
    tracking_destination_id: str = ""
    tracking_pair_fallback: bool = False
    tracking_pair_fallback_reason: str | None = None


class LiberoArrowResolutionService:
    """Resolve both selected endpoint masks from the exact submitted RGB frame."""

    def __init__(
        self,
        tracking_service: Any,
        segmentation_service: Any,
        graph_brain: Any | None,
        *,
        camera_id: str = "agentview",
        max_attempts: int = 3,
        attempt_timeout_s: float = 10.0,
        poll_interval_s: float = 0.01,
        geometry_rules: GeometricRelationConfig | dict[str, Any] | None = None,
        allow_partial_inventory: bool = False,
    ) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("live mask acquisition attempts must be in [1, 3]")
        self._tracker = tracking_service
        self._initializer = MaskSceneInitializer(
            segmentation_service, geometry_rules=geometry_rules
        )
        self._geometry_rules = geometry_rules
        self._allow_partial_inventory = bool(allow_partial_inventory)
        self._brain = graph_brain
        self._camera_id = camera_id
        self._max_attempts = max_attempts
        self._attempt_timeout_s = float(attempt_timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._lock = threading.RLock()
        self._episodes: dict[str, _Episode] = {}

    @staticmethod
    def _validate_rgb(rgb: np.ndarray, claimed_sha256: str) -> tuple[np.ndarray, str]:
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] != value.shape[1]:
            raise LiveResolutionError("LIBERO resolver requires square HxWx3 RGB")
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        if digest != str(claimed_sha256):
            raise LiveResolutionError("submitted RGB hash does not match decoded pixels")
        return value, digest

    @staticmethod
    def _validate_encoded_rgb(
        encoded_rgb: bytes, value: np.ndarray, digest: str
    ) -> bytes:
        """Require the detector/tracker payload to be the hashed RGB frame."""

        try:
            decoded = np.ascontiguousarray(
                np.asarray(
                    Image.open(io.BytesIO(bytes(encoded_rgb))).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        except Exception as exc:
            raise LiveResolutionError("encoded RGB payload is not a valid image") from exc
        if decoded.shape != value.shape:
            raise LiveResolutionError("encoded RGB payload shape does not match submitted RGB")
        encoded_digest = hashlib.sha256(decoded.tobytes()).hexdigest()
        if encoded_digest != digest:
            raise LiveResolutionError("encoded RGB payload does not match submitted RGB frame")
        return bytes(encoded_rgb)

    @staticmethod
    def _entity(scene: dict[str, Any], entity_id: str) -> dict[str, Any]:
        matches = [item for item in scene["instances"] if item.get("instance_id") == entity_id]
        if len(matches) != 1:
            raise LiveResolutionError(f"selected entity {entity_id!r} is absent or ambiguous")
        return matches[0]

    def _start_episode(
        self,
        *,
        episode_id: str,
        state_sequence: int,
        instruction: str,
        suite: str,
        rgb: np.ndarray,
        encoded_rgb: bytes,
        digest: str,
        selected_pair: tuple[str, str] | None = None,
    ) -> tuple[_Episode, np.ndarray, np.ndarray]:
        if self._brain is None and selected_pair is None:
            raise GraphBrainError("GRAPH_MODEL_UNAVAILABLE", "LIBERO graph brain is not configured")
        scene = self._initializer.initialize(
            rgb, instruction=instruction, suite=suite,
            visual_inventory=(suite == "object" and self._brain is None and selected_pair is not None
                              and getattr(self._tracker, "supports_unobserved_continuation", False)),
            allow_partial_inventory=self._allow_partial_inventory,
        )
        if scene.get("rgb_sha256") != digest:
            raise LiveResolutionError("mask-scene initializer did not use the submitted RGB frame")
        triplets = tuple(
            GraphTriplet(
                relation_id=str(item["relation_id"]),
                subject=str(item["subject"]),
                relation=str(item["relation"]),
                object=str(item["object"]),
            )
            for item in scene["triplets"]
        )
        if selected_pair is None:
            selection = self._brain.select_triplet(instruction=instruction, triplets=triplets)
        else:
            if len(selected_pair) != 2 or selected_pair[0] == selected_pair[1]:
                raise LiveResolutionError("explicit arrow pair must contain two distinct entity IDs")
            selection = SimpleNamespace(
                source_entity_id=selected_pair[0], destination_entity_id=selected_pair[1],
                origin="explicit_pair", model=None, response_id=None,
            )
        requested_source_id = str(selection.source_entity_id)
        requested_destination_id = str(selection.destination_entity_id)
        available_ids = sorted(str(item["instance_id"]) for item in scene["instances"])
        tracking_pair_fallback = False
        tracking_pair_fallback_reason = None
        if requested_source_id in available_ids and requested_destination_id in available_ids:
            tracking_source_id, tracking_destination_id = requested_source_id, requested_destination_id
        elif self._allow_partial_inventory and getattr(selection, "origin", None) == "explicit_pair":
            if len(available_ids) < 2:
                raise LiveResolutionError(
                    "partial public-RGB inventory requires two retained entities for tracking"
                )
            tracking_source_id, tracking_destination_id = available_ids[:2]
            tracking_pair_fallback = True
            tracking_pair_fallback_reason = "requested_pair_entity_missing_from_partial_inventory"
        else:
            # Keep the strict/default contract: an explicit or graph-selected
            # semantic endpoint must be present in the initialized scene.
            tracking_source_id, tracking_destination_id = requested_source_id, requested_destination_id
        source = self._entity(scene, tracking_source_id)
        destination = self._entity(scene, tracking_destination_id)
        source_mask = decode_mask(source["mask_png_base64"])
        destination_mask = decode_mask(destination["mask_png_base64"])
        source_session = f"libero-arrow:{episode_id}:source"
        destination_session = f"libero-arrow:{episode_id}:destination"
        # A repeated episode ID denotes a reset/restart, so remove only its two
        # prior synthetic sessions before creating the new generation.
        self._tracker.stop(source_session)
        self._tracker.stop(destination_session)
        frame_ts = float(state_sequence)
        self._tracker.start(
            env_id=source_session,
            camera_id=self._camera_id,
            query=str(source.get("prompt") or source.get("class_id") or selection.source_entity_id),
            frame_ts=frame_ts,
            jpeg=encoded_rgb,
            seed_mask=source_mask,
        )
        self._tracker.start(
            env_id=destination_session,
            camera_id=self._camera_id,
            query=str(destination.get("prompt") or destination.get("class_id") or selection.destination_entity_id),
            frame_ts=frame_ts,
            jpeg=encoded_rgb,
            seed_mask=destination_mask,
        )
        entity_sessions = {
            tracking_source_id: source_session,
            tracking_destination_id: destination_session,
        }
        for item in scene["instances"]:
            entity_id = str(item["instance_id"])
            if entity_id in entity_sessions:
                continue
            session_id = f"libero-arrow:{episode_id}:entity:{entity_id}"
            self._tracker.stop(session_id)
            self._tracker.start(
                env_id=session_id, camera_id=self._camera_id,
                query=str(item.get("prompt") or item.get("class_id") or entity_id),
                frame_ts=frame_ts, jpeg=encoded_rgb,
                seed_mask=decode_mask(item["mask_png_base64"]),
            )
            entity_sessions[entity_id] = session_id
        if hasattr(self._tracker, "bind_scene"):
            self._tracker.bind_scene([
                entity_sessions[str(item["instance_id"])] for item in scene["instances"]
            ])
        episode = _Episode(
            episode_id=episode_id,
            instruction=instruction,
            suite=suite,
            source_id=requested_source_id,
            destination_id=requested_destination_id,
            source_query=str(source.get("prompt") or source.get("class_id") or tracking_source_id),
            destination_query=str(destination.get("prompt") or destination.get("class_id") or tracking_destination_id),
            source_session=source_session,
            destination_session=destination_session,
            camera_id=self._camera_id,
            selection=selection,
            last_sequence=state_sequence,
            initial_scene=scene,
            entity_sessions=entity_sessions,
            tracking_source_id=tracking_source_id,
            tracking_destination_id=tracking_destination_id,
            tracking_pair_fallback=tracking_pair_fallback,
            tracking_pair_fallback_reason=tracking_pair_fallback_reason,
        )
        self._episodes[episode_id] = episode
        return episode, source_mask, destination_mask

    def _current_mask(self, session_id: str, frame_ts: float, *, allow_unobserved=False) -> tuple[np.ndarray | None, dict[str, Any]]:
        last: dict[str, Any] = {}
        for _attempt in range(self._max_attempts):
            deadline = time.monotonic() + self._attempt_timeout_s
            while time.monotonic() < deadline:
                snapshot = self._tracker.snapshot(session_id, camera_id=self._camera_id)
                last = snapshot
                error_source = getattr(self._tracker, "error_observation_source", "sam2_error")
                current_source = getattr(
                    self._tracker, "current_observation_source", "sam2_video_current"
                )
                if snapshot.get("status") == "error" or snapshot.get("observation_source") == error_source:
                    raise LiveResolutionError(
                        f"visual track failed: {snapshot.get('error', 'unknown error')}"
                    )
                if (
                    allow_unobserved and snapshot.get("status") == "unobserved"
                    and snapshot.get("source_frame_ts") == frame_ts
                    and snapshot.get("mask_present") is False
                    and snapshot.get("observation_source") == current_source
                    and snapshot.get("native_evidence", {}).get("identity_in_native_state") is True
                ):
                    return None, snapshot
                if (
                    snapshot.get("source_frame_ts") == frame_ts
                    and snapshot.get("mask_present") is True
                    and snapshot.get("observation_source") == current_source
                ):
                    mask = self._tracker.latest_mask(session_id, camera_id=self._camera_id)
                    if mask is not None and np.asarray(mask, dtype=bool).any():
                        return np.ascontiguousarray(mask, dtype=bool), snapshot
                time.sleep(self._poll_interval_s)
        raise LiveResolutionError(
            "visual tracker did not produce a current observed mask after "
            f"{self._max_attempts} attempts; last snapshot={last}"
        )

    @staticmethod
    def _live_scene(
        episode: _Episode,
        *,
        source_mask: np.ndarray | None,
        destination_mask: np.ndarray | None,
        state_sequence: int,
        rgb_sha256: str,
        source_snapshot: dict[str, Any],
        destination_snapshot: dict[str, Any],
        observations: dict[str, tuple[np.ndarray | None, dict[str, Any]]],
        geometry_rules: GeometricRelationConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rebuild the complete graph from the current visual frame.

        The graph brain is intentionally *not* called here.  It selected the
        stable semantic IDs from the initial SAM3.1 graph; every later frame
        mechanically recomputes geometry and relations using the same pure
        rules. Temporarily unobserved native identities remain in the scene,
        with null geometry and explicitly unknown relations. Initial masks
        are never relabelled as current geometry. Benchmark/brain mode still
        requires observed masks; only the explicit perception path accepts gaps.
        """

        initial = episode.initial_scene
        tracking_source_id = episode.tracking_source_id or episode.source_id
        tracking_destination_id = episode.tracking_destination_id or episode.destination_id
        instances = list(initial["instances"])
        masks = {
            str(item["instance_id"]): observations[str(item["instance_id"])][0]
            for item in instances if observations[str(item["instance_id"])][0] is not None
        }
        shape = (int(initial["height"]), int(initial["width"]))
        if any(mask.shape != shape for mask in masks.values()):
            raise LiveResolutionError("live graph masks do not share one native RGB resolution")
        entries = list(masks.items())
        for index, (left_id, left) in enumerate(entries):
            for right_id, right in entries[index + 1:]:
                union = np.logical_or(left, right).sum()
                if union and np.logical_and(left, right).sum() / union >= 0.9:
                    raise LiveResolutionError(f"duplicate live identities: {left_id}, {right_id}")

        live_instances: list[dict[str, Any]] = []
        for template in instances:
            instance_id = str(template["instance_id"])
            snapshot = observations[instance_id][1]
            if snapshot.get("source_frame_ts") != float(state_sequence):
                raise LiveResolutionError(f"stale graph entity: {instance_id}")
            observed = instance_id in masks
            state = "observed" if observed else "unobserved"
            evidence = str(snapshot["observation_source"])
            if observed:
                item = graph_instance(template, masks[instance_id], track_state=state,
                                      tracking_evidence=evidence)
                item["mask_sha256"] = hashlib.sha256(masks[instance_id].tobytes()).hexdigest()
                item["mask_png_base64"] = encode_mask(masks[instance_id])
            else:
                item = {**template, "area_px": 0, "center_xy": None, "bbox_xyxy": None,
                        "mask_sha256": None, "mask_png_base64": None, "track_state": state,
                        "tracking_evidence": evidence}
            item["detected_this_frame"] = snapshot.get("tracking_method") == "seed_mask"
            item["tracking_method"] = snapshot.get("tracking_method")
            item["current_geometry_valid"] = observed
            item["source_frame_ts"] = snapshot.get("source_frame_ts")
            for key in ("native_session_id", "object_id", "native_frame_index", "native_evidence",
                        "initialization_prompts", "subsequent_prompts", "future_frames_consumed",
                        "visibility_reason"):
                item[key] = snapshot.get(key)
            live_instances.append(item)

        visible = [item for item in live_instances if item["current_geometry_valid"]]
        if len(visible) >= 2:
            _relations, triplets, evidence = build_directed_relations(
                visible, masks, suite=episode.suite, instruction=episode.instruction, shape=shape,
                geometry_rules=geometry_rules,
                allow_partial_visibility=True,
            )
        else:
            triplets, evidence = [], []
        by_pair = {(item["subject"], item["object"]): item for item in triplets}
        triplets = []
        for subject in live_instances:
            for destination in live_instances:
                a, b = subject["instance_id"], destination["instance_id"]
                if a == b:
                    continue
                relation = by_pair.get((a, b))
                triplets.append({
                    **(relation or {"subject": a, "object": b, "relation": "unknown"}),
                    "relation_id": f"r{len(triplets):03d}",
                    "current_geometry_valid": relation is not None,
                    "unavailable_reason": None if relation else "endpoint_not_visible_in_native_output",
                })
        rules = geometric_relation_rules(geometry_rules)
        return {
            "schema": "samgraph.mask_scene.v1",
            "graph_schema": GEOMETRIC_GRAPH_SCHEMA,
            "width": int(shape[1]),
            "height": int(shape[0]),
            "pixel_frame": initial.get("pixel_frame", "agentview_raw_rgb_top_left_xyxy"),
            "rgb_sha256": rgb_sha256,
            "episode_id": episode.episode_id,
            "state_sequence": int(state_sequence),
            "instances": live_instances,
            "triplets": triplets,
            "visibility_schema": "samgraph.scene_visibility.v1",
            "all_entity_current_geometry_valid": len(visible) == len(live_instances),
            "unobserved_entity_ids": [item["instance_id"] for item in live_instances if not item["current_geometry_valid"]],
            "unknown_relation_count": sum(not item["current_geometry_valid"] for item in triplets),
            "rules": rules,
            "rules_sha256": geometric_relation_rules_sha256(geometry_rules),
            "relation_revision": geometric_relation_revision(geometry_rules),
            "relation_evidence": evidence,
            "target_ids": [episode.source_id, episode.destination_id],
            "selected_pair": [episode.source_id, episode.destination_id],
            "tracking_pair": {
                "requested": [episode.source_id, episode.destination_id],
                "effective": [tracking_source_id, tracking_destination_id],
                "fallback": episode.tracking_pair_fallback,
                "fallback_reason": episode.tracking_pair_fallback_reason,
                "diagnostic_only": True,
            },
            "graph_brain_selection_frozen": getattr(episode.selection, "origin", None) != "explicit_pair",
            "selection_origin": getattr(episode.selection, "origin", "graph_brain"),
            "inventory_mode": initial.get("inventory_mode", "semantic_catalog"),
            "allow_partial_inventory": initial.get("allow_partial_inventory", False),
            "inventory_completeness": initial.get("inventory_completeness", {}),
            "inventory_coverage": initial.get("inventory_coverage", {}),
            "missing_class_ids": initial.get("missing_class_ids", []),
            "semantic_task_identity_established": initial.get("semantic_task_identity_established", True),
            "simulator_geometry_consumed": False,
            "frame_provenance": {
                "camera_id": str(getattr(episode, "camera_id", "agentview")),
                "state_sequence": int(state_sequence),
                "rgb_sha256": rgb_sha256,
                "source_frame_ts": {
                    "source": source_snapshot.get("source_frame_ts"),
                    "destination": destination_snapshot.get("source_frame_ts"),
                },
            },
        }

    def resolve(
        self,
        *,
        episode_id: str,
        state_sequence: int,
        instruction: str,
        suite: str,
        rgb: np.ndarray,
        encoded_rgb: bytes,
        rgb_sha256: str,
        selected_pair: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        cleaned_episode = str(episode_id).strip()
        cleaned_instruction = " ".join(str(instruction).split())
        if not cleaned_episode or not cleaned_instruction:
            raise LiveResolutionError("episode_id and instruction are required")
        if suite not in {"spatial", "object"}:
            raise LiveResolutionError("suite must be spatial or object")
        try:
            sequence = int(state_sequence)
        except (TypeError, ValueError) as exc:
            raise LiveResolutionError("state_sequence must be an integer") from exc
        value, digest = self._validate_rgb(rgb, rgb_sha256)
        encoded = self._validate_encoded_rgb(encoded_rgb, value, digest)
        with self._lock:
            episode = self._episodes.get(cleaned_episode)
            if episode is None:
                episode, source_mask, destination_mask = self._start_episode(
                    episode_id=cleaned_episode,
                    state_sequence=sequence,
                    instruction=cleaned_instruction,
                    suite=suite,
                    rgb=value,
                    encoded_rgb=encoded,
                    digest=digest,
                    selected_pair=selected_pair,
                )
                initial_snapshot = {
                    "status": "observed",
                    "observation_source": episode.initial_scene["initialization_provider"]
                    + "_initial_current",
                    "source_frame_ts": float(sequence),
                    "tracking_method": "seed_mask",
                    "reacquisition_attempts": 0,
                }
                source_snapshot = dict(initial_snapshot)
                destination_snapshot = dict(initial_snapshot)
                observations = {
                    str(item["instance_id"]): (
                        decode_mask(item["mask_png_base64"]), {
                            **self._tracker.snapshot(episode.entity_sessions[str(item["instance_id"])], camera_id=self._camera_id),
                            **initial_snapshot,
                        }
                    ) for item in episode.initial_scene["instances"]
                }
                source_snapshot = observations[episode.tracking_source_id][1]
                destination_snapshot = observations[episode.tracking_destination_id][1]
            else:
                if (episode.instruction, episode.suite) != (cleaned_instruction, suite):
                    raise LiveResolutionError("episode instruction or suite changed without a reset")
                if selected_pair is not None and tuple(selected_pair) != (episode.source_id, episode.destination_id):
                    raise LiveResolutionError("explicit arrow pair changed without a reset")
                if sequence <= episode.last_sequence:
                    raise LiveResolutionError("state_sequence must increase for every submitted frame")
                frame_ts = float(sequence)
                observations = {}
                for entity_id, session_id in episode.entity_sessions.items():
                    self._tracker.submit_frame(
                        env_id=session_id, camera_id=self._camera_id,
                        frame_ts=frame_ts, jpeg=encoded,
                    )
                    observations[entity_id] = self._current_mask(
                        session_id, frame_ts,
                        allow_unobserved=(getattr(episode.selection, "origin", None) == "explicit_pair"
                                          and getattr(self._tracker, "supports_unobserved_continuation", False)),
                    )
                source_mask, source_snapshot = observations[episode.tracking_source_id]
                destination_mask, destination_snapshot = observations[episode.tracking_destination_id]
                episode.last_sequence = sequence
            selection = episode.selection
            live_graph = self._live_scene(
                episode,
                source_mask=source_mask,
                destination_mask=destination_mask,
                state_sequence=sequence,
                rgb_sha256=digest,
                source_snapshot=source_snapshot,
                destination_snapshot=destination_snapshot,
                observations=observations,
                geometry_rules=self._geometry_rules,
            )
            graph_overlay = render_graph_overlay(
                value,
                live_graph,
                selected_pair=(episode.tracking_source_id, episode.tracking_destination_id),
            )
            return {
                "schema": "samgraph.libero_arrow_resolution.v1",
                "provider": (
                    episode.initial_scene["initialization_provider"]
                    + "_" + getattr(selection, "origin", "graph_brain") + "_"
                    + str(getattr(self._tracker, "provider", "dual_sam2"))
                ),
                "rgb_sha256": digest,
                "episode_id": cleaned_episode,
                "state_sequence": sequence,
                "selection": {
                    "source_entity_id": episode.source_id,
                    "destination_entity_id": episode.destination_id,
                    "relation_id": getattr(selection, "relation_id", None),
                },
                "tracking_pair": {
                    "requested": [episode.source_id, episode.destination_id],
                    "effective": [episode.tracking_source_id, episode.tracking_destination_id],
                    "fallback": episode.tracking_pair_fallback,
                    "fallback_reason": episode.tracking_pair_fallback_reason,
                    "diagnostic_only": True,
                },
                "inventory": {
                    "allow_partial_inventory": episode.initial_scene.get("allow_partial_inventory", False),
                    "completeness": episode.initial_scene.get("inventory_completeness", {}),
                    "coverage": episode.initial_scene.get("inventory_coverage", {}),
                    "missing_class_ids": episode.initial_scene.get("missing_class_ids", []),
                },
                "arrow_available": source_mask is not None and destination_mask is not None,
                "instances": [
                    {
                        "instance_id": entity_id,
                        "mask_png_base64": encode_mask(mask) if mask is not None else None,
                        "current_geometry_valid": mask is not None,
                        "track_state": "observed" if mask is not None else "unobserved",
                        "observation_source": snapshot.get("observation_source"),
                        "source_frame_ts": snapshot.get("source_frame_ts"),
                        "tracking_method": snapshot.get("tracking_method"),
                        "reacquisition_attempts": snapshot.get("reacquisition_attempts", 0),
                        "reacquisition_iou": snapshot.get("reacquisition_iou"),
                    } for entity_id, mask, snapshot in (
                        (episode.tracking_source_id, source_mask, source_snapshot),
                        (episode.tracking_destination_id, destination_mask, destination_snapshot),
                    )
                ],
                "graph": live_graph,
                "graph_overlay_png_base64": _encode_rgb_png(graph_overlay),
                "provenance": {
                    "production_inputs": (
                        list(episode.initial_scene["production_inputs"])
                        + ["graph_triplets"]
                        + list(getattr(self._tracker, "production_inputs", ("sam2_live_masks",)))
                    ),
                    "initial_scene": episode.initial_scene,
                    "graph_triplets": list(episode.initial_scene.get("triplets", [])),
                    "live_graph_triplets": list(live_graph["triplets"]),
                    "live_graph_rules_revision": live_graph["relation_revision"],
                    "live_graph_frame": dict(live_graph["frame_provenance"]),
                    "graph_overlay_authoritative": False,
                    "graph_response_id": getattr(selection, "response_id", None),
                    "graph_brain_called_once": getattr(selection, "origin", None) != "explicit_pair",
                    "selection_origin": getattr(selection, "origin", "graph_brain"),
                    "graph_model": getattr(selection, "model", None),
                    "graph_prompt_revision": getattr(selection, "prompt_revision", None),
                    "source_and_destination_refreshed": source_mask is not None and destination_mask is not None,
                    "endpoint_observations": {
                        "source": dict(source_snapshot),
                        "destination": dict(destination_snapshot),
                    },
                    "simulator_semantic_endpoints_consumed": False,
                },
            }

    def close(self) -> None:
        with self._lock:
            episodes = tuple(self._episodes.values())
            self._episodes.clear()
        for episode in episodes:
            for session_id in episode.entity_sessions.values():
                self._tracker.stop(session_id)


__all__ = ["LiberoArrowResolutionService", "LiveResolutionError"]
