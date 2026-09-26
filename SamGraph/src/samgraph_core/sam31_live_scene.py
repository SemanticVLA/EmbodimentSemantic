"""Causal append bridge for the pinned SAM3.1 joint-object native tracker.

No future images or new point prompts are supplied after initialization. Native
conditioning and temporal memory survive each append. The fixed-resource public
API needs this explicitly version-pinned bridge; it is not an upstream API.
"""
from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .local_sam31 import (
    LocalSam31ReplayTracker, _decode_rgb, _mask_centroid_point, _normalise_masks,
)


@dataclass
class _Scene:
    session_id: str
    members: list[str]
    frame_ts: float
    digest: str
    failed: bool = False
    seed_conditioning: str = "centroid_point"
    seed_mask_sha256: dict[int, str] | None = None
    object_ids: dict[str, int] | None = None
    max_object_id: int = 0


_FULL_MASK_STAGE_SENTINEL = "_THIS_FRAME_HAS_OUTPUTS_"


def _mask_foreground_point(mask: np.ndarray) -> tuple[float, float]:
    """Return a deterministic *foreground* point, including for hollow masks."""
    arr = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        raise ValueError("cannot seed an empty mask")
    cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))
    if not (0 <= cy < arr.shape[0] and 0 <= cx < arr.shape[1] and arr[cy, cx]):
        distance = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest = np.lexsort((xs, ys, distance))[0]
        cx, cy = int(xs[nearest]), int(ys[nearest])
    return (float(cx) / float(arr.shape[1]), float(cy) / float(arr.shape[0]))


def _mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_is_nonempty(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_is_nonempty(v) for v in value)
    try:
        return bool(value.numel()) if hasattr(value, "numel") else bool(np.asarray(value).size)
    except Exception:
        return True


def _contains_exact_mask(value: Any, target: np.ndarray, key: str = "") -> bool:
    """Find a native mask record equal to the supplied bool seed."""
    if isinstance(value, dict):
        for name, child in value.items():
            child_key = f"{key}.{name}" if key else str(name).lower()
            if _contains_exact_mask(child, target, child_key.lower()):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_mask(child, target, key) for child in value)
    if "mask" not in key:
        return False
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        candidate = np.asarray(value)
        # Pinned SAM stores the singleton replacement mask as [N,C,H,W].
        # Accept only the exact documented singleton form (or the direct H,W
        # form used by small fake states); do not flatten arbitrary tensors.
        if candidate.shape == target.shape:
            normalized = candidate
        elif candidate.shape == (1, 1, *target.shape):
            normalized = candidate[0, 0]
        else:
            return False
        return np.array_equal(normalized.astype(bool), target)
    except Exception:
        return False


def _contains_point_input(value: Any, key: str = "") -> bool:
    if isinstance(value, dict):
        return any(_contains_point_input(
            child, (f"{key}.{name}" if key else str(name)).lower())
                   for name, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_point_input(child, key) for child in value)
    if "point" not in key:
        return False
    return _is_nonempty(value)


def _contains_memory_state(value: Any, key: str = "") -> bool:
    if isinstance(value, dict):
        for name, child in value.items():
            lower = str(name).lower()
            if ("maskmem_features" in lower or "maskmem_pos_enc" in lower) and _is_nonempty(child):
                return True
            if _contains_memory_state(child, lower):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_memory_state(child, key) for child in value)
    return False


def _assert_native_object_ids(state: dict, object_ids: list[int]) -> None:
    """Require every registered ID to remain present in native singleton state."""
    observed: set[int] = set()
    native_states = state.get("sam2_inference_states", [])
    if isinstance(native_states, dict):
        native_states = list(native_states.values())
    for native in native_states:
        if isinstance(native, dict) and isinstance(native.get("obj_id_to_idx"), dict):
            observed.update(int(oid) for oid in native["obj_id_to_idx"])
    if isinstance(state.get("obj_id_to_idx"), dict):
        observed.update(int(oid) for oid in state["obj_id_to_idx"])
    if not observed or not set(object_ids).issubset(observed):
        raise RuntimeError(f"native object IDs are not stable: expected={object_ids}, observed={sorted(observed)}")


def _private_singleton_state(owner: Any, state: Any, object_id: int) -> Any:
    getter = getattr(owner, "_get_sam2_inference_states_by_obj_ids", None)
    if not callable(getter):
        raise RuntimeError("pinned SAM multiplex state getter is unavailable")
    signature = inspect.signature(getter)
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        names = list(signature.parameters)
        if len(names) < 2:
            raise RuntimeError("unexpected singleton state getter signature")
    result = getter(state, [object_id])
    if isinstance(result, dict):
        if object_id in result:
            return result[object_id]
        if str(object_id) in result:
            return result[str(object_id)]
        if len(result) == 1:
            return next(iter(result.values()))
    if isinstance(result, (list, tuple)) and len(result) == 1:
        return result[0]
    return result


def _full_mask_tensor(mask: np.ndarray, state: dict) -> Any:
    import torch
    device = state.get("device")
    kwargs = {"dtype": torch.bool}
    if device is not None:
        kwargs["device"] = device
    return torch.as_tensor(np.ascontiguousarray(mask, dtype=np.bool_), **kwargs)


def _append_frame(predictor: Any, session_id: str, image: Any) -> int:
    """Normalize only the incoming image with the official loader, then append."""
    import torch
    from sam3.model.io_utils import load_resource_as_video_frames

    state = predictor._all_inference_states[session_id]["state"]
    index = state["num_frames"]
    # Bound retained image/memory growth explicitly until longer runs qualify.
    if index >= 256:
        raise RuntimeError("SAM3.1 live session reached its 256-frame memory limit; reset required")
    model = getattr(predictor.model, "model", predictor.model)
    # Only normalize the newly delivered pixels. No new predictor session,
    # capture, detector call, object prompt, or temporal state initialization.
    with torch.inference_mode():
        images, height, width = load_resource_as_video_frames(
            resource_path=[image], image_size=model.image_size,
            offload_video_to_cpu=False, img_mean=model.image_mean,
            img_std=model.image_std, async_loading_frames=False,
            video_loader_type="cv2",
        )
        if (state["orig_height"], state["orig_width"]) != (height, width):
            raise ValueError("live RGB dimensions changed")
        incoming = {"device": state["device"], "constants": {}}
        model._construct_initial_input_batch(incoming, images)
        batch, extra = state["input_batch"], incoming["input_batch"]
        batch.img_batch.tensors = torch.cat((batch.img_batch.tensors, extra.img_batch.tensors), dim=0)
        stage = extra.find_inputs[0]
        stage.img_ids = torch.full_like(stage.img_ids, index)
        stage.img_ids_np[:] = index
        batch.find_inputs.append(stage)
        batch.find_targets.append(None)
        batch.find_metadatas.append(None)
        for key in ("previous_stages_out", "per_frame_raw_point_input", "per_frame_raw_box_input",
                    "per_frame_visual_prompt", "per_frame_geometric_prompt"):
            state[key].append(None)
        state["per_frame_cur_step"].append(0)
        state["num_frames"] = index + 1
        for tracker_state in state["sam2_inference_states"]:
            tracker_state["num_frames"] = index + 1
        return index


def native_frame_evidence(state: dict, index: int) -> dict[int, dict]:
    """Read native current-frame scores/memories without changing inference.

    The pinned SAM3.1 implementation calls its internal states ``sam2``. They
    belong to the same SAM3.1 checkpoint, not a separate SAM2 fallback.
    """
    evidence = {}
    for native in state["sam2_inference_states"]:
        outputs = native["output_dict"]
        current = outputs["non_cond_frame_outputs"].get(index)
        if current is None:
            current = outputs["cond_frame_outputs"].get(index)
        for oid, position in native["obj_id_to_idx"].items():
            item = {
                "identity_in_native_state": True,
                "conditioning_frames": sorted(outputs["cond_frame_outputs"]),
                "memory_frames": sorted(outputs["non_cond_frame_outputs"]),
            }
            if current is not None:
                position = current.get("local_obj_id_to_idx", {}).get(oid, position)
                for key in ("object_score_logits", "pred_masks"):
                    value = current.get(key)
                    if value is not None and position < len(value):
                        row = value[position].detach().float()
                        item[key + "_min"] = float(row.min().cpu())
                        item[key + "_max"] = float(row.max().cpu())
                        if key == "pred_masks":
                            item["native_positive_mask_pixels"] = int((row > 0).sum().cpu())
            evidence[int(oid)] = item
    return evidence


class LocalSam31SceneTracker(LocalSam31ReplayTracker):
    """One persistent native session per scene; all entities advance together."""

    provider = "local_sam3_1_multiplex_video_persistent_scene"
    production_inputs = ("sam3_1_video_current_masks", "initial_public_rgb_mask_seeds")
    supports_unobserved_continuation = True

    def __init__(self, runtime, **kwargs):
        super().__init__(runtime, **kwargs)
        self._scenes: dict[str, _Scene] = {}
        self._dispatch_patched = False

    def _condition_full_mask(self, predictor: Any, model: Any, state: dict,
                             sessions: list[Any], object_ids: list[int],
                             frame_index: int = 0) -> dict[int, str]:
        """Replace each registration point with its exact native full-mask seed.

        This deliberately uses the pinned multiplex private APIs only after the
        public prompt has registered stable object IDs. Any contract mismatch
        raises and the caller closes the whole session; partial conditioning is
        never exposed as a usable scene.
        """
        tracker = getattr(model, "tracker", None)
        # SAM3.1 multiplex exposes the batched add_new_masks API on its
        # multiplex tracker wrapper.  The singular add_new_mask API belongs
        # to the non-multiplex SAM2 tracker and is not present here.
        add_masks = getattr(tracker, "add_new_masks", None)
        preflight = getattr(tracker, "propagate_in_video_preflight", None)
        if not callable(add_masks) or not callable(preflight):
            raise RuntimeError("pinned native full-mask tracker methods are unavailable")
        add_signature = inspect.signature(add_masks)
        required_add_parameters = {"obj_ids", "masks", "reconditioning"}
        add_accepts_kwargs = any(
            p.kind == p.VAR_KEYWORD for p in add_signature.parameters.values()
        )
        if not required_add_parameters.issubset(add_signature.parameters) or (
                "add_mask_to_memory" not in add_signature.parameters and not add_accepts_kwargs):
            raise RuntimeError("native multiplex add_new_masks contract is unavailable")
        pre_signature = inspect.signature(preflight)
        if "run_mem_encoder" not in pre_signature.parameters and not any(
                p.kind == p.VAR_KEYWORD for p in pre_signature.parameters.values()):
            raise RuntimeError("native preflight lacks run_mem_encoder contract")

        # The getter is pinned on the multiplex model in the qualified source;
        # accepting predictor as a fallback makes the adapter testable without
        # weakening the runtime contract.
        getter_owner = next((owner for owner in (model, predictor)
                             if callable(getattr(owner, "_get_sam2_inference_states_by_obj_ids", None))), None)
        if getter_owner is None:
            raise RuntimeError("pinned SAM multiplex state getter is unavailable")

        masks: list[np.ndarray] = []
        for oid, session in zip(object_ids, sessions):
            mask = np.asarray(session.last_mask, dtype=bool)
            if mask.ndim != 2 or mask.shape != (int(state["orig_height"]), int(state["orig_width"])):
                raise ValueError(f"full-mask seed shape mismatch for object {oid}: {mask.shape}")
            if not mask.any():
                raise ValueError(f"full-mask seed is empty for object {oid}")
            masks.append(mask)

        # The tracker API consumes a native tracker bucket, not the outer
        # detector/scene state. Objects can share a bucket, so recondition
        # every requested object in that bucket together before preflight.
        import torch

        buckets: dict[int, tuple[dict, list[int], list[np.ndarray]]] = {}
        for oid, mask in zip(object_ids, masks):
            native = _private_singleton_state(getter_owner, state, oid)
            if not isinstance(native, dict) or oid not in native.get("obj_id_to_idx", {}):
                raise RuntimeError(f"native tracker bucket is missing object {oid}")
            bucket = buckets.setdefault(id(native), (native, [], []))
            bucket[1].append(oid)
            bucket[2].append(mask)
        for native, native_ids, native_masks in buckets.values():
            mask_batch = torch.stack(
                [_full_mask_tensor(mask, native) for mask in native_masks], dim=0
            )
            add_masks(
                native, frame_index, native_ids, mask_batch,
                add_mask_to_memory=False, reconditioning=True,
            )
            preflight(native, run_mem_encoder=True)

        digests: dict[int, str] = {}
        for oid, mask in zip(object_ids, masks):
            singleton = _private_singleton_state(getter_owner, state, oid)
            if _contains_point_input(singleton):
                raise RuntimeError(f"native point conditioning was not cleared for object {oid}")
            if not _contains_exact_mask(singleton, mask):
                raise RuntimeError(f"native full-mask state mismatch for object {oid}")
            if not _contains_memory_state(singleton):
                raise RuntimeError(f"native memory encoder produced no observable state for object {oid}")
            digests[oid] = _mask_digest(mask)

        _assert_native_object_ids(state, object_ids)

        cache_owner = next((owner for owner in (model, predictor)
                            if callable(getattr(owner, "_cache_frame_outputs", None))), None)
        if cache_owner is None:
            raise RuntimeError("pinned outer frame cache method is unavailable")
        cache = cache_owner._cache_frame_outputs
        # The cache method replaces the whole frame entry. Keep unaffected
        # current-frame objects when inserting a late or replacement track.
        cached_masks = {
            int(oid): value for oid, value in
            state.get("cached_frame_outputs", {}).get(frame_index, {}).items()
        }
        for oid, session in zip(object_ids, sessions):
            cached_masks[oid] = _full_mask_tensor(
                np.asarray(session.last_mask, dtype=bool), state).unsqueeze(0)
        cache(state, frame_index, cached_masks)
        previous = state.get("previous_stages_out")
        if not isinstance(previous, list) or len(previous) <= frame_index:
            raise RuntimeError("pinned outer previous_stages_out state is unavailable")
        previous[frame_index] = _FULL_MASK_STAGE_SENTINEL
        return digests

    def bind_scene(self, members: list[str], *, seed_conditioning: str = "centroid_point") -> None:
        if seed_conditioning not in {"centroid_point", "full_mask"}:
            raise ValueError(f"unsupported seed_conditioning={seed_conditioning!r}")
        with self._lock, self._runtime._lock, self._runtime._inference_context():
            predictor = self._runtime._require_predictor()
            model = getattr(predictor.model, "model", predictor.model)
            if not self._dispatch_patched:
                original = model.parse_action_history_for_propagation

                def dispatch(state):
                    if "samgraph_live_object_ids" in state:
                        return "propagation_partial", list(state["samgraph_live_object_ids"])
                    return original(state)

                model.parse_action_history_for_propagation = dispatch
                self._dispatch_patched = True
            sessions = [self._sessions[(member, self.camera_id)] for member in members]
            first = sessions[0]
            if any(s.previous_rgb != first.previous_rgb or s.frame_ts != first.frame_ts for s in sessions):
                raise ValueError("joint tracking seeds must share one RGB frame")
            if seed_conditioning == "full_mask":
                expected_shape = _decode_rgb(first.previous_rgb).size[::-1]
                masks = [np.asarray(session.last_mask, dtype=bool) for session in sessions]
                if any(mask.ndim != 2 or mask.shape != expected_shape or not mask.any() for mask in masks):
                    raise ValueError("full-mask seeds must be nonempty and match the RGB dimensions")
                for index, mask in enumerate(masks):
                    if any(np.logical_and(mask, other).any() for other in masks[index + 1:]):
                        raise ValueError("full-mask seeds overlap; refusing ambiguous native identities")
            sid = predictor.handle_request({"type": "start_session", "resource_path": [_decode_rgb(first.previous_rgb)]})["session_id"]
            try:
                for oid, session in enumerate(sessions, 1):
                    x, y = (_mask_foreground_point(session.last_mask)
                            if seed_conditioning == "full_mask"
                            else _mask_centroid_point(session.last_mask))
                    predictor.handle_request({
                        "type": "add_prompt", "session_id": sid, "frame_index": 0,
                        "obj_id": oid, "points": [[x, y]], "point_labels": [1],
                        "rel_coordinates": True, "output_prob_thresh": 0.0,
                    })
                state = predictor._all_inference_states[sid]["state"]
                seed_digests = None
                if seed_conditioning == "full_mask":
                    seed_digests = self._condition_full_mask(
                        predictor, model, state, sessions, list(range(1, len(members) + 1)))
                state["samgraph_live_object_ids"] = list(range(1, len(members) + 1))
                scene = _Scene(sid, list(members), first.frame_ts, hashlib.sha256(first.previous_rgb).hexdigest(),
                               seed_conditioning=seed_conditioning, seed_mask_sha256=seed_digests,
                               object_ids={member: oid for oid, member in enumerate(members, 1)},
                               max_object_id=len(members))
                for member in members:
                    self._scenes[member] = scene
                for oid, session in enumerate(sessions, 1):
                    session.snapshot.update(
                        native_session_id=sid, object_id=oid, native_frame_index=0,
                        tracking_mode="persistent_scene_append", future_frames_consumed=False,
                        initialization_prompts=len(members), subsequent_prompts=0,
                        seed_conditioning=seed_conditioning,
                        seed_mask_sha256=(seed_digests or {}).get(oid),
                        seed_rgb_sha256=hashlib.sha256(first.previous_rgb).hexdigest(),
                    )
            except Exception:
                self._runtime._close_session(predictor, sid)
                raise

    def add_current_object(self, *, scene_member: str, env_id: str,
                           frame_ts: float, jpeg: bytes, seed_mask: np.ndarray,
                           query: str) -> None:
        """Register one automatically acquired current-frame full mask."""
        with self._lock, self._runtime._lock, self._runtime._inference_context():
            scene = self._scenes[str(scene_member)]
            if scene.failed or scene.seed_conditioning != "full_mask":
                raise RuntimeError("current-frame insertion requires a healthy full-mask scene")
            if float(frame_ts) != scene.frame_ts or hashlib.sha256(jpeg).hexdigest() != scene.digest:
                raise ValueError("current-frame insertion RGB does not match the tracked frame")
            if str(env_id) in self._scenes:
                raise ValueError("native scene member already exists")
            image = _decode_rgb(jpeg)
            mask = np.ascontiguousarray(np.asarray(seed_mask, dtype=bool))
            if mask.shape != (image.height, image.width) or not mask.any():
                raise ValueError("new full mask must be nonempty and match current RGB")
            predictor = self._runtime._require_predictor()
            model = getattr(predictor.model, "model", predictor.model)
            state = predictor._all_inference_states[scene.session_id]["state"]
            frame_index = int(state["num_frames"]) - 1
            scene.max_object_id += 1
            oid = scene.max_object_id
            try:
                x, y = _mask_foreground_point(mask)
                predictor.handle_request({
                    "type": "add_prompt", "session_id": scene.session_id,
                    "frame_index": frame_index, "obj_id": oid,
                    "points": [[x, y]], "point_labels": [1],
                    "rel_coordinates": True, "output_prob_thresh": 0.0,
                })
                temporary = SimpleNamespace(last_mask=mask)
                digests = self._condition_full_mask(
                    predictor, model, state, [temporary], [oid],
                    frame_index=frame_index,
                )
                self.start(
                    env_id=str(env_id), camera_id=self.camera_id, query=query,
                    frame_ts=frame_ts, jpeg=jpeg, seed_mask=mask,
                )
                session = self._sessions[(str(env_id), self.camera_id)]
                session.snapshot.update(
                    native_session_id=scene.session_id, object_id=oid,
                    native_frame_index=frame_index, tracking_mode="persistent_scene_append",
                    tracking_method="automatic_full_mask_reacquisition",
                    seed_conditioning="full_mask", seed_mask_sha256=digests[oid],
                    future_frames_consumed=False, subsequent_prompts=1,
                )
                scene.members.append(str(env_id))
                scene.object_ids[str(env_id)] = oid
                state["samgraph_live_object_ids"] = list(scene.object_ids.values())
                self._scenes[str(env_id)] = scene
            except Exception:
                scene.failed = True
                try:
                    predictor.remove_object(
                        scene.session_id, frame_idx=frame_index,
                        obj_id=oid, is_user_action=False,
                    )
                finally:
                    self._sessions.pop((str(env_id), self.camera_id), None)
                raise

    def remove_object(self, env_id: str, *, camera_id: str) -> None:
        """Quarantine one native generation after a rejected observation."""
        with self._lock, self._runtime._lock, self._runtime._inference_context():
            member = str(env_id)
            scene = self._scenes[member]
            if camera_id != self.camera_id:
                raise ValueError("invalid native scene camera")
            if len(scene.members) == 1:
                self.stop(member, camera_id=camera_id)
                return
            predictor = self._runtime._require_predictor()
            state = predictor._all_inference_states[scene.session_id]["state"]
            oid = scene.object_ids[member]
            try:
                predictor.remove_object(
                    scene.session_id, frame_idx=int(state["num_frames"]) - 1,
                    obj_id=oid, is_user_action=False,
                )
                scene.members.remove(member)
                del scene.object_ids[member]
                state["samgraph_live_object_ids"] = list(scene.object_ids.values())
                self._scenes.pop(member, None)
                self._sessions.pop((member, camera_id), None)
            except Exception:
                scene.failed = True
                raise

    def submit_frame(self, *, env_id, camera_id, frame_ts, jpeg):
        with self._lock, self._runtime._lock, self._runtime._inference_context():
            scene = self._scenes[str(env_id)]
            if camera_id != self.camera_id or scene.failed:
                raise RuntimeError("invalid camera or failed native scene; reset required")
            digest = hashlib.sha256(jpeg).hexdigest()
            if frame_ts == scene.frame_ts and digest == scene.digest:
                return True  # Other members already advanced atomically on this frame.
            if frame_ts <= scene.frame_ts:
                raise ValueError("live scene timestamp must increase")
            started = time.perf_counter()
            predictor = self._runtime._require_predictor()
            try:
                image = _decode_rgb(jpeg)
                index = _append_frame(predictor, scene.session_id, image)
                latest = None
                for result in predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": scene.session_id,
                    "propagation_direction": "forward", "start_frame_index": index,
                    "max_frame_num_to_track": 1, "output_prob_thresh": 0.0,
                }):
                    if int(result["frame_index"]) != index:
                        raise RuntimeError("native tracker returned a non-current frame")
                    latest = result
                if latest is None:
                    raise RuntimeError("native tracker returned no current frame")
                masks, ids, _ = _normalise_masks(latest["outputs"], shape=(image.height, image.width))
                expected = set((scene.object_ids or {}).values())
                if len(ids) != len(set(ids.tolist())) or not set(ids.tolist()) <= expected:
                    raise RuntimeError(f"native tracker returned duplicate or unknown IDs: {ids.tolist()}")
                current = {int(oid): mask for oid, mask in zip(ids, masks) if mask.any()}
                state = predictor._all_inference_states[scene.session_id]["state"]
                evidence = native_frame_evidence(state, index)
                if set(evidence) != expected:
                    raise RuntimeError("native temporal state lost an identity")
                elapsed = (time.perf_counter() - started) * 1000
                for member in scene.members:
                    oid = scene.object_ids[member]
                    session = self._sessions[(member, camera_id)]
                    mask = current.get(oid)
                    observed = mask is not None
                    # An omitted visible mask is NOT a failed video session.
                    # Preserve native memory/ID and process the next RGB. Do
                    # not expose the previous mask as a fresh observation.
                    session.last_mask = (np.ascontiguousarray(mask) if observed
                                         else np.zeros((image.height, image.width), dtype=bool))
                    session.frame_ts = float(frame_ts)
                    session.snapshot = {
                        **session.snapshot, "status": "observed" if observed else "unobserved",
                        "mask_present": observed,
                        "source_frame_ts": float(frame_ts), "latest_submitted_ts": float(frame_ts),
                        "area_px": int(mask.sum()) if observed else 0, "object_id": oid,
                        "tracking_mode": "persistent_scene_append",
                        "tracking_method": "native_scene_memory", "native_frame_index": index,
                        "future_frames_consumed": False, "processing_ms": elapsed,
                        "reacquisition_attempts": 0,
                        "native_session_id": scene.session_id, "subsequent_prompts": 0,
                        "native_evidence": evidence[oid],
                        "visibility_reason": None if observed else "native_tracker_no_visible_mask",
                    }
                scene.frame_ts, scene.digest = float(frame_ts), digest
                return True
            except Exception as exc:
                scene.failed = True
                for member in scene.members:
                    session = self._sessions[(member, camera_id)]
                    session.snapshot = {**session.snapshot, "status": "error", "mask_present": False,
                                        "error": str(exc), "observation_source": self.error_observation_source}
                raise

    def stop(self, env_id=None, camera_id=None):
        with self._lock, self._runtime._lock:
            affected = list(self._scenes) if env_id is None else [str(env_id)]
            for member in affected:
                scene = self._scenes.get(member)
                if scene is None:
                    continue
                if self._runtime._predictor is not None:
                    self._runtime._close_session(self._runtime._predictor, scene.session_id)
                for peer in scene.members:
                    self._scenes.pop(peer, None)
                    self._sessions.pop((peer, self.camera_id), None)
            return super().stop(env_id, camera_id)
