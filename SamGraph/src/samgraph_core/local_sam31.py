"""Pinned SAM 3.1 text-mask adapter and shared native predictor runtime.

Use :class:`LocalSam31SceneTracker` from ``sam31_live_scene`` for SamGraph.
It seeds a single joint native video session from the first RGB frame's
masks, then appends only each newly arriving RGB frame.  The replay tracker
class in this file is its compatibility base; it is not the SamGraph tracker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from PIL import Image


SAM31_SOURCE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM31_HF_REPOSITORY = "facebook/sam3.1"
SAM31_HF_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_CHECKPOINT_FILENAME = "sam3.1_multiplex.pt"
SAM31_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
SAM31_PROVIDER = "local_sam3_1_multiplex"
SAM31_TRACKING_MODE = "sliding_pair_replay"
SAM31_TRACKING_METHOD = "centroid_point_replay"


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* is inside *parent* after resolving symlinks."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _git_output(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"SAM 3.1 source provenance git command failed: {detail}")
    return result.stdout.strip()


def _sam31_package_spec() -> tuple[Path, Path]:
    """Resolve the package and model-builder paths before importing SAM code."""

    package_spec = importlib.util.find_spec("sam3")
    locations = list(package_spec.submodule_search_locations or []) if package_spec else []
    if len(locations) != 1:
        raise RuntimeError(
            "SAM 3.1 source provenance requires exactly one importable sam3 package root"
        )
    package_root = Path(locations[0]).resolve()
    if not package_root.is_dir():
        raise RuntimeError(f"SAM 3.1 package root is not a directory: {package_root}")

    builder_spec = importlib.util.find_spec("sam3.model_builder")
    builder_origin = Path(builder_spec.origin).resolve() if builder_spec and builder_spec.origin else None
    if builder_origin is None or not _path_is_within(builder_origin, package_root):
        raise RuntimeError(
            "SAM 3.1 model_builder resolves outside the single sam3 package root"
        )
    return package_root, builder_origin


def _sam31_git_identity(package_root: Path) -> dict[str, Any]:
    """Verify the imported package is a clean checkout at the qualified commit."""

    git_root = Path(
        _git_output(
            "-C", str(package_root), "rev-parse", "--show-toplevel", cwd=package_root
        )
    ).resolve()
    if not _path_is_within(package_root, git_root):
        raise RuntimeError("SAM 3.1 package root is not inside its Git worktree")
    revision = _git_output("-C", str(package_root), "rev-parse", "HEAD", cwd=package_root)
    if revision != SAM31_SOURCE_REVISION:
        raise RuntimeError(
            "SAM 3.1 source revision does not match the qualified pin: "
            f"expected {SAM31_SOURCE_REVISION}, got {revision}"
        )
    relative_package = package_root.relative_to(git_root)
    status = _git_output(
        "-C",
        str(git_root),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "." if str(relative_package) == "." else str(relative_package),
        cwd=git_root,
    )
    if status:
        raise RuntimeError(
            "SAM 3.1 source package is dirty; tracked or untracked changes are not allowed: "
            f"{status.splitlines()[0]}"
        )
    relative_package_text = relative_package.as_posix()
    return {
        "source_revision": revision,
        # Keep hostnames/usernames out of manifests while retaining enough
        # information to identify the exact source tree inside the checkout.
        "source_path": relative_package_text,
        "source_git_root": "git_worktree",
        "source_git_clean": True,
        "source_verification": "verified_git_checkout",
    }


def _verify_loaded_sam31_modules(package_root: Path) -> None:
    """Reject mixed-origin ``sam3`` modules loaded by dynamic imports."""

    for name, module in tuple(sys.modules.items()):
        if name != "sam3" and not name.startswith("sam3."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        module_path = Path(origin).resolve()
        if not _path_is_within(module_path, package_root):
            raise RuntimeError(
                "SAM 3.1 imported modules have mixed origins: "
                f"{name} resolves to {module_path}, outside {package_root}"
            )


def _verify_sam31_source() -> dict[str, Any]:
    """Verify the actual importable SAM tree before/after upstream loading."""

    package_root, builder_origin = _sam31_package_spec()
    identity = _sam31_git_identity(package_root)
    relative_builder = builder_origin.relative_to(package_root).as_posix()
    package_path = identity["source_path"]
    identity["source_model_builder_path"] = (
        relative_builder if package_path == "." else f"{package_path}/{relative_builder}"
    )
    _verify_loaded_sam31_modules(package_root)
    return identity


def _decode_rgb(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
    except Exception as exc:
        raise ValueError("SAM 3.1 input must be a valid encoded RGB image") from exc
    return image


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalise_masks(
    outputs: Any,
    *,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(outputs, dict):
        raise ValueError("SAM 3.1 response outputs must be a mapping")
    raw_masks = outputs.get("out_binary_masks")
    raw_ids = outputs.get("out_obj_ids")
    if raw_masks is None or raw_ids is None:
        raise ValueError("SAM 3.1 response lacks object IDs or binary masks")
    masks = _as_numpy(raw_masks).astype(bool, copy=False)
    object_ids = _as_numpy(raw_ids).reshape(-1)
    raw_scores = outputs.get("out_probs")
    scores: np.ndarray | None = None
    if raw_scores is not None:
        scores = _as_numpy(raw_scores).reshape(-1).astype(np.float64, copy=False)
        if len(scores) != len(object_ids):
            raise ValueError(
                "SAM 3.1 response confidence/object counts differ: "
                f"{scores.shape}, {object_ids.shape}"
            )
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or len(masks) != len(object_ids):
        raise ValueError(
            f"unexpected SAM 3.1 mask/object shapes: {masks.shape}, {object_ids.shape}"
        )
    if masks.shape[1:] != shape:
        resized = []
        for mask in masks:
            image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            resized.append(
                np.asarray(image.resize(shape[::-1], Image.Resampling.NEAREST), dtype=np.uint8) > 0
            )
        masks = np.stack(resized, axis=0) if resized else np.zeros((0, *shape), dtype=bool)
    return (
        np.ascontiguousarray(masks, dtype=bool),
        object_ids.astype(np.int64, copy=False),
        None if scores is None else np.ascontiguousarray(scores, dtype=np.float64),
    )


def _mask_png(mask: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(output, format="PNG")
    return output.getvalue()


def _mask_centroid_point(mask: np.ndarray) -> tuple[float, float]:
    """Return a normalized positive point for the official SAM3.1 tracker."""

    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    if array.ndim != 2 or not array.any():
        raise ValueError("SAM 3.1 replay seed must be a nonempty 2-D mask")
    ys, xs = np.nonzero(array)
    height, width = array.shape
    if width <= 0 or height <= 0:
        raise ValueError("SAM 3.1 replay seed has invalid dimensions")
    # ``rel_coordinates=True`` is the official point-prompt contract.  Keep
    # subpixel means instead of rounding small LIBERO masks to a biased pixel.
    return float(xs.mean() / width), float(ys.mean() / height)


def _select_replay_mask(
    masks: np.ndarray,
    object_ids: np.ndarray,
    previous_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Select the current visual mask associated with a replay seed."""

    matching = np.flatnonzero(object_ids == 1)
    if len(matching) == 1:
        selected_index = int(matching[0])
    else:
        # Multiplex tracking can remap a point-prompt object ID while the
        # prompt is associated with only one current mask.  Preserve visual
        # identity from the previous mask rather than assuming the caller's
        # ID survived the predictor's internal association.
        valid = [
            index for index, candidate in enumerate(masks)
            if np.asarray(candidate, dtype=bool).any()
        ]
        if not valid:
            raise RuntimeError(
                "SAM 3.1 replay returned no nonempty current mask for selected "
                f"object 1 (available_object_ids={object_ids.tolist()})"
            )
        if len(valid) == 1:
            selected_index = valid[0]
        else:
            previous_bool = np.asarray(previous_mask, dtype=bool)
            scored = []
            for index in valid:
                candidate = np.asarray(masks[index], dtype=bool)
                union = np.logical_or(previous_bool, candidate).sum()
                iou = (
                    float(np.logical_and(previous_bool, candidate).sum() / union)
                    if union
                    else 0.0
                )
                scored.append((iou, index))
            scored.sort(reverse=True)
            if len(scored) > 1 and scored[0][0] <= scored[1][0]:
                raise RuntimeError(
                    "SAM 3.1 replay returned ambiguous current masks for selected "
                    f"object 1 (available_object_ids={object_ids.tolist()})"
                )
            selected_index = scored[0][1]
    mask = np.ascontiguousarray(masks[selected_index], dtype=bool)
    if not mask.any():
        raise RuntimeError("SAM 3.1 replay returned an empty current mask")
    return mask, int(object_ids[selected_index])


def _summarise_replay_outputs(response: Any) -> dict[str, Any]:
    """Return JSON-safe visibility into one official replay response."""

    outputs = response.get("outputs", {}) if isinstance(response, dict) else {}
    summary: dict[str, Any] = {
        "frame_index": response.get("frame_index") if isinstance(response, dict) else None,
        "output_keys": sorted(outputs) if isinstance(outputs, dict) else [],
    }
    if not isinstance(outputs, dict):
        return summary

    object_ids = outputs.get("out_obj_ids")
    if object_ids is not None:
        if hasattr(object_ids, "detach"):
            object_ids = object_ids.detach().cpu().numpy()
        summary["object_ids"] = [int(value) for value in np.asarray(object_ids).reshape(-1)]

    masks = outputs.get("out_binary_masks")
    if masks is not None:
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks_array = np.asarray(masks)
        summary["mask_shape"] = list(masks_array.shape)
        if masks_array.ndim >= 3 and len(masks_array) > 0:
            summary["nonempty_mask_count"] = int(
                np.asarray(masks_array, dtype=bool)
                .reshape(len(masks_array), -1)
                .any(axis=1)
                .sum()
            )
        else:
            summary["nonempty_mask_count"] = 0
    return summary


def _patch_multiplex_init_state(predictor: Any) -> None:
    """Bridge the pinned SAM 3.1 predictor/model signature mismatch.

    The pinned multiplex model does not implement the optional
    ``offload_state_to_cpu`` argument, while the shared predictor dispatcher
    always forwards it.  Keep the official source checkout byte-for-byte
    pinned and adapt only the in-memory model instance.  A request that asks
    for unsupported CPU state offloading fails explicitly rather than being
    silently ignored.
    """

    model = getattr(predictor, "model", None)
    if model is None:
        # Dependency-injected protocol fakes used by unit tests do not expose
        # the official model object; they already accept the request contract.
        return
    original_init_state = model.init_state
    if "offload_state_to_cpu" in inspect.signature(original_init_state).parameters:
        return

    def init_state_compat(*args: Any, offload_state_to_cpu: bool = False, **kwargs: Any):
        if offload_state_to_cpu:
            raise ValueError("SAM 3.1 does not support offloading inference state to CPU")
        return original_init_state(*args, **kwargs)

    model.init_state = init_state_compat


def _patch_multiplex_short_replay_confirmation(predictor: Any) -> None:
    """Keep the selected point track visible in a causal two-frame replay.

    The official multiplex builder enables masklet confirmation with a
    three-consecutive-detection threshold.  A sliding pair intentionally has
    only the seed frame and one current frame, so the official postprocessor
    would otherwise classify the valid current mask as ``unconfirmed`` and
    hide it from ``out_binary_masks``.  Disable only that output suppression;
    the predictor still performs the official detector/tracker inference and
    the caller still rejects empty or stale current masks.
    """

    model = getattr(predictor, "model", None)
    if model is None:
        return
    # ``Sam3MultiplexVideoPredictor.model`` is a lightweight wrapper whose
    # own ``model`` field contains the actual tracking module.  Setting the
    # proxy attribute would only shadow it and leave postprocessing enabled.
    targets = [model]
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        targets.append(nested)
    for target in targets:
        if hasattr(target, "masklet_confirmation_enable"):
            target.masklet_confirmation_enable = False
        # The multiplex demo also buffers the first 15 frames before yielding
        # output.  A causal pair has exactly two frames, so the buffer would
        # make a valid propagated mask look absent even with confirmation
        # disabled.  Disable only these temporal presentation heuristics; the
        # detector/tracker itself still sees both frames.
        for name in ("hotstart_delay", "hotstart_unmatch_thresh", "hotstart_dup_thresh"):
            if hasattr(target, name):
                setattr(target, name, 0)


def _patch_multiplex_uncached_interactive_output(predictor: Any) -> None:
    """Retain actual point/propagation masks on previously uncached frames.

    The pinned upstream _build_sam2_output returns {} immediately when a frame
    has no detector cache, discarding even the supplied native tracker masks.
    Point-only causal sessions have no detector cache. Initialize only an empty
    cache entry; the original merger must still obtain masks from inference.
    No masks, scores, object IDs, or prior-frame geometry are synthesized.
    """
    model = getattr(predictor, "model", None)
    model = getattr(model, "model", model)
    original = getattr(model, "_build_sam2_output", None)
    if original is None:
        return

    def merge_current_masks(inference_state, frame_idx, refined_obj_id_to_mask=None):
        if refined_obj_id_to_mask is not None:
            inference_state.setdefault("cached_frame_outputs", {}).setdefault(frame_idx, {})
        return original(inference_state, frame_idx, refined_obj_id_to_mask)

    model._build_sam2_output = merge_current_masks


class OfficialSam31Runtime:
    """Own exactly one official, pinned SAM 3.1 multiplex predictor."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        source_revision: str = SAM31_SOURCE_REVISION,
        hf_revision: str = SAM31_HF_REVISION,
        confidence_threshold: float = 0.5,
        text_detection_threshold: float | None = None,
        predictor_factory: Any | None = None,
        allow_point_reacquisition: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.source_revision = str(source_revision)
        self.hf_revision = str(hf_revision)
        self.confidence_threshold = float(confidence_threshold)
        self.text_detection_threshold = (None if text_detection_threshold is None
                                         else float(text_detection_threshold))
        if self.text_detection_threshold is not None and not 0 < self.text_detection_threshold < 1:
            raise ValueError("text detection threshold must be in (0, 1)")
        if self.source_revision != SAM31_SOURCE_REVISION:
            raise ValueError("local SAM 3.1 source revision is not the qualified pin")
        if self.hf_revision != SAM31_HF_REVISION:
            raise ValueError("local SAM 3.1 checkpoint revision is not the qualified pin")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("SAM 3.1 confidence threshold must be in (0, 1)")
        self._predictor_factory = predictor_factory
        self.allow_point_reacquisition = bool(allow_point_reacquisition)
        self._predictor: Any | None = None
        self._source_identity: dict[str, Any] | None = None
        self._checkpoint_sha256_value: str | None = None
        self._lock = threading.RLock()
        self._last_tracking_method: str | None = None
        self._last_tracking_debug: dict[str, Any] | None = None
        self._legacy_gpu_attention_fallback = False

    @property
    def last_tracking_method(self) -> str | None:
        return self._last_tracking_method

    @property
    def last_tracking_debug(self) -> dict[str, Any] | None:
        return self._last_tracking_debug

    @property
    def model_identity(self) -> dict[str, Any]:
        return {
            "provider": SAM31_PROVIDER,
            # The constructor argument is configuration only.  It is never
            # evidence of the source that Python imported.  A real source
            # revision appears here only after warmup verifies the checkout;
            # injected predictor factories are explicitly unverified.
            "source_revision": (
                self._source_identity.get("source_revision")
                if self._source_identity is not None
                else None
            ),
            "requested_source_revision": self.source_revision,
            "source_verification": (
                self._source_identity.get("source_verification")
                if self._source_identity is not None
                else "pending_warmup"
            ),
            "checkpoint_repository": SAM31_HF_REPOSITORY,
            "checkpoint_revision": self.hf_revision,
            "checkpoint_filename": SAM31_CHECKPOINT_FILENAME,
            "checkpoint_sha256": self._checkpoint_sha256(),
            "legacy_gpu_attention_fallback": self._legacy_gpu_attention_fallback,
            "text_detection_threshold": self.text_detection_threshold,
            **({} if self._source_identity is None else self._source_identity),
        }

    def _checkpoint_sha256(self) -> str | None:
        if self._checkpoint_sha256_value is not None:
            return self._checkpoint_sha256_value
        if not self.checkpoint_path.is_file():
            return None
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self._checkpoint_sha256_value = digest.hexdigest()
        return self._checkpoint_sha256_value

    def warmup(self) -> None:
        with self._lock:
            if self._predictor is not None:
                if self._predictor_factory is None:
                    # Dynamic imports can happen during upstream predictor
                    # construction. Re-check the actual loaded module tree on
                    # every reuse so a mixed-origin import cannot be hidden
                    # behind an already-warmed runtime.
                    self._source_identity = _verify_sam31_source()
                return
            if not self.checkpoint_path.is_file():
                raise RuntimeError(f"pinned SAM 3.1 checkpoint is absent: {self.checkpoint_path}")
            # Verify the bytes immediately before handing the path to upstream
            # code, even if provenance was queried earlier in this process.
            self._checkpoint_sha256_value = None
            checkpoint_sha256 = self._checkpoint_sha256()
            if checkpoint_sha256 != SAM31_CHECKPOINT_SHA256:
                raise RuntimeError(
                    "SAM 3.1 checkpoint SHA-256 does not match the qualified pin: "
                    f"expected {SAM31_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
                )
            if self._predictor_factory is None:
                # Resolve and verify the source checkout before importing the
                # upstream model builder. The caller-provided revision string
                # is intentionally not used as provenance evidence.
                before_load = _verify_sam31_source()
                try:
                    from sam3.model_builder import build_sam3_multiplex_video_predictor
                except Exception as exc:
                    raise RuntimeError("the pinned official SAM 3.1 package is unavailable") from exc
                factory = build_sam3_multiplex_video_predictor
            else:
                factory = self._predictor_factory
                before_load = {
                    "source_revision": None,
                    "source_path": None,
                    "source_git_root": None,
                    "source_git_clean": None,
                    "source_verification": "unverified_injected_predictor_factory",
                    "predictor_factory_injected": True,
                }
            self._predictor = factory(
                checkpoint_path=str(self.checkpoint_path),
                use_fa3=False,
                use_rope_real=True,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
                default_output_prob_thresh=self.confidence_threshold,
            )
            if self._predictor_factory is None:
                # Validate again after construction because the official
                # builder may import additional sam3 modules dynamically.
                after_load = _verify_sam31_source()
                if after_load["source_revision"] != before_load["source_revision"]:
                    raise RuntimeError("SAM 3.1 source changed while constructing predictor")
                self._source_identity = after_load
                # Pinned decoder.py requests only FLASH_ATTENTION for video
                # propagation. Turing/T4 lacks that kernel; allow PyTorch's
                # efficient or math backend without changing SAM weights.
                import torch
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                    from torch.nn.attention import SDPBackend
                    import sam3.model.decoder as decoder
                    if not getattr(decoder, "_samgraph_turing_sdpa_fallback", False):
                        original_sdpa_kernel = decoder.sdpa_kernel

                        def compatible_sdpa_kernel(requested):
                            requested_backends = ([requested] if isinstance(requested, SDPBackend)
                                                  else list(requested))
                            for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
                                if backend not in requested_backends:
                                    requested_backends.append(backend)
                            return original_sdpa_kernel(requested_backends)

                        decoder.sdpa_kernel = compatible_sdpa_kernel
                        decoder._samgraph_turing_sdpa_fallback = True
                    self._legacy_gpu_attention_fallback = True
            else:
                self._source_identity = before_load
            _patch_multiplex_init_state(self._predictor)
            _patch_multiplex_short_replay_confirmation(self._predictor)
            _patch_multiplex_uncached_interactive_output(self._predictor)

    def _require_predictor(self) -> Any:
        self.warmup()
        assert self._predictor is not None
        return self._predictor

    def _inference_context(self):
        # Autocast is thread-local. The official predictor enters it only in
        # its constructor, but HTTP requests execute on other threads.
        if self._predictor_factory is not None:
            return nullcontext()
        import torch
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _close_session(predictor: Any, session_id: str) -> None:
        predictor.handle_request({
            "type": "close_session",
            "session_id": session_id,
            "run_gc_collect": False,
        })

    @contextmanager
    def _text_detection_context(self):
        """Configure image-only detection; restore native video settings on exit.

        In the pinned SAM3.1 implementation add_prompt does not use its
        output_prob_thresh argument to admit detections. The detector first
        filters with score_threshold_detection, then image_only_det_thresh.
        The caller holds the runtime lock throughout this context.
        """
        if self.text_detection_threshold is None:
            yield
            return
        model = self._require_predictor().model
        names = ("score_threshold_detection", "image_only_det_thresh")
        original = {name: getattr(model, name) for name in names}
        try:
            for name in names:
                setattr(model, name, self.text_detection_threshold)
            yield
        finally:
            for name, value in original.items():
                setattr(model, name, value)

    def segment(self, *, prompt: str, encoded_rgb: bytes) -> list[tuple[np.ndarray, float, int]]:
        cleaned = " ".join(str(prompt).split())
        if not cleaned:
            raise ValueError("SAM 3.1 text prompt cannot be empty")
        image = _decode_rgb(encoded_rgb)
        shape = (image.height, image.width)
        with self._lock, self._inference_context(), self._text_detection_context():
            predictor = self._require_predictor()
            response = predictor.handle_request({"type": "start_session", "resource_path": [image]})
            session_id = str(response["session_id"])
            try:
                result = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": cleaned,
                    "output_prob_thresh": self.confidence_threshold,
                })
                masks, object_ids, scores = _normalise_masks(result["outputs"], shape=shape)
                return [
                    (
                        np.ascontiguousarray(mask, dtype=bool),
                        float(scores[index]) if scores is not None else 1.0,
                        int(object_id),
                    )
                    for index, (mask, object_id) in enumerate(zip(masks, object_ids))
                    if mask.any()
                ]
            finally:
                self._close_session(predictor, session_id)

    def segment_candidates(
        self, *, prompt: str, encoded_rgb: bytes,
    ) -> list[tuple[np.ndarray, float | None, int]]:
        """Full-frame text proposals with nullable, unmodified SAM scores."""
        cleaned = " ".join(str(prompt).split())
        if not cleaned:
            raise ValueError("SAM 3.1 text prompt cannot be empty")
        image = _decode_rgb(encoded_rgb)
        with self._lock, self._inference_context(), self._text_detection_context():
            predictor = self._require_predictor()
            response = predictor.handle_request({
                "type": "start_session", "resource_path": [image],
            })
            session_id = str(response["session_id"])
            try:
                result = predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id,
                    "frame_index": 0, "text": cleaned,
                    "output_prob_thresh": self.confidence_threshold,
                })
                masks, object_ids, scores = _normalise_masks(
                    result["outputs"], shape=(image.height, image.width),
                )
                return [
                    (np.ascontiguousarray(mask, dtype=bool),
                     float(scores[index]) if scores is not None else None,
                     int(object_id))
                    for index, (mask, object_id) in enumerate(zip(masks, object_ids))
                    if mask.any()
                ]
            finally:
                self._close_session(predictor, session_id)

    def segment_many(
        self, *, prompts: tuple[str, ...], encoded_rgb: bytes,
    ) -> dict[str, list[tuple[np.ndarray, float | None, int]]]:
        """Ground shared vocabulary against one current RGB session."""
        if not prompts or any(not " ".join(str(prompt).split()) for prompt in prompts):
            raise ValueError("segment_many requires nonempty text prompts")
        image = _decode_rgb(encoded_rgb)
        with self._lock, self._inference_context(), self._text_detection_context():
            predictor = self._require_predictor()
            response = predictor.handle_request({
                "type": "start_session", "resource_path": [image],
            })
            session_id = str(response["session_id"])
            try:
                results = {}
                for prompt in prompts:
                    result = predictor.handle_request({
                        "type": "add_prompt", "session_id": session_id,
                        "frame_index": 0, "text": prompt,
                        "output_prob_thresh": self.confidence_threshold,
                    })
                    masks, ids, scores = _normalise_masks(
                        result["outputs"], shape=(image.height, image.width),
                    )
                    results[prompt] = [
                        (np.ascontiguousarray(mask, dtype=bool),
                         float(scores[index]) if scores is not None else None,
                         int(oid))
                        for index, (mask, oid) in enumerate(zip(masks, ids))
                        if mask.any()
                    ]
                return results
            finally:
                self._close_session(predictor, session_id)

    def segment_point(
        self,
        *,
        point_xy_rel: tuple[float, float],
        encoded_rgb: bytes,
    ) -> list[tuple[np.ndarray, float, int]]:
        """Return masks from one positive normalized point prompt.

        This is a one-frame official SAM request.  Coordinates are relative
        to the submitted RGB image and use the same ``rel_coordinates``
        contract as the existing video replay path.
        """
        if not isinstance(point_xy_rel, tuple) or len(point_xy_rel) != 2:
            raise ValueError("point_xy_rel must be a two-element tuple")
        try:
            point_x, point_y = (float(point_xy_rel[0]), float(point_xy_rel[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("point_xy_rel coordinates must be finite numbers") from exc
        if not np.isfinite(point_x) or not np.isfinite(point_y):
            raise ValueError("point_xy_rel coordinates must be finite numbers")
        if not (0.0 <= point_x < 1.0 and 0.0 <= point_y < 1.0):
            raise ValueError("point_xy_rel coordinates must be in [0, 1)")
        image = _decode_rgb(encoded_rgb)
        shape = (image.height, image.width)
        with self._lock, self._inference_context():
            predictor = self._require_predictor()
            response = predictor.handle_request({"type": "start_session", "resource_path": [image]})
            session_id = str(response["session_id"])
            try:
                result = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": 1,
                    "points": [[point_x, point_y]],
                    "point_labels": [1],
                    "clear_old_points": True,
                    "rel_coordinates": True,
                    "output_prob_thresh": 0.0,
                })
                masks, object_ids, scores = _normalise_masks(result["outputs"], shape=shape)
                return [
                    (
                        np.ascontiguousarray(mask, dtype=bool),
                        float(scores[index]) if scores is not None else 1.0,
                        int(object_id),
                    )
                    for index, (mask, object_id) in enumerate(zip(masks, object_ids))
                    if mask.any()
                ]
            finally:
                self._close_session(predictor, session_id)

    def track_pair(
        self,
        *,
        previous_rgb: bytes,
        current_rgb: bytes,
        previous_mask: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        previous = _decode_rgb(previous_rgb)
        current = _decode_rgb(current_rgb)
        if previous.size != current.size:
            raise ValueError("SAM 3.1 replay frames must have identical dimensions")
        shape = (previous.height, previous.width)
        seed = np.ascontiguousarray(np.asarray(previous_mask, dtype=bool))
        if seed.shape != shape or not seed.any():
            raise ValueError("SAM 3.1 replay seed must be a nonempty previous-frame mask")
        with self._lock, self._inference_context():
            predictor = self._require_predictor()
            self._last_tracking_method = SAM31_TRACKING_METHOD
            self._last_tracking_debug = {
                "propagation_request": {
                    "start_frame_index": 1,
                    "max_frame_num_to_track": 1,
                    "output_prob_thresh": 0.0,
                },
                "propagation_results": [],
            }
            response = predictor.handle_request({
                "type": "start_session",
                "resource_path": [previous, current],
            })
            session_id = str(response["session_id"])
            try:
                point_x, point_y = _mask_centroid_point(seed)
                prompted = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": 1,
                    "points": [[point_x, point_y]],
                    "point_labels": [1],
                    "clear_old_points": True,
                    "rel_coordinates": True,
                    # A point-conditioned current mask is validated below by
                    # nonemptiness and visual identity.  Do not let the
                    # multiplex score gate hide a valid short-replay track
                    # before the caller can inspect it.
                    "output_prob_thresh": 0.0,
                })
                self._last_tracking_debug["seed_prompt"] = _summarise_replay_outputs(prompted)
                latest = prompted if int(prompted.get("frame_index", 0)) == 1 else None
                for result in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    # ``add_prompt`` has already conditioned frame 0.  Replay
                    # only the newly arriving current frame; asking the
                    # multiplex predictor to process the seed frame again can
                    # re-run association and discard the short-lived point
                    # track before frame 1 is emitted.
                    "start_frame_index": 1,
                    "max_frame_num_to_track": 1,
                    "output_prob_thresh": 0.0,
                    "evict_cached_frame_outputs": True,
                }):
                    self._last_tracking_debug["propagation_results"].append(
                        _summarise_replay_outputs(result)
                    )
                    if int(result.get("frame_index", -1)) == 1:
                        latest = result
                if latest is None:
                    latest = {
                        "outputs": {
                            "out_binary_masks": np.zeros((0, *shape), dtype=bool),
                            "out_obj_ids": np.zeros((0,), dtype=np.int64),
                        }
                    }
                try:
                    masks, object_ids, _scores = _normalise_masks(
                        latest["outputs"], shape=shape
                    )
                    return _select_replay_mask(masks, object_ids, seed)
                except (RuntimeError, ValueError) as propagation_error:
                    if not self.allow_point_reacquisition:
                        raise
                    self._last_tracking_debug["propagation_error"] = (
                        f"{type(propagation_error).__name__}: {propagation_error}"
                    )
                    # The official multiplex postprocessor may return an
                    # empty result for a two-frame window while its tracker
                    # still accepts a current-frame point prompt.  Re-seed
                    # only the current public RGB frame at the previous
                    # visual centroid; never use simulator geometry or hold
                    # the previous mask as if it were current.
                    current_prompt = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 1,
                        "obj_id": 1,
                        "points": [[point_x, point_y]],
                        "point_labels": [1],
                        "clear_old_points": True,
                        "rel_coordinates": True,
                        "output_prob_thresh": 0.0,
                    })
                    self._last_tracking_debug["current_prompt"] = _summarise_replay_outputs(
                        current_prompt
                    )
                    if int(current_prompt.get("frame_index", -1)) != 1:
                        raise propagation_error
                    masks, object_ids, _scores = _normalise_masks(
                        current_prompt["outputs"], shape=shape
                    )
                    mask, object_id = _select_replay_mask(masks, object_ids, seed)
                    self._last_tracking_method = "current_centroid_point_reacquisition"
                    return mask, object_id
            finally:
                self._close_session(predictor, session_id)

    def close(self) -> None:
        with self._lock:
            predictor, self._predictor = self._predictor, None
        if predictor is not None and hasattr(predictor, "shutdown"):
            predictor.shutdown()


class LocalSam31Segmenter:
    """SegmentationService-compatible text mask adapter."""

    provider = SAM31_PROVIDER
    production_inputs = ("public_rgb", "sam3_1_text_masks")

    def __init__(self, runtime: OfficialSam31Runtime) -> None:
        self._runtime = runtime

    @property
    def provenance(self) -> dict[str, Any]:
        """Expose the pinned source/checkpoint identity in every scene."""

        return dict(self._runtime.model_identity)

    def warmup(self) -> None:
        self._runtime.warmup()

    def segment(self, *, prompt: str, jpeg: bytes) -> Any:
        masks = self._runtime.segment(prompt=prompt, encoded_rgb=jpeg)
        return SimpleNamespace(
            masks=[
                SimpleNamespace(png=_mask_png(mask), score=score, object_id=object_id)
                for mask, score, object_id in masks
            ]
        )

    def segment_candidates(self, *, prompt: str, jpeg: bytes) -> Any:
        proposals = self._runtime.segment_candidates(
            prompt=prompt, encoded_rgb=jpeg,
        )
        return SimpleNamespace(masks=[
            SimpleNamespace(png=_mask_png(mask), score=score, object_id=object_id)
            for mask, score, object_id in proposals
        ])

    def segment_many(self, *, prompts: tuple[str, ...], jpeg: bytes) -> dict[str, Any]:
        results = self._runtime.segment_many(
            prompts=prompts, encoded_rgb=jpeg,
        )
        return {
            prompt: SimpleNamespace(masks=[
                SimpleNamespace(png=_mask_png(mask), score=score, object_id=oid)
                for mask, score, oid in proposals
            ])
            for prompt, proposals in results.items()
        }

    def segment_point(
        self,
        *,
        point_xy_rel: tuple[float, float],
        encoded_rgb: bytes,
    ) -> list[tuple[np.ndarray, float, int]]:
        """Forward a one-frame point prompt to the pinned runtime."""
        return self._runtime.segment_point(
            point_xy_rel=point_xy_rel,
            encoded_rgb=encoded_rgb,
        )

    def close(self) -> None:
        # The shared runtime is owned by ResolverRuntime.
        return None


@dataclass(slots=True)
class _ReplaySession:
    env_id: str
    camera_id: str
    query: str
    track_id: str
    frame_ts: float
    previous_rgb: bytes
    last_mask: np.ndarray
    snapshot: dict[str, Any]


class LocalSam31ReplayTracker:
    """Sam2LiveStreamService-compatible causal SAM 3.1 video adapter."""

    enabled = True
    provider = SAM31_PROVIDER + "_video_" + SAM31_TRACKING_MODE
    production_inputs = ("sam3_1_video_current_masks", "previous_public_rgb_mask_seed")
    current_observation_source = "sam3_1_video_current"
    error_observation_source = "sam3_1_error"

    def __init__(
        self,
        runtime: OfficialSam31Runtime,
        *,
        camera_id: str = "agentview",
        max_reacquisition_attempts: int = 1,
        min_reacquisition_iou: float = 0.05,
    ) -> None:
        if int(max_reacquisition_attempts) < 0:
            raise ValueError("SAM 3.1 reacquisition attempts must be nonnegative")
        if not 0.0 <= float(min_reacquisition_iou) <= 1.0:
            raise ValueError("SAM 3.1 reacquisition IoU must be in [0, 1]")
        self._runtime = runtime
        self._camera_id = str(camera_id)
        self._max_reacquisition_attempts = int(max_reacquisition_attempts)
        self._min_reacquisition_iou = float(min_reacquisition_iou)
        self._lock = threading.RLock()
        self._sessions: dict[tuple[str, str], _ReplaySession] = {}

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def warmup(self) -> None:
        self._runtime.warmup()

    def start(
        self,
        *,
        env_id: str,
        query: str,
        frame_ts: float,
        jpeg: bytes,
        seed_mask: np.ndarray | None = None,
        seed_box: tuple[float, float, float, float] | None = None,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        selected_camera = camera_id or self._camera_id
        if selected_camera != self._camera_id:
            raise ValueError(f"SAM 3.1 tracker accepts camera {self._camera_id!r}")
        if seed_mask is None or seed_box is not None:
            raise ValueError("SAM 3.1 replay requires exactly one public-RGB seed mask")
        image = _decode_rgb(jpeg)
        mask = np.ascontiguousarray(np.asarray(seed_mask, dtype=bool))
        if mask.shape != (image.height, image.width) or not mask.any():
            raise ValueError("SAM 3.1 seed mask must match the public RGB frame")
        snapshot = {
            "active": True,
            "status": "observed",
            "track_id": uuid.uuid4().hex,
            "query": " ".join(str(query).split()),
            "camera_id": selected_camera,
            "source_frame_ts": float(frame_ts),
            "latest_submitted_ts": float(frame_ts),
            "mask_present": True,
            "area_px": int(mask.sum()),
            "observation_source": self.current_observation_source,
            "backend": self.provider,
            "tracking_mode": SAM31_TRACKING_MODE,
            # The initial scene mask is supplied by the text segmenter.  No
            # video prompt has run yet, so keep this distinct from the
            # centroid-point replay used on later submitted frames.
            "tracking_method": "seed_mask",
            "reacquisition_attempts": 0,
            "simulator_state_consumed": False,
        }
        key = (str(env_id), selected_camera)
        with self._lock:
            self._sessions[key] = _ReplaySession(
                env_id=str(env_id),
                camera_id=selected_camera,
                query=snapshot["query"],
                track_id=snapshot["track_id"],
                frame_ts=float(frame_ts),
                previous_rgb=bytes(jpeg),
                last_mask=mask,
                snapshot=snapshot,
            )
        return dict(snapshot)

    def submit_frame(
        self,
        *,
        env_id: str,
        camera_id: str,
        frame_ts: float,
        jpeg: bytes,
    ) -> bool:
        key = (str(env_id), str(camera_id))
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                raise KeyError(f"unknown SAM 3.1 replay session {env_id!r}")
            if float(frame_ts) <= session.frame_ts:
                raise ValueError("SAM 3.1 submitted frame timestamp must increase")
            previous_rgb = session.previous_rgb
            previous_mask = np.array(session.last_mask, copy=True)
        started = time.perf_counter()
        reacquisition_attempts = 0
        tracking_method = SAM31_TRACKING_METHOD
        reacquisition_iou: float | None = None
        try:
            try:
                mask, object_id = self._runtime.track_pair(
                    previous_rgb=previous_rgb,
                    current_rgb=bytes(jpeg),
                    previous_mask=previous_mask,
                )
                tracking_method = (
                    getattr(self._runtime, "last_tracking_method", None)
                    or tracking_method
                )
            except Exception as track_exc:
                if self._max_reacquisition_attempts == 0:
                    raise
                last_error: Exception = track_exc
                mask = None
                object_id = None
                for attempt in range(1, self._max_reacquisition_attempts + 1):
                    reacquisition_attempts = attempt
                    try:
                        candidates = self._runtime.segment(
                            prompt=session.query,
                            encoded_rgb=bytes(jpeg),
                        )
                        mask, object_id, reacquisition_iou = self._select_reacquired_mask(
                            candidates,
                            previous_mask,
                        )
                    except Exception as reacquire_exc:
                        last_error = reacquire_exc
                        continue
                    tracking_method = "text_reacquisition"
                    break
                if mask is None or object_id is None:
                    raise RuntimeError(
                        "SAM 3.1 replay failed and bounded text reacquisition failed: "
                        f"{last_error}"
                    ) from track_exc
            snapshot = {
                **session.snapshot,
                "status": "observed",
                "source_frame_ts": float(frame_ts),
                "latest_submitted_ts": float(frame_ts),
                "mask_present": True,
                "area_px": int(mask.sum()),
                "object_id": int(object_id),
                "processing_ms": (time.perf_counter() - started) * 1000.0,
                "observation_source": self.current_observation_source,
                "tracking_method": tracking_method,
                "reacquisition_attempts": reacquisition_attempts,
                "reacquisition_iou": reacquisition_iou,
            }
        except Exception as exc:
            with self._lock:
                current = self._sessions.get(key)
                if current is session:
                    current.snapshot = {
                        **session.snapshot,
                        "status": "error",
                        "mask_present": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "observation_source": self.error_observation_source,
                        "tracking_method": tracking_method,
                        "reacquisition_attempts": reacquisition_attempts,
                        "reacquisition_iou": reacquisition_iou,
                    }
            raise
        with self._lock:
            current = self._sessions.get(key)
            if current is not session:
                raise RuntimeError("SAM 3.1 replay session changed during inference")
            current.frame_ts = float(frame_ts)
            current.previous_rgb = bytes(jpeg)
            current.last_mask = mask
            current.snapshot = snapshot
        return True

    def _select_reacquired_mask(
        self,
        candidates: Iterable[tuple[np.ndarray, float, int]],
        previous_mask: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        previous = np.ascontiguousarray(np.asarray(previous_mask, dtype=bool))
        scored: list[tuple[float, np.ndarray, int]] = []
        for candidate, _score, object_id in candidates:
            mask = np.ascontiguousarray(np.asarray(candidate, dtype=bool))
            if mask.shape != previous.shape or not mask.any():
                continue
            union = np.logical_or(mask, previous).sum()
            if not union:
                continue
            iou = float(np.logical_and(mask, previous).sum() / union)
            scored.append((iou, mask, int(object_id)))
        if not scored:
            raise RuntimeError("SAM 3.1 text reacquisition returned no usable mask")
        scored.sort(key=lambda item: item[0], reverse=True)
        best_iou, best_mask, best_id = scored[0]
        if best_iou < self._min_reacquisition_iou:
            raise RuntimeError(
                "SAM 3.1 text reacquisition had insufficient mask overlap: "
                f"{best_iou:.4f} < {self._min_reacquisition_iou:.4f}"
            )
        if len(scored) > 1 and abs(scored[1][0] - best_iou) <= 1e-6:
            raise RuntimeError("SAM 3.1 text reacquisition selected ambiguous masks")
        return best_mask, best_id, best_iou

    def snapshot(self, env_id: str, camera_id: str | None = None) -> dict[str, Any]:
        key = (str(env_id), camera_id or self._camera_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return {
                    "active": False,
                    "status": "idle",
                    "mask_present": False,
                    "observation_source": None,
                    "backend": self.provider,
                }
            return dict(session.snapshot)

    def latest_mask(
        self,
        env_id: str,
        *,
        camera_id: str | None = None,
        include_held: bool = False,
    ) -> np.ndarray | None:
        # Keep the previous successful mask internally so a caller can retry
        # the next frame, but never expose it as a current observation after
        # an inference error.  Returning a stale mask here would let a caller
        # bypass the snapshot contract and silently turn a failed track into
        # apparently valid geometry.
        del include_held
        key = (str(env_id), camera_id or self._camera_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or session.snapshot.get("status") != "observed":
                return None
            return np.array(session.last_mask, copy=True)

    def stop(self, env_id: str | None = None, camera_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if env_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop((str(env_id), camera_id or self._camera_id), None)
        return {"active": False, "status": "idle", "backend": self.provider}

    def close(self) -> None:
        self.stop()


__all__ = [
    "LocalSam31ReplayTracker",
    "LocalSam31Segmenter",
    "OfficialSam31Runtime",
    "SAM31_CHECKPOINT_FILENAME",
    "SAM31_HF_REPOSITORY",
    "SAM31_HF_REVISION",
    "SAM31_PROVIDER",
    "SAM31_SOURCE_REVISION",
    "SAM31_TRACKING_MODE",
    "SAM31_TRACKING_METHOD",
]
