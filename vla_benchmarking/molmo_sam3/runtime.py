"""A minimal, project-owned SAM3.1 image predictor boundary.

The model is loaded only on the first call to :meth:`Sam3Runtime.predict`.
Importing this module therefore remains safe in the existing CPU test and v9d
environments.  The actual model source must be a local checkout of the pinned
SAM3 repository; no Omnis path or service is consulted.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


# These identifiers are deliberately duplicated in sam3_source.lock.json so a
# release manifest can be checked without importing Python.
SAM3_SOURCE_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
DEFAULT_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


class Sam3RuntimeError(RuntimeError):
    """Base error for fail-closed local runtime setup or prediction."""


@dataclass(frozen=True)
class Sam3RuntimeConfig:
    """Explicit local model provenance and inference settings.

    ``sam3_source_dir`` is the root containing the local ``sam3`` Python
    package.  It is required for the real builder, so an installed or Omnis
    copy cannot be selected accidentally.  ``checkpoint_path`` is checked by
    SHA-256 before model construction.
    """

    sam3_source_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256
    device: str = "cuda"
    bpe_path: Path | None = None
    score_threshold: float = 0.50
    mask_threshold: float = 0.50
    max_detections: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "sam3_source_dir", Path(self.sam3_source_dir))
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.bpe_path is not None:
            object.__setattr__(self, "bpe_path", Path(self.bpe_path))
        digest = self.checkpoint_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest")
        object.__setattr__(self, "checkpoint_sha256", digest)
        validate_threshold(self.score_threshold, name="score_threshold")
        validate_threshold(self.mask_threshold, name="mask_threshold")
        if not isinstance(self.max_detections, (int, np.integer)) or self.max_detections < 1:
            raise ValueError("max_detections must be a positive integer")


@dataclass(frozen=True)
class Sam3Request:
    """One original-resolution RGB observation and text segmentation query."""

    rgb: np.ndarray
    prompt: str = "bowl"
    score_threshold: float | None = None
    mask_threshold: float | None = None
    max_detections: int | None = None

    def __post_init__(self) -> None:
        validate_rgb_image(self.rgb)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.score_threshold is not None:
            validate_threshold(self.score_threshold, name="score_threshold")
        if self.mask_threshold is not None:
            validate_threshold(self.mask_threshold, name="mask_threshold")
        if self.max_detections is not None and (
            not isinstance(self.max_detections, (int, np.integer)) or self.max_detections < 1
        ):
            raise ValueError("max_detections must be a positive integer")


@dataclass(frozen=True)
class Sam3Detection:
    """One SAM3 detection in the request's original pixel frame."""

    mask: np.ndarray
    score: float
    box_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError("detection mask must be a two-dimensional boolean array")
        if not np.isfinite(self.score) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("detection score must be finite and in [0, 1]")
        if len(self.box_xyxy) != 4 or not np.all(np.isfinite(self.box_xyxy)):
            raise ValueError("detection box must contain four finite values")
        x0, y0, x1, y1 = map(float, self.box_xyxy)
        if x1 < x0 or y1 < y0:
            raise ValueError("detection box must be xyxy with non-negative extent")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "box_xyxy", (x0, y0, x1, y1))


@dataclass(frozen=True)
class Sam3Result:
    """Original-size detections and immutable provenance for one prediction."""

    image_height: int
    image_width: int
    detections: tuple[Sam3Detection, ...]
    prompt: str
    source_commit: str = SAM3_SOURCE_COMMIT
    checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256


def validate_rgb_image(rgb: Any) -> np.ndarray:
    """Validate and return an HxWx3 uint8 image without silently reshaping it."""

    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy.ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("rgb must have positive height and width")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb must have dtype uint8")
    return rgb


def validate_threshold(value: float, *, name: str = "threshold") -> float:
    """Validate a closed-unit-interval score/mask threshold."""

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint in bounded memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Sam3RuntimeError(f"cannot read SAM3 checkpoint {path}: {exc}") from exc
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/numpy-like model outputs without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _extract_output(output: Mapping[str, Any], *, height: int, width: int, score_threshold: float, mask_threshold: float, max_detections: int) -> tuple[Sam3Detection, ...]:
    try:
        raw_masks = _to_numpy(output["masks"])
        raw_scores = _to_numpy(output["scores"]).reshape(-1)
        raw_boxes = _to_numpy(output["boxes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Sam3RuntimeError("SAM3 output must provide masks, scores, and boxes") from exc

    if raw_masks.ndim == 4 and raw_masks.shape[1] == 1:
        raw_masks = raw_masks[:, 0]
    if raw_masks.ndim != 3 or raw_masks.shape[1:] != (height, width):
        raise Sam3RuntimeError("SAM3 masks are not at the original image resolution")
    if raw_boxes.ndim != 2 or raw_boxes.shape[1:] != (4,):
        raise Sam3RuntimeError("SAM3 boxes must have shape Nx4 in xyxy pixel coordinates")
    count = raw_masks.shape[0]
    if len(raw_scores) != count or len(raw_boxes) != count:
        raise Sam3RuntimeError("SAM3 masks, scores, and boxes have inconsistent counts")
    if not np.all(np.isfinite(raw_scores)) or not np.all(np.isfinite(raw_boxes)):
        raise Sam3RuntimeError("SAM3 returned non-finite scores or boxes")
    if np.any(raw_scores < 0.0) or np.any(raw_scores > 1.0):
        raise Sam3RuntimeError("SAM3 scores must be probabilities in [0, 1]")

    order = np.argsort(-raw_scores, kind="stable")
    detections: list[Sam3Detection] = []
    for index in order:
        score = float(raw_scores[index])
        if score < score_threshold:
            continue
        box = np.asarray(raw_boxes[index], dtype=np.float64)
        x0, y0, x1, y1 = map(float, box)
        # Permit only numerical overhang; grossly invalid coordinates indicate
        # a wrong processor frame and must not reach grasp geometry.
        if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1 or x1 < x0 or y1 < y0:
            raise Sam3RuntimeError("SAM3 returned a box outside the original image frame")
        mask = raw_masks[index]
        if np.issubdtype(mask.dtype, np.floating):
            if not np.all(np.isfinite(mask)):
                raise Sam3RuntimeError("SAM3 returned a non-finite mask")
            mask = mask >= mask_threshold
        else:
            if np.issubdtype(mask.dtype, np.integer) and np.any((mask != 0) & (mask != 1)):
                raise Sam3RuntimeError("SAM3 integer masks must contain only 0 and 1")
            mask = mask.astype(bool, copy=False)
        mask = np.asarray(mask, dtype=np.bool_)
        if not np.any(mask):
            continue
        detections.append(
            Sam3Detection(
                mask=mask,
                score=score,
                box_xyxy=(max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)),
            )
        )
        if len(detections) >= max_detections:
            break
    return tuple(detections)


class Sam3Runtime:
    """Lazy local SAM3 predictor with injectable factories for tests."""

    def __init__(
        self,
        config: Sam3RuntimeConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
        processor_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._processor: Any | None = None
        self._predictor: Any | None = None
        self._predictor_mode = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._processor is not None or self._predictor is not None

    def _validate_local_install(self) -> None:
        source = self.config.sam3_source_dir.resolve()
        if any("omnis" in part.lower() for part in source.parts):
            raise Sam3RuntimeError("SAM3 source must not be loaded from an Omnis directory")
        if not source.is_dir() or not (source / "sam3").is_dir():
            raise Sam3RuntimeError(f"local SAM3 source directory is missing sam3/: {source}")
        if not self.config.checkpoint_path.is_file():
            raise Sam3RuntimeError(f"SAM3 checkpoint does not exist: {self.config.checkpoint_path}")
        actual = compute_file_sha256(self.config.checkpoint_path)
        if actual != self.config.checkpoint_sha256:
            raise Sam3RuntimeError(
                f"SAM3 checkpoint SHA-256 {actual} does not match expected {self.config.checkpoint_sha256}"
            )
        # Injected factories are the explicit dependency-injection seam used by
        # CPU tests. They do not import a source checkout, so there is no git
        # revision to verify in that path; production construction below always
        # verifies the pinned checkout before importing it.
        if self._model_factory is not None and self._processor_factory is not None:
            return
        revision_marker = source / ".canary_source_commit"
        revision = revision_marker.read_text(encoding="utf-8").strip() if revision_marker.is_file() else None
        if revision is None:
            try:
                revision = subprocess.run(
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError) as exc:
                raise Sam3RuntimeError("cannot verify local SAM3 source revision") from exc
        if revision != SAM3_SOURCE_COMMIT:
            raise Sam3RuntimeError(
                f"SAM3 source revision {revision!r} does not match pinned {SAM3_SOURCE_COMMIT}"
            )

    def _load(self) -> Any:
        self._validate_local_install()
        if self._model_factory is None:
            source = str(self.config.sam3_source_dir.resolve())
            source_path = Path(source)
            for module_name, module in tuple(sys.modules.items()):
                if module_name != "sam3" and not module_name.startswith("sam3."):
                    continue
                module_path = getattr(module, "__file__", None)
                if module_path is None:
                    continue
                try:
                    Path(module_path).resolve().relative_to(source_path)
                except ValueError as exc:
                    raise Sam3RuntimeError(
                        f"a non-local {module_name} module is already loaded; refusing to reuse it"
                    ) from exc
            sys.path.insert(0, source)
            try:
                builder_module = importlib.import_module("sam3.model_builder")
                processor_module = importlib.import_module("sam3.model.sam3_image_processor")
            except Exception as exc:
                raise Sam3RuntimeError("cannot import pinned local SAM3 source") from exc
            finally:
                if sys.path and sys.path[0] == source:
                    sys.path.pop(0)
            predictor_factory = getattr(builder_module, "build_sam3_predictor", None)
            if callable(predictor_factory):
                try:
                    self._predictor = predictor_factory(
                        checkpoint_path=str(self.config.checkpoint_path),
                        bpe_path=None if self.config.bpe_path is None else str(self.config.bpe_path),
                        version="sam3.1", compile=False, warm_up=False,
                        max_num_objects=16, multiplex_count=16,
                        use_fa3=False, use_rope_real=False,
                        async_loading_frames=False,
                    )
                except Exception as exc:
                    raise Sam3RuntimeError("pinned local SAM3.1 predictor construction failed") from exc
                self._predictor_mode = True
                return self._predictor
            self._model_factory = getattr(builder_module, "build_sam3_image_model")
            self._processor_factory = getattr(processor_module, "Sam3Processor")
        kwargs = {"device": self.config.device, "eval_mode": True, "checkpoint_path": str(self.config.checkpoint_path)}
        if self.config.bpe_path is not None:
            kwargs["bpe_path"] = str(self.config.bpe_path)
        try:
            model = self._model_factory(**kwargs)
            # Keep all detections available here; this boundary applies the
            # request threshold itself, including per-request overrides.
            self._processor = self._processor_factory(
                model, device=self.config.device, confidence_threshold=0.0
            )
        except Exception as exc:
            raise Sam3RuntimeError("pinned local SAM3 model construction failed") from exc
        return self._processor

    @staticmethod
    def _extract_predictor_output(output: Mapping[str, Any], *, height: int, width: int) -> dict[str, Any]:
        """Adapt the official SAM3.1 predictor's output names to this boundary."""
        outputs = output.get("outputs", output) if isinstance(output, Mapping) else output
        if not isinstance(outputs, Mapping):
            raise Sam3RuntimeError("SAM3.1 predictor did not return an output mapping")
        masks = _to_numpy(outputs.get("out_binary_masks", ()))
        if masks.size == 0:
            return {"masks": np.empty((0, height, width), dtype=np.bool_), "scores": np.empty((0,), dtype=np.float32), "boxes": np.empty((0, 4), dtype=np.float32)}
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3 or masks.shape[1:] != (height, width):
            raise Sam3RuntimeError("SAM3.1 predictor masks are not at original resolution")
        scores = _to_numpy(outputs.get("out_probs", np.ones((masks.shape[0],), dtype=np.float32))).reshape(-1)
        if len(scores) != len(masks):
            raise Sam3RuntimeError("SAM3.1 predictor scores and masks have inconsistent counts")
        boxes = np.zeros((len(masks), 4), dtype=np.float32)
        for index, mask in enumerate(masks.astype(bool, copy=False)):
            ys, xs = np.nonzero(mask)
            if len(xs):
                boxes[index] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        return {"masks": masks, "scores": scores, "boxes": boxes}

    def predict(self, request: Sam3Request) -> Sam3Result:
        """Run text-prompt segmentation and return original-size detections."""

        rgb = validate_rgb_image(request.rgb)
        score_threshold = self.config.score_threshold if request.score_threshold is None else validate_threshold(request.score_threshold, name="score_threshold")
        mask_threshold = self.config.mask_threshold if request.mask_threshold is None else validate_threshold(request.mask_threshold, name="mask_threshold")
        max_detections = self.config.max_detections if request.max_detections is None else request.max_detections
        if self._processor is None and self._predictor is None:
            with self._lock:
                if self._processor is None and self._predictor is None:
                    self._load()
        if self._predictor_mode:
            predictor = self._predictor
            if predictor is None:
                raise Sam3RuntimeError("SAM3.1 predictor was not initialized")
            session_id = None
            try:
                from PIL import Image
                session = predictor.handle_request({"type": "start_session", "resource_path": [Image.fromarray(rgb, mode="RGB")]})
                session_id = str(session["session_id"])
                output = predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": request.prompt.strip()})
                normalized = self._extract_predictor_output(output, height=rgb.shape[0], width=rgb.shape[1])
            except Sam3RuntimeError:
                raise
            except Exception as exc:
                raise Sam3RuntimeError("SAM3.1 prediction failed") from exc
            finally:
                if session_id is not None:
                    try:
                        predictor.handle_request({"type": "close_session", "session_id": session_id})
                    except Exception:
                        pass
            detections = _extract_output(
                normalized, height=rgb.shape[0], width=rgb.shape[1],
                score_threshold=score_threshold, mask_threshold=mask_threshold,
                max_detections=int(max_detections),
            )
            return Sam3Result(
                image_height=rgb.shape[0], image_width=rgb.shape[1],
                detections=detections, prompt=request.prompt.strip(),
                checkpoint_sha256=self.config.checkpoint_sha256,
            )
        try:
            from PIL import Image

            state = self._processor.set_image(Image.fromarray(rgb, mode="RGB"))
            output = self._processor.set_text_prompt(prompt=request.prompt.strip(), state=state)
        except Exception as exc:
            raise Sam3RuntimeError("SAM3 prediction failed") from exc
        if not isinstance(output, Mapping):
            raise Sam3RuntimeError("SAM3 prediction did not return a mapping")
        detections = _extract_output(
            output,
            height=rgb.shape[0],
            width=rgb.shape[1],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            max_detections=int(max_detections),
        )
        return Sam3Result(
            image_height=rgb.shape[0],
            image_width=rgb.shape[1],
            detections=detections,
            prompt=request.prompt.strip(),
            checkpoint_sha256=self.config.checkpoint_sha256,
        )

    def close(self) -> None:
        """Release the processor/model reference; next predict reloads lazily."""

        with self._lock:
            self._processor = None
            self._predictor = None
            self._predictor_mode = False
