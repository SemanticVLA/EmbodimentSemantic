"""Project-local MolmoPoint image-pointing runtime.

This module is deliberately independent of the controller and of the Omnis
checkout.  It owns only the VLM boundary: an original-resolution RGB image is
turned into MolmoPoint's decoded image points.  Point coordinates are returned
in the input image's pixel frame; no confidence is invented because the
official MolmoPoint decoder does not provide one.

The Transformers and PyTorch imports are lazy.  This keeps the frozen v9d
runtime and CPU-only tests importable on hosts that do not have the VLM
dependencies installed.  ``model_factory``, ``processor_factory`` and
``torch_factory`` are explicit test seams and are not used by production code.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


# The SHA is the current verified revision of the official HF model repository
# at the time this canary runtime was prepared.  A release may override it only
# by creating a new, explicitly identified MolmoPoint experiment.
MOLMOPOINT_MODEL_ID = "allenai/MolmoPoint-8B"
MOLMOPOINT_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
MOLMOPOINT_TRANSFORMERS_VERSION = "4.57.1"
MOLMOPOINT_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_MAX_POINTS = 64
PROMPT_VARIANTS = {
    "rim_contact": (
        "Point to the leftmost, rightmost, frontmost, and backmost visible tips "
        "of the rim of the bowl highlighted in red."
    ),
    "rim_downward_approach": (
        "Point to distinct contact locations on the visible rim of the bowl "
        "highlighted in red. These are places a parallel-jaw gripper could pinch from above."
    ),
    "rim_clearance": (
        "Point to exposed parts of the rim of the bowl highlighted in red where "
        "robot fingers could grasp it without touching nearby objects."
    ),
}
DEFAULT_PROMPT_ID = "rim_downward_approach"
DEFAULT_PROMPT = PROMPT_VARIANTS[DEFAULT_PROMPT_ID]


class MolmoPointRuntimeError(RuntimeError):
    """Raised when model setup or decoded-point validation fails."""


@dataclass(frozen=True)
class MolmoPointRuntimeConfig:
    """Pinned model and deterministic inference settings."""

    model_id: str = MOLMOPOINT_MODEL_ID
    model_revision: str = MOLMOPOINT_MODEL_REVISION
    transformers_version: str = MOLMOPOINT_TRANSFORMERS_VERSION
    dtype: str = MOLMOPOINT_DTYPE
    device: str = "cuda"
    device_map: str | Mapping[str, Any] | None = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_points: int = DEFAULT_MAX_POINTS
    padding_side: str = "left"
    prompt_id: str = DEFAULT_PROMPT_ID
    prompt: str = DEFAULT_PROMPT

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.model_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", self.model_revision):
            raise ValueError("model_revision must be a pinned 40-character commit SHA")
        if not isinstance(self.transformers_version, str) or not self.transformers_version.strip():
            raise ValueError("transformers_version must be a non-empty string")
        if self.dtype != MOLMOPOINT_DTYPE:
            raise ValueError("MolmoPoint canary inference is pinned to bfloat16")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device_map is not None and not isinstance(self.device_map, (str, Mapping)):
            raise TypeError("device_map must be a string, mapping, or None")
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.max_points, int) or isinstance(self.max_points, bool) or self.max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        if self.prompt_id not in PROMPT_VARIANTS:
            raise ValueError(f"prompt_id must be one of {tuple(PROMPT_VARIANTS)}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")

    def provenance(self) -> dict[str, Any]:
        """Return JSON-safe immutable-run metadata, including a config hash."""

        payload = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "transformers_version": self.transformers_version,
            "dtype": self.dtype,
            "device": self.device,
            "device_map": self.device_map,
            "max_new_tokens": self.max_new_tokens,
            "max_points": self.max_points,
            "padding_side": self.padding_side,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return {**payload, "config_sha256": hashlib.sha256(encoded).hexdigest()}


def _validate_rgb(rgb: Any) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("rgb must be a numpy.ndarray")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("rgb must have positive height and width")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb must have dtype uint8")
    return rgb


def build_mask_highlight(rgb: np.ndarray, mask: np.ndarray, *, alpha: float = 0.35, color: Sequence[int] = (255, 32, 32)) -> np.ndarray:
    """Overlay an observed region without changing resolution or pixel frame."""

    image = _validate_rgb(rgb)
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.shape != image.shape[:2]:
        raise ValueError("mask must be a 2-D array matching rgb height and width")
    if mask_array.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    color_array = np.asarray(tuple(color), dtype=np.float64)
    if color_array.shape != (3,) or not np.all(np.isfinite(color_array)) or np.any(color_array < 0) or np.any(color_array > 255):
        raise ValueError("color must contain three values in [0, 255]")
    output = image.copy()
    if np.any(mask_array) and float(alpha) > 0:
        blended = (1.0 - float(alpha)) * output[mask_array].astype(np.float64) + float(alpha) * color_array
        output[mask_array] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return output


@dataclass(frozen=True)
class MolmoPointRequest:
    """One full image request; ``mask`` is optional for the mask-highlight arm."""

    rgb: np.ndarray
    mask: np.ndarray | None = None
    prompt: str | None = None
    mask_alpha: float = 0.35

    def __post_init__(self) -> None:
        _validate_rgb(self.rgb)
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2 or mask.shape != self.rgb.shape[:2] or mask.dtype != np.bool_:
                raise ValueError("mask must be boolean and match rgb height and width")
        if self.prompt is not None and (not isinstance(self.prompt, str) or not self.prompt.strip()):
            raise ValueError("prompt must be a non-empty string when provided")


@dataclass(frozen=True)
class MolmoImagePoint:
    """A decoded official MolmoPoint row: object id, image id, original x/y."""

    object_id: int
    image_num: int
    x: float
    y: float

    @property
    def uv(self) -> tuple[float, float]:
        """Return ``(u, v) == (x, y)`` in original-image pixel coordinates."""

        return (self.x, self.y)


@dataclass(frozen=True)
class MolmoPointResult:
    image_height: int
    image_width: int
    points: tuple[MolmoImagePoint, ...]
    prompt: str
    highlighted: bool
    provenance: Mapping[str, Any]


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on runtime host
        raise MolmoPointRuntimeError("PyTorch is required for MolmoPoint inference") from exc


class MolmoPointRuntime:
    """Lazy, persistent local Transformers MolmoPoint inference worker."""

    def __init__(
        self,
        config: MolmoPointRuntimeConfig | None = None,
        *,
        model_factory: Callable[..., Any] | None = None,
        processor_factory: Callable[..., Any] | None = None,
        torch_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or MolmoPointRuntimeConfig()
        self._model_factory = model_factory
        self._processor_factory = processor_factory
        self._torch_factory = torch_factory or _torch_module
        self._model: Any | None = None
        self._processor: Any | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _load(self) -> None:
        torch = self._torch_factory()
        if self._model_factory is None or self._processor_factory is None:
            try:
                transformers = importlib.import_module("transformers")
                installed_version = importlib.metadata.version("transformers")
                if installed_version != self.config.transformers_version:
                    raise MolmoPointRuntimeError(
                        f"Transformers {installed_version} does not match pinned {self.config.transformers_version}"
                    )
                self._model_factory = transformers.AutoModelForImageTextToText.from_pretrained
                self._processor_factory = transformers.AutoProcessor.from_pretrained
            except Exception as exc:  # pragma: no cover - depends on runtime host
                raise MolmoPointRuntimeError("Transformers with MolmoPoint remote code is required") from exc

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "revision": self.config.model_revision,
            "dtype": getattr(torch, "bfloat16", self.config.dtype),
        }
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map
        processor_kwargs = {
            "trust_remote_code": True,
            "revision": self.config.model_revision,
            "padding_side": self.config.padding_side,
        }
        try:
            model = self._model_factory(self.config.model_id, **model_kwargs)
            if self.config.device_map is None and hasattr(model, "to"):
                model = model.to(self.config.device)
            if hasattr(model, "eval"):
                model.eval()
            processor = self._processor_factory(self.config.model_id, **processor_kwargs)
        except Exception as exc:
            raise MolmoPointRuntimeError("MolmoPoint model/processor construction failed") from exc
        self._model, self._processor = model, processor

    @staticmethod
    def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved

    @staticmethod
    def _sequence_length(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
        size = getattr(input_ids, "size", None)
        if callable(size):
            return int(size(1))
        raise MolmoPointRuntimeError("processor inputs must provide 2-D input_ids")

    @staticmethod
    def _first_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)) and value:
            if isinstance(value[0], str):
                return value[0]
        raise MolmoPointRuntimeError("MolmoPoint text post-processing returned no text")

    def _decode_points(self, raw_points: Any, *, height: int, width: int) -> tuple[MolmoImagePoint, ...]:
        try:
            rows = np.asarray(raw_points, dtype=object)
        except Exception as exc:
            raise MolmoPointRuntimeError("MolmoPoint returned non-array points") from exc
        if rows.size == 0:
            return ()
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != 4:
            raise MolmoPointRuntimeError("MolmoPoint points must be rows of [object_id, image_num, x, y]")
        if len(rows) > self.config.max_points:
            raise MolmoPointRuntimeError(f"MolmoPoint returned more than max_points={self.config.max_points}")

        decoded: list[MolmoImagePoint] = []
        for row in rows:
            object_id = self._index_value(row[0], "object_id")
            image_num = self._index_value(row[1], "image_num")
            if image_num != 0:
                raise MolmoPointRuntimeError(f"MolmoPoint image_num {image_num} is outside single-image frame")
            try:
                x, y = float(row[2]), float(row[3])
            except (TypeError, ValueError) as exc:
                raise MolmoPointRuntimeError("MolmoPoint coordinates must be numeric") from exc
            if not np.all(np.isfinite((x, y))):
                raise MolmoPointRuntimeError("MolmoPoint coordinates must be finite")
            if not (0.0 <= x < float(width) and 0.0 <= y < float(height)):
                raise MolmoPointRuntimeError("MolmoPoint coordinate is outside the original image frame")
            decoded.append(MolmoImagePoint(object_id, image_num, x, y))
        return tuple(decoded)

    @staticmethod
    def _index_value(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise MolmoPointRuntimeError(f"MolmoPoint {name} must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise MolmoPointRuntimeError(f"MolmoPoint {name} must be an integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
            raise MolmoPointRuntimeError(f"MolmoPoint {name} must be a non-negative integer")
        return int(numeric)

    def predict(self, request: MolmoPointRequest) -> MolmoPointResult:
        """Generate and decode pointing tokens in original-image pixel units."""

        rgb = _validate_rgb(request.rgb)
        prompt = (request.prompt or self.config.prompt).strip()
        model = self._model
        processor = self._processor
        if model is None or processor is None:
            with self._lock:
                if self._model is None or self._processor is None:
                    self._load()
                model, processor = self._model, self._processor
        assert model is not None and processor is not None

        image = build_mask_highlight(rgb, request.mask, alpha=request.mask_alpha) if request.mask is not None else rgb.copy()
        try:
            from PIL import Image

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": Image.fromarray(image, mode="RGB")}]}]
            prepared = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                return_pointing_metadata=True,
            )
            if not isinstance(prepared, Mapping):
                raise MolmoPointRuntimeError("MolmoPoint processor did not return a mapping")
            inputs = dict(prepared)
            try:
                metadata = inputs.pop("metadata")
            except KeyError as exc:
                raise MolmoPointRuntimeError("MolmoPoint processor omitted pointing metadata") from exc
            if not isinstance(metadata, Mapping):
                raise MolmoPointRuntimeError("MolmoPoint metadata must be a mapping")
            for key in ("token_pooling", "subpatch_mapping", "image_sizes"):
                if key not in metadata:
                    raise MolmoPointRuntimeError(f"MolmoPoint metadata omitted {key}")
            inputs = self._move_inputs(inputs, self.config.device)
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise MolmoPointRuntimeError("MolmoPoint processor omitted input_ids")
            input_length = self._sequence_length(input_ids)
            torch = self._torch_factory()
            inference = getattr(torch, "inference_mode", None)
            autocast = getattr(torch, "autocast", None)
            with contextlib.ExitStack() as stack:
                stack.enter_context(inference() if callable(inference) else contextlib.nullcontext())
                if self.config.device.startswith("cuda") and callable(autocast):
                    stack.enter_context(autocast(device_type="cuda", dtype=getattr(torch, "bfloat16", self.config.dtype)))
                logits_processor = model.build_logit_processor_from_inputs(inputs)
                output = model.generate(
                    **inputs,
                    logits_processor=logits_processor,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            generated_tokens = output[:, input_length:]
            generated_text = self._first_text(
                processor.post_process_image_text_to_text(
                    generated_tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            raw_points = model.extract_image_points(
                generated_text,
                metadata["token_pooling"],
                metadata["subpatch_mapping"],
                metadata["image_sizes"],
            )
            points = self._decode_points(raw_points, height=rgb.shape[0], width=rgb.shape[1])
        except MolmoPointRuntimeError:
            raise
        except Exception as exc:
            raise MolmoPointRuntimeError("MolmoPoint inference or decoding failed") from exc
        provenance = {**self.config.provenance(), "effective_prompt": prompt,
                      "generated_text": generated_text, "decoded_point_count": len(points)}
        return MolmoPointResult(rgb.shape[0], rgb.shape[1], points, prompt, request.mask is not None, provenance)

    def close(self) -> None:
        """Release model and processor; the next prediction lazily reloads them."""

        with self._lock:
            self._model = None
            self._processor = None


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_PROMPT",
    "DEFAULT_PROMPT_ID",
    "PROMPT_VARIANTS",
    "MOLMOPOINT_DTYPE",
    "MOLMOPOINT_MODEL_ID",
    "MOLMOPOINT_MODEL_REVISION",
    "MOLMOPOINT_TRANSFORMERS_VERSION",
    "MolmoImagePoint",
    "MolmoPointRequest",
    "MolmoPointResult",
    "MolmoPointRuntime",
    "MolmoPointRuntimeConfig",
    "MolmoPointRuntimeError",
    "build_mask_highlight",
]
