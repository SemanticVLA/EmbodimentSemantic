from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

from ..proxy.artifacts import ArtifactStore
from ..proxy.dataset import load_episode_index, load_sampled_index
from ..proxy.metadata_signals import load_metadata_frames
from ..proxy.schemas import BBoxFrame, BBoxObject, MetadataFrame, SampledFrameRecord, read_jsonl
from ..proxy.bbox_geometry import bbox_area, bbox_iomin, bbox_iou
from ..proxy.task_priors import CANONICAL_OBJECTS, task_prior
from .base import VisionDependencyError
from .detection_cache import DetectionCache
from .optical_flow import track_missing_bowl


PROMPT = (
    "black shallow round bowl. red plastic drawer cabinet. black rectangular stove platform. "
    "orange-edged white plastic cookie bag. white round plate."
)
BOWL_PROMPT = "black shallow bowl."
STATIC_OBJECTS = frozenset(CANONICAL_OBJECTS) - {"black_bowl"}
RECOVERY_PROMPTS = {
    "black_bowl": "black shallow bowl.",
    "red_drawer": "red plastic drawer cabinet.",
    "black_stove": "black rectangular stove platform.",
    "cookie": "orange-edged white plastic cookie bag. orange and white cookie packet.",
    "white_plate": "white round plate. white circular dish.",
}


@dataclass(frozen=True)
class AgentEpisodePlan:
    reference_frame: int
    bowl_detection_frames: frozenset[int]
    movement_start: int | None
    movement_end: int | None
    release_frame: int | None
    metadata_reliable: bool


def plan_agent_episode(
    frames: list[SampledFrameRecord],
    signals: list[MetadataFrame],
) -> AgentEpisodePlan:
    """Choose sparse bowl detections without using metadata for relation direction."""
    ordered = sorted(frames, key=lambda item: item.frame)
    if not ordered:
        raise ValueError("Agent episode planning requires at least one sampled frame")
    reference = ordered[0].frame
    reliable = bool(signals) and all(item.metadata_reliable for item in signals)
    if not reliable:
        return AgentEpisodePlan(
            reference_frame=reference,
            bowl_detection_frames=frozenset(item.frame for item in ordered if item.frame != reference),
            movement_start=None,
            movement_end=None,
            release_frame=None,
            metadata_reliable=False,
        )

    moving = [item.frame for item in signals if item.held or item.lifted]
    if not moving:
        return AgentEpisodePlan(reference, frozenset(), None, None, None, True)

    movement_start = min(moving)
    movement_end = max(moving)
    dynamic = {
        item.frame
        for item in ordered
        if movement_start <= item.frame <= movement_end and item.frame != reference
    }
    released = sorted(
        item.frame for item in signals
        if item.frame >= movement_start and (item.released or item.phase == "released")
    )
    release_event = released[0] if released else movement_end + 1
    release_frame = next((item.frame for item in ordered if item.frame >= release_event), None)
    if release_frame is not None and release_frame != reference:
        dynamic.add(release_frame)
    return AgentEpisodePlan(
        reference_frame=reference,
        bowl_detection_frames=frozenset(dynamic),
        movement_start=movement_start,
        movement_end=movement_end,
        release_frame=release_frame,
        metadata_reliable=True,
    )


def select_bowl_model_frames(plan: AgentEpisodePlan, stride: int) -> frozenset[int]:
    if not plan.metadata_reliable:
        return plan.bowl_detection_frames
    stride = max(1, int(stride))
    moving = sorted(
        frame
        for frame in plan.bowl_detection_frames
        if frame != plan.release_frame
    )
    selected = set(moving[::stride])
    if moving:
        selected.add(moving[-1])
    if plan.release_frame is not None:
        selected.add(plan.release_frame)
    return frozenset(selected)


def _canonical_label(label: str) -> str | None:
    value = label.lower()
    if "bowl" in value and "cylinder" not in value:
        return "black_bowl"
    if "drawer" in value or "cabinet" in value:
        return "red_drawer"
    if "stove" in value or "platform" in value:
        return "black_stove"
    if "cookie" in value or "package" in value:
        return "cookie"
    if "plate" in value or "dish" in value:
        return "white_plate"
    return None


class GroundedSam2Detector:
    base_name = "grounding-dino-sam2-v7-agent-static-flow-stride"
    name = base_name

    @classmethod
    def cache_name(cls, config: dict[str, Any]) -> str:
        vision = config["vision"]
        cache_settings = {
            "prompt": PROMPT,
            "recovery_prompts": RECOVERY_PROMPTS,
            "model": str(vision["grounding_dino_model"]),
            "model_revision": str(vision.get("grounding_dino_revision", "")),
            "processor_use_fast": bool(vision.get("processor_use_fast", True)),
            "box_threshold": float(vision["box_threshold"]),
            "wrist_box_threshold": float(vision.get("wrist_box_threshold", 0.50)),
            "text_threshold": float(vision["text_threshold"]),
            "wrist_text_threshold": float(vision.get("wrist_text_threshold", 0.20)),
            "maximum_box_area_fraction": float(vision.get("maximum_box_area_fraction", 0.65)),
            "maximum_bowl_area_fraction": float(vision.get("maximum_bowl_area_fraction", 0.15)),
            "wrist_minimum_box_area_fraction": float(
                vision.get("wrist_minimum_box_area_fraction", 0.001)
            ),
            "wrist_maximum_box_area_fraction": float(
                vision.get("wrist_maximum_box_area_fraction", 0.90)
            ),
            "wrist_maximum_bowl_area_fraction": float(
                vision.get("wrist_maximum_bowl_area_fraction", 0.80)
            ),
            "wrist_minimum_bowl_aspect_ratio": float(
                vision.get("wrist_minimum_bowl_aspect_ratio", 0.70)
            ),
            "wrist_maximum_plate_aspect_ratio": float(
                vision.get("wrist_maximum_plate_aspect_ratio", 2.00)
            ),
            "wrist_minimum_cookie_orange_fraction": float(
                vision.get("wrist_minimum_cookie_orange_fraction", 0.008)
            ),
            "sam2": bool(str(vision.get("sam2_config", "")).strip())
            and bool(str(vision.get("sam2_checkpoint", "")).strip()),
            "optical_flow_missing_bowl": bool(vision.get("optical_flow_missing_bowl", True)),
            "dynamic_bowl_dino_stride": int(vision.get("dynamic_bowl_dino_stride", 2)),
            "optical_flow_minimum_bidirectional_iou": float(
                vision.get("optical_flow_minimum_bidirectional_iou", 0.15)
            ),
        }
        fingerprint = hashlib.sha256(
            json.dumps(cache_settings, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        return f"{cls.base_name}-{fingerprint}"

    def __init__(self, config: dict[str, Any]):
        vision = config["vision"]
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise VisionDependencyError(
                "Grounding DINO detection requires torch and transformers. "
                "Install the optional vision dependencies or use import-bboxes."
            ) from exc

        requested = str(vision.get("device", "auto"))
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else (
            "cpu" if requested == "auto" else requested
        )
        self.torch = torch
        model_id = str(vision["grounding_dino_model"])
        revision = str(vision.get("grounding_dino_revision", "")).strip() or None
        pretrained_options: dict[str, Any] = {}
        if revision is not None:
            pretrained_options["revision"] = revision
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=bool(vision.get("processor_use_fast", True)),
            **pretrained_options,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            **pretrained_options,
        ).to(self.device).eval()
        self.box_threshold = float(vision["box_threshold"])
        self.wrist_box_threshold = float(vision.get("wrist_box_threshold", 0.50))
        self.text_threshold = float(vision["text_threshold"])
        self.wrist_text_threshold = float(vision.get("wrist_text_threshold", 0.20))
        self.maximum_box_area_fraction = float(vision.get("maximum_box_area_fraction", 0.65))
        self.maximum_bowl_area_fraction = float(vision.get("maximum_bowl_area_fraction", 0.15))
        self.wrist_minimum_box_area_fraction = float(
            vision.get("wrist_minimum_box_area_fraction", 0.001)
        )
        self.wrist_maximum_box_area_fraction = float(
            vision.get("wrist_maximum_box_area_fraction", 0.90)
        )
        self.wrist_maximum_bowl_area_fraction = float(
            vision.get("wrist_maximum_bowl_area_fraction", 0.80)
        )
        self.wrist_minimum_bowl_aspect_ratio = float(
            vision.get("wrist_minimum_bowl_aspect_ratio", 0.70)
        )
        self.wrist_maximum_plate_aspect_ratio = float(
            vision.get("wrist_maximum_plate_aspect_ratio", 2.00)
        )
        self.wrist_minimum_cookie_orange_fraction = float(
            vision.get("wrist_minimum_cookie_orange_fraction", 0.008)
        )
        self.sam_predictor = self._load_sam2(vision)
        self.name = self.cache_name(config)

    def _load_sam2(self, vision: dict[str, Any]):
        config_path = str(vision.get("sam2_config", "")).strip()
        checkpoint = str(vision.get("sam2_checkpoint", "")).strip()
        if not config_path or not checkpoint:
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise VisionDependencyError(
                "SAM 2 paths were configured but the sam2 package is not installed"
            ) from exc
        model = build_sam2(config_path, checkpoint, device=self.device)
        return SAM2ImagePredictor(model)

    def _refine_boxes(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        if self.sam_predictor is None or len(boxes) == 0:
            return boxes
        self.sam_predictor.set_image(np.asarray(image.convert("RGB")))
        masks, _, _ = self.sam_predictor.predict(box=boxes, multimask_output=False)
        refined: list[list[float]] = []
        for box, mask in zip(boxes, masks):
            mask_2d = np.asarray(mask).squeeze()
            ys, xs = np.nonzero(mask_2d)
            if len(xs) == 0:
                refined.append(box.tolist())
            else:
                refined.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        return np.asarray(refined, dtype=float)

    @staticmethod
    def _task_from_image_path(image_path: Path | None) -> str:
        if image_path is None or len(image_path.parents) < 4:
            return ""
        return image_path.parents[3].name

    @staticmethod
    def _select_bowl(
        candidates: list[BBoxObject],
        selected: dict[str, BBoxObject],
        task: str,
    ) -> BBoxObject | None:
        if not candidates:
            return None
        prior = task_prior(task)
        support_boxes = [
            selected[name].bbox
            for name in prior.allowed_supports
            if name in selected
        ]

        def rank(item: BBoxObject) -> float:
            x1, y1, x2, y2 = item.bbox
            aspect = max(0.0, x2 - x1) / max(1.0, y2 - y1)
            shallow_bonus = min(0.30, 0.18 * max(0.0, aspect - 0.8))
            cylinder_penalty = 0.30 if aspect < 0.85 else 0.0
            support_bonus = 0.20 * max(
                (bbox_iomin(item.bbox, support) for support in support_boxes),
                default=0.0,
            )
            return item.confidence + shallow_bonus + support_bonus - cylinder_penalty

        return max(candidates, key=rank)

    def _filter_objects(
        self,
        objects: dict[str, BBoxObject],
        image: Image.Image,
        image_path: Path | None,
        minimum_override: float | None = None,
    ) -> dict[str, BBoxObject]:
        camera = image_path.parents[1].name if image_path is not None and len(image_path.parents) >= 2 else ""
        minimum = (
            float(minimum_override)
            if minimum_override is not None
            else (self.wrist_box_threshold if camera == "wrist" else self.box_threshold)
        )
        image_area = float(image.width * image.height)
        minimum_area = (
            getattr(self, "wrist_minimum_box_area_fraction", 0.001)
            if camera == "wrist"
            else 0.0
        )
        maximum_area = (
            getattr(self, "wrist_maximum_box_area_fraction", 0.90)
            if camera == "wrist"
            else self.maximum_box_area_fraction
        )
        maximum_bowl_area = (
            getattr(self, "wrist_maximum_bowl_area_fraction", 0.80)
            if camera == "wrist"
            else getattr(self, "maximum_bowl_area_fraction", 0.15)
        )
        candidates = [
            (name, item)
            for name, item in objects.items()
            if item.confidence >= minimum
            and minimum_area
            <= bbox_area(item.bbox) / max(1.0, image_area)
            <= maximum_area
            and (
                name != "black_bowl"
                or bbox_area(item.bbox) / max(1.0, image_area)
                <= maximum_bowl_area
            )
        ]
        selected: dict[str, BBoxObject] = {}
        for name, item in sorted(candidates, key=lambda value: value[1].confidence, reverse=True):
            if any(bbox_iou(item.bbox, kept.bbox) >= 0.90 for kept in selected.values()):
                continue
            selected[name] = item
        return selected

    def _candidate_area_allowed(
        self,
        name: str,
        item: BBoxObject,
        image: Image.Image,
        image_path: Path | None = None,
    ) -> bool:
        fraction = bbox_area(item.bbox) / max(1.0, float(image.width * image.height))
        camera = (
            image_path.parents[1].name
            if image_path is not None and len(image_path.parents) >= 2
            else ""
        )
        minimum = (
            getattr(self, "wrist_minimum_box_area_fraction", 0.001)
            if camera == "wrist"
            else 0.0
        )
        maximum = (
            getattr(self, "wrist_maximum_box_area_fraction", 0.90)
            if camera == "wrist"
            else self.maximum_box_area_fraction
        )
        bowl_maximum = (
            getattr(self, "wrist_maximum_bowl_area_fraction", 0.80)
            if camera == "wrist"
            else self.maximum_bowl_area_fraction
        )
        if not minimum <= fraction <= maximum:
            return False
        width = max(0.0, item.bbox[2] - item.bbox[0])
        height = max(1.0, item.bbox[3] - item.bbox[1])
        aspect = width / height
        if camera == "wrist" and name == "white_plate":
            maximum_plate_aspect = getattr(
                self, "wrist_maximum_plate_aspect_ratio", 2.00
            )
            if not 1.0 / maximum_plate_aspect <= aspect <= maximum_plate_aspect:
                return False
        if name != "black_bowl":
            return True
        minimum_aspect = (
            getattr(self, "wrist_minimum_bowl_aspect_ratio", 0.70)
            if camera == "wrist"
            else 0.0
        )
        return fraction <= bowl_maximum and aspect >= minimum_aspect

    def _candidate_appearance_allowed(
        self,
        name: str,
        item: BBoxObject,
        image: Image.Image,
        image_path: Path | None,
    ) -> bool:
        camera = (
            image_path.parents[1].name
            if image_path is not None and len(image_path.parents) >= 2
            else ""
        )
        if camera != "wrist" or name != "cookie":
            return True
        x1, y1, x2, y2 = item.bbox
        left = max(0, min(image.width, int(np.floor(x1))))
        top = max(0, min(image.height, int(np.floor(y1))))
        right = max(0, min(image.width, int(np.ceil(x2))))
        bottom = max(0, min(image.height, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            return False
        patch = np.asarray(image.convert("RGB"))[top:bottom, left:right].astype(float)
        red, green, blue = (patch[:, :, index] for index in range(3))
        orange = (
            (red > 150)
            & (red > green * 1.15)
            & (green > 55)
            & (green < 200)
            & (green > blue * 1.25)
            & (blue < 140)
        )
        minimum = getattr(self, "wrist_minimum_cookie_orange_fraction", 0.008)
        return float(orange.mean()) >= minimum

    def _objects_from_result(
        self,
        image: Image.Image,
        image_path: Path | None,
        result: dict[str, Any],
        allowed_objects: frozenset[str] | None = None,
        filter_minimum: float | None = None,
    ) -> dict[str, BBoxObject]:
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", []))
        boxes = self._refine_boxes(image, boxes)
        candidates: dict[str, list[BBoxObject]] = {}
        for box, score, label in zip(boxes, scores, labels):
            canonical = _canonical_label(str(label))
            if canonical not in CANONICAL_OBJECTS:
                continue
            if allowed_objects is not None and canonical not in allowed_objects:
                continue
            item = BBoxObject(
                bbox=tuple(float(value) for value in box),
                confidence=float(score),
                tracking_confidence=float(score),
                visible=True,
            )
            if not self._candidate_area_allowed(canonical, item, image, image_path):
                continue
            if not self._candidate_appearance_allowed(
                canonical, item, image, image_path
            ):
                continue
            candidates.setdefault(canonical, []).append(item)

        objects = {
            name: max(items, key=lambda item: item.confidence)
            for name, items in candidates.items()
            if name != "black_bowl"
        }
        bowl = self._select_bowl(
            candidates.get("black_bowl", []),
            objects,
            self._task_from_image_path(image_path),
        )
        if bowl is not None:
            objects["black_bowl"] = bowl
        return self._filter_objects(objects, image, image_path, filter_minimum)

    def detect_batch(
        self,
        images: list[Image.Image],
        image_paths: list[Path | None] | None = None,
        *,
        prompt: str = PROMPT,
        allowed_objects: frozenset[str] | None = None,
        threshold: float | None = None,
        text_threshold: float | None = None,
        filter_minimum: float | None = None,
    ) -> list[dict[str, BBoxObject]]:
        paths = image_paths or [None] * len(images)
        inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image.size[::-1] for image in images], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=self.box_threshold if threshold is None else float(threshold),
            text_threshold=(
                self.text_threshold if text_threshold is None else float(text_threshold)
            ),
            target_sizes=target_sizes,
        )
        return [
            self._objects_from_result(
                image, path, result, allowed_objects, filter_minimum
            )
            for image, path, result in zip(images, paths, results)
        ]

    def detect_bowl_candidates_batch(
        self,
        images: list[Image.Image],
        *,
        threshold: float = 0.05,
        minimum_area_fraction: float = 0.002,
        maximum_area_fraction: float = 0.04,
    ) -> list[list[BBoxObject]]:
        """Return every plausible proposal from a bowl-only prompt.

        Grounding DINO may leave the text label empty for low-confidence but
        geometrically accurate proposals. Because the prompt contains only the
        bowl class, those proposals are intentionally retained here.
        """
        inputs = self.processor(
            images=images,
            text=[BOWL_PROMPT] * len(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor(
            [image.size[::-1] for image in images], device=self.device
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=float(threshold),
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )
        output: list[list[BBoxObject]] = []
        for image, result in zip(images, results):
            boxes = self._refine_boxes(image, result["boxes"].detach().cpu().numpy())
            scores = result["scores"].detach().cpu().numpy()
            image_area = max(1.0, float(image.width * image.height))
            candidates: list[BBoxObject] = []
            for box, score in zip(boxes, scores):
                item = _clip_object(BBoxObject(
                    bbox=tuple(float(value) for value in box),
                    confidence=float(score),
                    tracking_confidence=float(score),
                    visible=True,
                    source="grounding_dino_low_threshold_candidate",
                ), image.width, image.height)
                if item is None:
                    continue
                area_fraction = bbox_area(item.bbox) / image_area
                if not minimum_area_fraction <= area_fraction <= maximum_area_fraction:
                    continue
                if any(bbox_iou(item.bbox, kept.bbox) >= 0.95 for kept in candidates):
                    continue
                candidates.append(item)
            output.append(sorted(candidates, key=lambda item: item.confidence, reverse=True))
        return output

    def detect(self, image: Image.Image, image_path: Path | None = None) -> dict[str, BBoxObject]:
        return self.detect_batch([image], [image_path])[0]


def _read_existing_frames(path: Path) -> list[BBoxFrame]:
    if not path.exists():
        return []
    records: list[BBoxFrame] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(BBoxFrame.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print(f"Ignoring malformed bbox checkpoint line {line_number}: {path}", file=sys.stderr)
    return records


def _load_batch(samples: list[SampledFrameRecord]) -> tuple[list[Image.Image], list[Path]]:
    paths = [Path(sample.image_path) for sample in samples]
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    return images, paths


def _clip_object(item: BBoxObject, width: int, height: int) -> BBoxObject | None:
    x1, y1, x2, y2 = item.bbox
    clipped = (
        min(max(float(x1), 0.0), float(width)),
        min(max(float(y1), 0.0), float(height)),
        min(max(float(x2), 0.0), float(width)),
        min(max(float(y2), 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return replace(item, bbox=clipped)


def _direct_record(
    sample: SampledFrameRecord,
    objects: dict[str, BBoxObject],
    detector_name: str,
    source: str,
) -> BBoxFrame:
    clipped = {
        name: value
        for name, item in objects.items()
        if (value := _clip_object(replace(item, source=source), sample.width, sample.height))
        is not None
    }
    return BBoxFrame(
        task=sample.task,
        episode=sample.episode,
        frame=sample.frame,
        timestamp=sample.timestamp,
        camera=sample.camera,
        width=sample.width,
        height=sample.height,
        objects=clipped,
        detector=detector_name,
    )


def _compose_agent_record(
    sample: SampledFrameRecord,
    reference: BBoxFrame,
    bowl: BBoxObject | None,
    bowl_source: str,
    detector_name: str,
) -> BBoxFrame:
    objects: dict[str, BBoxObject] = {}
    for name, item in reference.objects.items():
        if name not in STATIC_OBJECTS:
            continue
        clipped = _clip_object(
            replace(item, source="agent_static_reference"), sample.width, sample.height
        )
        if clipped is not None:
            objects[name] = clipped
    if bowl is not None:
        clipped_bowl = _clip_object(
            replace(bowl, source=bowl_source), sample.width, sample.height
        )
        if clipped_bowl is not None:
            objects["black_bowl"] = clipped_bowl
    return BBoxFrame(
        task=sample.task,
        episode=sample.episode,
        frame=sample.frame,
        timestamp=sample.timestamp,
        camera=sample.camera,
        width=sample.width,
        height=sample.height,
        objects=objects,
        detector=detector_name,
    )


def _interpolate_bowl(
    frame: int,
    anchors: dict[int, tuple[BBoxObject, str]],
) -> tuple[BBoxObject | None, str]:
    exact = anchors.get(frame)
    if exact is not None:
        return exact
    previous = [number for number in anchors if number < frame]
    following = [number for number in anchors if number > frame]
    if previous and following:
        left_frame = max(previous)
        right_frame = min(following)
        left = anchors[left_frame][0]
        right = anchors[right_frame][0]
        alpha = (frame - left_frame) / max(1, right_frame - left_frame)
        bbox = tuple(
            float((1.0 - alpha) * left_value + alpha * right_value)
            for left_value, right_value in zip(left.bbox, right.bbox)
        )
        return (
            BBoxObject(
                bbox=bbox,
                confidence=0.75 * min(left.confidence, right.confidence),
                tracking_confidence=0.65
                * min(left.tracking_confidence, right.tracking_confidence),
                visible=left.visible and right.visible,
                source="agent_bowl_dynamic_interpolation",
            ),
            "agent_bowl_dynamic_interpolation",
        )
    if previous:
        left = anchors[max(previous)][0]
        return (
            replace(left, tracking_confidence=min(0.25, left.tracking_confidence)),
            "agent_bowl_dynamic_carry_low_confidence",
        )
    if following:
        right = anchors[min(following)][0]
        return (
            replace(right, tracking_confidence=min(0.25, right.tracking_confidence)),
            "agent_bowl_dynamic_backfill_low_confidence",
        )
    return None, "agent_bowl_missing"


def _export_detected(
    cache: DetectionCache,
    detector_name: str,
    artifacts: ArtifactStore,
) -> dict[tuple[str, str, int, str], BBoxFrame]:
    completed = cache.load(detector_name)
    artifacts.write_jsonl(
        "bboxes/detected.jsonl",
        (item.to_dict() for item in sorted(completed.values(), key=lambda value: value.key())),
    )
    return completed


def run_sampled_detection(
    sampled_index_path: str | Path,
    metadata_path: str | Path,
    episode_index_path: str | Path,
    artifacts: ArtifactStore,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    camera: str | None = None,
    task: str | None = None,
    episode: str | None = None,
) -> dict[str, Any]:
    detector_name = GroundedSam2Detector.cache_name(config)
    detector: GroundedSam2Detector | None = None

    def get_detector() -> GroundedSam2Detector:
        nonlocal detector
        if detector is None:
            detector = GroundedSam2Detector(config)
        return detector

    frames = [
        frame for frame in load_sampled_index(sampled_index_path)
        if (camera is None or frame.camera == camera)
        and (task is None or frame.task == task)
        and (episode is None or frame.episode == episode)
    ]
    episode_records = {
        (item.task, item.episode): item for item in load_episode_index(episode_index_path)
    }
    output_path = artifacts.path("bboxes/detected.jsonl")
    cache_path = artifacts.path("cache/detection.sqlite3")
    batch_size = max(1, int(config["vision"].get("batch_size", 4)))

    inference_counts = {
        "agent_reference": 0,
        "agent_dynamic_bowl": 0,
        "agent_anchor_recovery": 0,
        "wrist_full_frame": 0,
    }
    reused_legacy = 0
    newly_materialized = 0
    optical_flow_fills = 0
    release_recovery_count = 0
    release_recovery_flow_fills = 0

    with DetectionCache(cache_path) as cache:
        existing_records = _read_existing_frames(output_path)
        cache.upsert(existing_records)
        completed = cache.load(detector_name)
        legacy: dict[tuple[str, str, int, str], BBoxFrame] = {}
        for version in cache.detector_counts():
            if version.startswith("grounding-dino-sam2-v3"):
                legacy.update(cache.load(version))

        pending = [frame for frame in frames if frame.key() not in completed]
        if limit is not None:
            pending = pending[:limit]
        pending_keys = {frame.key() for frame in pending}
        signals_by_episode: dict[tuple[str, str], list[MetadataFrame]] = defaultdict(list)
        if any(frame.camera == "agent_view" for frame in frames):
            metadata = load_metadata_frames(str(metadata_path)) if Path(metadata_path).exists() else {}
            for signal in metadata.values():
                signals_by_episode[(signal.task, signal.episode)].append(signal)
        all_by_episode: dict[tuple[str, str, str], list[SampledFrameRecord]] = defaultdict(list)
        pending_by_episode: dict[tuple[str, str, str], list[SampledFrameRecord]] = defaultdict(list)
        for frame in frames:
            all_by_episode[(frame.task, frame.episode, frame.camera)].append(frame)
        for frame in pending:
            pending_by_episode[(frame.task, frame.episode, frame.camera)].append(frame)
        for values in all_by_episode.values():
            values.sort(key=lambda item: item.frame)

        try:
            agent_groups = sorted(key for key in pending_by_episode if key[2] == "agent_view")
            for chunk_start in range(0, len(agent_groups), batch_size):
                group_chunk = agent_groups[chunk_start:chunk_start + batch_size]
                references: dict[tuple[str, str, str], BBoxFrame] = {}
                missing_reference_samples: list[SampledFrameRecord] = []
                missing_reference_groups: list[tuple[str, str, str]] = []
                plans: dict[tuple[str, str, str], AgentEpisodePlan] = {}

                for key in group_chunk:
                    episode_frames = all_by_episode[key]
                    plan = plan_agent_episode(episode_frames, signals_by_episode.get(key[:2], []))
                    plans[key] = plan
                    reference_sample = next(
                        item for item in episode_frames if item.frame == plan.reference_frame
                    )
                    reference_key = reference_sample.key()
                    reference = completed.get(reference_key) or legacy.get(reference_key)
                    if reference is None:
                        missing_reference_samples.append(reference_sample)
                        missing_reference_groups.append(key)
                    else:
                        references[key] = reference
                        if reference.detector != detector_name:
                            reused_legacy += 1

                if missing_reference_samples:
                    images, paths = _load_batch(missing_reference_samples)
                    detections = get_detector().detect_batch(images, paths)
                    inference_counts["agent_reference"] += len(missing_reference_samples)
                    for key, sample, objects in zip(
                        missing_reference_groups, missing_reference_samples, detections
                    ):
                        references[key] = _direct_record(
                            sample, objects, detector_name, "agent_reference_detection"
                        )

                bowl_anchors: dict[tuple[str, str, str], dict[int, tuple[BBoxObject, str]]] = defaultdict(dict)
                required_by_group: dict[tuple[str, str, str], set[int]] = {}
                bowl_requests: list[SampledFrameRecord] = []
                bowl_request_groups: list[tuple[str, str, str]] = []
                for key in group_chunk:
                    plan = plans[key]
                    reference = references[key]
                    if "black_bowl" in reference.objects:
                        bowl_anchors[key][plan.reference_frame] = (
                            reference.objects["black_bowl"],
                            "agent_bowl_stationary_pregrasp",
                        )
                    sample_by_frame = {item.frame: item for item in all_by_episode[key]}
                    required = {
                        item.frame
                        for item in pending_by_episode[key]
                        if item.frame in plan.bowl_detection_frames
                    }
                    if plan.release_frame is not None and any(
                        item.frame >= plan.release_frame for item in pending_by_episode[key]
                    ):
                        required.add(plan.release_frame)
                    required_by_group[key] = required
                    model_frames = (
                        select_bowl_model_frames(
                            plan, int(config["vision"].get("dynamic_bowl_dino_stride", 2))
                        )
                        if required
                        else frozenset()
                    )
                    for frame_number in sorted(model_frames):
                        sample = sample_by_frame[frame_number]
                        existing = completed.get(sample.key()) or legacy.get(sample.key())
                        if existing is not None and "black_bowl" in existing.objects:
                            source = (
                                "agent_bowl_dynamic_cached"
                                if existing.detector == detector_name
                                else "agent_bowl_dynamic_reused_v3"
                            )
                            bowl_anchors[key][frame_number] = (
                                existing.objects["black_bowl"], source
                            )
                            if existing.detector != detector_name:
                                reused_legacy += 1
                        else:
                            bowl_requests.append(sample)
                            bowl_request_groups.append(key)

                for start in range(0, len(bowl_requests), batch_size):
                    request_batch = bowl_requests[start:start + batch_size]
                    group_batch = bowl_request_groups[start:start + batch_size]
                    images, paths = _load_batch(request_batch)
                    detections = get_detector().detect_batch(
                        images,
                        paths,
                        prompt=BOWL_PROMPT,
                        allowed_objects=frozenset({"black_bowl"}),
                    )
                    inference_counts["agent_dynamic_bowl"] += len(request_batch)
                    for key, sample, objects in zip(group_batch, request_batch, detections):
                        bowl = objects.get("black_bowl")
                        if bowl is not None:
                            source = (
                                "agent_bowl_release_detection"
                                if plans[key].release_frame == sample.frame
                                else "agent_bowl_dynamic_detection"
                            )
                            bowl_anchors[key][sample.frame] = (bowl, source)

                if bool(config["vision"].get("optical_flow_missing_bowl", True)):
                    minimum_flow_iou = float(
                        config["vision"].get("optical_flow_minimum_bidirectional_iou", 0.15)
                    )
                    for key in group_chunk:
                        anchors = bowl_anchors[key]
                        episode_record = episode_records.get(key[:2])
                        video = episode_record.videos.get("agent_view", {}) if episode_record else {}
                        if not video.get("exists"):
                            continue
                        for frame_number in sorted(required_by_group[key] - set(anchors)):
                            previous_frames = [number for number in anchors if number < frame_number]
                            following_frames = [number for number in anchors if number > frame_number]
                            if not previous_frames or not following_frames:
                                continue
                            previous_frame = max(previous_frames)
                            next_frame = min(following_frames)
                            tracked = track_missing_bowl(
                                video_path=video["path"],
                                video_start_timestamp=float(video.get("from_timestamp", 0.0)),
                                fps=float(episode_record.fps),
                                previous_frame=previous_frame,
                                target_frame=frame_number,
                                next_frame=next_frame,
                                previous=anchors[previous_frame][0],
                                following=anchors[next_frame][0],
                                minimum_bidirectional_iou=minimum_flow_iou,
                            )
                            if tracked is not None:
                                anchors[frame_number] = (
                                    tracked,
                                    "agent_bowl_dynamic_optical_flow",
                                )
                                optical_flow_fills += 1

                records: list[BBoxFrame] = []
                for key in group_chunk:
                    plan = plans[key]
                    reference = references[key]
                    samples_to_save = list(pending_by_episode[key])
                    reference_sample = next(
                        item for item in all_by_episode[key] if item.frame == plan.reference_frame
                    )
                    if reference_sample.key() not in completed:
                        samples_to_save.append(reference_sample)
                    if plan.release_frame is not None:
                        release_sample = next(
                            (item for item in all_by_episode[key] if item.frame == plan.release_frame),
                            None,
                        )
                        if release_sample is not None and release_sample.key() not in completed:
                            samples_to_save.append(release_sample)

                    unique_samples = {item.key(): item for item in samples_to_save}
                    anchors = bowl_anchors[key]
                    for sample in sorted(unique_samples.values(), key=lambda item: item.frame):
                        exact = anchors.get(sample.frame)
                        if exact is not None:
                            bowl, bowl_source = exact
                        elif plan.release_frame is not None and sample.frame >= plan.release_frame:
                            bowl, bowl_source = anchors.get(
                                plan.release_frame,
                                anchors.get(plan.reference_frame, (None, "agent_bowl_missing")),
                            )
                            if bowl is not None and bowl_source == "agent_bowl_missing":
                                bowl_source = "agent_bowl_stationary_postrelease"
                            elif bowl is not None:
                                bowl_source = "agent_bowl_stationary_postrelease"
                        elif plan.movement_start is None or sample.frame < plan.movement_start:
                            bowl, _ = anchors.get(
                                plan.reference_frame, (None, "agent_bowl_missing")
                            )
                            bowl_source = "agent_bowl_stationary_pregrasp"
                        else:
                            bowl, bowl_source = _interpolate_bowl(sample.frame, anchors)
                        record = _compose_agent_record(
                            sample, reference, bowl, bowl_source, detector_name
                        )
                        records.append(record)

                cache.upsert(records)
                for record in records:
                    completed[record.key()] = record
                newly_materialized += sum(1 for record in records if record.key() in pending_keys)
                print(
                    f"Agent view: {min(chunk_start + len(group_chunk), len(agent_groups))} / "
                    f"{len(agent_groups)} episodes; {len(completed)} frames checkpointed",
                    file=sys.stderr,
                    flush=True,
                )

            wrist_pending = [item for item in pending if item.camera == "wrist"]
            wrist_missing: list[SampledFrameRecord] = []
            wrist_reused: list[BBoxFrame] = []
            for sample in wrist_pending:
                previous = legacy.get(sample.key())
                if previous is None:
                    wrist_missing.append(sample)
                    continue
                wrist_reused.append(
                    _direct_record(
                        sample,
                        previous.objects,
                        detector_name,
                        "wrist_direct_detection_reused_v3",
                    )
                )
                reused_legacy += 1
            cache.upsert(wrist_reused)
            for record in wrist_reused:
                completed[record.key()] = record
            newly_materialized += len(wrist_reused)

            for start in range(0, len(wrist_missing), batch_size):
                request_batch = wrist_missing[start:start + batch_size]
                images, paths = _load_batch(request_batch)
                detections = get_detector().detect_batch(images, paths)
                records = [
                    _direct_record(
                        sample, objects, detector_name, "wrist_direct_detection"
                    )
                    for sample, objects in zip(request_batch, detections)
                ]
                cache.upsert(records)
                for record in records:
                    completed[record.key()] = record
                inference_counts["wrist_full_frame"] += len(records)
                newly_materialized += len(records)
                if (start + len(records)) % 100 == 0 or start + len(records) == len(wrist_missing):
                    print(
                        f"Wrist: {start + len(records)} / {len(wrist_missing)} new frames; "
                        f"{len(completed)} frames checkpointed",
                        file=sys.stderr,
                        flush=True,
                    )

            recovery_objects: dict[tuple[str, str, str], dict[str, BBoxObject]] = defaultdict(dict)
            unresolved_recovery: dict[str, int] = Counter()
            if camera in (None, "agent_view"):
                agent_keys = sorted(key for key in all_by_episode if key[2] == "agent_view")
                generic_attempts = range(3) if pending else range(0)
                for attempt in generic_attempts:
                    requests: list[SampledFrameRecord] = []
                    request_keys: list[tuple[str, str, str]] = []
                    missing_by_key: dict[tuple[str, str, str], set[str]] = {}
                    for key in agent_keys:
                        episode_frames = all_by_episode[key]
                        frame_records = [
                            completed[item.key()]
                            for item in episode_frames
                            if item.key() in completed
                        ]
                        if not frame_records:
                            continue
                        present_static = {
                            name
                            for record in frame_records
                            for name in record.objects
                            if name in STATIC_OBJECTS
                        } | set(recovery_objects[key])
                        missing = set(STATIC_OBJECTS) - present_static
                        bowl_missing_samples = [
                            item
                            for item in episode_frames
                            if item.key() in completed
                            and "black_bowl" not in completed[item.key()].objects
                            and "black_bowl" not in recovery_objects[key]
                        ]
                        if bowl_missing_samples:
                            missing.add("black_bowl")
                        if not missing:
                            continue
                        if bowl_missing_samples:
                            candidate_index = (
                                len(bowl_missing_samples) // 2
                                if attempt == 0
                                else (-1 if attempt == 1 else len(bowl_missing_samples) // 4)
                            )
                            candidate = bowl_missing_samples[candidate_index]
                        else:
                            candidate_index = (
                                -1
                                if attempt == 0
                                else (
                                    len(episode_frames) // 2
                                    if attempt == 1
                                    else len(episode_frames) // 4
                                )
                            )
                            candidate = episode_frames[candidate_index]
                        requests.append(candidate)
                        request_keys.append(key)
                        missing_by_key[key] = missing

                    for start in range(0, len(requests), batch_size):
                        request_batch = requests[start:start + batch_size]
                        key_batch = request_keys[start:start + batch_size]
                        images, paths = _load_batch(request_batch)
                        recovery_threshold = float(
                            config["vision"].get("recovery_box_threshold", 0.18)
                        )
                        use_low_threshold = attempt == 2 or not pending
                        detections = get_detector().detect_batch(
                            images,
                            paths,
                            threshold=recovery_threshold if use_low_threshold else None,
                            filter_minimum=recovery_threshold if use_low_threshold else None,
                        )
                        inference_counts["agent_anchor_recovery"] += len(request_batch)
                        for key, objects in zip(key_batch, detections):
                            for name in missing_by_key[key]:
                                if name in objects:
                                    source = (
                                        "agent_bowl_reference_recovery"
                                        if name == "black_bowl"
                                        else "agent_static_recovery"
                                    )
                                    recovery_objects[key][name] = replace(
                                        objects[name], source=source
                                    )

                targeted_threshold = float(
                    config["vision"].get("targeted_recovery_box_threshold", 0.12)
                )
                for target_name in sorted(CANONICAL_OBJECTS):
                    requests = []
                    request_keys = []
                    for key in agent_keys:
                        episode_frames = all_by_episode[key]
                        frame_records = [
                            completed[item.key()]
                            for item in episode_frames
                            if item.key() in completed
                        ]
                        if target_name in STATIC_OBJECTS:
                            already_present = any(
                                target_name in item.objects for item in frame_records
                            ) or target_name in recovery_objects[key]
                            if already_present:
                                continue
                            candidate = episode_frames[-1]
                        else:
                            missing_samples = [
                                item
                                for item in episode_frames
                                if item.key() in completed
                                and target_name not in completed[item.key()].objects
                            ]
                            if not missing_samples or target_name in recovery_objects[key]:
                                continue
                            candidate = missing_samples[len(missing_samples) // 2]
                        requests.append(candidate)
                        request_keys.append(key)

                    for start in range(0, len(requests), batch_size):
                        request_batch = requests[start:start + batch_size]
                        key_batch = request_keys[start:start + batch_size]
                        images, paths = _load_batch(request_batch)
                        detections = get_detector().detect_batch(
                            images,
                            paths,
                            prompt=RECOVERY_PROMPTS[target_name],
                            allowed_objects=frozenset({target_name}),
                            threshold=targeted_threshold,
                            filter_minimum=targeted_threshold,
                        )
                        inference_counts["agent_anchor_recovery"] += len(request_batch)
                        for key, objects in zip(key_batch, detections):
                            if target_name not in objects:
                                continue
                            source = (
                                "agent_bowl_reference_recovery"
                                if target_name == "black_bowl"
                                else "agent_static_recovery"
                            )
                            recovery_objects[key][target_name] = replace(
                                objects[target_name], source=source
                            )

                repaired_records: list[BBoxFrame] = []
                for key in agent_keys:
                    recovered = recovery_objects.get(key, {})
                    for sample in all_by_episode[key]:
                        record = completed.get(sample.key())
                        if record is None:
                            continue
                        objects: dict[str, BBoxObject] = {}
                        for name, item in record.objects.items():
                            clipped = _clip_object(item, record.width, record.height)
                            if clipped is not None:
                                objects[name] = clipped
                        for name, item in recovered.items():
                            if name in STATIC_OBJECTS or name not in objects:
                                clipped = _clip_object(item, record.width, record.height)
                                if clipped is not None:
                                    objects[name] = clipped
                        updated = replace(record, objects=objects)
                        if updated != record:
                            repaired_records.append(updated)
                            completed[updated.key()] = updated
                cache.upsert(repaired_records)

                release_requests: list[SampledFrameRecord] = []
                release_request_keys: list[tuple[str, str, str]] = []
                release_plans: dict[tuple[str, str, str], AgentEpisodePlan] = {}
                for key in agent_keys:
                    plan = plan_agent_episode(
                        all_by_episode[key], signals_by_episode.get(key[:2], [])
                    )
                    if plan.release_frame is None:
                        continue
                    release_sample = next(
                        (
                            item
                            for item in all_by_episode[key]
                            if item.frame == plan.release_frame
                        ),
                        None,
                    )
                    release_record = completed.get(release_sample.key()) if release_sample else None
                    release_source = (
                        release_record.objects["black_bowl"].source
                        if release_record and "black_bowl" in release_record.objects
                        else "missing"
                    )
                    if release_source in {
                        "agent_bowl_release_detection",
                        "agent_bowl_release_recovery",
                    }:
                        continue
                    release_requests.append(all_by_episode[key][-1])
                    release_request_keys.append(key)
                    release_plans[key] = plan

                recovered_release: dict[tuple[str, str, str], BBoxObject] = {}
                recovery_threshold = float(
                    config["vision"].get("recovery_box_threshold", 0.18)
                )
                for start in range(0, len(release_requests), batch_size):
                    request_batch = release_requests[start:start + batch_size]
                    key_batch = release_request_keys[start:start + batch_size]
                    images, paths = _load_batch(request_batch)
                    detections = get_detector().detect_batch(
                        images,
                        paths,
                        threshold=recovery_threshold,
                        filter_minimum=recovery_threshold,
                    )
                    inference_counts["agent_anchor_recovery"] += len(request_batch)
                    for key, objects in zip(key_batch, detections):
                        if "black_bowl" in objects:
                            recovered_release[key] = replace(
                                objects["black_bowl"], source="agent_bowl_release_recovery"
                            )

                release_updates: list[BBoxFrame] = []
                for key, release_bowl in recovered_release.items():
                    plan = release_plans[key]
                    episode_frames = all_by_episode[key]
                    frame_records = {
                        item.frame: completed[item.key()]
                        for item in episode_frames
                        if item.key() in completed
                    }
                    trustworthy = {
                        frame: record.objects["black_bowl"]
                        for frame, record in frame_records.items()
                        if "black_bowl" in record.objects
                        and record.objects["black_bowl"].source
                        != "agent_bowl_dynamic_carry_low_confidence"
                        and frame < plan.release_frame
                    }
                    trustworthy[plan.release_frame] = release_bowl
                    episode_record = episode_records.get(key[:2])
                    video = episode_record.videos.get("agent_view", {}) if episode_record else {}
                    for frame, record in sorted(frame_records.items()):
                        bowl = record.objects.get("black_bowl")
                        replacement: BBoxObject | None = None
                        if frame >= plan.release_frame:
                            replacement = replace(
                                release_bowl,
                                source=(
                                    "agent_bowl_release_recovery"
                                    if frame == plan.release_frame
                                    else "agent_bowl_stationary_postrelease_recovery"
                                ),
                            )
                        elif bowl and bowl.source == "agent_bowl_dynamic_carry_low_confidence":
                            previous_frames = [value for value in trustworthy if value < frame]
                            next_frames = [value for value in trustworthy if value > frame]
                            if previous_frames and next_frames and video.get("exists"):
                                previous_frame = max(previous_frames)
                                next_frame = min(next_frames)
                                replacement = track_missing_bowl(
                                    video_path=video["path"],
                                    video_start_timestamp=float(
                                        video.get("from_timestamp", 0.0)
                                    ),
                                    fps=float(episode_record.fps),
                                    previous_frame=previous_frame,
                                    target_frame=frame,
                                    next_frame=next_frame,
                                    previous=trustworthy[previous_frame],
                                    following=trustworthy[next_frame],
                                    minimum_bidirectional_iou=float(
                                        config["vision"].get(
                                            "optical_flow_minimum_bidirectional_iou", 0.15
                                        )
                                    ),
                                )
                                if replacement is not None:
                                    replacement = replace(
                                        replacement,
                                        source="agent_bowl_dynamic_optical_flow_release_recovery",
                                    )
                                    release_recovery_flow_fills += 1
                            if replacement is None:
                                anchors = {
                                    value: (item, item.source)
                                    for value, item in trustworthy.items()
                                }
                                replacement, _ = _interpolate_bowl(frame, anchors)
                                if replacement is not None:
                                    replacement = replace(
                                        replacement,
                                        source="agent_bowl_dynamic_release_interpolation",
                                    )
                        if replacement is None:
                            continue
                        objects = dict(record.objects)
                        clipped = _clip_object(replacement, record.width, record.height)
                        if clipped is None:
                            continue
                        objects["black_bowl"] = clipped
                        updated = replace(record, objects=objects)
                        completed[updated.key()] = updated
                        release_updates.append(updated)
                    release_recovery_count += 1
                cache.upsert(release_updates)

                for key in agent_keys:
                    frame_records = [
                        completed[item.key()]
                        for item in all_by_episode[key]
                        if item.key() in completed
                    ]
                    for name in STATIC_OBJECTS:
                        if not any(name in item.objects for item in frame_records):
                            unresolved_recovery[name] += 1
                    if any("black_bowl" not in item.objects for item in frame_records):
                        unresolved_recovery["black_bowl_frames"] += 1
        finally:
            completed = _export_detected(cache, detector_name, artifacts)

        report = {
            "detector": detector_name,
            "requested_frames": len(frames),
            "target_pending_frames": len(pending),
            "new_frames": newly_materialized,
            "completed_frames": len(completed),
            "inferences": inference_counts,
            "reused_legacy_frames": reused_legacy,
            "optical_flow_fills": optical_flow_fills,
            "release_recovery": {
                "episodes": release_recovery_count,
                "optical_flow_fills": release_recovery_flow_fills,
                "unresolved_episodes": len(release_requests) - len(recovered_release),
            },
            "anchor_recovery": {
                "recovered_episode_objects": dict(
                    Counter(
                        name
                        for objects in recovery_objects.values()
                        for name in objects
                    )
                ),
                "unresolved": dict(unresolved_recovery),
            },
            "cache": str(cache_path),
            "cache_counts": cache.detector_counts(),
            "agent_strategy": {
                "static_objects": "one fixed-camera reference detection per episode",
                "black_bowl": "metadata-guided held/lifted and release detections",
                "metadata_fallback": "per-sampled-frame bowl detection",
            },
            "wrist_strategy": "full per-frame detection because the camera moves",
            "sam2_refinement": detector is not None and detector.sam_predictor is not None,
            "filters": {"camera": camera, "task": task, "episode": episode},
        }
        artifacts.write_json("reports/detection_report.json", report)
        return report
