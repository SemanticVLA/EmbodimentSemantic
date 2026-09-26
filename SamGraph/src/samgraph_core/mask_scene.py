"""Automatic public-RGB mask scene construction for LIBERO arrow tasks."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from .geometric_graph import (
    GeometricRelationConfig,
    GEOMETRIC_RELATION_RULES_REVISION,
    PIXEL_FRAME,
    build_directed_relations,
    convex_hull_mask,
    direction_relation,
    geometric_relation_rules,
    geometric_relation_revision,
    geometric_relation_rules_sha256,
    mask_centroid,
    mask_iou,
)


@dataclass(frozen=True, slots=True)
class Concept:
    class_id: str
    prompt: str
    role: str
    fallbacks: tuple[str, ...] = ()
    multiple: bool = False
    required_count: int | None = 1


SPATIAL_CATALOG = (
    Concept("black_bowl", "small metal bowl", "bowl", ("metal bowl", "bowl object"), True, 2),
    Concept("cookies", "small box", "support", ("food box",)),
    Concept("plate", "dinner plate", "support", ("round plate",)),
    Concept("white_ramekin", "small metal bowl which is white", "support"),
    Concept("flat_stove", "white stove", "support", ("burner",)),
    Concept("wooden_cabinet", "wooden cabinet", "cabinet", ("dark wooden cabinet",)),
)

INVERSE = {
    "is_left_of": "is_right_of",
    "is_right_of": "is_left_of",
    "is_in_front_of": "is_behind",
    "is_behind": "is_in_front_of",
    "is_on_top_of": "is_below_of",
    "is_below_of": "is_on_top_of",
    "is_inside": "contains",
    "contains": "is_inside",
}


def encode_mask(mask: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_mask(value: str) -> np.ndarray:
    try:
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(value, validate=True))).convert("L"))
    except Exception as exc:
        raise ValueError("mask scene contains an invalid PNG") from exc
    mask = np.ascontiguousarray(pixels > 0, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask scene requires nonempty binary masks")
    return mask


def _mask_from_result(value: Any, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(io.BytesIO(value.png))
    channel = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
    mask = np.ascontiguousarray(np.asarray(channel, dtype=np.uint8) > 0, dtype=bool)
    if mask.shape != shape or not mask.any():
        raise ValueError("segmentation mask does not match the submitted RGB frame")
    return mask


def _centroid(mask: np.ndarray) -> np.ndarray:
    return mask_centroid(mask)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    return mask_iou(left, right)


def _rgb_std(rgb: np.ndarray, mask: np.ndarray) -> float:
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    return float(np.asarray(pixels, dtype=np.float32).std()) if pixels.size else 0.0


def _hull(mask: np.ndarray) -> np.ndarray:
    return convex_hull_mask(mask)


def _direction(left: np.ndarray, right: np.ndarray) -> str:
    # Compatibility wrapper for historical callers.  New graph code uses the
    # versioned shared contract directly.
    return direction_relation(left, right)


def _object_terms(instruction: str) -> tuple[str, str]:
    normalized = " ".join(str(instruction).lower().split())
    match = re.fullmatch(
        r"pick(?: up)? the (.+?) and place it (?:on top of|on|in) (?:the )?(.+)",
        normalized,
    )
    if match is None:
        # Match the multi-word preposition first.  Otherwise the ``on`` branch
        # consumes the prefix of ``on top of`` and turns the destination into
        # ``top of ...``.
        match = re.fullmatch(r"put the (.+?) (?:on top of|on|in) (?:the )?(.+)", normalized)
    if match is None:
        raise ValueError("Object instruction does not match the supported place grammar")
    return match.group(1), match.group(2)


class MaskSceneInitializer:
    """Build a complete directed scene graph using only RGB segmentation masks."""

    def __init__(
        self,
        segmentation_service: Any,
        *,
        geometry_rules: GeometricRelationConfig | dict[str, Any] | None = None,
    ) -> None:
        self._segmenter = segmentation_service
        self._geometry_rules = geometry_rules

    @staticmethod
    def _encode_rgb(rgb: np.ndarray) -> bytes:
        output = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output, format="PNG")
        return output.getvalue()

    def _segment(self, rgb: np.ndarray, concept: Concept) -> list[tuple[np.ndarray, float]]:
        encoded = self._encode_rgb(rgb)
        prompts = (concept.prompt, *concept.fallbacks)
        for prompt in prompts:
            result = self._segmenter.segment(prompt=prompt, jpeg=encoded)
            masks = [(_mask_from_result(item, rgb.shape[:2]), float(item.score)) for item in result.masks]
            if masks:
                masks.sort(key=lambda item: (-item[1], -int(item[0].sum())))
                return masks if concept.multiple else masks[:1]
        return []

    def initialize(self, rgb: np.ndarray, *, instruction: str, suite: str,
                   visual_inventory: bool = False,
                   allow_partial_inventory: bool = False) -> dict[str, Any]:
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] != value.shape[1]:
            raise ValueError("LIBERO graph initialization requires square HxWx3 RGB")
        height, width = value.shape[:2]
        rules = geometric_relation_rules(self._geometry_rules)
        if suite == "spatial":
            catalog = SPATIAL_CATALOG
        elif suite == "object" and visual_inventory:
            # Perception-only mode inventories every returned visual instance;
            # it does not pretend a can's appearance establishes its task name.
            # Validated against actual Object RGB with all seven visible items.
            catalog = (Concept("visual_object", "small object", "visual_entity",
                               multiple=True, required_count=None),)
        elif suite == "object":
            source, destination = _object_terms(instruction)
            catalog = (
                Concept(re.sub(r"[^a-z0-9]+", "_", source).strip("_"), source, "object"),
                Concept(re.sub(r"[^a-z0-9]+", "_", destination).strip("_"), destination, "destination"),
            )
        else:
            raise ValueError(f"unsupported LIBERO suite: {suite!r}")

        segmented: dict[str, list[tuple[np.ndarray, float]]] = {
            concept.class_id: self._segment(value, concept) for concept in catalog
        }
        missing = [concept.class_id for concept in catalog if not segmented[concept.class_id]]
        if missing and not allow_partial_inventory:
            raise ValueError(f"public-RGB segmentation missed required classes: {missing}")

        # The broad metal-bowl prompt also sees the white ramekin. Remove that
        # cross-prompt duplicate using only the returned visual masks.
        if suite == "spatial":
            white_items = segmented["white_ramekin"]
            white = white_items[0][0] if white_items else None
            black = [
                item for item in segmented["black_bowl"]
                if (white is None or _iou(item[0], white) < rules["quality"]["mask_overlap_iou_max"])
                and _rgb_std(value, item[0]) >= rules["quality"]["black_bowl_rgb_std_min"]
            ]
            segmented["black_bowl"] = black

        wrong_counts = {
            concept.class_id: len(segmented[concept.class_id])
            for concept in catalog
            if concept.required_count is not None and (
                len(segmented[concept.class_id]) > concept.required_count
                if allow_partial_inventory
                else len(segmented[concept.class_id]) != concept.required_count
            )
        }
        if wrong_counts:
            raise ValueError(f"public-RGB segmentation inventory is ambiguous or incomplete: {wrong_counts}")

        retained_count = sum(len(items) for items in segmented.values())
        if allow_partial_inventory and retained_count < 2:
            raise ValueError(
                "partial public-RGB segmentation requires at least two retained visual entities"
            )

        inventory_completeness = {
            concept.class_id: {
                "expected_count": concept.required_count,
                "observed_count": len(segmented[concept.class_id]),
                "complete": (
                    concept.required_count is None
                    or len(segmented[concept.class_id]) == concept.required_count
                ),
                "missing": (
                    concept.required_count is not None
                    and len(segmented[concept.class_id]) < concept.required_count
                ),
            }
            for concept in catalog
        }
        expected_instances = sum(
            concept.required_count or 0 for concept in catalog
        )
        observed_classes = sum(bool(segmented[concept.class_id]) for concept in catalog)
        inventory_coverage = {
            "expected_class_count": len(catalog),
            "observed_class_count": observed_classes,
            "class_fraction": observed_classes / len(catalog) if catalog else 1.0,
            "expected_instance_count": expected_instances,
            "observed_instance_count": retained_count,
            "instance_fraction": (
                retained_count / expected_instances if expected_instances else 1.0
            ),
        }

        records: list[dict[str, Any]] = []
        for concept in catalog:
            items = segmented[concept.class_id]
            items.sort(key=lambda item: tuple(_centroid(item[0])))
            for index, (mask, score) in enumerate(items, start=1):
                center = _centroid(mask)
                records.append({
                    "instance_id": f"{concept.class_id}_{index}",
                    "class_id": concept.class_id,
                    "role": concept.role,
                    "prompt": concept.prompt,
                    "score": score,
                    "area_px": int(mask.sum()),
                    "center_xy": center.tolist(),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "mask_png_base64": encode_mask(mask),
                })

        masks = {item["instance_id"]: decode_mask(item["mask_png_base64"]) for item in records}
        centers = {item["instance_id"]: np.asarray(item["center_xy"], dtype=float) for item in records}
        for index, left in enumerate(records):
            for right in records[index + 1:]:
                overlap = _iou(masks[left["instance_id"]], masks[right["instance_id"]])
                if overlap >= rules["quality"]["mask_overlap_iou_max"]:
                    raise ValueError("ambiguous overlapping masks from separate visual instances")
        _relations, triplets, relation_evidence = build_directed_relations(
            records,
            masks,
            suite=suite,
            instruction=instruction,
            shape=value.shape[:2],
            geometry_rules=self._geometry_rules,
            allow_partial_visibility=allow_partial_inventory,
        )
        return {
            "schema": "samgraph.mask_scene.v1",
            "width": int(width),
            "height": int(height),
            "pixel_frame": PIXEL_FRAME,
            "rgb_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "instances": records,
            "triplets": triplets,
            "rules": rules,
            "rules_sha256": geometric_relation_rules_sha256(self._geometry_rules),
            "production_inputs": list(getattr(self._segmenter, "production_inputs", ("public_rgb", "sam3_text_masks"))),
            "initialization_provider": getattr(self._segmenter, "provider", "sam3"),
            "initialization_provenance": dict(getattr(self._segmenter, "provenance", {})),
            "relation_revision": geometric_relation_revision(self._geometry_rules),
            "relation_evidence": relation_evidence,
            "initial_mask_quality_gate": {
                "revision": "public_rgb_mask_quality_gate_v1",
                "min_black_bowl_rgb_std": (
                    rules["quality"]["black_bowl_rgb_std_min"] if suite == "spatial" else None
                ),
                "status": "PASS_PARTIAL" if allow_partial_inventory else "PASS",
            },
            "simulator_geometry_consumed": False,
            "inventory_mode": (
                "public_rgb_visual_instances" if visual_inventory
                else "semantic_catalog_partial" if allow_partial_inventory
                else "semantic_catalog"
            ),
            "semantic_task_identity_established": not visual_inventory,
            "allow_partial_inventory": bool(allow_partial_inventory),
            "inventory_completeness": inventory_completeness,
            "inventory_coverage": inventory_coverage,
            "missing_class_ids": [
                class_id for class_id, status in inventory_completeness.items()
                if status["missing"]
            ],
        }


# Compatibility for callers of the historical SAM3-only initializer name.
Sam3MaskSceneInitializer = MaskSceneInitializer


__all__ = [
    "Concept",
    "SPATIAL_CATALOG",
    "Sam3MaskSceneInitializer",
    "MaskSceneInitializer",
    "decode_mask",
    "encode_mask",
]
