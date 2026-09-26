"""Review rendering for SO101 observed and remembered masks."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


COLORS = (
    (255, 60, 220), (30, 225, 240), (255, 130, 30),
    (30, 240, 80), (255, 225, 30), (60, 145, 255),
)


def render_mask_overlay(rgb: np.ndarray, masks: Mapping[str, np.ndarray],
                        states: list[Mapping[str, Any]]) -> np.ndarray:
    """Solid-fill observations and hatch remembered/estimated geometry."""
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    labels = []
    for index, state in enumerate(states):
        key = str(state.get("output_id") or state.get("semantic_id") or state["track_id"])
        if key not in masks:
            continue
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape != overlay.shape[:2] or not mask.any():
            continue
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        inside = mask.copy()
        inside[1:] &= mask[:-1]
        inside[:-1] &= mask[1:]
        inside[:, 1:] &= mask[:, :-1]
        inside[:, :-1] &= mask[:, 1:]
        if state.get("status") == "observed":
            overlay[mask] = (overlay[mask].astype(np.float32) * 0.76 + color * 0.24).astype(np.uint8)
        else:
            yy, xx = np.indices(mask.shape)
            overlay[mask & ((xx + yy) % 14 < 3)] = color.astype(np.uint8)
        overlay[mask & ~inside] = color.astype(np.uint8)
        labels.append((key, tuple(int(x) for x in color), str(state.get("status"))))
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, (key, color, status) in enumerate(labels):
        y = 4 + index * 14
        draw.rectangle((4, y + 2, 12, y + 10), fill=color)
        draw.text((16, y), f"{key}: {status}", fill=(255, 255, 255), stroke_width=2,
                  stroke_fill=(0, 0, 0))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


__all__ = ["render_mask_overlay"]
