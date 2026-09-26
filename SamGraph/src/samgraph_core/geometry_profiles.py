"""Canonical geometry profiles shared by SamGraph inference and replay.

The standalone SamGraph benchmark has one frozen mask-derived geometry
calibration.  Keeping the preset here prevents the predictor, CLI, and cached
mask tuner from silently drifting apart.  The ordinary SamGraph geometry config
remains defined in :mod:`geometric_graph` and is intentionally not changed by
this module.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE = "samgraph_spatial_mask_geometry"

# These values are the best current exploratory calibration from the cached
# 92-frame subset.  They are a single shared preset, not a candidate grid.
_SAMGRAPH_SPATIAL_MASK_GEOMETRY: dict[str, Any] = {
    "profile": SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE,
    "bbox_padding_x": 0.0,
    "bbox_padding_y": 0.0,
    "containment_iomin_threshold": 0.8,
    "direction_horizontal_axis": "x",
    "direction_horizontal_sign": 1,
    "direction_vertical_axis": "y",
    "direction_vertical_sign": -1,
    "direction_vertical_scale": 2.0,
    "direction_tie_axis": "vertical",
    "stack_order": "lower_vertical_is_top",
    "drawer_stack_order": "higher_vertical_is_inside",
    "drawer_mode": "instruction_overlap_cabinet_band",
}

# Expose an immutable view so callers cannot mutate the process-wide source of
# truth.  The accessor below is intentionally the only normal way to pass a
# profile into a caller that may need to add candidate overrides.
SAMGRAPH_SPATIAL_MASK_GEOMETRY: Mapping[str, Any] = MappingProxyType(
    _SAMGRAPH_SPATIAL_MASK_GEOMETRY
)


def samgraph_spatial_mask_geometry() -> dict[str, Any]:
    """Return a fresh copy of the canonical SamGraph geometry preset."""

    return dict(SAMGRAPH_SPATIAL_MASK_GEOMETRY)


__all__ = [
    "SAMGRAPH_SPATIAL_MASK_GEOMETRY_PROFILE",
    "SAMGRAPH_SPATIAL_MASK_GEOMETRY",
    "samgraph_spatial_mask_geometry",
]
