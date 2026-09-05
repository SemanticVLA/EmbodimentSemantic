"""Canonical RGB-D grasp controller components.

This package is the sole active grasp policy.  It keeps MolmoPoint inference,
RGB-D geometry and gripper preshaping as small independently
testable modules while the episode/matrix engines remain internal adapters.
SAM and experimental treatment selection are intentionally not part of this
package.
"""

from .grasp_candidates import *  # noqa: F401,F403
from .molmopoint import (  # noqa: F401
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_POINTS,
    DEFAULT_PROMPT,
    DEFAULT_PROMPT_ID,
    MOLMOPOINT_DTYPE,
    MOLMOPOINT_MODEL_ID,
    MOLMOPOINT_MODEL_REVISION,
    MOLMOPOINT_TRANSFORMERS_VERSION,
    PROMPT_VARIANTS,
    MolmoImagePoint,
    MolmoPointRequest,
    MolmoPointResult,
    MolmoPointRuntime,
    MolmoPointRuntimeConfig,
    MolmoPointRuntimeError,
    build_mask_highlight,
)

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS", "DEFAULT_MAX_POINTS", "DEFAULT_PROMPT",
    "DEFAULT_PROMPT_ID", "MOLMOPOINT_DTYPE", "MOLMOPOINT_MODEL_ID",
    "MOLMOPOINT_MODEL_REVISION", "MOLMOPOINT_TRANSFORMERS_VERSION",
    "PROMPT_VARIANTS", "MolmoImagePoint", "MolmoPointRequest",
    "MolmoPointResult", "MolmoPointRuntime", "MolmoPointRuntimeConfig",
    "MolmoPointRuntimeError", "build_mask_highlight",
]
