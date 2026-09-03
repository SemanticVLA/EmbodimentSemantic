"""Project-local, lazy SAM3.1 image segmentation runtime.

This package intentionally contains no Omnis integration.  It exposes a small
typed boundary that keeps image validation and mask normalization testable on
machines without PyTorch or SAM3 installed.
"""

from .runtime import (
    DEFAULT_CHECKPOINT_SHA256,
    SAM3_SOURCE_COMMIT,
    Sam3Detection,
    Sam3Request,
    Sam3Result,
    Sam3Runtime,
    Sam3RuntimeConfig,
    Sam3RuntimeError,
    compute_file_sha256,
    validate_rgb_image,
    validate_threshold,
)

__all__ = [
    "DEFAULT_CHECKPOINT_SHA256",
    "SAM3_SOURCE_COMMIT",
    "Sam3Detection",
    "Sam3Request",
    "Sam3Result",
    "Sam3Runtime",
    "Sam3RuntimeConfig",
    "Sam3RuntimeError",
    "compute_file_sha256",
    "validate_rgb_image",
    "validate_threshold",
]
