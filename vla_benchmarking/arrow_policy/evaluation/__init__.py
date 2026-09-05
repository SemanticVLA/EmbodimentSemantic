"""Sealed 100-cell evaluation for the language-free ArrowStudent policy.

The package is intentionally a removable experiment corner.  It owns the
student inference seam and protocol binding while the existing matrix module
owns durable cell bookkeeping and resume semantics.
"""

from .reference_protocol import ReferenceProtocol, load_reference_protocol
from .runner import ArrowStudentEpisodeRunner, ArrowStudentRuntime

__all__ = [
    "ArrowStudentEpisodeRunner",
    "ArrowStudentRuntime",
    "ReferenceProtocol",
    "load_reference_protocol",
]
