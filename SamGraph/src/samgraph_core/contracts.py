"""Small SamGraph graph value types needed by the SamGraph perception modules.

This package does not include or invoke an LLM graph selector.
"""

from dataclasses import dataclass


class GraphBrainError(ValueError):
    """Compatibility error type raised when a required graph input is absent."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class GraphTriplet:
    relation_id: str
    subject: str
    relation: str
    object: str
