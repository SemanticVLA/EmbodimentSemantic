"""Pi0.5 no-arrow fine-tuning and evaluation integration.

The module intentionally has no import-time dependency on LeRobot.  Importing
the contract and dataset helpers is safe on CPU-only development hosts; the
optional runtime is loaded only by :class:`Pi05Adapter.load`.
"""

from .contracts import (
    PI05_ARTIFACT,
    PI05_IO,
    PI05_TRAINING,
    PolicyArtifact,
    PolicyIOContract,
    TrainingSpec,
    micro_steps_for,
    optimizer_updates_for,
    validate_observation,
    validate_action_chunk,
)
from .policy import Pi05Adapter

__all__ = [
    "PI05_ARTIFACT",
    "PI05_IO",
    "PI05_TRAINING",
    "PolicyArtifact",
    "PolicyIOContract",
    "TrainingSpec",
    "optimizer_updates_for",
    "micro_steps_for",
    "Pi05Adapter",
    "validate_observation",
    "validate_action_chunk",
]
