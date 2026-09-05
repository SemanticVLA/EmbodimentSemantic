"""Shared LIBERO Spatial evaluation contracts and backends.

Policy packages own perception/action adapters; this package owns the
evaluation condition, task/seed matrix, manifest identity, and result
aggregation contracts used by the canonical grasp controller, fine-tuned VLA
policies, language-free ArrowStudent, and ordinary LeRobot policies.
"""

from .contracts import (
    DEFAULT_EPISODES_PER_TASK,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED_BASE,
    DEFAULT_TASK_IDS,
    PUBLIC_SUITE_MODES,
    SUITE_MODES,
    TEXT_CONTEXTS,
    VISUAL_INPUTS,
    EvaluationCell,
    EvaluationCondition,
    EvaluationResult,
    aggregate_results,
    build_task_seed_matrix,
    parse_suite_mode,
)
from .backends import (
    BackendCapabilities,
    DIRECT_MATRIX,
    LEROBOT,
    validate_backend_condition,
)
from .registry import (
    POLICIES,
    PolicyCapabilities,
    get_policy_capabilities,
    validate_policy_condition,
)

__all__ = [
    "DEFAULT_EPISODES_PER_TASK",
    "DEFAULT_RESOLUTION",
    "DEFAULT_SEED_BASE",
    "DEFAULT_TASK_IDS",
    "PUBLIC_SUITE_MODES",
    "SUITE_MODES",
    "TEXT_CONTEXTS",
    "VISUAL_INPUTS",
    "EvaluationCell",
    "EvaluationCondition",
    "EvaluationResult",
    "aggregate_results",
    "build_task_seed_matrix",
    "parse_suite_mode",
    "BackendCapabilities",
    "DIRECT_MATRIX",
    "LEROBOT",
    "validate_backend_condition",
    "POLICIES",
    "PolicyCapabilities",
    "get_policy_capabilities",
    "validate_policy_condition",
]
