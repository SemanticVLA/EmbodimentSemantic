"""Self-contained runtime contracts for the seven-policy Arrow benchmark."""

from .contracts import (
    ACTION_DIM, ACTION_MAX, ACTION_MIN, CANONICAL_OBSERVATION_SCHEMA, POLICY_IDS,
    ActionProposal, ContractError, EpisodeSnapshot, GeometryAnchors, ObservationFrame,
    PolicyDecision, StepRecord, assert_student_observation, clip_action, digest, state8,
    validate_action, validate_student_observation,
)
from .controller import InterruptibleTeacher, Policy, policy_identifier
from .runtime import (
    Coordinator, Environment, EpisodeRecord, EpisodeResult, EpisodeStats, EventTracker,
    PendingStep, ProgressTracker, RuntimeEvent, TransactionalCoordinator, make_proposal,
)
from .policies import (
    ApprenticePolicy, EditorPolicy, MinimalPolicy, OnCallPolicy, TogetherPolicy, make_policy,
)
from .fast import GraphFastCorrector
from .trace import TracePolicy, extract_state_route, warp_route
from .learning import state_only_routes
from .branching import BranchEnvironment, BranchResult, BranchRunner, MinimalBranchExecutor, StatefulPolicy
from .policies import (
    ApprenticePolicy, BranchOutcome, EditorPolicy, MinimalPolicy, OnCallPolicy,
    ResidualPolicy, TogetherPolicy, make_policy,
)
from .fast import FastPolicy, GraphFastCorrector
from .trace import RoutePoint, TracePolicy, TraceRoute, extract_state_route, warp_route
from .native_learned_training import (
    NativeResidualReceipt, NativeResidualTrainingConfig, build_residual_batch,
    default_residual_features, load_native_residual, train_editor, train_minimal_learned,
    train_native_residual,
)
from .learning import (
    DATASET_VIEW_SCHEMA, TRANSITION_SCHEMA, DatasetManifest, DatasetView, InterventionRow,
    eligible_intervention_rows, fit_with_callback, intervention_rows, manifest_for_rows,
    shared_transition_filter, state_only_routes, transition_eligibility, write_dataset_view,
)
from .splits import (
    ResetIdentity, SplitError, SplitManifest, assert_disjoint, build_split_manifest,
    read_split_manifest, validate_split_manifests, validate_complete_split_manifests,
    validate_study_split_manifests, split_set_sha256, write_split_manifest,
)
from .artifacts import (
    ArtifactError, ArtifactRef, LineageManifest, canonical_bytes, read_lineage_manifest, verify_artifact,
    write_artifact, write_json_artifact, write_jsonl_artifact, write_lineage_manifest,
)
from .libero_adapter import (
    AdapterUnavailableError, LiberoEnvironmentAdapter, LiberoSnapshot,
    canonicalize_observation,
)
from .smolvla_adapter import SmolVLAAdapter, SmolVLASnapshot, SmolVLAInference
from .arrow_adapter import ArrowAdapter, ArrowSnapshot, InterruptibleArrowController, UnavailableArrowAdapter
from .interruptible_arrow import ArrowAvailability, ArrowPerceptionUnavailable, InterruptibleArrow
from .native_host import NativeHost, NativeHostSnapshot, NativeStep, TeacherStatus
from .native_runtime import (
    NativeCanaryError, NativeCanaryReceipt, NativeEnvironment, NativeEnvironmentFactory,
    NativeStepReceipt, NativeTeacher, NativeTeacherFactory, NativeVLA, NativeVLAFactory,
    run_native_canary, run_native_host_canary,
)
from .config import FrozenStudyManifest, ProtocolSeal, StudyConfig
from .native_factory import NativeHostSpec, action_selector_for, build_native_host, build_policy
from .native_legion_factory import (
    build_host as build_native_legion_host,
    build_minimal_branch_runner,
)
from .native_arrow_teacher import PerFrameArrowTeacher, build_rgbd_perception
from .native_executor import NativeExecutionReceipt, execute_native, import_callable, production_preflight
from .benchmark import (
    DEFAULT_CASES, EvaluationRow, PolicyCase, evaluate_paired_suite,
    evaluate_suite, rank_rows, row_from_result,
)
from .collection import (
    AttemptIdentity, CollectionArchive, CollectionAttempt, CollectionManifest, MasterLogWriter,
    TraceEpisode, TraceView, collect_on_call, derive_trace_view, make_collection_archive,
    write_collection_archive, write_master_log,
)
from .reporting import aggregate_rows, aggregate_evaluation_rows, paired_report, write_report

__all__ = [
    "ACTION_DIM", "ACTION_MAX", "ACTION_MIN", "CANONICAL_OBSERVATION_SCHEMA", "POLICY_IDS",
    "ActionProposal", "BranchEnvironment",
    "BranchResult", "BranchRunner", "MinimalBranchExecutor", "StatefulPolicy", "ContractError", "Coordinator", "Environment",
    "EpisodeRecord", "EpisodeResult", "EpisodeSnapshot", "EpisodeStats", "EventTracker",
    "InterruptibleTeacher", "ObservationFrame", "PendingStep", "Policy", "PolicyDecision",
    "ProgressTracker", "RuntimeEvent", "StepRecord", "TransactionalCoordinator", "make_proposal",
    "ApprenticePolicy", "EditorPolicy", "MinimalPolicy", "OnCallPolicy", "TogetherPolicy", "make_policy",
    "GraphFastCorrector", "TracePolicy", "extract_state_route", "state_only_routes", "warp_route",
    "policy_identifier", "assert_student_observation", "clip_action", "digest", "state8",
    "validate_action", "validate_student_observation", "GeometryAnchors",
    "ApprenticePolicy", "BranchOutcome", "EditorPolicy", "MinimalPolicy", "OnCallPolicy",
    "ResidualPolicy", "TogetherPolicy", "make_policy", "FastPolicy", "GraphFastCorrector",
    "RoutePoint", "TracePolicy", "TraceRoute", "extract_state_route", "warp_route",
    "NativeResidualReceipt", "NativeResidualTrainingConfig", "build_residual_batch",
    "default_residual_features", "load_native_residual", "train_editor", "train_minimal_learned", "train_native_residual",
    "TRANSITION_SCHEMA", "DATASET_VIEW_SCHEMA", "DatasetManifest", "DatasetView", "InterventionRow",
    "fit_with_callback", "intervention_rows", "eligible_intervention_rows", "manifest_for_rows",
    "shared_transition_filter", "transition_eligibility", "write_dataset_view", "state_only_routes",
    "ResetIdentity", "SplitError", "SplitManifest", "assert_disjoint", "build_split_manifest", "read_split_manifest",
    "validate_split_manifests", "validate_complete_split_manifests", "validate_study_split_manifests",
    "split_set_sha256", "write_split_manifest", "ArtifactError", "ArtifactRef", "LineageManifest",
    "canonical_bytes", "verify_artifact", "write_artifact", "write_json_artifact", "write_jsonl_artifact", "read_lineage_manifest", "write_lineage_manifest",
    "AdapterUnavailableError", "LiberoEnvironmentAdapter", "LiberoSnapshot", "canonicalize_observation",
    "SmolVLAAdapter", "SmolVLASnapshot", "SmolVLAInference", "ArrowAdapter", "ArrowSnapshot",
    "InterruptibleArrowController", "UnavailableArrowAdapter",
    "ArrowAvailability", "ArrowPerceptionUnavailable", "InterruptibleArrow",
    "NativeHost", "NativeHostSnapshot", "NativeStep", "TeacherStatus",
    "NativeCanaryError", "NativeCanaryReceipt", "NativeEnvironment", "NativeEnvironmentFactory",
    "NativeStepReceipt", "NativeTeacher", "NativeTeacherFactory", "NativeVLA", "NativeVLAFactory",
    "run_native_canary", "run_native_host_canary",
    "StudyConfig", "FrozenStudyManifest", "ProtocolSeal", "NativeHostSpec", "action_selector_for",
    "build_native_host", "build_policy", "NativeExecutionReceipt", "execute_native", "import_callable",
    "production_preflight", "build_native_legion_host", "build_minimal_branch_runner", "PerFrameArrowTeacher", "build_rgbd_perception", "DEFAULT_CASES", "EvaluationRow", "PolicyCase", "evaluate_suite",
    "evaluate_paired_suite", "rank_rows", "row_from_result", "AttemptIdentity",
    "CollectionAttempt", "CollectionManifest", "CollectionArchive", "MasterLogWriter", "TraceEpisode",
    "TraceView", "collect_on_call", "derive_trace_view", "make_collection_archive", "write_collection_archive", "write_master_log",
    "aggregate_rows", "aggregate_evaluation_rows", "paired_report", "write_report",
]
