"""Automatic Arrow-teacher test-time training for LIBERO.

The package keeps runtime and serialized contracts separate.  ``TransitionRecord``
is the atomic executed-step record; the dataset row is ``TrainingTransition``.
The declarative multi-VLA configuration is ``ExperimentConfig`` while the
single-task runtime helper is ``EpisodeRunConfig``.
"""

from .contracts import (
    ACTION_DIM, ACTION_MAX, ACTION_MIN, SCHEMA_VERSION, ActionChunk, Actor, ContractError,
    EpisodeSpec, EpisodeStatus, LiveEnvironment, RecoveryTeacher, SourceState,
    TeacherRecoveryRequest, TeacherRecoveryResult, TransitionRecord,
    assert_student_observation, normalize_action_chunk, serialize_records,
    validate_action,
)
from .config import (
    ExperimentConfig, FIDELITY_MODES, PAPER_KNOWN_SETTINGS, ProvenanceConfig,
    SplitConfig, VLA_ALIASES, VLA_DISPLAY_NAMES, VLA_NAMES, load_config,
    save_config, STUDY_PROTOCOL_DEFAULTS, validate_study_protocol,
)
from .dataset import (
    CANONICAL_OBSERVATION_SCHEMA, PEFT_EXPORT_METHOD, EpisodeRecord, ObservationSchemaError,
    TTTDataset, TransitionRecord as TrainingTransition,
    episode_from_executed_records, validate_student_observation_schema,
)
from .episode import EpisodeCoordinator, RolloutResult
from .experiment import (
    EnvironmentFactory, EnvironmentHandle, EpisodeOutcome, EpisodeRunConfig,
    Policy, SplitManifest, Updater, collect_from_config, initial_state_hash,
    run_episode, summarize, wilson_interval,
)
from .recording import EpisodeManifest, ExperimentProvenance, JSONLTransitionWriter
from .teacher import (
    ArrowGraspControllerTeacher, PrivilegedTakeoverEnvironmentView,
    TakeoverEnvironmentView, validate_teacher_transitions,
)
from .fidelity import (
    ExactFidelityError, PAPER_ID, PAPER_URL, ReferenceArtifactManifest,
    REQUIRED_ARTIFACT_FIELDS, RUNTIME_REQUIRED_FIELDS, ExactRuntimeAttestation,
    RuntimeAttestation,
    capture_runtime_attestation, validate_runtime_receipt,
)
from .model import FastState, TTTKVBConfig, build_ttt_kvb_module, model_parameter_metadata
from .training import (
    OptimizerMetadata, PRETRAIN_METADATA, POSTTRAIN_METADATA,
    RoboTTTTrainer, TBPTTState, TrainingContractError, build_optimizer,
    build_scheduler, flow_matching_target, masked_action_targets,
    masked_flow_matching_loss, run_tbptt_raw_segments, run_tbptt_segments,
    sample_flow_matching_tau, set_parameter_mask, train_from_config,
)
from .evaluation import TrialMetric, evaluate_from_config, paired_report, write_metrics
from .adapters import (
    AdaptableVLA, AdapterMetadata, AdapterRegistration, LegacyPolicyBridge,
    SUPPORTED_VLAS, VLAAdapterRegistry,
)
from .arrow_bridge import ArrowCanaryBridge
from .live_collection import (
    COLLECTION_SCHEMA, PEFT_METHOD_LABEL, SOURCE_KIND, CanonicalLiveEnvironment,
    CollectionResult, canonical_student_observation, collect_task_corrections,
    export_correction_only_lerobot_dataset,
)
from .demonstrations import (
    DemonstrationRecord, DemonstrationReceipt, DemonstrationValidationError,
    ValidatedDemonstration, validate_and_build_demonstration,
)
from .protocol import (
    ADAPTATION_TASKS, CHECKPOINTS, SEEDS, TRANSFER_TASKS, CostReceipt,
    ProtocolError, ProtocolLockReceipt, ResetStateEntry, RoboTTTProtocol,
    count_accepted_arrow_trajectories, make_default_protocol, require_arrow_target,
)
from .statistics import (
    BootstrapResult, PairedOutcome, StatisticsError, hierarchical_bootstrap,
    macro_task_success_delta, paired_success_delta, validate_paired_alignment,
)
from .provenance import (
    AUDIT_SCHEMA, ZeroShotAuditAttestation, ZeroShotEvidence,
    ZeroShotProvenanceError, validate_zero_shot_attestation,
    verify_zero_shot_evidence,
)
from .runtime_host import (
    EnvironmentIdentity, EnvironmentLease, EnvironmentView, OperationResult,
    OperationStatus, RuntimeContractError, RuntimeHost, RuntimeHostError,
    RuntimePreflightReceipt, RuntimeUnavailableError, UnavailableRuntimeFactory,
    observation_hash,
)
from .costs import (
    CostAccountingError, CostAggregate, CostMeter, PhaseReceipt, SensorReading,
    TrialCostAggregate,
)
from .peft_artifacts import (
    PEFTArtifactError, PEFTArtifactManifest, load_peft_manifest, save_peft_adapter,
    sha256_file, sha256_path, tree_sha256,
)

__all__ = [
    "ACTION_DIM", "ACTION_MAX", "ACTION_MIN", "SCHEMA_VERSION", "ActionChunk", "Actor",
    "ContractError", "EpisodeSpec", "EpisodeStatus", "LiveEnvironment",
    "RecoveryTeacher", "SourceState", "TeacherRecoveryRequest",
    "TeacherRecoveryResult", "TransitionRecord", "assert_student_observation",
    "normalize_action_chunk", "serialize_records", "validate_action",
    "ExperimentConfig", "FIDELITY_MODES", "PAPER_KNOWN_SETTINGS",
    "STUDY_PROTOCOL_DEFAULTS", "validate_study_protocol",
    "ProvenanceConfig", "SplitConfig", "VLA_ALIASES", "VLA_DISPLAY_NAMES",
    "VLA_NAMES", "load_config", "save_config", "CANONICAL_OBSERVATION_SCHEMA", "PEFT_EXPORT_METHOD",
    "EpisodeRecord", "ObservationSchemaError", "TTTDataset", "TrainingTransition",
    "episode_from_executed_records", "validate_student_observation_schema",
    "EpisodeCoordinator", "RolloutResult", "EnvironmentFactory", "EnvironmentHandle",
    "EpisodeOutcome", "EpisodeRunConfig", "Policy", "SplitManifest", "Updater",
    "collect_from_config", "initial_state_hash", "run_episode", "summarize",
    "wilson_interval", "EpisodeManifest", "ExperimentProvenance",
    "JSONLTransitionWriter", "ArrowGraspControllerTeacher",
    "PrivilegedTakeoverEnvironmentView", "TakeoverEnvironmentView",
    "validate_teacher_transitions", "ExactFidelityError", "PAPER_ID", "PAPER_URL",
    "ReferenceArtifactManifest", "REQUIRED_ARTIFACT_FIELDS", "RUNTIME_REQUIRED_FIELDS",
    "RuntimeAttestation", "ExactRuntimeAttestation", "capture_runtime_attestation", "validate_runtime_receipt",
    "FastState", "TTTKVBConfig", "build_ttt_kvb_module",
    "model_parameter_metadata", "OptimizerMetadata", "PRETRAIN_METADATA",
    "POSTTRAIN_METADATA", "RoboTTTTrainer", "TBPTTState", "TrainingContractError",
    "build_optimizer", "build_scheduler", "flow_matching_target",
    "masked_action_targets", "masked_flow_matching_loss", "run_tbptt_raw_segments",
    "run_tbptt_segments", "sample_flow_matching_tau", "set_parameter_mask",
    "train_from_config", "TrialMetric", "evaluate_from_config", "paired_report",
    "write_metrics", "AdaptableVLA", "AdapterMetadata", "AdapterRegistration",
    "LegacyPolicyBridge", "SUPPORTED_VLAS", "VLAAdapterRegistry", "ArrowCanaryBridge",
    "COLLECTION_SCHEMA", "SOURCE_KIND", "PEFT_METHOD_LABEL", "CanonicalLiveEnvironment",
    "CollectionResult", "canonical_student_observation", "collect_task_corrections",
    "export_correction_only_lerobot_dataset",
    "DemonstrationRecord", "DemonstrationReceipt", "DemonstrationValidationError",
    "ValidatedDemonstration", "validate_and_build_demonstration", "ADAPTATION_TASKS",
    "CHECKPOINTS", "SEEDS", "TRANSFER_TASKS", "CostReceipt", "ProtocolError",
    "ProtocolLockReceipt", "ResetStateEntry", "RoboTTTProtocol", "make_default_protocol",
    "count_accepted_arrow_trajectories", "require_arrow_target",
    "BootstrapResult", "PairedOutcome", "StatisticsError", "hierarchical_bootstrap",
    "macro_task_success_delta", "paired_success_delta", "validate_paired_alignment",
    "AUDIT_SCHEMA", "ZeroShotAuditAttestation", "ZeroShotEvidence", "ZeroShotProvenanceError",
    "validate_zero_shot_attestation", "verify_zero_shot_evidence", "EnvironmentIdentity",
    "EnvironmentLease", "EnvironmentView", "OperationResult", "OperationStatus",
    "RuntimeContractError", "RuntimeHost", "RuntimeHostError", "RuntimePreflightReceipt",
    "RuntimeUnavailableError", "UnavailableRuntimeFactory", "observation_hash",
    "CostAccountingError", "CostAggregate", "CostMeter", "PhaseReceipt", "SensorReading",
    "TrialCostAggregate",
    "PEFTArtifactError", "PEFTArtifactManifest", "load_peft_manifest", "save_peft_adapter",
    "sha256_file", "sha256_path", "tree_sha256",
]
