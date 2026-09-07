"""Configuration and provenance contracts for the LIBERO automatic-TTT study.

This module intentionally has no PyTorch, LIBERO, robosuite, or model imports.  It
is safe to use on a login node when constructing or auditing an experiment.  The
default mode is ``exact``: training/evaluation must stop unless the released
RoboTTT artifact and all unresolved paper details are supplied in a manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

FIDELITY_MODES = ("exact", "algorithmic_port")
VLA_NAMES = ("openvla", "pi05", "smolvla", "ours")
VLA_DISPLAY_NAMES = {
    "openvla": "OpenVLA",
    "pi05": "Pi0.5",
    "smolvla": "SmolVLA",
    "ours": "Ours/Arrow-compatible",
}
VLA_ALIASES = {
    "open_vla": "openvla",
    "openvla": "openvla",
    "pi0.5": "pi05",
    "pi05": "pi05",
    "smolvla": "smolvla",
    "smol_vla": "smolvla",
    "ours": "ours",
    "arrow": "ours",
    "arrow_compatible": "ours",
}

# Frozen study-design values.  These are explicit proposed LIBERO settings; the
# NVIDIA paper's DAgger section reports a 100-trajectory pool (50 from each of
# two base policies), but does not prescribe our LIBERO task/round layout.
STUDY_PROTOCOL_DEFAULTS: dict[str, Any] = {
    "protocol_id": "automatic-robottt-libero-v1",
    "paper_dagger_pool_trajectories": 100,
    "paper_dagger_trajectories_per_base_policy": 50,
    "accepted_arrow_corrections_per_round": 100,
    "adaptation_tasks": [0, 2, 4, 6, 8],
    "transfer_tasks": [1, 3, 5, 7, 9],
    "adaptation_seeds": [17, 29, 43],
    "rounds": 3,
    "attempts_per_task_per_round": 20,
    "query_episodes_per_task": 50,
    "query_checkpoints": [0, 1, 2, 3],
    "outer_training_steps_per_round": 20000,
    "context_window": 1000,
    "effective_batch_size": 8,
    "expected_rollouts_per_arm": 6000,
    "bootstrap_resamples": 2000,
    "improvement_target": 0.10,
}

# These values are reported in the paper.  A null field is deliberate: it marks
# a detail that cannot be reconstructed from the paper alone and therefore must
# come from the authors' released artifact before an ``exact`` run is allowed.
PAPER_KNOWN_SETTINGS: dict[str, Any] = {
    "paper": "RoboTTT: Test-Time Training for Generalizable Robot Manipulation",
    "paper_identifier": "arXiv:2607.15275v1",
    "ttt_layer_placement": "after_self_and_cross_attention_each_dit_layer",
    "num_dit_layers": 16,
    "register_tokens_per_timestep": 16,
    "gate": "tanh(alpha) * ttt_output + attention_output",
    "gate_alpha_initialization": 0.001,
    "fast_loss": "mean_squared_error(f_W(K_t), V_t)",
    "post_update_query": "f_Wt(Q_t)",
    "flow_matching_path": "A_tau = tau*A + (1-tau)*epsilon",
    "flow_matching_target": "A-epsilon",
    "tau_sampling": "0.999*(1-Beta(1.5,1))",
    "pretrain_steps": 30000,
    "posttrain_steps": 20000,
    # Appendix A.2 reports per-device batch 1 on 8 GPUs for post-training;
    # 1K is the context length, not the batch size.
    "posttrain_global_batch_size": 8,
    "posttrain_context_length": 1000,
    "posttrain_num_gpus": 8,
    "pretrain_weight_decay": 1e-5,
    "posttrain_weight_decay": 1e-5,
    "pretrain_peak_learning_rate": 2e-5,
    "posttrain_peak_learning_rate": 5e-5,
    "optimizer": "AdamW",
    "fast_mlp_activation": "GELU",
    "fast_mlp_width": None,
    "ttt_projection_dim": None,
    "qkv_normalization": None,
    "learned_step_size_parameterization": None,
    "tbptt_segment_length": None,
    "action_horizon": None,
    "denoising_steps": None,
    "token_packing": None,
    "optimizer_betas_eps_clip": None,
    "image_crop_stride_preprocessing": None,
    "checkpoint": None,
    "checkpoint_sha256": None,
    "official_code_commit": None,
}

EXACT_REQUIRED_ARTIFACT_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "official_code_commit",
    "fast_mlp_width",
    "ttt_projection_dim",
    "qkv_normalization",
    "learned_step_size_parameterization",
    "tbptt_segment_length",
    "action_horizon",
    "denoising_steps",
    "token_packing",
    "optimizer_betas_eps_clip",
    "image_crop_stride_preprocessing",
)


@dataclass
class SplitConfig:
    """A fixed, disjoint split contract.

    Episode IDs are explicit rather than sampled implicitly.  ``train`` contains
    only collection episodes, while ``eval`` is never written to by collection
    or training.  The same IDs must be used for every VLA and condition.
    """

    train_episode_ids: list[str] = field(default_factory=list)
    validation_episode_ids: list[str] = field(default_factory=list)
    eval_episode_ids: list[str] = field(default_factory=list)
    seed: int = 1000

    def validate(self, *, task_ids: list[int] | None = None, episodes_per_task: int | None = None) -> list[str]:
        errors: list[str] = []
        allowed_tasks = set(task_ids) if task_ids is not None else None
        buckets = {
            "train": self.train_episode_ids,
            "validation": self.validation_episode_ids,
            "eval": self.eval_episode_ids,
        }
        seen: dict[str, str] = {}
        task_seed_seen: dict[tuple[int, int], str] = {}
        eval_task_counts: dict[int, int] = {}
        for bucket, ids in buckets.items():
            if len(ids) != len(set(ids)):
                errors.append(f"{bucket}_episode_ids contains duplicates")
            bucket_task_seeds: set[tuple[int, int]] = set()
            for episode_id in ids:
                previous = seen.get(episode_id)
                if previous is not None:
                    errors.append(f"episode {episode_id!r} occurs in {previous} and {bucket}")
                seen[episode_id] = bucket
                match = re.match(r"^task(?P<task>\d+)_seed(?P<seed>\d+)(?:_|$)", episode_id)
                if match:
                    key = (int(match.group("task")), int(match.group("seed")))
                    if bucket == "eval":
                        eval_task_counts[key[0]] = eval_task_counts.get(key[0], 0) + 1
                    if allowed_tasks is not None and key[0] not in allowed_tasks:
                        errors.append(
                            f"{bucket} episode id {episode_id!r} parses task {key[0]}, "
                            f"which is not present in task_ids {sorted(allowed_tasks)}"
                        )
                    if key in bucket_task_seeds:
                        errors.append(f"{bucket} repeats task/seed pair {key}; deterministic initial states would duplicate")
                    bucket_task_seeds.add(key)
                    previous = task_seed_seen.get(key)
                    if previous is not None and previous != bucket:
                        errors.append(f"task/seed pair {key} occurs in {previous} and {bucket}")
                    task_seed_seen[key] = bucket
                elif task_ids is not None:
                    errors.append(f"episode id {episode_id!r} must encode task and seed")
        if not self.eval_episode_ids:
            errors.append("eval_episode_ids must be explicit and non-empty")
        if task_ids is not None and episodes_per_task is not None:
            for task_id in task_ids:
                actual = eval_task_counts.get(task_id, 0)
                if actual != episodes_per_task:
                    errors.append(
                        f"eval task {task_id} has {actual} episode IDs; "
                        f"expected exactly {episodes_per_task}"
                    )
            extra_tasks = sorted(set(eval_task_counts) - set(task_ids))
            if extra_tasks:
                errors.append(f"eval_episode_ids contains unconfigured tasks: {extra_tasks}")
        return errors


@dataclass
class ProvenanceConfig:
    """Paths and immutable identifiers that must be copied into every run."""

    source_repo: str = "unknown"
    source_commit: str = "unknown"
    libero_commit: str = "unknown"
    robosuite_version: str = "unknown"
    python_version: str = "unknown"
    cuda_version: str = "unknown"
    controller_config: str = ""
    vla_checkpoints: dict[str, str] = field(default_factory=dict)
    vla_checkpoint_sha256: dict[str, str] = field(default_factory=dict)
    reference_artifact_manifest: str = ""


@dataclass
class RuntimeInjectionConfig:
    """Import paths for real host-side LIBERO/model integrations.

    Values use ``python.module:callable`` notation.  No simulator, checkpoint,
    or fake adapter is constructed by this package.
    """

    collection_factory: str = ""
    training_factory: str = ""
    evaluation_factory: str = ""


@dataclass
class ExperimentConfig:
    """Complete experiment declaration consumed by preflight and run backends."""

    fidelity_mode: Literal["exact", "algorithmic_port"] = "exact"
    vla_names: list[str] = field(default_factory=lambda: list(VLA_NAMES))
    task_ids: list[int] = field(default_factory=lambda: list(range(10)))
    seeds: list[int] = field(default_factory=lambda: [1000])
    episodes_per_task: int = 10
    student_step_budget: int = 220
    teacher_step_budget: int = 1200
    output_root: str = "vla_benchmarking/libero/automatic_ttt/runs"
    dataset_root: str = "vla_benchmarking/libero/automatic_ttt/datasets"
    split: SplitConfig = field(default_factory=SplitConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    runtime: RuntimeInjectionConfig = field(default_factory=RuntimeInjectionConfig)
    paper_settings: dict[str, Any] = field(default_factory=lambda: dict(PAPER_KNOWN_SETTINGS))
    controls: list[str] = field(
        default_factory=lambda: [
            "frozen_baseline",
            "adapted",
            "hybrid",
            "correction_only",
            "full_failure_context",
            "shuffled_failure_context",
            "reset_fast_state",
            "gdn",
        ]
    )
    # Kept as a JSON mapping for backwards-compatible config files.  Validation
    # below is strict and checks every frozen design value; no host may infer a
    # missing count at execution time.
    study_protocol: dict[str, Any] = field(default_factory=lambda: dict(STUDY_PROTOCOL_DEFAULTS))
    zero_shot_audit_evidence: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.fidelity_mode not in FIDELITY_MODES:
            errors.append(f"fidelity_mode must be one of {FIDELITY_MODES}")
        normalized: list[str] = []
        for name in self.vla_names:
            canonical = VLA_ALIASES.get(name.lower())
            if canonical is None:
                errors.append(f"unknown VLA {name!r}; choose from {VLA_NAMES}")
            elif canonical not in normalized:
                normalized.append(canonical)
        if not normalized:
            errors.append("vla_names must contain at least one supported VLA")
        if not self.task_ids:
            errors.append("task_ids must not be empty")
        if len(self.task_ids) != len(set(self.task_ids)):
            errors.append("task_ids contains duplicates")
        if any(task < 0 for task in self.task_ids):
            errors.append("task_ids must be non-negative")
        if any(seed < 0 for seed in self.seeds):
            errors.append("seeds must be non-negative")
        for field_name in ("episodes_per_task", "student_step_budget", "teacher_step_budget"):
            if getattr(self, field_name) <= 0:
                errors.append(f"{field_name} must be positive")
        errors.extend(self.split.validate(task_ids=self.task_ids, episodes_per_task=self.episodes_per_task))
        if self.fidelity_mode == "exact":
            for field_name in (
                "source_commit",
                "libero_commit",
                "robosuite_version",
                "python_version",
                "cuda_version",
                "controller_config",
            ):
                value = getattr(self.provenance, field_name)
                if not value or value == "unknown":
                    errors.append(f"exact mode requires provenance.{field_name}")
        errors.extend(validate_vla_checkpoints(self.provenance, self.fidelity_mode, self.vla_names))
        errors.extend(validate_artifact_manifest(self.paper_settings, self.provenance, self.fidelity_mode))
        errors.extend(validate_study_protocol(self.study_protocol, self.task_ids))
        if self.fidelity_mode == "exact":
            canonical_names = {VLA_ALIASES.get(name.lower(), name.lower()) for name in self.vla_names}
            normalized_audits = {
                VLA_ALIASES.get(str(name).lower(), str(name).lower()): path
                for name, path in self.zero_shot_audit_evidence.items()
            }
            missing_audits = sorted(name for name in canonical_names if not normalized_audits.get(name))
            if missing_audits:
                errors.append(
                    "exact zero-shot mode requires audit evidence paths (host-verifies SHA-256) for: "
                    + ", ".join(missing_audits)
                )
        return errors

    def canonical_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["vla_names"] = sorted({VLA_ALIASES.get(v.lower(), v.lower()) for v in self.vla_names})
        return result

    def digest(self) -> str:
        encoded = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def validate_artifact_manifest(
    paper_settings: Mapping[str, Any],
    provenance: ProvenanceConfig,
    fidelity_mode: str,
) -> list[str]:
    """Return exact-fidelity blockers; never infer missing paper details."""

    if fidelity_mode != "exact":
        return []
    errors: list[str] = []
    manifest_path = provenance.reference_artifact_manifest
    if not manifest_path:
        return ["exact mode requires provenance.reference_artifact_manifest"]
    if not Path(manifest_path).is_file():
        return [f"reference artifact manifest does not exist: {manifest_path}"]

    # Use the canonical fidelity implementation for the security-critical
    # checks (checkpoint URI/hash, code commit, unresolved fields, and local
    # checkpoint bytes).  Do not duplicate those checks here: a future change
    # to the artifact contract must fail closed in this preflight as well.
    try:
        from .fidelity import ExactFidelityError, ReferenceArtifactManifest

        reference = ReferenceArtifactManifest.from_json(manifest_path)
        reference.require_exact()
    except ExactFidelityError as exc:
        errors.append(str(exc))
        return errors
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"could not parse reference artifact manifest: {exc}")
        return errors
    if reference.paper_id and "2607.15275" not in reference.paper_id and "2607.15275" not in reference.paper_url:
        errors.append(f"reference artifact is not RoboTTT arXiv:2607.15275: {reference.paper_id}")

    # Resolve the names used by this package to the official manifest names.
    # A caller may leave these config values null, but may not replace an
    # official value with an invented value.  This prevents a seemingly
    # complete config from silently drifting away from the paper artifact.
    resolved: dict[str, Any] = dict(reference.fields)
    resolved.update(
        {
            "checkpoint": reference.checkpoint_uri,
            "checkpoint_sha256": reference.checkpoint_sha256,
            "official_code_commit": reference.code_commit,
            "fast_mlp_width": reference.fields.get("fast_mlp_hidden_dim"),
            "ttt_projection_dim": reference.fields.get("qkv_dimensions"),
            "learned_step_size_parameterization": reference.fields.get("inner_learning_rate_parameterization"),
        }
    )
    for field_name in EXACT_REQUIRED_ARTIFACT_FIELDS:
        official_value = resolved.get(field_name)
        caller_value = paper_settings.get(field_name)
        if official_value is None or official_value == "":
            errors.append(f"official artifact does not resolve required field {field_name}")
            continue
        if caller_value is not None and caller_value != "" and caller_value != official_value:
            errors.append(
                f"paper_settings.{field_name}={caller_value!r} conflicts with official artifact value {official_value!r}"
            )
    return errors


def validate_vla_checkpoints(
    provenance: ProvenanceConfig,
    fidelity_mode: str,
    vla_names: Iterable[str],
) -> list[str]:
    """Validate checkpoint paths and optional SHA-256 identity records."""

    errors: list[str] = []
    canonical = {VLA_ALIASES.get(name.lower(), name.lower()) for name in vla_names}
    for name in canonical:
        checkpoint = provenance.vla_checkpoints.get(name)
        if not checkpoint:
            if fidelity_mode == "exact":
                errors.append(f"exact mode requires provenance.vla_checkpoints.{name}")
            continue
        # URI-backed model references are verified by the host launcher.  Any
        # other value is a local path and must already exist before execution.
        if "://" not in checkpoint:
            path = Path(checkpoint)
            if not path.exists():
                errors.append(f"VLA checkpoint for {name} does not exist: {checkpoint}")
                continue
            digest = provenance.vla_checkpoint_sha256.get(name)
            if fidelity_mode == "exact" and not digest:
                errors.append(f"exact mode requires provenance.vla_checkpoint_sha256.{name} for local checkpoint")
            if digest:
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    errors.append(f"VLA checkpoint digest for {name} must be a 64-character SHA-256")
                else:
                    try:
                        from .fidelity import verify_checkpoint_hash

                        verify_checkpoint_hash(path, digest)
                    except Exception as exc:
                        errors.append(f"VLA checkpoint hash validation failed for {name}: {exc}")
        elif name in provenance.vla_checkpoint_sha256:
            digest = provenance.vla_checkpoint_sha256[name]
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                errors.append(f"VLA checkpoint digest for {name} must be a 64-character SHA-256")
    return errors


def _merge_dataclass(dataclass_type: Any, value: Mapping[str, Any] | None) -> Any:
    value = dict(value or {})
    defaults = dataclass_type()
    for key, item in value.items():
        if not hasattr(defaults, key):
            raise ValueError(f"unknown {dataclass_type.__name__} field: {key}")
        setattr(defaults, key, item)
    return defaults


def config_from_dict(payload: Mapping[str, Any]) -> ExperimentConfig:
    """Build a typed configuration and reject typoed keys."""

    known = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown ExperimentConfig fields: {sorted(unknown)}")
    raw = dict(payload)
    raw["split"] = _merge_dataclass(SplitConfig, raw.get("split"))
    raw["provenance"] = _merge_dataclass(ProvenanceConfig, raw.get("provenance"))
    raw["runtime"] = _merge_dataclass(RuntimeInjectionConfig, raw.get("runtime"))
    provided_protocol = raw.get("study_protocol") or {}
    unknown_protocol = set(provided_protocol) - set(STUDY_PROTOCOL_DEFAULTS)
    if unknown_protocol:
        raise ValueError(f"unknown study_protocol fields: {sorted(unknown_protocol)}")
    protocol = dict(STUDY_PROTOCOL_DEFAULTS)
    protocol.update(provided_protocol)
    raw["study_protocol"] = protocol
    audits = raw.get("zero_shot_audit_evidence") or {}
    if not isinstance(audits, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in audits.items()):
        raise ValueError("zero_shot_audit_evidence must map VLA names to evidence paths")
    raw["zero_shot_audit_evidence"] = dict(audits)
    provided_settings = raw.get("paper_settings") or {}
    unknown_settings = set(provided_settings) - set(PAPER_KNOWN_SETTINGS)
    if unknown_settings:
        raise ValueError(f"unknown paper_settings fields: {sorted(unknown_settings)}")
    settings = dict(PAPER_KNOWN_SETTINGS)
    settings.update(provided_settings)
    raw["paper_settings"] = settings
    return ExperimentConfig(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("config JSON must contain an object")
    return config_from_dict(payload)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_study_protocol(protocol: Mapping[str, Any], task_ids: Sequence[int]) -> list[str]:
    """Validate the immutable paper-count/proposed-LIBERO study contract."""
    errors: list[str] = []
    unknown = set(protocol) - set(STUDY_PROTOCOL_DEFAULTS)
    if unknown:
        errors.append(f"study_protocol has unknown fields: {sorted(unknown)}")
    for key, expected in STUDY_PROTOCOL_DEFAULTS.items():
        if key not in protocol:
            errors.append(f"study_protocol.{key} is required; no hidden defaults are allowed")
            continue
        actual = protocol[key]
        if isinstance(expected, list):
            if not isinstance(actual, (list, tuple)) or list(actual) != expected:
                errors.append(f"study_protocol.{key} must equal the frozen value {expected!r}")
        elif isinstance(expected, float):
            if actual != expected:
                errors.append(f"study_protocol.{key} must equal {expected!r}")
        elif actual != expected:
            errors.append(f"study_protocol.{key} must equal {expected!r}")
    adaptation = set(protocol.get("adaptation_tasks", ()))
    transfer = set(protocol.get("transfer_tasks", ()))
    configured = set(task_ids)
    if adaptation | transfer != set(range(10)):
        errors.append("study_protocol adaptation/transfer tasks must partition LIBERO task IDs 0..9")
    if not configured.issubset(adaptation | transfer):
        errors.append("study_protocol does not cover one or more configured task_ids")
    if adaptation & transfer:
        errors.append("study_protocol adaptation/transfer tasks overlap")
    return errors
