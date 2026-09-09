"""Explicit, serializable study contract for paired policy comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import CANONICAL_OBSERVATION_SCHEMA, ContractError, _safe
from .splits import SplitError, SplitManifest, validate_study_split_manifests


@dataclass(frozen=True)
class StudyConfig:
    task_ids: tuple[int, ...] = tuple(range(10))
    test_reset_ids: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    validation_reset_ids: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    on_call_attempts_per_task: int = 50
    primary_horizon: int = 1200
    secondary_horizon: int = 280
    learned_seeds: tuple[int, ...] = (1000, 1001, 1002)
    trace_geometry_provider_revision: str = "unresolved"
    observation_schema: str = CANONICAL_OBSERVATION_SCHEMA
    split_manifest_schema: str = "arrow_policy_suite.reset_split.v1"
    artifact_lineage_schema: str = "arrow_policy_suite.artifact_lineage.v1"
    create_only_artifacts: bool = True
    # All four revisions are deliberately separate: changing any one of them
    # invalidates comparability with a previously sealed study.
    model_revision: str = "unresolved"
    controller_revision: str = "unresolved"
    calibration_revision: str = "unresolved"
    environment_revision: str = "unresolved"
    # Explicit frozen pins for the policy/evaluation interface.  These fields
    # are appended to keep older positional StudyConfig constructors valid;
    # validation intentionally rejects their unresolved defaults.
    processor_revision: str = "unresolved"
    action_encoding: str = "unresolved"
    n_action_steps: int = 1
    camera_names: tuple[str, ...] = ()
    image_resolution: tuple[int, int] = (0, 0)
    depth_units: str = "unresolved"
    renderer_revision: str = "unresolved"
    metric_revision: str = "unresolved"
    # Predeclared policy protocol.  These values are part of the study seal so
    # a launcher cannot silently change the intervention while reusing split
    # identities.
    minimal_representative_variant: str = "runtime_oracle"
    together_pose_weight: float = 0.5
    on_call_progress_window: int = 20
    on_call_min_teacher_steps: int = 20
    on_call_progress_threshold: float = 0.10
    minimal_branch_hold_steps: int = 20
    fast_support_mode: str = "one_observation"
    fast_support_attempts: int = 1
    fast_feature_dim: int = 32
    fast_fast_parameter_count: int = 448
    trace_lookahead: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.task_ids, (str, bytes)):
            raise ContractError("task_ids must be a sequence of integers")
        tasks = tuple(int(task) for task in self.task_ids)
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(self, "learned_seeds", tuple(int(seed) for seed in self.learned_seeds))
        if isinstance(self.camera_names, (str, bytes)):
            camera_names = (str(self.camera_names),)
        else:
            camera_names = tuple(str(name) for name in self.camera_names)
        object.__setattr__(self, "camera_names", camera_names)
        if isinstance(self.image_resolution, (int, float)) and not isinstance(self.image_resolution, bool):
            resolution = (int(self.image_resolution), int(self.image_resolution))
        else:
            try:
                resolution = tuple(int(value) for value in self.image_resolution)
            except (TypeError, ValueError):
                resolution = ()
        object.__setattr__(self, "image_resolution", resolution)

        def normalize_reset_map(value: Mapping[int, Sequence[str]], name: str) -> Mapping[int, tuple[str, ...]]:
            if not isinstance(value, Mapping):
                raise ContractError(f"{name} must be a mapping from task id to reset IDs")
            normalized: dict[int, tuple[str, ...]] = {}
            for raw_task, raw_ids in value.items():
                task = int(raw_task)
                if isinstance(raw_ids, (str, bytes)):
                    raise ContractError(f"{name}[{task}] must contain reset IDs, not text")
                normalized[task] = tuple(str(reset_id) for reset_id in raw_ids)
            return MappingProxyType(normalized)

        object.__setattr__(self, "test_reset_ids", normalize_reset_map(self.test_reset_ids, "test_reset_ids"))
        object.__setattr__(self, "validation_reset_ids", normalize_reset_map(self.validation_reset_ids, "validation_reset_ids"))

    def validate(self) -> None:
        if self.observation_schema != CANONICAL_OBSERVATION_SCHEMA:
            raise ContractError("the canonical student observation schema is required")
        if self.split_manifest_schema != "arrow_policy_suite.reset_split.v1":
            raise ContractError("unsupported split manifest schema")
        if self.artifact_lineage_schema != "arrow_policy_suite.artifact_lineage.v1":
            raise ContractError("unsupported artifact lineage schema")
        if self.create_only_artifacts is not True:
            raise ContractError("experiment artifacts must be create-only")
        for name in (
            "model_revision", "controller_revision", "calibration_revision", "environment_revision",
            "trace_geometry_provider_revision", "processor_revision", "depth_units", "renderer_revision",
            "metric_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"unresolved", "unknown", "latest"}:
                raise ContractError(f"{name} must be an explicit frozen revision")
        if not isinstance(self.action_encoding, str) or not self.action_encoding.strip() or self.action_encoding.strip().lower() in {"unresolved", "unknown", "latest"}:
            raise ContractError("action_encoding must be an explicit frozen encoding")
        if self.n_action_steps != 1:
            raise ContractError("the causal policy contract requires n_action_steps=1")
        if not self.camera_names or any(not name.strip() for name in self.camera_names) or len(set(self.camera_names)) != len(self.camera_names):
            raise ContractError("camera_names must contain unique explicit camera names")
        if len(self.image_resolution) != 2 or any(value <= 0 for value in self.image_resolution):
            raise ContractError("image_resolution must be two positive dimensions")
        if self.primary_horizon < 1200 or self.secondary_horizon <= 0 or self.secondary_horizon > self.primary_horizon:
            raise ContractError("invalid evaluation horizons")
        if self.on_call_attempts_per_task != 50:
            raise ContractError("the paired pilot requires exactly 50 On-Call attempts per task")
        if self.minimal_representative_variant not in {"runtime_oracle", "learned"}:
            raise ContractError("minimal_representative_variant must be runtime_oracle or learned")
        if not 0.0 <= float(self.together_pose_weight) <= 1.0:
            raise ContractError("together_pose_weight must be in [0, 1]")
        if int(self.on_call_progress_window) != 20 or int(self.on_call_min_teacher_steps) != 20:
            raise ContractError("On-Call protocol requires a 20-step progress/takeover window")
        if abs(float(self.on_call_progress_threshold) - 0.10) > 1e-12:
            raise ContractError("On-Call progress threshold is frozen at 0.10")
        if int(self.minimal_branch_hold_steps) != 20:
            raise ContractError("Minimal branch hold is frozen at 20 steps")
        if self.fast_support_mode != "one_observation" or int(self.fast_support_attempts) != 1:
            raise ContractError("Fast primary protocol requires exactly one observation and one support attempt")
        if int(self.fast_feature_dim) != 32 or int(self.fast_fast_parameter_count) != 448:
            raise ContractError("Fast protocol requires feature_dim=32 and exactly 448 fast parameters")
        if int(self.trace_lookahead) != 2:
            raise ContractError("Trace lookahead is frozen at two route points")
        if len(set(self.task_ids)) != len(self.task_ids) or any(task < 0 for task in self.task_ids):
            raise ContractError("task_ids must be unique non-negative integers")
        unknown_test_tasks = set(self.test_reset_ids) - set(self.task_ids)
        unknown_validation_tasks = set(self.validation_reset_ids) - set(self.task_ids)
        if unknown_test_tasks or unknown_validation_tasks:
            raise ContractError(
                "reset identity manifests contain tasks outside task_ids: "
                f"test={sorted(unknown_test_tasks)} validation={sorted(unknown_validation_tasks)}"
            )
        all_test: list[str] = []
        all_validation: list[str] = []
        for task in self.task_ids:
            test = set(self.test_reset_ids.get(task, ()))
            validation = set(self.validation_reset_ids.get(task, ()))
            if len(test) != 10:
                raise ContractError(f"task {task} needs exactly ten frozen test reset identities")
            if test & validation:
                raise ContractError(f"task {task} reuses a test and validation reset identity")
            if len(validation) != 10:
                raise ContractError(f"task {task} needs ten frozen validation reset identities")
            all_test.extend(self.test_reset_ids.get(task, ()))
            all_validation.extend(self.validation_reset_ids.get(task, ()))
        if len(all_test) != len(set(all_test)):
            raise ContractError("test reset identity is reused across tasks")
        if len(all_validation) != len(set(all_validation)):
            raise ContractError("validation reset identity is reused across tasks")
        overlap = set(all_test) & set(all_validation)
        if overlap:
            raise ContractError(f"test and validation reset identities overlap: {sorted(overlap)[:3]}")

    def _canonical_payload(self) -> dict[str, Any]:
        return _safe(self.__dict__)

    def config_sha256(self) -> str:
        self.validate()
        encoded = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def identity_seal_sha256(self) -> str:
        self.validate()
        identity_payload = {
            "task_ids": self.task_ids,
            "test_reset_ids": self.test_reset_ids,
            "validation_reset_ids": self.validation_reset_ids,
            "model_revision": self.model_revision,
            "controller_revision": self.controller_revision,
            "calibration_revision": self.calibration_revision,
            "environment_revision": self.environment_revision,
            "processor_revision": self.processor_revision,
            "action_encoding": self.action_encoding,
            "n_action_steps": self.n_action_steps,
            "camera_names": self.camera_names,
            "image_resolution": self.image_resolution,
            "depth_units": self.depth_units,
            "renderer_revision": self.renderer_revision,
            "metric_revision": self.metric_revision,
        }
        encoded = json.dumps(_safe(identity_payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> dict[str, Any]:
        payload = self._canonical_payload()
        return {"schema": "arrow_policy_suite.study_config.v1", **payload,
                "config_sha256": self.config_sha256(),
                "identity_seal_sha256": self.identity_seal_sha256()}

    @staticmethod
    def verify_manifest(manifest: Mapping[str, Any]) -> None:
        """Verify a serialized config without trusting its supplied digests."""
        if not isinstance(manifest, Mapping) or manifest.get("schema") != "arrow_policy_suite.study_config.v1":
            raise ContractError("unsupported study config manifest schema")
        supplied_config = manifest.get("config_sha256")
        supplied_identity = manifest.get("identity_seal_sha256")
        payload = {key: value for key, value in manifest.items()
                   if key not in {"schema", "config_sha256", "identity_seal_sha256"}}
        encoded = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if supplied_config != hashlib.sha256(encoded).hexdigest():
            raise ContractError("study config_sha256 does not match canonical payload")
        kwargs = dict(payload)
        kwargs["task_ids"] = tuple(kwargs.get("task_ids", ()))
        kwargs["learned_seeds"] = tuple(kwargs.get("learned_seeds", ()))
        for name in ("test_reset_ids", "validation_reset_ids"):
            kwargs[name] = {int(key): tuple(value) for key, value in dict(kwargs.get(name, {})).items()}
        candidate = StudyConfig(**kwargs)
        if supplied_identity != candidate.identity_seal_sha256():
            raise ContractError("study identity seal does not match reset/revision identities")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "StudyConfig":
        cls.verify_manifest(manifest)
        kwargs = {key: value for key, value in manifest.items()
                  if key not in {"schema", "config_sha256", "identity_seal_sha256"}}
        kwargs["task_ids"] = tuple(kwargs.get("task_ids", ()))
        kwargs["learned_seeds"] = tuple(kwargs.get("learned_seeds", ()))
        for name in ("test_reset_ids", "validation_reset_ids"):
            kwargs[name] = {int(key): tuple(value) for key, value in dict(kwargs.get(name, {})).items()}
        return cls(**kwargs)


@dataclass(frozen=True)
class FrozenStudyManifest:
    """Sealed composition of config, reset splits, and training lineage.

    ``collection`` is accepted under the explicit ``collection`` split name
    or the historical ``train`` spelling.  Counts are fixed by the study
    contract: 50 collection, 10 validation, and 10 test identities per task.
    """

    config: StudyConfig
    collection: SplitManifest
    validation: SplitManifest
    test: SplitManifest
    training_manifest_digests: tuple[str, ...] = ()
    schema: str = "arrow_policy_suite.frozen_study.v1"
    composite_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != "arrow_policy_suite.frozen_study.v1":
            raise ContractError("unsupported frozen study manifest schema")
        if not isinstance(self.config, StudyConfig):
            raise ContractError("FrozenStudyManifest requires a StudyConfig")
        if not all(isinstance(item, SplitManifest) for item in (self.collection, self.validation, self.test)):
            raise ContractError("FrozenStudyManifest requires typed SplitManifest values")
        self.config.validate()
        # Reset identities are only valid for the exact sealed environment.
        # This prevents a split assembled from a different simulator build
        # from being accepted merely because its counts and hashes are valid.
        for split_name, manifest in (("collection", self.collection), ("validation", self.validation), ("test", self.test)):
            mismatched = sorted({identity.environment_fingerprint for identity in manifest.identities
                                 if identity.environment_fingerprint != self.config.environment_revision})
            if mismatched:
                raise ContractError(
                    f"{split_name} reset identities do not match frozen environment_revision "
                    f"{self.config.environment_revision!r}: {mismatched[:3]}"
                )
        try:
            validate_study_split_manifests(
                self.collection, self.validation, self.test, self.config.task_ids,
                collection_per_task=50, validation_per_task=10, test_per_task=10,
            )
        except SplitError as exc:
            raise ContractError(str(exc)) from exc
        # The typed validation/test manifests must agree with the reset-ID
        # maps carried by StudyConfig.  Compare the stable episode/reset IDs,
        # not list order, so serialization ordering cannot alter the seal.
        for split_name, manifest, expected_map in (
            ("validation", self.validation, self.config.validation_reset_ids),
            ("test", self.test, self.config.test_reset_ids),
        ):
            if expected_map:
                for task in self.config.task_ids:
                    expected = set(expected_map.get(task, ()))
                    actual = {identity.episode_id for identity in manifest.identities if identity.task_id == task}
                    if expected != actual:
                        raise ContractError(
                            f"{split_name} reset identities for task {task} do not match StudyConfig maps"
                        )
        digests = tuple(str(value) for value in self.training_manifest_digests)
        if not digests:
            raise ContractError("FrozenStudyManifest requires at least one retained training manifest digest")
        for value in digests:
            if len(value) != 64 or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
                raise ContractError("training manifest digests must be lowercase SHA-256 values")
        if len(set(digests)) != len(digests):
            raise ContractError("training manifest digests must be unique")
        object.__setattr__(self, "training_manifest_digests", digests)
        computed = _sha256_payload(self.payload())
        if self.composite_sha256 and self.composite_sha256 != computed:
            raise ContractError("frozen study composite digest does not match contents")
        object.__setattr__(self, "composite_sha256", computed)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": self.config.manifest(),
            "collection": self.collection.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "training_manifest_digests": list(self.training_manifest_digests),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "composite_sha256": self.composite_sha256}

    manifest = to_dict

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenStudyManifest":
        if not isinstance(value, Mapping) or value.get("schema") != "arrow_policy_suite.frozen_study.v1":
            raise ContractError("unsupported frozen study manifest schema")
        try:
            config = StudyConfig.from_manifest(value["config"])
            collection = SplitManifest.from_dict(value["collection"])
            validation = SplitManifest.from_dict(value["validation"])
            test = SplitManifest.from_dict(value["test"])
            digests = tuple(value.get("training_manifest_digests", ()))
            composite = str(value.get("composite_sha256", ""))
        except (KeyError, TypeError, ValueError, SplitError) as exc:
            raise ContractError("malformed frozen study manifest") from exc
        return cls(config, collection, validation, test, digests, str(value["schema"]), composite)

    @staticmethod
    def verify_manifest(value: Mapping[str, Any]) -> None:
        """Reconstruct and verify all nested config, split, and composite digests."""
        FrozenStudyManifest.from_dict(value)

    def write_artifact(self, path: str) -> Any:
        """Persist this seal using the package's create-only artifact writer."""
        from .artifacts import write_json_artifact

        return write_json_artifact(
            path, self.to_dict(), kind="frozen-study-manifest",
            lineage=self.training_manifest_digests,
        )

    @classmethod
    def read_artifact(cls, path: str | Any) -> "FrozenStudyManifest":
        from pathlib import Path
        import json

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("frozen study manifest artifact is unreadable") from exc
        return cls.from_dict(value)


@dataclass(frozen=True)
class ProtocolSeal:
    """Pre-collection seal that does not depend on learned artifacts.

    The old frozen-study artifact required a training-manifest digest, which
    made it impossible to seal the protocol before collecting the teacher
    data.  ``ProtocolSeal`` binds the configuration and reset identities first;
    the post-collection run manifest can then add training/derived artifact
    lineage without changing this seal.
    """

    config: StudyConfig
    collection: SplitManifest
    validation: SplitManifest
    test: SplitManifest
    schema: str = "arrow_policy_suite.protocol_seal.v1"
    protocol_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != "arrow_policy_suite.protocol_seal.v1":
            raise ContractError("unsupported protocol seal schema")
        self.config.validate()
        if not all(isinstance(item, SplitManifest) for item in (self.collection, self.validation, self.test)):
            raise ContractError("ProtocolSeal requires typed split manifests")
        try:
            validate_study_split_manifests(
                self.collection, self.validation, self.test, self.config.task_ids,
                collection_per_task=50, validation_per_task=10, test_per_task=10,
            )
        except SplitError as exc:
            raise ContractError(str(exc)) from exc
        for split_name, manifest in (("collection", self.collection), ("validation", self.validation), ("test", self.test)):
            mismatched = sorted({identity.environment_fingerprint for identity in manifest.identities
                                 if identity.environment_fingerprint != self.config.environment_revision})
            if mismatched:
                raise ContractError(
                    f"{split_name} reset identities do not match frozen environment_revision "
                    f"{self.config.environment_revision!r}: {mismatched[:3]}"
                )
        for split_name, manifest, expected_map in (
            ("validation", self.validation, self.config.validation_reset_ids),
            ("test", self.test, self.config.test_reset_ids),
        ):
            if expected_map:
                for task in self.config.task_ids:
                    expected = set(expected_map.get(task, ()))
                    actual = {identity.episode_id for identity in manifest.identities if identity.task_id == task}
                    if expected != actual:
                        raise ContractError(f"{split_name} reset identities for task {task} do not match StudyConfig maps")
        computed = _sha256_payload(self.payload())
        if self.protocol_sha256 and self.protocol_sha256 != computed:
            raise ContractError("protocol seal digest does not match contents")
        object.__setattr__(self, "protocol_sha256", computed)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": self.config.manifest(),
            "collection": self.collection.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "protocol_sha256": self.protocol_sha256}

    manifest = to_dict

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolSeal":
        if not isinstance(value, Mapping) or value.get("schema") != "arrow_policy_suite.protocol_seal.v1":
            raise ContractError("unsupported protocol seal schema")
        try:
            return cls(
                StudyConfig.from_manifest(value["config"]),
                SplitManifest.from_dict(value["collection"]),
                SplitManifest.from_dict(value["validation"]),
                SplitManifest.from_dict(value["test"]),
                str(value["schema"]), str(value.get("protocol_sha256", "")),
            )
        except (KeyError, TypeError, ValueError, SplitError) as exc:
            raise ContractError("malformed protocol seal") from exc

    @staticmethod
    def verify_manifest(value: Mapping[str, Any]) -> None:
        ProtocolSeal.from_dict(value)

    def write_artifact(self, path: str | Any) -> Any:
        from .artifacts import write_json_artifact
        return write_json_artifact(path, self.to_dict(), kind="protocol-seal")


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
