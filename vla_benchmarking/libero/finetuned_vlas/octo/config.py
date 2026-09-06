"""Pinned Octo artifacts and matched-training configuration.

This module is declarative only: importing it never downloads a checkpoint or
initializes a JAX device.  The community evaluation and matched retraining
recipes are deliberately separate because the former is an unmatched,
multi-suite artifact while the latter is the paper's controlled condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal


@dataclass(frozen=True)
class CheckpointRef:
    """Immutable Hub artifact identity using Octo's published root/step layout.

    ``checkpoint_subpath`` is retained as a compatibility view for older
    manifests.  New callers should use ``experiment_root`` and ``step`` so a
    loader cannot accidentally pass the nested ``default/checkpoint`` leaf as
    the model root.
    """

    repository: str
    revision: str
    checkpoint_subpath: str | None = None
    experiment_root: str | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if self.step is not None and (isinstance(self.step, bool) or int(self.step) <= 0):
            raise ValueError("Octo checkpoint step must be a positive integer")
        if self.experiment_root is not None and not str(self.experiment_root).strip():
            raise ValueError("Octo experiment_root must be non-empty when supplied")
        if self.checkpoint_subpath:
            parts = tuple(part for part in self.checkpoint_subpath.strip("/").split("/") if part)
            if len(parts) < 3 or parts[-2:] != ("default", "checkpoint"):
                raise ValueError("Octo checkpoint_subpath must end in <step>/default/checkpoint")
            parsed_step = int(parts[-3])
            parsed_root = "/".join(parts[:-3])
            if self.step is not None and int(self.step) != parsed_step:
                raise ValueError("checkpoint step disagrees with checkpoint_subpath")
            if self.experiment_root is not None and str(self.experiment_root).strip("/") != parsed_root:
                raise ValueError("experiment_root disagrees with checkpoint_subpath")
            object.__setattr__(self, "step", parsed_step)
            object.__setattr__(self, "experiment_root", parsed_root or None)
        elif self.step is not None:
            root = str(self.experiment_root or "").strip("/")
            suffix = f"{int(self.step)}/default/checkpoint"
            object.__setattr__(self, "checkpoint_subpath", f"{root}/{suffix}" if root else suffix)

    def identifier(self) -> str:
        base = f"hf://{self.repository}@{self.revision}"
        if self.checkpoint_subpath:
            return f"{base}/{self.checkpoint_subpath.strip('/')}"
        return base

    def root_identifier(self) -> str:
        """Return the Hub/local model root passed to ``load_pretrained``."""

        return str(self.experiment_root or "").strip("/")


@dataclass(frozen=True)
class BatchCandidate:
    """One predeclared A40 memory option; effective batch stays fixed at 32."""

    microbatch: int
    gradient_accumulation_steps: int

    @property
    def effective_batch(self) -> int:
        return self.microbatch * self.gradient_accumulation_steps


@dataclass(frozen=True)
class OctoConfig:
    name: Literal["community_eval", "matched_train"]
    checkpoint: CheckpointRef
    mode: Literal["community_eval", "matched_train"]
    dataset_name: str
    episodes: int
    timesteps: int
    epochs: int
    optimizer_updates: int | None
    seed: int
    learning_rate: float | None
    warmup_steps: int | None
    image_key: str
    language_key: str
    action_dim: int
    observation_window: int
    action_horizon: int
    use_wrist_camera: bool
    use_proprioception: bool
    freeze_t5: bool
    full_transformer_finetune: bool
    policy_kind: str
    provenance: str
    notes: str

    def validate(self) -> None:
        if self.episodes <= 0 or self.timesteps <= 0:
            raise ValueError("dataset counts must be positive")
        if self.action_dim != 7:
            raise ValueError("LIBERO requires a seven-dimensional action")
        if self.observation_window not in (1, 2):
            raise ValueError("Octo LIBERO contract supports only one- or two-frame windows")
        if self.action_horizon != 4:
            raise ValueError("Octo-Base 1.5 LIBERO contract uses a four-step horizon")
        if self.use_wrist_camera or self.use_proprioception:
            raise ValueError("selected Octo checkpoint contract is image-and-language only")
        if self.mode == "matched_train":
            if self.policy_kind != "octo_base15_spatial_no_arrow_matched":
                raise ValueError("matched training policy kind is not sealed")
            if self.episodes != 500 or self.timesteps != 62250:
                raise ValueError("matched training must use the canonical 500/62250 source exposure")
            if self.optimizer_updates is not None or self.epochs != 15:
                raise ValueError("matched training must preserve the sealed 15-epoch exposure")
            if self.learning_rate != 3e-4 or self.warmup_steps != 2000:
                raise ValueError("matched training must preserve the sealed Octo schedule")
        elif self.mode == "community_eval":
            if self.policy_kind != "octo_community_multisuite_190k":
                raise ValueError("community evaluation policy kind is not sealed")
            if self.episodes != 432 or self.timesteps != 52970:
                raise ValueError("community evaluation must record its 432/52970 provenance")
            if self.observation_window != 2:
                raise ValueError("community checkpoint config requires a two-frame observation window")
            if self.optimizer_updates is not None:
                raise ValueError("community evaluation must not declare training updates")
        else:
            raise ValueError(f"unsupported Octo mode: {self.mode}")


COMMUNITY_CHECKPOINT = CheckpointRef(
    repository="cyrusneary/octo-finetuned-libero",
    revision="f8a0888cfa7ef3be072417eb012339464a9bb6dc",
    checkpoint_subpath=(
        "2025-06-21_octo_base_1p5_libero_finetune/octo_finetune/"
        "experiment_20250621_094538/190000/default/checkpoint"
    ),
    experiment_root=(
        "2025-06-21_octo_base_1p5_libero_finetune/octo_finetune/"
        "experiment_20250621_094538"
    ),
    step=190000,
)

OFFICIAL_BASE_CHECKPOINT = CheckpointRef(
    repository="rail-berkeley/octo-base-1.5",
    revision="ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48",
    checkpoint_subpath="300000/default/checkpoint",
    experiment_root=None,
    step=300000,
)

# This name is registered by dataset.register_tfds_builder and is passed
# directly to the pinned Octo make_single_dataset/tfds.builder path.
OCTO_DATASET_NAME = "libero_spatial_no_arrows"
COMMUNITY_DATASET_NAME = "libero_spatial_community_reference_432"
OCTO_IMAGE_KEY = "image_primary"
OCTO_LANGUAGE_KEY = "language_instruction"
OCTO_ACTION_DIM = 7
OCTO_OBSERVATION_WINDOW = 1
OCTO_ACTION_HORIZON = 4
EFFECTIVE_BATCH = 32
A40_MEMORY_LIMIT_GB = 43.2  # 90% of a 48 GB A40, sealed before full training.

A40_BATCH_LADDER = (
    BatchCandidate(microbatch=32, gradient_accumulation_steps=1),
    BatchCandidate(microbatch=16, gradient_accumulation_steps=2),
    BatchCandidate(microbatch=8, gradient_accumulation_steps=4),
)

COMMUNITY_EVAL_CONFIG = OctoConfig(
    name="community_eval",
    checkpoint=COMMUNITY_CHECKPOINT,
    mode="community_eval",
    dataset_name=COMMUNITY_DATASET_NAME,
    episodes=432,
    timesteps=52970,
    epochs=0,
    optimizer_updates=None,
    seed=1000,
    learning_rate=None,
    warmup_steps=None,
    image_key=OCTO_IMAGE_KEY,
    language_key=OCTO_LANGUAGE_KEY,
    action_dim=OCTO_ACTION_DIM,
    observation_window=2,
    action_horizon=OCTO_ACTION_HORIZON,
    use_wrist_camera=False,
    use_proprioception=False,
    freeze_t5=True,
    full_transformer_finetune=False,
    policy_kind="octo_community_multisuite_190k",
    provenance="community_multisuite_libero_190k",
    notes="Unmatched community multi-suite LIBERO checkpoint at 190000 updates.",
)

MATCHED_TRAIN_CONFIG = OctoConfig(
    name="matched_train",
    checkpoint=OFFICIAL_BASE_CHECKPOINT,
    mode="matched_train",
    dataset_name=OCTO_DATASET_NAME,
    episodes=500,
    timesteps=62250,
    epochs=15,
    # Derived by preflight from the verified transition count and effective
    # batch; never duplicated as a hand-entered schedule constant.
    optimizer_updates=None,
    seed=1000,
    learning_rate=3e-4,
    warmup_steps=2000,
    image_key=OCTO_IMAGE_KEY,
    language_key=OCTO_LANGUAGE_KEY,
    action_dim=OCTO_ACTION_DIM,
    observation_window=OCTO_OBSERVATION_WINDOW,
    action_horizon=OCTO_ACTION_HORIZON,
    use_wrist_camera=False,
    use_proprioception=False,
    freeze_t5=True,
    full_transformer_finetune=True,
    policy_kind="octo_base15_spatial_no_arrow_matched",
    provenance="canonical_no_arrow_500_62250",
    notes="Controlled no-arrow retraining from official Octo-Base 1.5.",
)


def build_community_eval_config() -> OctoConfig:
    """Return the immutable community-checkpoint evaluation contract."""

    return COMMUNITY_EVAL_CONFIG


def build_matched_train_config() -> OctoConfig:
    """Return the immutable canonical no-arrow training contract."""

    return MATCHED_TRAIN_CONFIG


def checkpoint_download_patterns(checkpoint: CheckpointRef) -> tuple[str, ...]:
    """Return a fail-closed allow-list for a single Octo checkpoint tree.

    Historical checkpoints in the community repository must never be pulled
    into the local snapshot accidentally.  The selected nested checkpoint is
    the only recursive pattern; metadata is limited to the files consumed by
    the audit/loader.
    """

    metadata = (
        "config.json",
        "finetune_config.json",
        "dataset_statistics.json",
        "example_batch.msgpack",
    )
    if checkpoint.checkpoint_subpath:
        selected = checkpoint.checkpoint_subpath.strip("/")
        parts = selected.split("/")
        run_root = "/".join(parts[:-3]) if len(parts) >= 3 else ""
        scoped_metadata = tuple(f"{run_root}/{name}" for name in metadata) if run_root else ()
        # Keep both forms: Hub snapshots may expose ``checkpoint`` as one
        # large file, or as a directory whose contents must be fetched.
        return (*metadata, *scoped_metadata, selected, f"{selected}/**")
    return (*metadata, "checkpoint/**")


def choose_a40_batch_candidate(measured_peak_vram_gb: Mapping[int, float]) -> BatchCandidate:
    """Choose the first sealed ladder entry that fits measured VRAM.

    Selection must happen after the two-update memory smoke test and before a
    full run.  The function intentionally has no automatic alternative outside
    the declared ladder.

    ``measured_peak_vram_gb`` maps each candidate microbatch to its measured
    peak allocation.  Requiring measurements keyed by the declared ladder
    prevents a caller from silently selecting a different training recipe.
    """

    if not isinstance(measured_peak_vram_gb, Mapping):
        raise TypeError("measured peak VRAM must be a mapping keyed by microbatch")
    for candidate in A40_BATCH_LADDER:
        if candidate.effective_batch != EFFECTIVE_BATCH:
            raise AssertionError("A40 ladder changed the effective batch")
        if candidate.microbatch not in measured_peak_vram_gb:
            raise ValueError(f"missing A40 memory measurement for microbatch {candidate.microbatch}")
        peak_vram_gb = float(measured_peak_vram_gb[candidate.microbatch])
        if peak_vram_gb < 0:
            raise ValueError("peak VRAM cannot be negative")
        if peak_vram_gb <= A40_MEMORY_LIMIT_GB:
            return candidate
    raise RuntimeError(
        f"all measured A40 candidates exceed the sealed {A40_MEMORY_LIMIT_GB:.2f} GB limit"
    )


COMMUNITY_EVAL_CONFIG.validate()
MATCHED_TRAIN_CONFIG.validate()
