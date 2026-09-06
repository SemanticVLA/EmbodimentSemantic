"""Octo-Base 1.5 integration contracts for the LIBERO benchmark.

The package intentionally has no import-time dependency on JAX, TensorFlow,
Octo, or Hugging Face Hub.  Compute-node entry points import those packages
only after the relevant runtime has been selected and preflighted.
"""

from .config import (
    A40_BATCH_LADDER,
    COMMUNITY_DATASET_NAME,
    OCTO_DATASET_NAME,
    COMMUNITY_CHECKPOINT,
    COMMUNITY_EVAL_CONFIG,
    OFFICIAL_BASE_CHECKPOINT,
    MATCHED_TRAIN_CONFIG,
    build_community_eval_config,
    build_matched_train_config,
    checkpoint_download_patterns,
)
from .contracts import (
    OCTO_ACTION_DIM,
    OCTO_ACTION_HORIZON,
    OCTO_IMAGE_KEY,
    OCTO_OBSERVATION_WINDOW,
    convert_libero_to_octo_action,
    convert_octo_to_libero_action,
    rotate_stored_frame_180,
)
from .manifest import compute_optimizer_updates
from .dataset import (
    build_tfds_dataset,
    make_builder_class,
    register_tfds_builder,
    serialize_canonical_episode,
    smoke_test_make_single_dataset,
    write_rlds_source_jsonl,
)

__all__ = [
    "A40_BATCH_LADDER",
    "COMMUNITY_DATASET_NAME",
    "OCTO_DATASET_NAME",
    "COMMUNITY_CHECKPOINT",
    "COMMUNITY_EVAL_CONFIG",
    "MATCHED_TRAIN_CONFIG",
    "OFFICIAL_BASE_CHECKPOINT",
    "build_community_eval_config",
    "build_matched_train_config",
    "checkpoint_download_patterns",
    "OCTO_ACTION_DIM",
    "OCTO_ACTION_HORIZON",
    "OCTO_IMAGE_KEY",
    "OCTO_OBSERVATION_WINDOW",
    "convert_libero_to_octo_action",
    "convert_octo_to_libero_action",
    "rotate_stored_frame_180",
    "compute_optimizer_updates",
    "serialize_canonical_episode",
    "write_rlds_source_jsonl",
    "make_builder_class",
    "register_tfds_builder",
    "build_tfds_dataset",
    "smoke_test_make_single_dataset",
]
