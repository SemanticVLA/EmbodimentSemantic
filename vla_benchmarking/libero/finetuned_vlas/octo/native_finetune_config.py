"""Checked-in Octo config overlay for the matched LIBERO retraining.

This module is loaded by Octo's pinned ``scripts/finetune.py`` through its
ml-collections config flag.  Dataset paths and the verified update count are
still supplied as command-line overrides by ``train.py``; no scientific
defaults are duplicated in the launcher.
"""

from __future__ import annotations

from collections.abc import Mapping


def libero_identity_transform(trajectory):
    """Validate canonical LIBERO fields without Bridge state/action rewriting.

    The upstream Octo config defaults to a Bridge transform.  LIBERO already
    supplies canonical seven-dimensional actions and an agent-view image, so
    applying that transform would silently alter the experiment contract.
    """

    if not isinstance(trajectory, Mapping):
        raise TypeError("LIBERO Octo trajectory must be a mapping")
    observation = trajectory.get("observation")
    if not isinstance(observation, Mapping) or "image_primary" not in observation:
        raise ValueError("LIBERO trajectory requires observation.image_primary")
    action = trajectory.get("action")
    shape = getattr(action, "shape", None)
    if shape is None or len(shape) == 0:
        raise ValueError("LIBERO trajectory requires a tensor action with a static final dimension")
    # Time/trajectory dimensions may be None or otherwise unknown.  Only the
    # final action dimension is checked statically and it must be known.
    final_dim = shape[-1]
    if final_dim is None or int(final_dim) != 7:
        raise ValueError(f"LIBERO trajectory action must end in seven dimensions, got {shape}")
    language = trajectory.get("language_instruction")
    task = trajectory.get("task")
    if language is None and isinstance(task, Mapping):
        language = task.get("language_instruction")
    if language is None:
        raise ValueError("LIBERO trajectory requires language_instruction")
    return trajectory


def smoke_test_config_and_dataset(
    data_dir,
    *,
    config=None,
    make_single_dataset_fn=None,
):
    """One-batch seam proving the exact native config reaches Octo's loader."""

    if config is None:
        config = get_config()
    dataset_kwargs = config.dataset_kwargs.to_dict() if hasattr(config.dataset_kwargs, "to_dict") else dict(config.dataset_kwargs)
    raw_traj = getattr(config, "traj_transform_kwargs", {})
    raw_frame = getattr(config, "frame_transform_kwargs", {})
    traj_kwargs = raw_traj.to_dict() if hasattr(raw_traj, "to_dict") else dict(raw_traj)
    frame_kwargs = raw_frame.to_dict() if hasattr(raw_frame, "to_dict") else dict(raw_frame)
    standardize = dataset_kwargs.get("standardize_fn")
    if standardize is None:
        raise ValueError("native Octo config lost the LIBERO identity standardize_fn")
    dataset_kwargs["name"] = str(dataset_kwargs.get("name", "libero_spatial_no_arrows"))
    dataset_kwargs["data_dir"] = str(data_dir)
    from vla_benchmarking.libero.finetuned_vlas.octo.dataset import smoke_test_make_single_dataset

    return smoke_test_make_single_dataset(
        data_dir,
        make_single_dataset_fn=make_single_dataset_fn,
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=traj_kwargs,
        frame_transform_kwargs=frame_kwargs,
    )


def get_config(config_string: str = "full,language_conditioned"):
    # The pinned Octo trainer resolves datasets through ``tfds.builder`` in a
    # fresh process.  Register our checked-in local builder before its config
    # is evaluated; the builder is lazy and performs no data work here.
    from vla_benchmarking.libero.finetuned_vlas.octo.config import OCTO_DATASET_NAME
    from vla_benchmarking.libero.finetuned_vlas.octo.dataset import register_tfds_builder
    from octo.utils.spec import ModuleSpec

    register_tfds_builder(dataset_name=OCTO_DATASET_NAME)
    try:
        from scripts.configs.finetune_config import get_config as upstream_get_config
    except ImportError as exc:  # pragma: no cover - only on the Octo runtime
        raise RuntimeError(
            "native_finetune_config.py must be evaluated from the pinned Octo repository"
        ) from exc

    config = upstream_get_config(config_string)
    # LIBERO provides one agent-view RGB stream and language.  The matched
    # policy deliberately excludes wrist/proprio inputs, while retaining the
    # native 4-action chunk and unscaled gripper dimension.
    config.dataset_kwargs.name = OCTO_DATASET_NAME
    config.dataset_kwargs.image_obs_keys = {"primary": "image_primary", "wrist": None}
    config.dataset_kwargs.proprio_obs_key = None
    config.dataset_kwargs.language_key = "language_instruction"
    config.dataset_kwargs.standardize_fn = ModuleSpec.create(
        "vla_benchmarking.libero.finetuned_vlas.octo.native_finetune_config:libero_identity_transform"
    )
    config.dataset_kwargs.action_normalization_mask = [True, True, True, True, True, True, False]
    config.window_size = 1
    config.traj_transform_kwargs.window_size = 1
    config.traj_transform_kwargs.action_horizon = 4
    config.seed = 1000
    config.batch_size = 32
    config.optimizer.learning_rate.peak_value = 3e-4
    config.optimizer.learning_rate.warmup_steps = 2000
    config.optimizer.learning_rate.decay_steps = int(getattr(config, "num_steps", 1))
    config.optimizer.weight_decay = 0.01
    config.optimizer.clip_gradient = 1.0
    config.optimizer.frozen_keys = None
    return config
