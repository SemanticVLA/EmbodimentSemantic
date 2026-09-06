"""Pinned LeRobot config factory used by native Pi0.5 evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def build_policy_config(*, checkpoint_path: str, artifact: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the native PI05Config and dataset metadata from one snapshot.

    The fine-tuned Hub artifact is evaluated directly.  No optimizer or
    adapter-training path is called here.  ``env_config`` is intentionally
    ``None`` because LIBERO is constructed by the shared evaluator rather than
    by LeRobot's policy factory.
    """

    del artifact, plan
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except ImportError as exc:  # pragma: no cover - compute-node runtime
        raise RuntimeError("the pinned LeRobot installation lacks the Pi0.5 runtime") from exc

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    policy_config = PI05Config.from_pretrained(
        checkpoint,
        device="cuda",
        pretrained_path=checkpoint,
        dtype="bfloat16",
    )
    dataset_repo = os.environ.get("PI05_DATASET_REPO", "HuggingFaceVLA/libero")
    dataset_revision = os.environ.get("PI05_DATASET_REVISION", "v3.0")
    dataset_meta = LeRobotDatasetMetadata(dataset_repo, revision=dataset_revision)
    return {"policy_config": policy_config, "dataset_meta": dataset_meta, "env_config": None}


__all__ = ["build_policy_config"]
