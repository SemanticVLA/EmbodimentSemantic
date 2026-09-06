"""Create immutable 100-episode plans for the Legion VLA evaluation matrix.

The script is intentionally run after the exact Hub snapshots have been
resolved on the compute node.  Plans therefore bind the resolved Hub commit
and the local checkpoint-tree digest instead of the mutable ``main`` ref.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vla_benchmarking.libero.evaluation.plan import build_evaluation_plan, shared_source_hashes
from vla_benchmarking.libero.evaluation.policy_adapter import (
    derive_checkpoint_receipt,
    derive_dataset_manifest_receipt,
    derive_io_receipt,
    derive_runtime_receipt,
)
from vla_benchmarking.libero.evaluation.native_vla_eval import _runtime_closure_paths


def _runtime_receipt(model: str) -> dict[str, Any]:
    return derive_runtime_receipt(closure_paths=_runtime_closure_paths(model), require_clean=True)


def _io_receipt(model: str) -> dict[str, Any]:
    if model == "pi05":
        return derive_io_receipt(
            {
                "action_horizon": 50,
                "action_dim": 7,
                "input_resolution": 256,
                "camera_keys": ["observation.images.image", "observation.images.image2"],
                "state_dim": 8,
                "visual_input": "none",
            }
        )
    if model == "openvla_oft":
        return derive_io_receipt(
            {
                "action_horizon": 8,
                "action_dim": 7,
                "input_resolution": 256,
                "camera_keys": ["observation.images.image", "observation.images.image2"],
                "state_dim": 8,
                "visual_input": "none",
            }
        )
    if model == "openvla":
        return derive_io_receipt(
            {
                "action_horizon": 1,
                "action_dim": 7,
                "input_resolution": 224,
                "camera_keys": ["agentview"],
                "state_dim": None,
                "visual_input": "none",
            }
        )
    raise ValueError(f"unsupported VLA model: {model}")


def _artifact(model: str, checkpoint: Path, revision: str) -> dict[str, Any]:
    model_id = {
        "pi05": "lerobot/pi05_libero_finetuned_v044",
        "openvla_oft": "openvla/openvla-7b-finetuned-libero-spatial",
        "openvla": "openvla/openvla-7b-finetuned-libero-spatial",
    }[model]
    return derive_checkpoint_receipt(
        checkpoint,
        artifact_id=model_id,
        checkpoint_revision=revision,
    )


def _write_plan(
    *,
    model: str,
    suite_mode: str,
    artifact: dict[str, Any],
    runtime: dict[str, Any],
    io: dict[str, Any],
    dataset: dict[str, Any],
    output: Path,
) -> None:
    plan = build_evaluation_plan(
        policy_kind=model,
        suite_mode=suite_mode,
        task_ids=tuple(range(10)),
        episodes_per_task=10,
        seed_base=1000,
        camera="agentview",
        resolution=256,
        text_context="none",
        visual_input="none",
        source_hashes=shared_source_hashes(),
        bindings={
            "artifact": artifact,
            "runtime": runtime,
            "io": io,
            "dataset_manifest": dataset,
            "adapter_kind": model,
        },
    )
    if len(plan["schedule"]["cells"]) != 100:
        raise AssertionError("VLA evaluation plan must contain exactly 100 cells")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "openvla", "openvla_oft"), required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint_path.expanduser().resolve()
    dataset_manifest = args.dataset_manifest.expanduser().resolve()
    artifact = _artifact(args.model, checkpoint, args.checkpoint_revision)
    runtime = _runtime_receipt(args.model)
    io = _io_receipt(args.model)
    dataset = derive_dataset_manifest_receipt(dataset_manifest)
    for suite_mode, name in (("vanilla", "vanilla"), ("sealed_randomized", "sealed")):
        _write_plan(
            model=args.model,
            suite_mode=suite_mode,
            artifact=artifact,
            runtime=runtime,
            io=io,
            dataset=dataset,
            output=args.output_dir.expanduser().resolve() / f"{args.model}_{name}_100.json",
        )
    print(json.dumps({"model": args.model, "checkpoint": artifact, "dataset": dataset}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
