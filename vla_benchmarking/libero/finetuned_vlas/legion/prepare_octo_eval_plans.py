"""Create the matched Octo 100+100 episode evaluation plans."""

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
from vla_benchmarking.libero.finetuned_vlas.octo.config import MATCHED_TRAIN_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    closure = (
        root / "vla_benchmarking/libero/evaluation",
        root / "vla_benchmarking/libero/finetuned_vlas/common",
        root / "vla_benchmarking/libero/finetuned_vlas/octo",
    )
    artifact = derive_checkpoint_receipt(
        args.checkpoint_path,
        artifact_id="octo_base15_spatial_no_arrow_matched_finetuned",
        checkpoint_revision=args.checkpoint_revision,
    )
    runtime = derive_runtime_receipt(closure_paths=closure, require_clean=True)
    io = derive_io_receipt(
        {
            "action_horizon": 4,
            "action_dim": 7,
            "input_resolution": 256,
            "camera_keys": ["image_primary"],
            "state_dim": None,
            "visual_input": "none",
        }
    )
    dataset = derive_dataset_manifest_receipt(args.dataset_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suite_mode, name in (("vanilla", "vanilla"), ("sealed_randomized", "sealed")):
        plan = build_evaluation_plan(
            policy_kind=MATCHED_TRAIN_CONFIG.policy_kind,
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
                "adapter_kind": MATCHED_TRAIN_CONFIG.policy_kind,
            },
        )
        if len(plan["schedule"]["cells"]) != 100:
            raise AssertionError("Octo evaluation plan must contain exactly 100 cells")
        (args.output_dir / f"octo_{name}_100.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"artifact": artifact, "dataset": dataset}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
