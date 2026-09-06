"""Write the sealed A40 two-update memory-smoke receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vla_benchmarking.libero.finetuned_vlas.octo.config import A40_BATCH_LADDER, A40_MEMORY_LIMIT_GB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--octo-commit", required=True)
    parser.add_argument("--peaks", nargs=3, type=float, metavar=("MB32", "MB16", "MB8"), required=True)
    parser.add_argument("--statuses", nargs=3, choices=("PASS", "FAILED"), metavar=("MB32", "MB16", "MB8"), required=True)
    parser.add_argument("--smoke-status", choices=("PASS", "FAIL"), default="PASS")
    args = parser.parse_args()
    peaks = {candidate.microbatch: float(value) for candidate, value in zip(A40_BATCH_LADDER, args.peaks)}
    statuses = {candidate.microbatch: status for candidate, status in zip(A40_BATCH_LADDER, args.statuses)}
    if any(not math.isfinite(value) or value < 0 for value in peaks.values()):
        raise SystemExit("all peak VRAM values must be finite and non-negative")
    successful_peaks = {microbatch: peak for microbatch, peak in peaks.items() if statuses[microbatch] == "PASS"}
    selected = next(
        (
            candidate for candidate in A40_BATCH_LADDER
            if statuses[candidate.microbatch] == "PASS"
            and successful_peaks[candidate.microbatch] <= A40_MEMORY_LIMIT_GB
        ),
        None,
    ) if args.smoke_status == "PASS" else None
    payload = {
        "schema": "octo_a40_memory_receipt.v1",
        "device": {"model": "NVIDIA A40", "memory_gb": 48.0},
        "run": {
            "checkpoint_revision": str(args.checkpoint_revision),
            "dataset_manifest_sha256": str(args.dataset_manifest_sha256),
            "config_sha256": str(args.config_sha256),
            "runtime_sha256": str(args.runtime_sha256),
            "octo_commit": str(args.octo_commit),
        },
        "smoke_test": {"completed": args.smoke_status == "PASS", "updates": 2},
        "candidates": [
            {
                "microbatch": candidate.microbatch,
                "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
                "microsteps": 2 * candidate.gradient_accumulation_steps,
                "status": statuses[candidate.microbatch],
                **({"peak_vram_gb": peaks[candidate.microbatch]} if statuses[candidate.microbatch] == "PASS" else {}),
            }
            for candidate in A40_BATCH_LADDER
        ],
        "selected_candidate": (
            {"microbatch": selected.microbatch, "gradient_accumulation_steps": selected.gradient_accumulation_steps}
            if selected is not None else {"microbatch": 0, "gradient_accumulation_steps": 0}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if selected is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
