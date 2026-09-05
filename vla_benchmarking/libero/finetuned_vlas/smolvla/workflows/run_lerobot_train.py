"""Run LeRobot training with deterministic flags and optional runtime evidence.

Set ``TRAINING_RUNTIME_EVIDENCE`` to a JSON path when a smoke or cluster job
needs process-local loss and CUDA peak-memory evidence.  Keeping this inside
the training process is important: querying CUDA from a later Python process
would incorrectly report zero for the completed workload.
"""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _install_runtime_evidence(lerobot_train, output: Path) -> None:
    """Instrument LeRobot's update boundary without changing training math."""
    state = {
        "updates_observed": 0,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "min_loss": None,
        "max_loss": None,
        "last_loss": None,
        "last_grad_norm": None,
    }
    original_update_policy = lerobot_train.update_policy

    def observed_update_policy(*args, **kwargs):
        tracker, output_dict = original_update_policy(*args, **kwargs)
        metrics = tracker.to_dict(use_avg=False)
        loss = float(metrics["loss"])
        grad_norm = float(metrics["grad_norm"])
        state["updates_observed"] += 1
        state["all_losses_finite"] = bool(state["all_losses_finite"] and math.isfinite(loss))
        state["all_grad_norms_finite"] = bool(
            state["all_grad_norms_finite"] and math.isfinite(grad_norm)
        )
        state["min_loss"] = loss if state["min_loss"] is None else min(state["min_loss"], loss)
        state["max_loss"] = loss if state["max_loss"] is None else max(state["max_loss"], loss)
        state["last_loss"] = loss
        state["last_grad_norm"] = grad_norm
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite training metric at update {state['updates_observed']}: "
                f"loss={loss}, grad_norm={grad_norm}"
            )
        return tracker, output_dict

    lerobot_train.update_policy = observed_update_policy
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def write_evidence() -> None:
        payload = dict(state)
        payload.update(
            {
                "schema_version": 1,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0,
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(output.name + f".tmp.{os.getpid()}")
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.replace(output)

    atexit.register(write_evidence)


def main() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    evidence = os.environ.get("TRAINING_RUNTIME_EVIDENCE")
    if evidence:
        _install_runtime_evidence(lerobot_train, Path(evidence).expanduser().resolve())
    lerobot_train.main()


if __name__ == "__main__":
    main()
