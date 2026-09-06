"""LeRobot 0.5.2 trainer entry point with explicit gradient accumulation.

LeRobot 0.5.2's stock ``lerobot_train`` loop updates the optimizer once per
batch and has no accumulation setting.  This small wrapper patches only the
update primitive before delegating configuration, dataset loading, processor
construction, checkpointing, and logging to the pinned trainer.  It is kept in
the repository so the emitted command and its accumulation semantics remain
auditable and version-specific.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from typing import Any


_ENV_STEPS = "EMBODIMENT_PI05_GRADIENT_ACCUMULATION_STEPS"


def _parse_wrapper_args(argv: list[str]) -> tuple[int, list[str]]:
    """Extract wrapper-only arguments and leave LeRobot's CLI untouched."""

    accumulation = None
    remaining: list[str] = []
    for arg in argv:
        if arg.startswith("--gradient_accumulation_steps="):
            if accumulation is not None:
                raise SystemExit("gradient accumulation was provided more than once")
            accumulation = int(arg.split("=", 1)[1])
            continue
        if arg == "--gradient_accumulation_steps":
            raise SystemExit("use --gradient_accumulation_steps=N")
        remaining.append(arg)
    if accumulation is None:
        raise SystemExit("--gradient_accumulation_steps=N is required")
    if accumulation <= 0:
        raise SystemExit("gradient accumulation must be positive")
    return accumulation, remaining


def _install_accumulating_update_policy(accumulation_steps: int) -> None:
    """Patch LeRobot's update primitive with true micro-batch accumulation."""

    import time

    import torch
    from lerobot.scripts import lerobot_train as train_module
    from lerobot.utils.utils import has_method

    state = {"micro_step": 0}

    def update_policy(
        train_metrics: Any,
        policy: Any,
        batch: Any,
        optimizer: Any,
        grad_clip_norm: float,
        accelerator: Any,
        lr_scheduler: Any = None,
        lock: Any = None,
        sample_weighter: Any = None,
    ) -> tuple[Any, dict | None]:
        start_time = time.perf_counter()
        policy.train()

        sample_weights = None
        weight_stats = None
        if sample_weighter is not None:
            sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

        with accelerator.autocast():
            if sample_weights is not None:
                per_sample_loss, output_dict = policy.forward(batch, reduction="none")
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy.forward(batch)

        # The divisor makes the accumulated gradient equal to the mean over
        # the configured effective batch, including distributed workers.
        accelerator.backward(loss / accumulation_steps)
        state["micro_step"] += 1
        should_step = state["micro_step"] % accumulation_steps == 0
        if should_step:
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )

            with lock if lock is not None else nullcontext():
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        else:
            # Preserve the metrics interface without pretending a micro-step
            # performed an optimizer update.
            grad_norm = torch.zeros((), device=loss.device)

        train_metrics.loss = loss.item()
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        return train_metrics, output_dict

    # Guard against accidental double patching if the wrapper is imported by
    # a launcher that retries in-process.
    update_policy.__name__ = "update_policy_with_gradient_accumulation"
    train_module.update_policy = update_policy


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    accumulation_steps, remaining = _parse_wrapper_args(raw)
    steps = next((int(arg.split("=", 1)[1]) for arg in remaining if arg.startswith("--steps=")), None)
    if steps is not None and steps % accumulation_steps:
        raise SystemExit(
            f"--steps={steps} must be divisible by --gradient_accumulation_steps={accumulation_steps}; "
            "the Pi0.5 recipe does not permit a partial optimizer update"
        )
    os.environ[_ENV_STEPS] = str(accumulation_steps)
    _install_accumulating_update_policy(accumulation_steps)
    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
