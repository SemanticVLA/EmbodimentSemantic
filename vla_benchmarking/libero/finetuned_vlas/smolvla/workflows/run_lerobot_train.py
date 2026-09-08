"""Run LeRobot training with deterministic flags and optional runtime evidence.

Set ``TRAINING_RUNTIME_EVIDENCE`` to a JSON path when a smoke or cluster job
needs process-local loss and CUDA peak-memory evidence.  Keeping this inside
the training process is important: querying CUDA from a later Python process
would incorrectly report zero for the completed workload.
"""
from __future__ import annotations

import atexit
import inspect
import json
import math
import os
from pathlib import Path

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _required_runtime_setting(name: str, cast):
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} is required when runtime evidence is enabled")
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} is invalid: {value!r}") from exc


def _install_runtime_evidence(lerobot_train, output: Path):
    """Instrument LeRobot's update boundary without changing training math."""
    state = {
        "updates_observed": 0,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "min_loss": None,
        "max_loss": None,
        "last_loss": None,
        "last_grad_norm": None,
        "min_learning_rate": None,
        "max_learning_rate": None,
        "last_learning_rate": None,
        "optimizer_class": None,
        "optimizer_hyperparameters": None,
        "scheduler_class": None,
        "scheduler_config": None,
        "lr_milestones_consistent": False,
        "attestation_status": "RUNNING",
    }
    expected = {
        "updates": _required_runtime_setting("TRAIN_EXPECTED_UPDATES", int),
        "base_lr": _required_runtime_setting("TRAIN_EXPECTED_BASE_LR", float),
        "weight_decay": _required_runtime_setting("TRAIN_EXPECTED_WEIGHT_DECAY", float),
        "betas": tuple(float(x) for x in json.loads(_required_runtime_setting("TRAIN_EXPECTED_BETAS", str))),
        "eps": _required_runtime_setting("TRAIN_EXPECTED_EPS", float),
        "grad_clip_norm": _required_runtime_setting("TRAIN_EXPECTED_GRAD_CLIP", float),
        "scheduler": _required_runtime_setting("TRAIN_EXPECTED_SCHEDULER", str),
        "warmup_steps": _required_runtime_setting("TRAIN_EXPECTED_WARMUP", int),
        "decay_steps": _required_runtime_setting("TRAIN_EXPECTED_DECAY_STEPS", int),
        "decay_lr": _required_runtime_setting("TRAIN_EXPECTED_DECAY_LR", float),
    }
    if len(expected["betas"]) != 2:
        raise RuntimeError("TRAIN_EXPECTED_BETAS must contain exactly two values")
    original_update_policy = lerobot_train.update_policy
    bound_signature = inspect.signature(original_update_policy)
    objects_checked = False

    def _verify_optimizer_and_scheduler(args, kwargs) -> tuple[object, object]:
        nonlocal objects_checked
        bound = bound_signature.bind_partial(*args, **kwargs)
        optimizer = bound.arguments.get("optimizer")
        scheduler = bound.arguments.get("lr_scheduler")
        grad_clip_norm = bound.arguments.get("grad_clip_norm")
        if optimizer is None or scheduler is None:
            raise RuntimeError("runtime evidence could not bind optimizer and lr_scheduler")
        if grad_clip_norm is None or abs(float(grad_clip_norm) - expected["grad_clip_norm"]) > 1e-12:
            raise RuntimeError("gradient clipping differs from sealed value")
        optimizer_wrapper = optimizer
        if (optimizer.__class__.__name__, optimizer.__class__.__module__) == (
            "AcceleratedOptimizer", "accelerate.optimizer"
        ):
            optimizer = getattr(optimizer, "optimizer", None)
            if optimizer is None or optimizer is optimizer_wrapper:
                raise RuntimeError("AcceleratedOptimizer does not expose its wrapped optimizer")
        scheduler_wrapper = scheduler
        if (scheduler.__class__.__name__, scheduler.__class__.__module__) == (
            "AcceleratedScheduler", "accelerate.scheduler"
        ):
            wrapped_optimizers = list(getattr(scheduler, "optimizers", ()))
            if optimizer_wrapper not in wrapped_optimizers:
                raise RuntimeError("AcceleratedScheduler is not bound to the training optimizer wrapper")
            scheduler = getattr(scheduler, "scheduler", None)
            if scheduler is None or scheduler is scheduler_wrapper:
                raise RuntimeError("AcceleratedScheduler does not expose its wrapped scheduler")
        if optimizer.__class__.__name__ != "AdamW" or optimizer.__class__.__module__ != "torch.optim.adamw":
            raise RuntimeError(f"optimizer is not torch.optim.AdamW: {type(optimizer)!r}")
        groups = list(getattr(optimizer, "param_groups", ()))
        if not groups:
            raise RuntimeError("AdamW has no parameter groups")
        for group in groups:
            if abs(float(group.get("weight_decay", float("nan"))) - expected["weight_decay"]) > 1e-12:
                raise RuntimeError("AdamW weight decay differs from sealed value")
            if tuple(float(x) for x in group.get("betas", ())) != expected["betas"]:
                raise RuntimeError("AdamW betas differ from sealed value")
            if abs(float(group.get("eps", float("nan"))) - expected["eps"]) > 1e-15:
                raise RuntimeError("AdamW epsilon differs from sealed value")
        if abs(float(getattr(optimizer, "defaults", {}).get("lr", float("nan"))) - expected["base_lr"]) > 1e-12:
            raise RuntimeError("AdamW base learning rate differs from sealed peak LR")
        if scheduler.__class__.__name__ not in {"LambdaLR", "SequentialLR", "CosineAnnealingLR"}:
            raise RuntimeError(f"unexpected scheduler class: {type(scheduler)!r}")
        if getattr(scheduler, "optimizer", None) is not optimizer:
            raise RuntimeError("scheduler is not bound to the training AdamW optimizer")
        base_lrs = [float(x) for x in getattr(scheduler, "base_lrs", ())]
        if not base_lrs or any(abs(x - expected["base_lr"]) > 1e-12 for x in base_lrs):
            raise RuntimeError(f"scheduler base LR differs from sealed peak LR: {base_lrs!r}")
        state["optimizer_class"] = f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}"
        first_group = groups[0]
        state["optimizer_hyperparameters"] = {
            "lr": float(optimizer.defaults["lr"]),
            "weight_decay": float(first_group["weight_decay"]),
            "betas": [float(x) for x in first_group["betas"]],
            "eps": float(first_group["eps"]),
            "grad_clip_norm": float(grad_clip_norm),
        }
        state["scheduler_class"] = f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}"
        state["scheduler_config"] = {
            "name": expected["scheduler"], "base_lrs": base_lrs,
            "warmup_steps": expected["warmup_steps"], "decay_steps": expected["decay_steps"],
            "decay_lr": expected["decay_lr"],
            "last_epoch": int(getattr(scheduler, "last_epoch", -1)),
            "last_lr": [float(x) for x in getattr(scheduler, "_last_lr", ())],
        }
        objects_checked = True
        return optimizer, scheduler

    def observed_update_policy(*args, **kwargs):
        optimizer, _scheduler = _verify_optimizer_and_scheduler(args, kwargs)
        tracker, output_dict = original_update_policy(*args, **kwargs)
        metrics = tracker.to_dict(use_avg=False)
        loss = float(metrics["loss"])
        grad_norm = float(metrics["grad_norm"])
        learning_rate = float(metrics["lr"])
        state["updates_observed"] += 1
        state["all_losses_finite"] = bool(state["all_losses_finite"] and math.isfinite(loss))
        state["all_grad_norms_finite"] = bool(
            state["all_grad_norms_finite"] and math.isfinite(grad_norm)
        )
        state["min_loss"] = loss if state["min_loss"] is None else min(state["min_loss"], loss)
        state["max_loss"] = loss if state["max_loss"] is None else max(state["max_loss"], loss)
        state["last_loss"] = loss
        state["last_grad_norm"] = grad_norm
        state["min_learning_rate"] = learning_rate if state["min_learning_rate"] is None else min(state["min_learning_rate"], learning_rate)
        state["max_learning_rate"] = learning_rate if state["max_learning_rate"] is None else max(state["max_learning_rate"], learning_rate)
        state["last_learning_rate"] = learning_rate
        update = state["updates_observed"]
        # LeRobot 0.5.2's scheduler closures use the scheduler's post-step
        # ``last_epoch`` as current_step.  Warmup is (step+1)/(warmup+1);
        # cosine decay is intentionally *not* warmup-offset.
        current_step = int(getattr(_scheduler, "last_epoch", update))
        warmup, decay = expected["warmup_steps"], expected["decay_steps"]
        if current_step < warmup:
            expected_lr = expected["base_lr"] * (current_step + 1) / max(1, warmup + 1)
        elif current_step >= decay:
            expected_lr = expected["decay_lr"]
        else:
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * current_step / max(1, decay)))
            expected_lr = expected["decay_lr"] + (expected["base_lr"] - expected["decay_lr"]) * cosine_factor
        if abs(learning_rate - expected_lr) > max(1e-8, expected["base_lr"] * 2e-3):
            raise RuntimeError(f"scheduler LR milestone mismatch at update {update}: got {learning_rate}, expected {expected_lr}")
        if not math.isfinite(loss) or not math.isfinite(grad_norm) or not math.isfinite(learning_rate):
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
        payload["expected_contract"] = expected
        payload["optimizer_scheduler_objects_checked"] = bool(objects_checked)
        if state.get("optimizer_hyperparameters"):
            opt = dict(state["optimizer_hyperparameters"])
            payload["optimizer"] = {
                "optimizer": "AdamW", "weight_decay": opt["weight_decay"],
                "peak_learning_rate": opt["lr"], "betas": opt["betas"],
                "epsilon": opt["eps"], "gradient_clip_norm": opt["grad_clip_norm"],
            }
        if state.get("scheduler_config"):
            sched = dict(state["scheduler_config"])
            payload["scheduler"] = {
                "scheduler": sched["name"], "warmup_steps": sched["warmup_steps"],
                "decay_steps": sched["decay_steps"], "decay_lr": sched["decay_lr"],
                "class": state.get("scheduler_class"), "base_lrs": sched["base_lrs"],
                "last_epoch": sched["last_epoch"], "last_lr": sched["last_lr"],
            }
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

    def finalize() -> None:
        floor_required = expected["updates"] >= expected["decay_steps"]
        floor_reached = abs(float(state["last_learning_rate"] or float("nan")) - expected["decay_lr"]) <= max(1e-8, expected["base_lr"] * 2e-3)
        state["lr_milestones_consistent"] = bool(
            objects_checked and state["updates_observed"] == expected["updates"]
            and state["all_losses_finite"] and state["all_grad_norms_finite"]
            and all(math.isfinite(float(state[key])) for key in ("min_learning_rate", "max_learning_rate", "last_learning_rate"))
            and (not floor_required or floor_reached)
        )
        state["lr_floor_required"] = floor_required
        state["lr_floor_reached"] = floor_reached
        state["attestation_status"] = "VERIFIED" if state["lr_milestones_consistent"] else "FAILED"
        write_evidence()
        if state["attestation_status"] != "VERIFIED":
            raise RuntimeError(f"runtime training evidence failed attestation: {state}")

    atexit.register(write_evidence)
    return finalize


def main() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    evidence = os.environ.get("TRAINING_RUNTIME_EVIDENCE")
    finalize = None
    if evidence:
        finalize = _install_runtime_evidence(lerobot_train, Path(evidence).expanduser().resolve())
    lerobot_train.main()
    if finalize is not None:
        finalize()


if __name__ == "__main__":
    main()
