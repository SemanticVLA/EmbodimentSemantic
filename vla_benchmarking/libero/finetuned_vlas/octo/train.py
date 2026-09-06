"""Guarded Octo matched-training command wrapper.

By default this performs preflight and prints the command.  Native training
is only invoked with explicit ``--execute`` and an executable entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import (
    A40_BATCH_LADDER,
    A40_MEMORY_LIMIT_GB,
    BatchCandidate,
    EFFECTIVE_BATCH,
    MATCHED_TRAIN_CONFIG,
    choose_a40_batch_candidate,
)
from .preflight import run_preflight
from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt


DEFAULT_NATIVE_ENTRYPOINT = Path(__file__).with_name("native_train.py")


MEMORY_RECEIPT_SCHEMA = "octo_a40_memory_receipt.v1"


def _read_memory_receipt(path: str | Path) -> tuple[dict[int, float], BatchCandidate, dict[str, Any]]:
    """Read and validate the complete two-update A40 measurement receipt."""

    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"A40 memory measurement receipt is unreadable: {target}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MEMORY_RECEIPT_SCHEMA:
        raise ValueError(f"A40 memory receipt must declare schema {MEMORY_RECEIPT_SCHEMA!r}")
    device = payload.get("device")
    if not isinstance(device, dict) or "model" not in device or "memory_gb" not in device:
        raise ValueError("A40 receipt requires device.model and device.memory_gb")
    if "A40" not in str(device["model"]).upper() or float(device["memory_gb"]) != 48.0:
        raise ValueError("A40 receipt device identity/memory must be NVIDIA A40 with 48 GB")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ValueError("A40 receipt requires exact run identities")
    for field in ("checkpoint_revision", "dataset_manifest_sha256", "config_sha256", "runtime_sha256", "octo_commit"):
        value = str(run.get(field, "")).strip()
        if not value or (field.endswith("sha256") and len(value) != 64) or value.lower() == "unknown":
            raise ValueError(f"A40 receipt run identity is missing/invalid: {field}")
    smoke = payload.get("smoke_test")
    if not isinstance(smoke, dict) or smoke.get("completed") is not True or int(smoke.get("updates", -1)) != 2:
        raise ValueError("A40 receipt requires a completed two-update smoke test")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("A40 receipt requires a candidate list")
    measurements: dict[int, float] = {}
    ladder_by_microbatch = {candidate.microbatch: candidate for candidate in A40_BATCH_LADDER}
    for item in candidates:
        if not isinstance(item, dict) or not {"microbatch", "gradient_accumulation_steps", "status", "microsteps"}.issubset(item):
            raise ValueError("A40 receipt candidates require microbatch, accumulation, status, and microsteps")
        microbatch = int(item["microbatch"])
        if microbatch not in ladder_by_microbatch:
            raise ValueError("A40 receipt contains a candidate outside the sealed ladder")
        if int(item["gradient_accumulation_steps"]) != ladder_by_microbatch[microbatch].gradient_accumulation_steps:
            raise ValueError("A40 receipt candidate accumulation disagrees with the sealed ladder")
        if microbatch in measurements:
            raise ValueError("A40 receipt contains duplicate candidate measurements")
        expected_microsteps = 2 * ladder_by_microbatch[microbatch].gradient_accumulation_steps
        if int(item["microsteps"]) != expected_microsteps:
            raise ValueError("A40 receipt candidate did not run two optimizer updates")
        status = str(item["status"]).upper()
        if status == "FAILED":
            continue
        if status != "PASS" or "peak_vram_gb" not in item:
            raise ValueError("A40 candidate must record PASS plus peak_vram_gb, or FAILED without selection")
        peak = float(item["peak_vram_gb"])
        if not math.isfinite(peak) or peak < 0:
            raise ValueError("A40 candidate peak_vram_gb must be finite and non-negative")
        measurements[microbatch] = peak
    selected = payload.get("selected_candidate")
    if not isinstance(selected, dict) or not {"microbatch", "gradient_accumulation_steps"}.issubset(selected):
        raise ValueError("A40 receipt requires selected_candidate")
    selected_candidate = BatchCandidate(int(selected["microbatch"]), int(selected["gradient_accumulation_steps"]))
    if selected_candidate not in A40_BATCH_LADDER:
        raise ValueError("A40 receipt selected_candidate is outside the sealed ladder")
    if selected_candidate.microbatch not in measurements:
        raise ValueError("A40 receipt selected_candidate was not a successful smoke candidate")
    return measurements, selected_candidate, run


def _load_memory_measurements(path: str | Path) -> dict[int, float]:
    """Read measured peaks from a strict sealed A40 receipt."""

    return _read_memory_receipt(path)[0]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_two_update_memory_smoke(candidate: BatchCandidate, step_fn: Any) -> dict[str, Any]:
    """Run exactly two optimizer updates and record candidate status.

    ``step_fn`` is invoked once per microstep and may return a mapping with a
    measured ``peak_vram_gb`` value.  A failed candidate is explicitly marked
    and is excluded by receipt parsing/selection.
    """

    microsteps = 2 * int(candidate.gradient_accumulation_steps)
    peaks: list[float] = []
    try:
        for index in range(microsteps):
            result = step_fn(index)
            if isinstance(result, Mapping) and result.get("peak_vram_gb") is not None:
                peak = float(result["peak_vram_gb"])
                if not math.isfinite(peak) or peak < 0:
                    raise ValueError("peak_vram_gb must be finite and non-negative")
                peaks.append(peak)
        if len(peaks) == 0:
            raise ValueError("memory smoke did not report peak_vram_gb")
        return {
            "microbatch": candidate.microbatch,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "microsteps": microsteps,
            "status": "PASS",
            "peak_vram_gb": max(peaks),
        }
    except Exception as exc:
        return {
            "microbatch": candidate.microbatch,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "microsteps": microsteps,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }


def resolve_batch_candidate(
    *,
    measurements: str | Path | None = None,
    microbatch: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> BatchCandidate:
    """Resolve one measured candidate and reject unsealed batch recipes."""

    if measurements is None:
        raise ValueError("actual Octo training requires an A40 memory measurement receipt")
    measured_values, declared_candidate, _run_identity = _read_memory_receipt(measurements)
    measured = next(
        (candidate for candidate in A40_BATCH_LADDER
         if candidate.microbatch in measured_values and measured_values[candidate.microbatch] <= A40_MEMORY_LIMIT_GB),
        None,
    )
    if measured is None:
        raise RuntimeError("all successful A40 memory-smoke candidates exceed the sealed VRAM limit")
    if declared_candidate != measured:
        raise ValueError("A40 receipt selected_candidate disagrees with measured ladder selection")
    explicit = None
    if microbatch is not None or gradient_accumulation_steps is not None:
        if microbatch is None or gradient_accumulation_steps is None:
            raise ValueError("--microbatch and --gradient-accumulation-steps must be supplied together")
        explicit = next(
            (
                candidate
                for candidate in A40_BATCH_LADDER
                if candidate.microbatch == int(microbatch)
                and candidate.gradient_accumulation_steps == int(gradient_accumulation_steps)
            ),
            None,
        )
        if explicit is None:
            raise ValueError("requested batch recipe is not in the sealed A40 ladder")
    if explicit is not None and measured != explicit:
        raise ValueError("explicit A40 batch candidate disagrees with the first measured candidate that fits")
    selected = measured
    if selected.effective_batch != EFFECTIVE_BATCH:
        raise AssertionError("selected A40 recipe changed the sealed effective batch")
    return selected


def build_command(
    *,
    entrypoint: str | None,
    dataset_manifest: str | Path,
    checkpoint_path: str | Path,
    evidence: dict[str, Any],
    python_executable: str = sys.executable,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_candidate: BatchCandidate | None = None,
    dataset_root: str | Path | None = None,
) -> list[str]:
    if not entrypoint or not str(entrypoint).strip():
        raise ValueError("a concrete native training entrypoint is required to print or execute a command")
    updates = evidence.get("optimizer_updates")
    if updates is None or int(updates) <= 0:
        raise ValueError("training command requires optimizer updates derived by preflight")
    manifest = Path(dataset_manifest).expanduser().resolve()
    materialized_root = Path(dataset_root or manifest.parent).expanduser().resolve()
    config = Path(config_path or (Path(__file__).resolve().parent / "native_finetune_config.py")).resolve()
    destination = Path(output_dir or (manifest.parent / "octo_runs")).expanduser().resolve()
    # These are the actual ml-collections overrides consumed by the pinned
    # Octo ``scripts/finetune.py``.  The verified completion manifest remains
    # the source of the update count; callers cannot substitute a hand value.
    entrypoint_path = Path(entrypoint).expanduser().resolve()
    if batch_candidate is None and entrypoint_path.name == DEFAULT_NATIVE_ENTRYPOINT.name:
        raise ValueError("native Octo training command requires a measured A40 batch candidate")
    command_prefix = [python_executable, str(entrypoint_path)] if entrypoint_path.suffix == ".py" else [str(entrypoint_path)]
    checkpoint_root = Path(evidence.get("checkpoint_root_path", checkpoint_path)).expanduser().resolve()
    checkpoint_step = evidence.get("checkpoint_step")
    native_steps = int(updates) * (batch_candidate.gradient_accumulation_steps if batch_candidate else 1)
    accumulation = batch_candidate.gradient_accumulation_steps if batch_candidate else 1
    command = command_prefix + [
        f"--config={config}:full,language_conditioned",
        f"--config.pretrained_path={checkpoint_root}",
        *([f"--config.pretrained_step={int(checkpoint_step)}"] if checkpoint_step is not None else []),
        f"--config.dataset_kwargs.data_dir={materialized_root}",
        f"--config.save_dir={destination}",
        f"--config.num_steps={int(updates)}",
        f"--config.save_interval={native_steps}",
        f"--config.batch_size={(batch_candidate.microbatch if batch_candidate else EFFECTIVE_BATCH)}",
        "--config.seed=1000",
        "--config.window_size=1",
        # Optax MultiSteps advances the wrapped schedule only on optimizer
        # updates, so keep the sealed schedule in optimizer-update units even
        # though the outer native loop runs one step per microbatch.
        f"--config.optimizer.learning_rate.warmup_steps={int(MATCHED_TRAIN_CONFIG.warmup_steps)}",
        f"--config.optimizer.learning_rate.decay_steps={int(updates)}",
        "--config.optimizer.weight_decay=0.01",
        "--config.optimizer.clip_gradient=1.0",
    ]
    if batch_candidate is not None:
        command[command.index(f"--config.num_steps={int(updates)}")] = (
            f"--config.num_steps={native_steps}"
        )
        command.append(
            f"--config.optimizer.grad_accumulation_steps={batch_candidate.gradient_accumulation_steps}"
        )
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--entrypoint",
        default=str(DEFAULT_NATIVE_ENTRYPOINT),
        help="checked-in native Octo trainer wrapper (override only for the pinned upstream script)",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, help="materialized TFDS/RLDS root (defaults to manifest directory)")
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="where to persist the immutable preflight evidence (default: sibling preflight_receipt.json)",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true", help="execute only after preflight and explicit entrypoint resolution")
    parser.add_argument("--memory-measurements", type=Path, help="JSON receipt of measured A40 peak VRAM by microbatch")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--gradient-accumulation-steps", "--accumulation", dest="gradient_accumulation_steps", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.execute and args.preflight_only:
        raise SystemExit("--execute and --preflight-only are mutually exclusive")
    evidence = run_preflight(
        mode="matched_train",
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
    )
    if args.execute:
        from .preflight import runtime_closure_paths

        runtime = derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)
        if str(runtime["sha256"]) != str(evidence["runtime_sha256"]):
            raise SystemExit("native Octo runtime closure changed after preflight")
    receipt_path = args.receipt_output or (Path(args.dataset_manifest).expanduser().resolve().parent / "preflight_receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not (args.print_command or args.execute):
        return 0
    if (args.print_command or args.execute):
        try:
            batch_candidate = resolve_batch_candidate(
                measurements=args.memory_measurements,
                microbatch=args.microbatch,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
    else:
        batch_candidate = None
    if args.entrypoint is not None and args.execute and not Path(args.entrypoint).expanduser().is_file():
        raise SystemExit(
            "--entrypoint must point to the checked-in native Octo wrapper or pinned Octo scripts/finetune.py; "
            f"not found: {args.entrypoint}"
        )
    if args.memory_measurements is not None:
        _, _, receipt_run = _read_memory_receipt(args.memory_measurements)
        if str(receipt_run["checkpoint_revision"]) != str(evidence["checkpoint_revision"]):
            raise SystemExit("A40 receipt checkpoint_revision does not match Octo preflight")
        if str(receipt_run["dataset_manifest_sha256"]) != str(evidence.get("dataset_manifest_sha256")):
            raise SystemExit("A40 receipt dataset_manifest_sha256 does not match the completed manifest")
        if str(receipt_run["runtime_sha256"]) != str(evidence["runtime_sha256"]):
            raise SystemExit("A40 receipt runtime_sha256 does not match the native runtime receipt")
        config_file = Path(args.config_path or (Path(__file__).resolve().parent / "native_finetune_config.py")).resolve()
        if str(receipt_run["config_sha256"]) != _sha256_file(config_file):
            raise SystemExit("A40 receipt config_sha256 does not match native_finetune_config.py")
    command = build_command(
        entrypoint=args.entrypoint,
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
        evidence=evidence,
        python_executable=args.python_executable,
        config_path=args.config_path,
        output_dir=args.output_dir,
        batch_candidate=batch_candidate,
        dataset_root=args.dataset_root,
    )
    if args.print_command or not args.execute:
        print(shlex.join(command))
    if not args.execute:
        return 0
    executable = shutil.which(command[0]) or (command[0] if Path(command[0]).is_file() else None)
    if executable is None:
        raise SystemExit(f"--execute requires a resolvable executable entrypoint: {command[0]}")
    command[0] = executable
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
