"""Preflight and receipt generation for the Pi0.5 experiment."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .contracts import PI05_ARTIFACT, PI05_IO, PI05_TRAINING, micro_steps_for, optimizer_updates_for
from vla_benchmarking.libero.evaluation.policy_adapter import (
    derive_checkpoint_receipt,
    derive_io_receipt,
    derive_runtime_receipt,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Pi0.5 manifest does not exist: {target}")
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pi0.5 manifest is not valid JSON: {target}") from exc
    if manifest.get("model") != "pi05":
        raise ValueError("Pi0.5 preflight requires a pi05 dataset manifest")
    if manifest.get("arrow_condition") != "none":
        raise ValueError("Pi0.5 manifest is not no-arrow")
    for key in ("content_sha256",):
        if not SHA256_RE.fullmatch(str(manifest.get(key, ""))):
            raise ValueError(f"manifest {key} must be a lowercase SHA-256 digest")
    episodes = int(manifest.get("episodes", 0))
    timesteps = int(manifest.get("timesteps", manifest.get("frames", 0)))
    if episodes <= 0 or timesteps <= 0:
        raise ValueError("Pi0.5 manifest must contain positive episode and timestep counts")
    lengths = manifest.get("episode_lengths", {})
    if not isinstance(lengths, Mapping) or sum(int(value) for value in lengths.values()) != timesteps:
        raise ValueError("Pi0.5 episode_lengths do not sum to timesteps")
    return manifest, _file_sha256(target)


def validate_runtime(
    python_executable: str | Path = sys.executable,
    *,
    device: str = "cuda",
    require_cuda: bool = False,
) -> dict[str, Any]:
    executable = Path(python_executable).expanduser()
    resolved = str(executable.resolve()) if executable.exists() else shutil.which(str(python_executable))
    if not resolved:
        raise FileNotFoundError(f"runtime Python executable was not found: {python_executable}")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    cuda_available = None
    if require_cuda:
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
        except ImportError as exc:
            raise RuntimeError("--require-cuda requires torch in the preflight environment") from exc
        if not cuda_available:
            raise RuntimeError("CUDA was required but is not available")
    return {"python": str(resolved), "device": device, "cuda_available": cuda_available}


def build_receipt(
    *,
    manifest_path: str | Path,
    checkpoint_revision: str,
    checkpoint_path: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    device: str = "cuda",
    require_cuda: bool = False,
) -> dict[str, Any]:
    if not REVISION_RE.fullmatch(str(checkpoint_revision)):
        raise ValueError("checkpoint_revision must be an immutable 40-character commit SHA")
    manifest, manifest_sha256 = validate_manifest(manifest_path)
    runtime = {
        **validate_runtime(python_executable, device=device, require_cuda=require_cuda),
        # Preflight may be inspected from the active development worktree;
        # the receipt still hashes the complete closure and records its exact
        # content state. Native execution remains strict about clean sources.
        **derive_runtime_receipt(python_executable=python_executable, require_clean=False),
    }
    resolved_checkpoint = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Pi0.5 checkpoint path does not exist: {checkpoint}")
        resolved_checkpoint = str(checkpoint)
        checkpoint_sha256 = derive_checkpoint_receipt(
            checkpoint,
            artifact_id=PI05_ARTIFACT.model_id,
            checkpoint_revision=checkpoint_revision,
        )["checkpoint_sha256"]
    else:
        checkpoint_sha256 = None
    io = derive_io_receipt(
        {
            "action_horizon": PI05_IO.action_horizon,
            "action_dim": PI05_IO.action_dim,
            "input_resolution": PI05_IO.image_size[0],
            "camera_keys": list(PI05_IO.camera_keys),
            "state_dim": PI05_IO.state_dim,
            "visual_input": "none",
        }
    )
    timesteps = int(manifest.get("timesteps") or manifest.get("frames", 0))
    updates = optimizer_updates_for(timesteps, PI05_TRAINING.epochs, PI05_TRAINING.effective_batch_size)
    training = PI05_TRAINING.as_dict()
    training.update(
        {
            "source_timesteps": timesteps,
            "optimizer_updates": updates,
            "micro_steps": micro_steps_for(
                timesteps,
                PI05_TRAINING.epochs,
                PI05_TRAINING.effective_batch_size,
                PI05_TRAINING.gradient_accumulation_steps,
            ),
        }
    )
    return {
        "schema_version": 1,
        "status": "preflight_ok",
        "model": PI05_ARTIFACT.model_id,
        "checkpoint_revision": str(checkpoint_revision).lower(),
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_file_sha256": manifest_sha256,
        "manifest_content_sha256": manifest["content_sha256"],
        "dataset_manifest_id": f"dataset:{manifest_sha256}",
        "dataset_manifest_sha256": manifest_sha256,
        "dataset": {"episodes": int(manifest["episodes"]), "timesteps": timesteps},
        "runtime": runtime,
        "io": io,
        "training": training,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()
    receipt = build_receipt(
        manifest_path=args.manifest,
        checkpoint_revision=args.checkpoint_revision,
        checkpoint_path=args.checkpoint_path,
        python_executable=args.python_executable,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.receipt_output:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
