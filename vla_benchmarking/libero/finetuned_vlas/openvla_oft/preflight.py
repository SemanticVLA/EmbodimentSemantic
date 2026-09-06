"""Preflight and receipt generation for OpenVLA-OFT."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .contracts import OPENVLA_ARTIFACT, OPENVLA_DATASET_NAME, OPENVLA_IO, OPENVLA_TRAINING, optimizer_updates_for
from .dataset import NOOP_FILTER_NAME, NOOP_FILTER_THRESHOLD
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
        raise FileNotFoundError(f"OpenVLA-OFT manifest does not exist: {target}")
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenVLA-OFT manifest is not valid JSON: {target}") from exc
    if manifest.get("model") != "openvla_oft" or manifest.get("format") != "rlds":
        raise ValueError("OpenVLA-OFT preflight requires an openvla_oft RLDS manifest")
    if manifest.get("dataset_name", OPENVLA_DATASET_NAME) != OPENVLA_DATASET_NAME:
        raise ValueError(f"OpenVLA-OFT manifest dataset_name must be {OPENVLA_DATASET_NAME!r}")
    if manifest.get("arrow_condition") != "none":
        raise ValueError("OpenVLA-OFT manifest is not no-arrow")
    if not SHA256_RE.fullmatch(str(manifest.get("content_sha256", ""))):
        raise ValueError("manifest content_sha256 must be a lowercase SHA-256 digest")
    episodes = int(manifest.get("episodes", 0))
    timesteps = int(manifest.get("timesteps", manifest.get("frames", 0)))
    lengths = manifest.get("episode_lengths", {})
    if episodes <= 0 or timesteps <= 0 or not isinstance(lengths, Mapping):
        raise ValueError("OpenVLA-OFT manifest must contain positive episode/timestep counts")
    if sum(int(value) for value in lengths.values()) != timesteps:
        raise ValueError("OpenVLA-OFT episode_lengths do not sum to timesteps")
    noop_filter = manifest.get("noop_filter")
    if not isinstance(noop_filter, Mapping):
        raise ValueError("OpenVLA-OFT manifest requires verified no-op filter provenance")
    if noop_filter.get("dataset_name") != OPENVLA_DATASET_NAME or noop_filter.get("filter") != NOOP_FILTER_NAME:
        raise ValueError("OpenVLA-OFT manifest no-op provenance is not the pinned filter")
    if float(noop_filter.get("threshold", -1)) != NOOP_FILTER_THRESHOLD:
        raise ValueError("OpenVLA-OFT manifest no-op threshold is not pinned")
    if int(noop_filter.get("output_frames", -1)) != timesteps:
        raise ValueError("OpenVLA-OFT manifest no-op output count does not match timesteps")
    source_frames = int(noop_filter.get("source_frames", -1))
    if source_frames < timesteps or int(noop_filter.get("dropped_frames", -1)) != source_frames - timesteps:
        raise ValueError("OpenVLA-OFT manifest no-op before/after counts are inconsistent")
    if not all(isinstance(noop_filter.get(key), str) and SHA256_RE.fullmatch(noop_filter[key]) for key in ("source_sha256", "output_sha256")):
        raise ValueError("OpenVLA-OFT manifest no-op provenance requires source/output SHA-256 values")
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
        # Preflight is a development-time inspection boundary; the complete
        # closure digest remains immutable even when this worktree is dirty.
        # Native execution performs the strict clean-source check separately.
        **derive_runtime_receipt(python_executable=python_executable, require_clean=False),
    }
    resolved_checkpoint = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"OpenVLA-OFT checkpoint path does not exist: {checkpoint}")
        resolved_checkpoint = str(checkpoint)
        checkpoint_sha256 = derive_checkpoint_receipt(
            checkpoint,
            artifact_id=OPENVLA_ARTIFACT.model_id,
            checkpoint_revision=checkpoint_revision,
        )["checkpoint_sha256"]
    else:
        checkpoint_sha256 = None
    io = derive_io_receipt(
        {
            "action_horizon": OPENVLA_IO.action_horizon,
            "action_dim": OPENVLA_IO.action_dim,
            "input_resolution": OPENVLA_IO.image_size[0],
            "camera_keys": list(OPENVLA_IO.camera_keys),
            "state_dim": OPENVLA_IO.state_dim,
            "visual_input": "none",
        }
    )
    timesteps = int(manifest["timesteps"])
    updates = optimizer_updates_for(timesteps, OPENVLA_TRAINING.epochs, OPENVLA_TRAINING.effective_batch_size)
    training = OPENVLA_TRAINING.as_dict()
    training.update({"source_timesteps": timesteps, "optimizer_updates": updates})
    return {
        "schema_version": 1,
        "status": "preflight_ok",
        "model": OPENVLA_ARTIFACT.model_id,
        "checkpoint_revision": str(checkpoint_revision).lower(),
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_file_sha256": manifest_sha256,
        "manifest_content_sha256": manifest["content_sha256"],
        "dataset_manifest_id": f"dataset:{manifest_sha256}",
        "dataset_manifest_sha256": manifest_sha256,
        "dataset": {
            "name": OPENVLA_DATASET_NAME,
            "episodes": int(manifest["episodes"]),
            "timesteps": int(manifest["timesteps"]),
        },
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
