"""Materialize the canonical 500-demo LIBERO source for Octo training.

The conversion is deterministic and streams one episode at a time.  It uses
the same HDF5 reader and action/frame conventions as the checked-in Pi0.5 and
OpenVLA integrations, then writes the RLDS source and the strict completion
manifest consumed by Octo preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from vla_benchmarking.libero.finetuned_vlas.common.libero_hdf5 import iter_source_frames
from vla_benchmarking.libero.finetuned_vlas.octo.config import MATCHED_TRAIN_CONFIG
from vla_benchmarking.libero.finetuned_vlas.octo.dataset import (
    _encoded_image,
    _decode_image,
    build_tfds_dataset,
    fingerprint_steps,
    serialize_canonical_episode,
)
from vla_benchmarking.libero.finetuned_vlas.octo.manifest import (
    build_completion_manifest,
    write_completion_manifest,
)


def _source_revision(data_root: Path) -> str:
    manifest = data_root / "libero_spatial_v5_source_manifest.json"
    if manifest.is_file():
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(data_root.glob("*.hdf5")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _iter_serialized_steps(data_root: Path, *, demos_per_task: int) -> Iterator[dict[str, Any]]:
    frames = iter_source_frames(data_root, task_ids=range(10), demos_per_task=demos_per_task, image_size=256)
    current_id: str | None = None
    current_frames: list[Any] = []
    current_actions: list[Any] = []
    current_instruction = ""

    def flush() -> Iterable[dict[str, Any]]:
        if current_id is None:
            return ()
        return serialize_canonical_episode(
            (frame.image_primary for frame in current_frames),
            current_actions,
            instruction=current_instruction,
            episode_id=current_id,
        )

    for frame in frames:
        if current_id is not None and frame.episode_id != current_id:
            yield from flush()
            current_frames.clear()
            current_actions.clear()
        current_id = frame.episode_id
        current_instruction = frame.language_instruction
        current_frames.append(frame)
        current_actions.append(frame.action)
    if current_id is not None:
        yield from flush()


def _decoded_records(source_jsonl: Path) -> Iterator[dict[str, Any]]:
    with source_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield {
                "observation": {"image_primary": _decode_image(record["image"])},
                "action": record["action"],
                "task": {"language_instruction": record["language_instruction"]},
                "episode_id": record["episode_id"],
                "frame_id": record["frame_id"],
                "is_first": record["is_first"],
                "is_last": record["is_last"],
                "is_terminal": record["is_terminal"],
                "frame_provenance": record.get("frame_provenance", {}),
                "action_space": record.get("action_space", "octo_libero_dataset_v1"),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--demos-per-task", type=int, default=50)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    source_jsonl = run_root / "octo_source.jsonl"
    manifest_path = run_root / "octo_completion_manifest.json"
    materialized = args.dataset_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Stream the JSONL file; only one source episode is held in memory.
    episode_count = 0
    step_count = 0
    last_episode = None
    with source_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for step in _iter_serialized_steps(data_root, demos_per_task=int(args.demos_per_task)):
            episode_id = str(step["episode_id"])
            if episode_id != last_episode:
                episode_count += 1
                last_episode = episode_id
            step_count += 1
            encoded = {
                "episode_id": episode_id,
                "frame_id": int(step["frame_id"]),
                "image": _encoded_image(step["observation"]["image_primary"]),
                "action": [float(value) for value in step["action"]],
                "language_instruction": str(step["task"]["language_instruction"]),
                "is_first": bool(step["is_first"]),
                "is_last": bool(step["is_last"]),
                "is_terminal": bool(step["is_terminal"]),
                "frame_provenance": dict(step.get("frame_provenance") or {}),
                "action_space": str(step.get("action_space", "octo_libero_dataset_v1")),
            }
            handle.write(json.dumps(encoded, sort_keys=True, separators=(",", ":")) + "\n")
    if episode_count != MATCHED_TRAIN_CONFIG.episodes:
        raise RuntimeError(f"LIBERO source did not contain exactly 500 episodes; got {episode_count}")
    if step_count != MATCHED_TRAIN_CONFIG.timesteps:
        raise RuntimeError(
            f"LIBERO source did not contain exactly {MATCHED_TRAIN_CONFIG.timesteps} transitions; got {step_count}"
        )
    fingerprint = fingerprint_steps(_decoded_records(source_jsonl))
    manifest = build_completion_manifest(
        config=MATCHED_TRAIN_CONFIG,
        fingerprint=fingerprint,
        source_revision=_source_revision(data_root),
    )
    write_completion_manifest(manifest_path, manifest)
    build_tfds_dataset(source_jsonl, materialized)
    # The output is a compact receipt; the source and materialization are
    # archived by the enclosing SLURM wrapper.
    receipt = {
        "schema": "octo_prepare_receipt.v1",
        "source_jsonl": str(source_jsonl),
        "completion_manifest": str(manifest_path),
        "dataset_root": str(materialized),
        "fingerprint": fingerprint,
    }
    (run_root / "prepare_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
