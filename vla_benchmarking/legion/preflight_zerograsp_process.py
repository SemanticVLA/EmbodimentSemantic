"""Bounded ZeroGrasp JSONL process ready/close preflight.

This verifies the explicitly supplied checkpoint digest before importing the
adapter or starting the evaluation-repository worker. It then starts the real
worker, waits for its protocol ``ready`` reply (which includes official
backend/checkpoint construction), and closes it without sending inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def verify_checkpoint_sha256(path: str | Path, expected: str) -> tuple[Path, str]:
    """Canonicalize and hash a checkpoint before any worker can deserialize it."""
    expected = str(expected)
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("--checkpoint-sha256 must be a lowercase 64-character SHA-256")
    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is not a regular file: {checkpoint}")
    digestor = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digestor.update(chunk)
    actual = digestor.hexdigest()
    if actual != expected:
        raise ValueError("checkpoint SHA-256 mismatch")
    return checkpoint, actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args(argv)

    try:
        # This must remain before adapter import and process construction.
        checkpoint, checkpoint_sha256 = verify_checkpoint_sha256(args.checkpoint, args.checkpoint_sha256)
        evaluation_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(evaluation_root))
        from zerograsp_adapter import ZeroGraspProcessAdapter
        from zerograsp_contracts import ZeroGraspConfig

        config = ZeroGraspConfig(
            external_repo=str(Path(args.repo).resolve()),
            checkpoint=str(checkpoint),
            runtime_config=str(Path(args.config).resolve()),
            request_timeout_s=args.timeout_s,
        ).validate()
        command = [
            str(Path(args.python).resolve()),
            str(Path(args.worker).resolve()),
            "--repo",
            str(Path(args.repo).resolve()),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(Path(args.config).resolve()),
        ]
        adapter = ZeroGraspProcessAdapter(command, config, timeout_s=args.timeout_s)
        ready = False
        try:
            adapter.start()
            ready = True
        finally:
            adapter.close()
        print(json.dumps({"protocol": adapter.protocol, "protocol_ready": ready, "closed": True, "runtime_hash": adapter.runtime_hash, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ready": False, "closed": True, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
