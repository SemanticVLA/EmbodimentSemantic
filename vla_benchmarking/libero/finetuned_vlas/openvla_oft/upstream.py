"""Validation helpers for the pinned OpenVLA-OFT source checkout."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path

from .contracts import (
    OPENVLA_EVAL_ENTRYPOINT,
    OPENVLA_FINETUNE_ENTRYPOINT,
    OPENVLA_OFT_UPSTREAM_COMMIT,
)


@dataclass(frozen=True)
class PinnedUpstreamCheckout:
    root: Path
    commit: str
    finetune_entrypoint: Path
    eval_entrypoint: Path


def validate_upstream_checkout(
    checkout: str | Path,
    *,
    expected_commit: str = OPENVLA_OFT_UPSTREAM_COMMIT,
) -> PinnedUpstreamCheckout:
    """Validate a local, already-cloned pinned fork without network access."""

    root = Path(checkout).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"OpenVLA-OFT upstream checkout is not a directory: {root}")
    expected = str(expected_commit).strip().lower()
    if expected != OPENVLA_OFT_UPSTREAM_COMMIT:
        raise ValueError(
            "OpenVLA-OFT upstream commit must be the pinned revision "
            f"{OPENVLA_OFT_UPSTREAM_COMMIT}, got {expected_commit!r}"
        )
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError("git is required to validate the OpenVLA-OFT checkout") from exc
    actual = result.stdout.strip().lower()
    if result.returncode != 0 or actual != expected:
        detail = result.stderr.strip() or "not a git checkout"
        raise ValueError(
            f"OpenVLA-OFT checkout commit mismatch at {root}: expected {expected}, "
            f"got {actual or detail}"
        )
    finetune = root / OPENVLA_FINETUNE_ENTRYPOINT
    evaluation = root / OPENVLA_EVAL_ENTRYPOINT
    missing = [str(path) for path in (finetune, evaluation) if not path.is_file()]
    if missing:
        raise ValueError(f"pinned OpenVLA-OFT entrypoint(s) missing: {', '.join(missing)}")
    return PinnedUpstreamCheckout(root, actual, finetune, evaluation)
