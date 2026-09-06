"""Checked-in Octo trainer bridge for the pinned upstream ``finetune.py``.

The repository owns the dataset registration/config contract; the pinned Octo
repository still owns its native training loop.  This bridge resolves that
script from an explicit ``OCTO_FINETUNE_ENTRYPOINT`` or ``OCTO_REPO`` and
forwards the already validated ml-collections overrides from ``train.py``.
It never downloads a checkpoint or silently selects an arbitrary trainer.
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
import subprocess
import sys


def _flag_value(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(argv):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def select_final_checkpoint(save_dir: str | Path, *, expected_step: int) -> Path:
    """Select the numerically latest native checkpoint, never lexicographically."""

    root = Path(save_dir).expanduser().resolve()
    candidates: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        name = path.name.lower()
        if path.is_dir() and name not in {"checkpoint", "ckpt"}:
            continue
        if path.is_file() and name not in {"checkpoint", "ckpt"}:
            continue
        if path.is_dir() and not any(child.is_file() for child in path.rglob("*")):
            continue
        parts = [part for part in path.parts if part.isdigit()]
        if path.parent.name.isdigit():
            parts.append(path.parent.name)
        if not parts:
            match = re.search(r"(?:step|checkpoint|ckpt)[_-]?(\d+)", path.name, re.IGNORECASE)
            if match:
                parts.append(match.group(1))
        if parts:
            candidates.append((int(parts[-1]), path))
    if not candidates:
        raise RuntimeError(f"no numeric Octo checkpoints found under {root}")
    step, selected = max(candidates, key=lambda item: (item[0], str(item[1])))
    if int(step) != int(expected_step):
        raise RuntimeError(f"final Octo checkpoint step {step} does not equal requested final step {expected_step}")
    return selected


def _upstream_commit(entrypoint: Path) -> str:
    repository = Path(os.environ.get("OCTO_REPO", "")).expanduser().resolve()
    if not repository.is_dir():
        repository = entrypoint.parent.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = os.environ.get("OCTO_COMMIT", "").strip()
        if not value:
            raise RuntimeError("unable to determine pinned Octo upstream commit")
        return value


def write_final_checkpoint_receipt(save_dir: str | Path, checkpoint: Path, *, step: int, entrypoint: Path) -> Path:
    destination = Path(save_dir).expanduser().resolve() / "octo_final_checkpoint_receipt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    from .preflight import _checkpoint_tree_sha256, runtime_closure_paths
    from vla_benchmarking.libero.evaluation.policy_adapter import derive_runtime_receipt

    runtime = derive_runtime_receipt(closure_paths=runtime_closure_paths(), require_clean=True)
    payload = {
        "schema": "octo_final_checkpoint_receipt.v1",
        "checkpoint_path": str(checkpoint),
        "step": int(step),
        "checkpoint_sha256": _checkpoint_tree_sha256(checkpoint),
        "octo_commit": _upstream_commit(entrypoint),
        "runtime_sha256": runtime["sha256"],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def resolve_upstream_entrypoint() -> Path:
    explicit = os.environ.get("OCTO_FINETUNE_ENTRYPOINT", "").strip()
    if explicit:
        target = Path(explicit).expanduser().resolve()
    else:
        repository = os.environ.get("OCTO_REPO", "").strip()
        if not repository:
            raise SystemExit(
                "native Octo training requires OCTO_FINETUNE_ENTRYPOINT or OCTO_REPO "
                "pointing to the pinned Octo checkout"
            )
        target = Path(repository).expanduser().resolve() / "scripts" / "finetune.py"
    if not target.is_file():
        raise SystemExit(f"pinned Octo scripts/finetune.py was not found: {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    entrypoint = resolve_upstream_entrypoint()
    forwarded = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, str(entrypoint), *forwarded]
    result = int(subprocess.run(command, check=False).returncode)
    if result != 0:
        return result
    save_dir = _flag_value(forwarded, "--config.save_dir")
    final_steps = _flag_value(forwarded, "--config.num_steps")
    if not save_dir or final_steps is None:
        raise SystemExit("native Octo training must declare save_dir and final num_steps")
    selected = select_final_checkpoint(save_dir, expected_step=int(final_steps))
    receipt = write_final_checkpoint_receipt(save_dir, selected, step=int(final_steps), entrypoint=entrypoint)
    print(json.dumps({"final_checkpoint": str(selected), "receipt": str(receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - runtime boundary
    raise SystemExit(main())
