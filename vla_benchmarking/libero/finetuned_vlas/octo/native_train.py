"""Checked-in Octo trainer bridge for the pinned upstream ``finetune.py``.

The repository owns the dataset registration/config contract; the pinned Octo
repository still owns its native training loop.  This bridge resolves that
script from an explicit ``OCTO_FINETUNE_ENTRYPOINT`` or ``OCTO_REPO`` and
forwards the already validated ml-collections overrides from ``train.py``.
It never downloads a checkpoint or silently selects an arbitrary trainer.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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
    command = [sys.executable, str(entrypoint), *(sys.argv[1:] if argv is None else argv)]
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":  # pragma: no cover - runtime boundary
    raise SystemExit(main())
