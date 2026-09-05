#!/usr/bin/env python3
"""Run the sole frozen canonical RGB-D/MolmoPoint grasp controller."""

from __future__ import annotations

import sys
from pathlib import Path

# Running by path is the supported Legion/local entrypoint.  Make the checkout
# root visible before importing the package implementation.
if __package__ in {None, ""}:  # pragma: no cover - direct script smoke
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

try:
    from .controller.entrypoint import main
except ImportError:  # pragma: no cover - direct script execution
    from vla_benchmarking.arrow_grasp_controller.controller.entrypoint import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
