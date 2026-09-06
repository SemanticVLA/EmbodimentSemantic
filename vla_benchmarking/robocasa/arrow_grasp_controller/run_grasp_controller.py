#!/usr/bin/env python3
"""Run RoboCasa Pick & Place preflight or terminal-accounting skeleton."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

try:
    from .controller.entrypoint import main
except ImportError:  # pragma: no cover - direct script execution
    from vla_benchmarking.robocasa.arrow_grasp_controller.controller.entrypoint import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
