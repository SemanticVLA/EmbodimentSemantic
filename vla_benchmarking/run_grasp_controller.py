#!/usr/bin/env python3
"""Run the sole frozen canonical RGB-D/MolmoPoint grasp controller."""

from __future__ import annotations

try:
    from .grasp_controller.entrypoint import main
except ImportError:  # pragma: no cover - direct script execution
    from grasp_controller.entrypoint import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
