"""Install or validate the isolated RoboCasa benchmark environment.

This helper deliberately does not share the LIBERO environment: RoboCasa
requires a newer robosuite/MuJoCo stack.  Running without ``--install`` is a
read-only import/version check.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PACKAGE_ROOT / "requirements.txt"


def _version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def check_environment() -> int:
    """Check Python and required imports without creating an environment."""

    if sys.version_info < (3, 11):
        print(f"ERROR: RoboCasa requires Python 3.11+; found {sys.version}")
        return 1
    missing: list[str] = []
    for module_name in ("robocasa", "robosuite", "mujoco", "cv2"):
        try:
            print(f"{module_name}: {_version(module_name)}")
        except (ImportError, OSError) as exc:
            missing.append(f"{module_name} ({exc})")
    if missing:
        print("Missing dependencies:")
        for item in missing:
            print(f"  - {item}")
        print(f"Install with: {sys.executable} -m pip install -r {REQUIREMENTS}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="install the pinned RoboCasa requirements before checking imports",
    )
    args = parser.parse_args(argv)
    if args.install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            check=True,
        )
    return check_environment()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["check_environment", "main"]
