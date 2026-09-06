"""CLI entrypoint for the RoboCasa arrow-controller evaluation."""

from __future__ import annotations

from typing import Sequence

from vla_benchmarking.robocasa.evaluation.runner import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
