"""Dry-run utilities for the Pi0.5 integration."""

from __future__ import annotations

import argparse
import json

from .policy import Pi05Adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print sealed config; never load/train")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is available until a training launcher is approved")
    print(json.dumps(Pi05Adapter().dry_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
