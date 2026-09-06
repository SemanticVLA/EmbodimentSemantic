"""Dry-run utilities for OpenVLA-OFT integration."""

from __future__ import annotations

import argparse
import json

from .policy import OpenVLAOFTAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print sealed config; never load/train")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is available until a training launcher is approved")
    print(json.dumps(OpenVLAOFTAdapter().dry_run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
