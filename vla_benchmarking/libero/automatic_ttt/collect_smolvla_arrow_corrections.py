"""CLI front door for genuine Arrow collection.

The production environment, SmolVLA action function, and Arrow bridge are
explicitly injected as a ``module:callable`` factory.  Invoking this command
without that factory is a fail-closed preflight; it never substitutes HDF5
expert demonstrations or a mocked controller.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from .live_collection import (
    FRESH_PEFT_METHOD_LABEL, FRESH_SOURCE_KIND, PEFT_METHOD_LABEL, SOURCE_KIND,
)


def _load(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("factory must use module:callable notation")
    module_name, attribute = spec.split(":", 1)
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory is not callable: {spec}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accepted-target", type=int, default=50)
    parser.add_argument("--adaptation-seed-start", type=int, default=3000)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--vla-step-budget", type=int, default=280)
    parser.add_argument("--arrow-step-budget", type=int, default=1200)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--controller-config-hash", required=True)
    parser.add_argument(
        "--collection-mode", choices=("fresh_arrow", "same_episode_takeover"),
        default="fresh_arrow",
        help="fresh reset Arrow demonstrations (production) or legacy takeover ablation",
    )
    parser.add_argument(
        "--reserved-eval-contract", type=Path,
        help="JSON containing reserved_eval_init_state_indices and reserved_eval_init_state_hashes",
    )
    parser.add_argument(
        "--factory",
        help="explicit module:callable that receives the collection arguments",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    is_fresh = args.collection_mode == "fresh_arrow"
    contract = {
        "source_kind": FRESH_SOURCE_KIND if is_fresh else SOURCE_KIND,
        "method_label": FRESH_PEFT_METHOD_LABEL if is_fresh else PEFT_METHOD_LABEL,
        "task_ids": [args.task_id],
        "accepted_target": args.accepted_target,
        "adaptation_seed_namespace": args.adaptation_seed_start,
        "collection_mode": args.collection_mode,
        "starts_from_reset": is_fresh,
        "vla_called": not is_fresh,
        "hdf5_fallback": False,
        "max_attempts": args.max_attempts,
        "vla_step_budget": args.vla_step_budget,
        "arrow_step_budget": args.arrow_step_budget,
        "resolution": args.resolution,
    }
    if args.dry_run:
        print(json.dumps({"status": "READY_FOR_EXPLICIT_FACTORY", **contract}, sort_keys=True))
        return 0
    if not args.factory:
        print(json.dumps({
            "status": "BLOCKED_NEEDS_LIVE_RUNTIME_FACTORY",
            "message": "Provide the native SmolVLA environment/action/Arrow factory; no fallback is allowed.",
            **contract,
        }, sort_keys=True))
        return 3
    try:
        factory = _load(args.factory)
        reserved_contract = {}
        if args.reserved_eval_contract is not None:
            reserved_contract = json.loads(args.reserved_eval_contract.read_text(encoding="utf-8"))
            if not isinstance(reserved_contract, dict):
                raise ValueError("reserved eval contract must be a JSON object")
        result = factory(
            task_id=args.task_id,
            task_description=args.task_description,
            output_root=args.output_root,
            accepted_target=args.accepted_target,
            adaptation_seed_start=args.adaptation_seed_start,
            max_attempts=args.max_attempts,
            vla_step_budget=args.vla_step_budget,
            arrow_step_budget=args.arrow_step_budget,
            resolution=args.resolution,
            controller_config_hash=args.controller_config_hash,
            collection_mode=args.collection_mode,
            reserved_eval_init_state_indices=reserved_contract.get("reserved_eval_init_state_indices"),
            reserved_eval_init_state_hashes=reserved_contract.get("reserved_eval_init_state_hashes"),
        )
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}))
        return 4
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
