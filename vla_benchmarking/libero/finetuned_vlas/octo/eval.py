"""Guarded native Octo evaluation command wrapper."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from .preflight import run_preflight


DEFAULT_NATIVE_ENTRYPOINT = Path(__file__).with_name("native_eval.py")


def build_command(*, entrypoint: str | None, mode: Literal["community_eval", "matched_train"], dataset_manifest: str | Path | None, checkpoint_path: str | Path, evidence: dict[str, Any], plan: str | Path | None = None, python_executable: str = sys.executable) -> list[str]:
    if not entrypoint or not str(entrypoint).strip():
        raise ValueError("a concrete native evaluation entrypoint is required to print or execute a command")
    entrypoint_path = Path(entrypoint).expanduser().resolve()
    command_prefix = [python_executable, str(entrypoint_path)] if entrypoint_path.suffix == ".py" else [str(entrypoint_path)]
    command = command_prefix + [
        "--mode", mode,
        "--checkpoint-path", str(Path(checkpoint_path).expanduser().resolve()),
        "--policy-kind", str(evidence["policy_kind"]),
        "--checkpoint-revision", str(evidence["checkpoint_revision"]),
        "--action-horizon", str(evidence["action_horizon"]),
        "--execute-horizon", str(evidence["action_horizon"]),
    ]
    if evidence.get("checkpoint_root_path") is not None:
        command.extend(["--checkpoint-root", str(Path(evidence["checkpoint_root_path"]).expanduser().resolve())])
    if evidence.get("checkpoint_step") is not None:
        command.extend(["--checkpoint-step", str(int(evidence["checkpoint_step"]))])
    if evidence.get("runtime_sha256") is not None:
        command.extend(["--runtime-sha256", str(evidence["runtime_sha256"])])
    if evidence.get("io_sha256") is not None:
        command.extend(["--io-sha256", str(evidence["io_sha256"])])
    checkpoint_sha256 = evidence.get("checkpoint_sha256") or evidence.get("checkpoint_tree_sha256")
    if checkpoint_sha256:
        command.extend(["--checkpoint-sha256", str(checkpoint_sha256)])
    if dataset_manifest is not None:
        command[3:3] = ["--dataset-manifest", str(Path(dataset_manifest).expanduser().resolve())]
    if plan is not None:
        command.extend(["--plan", str(Path(plan).expanduser().resolve())])
    dataset_identity = evidence.get("dataset_manifest_sha256") or evidence.get("dataset_statistics_sha256")
    if dataset_identity:
        command.extend(["--dataset-manifest-sha256", str(dataset_identity)])
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("community_eval", "matched_train"), required=True)
    parser.add_argument("--dataset-manifest", required=False)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--entrypoint",
        default=str(DEFAULT_NATIVE_ENTRYPOINT),
        help="checked-in native Octo evaluator wrapper (override only for a pinned runtime)",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="where to persist the immutable preflight evidence (default: sibling preflight_receipt.json)",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, help="sealed shared evaluation plan to validate and pass to the native evaluator")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.execute and args.preflight_only:
        raise SystemExit("--execute and --preflight-only are mutually exclusive")
    if args.execute and args.plan is None:
        raise SystemExit("--plan is required for native Octo execution")
    evidence = run_preflight(
        mode=args.mode,
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
    )
    if args.plan is not None:
        if not args.plan.is_file():
            raise SystemExit(f"evaluation plan does not exist: {args.plan}")
        from vla_benchmarking.libero.evaluation.plan import validate_plan

        validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    receipt_path = args.receipt_output
    if receipt_path is None and args.dataset_manifest is not None:
        receipt_path = Path(args.dataset_manifest).expanduser().resolve().parent / "preflight_receipt.json"
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if args.execute and not Path(args.entrypoint).expanduser().is_file():
        raise SystemExit(
            "--entrypoint must point to the checked-in native Octo evaluator wrapper; "
            f"not found: {args.entrypoint}"
        )
    command = build_command(
        entrypoint=args.entrypoint,
        mode=args.mode,
        dataset_manifest=args.dataset_manifest,
        checkpoint_path=args.checkpoint_path,
        evidence=evidence,
        plan=args.plan,
        python_executable=sys.executable,
    )
    if args.print_command or not args.execute:
        print(shlex.join(command))
    if not args.execute:
        return 0
    executable = shutil.which(command[0]) or (command[0] if Path(command[0]).is_file() else None)
    if executable is None:
        raise SystemExit(f"--execute requires a resolvable executable entrypoint: {command[0]}")
    command[0] = executable
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
