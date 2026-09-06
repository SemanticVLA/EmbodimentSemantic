"""Guarded OpenVLA-OFT evaluation command builder and entrypoint."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .contracts import OPENVLA_ARTIFACT
from .preflight import build_receipt
from .upstream import validate_upstream_checkout


def _validate_plan_file(path: Path | None) -> None:
    if path is None:
        return
    from vla_benchmarking.libero.evaluation.plan import validate_plan

    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(payload)


def build_command(
    *, python_executable: str, checkpoint: str, output_root: str,
    plan: str | Path | None = None, native: bool = False,
    dataset_manifest: str | Path | None = None, checkpoint_revision: str | None = None,
    upstream_repo: str | Path | None = None, upstream_commit: str | None = None,
) -> list[str]:
    checkout = None
    if upstream_repo is not None or upstream_commit is not None:
        if upstream_repo is None or upstream_commit is None:
            raise ValueError("upstream_repo and upstream_commit must be supplied together")
        checkout = validate_upstream_checkout(upstream_repo, expected_commit=upstream_commit)
    if native or plan is not None:
        if plan is None:
            raise ValueError("native OpenVLA-OFT evaluation requires a shared plan")
        command = [
            python_executable,
            "-m",
            "vla_benchmarking.libero.evaluation.native_vla_eval",
            "--model", "openvla_oft",
            "--plan", str(plan),
            "--checkpoint-path", checkpoint,
            "--output-jsonl", str(Path(output_root) / "native_results.jsonl"),
        ]
        if dataset_manifest is not None:
            command.extend(["--dataset-manifest", str(dataset_manifest)])
        if checkpoint_revision is not None:
            command.extend(["--checkpoint-revision", str(checkpoint_revision)])
        return command
    return [
        python_executable,
        str(checkout.eval_entrypoint) if checkout is not None else "experiments/robot/libero/run_libero_eval.py",
        "--pretrained_checkpoint", checkpoint,
        "--task_suite_name", "libero_spatial",
        "--num_trials_per_task", "50",
        "--num_open_loop_steps", "8",
        "--seed", "1000",
        "--local_log_dir", output_root,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--print-command", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint", default=OPENVLA_ARTIFACT.model_id)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="where to persist the immutable preflight receipt (default: <output-root>/preflight_receipt.json)",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--plan", type=Path, help="sealed shared evaluation plan to validate before launch")
    parser.add_argument("--upstream-repo", type=Path, help="explicit local pinned OpenVLA-OFT checkout")
    parser.add_argument("--upstream-commit", help="expected pinned OpenVLA-OFT checkout commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.plan is not None and not args.plan.is_file():
        raise SystemExit(f"evaluation plan does not exist: {args.plan}")
    if args.execute and args.plan is not None and args.checkpoint_path is None:
        raise SystemExit("native OpenVLA-OFT execution requires --checkpoint-path for the resolved local checkpoint")
    if not args.print_command and (args.upstream_repo is None or args.upstream_commit is None):
        raise SystemExit("preflight/execute requires --upstream-repo and --upstream-commit")
    if args.upstream_repo is not None or args.upstream_commit is not None:
        if args.upstream_repo is None or args.upstream_commit is None:
            raise SystemExit("--upstream-repo and --upstream-commit must be supplied together")
        validate_upstream_checkout(args.upstream_repo, expected_commit=args.upstream_commit)
    _validate_plan_file(args.plan)
    checkpoint = str(args.checkpoint_path) if args.checkpoint_path is not None else args.checkpoint
    command = build_command(
        python_executable=args.python_executable,
        checkpoint=checkpoint,
        output_root=args.output_root,
        plan=args.plan,
        native=args.plan is not None,
        dataset_manifest=args.manifest,
        checkpoint_revision=args.checkpoint_revision,
        upstream_repo=args.upstream_repo,
        upstream_commit=args.upstream_commit,
    )
    if args.print_command:
        print(shlex.join(command))
        return 0
    if not args.manifest or not args.checkpoint_revision:
        raise SystemExit("--manifest and --checkpoint-revision are required for preflight/execute")
    receipt = build_receipt(
        manifest_path=args.manifest,
        checkpoint_revision=args.checkpoint_revision,
        checkpoint_path=args.checkpoint_path,
        python_executable=args.python_executable,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    receipt_path = args.receipt_output or (Path(args.output_root) / "preflight_receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
