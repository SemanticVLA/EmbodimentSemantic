from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .proxy.artifacts import ArtifactStore
from .proxy.config import DEFAULT_CONFIG_PATH, load_config
from .proxy.dataset import write_dataset_index
from .proxy.fusion import generate_proxy_artifacts
from .proxy.metadata_signals import extract_metadata
from .proxy.validation import validate_proxy_files
from .vision.import_bboxes import import_bbox_jsonl
from .vision.agent_bbox_rules import generate_agent_bboxes
from .vision.wrist_bbox_rules import generate_wrist_bboxes


def _paths(config: dict[str, Any]) -> tuple[ArtifactStore, dict[str, Path]]:
    paths = {name: Path(value).resolve() for name, value in config["paths"].items()}
    artifacts = ArtifactStore(paths["artifacts"])
    artifacts.ensure_layout()
    return artifacts, paths


def _ensure_index(config: dict[str, Any], artifacts: ArtifactStore, paths: dict[str, Path]) -> Path:
    index_path = artifacts.path("index/episodes.jsonl", create_parent=False)
    if not index_path.exists():
        write_dataset_index(paths["so101_dataset"], artifacts)
    return index_path


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _default_bbox_path(artifacts: ArtifactStore) -> Path:
    agent_view = artifacts.path("bboxes/agent_view.jsonl", create_parent=False)
    imported = artifacts.path("bboxes/imported.jsonl", create_parent=False)
    if agent_view.exists():
        return agent_view
    if imported.exists():
        return imported
    raise FileNotFoundError(
        "No bbox artifact found. Run 'detect --sampled' or 'import-bboxes <path>' first."
    )


def _default_wrist_bbox_path(artifacts: ArtifactStore) -> Path | None:
    path = artifacts.path("bboxes/wrist.jsonl", create_parent=False)
    return path if path.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo.so101_proxy_demo",
        description="Isolated SO101 2D Proxy GT pipeline",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit-data", help="Index SO101 episodes and sampled JPEGs")
    subparsers.add_parser("extract-metadata", help="Extract gripper, lift, and phase signals")

    detect = subparsers.add_parser("detect", help="Run optional Grounding DINO/SAM2 bbox detection")
    detect.add_argument("--sampled", action="store_true", help="Detect the existing sampled JPEGs")
    detect.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional agent episode limit (or wrist frame limit)",
    )
    detect.add_argument(
        "--camera",
        choices=("agent_view", "wrist"),
        default="agent_view",
        help="Camera to process; defaults to optimized fixed agent view",
    )
    detect.add_argument("--task", default=None)
    detect.add_argument("--episode", default=None)
    import_boxes = subparsers.add_parser("import-bboxes", help="Import versioned bbox JSONL")
    import_boxes.add_argument("source")

    generate = subparsers.add_parser("generate-proxy", help="Generate merged Proxy GT")
    generate.add_argument("--bboxes", default=None)
    generate.add_argument(
        "--wrist-bboxes",
        default=None,
        help="Optional wrist bbox JSONL; defaults to artifacts/bboxes/wrist.jsonl when present",
    )

    validate = subparsers.add_parser("validate", help="Validate generated Proxy GT artifacts")
    validate.add_argument(
        "--camera",
        choices=("agent_view", "wrist", "all"),
        default="agent_view",
        help="Camera coverage to validate; defaults to agent view",
    )

    serve = subparsers.add_parser("serve", help="Launch the standalone browser demo")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--no-open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    artifacts, paths = _paths(config)

    if args.command == "audit-data":
        _print(write_dataset_index(paths["so101_dataset"], artifacts))
        return 0

    episode_index = _ensure_index(config, artifacts, paths)
    sampled_index = artifacts.path("index/sampled_frames.jsonl", create_parent=False)

    if args.command == "extract-metadata":
        _print(
            extract_metadata(
                str(paths["so101_dataset"]),
                str(episode_index),
                artifacts,
                config,
            )
        )
        return 0

    if args.command == "detect":
        if not args.sampled:
            raise SystemExit("Only sampled detection is supported initially; pass --sampled")
        metadata_path = artifacts.path("metadata/frame_signals.jsonl", create_parent=False)
        if not metadata_path.exists():
            extract_metadata(
                str(paths["so101_dataset"]),
                str(episode_index),
                artifacts,
                config,
            )
        if args.camera == "agent_view":
            _print(
                generate_agent_bboxes(
                    sampled_index,
                    metadata_path,
                    episode_index,
                    artifacts,
                    config,
                    task=args.task,
                    episode=args.episode,
                    limit_episodes=args.limit,
                )
            )
        else:
            _print(
                generate_wrist_bboxes(
                    sampled_index,
                    metadata_path,
                    episode_index,
                    artifacts,
                    config,
                    task=args.task,
                    episode=args.episode,
                    limit_frames=args.limit,
                )
            )
        return 0

    if args.command == "import-bboxes":
        _print(import_bbox_jsonl(args.source, artifacts))
        return 0

    if args.command == "generate-proxy":
        metadata_path = artifacts.path("metadata/frame_signals.jsonl", create_parent=False)
        if not metadata_path.exists():
            extract_metadata(
                str(paths["so101_dataset"]),
                str(episode_index),
                artifacts,
                config,
            )
        bbox_path = Path(args.bboxes).resolve() if args.bboxes else _default_bbox_path(artifacts)
        wrist_bbox_path = (
            Path(args.wrist_bboxes).resolve()
            if args.wrist_bboxes
            else _default_wrist_bbox_path(artifacts)
        )
        _print(
            generate_proxy_artifacts(
                bbox_path,
                metadata_path,
                episode_index,
                artifacts,
                config,
                wrist_bbox_path=wrist_bbox_path,
            )
        )
        return 0

    if args.command == "validate":
        cameras = None if args.camera == "all" else (args.camera,)
        _print(validate_proxy_files(artifacts.root, artifacts, cameras=cameras))
        return 0

    if args.command == "serve":
        from demo.so101_backend import serve

        serve(
            config,
            artifacts,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open_browser,
        )
        return 0
    raise AssertionError(args.command)
