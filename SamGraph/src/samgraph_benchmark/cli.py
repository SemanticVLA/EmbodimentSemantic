"""Small CLI for cache inspection, prediction persistence, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import importlib

from .artifacts import ArtifactStore
from .frames import discover_archives, iter_zip_frames, parse_episode_selection
from .runner import read_predictions_jsonl, write_predictions_jsonl
from .predictor import SamGraphSamPredictor
from samgraph_core.geometric_graph import geometric_relation_rules, geometric_relation_rules_sha256
from samgraph_core.geometry_profiles import samgraph_spatial_mask_geometry


def _resolution(value: str) -> int | None:
    parsed = int(value)
    return None if parsed == 0 else parsed


def _scope_task_prompts(task_prompts: object, input_tasks: set[str]) -> dict[str, object]:
    """Select prompts for the input tasks while permitting a full-suite config."""
    if not isinstance(task_prompts, dict):
        raise ValueError("name configuration must be a task-to-prompts object")
    missing_tasks = input_tasks - set(task_prompts)
    if missing_tasks:
        raise ValueError(
            f"name configuration is missing input tasks: {sorted(missing_tasks)!r}"
        )
    return {task: task_prompts[task] for task in sorted(input_tasks)}


def _artifact_output(store: ArtifactStore, requested: Path | None, default: str) -> Path:
    if requested is None:
        return store.path(default)
    # README examples conventionally spell this as ``artifacts/name.json``;
    # normalize that prefix while still rejecting every path outside the store.
    relative = requested
    if not relative.is_absolute() and relative.parts and relative.parts[0].lower() == "artifacts":
        relative = Path(*relative.parts[1:])
    return store.path(relative)


def _mask_cache_for_output(store: ArtifactStore, output: Path) -> Path:
    """Keep per-frame caches adjacent to the selected JSONL within artifacts."""
    relative_parent = output.parent.resolve().relative_to(store.root)
    return store.path(relative_parent / f"{output.stem}_masks")


def _manifest_for_output(store: ArtifactStore, output: Path) -> Path:
    """Return the one-to-one manifest path for a prediction JSONL."""
    relative_parent = output.parent.resolve().relative_to(store.root)
    return store.path(relative_parent / f"{output.stem}.manifest.json")


def _input_archive_hashes(frames_root: Path, archives: list[Path]) -> dict[str, str]:
    """Hash inputs using lexical in-root labels, following file symlinks."""
    hashes = {}
    for archive in archives:
        label = _input_archive_label(frames_root, archive)
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[label] = digest.hexdigest()
    return hashes


def _input_archive_label(frames_root: Path, archive: Path) -> str:
    """Return a relative label without resolving archive symlink targets."""
    root = Path(os.path.abspath(os.fspath(frames_root)))
    lexical_archive = Path(os.path.abspath(os.fspath(archive)))
    try:
        relative = lexical_archive.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"input archive path escapes frames root: {archive} not under {frames_root}"
        ) from exc
    if not relative.parts:
        raise ValueError(f"input archive path is the frames root, not a file: {archive}")
    return relative.as_posix()


def _source_tree_identity() -> dict[str, object]:
    """Hash loaded SamGraph Python sources using package-relative labels only."""
    digest = hashlib.sha256()
    labels = []
    for package_name in ("samgraph_core", "samgraph_benchmark"):
        package = importlib.import_module(package_name)
        package_root = Path(package.__file__).resolve().parent
        for source in sorted(package_root.rglob("*.py")):
            label = f"{package_name}/{source.relative_to(package_root).as_posix()}"
            labels.append(label)
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
    return {"sha256": digest.hexdigest(), "files": labels}


def _prediction_frame_stride(path: Path) -> int:
    """Infer the saved prediction cadence, rejecting mixed-cadence JSONL files."""
    values = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("frame_stride", 1)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"invalid frame_stride in {path}: {value!r}")
            values.add(value)
    if len(values) > 1:
        raise ValueError(f"prediction JSONL contains mixed frame strides: {sorted(values)}")
    return next(iter(values), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m samgraph_benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="validate ZIP frame ordering and emit a manifest")
    inspect.add_argument("--frames-root", type=Path, required=True)
    inspect.add_argument("--resolution", type=_resolution, default=None, help="square pre-resize; 0 keeps native JPEG size")
    inspect.add_argument("--output", type=Path, default=None)
    predict = sub.add_parser("predict", help="run pinned SAM 3.1 on sampled ZIP frames")
    predict.add_argument("--frames-root", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, default=None, help="SAM checkpoint; defaults to SAM31_CHECKPOINT")
    predict.add_argument("--resolution", type=_resolution, default=None, help="square pre-resize; 0 keeps native JPEG size")
    predict.add_argument("--frame-stride", type=int, default=1,
                         help="emit zero-based frame indices divisible by this stride")
    predict.add_argument("--tracking-stride", type=int, default=None,
                         help="process every Nth RGB frame; automatic mode requires 1")
    predict.add_argument("--mode", choices=("legacy", "automatic"), default="legacy")
    predict.add_argument("--names-config", type=Path, help="task -> object class -> ordered text descriptions")
    predict.add_argument("--episodes", default="0", help="index/list, half-open range 0:50, or all")
    predict.add_argument("--max-raw-frames", type=int, default=None,
                         help="bounded prefix for an automatic runtime canary")
    predict.add_argument("--geometry-config", type=Path, default=None, help="JSON flat geometry-rule overrides")
    predict.add_argument("--output", type=Path, default=None)
    score = sub.add_parser("score", help="evaluate persisted predictions against HDF5 GT")
    score.add_argument("--hdf5-root", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--frame-stride", type=int, default=None,
                        help="score zero-based frame indices; defaults to the saved prediction cadence")
    score.add_argument("--output", type=Path, default=None)
    tune = sub.add_parser("tune", help="offline geometry sweep over cached masks; no SAM rerun")
    tune.add_argument("--hdf5-root", type=Path, required=True)
    tune.add_argument("--predictions", type=Path, required=True, help="prediction JSONL containing mask_cache paths")
    tune.add_argument("--candidates", type=Path, required=True, help="bounded JSON list or grid of geometry candidates")
    tune.add_argument("--max-candidates", type=int, default=256)
    tune.add_argument("--frame-stride", type=int, default=None,
                       help="score zero-based frame indices; defaults to the saved prediction cadence")
    tune.add_argument("--workers", type=int, default=1,
                       help="bounded candidate-parallel CPU workers (default: 1/serial; max: 32)")
    tune.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        store = ArtifactStore()
        rows = []
        for archive in discover_archives(args.frames_root):
            frames = list(iter_zip_frames(archive, resolution=args.resolution))
            archive_label = _input_archive_label(args.frames_root, archive)
            rows.append({"task": archive.parent.name, "archive": archive_label, "frames": len(frames),
                         "source_resolution": list(frames[0].original_size), "input_resolution": [frames[0].rgb.shape[1], frames[0].rgb.shape[0]]})
        value = {"camera": "agentview", "pre_resize_resolution": args.resolution, "tasks": rows, "frames_total": sum(x["frames"] for x in rows)}
        target = _artifact_output(store, args.output, "frame_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "score":
        from .ground_truth import hdf5_task_paths, load_ground_truth
        from .metrics import evaluate_predictions
        store = ArtifactStore()
        frame_stride = args.frame_stride if args.frame_stride is not None else _prediction_frame_stride(args.predictions)
        report = evaluate_predictions(
            load_ground_truth(hdf5_task_paths(args.hdf5_root)),
            read_predictions_jsonl(args.predictions),
            frame_stride=frame_stride,
        )
        value = report.as_dict()
        target = _artifact_output(store, args.output, "evaluation.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "predict":
        store = ArtifactStore()
        checkpoint = args.checkpoint or (Path(os.environ["SAM31_CHECKPOINT"]) if os.environ.get("SAM31_CHECKPOINT") else None)
        if checkpoint is None:
            raise SystemExit("predict requires --checkpoint or SAM31_CHECKPOINT")
        geometry = samgraph_spatial_mask_geometry()
        if args.geometry_config is not None:
            geometry = json.loads(args.geometry_config.read_text(encoding="utf-8"))
        resolved_geometry = geometric_relation_rules(geometry)
        resolved_geometry_sha256 = geometric_relation_rules_sha256(geometry)
        episodes = parse_episode_selection(args.episodes)
        archives = discover_archives(args.frames_root, episodes=episodes)
        hashes = _input_archive_hashes(args.frames_root, archives)
        output = _artifact_output(store, args.output, "predictions.jsonl")
        mask_cache = _mask_cache_for_output(store, output)
        tracking_stride = args.tracking_stride
        if args.mode == "automatic":
            tracking_stride = 1 if tracking_stride is None else tracking_stride
            if tracking_stride != 1 or args.frame_stride != 5:
                raise ValueError("automatic benchmark requires tracking stride 1 and evaluation stride 5")
            if output.exists() or mask_cache.exists() or _manifest_for_output(store, output).exists():
                raise FileExistsError("automatic run output already exists; choose a new output name")
        elif args.max_raw_frames is not None:
            raise ValueError("--max-raw-frames is reserved for automatic mode")
        source_tree_before = _source_tree_identity()
        predictor_kwargs = {"geometry_rules": geometry}
        names_sha256 = None
        if args.names_config is not None:
            if args.mode != "automatic":
                raise ValueError("--names-config requires automatic mode")
            names_bytes = args.names_config.read_bytes()
            task_prompts = json.loads(names_bytes)
            input_tasks = {p.parent.name for p in archives}
            scoped_task_prompts = _scope_task_prompts(task_prompts, input_tasks)
            from samgraph_core.automatic_scene import AutomaticMaskAcquirer
            for prompts in scoped_task_prompts.values():
                AutomaticMaskAcquirer(None, prompts_by_class=prompts)
            names_sha256 = hashlib.sha256(names_bytes).hexdigest()
            # A full ten-task name map is valid for a scoped one-task repair run;
            # only input-task prompts are passed to the predictor.
            predictor_kwargs["task_prompts"] = scoped_task_prompts
        if args.mode == "automatic":
            predictor_kwargs["mode"] = "automatic"
        predictor = SamGraphSamPredictor(checkpoint, **predictor_kwargs)
        predictor.warmup()
        try:
            from .runner import PredictionRunner
            runner = PredictionRunner(predictor, resolution=args.resolution,
                                      mask_cache_dir=mask_cache,
                                      frame_stride=args.frame_stride,
                                      tracking_stride=tracking_stride,
                                      no_clobber=args.mode == "automatic",
                                      max_raw_frames=args.max_raw_frames, episodes=episodes)
            runner.run_root_to_jsonl(args.frames_root, output)
        finally:
            predictor.close()
        source_tree_after = _source_tree_identity()
        if source_tree_before != source_tree_after:
            raise RuntimeError("SamGraph source tree changed during prediction; refusing to finalize manifest")
        manifest = {
            "schema": ("samgraph.benchmark.run.v2" if args.mode == "automatic"
                       else "samgraph.benchmark.run.v1"),
            "camera": "agentview", "suite": "spatial", "mode": args.mode,
            "pre_resize_resolution": args.resolution, "frame_stride": args.frame_stride,
            "tracking_stride": tracking_stride if tracking_stride is not None else args.frame_stride,
            "max_raw_frames": args.max_raw_frames,
            "episode_selection": args.episodes,
            "names_config_sha256": names_sha256,
            "sam_internal_image_size": 1008,
            "frames_root": {"label": "agentview_frame_cache", "relative_reference": "."},
            "input_archive_sha256": hashes,
            "geometry_config": geometry, "predictor": predictor.provenance,
            "resolved_geometry_rules": resolved_geometry,
            "geometry_rules_sha256": resolved_geometry_sha256,
            "source_tree": source_tree_before,
            "source_tree_after_sha256": source_tree_after["sha256"],
            "source_tree_unchanged_after_run": True,
            "predictions": str(output.relative_to(store.root)),
            "mask_cache": str(mask_cache.relative_to(store.root)),
        }
        manifest_path = _manifest_for_output(store, output)
        store.write_json(manifest_path.relative_to(store.root), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "tune":
        from .ground_truth import hdf5_task_paths, load_ground_truth
        from .tuning import evaluate_cached_geometry, load_candidate_configs
        store = ArtifactStore()
        frame_stride = args.frame_stride if args.frame_stride is not None else _prediction_frame_stride(args.predictions)
        candidates = load_candidate_configs(args.candidates, max_candidates=args.max_candidates)
        result = evaluate_cached_geometry(
            load_ground_truth(hdf5_task_paths(args.hdf5_root)), args.predictions, candidates,
            frame_stride=frame_stride, workers=args.workers,
        )
        target = _artifact_output(store, args.output, "geometry_tuning.json")
        store.write_json(target.relative_to(store.root), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
