#!/usr/bin/env python3
import argparse
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Maps --type to the adapter class (module_path, class_name)
_TYPE_TO_CLASS = {
    "vllm":      ("vlm_bench.models.vllm_local", "VllmVLM"),
    "nvidia":    ("vlm_bench.models.nvidia",      "NvidiaVLM"),
    "nvidia-paligemma": ("vlm_bench.models.nvidia_paligemma", "NvidiaPaligemmaVLM"),
    "gemini":    ("vlm_bench.models.gemini",      "GeminiVLM"),
    "openai":    ("vlm_bench.models.openai",      "OpenAIVLM"),
    "anthropic": ("vlm_bench.models.anthropic",   "AnthropicVLM"),
    "hf":        ("vlm_bench.models.hf_local",    "HFLocalVLM"),
    "deepseek":  ("vlm_bench.models.deepseek_local", "DeepseekLocalVLM"),
}


def _build_model_from_args(args):
    import importlib
    from vlm_bench.models.base import BaseVLM

    if args.type not in _TYPE_TO_CLASS:
        raise ValueError(
            f"Unknown --type '{args.type}'. Valid types: {list(_TYPE_TO_CLASS)}"
        )

    module_path, cls_name = _TYPE_TO_CLASS[args.type]
    cls = getattr(importlib.import_module(module_path), cls_name)

    kwargs = dict(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )

    # vLLM-specific extras
    if args.type == "vllm":
        kwargs["max_model_len"] = args.max_model_len
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        if args.max_pixels is not None:
            kwargs["max_pixels"] = args.max_pixels
        if args.quantization is not None:
            kwargs["quantization"] = args.quantization

    # Gemini-specific
    if args.type == "gemini" and args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget

    instance = cls(args.model, **kwargs)
    # registry_name is used as the output subfolder name
    instance.registry_name = args.name or args.model.replace("/", "--")
    return instance


def _model_jsonl_paths(model_out_dir: Path, cameras: list[str], prompt_version: str) -> list[Path]:
    jsonl_paths = []
    for cam in cameras:
        jsonl_paths.extend(sorted((model_out_dir / cam / "json").glob(f"*_{cam}_{prompt_version}.jsonl")))
    return sorted(list(set(jsonl_paths)))


class _Tee:
    """Writes to both a file and the original stream simultaneously."""
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def close(self):
        self._file.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM Benchmarking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EXAMPLES\n"
            "  python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --type vllm --input-dir /data\n"
            "  python run.py --model mistralai/mistral-large-3-675b-instruct-2512 --type nvidia --input-dir /data\n"
            "  python run.py --model gemini-3.1-pro-preview --type gemini --input-dir /data\n"
            "  python run.py --list-models   # show known model IDs from models.yaml\n"
        ),
    )

    # ── Core ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model",   help="Model ID (HuggingFace path or API model name)")
    parser.add_argument("--type",    choices=list(_TYPE_TO_CLASS), help="Backend type")
    parser.add_argument("--name",    help="Override output folder name (default: model ID with / replaced by --)")
    parser.add_argument("--input-dir",  help="Directory containing HDF5 files or SO101 task directories")
    parser.add_argument("--output-dir", default="output", help="Root output directory")

    # ── Inference params (all optional, sensible defaults) ────────────────────
    parser.add_argument("--batch-size",   type=int,   default=8,    help="Batch size (default: 8)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max output tokens (default: 4096)")
    parser.add_argument("--temperature",  type=float, default=0.2,  help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-retries",  type=int,   default=5,    help="API retry attempts (default: 5)")

    # ── vLLM-specific ─────────────────────────────────────────────────────────
    parser.add_argument("--max-model-len",          type=int,   default=8192, help="vLLM max context length (default: 8192)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory fraction (default: 0.90)")
    parser.add_argument("--max-pixels",             type=int,   default=None, help="vLLM vision max pixels (e.g. 1048576 for Qwen)")
    parser.add_argument("--quantization",           default=None,             help="vLLM quantization (e.g. awq, fp8)")

    # ── Gemini-specific ───────────────────────────────────────────────────────
    parser.add_argument("--thinking-budget", type=int, default=None, help="Gemini thinking budget tokens")

    # ── Run control ───────────────────────────────────────────────────────────
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--models-config", default="models.yaml")
    parser.add_argument("--tasks",  type=int, default=None, help="Limit number of tasks (HDF5 files or SO101 task dirs) processed")
    parser.add_argument("--task-id", type=int, default=None, help="Run only task at this 0-based index")
    parser.add_argument("--demos",  type=int, default=None, help="Limit demos/episodes per task")
    parser.add_argument("--frames", type=int, default=None, help="Use only first N frame indices from config.yaml")
    parser.add_argument("--cameras", nargs="+", help="Override cameras (HDF5: agentview eye_in_hand; SO101: agent_view wrist)")
    parser.add_argument("--dataset-type", choices=["hdf5", "so101"], default=None,
                        help="Dataset format: hdf5 (default, auto-detected) or so101 (MP4-based LeRobot format)")
    parser.add_argument("--save-frames", nargs="?", const="debug_frames", default=None,
                        help="Save sampled input frames to a folder for inspection (default folder: debug_frames)")
    parser.add_argument("--list-models", action="store_true", help="Print known model IDs from models.yaml and exit")
    parser.add_argument("--verbose", action="store_true")

    # deprecated aliases
    parser.add_argument("--max-files", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-demos", type=int, default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def _print_model_list(models_config: dict) -> None:
    print("\nKnown models by type (models.yaml reference):\n")
    for backend_type, model_ids in models_config.items():
        if not model_ids:
            continue
        print(f"  --type {backend_type}")
        for m in model_ids:
            print(f"      {m}")
        print()


def main():
    args = parse_args()

    if args.list_models:
        with open(args.models_config) as f:
            models_config = yaml.safe_load(f)
        _print_model_list(models_config)
        return

    if not args.model:
        print("error: --model is required", file=sys.stderr)
        sys.exit(1)
    if not args.type:
        print("error: --type is required", file=sys.stderr)
        sys.exit(1)
    if not args.input_dir:
        print("error: --input-dir is required", file=sys.stderr)
        sys.exit(1)

    registry_name = args.name or args.model.replace("/", "--")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    task_suffix = f"_task{args.task_id}" if args.task_id is not None else ""
    log_path = log_dir / f"{ts}_{registry_name}{task_suffix}.log"

    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = tee_out
    sys.stderr = tee_err

    print(f"{'='*64}")
    print(f"  Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model       : {args.model}")
    print(f"  Type        : {args.type}")
    print(f"  Output name : {registry_name}")
    print(f"  Input dir   : {args.input_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Tasks       : {f'task-id {args.task_id}' if args.task_id is not None else args.tasks or 'all'}")
    print(f"  Demos       : {args.demos or 'all'}")
    print(f"  Frames      : {args.frames or 'all (from config)'}")
    print(f"  Log file    : {log_path}")
    print(f"{'='*64}\n")

    t_total = time.perf_counter()
    try:
        _main(args)
    finally:
        elapsed = time.perf_counter() - t_total
        print(f"\n{'='*64}")
        print(f"  Run finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total time   : {elapsed/3600:.2f}h ({elapsed:.0f}s)")
        print(f"{'='*64}")
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()


def _is_so101_input(input_dir: str) -> bool:
    """Detect SO101 dataset by presence of task subdirs containing a 'videos' folder."""
    root = Path(input_dir)
    for d in root.iterdir():
        if d.is_dir() and (d / "videos").is_dir():
            return True
    return False


def _main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    max_tasks = args.tasks if args.tasks is not None else args.max_files
    max_demos = args.demos if args.demos is not None else args.max_demos

    # Determine dataset type: explicit flag > auto-detect
    dataset_type = args.dataset_type
    if dataset_type is None:
        dataset_type = "so101" if _is_so101_input(args.input_dir) else "hdf5"
    print(f"  Dataset type : {dataset_type}\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model: {args.model} ...")
    t_load = time.perf_counter()
    model = _build_model_from_args(args)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded in {time.perf_counter()-t_load:.1f}s\n")

    if dataset_type == "so101":
        from vlm_bench.so101_runner import run_so101_experiment
        try:
            run_so101_experiment(
                model=model,
                config=config,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                max_tasks=max_tasks,
                max_demos=max_demos,
                max_frames=args.frames,
                task_id=args.task_id,
                cameras_to_run=args.cameras,
                save_frames_dir=args.save_frames,
                verbose=args.verbose,
            )
        finally:
            model.close()
        print("\nSO101 run complete. No ground-truth evaluation available for this dataset.")
        return

    # HDF5 path
    from vlm_bench.runner import run_experiment
    from vlm_bench.eval import aggregate, evaluate_jsonl, print_aggregate_report

    active_cameras = args.cameras if args.cameras is not None else config["extraction"].get("cameras", ["agentview"])
    config_prompt_version = config["extraction"].get("prompt_version", "v1")

    if "frame_step" in config["extraction"]:
        step = config["extraction"]["frame_step"]
        step = 1 if step == 0 else step
        frame_indices = list(range(0, config["extraction"].get("frame_max", 10000), step))
    else:
        frame_indices = config["extraction"].get("frame_indices", [0])
    active_frame_indices = frame_indices[:args.frames] if args.frames is not None else frame_indices

    try:
        run_experiment(
            model=model,
            config=config,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_tasks=max_tasks,
            max_demos=max_demos,
            max_frames=args.frames,
            task_id=args.task_id,
            cameras_to_run=active_cameras,
            output_prompt_version=config_prompt_version,
            verbose=args.verbose,
        )
    finally:
        model.close()

    model_out_dir = Path(args.output_dir) / model.registry_name
    all_results = []
    jsonl_paths = _model_jsonl_paths(model_out_dir, active_cameras, config_prompt_version)

    for jsonl_path in jsonl_paths:
        results = list(evaluate_jsonl(
            jsonl_path, Path(args.input_dir), active_cameras, active_frame_indices,
        ))
        all_results.extend(results)
        if results and (len(jsonl_paths) == 1 or args.verbose):
            print_aggregate_report(
                aggregate(results),
                label=jsonl_path.name,
            )
    if len(jsonl_paths) > 1 and all_results:
        print_aggregate_report(
            aggregate(all_results),
            label=f"{model.registry_name} — ALL FILES COMBINED",
        )


if __name__ == "__main__":
    main()
