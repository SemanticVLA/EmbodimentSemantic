"""Legion entry point for the language-free arrow SmolVLA experiment.

The SLURM wrapper invokes this module once per A/B/C/D stage.  It intentionally
does not use ``lerobot-train``: the policy input contract has changed, so the
small loop below owns the bridge losses and the durable checkpoint protocol.
LeRobot is imported only after argument validation and only on a compute node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from .config import STAGE_SCHEDULE, StageName
from .lerobot_integration import (
    ArrowSmolVLAPolicy,
    ArrowStageTrainer,
    SmolVLATeacherAdapter,
    TeacherBatch,
    _pad_last,
    load_pinned_smolvla,
)


def _checkpoint_dir(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if (resolved / "pretrained_model").is_dir():
        resolved = resolved / "pretrained_model"
    if not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {resolved}")
    return resolved


def _require_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - compute-node only
        raise RuntimeError("LeRobot dataset support is missing; install requirements-lora.txt") from exc
    return LeRobotDataset


class _PairedDataset(Dataset):
    def __init__(self, arrow: Dataset, clean: Dataset) -> None:
        if len(arrow) != len(clean):
            raise ValueError(f"arrow/clean datasets have different frame counts: {len(arrow)} != {len(clean)}")
        self.arrow, self.clean = arrow, clean

    def __len__(self) -> int:
        return len(self.arrow)

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrow, clean = self.arrow[index], self.clean[index]
        if arrow.get("task") != clean.get("task"):
            raise ValueError(f"paired dataset task mismatch at frame {index}")
        # The two views are allowed to differ only in their rendered image.
        # Check lineage and supervision before adding the clean teacher image;
        # otherwise a shuffled or mismatched pair would create silent leakage.
        for key in ("episode_index", "frame_index", "timestamp"):
            present_arrow, present_clean = key in arrow, key in clean
            if present_arrow != present_clean:
                raise ValueError(f"paired dataset {key} mismatch at frame {index}")
            if present_arrow:
                left, right = torch.as_tensor(arrow[key]), torch.as_tensor(clean[key])
                if left.shape != right.shape or not torch.equal(left, right):
                    raise ValueError(f"paired dataset {key} mismatch at frame {index}")
        for key in ("observation.state", "action"):
            if key not in arrow or key not in clean:
                raise ValueError(f"paired dataset is missing raw {key} at frame {index}")
            left, right = torch.as_tensor(arrow[key]), torch.as_tensor(clean[key])
            if left.shape != right.shape or not torch.equal(left, right):
                raise ValueError(f"paired dataset raw {key} mismatch at frame {index}")
        return {**arrow, "teacher_image": clean["observation.images.image"], "teacher_state": clean["observation.state"]}


def _find_clean_root(arrow_root: Path, explicit: str | None) -> Path:
    if explicit:
        clean = Path(explicit).expanduser().resolve()
    else:
        sibling = arrow_root.parent / "control"
        if not sibling.is_dir():
            raise FileNotFoundError(
                f"clean teacher dataset is missing beside the arrow dataset: {sibling}; "
                "pass --teacher-dataset-root explicitly"
            )
        clean = sibling
    if not clean.is_dir():
        raise FileNotFoundError(f"clean teacher dataset is missing: {clean}; pass --teacher-dataset-root")
    return clean


def _dataset(repo_root: Path, *, clean: bool = False) -> Dataset:
    LeRobotDataset = _require_lerobot_dataset()
    name = f"local/{repo_root.name}"
    # SmolVLA is trained on an action chunk, not a single frame.  Request the
    # pinned 20 Hz, 50-step horizon explicitly; constructor fallback keeps the
    # launcher compatible with older LeRobot releases that read this from the
    # dataset configuration.
    delta = {
        "observation.images.image": [0.0],
        "observation.state": [0.0],
        "action": [i / 20.0 for i in range(50)],
    }
    try:
        return LeRobotDataset(repo_id=name, root=repo_root, delta_timestamps=delta)
    except TypeError:
        return LeRobotDataset(repo_id=name, root=repo_root)


def _tokenize_tasks(policy: Any, tasks: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    normalized_tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
    encoded = tokenizer(
        normalized_tasks,
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    tokens = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].bool().to(device)
    return tokens, mask


def _normalizer_path(checkpoint: Path) -> Path | None:
    path = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    return path if path.is_file() else None


def _resolve_normalizer_path(teacher_checkpoint: Path, initial_checkpoint: Path, base_policy: Path | None) -> tuple[Path, dict[str, Any]]:
    teacher_path = _normalizer_path(teacher_checkpoint)
    initial_path = _normalizer_path(initial_checkpoint)
    if teacher_path and initial_path:
        teacher_hash = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
        initial_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        if teacher_hash != initial_hash:
            raise ValueError("teacher and initial checkpoints contain different normalizer statistics")
        return teacher_path, {"source": "teacher_and_initial_checkpoint", "sha256": teacher_hash}
    if teacher_path or initial_path:
        path = teacher_path or initial_path
        assert path is not None
        return path, {"source": "single_checkpoint_normalizer", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if base_policy is None:
        raise FileNotFoundError("checkpoint normalizer statistics are absent; --base-policy is required for the documented base fallback")
    fallback = _normalizer_path(base_policy)
    if fallback is None:
        raise FileNotFoundError(f"normalizer statistics are absent from both checkpoints and base policy: {base_policy}")
    return fallback, {"source": "base_policy_fallback_checkpoint_stats_absent", "sha256": hashlib.sha256(fallback.read_bytes()).hexdigest()}


class _PolicyNormalizer:
    """Apply the frozen SmolVLA MEAN_STD state/action contract."""

    def __init__(self, source_path: Path, device: torch.device) -> None:
        path = source_path
        if not path.is_file():
            raise FileNotFoundError(f"normalizer state is missing from the selected checkpoint source: {path}")
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - compute-node only
            raise RuntimeError("safetensors is required to load the pinned SmolVLA normalizer") from exc
        values = load_file(str(path), device="cpu")
        try:
            # Keep statistics on CPU: DataLoader collation happens before the
            # trainer moves tensors to CUDA. Each call follows the input device.
            self.state_mean = values["observation.state.mean"].float()
            self.state_std = values["observation.state.std"].float().clamp_min(1e-8)
            self.action_mean = values["action.mean"].float()
            self.action_std = values["action.std"].float().clamp_min(1e-8)
        except KeyError as exc:
            raise ValueError(f"pinned normalizer is missing required tensor: {exc}") from exc

    def state(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() - self.state_mean.to(value.device)) / self.state_std.to(value.device)

    def action(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() - self.action_mean.to(value.device)) / self.action_std.to(value.device)


def _collate_factory(policy: Any, device: torch.device, normalizer: _PolicyNormalizer):
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        from torch.utils.data._utils.collate import default_collate

        batch = default_collate(items)
        tasks = items[0].get("task")
        if tasks is None:
            raise KeyError("LeRobot arrow dataset must contain task strings for teacher distillation")
        task_list = [item["task"] for item in items]
        tokens, mask = _tokenize_tasks(policy, task_list, device)
        # A one-element delta timestamp is still returned with a time axis by
        # LeRobot.  The student contract consumes the latest agentview frame
        # and one proprioceptive state vector, while actions retain the full
        # 50-step chunk.
        if batch["observation.images.image"].ndim == 5:
            batch["observation.images.image"] = batch["observation.images.image"][:, -1]
        if batch["observation.state"].ndim == 3:
            batch["observation.state"] = batch["observation.state"][:, -1]
        batch["observation.state"] = normalizer.state(batch["observation.state"])
        if batch["observation.state"].shape[-1] < 32:
            padded_state = torch.zeros(*batch["observation.state"].shape[:-1], 32, dtype=batch["observation.state"].dtype)
            padded_state[..., : batch["observation.state"].shape[-1]] = batch["observation.state"]
            batch["observation.state"] = padded_state
        if "action" not in batch or batch["action"].ndim != 3:
            raise ValueError("LeRobot dataset must return a [batch, 50, 7] action chunk")
        batch["action"] = normalizer.action(batch["action"])
        teacher_image = batch["teacher_image"]
        if teacher_image.ndim == 5:
            teacher_image = teacher_image[:, -1]
        teacher_image = teacher_image.float()
        if float(teacher_image.detach().max()) > 1.5:
            teacher_image = teacher_image / 255.0
        prepared_teacher_images, prepared_teacher_masks = policy.prepare_images(
            {"observation.images.image": teacher_image}
        )
        teacher_image = prepared_teacher_images[0]
        teacher_image_mask = prepared_teacher_masks[0].to(device)
        teacher_state = batch["teacher_state"].float()
        if teacher_state.ndim == 3:
            teacher_state = teacher_state[:, -1]
        teacher_state = normalizer.state(teacher_state)
        if teacher_state.shape[-1] < 32:
            padded = torch.zeros(*teacher_state.shape[:-1], 32, dtype=teacher_state.dtype)
            padded[..., : teacher_state.shape[-1]] = teacher_state
            teacher_state = padded
        batch["teacher"] = TeacherBatch(
            images=[teacher_image.to(device)],
            image_masks=[teacher_image_mask],
            language_tokens=tokens,
            language_mask=mask,
            state=teacher_state.to(device),
        )
        return batch

    return collate


def _iter_dataset(arrow_root: Path, clean_root: Path, policy: Any, device: torch.device, batch_size: int, normalizer: _PolicyNormalizer) -> DataLoader:
    arrow = _dataset(arrow_root)
    clean = _dataset(clean_root)
    paired = _PairedDataset(arrow, clean)
    return DataLoader(paired, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", collate_fn=_collate_factory(policy, device, normalizer))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=False, choices=[stage.name.value for stage in STAGE_SCHEDULE])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--teacher-dataset-root", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--base-policy", default=os.environ.get("ARROW_BASE_POLICY"), help="local pinned SmolVLA snapshot used to merge PEFT checkpoints")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--libero-config-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="load real checkpoints/data and run one update of every stage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.smoke and args.stage is None:
        raise ValueError("--stage is required unless --smoke is supplied")
    stage = StageName(args.stage) if args.stage is not None else None
    if args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("batch size and save frequency must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    arrow_root = Path(args.dataset_root).expanduser().resolve()
    clean_root = _find_clean_root(arrow_root, args.teacher_dataset_root)
    for required in (run_root, output_root, checkpoint_root):
        required.mkdir(parents=True, exist_ok=True)
    if stage is not None:
        _write_json(run_root / "stages" / f"{stage.name}.started.json", {"stage": stage.value, "seed": args.seed})

    teacher_checkpoint = _checkpoint_dir(args.teacher_checkpoint)
    initial_checkpoint = _checkpoint_dir(args.initial_checkpoint)
    base_policy = Path(args.base_policy).expanduser().resolve() if args.base_policy else None
    normalizer_path, normalizer_provenance = _resolve_normalizer_path(teacher_checkpoint, initial_checkpoint, base_policy)
    _write_json(run_root / "normalizer_provenance.json", {"path": str(normalizer_path), **normalizer_provenance})
    teacher_policy = load_pinned_smolvla(teacher_checkpoint, base_policy=args.base_policy, device=str(device))
    initial_policy = load_pinned_smolvla(initial_checkpoint, base_policy=args.base_policy, device=str(device))
    normalizer = _PolicyNormalizer(normalizer_path, device)
    # Keep trainable vision/connector weights in FP32.  The pinned VLM loads
    # those modules in BF16; AdamW updates at 1e-6 would otherwise be below one
    # BF16 ULP and Stage D would appear to run while leaving them unchanged.
    student = ArrowSmolVLAPolicy(initial_policy).float()
    teacher = SmolVLATeacherAdapter(teacher_policy)
    loader = _iter_dataset(arrow_root, clean_root, teacher_policy, device, args.batch_size, normalizer)
    trainer = ArrowStageTrainer(student, checkpoint_root, seed=args.seed, save_every=args.save_every, device=device)
    if args.smoke:
        smoke_root = run_root / "integration_smoke"
        smoke_trainer = ArrowStageTrainer(
            student,
            smoke_root,
            seed=args.seed,
            save_every=1,
            device=device,
        )
        smoke_results = []
        for stage_config in STAGE_SCHEDULE:
            result = smoke_trainer.run_stage(
                stage_config.name,
                loader,
                teacher=teacher,
                updates=1,
                resume=False,
            )
            smoke_results.append(result)
        _write_json(run_root / "integration_smoke.json", {"status": "passed", "stages": smoke_results})
        return 0
    assert stage is not None
    result = trainer.run_stage(stage, loader, teacher=teacher, updates=args.updates, resume=args.resume)
    _write_json(run_root / "stages" / f"{stage.name}.result.json", result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
