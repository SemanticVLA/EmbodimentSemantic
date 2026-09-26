"""LIBERO Spatial HDF5 ground-truth access, isolated to evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


LABEL_ALIASES = {
    "black_bowl_1": "akita_black_bowl_1",
    "black_bowl_2": "akita_black_bowl_2",
    "bowl_1": "akita_black_bowl_1",
    "bowl_2": "akita_black_bowl_2",
    "white_ramekin_1": "glazed_rim_porcelain_ramekin_1",
}


def canonical_label(value: object) -> str:
    text = str(value).strip()
    return LABEL_ALIASES.get(text, text)


def canonical_triplet(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return (canonical_label(value[0]), str(value[1]).strip(), canonical_label(value[2]))


def _decode_json_scalar(value: object) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "tobytes"):
        value = value.tobytes().decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"expected JSON scalar bytes/string, got {type(value).__name__}")
    return json.loads(value)


@dataclass(frozen=True)
class GroundTruthIndex:
    """Immutable index keyed by ``(task, demo, frame)``."""

    frames: dict[tuple[str, str, int], frozenset[tuple[str, str, str]]]

    def get(self, task: str, demo: str, frame: int) -> frozenset[tuple[str, str, str]]:
        return self.frames.get((task, demo, frame), frozenset())

    def keys(self):
        return self.frames.keys()


def _task_from_path(path: Path) -> str:
    return path.stem.removesuffix("_demo")


def load_ground_truth(hdf5_paths: str | Path | list[str | Path], *, camera: str = "agentview") -> GroundTruthIndex:
    """Load only the requested camera graph datasets for the evaluation phase."""

    if camera != "agentview":
        raise ValueError("SamGraph exploratory harness currently supports agentview only")
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("evaluation requires h5py; install SamGraph benchmark dependencies") from exc
    paths = [Path(hdf5_paths)] if isinstance(hdf5_paths, (str, Path)) else [Path(item) for item in hdf5_paths]
    result: dict[tuple[str, str, int], frozenset[tuple[str, str, str]]] = {}
    for path in sorted(paths):
        task = _task_from_path(path)
        with h5py.File(path, "r") as handle:
            demos = sorted(handle["data"].keys())
            for demo in demos:
                dataset_path = f"data/{demo}/obs/agentview_scene_graph"
                if dataset_path not in handle:
                    raise KeyError(f"missing {dataset_path} in {path}")
                frames = _decode_json_scalar(handle[dataset_path][()])
                if not isinstance(frames, list):
                    raise ValueError(f"{dataset_path} is not a frame list")
                for frame, raw_triplets in enumerate(frames):
                    triplets = frozenset(
                        item for value in raw_triplets for item in [canonical_triplet(value)] if item is not None
                    )
                    result[(task, str(demo), frame)] = triplets
    return GroundTruthIndex(result)


def hdf5_task_paths(hdf5_root: str | Path) -> list[Path]:
    root = Path(hdf5_root)
    if root.is_file():
        return [root]
    paths = sorted(root.glob("*_demo.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no *_demo.hdf5 files under {root}")
    return paths
