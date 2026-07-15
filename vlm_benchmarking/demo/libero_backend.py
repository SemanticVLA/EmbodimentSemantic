from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import mimetypes
import os
import pickle
import queue
import sys
import threading
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import h5py
import numpy as np
from PIL import Image

from .http_helpers import safe_child_path, send_bytes, send_file, send_html, send_json
from vlm_bench.io_utils import list_hdf5_files, parse_triplets


CAMERA_INFO = {
    "agentview": {
        "rgb": "agentview_rgb",
        "bboxes": "agentview_bboxes",
        "scene_graph": "agentview_scene_graph",
        "sim": "agentview",
    },
    "eye_in_hand": {
        "rgb": "eye_in_hand_rgb",
        "bboxes": "robot0_eye_in_hand_bboxes",
        "scene_graph": "robot0_eye_in_hand_scene_graph",
        "sim": "robot0_eye_in_hand",
    },
}

RELATION_LABELS = {
    "is_left_of": "left",
    "is_right_of": "right",
    "is_in_front_of": "front",
    "is_behind": "behind",
    "is_on_top_of": "top",
    "is_below_of": "below",
    "is_inside": "inside",
    "contains": "contains",
}

DEFAULT_SUBJECT = "akita_black_bowl_1"
DEMO_BUILD = "scene-graph-demo-2026-07-13-default-1024-v27"
RESOLUTION_OPTIONS = [512, 768, 1024]
PREDICTION_FRAME_STRIDE = 5
DEMO_ROOT = Path(__file__).resolve().parent
COMMON_STATIC_ROOT = DEMO_ROOT / "common"

ICLR_PRESETS = [
    {
        "id": "gt-agentview",
        "label": "GT Agentview",
        "description": "Clean ground-truth overlay",
        "camera": "agentview",
        "res": 1024,
        "mode": "gt",
        "task": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "demo": "demo_0",
        "frame": 0,
        "subject": DEFAULT_SUBJECT,
    },
    {
        "id": "prediction-agentview",
        "label": "Agentview Prediction",
        "description": "GT vs prediction from the fixed scene camera",
        "camera": "agentview",
        "res": 1024,
        "mode": "compare",
        "preferred_model": "gemini-3.1-pro-preview",
        "task": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "demo": "demo_0",
        "frame": 0,
        "subject": DEFAULT_SUBJECT,
    },
    {
        "id": "wrist-view",
        "label": "Eye-in-Hand GT",
        "description": "Wrist-camera ground truth",
        "camera": "eye_in_hand",
        "res": 1024,
        "mode": "gt",
        "task": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "demo": "demo_0",
        "frame": 20,
        "subject": DEFAULT_SUBJECT,
    },
    {
        "id": "prediction-eye-in-hand",
        "label": "Eye-in-Hand Prediction",
        "description": "GT vs prediction from the wrist camera",
        "camera": "eye_in_hand",
        "res": 1024,
        "mode": "compare",
        "preferred_model": "gemini-3.1-pro-preview",
        "task": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "demo": "demo_0",
        "frame": 0,
        "subject": DEFAULT_SUBJECT,
    },
]

Triplet = tuple[str, str, str]
_DUPLICATE_BOWL_OBJECTS = ("akita_black_bowl_1", "akita_black_bowl_2")
_DUPLICATE_BOWL_SWAP = {
    _DUPLICATE_BOWL_OBJECTS[0]: _DUPLICATE_BOWL_OBJECTS[1],
    _DUPLICATE_BOWL_OBJECTS[1]: _DUPLICATE_BOWL_OBJECTS[0],
}


def _decode_json_dataset(dataset: h5py.Dataset) -> Any:
    raw = dataset[()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.ndarray):
        text = raw.tobytes().decode("utf-8")
    else:
        text = str(raw)
    return json.loads(text)


def relation_label(relation: str) -> str:
    return RELATION_LABELS.get(relation, relation.removeprefix("is_").replace("_of", "").replace("_", " "))


def _frame_f1(gt: set[Triplet], pred: set[Triplet]) -> float:
    if not gt and not pred:
        return 1.0
    tp = len(gt & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _swap_duplicate_bowls(triplets: list[Triplet] | set[Triplet]) -> list[Triplet]:
    return [
        (_DUPLICATE_BOWL_SWAP.get(a, a), rel, _DUPLICATE_BOWL_SWAP.get(b, b))
        for a, rel, b in triplets
    ]


def duplicate_bowl_invariant_prediction(gt: set[Triplet], pred: list[Triplet]) -> tuple[list[Triplet], bool]:
    """Mirror eval.py's frame-level duplicate-bowl assignment for visual correctness colors."""
    pred_set = set(pred)
    swapped = _swap_duplicate_bowls(pred)
    swapped_set = set(swapped)
    if swapped_set != pred_set and _frame_f1(gt, swapped_set) > _frame_f1(gt, pred_set):
        deduped = list(dict.fromkeys(swapped))
        return deduped, True
    return list(dict.fromkeys(pred)), False


def prediction_metrics(gt: set[Triplet], pred: set[Triplet]) -> dict[str, Any]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    recall = tp / len(gt) if gt else (1.0 if not pred else 0.0)
    f1 = _frame_f1(gt, pred)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def task_name_from_hdf5(path: Path) -> str:
    return path.stem.removesuffix("_demo")


def rotate_bbox_180(
    bbox: list[float] | tuple[float, float, float, float],
    source_size: tuple[int, int],
) -> list[float]:
    width, height = source_size
    x1, y1, x2, y2 = bbox
    return [width - x2, height - y2, width - x1, height - y1]


def scale_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    rotate180: bool = False,
) -> list[float]:
    if rotate180:
        bbox = rotate_bbox_180(bbox, source_size)
    source_w, source_h = source_size
    target_w, target_h = target_size
    x_scale = target_w / source_w
    y_scale = target_h / source_h
    x1, y1, x2, y2 = bbox
    return [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def mat_to_quat_wxyz(mat: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(mat, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )

    diagonal = np.diag(matrix)
    axis = int(np.argmax(diagonal))
    if axis == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=float,
        )
    elif axis == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )

    norm = np.linalg.norm(quat)
    return quat / norm if norm else quat


def relation_edges(
    triplets: list[list[str]] | list[tuple[str, str, str]],
    bboxes: dict[str, list[float]],
    subject: str,
    correct_triplets: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    edges = []
    subject_filter = subject.strip()
    use_all = subject_filter.lower() == "all"
    for item in triplets:
        if len(item) != 3:
            continue
        obj_a, relation, obj_b = item
        if not use_all and obj_a != subject_filter:
            continue
        if obj_a not in bboxes or obj_b not in bboxes:
            continue
        triplet = (obj_a, relation, obj_b)
        start = bbox_center(bboxes[obj_a])
        end = bbox_center(bboxes[obj_b])
        edges.append(
            {
                "subject": obj_a,
                "relation": relation,
                "label": relation_label(relation),
                "object": obj_b,
                "correct": True if correct_triplets is None else triplet in correct_triplets,
                "start": [round(start[0], 3), round(start[1], 3)],
                "end": [round(end[0], 3), round(end[1], 3)],
            }
        )
    return edges


def relation_triplets(
    triplets: list[list[str]] | list[tuple[str, str, str]],
    bboxes: dict[str, list[float]],
    subject: str,
    correct_triplets: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    subject_filter = subject.strip()
    use_all = subject_filter.lower() == "all"
    items = []
    for item in triplets:
        if len(item) != 3:
            continue
        obj_a, relation, obj_b = item
        triplet = (obj_a, relation, obj_b)
        visible = obj_a in bboxes and obj_b in bboxes
        selected = use_all or obj_a == subject_filter
        items.append(
            {
                "subject": obj_a,
                "relation": relation,
                "label": relation_label(relation),
                "object": obj_b,
                "visible": visible,
                "selected": selected,
                "correct": True if correct_triplets is None else triplet in correct_triplets,
            }
        )
    return items


@dataclass(frozen=True)
class TaskInfo:
    id: str
    name: str
    path: Path


@dataclass
class DemoData:
    states: np.ndarray
    frame_count: int
    source_size: tuple[int, int]
    bboxes: list[dict[str, list[int]]]
    graphs: list[list[list[str]]]
    world_coords: list[dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class PredictionRunInfo:
    id: str
    name: str
    model: str
    camera: str
    task: str
    path: Path


class PredictionStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._runs: dict[str, PredictionRunInfo] = {}
        self._run_cache: dict[str, dict[tuple[str, str, int], list[tuple[str, str, str]]]] = {}
        self._scan_runs()

    def _scan_runs(self) -> None:
        if not self.output_dir.exists():
            return
        for path in sorted(self.output_dir.glob("*/*/json/*.jsonl")):
            try:
                model, camera = path.relative_to(self.output_dir).parts[:2]
            except ValueError:
                continue
            if camera not in CAMERA_INFO:
                continue
            stem = path.stem
            marker = f"_{camera}_"
            if marker not in stem:
                continue
            task = stem.split(marker, 1)[0]
            run_id = path.relative_to(self.output_dir).as_posix()
            version = stem.split(marker, 1)[1]
            display_model = model.replace("--", "/").replace("_", " ")
            self._runs[run_id] = PredictionRunInfo(
                id=run_id,
                name=f"{display_model} ({camera.replace('_', ' ')}, {version})",
                model=model,
                camera=camera,
                task=task,
                path=path,
            )

    def list_runs(self, task: str, camera: str) -> list[dict[str, str]]:
        runs = [
            run
            for run in self._runs.values()
            if run.task == task and run.camera == camera
        ]
        return [
            {"id": run.id, "name": run.name, "model": run.model, "camera": run.camera}
            for run in sorted(runs, key=lambda item: (item.model.lower(), item.name.lower()))
        ]

    def triplets_for_frame(
        self,
        run_id: str,
        demo: str,
        camera: str,
        frame: int,
    ) -> list[tuple[str, str, str]]:
        if not run_id:
            return []
        index = self._load_run(run_id)
        return index.get((demo, camera, frame), [])

    def has_frame(self, run_id: str, demo: str, camera: str, frame: int) -> bool:
        if not run_id:
            return False
        index = self._load_run(run_id)
        return (demo, camera, frame) in index

    def _load_run(self, run_id: str) -> dict[tuple[str, str, int], list[tuple[str, str, str]]]:
        cached = self._run_cache.get(run_id)
        if cached is not None:
            return cached
        if run_id not in self._runs:
            raise KeyError(f"Unknown prediction run '{run_id}'")
        index: dict[tuple[str, str, int], list[tuple[str, str, str]]] = {}
        with open(self._runs[run_id].path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_demo = str(rec.get("demo", ""))
                rec_camera = str(rec.get("camera", ""))
                try:
                    rec_frame = int(rec.get("frame", 0))
                except (TypeError, ValueError):
                    continue
                triplets = parse_triplets(rec.get("response", ""))
                key = (rec_demo, rec_camera, rec_frame)
                existing = index.setdefault(key, [])
                seen = set(existing)
                for triplet in triplets:
                    if triplet not in seen:
                        seen.add(triplet)
                        existing.append(triplet)
        self._run_cache[run_id] = index
        return index


class Hdf5SemanticCache:
    def __init__(self, cache_dir: str | Path | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _token(self, hdf5_path: Path, demo: str) -> str:
        resolved = hdf5_path.resolve()
        stat = resolved.stat()
        key = json.dumps(
            {
                "path": str(resolved),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "demo": demo,
                "build": DEMO_BUILD,
                "camera_info": CAMERA_INFO,
            },
            sort_keys=True,
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def path_for(self, hdf5_path: Path, demo: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._token(hdf5_path, demo)}.pkl"

    def get(self, hdf5_path: Path, demo: str) -> dict[str, DemoData] | None:
        path = self.path_for(hdf5_path, demo)
        if path is None or not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            return None
        if not isinstance(value, dict) or not all(camera in value for camera in CAMERA_INFO):
            return None
        return value

    def put(self, hdf5_path: Path, demo: str, data: dict[str, DemoData]) -> None:
        path = self.path_for(hdf5_path, demo)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)


class Hdf5SceneGraphStore:
    _shared_demo_cache: dict[tuple[str, str], dict[str, DemoData]] = {}
    _shared_demo_lock = threading.Lock()

    def __init__(
        self,
        input_dir: str | Path,
        camera: str,
        rotate_agentview: bool = True,
        allowed_demos: set[str] | frozenset[str] | None = None,
        semantic_cache_dir: str | Path | None = None,
    ):
        if camera not in CAMERA_INFO:
            raise ValueError(f"Unknown camera '{camera}'. Expected one of {sorted(CAMERA_INFO)}")
        self.input_dir = Path(input_dir)
        self.camera = camera
        self.rotate_agentview = rotate_agentview
        self.allowed_demos = frozenset(allowed_demos) if allowed_demos else None
        self.semantic_cache = Hdf5SemanticCache(semantic_cache_dir)
        self._task_index = {
            task_name_from_hdf5(path): TaskInfo(
                id=task_name_from_hdf5(path),
                name=task_name_from_hdf5(path).replace("_", " "),
                path=path,
            )
            for path in list_hdf5_files(str(self.input_dir))
        }
        if not self._task_index:
            raise FileNotFoundError(f"No HDF5 files found in {self.input_dir}")
        self._demo_cache: dict[tuple[str, str], DemoData] = {}
        self._demo_list_cache: dict[str, list[dict[str, Any]]] = {}

    @property
    def camera_info(self) -> dict[str, str]:
        return CAMERA_INFO[self.camera]

    @property
    def rotate_current_camera(self) -> bool:
        return self.camera == "agentview" and self.rotate_agentview

    def list_tasks(self) -> list[dict[str, str]]:
        return [
            {"id": info.id, "name": info.name, "path": str(info.path)}
            for info in sorted(self._task_index.values(), key=lambda item: item.id)
        ]

    def task_path(self, task_id: str) -> Path:
        try:
            return self._task_index[task_id].path
        except KeyError as exc:
            raise KeyError(f"Unknown task '{task_id}'") from exc

    def list_demos(self, task_id: str) -> list[dict[str, Any]]:
        if task_id in self._demo_list_cache:
            return self._demo_list_cache[task_id]
        path = self.task_path(task_id)
        rgb_key = self.camera_info["rgb"]
        with h5py.File(path, "r") as f:
            demos = []
            for demo in sorted(f["data"].keys(), key=_demo_sort_key):
                if self.allowed_demos is not None and demo not in self.allowed_demos:
                    continue
                frame_count = int(f[f"data/{demo}/obs/{rgb_key}"].shape[0])
                demos.append({"id": demo, "frame_count": frame_count})
        self._demo_list_cache[task_id] = demos
        return demos

    def demo_data(self, task_id: str, demo: str) -> DemoData:
        cache_key = (task_id, demo)
        cached = self._demo_cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.task_path(task_id)
        shared_key = (str(path.resolve()), demo)
        with self._shared_demo_lock:
            shared = self._shared_demo_cache.get(shared_key)
            if shared is None:
                shared = self.semantic_cache.get(path, demo)
                if shared is None:
                    shared = self._load_demo_data_from_hdf5(path, demo)
                    self.semantic_cache.put(path, demo, shared)
                self._shared_demo_cache[shared_key] = shared
            data = shared[self.camera]

        self._demo_cache[cache_key] = data
        return data

    def _load_demo_data_from_hdf5(self, path: Path, demo: str) -> dict[str, DemoData]:
        shared = {}
        with h5py.File(path, "r") as f:
            demo_group = f[f"data/{demo}"]
            obs = demo_group["obs"]
            states = demo_group["states"][:]
            for camera, info in CAMERA_INFO.items():
                source_shape = obs[info["rgb"]].shape
                shared[camera] = DemoData(
                    states=states,
                    frame_count=int(source_shape[0]),
                    source_size=(int(source_shape[2]), int(source_shape[1])),
                    bboxes=_decode_json_dataset(obs[info["bboxes"]]),
                    graphs=_decode_json_dataset(obs[info["scene_graph"]]),
                    world_coords=_decode_json_dataset(obs[f"{info['bboxes'].removesuffix('_bboxes')}_world_coords"]),
                )
        return shared

    def frame_payload(
        self,
        task_id: str,
        demo: str,
        frame: int,
        target_size: tuple[int, int],
        subject: str,
        *,
        mode: str = "gt",
        prediction_triplets: list[tuple[str, str, str]] | None = None,
        prediction_run: str = "",
        prediction_available: bool = True,
    ) -> dict[str, Any]:
        data = self.demo_data(task_id, demo)
        if frame < 0 or frame >= data.frame_count:
            raise IndexError(f"Frame {frame} out of range for {demo}; valid range is 0-{data.frame_count - 1}")

        raw_bboxes = data.bboxes[frame]
        gt_triplets = [
            tuple(item)
            for item in (data.graphs[frame] if frame < len(data.graphs) else [])
            if len(item) == 3
        ]
        gt_set = set(gt_triplets)
        duplicate_bowl_swap_applied = False
        if mode in {"prediction", "compare"}:
            displayed_triplets, duplicate_bowl_swap_applied = duplicate_bowl_invariant_prediction(
                gt_set,
                prediction_triplets or [],
            )
        else:
            displayed_triplets = gt_triplets
        displayed_triplets = displayed_triplets or []
        displayed_set = set(displayed_triplets)
        scaled_bboxes = {
            obj: scale_bbox(
                bbox,
                data.source_size,
                target_size,
                rotate180=self.rotate_current_camera,
            )
            for obj, bbox in raw_bboxes.items()
        }
        correct_triplets = gt_set if mode in {"prediction", "compare"} else None
        edges = relation_edges(displayed_triplets, scaled_bboxes, subject, correct_triplets)
        gt_edges = relation_edges(gt_triplets, scaled_bboxes, subject)
        gt_triplet_items = relation_triplets(gt_triplets, scaled_bboxes, subject)
        metrics = prediction_metrics(gt_set, displayed_set) if mode in {"prediction", "compare"} else None
        return {
            "task": task_id,
            "demo": demo,
            "frame": frame,
            "frame_count": data.frame_count,
            "camera": self.camera,
            "mode": mode,
            "prediction_run": prediction_run,
            "prediction_available": prediction_available,
            "prediction_matching": "duplicate_bowl_invariant" if mode in {"prediction", "compare"} else "ground_truth",
            "duplicate_bowl_swap_applied": duplicate_bowl_swap_applied,
            "source_size": {"width": data.source_size[0], "height": data.source_size[1]},
            "target_size": {"width": target_size[0], "height": target_size[1]},
            "objects": sorted(scaled_bboxes),
            "bboxes": [
                {"object": obj, "bbox": [round(v, 3) for v in bbox]}
                for obj, bbox in sorted(scaled_bboxes.items())
            ],
            "edges": edges,
            "gt_edges": gt_edges,
            "triplets": relation_triplets(displayed_triplets, scaled_bboxes, subject, correct_triplets),
            "gt_triplets": gt_triplet_items,
            "gt_triplet_count": len(gt_triplets),
            "triplet_count": len(displayed_triplets),
            "visible_triplet_count": len(edges),
            "correct_visible_triplet_count": sum(1 for edge in edges if edge.get("correct", True)),
            "metrics": metrics,
        }

    def state_for_frame(self, task_id: str, demo: str, frame: int) -> np.ndarray:
        data = self.demo_data(task_id, demo)
        if frame < 0 or frame >= len(data.states):
            raise IndexError(f"Frame {frame} out of range for {demo}; valid range is 0-{len(data.states) - 1}")
        return data.states[frame]

    def fixed_body_transforms_for_frame(self, task_id: str, demo: str, frame: int) -> dict[str, dict[str, Any]]:
        data = self.demo_data(task_id, demo)
        if frame < 0 or frame >= data.frame_count:
            raise IndexError(f"Frame {frame} out of range for {demo}; valid range is 0-{data.frame_count - 1}")
        return {
            label: value
            for label, value in data.world_coords[frame].items()
            if isinstance(value, dict) and ("pos" in value or "mat" in value)
        }

    def render_inputs_for_range(
        self,
        task_id: str,
        demo: str,
        start: int,
        end: int,
    ) -> list[tuple[int, np.ndarray, dict[str, dict[str, Any]]]]:
        data = self.demo_data(task_id, demo)
        if start < 0 or end >= len(data.states) or start > end:
            raise IndexError(f"Frame range {start}-{end} out of range for {demo}; valid range is 0-{len(data.states) - 1}")
        return [
            (
                frame,
                data.states[frame],
                {
                    label: value
                    for label, value in data.world_coords[frame].items()
                    if isinstance(value, dict) and ("pos" in value or "mat" in value)
                },
            )
            for frame in range(start, end + 1)
        ]


class SimFrameRenderer:
    def __init__(self, hdf5_path: Path | None, max_res: int, rotate_agentview: bool):
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.max_res = max_res
        self.rotate_agentview = rotate_agentview
        self._env = None
        self._fixed_body_ids: dict[str, int | None] = {}

    def render_state(
        self,
        camera: str,
        res: int,
        hdf5_path: Path,
        state: np.ndarray,
        fixed_body_transforms: dict[str, Any] | None = None,
    ) -> Image.Image:
        env = self._get_env(hdf5_path)
        return self._render_after_state_set(env, camera, res, state, fixed_body_transforms)

    def render_states(
        self,
        camera: str,
        res: int,
        hdf5_path: Path,
        states: list[np.ndarray],
        fixed_body_transforms: list[dict[str, Any]] | None = None,
    ) -> list[Image.Image]:
        env = self._get_env(hdf5_path)
        transforms = fixed_body_transforms or [None] * len(states)
        return [
            self._render_after_state_set(env, camera, res, state, frame_transforms)
            for state, frame_transforms in zip(states, transforms)
        ]

    def _render_after_state_set(
        self,
        env,
        camera: str,
        res: int,
        state: np.ndarray,
        fixed_body_transforms: dict[str, Any] | None = None,
    ) -> Image.Image:
        env.sim.set_state_from_flattened(state)
        self._apply_fixed_body_transforms(env, fixed_body_transforms)
        env.sim.forward()
        img_arr = env.sim.render(
            camera_name=CAMERA_INFO[camera]["sim"],
            height=res,
            width=res,
            depth=False,
        )
        if camera == "agentview" and self.rotate_agentview:
            img_arr = np.rot90(img_arr, 2).copy()
        return Image.fromarray(img_arr.astype(np.uint8))

    def _apply_fixed_body_transforms(self, env, transforms: dict[str, Any] | None) -> None:
        if not transforms:
            return
        model = env.sim.model
        for label, transform in transforms.items():
            body_id = self._fixed_body_ids.get(label)
            if label not in self._fixed_body_ids:
                body_id = None
                for candidate in (f"{label}_main", label):
                    try:
                        candidate_id = model.body_name2id(candidate)
                    except Exception:
                        continue
                    if int(model.body_jntnum[candidate_id]) == 0:
                        body_id = int(candidate_id)
                    break
                self._fixed_body_ids[label] = body_id
            if body_id is not None:
                if isinstance(transform, dict):
                    pos = transform.get("pos")
                    mat = transform.get("mat")
                else:
                    pos = transform
                    mat = None
                if pos is not None:
                    model.body_pos[body_id] = np.asarray(pos, dtype=float)
                if mat is not None:
                    model.body_quat[body_id] = mat_to_quat_wxyz(mat)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self.hdf5_path = None
        self._fixed_body_ids.clear()

    def _get_env(self, hdf5_path: Path):
        requested_path = Path(hdf5_path)
        if self._env is not None and self.hdf5_path == requested_path:
            return self._env
        if self._env is not None:
            self.close()
        self.hdf5_path = requested_path
        self._env = self._create_env(requested_path)
        return self._env

    def _create_env(self, hdf5_path: Path):
        if os.name == "nt":
            os.environ.setdefault("MUJOCO_GL", "wgl")
        try:
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Fresh simulator rendering requires LIBERO in the active Python environment. "
                "Install/configure LIBERO, then rerun this demo."
            ) from exc

        bddl_file = resolve_bddl_file(hdf5_path, get_libero_path)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=self.max_res,
            camera_widths=self.max_res,
        )
        env.reset()
        return env


def resolve_bddl_file(hdf5_path: Path, get_libero_path) -> Path:
    task_name = hdf5_path.stem.removesuffix("_demo")
    root = Path(get_libero_path("bddl_files"))
    bddl_file = root / "libero_spatial" / f"{task_name}.bddl"
    if bddl_file.exists():
        return bddl_file
    matches = list(root.rglob(f"{task_name}.bddl"))
    if not matches:
        raise FileNotFoundError(f"Cannot find bddl file for task '{task_name}' under {root}")
    return matches[0]


class RendererManager:
    def __init__(self, camera: str = "agentview", res: int = 512, rotate_agentview: bool = True):
        self.camera = camera
        self.res = res
        self.rotate_agentview = rotate_agentview
        self._renderer = SimFrameRenderer(None, res, rotate_agentview)
        self._jobs: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name="scene-graph-render-worker", daemon=True)
        self._worker.start()

    @property
    def simulator_started(self) -> bool:
        return self._renderer._env is not None

    def render(
        self,
        hdf5_path: Path,
        state: np.ndarray,
        fixed_body_transforms: dict[str, Any] | None = None,
        *,
        camera: str | None = None,
        res: int | None = None,
        priority: int = 0,
    ) -> Image.Image:
        return self._submit(
            priority,
            lambda: self._renderer.render_state(
                camera or self.camera,
                int(res or self.res),
                hdf5_path,
                state,
                fixed_body_transforms,
            ),
        )

    def render_many(
        self,
        hdf5_path: Path,
        states: list[np.ndarray],
        fixed_body_transforms: list[dict[str, Any]] | None = None,
        *,
        camera: str | None = None,
        res: int | None = None,
        priority: int = 10,
    ) -> list[Image.Image]:
        transforms = fixed_body_transforms or [None] * len(states)
        return [
            self.render(
                hdf5_path,
                state,
                frame_transforms,
                camera=camera,
                res=res,
                priority=priority,
            )
            for state, frame_transforms in zip(states, transforms)
        ]

    def _submit(self, priority: int, fn):
        if self._closed:
            raise RuntimeError("RendererManager is closed")
        done = threading.Event()
        result: dict[str, Any] = {}
        self._jobs.put((priority, next(self._seq), (fn, done, result)))
        done.wait()
        if "exc" in result:
            raise result["exc"]
        return result["value"]

    def _worker_loop(self) -> None:
        while True:
            priority, seq, payload = self._jobs.get()
            if payload is None:
                self._jobs.task_done()
                break
            fn, done, result = payload
            try:
                result["value"] = fn()
            except BaseException as exc:
                result["exc"] = exc
            finally:
                done.set()
                self._jobs.task_done()

    def close_all(self) -> None:
        if not self._closed:
            self._closed = True
            self._jobs.put((sys.maxsize, next(self._seq), None))
            self._worker.join(timeout=5)
        self._renderer.close()


class DiskFrameCache:
    def __init__(self, cache_dir: str | Path | None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def image_token(self, hdf5_path: Path, camera: str, res: int, frame: int) -> str:
        resolved = hdf5_path.resolve()
        stat = resolved.stat()
        key = json.dumps(
            {
                "path": str(resolved),
                "mtime_ns": stat.st_mtime_ns,
                "camera": camera,
                "res": res,
                "frame": frame,
                "rotate_agentview": camera == "agentview",
            },
            sort_keys=True,
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def path_for(self, hdf5_path: Path, camera: str, res: int, frame: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / camera / f"{res}px" / f"{self.image_token(hdf5_path, camera, res, frame)}.png"

    def exists(self, hdf5_path: Path, camera: str, res: int, frame: int) -> bool:
        path = self.path_for(hdf5_path, camera, res, frame)
        return bool(path and path.exists())

    def get(self, hdf5_path: Path, camera: str, res: int, frame: int) -> Image.Image | None:
        path = self.path_for(hdf5_path, camera, res, frame)
        if path is None or not path.exists():
            return None
        with Image.open(path) as img:
            return img.convert("RGB")

    def put(self, hdf5_path: Path, camera: str, res: int, frame: int, image: Image.Image) -> None:
        path = self.path_for(hdf5_path, camera, res, frame)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", compress_level=1)


class BoundedOverlayCache(OrderedDict):
    def __init__(self, max_items: int = 4096):
        super().__init__()
        self.max_items = max_items

    def get(self, key, default=None):
        if key not in self:
            return default
        value = super().pop(key)
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value) -> None:
        if key in self:
            super().pop(key)
        super().__setitem__(key, value)
        while len(self) > self.max_items:
            self.popitem(last=False)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: "SceneGraphDemoServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
            elif parsed.path.startswith("/common/"):
                path = safe_child_path(COMMON_STATIC_ROOT, parsed.path.removeprefix("/common/"))
                self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream", cache=False)
            elif parsed.path == "/api/tasks":
                self._send_json(
                    {
                        "tasks": self.server.default_store.list_tasks(),
                        "cameras": self.server.camera_options(),
                        "default_camera": self.server.default_camera,
                    }
                )
            elif parsed.path == "/api/demos":
                query = parse_qs(parsed.query)
                task = _required_query(query, "task")
                camera = self._camera_from_query(query)
                self._send_json({"demos": self.server.store_for(camera).list_demos(task)})
            elif parsed.path == "/api/predictions":
                query = parse_qs(parsed.query)
                task = _required_query(query, "task")
                camera = self._camera_from_query(query)
                self._send_json({"runs": self.server.predictions.list_runs(task, camera)})
            elif parsed.path == "/api/health":
                self._send_json(
                    {
                        "build": DEMO_BUILD,
                        "default_camera": self.server.default_camera,
                        "cameras": self.server.camera_options(),
                        "default_res": self.server.default_res,
                        "res": self.server.default_res,
                        "resolutions": self.server.resolution_options(),
                        "prediction_frame_stride": PREDICTION_FRAME_STRIDE,
                        "presets": self.server.available_presets(),
                        "overlay_cache_size": len(self.server.overlay_cache),
                        "simulator_started": self.server.simulator_started(),
                        "disk_cache": self.server.disk_cache.enabled,
                    }
                )
            elif parsed.path == "/api/frame":
                self._handle_frame(parse_qs(parsed.query))
            elif parsed.path == "/api/preload":
                self._handle_preload(parse_qs(parsed.query))
            elif parsed.path == "/api/image":
                self._handle_image(parse_qs(parsed.query))
            elif parsed.path == "/favicon.ico":
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _handle_frame(self, query: dict[str, list[str]]) -> None:
        camera = self._camera_from_query(query)
        res = self._resolution_from_query(query)
        task = _required_query(query, "task")
        demo = _required_query(query, "demo")
        frame = int(query.get("frame", ["0"])[0])
        subject = query.get("subject", [self.server.default_subject])[0] or self.server.default_subject
        mode = self._mode_from_query(query)
        prediction = query.get("prediction", [""])[0]
        self._send_json(self._frame_response(camera, res, task, demo, frame, subject, mode, prediction))

    def _frame_response(
        self,
        camera: str,
        res: int,
        task: str,
        demo: str,
        frame: int,
        subject: str,
        mode: str,
        prediction: str,
    ) -> dict[str, Any]:
        store = self.server.store_for(camera)
        image_url, image_cached = self.server.ensure_image_url(camera, res, task, demo, frame)
        cache_key = (camera, res, task, demo, frame, subject, mode, prediction)
        cached_overlay = self.server.overlay_cache.get(cache_key)
        if cached_overlay is not None:
            return {"image_url": image_url, "image_cached": image_cached, "overlay": cached_overlay}
        prediction_available = True
        prediction_triplets = (
            self.server.predictions.triplets_for_frame(prediction, demo, camera, frame)
            if mode in {"prediction", "compare"} and prediction
            else None
        )
        if mode in {"prediction", "compare"}:
            prediction_available = bool(prediction) and self.server.predictions.has_frame(prediction, demo, camera, frame)
        overlay = store.frame_payload(
            task,
            demo,
            frame,
            (res, res),
            subject,
            mode=mode,
            prediction_triplets=prediction_triplets,
            prediction_run=prediction if mode in {"prediction", "compare"} else "",
            prediction_available=prediction_available,
        )
        self.server.overlay_cache[cache_key] = overlay
        return {"image_url": image_url, "image_cached": image_cached, "overlay": overlay}

    def _handle_preload(self, query: dict[str, list[str]]) -> None:
        camera = self._camera_from_query(query)
        res = self._resolution_from_query(query)
        task = _required_query(query, "task")
        demo = _required_query(query, "demo")
        start = int(query.get("start", ["0"])[0])
        end = int(query.get("end", [str(start)])[0])
        step = max(1, int(query.get("step", ["1"])[0]))
        subject = query.get("subject", [self.server.default_subject])[0] or self.server.default_subject
        mode = self._mode_from_query(query)
        prediction = query.get("prediction", [""])[0]
        store = self.server.store_for(camera)
        frames_by_index: dict[int, dict[str, Any]] = {}
        requested_frames = list(range(start, end + 1, step))
        image_urls, cached_images, rendered_images = self.server.ensure_image_urls(
            camera,
            res,
            task,
            demo,
            requested_frames,
        )
        for frame in requested_frames:
            cache_key = (camera, res, task, demo, frame, subject, mode, prediction)
            cached_overlay = self.server.overlay_cache.get(cache_key)
            if cached_overlay is None:
                prediction_available = True
                prediction_triplets = (
                    self.server.predictions.triplets_for_frame(prediction, demo, camera, frame)
                    if mode in {"prediction", "compare"} and prediction
                    else None
                )
                if mode in {"prediction", "compare"}:
                    prediction_available = bool(prediction) and self.server.predictions.has_frame(
                        prediction, demo, camera, frame
                    )
                cached_overlay = store.frame_payload(
                    task,
                    demo,
                    frame,
                    (res, res),
                    subject,
                    mode=mode,
                    prediction_triplets=prediction_triplets,
                    prediction_run=prediction if mode in {"prediction", "compare"} else "",
                    prediction_available=prediction_available,
                )
                self.server.overlay_cache[cache_key] = cached_overlay
            frames_by_index[frame] = {"image_url": image_urls[frame], "overlay": cached_overlay}

        frames = [frames_by_index[frame] for frame in requested_frames]
        self._send_json({"frames": frames, "cached": cached_images, "rendered": rendered_images})

    def _handle_image(self, query: dict[str, list[str]]) -> None:
        camera = self._camera_from_query(query)
        res = self._resolution_from_query(query)
        task = _required_query(query, "task")
        demo = _required_query(query, "demo")
        frame = int(query.get("frame", ["0"])[0])
        image_bytes = self.server._ensure_image_bytes(camera, res, task, demo, frame)
        send_bytes(self, image_bytes, "image/png", cache=True)

    def _camera_from_query(self, query: dict[str, list[str]]) -> str:
        camera = query.get("camera", [self.server.default_camera])[0] or self.server.default_camera
        if camera not in self.server.stores:
            raise ValueError(f"Unknown camera '{camera}'. Expected one of {sorted(self.server.stores)}")
        return camera

    def _resolution_from_query(self, query: dict[str, list[str]]) -> int:
        value = int(query.get("res", [str(self.server.default_res)])[0] or self.server.default_res)
        if value not in self.server.resolutions:
            raise ValueError(f"Unknown resolution '{value}'. Expected one of {self.server.resolutions}")
        return value

    def _mode_from_query(self, query: dict[str, list[str]]) -> str:
        mode = query.get("mode", ["gt"])[0] or "gt"
        if mode not in {"gt", "prediction", "compare"}:
            raise ValueError("Unknown overlay mode. Expected 'gt', 'prediction', or 'compare'")
        return mode

    def _send_html(self, html: str) -> None:
        send_html(self, html)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        send_json(self, payload, status)

    def _send_file(self, path: Path, content_type: str, *, cache: bool) -> None:
        send_file(self, path, content_type, cache=cache)


class SceneGraphDemoServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        store: Hdf5SceneGraphStore | dict[str, Hdf5SceneGraphStore],
        renderers: RendererManager | dict[str | tuple[str, int], RendererManager],
        default_subject: str,
        default_camera: str | None = None,
        default_res: int | None = None,
        resolutions: list[int] | None = None,
        predictions: PredictionStore | None = None,
        cache_dir: str | Path | None = ".cache/scene_graph_demo",
    ):
        super().__init__(server_address, DemoRequestHandler)
        self.stores = store if isinstance(store, dict) else {store.camera: store}
        self.default_res = int(default_res or _infer_renderer_res(renderers) or 1024)
        self.resolutions = sorted(set(resolutions or [self.default_res]))
        if self.default_res not in self.resolutions:
            self.resolutions.append(self.default_res)
            self.resolutions.sort()
        self.renderers_by_key: dict[tuple[str, int], Any] = {}
        if isinstance(renderers, dict):
            for key, value in renderers.items():
                if isinstance(key, tuple):
                    camera, res = key
                    self.renderers_by_key[(camera, int(res))] = value
                else:
                    renderer_res = int(getattr(value, "res", 0) or 0)
                    self.renderers_by_key[(key, renderer_res or self.default_res)] = value
        else:
            camera = next(iter(self.stores))
            renderer_res = int(getattr(renderers, "res", 0) or 0)
            self.renderers_by_key[(camera, renderer_res or self.default_res)] = renderers
        self.default_camera = default_camera or next(iter(self.stores))
        if self.default_camera not in self.stores:
            raise ValueError(f"Default camera '{self.default_camera}' is not configured")
        self.default_subject = default_subject
        self.predictions = predictions or PredictionStore("output")
        self.overlay_cache = BoundedOverlayCache()
        self._memory_image_cache: OrderedDict[tuple[str, int, str, str, int], bytes] = OrderedDict()
        self._memory_image_cache_max = 256
        self.disk_cache = DiskFrameCache(cache_dir)

    @property
    def default_store(self) -> Hdf5SceneGraphStore:
        return self.store_for(self.default_camera)

    @property
    def default_renderers(self) -> Any:
        return self.renderers_for(self.default_camera, self.default_res)

    def store_for(self, camera: str) -> Hdf5SceneGraphStore:
        try:
            return self.stores[camera]
        except KeyError as exc:
            raise KeyError(f"Unknown camera '{camera}'") from exc

    def renderers_for(self, camera: str, res: int | None = None) -> Any:
        target_res = int(res or self.default_res)
        key = (camera, target_res)
        if key in self.renderers_by_key:
            return self.renderers_by_key[key]
        camera_matches = [
            renderer
            for (renderer_camera, _), renderer in self.renderers_by_key.items()
            if renderer_camera == camera
        ]
        if len(camera_matches) == 1 and not hasattr(camera_matches[0], "res"):
            return camera_matches[0]
        raise KeyError(f"Unknown camera/resolution '{camera}/{target_res}'")

    def render_image(
        self,
        camera: str,
        res: int,
        hdf5_path: Path,
        frame: int,
        state: np.ndarray,
        fixed_body_transforms: dict[str, Any] | None = None,
        *,
        priority: int = 0,
    ) -> Image.Image:
        cached = self.disk_cache.get(hdf5_path, camera, res, frame)
        if cached is not None:
            return cached
        renderer = self.renderers_for(camera, res)
        try:
            image = renderer.render(hdf5_path, state, fixed_body_transforms, camera=camera, res=res, priority=priority)
        except TypeError:
            image = renderer.render(hdf5_path, state, fixed_body_transforms)
        self.disk_cache.put(hdf5_path, camera, res, frame, image)
        return image

    def ensure_image_url(
        self,
        camera: str,
        res: int,
        task: str,
        demo: str,
        frame: int,
    ) -> tuple[str, bool]:
        store = self.store_for(camera)
        hdf5_path = store.task_path(task)
        was_cached = self.disk_cache.exists(hdf5_path, camera, res, frame) or (
            camera,
            res,
            task,
            demo,
            frame,
        ) in self._memory_image_cache
        self._ensure_image_rendered(camera, res, task, demo, frame)
        token = self.disk_cache.image_token(hdf5_path, camera, res, frame)
        params = urlencode(
            {"camera": camera, "res": res, "task": task, "demo": demo, "frame": frame, "token": token}
        )
        return f"/api/image?{params}", was_cached

    def ensure_image_urls(
        self,
        camera: str,
        res: int,
        task: str,
        demo: str,
        frames: list[int],
    ) -> tuple[dict[int, str], int, int]:
        store = self.store_for(camera)
        hdf5_path = store.task_path(task)
        urls: dict[int, str] = {}
        missing: list[int] = []
        cached = 0
        for frame in frames:
            path = self.disk_cache.path_for(hdf5_path, camera, res, frame)
            memory_key = (camera, res, task, demo, frame)
            if (path is not None and path.exists()) or memory_key in self._memory_image_cache:
                cached += 1
            else:
                missing.append(frame)
            token = self.disk_cache.image_token(hdf5_path, camera, res, frame)
            params = urlencode(
                {"camera": camera, "res": res, "task": task, "demo": demo, "frame": frame, "token": token}
            )
            urls[frame] = f"/api/image?{params}"

        if missing:
            render_inputs = [
                (
                    frame,
                    store.state_for_frame(task, demo, frame),
                    store.fixed_body_transforms_for_frame(task, demo, frame),
                )
                for frame in missing
            ]
            rendered = self.render_images(camera, res, hdf5_path, render_inputs, priority=10)
            if not self.disk_cache.enabled:
                for frame, image in zip(missing, rendered):
                    buf = BytesIO()
                    image.save(buf, format="PNG", compress_level=1)
                    self._memory_image_cache[(camera, res, task, demo, frame)] = buf.getvalue()
                while len(self._memory_image_cache) > self._memory_image_cache_max:
                    self._memory_image_cache.popitem(last=False)
        return urls, cached, len(missing)

    def _ensure_image_rendered(self, camera: str, res: int, task: str, demo: str, frame: int) -> None:
        store = self.store_for(camera)
        hdf5_path = store.task_path(task)
        path = self.disk_cache.path_for(hdf5_path, camera, res, frame)
        memory_key = (camera, res, task, demo, frame)
        if (path is not None and path.exists()) or memory_key in self._memory_image_cache:
            return
        image = self.render_image(
            camera,
            res,
            hdf5_path,
            frame,
            store.state_for_frame(task, demo, frame),
            store.fixed_body_transforms_for_frame(task, demo, frame),
            priority=0,
        )
        if path is not None:
            return
        buf = BytesIO()
        image.save(buf, format="PNG", compress_level=1)
        self._memory_image_cache[memory_key] = buf.getvalue()
        while len(self._memory_image_cache) > self._memory_image_cache_max:
            self._memory_image_cache.popitem(last=False)

    def _ensure_image_bytes(self, camera: str, res: int, task: str, demo: str, frame: int) -> bytes:
        store = self.store_for(camera)
        hdf5_path = store.task_path(task)
        path = self.disk_cache.path_for(hdf5_path, camera, res, frame)
        if path is not None and path.exists():
            return path.read_bytes()

        memory_key = (camera, res, task, demo, frame)
        cached = self._memory_image_cache.get(memory_key)
        if cached is not None:
            self._memory_image_cache.move_to_end(memory_key)
            return cached

        image = self.render_image(
            camera,
            res,
            hdf5_path,
            frame,
            store.state_for_frame(task, demo, frame),
            store.fixed_body_transforms_for_frame(task, demo, frame),
            priority=0,
        )
        if path is not None and path.exists():
            return path.read_bytes()
        buf = BytesIO()
        image.save(buf, format="PNG", compress_level=1)
        data = buf.getvalue()
        self._memory_image_cache[memory_key] = data
        while len(self._memory_image_cache) > self._memory_image_cache_max:
            self._memory_image_cache.popitem(last=False)
        return data

    def render_images(
        self,
        camera: str,
        res: int,
        hdf5_path: Path,
        render_inputs: list[tuple[int, np.ndarray, dict[str, Any]]],
        *,
        priority: int = 10,
    ) -> list[Image.Image]:
        images_by_frame: dict[int, Image.Image] = {}
        missing: list[tuple[int, np.ndarray, dict[str, Any]]] = []
        for frame, state, transforms in render_inputs:
            cached = self.disk_cache.get(hdf5_path, camera, res, frame)
            if cached is not None:
                images_by_frame[frame] = cached
            else:
                missing.append((frame, state, transforms))

        if missing:
            renderer = self.renderers_for(camera, res)
            try:
                rendered = renderer.render_many(
                    hdf5_path,
                    [state for _, state, _ in missing],
                    [transforms for _, _, transforms in missing],
                    camera=camera,
                    res=res,
                    priority=priority,
                )
            except TypeError:
                rendered = renderer.render_many(
                    hdf5_path,
                    [state for _, state, _ in missing],
                    [transforms for _, _, transforms in missing],
                )
            for (frame, _, _), image in zip(missing, rendered):
                self.disk_cache.put(hdf5_path, camera, res, frame, image)
                images_by_frame[frame] = image

        return [images_by_frame[frame] for frame, _, _ in render_inputs]

    def camera_options(self) -> list[dict[str, str]]:
        return [
            {"id": camera, "name": camera.replace("_", " ")}
            for camera in sorted(self.stores)
        ]

    def resolution_options(self) -> list[dict[str, str]]:
        return [
            {"id": str(res), "name": f"{res} x {res}"}
            for res in self.resolutions
        ]

    def available_presets(self) -> list[dict[str, Any]]:
        tasks = {task["id"] for task in self.default_store.list_tasks()}
        return [preset for preset in ICLR_PRESETS if preset["task"] in tasks]

    def simulator_started(self) -> bool:
        return any(bool(getattr(renderers, "simulator_started", False)) for renderers in set(self.renderers_by_key.values()))

    def close_all_renderers(self) -> None:
        seen = set()
        for renderers in self.renderers_by_key.values():
            if id(renderers) in seen:
                continue
            seen.add(id(renderers))
            if hasattr(renderers, "close_all"):
                renderers.close_all()


def _infer_renderer_res(renderers: Any) -> int | None:
    if isinstance(renderers, dict):
        for key, value in renderers.items():
            if isinstance(key, tuple):
                return int(key[1])
            renderer_res = getattr(value, "res", None)
            if renderer_res:
                return int(renderer_res)
        return None
    renderer_res = getattr(renderers, "res", None)
    return int(renderer_res) if renderer_res else None


def _required_query(query: dict[str, list[str]], key: str) -> str:
    value = query.get(key, [""])[0]
    if not value:
        raise ValueError(f"Missing required query parameter '{key}'")
    return value


def _demo_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("demo_"):
        try:
            return int(value.removeprefix("demo_")), value
        except ValueError:
            pass
    return sys.maxsize, value


LIBERO_STATIC_ROOT = Path(__file__).resolve().parent / "libero"
INDEX_HTML = (LIBERO_STATIC_ROOT / "index.html").read_text(encoding="utf-8")


def serve(args: argparse.Namespace) -> None:
    rotate_agentview = not args.no_rotate_agentview
    semantic_cache_dir = None if args.no_disk_cache else Path(args.cache_dir) / "hdf5_semantics"
    stores = {
        camera: Hdf5SceneGraphStore(
            args.input_dir,
            camera,
            rotate_agentview=rotate_agentview,
            semantic_cache_dir=semantic_cache_dir,
        )
        for camera in CAMERA_INFO
    }
    resolutions = sorted(set([*RESOLUTION_OPTIONS, args.res]))
    shared_renderer = RendererManager(args.camera, max(resolutions), rotate_agentview=rotate_agentview)
    renderers = {
        (camera, res): shared_renderer
        for camera in CAMERA_INFO
        for res in resolutions
    }
    server = SceneGraphDemoServer(
        (args.host, args.port),
        stores,
        renderers,
        args.subject,
        default_camera=args.camera,
        default_res=args.res,
        resolutions=resolutions,
        predictions=PredictionStore(args.output_dir),
        cache_dir=None if args.no_disk_cache else args.cache_dir,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Scene graph scrubber: {url}")
    print("Fresh rendering uses LIBERO OffScreenRenderEnv. If rendering fails, fix the simulator env and reload.")
    if not args.no_disk_cache:
        print(f"Persistent rendered-frame cache: {args.cache_dir}")
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.close_all_renderers()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local browser scrubber for GT scene-graph overlays.")
    parser.add_argument("--input-dir", default="data/libero_spatial_v5")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--camera", default="agentview", choices=sorted(CAMERA_INFO))
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--cache-dir", default=".cache/scene_graph_demo")
    parser.add_argument("--no-disk-cache", action="store_true", help="Disable persistent PNG cache for rendered frames.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--no-rotate-agentview",
        action="store_true",
        help="Disable the 180-degree agentview display convention used by plot_frame.py --hires.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    serve(parser.parse_args(argv))
