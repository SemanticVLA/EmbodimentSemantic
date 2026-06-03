"""
Triplet-level evaluation: precision, recall, F1 of predicted scene graphs vs GT.

GT source  : HDF5 obs/{camera}_scene_graph (JSON-encoded list-of-lists per frame)
Pred source: JSONL logs produced by runner.py (raw model responses are re-parsed)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .io_utils import parse_triplets, list_hdf5_files
from .prompts import BIDIRECTIONAL_RELATIONS, INVERSE_MAP


Triplet = tuple[str, str, str]
_DUPLICATE_BOWL_OBJECTS = ("akita_black_bowl_1", "akita_black_bowl_2")
_DUPLICATE_BOWL_SWAP = {
    _DUPLICATE_BOWL_OBJECTS[0]: _DUPLICATE_BOWL_OBJECTS[1],
    _DUPLICATE_BOWL_OBJECTS[1]: _DUPLICATE_BOWL_OBJECTS[0],
}

_SCENE_GRAPH_KEY = {
    "agentview": "agentview_scene_graph",
    "eye_in_hand": "robot0_eye_in_hand_scene_graph",
}
_GT_INDEX_CACHE: dict[tuple, dict[tuple[str, str, int], set[Triplet]]] = {}
_DEMO_KEYS_CACHE: dict[str, list[str]] = {}


def _frame_f1(gt: set[Triplet], pred: set[Triplet]) -> float:
    if not gt and not pred:
        return 1.0
    tp = len(gt & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _swap_duplicate_bowls(triplets: set[Triplet]) -> set[Triplet]:
    return {
        (_DUPLICATE_BOWL_SWAP.get(a, a), rel, _DUPLICATE_BOWL_SWAP.get(b, b))
        for a, rel, b in triplets
    }


def _duplicate_bowl_invariant_pred(gt: set[Triplet], pred: set[Triplet]) -> set[Triplet]:
    """Choose the better frame-level assignment for visually identical black bowls."""
    swapped = _swap_duplicate_bowls(pred)
    if swapped == pred:
        return pred
    if _frame_f1(gt, swapped) > _frame_f1(gt, pred):
        return swapped
    return pred


@dataclass
class FrameResult:
    task: str
    demo: str
    frame: int
    camera: str
    gt: set[Triplet]
    pred: set[Triplet]

    def __post_init__(self) -> None:
        self.pred = _duplicate_bowl_invariant_pred(self.gt, self.pred)

    @property
    def tp(self) -> set[Triplet]:
        return self.gt & self.pred

    @property
    def fp(self) -> set[Triplet]:
        return self.pred - self.gt

    @property
    def fn(self) -> set[Triplet]:
        return self.gt - self.pred

    @property
    def precision(self) -> float:
        if not self.gt and not self.pred:
            return 1.0
        return len(self.tp) / len(self.pred) if self.pred else 0.0

    @property
    def recall(self) -> float:
        if not self.gt and not self.pred:
            return 1.0
        return len(self.tp) / len(self.gt) if self.gt else 0.0

    @property
    def f1(self) -> float:
        return _frame_f1(self.gt, self.pred)

    @property
    def reversals(self) -> set[Triplet]:
        """GT triplets that were missed but predicted with reversed semantics."""
        revs = set()
        for a, rel, b in self.fn:
            inv_rel = INVERSE_MAP.get(rel)
            if (b, rel, a) in self.pred or (inv_rel and (a, inv_rel, b) in self.pred):
                revs.add((a, rel, b))
        return revs


@dataclass
class AggregateMetrics:
    precision: float
    recall: float
    f1: float
    n_frames: int
    n_gt_triplets: int
    n_pred_triplets: int
    n_tp: int
    n_fp: int
    n_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_relation: dict[str, dict]
    coverage: float
    hallucination_rate: float
    n_reversals: int
    reversal_rate: float
    direction_consistency: float
    per_object_recall: dict[str, float]
    per_task_metrics: dict[str, dict]
    mean_per_task_f1: float


@dataclass
class PredictionCountStats:
    records_total: int = 0
    records_selected: int = 0
    scored_frames: int = 0
    parsed_triplets: int = 0
    scored_triplets: int = 0
    duplicate_records: int = 0
    dedup_triplets_dropped: int = 0
    empty_records: int = 0
    bad_json_lines: int = 0


def _load_all_gt(
    hdf5_path: Path,
    demos: list[str],
    cameras: list[str],
    frame_indices: list[int] | None = None,
) -> dict[tuple[str, str, int], set[Triplet]]:
    """Open HDF5 once, returning {(demo, camera, frame_idx): triplets}."""
    cache_key = (
        str(hdf5_path.resolve()),
        tuple(demos),
        tuple(cameras),
        tuple(sorted(frame_indices)) if frame_indices is not None else None,
    )
    if cache_key in _GT_INDEX_CACHE:
        return _GT_INDEX_CACHE[cache_key]

    index: dict[tuple[str, str, int], set[Triplet]] = {}
    frame_index_set = set(frame_indices) if frame_indices is not None else None
    with h5py.File(hdf5_path, "r") as f:
        for demo in demos:
            for camera in cameras:
                key = _SCENE_GRAPH_KEY.get(camera)
                if key is None:
                    continue
                dataset_path = f"data/{demo}/obs/{key}"
                if dataset_path not in f:
                    continue
                frames = json.loads(f[dataset_path][()].decode("utf-8"))
                if frame_index_set is None:
                    selected_indices = range(len(frames))
                else:
                    selected_indices = (idx for idx in frame_index_set if idx < len(frames))
                for frame_idx in selected_indices:
                    triplets = frames[frame_idx]
                    index[(demo, camera, frame_idx)] = {
                        tuple(t) for t in triplets if len(t) == 3
                    }
    _GT_INDEX_CACHE[cache_key] = index
    return index


def _load_pred_jsonl(
    jsonl_path: Path,
    cameras: list[str] | None = None,
    frame_indices: list[int] | None = None,
) -> dict[tuple[str, str, int], set[Triplet]]:
    """Re-parse JSONL responses into {(demo, camera, frame): triplets}."""
    index: dict[tuple[str, str, int], set[Triplet]] = {}
    frame_index_set = set(frame_indices) if frame_indices is not None else None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            camera = rec["camera"]
            frame = int(rec["frame"])
            if cameras is not None and camera not in cameras:
                continue
            if frame_index_set is not None and frame not in frame_index_set:
                continue
            key = (rec["demo"], camera, frame)
            triplets = {
                (a, rel, b)
                for a, rel, b in parse_triplets(rec.get("response", ""))
                if rel in BIDIRECTIONAL_RELATIONS
            }
            index.setdefault(key, set()).update(triplets)
    return index


def jsonl_prediction_counts(
    jsonl_path: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
) -> PredictionCountStats:
    """Count JSONL records and parsed predictions using evaluation filters."""
    frame_index_set = set(frame_indices)
    pred_index: dict[tuple[str, str, int], set[Triplet]] = {}
    key_counts: dict[tuple[str, str, int], int] = {}
    stats = PredictionCountStats()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.bad_json_lines += 1
                continue

            stats.records_total += 1
            camera = rec["camera"]
            frame = int(rec["frame"])
            if (cameras is not None and camera not in cameras) or frame not in frame_index_set:
                continue

            stats.records_selected += 1
            key = (rec["demo"], camera, frame)
            key_counts[key] = key_counts.get(key, 0) + 1
            triplets = {
                (a, rel, b)
                for a, rel, b in parse_triplets(rec.get("response", ""))
                if rel in BIDIRECTIONAL_RELATIONS
            }
            if not triplets:
                stats.empty_records += 1
            stats.parsed_triplets += len(triplets)
            pred_index.setdefault(key, set()).update(triplets)

    stats.scored_frames = len(pred_index)
    stats.scored_triplets = sum(len(v) for v in pred_index.values())
    stats.duplicate_records = sum(n - 1 for n in key_counts.values() if n > 1)
    stats.dedup_triplets_dropped = stats.parsed_triplets - stats.scored_triplets
    return stats


def _find_hdf5_for_artifact(path: Path, input_dir: Path) -> tuple[str | None, Path | None]:
    for hdf5_path in list_hdf5_files(str(input_dir)):
        stem = hdf5_path.stem.replace("_demo", "")
        if stem in path.stem:
            return stem, hdf5_path
    return None, None


def _camera_from_artifact_name(path: Path) -> str | None:
    stem = path.stem
    if "_eye_in_hand_" in stem or stem.endswith("_eye_in_hand"):
        return "eye_in_hand"
    if "_agentview_" in stem or stem.endswith("_agentview"):
        return "agentview"
    return None


def _demo_keys(hdf5_path: Path) -> list[str]:
    cache_key = str(hdf5_path.resolve())
    if cache_key in _DEMO_KEYS_CACHE:
        return _DEMO_KEYS_CACHE[cache_key]
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"] if "data" in f else f
        demos = sorted(data_group.keys())
    _DEMO_KEYS_CACHE[cache_key] = demos
    return demos


def evaluate_jsonl(
    jsonl_path: Path,
    input_dir: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
) -> Iterator[FrameResult]:
    """Score a JSONL artifact against every expected HDF5/config frame."""
    jsonl_path = Path(jsonl_path)
    input_dir = Path(input_dir)
    pred_index = _load_pred_jsonl(jsonl_path, cameras, frame_indices)

    task_name, hdf5_path = _find_hdf5_for_artifact(jsonl_path, input_dir)
    if hdf5_path is None:
        print(f"  [skip] {jsonl_path.name} - no matching HDF5 found in {input_dir}")
        return

    if cameras is not None:
        cam_list = sorted(cameras)
    else:
        inferred_camera = _camera_from_artifact_name(jsonl_path)
        if inferred_camera is not None:
            cam_list = [inferred_camera]
        else:
            cam_list = sorted({cam for _, cam, _ in pred_index}) or sorted(_SCENE_GRAPH_KEY)

    demos = _demo_keys(hdf5_path)
    gt_index = _load_all_gt(hdf5_path, demos, cam_list, frame_indices)

    for demo, camera, frame in sorted(gt_index):
        gt = {
            t for t in gt_index[(demo, camera, frame)]
            if t[1] in BIDIRECTIONAL_RELATIONS
        }
        pred = {
            t for t in pred_index.get((demo, camera, frame), set())
            if t[1] in BIDIRECTIONAL_RELATIONS
        }
        yield FrameResult(
            task=task_name or "",
            demo=demo,
            frame=frame,
            camera=camera,
            gt=gt,
            pred=pred,
        )


def _per_relation_metrics(results: list[FrameResult]) -> dict[str, dict]:
    out = {}
    for rel in sorted(BIDIRECTIONAL_RELATIONS):
        tp = sum(sum(1 for t in r.tp if t[1] == rel) for r in results)
        fp = sum(sum(1 for t in r.fp if t[1] == rel) for r in results)
        fn = sum(sum(1 for t in r.fn if t[1] == rel) for r in results)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        out[rel] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r_, 4),
            "f1": round(f, 4),
        }
    return out


def _coverage(results: list[FrameResult]) -> float:
    total, covered = 0, 0
    for r in results:
        pred_relations = {t[1] for t in r.pred}
        for t in r.gt:
            total += 1
            if t[1] in pred_relations:
                covered += 1
    return covered / total if total else 0.0


def _direction_consistency(results: list[FrameResult]) -> float:
    total, consistent = 0, 0
    for r in results:
        for a, rel, b in r.gt:
            inv_rel = INVERSE_MAP.get(rel)
            if inv_rel and (b, inv_rel, a) in r.gt:
                total += 1
                if (a, rel, b) in r.pred and (b, inv_rel, a) in r.pred:
                    consistent += 1
    return consistent / total if total else math.nan


def _per_object_recall(results: list[FrameResult]) -> dict[str, float]:
    obj_gt: dict[str, int] = {}
    obj_tp: dict[str, int] = {}
    for r in results:
        for a, _, b in r.gt:
            for obj in (a, b):
                obj_gt[obj] = obj_gt.get(obj, 0) + 1
                obj_tp.setdefault(obj, 0)
        for a, _, b in r.tp:
            for obj in (a, b):
                obj_tp[obj] = obj_tp.get(obj, 0) + 1
    return {
        obj: round(obj_tp.get(obj, 0) / count, 4)
        for obj, count in sorted(obj_gt.items())
    }


def aggregate(results: list[FrameResult]) -> AggregateMetrics:
    """Macro-average P/R/F1 across frames plus micro and diagnostic metrics."""
    if not results:
        empty_rel = {
            r: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            for r in sorted(BIDIRECTIONAL_RELATIONS)
        }
        return AggregateMetrics(
            precision=0,
            recall=0,
            f1=0,
            n_frames=0,
            n_gt_triplets=0,
            n_pred_triplets=0,
            n_tp=0,
            n_fp=0,
            n_fn=0,
            micro_precision=0,
            micro_recall=0,
            micro_f1=0,
            per_relation=empty_rel,
            coverage=0.0,
            hallucination_rate=0.0,
            n_reversals=0,
            reversal_rate=0.0,
            direction_consistency=math.nan,
            per_object_recall={},
            per_task_metrics={},
            mean_per_task_f1=0.0,
        )

    macro_p = float(np.mean([r.precision for r in results]))
    macro_r = float(np.mean([r.recall for r in results]))
    macro_f1 = float(np.mean([r.f1 for r in results]))
    n_tp = sum(len(r.tp) for r in results)
    n_fp = sum(len(r.fp) for r in results)
    n_fn = sum(len(r.fn) for r in results)
    n_rev = sum(len(r.reversals) for r in results)

    micro_p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    micro_r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    task_groups: dict[str, list[FrameResult]] = {}
    for r in results:
        task_groups.setdefault(r.task, []).append(r)

    per_task_metrics = {}
    for task, t_results in task_groups.items():
        t_macro_f1 = float(np.mean([r.f1 for r in t_results]))
        t_tp = sum(len(r.tp) for r in t_results)
        t_fp = sum(len(r.fp) for r in t_results)
        t_fn = sum(len(r.fn) for r in t_results)
        t_rev = sum(len(r.reversals) for r in t_results)
        t_micro_p = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
        t_micro_r = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
        t_micro_f = (
            2 * t_micro_p * t_micro_r / (t_micro_p + t_micro_r)
            if (t_micro_p + t_micro_r)
            else 0.0
        )

        t_demo_f1s: dict[str, list[float]] = {}
        for r in t_results:
            t_demo_f1s.setdefault(r.demo, []).append(r.f1)
        mean_demo_f1 = (
            float(np.mean([float(np.mean(f1s)) for f1s in t_demo_f1s.values()]))
            if t_demo_f1s
            else 0.0
        )

        t_dc = _direction_consistency(t_results)
        per_task_metrics[task] = {
            "demo_f1": round(mean_demo_f1, 4),
            "macro_f1": round(t_macro_f1, 4),
            "micro_f1": round(t_micro_f, 4),
            "coverage": round(_coverage(t_results), 4),
            "hallucination": round(t_fp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0, 4),
            "n_reversals": t_rev,
            "reversal_rate": round(t_rev / t_fn if t_fn else 0.0, 4),
            "dir_consistency": round(t_dc, 4) if not math.isnan(t_dc) else math.nan,
        }

    mean_per_task_f1 = (
        float(np.mean([m["demo_f1"] for m in per_task_metrics.values()]))
        if per_task_metrics
        else 0.0
    )
    dc = _direction_consistency(results)

    return AggregateMetrics(
        precision=macro_p,
        recall=macro_r,
        f1=macro_f1,
        n_frames=len(results),
        n_gt_triplets=sum(len(r.gt) for r in results),
        n_pred_triplets=sum(len(r.pred) for r in results),
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        micro_precision=round(micro_p, 4),
        micro_recall=round(micro_r, 4),
        micro_f1=round(micro_f, 4),
        per_relation=_per_relation_metrics(results),
        coverage=round(_coverage(results), 4),
        hallucination_rate=round(n_fp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0, 4),
        n_reversals=n_rev,
        reversal_rate=round(n_rev / n_fn if n_fn else 0.0, 4),
        direction_consistency=dc,
        per_object_recall=_per_object_recall(results),
        per_task_metrics=per_task_metrics,
        mean_per_task_f1=round(mean_per_task_f1, 4),
    )


def print_frame_report(r: FrameResult, verbose: bool = False) -> None:
    print(f"\n{'=' * 64}")
    print(f"  task   : {r.task}")
    print(f"  demo   : {r.demo}  |  frame : {r.frame}  |  camera : {r.camera}")
    print(f"  GT     : {len(r.gt)}  |  pred : {len(r.pred)}")
    print(f"  TP={len(r.tp)}  FP={len(r.fp)}  FN={len(r.fn)}")
    print(f"  Precision={r.precision:.3f}  Recall={r.recall:.3f}  F1={r.f1:.3f}")

    if verbose:
        if r.tp:
            print(f"\n  TRUE POSITIVES ({len(r.tp)}):")
            for t in sorted(r.tp):
                print(f"    {t[0]}, {t[1]}, {t[2]}")
        if r.fp:
            print(f"\n  FALSE POSITIVES - model predicted, not in GT ({len(r.fp)}):")
            for t in sorted(r.fp):
                print(f"    {t[0]}, {t[1]}, {t[2]}")
        if r.fn:
            print(f"\n  FALSE NEGATIVES - in GT, model missed ({len(r.fn)}):")
            for t in sorted(r.fn):
                print(f"    {t[0]}, {t[1]}, {t[2]}")


def print_aggregate_report(m: AggregateMetrics, label: str = "") -> None:
    header = f"AGGREGATE - {label}" if label else "AGGREGATE"
    print(f"\n{'=' * 64}")
    print(f"  {header}")
    print(f"  Frames   : {m.n_frames}")
    print(f"  GT total : {m.n_gt_triplets}  |  Pred total : {m.n_pred_triplets}")
    print(f"  TP={m.n_tp}  FP={m.n_fp}  FN={m.n_fn}")
    print(f"  Macro  Precision={m.precision:.3f}  Recall={m.recall:.3f}  F1={m.f1:.3f}")
    print(f"  Micro  Precision={m.micro_precision:.3f}  Recall={m.micro_recall:.3f}  F1={m.micro_f1:.3f}")
    print(
        f"  Coverage={m.coverage:.3f}  Hallucination={m.hallucination_rate:.3f}  "
        f"Reversals={m.n_reversals} ({m.reversal_rate:.3f})"
    )
    if math.isnan(m.direction_consistency):
        print("  Direction Consistency=N/A")
    else:
        print(f"  Direction Consistency={m.direction_consistency:.3f}")

    if m.per_relation:
        print("\n  Per-Relation F1:")
        for rel, rm in m.per_relation.items():
            print(
                f"    {rel:<22}  P={rm['precision']:.3f}  R={rm['recall']:.3f}  "
                f"F1={rm['f1']:.3f}  (TP={rm['tp']} FP={rm['fp']} FN={rm['fn']})"
            )

    if m.per_object_recall:
        print("\n  Per-Object Recall:")
        for obj, rec in m.per_object_recall.items():
            print(f"    {obj:<40}  Recall={rec:.3f}")

    if m.per_task_metrics:
        print("\n  Task-Based Details:")
        print(
            f"    {'Task':<60} | {'mF1':<7} | {'Macro F1':<8} | {'Micro F1':<8} | "
            f"{'Cov':<5} | {'Hal':<5} | {'Rev':<5} | {'DC':<5}"
        )
        print("    " + "-" * 124)

        for task, tm in sorted(m.per_task_metrics.items()):
            t_name = task if len(task) <= 60 else task[:57] + "..."
            dc_str = (
                f"{tm['dir_consistency']:.3f}"
                if not math.isnan(tm["dir_consistency"])
                else "N/A"
            )
            print(
                f"    {t_name:<60} | {tm['demo_f1']:<7.3f} | "
                f"{tm['macro_f1']:<8.3f} | {tm['micro_f1']:<8.3f} | "
                f"{tm['coverage']:<5.3f} | {tm['hallucination']:<5.3f} | "
                f"{tm['n_reversals']:<3} ({tm['reversal_rate']:.3f}) | {dc_str:<5}"
            )

        print(f"\n  > {'Mean Per-Task (Demo) F1':<58}  F1={m.mean_per_task_f1:.3f}")
