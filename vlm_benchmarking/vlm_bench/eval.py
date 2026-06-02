"""
Triplet-level evaluation: precision, recall, F1 of predicted scene graphs vs GT.

GT source  : HDF5 obs/{camera}_scene_graph  (JSON-encoded list-of-lists per frame)
Pred source: CSV produced by runner.py       (task, demo, frame, camera, objectA, relation, objectB)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .prompts import BIDIRECTIONAL_RELATIONS, UNIDIRECTIONAL_RELATIONS, INVERSE_MAP
from .io_utils import canonicalize_relation, parse_triplets, list_hdf5_files


Triplet = tuple[str, str, str]

_SCENE_GRAPH_KEY = {
    "agentview": "agentview_scene_graph",
    "eye_in_hand": "robot0_eye_in_hand_scene_graph",
}


# ─── data classes ────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    task: str
    demo: str
    frame: int
    camera: str
    gt: set[Triplet]
    pred: set[Triplet]

    @property
    def tp(self) -> set[Triplet]: return self.gt & self.pred
    @property
    def fp(self) -> set[Triplet]: return self.pred - self.gt
    @property
    def fn(self) -> set[Triplet]: return self.gt - self.pred

    @property
    def precision(self) -> float:
        return len(self.tp) / len(self.pred) if self.pred else 0.0

    @property
    def recall(self) -> float:
        return len(self.tp) / len(self.gt) if self.gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def reversals(self) -> set[Triplet]:
        """Triplets in GT that were missed (FN) but predicted in reverse."""
        revs = set()
        for (a, rel, b) in self.fn:
            inv_rel = INVERSE_MAP.get(rel)
            if (b, rel, a) in self.pred or (inv_rel and (a, inv_rel, b) in self.pred):
                revs.add((a, rel, b))
        return revs


@dataclass
class AggregateMetrics:
    # macro-averaged across frames
    precision: float
    recall: float
    f1: float
    n_frames: int
    n_gt_triplets: int
    n_pred_triplets: int
    n_tp: int
    n_fp: int
    n_fn: int
    # micro-averaged (pooled TP/FP/FN then compute)
    micro_precision: float
    micro_recall: float
    micro_f1: float
    # per-relation: rel -> {tp, fp, fn, precision, recall, f1}
    per_relation: dict[str, dict]
    # additional metrics
    coverage: float           # fraction of GT triplets whose relation type appears in pred
    hallucination_rate: float # FP / (TP + FP)
    n_reversals: int          # total count of FNs that were predicted in reverse
    reversal_rate: float      # fraction of FNs that were predicted in reverse
    direction_consistency: float  # fraction of inverse pairs both predicted; nan in u-mode
    per_object_recall: dict[str, float]
    per_task_metrics: dict[str, dict]
    mean_per_task_f1: float


# ─── loading helpers ──────────────────────────────────────────────────────────

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


def _load_gt(hdf5_path: Path, demo: str, frame: int, camera: str) -> set[Triplet]:
    key = _SCENE_GRAPH_KEY.get(camera)
    if key is None:
        raise ValueError(f"Unknown camera '{camera}'. Expected: {list(_SCENE_GRAPH_KEY)}")
    with h5py.File(hdf5_path, "r") as f:
        raw = f[f"data/{demo}/obs/{key}"][()]
    frames = json.loads(raw.decode("utf-8"))
    if frame >= len(frames):
        return set()
    return {tuple(t) for t in frames[frame]}


def _load_all_gt(hdf5_path: Path, demos: list[str], cameras: list[str]) -> dict[tuple, set[Triplet]]:
    """Open HDF5 once, return {(demo, camera, frame_idx): triplets} for all demos/cameras."""
    index: dict[tuple, set[Triplet]] = {}
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
                for frame_idx, triplets in enumerate(frames):
                    index[(demo, camera, frame_idx)] = {tuple(t) for t in triplets}
    return index


def _load_pred_jsonl(
    jsonl_path: Path,
    cameras: list[str] | None = None,
    frame_indices: list[int] | None = None,
    direction: str = "b",
) -> dict[tuple[str, str, int], set[Triplet]]:
    """Re-parse raw model responses from JSONL. Returns {(demo, camera, frame): set of triplets}."""
    index: dict[tuple[str, str, int], set[Triplet]] = {}
    frame_index_set = set(frame_indices) if frame_indices is not None else None
    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS
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
                for a, rel, b in parse_triplets(rec["response"])
                if rel in relations
            }
            index.setdefault(key, set()).update(triplets)
    return index


def _load_pred_csv(csv_path: Path) -> dict[tuple[str, str, int], set[Triplet]]:
    """Returns {(demo, camera, frame): set of triplets}."""
    index: dict[tuple[str, str, int], set[Triplet]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = canonicalize_relation(row["relation"])
            if rel is None:
                continue
            key = (row["demo"], row["camera"], int(row["frame"]))
            index.setdefault(key, set()).add(
                (row["objectA"], rel, row["objectB"])
            )
    return index


def jsonl_prediction_counts(
    jsonl_path: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
    direction: str = "b",
) -> PredictionCountStats:
    """Count JSONL records and parsed predictions using the same filters as evaluate_jsonl."""
    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS
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
                if rel in relations
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


def csv_prediction_counts(
    csv_path: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
    direction: str = "b",
) -> PredictionCountStats:
    """Count CSV rows and parsed predictions using the same filters as evaluate_csv."""
    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS
    frame_index_set = set(frame_indices)
    pred_index: dict[tuple[str, str, int], set[Triplet]] = {}
    stats = PredictionCountStats()

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats.records_total += 1
            camera = row["camera"]
            frame = int(row["frame"])
            if (cameras is not None and camera not in cameras) or frame not in frame_index_set:
                continue
            rel = canonicalize_relation(row["relation"])
            if rel is None or rel not in relations:
                continue
            stats.records_selected += 1
            key = (row["demo"], camera, frame)
            stats.parsed_triplets += 1
            pred_index.setdefault(key, set()).add((row["objectA"], rel, row["objectB"]))

    stats.scored_frames = len(pred_index)
    stats.scored_triplets = sum(len(v) for v in pred_index.values())
    stats.dedup_triplets_dropped = stats.parsed_triplets - stats.scored_triplets
    return stats


# ─── metric helpers ───────────────────────────────────────────────────────────

def _per_relation_metrics(results: list[FrameResult], relations: frozenset[str]) -> dict[str, dict]:
    out = {}
    for rel in sorted(relations):
        tp = sum(sum(1 for t in r.tp if t[1] == rel) for r in results)
        fp = sum(sum(1 for t in r.fp if t[1] == rel) for r in results)
        fn = sum(sum(1 for t in r.fn if t[1] == rel) for r in results)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        out[rel] = {"tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 4), "recall": round(r_, 4), "f1": round(f, 4)}
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
        for (a, rel, b) in r.gt:
            inv_rel = INVERSE_MAP.get(rel)
            if inv_rel and (b, inv_rel, a) in r.gt:
                total += 1
                if (a, rel, b) in r.pred and (b, inv_rel, a) in r.pred:
                    consistent += 1
    # each pair is counted twice (once per direction), result is the same ratio
    return consistent / total if total else math.nan


def _per_object_recall(results: list[FrameResult]) -> dict[str, float]:
    obj_gt: dict[str, int] = {}
    obj_tp: dict[str, int] = {}
    for r in results:
        for (a, _, b) in r.gt:
            for obj in (a, b):
                obj_gt[obj] = obj_gt.get(obj, 0) + 1
                obj_tp.setdefault(obj, 0)
        for (a, _, b) in r.tp:
            for obj in (a, b):
                obj_tp[obj] = obj_tp.get(obj, 0) + 1
    return {obj: round(obj_tp.get(obj, 0) / count, 4) for obj, count in sorted(obj_gt.items())}


# ─── core evaluation ──────────────────────────────────────────────────────────

def evaluate_csv(
    csv_path: Path,
    input_dir: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
    direction: str = "b",
) -> Iterator[FrameResult]:
    """
    Yields one FrameResult per (demo, camera, frame) found in the CSV.
    Matches each against the GT loaded from the corresponding HDF5 file.
    Triplets are filtered to the active relation set (bidirectional or unidirectional).
    """
    csv_path = Path(csv_path)
    input_dir = Path(input_dir)

    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS

    pred_index = _load_pred_csv(csv_path)
    if not pred_index:
        return

    task_name = None
    hdf5_path = None
    for f in list_hdf5_files(str(input_dir)):
        stem = f.stem.replace("_demo", "")
        if stem in csv_path.stem:
            task_name = stem
            hdf5_path = f
            break

    if hdf5_path is None:
        print(f"  [skip] {csv_path.name} — no matching HDF5 found in {input_dir}")
        return

    demos = sorted({demo for demo, _, _ in pred_index})
    cam_list = sorted({cam for _, cam, _ in pred_index if cameras is None or cam in (cameras or [])})
    gt_index = _load_all_gt(hdf5_path, demos, cam_list)
    frame_index_set = set(frame_indices)

    for (demo, camera, frame), pred in sorted(pred_index.items()):
        if (cameras is not None and camera not in cameras) or frame not in frame_index_set:
            continue
        gt = gt_index.get((demo, camera, frame), set())
        gt = {t for t in gt if t[1] in relations}
        pred = {t for t in pred if t[1] in relations}
        yield FrameResult(
            task=task_name,
            demo=demo,
            frame=frame,
            camera=camera,
            gt=gt,
            pred=pred,
        )


def evaluate_jsonl(
    jsonl_path: Path,
    input_dir: Path,
    cameras: list[str] | None,
    frame_indices: list[int],
    direction: str = "b",
) -> Iterator[FrameResult]:
    """Same as evaluate_csv but reads raw model responses from a JSONL log and re-parses them."""
    jsonl_path = Path(jsonl_path)
    input_dir = Path(input_dir)

    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS

    pred_index = _load_pred_jsonl(jsonl_path, cameras, frame_indices, direction=direction)
    if not pred_index:
        return

    # derive task name from filename (same convention as CSV: {task}_{camera}_{ver}.jsonl)
    task_name = None
    hdf5_path = None
    for f in list_hdf5_files(str(input_dir)):
        stem = f.stem.replace("_demo", "")
        if stem in jsonl_path.stem:
            task_name = stem
            hdf5_path = f
            break

    if hdf5_path is None:
        print(f"  [skip] {jsonl_path.name} — no matching HDF5 found in {input_dir}")
        return

    demos = sorted({demo for demo, _, _ in pred_index})
    cam_list = sorted({cam for _, cam, _ in pred_index if cameras is None or cam in (cameras or [])})
    gt_index = _load_all_gt(hdf5_path, demos, cam_list)

    for (demo, camera, frame), pred in sorted(pred_index.items()):
        gt = gt_index.get((demo, camera, frame), set())
        gt = {t for t in gt if t[1] in relations}
        yield FrameResult(
            task=task_name,
            demo=demo,
            frame=frame,
            camera=camera,
            gt=gt,
            pred=pred,
        )


def aggregate(results: list[FrameResult], direction: str = "b") -> AggregateMetrics:
    """Macro-average P/R/F1 across frames plus micro and per-relation metrics."""
    relations = BIDIRECTIONAL_RELATIONS if direction == "b" else UNIDIRECTIONAL_RELATIONS

    if not results:
        empty_rel = {r: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                     for r in sorted(relations)}
        return AggregateMetrics(
            precision=0, recall=0, f1=0,
            n_frames=0, n_gt_triplets=0, n_pred_triplets=0,
            n_tp=0, n_fp=0, n_fn=0,
            micro_precision=0, micro_recall=0, micro_f1=0,
            per_relation=empty_rel,
            coverage=0.0, hallucination_rate=0.0, n_reversals=0, reversal_rate=0.0,
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
        t_micro_f = 2 * t_micro_p * t_micro_r / (t_micro_p + t_micro_r) if (t_micro_p + t_micro_r) else 0.0
        
        t_cov = _coverage(t_results)
        t_n_rev = sum(len(r.reversals) for r in t_results)
        t_hal = t_fp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
        t_rev_rate = t_rev / t_fn if t_fn else 0.0
        t_dc = _direction_consistency(t_results) if direction == "b" else math.nan
        
        t_demo_f1s = {}
        for r in t_results:
            t_demo_f1s.setdefault(r.demo, []).append(r.f1)
        mean_demo_f1 = float(np.mean([float(np.mean(f1s)) for f1s in t_demo_f1s.values()])) if t_demo_f1s else 0.0
        
        per_task_metrics[task] = {
            "demo_f1": round(mean_demo_f1, 4),
            "macro_f1": round(t_macro_f1, 4),
            "micro_f1": round(t_micro_f, 4),
            "coverage": round(t_cov, 4),
            "hallucination": round(t_hal, 4),
            "n_reversals": t_n_rev,
            "reversal_rate": round(t_rev_rate, 4),
            "dir_consistency": round(t_dc, 4) if not math.isnan(t_dc) else math.nan,
        }

    mean_per_task_f1 = float(np.mean([m["demo_f1"] for m in per_task_metrics.values()])) if per_task_metrics else 0.0

    dc = _direction_consistency(results) if direction == "b" else math.nan

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
        per_relation=_per_relation_metrics(results, relations),
        coverage=round(_coverage(results), 4),
        hallucination_rate=round(n_fp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0, 4),
        n_reversals=n_rev,
        reversal_rate=round(n_rev / n_fn if n_fn else 0.0, 4),
        direction_consistency=dc,
        per_object_recall=_per_object_recall(results),
        per_task_metrics=per_task_metrics,
        mean_per_task_f1=round(mean_per_task_f1, 4),
    )



# ─── reporting ────────────────────────────────────────────────────────────────

def print_frame_report(r: FrameResult, verbose: bool = False):
    print(f"\n{'='*64}")
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
            print(f"\n  FALSE POSITIVES — model predicted, not in GT ({len(r.fp)}):")
            for t in sorted(r.fp):
                print(f"    {t[0]}, {t[1]}, {t[2]}")
        if r.fn:
            print(f"\n  FALSE NEGATIVES — in GT, model missed ({len(r.fn)}):")
            for t in sorted(r.fn):
                print(f"    {t[0]}, {t[1]}, {t[2]}")


def print_aggregate_report(m: AggregateMetrics, label: str = "", direction: str = "b"):
    header = f"AGGREGATE{f' — {label}' if label else ''}"
    print(f"\n{'='*64}")
    print(f"  {header}")
    print(f"  Frames   : {m.n_frames}")
    print(f"  GT total : {m.n_gt_triplets}  |  Pred total : {m.n_pred_triplets}")
    print(f"  TP={m.n_tp}  FP={m.n_fp}  FN={m.n_fn}")
    print(f"  Macro  Precision={m.precision:.3f}  Recall={m.recall:.3f}  F1={m.f1:.3f}")
    print(f"  Micro  Precision={m.micro_precision:.3f}  Recall={m.micro_recall:.3f}  F1={m.micro_f1:.3f}")
    print(f"  Coverage={m.coverage:.3f}  Hallucination={m.hallucination_rate:.3f}  Reversals={m.n_reversals} ({m.reversal_rate:.3f})")
    if direction == "b":
        dc = m.direction_consistency
        if math.isnan(dc):
            print(f"  Direction Consistency=N/A")
        else:
            print(f"  Direction Consistency={dc:.3f}")

    if m.per_relation:
        print(f"\n  Per-Relation F1:")
        for rel, rm in m.per_relation.items():
            print(f"    {rel:<22}  P={rm['precision']:.3f}  R={rm['recall']:.3f}  "
                  f"F1={rm['f1']:.3f}  (TP={rm['tp']} FP={rm['fp']} FN={rm['fn']})")

    if m.per_object_recall:
        print(f"\n  Per-Object Recall:")
        for obj, rec in m.per_object_recall.items():
            print(f"    {obj:<40}  Recall={rec:.3f}")

    if m.per_task_metrics:
        print(f"\n  Task-Based Details:")
        if direction == "b":
            print(f"    {'Task':<60} | {'mF1':<7} | {'Macro F1':<8} | {'Micro F1':<8} | {'Cov':<5} | {'Hal':<5} | {'Rev':<5} | {'DC':<5}")
            print("    " + "-" * 124)
        else:
            print(f"    {'Task':<60} | {'mF1':<7} | {'Macro F1':<8} | {'Micro F1':<8} | {'Cov':<5} | {'Hal':<5} | {'Rev':<5}")
            print("    " + "-" * 116)
            
        for task, tm in sorted(m.per_task_metrics.items()):
            t_name = task if len(task) <= 60 else task[:57] + "..."
            if direction == "b":
                dc_str = f"{tm['dir_consistency']:.3f}" if not math.isnan(tm['dir_consistency']) else "N/A"
                print(f"    {t_name:<60} | {tm['demo_f1']:<7.3f} | {tm['macro_f1']:<8.3f} | {tm['micro_f1']:<8.3f} | {tm['coverage']:<5.3f} | {tm['hallucination']:<5.3f} | {tm['n_reversals']:<3} ({tm['reversal_rate']:.3f}) | {dc_str:<5}")
            else:
                print(f"    {t_name:<60} | {tm['demo_f1']:<7.3f} | {tm['macro_f1']:<8.3f} | {tm['micro_f1']:<8.3f} | {tm['coverage']:<5.3f} | {tm['hallucination']:<5.3f} | {tm['n_reversals']:<3} ({tm['reversal_rate']:.3f})")
                
        print(f"\n  > {'Mean Per-Task (Demo) F1':<58}  F1={m.mean_per_task_f1:.3f}")
