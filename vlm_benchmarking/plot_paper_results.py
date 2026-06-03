#!/usr/bin/env python3
"""
Create paper-ready bar charts from paper_results.csv.

Default output:
  figures/paper_results/
    summary/*.png
    relations/*.png
    aggregate/*.png
    chart_manifest.csv

Usage:
  python plot_paper_results.py
  python plot_paper_results.py --input paper_results.csv --out-dir figures/paper_results
  python plot_paper_results.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAMERAS = [
    ("agentview", "Agent view", "#4C78A8"),
    ("eye_in_hand", "Wrist view", "#F58518"),
]

FONT_TITLE = 18
FONT_AXIS = 20
FONT_TICK = 18
FONT_MODEL = 18
FONT_LEGEND = 18

SUMMARY_METRICS = [
    ("macro_f1", "Macro F1"),
    ("macro_precision", "Macro precision"),
    ("macro_recall", "Macro recall"),
    ("micro_f1", "Micro F1"),
    ("micro_precision", "Micro precision"),
    ("micro_recall", "Micro recall"),
    ("mean_per_task_f1", "Mean per-task F1"),
    ("coverage", "Relation-type coverage"),
    ("hallucination_rate", "Hallucination rate"),
    ("reversal_rate", "Reversal rate"),
    ("direction_consistency", "Direction consistency"),
]

RELATION_METRICS = [
    ("contains_f1", "contains F1"),
    ("is_behind_f1", "is_behind F1"),
    ("is_below_of_f1", "is_below_of F1"),
    ("is_in_front_of_f1", "is_in_front_of F1"),
    ("is_inside_f1", "is_inside F1"),
    ("is_left_of_f1", "is_left_of F1"),
    ("is_on_top_of_f1", "is_on_top_of F1"),
    ("is_right_of_f1", "is_right_of F1"),
]

DIRECT_METRICS = SUMMARY_METRICS + RELATION_METRICS


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_value(row: dict[str, str], metric: str, camera: str) -> float | None:
    return _to_float(row.get(f"{metric}_{camera}"))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _model_label(model: str, max_len: int = 52) -> str:
    label = model.replace("--", "/")
    return label if len(label) <= max_len else label[: max_len - 1] + "..."


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _has_any_metric(row: dict[str, str], metrics: Iterable[tuple[str, str]]) -> bool:
    for metric, _ in metrics:
        for camera, _, _ in CAMERAS:
            if _metric_value(row, metric, camera) is not None:
                return True
    return False


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def score(row: dict[str, str]) -> float:
        value = _metric_value(row, "mean_per_task_f1", "agentview")
        return value if value is not None else -math.inf

    return sorted(rows, key=score, reverse=True)


def _save_figure(fig, out_base: Path, formats: list[str], dpi: int) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        written.append(out_path)
    plt.close(fig)
    return written


def _style_horizontal_axis(ax, y: np.ndarray, labels: list[str], xlabel: str, title: str) -> None:
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_MODEL)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelleft=True, labelright=False, left=False, right=False, pad=10)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    ax.grid(axis="x", color="#dddddd", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.36, -0.02),
        borderaxespad=0.0,
        frameon=True,
        fontsize=FONT_LEGEND,
    )


def _plot_grouped_metric(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    models = [_model_label(row["model"]) for row in rows]
    y = np.arange(len(rows), dtype=float)
    bar_h = 0.34
    fig_h = max(6.0, 0.48 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    all_values: list[float] = []
    for cam_i, (camera, camera_label, color) in enumerate(CAMERAS):
        offset = (cam_i - 0.5) * bar_h
        plot_y = []
        values = []
        for row_i, row in enumerate(rows):
            value = _metric_value(row, metric, camera)
            if value is None:
                continue
            plot_y.append(y[row_i] + offset)
            values.append(value)
        if not values:
            continue
        ax.barh(plot_y, values, height=bar_h, label=camera_label, color=color, alpha=0.9)
        all_values.extend(values)

    x_lim = 1.0
    if all_values:
        x_max = max(all_values)
        x_lim = 1.0 if x_max <= 1.0 else x_max * 1.16
        ax.set_xlim(0, x_lim)

    _style_horizontal_axis(ax, y, models, title, f"{title} by model and camera")
    return _save_figure(fig, out_base, formats, dpi)


def _plot_completeness(
    rows: list[dict[str, str]],
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    models = [_model_label(row["model"]) for row in rows]
    y = np.arange(len(rows), dtype=float)
    bar_h = 0.34
    fig_h = max(6.0, 0.48 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    for cam_i, (camera, camera_label, color) in enumerate(CAMERAS):
        offset = (cam_i - 0.5) * bar_h
        plot_y = []
        values = []
        for row_i, row in enumerate(rows):
            done = _to_float(row.get(f"n_frames_done_{camera}"))
            expected = _to_float(row.get(f"n_frames_expected_{camera}"))
            if done is None or expected is None or expected <= 0:
                continue
            plot_y.append(y[row_i] + offset)
            values.append(100.0 * done / expected)
        if values:
            ax.barh(plot_y, values, height=bar_h, label=camera_label, color=color, alpha=0.9)

    ax.set_xlim(0, 105)
    _style_horizontal_axis(ax, y, models, "Completed frames (%)", "Evaluation coverage by model and camera")
    return _save_figure(fig, out_base, formats, dpi)


def _plot_relation_means(
    rows: list[dict[str, str]],
    out_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    labels = [label.replace(" F1", "") for _, label in RELATION_METRICS]
    y = np.arange(len(labels), dtype=float)
    bar_h = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 6.4))

    for cam_i, (camera, camera_label, color) in enumerate(CAMERAS):
        values = []
        for metric, _ in RELATION_METRICS:
            vals = [
                _metric_value(row, metric, camera)
                for row in rows
                if _metric_value(row, metric, camera) is not None
            ]
            values.append(float(np.mean(vals)) if vals else 0.0)
        offset = (cam_i - 0.5) * bar_h
        ax.barh(y + offset, values, height=bar_h, label=camera_label, color=color, alpha=0.9)

    ax.set_xlim(0, 1.0)
    _style_horizontal_axis(ax, y, labels, "Mean F1 across evaluated models", "Relation difficulty by camera")
    return _save_figure(fig, out_base, formats, dpi)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chart", "category", "description", "files"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bar charts from paper_results.csv")
    parser.add_argument("--input", default="paper_results.csv", help="Input paper_results.csv path")
    parser.add_argument("--out-dir", default="figures/paper_results", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "svg"],
                        help="Figure formats to write")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    rows_all = _read_rows(input_path)
    rows = [row for row in rows_all if _has_any_metric(row, DIRECT_METRICS)]
    rows = _sort_rows(rows)

    if not rows:
        raise SystemExit(f"No plottable metric rows found in {input_path}")

    manifest = []

    for metric, title in SUMMARY_METRICS:
        files = _plot_grouped_metric(
            rows,
            metric,
            title,
            out_dir / "summary" / f"{_safe_name(metric)}_by_model",
            args.formats,
            args.dpi,
        )
        manifest.append({
            "chart": f"{metric}_by_model",
            "category": "summary",
            "description": f"{title} by model, grouped by camera",
            "files": ";".join(str(p) for p in files),
        })

    for metric, title in RELATION_METRICS:
        files = _plot_grouped_metric(
            rows,
            metric,
            title,
            out_dir / "relations" / f"{_safe_name(metric)}_by_model",
            args.formats,
            args.dpi,
        )
        manifest.append({
            "chart": f"{metric}_by_model",
            "category": "relations",
            "description": f"{title} by model, grouped by camera",
            "files": ";".join(str(p) for p in files),
        })

    files = _plot_relation_means(
        rows,
        out_dir / "aggregate" / "mean_relation_f1_by_camera",
        args.formats,
        args.dpi,
    )
    manifest.append({
        "chart": "mean_relation_f1_by_camera",
        "category": "aggregate",
        "description": "Mean relation F1 across evaluated models, grouped by camera",
        "files": ";".join(str(p) for p in files),
    })

    files = _plot_completeness(
        rows,
        out_dir / "aggregate" / "completeness_by_model",
        args.formats,
        args.dpi,
    )
    manifest.append({
        "chart": "completeness_by_model",
        "category": "aggregate",
        "description": "Completed evaluation frames as a percentage of expected frames",
        "files": ";".join(str(p) for p in files),
    })

    manifest_path = out_dir / "chart_manifest.csv"
    _write_manifest(manifest_path, manifest)

    n_direct = len(SUMMARY_METRICS) + len(RELATION_METRICS)
    print(f"Read {len(rows_all)} model rows from {input_path}; plotted {len(rows)} rows with metrics.")
    print(f"Created {n_direct} direct metric bar charts + 2 aggregate bar charts = {n_direct + 2} chart designs.")
    print(f"Wrote {len(manifest) * len(args.formats)} figure files and manifest: {manifest_path}")


if __name__ == "__main__":
    main()
