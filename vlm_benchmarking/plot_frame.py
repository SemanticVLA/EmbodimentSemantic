#!/usr/bin/env python3
"""
Save frame images and scene graph CSVs for a specific (task, demo, frame).

Outputs into figures/<stem>/:
    agentview.png          — raw frame image
    eye_in_hand.png        — raw frame image
    agentview_gt.csv       — ground truth triplets with TP/FN label
    agentview_pred.csv     — predicted triplets with TP/FP label
    eye_in_hand_gt.csv
    eye_in_hand_pred.csv

Usage:
    python plot_frame.py `
        --model gemini-3.1-pro-preview `
        --task pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate `
        --demo demo_0 `
        --frame 0 `
        --input-dir data/libero_spatial_v5/
"""

import argparse
import csv as csv_mod
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import yaml
from PIL import Image

from vlm_bench.eval import _load_gt, _load_pred_csv
from vlm_bench.io_utils import frame_to_pil, list_hdf5_files

CAMERAS = ["agentview", "eye_in_hand"]

TASK_NAMES = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}


def _find_hdf5(task: str, input_dir: Path) -> Path:
    task = task.strip()
    for p in list_hdf5_files(str(input_dir)):
        if task in p.stem:
            return p
    raise FileNotFoundError(f"No HDF5 found for task '{task}' in {input_dir}")


def _load_image(hdf5_path: Path, demo: str, frame: int, camera: str, rotate180: bool):
    key_map = {"agentview": "agentview_rgb", "eye_in_hand": "eye_in_hand_rgb"}
    with h5py.File(hdf5_path, "r") as f:
        arr = f[f"data/{demo}/obs/{key_map[camera]}"][frame]
    return frame_to_pil(arr, rotate180=(rotate180 and camera == "agentview"))


def _load_pred(model_dir: Path, camera: str, task: str, demo: str, frame: int) -> set[tuple]:
    task = task.strip()
    csv_dir = model_dir / camera / "csv"
    if not csv_dir.is_dir():
        return set()
    for csv_path in sorted(csv_dir.glob("*.csv")):
        if task in csv_path.stem:
            index = _load_pred_csv(csv_path)
            return index.get((demo, camera, frame), set())
    return set()


def _make_figure(camera: str, img_path: Path, gt_path: Path, pred_path: Path, out_dir: Path,
                 task: str = "", demo: str = "", frame: int = 0):
    """Render image + GT/pred tables side-by-side and save as PNG."""
    hires_path = out_dir / f"{camera}_hires.png"
    src = hires_path if hires_path.exists() else img_path
    img = np.array(Image.open(src))

    def _read_csv(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv_mod.DictReader(f))

    gt_rows = _read_csv(gt_path)
    pred_rows = _read_csv(pred_path)

    def _fmt(r):
        return f"{r['objectA']}  {r['relation']}  {r['objectB']}"

    COL_TP = "#c8e6c9"   # green
    COL_FP = "#ffcdd2"   # red
    COL_FN = "#fff9c4"   # yellow
    COL_HDR_GT   = "#2e7d32"
    COL_HDR_PRED = "#c62828"

    def _table_data(rows, label_col):
        if not rows:
            return [["(none)", ""]], [["#f5f5f5", "#f5f5f5"]]
        data, colors = [], []
        for r in rows:
            lbl = r[label_col]
            c = COL_TP if lbl == "TP" else (COL_FP if lbl == "FP" else COL_FN)
            data.append([_fmt(r), lbl])
            colors.append([c, c])
        return data, colors

    gt_data,   gt_colors   = _table_data(gt_rows,   "label")
    pred_data, pred_colors = _table_data(pred_rows, "label")

    n_gt   = len(gt_data)
    n_pred = len(pred_data)
    row_h  = 0.28          # inches per table row
    pad    = 1.0           # inches between sections
    hdr    = 0.45          # inches for section header
    img_h  = max(5.0, img.shape[0] / img.shape[1] * 5.0)

    right_h = hdr + n_gt * row_h + pad + hdr + n_pred * row_h + 0.6
    fig_h   = max(img_h + 1.0, right_h + 1.2)
    fig_w   = 18.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Two columns: image (48%) | tables (52%)
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.48, 0.52],
        left=0.02, right=0.98,
        top=0.91, bottom=0.04,
    )

    # ── image ────────────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    is_hires = hires_path.exists()
    cam_label = camera.replace("_", " ")
    subtitle = f"{task}   |   {demo}   |   frame {frame}"
    ax_img.set_title(f"{cam_label}\n{subtitle}", fontsize=11, fontweight="bold",
                     pad=8, color="#212121", linespacing=1.6)

    # ── right column: stacked axes for GT then Pred ──────────────────────────
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.axis("off")

    right_total = hdr + n_gt * row_h + pad + hdr + n_pred * row_h
    y = 1.0  # normalised [0,1] within ax_right, drawn top-down

    def _draw_table(ax, data, colors, title, title_color, y_top, section_h):
        # section title
        ax.text(
            0.5, y_top, title,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=title_color,
        )
        y_table = y_top - hdr / section_h * (1 / (right_total / section_h + 0.001))

        # compute bbox in axes-fraction coords
        title_frac = hdr / right_total
        table_frac = len(data) * row_h / right_total

        tbl = ax.table(
            cellText=data,
            colLabels=["triplet", ""],
            cellColours=colors,
            colColours=[title_color + "33", title_color + "33"],
            cellLoc="left",
            bbox=[0.01, y_top - title_frac - table_frac, 0.98, table_frac],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.auto_set_column_width([0, 1])
        # make label column narrow
        for (row, col), cell in tbl.get_celld().items():
            if col == 1:
                cell.set_width(0.08)
            cell.set_edgecolor("#bdbdbd")
            cell.set_linewidth(0.5)
        return y_top - title_frac - table_frac

    gt_section_h   = hdr + n_gt   * row_h
    pred_section_h = hdr + n_pred * row_h

    gt_title_frac   = hdr / right_total
    gt_table_frac   = n_gt * row_h / right_total
    pad_frac        = pad / right_total
    pred_title_frac = hdr / right_total
    pred_table_frac = n_pred * row_h / right_total

    # GT block
    y_gt_title = 1.0
    ax_right.text(0.5, y_gt_title, "Ground Truth",
                  transform=ax_right.transAxes,
                  ha="center", va="top",
                  fontsize=12, fontweight="bold", color=COL_HDR_GT)

    y_gt_table_top = y_gt_title - gt_title_frac
    tbl_gt = ax_right.table(
        cellText=gt_data,
        colLabels=["triplet", ""],
        cellColours=gt_colors,
        colColours=["#a5d6a7", "#a5d6a7"],
        cellLoc="left",
        bbox=[0.01, y_gt_table_top - gt_table_frac, 0.97, gt_table_frac],
    )
    tbl_gt.auto_set_font_size(False)
    tbl_gt.set_fontsize(8.5)
    for (row, col), cell in tbl_gt.get_celld().items():
        if col == 1:
            cell.set_width(0.07)
        cell.set_edgecolor("#9e9e9e")
        cell.set_linewidth(0.4)

    # Pred block
    y_pred_title = y_gt_table_top - gt_table_frac - pad_frac
    ax_right.text(0.5, y_pred_title, "Predictions",
                  transform=ax_right.transAxes,
                  ha="center", va="top",
                  fontsize=12, fontweight="bold", color=COL_HDR_PRED)

    y_pred_table_top = y_pred_title - pred_title_frac
    tbl_pred = ax_right.table(
        cellText=pred_data,
        colLabels=["triplet", ""],
        cellColours=pred_colors,
        colColours=["#ef9a9a", "#ef9a9a"],
        cellLoc="left",
        bbox=[0.01, y_pred_table_top - pred_table_frac, 0.97, pred_table_frac],
    )
    tbl_pred.auto_set_font_size(False)
    tbl_pred.set_fontsize(8.5)
    for (row, col), cell in tbl_pred.get_celld().items():
        if col == 1:
            cell.set_width(0.07)
        cell.set_edgecolor("#9e9e9e")
        cell.set_linewidth(0.4)

    # legend — placed in the gap between GT table and Pred title
    y_legend = y_gt_table_top - gt_table_frac - pad_frac * 0.5  # vertical midpoint of gap
    legend_patches = [
        mpatches.Patch(color=COL_TP, label="TP — correct"),
        mpatches.Patch(color=COL_FP, label="FP — hallucinated"),
        mpatches.Patch(color=COL_FN, label="FN — missed"),
    ]
    ax_right.legend(
        handles=legend_patches,
        loc="center", bbox_to_anchor=(0.5, y_legend),
        ncol=3, fontsize=9, framealpha=0.9, edgecolor="#bdbdbd",
        bbox_transform=ax_right.transAxes,
    )

    out_path = out_dir / f"{camera}_figure.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure {out_path}")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("objectA,relation,objectB,label\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=["objectA", "relation", "objectB", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-id", type=int, choices=list(TASK_NAMES.keys()),
                            metavar="{0-9}", help=f"Task index: {', '.join(f'{k}={v[:30]}...' for k, v in TASK_NAMES.items())}")
    task_group.add_argument("--task", help="Full task name string")
    parser.add_argument("--demo", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--input-dir", default="data/libero_spatial_v5/")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--hires", type=int, default=None, metavar="RES",
                        help="Also render at RES×RES via simulator (requires vla_bench_py312). E.g. --hires 512")
    parser.add_argument("--conda-env", default="vla_bench_py312",
                        help="Conda env that has libero installed (default: vla_bench_py312)")
    args = parser.parse_args()

    if args.task_id is not None:
        args.task = TASK_NAMES[args.task_id]
    else:
        args.task = args.task.strip()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rotate180 = cfg["extraction"].get("rotate_agentview_180", True)

    input_dir = Path(args.input_dir)
    model_dir = Path(args.output_dir) / args.model

    if not model_dir.is_dir():
        print(f"ERROR: model directory not found: {model_dir}")
        sys.exit(1)

    hdf5_path = _find_hdf5(args.task, input_dir)

    # output folder: figures/<model>_<task>_<demo>_f<frame>/
    safe_task = args.task[:50]
    out_dir = Path("figures") / f"{args.model}_{safe_task}_{args.demo}_f{args.frame}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for camera in CAMERAS:
        # ── image ─────────────────────────────────────────────────────────────
        try:
            img = _load_image(hdf5_path, args.demo, args.frame, camera, rotate180)
            img_path = out_dir / f"{camera}.png"
            img.save(img_path)
            print(f"Saved {img_path}")
        except Exception as e:
            print(f"WARNING: could not save {camera} image: {e}")

        # ── graphs ────────────────────────────────────────────────────────────
        gt = _load_gt(hdf5_path, args.demo, args.frame, camera)
        pred = _load_pred(model_dir, camera, args.task, args.demo, args.frame)

        tp = gt & pred
        fp = pred - gt
        fn = gt - pred

        gt_rows = [
            {"objectA": a, "relation": r, "objectB": b, "label": "TP" if (a, r, b) in tp else "FN"}
            for a, r, b in sorted(gt)
        ]
        pred_rows = [
            {"objectA": a, "relation": r, "objectB": b, "label": "TP" if (a, r, b) in tp else "FP"}
            for a, r, b in sorted(pred)
        ]

        gt_path = out_dir / f"{camera}_gt.csv"
        pred_path = out_dir / f"{camera}_pred.csv"
        _write_csv(gt_path, gt_rows)
        _write_csv(pred_path, pred_rows)
        print(f"Saved {gt_path}  ({len(gt)} triplets: {len(tp)} TP, {len(fn)} FN)")
        print(f"Saved {pred_path}  ({len(pred)} triplets: {len(tp)} TP, {len(fp)} FP)")

    # ── hi-res re-render via simulator ───────────────────────────────────────
    if args.hires:
        import os
        render_script = Path(__file__).parent / "render_hires.py"
        rotate_flag = ["--rotate-agentview"] if rotate180 else []
        # Call the conda env's python directly — avoids conda run resetting sys.path
        python_exe = Path(f"C:/Users/hassa/anaconda3/envs/{args.conda_env}/python.exe")
        if not python_exe.exists():
            print(f"WARNING: python not found at {python_exe}, skipping hi-res render")
            args.hires = None
        env = os.environ.copy()
        cmd = [
            str(python_exe), str(render_script),
            "--hdf5", str(hdf5_path),
            "--demo", args.demo,
            "--frame", str(args.frame),
            "--res", str(args.hires),
            "--out-dir", str(out_dir),
            *rotate_flag,
        ]
        print(f"\nRunning hi-res render ({args.hires}×{args.hires}) via {args.conda_env}...")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"WARNING: hi-res render failed (exit {result.returncode})")

    # ── matplotlib figures ────────────────────────────────────────────────────
    for camera in CAMERAS:
        img_path = out_dir / f"{camera}.png"
        gt_path = out_dir / f"{camera}_gt.csv"
        pred_path = out_dir / f"{camera}_pred.csv"
        if img_path.exists() and gt_path.exists() and pred_path.exists():
            _make_figure(camera, img_path, gt_path, pred_path, out_dir,
                         task=args.task, demo=args.demo, frame=args.frame)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
