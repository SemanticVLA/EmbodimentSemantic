#!/usr/bin/env python3
"""
Render GT scene graph(s) as Graphviz diagrams for a specific (model, task, demo, frame).

Outputs into the same figures/<stem>/ folder as plot_frame.py:
    agentview_gt_graph.png
    eye_in_hand_gt_graph.png

Usage:
    python plot_graph.py --model gemini-3.1-pro-preview --task-id 7 --demo demo_0 --frame 0
    python plot_graph.py --model gemini-3.1-pro-preview --task pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate --demo demo_0 --frame 0
"""

import argparse
import csv
from pathlib import Path

import graphviz

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

NODE_PALETTE = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A",
    "#D7BDE2", "#FAD7A0", "#A2D9CE", "#FDEBD0",
]

OBJ_ABBR = {
    "akita_black_bowl":             "bowl",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "wooden_cabinet":               "cabinet",
    "flat_stove":                   "stove",
    "cookies":                      "cookies",
    "plate":                        "plate",
}


def _short_obj(name: str) -> str:
    parts = name.rsplit("_", 1)
    base = OBJ_ABBR.get(parts[0], parts[0].replace("_", " "))
    idx  = parts[1] if len(parts) > 1 else ""
    return f"{base} {idx}" if idx else base


def _short_rel(rel: str) -> str:
    return rel.replace("is_", "").replace("_of", "").replace("_", " ")


def _render_graph(gt_path: Path, pred_path: Path, out_path: Path, title: str):
    with open(gt_path, newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))
    with open(pred_path, newline="", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))

    pred_set = {(r["objectA"], r["relation"], r["objectB"]) for r in pred_rows}

    nodes = sorted({r["objectA"] for r in gt_rows} | {r["objectB"] for r in gt_rows})
    node_color = {n: NODE_PALETTE[i % len(NODE_PALETTE)] for i, n in enumerate(nodes)}

    dot = graphviz.Digraph(
        graph_attr={
            "label":     title,
            "labelloc":  "t",
            "fontname":  "Helvetica",
            "fontsize":  "14",
            "rankdir":   "LR",
            "splines":   "spline",
            "bgcolor":   "white",
            "nodesep":   "0.6",
            "ranksep":   "1.4",
            "pad":       "0.4",
        },
        node_attr={
            "shape":    "box",
            "style":    "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "12",
            "margin":   "0.2,0.1",
            "penwidth": "1.5",
        },
        edge_attr={
            "fontname":  "Helvetica",
            "fontsize":  "9",
            "arrowsize": "0.6",
        },
    )

    for n in nodes:
        dot.node(n, label=_short_obj(n), fillcolor=node_color[n])

    n_tp, n_fn = 0, 0
    for r in gt_rows:
        a, rel, b = r["objectA"], r["relation"], r["objectB"]
        is_tp = (a, rel, b) in pred_set
        n_tp += is_tp
        n_fn += not is_tp
        dot.edge(
            a, b,
            label=_short_rel(rel),
            color="#27AE60" if is_tp else "#E74C3C",
            fontcolor="#555555",
            penwidth="2.5" if is_tp else "1.2",
            style="solid" if is_tp else "dashed",
        )

    dot.render(str(out_path.with_suffix("")), format="png", cleanup=True)
    print(f"Saved {out_path}  (TP={n_tp} FN={n_fn})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-id", type=int, choices=list(TASK_NAMES.keys()),
                            metavar="{0-9}")
    task_group.add_argument("--task")
    parser.add_argument("--demo", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS,
                        choices=CAMERAS, metavar="CAM")
    args = parser.parse_args()

    task = TASK_NAMES[args.task_id] if args.task_id is not None else args.task.strip()
    safe_task = task[:50]
    fig_dir = Path("figures") / f"{args.model}_{safe_task}_{args.demo}_f{args.frame}"

    if not fig_dir.exists():
        print(f"ERROR: figures folder not found: {fig_dir}")
        print("Run plot_frame.py first to generate GT/pred CSVs.")
        raise SystemExit(1)

    for camera in args.cameras:
        gt_path   = fig_dir / f"{camera}_gt.csv"
        pred_path = fig_dir / f"{camera}_pred.csv"
        if not gt_path.exists():
            print(f"WARNING: {gt_path} not found, skipping {camera}")
            continue
        title = f"{camera}  |  {task[:60]}  |  {args.demo}  |  frame {args.frame}"
        _render_graph(gt_path, pred_path, fig_dir / f"{camera}_gt_graph.png", title)

    print(f"\nAll graphs in: {fig_dir}")


if __name__ == "__main__":
    main()
