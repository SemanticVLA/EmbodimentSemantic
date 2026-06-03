"""Visualize scene graph triplets for tasks 2 and 6, episode 0, step 0."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "wgl")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_libero_root = os.path.join(_repo_root, "LIBERO")
os.environ["LIBERO_CONFIG_PATH"] = os.path.join(_repo_root, ".libero")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _libero_root)
sys.path.insert(0, os.path.join(_repo_root, "dataset_eval"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robosuite.utils import camera_utils as CU

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_live_semantic_context import get_bbox, generate_frame_graph, discover_objects
from radomize_scenes import settle_physics
from config import (
    BENCHMARK_NAME, CAMERA_HEIGHT, CAMERA_WIDTH, SETTLE_STEPS_INIT,
    TASK_CAMERA_OVERRIDE,
)

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]

OUT_DIR = os.path.join(_repo_root, "dataset_eval", "swap_outputs")
GRAPH_CAMERA = "agentview"


def get_bboxes(env, camera, img_h, img_w):
    objects = discover_objects(env)
    K = CU.get_camera_intrinsic_matrix(env.sim, camera, img_h, img_w)
    ext_inv = np.linalg.inv(CU.get_camera_extrinsic_matrix(env.sim, camera))
    result = {}
    for label, body in objects.items():
        bbox = get_bbox(env, body, ext_inv, K, img_h, img_w)
        if bbox is not None:
            result[label] = list(bbox)
    return result


def get_world(env):
    objects = discover_objects(env)
    world = {}
    for label, body in objects.items():
        bid = env.sim.model.body_name2id(body)
        world[label] = {"pos": env.sim.data.body_xpos[bid].tolist()}
    return world


def draw_graph(ax, img, bboxes, triplets, title):
    """Render image with bbox overlays and print triplets as text below."""
    ax.imshow(img)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    label_colors = {}
    for i, label in enumerate(sorted(bboxes.keys())):
        label_colors[label] = COLORS[i % len(COLORS)]

    for label, (x1, y1, x2, y2) in bboxes.items():
        color = label_colors[label]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 3, 0), label,
                fontsize=6, color=color,
                bbox=dict(fc="black", alpha=0.5, pad=1, boxstyle="round,pad=0.1"))

    # Print triplets as text block inside the image bottom
    if triplets:
        lines = [f"{a}  —[{r}]→  {b}" for a, r, b in triplets]
        text = "\n".join(lines)
        ax.text(4, CAMERA_HEIGHT - 4, text,
                fontsize=5.5, color="white", va="bottom",
                bbox=dict(fc="black", alpha=0.6, pad=2, boxstyle="round,pad=0.3"))


def run_task(task_id):
    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[BENCHMARK_NAME]().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
    )
    env.reset()
    settle_physics(env, max_steps=SETTLE_STEPS_INIT)

    # Scene graph always computed from agentview
    bboxes_agentview = get_bboxes(env, GRAPH_CAMERA, CAMERA_HEIGHT, CAMERA_WIDTH)
    world = get_world(env)
    triplets = generate_frame_graph(bboxes_agentview, world, object_filter=set(bboxes_agentview.keys()))

    # Display on override cameras (or agentview if no override)
    override_str = TASK_CAMERA_OVERRIDE.get(task_id, "")
    override_cameras = [c.strip() for c in override_str.split(",") if c.strip()]
    display_cameras = override_cameras if override_cameras else [GRAPH_CAMERA]

    n_cams = len(display_cameras)
    fig, axes = plt.subplots(1, n_cams, figsize=(7 * n_cams, 7))
    if n_cams == 1:
        axes = [axes]

    for ax, camera in zip(axes, display_cameras):
        # Bboxes projected for this display camera
        bboxes_raw = get_bboxes(env, camera, CAMERA_HEIGHT, CAMERA_WIDTH)
        bboxes_display = {
            label: (x1, (CAMERA_HEIGHT - 1) - y2, x2, (CAMERA_HEIGHT - 1) - y1)
            for label, (x1, y1, x2, y2) in bboxes_raw.items()
        }
        img = env.sim.render(camera_name=camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)[::-1]
        draw_graph(ax, img, bboxes_display, triplets, f"Task {task_id} | {camera} | graph from agentview")

    fig.suptitle(f"Task {task_id}: {task.language}", fontsize=9)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"task_{task_id:02d}_scene_graph.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTask {task_id} — {len(triplets)} triplets:")
    for a, r, b in triplets:
        print(f"  {a}  [{r}]  {b}")
    print(f"Saved: {out_path}")

    env.close()


def check_graph_consistency(task_id):
    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[BENCHMARK_NAME]().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAMERA_HEIGHT, camera_widths=CAMERA_WIDTH)
    env.reset()
    settle_physics(env, max_steps=SETTLE_STEPS_INIT)

    world = get_world(env)
    override_str = TASK_CAMERA_OVERRIDE.get(task_id, "")
    all_cameras = [GRAPH_CAMERA] + [c.strip() for c in override_str.split(",") if c.strip()]

    graphs = {}
    for camera in all_cameras:
        bboxes = get_bboxes(env, camera, CAMERA_HEIGHT, CAMERA_WIDTH)
        triplets = generate_frame_graph(bboxes, world, object_filter=set(bboxes.keys()))
        graphs[camera] = (set(bboxes.keys()), triplets)
        print(f"  {camera}: {len(bboxes)} objects visible, {len(triplets)} triplets")

    ref_objects, ref_triplets = graphs[GRAPH_CAMERA]
    ref_set = set(ref_triplets)
    all_match = True
    for camera, (objs, triplets) in graphs.items():
        if camera == GRAPH_CAMERA:
            continue
        if objs != ref_objects:
            print(f"  MISMATCH objects {camera}: {objs ^ ref_objects}")
            all_match = False
        if set(triplets) != ref_set:
            print(f"  MISMATCH triplets {camera}")
            all_match = False
    if all_match:
        print(f"  ALL CAMERAS PRODUCE IDENTICAL GRAPH")

    # Extra: check if triplets for the common objects are identical
    for camera, (objs, triplets) in graphs.items():
        if camera == GRAPH_CAMERA:
            continue
        common = ref_objects & objs
        ref_common = {t for t in ref_triplets if t[0] in common and t[2] in common}
        cam_common = {t for t in triplets if t[0] in common and t[2] in common}
        if ref_common == cam_common:
            print(f"  Triplets for common objects: IDENTICAL between {GRAPH_CAMERA} and {camera}")
        else:
            print(f"  Triplets for common objects DIFFER between {GRAPH_CAMERA} and {camera}:")
            for t in sorted(ref_common ^ cam_common):
                print(f"    {t}")

    env.close()


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        for tid in [2, 6]:
            print(f"\n--- Task {tid} consistency check ---")
            check_graph_consistency(tid)
    else:
        for tid in [2, 6]:
            print(f"\n--- Task {tid} ---")
            run_task(tid)
