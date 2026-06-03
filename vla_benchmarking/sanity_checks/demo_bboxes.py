"""Visualize bounding boxes on rendered frames for tasks 2 and 6, episode 0, step 0."""
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
from libero_live_semantic_context import get_bbox, _camera_name_for_mujoco
from radomize_scenes import settle_physics, discover_objects
from config import (
    BENCHMARK_NAME, CAMERA_HEIGHT, CAMERA_WIDTH, SETTLE_STEPS_INIT,
    TASK_CAMERA_OVERRIDE,
)

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]

OUT_DIR = os.path.join(_repo_root, "dataset_eval", "swap_outputs")


def get_bboxes_for_camera(env, camera, img_h, img_w):
    """Return {label: (x1,y1,x2,y2)} for all objects in the scene."""
    objects = discover_objects(env)
    K = CU.get_camera_intrinsic_matrix(env.sim, camera, img_h, img_w)
    ext_inv = np.linalg.inv(CU.get_camera_extrinsic_matrix(env.sim, camera))
    result = {}
    for label, body in objects.items():
        bbox = get_bbox(env, body, ext_inv, K, img_h, img_w)
        if bbox is not None:
            result[label] = bbox
    return result


def draw_bboxes(ax, img, bboxes, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    for i, (label, (x1, y1, x2, y2)) in enumerate(bboxes.items()):
        color = COLORS[i % len(COLORS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 3, 0), label,
                fontsize=6, color=color,
                bbox=dict(fc="black", alpha=0.5, pad=1, boxstyle="round,pad=0.1"))


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

    override_str = TASK_CAMERA_OVERRIDE.get(task_id, "")
    override_cameras = [c.strip() for c in override_str.split(",") if c.strip()]
    all_cameras = list(dict.fromkeys(["agentview"] + override_cameras))

    n_cams = len(all_cameras)
    fig, axes = plt.subplots(1, n_cams, figsize=(6 * n_cams, 6))
    if n_cams == 1:
        axes = [axes]

    for ax, camera in zip(axes, all_cameras):
        img_raw = env.sim.render(camera_name=camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        bboxes_raw = get_bboxes_for_camera(env, camera, CAMERA_HEIGHT, CAMERA_WIDTH)
        img = img_raw[::-1]
        bboxes = {
            label: (x1, (CAMERA_HEIGHT - 1) - y2, x2, (CAMERA_HEIGHT - 1) - y1)
            for label, (x1, y1, x2, y2) in bboxes_raw.items()
        }
        tag = " [scene-graph cam]" if camera == "agentview" else " [bbox only]"
        draw_bboxes(ax, img, bboxes, f"Task {task_id} | {camera}{tag}")

    fig.suptitle(f"Task {task_id}: {task.language}", fontsize=9)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"task_{task_id:02d}_bboxes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    for tid in [2, 6]:
        print(f"\n--- Task {tid} ---")
        run_task(tid)
