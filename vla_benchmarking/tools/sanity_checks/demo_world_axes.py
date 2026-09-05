"""Visualize world axes projected onto the raw model-view image to verify scene graph directions."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "wgl")

_vla_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_repo_root = os.path.dirname(_vla_root)
_libero_root = os.path.join(_vla_root, "LIBERO")
os.environ["LIBERO_CONFIG_PATH"] = os.path.join(_repo_root, ".libero")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _libero_root)
sys.path.insert(0, _vla_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from robosuite.utils import camera_utils as CU

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla_benchmarking.evaluation.libero_live_semantic_context import world_to_pixel, discover_objects
from vla_benchmarking.evaluation.randomize_scenes import settle_physics, get_object_pose
from vla_benchmarking.shared.config import (
    BENCHMARK_NAME, CAMERA_HEIGHT, CAMERA_WIDTH, SETTLE_STEPS_INIT,
)

OUT_DIR = os.path.join(_vla_root, "swap_outputs")
CAMERA = "agentview"
AXIS_LEN = 0.15  # meters


def draw_axes(ax, env, camera, img_h, img_w):
    K = CU.get_camera_intrinsic_matrix(env.sim, camera, img_h, img_w)
    ext_inv = np.linalg.inv(CU.get_camera_extrinsic_matrix(env.sim, camera))

    # Use scene center as origin (mean of all object positions)
    objects = discover_objects(env)
    positions = []
    for label, body in objects.items():
        bid = env.sim.model.body_name2id(body)
        positions.append(env.sim.data.body_xpos[bid].copy())
    origin = np.mean(positions, axis=0)

    axes_def = [
        (np.array([1, 0, 0]), "red",   "+X (front/behind)"),
        (np.array([0, 1, 0]), "green", "+Y (left/right)"),
        (np.array([0, 0, 1]), "blue",  "+Z (up)"),
    ]

    origin_px = world_to_pixel(origin, ext_inv, K, img_h, img_w)
    if origin_px is None:
        print("  WARNING: scene center not visible")
        return

    ox, oy = origin_px
    ax.plot(ox, oy, "wo", markersize=6)

    for direction, color, label in axes_def:
        tip = origin + direction * AXIS_LEN
        tip_px = world_to_pixel(tip, ext_inv, K, img_h, img_w)
        if tip_px is None:
            continue
        tx, ty = tip_px
        ax.annotate("", xy=(tx, ty), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        ax.text(tx + 4, ty, label, color=color, fontsize=7,
                bbox=dict(fc="black", alpha=0.5, pad=1, boxstyle="round,pad=0.1"))

    # Also label each object with its world pos
    for label, body in objects.items():
        bid = env.sim.model.body_name2id(body)
        pos = env.sim.data.body_xpos[bid]
        px = world_to_pixel(pos, ext_inv, K, img_h, img_w)
        if px is None:
            continue
        ax.plot(px[0], px[1], "w+", markersize=8, markeredgewidth=1.5)
        ax.text(px[0] + 4, px[1], f"{label}\n{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}",
                fontsize=5, color="white",
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

    img_raw = env.sim.render(camera_name=CAMERA, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_raw)
    ax.set_title(f"Task {task_id} | {CAMERA} | RAW (model view) + world axes", fontsize=8)
    ax.axis("off")
    draw_axes(ax, env, CAMERA, CAMERA_HEIGHT, CAMERA_WIDTH)

    fig.suptitle(f"Task {task_id}: {task.language}\n"
                 f"Red=+X(front/behind)  Green=+Y(left/right)  Blue=+Z(up)", fontsize=8)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"task_{task_id:02d}_world_axes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    for tid in [2, 6]:
        print(f"\n--- Task {tid} ---")
        run_task(tid)
