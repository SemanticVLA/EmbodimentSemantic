import os
import sys

# Must be set before any robosuite/mujoco import — robosuite defaults to "egl" which is Linux-only
os.environ.setdefault("MUJOCO_GL", "wgl")

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_libero_root = os.path.join(_repo_root, "LIBERO")

# Point LIBERO at our project-local config so it doesn't use ~/.libero (which may point elsewhere)
os.environ["LIBERO_CONFIG_PATH"] = os.path.join(_repo_root, ".libero")

# Insert at 0 so our LIBERO clone wins over any other libero on sys.path/PYTHONPATH
sys.path.insert(0, _repo_root)
sys.path.insert(0, _libero_root)

from radomize_scenes import swap_objects, move_object, settle_physics
from bddl_utils import make_filtered_bddl

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    BENCHMARK_NAME, CAMERA_NAME, CAMERA_HEIGHT, CAMERA_WIDTH,
    SETTLE_STEPS_INIT, SETTLE_STEPS_SWAP, TASK_SWAP_CONFIG,
    TASK_NAMES, OUTPUT_DIR, DEFAULT_CAMERAS,
    TASK_REMOVE_CONFIG, TASK_PROMPT_OVERRIDE, TASK_CAMERA_OVERRIDE,
)


def _get_cameras_for_task(task_id: int) -> list[str]:
    """Return after-side camera list: override if configured, else DEFAULT_CAMERAS."""
    if task_id in TASK_CAMERA_OVERRIDE:
        return [c.strip() for c in TASK_CAMERA_OVERRIDE[task_id].split(",") if c.strip()]
    return DEFAULT_CAMERAS


def _render(env, cameras: list[str]) -> list:
    """Render all cameras and return list of (camera_name, image) tuples."""
    return [
        (cam, env.sim.render(camera_name=cam, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)[::-1])
        for cam in cameras
    ]


def _make_env(bddl_path: str) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
    )
    env.reset()
    settle_physics(env, max_steps=SETTLE_STEPS_INIT)
    return env


def run_task_demo(task_id: int, output_dir: str = OUTPUT_DIR):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[BENCHMARK_NAME]()
    task = task_suite.get_task(task_id)

    base_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    remove_objects = TASK_REMOVE_CONFIG.get(task_id, [])
    cameras = _get_cameras_for_task(task_id)
    swaps = TASK_SWAP_CONFIG.get(task_id, [])

    if task_id in TASK_CAMERA_OVERRIDE:
        print(f"  [CAMERA] Override: {DEFAULT_CAMERAS} → {cameras}")

    # --- Before: always original BDDL, lerobot default cameras, no swaps ---
    env_before = _make_env(base_bddl)
    frames_before = _render(env_before, DEFAULT_CAMERAS)
    env_before.close()

    # --- After: filtered BDDL (if any) + swaps applied ---
    if remove_objects:
        filtered_bddl = make_filtered_bddl(base_bddl, remove_objects)
        print(f"  [REMOVE] Removed {remove_objects} from BDDL")
    else:
        filtered_bddl = base_bddl

    env_after = _make_env(filtered_bddl)
    for obj_a, obj_b in swaps:
        if isinstance(obj_b, str):
            swap_objects(env_after, obj_a, obj_b, settle_steps=SETTLE_STEPS_SWAP, verbose=False)
        else:
            move_object(env_after, obj_a, obj_b, settle_steps=SETTLE_STEPS_SWAP, verbose=False)
    frames_after = _render(env_after, cameras)
    env_after.close()

    # --- Prompt ---
    base_prompt = TASK_NAMES.get(task_id, f"task_{task_id}")
    final_prompt = TASK_PROMPT_OVERRIDE.get(task_id, base_prompt)
    prompt_changed = task_id in TASK_PROMPT_OVERRIDE
    if prompt_changed:
        print(f"  [PROMPT] Override: {final_prompt!r}")
    else:
        print(f"  [PROMPT] Default:  {final_prompt!r}")

    # --- Build figure ---
    # Before: one row per DEFAULT_CAMERAS (agentview + robot0_eye_in_hand).
    # After:  one row per cameras (TASK_CAMERA_OVERRIDE if set, else DEFAULT_CAMERAS).
    # Rows are paired by index; if counts differ each side fills its own rows.
    n_before = len(frames_before)
    n_after  = len(frames_after)
    n_rows   = max(n_before, n_after)
    has_prompt_banner = prompt_changed or bool(remove_objects)

    fig = plt.figure(figsize=(10, n_rows * 4 + (1.2 if has_prompt_banner else 0)))
    gs = gridspec.GridSpec(
        n_rows + (1 if has_prompt_banner else 0),
        2,
        height_ratios=([0.3] + [1] * n_rows if has_prompt_banner else [1] * n_rows),
        hspace=0.4,
    )

    row_offset = 0
    if has_prompt_banner:
        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        info_lines = []
        if remove_objects:
            info_lines.append(f"REMOVED: {', '.join(remove_objects)}")
        if prompt_changed:
            info_lines.append(f"PROMPT: {final_prompt}")
        ax_text.text(0.01, 0.5, "\n".join(info_lines), transform=ax_text.transAxes,
                     fontsize=8, va="center", wrap=True,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffc107"))
        row_offset = 1

    before_label = "Before" + (" Swap" if swaps else "") + (" / Remove" if remove_objects else "")
    after_label  = "After"  + (" Swap" if swaps else "") + (" / Remove" if remove_objects else "")

    for row_idx in range(n_rows):
        row = row_idx + row_offset

        if row_idx < n_before:
            cam_name_b, img_b = frames_before[row_idx]
            ax_b = fig.add_subplot(gs[row, 0])
            ax_b.imshow(img_b)
            ax_b.set_title(f"{before_label}\n[{cam_name_b}]", fontsize=8)
            ax_b.axis("off")

        if row_idx < n_after:
            cam_name_a, img_a = frames_after[row_idx]
            ax_a = fig.add_subplot(gs[row, 1])
            ax_a.imshow(img_a)
            ax_a.set_title(f"{after_label}\n[{cam_name_a}]", fontsize=8)
            ax_a.axis("off")

    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    fig.suptitle(f"Task {task_id}: {task_name}", fontsize=9, y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"task_{task_id:02d}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    import argparse

    all_tasks = sorted(set(
        list(TASK_SWAP_CONFIG.keys())
        + list(TASK_REMOVE_CONFIG.keys())
        + list(TASK_PROMPT_OVERRIDE.keys())
        + list(TASK_CAMERA_OVERRIDE.keys())
    ))
    if not all_tasks:
        all_tasks = list(range(10))

    parser = argparse.ArgumentParser(description="Visualize all LIBERO randomizations for a task.")
    parser.add_argument(
        "tasks",
        nargs="*",
        type=int,
        metavar="TASK_ID",
        help=f"Task IDs to run (default: all configured). Configured: {all_tasks}",
    )
    args = parser.parse_args()

    task_ids = args.tasks if args.tasks else all_tasks

    for t_id in task_ids:
        print(f"\n--- Task {t_id} ---")
        run_task_demo(t_id)
