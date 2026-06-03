#!/usr/bin/env python3
"""
Re-render a specific (demo, frame) from the HDF5 states at high resolution.
Run this under the vla_bench_py312 conda env (the one that has libero).

Called by plot_frame.py --hires, but can also be used standalone:
    conda run -n vla_bench_py312 python render_hires.py \
        --hdf5 data/libero_spatial_v5/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
        --demo demo_0 --frame 0 --res 512 --out-dir figures/my_output

For plot_frame.py's nested layout, add:
        --nested-output --frame-stem frame_000000
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv


CAMERA_MAP = {
    "agentview": "agentview",
    "eye_in_hand": "robot0_eye_in_hand",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--res", type=int, default=512, help="Square resolution for hi-res render")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nested-output", action="store_true",
                        help="Write <out-dir>/<camera>/<frame-stem>_hires.png")
    parser.add_argument("--frame-stem", default=None,
                        help="Filename stem for nested output, e.g. frame_000000")
    parser.add_argument("--rotate-agentview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_stem = args.frame_stem or f"frame_{args.frame:06d}"

    hdf5_path = Path(args.hdf5)
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        states = f[f"data/{args.demo}/states"][:]

    if args.frame >= len(states):
        raise ValueError(f"Frame {args.frame} out of range (only {len(states)} states in {args.demo})")

    state = states[args.frame]

    # Resolve the bddl file from libero's actual install path.
    # The path stored in HDF5 env_args is from the original collection machine — don't trust it.
    from libero.libero import get_libero_path
    task_name = Path(args.hdf5).stem.replace("_demo", "")
    bddl_dir = Path(get_libero_path("bddl_files")) / "libero_spatial"
    bddl_file = bddl_dir / f"{task_name}.bddl"
    if not bddl_file.exists():
        # fallback: search all bddl subdirs
        matches = list(Path(get_libero_path("bddl_files")).rglob(f"{task_name}.bddl"))
        if not matches:
            raise FileNotFoundError(f"Cannot find bddl file for task '{task_name}'")
        bddl_file = matches[0]
    print(f"Using bddl: {bddl_file}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.res,
        camera_widths=args.res,
    )

    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    for cam_key, cam_sim_name in CAMERA_MAP.items():
        # robosuite render returns (H, W, 3) uint8, flipped vertically (OpenGL convention)
        img_arr = env.sim.render(
            camera_name=cam_sim_name,
            height=args.res,
            width=args.res,
            depth=False,
        )
        # eye_in_hand: original HDF5 stores raw OpenGL output — no flip needed
        # agentview: original HDF5 stores raw OpenGL output too, but the eval pipeline
        #   applies rot90(2) (180° rotation) via frame_to_pil when rotate_agentview=True.
        #   rot90(2) on the raw upside-down OpenGL image = correct orientation for agentview.
        if args.rotate_agentview and cam_key == "agentview":
            img_arr = np.rot90(img_arr, 2).copy()
        # else: keep raw — matches original HDF5 storage

        img = Image.fromarray(img_arr.astype(np.uint8))
        if args.nested_output:
            cam_dir = out_dir / cam_key
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_path = cam_dir / f"{frame_stem}_hires.png"
        else:
            out_path = out_dir / f"{cam_key}_hires.png"
        img.save(out_path)
        print(f"Saved {out_path}")

    env.close()


if __name__ == "__main__":
    main()
