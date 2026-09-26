#!/usr/bin/env python3
"""
Re-render a specific (demo, frame) from HDF5 states at high resolution.

This keeps the original plot_frame.py --hires CLI contract while sharing the
same simulator renderer used by the unified demo.
"""

import argparse
import json
from pathlib import Path

import h5py

from demo.libero_backend import CAMERA_INFO, SimFrameRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--res", type=int, default=512, help="Square resolution for hi-res render")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--nested-output",
        action="store_true",
        help="Write <out-dir>/<camera>/<frame-stem>_hires.png",
    )
    parser.add_argument("--frame-stem", default=None, help="Filename stem, e.g. frame_000000")
    parser.add_argument("--rotate-agentview", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_stem = args.frame_stem or f"frame_{args.frame:06d}"

    hdf5_path = Path(args.hdf5)
    with h5py.File(hdf5_path, "r") as f:
        states = f[f"data/{args.demo}/states"][:]
        obs = f[f"data/{args.demo}/obs"]
        world_coords = {}
        for camera in CAMERA_INFO:
            world_key = CAMERA_INFO[camera]["bboxes"].removesuffix("_bboxes") + "_world_coords"
            if world_key in obs:
                world_coords[camera] = json.loads(obs[world_key][()].decode("utf-8"))

    if args.frame >= len(states):
        raise ValueError(f"Frame {args.frame} out of range (only {len(states)} states in {args.demo})")

    for camera in CAMERA_INFO:
        renderer = SimFrameRenderer(
            hdf5_path=hdf5_path,
            res=args.res,
            camera=camera,
            rotate_agentview=args.rotate_agentview,
        )
        try:
            fixed_transforms = {
                label: value
                for label, value in world_coords.get(camera, [{}])[args.frame].items()
                if isinstance(value, dict) and ("pos" in value or "mat" in value)
            } if camera in world_coords else None
            img = renderer.render_state(states[args.frame], fixed_transforms)
            if args.nested_output:
                camera_dir = out_dir / camera
                camera_dir.mkdir(parents=True, exist_ok=True)
                out_path = camera_dir / f"{frame_stem}_hires.png"
            else:
                out_path = out_dir / f"{camera}_hires.png"
            img.save(out_path)
            print(f"Saved {out_path}")
        finally:
            renderer.close()


if __name__ == "__main__":
    main()
