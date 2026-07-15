from __future__ import annotations

import argparse
from pathlib import Path

import h5py


DEMO_ID = "demo_0"
RGB_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
SEMANTIC_KEYS = (
    "agentview_bboxes",
    "agentview_scene_graph",
    "agentview_world_coords",
    "robot0_eye_in_hand_bboxes",
    "robot0_eye_in_hand_scene_graph",
    "robot0_eye_in_hand_world_coords",
)


def build(source_dir: Path, output_dir: Path) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"LIBERO HDF5 directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected = {path.name for path in source_dir.glob("*.hdf5")}
    if not expected:
        raise FileNotFoundError(f"No HDF5 files found in {source_dir}")
    for stale in output_dir.glob("*.hdf5"):
        if stale.name not in expected:
            stale.unlink()

    for source_path in sorted(source_dir.glob("*.hdf5")):
        destination = output_dir / source_path.name
        temporary = destination.with_suffix(".hdf5.tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(source_path, "r") as source:
            source_demo_path = f"data/{DEMO_ID}"
            if source_demo_path not in source:
                raise KeyError(f"{source_path.name} does not contain {source_demo_path}")
            source_demo = source[source_demo_path]
            source_obs = source_demo["obs"]
            with h5py.File(temporary, "w") as target:
                target_data = target.create_group("data")
                for key, value in source["data"].attrs.items():
                    target_data.attrs[key] = value
                target_demo = target_data.create_group(DEMO_ID)
                for key, value in source_demo.attrs.items():
                    target_demo.attrs[key] = value
                source.copy(source_demo["states"], target_demo, name="states")
                target_obs = target_demo.create_group("obs")
                for key in SEMANTIC_KEYS:
                    source.copy(source_obs[key], target_obs, name=key)
                for key in RGB_KEYS:
                    rgb = source_obs[key]
                    # The backend only reads this dataset's shape. No image chunks
                    # are allocated; MuJoCo renders pixels from the stored states.
                    target_obs.create_dataset(key, shape=rgb.shape, dtype=rgb.dtype)
        temporary.replace(destination)
        print(f"Wrote {destination.name} ({destination.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache LIBERO demo_0 state and semantic data without RGB frames."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
