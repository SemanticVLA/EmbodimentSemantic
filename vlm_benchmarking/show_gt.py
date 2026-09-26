import argparse
import sys
from pathlib import Path

from vlm_bench.eval import _load_gt

def main():
    parser = argparse.ArgumentParser(description="Show the ground truth scene graph for a specific task, demo, and frame.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., pick_up_the_black_bowl_between...)")
    parser.add_argument("--demo", type=str, required=True, help="Demo identifier (e.g., demo_0)")
    parser.add_argument("--frame", type=int, required=True, help="Frame index (e.g., 10)")
    parser.add_argument("--camera", type=str, default="agentview", help="Camera view (e.g., agentview or eye_in_hand)")
    parser.add_argument("--input-dir", type=str, default="data/libero_spatial_v5/", help="Directory containing HDF5 files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    print(f"\n=== Ground Truth Graph ===")
    print(f"Task:  {args.task}")
    print(f"Demo:  {args.demo}")
    print(f"Frame: {args.frame}")
    print(f"Camera: {args.camera}")
    print("-" * 60)
    
    try:
        hdf5_path = input_dir / f"{args.task}_demo.hdf5"
        if not hdf5_path.exists():
            matches = [p for p in input_dir.glob("*.hdf5") if args.task in p.name]
            if matches:
                hdf5_path = matches[0]
            else:
                print(f"Error: Could not find HDF5 file for task '{args.task}' in {input_dir}")
                sys.exit(1)

        gt_triplets = sorted(list(_load_gt(hdf5_path, args.demo, args.frame, args.camera)))
        
        if not gt_triplets:
            print("No ground truth triplets found for this frame.")
        else:
            for objA, rel, objB in gt_triplets:
                print(f"{objA}  |  {rel}  |  {objB}")
            
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()