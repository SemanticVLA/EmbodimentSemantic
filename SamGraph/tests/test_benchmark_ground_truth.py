import json

import h5py

from samgraph_benchmark.ground_truth import load_ground_truth


def test_gt_parser_reads_only_eval_graph_dataset(tmp_path):
    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        payload = [["black_bowl_1", "is_left_of", "plate_1"]]
        obs.create_dataset("agentview_scene_graph", data=json.dumps([payload]).encode("utf-8"))
        obs.create_dataset("agentview_rgb", data=0, dtype="u1")
    index = load_ground_truth(path)
    assert index.get("task", "demo_0", 0) == frozenset({("akita_black_bowl_1", "is_left_of", "plate_1")})
