import json
import hashlib
import shutil
import pytest

import h5py
import numpy as np

from samgraph_benchmark.ground_truth import load_ground_truth
from samgraph_benchmark.metrics import evaluate_predictions
from samgraph_benchmark.tuning import (
    config_hash,
    evaluate_cached_geometry,
    evaluate_cached_geometry_leave_one_task_out,
    load_candidate_configs,
    validate_geometry_candidate,
)
from samgraph_core.geometric_graph import build_directed_relations


def test_candidate_grid_is_explicit_and_bounded(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"grid": {
        "profile": ["samgraph_spatial_mask_geometry"],
        "direction_vertical_scale": [1.0, 2.0],
    }}))
    candidates = load_candidate_configs(path)
    assert len(candidates) == 2
    assert {candidate["profile"] for candidate in candidates} == {"samgraph_spatial_mask_geometry"}


def test_notebook_proxy_candidate_parses_typed_fields_and_requires_profile(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{
        "profile": "samgraph_spatial_mask_geometry",
        "bbox_padding_x": "0",
        "bbox_padding_y": 0,
        "containment_iomin_threshold": "0.8",
        "direction_horizontal_sign": "1",
        "direction_vertical_scale": "2",
        "stack_order": "lower_vertical_is_top",
    }]))
    candidate = load_candidate_configs(path)[0]
    assert candidate["bbox_padding_x"] == "0"
    assert validate_geometry_candidate(candidate)["profile"] == "samgraph_spatial_mask_geometry"
    with pytest.raises(ValueError, match="require profile"):
        validate_geometry_candidate({"bbox_padding_x": 5})


def test_notebook_proxy_variants_are_not_selectable():
    with pytest.raises(ValueError, match="unsupported geometry profile"):
        validate_geometry_candidate({"profile": "notebook_mask_proxy_v1"})
    with pytest.raises(ValueError, match="unsupported geometry profile"):
        validate_geometry_candidate({"profile": "notebook_mask_proxy_v2"})
    with pytest.raises(ValueError, match="unsupported"):
        validate_geometry_candidate({
            "profile": "samgraph_spatial_mask_geometry",
            "multiple_support_policy": "reject",
        })


@pytest.mark.parametrize("key", [
    "hull_coverage_min",
    "hull_coverage_near_min",
    "near_distance_max_normalized",
    "cookies_reverse_coverage_min",
    "cabinet_top_layer_quantile",
])
def test_ignored_legacy_geometry_keys_are_rejected(key):
    with pytest.raises(ValueError, match="ignored legacy geometry keys"):
        validate_geometry_candidate({
            "profile": "samgraph_spatial_mask_geometry",
            key: 0.5,
        })


def test_vertical_scale_legacy_alias_remains_an_active_proxy_override():
    candidate = validate_geometry_candidate({
        "profile": "samgraph_spatial_mask_geometry",
        "vertical_scale": 1.25,
    })
    assert candidate["vertical_scale"] == 1.25
    assert candidate["direction_vertical_scale"] == 1.25


@pytest.mark.parametrize("key", [
    "padding_x",
    "padding_x_px",
    "padding_y",
    "padding_y_px",
    "iomin_threshold",
    "horizontal_axis",
    "horizontal_sign",
    "vertical_axis",
    "vertical_sign",
    "scale",
    "tie_axis",
])
def test_noncanonical_samgraph_aliases_are_rejected_before_merge(key):
    with pytest.raises(ValueError, match="non-canonical SamGraph geometry aliases"):
        validate_geometry_candidate({
            "profile": "samgraph_spatial_mask_geometry",
            key: 1,
        })


def test_non_replayable_quality_candidate_is_rejected(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"black_bowl_rgb_std_min": 8.0}]))
    with pytest.raises(ValueError, match="non-replayable"):
        load_candidate_configs(path)


def test_geometry_tuning_reloads_cached_masks_without_sam(tmp_path):
    masks = {
        "black_bowl_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (0, 4))),
        "plate_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (4, 0))),
    }
    mask_path = tmp_path / "masks.npz"
    np.savez_compressed(mask_path, **masks)
    instances = [
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
    ]
    _relations, triplets, _evidence = build_directed_relations(
        instances, masks, suite="spatial", instruction="task", shape=(4, 6)
    )
    h5_path = tmp_path / "task_demo.hdf5"
    with h5py.File(h5_path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        frame_triplets = [[item["subject"], item["relation"], item["object"]] for item in triplets]
        obs.create_dataset("agentview_scene_graph", data=json.dumps([frame_triplets, frame_triplets]).encode())
    run = tmp_path / "run"
    run.mkdir()
    portable_masks = run / "masks" / "task" / "000000.npz"
    portable_masks.parent.mkdir(parents=True)
    shutil.copy2(mask_path, portable_masks)
    prediction_path = run / "predictions.jsonl"
    prediction_path.write_text(json.dumps({"task": "task", "demo": "demo_0", "frame": 0,
                                            "mask_cache": "masks/task/000000.npz",
                                            "mask_cache_sha256": hashlib.sha256(portable_masks.read_bytes()).hexdigest()}) + "\n")
    copied = tmp_path / "copied_run"
    shutil.copytree(run, copied)
    result = evaluate_cached_geometry(
        load_ground_truth(h5_path), copied / "predictions.jsonl", [{}, {}], frame_stride=2
    )
    assert result["exploratory"] is True
    assert result["results"][0]["frames_with_cached_masks"] == 1
    assert result["results"][0]["primary"]["f1"] == 1.0
    assert result["results"][0]["config_hash"]
    assert result["frame_stride"] == 2
    assert result["results"][0]["frames_expected"] == 1
    assert result["candidate_count"] == 1
    provenance = result["input_provenance"]
    assert len(provenance["predictions_jsonl_sha256"]) == 64
    assert provenance["expected_keys"] == [["task", "demo_0", 0]]
    assert provenance["cached_keys"] == [["task", "demo_0", 0]]
    assert provenance["cache_file_count"] == 1
    assert len(provenance["cache_digest"]) == 64
    assert provenance["cache_entries"][0]["key"] == ["task", "demo_0", 0]
    assert result["results"][0]["input_provenance_sha256"] == provenance["cache_digest"]


def test_parallel_cached_tuning_matches_serial_and_preserves_hash_order(tmp_path):
    """Candidate-parallel replay must be a deterministic serial equivalent."""
    masks = {
        "black_bowl_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (0, 4))),
        "plate_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (4, 0))),
    }
    mask_path = tmp_path / "masks.npz"
    np.savez_compressed(mask_path, **masks)
    instances = [
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
    ]
    _relations, triplets, _evidence = build_directed_relations(
        instances, masks, suite="spatial", instruction="task", shape=(4, 6)
    )
    h5_path = tmp_path / "task_demo.hdf5"
    with h5py.File(h5_path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        frame_triplets = [[item["subject"], item["relation"], item["object"]] for item in triplets]
        obs.create_dataset(
            "agentview_scene_graph",
            data=json.dumps([frame_triplets]).encode(),
        )

    run = tmp_path / "run"
    portable_masks = run / "masks" / "task" / "000000.npz"
    portable_masks.parent.mkdir(parents=True)
    shutil.copy2(mask_path, portable_masks)
    prediction_path = run / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({
            "task": "task",
            "demo": "demo_0",
            "frame": 0,
            "mask_cache": "masks/task/000000.npz",
            "mask_cache_sha256": hashlib.sha256(portable_masks.read_bytes()).hexdigest(),
        }) + "\n"
    )

    profile = "samgraph_spatial_mask_geometry"
    candidates = [
        {"profile": profile, "direction_horizontal_sign": 1},
        {"profile": profile, "direction_horizontal_sign": -1},
        {"profile": profile, "direction_vertical_scale": 1.25},
        # Duplicate hashes are intentionally present to exercise de-duplication.
        {"profile": profile, "direction_vertical_scale": 1.25},
    ]
    ground_truth = load_ground_truth(h5_path)
    serial = evaluate_cached_geometry(ground_truth, prediction_path, candidates)
    parallel = evaluate_cached_geometry(ground_truth, prediction_path, candidates, workers=2)
    parallel_reversed = evaluate_cached_geometry(
        ground_truth, prediction_path, list(reversed(candidates)), workers=2
    )

    assert serial["execution"] == {
        "backend": "serial", "workers": 1, "requested_workers": 1,
    }
    assert parallel["execution"] == {
        "backend": "thread_pool", "workers": 2, "requested_workers": 2,
    }
    assert serial["candidate_count"] == parallel["candidate_count"] == 3
    assert serial["results"] == parallel["results"]
    assert parallel["results"] == parallel_reversed["results"]
    expected_hashes = {
        config_hash(validate_geometry_candidate(candidate)) for candidate in candidates
    }
    assert {item["config_hash"] for item in serial["results"]} == expected_hashes
    for item in parallel["results"]:
        assert item["frames_expected"] == 1
        assert item["frames_with_cached_masks"] == 1
        assert item["frames_scored_from_masks"] == 1
        assert item["frames_missing_mask_cache"] == 0
        assert item["mask_coverage"] == 1.0
        assert item["geometry_errors"] == 0


@pytest.mark.parametrize("workers", [0, -1, 33, True, "2"])
def test_cached_tuning_rejects_invalid_worker_counts(tmp_path, workers):
    from samgraph_benchmark.ground_truth import GroundTruthIndex

    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text("{}\n")
    ground_truth = GroundTruthIndex({("task", "demo_0", 0): frozenset()})
    with pytest.raises(ValueError, match="workers"):
        evaluate_cached_geometry(ground_truth, prediction_path, [{}], workers=workers)


def test_single_cached_mask_matches_live_empty_triplet_semantics(tmp_path):
    """A one-mask frame is a valid empty graph, not an offline geometry error."""
    two_masks = {
        "black_bowl_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (0, 4))),
        "plate_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (4, 0))),
    }
    one_mask = {"black_bowl_1": two_masks["black_bowl_1"]}
    _relations, triplets, _evidence = build_directed_relations(
        [
            {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
            {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
        ],
        two_masks,
        suite="spatial",
        instruction="task",
        shape=(4, 6),
    )
    frame_triplets = [[item["subject"], item["relation"], item["object"]] for item in triplets]
    h5_path = tmp_path / "task_demo.hdf5"
    with h5py.File(h5_path, "w") as handle:
        obs = handle.create_group("data/demo_0/obs")
        # With stride two, frame 0 exercises the one-mask path and frame 2
        # exercises the ordinary two-mask geometry path.
        obs.create_dataset(
            "agentview_scene_graph",
            data=json.dumps([[], frame_triplets, frame_triplets]).encode(),
        )

    run = tmp_path / "run"
    (run / "masks" / "task").mkdir(parents=True)
    one_path = run / "masks" / "task" / "000000.npz"
    two_path = run / "masks" / "task" / "000002.npz"
    np.savez_compressed(one_path, **one_mask)
    np.savez_compressed(two_path, **two_masks)
    prediction_path = run / "predictions.jsonl"
    rows = []
    for frame, path in ((0, one_path), (2, two_path)):
        rows.append({
            "task": "task",
            "demo": "demo_0",
            "frame": frame,
            "mask_cache": str(path.relative_to(run)).replace("\\", "/"),
            "mask_cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    prediction_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    ground_truth = load_ground_truth(h5_path)
    result = evaluate_cached_geometry(ground_truth, prediction_path, [{}], frame_stride=2)
    offline = result["results"][0]
    assert offline["geometry_errors"] == 0
    assert offline["frames_scored_from_masks"] == 2
    assert offline["frames_missing_mask_cache"] == 0

    # This is the live/default comparison: one-mask frame emits no triplets,
    # while the two-mask frame uses the ordinary geometric prediction.
    live = evaluate_predictions(
        ground_truth,
        {
            ("task", "demo_0", 0): [],
            ("task", "demo_0", 2): frame_triplets,
        },
        frame_stride=2,
    )
    assert offline["primary"]["f1"] == live.primary["f1"]


def test_declared_cache_missing_and_hash_mismatch_fail_closed(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(json.dumps({"task": "task", "demo": "demo_0", "frame": 0,
                                            "mask_cache": "missing.npz", "mask_cache_sha256": "bad"}) + "\n")
    from samgraph_benchmark.ground_truth import GroundTruthIndex
    ground_truth = GroundTruthIndex({("task", "demo_0", 0): frozenset()})
    with pytest.raises(FileNotFoundError, match="missing"):
        evaluate_cached_geometry(ground_truth, prediction_path, [{}])
    cache = tmp_path / "cache.npz"
    np.savez_compressed(cache, black_bowl_1=np.ones((2, 2), dtype=bool), plate_1=np.ones((2, 2), dtype=bool))
    prediction_path.write_text(json.dumps({"task": "task", "demo": "demo_0", "frame": 0,
                                            "mask_cache": str(cache), "mask_cache_sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_cached_geometry(ground_truth, prediction_path, [{}])


def test_zero_mask_coverage_is_an_error(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(json.dumps({"task": "task", "demo": "demo_0", "frame": 0,
                                            "triplets": [], "mask_cache": None}) + "\n")
    from samgraph_benchmark.ground_truth import GroundTruthIndex
    ground_truth = GroundTruthIndex({("task", "demo_0", 0): frozenset()})
    with pytest.raises(ValueError, match="zero cached-mask coverage"):
        evaluate_cached_geometry(ground_truth, prediction_path, [{}])


def test_affine_control_freezes_the_05491_winner_non_direction_contract():
    """The affine control keeps the established 0.5491 winner's fixed rules."""
    from samgraph_core.geometric_graph import geometric_relation_rules

    control = validate_geometry_candidate({
        "profile": "samgraph_spatial_mask_geometry",
        "containment_iomin_threshold": 0.5,
        "direction_horizontal_axis": "x",
        "direction_vertical_axis": "y",
        "direction_horizontal_sign": 1,
        "direction_vertical_sign": -1,
        "direction_vertical_scale": 2.0,
        "direction_tie_axis": "vertical",
        "stack_order": "lower_vertical_is_top",
        "drawer_stack_order": "lower_vertical_is_inside",
        "drawer_mode": "instruction_overlap_cabinet_band",
        "direction_front_coefficients": [0.0, -2.0],
        "direction_left_coefficients": [1.0, 0.0],
    })
    resolved = geometric_relation_rules(control)
    assert resolved["containment"]["threshold"] == 0.5
    assert resolved["coordinate_adjustment"] | {
        "front_score_coefficients": [0.0, -2.0],
        "left_score_coefficients": [1.0, 0.0],
    } == resolved["coordinate_adjustment"]
    assert {
        key: value
        for key, value in resolved["coordinate_adjustment"].items()
        if key not in {"front_score_coefficients", "left_score_coefficients"}
    } == {
        "direction_coordinates": "notebook_mask_proxy_pixels",
        "dominant_axis_tie": "vertical",
        "horizontal_axis": "x",
        "horizontal_sign": 1,
        "vertical_axis": "y",
        "vertical_scale": 2.0,
        "vertical_sign": -1,
    }
    assert resolved["stack_order"] == {
        "name": "lower_vertical_is_top",
        "drawer_name": "lower_vertical_is_inside",
        "drawer_mode": "instruction_overlap_cabinet_band",
    }


def test_affine_candidates_require_both_rows_and_accept_nested_contract():
    with pytest.raises(ValueError, match="supplied together"):
        validate_geometry_candidate({
            "profile": "samgraph_spatial_mask_geometry",
            "direction_front_coefficients": [0.0, -2.0],
        })
    candidate = validate_geometry_candidate({
        "profile": "samgraph_spatial_mask_geometry",
        "coordinate_adjustment": {
            "direction_front_coefficients": [0.0, -2.0],
            "direction_left_coefficients": [1.0, 0.0],
        },
    })
    assert candidate["coordinate_adjustment"]["direction_front_coefficients"] == [0.0, -2.0]


def test_leave_one_task_out_replays_masks_without_label_inference(tmp_path):
    """CV selects on one task and scores the other using the same cache contract."""
    masks = {
        "black_bowl_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (0, 4))),
        "plate_1": np.pad(np.ones((2, 2), dtype=bool), ((1, 1), (4, 0))),
    }
    run = tmp_path / "run"
    cache = run / "masks"
    cache.mkdir(parents=True)
    mask_path = cache / "shared.npz"
    np.savez_compressed(mask_path, **masks)
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    rows = []
    hdf5_paths = []
    for task in ("task_a", "task_b"):
        hdf5_path = tmp_path / f"{task}_demo.hdf5"
        hdf5_paths.append(hdf5_path)
        with h5py.File(hdf5_path, "w") as handle:
            obs = handle.create_group("data/demo_0/obs")
            obs.create_dataset("agentview_scene_graph", data=json.dumps([[]]).encode())
        rows.append(json.dumps({
            "task": task,
            "demo": "demo_0",
            "frame": 0,
            "mask_cache": "masks/shared.npz",
            "mask_cache_sha256": digest,
        }))
    prediction_path = run / "predictions.jsonl"
    prediction_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ground_truth = load_ground_truth(hdf5_paths)
    result = evaluate_cached_geometry_leave_one_task_out(
        ground_truth,
        prediction_path,
        [{
            "profile": "samgraph_spatial_mask_geometry",
            "direction_front_coefficients": [0.0, -2.0],
            "direction_left_coefficients": [1.0, 0.0],
        }],
        frame_stride=1,
    )
    assert result["schema"] == "samgraph.geometry_tuning.leave_one_task_out.v1"
    assert result["resubstitution"] is False
    assert result["candidate_count"] == 1
    assert {fold["held_out_task"] for fold in result["folds"]} == {"task_a", "task_b"}
    assert all(fold["held_out"]["frames_expected"] == 1 for fold in result["folds"])
    assert result["selection"]["labels_used_for_inference"] is False
