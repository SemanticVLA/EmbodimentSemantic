from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from config import TASK_PROMPT_OVERRIDE
from hdf5_to_lerobot_dataset import (
    FEATURES,
    GRAPH_CONTRACT,
    GRAPH_EXTRACTOR_PATH,
    GRAPH_PAIR_KIND,
    REPO_ROOT,
    SEALED_LORA_IMAGE_SIZE,
    SEALED_LORA_VISUAL_CONTRACT,
    TARGET_ARROW_PAIR_KIND,
    TARGET_ARROW_VARIANT,
    assert_paired_frame_invariants,
    build_frame,
    build_graph_paired_frames,
    build_paired_frames,
    sha256_file,
    expected_drawable_relations,
    filter_by_subject,
    flip180,
    main_image_change_mask,
    resize_rgb_image,
    run_convert_pair,
    run_verify,
    sealed_pair_manifest_path,
    sealed_pair_sentinel_path,
    sealed_target_arrow_pair_manifest_path,
    sealed_target_arrow_pair_sentinel_path,
    scale_and_clamp_bboxes,
    task_text_for,
    validate_verified_pair,
    _load_sealed_manifest,
    _assert_episode_expectations,
    _full_experiment_ready,
)


def _frame_kwargs(**overrides):
    base = dict(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        eye_in_hand_rgb=np.full((128, 128, 3), 7, dtype=np.uint8),
        bboxes={"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [80, 80, 100, 100]},
        relations=[
            ["akita_black_bowl_1", "is_left_of", "plate_1"],
            ["plate_1", "is_right_of", "akita_black_bowl_1"],
        ],
        ee_pos=np.array([0.1, 0.2, 0.3]),
        ee_ori=np.array([1.0, -1.0, 0.5]),
        gripper_states=np.array([0.04, -0.04]),
        action=np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, -1.0]),
        task_text="pick up the black bowl and place it on the plate",
    )
    base.update(overrides)
    return base


def test_task_text_for_uses_every_configured_prompt_override(tmp_path):
    source = tmp_path / "task.hdf5"
    with h5py.File(source, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "canonical task"})

    with h5py.File(source, "r") as hdf5_file:
        for task_id, expected_prompt in TASK_PROMPT_OVERRIDE.items():
            assert task_text_for(task_id, hdf5_file) == expected_prompt


def test_flip180_rotates_both_axes():
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    flipped = flip180(frame)
    assert flipped.shape == frame.shape
    assert np.array_equal(flipped, frame[::-1, ::-1])
    assert np.array_equal(flipped[-1, -1], frame[0, 0])


def test_filter_by_subject_keeps_only_matching_subject():
    relations = [
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("plate_1", "is_right_of", "akita_black_bowl_1"),
        ("cookies_1", "is_on_top_of", "wooden_cabinet_1"),
    ]
    filtered = filter_by_subject(relations, "akita_black_bowl_1")
    assert filtered == [("akita_black_bowl_1", "is_left_of", "plate_1")]


def test_filter_by_subject_none_keeps_everything():
    relations = [("a", "r", "b"), ("b", "r", "a")]
    assert filter_by_subject(relations, None) == [("a", "r", "b"), ("b", "r", "a")]


def test_build_frame_state_vector_is_ee_pos_ori_gripper_concat():
    frame = build_frame(**_frame_kwargs(), variant="control")
    expected = np.array([0.1, 0.2, 0.3, 1.0, -1.0, 0.5, 0.04, -0.04], dtype=np.float32)
    assert frame["observation.state"].dtype == np.float32
    assert np.allclose(frame["observation.state"], expected)
    assert frame["observation.state"].shape == FEATURES["observation.state"]["shape"]


def test_build_frame_action_dtype_and_shape():
    frame = build_frame(**_frame_kwargs(), variant="control")
    assert frame["action"].dtype == np.float32
    assert frame["action"].shape == FEATURES["action"]["shape"]


def test_build_frame_control_never_draws_arrows():
    kwargs = _frame_kwargs(agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8))
    frame = build_frame(**kwargs, variant="control")
    # Control must be resize+flip(raw) and nothing else -- no arrow pixels introduced.
    assert np.array_equal(
        frame["observation.images.image"],
        flip180(resize_rgb_image(kwargs["agentview_rgb"])),
    )


def test_build_frame_seals_both_cameras_to_256_and_scales_clamps_bboxes():
    source = np.zeros((128, 128, 3), dtype=np.uint8)
    bboxes = {"a": [-10, 2, 64, 200], "b": [96, 32, 128, 64]}
    assert scale_and_clamp_bboxes(bboxes) == {
        "a": [0, 4, 128, 255],
        "b": [192, 64, 255, 128],
    }
    frame = build_frame(**_frame_kwargs(agentview_rgb=source, eye_in_hand_rgb=source), variant="control")
    assert frame["observation.images.image"].shape == (SEALED_LORA_IMAGE_SIZE,) * 2 + (3,)
    assert frame["observation.images.image2"].shape == (SEALED_LORA_IMAGE_SIZE,) * 2 + (3,)
    assert FEATURES["observation.images.image"]["dtype"] == "image"
    assert FEATURES["observation.images.image2"]["dtype"] == "image"


def test_build_frame_treatment_draws_arrows_only_for_target_subject():
    kwargs = _frame_kwargs(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        relations=[
            ["akita_black_bowl_1", "is_left_of", "plate_1"],
            ["cookies_1", "is_on_top_of", "plate_1"],  # wrong subject, must be dropped
        ],
    )
    control = build_frame(**kwargs, variant="control")
    treatment = build_frame(**kwargs, variant="treatment")

    assert not np.array_equal(control["observation.images.image"], treatment["observation.images.image"])
    # Everything except the main image must stay identical between variants.
    assert np.array_equal(
        control["observation.images.image2"], treatment["observation.images.image2"]
    )
    assert np.allclose(control["observation.state"], treatment["observation.state"])
    assert np.allclose(control["action"], treatment["action"])
    assert control["task"] == treatment["task"]


def test_target_arrow_treatment_draws_exactly_one_bowl_to_goal_arrow():
    kwargs = _frame_kwargs(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        relations=[
            ["akita_black_bowl_1", "is_left_of", "plate_1"],
            ["akita_black_bowl_1", "is_behind", "cookies_1"],
            ["cookies_1", "is_on_top_of", "plate_1"],
        ],
        bboxes={
            "akita_black_bowl_1": [10, 10, 20, 20],
            "cookies_1": [40, 40, 50, 50],
            "plate_1": [80, 80, 100, 100],
        },
    )
    control = build_frame(**kwargs, variant="control")
    target = build_frame(**kwargs, variant=TARGET_ARROW_VARIANT)

    assert expected_drawable_relations(
        kwargs,
        treatment_variant=TARGET_ARROW_VARIANT,
        goal_object="plate_1",
    ) == [("akita_black_bowl_1", "goal", "plate_1")]
    assert not np.array_equal(
        control["observation.images.image"], target["observation.images.image"]
    )
    assert np.array_equal(control["observation.images.image2"], target["observation.images.image2"])
    assert np.array_equal(control["observation.state"], target["observation.state"])
    assert np.array_equal(control["action"], target["action"])
    assert control["task"] == target["task"]


def test_target_arrow_profile_is_distinct_from_all_arrows_profile():
    from hdf5_to_lerobot_dataset import _sealed_profile

    all_arrows = _sealed_profile(False)
    target_arrow = _sealed_profile(True)
    assert target_arrow["pair_kind"] == TARGET_ARROW_PAIR_KIND
    assert target_arrow["pair_kind"] != all_arrows["pair_kind"]
    assert target_arrow["manifest_name"] != all_arrows["manifest_name"]
    assert target_arrow["sentinel_name"] != all_arrows["sentinel_name"]
    assert target_arrow["treatment_variant"] == TARGET_ARROW_VARIANT
    assert target_arrow["variants"] == ("control", TARGET_ARROW_VARIANT)
    assert target_arrow["visual_contract"]["relation_selection"] == "single_subject_to_task_goal"


def test_build_frame_treatment_with_no_target_relations_matches_control():
    kwargs = _frame_kwargs(
        agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8),
        relations=[["cookies_1", "is_on_top_of", "plate_1"]],  # no akita_black_bowl_1 subject
    )
    control = build_frame(**kwargs, variant="control")
    treatment = build_frame(**kwargs, variant="treatment")
    assert np.array_equal(control["observation.images.image"], treatment["observation.images.image"])


def test_graph_profiles_preserve_historical_arrow_and_control_pixels():
    kwargs = _frame_kwargs()
    control = build_frame(**kwargs, variant="control")
    treatment = build_frame(**kwargs, variant="treatment")
    graph, arrow_graph, expected_mask, _ = build_graph_paired_frames(**kwargs)
    assert np.array_equal(graph["observation.images.image"], control["observation.images.image"])
    assert np.array_equal(arrow_graph["observation.images.image"], treatment["observation.images.image"])
    assert np.any(expected_mask)


def test_graph_pair_changes_only_main_image_and_keeps_text_actions_state_wrist_identical():
    kwargs = _frame_kwargs(
        bboxes={
            "akita_black_bowl_1": [10, 10, 20, 20],
            "akita_black_bowl_2": [35, 10, 45, 20],
            "plate_1": [80, 80, 100, 100],
            "cookies_1": [60, 20, 70, 30],
        },
        relations=[
            ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_1", "is_behind", "cookies_1"),
        ],
    )
    graph, arrow_graph, expected_mask, relations = build_graph_paired_frames(**kwargs)
    assert relations == [
        ("akita_black_bowl_1", "is_left_of", "akita_black_bowl_2"),
        ("akita_black_bowl_1", "is_left_of", "plate_1"),
        ("akita_black_bowl_1", "is_behind", "cookies_1"),
    ]
    assert graph["task"] == arrow_graph["task"]
    assert np.array_equal(graph["action"], arrow_graph["action"])
    assert np.array_equal(graph["observation.state"], arrow_graph["observation.state"])
    assert np.array_equal(graph["observation.images.image2"], arrow_graph["observation.images.image2"])
    assert np.array_equal(
        graph["observation.images.image"],
        build_frame(**kwargs, variant="control")["observation.images.image"],
    )
    assert np.count_nonzero(expected_mask) > 0


def test_graph_pair_drops_relations_to_objects_absent_from_the_source_frame():
    kwargs = _frame_kwargs(
        bboxes={"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [80, 80, 100, 100]},
        relations=[
            ("akita_black_bowl_1", "is_left_of", "plate_1"),
            ("akita_black_bowl_1", "is_left_of", "removed_cookie_box"),
        ],
    )
    graph, arrow_graph, _mask, relations = build_graph_paired_frames(**kwargs)
    assert relations == [("akita_black_bowl_1", "is_left_of", "plate_1")]
    assert "removed_cookie_box" not in graph["task"]
    assert graph["task"] == arrow_graph["task"]


def test_graph_pair_recomputes_relations_from_bbox_world_and_rejects_stored_mismatch():
    """Offline graph supervision must agree with the canonical live extractor."""
    kwargs = _frame_kwargs(
        bboxes={
            "akita_black_bowl_1": [10, 10, 20, 20],
            "plate_1": [80, 80, 100, 100],
        },
        relations=[("akita_black_bowl_1", "is_left_of", "plate_1")],
    )
    world = {
        "akita_black_bowl_1": {"pos": [0.0, 0.0, 0.0]},
        "plate_1": {"pos": [0.0, -1.0, 0.0]},
    }
    _graph, _arrow_graph, _mask, relations = build_graph_paired_frames(**kwargs, world=world)
    assert relations == [("akita_black_bowl_1", "is_left_of", "plate_1")]

    tampered = dict(kwargs)
    tampered["relations"] = [("akita_black_bowl_1", "is_right_of", "plate_1")]
    with pytest.raises(AssertionError, match="canonical.*extractor"):
        build_graph_paired_frames(**tampered, world=world)


def test_graph_manifest_records_extractor_digest_and_verifier_rechecks_live_file():
    """A graph pair must be invalidated if the canonical extractor source drifts."""
    converter = Path(__file__).resolve().parents[1] / "hdf5_to_lerobot_dataset.py"
    source = converter.read_text(encoding="utf-8")
    assert '"graph_extractor_sha256": sha256_file(GRAPH_EXTRACTOR_PATH)' in source
    verify_start = source.index("def _load_sealed_manifest")
    verify_end = source.index("def validate_verified_pair", verify_start)
    verifier = source[verify_start:verify_end]
    assert "manifest.get(\"graph_extractor_sha256\")" in verifier
    assert "sha256_file(GRAPH_EXTRACTOR_PATH)" in verifier


def test_graph_episode_frame_metadata_mutation_is_rejected():
    kwargs = _frame_kwargs()
    graph, arrow_graph, _mask, _relations = build_graph_paired_frames(**kwargs)
    graph["episode_index"] = np.asarray(0)
    graph["frame_index"] = np.asarray(0)
    arrow_graph["episode_index"] = np.asarray(0)
    arrow_graph["frame_index"] = np.asarray(0)
    graph["episode_index"] = np.asarray(99)
    with pytest.raises(AssertionError, match="episode_index mismatch"):
        _assert_episode_expectations(
            graph,
            arrow_graph,
            episode_index=0,
            frame_index=0,
            frame_label="graph-tampered",
        )


def test_graph_manifest_rejects_noncanonical_tokenizer_contract(tmp_path):
    """A verified graph pair must not silently downgrade its 96-token budget."""
    from hdf5_to_lerobot_dataset import FPS, _source_snapshot_identity

    manifest = {
        "schema_version": 1,
        "pair_kind": GRAPH_PAIR_KIND,
        "visual_contract": SEALED_LORA_VISUAL_CONTRACT,
        "graph_contract": GRAPH_CONTRACT,
        "graph_formatter_sha256": sha256_file(REPO_ROOT / "scene_graph_formats.py"),
        "graph_extractor_sha256": sha256_file(GRAPH_EXTRACTOR_PATH),
        "storage_contract": {"image_dtype": "image", "use_videos": False, "fps": FPS},
        "source_snapshot_identity": _source_snapshot_identity([]),
        "full_experiment_ready": False,
        "launch_eligibility": "subset_smoke_not_launchable",
        "task_ids": [],
        "tasks": [],
        "total_episodes": 0,
        "total_frames": 0,
        "tokenizer_contract": {
            "model_id": "HuggingFaceTB/SmolVLM2-500M-Instruct",
            "max_length": 48,
            "truncation_allowed": False,
            "task_instruction_must_be_retained": True,
        },
    }
    path = tmp_path / "sealed_lora_graph_pair_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="tokenizer"):
        _load_sealed_manifest(tmp_path, graph=True)


def test_full_graph_manifest_requires_nonempty_source_grounded_frames():
    """500 empty demos must not satisfy the launchable full-experiment gate."""
    manifest = {
        "task_ids": list(range(10)),
        "total_episodes": 500,
        "total_frames": 0,
        "tasks": [
            {
                "task_id": task_id,
                "demos": [{"frame_count": 0} for _ in range(50)],
            }
            for task_id in range(10)
        ],
    }
    assert not _full_experiment_ready(manifest)


def test_paired_prewrite_invariants_reject_misplaced_arrow_pixels():
    kwargs = _frame_kwargs(agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8))
    control, treatment, expected_mask = build_paired_frames(**kwargs)
    assert expected_mask.any()
    assert_paired_frame_invariants(control, treatment, expected_arrow_mask=expected_mask)

    bad_treatment = dict(treatment)
    bad_main = treatment["observation.images.image"].copy()
    bad_main[0, 0] = [255, 0, 0]
    bad_treatment["observation.images.image"] = bad_main
    try:
        assert_paired_frame_invariants(
            control,
            bad_treatment,
            expected_arrow_mask=expected_mask,
            frame_label="tampered",
        )
    except AssertionError as exc:
        assert "localized" in str(exc)
    else:
        raise AssertionError("expected verifier invariant rejection for misplaced pixel")


def test_paired_mask_is_exactly_the_only_main_image_difference():
    control, treatment, expected_mask = build_paired_frames(
        **_frame_kwargs(agentview_rgb=np.zeros((128, 128, 3), dtype=np.uint8))
    )
    assert np.array_equal(
        expected_mask,
        main_image_change_mask(
            control["observation.images.image"], treatment["observation.images.image"]
        ),
    )


def test_pair_converter_verifier_rejects_tampered_stored_main_image(tmp_path):
    """Exercise the lossless pair path and prove verify rejects misplaced pixels."""
    source = tmp_path / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    with h5py.File(source, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["num_demos"] = 1
        data.attrs["problem_info"] = json.dumps({"language_instruction": "test task"})
        demo = data.create_group("demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.full((2, 128, 128, 3), 7, dtype=np.uint8))
        obs.create_dataset("ee_pos", data=np.zeros((2, 3), dtype=np.float32))
        obs.create_dataset("ee_ori", data=np.zeros((2, 3), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((2, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        bboxes = [
            {"akita_black_bowl_1": [10, 10, 20, 20], "plate_1": [90, 90, 100, 100]},
            {"akita_black_bowl_1": [20, 20, 30, 30], "plate_1": [90, 90, 100, 100]},
        ]
        relations = [["akita_black_bowl_1", "is_left_of", "plate_1"]]
        obs.create_dataset("agentview_bboxes", data=json.dumps(bboxes), dtype=h5py.string_dtype())
        obs.create_dataset(
            "agentview_scene_graph",
            data=json.dumps([relations, relations]),
            dtype=h5py.string_dtype(),
        )
        obs.create_dataset(
            "agentview_world_coords",
            data=json.dumps([
                {
                    "akita_black_bowl_1": {"pos": [0.0, 0.0, 0.0]},
                    "plate_1": {"pos": [0.0, -1.0, 0.0]},
                },
                {
                    "akita_black_bowl_1": {"pos": [0.1, 0.1, 0.0]},
                    "plate_1": {"pos": [0.0, -1.0, 0.0]},
                },
            ]),
            dtype=h5py.string_dtype(),
        )

    args = argparse.Namespace(
        data_dir=tmp_path,
        output_root=tmp_path / "pair",
        tasks=[0],
        demos_per_task=1,
        allow_subset=True,
    )
    args.output_root.mkdir()
    sealed_pair_sentinel_path(args.output_root).write_text('{"stale": true}\n', encoding="utf-8")
    run_convert_pair(args)
    assert not sealed_pair_sentinel_path(args.output_root).exists()
    run_verify(args)
    assert sealed_pair_sentinel_path(args.output_root).is_file()
    sentinel = validate_verified_pair(args.output_root, require_full_experiment=False)
    assert sentinel["full_experiment_ready"] is False
    assert sentinel["launch_eligibility"] == "subset_smoke_not_launchable"
    assert sentinel["manifest_path"] == "sealed_lora_pair_manifest.json"
    assert set(sentinel["dataset_fingerprints"]) == {"control", "treatment"}

    target_args = argparse.Namespace(
        data_dir=tmp_path,
        output_root=tmp_path / "target_pair",
        tasks=[0],
        demos_per_task=1,
        allow_subset=True,
    )
    target_args.output_root.mkdir()
    run_convert_pair(target_args, target_arrow=True)
    assert sealed_target_arrow_pair_manifest_path(target_args.output_root).is_file()
    assert not sealed_target_arrow_pair_sentinel_path(target_args.output_root).exists()
    run_verify(target_args, target_arrow=True)
    assert sealed_target_arrow_pair_sentinel_path(target_args.output_root).is_file()
    target_sentinel = validate_verified_pair(
        target_args.output_root,
        require_full_experiment=False,
        target_arrow=True,
    )
    assert target_sentinel["manifest_path"] == "sealed_lora_target_arrow_pair_manifest.json"
    assert set(target_sentinel["dataset_fingerprints"]) == {"control", TARGET_ARROW_VARIANT}
    assert target_sentinel["pair_kind"] == TARGET_ARROW_PAIR_KIND
    try:
        validate_verified_pair(target_args.output_root, target_arrow=True)
    except AssertionError as exc:
        assert "subset smoke" in str(exc)
    else:
        raise AssertionError("one-task target-arrow sentinel must be rejected for launch")

    try:
        validate_verified_pair(args.output_root)
    except AssertionError as exc:
        assert "subset smoke" in str(exc)
    else:
        raise AssertionError("one-task sealed sentinel must be rejected for launch")

    manifest_path = sealed_pair_manifest_path(args.output_root)
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(original_manifest + b" ")
    try:
        validate_verified_pair(args.output_root, require_full_experiment=False)
    except AssertionError as exc:
        assert "manifest bytes" in str(exc)
    else:
        raise AssertionError("manifest edit must invalidate its verified sentinel")
    manifest_path.write_bytes(original_manifest)

    import pyarrow as pa
    import pyarrow.parquet as pq

    stored_main = args.output_root / "treatment" / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(stored_main)
    image_column = "observation.images.image"
    image_entries = table[image_column].to_pylist()
    with Image.open(io.BytesIO(image_entries[0]["bytes"])) as image:
        tampered = np.asarray(image).copy()
    tampered[0, 0] = [255, 0, 0]
    encoded = io.BytesIO()
    Image.fromarray(tampered).save(encoded, format="PNG")
    image_entries[0]["bytes"] = encoded.getvalue()
    image_type = table.schema.field(image_column).type
    table = table.set_column(
        table.schema.get_field_index(image_column),
        image_column,
        pa.array(image_entries, type=image_type),
    )
    pq.write_table(table, stored_main)

    try:
        validate_verified_pair(args.output_root, require_full_experiment=False)
    except AssertionError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("dataset mutation must invalidate its verified sentinel")

    try:
        run_verify(args)
    except AssertionError as exc:
        assert "observation.images.image" in str(exc) or "main-image changes" in str(exc)
    else:
        raise AssertionError("expected source-grounded verifier rejection for tampered stored image")
    assert not sealed_pair_sentinel_path(args.output_root).exists()


def test_build_frame_rejects_unknown_variant():
    try:
        build_frame(**_frame_kwargs(), variant="bogus")
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown variant")


def test_camera_key_mapping_matches_hdf5_naming():
    # obs/eye_in_hand_rgb has NO robot0_ prefix, unlike obs/robot0_eye_in_hand_bboxes /
    # _scene_graph / _world_coords (verified directly against a real HDF5 file). This
    # test pins build_frame's kwarg name to that convention so a future edit that
    # accidentally renames the kwarg to "robot0_eye_in_hand_rgb" fails loudly.
    import inspect

    params = inspect.signature(build_frame).parameters
    assert "eye_in_hand_rgb" in params
    assert "robot0_eye_in_hand_rgb" not in params
