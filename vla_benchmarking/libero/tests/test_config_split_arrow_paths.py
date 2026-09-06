"""Regression guards for separating graph filters from arrow sources."""

from __future__ import annotations

from pathlib import Path

from vla_benchmarking.libero.evaluation.graph_relation_extractor import generate_frame_graph
from vla_benchmarking.libero.shared.config import ARROW_SOURCE_OBJECT, SCENE_GRAPH_SUBJECT_FILTER


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_shared_config_exposes_unfiltered_graph_and_legacy_arrow_source() -> None:
    assert SCENE_GRAPH_SUBJECT_FILTER is None
    assert ARROW_SOURCE_OBJECT == "akita_black_bowl_1"


def test_unfiltered_graph_has_all_ordered_object_pairs() -> None:
    names = [
        "akita_black_bowl_1",
        "akita_black_bowl_2",
        "cookies_1",
        "plate_1",
        "glazed_rim_porcelain_ramekin_1",
        "stove_1",
        "wooden_cabinet_1",
    ]
    bboxes = {name: [0, 0, 10, 10] for name in names}
    world = {name: {"pos": [float(index), 0.0, 0.0]} for index, name in enumerate(names)}

    relations = generate_frame_graph(
        bboxes,
        world,
        object_filter=set(names),
        subject_filter=SCENE_GRAPH_SUBJECT_FILTER,
    )

    assert len(relations) == 7 * 6
    assert sum(subject == ARROW_SOURCE_OBJECT for subject, _, _ in relations) == 6


def test_preview_arrow_path_uses_full_graph_filter() -> None:
    source = (
        REPO_ROOT / "vla_benchmarking" / "libero" / "evaluation" / "preview_visual_arrows.py"
    ).read_text(encoding="utf-8")

    assert "SCENE_GRAPH_SUBJECT_FILTER" in source
    assert "generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER" in source


def test_visual_pair_generates_full_graph_and_selects_goal_arrow_source() -> None:
    source = (
        REPO_ROOT / "vla_benchmarking" / "libero" / "evaluation" / "render_visual_arrow_pair.py"
    ).read_text(encoding="utf-8")

    assert "ARROW_SOURCE_OBJECT" in source
    assert "SCENE_GRAPH_SUBJECT_FILTER" in source
    assert "generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER" in source


def test_visual_eval_keeps_graph_filter_and_goal_arrow_source_separate() -> None:
    source = (
        REPO_ROOT
        / "vla_benchmarking"
        / "libero"
        / "evaluation"
        / "run_lerobot_eval_with_context.py"
    ).read_text(encoding="utf-8")

    assert "live_generator.scene_graph_subject_filter = SCENE_GRAPH_SUBJECT_FILTER" in source
    assert "arrow_subject=" in source
    assert "ARROW_SOURCE_OBJECT" in source
