import numpy as np
import pytest

from samgraph_core.geometric_graph import (
    GEOMETRIC_RELATION_RULES_REVISION,
    GeometricRelationConfig,
    NotebookMaskProxyConfig,
    build_directed_relations,
    direction_relation,
    geometric_relation_revision,
    geometric_relation_rules,
    geometric_relation_rules_sha256,
    bbox_iomin,
    mask_bbox_xyxy_half_open,
    notebook_mask_bbox_xyxy,
    notebook_direction_relation,
    support_relation,
)
from samgraph_core.geometry_profiles import (
    SAMGRAPH_SPATIAL_MASK_GEOMETRY,
    samgraph_spatial_mask_geometry,
)


def _rect(size: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _instances(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {"instance_id": instance_id, "class_id": class_id, "role": role}
        for instance_id, class_id, role in items
    ]


def test_notebook_pair_semantics_includes_bowl_bowl_without_changing_legacy():
    instances = _instances(("black_bowl_1", "bowl", "bowl"),
                           ("black_bowl_2", "bowl", "bowl"))
    masks = {"black_bowl_1": _rect(40, 10, 10, 20, 20),
             "black_bowl_2": _rect(40, 9, 9, 22, 22)}
    kwargs = dict(suite="spatial", instruction="place the bowl on the plate", shape=(40, 40))
    legacy, _, _ = build_directed_relations(instances, masks,
        geometry_rules=NotebookMaskProxyConfig(), **kwargs)
    aligned, _, _ = build_directed_relations(instances, masks,
        geometry_rules=NotebookMaskProxyConfig(notebook_pair_semantics=True), **kwargs)
    assert not legacy
    assert len(aligned) == 2
    assert set(aligned.values()) == {"is_on_top_of", "is_below_of"}


def test_notebook_pair_semantics_coincident_direction_and_serialization():
    config = NotebookMaskProxyConfig(notebook_pair_semantics=True)
    assert notebook_direction_relation((2, 2), (2, 2), geometry_rules=config) == "is_behind"
    rules = geometric_relation_rules(config)
    assert rules["notebook_pair_semantics"] is True
    assert notebook_direction_relation((2, 2), (2, 2), geometry_rules=rules) == "is_behind"
    assert geometric_relation_rules_sha256(config) != geometric_relation_rules_sha256(NotebookMaskProxyConfig())


def test_default_direction_rule_is_stable_and_vertical_tie_wins() -> None:
    assert GEOMETRIC_RELATION_RULES_REVISION == "libero_geometric_relation_rules_v2"
    assert "top drawer" in geometric_relation_rules()["object_goal"]["cabinet_storage_tokens"]
    assert direction_relation((10, 10), (0, 0)) == "is_behind"
    assert direction_relation((20, 10), (0, 0)) == "is_left_of"


def test_support_and_cabinet_containment_use_visual_hulls() -> None:
    bowl = _rect(64, 25, 20, 39, 34)
    support = _rect(64, 20, 15, 45, 42)
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
        bowl,
        support,
        instruction="pick up the bowl and place it on the plate",
        shape=bowl.shape,
    )
    assert relation == "is_on_top_of"
    assert evidence["qualifies"] is True

    cabinet = _rect(64, 20, 10, 45, 50)
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl,
        cabinet,
        instruction="put the bowl in the top layer of the wooden cabinet",
        shape=bowl.shape,
    )
    assert relation == "is_inside"
    assert evidence["cabinet_exception_applied"] is True


def test_top_drawer_task_slug_uses_containment_without_simulator_geometry() -> None:
    bowl = _rect(128, 48, 44, 64, 60)
    cabinet = _rect(128, 40, 32, 88, 96)
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl,
        cabinet,
        instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=bowl.shape,
    )
    assert relation == "is_inside"
    assert evidence["cabinet_exception_applied"] is True

    relation, _ = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl,
        cabinet,
        instruction="put the bowl on the wooden cabinet",
        shape=bowl.shape,
    )
    assert relation == "is_on_top_of"

    relations, triplets, _ = build_directed_relations(
        _instances(
            ("bowl_1", "black_bowl", "bowl"),
            ("cabinet_1", "wooden_cabinet", "cabinet"),
        ),
        {"bowl_1": bowl, "cabinet_1": cabinet},
        suite="spatial",
        instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=bowl.shape,
    )
    assert relations[("bowl_1", "cabinet_1")] == "is_inside"
    assert any(
        item["subject"] == "bowl_1"
        and item["object"] == "cabinet_1"
        and item["relation"] == "is_inside"
        for item in triplets
    )

    above_edge_bowl = _rect(128, 48, 30, 64, 40)
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        above_edge_bowl,
        cabinet,
        instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=cabinet.shape,
    )
    assert relation == "is_on_top_of"
    assert evidence["qualifies"] is True
    assert evidence["cabinet_exception_applied"] is False

    partial_instances = _instances(
        ("bowl_1", "black_bowl", "bowl"),
        ("cabinet_1", "wooden_cabinet", "cabinet"),
    )
    with pytest.raises(ValueError, match="no unique visually supported bowl"):
        build_directed_relations(
            partial_instances,
            {"bowl_1": above_edge_bowl, "cabinet_1": cabinet},
            suite="spatial",
            instruction="put the bowl in the top drawer of the wooden cabinet",
            shape=cabinet.shape,
        )
    _, partial_triplets, _ = build_directed_relations(
        partial_instances,
        {"bowl_1": above_edge_bowl, "cabinet_1": cabinet},
        suite="spatial",
        instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=cabinet.shape,
        allow_partial_visibility=True,
    )
    assert any(
        item["subject"] == "bowl_1"
        and item["object"] == "cabinet_1"
        and item["relation"] == "is_on_top_of"
        for item in partial_triplets
    )

    second_inside_bowl = _rect(128, 48, 52, 64, 68)
    with pytest.raises(ValueError, match="no unique visually supported bowl"):
        build_directed_relations(
            _instances(
                ("bowl_1", "black_bowl", "bowl"),
                ("bowl_2", "black_bowl", "bowl"),
                ("cabinet_1", "wooden_cabinet", "cabinet"),
            ),
            {"bowl_1": bowl, "bowl_2": second_inside_bowl, "cabinet_1": cabinet},
            suite="spatial",
            instruction="put the bowl in the top drawer of the wooden cabinet",
            shape=cabinet.shape,
            allow_partial_visibility=True,
        )


def test_custom_geometry_config_changes_revision_and_digest() -> None:
    custom = GeometricRelationConfig(vertical_scale=2.0, hull_coverage_min=0.9)
    assert geometric_relation_revision() == GEOMETRIC_RELATION_RULES_REVISION
    assert geometric_relation_revision(custom) != GEOMETRIC_RELATION_RULES_REVISION
    assert geometric_relation_rules_sha256(custom) != geometric_relation_rules_sha256()


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_direction_and_object_goal_are_resolution_invariant(size: int) -> None:
    object_mask = _rect(size, size // 4, size // 4, size // 4 + size // 16, size // 4 + size // 16)
    destination_mask = _rect(size, size // 2, size // 4, size // 2 + size // 16, size // 4 + size // 16)
    instances = _instances(("object_1", "object", "object"), ("destination_1", "plate", "destination"))
    _, triplets, _ = build_directed_relations(
        instances,
        {"object_1": object_mask, "destination_1": destination_mask},
        suite="object",
        instruction="pick the object and place it on the plate",
        shape=(size, size),
    )
    assert triplets[0]["relation"] == "is_on_top_of"
    assert triplets[1]["relation"] == "is_below_of"


def test_shape_mismatch_is_rejected() -> None:
    instances = _instances(("a", "object", "object"), ("b", "plate", "destination"))
    with pytest.raises(ValueError, match="inconsistent image shapes"):
        build_directed_relations(
            instances,
            {"a": np.ones((256, 256), dtype=bool), "b": np.ones((512, 512), dtype=bool)},
            suite="object",
            instruction="put the object on the plate",
            shape=(256, 256),
        )


def test_notebook_mask_proxy_uses_half_open_iomin_and_strict_threshold() -> None:
    bowl = _rect(32, 0, 8, 10, 18)
    support = _rect(32, 5, 8, 15, 18)
    assert mask_bbox_xyxy_half_open(bowl) == (0, 8, 10, 18)
    assert bbox_iomin((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
    proxy = {
        "profile": "samgraph_spatial_mask_geometry",
        "bbox_padding_x": 0,
        "bbox_padding_y": 0,
        "containment_iomin_threshold": "0.5",
    }
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
        bowl,
        support,
        instruction="put the bowl on the plate",
        shape=bowl.shape,
        geometry_rules=proxy,
    )
    assert relation is None
    assert evidence["bbox_iomin"] == 0.5
    proxy["containment_iomin_threshold"] = 0.49
    relation, _ = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
        bowl,
        support,
        instruction="put the bowl on the plate",
        shape=bowl.shape,
        geometry_rules=proxy,
    )
    assert relation == "is_on_top_of"


def test_notebook_mask_proxy_direction_axes_signs_scale_and_tie_are_explicit() -> None:
    proxy = {"profile": "samgraph_spatial_mask_geometry", "direction_tie_axis": "vertical"}
    assert notebook_direction_relation((10, 10), (0, 0), geometry_rules=proxy) == "is_behind"
    proxy["direction_tie_axis"] = "horizontal"
    assert notebook_direction_relation((10, 10), (0, 0), geometry_rules=proxy) == "is_behind"
    proxy.update({"direction_horizontal_axis": "y", "direction_vertical_axis": "x",
                  "direction_horizontal_sign": "-1", "direction_vertical_sign": "1",
                  "direction_vertical_scale": "2"})
    assert notebook_direction_relation((10, 0), (0, 0), geometry_rules=proxy) == "is_in_front_of"


def test_notebook_mask_proxy_affine_direction_map_uses_cross_axis_scores() -> None:
    # front = du + dv; left = du - dv.  The diagonal displacement therefore
    # exercises both off-diagonal terms rather than only an axis/sign preset.
    affine = {
        "profile": "samgraph_spatial_mask_geometry",
        "front_score_coefficients": [1.0, 1.0],
        "left_score_coefficients": [1.0, -1.0],
    }
    assert notebook_direction_relation((0, 1), (0, 0), geometry_rules=affine) == "is_in_front_of"
    assert notebook_direction_relation((1, -1), (0, 0), geometry_rules=affine) == "is_left_of"
    assert notebook_direction_relation((-1, 1), (0, 0), geometry_rules=affine) == "is_right_of"


def test_notebook_mask_proxy_affine_map_preserves_dominant_axis_tie_semantics() -> None:
    affine = {
        "profile": "samgraph_spatial_mask_geometry",
        # front = -2*dv; left = du, so (du, dv)=(2, 1) ties at magnitude 2.
        "direction_front_coefficients": [0.0, -2.0],
        "direction_left_coefficients": [1.0, 0.0],
    }
    assert notebook_direction_relation((2, 1), (0, 0), geometry_rules=affine) == "is_behind"
    affine["direction_tie_axis"] = "horizontal"
    assert notebook_direction_relation((2, 1), (0, 0), geometry_rules=affine) == "is_left_of"


def test_notebook_mask_proxy_affine_map_round_trips_and_changes_revision() -> None:
    config = NotebookMaskProxyConfig(
        direction_front_coefficients=(0.2, -1.5),
        direction_left_coefficients=(1.0, 0.3),
    )
    serialized = geometric_relation_rules(config)
    coordinate = serialized["coordinate_adjustment"]
    assert coordinate["front_score_coefficients"] == [0.2, -1.5]
    assert coordinate["left_score_coefficients"] == [1.0, 0.3]
    assert geometric_relation_rules(serialized) == serialized
    assert geometric_relation_revision(config) != "samgraph_spatial_mask_geometry"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"direction_front_coefficients": (1.0, 0.0)}, "supplied together"),
        ({"direction_front_coefficients": (0.0, 0.0), "direction_left_coefficients": (1.0, 0.0)}, "all-zero"),
        ({"direction_front_coefficients": (1.0, 0.0), "direction_left_coefficients": (2.0, 0.0)}, "non-degenerate"),
        ({"direction_front_coefficients": (1.0, 2.0, 3.0), "direction_left_coefficients": (0.0, 1.0)}, "exactly two"),
    ],
)
def test_notebook_mask_proxy_rejects_invalid_affine_coefficients(
    kwargs: dict[str, object], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        NotebookMaskProxyConfig(**kwargs)


def test_notebook_mask_proxy_is_resolution_invariant_without_absolute_padding() -> None:
    proxy = {
        "profile": "samgraph_spatial_mask_geometry",
        "bbox_padding_x": 0,
        "bbox_padding_y": 0,
        "containment_iomin_threshold": 0.8,
    }
    small_bowl = _rect(32, 8, 8, 16, 16)
    small_plate = _rect(32, 8, 8, 16, 16)
    large_bowl = _rect(64, 16, 16, 32, 32)
    large_plate = _rect(64, 16, 16, 32, 32)
    args = {
        "instruction": "put the bowl on the plate",
        "suite": "spatial",
    }
    _, small_triplets, _ = build_directed_relations(
        _instances(("bowl_1", "black_bowl", "bowl"), ("plate_1", "plate", "support")),
        {"bowl_1": small_bowl, "plate_1": small_plate},
        shape=small_bowl.shape, geometry_rules=proxy, **args,
    )
    _, large_triplets, _ = build_directed_relations(
        _instances(("bowl_1", "black_bowl", "bowl"), ("plate_1", "plate", "support")),
        {"bowl_1": large_bowl, "plate_1": large_plate},
        shape=large_bowl.shape, geometry_rules=proxy, **args,
    )
    assert [(item["subject"], item["relation"], item["object"]) for item in small_triplets] == [
        (item["subject"], item["relation"], item["object"]) for item in large_triplets
    ]


def test_notebook_mask_proxy_defaults_to_unpadded_mask_extents() -> None:
    assert NotebookMaskProxyConfig().bbox_padding_x == 0.0
    assert NotebookMaskProxyConfig().bbox_padding_y == 0.0
    mask = _rect(128, 20, 30, 40, 50)
    assert notebook_mask_bbox_xyxy(mask) == mask_bbox_xyxy_half_open(mask) == (20, 30, 40, 50)


def test_notebook_mask_proxy_unpadded_geometry_is_resolution_invariant() -> None:
    decisions = []
    for size in (128, 256, 1024):
        scale = size / 128
        bowl = _rect(size, int(20 * scale), int(20 * scale), int(40 * scale), int(40 * scale))
        support = _rect(size, int(20 * scale), int(20 * scale), int(42 * scale), int(42 * scale))
        relation, evidence = support_relation(
            {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
            {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
            bowl, support, instruction="put the bowl on the plate", shape=bowl.shape,
            geometry_rules={"profile": "samgraph_spatial_mask_geometry"},
        )
        decisions.append((relation, evidence["bbox_iomin"]))
    assert all(relation == decisions[0][0] for relation, _ in decisions)
    assert all(abs(iomin - decisions[0][1]) < 0.02 for _, iomin in decisions)


def test_notebook_mask_proxy_rules_round_trip_hash_and_direct_config() -> None:
    original = NotebookMaskProxyConfig(
        bbox_padding_x=0.03125,
        bbox_padding_y=0.0625,
        stack_order="lower_vertical_is_top",
        drawer_stack_order="lower_vertical_is_inside",
        drawer_mode="instruction_overlap",
    )
    serialized = geometric_relation_rules(original)
    assert serialized["simulator_geometry_consumed"] is False
    assert serialized["stack_order"]["drawer_name"] == "lower_vertical_is_inside"
    assert geometric_relation_rules(serialized) == serialized
    assert geometric_relation_rules_sha256(serialized) == geometric_relation_rules_sha256(original)
    assert geometric_relation_revision(serialized) == geometric_relation_revision(original)
    assert geometric_relation_rules_sha256(
        {**serialized, "drawer_mode": "disabled"}
    ) != geometric_relation_rules_sha256(serialized)
    _, triplets, _ = build_directed_relations(
        _instances(("bowl_1", "black_bowl", "bowl"), ("plate_1", "plate", "support")),
        {"bowl_1": _rect(32, 8, 8, 16, 16), "plate_1": _rect(32, 8, 8, 16, 16)},
        suite="spatial", instruction="put the bowl on the plate", shape=(32, 32),
        geometry_rules=original,
    )
    assert triplets


def test_notebook_mask_proxy_all_qualified_keeps_each_support_pair_and_inverse() -> None:
    bowl = _rect(64, 24, 24, 32, 32)
    plate = _rect(64, 22, 22, 34, 34)
    ramekin = _rect(64, 24, 23, 34, 33)
    instances = _instances(
        ("bowl_1", "black_bowl", "bowl"),
        ("plate_1", "plate", "support"),
        ("white_ramekin_1", "white_ramekin", "support"),
    )
    relations, triplets, _ = build_directed_relations(
        instances,
        {"bowl_1": bowl, "plate_1": plate, "white_ramekin_1": ramekin},
        suite="spatial",
        instruction="put the bowl on the plate",
        shape=bowl.shape,
        geometry_rules={"profile": "samgraph_spatial_mask_geometry"},
    )
    assert relations[("bowl_1", "plate_1")] == "is_on_top_of"
    assert relations[("bowl_1", "white_ramekin_1")] == "is_on_top_of"
    assert relations[("plate_1", "bowl_1")] == "is_below_of"
    assert relations[("white_ramekin_1", "bowl_1")] == "is_below_of"
    assert len(triplets) == len(instances) * (len(instances) - 1)
    assert len({(item["subject"], item["object"]) for item in triplets}) == len(triplets)


def test_notebook_mask_proxy_and_live_profile_differ_on_ambiguous_supports() -> None:
    bowl = _rect(64, 24, 24, 32, 32)
    plate = _rect(64, 22, 22, 34, 34)
    ramekin = _rect(64, 24, 23, 34, 33)
    instances = _instances(
        ("bowl_1", "black_bowl", "bowl"),
        ("plate_1", "plate", "support"),
        ("white_ramekin_1", "white_ramekin", "support"),
    )
    masks = {"bowl_1": bowl, "plate_1": plate, "white_ramekin_1": ramekin}
    relations, _, _ = build_directed_relations(
        instances, masks, suite="spatial", instruction="put the bowl on the plate",
        shape=bowl.shape,
        geometry_rules={"profile": "samgraph_spatial_mask_geometry"},
    )
    assert ("bowl_1", "plate_1") in relations
    assert ("bowl_1", "white_ramekin_1") in relations
    with pytest.raises(ValueError, match="multiple visual supports"):
        build_directed_relations(
            instances, masks, suite="spatial", instruction="put the bowl on the plate",
            shape=bowl.shape,
        )


def test_notebook_mask_proxy_profile_is_canonical_and_has_best_preset() -> None:
    rules = geometric_relation_rules({"profile": "samgraph_spatial_mask_geometry"})
    assert rules["profile"] == "samgraph_spatial_mask_geometry"
    assert rules["revision"] == "samgraph_spatial_mask_geometry"
    assert rules["mask_bbox"]["padding_x_fraction"] == 0.0
    assert rules["mask_bbox"]["padding_y_fraction"] == 0.0
    assert rules["containment"]["threshold"] == 0.8
    assert rules["coordinate_adjustment"]["vertical_scale"] == 2.0
    with pytest.raises(ValueError, match="unsupported geometry profile"):
        geometric_relation_rules({"profile": "notebook_mask_proxy_v1"})


def test_samgraph_geometry_profile_is_immutable_and_accessor_is_fresh() -> None:
    with pytest.raises(TypeError):
        SAMGRAPH_SPATIAL_MASK_GEOMETRY["bbox_padding_x"] = 1.0
    first = samgraph_spatial_mask_geometry()
    first["bbox_padding_x"] = 1.0
    assert samgraph_spatial_mask_geometry()["bbox_padding_x"] == 0.0


def test_notebook_mask_proxy_rejects_unknown_stack_order() -> None:
    with pytest.raises(ValueError, match="unsupported notebook stack-order"):
        NotebookMaskProxyConfig(stack_order="invented_order")
    with pytest.raises(ValueError, match="unsupported notebook drawer stack-order"):
        NotebookMaskProxyConfig(drawer_stack_order="invented_order")


def test_notebook_mask_proxy_drawer_gate_has_separate_stack_order() -> None:
    bowl = _rect(64, 20, 24, 36, 40)
    cabinet = _rect(64, 16, 16, 48, 52)
    proxy = {"profile": "samgraph_spatial_mask_geometry", "bbox_padding_x": 0, "bbox_padding_y": 0}
    relation, evidence = support_relation(
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction="put the bowl on the wooden cabinet", shape=bowl.shape,
        geometry_rules=proxy,
    )
    assert relation == "is_below_of"
    assert evidence["cabinet_exception_applied"] is False

    relation, evidence = support_relation(
        {"instance_id": "akita_black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction="put the bowl in the top drawer of the wooden cabinet", shape=bowl.shape,
        geometry_rules=proxy,
    )
    assert relation == "is_inside"
    assert evidence["cabinet_exception_applied"] is True

    # Instruction-overlap alone must not turn an ordinary support into a
    # drawer relation merely because the text mentions a cabinet drawer.
    relation, evidence = support_relation(
        {"instance_id": "bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "plate_1", "class_id": "plate", "role": "support"},
        bowl, cabinet, instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=bowl.shape,
        geometry_rules={**proxy, "drawer_mode": "instruction_overlap"},
    )
    assert relation == "is_below_of"
    assert evidence["cabinet_exception_applied"] is False

    lower = {**proxy, "drawer_stack_order": "lower_vertical_is_inside"}
    relation, _ = support_relation(
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=bowl.shape, geometry_rules=lower,
    )
    assert relation == "contains"

    disabled = {**proxy, "drawer_mode": "disabled"}
    relation, _ = support_relation(
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction="put the bowl in the top drawer of the wooden cabinet",
        shape=bowl.shape, geometry_rules=disabled,
    )
    assert relation == "is_below_of"


def test_notebook_drawer_exception_requires_exact_normalized_identity_pair() -> None:
    bowl = _rect(64, 20, 24, 36, 40)
    cabinet = _rect(64, 16, 16, 48, 52)
    proxy = {"profile": "samgraph_spatial_mask_geometry"}
    instruction = "put the bowl in the top drawer of the wooden cabinet"

    # The notebook's asset-prefixed identity normalizes to the calibrated pair.
    relation, evidence = support_relation(
        {"instance_id": "akita_black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction=instruction, shape=bowl.shape, geometry_rules=proxy,
    )
    assert relation == "is_inside"
    assert evidence["cabinet_exception_applied"] is True

    # Other bowl/cabinet IDs keep ordinary on-top/below behavior even when
    # their masks and instruction are otherwise identical.
    relation, evidence = support_relation(
        {"instance_id": "black_bowl_2", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl, cabinet, instruction=instruction, shape=bowl.shape, geometry_rules=proxy,
    )
    assert relation == "is_below_of"
    assert evidence["cabinet_exception_applied"] is False

    # A centroid on the boundary is not a strict overlap.
    boundary_bowl = _rect(64, 8, 24, 25, 40)
    relation, evidence = support_relation(
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        boundary_bowl, cabinet, instruction=instruction, shape=bowl.shape,
        geometry_rules={**proxy, "containment_iomin_threshold": 0.5},
    )
    assert relation == "is_below_of"
    assert evidence["cabinet_exception_applied"] is False


def test_notebook_drawer_exception_does_not_reinterpret_top_layer_instruction() -> None:
    bowl = _rect(64, 20, 24, 36, 40)
    cabinet = _rect(64, 16, 16, 48, 52)
    relation, evidence = support_relation(
        {"instance_id": "black_bowl_1", "class_id": "black_bowl", "role": "bowl"},
        {"instance_id": "wooden_cabinet_1", "class_id": "wooden_cabinet", "role": "cabinet"},
        bowl,
        cabinet,
        instruction="put the bowl in the top layer of the wooden cabinet",
        shape=bowl.shape,
        geometry_rules={"profile": "samgraph_spatial_mask_geometry"},
    )
    assert relation == "is_below_of"
    assert evidence["cabinet_exception_applied"] is False
