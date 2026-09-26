import io
from types import SimpleNamespace
from types import SimpleNamespace

import numpy as np
from PIL import Image

from samgraph_core.geometric_graph import GeometricRelationConfig
from samgraph_core.live_resolution import LiberoArrowResolutionService, _Episode
from samgraph_core.mask_scene import MaskSceneInitializer, encode_mask


def _png(mask: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output, format="PNG")
    return output.getvalue()


class _ObjectSegmenter:
    provider = "test_segmenter"
    production_inputs = ("test_rgb",)
    provenance = {}

    def __init__(self, size: int) -> None:
        self._masks = {
            "object": np.pad(
                np.ones((size // 16, size // 16), dtype=bool),
                ((size // 4, size - size // 4 - size // 16),) * 2,
            ),
            "plate": np.pad(
                np.ones((size // 16, size // 16), dtype=bool),
                ((size // 2, size - size // 2 - size // 16),) * 2,
            ),
        }

    def segment(self, *, prompt: str, jpeg: bytes):
        key = "object" if prompt == "object" else "plate"
        return SimpleNamespace(masks=[SimpleNamespace(png=_png(self._masks[key]), score=1.0)])


def test_mask_scene_initializer_accepts_native_square_resolutions() -> None:
    for size in (256, 512, 1024):
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        scene = MaskSceneInitializer(_ObjectSegmenter(size)).initialize(
            rgb, instruction="pick the object and place it on the plate", suite="object"
        )
        assert scene["width"] == size
        assert scene["height"] == size
        assert scene["rules_sha256"]


def test_mask_scene_initializer_records_custom_rule_revision() -> None:
    custom = GeometricRelationConfig(mask_overlap_iou_max=0.75)
    size = 256
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    scene = MaskSceneInitializer(
        _ObjectSegmenter(size), geometry_rules=custom
    ).initialize(
        rgb, instruction="pick the object and place it on the plate", suite="object"
    )
    assert scene["relation_revision"].startswith("libero_geometric_relation_rules_v2+")
    assert scene["rules"]["quality"]["mask_overlap_iou_max"] == 0.75


class _PartialSpatialSegmenter:
    provider = "test_segmenter"
    production_inputs = ("test_rgb",)
    provenance = {}

    def __init__(self, size: int, *, extra_bowl: bool = False, include_bowl: bool = True,
                 include_cabinet: bool = False) -> None:
        self._size = size
        self._masks = {
            "dinner plate": self._rectangle(56, 56, 76, 76),
        }
        if include_bowl:
            self._masks["small metal bowl"] = self._rectangle(8, 8, 24, 24)
        else:
            self._masks["small box"] = self._rectangle(88, 88, 108, 108)
        if include_cabinet:
            self._masks["wooden cabinet"] = self._rectangle(40, 40, 110, 110)
        if extra_bowl:
            self._masks["small metal bowl"] = [
                self._rectangle(8, 8, 24, 24),
                self._rectangle(32, 8, 48, 24),
                self._rectangle(8, 32, 24, 48),
            ]

    def _rectangle(self, top: int, left: int, bottom: int, right: int) -> object:
        mask = np.zeros((self._size, self._size), dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    def segment(self, *, prompt: str, jpeg: bytes):
        value = self._masks.get(prompt, [])
        if not isinstance(value, list):
            value = [value]
        return SimpleNamespace(
            masks=[SimpleNamespace(png=_png(mask), score=1.0) for mask in value]
        )


def test_partial_spatial_inventory_records_only_observed_sam_classes() -> None:
    size = 128
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = (np.indices((size, size))[0] * 2).astype(np.uint8)
    scene = MaskSceneInitializer(_PartialSpatialSegmenter(size)).initialize(
        rgb,
        instruction="put the bowl in the cabinet",
        suite="spatial",
        allow_partial_inventory=True,
    )
    assert {item["class_id"] for item in scene["instances"]} == {"black_bowl", "plate"}
    assert scene["allow_partial_inventory"] is True
    assert set(scene["missing_class_ids"]) == {
        "black_bowl", "cookies", "white_ramekin", "flat_stove", "wooden_cabinet"
    }
    assert scene["inventory_coverage"]["observed_instance_count"] == 2
    assert scene["initial_mask_quality_gate"]["status"] == "PASS_PARTIAL"
    try:
        MaskSceneInitializer(_PartialSpatialSegmenter(size)).initialize(
            rgb, instruction="put the bowl in the cabinet", suite="spatial"
        )
    except ValueError as error:
        assert "missed required classes" in str(error)
    else:
        raise AssertionError("strict spatial inventory unexpectedly accepted missing classes")


def test_partial_spatial_inventory_still_rejects_overcount() -> None:
    size = 128
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = (np.indices((size, size))[0] * 2).astype(np.uint8)
    try:
        MaskSceneInitializer(_PartialSpatialSegmenter(size, extra_bowl=True)).initialize(
            rgb,
            instruction="put the bowl in the cabinet",
            suite="spatial",
            allow_partial_inventory=True,
        )
    except ValueError as error:
        assert "ambiguous or incomplete" in str(error)
    else:
        raise AssertionError("partial spatial inventory accepted an over-count")


def test_partial_spatial_top_drawer_initialization_allows_no_inside_candidate() -> None:
    size = 128
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = (np.indices((size, size))[0] * 2).astype(np.uint8)
    segmenter = _PartialSpatialSegmenter(size, include_cabinet=True)
    # Move the retained bowl above the cabinet so the visual proxy has no
    # inside candidate; partial visibility must keep the frame usable.
    segmenter._masks["small metal bowl"] = segmenter._rectangle(8, 8, 24, 24)
    scene = MaskSceneInitializer(segmenter).initialize(
        rgb,
        instruction="put the bowl in the top drawer of the wooden cabinet",
        suite="spatial",
        allow_partial_inventory=True,
    )
    assert {item["class_id"] for item in scene["instances"]} == {
        "black_bowl", "plate", "wooden_cabinet"
    }
    assert scene["triplets"]


class _StartTracker:
    supports_unobserved_continuation = False

    def __init__(self) -> None:
        self.started = []

    def stop(self, session_id: str) -> None:
        return None

    def start(self, **kwargs) -> None:
        self.started.append(kwargs)

    def bind_scene(self, sessions) -> None:
        return None


def test_partial_explicit_pair_uses_deterministic_tracking_fallback_only() -> None:
    size = 128
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    digest = __import__("hashlib").sha256(rgb.tobytes()).hexdigest()
    tracker = _StartTracker()
    resolver = LiberoArrowResolutionService(
        tracking_service=tracker,
        segmentation_service=_PartialSpatialSegmenter(size, include_bowl=False),
        graph_brain=None,
        allow_partial_inventory=True,
    )
    episode, source_mask, destination_mask = resolver._start_episode(
        episode_id="task/demo_0",
        state_sequence=0,
        instruction="put the bowl in the cabinet",
        suite="spatial",
        rgb=rgb,
        encoded_rgb=_png(rgb[:, :, 0].astype(bool)),
        digest=digest,
        selected_pair=("black_bowl_1", "plate_1"),
    )
    assert episode.source_id == "black_bowl_1"
    assert episode.destination_id == "plate_1"
    assert (episode.tracking_source_id, episode.tracking_destination_id) == (
        "cookies_1", "plate_1"
    )
    assert episode.tracking_pair_fallback is True
    assert source_mask.any() and destination_mask.any()


def test_live_rgb_validation_accepts_square_native_dimensions() -> None:
    for size in (256, 512, 1024):
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        import hashlib

        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        value, actual = LiberoArrowResolutionService._validate_rgb(rgb, digest)
        assert value.shape == (size, size, 3)
        assert actual == digest


def test_live_graph_allows_zero_visible_top_drawer_inside_candidates() -> None:
    size = 128
    bowl = np.zeros((size, size), dtype=bool)
    bowl[30:40, 48:64] = True  # overlaps the cabinet edge, centroid is above it
    cabinet = np.zeros((size, size), dtype=bool)
    cabinet[32:96, 40:88] = True
    instruction = "put the bowl in the top drawer of the wooden cabinet"
    initial = {
        "height": size,
        "width": size,
        "instances": [
            {
                "instance_id": "bowl_1",
                "class_id": "black_bowl",
                "role": "bowl",
                "mask_png_base64": encode_mask(bowl),
            },
            {
                "instance_id": "cabinet_1",
                "class_id": "wooden_cabinet",
                "role": "cabinet",
                "mask_png_base64": encode_mask(cabinet),
            },
        ],
    }
    episode = _Episode(
        episode_id="episode-1",
        instruction=instruction,
        suite="spatial",
        source_id="bowl_1",
        destination_id="cabinet_1",
        source_query="bowl",
        destination_query="cabinet",
        source_session="source",
        destination_session="destination",
        camera_id="agentview",
        selection=SimpleNamespace(origin="explicit_pair"),
        last_sequence=0,
        initial_scene=initial,
        entity_sessions={"bowl_1": "source", "cabinet_1": "destination"},
    )
    snapshot = {
        "source_frame_ts": 0.0,
        "observation_source": "test_current",
        "tracking_method": "seed_mask",
    }
    graph = LiberoArrowResolutionService._live_scene(
        episode,
        source_mask=bowl,
        destination_mask=cabinet,
        state_sequence=0,
        rgb_sha256="rgb",
        source_snapshot=snapshot,
        destination_snapshot=snapshot,
        observations={"bowl_1": (bowl, snapshot), "cabinet_1": (cabinet, snapshot)},
    )
    assert any(
        item["subject"] == "bowl_1"
        and item["object"] == "cabinet_1"
        and item["relation"] == "is_on_top_of"
        for item in graph["triplets"]
    )
