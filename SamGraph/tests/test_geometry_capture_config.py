import base64
import hashlib
import io
import json
import numpy as np
from PIL import Image

from samgraph_core.centroid_capture import HttpGraphSnapshotResolver
from samgraph_core.geometric_graph import (
    GEOMETRIC_RELATION_RULES_REVISION,
    GeometricRelationConfig,
    geometric_relation_revision,
    geometric_relation_rules,
    geometric_relation_rules_sha256,
)


def _mask_png(mask: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class _Response:
    def __init__(self, snapshot: dict) -> None:
        self._payload = json.dumps(snapshot).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return self._payload


def _snapshot(*, size: int, geometry_rules) -> tuple[np.ndarray, dict]:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    source = np.zeros((size, size), dtype=bool)
    source[20:30, 20:30] = True
    destination = np.zeros((size, size), dtype=bool)
    destination[80:90, 80:90] = True
    source_center = [24.5, 24.5]
    destination_center = [84.5, 84.5]
    source_item = {
        "instance_id": "object_1",
        "mask_png_base64": _mask_png(source),
        "current_geometry_valid": True,
        "track_state": "observed",
        "source_frame_ts": 0,
    }
    destination_item = {
        "instance_id": "destination_1",
        "mask_png_base64": _mask_png(destination),
        "current_geometry_valid": True,
        "track_state": "observed",
        "source_frame_ts": 0,
    }
    graph = {
        "rgb_sha256": digest,
        "relation_revision": geometric_relation_revision(geometry_rules),
        "rules_sha256": geometric_relation_rules_sha256(geometry_rules),
        "rules": geometric_relation_rules(geometry_rules),
        "simulator_geometry_consumed": False,
        "selected_pair": ["object_1", "destination_1"],
        "instances": [
            {**source_item, "center_xy": source_center},
            {**destination_item, "center_xy": destination_center},
        ],
    }
    snapshot = {
        "rgb_sha256": digest,
        "episode_id": "episode-1",
        "state_sequence": 0,
        "provider": "test",
        "selection": {
            "source_entity_id": "object_1",
            "destination_entity_id": "destination_1",
        },
        "instances": [source_item, destination_item],
        "provenance": {"graph_model": "test-model"},
        "graph": graph,
    }
    return rgb, snapshot


def test_http_capture_verifies_custom_config_and_serialized_contract(monkeypatch) -> None:
    custom = GeometricRelationConfig(vertical_scale=2.0, hull_coverage_min=0.9)
    rgb, snapshot = _snapshot(size=256, geometry_rules=custom)
    monkeypatch.setattr(
        "samgraph_core.centroid_capture.urllib.request.urlopen",
        lambda request, timeout: _Response(snapshot),
    )
    capture = {"episode_id": "episode-1", "state_sequence": 0}

    configured = HttpGraphSnapshotResolver(
        "http://example.invalid",
        instruction="test",
        suite="spatial",
        geometry_rules=custom,
    )
    result = configured.resolve(rgb, capture=capture)
    assert result.graph["relation_revision"] == geometric_relation_revision(custom)

    serialized = HttpGraphSnapshotResolver(
        "http://example.invalid",
        instruction="test",
        suite="spatial",
        geometry_rules=snapshot["graph"]["rules"],
    )
    assert serialized.resolve(rgb, capture=capture).graph["rules_sha256"] == snapshot["graph"]["rules_sha256"]


def test_http_capture_preserves_default_contract(monkeypatch) -> None:
    rgb, snapshot = _snapshot(size=256, geometry_rules=None)
    monkeypatch.setattr(
        "samgraph_core.centroid_capture.urllib.request.urlopen",
        lambda request, timeout: _Response(snapshot),
    )
    result = HttpGraphSnapshotResolver(
        "http://example.invalid",
        instruction="test",
        suite="spatial",
    ).resolve(rgb, capture={"episode_id": "episode-1", "state_sequence": 0})
    assert result.graph["relation_revision"] == GEOMETRIC_RELATION_RULES_REVISION
    assert result.graph["rules"] == geometric_relation_rules()
