"""Fresh-RGB mask-centroid arrow capture for the LIBERO controller.

The benchmark service owns camera, depth, calibration, robot telemetry, and
stepping. A resolver operating on the exact clean RGB frame owns visual
identity. This module joins the two without accepting simulator object names,
boxes, sites, poses, goal geometry, or evaluator state.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .contracts import GraphTriplet
from .geometric_graph import (
    GeometricRelationConfig,
    geometric_relation_revision,
    geometric_relation_rules,
    geometric_relation_rules_sha256,
)


CENTROID_ARROW_INPUT_SOURCE = "public_rgb_mask_centroids"
CENTROID_ARROW_CONTRACT = "libero_arrow_v1"
RGBD_FRAME_CONTRACT = "libero_rgbd_frame_v1"


class MaskResolutionError(ValueError):
    """The visual resolver could not produce current admissible endpoint masks."""


@dataclass(frozen=True, slots=True)
class SelectedMasks:
    source_entity_id: str
    destination_entity_id: str
    source_mask: np.ndarray
    destination_mask: np.ndarray
    rgb_sha256: str
    provider: str
    provenance: Mapping[str, Any]
    graph: Mapping[str, Any] | None = None
    graph_overlay_png_base64: str | None = None


class MaskResolver(Protocol):
    def resolve(self, rgb: np.ndarray, *, capture: Mapping[str, Any]) -> SelectedMasks: ...


def rgb_sha256(rgb: np.ndarray) -> str:
    value = np.ascontiguousarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise MaskResolutionError("clean RGB must be HxWx3 uint8")
    return hashlib.sha256(value.tobytes()).hexdigest()


def mask_centroid(mask: np.ndarray, *, shape: tuple[int, int] | None = None) -> tuple[float, float]:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise MaskResolutionError(f"mask has invalid shape {value.shape}; expected {shape}")
    ys, xs = np.nonzero(value)
    if len(xs) == 0:
        raise MaskResolutionError("selected mask is empty")
    return float(xs.mean()), float(ys.mean())


def capture_endpoint_pixels(capture: Mapping[str, Any], decoded_arrow: Any):
    """Return the pixels authoritative for this capture's endpoint mode.

    The rendered arrow is intentionally decoded in every mode so a corrupted
    or mismatched overlay fails closed. In automatic mask-centroid mode, its
    rounded raster endpoints are only a visual transport format: the lossless
    mask means are the authoritative subpixel endpoints used for RGB-D depth
    sampling and deprojection. Historical reference captures do not carry
    mask means and continue to use the decoded arrow command.
    """
    if capture.get("arrow_input_source") != CENTROID_ARROW_INPUT_SOURCE:
        return (tuple(float(value) for value in decoded_arrow.source_xy),
                tuple(float(value) for value in decoded_arrow.target_xy),
                "decoded_arrow")

    def _centroid(name: str) -> tuple[float, float]:
        value = np.asarray(capture.get(name), dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite two-element centroid")
        if np.any(value < 0.0) or np.any(value >= 256.0):
            raise ValueError(f"{name} lies outside the native RGB image")
        return (float(value[0]), float(value[1]))

    source = _centroid("source_mask_centroid_xy")
    target = _centroid("destination_mask_centroid_xy")
    if float(np.linalg.norm(np.asarray(target) - np.asarray(source))) <= 1.0:
        raise ValueError("mask centroids are too close to define an endpoint pair")
    return source, target, "mask_centroid"


def draw_centroid_arrow(
    rgb: np.ndarray,
    source_center: Sequence[float],
    destination_center: Sequence[float],
) -> np.ndarray:
    """Render the controller-compatible adaptive green arrow."""

    canvas = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    start = tuple(int(round(float(value))) for value in source_center)
    end = tuple(int(round(float(value))) for value in destination_center)
    span = float(np.linalg.norm(np.asarray(end, dtype=float) - start))
    if span <= 1.0:
        raise MaskResolutionError("mask centroids are too close to define an arrow")
    width = 1 if span < 32.0 else 2
    head = max(3, min(16, round(0.35 * span))) if span < 32.0 else 16
    angle = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
    wings = [
        (
            round(end[0] - head * np.cos(angle + sign * np.pi / 6)),
            round(end[1] - head * np.sin(angle + sign * np.pi / 6)),
        )
        for sign in (-1, 1)
    ]
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.line([start, end], fill=(0, 166, 107), width=width)
    draw.polygon([end, *wings], fill=(0, 166, 107))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _validate_mask(mask: Any, *, shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    if value.shape != shape or not value.any():
        raise MaskResolutionError(f"{label} mask must be nonempty with native RGB shape {shape}")
    return value


class AutomaticMaskGraphResolver:
    """Initialize from RGB once, select once, and refresh both endpoints per capture.

    ``initialize`` returns a complete public-RGB mask scene. ``refresh``
    segments one selected entity from the current frame. The callbacks keep
    model construction outside this benchmark-independent orchestration.
    """

    def __init__(
        self,
        *,
        instruction: str,
        graph_brain: Any,
        initialize: Callable[[np.ndarray], Mapping[str, Any]],
        refresh: Callable[[np.ndarray, str, Mapping[str, Any]], np.ndarray],
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1 or max_attempts > 3:
            raise ValueError("mask refresh attempts must be in [1, 3]")
        self._instruction = " ".join(str(instruction).split())
        self._brain = graph_brain
        self._initialize = initialize
        self._refresh = refresh
        self._max_attempts = max_attempts
        self._templates: dict[str, Mapping[str, Any]] = {}
        self._selection: Any | None = None

    def _ensure_initialized(self, rgb: np.ndarray) -> None:
        if self._selection is not None:
            return
        scene = self._initialize(np.array(rgb, copy=True))
        instances = scene.get("instances") if isinstance(scene, Mapping) else None
        raw_triplets = scene.get("triplets") if isinstance(scene, Mapping) else None
        production_inputs = scene.get("production_inputs") if isinstance(scene, Mapping) else None
        if not isinstance(instances, list) or len(instances) < 2 or not isinstance(raw_triplets, list):
            raise MaskResolutionError("initializer did not return a complete mask scene")
        if not isinstance(production_inputs, list) or not any("rgb" in str(v).lower() for v in production_inputs):
            raise MaskResolutionError("initializer must declare public RGB production inputs")
        if _declares_privileged_endpoint_input(scene):
            raise MaskResolutionError("initializer scene declares privileged endpoint provenance")
        templates: dict[str, Mapping[str, Any]] = {}
        for item in instances:
            entity_id = str(item.get("instance_id", ""))
            if not entity_id or entity_id in templates:
                raise MaskResolutionError("initializer returned invalid or duplicate entity IDs")
            templates[entity_id] = dict(item)
        triplets = tuple(
            GraphTriplet(
                relation_id=str(item["relation_id"]),
                subject=str(item["subject"]),
                relation=str(item["relation"]),
                object=str(item["object"]),
            )
            for item in raw_triplets
        )
        selection = self._brain.select_triplet(instruction=self._instruction, triplets=triplets)
        if selection.source_entity_id not in templates or selection.destination_entity_id not in templates:
            raise MaskResolutionError("graph brain selected an entity absent from the RGB scene")
        self._templates = templates
        self._selection = selection

    def _refresh_selected(self, rgb: np.ndarray, entity_id: str) -> np.ndarray:
        last_error: Exception | None = None
        for _attempt in range(self._max_attempts):
            try:
                return _validate_mask(
                    self._refresh(np.array(rgb, copy=True), entity_id, self._templates[entity_id]),
                    shape=rgb.shape[:2],
                    label=entity_id,
                )
            except (MaskResolutionError, ValueError, TypeError) as exc:
                last_error = exc
        raise MaskResolutionError(
            f"visual refresh failed for {entity_id!r} after {self._max_attempts} attempts: {last_error}"
        )

    def resolve(self, rgb: np.ndarray, *, capture: Mapping[str, Any]) -> SelectedMasks:
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        self._ensure_initialized(value)
        selection = self._selection
        source = self._refresh_selected(value, selection.source_entity_id)
        destination = self._refresh_selected(value, selection.destination_entity_id)
        return SelectedMasks(
            source_entity_id=selection.source_entity_id,
            destination_entity_id=selection.destination_entity_id,
            source_mask=source,
            destination_mask=destination,
            rgb_sha256=rgb_sha256(value),
            provider="automatic_rgb_graph_sam",
            provenance={
                "graph_brain_called_once": True,
                "graph_model": getattr(selection, "model", None),
                "graph_prompt_revision": getattr(selection, "prompt_revision", None),
                "relation_id": getattr(selection, "relation_id", None),
                "refresh_attempt_limit": self._max_attempts,
                "simulator_semantic_endpoints_consumed": False,
            },
        )


def _decode_mask(value: str) -> np.ndarray:
    try:
        payload = base64.b64decode(value, validate=True)
        return np.asarray(Image.open(io.BytesIO(payload)).convert("L"), dtype=np.uint8) > 0
    except Exception as exc:
        raise MaskResolutionError("graph snapshot contains an invalid mask PNG") from exc


def _encode_mask(mask: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _declares_privileged_endpoint_input(value: Any, *, key: str = "") -> bool:
    """Reject consumed privileged inputs without rejecting explicit false audit flags."""

    normalized_key = key.lower()
    privileged = ("simulator", "oracle", "object_state", "goal_state", "containment_site", "pose")
    if isinstance(value, Mapping):
        return any(_declares_privileged_endpoint_input(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_declares_privileged_endpoint_input(item, key=key) for item in value)
    if normalized_key in {"production_inputs", "inputs", "sources"} and isinstance(value, str):
        return any(token in value.lower() for token in privileged)
    if not any(token in normalized_key for token in privileged):
        return False
    if isinstance(value, bool):
        return value
    if value is None or value == 0 or str(value).strip().lower() in {"", "false", "none", "no"}:
        return False
    return True


class HttpGraphSnapshotResolver:
    """Resolve both current endpoint masks from a live SamGraph graph endpoint."""

    def __init__(
        self,
        url: str,
        *,
        instruction: str,
        suite: str,
        expected_model: str | None = None,
        timeout_s: float = 60.0,
        geometry_rules: GeometricRelationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._url = str(url)
        self._instruction = " ".join(str(instruction).split())
        self._suite = str(suite)
        self._expected_model = str(expected_model).strip() if expected_model else None
        self._timeout_s = float(timeout_s)
        self._geometry_rules = geometry_rules

    def resolve(self, rgb: np.ndarray, *, capture: Mapping[str, Any]) -> SelectedMasks:
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        output = io.BytesIO()
        Image.fromarray(value, mode="RGB").save(output, format="PNG", optimize=False)
        request = urllib.request.Request(
            self._url,
            data=json.dumps({
                "rgb_png_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                "rgb_sha256": rgb_sha256(value),
                "episode_id": capture.get("episode_id"),
                "state_sequence": capture.get("state_sequence"),
                "instruction": self._instruction,
                "suite": self._suite,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                snapshot = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise MaskResolutionError(f"live graph endpoint failed: {exc}") from exc
        digest = rgb_sha256(value)
        if not isinstance(snapshot, Mapping):
            raise MaskResolutionError("live graph response must be an object")
        if snapshot.get("rgb_sha256") != digest:
            raise MaskResolutionError("live graph response does not match the submitted RGB frame")
        # Identical RGB pixels can recur across resets or stationary frames.
        # A hash alone does not prove episode or observation freshness.
        for field in ("episode_id", "state_sequence"):
            if capture.get(field) is None or snapshot.get(field) != capture[field]:
                raise MaskResolutionError(f"live graph response {field} does not match the capture")
        provenance = snapshot.get("provenance", {})
        if not isinstance(provenance, Mapping) or _declares_privileged_endpoint_input(snapshot):
            raise MaskResolutionError("live graph response declares invalid or privileged endpoint provenance")
        actual_model = str(provenance.get("graph_model", ""))
        if (
            self._expected_model is not None
            and actual_model != self._expected_model
            and not actual_model.startswith(self._expected_model + "-")
        ):
            raise MaskResolutionError(
                "live graph response model does not match the configured qualification treatment"
            )
        selection = snapshot.get("selection", {})
        source_id = str(selection.get("source_entity_id", ""))
        destination_id = str(selection.get("destination_entity_id", ""))
        by_id = {str(item.get("instance_id")): item for item in snapshot.get("instances", [])}
        source = by_id.get(source_id)
        destination = by_id.get(destination_id)
        if not source_id or not destination_id or source is None or destination is None:
            raise MaskResolutionError("live graph response lacks selected endpoint entities")
        endpoint_masks: dict[str, np.ndarray] = {}
        for item in (source, destination):
            if not item.get("current_geometry_valid") or item.get("track_state") != "observed":
                raise MaskResolutionError("live graph endpoint returned held or static endpoint geometry")
            endpoint_masks[str(item["instance_id"])] = _decode_mask(item["mask_png_base64"])
        graph = snapshot.get("graph")
        if graph is not None:
            if not isinstance(graph, Mapping):
                raise MaskResolutionError("live graph response graph must be an object")
            if graph.get("rgb_sha256") != digest:
                raise MaskResolutionError("live graph geometry does not match the submitted RGB frame")
            expected_revision = geometric_relation_revision(self._geometry_rules)
            expected_digest = geometric_relation_rules_sha256(self._geometry_rules)
            if graph.get("relation_revision") != expected_revision:
                raise MaskResolutionError(
                    "live graph relation rules do not match the configured revision"
                )
            if graph.get("rules_sha256") != expected_digest:
                raise MaskResolutionError(
                    "live graph relation rules digest differs from the configured contract"
                )
            serialized_rules = graph.get("rules")
            if serialized_rules is not None and not isinstance(serialized_rules, Mapping):
                raise MaskResolutionError("live graph relation rules manifest is invalid")
            if (
                isinstance(serialized_rules, Mapping)
                and serialized_rules != geometric_relation_rules(self._geometry_rules)
            ):
                raise MaskResolutionError(
                    "live graph relation rules manifest differs from the configured contract"
                )
            if graph.get("simulator_geometry_consumed") is not False:
                raise MaskResolutionError("live graph geometry declares simulator provenance")
            selected_pair = graph.get("selected_pair")
            if selected_pair != [source_id, destination_id]:
                raise MaskResolutionError(
                    "live graph selected pair does not match endpoint selection"
                )
            graph_instances = {
                str(item.get("instance_id")): item
                for item in graph.get("instances", [])
                if isinstance(item, Mapping)
            }
            for entity_id in (source_id, destination_id):
                item = graph_instances.get(entity_id)
                if item is None or item.get("track_state") != "observed":
                    raise MaskResolutionError("live graph selected geometry is not current observed data")
                graph_mask = _decode_mask(item.get("mask_png_base64", ""))
                if not np.array_equal(graph_mask, endpoint_masks[entity_id]):
                    raise MaskResolutionError(
                        "live graph selected mask does not match endpoint mask"
                    )
                center = np.asarray(item.get("center_xy"), dtype=np.float64)
                expected_center = np.asarray(mask_centroid(graph_mask), dtype=np.float64)
                if center.shape != (2,) or not np.isfinite(center).all() \
                        or not np.allclose(center, expected_center, atol=1e-6):
                    raise MaskResolutionError(
                        "live graph selected centroid does not match endpoint mask"
                    )
                endpoint_item = source if entity_id == source_id else destination
                if item.get("source_frame_ts") != endpoint_item.get("source_frame_ts"):
                    raise MaskResolutionError(
                        "live graph selected frame timestamp does not match endpoint"
                    )
        return SelectedMasks(
            source_entity_id=source_id,
            destination_entity_id=destination_id,
            source_mask=endpoint_masks[source_id],
            destination_mask=endpoint_masks[destination_id],
            rgb_sha256=digest,
            provider=str(snapshot.get("provider", "samgraph_live_graph")),
            provenance=dict(provenance),
            graph=graph,
            graph_overlay_png_base64=(
                str(snapshot["graph_overlay_png_base64"])
                if snapshot.get("graph_overlay_png_base64") else None
            ),
        )


class CentroidCaptureClient:
    """Benchmark-client facade that composes a fresh RGB-D frame with masks."""

    def __init__(
        self,
        client: Any,
        resolver: MaskResolver,
        *,
        diagnostic_reference: Callable[[Mapping[str, Any], Mapping[str, Any] | None], Mapping[str, Any]] | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._diagnostic_reference = diagnostic_reference

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @staticmethod
    def is_transport_timeout(exc: BaseException) -> bool:
        return type(exc).__name__ == "TransportTimeout"

    def capture(self, camera: str = "agentview", input_mode: str = "mask_centroid_arrow") -> dict[str, Any]:
        if camera != "agentview" or input_mode != "mask_centroid_arrow":
            raise ValueError("centroid capture supports agentview/mask_centroid_arrow only")
        frame = self._client.capture(camera=camera, input_mode="rgbd_frame")
        if frame.get("frame_contract_version") != RGBD_FRAME_CONTRACT:
            raise MaskResolutionError("benchmark service lacks the RGB-D frame contract")
        rgb = np.ascontiguousarray(np.asarray(frame.get("rgb"), dtype=np.uint8))
        digest = rgb_sha256(rgb)
        if digest != frame.get("clean_frame_sha256"):
            raise MaskResolutionError("benchmark clean-frame hash mismatch")
        resolved = self._resolver.resolve(rgb, capture=frame)
        if resolved.rgb_sha256 != digest:
            raise MaskResolutionError("mask resolver used a different RGB frame")
        if _declares_privileged_endpoint_input(resolved.provenance):
            raise MaskResolutionError("mask resolver declares privileged endpoint provenance")
        source_mask = _validate_mask(resolved.source_mask, shape=rgb.shape[:2], label="source")
        destination_mask = _validate_mask(resolved.destination_mask, shape=rgb.shape[:2], label="destination")
        source_center = mask_centroid(source_mask, shape=rgb.shape[:2])
        destination_center = mask_centroid(destination_mask, shape=rgb.shape[:2])
        result = {
            **frame,
            "arrow_contract_version": CENTROID_ARROW_CONTRACT,
            "arrow_rgb": draw_centroid_arrow(rgb, source_center, destination_center),
            "oracle": False,
            "input_source": "mask_centroid_arrow",
            "arrow_input_source": CENTROID_ARROW_INPUT_SOURCE,
            "source_entity_id": resolved.source_entity_id,
            "destination_entity_id": resolved.destination_entity_id,
            "source_mask_centroid_xy": list(source_center),
            "destination_mask_centroid_xy": list(destination_center),
            # The backend persists these lossless masks in capture.json, so
            # endpoint means can be independently checked after a run.
            "source_mask_png_base64": _encode_mask(source_mask),
            "destination_mask_png_base64": _encode_mask(destination_mask),
            "endpoint_provider": resolved.provider,
            "endpoint_provenance": dict(resolved.provenance),
            "graph": dict(resolved.graph) if isinstance(resolved.graph, Mapping) else None,
            "graph_overlay_png_base64": resolved.graph_overlay_png_base64,
            "simulator_semantic_endpoints_consumed": False,
        }
        if self._diagnostic_reference is not None:
            if not isinstance(resolved.graph, Mapping):
                raise MaskResolutionError("geometric graph comparison requires the live graph artifact")
            diagnostic = self._diagnostic_reference(frame, resolved.graph)
            if not isinstance(diagnostic, Mapping):
                raise MaskResolutionError("geometric graph diagnostic must return an object")
            result["geometric_graph_reference"] = diagnostic.get("reference_graph")
            result["geometric_graph_comparison"] = diagnostic.get("comparison")
            result["geometric_graph_diagnostic_provenance"] = {
                "role": "post_prediction_diagnostic_only",
                "production_endpoint_source": "public_rgb_mask_centroids",
            }
        return result


__all__ = [
    "AutomaticMaskGraphResolver",
    "CENTROID_ARROW_INPUT_SOURCE",
    "CentroidCaptureClient",
    "HttpGraphSnapshotResolver",
    "MaskResolutionError",
    "SelectedMasks",
    "draw_centroid_arrow",
    "mask_centroid",
    "rgb_sha256",
]
