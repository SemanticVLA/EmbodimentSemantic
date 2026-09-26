"""SAM 3.1-backed SamGraph predictor.

Imports for the gated SAM runtime are intentionally lazy: importing the
benchmark package remains possible on a CPU-only machine, while constructing
this predictor fails clearly when the pinned runtime/checkpoint is unavailable.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


def _encode_png(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(
        output, format="PNG", optimize=False
    )
    return output.getvalue()


class SamGraphSamPredictor:
    """One native SAM runtime and persistent resolver across task archives."""

    def __init__(self, checkpoint_path: str | Path, *, geometry_rules: Mapping[str, Any] | None = None,
                 runtime: Any | None = None, tracker: Any | None = None,
                 segmenter: Any | None = None, resolver: Any | None = None,
                 mode: str = "legacy", task_prompts: Mapping[str, Any] | None = None) -> None:
        if mode not in {"legacy", "automatic"}:
            raise ValueError("predictor mode must be legacy or automatic")
        if geometry_rules is None:
            from samgraph_core.geometry_profiles import samgraph_spatial_mask_geometry
            geometry_rules = samgraph_spatial_mask_geometry()
        if runtime is None:
            from samgraph_core import OfficialSam31Runtime
            # Apply the image-only text detection cutoff used by the all-task
            # localization probe. The runtime restores native video thresholds
            # after each text query; legacy inference stays unchanged.
            runtime = OfficialSam31Runtime(
                checkpoint_path,
                text_detection_threshold=0.2 if mode == "automatic" else None,
            )
        if tracker is None:
            from samgraph_core import LocalSam31SceneTracker
            tracker = LocalSam31SceneTracker(runtime, camera_id="agentview")
        if segmenter is None:
            from samgraph_core import LocalSam31Segmenter
            segmenter = LocalSam31Segmenter(runtime)
        if resolver is None and mode == "legacy":
            from samgraph_core import LiberoArrowResolutionService
            resolver = LiberoArrowResolutionService(
                tracking_service=tracker,
                segmentation_service=segmenter,
                graph_brain=None,
                camera_id="agentview",
                geometry_rules=geometry_rules,
                allow_partial_inventory=True,
            )
        self.runtime = runtime
        self.tracker = tracker
        self.segmenter = segmenter
        self.resolver = resolver
        self.geometry_rules = geometry_rules
        self.mode = mode
        self.task_prompts = task_prompts
        self.automatic = None
        if mode == "automatic":
            from samgraph_core.automatic_scene import AutomaticSceneController
            self.automatic = AutomaticSceneController(
                segmenter, tracker, geometry_rules=geometry_rules, task_prompts=task_prompts,
            )
        self._archive_open = False
        self._demo = "demo_0"

    def on_episode_start(self, task: str, demo: str) -> None:
        self.on_archive_start(task)
        self._demo = demo

    @property
    def provenance(self) -> dict[str, Any]:
        identity = dict(getattr(self.runtime, "model_identity", {}))
        return {"sam": identity, "camera": "agentview", "suite": "spatial",
                "mode": self.mode,
                "task_prompts": self.task_prompts,
                "selected_pair": (["black_bowl_1", "plate_1"] if self.mode == "legacy" else None),
                "prediction_input": "ZIP JPEG -> RGB -> undo_agentview_rotation -> optional_resize -> PNG",
                "geometry_rules": self.geometry_rules}

    def warmup(self) -> None:
        warmup = getattr(self.runtime, "warmup", None)
        if callable(warmup):
            warmup()

    def on_archive_start(self, task: str) -> None:
        # Resolver.close() terminates all native per-archive sessions. Runtime
        # remains shared so model weights are loaded once per process.
        if self._archive_open:
            if self.mode == "automatic":
                self.automatic.close_episode()
            else:
                self.resolver.close()
        self._archive_open = True

    def on_archive_end(self) -> None:
        if self._archive_open:
            if self.mode == "automatic":
                self.automatic.close_episode()
            else:
                self.resolver.close()
            self._archive_open = False

    @staticmethod
    def _triplets(graph: Any) -> list[list[str]]:
        if not isinstance(graph, Mapping):
            return []
        result = []
        for item in graph.get("triplets", []):
            if not isinstance(item, Mapping) or item.get("current_geometry_valid", True) is False:
                continue
            subject, relation, object_ = item.get("subject"), item.get("relation"), item.get("object")
            if all(isinstance(value, str) and value and value != "unknown" for value in (subject, relation, object_)):
                result.append([subject, relation, object_])
        return result

    @staticmethod
    def _masks(graph: Any) -> dict[str, np.ndarray]:
        if not isinstance(graph, Mapping):
            return {}
        from samgraph_core.mask_scene import decode_mask
        masks = {}
        for item in graph.get("instances", []):
            if not isinstance(item, Mapping) or not item.get("current_geometry_valid"):
                continue
            encoded = item.get("mask_png_base64")
            if encoded:
                masks[str(item["instance_id"])] = decode_mask(encoded)
        return masks

    def __call__(self, rgb: np.ndarray, task: str, frame: int) -> dict[str, Any]:
        value = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        encoded = _encode_png(value)
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        episode_id = f"{task}/{self._demo}"
        instruction = " ".join(str(task).replace("_", " ").split())
        if self.mode == "automatic":
            output = (self.automatic.start_episode(
                task=episode_id, instruction=instruction, frame=int(frame), rgb=value,
            ) if frame == 0 else self.automatic.step(frame=int(frame), rgb=value))
            return {**output, "rgb_sha256": digest, "tracking_pair": {},
                    "sam_result_provenance": self.provenance}
        result = self.resolver.resolve(
            episode_id=episode_id,
            state_sequence=int(frame),
            instruction=instruction,
            suite="spatial",
            rgb=value,
            encoded_rgb=encoded,
            rgb_sha256=digest,
            selected_pair=("black_bowl_1", "plate_1"),
        )
        graph = result.get("graph", {})
        tracking_pair = result.get("tracking_pair", {})
        inventory = result.get("inventory", {})
        return {"triplets": self._triplets(graph), "masks": self._masks(graph),
                "graph": graph, "rgb_sha256": digest,
                "geometry_rules": graph.get("rules"),
                "geometry_rules_sha256": graph.get("rules_sha256"),
                "relation_revision": graph.get("relation_revision"),
                "tracking_pair": tracking_pair,
                "inventory": inventory,
                "sam_result_provenance": result.get("provenance", {})}

    def close(self) -> None:
        self.on_archive_end()
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()


__all__ = ["SamGraphSamPredictor"]
