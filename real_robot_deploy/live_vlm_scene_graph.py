"""Live scene-graph generation for real-robot deployment, using a VLM API.

Unlike vla_benchmarking/libero_live_semantic_context.py (which reads
ground-truth object poses from the MuJoCo simulator), there is no privileged
state on the real robot — so the scene graph is derived by sending the
current camera frame to a VLM (vlm_bench.models.*) with the same spatial
scene-graph prompt used in vlm_benchmarking, and parsing its triplet response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from scene_graph_format import Triplet, context_label, format_triplets

VALID_MODES = {"standard", "scene_graph"}


@dataclass
class LiveVLMSceneGraphGenerator:
    vlm: Any  # vlm_bench.models.base.BaseVLM
    objects: list[str]
    scene_graph_format: str = "triplet"
    max_chars: int = 2500
    # Optional tokenizer + budget so `task + suffix` fits the policy's prompt
    # length (mirrors libero_live_semantic_context.LiveSemanticContextGenerator).
    # If tokenizer is None, falls back to a simple character-length cap.
    tokenizer: Any | None = None
    max_prompt_tokens: int = 48
    debug: bool = False
    _last_response: str = field(default="", init=False)

    def query_triplets(self, image: Image.Image, task_hint: str) -> list[Triplet]:
        from vlm_bench.io_utils import parse_triplets
        from vlm_bench.prompts import build_agentview_prompt

        prompt = build_agentview_prompt(self.objects, task_hint)
        response = self.vlm.query(image, prompt)
        self._last_response = response or ""
        return parse_triplets(self._last_response)

    def _build_suffix(self, triplets: list[Triplet], n: int | None = None) -> str:
        if n is not None:
            triplets = triplets[:n]
        text = format_triplets(triplets, fmt=self.scene_graph_format)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "...<truncated>"
        return f"\n{context_label(self.scene_graph_format)}: {text}"

    def suffix_for_frame(self, image: Image.Image, task_hint: str, mode: str) -> str:
        """Query the VLM for `image` and return a task-suffix for `mode`.

        Drops trailing triplets (longest-first) until `task_hint + suffix`
        fits `max_prompt_tokens`, the same truncation strategy used by
        LiveSemanticContextGenerator.fit_suffix(). Falls back to an
        untruncated suffix if no tokenizer is configured.
        """
        mode = mode.strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if mode == "standard":
            return ""

        triplets = self.query_triplets(image, task_hint)

        if self.tokenizer is None:
            return self._build_suffix(triplets)

        def fits(suffix: str) -> bool:
            candidate = task_hint + suffix
            if not candidate.endswith("\n"):
                candidate += "\n"
            n_tokens = len(self.tokenizer(candidate, add_special_tokens=True)["input_ids"])
            return n_tokens <= self.max_prompt_tokens

        for n in range(len(triplets), -1, -1):
            suffix = self._build_suffix(triplets, n=n)
            if fits(suffix):
                return suffix
        return ""
