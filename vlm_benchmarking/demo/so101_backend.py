from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .http_helpers import safe_child_path, send_file, send_json, send_range_file
from .so101_proxy_demo import PACKAGE_VERSION
from .so101_proxy_demo.proxy.artifacts import ArtifactStore
from .so101_proxy_demo.proxy.dataset import load_episode_index, load_sampled_index
from .so101_proxy_demo.proxy.metadata_signals import load_metadata_frames
from .so101_proxy_demo.proxy.schemas import (
    BBoxFrame,
    EpisodeRecord,
    ProxyFrame,
    SampledFrameRecord,
    read_jsonl,
)
from .so101_proxy_demo.proxy.task_priors import canonical_object_name


STATIC_ROOT = Path(__file__).resolve().parent / "so101"
COMMON_STATIC_ROOT = Path(__file__).resolve().parent / "common"
RELATIONS = {
    "is_left_of",
    "is_right_of",
    "is_in_front_of",
    "is_behind",
    "is_on_top_of",
    "is_below_of",
    "is_inside",
    "contains",
}
INVERSE_RELATIONS = {
    "is_left_of": "is_right_of",
    "is_right_of": "is_left_of",
    "is_in_front_of": "is_behind",
    "is_behind": "is_in_front_of",
    "is_on_top_of": "is_below_of",
    "is_below_of": "is_on_top_of",
    "is_inside": "contains",
    "contains": "is_inside",
}
EDIT_SCHEMA_VERSION = "so101-demo-graph-edits-v2"
REVIEW_SCHEMA_VERSION = "so101-demo-review-status-v1"
REVIEW_STATUSES = {"reviewed", "needs_attention"}
CSV_FIELDS = (
    "task",
    "episode",
    "frame",
    "timestamp",
    "camera",
    "mode",
    "subject",
    "relation",
    "object",
    "edited",
    "original_relation",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_prediction(text: str) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[1] not in RELATIONS:
            continue
        subject = canonical_object_name(parts[0])
        obj = canonical_object_name(parts[2])
        if subject is None or obj is None or subject == obj:
            continue
        triplet = (subject, parts[1], obj)
        if triplet not in seen:
            seen.add(triplet)
            output.append(triplet)
    return output


class StaleGraphEditError(ValueError):
    pass


class StaleReviewStatusError(ValueError):
    pass


class DemoRepository:
    def __init__(
        self,
        config: dict[str, Any],
        artifacts: ArtifactStore,
        *,
        api_prefix: str = "/api",
        allowed_episodes: set[str] | frozenset[str] | None = None,
        graph_output_dir: str | Path | None = None,
        read_only: bool = False,
    ):
        self.config = config
        self.artifacts = artifacts
        self.dataset_root = Path(config["paths"]["so101_dataset"]).resolve()
        self.prediction_root = Path(config["paths"]["gemini_predictions"]).resolve()
        self.api_prefix = "/" + api_prefix.strip("/")
        self.read_only = read_only
        self.graph_output_root = Path(graph_output_dir or "output").resolve()
        self.graph_edit_path = self.graph_output_root / "so101_graph_edits.jsonl"
        self.review_status_path = self.graph_output_root / "so101_review_status.jsonl"
        self._edit_lock = threading.Lock()
        self._review_lock = threading.Lock()
        self.edits: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        self.reviews: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        if not self.read_only:
            self.graph_output_root.mkdir(parents=True, exist_ok=True)
        self.allowed_episodes = (
            frozenset(allowed_episodes) if allowed_episodes else None
        )
        self.episodes = load_episode_index(artifacts.path("index/episodes.jsonl", create_parent=False))
        self.sampled = load_sampled_index(artifacts.path("index/sampled_frames.jsonl", create_parent=False))
        metadata_path = artifacts.path("metadata/frame_signals.jsonl", create_parent=False)
        self.metadata = load_metadata_frames(str(metadata_path)) if metadata_path.exists() else {}
        self._episode_index = {(item.task, item.episode): item for item in self.episodes}
        self._sample_index = {(item.task, item.episode, item.frame, item.camera): item for item in self.sampled}
        self._frames_by_sequence: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for item in self.sampled:
            self._frames_by_sequence[(item.task, item.episode, item.camera)].append(item.frame)
        for frames in self._frames_by_sequence.values():
            frames.sort()
        self.proxy: dict[tuple[str, str, int, str, str], ProxyFrame] = {}
        self.bboxes: dict[tuple[str, str, int, str], BBoxFrame] = {}
        self.predictions: dict[tuple[str, str, int, str], list[tuple[str, str, str]]] = {}
        self.bbox_source: str | None = None
        self.bbox_sources: list[str] = []
        self._load_bboxes()
        self._load_proxy()
        self._load_predictions()
        if not self.read_only:
            self._load_graph_edits()
            self._load_review_status()

    def _load_bboxes(self) -> None:
        for name in ("agent_view.jsonl", "imported.jsonl"):
            path = self.artifacts.path(f"bboxes/{name}", create_parent=False)
            if not path.exists():
                continue
            for value in read_jsonl(path):
                item = BBoxFrame.from_dict(value)
                self.bboxes[item.key()] = item
            self.bbox_sources.append(name)
            break
        wrist_path = self.artifacts.path("bboxes/wrist.jsonl", create_parent=False)
        if wrist_path.exists():
            for value in read_jsonl(wrist_path):
                item = BBoxFrame.from_dict(value)
                if item.camera == "wrist":
                    self.bboxes[item.key()] = item
            self.bbox_sources.append("wrist.jsonl")
        self.bbox_source = ", ".join(self.bbox_sources) if self.bbox_sources else None

    def _load_proxy(self) -> None:
        for path in sorted((self.artifacts.root / "proxy_graphs").glob("*/*.jsonl")):
            for value in read_jsonl(path):
                item = ProxyFrame.from_dict(value)
                self.proxy[item.key()] = item

    def _load_predictions(self) -> None:
        if not self.prediction_root.exists():
            return
        for path in sorted(self.prediction_root.glob("*/json/*.jsonl")):
            for value in read_jsonl(path):
                task = str(value.get("task", ""))
                episode = str(value.get("demo", ""))
                camera = str(value.get("camera", path.parent.parent.name))
                frame = int(value.get("frame", 0))
                self.predictions[(task, episode, frame, camera)] = _parse_prediction(
                    str(value.get("response", ""))
                )

    def _load_graph_edits(self) -> None:
        if not self.graph_edit_path.exists():
            return
        for value in read_jsonl(self.graph_edit_path):
            try:
                key = self._graph_key(
                    str(value["task"]),
                    str(value["episode"]),
                    int(value["frame"]),
                    str(value["camera"]),
                    str(value.get("mode", "gt")),
                )
                relations = self._coerce_relation_rows(list(value.get("relations", [])))
            except (KeyError, TypeError, ValueError):
                continue
            base_graph_hash = str(value.get("base_graph_hash") or self._base_graph_hash_for_key(key))
            revision = int(value.get("edit_revision", 0) or 0)
            errors = self._validate_edit_relations(key, relations, base_graph_hash)
            self.edits[key] = {
                "schema_version": EDIT_SCHEMA_VERSION,
                "task": key[0],
                "episode": key[1],
                "frame": key[2],
                "camera": key[3],
                "mode": key[4],
                "base_graph_hash": base_graph_hash,
                "edit_revision": revision,
                "relations": relations,
                "updated_at": str(value.get("updated_at", "")),
                "validation_status": "invalid" if errors else "valid",
                "validation_errors": errors,
            }

    def _load_review_status(self) -> None:
        if not self.review_status_path.exists():
            return
        for value in read_jsonl(self.review_status_path):
            try:
                key = self._graph_key(
                    str(value["task"]),
                    str(value["episode"]),
                    int(value["frame"]),
                    str(value["camera"]),
                    str(value.get("mode", "gt")),
                )
                self.sampled_frame(key[0], key[1], key[2], key[3])
                review_status = str(value.get("review_status", "")).strip()
                if review_status not in REVIEW_STATUSES:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            self.reviews[key] = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "task": key[0],
                "episode": key[1],
                "frame": key[2],
                "camera": key[3],
                "mode": key[4],
                "base_graph_hash": str(value.get("base_graph_hash") or self._base_graph_hash_for_key(key)),
                "review_status": review_status,
                "reviewed_at": str(value.get("reviewed_at", "")),
                "reviewer": str(value.get("reviewer", "")),
                "note": str(value.get("note", "")),
            }

    @staticmethod
    def _graph_key(task: str, episode: str, frame: int, camera: str, mode: str) -> tuple[str, str, int, str, str]:
        return task, episode, int(frame), camera, mode or "gt"

    def _write_graph_edits_locked(self) -> None:
        if self.read_only:
            raise PermissionError("Graph editing is disabled in online cached-demo mode.")
        self.graph_output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.graph_edit_path.with_name(f".{self.graph_edit_path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for key in sorted(self.edits):
                    handle.write(json.dumps(self.edits[key], sort_keys=True) + "\n")
            temporary.replace(self.graph_edit_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_review_status_locked(self) -> None:
        if self.read_only:
            raise PermissionError("Review status is disabled in online cached-demo mode.")
        self.graph_output_root.mkdir(parents=True, exist_ok=True)
        temporary = self.review_status_path.with_name(f".{self.review_status_path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for key in sorted(self.reviews):
                    handle.write(json.dumps(self.reviews[key], sort_keys=True) + "\n")
            temporary.replace(self.review_status_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _visible_names_for_frame(self, task: str, episode: str, frame: int, camera: str, mode: str) -> set[str]:
        proxy = self.proxy.get(self._graph_key(task, episode, frame, camera, mode))
        if proxy:
            return set(proxy.visible_objects)
        bbox_frame = self.bboxes.get((task, episode, frame, camera))
        if bbox_frame:
            return set(bbox_frame.objects)
        self.sampled_frame(task, episode, frame, camera)
        return set()

    @staticmethod
    def _coerce_relation_row(item: dict[str, Any]) -> dict[str, str]:
        subject = canonical_object_name(str(item.get("subject", ""))) or str(item.get("subject", "")).strip()
        obj = canonical_object_name(str(item.get("object", ""))) or str(item.get("object", "")).strip()
        return {
            "subject": subject,
            "relation": str(item.get("relation", "")).strip(),
            "object": obj,
        }

    def _coerce_relation_rows(self, relations: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            self._coerce_relation_row(item)
            for item in relations
            if isinstance(item, dict)
        ]

    @staticmethod
    def _hash_json(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _base_graph_hash_for_key(self, key: tuple[str, str, int, str, str]) -> str:
        task, episode, frame, camera, mode = key
        proxy = self.proxy.get(key)
        bbox_frame = self.bboxes.get((task, episode, frame, camera))
        if proxy:
            visible_objects = list(proxy.visible_objects)
            relations = [
                {
                    "subject": item.subject,
                    "relation": item.relation,
                    "object": item.object,
                    "source": item.source,
                    "confidence": item.confidence,
                    "metadata_gates": list(item.metadata_gates),
                    "evidence": item.evidence,
                }
                for item in proxy.relations
            ]
            bboxes = {name: item.to_dict() for name, item in proxy.bboxes.items()}
            model_version = proxy.model_version
        else:
            visible_objects = sorted(bbox_frame.objects) if bbox_frame else []
            relations = []
            bboxes = {name: item.to_dict() for name, item in bbox_frame.objects.items()} if bbox_frame else {}
            model_version = "none"
        return self._hash_json(
            {
                "task": task,
                "episode": episode,
                "frame": frame,
                "camera": camera,
                "mode": mode,
                "visible_objects": visible_objects,
                "bboxes": bboxes,
                "relations": relations,
                "model_version": model_version,
            }
        )

    def _validate_edit_relations(
        self,
        key: tuple[str, str, int, str, str],
        relations: list[dict[str, str]],
        base_graph_hash: str | None = None,
    ) -> list[str]:
        task, episode, frame, camera, mode = key
        errors: list[str] = []
        try:
            visible = self._visible_names_for_frame(task, episode, frame, camera, mode)
        except (KeyError, ValueError) as exc:
            visible = set()
            errors.append(str(exc))
        if base_graph_hash:
            current_hash = self._base_graph_hash_for_key(key)
            if base_graph_hash != current_hash:
                errors.append("Edit was created against a stale generated graph.")
        if relations and not visible:
            errors.append("No visible object endpoints are available for this frame.")

        seen_directed: set[tuple[str, str, str]] = set()
        by_unordered_pair: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for index, item in enumerate(relations, 1):
            subject = item.get("subject", "")
            relation = item.get("relation", "")
            obj = item.get("object", "")
            if not subject or not relation or not obj:
                errors.append(f"Relation {index} requires subject, relation, and object.")
                continue
            if relation not in RELATIONS:
                errors.append(f"Relation {index} uses unknown predicate '{relation}'.")
            if subject == obj:
                errors.append(f"Relation {index} uses the same subject and object.")
            if visible and (subject not in visible or obj not in visible):
                errors.append(
                    f"Relation {index} endpoint is not visible in this frame: {subject}, {obj}."
                )
            triplet = (subject, relation, obj)
            if triplet in seen_directed:
                errors.append(
                    f"Duplicate directed triplet: {subject}, {relation}, {obj}."
                )
            seen_directed.add(triplet)
            by_unordered_pair[tuple(sorted((subject, obj)))].append(triplet)

        for pair, pair_rows in sorted(by_unordered_pair.items()):
            if len(pair_rows) != 2:
                errors.append(
                    f"Object pair {pair[0]} / {pair[1]} must have exactly two inverse directed triplets."
                )
                continue
            for subject, relation, obj in pair_rows:
                inverse = (obj, INVERSE_RELATIONS.get(relation, ""), subject)
                if inverse not in pair_rows:
                    errors.append(
                        f"Relation {subject}, {relation}, {obj} is missing inverse {inverse[0]}, {inverse[1]}, {inverse[2]}."
                    )
        return errors

    def _normalize_edit_pairs(
        self,
        task: str,
        episode: str,
        frame: int,
        camera: str,
        mode: str,
        pairs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        key = self._graph_key(task, episode, frame, camera, mode)
        visible = self._visible_names_for_frame(task, episode, frame, camera, mode)
        output: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for index, item in enumerate(pairs, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Pair {index} must be an object")
            row = self._coerce_relation_row(item)
            subject = row["subject"]
            relation = row["relation"]
            obj = row["object"]
            if not subject or not obj or not relation:
                raise ValueError(f"Pair {index} requires subject, relation, and object")
            if relation not in RELATIONS:
                raise ValueError(f"Pair {index} uses unknown predicate '{relation}'")
            if subject == obj:
                raise ValueError(f"Pair {index} uses the same subject and object")
            if visible and (subject not in visible or obj not in visible):
                raise ValueError(
                    f"Pair {index} endpoint is not visible in this frame: {subject}, {obj}"
                )
            pair_key = tuple(sorted((subject, obj)))
            if pair_key in seen_pairs:
                raise ValueError(f"Object pair {subject} / {obj} appears more than once")
            seen_pairs.add(pair_key)
            output.append({"subject": subject, "relation": relation, "object": obj})
            output.append({"subject": obj, "relation": INVERSE_RELATIONS[relation], "object": subject})
        errors = self._validate_edit_relations(key, output)
        if errors:
            raise ValueError("; ".join(errors))
        return output

    def _normalize_edit_relations(
        self,
        task: str,
        episode: str,
        frame: int,
        camera: str,
        mode: str,
        relations: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        key = self._graph_key(task, episode, frame, camera, mode)
        for index, item in enumerate(relations, 1):
            if not isinstance(item, dict):
                raise ValueError(f"Relation {index} must be an object")
        output = self._coerce_relation_rows(relations)
        errors = self._validate_edit_relations(key, output)
        if errors:
            raise ValueError("; ".join(errors))
        return output

    def _edited_proxy_relations(self, proxy: ProxyFrame | None, edit: dict[str, Any] | None) -> list[dict[str, Any]]:
        original = [item.to_dict() for item in proxy.relations] if proxy else []
        if edit is None:
            return original
        original_by_pair = {
            (item["subject"], item["object"]): item
            for item in original
        }
        output = []
        for item in edit["relations"]:
            base = dict(original_by_pair.get((item["subject"], item["object"]), {}))
            base.update(
                {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                    "source": "manual_edit",
                    "confidence": float(base.get("confidence", 1.0)),
                    "metadata_gates": list(base.get("metadata_gates", [])),
                    "evidence": dict(base.get("evidence", {})),
                }
            )
            output.append(base)
        return output

    @staticmethod
    def _graph_pairs_from_relations(
        relations: list[dict[str, Any]],
        original_relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        original_by_direction = {
            (item["subject"], item["object"]): item.get("relation", "")
            for item in original_relations
        }
        by_direction = {
            (item["subject"], item["object"]): item
            for item in relations
        }
        pairs: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for item in relations:
            subject = item["subject"]
            obj = item["object"]
            pair_key = tuple(sorted((subject, obj)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            inverse = by_direction.get((obj, subject))
            pairs.append(
                {
                    "subject": subject,
                    "relation": item["relation"],
                    "object": obj,
                    "inverse_relation": inverse.get("relation", "") if inverse else "",
                    "edited": item.get("source") == "manual_edit"
                    or (inverse is not None and inverse.get("source") == "manual_edit"),
                    "original_relation": original_by_direction.get((subject, obj), ""),
                    "original_inverse_relation": original_by_direction.get((obj, subject), ""),
                }
            )
        return pairs

    def _invalid_graph_edits(self) -> list[dict[str, Any]]:
        invalid: list[dict[str, Any]] = []
        for key, edit in sorted(self.edits.items()):
            errors = self._validate_edit_relations(
                key,
                list(edit.get("relations", [])),
                str(edit.get("base_graph_hash", "")),
            )
            edit["validation_errors"] = errors
            edit["validation_status"] = "invalid" if errors else "valid"
            if errors:
                invalid.append(
                    {
                        "task": key[0],
                        "episode": key[1],
                        "frame": key[2],
                        "camera": key[3],
                        "mode": key[4],
                        "errors": errors,
                    }
                )
        return invalid

    def _review_payload_for_key(self, key: tuple[str, str, int, str, str]) -> dict[str, Any]:
        review = self.reviews.get(key)
        if review is None:
            return {
                "review_status": "unreviewed",
                "reviewed_at": "",
                "reviewer": "",
                "review_note": "",
                "stale_review": False,
            }
        stale = bool(review.get("base_graph_hash") and review.get("base_graph_hash") != self._base_graph_hash_for_key(key))
        return {
            "review_status": str(review.get("review_status", "unreviewed")),
            "reviewed_at": str(review.get("reviewed_at", "")),
            "reviewer": str(review.get("reviewer", "")),
            "review_note": str(review.get("note", "")),
            "stale_review": stale,
        }

    def _stale_reviews(self) -> list[dict[str, Any]]:
        stale: list[dict[str, Any]] = []
        for key, review in sorted(self.reviews.items()):
            current_hash = self._base_graph_hash_for_key(key)
            if str(review.get("base_graph_hash", "")) != current_hash:
                stale.append(
                    {
                        "task": key[0],
                        "episode": key[1],
                        "frame": key[2],
                        "camera": key[3],
                        "mode": key[4],
                        "review_base_graph_hash": str(review.get("base_graph_hash", "")),
                        "base_graph_hash": current_hash,
                    }
                )
        return stale

    def _all_worklist_keys(self) -> list[tuple[str, str, int, str, str]]:
        sampled_keys = {
            self._graph_key(item.task, item.episode, item.frame, item.camera, "gt")
            for item in self.sampled
            if self.allowed_episodes is None or item.episode in self.allowed_episodes
        }
        return sorted(sampled_keys | set(self.proxy) | set(self.edits) | set(self.reviews))

    def _worklist_entry(self, key: tuple[str, str, int, str, str]) -> dict[str, Any]:
        edit = self.edits.get(key)
        edit_errors = (
            self._validate_edit_relations(
                key,
                list(edit.get("relations", [])),
                str(edit.get("base_graph_hash", "")),
            )
            if edit
            else []
        )
        if edit is not None:
            edit["validation_errors"] = edit_errors
            edit["validation_status"] = "invalid" if edit_errors else "valid"
        review = self._review_payload_for_key(key)
        task, episode, frame, camera, mode = key
        return {
            "task": task,
            "episode": episode,
            "frame": frame,
            "camera": camera,
            "mode": mode,
            "proxy_available": key in self.proxy,
            "bbox_available": (task, episode, frame, camera) in self.bboxes,
            "base_graph_hash": self._base_graph_hash_for_key(key),
            "edited": edit is not None,
            "edit_revision": int(edit.get("edit_revision", 0)) if edit else 0,
            "validation_status": "invalid" if edit_errors else "valid",
            "validation_errors": edit_errors,
            **review,
        }

    def _progress_counts(
        self,
        *,
        invalid_edit_count: int | None = None,
        stale_review_count: int | None = None,
    ) -> dict[str, Any]:
        keys = set(self._all_worklist_keys())
        reviewed = sum(
            1
            for key, review in self.reviews.items()
            if key in keys and review.get("review_status") == "reviewed"
        )
        needs_attention = sum(
            1
            for key, review in self.reviews.items()
            if key in keys and review.get("review_status") == "needs_attention"
        )
        edited = sum(1 for key in self.edits if key in keys)
        invalid = invalid_edit_count if invalid_edit_count is not None else len(self._invalid_graph_edits())
        stale_review = stale_review_count if stale_review_count is not None else len(self._stale_reviews())
        return {
            "frames": len(keys),
            "reviewed_frames": reviewed,
            "needs_attention_frames": needs_attention,
            "unreviewed_frames": max(0, len(keys) - reviewed - needs_attention),
            "edited_frames": edited,
            "invalid_edit_frames": invalid,
            "stale_review_frames": stale_review,
            "remaining_frames": max(0, len(keys) - reviewed),
        }

    def worklist(
        self,
        *,
        task: str | None = None,
        episode: str | None = None,
        camera: str | None = None,
        mode: str | None = None,
        edit_status: str = "all",
        review_status: str = "all",
        validation_status: str = "all",
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for key in self._all_worklist_keys():
            if task and key[0] != task:
                continue
            if episode and key[1] != episode:
                continue
            if camera and key[3] != camera:
                continue
            if mode and key[4] != mode:
                continue
            entry = self._worklist_entry(key)
            if edit_status == "edited" and not entry["edited"]:
                continue
            if edit_status == "generated" and entry["edited"]:
                continue
            if review_status != "all" and entry["review_status"] != review_status:
                continue
            if validation_status == "invalid" and entry["validation_status"] != "invalid":
                continue
            if validation_status == "valid" and entry["validation_status"] != "valid":
                continue
            if validation_status == "stale" and not entry["stale_review"]:
                continue
            items.append(entry)
        return {
            "items": items,
            "count": len(items),
            "summary": self._progress_counts(),
            "filters": {
                "task": task or "",
                "episode": episode or "",
                "camera": camera or "",
                "mode": mode or "",
                "edit_status": edit_status,
                "review_status": review_status,
                "validation_status": validation_status,
            },
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _report_paths(self) -> dict[str, str]:
        report_dir = self.artifacts.root / "reports"
        if not report_dir.exists():
            return {}
        return {
            path.name: str(path.resolve())
            for path in sorted(report_dir.glob("*.json"))
        }

    def pipeline_status(self, *, include_hashes: bool = False) -> dict[str, Any]:
        report_paths = self._report_paths()
        source_artifacts = {}
        for relative in (
            "index/episodes.jsonl",
            "index/sampled_frames.jsonl",
            "metadata/frame_signals.jsonl",
            "bboxes/agent_view.jsonl",
            "bboxes/wrist.jsonl",
            "proxy_graphs/gt/agent_view.jsonl",
            "proxy_graphs/gt/wrist.jsonl",
        ):
            path = self.artifacts.path(relative, create_parent=False)
            source_artifacts[relative] = {
                "path": str(path),
                "exists": path.exists(),
                "sha256": self._file_sha256(path) if include_hashes else "",
            }
        coverage = self._coverage_by_camera()
        return {
            "generated_at": _now_utc(),
            "coverage_by_camera": coverage,
            "wrist_complete": bool(coverage.get("wrist", {}).get("complete", False)),
            "wrist_status": coverage.get("wrist", {}).get("status", "missing"),
            "wrist_gap": max(
                0,
                int(coverage.get("wrist", {}).get("sampled_frames", 0))
                - int(coverage.get("wrist", {}).get("proxy_frames", 0)),
            ),
            "graph_edit_path": str(self.graph_edit_path),
            "review_status_path": str(self.review_status_path),
            "report_paths": report_paths,
            "source_artifacts": source_artifacts,
            "progress": self._progress_counts(),
            "stale_reviews": self._stale_reviews(),
        }

    def save_review_status(
        self,
        task: str,
        episode: str,
        frame: int,
        camera: str,
        mode: str,
        *,
        base_graph_hash: str,
        review_status: str,
        reviewer: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("Review status is disabled in online cached-demo mode.")
        key = self._graph_key(task, episode, frame, camera, mode)
        self.sampled_frame(task, episode, frame, camera)
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review_status '{review_status}'")
        if not base_graph_hash:
            raise ValueError("base_graph_hash is required")
        current_hash = self._base_graph_hash_for_key(key)
        if base_graph_hash != current_hash:
            raise StaleReviewStatusError("Generated graph changed; reload this frame before saving review status.")
        with self._review_lock:
            self.reviews[key] = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "task": key[0],
                "episode": key[1],
                "frame": key[2],
                "camera": key[3],
                "mode": key[4],
                "base_graph_hash": base_graph_hash,
                "review_status": review_status,
                "reviewed_at": _now_utc(),
                "reviewer": reviewer.strip(),
                "note": note.strip(),
            }
            self._write_review_status_locked()
        return self.frame_payload(task, episode, frame, camera, mode)

    def reset_review_status(self, task: str, episode: str, frame: int, camera: str, mode: str) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("Review status is disabled in online cached-demo mode.")
        key = self._graph_key(task, episode, frame, camera, mode)
        self.sampled_frame(task, episode, frame, camera)
        with self._review_lock:
            self.reviews.pop(key, None)
            self._write_review_status_locked()
        return self.frame_payload(task, episode, frame, camera, mode)

    def save_graph_edit(
        self,
        task: str,
        episode: str,
        frame: int,
        camera: str,
        mode: str,
        *,
        base_graph_hash: str,
        pairs: list[dict[str, Any]] | None = None,
        relations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("Graph editing is disabled in online cached-demo mode.")
        key = self._graph_key(task, episode, frame, camera, mode)
        if not base_graph_hash:
            raise ValueError("base_graph_hash is required")
        current_hash = self._base_graph_hash_for_key(key)
        if base_graph_hash != current_hash:
            raise StaleGraphEditError("Generated graph changed; reload this frame before saving edits.")
        if pairs is not None:
            normalized = self._normalize_edit_pairs(task, episode, frame, camera, mode, pairs)
        else:
            normalized = self._normalize_edit_relations(task, episode, frame, camera, mode, relations or [])
        validation_errors = self._validate_edit_relations(key, normalized, base_graph_hash)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        previous_revision = int(self.edits.get(key, {}).get("edit_revision", 0))
        with self._edit_lock:
            self.edits[key] = {
                "schema_version": EDIT_SCHEMA_VERSION,
                "task": key[0],
                "episode": key[1],
                "frame": key[2],
                "camera": key[3],
                "mode": key[4],
                "base_graph_hash": base_graph_hash,
                "edit_revision": previous_revision + 1,
                "relations": normalized,
                "updated_at": _now_utc(),
                "validation_status": "valid",
                "validation_errors": [],
            }
            self._write_graph_edits_locked()
        return self.frame_payload(task, episode, frame, camera, mode)

    def reset_graph_edit(self, task: str, episode: str, frame: int, camera: str, mode: str) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("Graph editing is disabled in online cached-demo mode.")
        key = self._graph_key(task, episode, frame, camera, mode)
        with self._edit_lock:
            self.edits.pop(key, None)
            self._write_graph_edits_locked()
        return self.frame_payload(task, episode, frame, camera, mode)

    def _export_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        keys = sorted(set(self.proxy) | set(self.edits))
        for key in keys:
            task, episode, frame, camera, mode = key
            proxy = self.proxy.get(key)
            edit = self.edits.get(key)
            relations = self._edited_proxy_relations(proxy, edit)
            original_by_direction = {
                (item.subject, item.object): item.relation
                for item in proxy.relations
            } if proxy else {}
            timestamp = proxy.timestamp if proxy else self.sampled_frame(task, episode, frame, camera).timestamp
            metadata_reliable = proxy.metadata_reliable if proxy else False
            gripper_phase = proxy.gripper_phase if proxy else "unknown"
            base_graph_hash = str(edit.get("base_graph_hash", "")) if edit else self._base_graph_hash_for_key(key)
            validation_errors = list(edit.get("validation_errors", [])) if edit else []
            review = self._review_payload_for_key(key)
            for relation in relations:
                original_relation = original_by_direction.get(
                    (relation["subject"], relation["object"]),
                    "",
                )
                row_edited = bool(edit) and (
                    not original_relation or original_relation != relation["relation"]
                )
                rows.append(
                    {
                        "task": task,
                        "episode": episode,
                        "frame": frame,
                        "timestamp": timestamp,
                        "camera": camera,
                        "mode": mode,
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"],
                        "source": relation.get("source", ""),
                        "confidence": relation.get("confidence", ""),
                        "metadata_reliable": metadata_reliable,
                        "gripper_phase": gripper_phase,
                        "edited": "yes" if row_edited else "no",
                        "original_relation": original_relation if row_edited else "",
                        "base_graph_hash": base_graph_hash,
                        "validation_errors": " | ".join(validation_errors),
                        "review_status": review["review_status"],
                        "reviewed_at": review["reviewed_at"],
                        "reviewer": review["reviewer"],
                        "note": review["review_note"],
                    }
                )
        return rows

    def _coverage_by_camera(self) -> dict[str, dict[str, Any]]:
        cameras = sorted(
            {item.camera for item in self.sampled}
            | {key[3] for key in self.proxy}
            | {key[3] for key in self.bboxes}
        )
        coverage: dict[str, dict[str, Any]] = {}
        for camera in cameras:
            sampled = sum(1 for item in self.sampled if item.camera == camera)
            bbox = sum(1 for key in self.bboxes if key[3] == camera)
            proxy_by_mode = {
                mode: sum(1 for key in self.proxy if key[3] == camera and key[4] == mode)
                for mode in sorted({key[4] for key in self.proxy if key[3] == camera})
            }
            proxy_total = max(proxy_by_mode.values(), default=0)
            complete = sampled > 0 and proxy_total >= sampled
            coverage[camera] = {
                "sampled_frames": sampled,
                "bbox_frames": bbox,
                "proxy_frames": proxy_total,
                "proxy_frames_by_mode": proxy_by_mode,
                "complete": complete,
                "status": "complete" if complete else "partial",
            }
        return coverage

    def _write_json_artifact(self, path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _unique_export_dir(self, export_root: Path, timestamp: str) -> Path:
        export_root.mkdir(parents=True, exist_ok=True)
        candidate = export_root / timestamp
        suffix = 1
        while candidate.exists():
            candidate = export_root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        return candidate

    def _write_export_dir(
        self,
        export_dir: Path,
        rows: list[dict[str, Any]],
        *,
        exported_at: str,
        timestamp: str,
    ) -> dict[str, Any]:
        export_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["camera"])].append(row)

        files: list[dict[str, Any]] = []
        csv_outputs: list[tuple[Path, list[dict[str, Any]], str]] = [
            (export_dir / f"{camera}.csv", group_rows, camera)
            for camera, group_rows in sorted(grouped.items())
        ]

        for path, csv_rows, camera in csv_outputs:
            temporary = path.with_name(f".{path.name}.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(csv_rows)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            files.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "rows": len(csv_rows),
                    "camera": camera,
                }
            )

        return {
            "exported_at": exported_at,
            "timestamp": timestamp,
            "output_dir": str(export_dir),
            "files": files,
            "frames": len(set((row["task"], row["episode"], row["frame"], row["camera"], row["mode"]) for row in rows)),
            "edited_frames": len({
                (row["task"], row["episode"], row["frame"], row["camera"], row["mode"])
                for row in rows
                if str(row["edited"]).lower() == "yes"
            }),
            "reviewed_frames": len({
                (row["task"], row["episode"], row["frame"], row["camera"], row["mode"])
                for row in rows
                if row["review_status"] == "reviewed"
            }),
            "needs_attention_frames": len({
                (row["task"], row["episode"], row["frame"], row["camera"], row["mode"])
                for row in rows
                if row["review_status"] == "needs_attention"
            }),
            "rows": len(rows),
            "schema": list(CSV_FIELDS),
        }

    def export_graph_csvs(self) -> dict[str, Any]:
        if self.read_only:
            raise PermissionError("CSV export is disabled in online cached-demo mode.")
        invalid = self._invalid_graph_edits()
        if invalid:
            first = invalid[0]
            raise ValueError(
                "Cannot export CSVs because saved graph edits are invalid: "
                f"{first['task']}/{first['episode']}/{first['camera']}/frame {first['frame']} "
                f"({'; '.join(first['errors'])})"
            )
        stale_reviews = self._stale_reviews()
        if stale_reviews:
            first = stale_reviews[0]
            raise ValueError(
                "Cannot export CSVs because saved review status is stale: "
                f"{first['task']}/{first['episode']}/{first['camera']}/frame {first['frame']}"
            )
        exported_at = _now_utc()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_root = self.graph_output_root / "annotated_graphs"
        export_dir = self._unique_export_dir(export_root, timestamp)
        rows = self._export_rows()
        return self._write_export_dir(
            export_dir,
            rows,
            exported_at=exported_at,
            timestamp=timestamp,
        )

    def health(self) -> dict[str, Any]:
        modes = sorted({key[4] for key in self.proxy})
        invalid_edits = self._invalid_graph_edits()
        stale_reviews = self._stale_reviews()
        progress = self._progress_counts(
            invalid_edit_count=len(invalid_edits),
            stale_review_count=len(stale_reviews),
        )
        pipeline = self.pipeline_status()
        return {
            "name": "SO101 Demo",
            "version": PACKAGE_VERSION,
            "tasks": len({item.task for item in self.episodes}),
            "episodes": len(self.episodes),
            "sampled_frames": len(self.sampled),
            "sampled_per_camera": {
                camera: sum(1 for item in self.sampled if item.camera == camera)
                for camera in ("agent_view", "wrist")
            },
            "bbox_frames": len(self.bboxes),
            "bbox_source": self.bbox_source,
            "bbox_sources": list(self.bbox_sources),
            "proxy_frames": len(self.proxy),
            "proxy_modes": modes,
            "prediction_frames": len(self.predictions),
            "metadata_frames": len(self.metadata),
            "graph_editing_enabled": not self.read_only,
            "graph_export_enabled": not self.read_only,
            "graph_edits": len(self.edits),
            "graph_invalid_edits": len(invalid_edits),
            "graph_edit_path": str(self.graph_edit_path),
            "graph_output_root": str(self.graph_output_root),
            "graph_export_root": str(self.graph_output_root / "annotated_graphs"),
            "review_status_enabled": not self.read_only,
            "review_records": len(self.reviews),
            "reviewed_frames": progress["reviewed_frames"],
            "needs_attention_frames": progress["needs_attention_frames"],
            "unreviewed_frames": progress["unreviewed_frames"],
            "stale_review_frames": len(stale_reviews),
            "review_status_path": str(self.review_status_path),
            "pipeline_status": {
                "wrist_complete": pipeline["wrist_complete"],
                "wrist_status": pipeline["wrist_status"],
                "wrist_gap": pipeline["wrist_gap"],
                "report_paths": pipeline["report_paths"],
            },
            "demo_mode": "online" if self.read_only else "offline",
            "coverage_by_camera": self._coverage_by_camera(),
        }

    def tasks(self) -> list[dict[str, str]]:
        proxy_tasks = {key[0] for key in self.proxy if key[3] == "agent_view"}
        names = sorted({item.task for item in self.episodes}, key=lambda name: (name not in proxy_tasks, name))
        return [{"id": name, "name": name.replace("-", " ")} for name in names]

    def episodes_for(self, task: str) -> list[dict[str, Any]]:
        output = []
        proxy_episodes = {
            key[1] for key in self.proxy if key[0] == task and key[3] == "agent_view"
        }
        for item in sorted(
            (
                episode
                for episode in self.episodes
                if episode.task == task
                and (
                    self.allowed_episodes is None
                    or episode.episode in self.allowed_episodes
                )
            ),
            key=lambda value: (value.episode not in proxy_episodes, value.episode_index),
        ):
            output.append(
                {
                    "id": item.episode,
                    "name": f"{item.episode} ({item.length} frames)",
                    "length": item.length,
                    "fps": item.fps,
                    "sampled_frames": len(self._frames_by_sequence.get((task, item.episode, "agent_view"), [])),
                }
            )
        return output

    def frames_for(self, task: str, episode: str, camera: str) -> list[int]:
        return self._frames_by_sequence.get((task, episode, camera), [])

    def sampled_frame(self, task: str, episode: str, frame: int, camera: str) -> SampledFrameRecord:
        try:
            return self._sample_index[(task, episode, frame, camera)]
        except KeyError as exc:
            raise KeyError(f"Unknown sampled frame {task}/{episode}/{camera}/{frame}") from exc

    def episode(self, task: str, episode: str) -> EpisodeRecord:
        try:
            return self._episode_index[(task, episode)]
        except KeyError as exc:
            raise KeyError(f"Unknown episode {task}/{episode}") from exc

    def frame_payload(self, task: str, episode: str, frame: int, camera: str, mode: str) -> dict[str, Any]:
        sample = self.sampled_frame(task, episode, frame, camera)
        proxy = self.proxy.get((task, episode, frame, camera, mode))
        bbox_frame = self.bboxes.get((task, episode, frame, camera))
        metadata = self.metadata.get((task, episode, frame))
        predictions = self.predictions.get((task, episode, frame, camera), [])
        edit_key = self._graph_key(task, episode, frame, camera, mode)
        edit = self.edits.get(edit_key)
        proxy_relations = self._edited_proxy_relations(proxy, edit)
        original_relations = [item.to_dict() for item in proxy.relations] if proxy else []
        base_graph_hash = self._base_graph_hash_for_key(edit_key)
        validation_errors = (
            self._validate_edit_relations(
                edit_key,
                list(edit.get("relations", [])),
                str(edit.get("base_graph_hash", "")),
            )
            if edit
            else []
        )
        if edit is not None:
            edit["validation_errors"] = validation_errors
            edit["validation_status"] = "invalid" if validation_errors else "valid"
        review = self._review_payload_for_key(edit_key)
        proxy_set = {
            (str(item["subject"]), str(item["relation"]), str(item["object"]))
            for item in proxy_relations
        }
        predicted = [
            {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "correct": (subject, relation, obj) in proxy_set if proxy else None,
            }
            for subject, relation, obj in predictions
        ]
        # Proxy records retain smoothed geometry used by the rule engine. The
        # synchronized detector stream is the correct geometry for video overlays.
        bboxes = dict(bbox_frame.objects if bbox_frame else (proxy.bboxes if proxy else {}))
        tp = sum(1 for item in predicted if item["correct"] is True)
        fp = sum(1 for item in predicted if item["correct"] is False)
        fn = max(0, len(proxy_set) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        episode_record = self.episode(task, episode)
        video = episode_record.videos.get(camera, {})
        query = urlencode({"task": task, "episode": episode, "camera": camera})
        return {
            "task": task,
            "episode": episode,
            "frame": frame,
            "camera": camera,
            "mode": mode,
            "width": sample.width,
            "height": sample.height,
            "image_url": f"{self.api_prefix}/image?{urlencode({'task': task, 'episode': episode, 'frame': frame, 'camera': camera})}",
            "video_url": f"{self.api_prefix}/video?{query}" if video.get("exists") else None,
            "video_start": float(video.get("from_timestamp", 0.0)),
            "video_end": float(video.get("to_timestamp", 0.0)),
            "fps": episode_record.fps,
            "proxy_available": proxy is not None or edit is not None,
            "prediction_available": bool(predictions),
            "editable": not self.read_only and (proxy is not None or edit is not None),
            "manual_edit": edit is not None,
            "manual_edit_updated_at": str(edit.get("updated_at", "")) if edit else "",
            "base_graph_hash": base_graph_hash,
            "edit_revision": int(edit.get("edit_revision", 0)) if edit else 0,
            "validation_errors": validation_errors,
            "validation_status": "invalid" if validation_errors else "valid",
            "stale_edit": bool(validation_errors and any("stale generated graph" in item for item in validation_errors)),
            **review,
            "visible_objects": list(proxy.visible_objects) if proxy else sorted(bboxes),
            "bboxes": {name: item.to_dict() for name, item in bboxes.items()},
            "proxy_relations": proxy_relations,
            "graph_pairs": self._graph_pairs_from_relations(proxy_relations, original_relations),
            "prediction_relations": predicted,
            "metadata": metadata.to_dict() if metadata else None,
            "metrics": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: "ProxyDemoServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/so101/api" or parsed.path.startswith("/so101/api/"):
                original_path = self.path
                self.path = self.path.removeprefix("/so101")
                try:
                    self.do_GET()
                finally:
                    self.path = original_path
            elif parsed.path == "/":
                self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8", cache=False)
            elif parsed.path in {"/so101", "/so101/"}:
                self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8", cache=False)
            elif parsed.path.startswith("/common/"):
                path = safe_child_path(COMMON_STATIC_ROOT, parsed.path.removeprefix("/common/"))
                self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream", cache=False)
            elif parsed.path.startswith("/so101/"):
                name = parsed.path.removeprefix("/so101/")
                path = safe_child_path(STATIC_ROOT, name)
                self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream", cache=False)
            elif parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                path = safe_child_path(STATIC_ROOT, name)
                self._send_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream", cache=False)
            elif parsed.path == "/api/health":
                self._json(self.server.repository.health())
            elif parsed.path == "/api/tasks":
                self._json({"tasks": self.server.repository.tasks()})
            elif parsed.path == "/api/episodes":
                self._json({"episodes": self.server.repository.episodes_for(self._required(query, "task"))})
            elif parsed.path == "/api/frames":
                self._json(
                    {
                        "frames": self.server.repository.frames_for(
                            self._required(query, "task"),
                            self._required(query, "episode"),
                            self._required(query, "camera"),
                        )
                    }
                )
            elif parsed.path == "/api/worklist":
                self._json(
                    self.server.repository.worklist(
                        task=query.get("task", [""])[0] or None,
                        episode=query.get("episode", [""])[0] or None,
                        camera=query.get("camera", [""])[0] or None,
                        mode=query.get("mode", [""])[0] or None,
                        edit_status=query.get("edit_status", ["all"])[0] or "all",
                        review_status=query.get("review_status", ["all"])[0] or "all",
                        validation_status=query.get("validation_status", ["all"])[0] or "all",
                    )
                )
            elif parsed.path in {"/api/graph-edits", "/api/graph-edits/reset", "/api/review-status/reset", "/api/export-csv"}:
                self._json(
                    {"error": f"Use POST for {parsed.path}."},
                    HTTPStatus.METHOD_NOT_ALLOWED,
                )
            elif parsed.path == "/api/pipeline-status":
                self._json(self.server.repository.pipeline_status())
            elif parsed.path == "/api/review-status":
                if self.server.repository.read_only:
                    raise PermissionError("Review status is disabled in online cached-demo mode.")
                key = self.server.repository._graph_key(
                    self._required(query, "task"),
                    self._required(query, "episode"),
                    int(self._required(query, "frame")),
                    self._required(query, "camera"),
                    query.get("mode", ["gt"])[0],
                )
                self._json(self.server.repository._review_payload_for_key(key))
            elif parsed.path == "/api/frame":
                self._json(
                    self.server.repository.frame_payload(
                        self._required(query, "task"),
                        self._required(query, "episode"),
                        int(self._required(query, "frame")),
                        self._required(query, "camera"),
                        query.get("mode", ["gt"])[0],
                    )
                )
            elif parsed.path == "/api/image":
                sample = self.server.repository.sampled_frame(
                    self._required(query, "task"),
                    self._required(query, "episode"),
                    int(self._required(query, "frame")),
                    self._required(query, "camera"),
                )
                path = Path(sample.image_path)
                if not path.is_absolute():
                    path = self.server.repository.dataset_root / path
                path = path.resolve()
                path.relative_to(self.server.repository.dataset_root)
                self._send_file(path, "image/jpeg", cache=True)
            elif parsed.path == "/api/video":
                episode = self.server.repository.episode(
                    self._required(query, "task"), self._required(query, "episode")
                )
                camera = self._required(query, "camera")
                path = Path(episode.videos[camera]["path"]).resolve()
                path.relative_to(self.server.repository.dataset_root)
                self._send_range_file(path, "video/mp4")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/so101/api" or parsed.path.startswith("/so101/api/"):
                original_path = self.path
                self.path = self.path.removeprefix("/so101")
                try:
                    self.do_POST()
                finally:
                    self.path = original_path
            elif parsed.path == "/api/graph-edits":
                payload = self._read_json_body()
                pairs = payload.get("pairs")
                relations = payload.get("relations", [])
                if pairs is not None and not isinstance(pairs, list):
                    raise ValueError("pairs must be a list")
                if not isinstance(relations, list):
                    raise ValueError("relations must be a list")
                self._json(
                    self.server.repository.save_graph_edit(
                        str(payload["task"]),
                        str(payload["episode"]),
                        int(payload["frame"]),
                        str(payload["camera"]),
                        str(payload.get("mode", "gt")),
                        base_graph_hash=str(payload.get("base_graph_hash", "")),
                        pairs=pairs,
                        relations=relations,
                    )
                )
            elif parsed.path == "/api/graph-edits/reset":
                payload = self._read_json_body()
                self._json(
                    self.server.repository.reset_graph_edit(
                        str(payload["task"]),
                        str(payload["episode"]),
                        int(payload["frame"]),
                        str(payload["camera"]),
                        str(payload.get("mode", "gt")),
                    )
                )
            elif parsed.path == "/api/review-status":
                payload = self._read_json_body()
                self._json(
                    self.server.repository.save_review_status(
                        str(payload["task"]),
                        str(payload["episode"]),
                        int(payload["frame"]),
                        str(payload["camera"]),
                        str(payload.get("mode", "gt")),
                        base_graph_hash=str(payload.get("base_graph_hash", "")),
                        review_status=str(payload.get("review_status", "")),
                        reviewer=str(payload.get("reviewer", "")),
                        note=str(payload.get("note", "")),
                    )
                )
            elif parsed.path == "/api/review-status/reset":
                payload = self._read_json_body()
                self._json(
                    self.server.repository.reset_review_status(
                        str(payload["task"]),
                        str(payload["episode"]),
                        int(payload["frame"]),
                        str(payload["camera"]),
                        str(payload.get("mode", "gt")),
                    )
                )
            elif parsed.path == "/api/export-csv":
                self._read_json_body(required=False)
                self._json(self.server.repository.export_graph_csvs())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except (StaleGraphEditError, StaleReviewStatusError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    @staticmethod
    def _required(query: dict[str, list[str]], key: str) -> str:
        values = query.get(key)
        if not values or not values[0]:
            raise ValueError(f"Missing query parameter '{key}'")
        return values[0]

    def _read_json_body(self, *, required: bool = True) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            if required:
                raise ValueError("Request body is required")
            return {}
        if length > 1024 * 1024:
            raise ValueError("Request body is too large")
        data = self.rfile.read(length)
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object")
        return payload

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        send_json(self, payload, status)

    def _send_file(self, path: Path, content_type: str, *, cache: bool) -> None:
        send_file(self, path, content_type, cache=cache)

    def _send_range_file(self, path: Path, content_type: str) -> None:
        send_range_file(self, path, content_type)


class ProxyDemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], repository: DemoRepository):
        super().__init__(address, DemoRequestHandler)
        self.repository = repository


def serve(
    config: dict[str, Any],
    artifacts: ArtifactStore,
    *,
    host: str | None = None,
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    repository = DemoRepository(config, artifacts)
    settings = config["demo"]
    active_host = host or str(settings["host"])
    active_port = int(port or settings["port"])
    server = ProxyDemoServer((active_host, active_port), repository)
    url = f"http://{active_host}:{active_port}/"
    print(f"SO101 Demo: {url}", flush=True)
    print(f"PID: {__import__('os').getpid()}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
