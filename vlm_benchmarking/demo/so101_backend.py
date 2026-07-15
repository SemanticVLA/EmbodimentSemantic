from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from collections import defaultdict
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


class DemoRepository:
    def __init__(
        self,
        config: dict[str, Any],
        artifacts: ArtifactStore,
        *,
        api_prefix: str = "/api",
        allowed_episodes: set[str] | frozenset[str] | None = None,
    ):
        self.config = config
        self.artifacts = artifacts
        self.dataset_root = Path(config["paths"]["so101_dataset"]).resolve()
        self.prediction_root = Path(config["paths"]["gemini_predictions"]).resolve()
        self.api_prefix = "/" + api_prefix.strip("/")
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

    def health(self) -> dict[str, Any]:
        modes = sorted({key[4] for key in self.proxy})
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
        proxy_set = {item.triplet() for item in proxy.relations} if proxy else set()
        predicted = [
            {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "correct": (subject, relation, obj) in proxy_set if proxy else None,
            }
            for subject, relation, obj in predictions
        ]
        proxy_relations = [item.to_dict() for item in proxy.relations] if proxy else []
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
            "proxy_available": proxy is not None,
            "prediction_available": bool(predictions),
            "visible_objects": list(proxy.visible_objects) if proxy else sorted(bboxes),
            "bboxes": {name: item.to_dict() for name, item in bboxes.items()},
            "proxy_relations": proxy_relations,
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
            if parsed.path == "/":
                self._send_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8", cache=False)
            elif parsed.path.startswith("/common/"):
                path = safe_child_path(COMMON_STATIC_ROOT, parsed.path.removeprefix("/common/"))
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

    @staticmethod
    def _required(query: dict[str, list[str]], key: str) -> str:
        values = query.get(key)
        if not values or not values[0]:
            raise ValueError(f"Missing query parameter '{key}'")
        return values[0]

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
