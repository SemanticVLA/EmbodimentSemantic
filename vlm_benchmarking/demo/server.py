from __future__ import annotations

import argparse
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .libero_backend import (
    CAMERA_INFO,
    DEFAULT_SUBJECT,
    RESOLUTION_OPTIONS,
    DemoRequestHandler as LiberoRequestHandler,
    Hdf5SceneGraphStore,
    PredictionStore,
    RendererManager,
    SceneGraphDemoServer,
)
from .so101_proxy_demo.proxy.artifacts import ArtifactStore
from .so101_proxy_demo.proxy.config import load_config
from .so101_backend import (
    DemoRepository,
    DemoRequestHandler as So101RequestHandler,
)


DEMO_ROOT = Path(__file__).resolve().parent

SO101_API_ALIAS_PATHS = {
    "/api/graph-edits",
    "/api/graph-edits/reset",
    "/api/review-status",
    "/api/review-status/reset",
    "/api/export-csv",
}


def _so101_api_alias_path(path: str) -> str | None:
    for endpoint in sorted(SO101_API_ALIAS_PATHS, key=len, reverse=True):
        if path == endpoint or path == f"/so101{endpoint}" or path.endswith(f"/so101{endpoint}"):
            return endpoint
    return None

LAUNCH_MODES = {
    "online": {
        "label": "cached hosted demo",
        "dataset_scope": "LIBERO demo_0 and SO101 episode_0 bundled caches",
        "input_dir": DEMO_ROOT / "libero_demo_cache",
        "output_dir": DEMO_ROOT / "libero_prediction_cache",
        "so101_config": DEMO_ROOT / "so101_demo_cache" / "config.yaml",
        "so101_artifacts": None,
        "cache_dir": ".cache/scene_graph_demo",
        "graph_output_dir": DEMO_ROOT.parent / "output",
        "bundled_cache_dir": DEMO_ROOT / "libero_frame_cache",
        "cached_only": True,
        "no_disk_cache": True,
        "host": "0.0.0.0",
        "port": 7860,
        "no_open_browser": True,
    },
    "offline": {
        "label": "localhost scene-graph tool",
        "dataset_scope": "local LIBERO/SO101 datasets and writable proxy artifacts",
        "input_dir": "data/libero_spatial_v5",
        "output_dir": "output",
        "so101_config": DEMO_ROOT / "so101_proxy_demo" / "config" / "default.yaml",
        "so101_artifacts": None,
        "cache_dir": ".cache/scene_graph_demo",
        "graph_output_dir": "output",
        "bundled_cache_dir": None,
        "cached_only": False,
        "no_disk_cache": False,
        "host": "127.0.0.1",
        "port": 7860,
        "no_open_browser": False,
    },
}


class UnifiedDemoRequestHandler(LiberoRequestHandler, So101RequestHandler):
    server: "UnifiedDemoServer"

    def _dispatch_so101_api(self, method: str, api_path: str) -> None:
        original_path = self.path
        suffix = ""
        if "?" in original_path:
            suffix = "?" + original_path.split("?", 1)[1]
        self.path = api_path + suffix
        try:
            if method == "GET" and api_path == "/api/health":
                payload = self.server.repository.health()
                payload.update(self.server.mode_health())
                self._send_json(payload)
                return
            if method == "GET":
                So101RequestHandler.do_GET(self)
            else:
                So101RequestHandler.do_POST(self)
        finally:
            self.path = original_path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_demo_file(DEMO_ROOT / "index.html", cache=False)
            return
        if parsed.path == "/index.html":
            self._redirect("/")
            return
        if parsed.path in {"/libero", "/so101"}:
            self._redirect(parsed.path + "/")
            return
        if parsed.path == "/libero/":
            self._serve_demo_file(DEMO_ROOT / "libero" / "index.html", cache=False)
            return
        if parsed.path == "/so101/":
            self._serve_demo_file(DEMO_ROOT / "so101" / "index.html", cache=False)
            return
        if parsed.path in {"/portal.css", "/portal.js"}:
            self._serve_demo_file(DEMO_ROOT / parsed.path.removeprefix("/"), cache=False)
            return
        if parsed.path.startswith("/common/"):
            self._serve_scoped_file(parsed.path, "/common/", DEMO_ROOT / "common")
            return
        if parsed.path.startswith("/so101/") and not parsed.path.startswith("/so101/api/"):
            self._serve_scoped_file(parsed.path, "/so101/", DEMO_ROOT / "so101")
            return
        if parsed.path.startswith("/data/"):
            self._serve_scoped_file(parsed.path, "/data/", DEMO_ROOT / "data")
            return
        if parsed.path == "/so101/api" or parsed.path.startswith("/so101/api/"):
            self._dispatch_so101_api("GET", parsed.path.removeprefix("/so101"))
            return
        api_alias = _so101_api_alias_path(parsed.path)
        if api_alias:
            self._dispatch_so101_api("GET", api_alias)
            return
        LiberoRequestHandler.do_GET(self)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/so101/api" or parsed.path.startswith("/so101/api/"):
            self._dispatch_so101_api("POST", parsed.path.removeprefix("/so101"))
            return
        api_alias = _so101_api_alias_path(parsed.path)
        if api_alias:
            self._dispatch_so101_api("POST", api_alias)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_scoped_file(
        self,
        request_path: str,
        prefix: str,
        root: Path,
        *,
        cache: bool = False,
    ) -> None:
        relative = request_path.removeprefix(prefix)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_demo_file(path, cache=cache)

    def _serve_demo_file(self, path: Path, *, cache: bool) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type == "text/html":
            content_type += "; charset=utf-8"
        self._send_file(path, content_type, cache=cache)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


class UnifiedDemoServer(SceneGraphDemoServer):
    def __init__(
        self,
        *args: Any,
        so101_repository: DemoRepository,
        demo_mode: str,
        demo_mode_label: str,
        demo_dataset_scope: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.RequestHandlerClass = UnifiedDemoRequestHandler
        self.repository = so101_repository
        self.demo_mode = demo_mode
        self.demo_mode_label = demo_mode_label
        self.demo_dataset_scope = demo_dataset_scope

    def mode_health(self) -> dict[str, str]:
        return {
            "demo_mode": self.demo_mode,
            "demo_mode_label": self.demo_mode_label,
            "demo_dataset_scope": self.demo_dataset_scope,
        }


def serve(args: argparse.Namespace) -> None:
    rotate_agentview = not args.no_rotate_agentview
    semantic_cache_dir = None if args.no_disk_cache else Path(args.cache_dir) / "hdf5_semantics"
    stores = {
        camera: Hdf5SceneGraphStore(
            args.input_dir,
            camera,
            rotate_agentview=rotate_agentview,
            semantic_cache_dir=semantic_cache_dir,
        )
        for camera in CAMERA_INFO
    }
    resolutions = sorted(set([*RESOLUTION_OPTIONS, args.res]))
    renderers = {}
    if not args.cached_only:
        shared_renderer = RendererManager(
            args.camera,
            max(resolutions),
            rotate_agentview=rotate_agentview,
        )
        renderers = {
            (camera, resolution): shared_renderer
            for camera in CAMERA_INFO
            for resolution in resolutions
        }

    so101_config = load_config(args.so101_config)
    artifact_root = Path(args.so101_artifacts).resolve() if args.so101_artifacts else Path(
        so101_config["paths"]["artifacts"]
    ).resolve()
    so101_repository = DemoRepository(
        so101_config,
        ArtifactStore(artifact_root),
        api_prefix="/so101/api",
        graph_output_dir=args.graph_output_dir,
        read_only=args.demo_mode != "offline",
    )
    server = UnifiedDemoServer(
        (args.host, args.port),
        stores,
        renderers,
        args.subject,
        default_camera=args.camera,
        default_res=args.res,
        resolutions=resolutions,
        predictions=PredictionStore(args.output_dir),
        cache_dir=None if args.no_disk_cache else args.cache_dir,
        bundled_cache_dir=args.bundled_cache_dir,
        cached_only=args.cached_only,
        so101_repository=so101_repository,
        demo_mode=args.demo_mode,
        demo_mode_label=args.demo_mode_label,
        demo_dataset_scope=args.demo_dataset_scope,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"EmbodimentSemantic unified demo: {url}", flush=True)
    print(f"Mode: {args.demo_mode} ({args.demo_mode_label})", flush=True)
    print(f"Dataset scope: {args.demo_dataset_scope}", flush=True)
    print(f"PID: {__import__('os').getpid()}", flush=True)
    print("LIBERO: /libero/ | SO101: /so101/", flush=True)
    if not args.no_open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close_all_renderers()


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--so101-config", default=argparse.SUPPRESS)
    parser.add_argument("--so101-artifacts", default=argparse.SUPPRESS)
    parser.add_argument("--camera", default=argparse.SUPPRESS, choices=sorted(CAMERA_INFO))
    parser.add_argument("--res", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--subject", default=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", default=argparse.SUPPRESS)
    parser.add_argument("--graph-output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--no-disk-cache", dest="no_disk_cache", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--disk-cache", dest="no_disk_cache", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--bundled-cache-dir", default=argparse.SUPPRESS)
    parser.add_argument("--no-bundled-cache", dest="bundled_cache_dir", action="store_const", const=None, default=argparse.SUPPRESS)
    parser.add_argument("--cached-only", dest="cached_only", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--live-rendering", dest="cached_only", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--host", default=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--no-open-browser", dest="no_open_browser", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--open-browser", dest="no_open_browser", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--no-rotate-agentview", action="store_true", default=argparse.SUPPRESS)


def _apply_launch_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    mode = getattr(args, "demo_mode", None) or "offline"
    defaults = LAUNCH_MODES[mode]
    for key, value in defaults.items():
        if key in {"label", "dataset_scope"}:
            continue
        if not hasattr(args, key):
            setattr(args, key, value)
    if not hasattr(args, "res"):
        args.res = 1024
    if not hasattr(args, "camera"):
        args.camera = "agentview"
    if not hasattr(args, "subject"):
        args.subject = DEFAULT_SUBJECT
    if not hasattr(args, "no_rotate_agentview"):
        args.no_rotate_agentview = False
    args.demo_mode = mode
    args.demo_mode_label = str(defaults["label"])
    args.demo_dataset_scope = str(defaults["dataset_scope"])
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the unified LIBERO/SO101 scene-graph app. Use 'online' for "
            "the cached Docker/Fly demo and 'offline' for the localhost "
            "scene-graph tool."
        )
    )
    parser.set_defaults(demo_mode="offline")
    _add_serve_arguments(parser)

    subparsers = parser.add_subparsers(dest="command")
    online = subparsers.add_parser(
        "online",
        help="Serve the cached hosted demo dataset: LIBERO demo_0 and SO101 episode_0.",
    )
    online.set_defaults(demo_mode="online")
    _add_serve_arguments(online)

    offline = subparsers.add_parser(
        "offline",
        help="Serve the localhost scene-graph tool against local datasets/artifacts.",
    )
    offline.set_defaults(demo_mode="offline")
    _add_serve_arguments(offline)
    return parser


def main(argv: list[str] | None = None) -> None:
    serve(_apply_launch_mode_defaults(build_arg_parser().parse_args(argv)))
