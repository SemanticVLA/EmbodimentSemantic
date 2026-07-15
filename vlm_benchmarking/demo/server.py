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


class UnifiedDemoRequestHandler(LiberoRequestHandler, So101RequestHandler):
    server: "UnifiedDemoServer"

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
            original_path = self.path
            self.path = self.path.removeprefix("/so101")
            try:
                So101RequestHandler.do_GET(self)
            finally:
                self.path = original_path
            return
        LiberoRequestHandler.do_GET(self)

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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.RequestHandlerClass = UnifiedDemoRequestHandler
        self.repository = so101_repository


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
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"EmbodimentSemantic unified demo: {url}", flush=True)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the unified LIBERO and SO101 scene-graph demo."
    )
    parser.add_argument("--input-dir", default="data/libero_spatial_v5")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--so101-config", default=None)
    parser.add_argument("--so101-artifacts", default=None)
    parser.add_argument("--camera", default="agentview", choices=sorted(CAMERA_INFO))
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--cache-dir", default=".cache/scene_graph_demo")
    parser.add_argument("--no-disk-cache", action="store_true")
    parser.add_argument("--bundled-cache-dir", default=None)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--no-rotate-agentview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    serve(build_arg_parser().parse_args(argv))
