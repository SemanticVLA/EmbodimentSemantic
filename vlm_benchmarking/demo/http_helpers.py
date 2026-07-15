from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


def safe_child_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def guess_content_type(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if content_type == "text/html":
        return "text/html; charset=utf-8"
    if content_type in {"text/css", "text/javascript", "application/javascript"}:
        return f"{content_type}; charset=utf-8"
    return content_type


def send_json(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, Any],
    status: HTTPStatus = HTTPStatus.OK,
) -> None:
    data = json.dumps(payload).encode("utf-8")
    send_bytes(handler, data, "application/json", status=status, cache=False)


def send_html(handler: BaseHTTPRequestHandler, html: str) -> None:
    send_bytes(handler, html.encode("utf-8"), "text/html; charset=utf-8", cache=False)


def send_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str | None = None, *, cache: bool) -> None:
    send_bytes(handler, path.read_bytes(), content_type or guess_content_type(path), cache=cache)


def send_bytes(
    handler: BaseHTTPRequestHandler,
    data: bytes,
    content_type: str,
    *,
    status: HTTPStatus = HTTPStatus.OK,
    cache: bool,
    cache_control: str | None = None,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header(
        "Cache-Control",
        cache_control or ("public, max-age=31536000, immutable" if cache else "no-store"),
    )
    handler.end_headers()
    handler.wfile.write(data)


def send_range_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    size = path.stat().st_size
    range_header = handler.headers.get("Range")
    start, end = 0, size - 1
    status = HTTPStatus.OK
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        if match.group(1):
            start = int(match.group(1))
        if match.group(2):
            end = min(int(match.group(2)), size - 1)
        if start > end or start >= size:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        status = HTTPStatus.PARTIAL_CONTENT
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "public, max-age=3600")
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)
