from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from samgraph_benchmark.frames import iter_zip_frames, ordered_members


def _encoded(image: np.ndarray, fmt: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(stream, format=fmt)
    return stream.getvalue()


def test_zip_order_transform_and_resize(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    # Cache stores the display-rotated version; after undo, the red corner is
    # back at the original top-left. PNG avoids JPEG test noise.
    raw0 = np.zeros((6, 6, 3), dtype=np.uint8)
    raw0[0, 0] = (255, 0, 0)
    raw1 = np.zeros((6, 6, 3), dtype=np.uint8)
    raw1[5, 5] = (0, 255, 0)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000001.png", _encoded(np.rot90(raw1, 2)))
        z.writestr("000000.png", _encoded(np.rot90(raw0, 2)))
    rows = list(iter_zip_frames(archive, resolution=8))
    assert [row.frame for row in rows] == [0, 1]
    assert [row.source_name for row in rows] == ["000000.png", "000001.png"]
    assert rows[0].rgb.shape == (8, 8, 3)
    # Resizing preserves the corner's colour neighborhood.
    assert rows[0].rgb[0, 0, 0] > 200 and rows[0].rgb[0, 0, 1] < 50


def test_noncontiguous_zip_frames_fail(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000001.png", _encoded(image))
    with pytest.raises(ValueError, match="non-contiguous"):
        list(iter_zip_frames(archive))


def test_non_square_source_is_rejected_before_resize(tmp_path):
    archive = tmp_path / "task" / "demo_0.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("000000.png", _encoded(np.zeros((4, 6, 3), dtype=np.uint8)))
    with pytest.raises(ValueError, match="square"):
        list(iter_zip_frames(archive, resolution=None))
