from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VLA_ROOT = REPO_ROOT / "vla_benchmarking"
for _path in (REPO_ROOT, VLA_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
