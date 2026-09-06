from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "vla_benchmarking"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
