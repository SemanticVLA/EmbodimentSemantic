from __future__ import annotations

import sys
from pathlib import Path


VLA_ROOT = Path(__file__).resolve().parents[1]
if str(VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VLA_ROOT))
