from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    for key, value in config.get("paths", {}).items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        config["paths"][key] = str(candidate)
    return config
