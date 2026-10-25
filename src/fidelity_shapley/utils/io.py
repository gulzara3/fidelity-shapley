"""Small JSON helpers used for incremental, per-seed artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
