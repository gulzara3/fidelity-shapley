"""Idempotent, resumable stage tracking (``state.json``)."""
from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path


class StageManager:
    """Records completed pipeline stages so interrupted runs resume cleanly."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, stage: str, artifacts: Iterable[Path] = ()) -> bool:
        ok = self.state.get(stage, {}).get("done", False)
        ok = ok and all(Path(a).exists() for a in artifacts)
        if ok:
            print(f"[resume] '{stage}' complete — skipping")
        return ok

    def mark(self, stage: str) -> None:
        self.state[stage] = {"done": True, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))
        print(f"[stage] '{stage}' ✓")
