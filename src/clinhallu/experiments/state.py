"""Durable run-state tracking for safe resume behavior."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.value = {"stage": "created", "history": []}

    def update(self, stage: str, **details) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.value["stage"] = stage
        self.value["updated_at"] = timestamp
        self.value.setdefault("history", []).append(
            {"stage": stage, "timestamp": timestamp, **details}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.value, indent=2), encoding="utf-8")
        temporary.replace(self.path)
