"""A durable record of which units of work are already finished.

Training checkpoints make a *single* run resumable. A ledger makes a *sequence*
of runs resumable: five ablation variants, five pipeline stages, a dozen
baselines. Without one, a crash during the fourth ablation means the next
invocation re-derives everything from scratch just to discover the first three
were already done.

Each entry records a status, an attempt count, timestamps, and a fingerprint of
the inputs. The fingerprint is what makes skipping safe: if a config changes,
the fingerprint changes, the previous "completed" entry no longer matches, and
the unit runs again instead of silently reusing stale results.

Writes are atomic and happen immediately after every state change, so the file
on disk always reflects reality even if the process dies in the next instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_SKIPPED = "skipped"

#: Statuses that mean "do this again on the next invocation".
RESUMABLE_STATUSES = {STATUS_PENDING, STATUS_RUNNING, STATUS_FAILED, STATUS_INTERRUPTED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(payload: Any) -> str:
    """Stable hash of any JSON-serialisable description of a unit's inputs."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class Ledger:
    """Atomic, append-friendly completion record for a set of named units."""

    def __init__(self, path: str | Path, *, name: str = "run") -> None:
        self.path = Path(path)
        self.name = name
        self.data: Dict[str, Any] = {"name": name, "units": {}}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("units"), dict):
                    self.data = loaded
                else:
                    logger.warning("Ignoring malformed ledger at %s", self.path)
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable ledger at %s", self.path)

    def _flush(self) -> None:
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def entry(self, unit: str) -> Dict[str, Any]:
        return self.data["units"].get(unit, {})

    def status(self, unit: str) -> str:
        return self.entry(unit).get("status", STATUS_PENDING)

    def is_complete(self, unit: str, *, fingerprint_value: Optional[str] = None) -> bool:
        """True only when the unit finished *and* its inputs are unchanged."""
        entry = self.entry(unit)
        if entry.get("status") != STATUS_COMPLETED:
            return False
        if fingerprint_value is None:
            return True
        recorded = entry.get("fingerprint")
        if recorded is None:
            # Written before fingerprinting existed; trust it rather than
            # discarding hours of finished work.
            return True
        if recorded != fingerprint_value:
            logger.info(
                "Unit %r completed previously, but its configuration changed "
                "(%s -> %s); it will run again.",
                unit,
                recorded,
                fingerprint_value,
            )
            return False
        return True

    def pending(self, units: Iterable[str]) -> list[str]:
        return [unit for unit in units if not self.is_complete(unit)]

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for entry in self.data["units"].values():
            status = entry.get("status", STATUS_PENDING)
            counts[status] = counts.get(status, 0) + 1
        return {"path": str(self.path), "counts": counts, "units": self.data["units"]}

    def mark(self, unit: str, status: str, **details: Any) -> None:
        entry = dict(self.data["units"].get(unit, {}))
        entry.update(details)
        entry["status"] = status
        entry["updated_at"] = _now()
        if status == STATUS_RUNNING:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry["started_at"] = _now()
        if status == STATUS_COMPLETED:
            entry["completed_at"] = _now()
            entry.pop("error", None)
        self.data["units"][unit] = entry
        self._flush()

    def start(self, unit: str, *, fingerprint_value: Optional[str] = None, **details) -> None:
        if fingerprint_value is not None:
            details["fingerprint"] = fingerprint_value
        self.mark(unit, STATUS_RUNNING, **details)

    def complete(self, unit: str, **details) -> None:
        self.mark(unit, STATUS_COMPLETED, **details)

    def fail(self, unit: str, error: str, **details) -> None:
        self.mark(unit, STATUS_FAILED, error=str(error)[:2000], **details)

    def interrupt(self, unit: str, **details) -> None:
        self.mark(unit, STATUS_INTERRUPTED, **details)

    def reset(self, units: Optional[Iterable[str]] = None) -> None:
        """Forget recorded outcomes so the given units run again."""
        if units is None:
            self.data["units"] = {}
        else:
            for unit in units:
                self.data["units"].pop(unit, None)
        self._flush()
