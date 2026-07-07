"""AllocationTracker — records allocation changes to JSONL for historical tracking.

Storage: findings/allocation_history.jsonl (append-only JSONL).
"""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.allocation_history import AllocationRecord, AllocationSnapshot


class AllocationTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._path = findings_dir / "allocation_history.jsonl"
        self._snapshot_path = findings_dir / "allocation_snapshots.jsonl"

    def record(self, entry: AllocationRecord) -> None:
        """Append an allocation record to the JSONL file."""
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[dict]:
        """Read all allocation records."""
        if not self._path.exists():
            return []
        records: list[dict] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def record_snapshot(self, snapshot: AllocationSnapshot) -> None:
        """Append an allocation snapshot to the snapshots JSONL file."""
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._snapshot_path, "a") as f:
            f.write(json.dumps(snapshot.model_dump(mode="json"), default=str) + "\n")

    def load_snapshots(self) -> list[dict]:
        """Read all allocation snapshot records."""
        if not self._snapshot_path.exists():
            return []
        records: list[dict] = []
        with open(self._snapshot_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_latest_actuals(self) -> dict[str, float]:
        """Get most recent allocation_pct per bot_id from allocation_history.jsonl.

        Scans records sorted by date descending; first occurrence per bot wins.
        Returns empty dict if no history exists.
        """
        records = self.load_all()
        if not records:
            return {}
        # Sort by date descending
        records.sort(key=lambda r: r.get("date", ""), reverse=True)
        latest: dict[str, float] = {}
        for rec in records:
            bid = rec.get("bot_id", "")
            if bid and bid not in latest:
                latest[bid] = rec.get("allocation_pct", 0.0)
        return latest
