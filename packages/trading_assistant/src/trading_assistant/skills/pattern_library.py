# skills/pattern_library.py
"""Cross-bot pattern library — persistence and retrieval of structural innovations.

Storage: findings/pattern_library.jsonl (append-only JSONL).
Loaded into ContextBuilder for weekly analysis so Claude can propose transfers.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus


class PatternLibrary:
    """Manages the cross-bot pattern library."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._path = findings_dir / "pattern_library.jsonl"

    def add(self, entry: PatternEntry) -> PatternEntry:
        """Add a new pattern entry. Assigns pattern_id if not set."""
        if not entry.pattern_id:
            entry.pattern_id = uuid.uuid4().hex[:12]
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), default=str) + "\n")
        return entry

    def load_all(self) -> list[PatternEntry]:
        """Load all pattern entries."""
        if not self._path.exists():
            return []
        entries: list[PatternEntry] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(PatternEntry(**json.loads(line)))
        return entries

    def load_active(self) -> list[PatternEntry]:
        """Load patterns that are not rejected — candidates for transfer."""
        return [
            e for e in self.load_all()
            if e.status != PatternStatus.REJECTED
        ]

    def load_for_bot(self, bot_id: str) -> list[PatternEntry]:
        """Load patterns relevant to a specific bot (source or target)."""
        return [
            e for e in self.load_active()
            if e.source_bot == bot_id or bot_id in e.target_bots
        ]

    def update_status(self, pattern_id: str, status: PatternStatus) -> bool:
        """Update the status of a pattern. Rewrites the file."""
        entries = self.load_all()
        found = False
        for entry in entries:
            if entry.pattern_id == pattern_id:
                entry.status = status
                if status == PatternStatus.VALIDATED and not entry.validated_at:
                    from datetime import date
                    entry.validated_at = date.today().isoformat()
                found = True
                break

        if found:
            self._rewrite(entries)
        return found

    def validate_pattern(self, pattern_id: str) -> bool:
        """Promote a pattern to VALIDATED and set validated_at. Returns True if found."""
        return self.update_status(pattern_id, PatternStatus.VALIDATED)

    def _rewrite(self, entries: list[PatternEntry]) -> None:
        """Rewrite the entire JSONL file (atomic write)."""
        from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl

        atomic_rewrite_jsonl(self._path, entries)
