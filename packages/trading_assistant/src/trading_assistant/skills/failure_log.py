# skills/failure_log.py
"""Failure log — persists triage outcomes and PR rejections for Ralph Loop V2.

Written to `.assistant/failure-log.jsonl`. Future triage prompts include
past rejections for the same error type, so the system learns from mistakes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.bug_triage import TriageOutcome, TriageResult


@dataclass
class FailureEntry:
    bot_id: str
    error_type: str
    message: str
    outcome: TriageOutcome
    rejection_reason: str = ""
    pr_url: str = ""
    issue_url: str = ""
    suggested_fix: str = ""
    timestamp: str = ""

    @classmethod
    def from_triage_result(
        cls,
        result: TriageResult,
        rejection_reason: str = "",
    ) -> FailureEntry:
        return cls(
            bot_id=result.error_event.bot_id,
            error_type=result.error_event.error_type,
            message=result.error_event.message,
            outcome=result.outcome,
            rejection_reason=rejection_reason,
            pr_url=result.pr_url,
            issue_url=result.github_issue_url,
            suggested_fix=result.suggested_fix,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "error_type": self.error_type,
            "message": self.message,
            "outcome": self.outcome.value,
            "rejection_reason": self.rejection_reason,
            "pr_url": self.pr_url,
            "issue_url": self.issue_url,
            "suggested_fix": self.suggested_fix,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FailureEntry:
        return cls(
            bot_id=d["bot_id"],
            error_type=d["error_type"],
            message=d["message"],
            outcome=TriageOutcome(d["outcome"]),
            rejection_reason=d.get("rejection_reason", ""),
            pr_url=d.get("pr_url", ""),
            issue_url=d.get("issue_url", ""),
            suggested_fix=d.get("suggested_fix", ""),
            timestamp=d.get("timestamp", ""),
        )


class FailureLog:
    """Append-only JSONL log of triage outcomes and PR rejections."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record_triage(self, result: TriageResult) -> None:
        """Record a triage outcome."""
        entry = FailureEntry.from_triage_result(result)
        self._append(entry)

    def record_rejection(self, result: TriageResult, reason: str) -> None:
        """Record a PR rejection with the reason."""
        entry = FailureEntry.from_triage_result(result, rejection_reason=reason)
        self._append(entry)

    def get_past_rejections(
        self,
        error_type: str,
        limit: int = 10,
    ) -> list[FailureEntry]:
        """Load past rejections filtered by error_type."""
        if not self._path.exists():
            return []

        entries: list[FailureEntry] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("error_type") == error_type and data.get("rejection_reason"):
                entries.append(FailureEntry.from_dict(data))

        return entries[-limit:]

    def _append(self, entry: FailureEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
