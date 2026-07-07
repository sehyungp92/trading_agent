"""JSONL-based session persistence for analysis runs (H5)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.session import SessionRecord

logger = logging.getLogger(__name__)

_RECENT_SESSION_METADATA_KEYS = (
    "provider",
    "runtime",
    "requested_model",
    "effective_model",
    "cost_usd",
    "first_output_ms",
    "stream_event_count",
    "tool_call_count",
    "auth_mode",
    "run_id",
)


class SessionStore:
    """JSONL-based session persistence for analysis runs."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)

    def _session_dir(self, agent_type: str, date: str) -> Path:
        """Get the directory for a specific agent type and date."""
        d = self._base_dir / agent_type / date
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _session_file(self, agent_type: str, date: str) -> Path:
        return self._session_dir(agent_type, date) / "sessions.jsonl"

    def _recent_session_summary(self, record: SessionRecord) -> dict:
        """Build a compact summary dict for prompt injection and APIs."""
        summary = {
            "agent_type": record.agent_type,
            "date": record.timestamp.strftime("%Y-%m-%d"),
            "duration_ms": record.duration_ms,
            "token_usage": record.token_usage,
            "response_summary": record.response_summary[:200],
        }
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        for key in _RECENT_SESSION_METADATA_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                summary[key] = value
        return summary

    def record_invocation(
        self,
        session_id: str,
        agent_type: str,
        prompt_package: dict,
        response: str,
        token_usage: dict | None = None,
        duration_ms: int = 0,
        metadata: dict | None = None,
    ) -> SessionRecord:
        """Record an LLM invocation to JSONL."""
        prompt_json = json.dumps(prompt_package, sort_keys=True, default=str)
        prompt_hash = hashlib.sha256(prompt_json.encode()).hexdigest()[:16]

        record = SessionRecord(
            session_id=session_id,
            agent_type=agent_type,
            prompt_hash=prompt_hash,
            response_summary=response[:500],
            token_usage=token_usage or {},
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        date_str = record.timestamp.strftime("%Y-%m-%d")
        path = self._session_file(agent_type, date_str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        return record

    def get_session(
        self, session_id: str, agent_type: str, date: str
    ) -> list[SessionRecord]:
        """Get all records for a specific session."""
        path = self._session_file(agent_type, date)
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                record = SessionRecord.model_validate_json(line)
                if record.session_id == session_id:
                    records.append(record)
        return records

    def get_recent_sessions(
        self, agent_type: str, days: int = 7,
    ) -> list[dict]:
        """Get recent session summaries for an agent type.

        Returns list of dicts with: agent_type, date, duration_ms, token_usage,
        response_summary (first 200 chars).
        """
        results: list[dict] = []
        agent_dir = self._base_dir / agent_type
        if not agent_dir.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for date_dir in sorted(agent_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            # Quick date filter
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc,
                )
                if dir_date < cutoff:
                    break  # dirs are sorted descending, so we can stop
            except ValueError:
                continue

            session_file = date_dir / "sessions.jsonl"
            if not session_file.exists():
                continue

            for line in session_file.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    record = SessionRecord.model_validate_json(line)
                    results.append(self._recent_session_summary(record))
                except Exception:
                    continue

        return results

    def list_sessions(
        self, agent_type: str | None = None, date: str | None = None
    ) -> list[dict]:
        """List available sessions. Returns summary dicts."""
        results = []

        if not self._base_dir.exists():
            return results

        if agent_type and date:
            search_dirs = [self._base_dir / agent_type / date]
        elif agent_type:
            search_dirs = (
                list((self._base_dir / agent_type).glob("*"))
                if (self._base_dir / agent_type).exists()
                else []
            )
        else:
            search_dirs = [
                d
                for at in self._base_dir.iterdir()
                if at.is_dir()
                for d in at.iterdir()
                if d.is_dir()
            ]

        for d in search_dirs:
            session_file = d / "sessions.jsonl"
            if not session_file.exists():
                continue

            seen_sessions: set[str] = set()
            for line in session_file.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    record = SessionRecord.model_validate_json(line)
                    if record.session_id not in seen_sessions:
                        seen_sessions.add(record.session_id)
                        summary = {
                            "session_id": record.session_id,
                            "agent_type": record.agent_type,
                            "timestamp": record.timestamp.isoformat(),
                        }
                        metadata = record.metadata if isinstance(record.metadata, dict) else {}
                        for key in _RECENT_SESSION_METADATA_KEYS:
                            value = metadata.get(key)
                            if value not in (None, ""):
                                summary[key] = value
                        results.append(summary)
        return results
