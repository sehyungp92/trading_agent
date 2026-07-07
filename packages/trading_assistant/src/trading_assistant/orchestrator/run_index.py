# orchestrator/run_index.py
"""Searchable run index backed by SQLite FTS5.

Indexes agent run metadata, structured outputs, and validator notes to enable
cross-session recall and future replay evaluation.  Built on the existing
run folder structure — no changes to the run data format needed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    agent_type   TEXT NOT NULL,
    provider     TEXT DEFAULT '',
    model        TEXT DEFAULT '',
    bot_ids      TEXT DEFAULT '',
    date         TEXT DEFAULT '',
    success      INTEGER DEFAULT 1,
    duration_ms  INTEGER DEFAULT 0,
    cost_usd     REAL DEFAULT 0.0,
    created_at   TEXT NOT NULL,
    run_dir      TEXT DEFAULT '',
    response_preview TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(
    run_id,
    agent_type,
    bot_ids,
    response_text,
    instructions_text,
    validator_notes,
    structured_output,
    tokenize='porter unicode61'
);
"""


class RunIndex:
    """SQLite FTS5-backed index over agent run history.

    Usage::

        idx = RunIndex(Path("data/run_index.db"))
        idx.index_run(run_id, agent_type, run_dir, result)
        results = idx.search("regime mismatch filter threshold")
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def index_run(
        self,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        *,
        provider: str = "",
        model: str = "",
        bot_ids: str = "",
        date: str = "",
        success: bool = True,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Index a completed run from its run directory."""
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        try:
            existing = self._conn.execute(
                "SELECT created_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing and existing["created_at"]:
                created_at = existing["created_at"]
        except sqlite3.Error:
            created_at = now

        # Read available artifacts from run directory
        response_text = self._read_file(run_dir / "response.md")
        instructions_text = self._read_file(run_dir / "instructions.md")
        validator_notes = self._read_file(run_dir / "validator_notes.md")
        structured_output = self._read_file(run_dir / "parsed_analysis.json")

        response_preview = response_text[:500] if response_text else ""

        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, agent_type, provider, model, bot_ids, date,
                    success, duration_ms, cost_usd, created_at, run_dir,
                    response_preview, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, agent_type, provider, model, bot_ids, date,
                    1 if success else 0, duration_ms, cost_usd, created_at,
                    str(run_dir), response_preview,
                    json.dumps(metadata or {}, default=str),
                ),
            )

            # Delete any existing FTS entry for this run_id before inserting
            self._conn.execute(
                "DELETE FROM runs_fts WHERE run_id = ?", (run_id,),
            )
            self._conn.execute(
                """INSERT INTO runs_fts
                   (run_id, agent_type, bot_ids, response_text,
                    instructions_text, validator_notes, structured_output)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, agent_type, bot_ids,
                    response_text, instructions_text,
                    validator_notes, structured_output,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to index run %s: %s", run_id, exc)
            self._conn.rollback()

    def search(
        self,
        query: str,
        limit: int = 20,
        agent_type: str = "",
        bot_id: str = "",
        min_date: str = "",
    ) -> list[dict]:
        """Full-text search over indexed runs.

        Args:
            query: FTS5 query string (supports AND, OR, NEAR, etc.).
            limit: Maximum results to return.
            agent_type: Filter to a specific agent type.
            bot_id: Filter to runs involving a specific bot.
            min_date: Only include runs on or after this date.

        Returns:
            List of dicts with run metadata and matching snippets.
        """
        try:
            # Build WHERE clause for filters
            conditions = []
            params: list[str | int] = []

            if agent_type:
                conditions.append("r.agent_type = ?")
                params.append(agent_type)
            if bot_id:
                conditions.append("r.bot_ids LIKE ? ESCAPE '\\'")
                escaped = bot_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{escaped}%")
            if min_date:
                conditions.append("r.date >= ?")
                params.append(min_date)

            where_clause = ""
            if conditions:
                where_clause = "AND " + " AND ".join(conditions)

            sql = f"""
                SELECT r.run_id, r.agent_type, r.provider, r.model,
                       r.bot_ids, r.date, r.success, r.duration_ms,
                       r.cost_usd, r.created_at, r.response_preview,
                       snippet(runs_fts, -1, '<b>', '</b>', '...', 32) as snippet
                FROM runs_fts f
                JOIN runs r ON r.run_id = f.run_id
                WHERE runs_fts MATCH ?
                {where_clause}
                ORDER BY rank
                LIMIT ?
            """
            params_final: list[str | int] = [query, *params, limit]
            rows = self._conn.execute(sql, params_final).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error("FTS search failed: %s", exc)
            return []

    def get_recent_runs(
        self,
        agent_type: str = "",
        bot_id: str = "",
        limit: int = 20,
        days: int = 30,
    ) -> list[dict]:
        """Get recent runs without a search query."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conditions = ["created_at >= ?"]
        params: list[str | int] = [cutoff]

        if agent_type:
            conditions.append("agent_type = ?")
            params.append(agent_type)
        if bot_id:
            conditions.append("bot_ids LIKE ? ESCAPE '\\'")
            escaped = bot_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")

        sql = f"""
            SELECT run_id, agent_type, provider, model, bot_ids, date,
                   success, duration_ms, cost_usd, created_at,
                   response_preview
            FROM runs
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error("Recent runs query failed: %s", exc)
            return []

    def reindex_from_directory(self, runs_dir: Path) -> int:
        """Bulk-index all existing run directories.

        Scans ``runs_dir`` for subdirectories and indexes each one.
        Returns the count of newly indexed runs.
        """
        if not runs_dir.is_dir():
            return 0

        existing = set()
        try:
            rows = self._conn.execute("SELECT run_id FROM runs").fetchall()
            existing = {row[0] for row in rows}
        except sqlite3.Error:
            pass

        count = 0
        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir():
                continue
            run_id = entry.name
            if run_id in existing:
                continue

            # Try to extract metadata from run directory
            agent_type = self._infer_agent_type(run_id)
            meta = self._read_json(entry / "metadata.json") or {}

            self.index_run(
                run_id=run_id,
                agent_type=agent_type,
                run_dir=entry,
                provider=meta.get("provider", ""),
                model=meta.get("effective_model", ""),
                bot_ids=meta.get("bot_ids", ""),
                date=meta.get("date", ""),
                success=meta.get("success", True),
                duration_ms=meta.get("duration_ms", 0),
                cost_usd=meta.get("cost_usd", 0.0),
                metadata=meta,
            )
            count += 1

        return count

    @staticmethod
    def _read_file(path: Path, max_bytes: int = 256_000) -> str:
        """Read a file, returning empty string if missing.

        Caps at ``max_bytes`` (default 256 KB) to avoid loading unbounded
        response files into memory during bulk reindex.
        """
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return f.read(max_bytes)
            except Exception:
                return ""
        return ""

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        """Read a JSON file, returning None if missing."""
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    @staticmethod
    def _infer_agent_type(run_id: str) -> str:
        """Guess agent type from run_id prefix."""
        prefixes = {
            "daily": "daily_analysis",
            "weekly": "weekly_analysis",
            "monthly": "monthly_validation",
            "triage": "triage",
            "discovery": "discovery_analysis",
            "reasoning": "outcome_reasoning",
        }
        lower = run_id.lower()
        for prefix, agent_type in prefixes.items():
            if lower.startswith(prefix):
                return agent_type
        return "unknown"
