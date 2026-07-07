from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class PatternCount(BaseModel):
    """A single pattern with its count."""
    category: str
    key: str
    count: int

class ConsolidationSummary(BaseModel):
    """Summary of a memory consolidation run."""
    consolidated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_file: str
    total_entries: int
    patterns: list[PatternCount] = []
    top_bots: list[PatternCount] = []
    top_error_types: list[PatternCount] = []
    top_root_causes: list[PatternCount] = []


class MemoryIndex(BaseModel):
    """Lightweight index maintained by consolidation for O(1) existence checks.

    Answers "does data exist for X?" without scanning directories.
    """
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Curated data: which dates have data for which bots
    curated_dates_by_bot: dict[str, list[str]] = {}
    # Weekly curated data: which week_start dates exist
    weekly_dates: list[str] = []

    # Sessions: which agent_types have sessions on which dates
    sessions_by_agent_type: dict[str, list[str]] = {}

    # Runs: which run_ids exist
    run_ids: list[str] = []

    # Findings: entry counts per JSONL file
    findings_counts: dict[str, int] = {}

    # Heartbeats: last seen timestamp per bot
    heartbeat_last_seen: dict[str, str] = {}

    # Last consolidation timestamp
    last_consolidated: str = ""

    # Summary stats
    total_sessions: int = 0
    total_findings: int = 0
    total_curated_days: int = 0
