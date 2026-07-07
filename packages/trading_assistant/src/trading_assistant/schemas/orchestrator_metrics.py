"""Orchestrator observability metrics schema."""
from __future__ import annotations

from pydantic import BaseModel


class BotLatencyStats(BaseModel):
    """Per-bot delivery latency percentiles."""
    bot_id: str
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0
    sample_count: int = 0


class OrchestratorMetrics(BaseModel):
    """Point-in-time snapshot of orchestrator operational state."""
    queue_depth: int = 0
    dead_letter_count: int = 0
    active_agents: int = 0
    error_rate_1h: float = 0.0
    uptime_seconds: float = 0.0
    last_daily_analysis: str | None = None
    last_weekly_analysis: str | None = None
    delivery_latency_p50: float = 0.0
    delivery_latency_p95: float = 0.0
    delivery_latency_max: float = 0.0
    per_bot_latency: list[BotLatencyStats] = []
    # P2-8: backlog age — clearer than depth alone since 20 very-old events
    # are more dangerous than 200 fresh events during a normal relay drain.
    oldest_pending_age_seconds: float = 0.0
