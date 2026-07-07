"""Schemas for per-invocation cost tracking."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class CostRecord(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str
    workflow: str = ""
    model: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0
    success: bool = True
    run_id: str = ""


class CostSummary(BaseModel):
    """Aggregated cost summary over a time window."""
    total_cost_usd: float = 0.0
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_duration_ms: int = 0
    by_provider: dict[str, float] = Field(default_factory=dict)
    by_workflow: dict[str, float] = Field(default_factory=dict)
