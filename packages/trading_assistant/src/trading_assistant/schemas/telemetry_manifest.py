"""Telemetry coverage manifest for monthly validation."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TelemetryEligibility(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    INSUFFICIENT_LINEAGE = "insufficient_lineage"
    INSUFFICIENT_DATA = "insufficient_data"


class TelemetryManifest(BaseModel):
    """Lineage and event coverage for one bot/strategy/month."""

    manifest_id: str = ""
    bot_id: str
    strategy_id: str = ""
    run_month: str
    window_start: date
    window_end: date
    event_counts_by_type: dict[str, int] = Field(default_factory=dict)
    lineage_coverage_ratio: float = 0.0
    missing_field_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
    known_gaps: list[str] = Field(default_factory=list)
    authoritative_eligibility: TelemetryEligibility = TelemetryEligibility.INSUFFICIENT_DATA
    time_coverage: dict[str, str] = Field(default_factory=dict)
    total_events: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_version: str = "telemetry_manifest_v1"

    @model_validator(mode="after")
    def _normalize(self) -> TelemetryManifest:
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        if not self.total_events:
            self.total_events = sum(self.event_counts_by_type.values())
        self.lineage_coverage_ratio = max(0.0, min(float(self.lineage_coverage_ratio), 1.0))
        if not self.manifest_id:
            raw = "|".join([
                self.bot_id,
                self.strategy_id,
                self.run_month,
                self.window_start.isoformat(),
                self.window_end.isoformat(),
                str(self.total_events),
            ])
            self.manifest_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self

    @property
    def is_authoritative_ready(self) -> bool:
        return self.authoritative_eligibility == TelemetryEligibility.AUTHORITATIVE
