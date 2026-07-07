"""Canonical market-data coverage manifest schema."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MarketDataUsability(str, Enum):
    AUTHORITATIVE = "authoritative"
    DIAGNOSTICS_ONLY = "diagnostics_only"
    BLOCKED = "blocked"


class MissingRange(BaseModel):
    start_ts: datetime
    end_ts: datetime
    reason: str = ""


class MarketDataManifest(BaseModel):
    """Coverage and provenance for one market-data slice."""

    manifest_id: str = ""
    source: str
    market: str = ""
    symbol: str
    timeframe: str
    start_ts: datetime
    end_ts: datetime
    expected_bars: int = 0
    actual_bars: int = 0
    coverage_ratio: float = 0.0
    missing_ranges: list[MissingRange] = Field(default_factory=list)
    session_calendar: str = ""
    timezone: str = "UTC"
    checksum: str = ""
    schema_version: str = "market_data_manifest_v1"
    source_version: str = ""
    adjustment_policy: str = ""
    fee_model_version: str = ""
    slippage_model_version: str = ""
    usable_for_authoritative_validation: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize(self) -> MarketDataManifest:
        if self.end_ts < self.start_ts:
            raise ValueError("end_ts must be >= start_ts")
        if self.expected_bars < 0 or self.actual_bars < 0:
            raise ValueError("bar counts must be non-negative")
        if self.expected_bars and not self.coverage_ratio:
            self.coverage_ratio = self.actual_bars / self.expected_bars
        self.coverage_ratio = max(0.0, min(float(self.coverage_ratio), 1.0))
        if not self.manifest_id:
            raw = "|".join([
                self.source,
                self.market,
                self.symbol,
                self.timeframe,
                self.start_ts.isoformat(),
                self.end_ts.isoformat(),
                self.checksum,
            ])
            self.manifest_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        if self.blocking_reasons:
            self.usable_for_authoritative_validation = False
        return self

    @property
    def usability(self) -> MarketDataUsability:
        if self.usable_for_authoritative_validation:
            return MarketDataUsability.AUTHORITATIVE
        if self.actual_bars > 0:
            return MarketDataUsability.DIAGNOSTICS_ONLY
        return MarketDataUsability.BLOCKED
