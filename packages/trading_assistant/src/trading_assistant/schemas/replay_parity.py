"""Replay parity schemas and status classification."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ReplayParityStatus(str, Enum):
    PASS = "pass"
    PASS_WITH_KNOWN_GAPS = "pass_with_known_gaps"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class ReplayParityReport(BaseModel):
    """Live-vs-replay incumbent parity report."""

    bot_id: str
    strategy_id: str
    run_month: str = ""
    trade_count_live: int = 0
    trade_count_replay: int = 0
    entry_match_rate: float = 0.0
    exit_match_rate: float = 0.0
    side_quantity_match_rate: float = 0.0
    fill_price_delta_bps: float = 0.0
    pnl_delta_pct: float = 0.0
    fee_slippage_delta_bps: float = 0.0
    drawdown_delta_pct: float = 0.0
    missing_trade_explanations: list[str] = Field(default_factory=list)
    extra_simulated_trade_explanations: list[str] = Field(default_factory=list)
    known_gaps: list[str] = Field(default_factory=list)
    status: ReplayParityStatus = ReplayParityStatus.INSUFFICIENT_DATA
    evidence_paths: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _normalize(self) -> ReplayParityReport:
        for field_name in ("entry_match_rate", "exit_match_rate", "side_quantity_match_rate"):
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(float(value), 1.0)))
        return self

    @property
    def eligible_for_authoritative_validation(self) -> bool:
        return self.status in {
            ReplayParityStatus.PASS,
            ReplayParityStatus.PASS_WITH_KNOWN_GAPS,
        }
