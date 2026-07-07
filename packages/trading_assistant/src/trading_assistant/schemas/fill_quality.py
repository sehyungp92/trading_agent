"""Fill quality schemas — order fill analysis for adverse selection detection.

Used by FillQualityAnalyzer to assess momentum_nq_01 fill detail data.
"""
from __future__ import annotations

from pydantic import BaseModel


class FillStats(BaseModel):
    """Aggregated fill statistics for one side (entry or exit)."""

    sample_count: int = 0
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    avg_fill_latency_ms: float = 0.0
    adverse_fill_pct: float = 0.0  # % of fills where slippage was against us
    by_fill_type: dict[str, int] = {}  # e.g. {"limit": 5, "market": 10}


class SymbolFillQuality(BaseModel):
    """Per-symbol fill quality breakdown."""

    symbol: str
    entry_stats: FillStats = FillStats()
    exit_stats: FillStats = FillStats()
    net_adverse_impact_bps: float = 0.0


class FillQualityReport(BaseModel):
    """Aggregated fill quality report for one bot on one date."""

    bot_id: str
    date: str
    overall_entry: FillStats = FillStats()
    overall_exit: FillStats = FillStats()
    by_symbol: dict[str, SymbolFillQuality] = {}
    coverage_pct: float = 0.0
    adverse_selection_detected: bool = False
