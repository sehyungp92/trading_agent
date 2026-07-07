"""Engine-level metrics schemas — per-engine decomposition of bot performance.

Enables analysis like "Downturn's Fade engine has 28% win rate in NEUTRAL regime"
instead of generic "DownturnDominator has 45% win rate".
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RegimeEngineStats(BaseModel):
    """Performance stats for a single engine within a single regime."""

    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0


class EngineMetrics(BaseModel):
    """Aggregated metrics for a single sub-engine of a strategy."""

    engine: str
    strategy_id: str = ""
    bot_id: str = ""
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_exit_efficiency: float = 0.0
    avg_signal_strength: float = 0.0
    regime_breakdown: list[RegimeEngineStats] = Field(default_factory=list)


class EngineDecomposition(BaseModel):
    """Full engine-level decomposition for a bot on a given period."""

    bot_id: str
    period: str = ""
    engines: list[EngineMetrics] = Field(default_factory=list)
    unmapped_trades: int = 0
