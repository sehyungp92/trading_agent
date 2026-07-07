"""Exit efficiency schemas — measuring how well exits capture available moves."""
from __future__ import annotations

from pydantic import BaseModel


class ExitEfficiencyRecord(BaseModel):
    """Per-trade exit efficiency with post-exit price continuation data."""
    trade_id: str
    bot_id: str
    pair: str
    pnl: float
    exit_reason: str
    market_regime: str = ""
    exit_efficiency: float = 0.0
    continuation_1h: float | None = None
    continuation_4h: float | None = None


class ExitEfficiencyStats(BaseModel):
    """Aggregated exit efficiency stats for a bot's daily trades."""
    bot_id: str
    date: str
    avg_efficiency: float = 0.0
    premature_exit_pct: float = 0.0
    by_exit_reason: dict[str, float] = {}
    by_regime: dict[str, float] = {}
    total_trades_with_data: int = 0
