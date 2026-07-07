"""Signal factor attribution schemas — per-factor win rate, PnL, contribution."""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class FactorStats(BaseModel):
    """Aggregated stats for a single signal factor across daily trades."""
    factor_name: str
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    avg_contribution: float = 0.0

    @computed_field
    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0

    @computed_field
    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trade_count if self.trade_count > 0 else 0.0


class FactorAttribution(BaseModel):
    """Per-factor performance attribution for a bot's daily trades."""
    bot_id: str
    date: str
    factors: list[FactorStats] = []
