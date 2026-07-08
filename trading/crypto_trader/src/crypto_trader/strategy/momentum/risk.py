"""Per-session risk limits — consecutive losses, daily PnL, trade count."""

from __future__ import annotations

from datetime import date, datetime

from crypto_trader.strategy.momentum.config import DailyLimitParams


class SessionRiskManager:
    def __init__(self, config: DailyLimitParams) -> None:
        self._cfg = config
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date | None = None

    def record_trade(self, pnl: float, timestamp: datetime) -> None:
        self._maybe_reset(timestamp)
        self._daily_pnl += pnl
        self._daily_trades += 1
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def is_session_stopped(self, equity: float, current_time: datetime) -> tuple[bool, str]:
        self._maybe_reset(current_time)
        if self._consecutive_losses >= self._cfg.max_consecutive_losses:
            return True, "consecutive_losses"
        if equity > 0 and self._daily_pnl < 0 and abs(self._daily_pnl) / equity >= self._cfg.max_daily_loss_pct:
            return True, "daily_loss_limit"
        if self._daily_trades >= self._cfg.max_trades_per_day:
            return True, "daily_trade_limit"
        return False, ""

    def _maybe_reset(self, current_time: datetime) -> None:
        today = current_time.date()
        if self._last_reset_date is None or today > self._last_reset_date:
            self._consecutive_losses = 0
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = today
