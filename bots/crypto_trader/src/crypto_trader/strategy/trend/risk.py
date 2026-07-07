"""Trend session risk management — daily loss, consecutive loss, trade count limits."""

from __future__ import annotations

from datetime import date, datetime

from .config import TrendLimitParams


class RiskManager:
    """Track daily PnL, consecutive losses, and trade count."""

    def __init__(self, cfg: TrendLimitParams) -> None:
        self._cfg = cfg
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._current_date: date | None = None

    def is_session_stopped(self, equity: float, current_time: datetime) -> tuple[bool, str]:
        self._check_date_reset(current_time)

        # Daily loss limit
        if equity > 0 and self._daily_pnl < 0:
            if abs(self._daily_pnl) / equity >= self._cfg.max_daily_loss_pct:
                return True, "daily_loss_limit"

        # Consecutive loss limit
        if self._consecutive_losses >= self._cfg.max_consecutive_losses:
            return True, "consecutive_losses"

        # Trade count limit
        if self._trades_today >= self._cfg.max_trades_per_day:
            return True, "daily_trade_limit"

        return False, ""

    def record_trade(self, pnl: float, timestamp: datetime) -> None:
        self._check_date_reset(timestamp)
        self._daily_pnl += pnl
        self._trades_today += 1

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _check_date_reset(self, current_time: datetime) -> None:
        today = current_time.date() if hasattr(current_time, 'date') else current_time
        if self._current_date is None or today != self._current_date:
            self._current_date = today
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._consecutive_losses = 0
