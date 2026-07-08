"""Breakout session risk management — daily/weekly loss, consecutive loss, trade count."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from .config import BreakoutLimitParams


class RiskManager:
    """Track daily/weekly PnL, consecutive losses, and trade counts."""

    def __init__(self, cfg: BreakoutLimitParams) -> None:
        self._p = cfg
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trades_today: int = 0
        self._current_day: object | None = None  # date
        self._current_week: int | None = None  # ISO week number

    def snapshot_state(self) -> dict[str, Any]:
        """Return the mutable session-risk counters."""
        return {
            "_daily_pnl": self._daily_pnl,
            "_weekly_pnl": self._weekly_pnl,
            "_consecutive_losses": self._consecutive_losses,
            "_trades_today": self._trades_today,
            "_current_day": deepcopy(self._current_day),
            "_current_week": self._current_week,
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore counters captured by :meth:`snapshot_state`."""
        for name, value in snapshot.items():
            if hasattr(self, name):
                setattr(self, name, deepcopy(value))

    def is_session_stopped(self, equity: float, current_time: datetime) -> tuple[bool, str]:
        """Return ``(True, reason)`` if any session limit has been breached."""
        self._reset_if_new_day(current_time)
        self._reset_if_new_week(current_time)

        # Daily loss limit
        if equity > 0 and self._daily_pnl < 0:
            if abs(self._daily_pnl) / equity >= self._p.max_daily_loss_pct:
                return True, "daily_loss_limit"

        # Weekly loss limit
        if equity > 0 and self._weekly_pnl < 0:
            if abs(self._weekly_pnl) / equity >= self._p.max_weekly_loss_pct:
                return True, "weekly_loss_limit"

        # Consecutive loss limit
        if self._consecutive_losses >= self._p.max_consecutive_losses:
            return True, "consecutive_losses"

        # Trade count limit
        if self._trades_today >= self._p.max_trades_per_day:
            return True, "daily_trade_limit"

        return False, ""

    def record_trade_exit(self, pnl: float, timestamp: datetime) -> None:
        """Record a trade exit and update running counters."""
        self._reset_if_new_day(timestamp)
        self._reset_if_new_week(timestamp)

        self._trades_today += 1
        self._daily_pnl += pnl
        self._weekly_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _reset_if_new_day(self, ts: datetime) -> None:
        day = ts.date() if hasattr(ts, "date") and callable(ts.date) else ts
        if self._current_day != day:
            self._current_day = day
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._consecutive_losses = 0

    def _reset_if_new_week(self, ts: datetime) -> None:
        d = ts.date() if hasattr(ts, "date") and callable(ts.date) else ts
        week = d.isocalendar()[1]
        if self._current_week != week:
            self._current_week = week
            self._weekly_pnl = 0.0
