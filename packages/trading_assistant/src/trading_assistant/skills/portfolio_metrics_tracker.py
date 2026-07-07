# skills/portfolio_metrics_tracker.py
"""Portfolio metrics tracker — computes rolling portfolio-level risk-adjusted metrics.

Reads family daily snapshots and computes rolling 7d/30d/90d Sharpe, Sortino,
and Calmar ratios at the portfolio level. No LLM calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from trading_assistant.schemas.portfolio_metrics import PortfolioRollingMetrics


class PortfolioMetricsTracker:
    """Computes rolling portfolio metrics from curated family snapshot data."""

    def __init__(self, curated_dir: Path) -> None:
        self._curated_dir = curated_dir

    def compute(self, date: str, lookback_days: int = 90) -> PortfolioRollingMetrics:
        """Compute rolling metrics as of the given date."""
        from datetime import datetime, timedelta

        end = datetime.strptime(date, "%Y-%m-%d")
        daily_pnl: list[float] = []
        family_daily: dict[str, list[float]] = {}

        for d in range(lookback_days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            snapshot_path = self._curated_dir / date_str / "portfolio" / "family_snapshots.json"
            if not snapshot_path.exists():
                continue
            try:
                snapshots = json.loads(snapshot_path.read_text(encoding="utf-8"))
                day_total = 0.0
                for snap in snapshots:
                    pnl = snap.get("total_net_pnl", 0.0)
                    day_total += pnl
                    fam = snap.get("family", "unknown")
                    family_daily.setdefault(fam, []).append(pnl)
                daily_pnl.append(day_total)
            except (json.JSONDecodeError, OSError):
                continue

        # Reverse so oldest first
        daily_pnl.reverse()
        for fam in family_daily:
            family_daily[fam].reverse()

        return PortfolioRollingMetrics(
            date=date,
            sharpe_7d=self._sharpe(daily_pnl[-7:]) if len(daily_pnl) >= 7 else 0.0,
            sharpe_30d=self._sharpe(daily_pnl[-30:]) if len(daily_pnl) >= 14 else 0.0,
            sharpe_90d=self._sharpe(daily_pnl) if len(daily_pnl) >= 30 else 0.0,
            sortino_7d=self._sortino(daily_pnl[-7:]) if len(daily_pnl) >= 7 else 0.0,
            sortino_30d=self._sortino(daily_pnl[-30:]) if len(daily_pnl) >= 14 else 0.0,
            sortino_90d=self._sortino(daily_pnl) if len(daily_pnl) >= 30 else 0.0,
            calmar_30d=self._calmar(daily_pnl[-30:]) if len(daily_pnl) >= 14 else 0.0,
            calmar_90d=self._calmar(daily_pnl) if len(daily_pnl) >= 30 else 0.0,
            total_pnl_7d=sum(daily_pnl[-7:]) if len(daily_pnl) >= 7 else sum(daily_pnl),
            total_pnl_30d=sum(daily_pnl[-30:]) if len(daily_pnl) >= 14 else sum(daily_pnl),
            max_drawdown_30d=self._max_drawdown(daily_pnl[-30:]),
            max_drawdown_90d=self._max_drawdown(daily_pnl),
            family_metrics={
                fam: {
                    "sharpe_30d": self._sharpe(vals[-30:]) if len(vals) >= 14 else 0.0,
                    "pnl_30d": sum(vals[-30:]) if len(vals) >= 14 else sum(vals),
                    "max_drawdown_30d": self._max_drawdown(vals[-30:]),
                }
                for fam, vals in family_daily.items()
            },
        )

    @staticmethod
    def _sharpe(returns: list[float]) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _sortino(returns: list[float]) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 0.0 if mean <= 0 else 10.0  # cap
        down_var = sum(r ** 2 for r in downside) / len(returns)
        down_std = math.sqrt(down_var) if down_var > 0 else 0.0
        if down_std == 0:
            return 0.0
        return (mean / down_std) * math.sqrt(252)

    @staticmethod
    def _calmar(returns: list[float]) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if not returns:
            return 0.0
        total = sum(returns)
        annualized = total * 252 / max(len(returns), 1)
        max_dd = PortfolioMetricsTracker._max_drawdown(returns)
        if max_dd <= 0:
            return 0.0
        return annualized / (max_dd * 100)

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        """Max drawdown as a fraction (0-1)."""
        if not returns:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            equity += r
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd
