# skills/portfolio_what_if.py
"""Portfolio what-if analysis — simplified linear rescaling for allocation proposals.

Input: historical daily family PnL series + proposed allocation weights.
Method: multiply each family's daily PnL by (new_weight / old_weight).
Output: portfolio Calmar, Sharpe, max drawdown under new allocation.

When trade-level data is available, computes enriched metrics:
- Intra-day max drawdown from per-trade equity curve
- Sortino ratio (downside deviation)
- Profit factor and win rate
- PnL by regime breakdown

Limitation: ignores interaction effects from coordination rules, capacity constraints.
Good enough for: allocation weight changes (~80% of proposals).
Not sufficient for: coordination rule changes, drawdown tier changes.
"""
from __future__ import annotations

import math
from collections import defaultdict

_INF_CAP = 9999.0


def _safe_round(val: float, digits: int = 4) -> float:
    """Round a float, capping ±inf and NaN for JSON safety."""
    if math.isnan(val):
        return 0.0
    if math.isinf(val):
        return _INF_CAP if val > 0 else -_INF_CAP
    return round(val, digits)


class PortfolioWhatIf:
    """Simplified what-if analysis for portfolio allocation changes."""

    def __init__(
        self,
        family_daily_pnl: dict[str, list[float]],
        current_weights: dict[str, float],
        family_trades: dict[str, list] | None = None,
    ) -> None:
        """Initialize with historical daily PnL per family and current weights.

        Args:
            family_daily_pnl: family_name → list of daily PnL values (oldest first).
            current_weights: family_name → current allocation weight (0-1).
            family_trades: family_name → list of TradeEvent objects (optional).
                When provided, enables trade-level enriched metrics.
        """
        self._current_weights = current_weights
        self._family_trades = family_trades

        if family_trades:
            # Derive daily PnL from trades (more accurate than pre-aggregated)
            derived = self._compute_daily_pnl_from_trades(family_trades)
            # Use derived if non-empty, otherwise fall back to provided
            self._family_pnl = derived if derived else family_daily_pnl
        else:
            self._family_pnl = family_daily_pnl

    def evaluate(self, proposed_weights: dict[str, float]) -> dict:
        """Evaluate a proposed allocation change.

        Returns dict with standard metrics. When trades are available,
        includes enriched metrics (sortino, profit_factor, regime breakdown).
        """
        current_portfolio = self._build_portfolio_series(self._current_weights)
        proposed_portfolio = self._build_portfolio_series(proposed_weights)

        current_sharpe = self._sharpe(current_portfolio)
        proposed_sharpe = self._sharpe(proposed_portfolio)
        current_calmar = self._calmar(current_portfolio)
        proposed_calmar = self._calmar(proposed_portfolio)

        # Use trade-level max drawdown when trades available
        if self._family_trades:
            current_dd = self._trade_level_max_drawdown(self._current_weights)
            proposed_dd = self._trade_level_max_drawdown(proposed_weights)
        else:
            current_dd = self._max_drawdown(current_portfolio)
            proposed_dd = self._max_drawdown(proposed_portfolio)

        result = {
            "current_sharpe": round(current_sharpe, 4),
            "proposed_sharpe": round(proposed_sharpe, 4),
            "sharpe_delta": round(proposed_sharpe - current_sharpe, 4),
            "current_calmar": round(current_calmar, 4),
            "proposed_calmar": round(proposed_calmar, 4),
            "calmar_delta": round(proposed_calmar - current_calmar, 4),
            "current_max_drawdown": round(current_dd, 4),
            "proposed_max_drawdown": round(proposed_dd, 4),
            "drawdown_delta": round(proposed_dd - current_dd, 4),
            "days_analyzed": len(current_portfolio),
            "method": "trade_level_rescaling" if self._family_trades else "linear_rescaling",
        }

        if self._family_trades:
            result.update(self._compute_enriched_metrics(
                proposed_weights, current_portfolio, proposed_portfolio,
            ))

        return result

    def _compute_enriched_metrics(
        self,
        proposed_weights: dict[str, float],
        current_daily: list[float],
        proposed_daily: list[float],
    ) -> dict:
        """Compute trade-level enriched metrics for current and proposed weights."""
        # Sortino from daily series (reuse already-computed series)
        current_sortino = self._sortino(current_daily)
        proposed_sortino = self._sortino(proposed_daily)

        # Collect all scaled trade PnLs
        current_trade_pnls = self._scale_trade_pnls(self._current_weights)
        proposed_trade_pnls = self._scale_trade_pnls(proposed_weights)

        # Profit factor
        current_pf = self._profit_factor(current_trade_pnls)
        proposed_pf = self._profit_factor(proposed_trade_pnls)

        # Trade counts
        total_trades = sum(len(ts) for ts in self._family_trades.values())
        trades_by_family = {fam: len(ts) for fam, ts in self._family_trades.items()}

        # PnL by regime
        current_regime = self._pnl_by_regime(self._current_weights)
        proposed_regime = self._pnl_by_regime(proposed_weights)

        return {
            "current_sortino": _safe_round(current_sortino),
            "proposed_sortino": _safe_round(proposed_sortino),
            "sortino_delta": _safe_round(proposed_sortino - current_sortino),
            "current_profit_factor": _safe_round(current_pf),
            "proposed_profit_factor": _safe_round(proposed_pf),
            "profit_factor_delta": _safe_round(proposed_pf - current_pf),
            "total_trades": total_trades,
            "trades_by_family": trades_by_family,
            "current_pnl_by_regime": {k: round(v, 4) for k, v in current_regime.items()},
            "proposed_pnl_by_regime": {k: round(v, 4) for k, v in proposed_regime.items()},
        }

    def _scale_trade_pnls(self, weights: dict[str, float]) -> list[float]:
        """Get all trade PnLs scaled by family weight ratios."""
        pnls = []
        for family, trades in self._family_trades.items():
            current_w = self._current_weights.get(family, 0.0)
            proposed_w = weights.get(family, current_w)
            if current_w <= 0:
                continue
            scale = proposed_w / current_w
            for trade in trades:
                pnl = getattr(trade, "pnl", None)
                if pnl is not None:
                    pnls.append(pnl * scale)
        return pnls

    def _trade_level_max_drawdown(self, weights: dict[str, float]) -> float:
        """Compute max drawdown from per-trade equity curve (intra-day granularity)."""
        # Collect all trades with timestamps and scaled PnL
        timed_trades = []
        for family, trades in self._family_trades.items():
            current_w = self._current_weights.get(family, 0.0)
            proposed_w = weights.get(family, current_w)
            if current_w <= 0:
                continue
            scale = proposed_w / current_w
            for trade in trades:
                pnl = getattr(trade, "pnl", None)
                if pnl is None:
                    continue
                # Use exit_time for chronological ordering
                exit_time = getattr(trade, "exit_time", None)
                entry_time = getattr(trade, "entry_time", None)
                ts = exit_time or entry_time
                if ts is None:
                    continue
                timed_trades.append((ts, pnl * scale))

        if not timed_trades:
            return 0.0

        # Sort chronologically
        timed_trades.sort(key=lambda x: x[0])

        # Build equity curve and track max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for _, pnl in timed_trades:
            equity += pnl
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _pnl_by_regime(self, weights: dict[str, float]) -> dict[str, float]:
        """Sum scaled PnL by market regime across all families."""
        regime_pnl: dict[str, float] = defaultdict(float)
        for family, trades in self._family_trades.items():
            current_w = self._current_weights.get(family, 0.0)
            proposed_w = weights.get(family, current_w)
            if current_w <= 0:
                continue
            scale = proposed_w / current_w
            for trade in trades:
                regime = getattr(trade, "market_regime", "") or ""
                pnl = getattr(trade, "pnl", None)
                if regime and pnl is not None:
                    regime_pnl[regime] += pnl * scale
        return dict(regime_pnl)

    @staticmethod
    def _compute_daily_pnl_from_trades(
        family_trades: dict[str, list],
    ) -> dict[str, list[float]]:
        """Group trades by date and sum PnL per day per family.

        Returns family_name → list[float] in chronological order (oldest first).
        """
        family_date_pnl: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        all_dates: set[str] = set()

        for family, trades in family_trades.items():
            for trade in trades:
                date_str = PortfolioWhatIf._trade_date(trade)
                if date_str:
                    pnl = getattr(trade, "pnl", 0.0) or 0.0
                    family_date_pnl[family][date_str] += pnl
                    all_dates.add(date_str)

        if not all_dates:
            return {}

        sorted_dates = sorted(all_dates)
        result: dict[str, list[float]] = {}
        for family, date_pnl in family_date_pnl.items():
            result[family] = [date_pnl.get(d, 0.0) for d in sorted_dates]

        return result

    @staticmethod
    def _trade_date(trade) -> str:
        """Extract date string (YYYY-MM-DD) from trade's exit_time or entry_time."""
        exit_time = getattr(trade, "exit_time", None)
        entry_time = getattr(trade, "entry_time", None)
        ts = exit_time or entry_time
        if ts is None:
            return ""
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d")
        # Handle string timestamps
        ts_str = str(ts)
        if len(ts_str) >= 10:
            return ts_str[:10]
        return ""

    def _build_portfolio_series(self, weights: dict[str, float]) -> list[float]:
        """Build daily portfolio PnL by rescaling family PnL series."""
        if not self._family_pnl:
            return []

        lengths = [len(v) for v in self._family_pnl.values() if v]
        if not lengths:
            return []
        max_len = max(lengths)
        portfolio = [0.0] * max_len

        for family, pnl_series in self._family_pnl.items():
            current_w = self._current_weights.get(family, 0.0)
            proposed_w = weights.get(family, current_w)

            if current_w <= 0:
                continue  # Can't rescale from zero

            scale = proposed_w / current_w

            for i, pnl in enumerate(pnl_series):
                portfolio[i] += pnl * scale

        return portfolio

    @staticmethod
    def _sharpe(returns: list[float]) -> float:
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
        """Sortino ratio: mean / downside_deviation * sqrt(252)."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside_sq = [r ** 2 for r in returns if r < 0]
        if not downside_sq:
            return 0.0 if mean <= 0 else float("inf")
        downside_dev = math.sqrt(sum(downside_sq) / len(returns))
        if downside_dev == 0:
            return 0.0
        return (mean / downside_dev) * math.sqrt(252)

    @staticmethod
    def _calmar(returns: list[float]) -> float:
        if not returns:
            return 0.0
        total = sum(returns)
        annualized = total * 252 / max(len(returns), 1)
        max_dd = PortfolioWhatIf._max_drawdown(returns)
        if max_dd <= 0:
            return 0.0
        return annualized / (max_dd * 100)

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
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

    @staticmethod
    def _profit_factor(pnl_values: list[float]) -> float:
        """Profit factor: sum(wins) / abs(sum(losses))."""
        wins = sum(p for p in pnl_values if p > 0)
        losses = sum(p for p in pnl_values if p < 0)
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / abs(losses)
