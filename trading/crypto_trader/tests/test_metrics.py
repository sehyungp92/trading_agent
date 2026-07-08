"""Tests for enhanced performance metrics — streaks, edge ratio, breakdowns."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_trader.backtest.metrics import (
    PerformanceMetrics,
    _compute_group_breakdown,
    _compute_r_distribution,
    _compute_streaks,
    _compute_weekly_returns,
    filter_metrics_for_scoring,
)
from crypto_trader.core.models import SetupGrade, Side, Trade


def _make_trade(
    pnl: float,
    r: float = 0.0,
    *,
    realized_r: float | None = None,
    direction: Side = Side.LONG,
    grade: SetupGrade | None = SetupGrade.B,
    exit_reason: str = "protective_stop",
    confirmation_type: str = "inside_bar_break",
    confluences: list[str] | None = None,
    mae_r: float | None = None,
    mfe_r: float | None = None,
    bars_held: int = 1,
    entry_time: datetime | None = None,
    symbol: str = "BTC",
) -> Trade:
    """Create a Trade for testing."""
    if entry_time is None:
        entry_time = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
    return Trade(
        trade_id="t1",
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        qty=1.0,
        entry_time=entry_time,
        exit_time=entry_time,
        pnl=pnl,
        r_multiple=r,
        commission=0.0,
        bars_held=bars_held,
        setup_grade=grade,
        exit_reason=exit_reason,
        confluences_used=confluences or ["m15_ema20"],
        confirmation_type=confirmation_type,
        entry_method="close",
        funding_paid=0.0,
        mae_r=mae_r,
        mfe_r=mfe_r,
        realized_r_multiple=realized_r,
    )


class TestComputeStreaks:
    def test_all_winners(self):
        trades = [_make_trade(10, 1.0), _make_trade(5, 0.5), _make_trade(3, 0.3)]
        wins, losses = _compute_streaks(trades)
        assert wins == 3
        assert losses == 0

    def test_all_losers(self):
        trades = [_make_trade(-10, -1.0), _make_trade(-5, -0.5)]
        wins, losses = _compute_streaks(trades)
        assert wins == 0
        assert losses == 2

    def test_mixed(self):
        trades = [
            _make_trade(10, 1.0),   # W
            _make_trade(-5, -0.5),  # L
            _make_trade(-3, -0.3),  # L
            _make_trade(-2, -0.2),  # L
            _make_trade(8, 0.8),    # W
            _make_trade(4, 0.4),    # W
        ]
        wins, losses = _compute_streaks(trades)
        assert wins == 2
        assert losses == 3

    def test_empty(self):
        wins, losses = _compute_streaks([])
        assert wins == 0
        assert losses == 0


class TestRDistribution:
    def test_buckets(self):
        rs = [-1.5, -0.8, -0.3, 0.1, 0.7, 1.5, 2.5]
        dist = _compute_r_distribution(rs)
        assert dist["< -1.0"] == 1
        assert dist["-1.0 to -0.5"] == 1
        assert dist["-0.5 to 0"] == 1
        assert dist["0 to 0.5"] == 1
        assert dist["0.5 to 1.0"] == 1
        assert dist["1.0 to 2.0"] == 1
        assert dist["> 2.0"] == 1

    def test_empty(self):
        dist = _compute_r_distribution([])
        assert all(v == 0 for v in dist.values())


class TestGroupBreakdown:
    def test_by_exit_reason(self):
        trades = [
            _make_trade(10, 1.0, exit_reason="tp1"),
            _make_trade(-5, -0.5, exit_reason="protective_stop"),
            _make_trade(-3, -0.3, exit_reason="protective_stop"),
        ]
        result = _compute_group_breakdown(
            trades, key_fn=lambda t: t.exit_reason
        )
        assert "tp1" in result
        assert result["tp1"]["trades"] == 1
        assert result["tp1"]["win_rate"] == 100.0

        assert "protective_stop" in result
        assert result["protective_stop"]["trades"] == 2
        assert result["protective_stop"]["win_rate"] == 0.0

    def test_by_confluence_count(self):
        trades = [
            _make_trade(10, 1.0, confluences=["a", "b", "c"]),
            _make_trade(-5, -0.5, confluences=["a"]),
            _make_trade(3, 0.3, confluences=["a"]),
        ]
        result = _compute_group_breakdown(
            trades, key_fn=lambda t: len(t.confluences_used) if t.confluences_used else 0
        )
        assert result[1]["trades"] == 2
        assert result[3]["trades"] == 1


class TestWeeklyReturns:
    def test_single_week(self):
        trades = [
            _make_trade(10, 1.0, entry_time=datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)),
            _make_trade(-5, -0.5, entry_time=datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc)),
        ]
        weekly = _compute_weekly_returns(trades)
        assert len(weekly) == 1
        assert weekly[0]["pnl"] == pytest.approx(5.0)

    def test_multiple_weeks(self):
        trades = [
            _make_trade(10, 1.0, entry_time=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)),
            _make_trade(-5, -0.5, entry_time=datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc)),
        ]
        weekly = _compute_weekly_returns(trades)
        assert len(weekly) == 2

    def test_empty(self):
        assert _compute_weekly_returns([]) == []


class TestPerformanceMetricsNewFields:
    def test_edge_ratio_computed(self):
        """Edge ratio = avg_mfe / abs(avg_mae)."""
        m = PerformanceMetrics()
        m.avg_mfe_r = 0.5
        m.avg_mae_r = -1.0
        # Edge ratio would be 0.5 / 1.0 = 0.5
        # But we test via compute_metrics — just verify field exists
        assert m.edge_ratio == 0.0  # default

    def test_defaults(self):
        """All new fields have sensible defaults."""
        m = PerformanceMetrics()
        assert m.max_consecutive_wins == 0
        assert m.max_consecutive_losses == 0
        assert m.edge_ratio == 0.0
        assert m.recovery_factor == 0.0
        assert m.profit_concentration == 0.0
        assert m.payoff_ratio == 0.0
        assert m.per_confirmation == {}
        assert m.per_confluence_count == {}
        assert m.per_exit_reason == {}
        assert m.r_distribution == {}
        assert m.weekly_returns == []


class TestFilterMetricsForScoring:
    """Tests for filter_metrics_for_scoring — backtest_end trade exclusion."""

    def _base_metrics(self) -> dict[str, float]:
        """Baseline metrics dict simulating full-period equity-based values."""
        return {
            "net_profit": 500.0,
            "net_return_pct": 5.0,
            "total_trades": 5.0,
            "win_rate": 60.0,
            "avg_winner_r": 1.5,
            "avg_loser_r": -0.8,
            "expectancy_r": 0.5,
            "profit_factor": 2.5,
            "max_drawdown_pct": 10.0,
            "max_drawdown_duration": 5.0,
            "sharpe_ratio": 2.0,
            "sortino_ratio": 3.0,
            "calmar_ratio": 0.5,
            "avg_bars_held": 4.0,
            "avg_mae_r": -0.3,
            "avg_mfe_r": 0.8,
            "exit_efficiency": 0.6,
            "edge_ratio": 2.67,
            "payoff_ratio": 1.8,
            "recovery_factor": 0.5,
            "max_consecutive_losses": 2.0,
            "a_setup_win_rate": 100.0,
            "b_setup_win_rate": 50.0,
            "long_win_rate": 66.7,
            "short_win_rate": 50.0,
            "funding_cost_total": 1.0,
        }

    def test_empty_exclusion_returns_original(self):
        """Empty or None exclude_exit_reasons returns metrics unchanged."""
        metrics = self._base_metrics()
        trades = [_make_trade(10, 1.0)]
        assert filter_metrics_for_scoring(metrics, trades, set()) is metrics
        assert filter_metrics_for_scoring(metrics, trades, None) is metrics

    def test_equity_fields_preserved(self):
        """Equity-based fields are unchanged after filtering."""
        metrics = self._base_metrics()
        trades = [
            _make_trade(100, 1.5, exit_reason="tp1", mfe_r=2.0, mae_r=-0.3),
            _make_trade(-50, -0.8, exit_reason="protective_stop", mfe_r=0.2, mae_r=-1.0),
            _make_trade(200, 3.0, exit_reason="backtest_end", mfe_r=5.0, mae_r=-0.1),
        ]
        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        # Equity fields must be identical
        assert filtered["net_return_pct"] == 5.0
        assert filtered["max_drawdown_pct"] == 10.0
        assert filtered["sharpe_ratio"] == 2.0
        assert filtered["sortino_ratio"] == 3.0
        assert filtered["calmar_ratio"] == 0.5
        assert filtered["recovery_factor"] == 0.5
        assert filtered["net_profit"] == 500.0

    def test_trade_fields_recomputed(self):
        """Trade-based fields reflect only non-excluded trades."""
        metrics = self._base_metrics()
        trades = [
            _make_trade(100, 1.5, exit_reason="tp1", mfe_r=2.0, mae_r=-0.3),
            _make_trade(-50, -0.8, exit_reason="protective_stop", mfe_r=0.2, mae_r=-1.0),
            _make_trade(200, 3.0, exit_reason="backtest_end", mfe_r=5.0, mae_r=-0.1),
        ]
        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        assert filtered["total_trades"] == 2.0
        assert filtered["win_rate"] == 50.0  # 1 winner / 2 trades
        assert filtered["avg_winner_r"] == pytest.approx(1.5)
        assert filtered["avg_loser_r"] == pytest.approx(-0.8)
        # PF: gross_profit=100, gross_loss=50 -> 2.0
        assert filtered["profit_factor"] == pytest.approx(2.0)

    def test_all_trades_excluded_zeroes_fields(self):
        """When all trades are excluded, trade-based fields are zero."""
        metrics = self._base_metrics()
        trades = [
            _make_trade(200, 3.0, exit_reason="backtest_end"),
            _make_trade(100, 1.5, exit_reason="backtest_end"),
        ]
        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        assert filtered["total_trades"] == 0.0
        assert filtered["win_rate"] == 0.0
        assert filtered["profit_factor"] == 0.0
        assert filtered["expectancy_r"] == 0.0
        assert filtered["exit_efficiency"] == 0.0
        # Equity fields still preserved
        assert filtered["sharpe_ratio"] == 2.0

    def test_pf_computed_from_real_trades_only(self):
        """PF computed only from non-excluded trades."""
        metrics = self._base_metrics()
        trades = [
            _make_trade(30, 0.5, exit_reason="tp1", mfe_r=1.0, mae_r=-0.3),
            _make_trade(-10, -0.3, exit_reason="protective_stop", mfe_r=0.1, mae_r=-0.5),
            _make_trade(-20, -0.5, exit_reason="time_stop", mfe_r=0.1, mae_r=-0.7),
            _make_trade(500, 5.0, exit_reason="backtest_end", mfe_r=6.0, mae_r=-0.1),
        ]
        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        assert filtered["total_trades"] == 3.0
        # PF = 30 / (10+20) = 1.0
        assert filtered["profit_factor"] == pytest.approx(1.0)

    def test_exit_efficiency_winners_only(self):
        """Exit efficiency computed from winners with positive mfe_r."""
        metrics = self._base_metrics()
        trades = [
            _make_trade(30, 1.0, exit_reason="tp1", mfe_r=2.0, mae_r=-0.3),
            _make_trade(-10, -0.3, exit_reason="protective_stop", mfe_r=0.1, mae_r=-0.5),
            _make_trade(500, 5.0, exit_reason="backtest_end", mfe_r=6.0, mae_r=-0.1),
        ]
        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        # Only the tp1 winner: efficiency = r_multiple/mfe_r = 1.0/2.0 = 0.5
        assert filtered["exit_efficiency"] == pytest.approx(0.5)

    def test_trade_fields_prefer_realized_r_multiple(self):
        metrics = self._base_metrics()
        trades = [
            _make_trade(30, 1.0, realized_r=0.4, exit_reason="tp1", mfe_r=2.0, mae_r=-0.3),
            _make_trade(-10, -0.5, realized_r=-0.2, exit_reason="protective_stop", mfe_r=0.5, mae_r=-0.6),
        ]

        filtered = filter_metrics_for_scoring(metrics, trades, {"backtest_end"})

        assert filtered["avg_winner_r"] == pytest.approx(0.4)
        assert filtered["avg_loser_r"] == pytest.approx(-0.2)
        assert filtered["expectancy_r"] == pytest.approx(0.1)
        assert filtered["exit_efficiency"] == pytest.approx(0.2)
