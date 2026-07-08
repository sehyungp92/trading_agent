"""Tests for Sharpe/Sortino/net_return metric calculation fixes."""

from datetime import datetime, date, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from crypto_trader.backtest.diagnostics import generate_diagnostics
from crypto_trader.backtest.metrics import PerformanceMetrics, compute_metrics
from crypto_trader.core.models import Side, SetupGrade, TerminalMark, Trade


def _make_trade(pnl, entry_time=None, exit_time=None, r_multiple=None, **kwargs):
    """Create a minimal Trade for metrics testing."""
    if entry_time is None:
        entry_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    if exit_time is None:
        exit_time = entry_time + timedelta(hours=1)
    mae_r = kwargs.pop("mae_r", None)
    mfe_r = kwargs.pop("mfe_r", None)
    return Trade(
        trade_id="1",
        symbol="BTC",
        direction=Side.LONG,
        entry_price=50000.0,
        exit_price=51000.0 if pnl > 0 else 49000.0,
        qty=1.0,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl=pnl,
        r_multiple=r_multiple,
        commission=0.0,
        bars_held=10,
        setup_grade=None,
        exit_reason="stop",
        confluences_used=None,
        confirmation_type=None,
        entry_method=None,
        funding_paid=0.0,
        mae_r=mae_r,
        mfe_r=mfe_r,
        **kwargs,
    )


def _make_broker_mock(trades, equity_history, initial_equity=10000.0, liquidation_history=None):
    """Create a mock broker with given trades and equity history."""
    broker = SimpleNamespace(
        _closed_trades=trades,
        _equity_history=equity_history,
        _liquidation_equity_history=liquidation_history or [],
        initial_equity=initial_equity,
        _initial_equity=initial_equity,
    )
    return broker


class TestNetReturnFromEquityCurve:
    """Test that net_return_pct uses equity curve, not trade sums."""

    def test_net_return_from_equity_curve(self):
        """net_return_pct should be (final_equity - initial) / initial * 100."""
        trades = [_make_trade(pnl=500.0)]
        equity = [
            (datetime(2026, 3, 1, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 2, tzinfo=timezone.utc), 10800.0),
        ]
        broker = _make_broker_mock(trades, equity, initial_equity=10000.0)
        m = compute_metrics(broker)
        # Equity says 10800, so return = 8%
        assert m.net_return_pct == pytest.approx(8.0)
        assert m.net_profit == pytest.approx(800.0)

    def test_net_return_differs_from_trade_sum(self):
        """When equity curve differs from trade sum, equity wins."""
        # Trades say +200, equity says -100 (due to commission/funding not in trades)
        trades = [_make_trade(pnl=200.0)]
        equity = [
            (datetime(2026, 3, 1, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 2, tzinfo=timezone.utc), 9900.0),
        ]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)
        assert m.net_profit == pytest.approx(-100.0)
        assert m.net_return_pct == pytest.approx(-1.0)

    def test_net_return_fallback_no_equity(self):
        """Without equity history, falls back to trade sum."""
        trades = [_make_trade(pnl=500.0)]
        broker = _make_broker_mock(trades, equity_history=[])
        m = compute_metrics(broker)
        assert m.net_profit == pytest.approx(500.0)
        assert m.net_return_pct == pytest.approx(5.0)

    def test_net_return_prefers_liquidation_equity_curve(self):
        trades = [_make_trade(pnl=200.0)]
        raw_equity = [
            (datetime(2026, 3, 1, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 2, tzinfo=timezone.utc), 10500.0),
        ]
        liquidation_equity = [
            (datetime(2026, 3, 1, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 2, tzinfo=timezone.utc), 9800.0),
        ]
        broker = _make_broker_mock(
            trades,
            equity_history=raw_equity,
            liquidation_history=liquidation_equity,
        )

        m = compute_metrics(broker)

        assert m.net_profit == pytest.approx(-200.0)
        assert m.net_return_pct == pytest.approx(-2.0)
        assert m.max_drawdown_pct == pytest.approx(2.0)

    def test_metrics_prefer_realized_r_multiple(self):
        trades = [
            _make_trade(pnl=30.0, r_multiple=1.0, realized_r_multiple=0.4, mfe_r=2.0),
            _make_trade(pnl=-10.0, r_multiple=-0.5, realized_r_multiple=-0.2, mfe_r=0.5),
        ]
        broker = _make_broker_mock(trades, equity_history=[])

        m = compute_metrics(broker)

        assert m.avg_winner_r == pytest.approx(0.4)
        assert m.avg_loser_r == pytest.approx(-0.2)
        assert m.expectancy_r == pytest.approx(0.1)
        assert m.exit_efficiency == pytest.approx(0.2)

    def test_metrics_accept_duck_typed_realized_r_without_property(self):
        trade = SimpleNamespace(
            net_pnl=20.0,
            realized_r_multiple=0.3,
            r_multiple=1.1,
            bars_held=4,
            mae_r=-0.2,
            mfe_r=1.0,
            setup_grade=None,
            direction=Side.LONG,
            funding_paid=0.0,
            symbol="BTC",
            confirmation_type="inside_bar_break",
            confluences_used=["m15_ema20"],
            exit_reason="tp1",
            entry_time=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        broker = _make_broker_mock([trade], equity_history=[])

        m = compute_metrics(broker)

        assert m.avg_winner_r == pytest.approx(0.3)
        assert m.expectancy_r == pytest.approx(0.3)
        assert m.exit_efficiency == pytest.approx(0.3)


class TestSharpeDaily:
    """Test that Sharpe uses daily equity returns with sqrt(365)."""

    def test_sharpe_uses_daily_returns(self):
        """Sharpe should be calculated from daily equity changes."""
        # Create equity curve: 5 days of returns
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        equity = []
        values = [10000, 10100, 10050, 10200, 10150]
        for i, v in enumerate(values):
            # Multiple entries per day to test last-of-day logic
            ts = base + timedelta(days=i, hours=9)
            equity.append((ts, v - 50))
            ts2 = base + timedelta(days=i, hours=17)
            equity.append((ts2, v))

        trades = [_make_trade(pnl=150.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)

        # Manually compute expected Sharpe
        daily_rets = np.array([
            (10100 - 10000) / 10000,  # +1%
            (10050 - 10100) / 10100,  # -0.495%
            (10200 - 10050) / 10050,  # +1.493%
            (10150 - 10200) / 10200,  # -0.49%
        ])
        expected_sharpe = float(
            np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(365)
        )
        assert m.sharpe_ratio == pytest.approx(expected_sharpe, rel=1e-6)

    def test_sharpe_annualizes_365(self):
        """Sharpe should use sqrt(365) for crypto, not sqrt(252)."""
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        equity = [
            (base, 10000.0),
            (base + timedelta(days=1), 10100.0),
            (base + timedelta(days=2), 10200.0),
        ]
        trades = [_make_trade(pnl=200.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)

        # With all-positive returns and 2 data points, std > 0
        daily_rets = np.array([0.01, 100 / 10100])
        expected = float(np.mean(daily_rets) / np.std(daily_rets, ddof=1) * np.sqrt(365))
        assert m.sharpe_ratio == pytest.approx(expected, rel=1e-6)

    def test_single_trading_day_sharpe_zero(self):
        """With only one day in equity curve, Sharpe should be 0."""
        equity = [
            (datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc), 10000.0),
            (datetime(2026, 3, 1, 17, 0, tzinfo=timezone.utc), 10100.0),
        ]
        trades = [_make_trade(pnl=100.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)
        assert m.sharpe_ratio == 0.0

    def test_no_equity_history_sharpe_zero(self):
        """Without equity history, Sharpe should be 0."""
        trades = [_make_trade(pnl=100.0), _make_trade(pnl=-50.0)]
        broker = _make_broker_mock(trades, equity_history=[])
        m = compute_metrics(broker)
        assert m.sharpe_ratio == 0.0


class TestSortinoDaily:
    """Test that Sortino uses only downside daily returns."""

    def test_sortino_uses_downside_only(self):
        """Sortino should use std of negative daily returns only."""
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        equity = [
            (base, 10000.0),
            (base + timedelta(days=1), 10100.0),     # +1%
            (base + timedelta(days=2), 10050.0),      # -0.495%
            (base + timedelta(days=3), 10200.0),      # +1.49%
            (base + timedelta(days=4), 10100.0),      # -0.98%
        ]
        trades = [_make_trade(pnl=100.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)

        daily_rets = np.array([
            (10100 - 10000) / 10000,
            (10050 - 10100) / 10100,
            (10200 - 10050) / 10050,
            (10100 - 10200) / 10200,
        ])
        downside = daily_rets[daily_rets < 0]
        expected = float(
            np.mean(daily_rets) / np.std(downside, ddof=1) * np.sqrt(365)
        )
        assert m.sortino_ratio == pytest.approx(expected, rel=1e-6)

    def test_all_positive_returns_sortino_zero(self):
        """If all daily returns are positive, Sortino should be 0."""
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        equity = [
            (base, 10000.0),
            (base + timedelta(days=1), 10100.0),
            (base + timedelta(days=2), 10200.0),
            (base + timedelta(days=3), 10300.0),
        ]
        trades = [_make_trade(pnl=300.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)
        # No downside returns → no downside std → Sortino = 0
        assert m.sortino_ratio == 0.0


class TestCalmarAutoCorrect:
    """Test that Calmar auto-corrects via equity-based net_return_pct."""

    def test_calmar_uses_equity_return(self):
        """Calmar = net_return_pct / max_drawdown_pct, using equity-based return."""
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        equity = [
            (base, 10000.0),
            (base + timedelta(days=1), 10500.0),     # +5%
            (base + timedelta(days=2), 9500.0),       # DD from peak: 9.52%
            (base + timedelta(days=3), 10200.0),      # recover
        ]
        trades = [_make_trade(pnl=200.0)]
        broker = _make_broker_mock(trades, equity)
        m = compute_metrics(broker)
        # net_return_pct = (10200 - 10000) / 10000 * 100 = 2.0
        assert m.net_return_pct == pytest.approx(2.0)
        # max_drawdown_pct from 10500 to 9500 = (10500-9500)/10500 * 100 ≈ 9.524%
        expected_dd = (10500 - 9500) / 10500 * 100
        assert m.max_drawdown_pct == pytest.approx(expected_dd, rel=1e-3)
        assert m.calmar_ratio == pytest.approx(2.0 / expected_dd, rel=1e-3)


class TestTerminalMarkMetrics:
    def test_metrics_include_terminal_mark_pnl_without_realized_trades(self):
        broker = SimpleNamespace(
            _closed_trades=[],
            _terminal_marks=[
                TerminalMark(
                    symbol="BTC",
                    direction=Side.LONG,
                    qty=1.0,
                    timestamp=datetime(2026, 3, 2, tzinfo=timezone.utc),
                    entry_price=100.0,
                    mark_price_raw=105.0,
                    mark_price_net_liquidation=104.0,
                    unrealized_pnl_net=4.0,
                    unrealized_r_at_mark=0.4,
                )
            ],
            _equity_history=[],
            initial_equity=10_000.0,
            _initial_equity=10_000.0,
        )

        m = compute_metrics(broker)

        assert m.total_trades == 0
        assert m.realized_pnl_net == 0.0
        assert m.terminal_mark_count == 1
        assert m.terminal_mark_pnl_net == pytest.approx(4.0)
        assert m.net_profit == pytest.approx(4.0)

    def test_generate_diagnostics_reports_terminal_marks_without_backtest_end(self):
        terminal_marks = [
            TerminalMark(
                symbol="BTC",
                direction=Side.LONG,
                qty=1.0,
                timestamp=datetime(2026, 3, 2, tzinfo=timezone.utc),
                entry_price=100.0,
                mark_price_raw=105.0,
                mark_price_net_liquidation=104.0,
                unrealized_pnl_net=4.0,
                unrealized_r_at_mark=0.4,
            )
        ]

        text = generate_diagnostics(
            trades=[],
            initial_equity=10_000.0,
            terminal_marks=terminal_marks,
        )

        assert "Terminal marks: 1 open position(s)" in text
        assert "Net liquidation P&L" in text
        assert "backtest_end" not in text
