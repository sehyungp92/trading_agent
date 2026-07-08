"""Tests for portfolio state."""

from datetime import date, datetime, timezone

import pytest

from crypto_trader.core.models import Side
from crypto_trader.portfolio.state import OpenRisk, PortfolioState


class TestOpenRisk:
    def test_creation(self):
        r = OpenRisk(strategy_id="momentum", symbol="BTC", direction=Side.LONG, risk_R=1.0)
        assert r.strategy_id == "momentum"
        assert r.symbol == "BTC"
        assert r.direction == Side.LONG
        assert r.risk_R == 1.0
        assert r.entry_time is None


class TestPortfolioState:
    def _make_state(self, equity=10000.0):
        return PortfolioState(equity=equity, peak_equity=equity)

    def test_total_heat_empty(self):
        s = self._make_state()
        assert s.total_heat_R() == 0.0

    def test_total_heat_with_risks(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 0.5))
        assert s.total_heat_R() == 1.5

    def test_directional_risk(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 0.5))
        s.add_risk(OpenRisk("breakout", "SOL", Side.LONG, 0.8))
        assert s.directional_risk_R(Side.LONG) == 1.8
        assert s.directional_risk_R(Side.SHORT) == 0.5

    def test_symbol_risk_all_directions(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "BTC", Side.LONG, 0.5))
        assert s.symbol_risk_R("BTC") == 1.5
        assert s.symbol_risk_R("ETH") == 0.0

    def test_symbol_risk_with_direction(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "BTC", Side.SHORT, 0.5))
        assert s.symbol_risk_R("BTC", Side.LONG) == 1.0
        assert s.symbol_risk_R("BTC", Side.SHORT) == 0.5

    def test_strategy_position_count(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("momentum", "ETH", Side.LONG, 0.5))
        s.add_risk(OpenRisk("trend", "SOL", Side.SHORT, 0.8))
        assert s.strategy_position_count("momentum") == 2
        assert s.strategy_position_count("trend") == 1
        assert s.strategy_position_count("breakout") == 0

    def test_total_positions(self):
        s = self._make_state()
        assert s.total_positions() == 0
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 0.5))
        assert s.total_positions() == 2

    def test_dd_pct_no_drawdown(self):
        s = self._make_state(10000.0)
        assert s.dd_pct() == 0.0

    def test_dd_pct_with_drawdown(self):
        s = self._make_state(10000.0)
        s.equity = 9200.0
        assert s.dd_pct() == pytest.approx(0.08)

    def test_dd_pct_zero_peak(self):
        s = PortfolioState(equity=0.0, peak_equity=0.0)
        assert s.dd_pct() == 0.0

    def test_remove_risk_found(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        s.add_risk(OpenRisk("trend", "ETH", Side.SHORT, 0.5))
        removed = s.remove_risk("momentum", "BTC")
        assert removed is not None
        assert removed.strategy_id == "momentum"
        assert s.total_positions() == 1

    def test_remove_risk_not_found(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        assert s.remove_risk("trend", "BTC") is None
        assert s.total_positions() == 1

    def test_update_equity_tracks_peak(self):
        s = self._make_state(10000.0)
        s.update_equity(11000.0)
        assert s.equity == 11000.0
        assert s.peak_equity == 11000.0
        s.update_equity(10500.0)
        assert s.equity == 10500.0
        assert s.peak_equity == 11000.0

    def test_reset_daily(self):
        s = self._make_state()
        s.daily_pnl_R["momentum"] = -1.5
        s.portfolio_daily_pnl_R = -1.5
        s.reset_daily(date(2026, 4, 20))
        assert s.daily_pnl_R == {}
        assert s.portfolio_daily_pnl_R == 0.0
        assert s.current_day == date(2026, 4, 20)

    def test_strategy_daily_pnl_default(self):
        s = self._make_state()
        assert s.strategy_daily_pnl_R("unknown") == 0.0

    def test_to_dict(self):
        s = self._make_state()
        s.add_risk(OpenRisk("momentum", "BTC", Side.LONG, 1.0))
        d = s.to_dict()
        assert d["equity"] == 10000.0
        assert len(d["open_risks"]) == 1
        assert d["open_risks"][0]["direction"] == "LONG"

    def test_from_dict_restores_open_risks_and_daily_state(self):
        entry_time = datetime(2026, 6, 4, 12, tzinfo=timezone.utc)
        payload = {
            "equity": 10000.0,
            "peak_equity": 10500.0,
            "daily_pnl_R": {"momentum": -0.25},
            "portfolio_daily_pnl_R": -0.25,
            "current_day": "2026-06-04",
            "open_risks": [{
                "strategy_id": "momentum",
                "symbol": "BTC",
                "direction": "LONG",
                "risk_R": 0.4,
                "entry_time": entry_time.isoformat(),
                "risk_id": "intent_1",
                "intent_id": "intent_1",
                "client_order_id": "client_1",
                "order_qty": 1.0,
                "filled_qty": 0.4,
                "applied_fill_ids": ["fill_1"],
            }],
        }

        restored = PortfolioState.from_dict(payload)

        assert restored.peak_equity == 10500.0
        assert restored.strategy_daily_pnl_R("momentum") == -0.25
        assert restored.current_day == date(2026, 6, 4)
        assert len(restored.open_risks) == 1
        assert restored.open_risks[0].risk_id == "intent_1"
        assert restored.open_risks[0].filled_qty == pytest.approx(0.4)
