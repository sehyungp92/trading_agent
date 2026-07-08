"""Tests for position sizing, leverage, and risk checks."""

from __future__ import annotations

import pytest

from crypto_trader.core.models import Position, SetupGrade, Side
from crypto_trader.strategy.momentum.config import RiskParams
from crypto_trader.strategy.momentum.sizing import PositionSizer, SizingResult


class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(RiskParams(risk_pct_a=0.0075, risk_pct_b=0.004))

    def test_a_grade_sizing(self):
        result, reason = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=500,
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        # Risk amount = 10000 * 0.0075 = 75
        # Qty = 75 / 500 = 0.15
        assert result.qty == pytest.approx(0.15, rel=0.01)
        assert result.risk_pct_actual > 0

    def test_b_grade_sizing(self):
        result, reason = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=500,
            setup_grade=SetupGrade.B,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        # Risk amount = 10000 * 0.004 = 40
        # Qty = 40 / 500 = 0.08
        assert result.qty == pytest.approx(0.08, rel=0.01)

    def test_c_grade_rejected(self):
        result, reason = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=500,
            setup_grade=SetupGrade.C,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is None
        assert reason == "c_grade"

    def test_max_concurrent_positions(self):
        positions = [
            Position(symbol="ETH", direction=Side.LONG, qty=1.0, avg_entry=3000),
            Position(symbol="SOL", direction=Side.LONG, qty=10.0, avg_entry=100),
            Position(symbol="AVAX", direction=Side.LONG, qty=50.0, avg_entry=30),
        ]
        result, reason = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=500,
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=positions,
            direction=Side.LONG,
        )
        assert result is None
        assert reason == "max_concurrent"

    def test_leverage_clamped(self):
        # Very tight stop -> high leverage -> should clamp
        result, reason = self.sizer.compute(
            equity=1_000,
            entry_price=50_000,
            stop_distance=10,  # Very tight
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert result.leverage <= 10.0  # Max for BTC
        assert result.was_reduced

    def test_alt_leverage_lower(self):
        result, _ = self.sizer.compute(
            equity=1_000,
            entry_price=100,
            stop_distance=0.5,
            setup_grade=SetupGrade.A,
            symbol="DOGE",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert result.leverage <= 4.0  # Max for alts

    def test_zero_stop_distance_rejected(self):
        result, reason = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=0,
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is None
        assert reason == "zero_stop_distance"

    def test_sizing_result_fields(self):
        result, _ = self.sizer.compute(
            equity=10_000,
            entry_price=50_000,
            stop_distance=500,
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert isinstance(result, SizingResult)
        assert result.qty > 0
        assert result.leverage > 0
        assert result.notional > 0
        assert result.liquidation_price >= 0
