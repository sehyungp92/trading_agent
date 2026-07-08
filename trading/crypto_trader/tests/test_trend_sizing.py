"""Tests for trend position sizing."""

import pytest

from crypto_trader.core.models import Position, SetupGrade, Side
from crypto_trader.strategy.trend.config import TrendLimitParams, TrendRiskParams
from crypto_trader.strategy.trend.sizing import PositionSizer, SizingResult


class TestPositionSizer:
    def test_basic_sizing_b_grade(self):
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams())
        result, reason = sizer.compute(
            equity=10000,
            entry_price=50000,
            stop_distance=500,
            grade=SetupGrade.B,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        assert result.qty > 0
        assert result.risk_pct_actual == pytest.approx(0.01, abs=0.001)  # Baked from round 1

    def test_a_grade_larger_than_b(self):
        """A-grade sizes larger than B when risk_pct_a > risk_pct_b."""
        sizer = PositionSizer(
            TrendRiskParams(risk_pct_a=0.02, risk_pct_b=0.01, max_risk_pct=0.03),
            TrendLimitParams(),
        )
        result_a, _ = sizer.compute(10000, 50000, 500, SetupGrade.A, "BTC", [], Side.LONG)
        result_b, _ = sizer.compute(10000, 50000, 500, SetupGrade.B, "BTC", [], Side.LONG)
        assert result_a is not None and result_b is not None
        assert result_a.qty > result_b.qty

    def test_leverage_clamped_major(self):
        sizer = PositionSizer(
            TrendRiskParams(risk_pct_a=0.05, max_leverage_major=3.0),
            TrendLimitParams()
        )
        result, _ = sizer.compute(10000, 50000, 50, SetupGrade.A, "BTC", [], Side.LONG)
        assert result is not None
        assert result.leverage <= 3.0
        assert result.was_reduced

    def test_leverage_clamped_alt(self):
        sizer = PositionSizer(
            TrendRiskParams(max_leverage_alt=3.0),
            TrendLimitParams()
        )
        result, _ = sizer.compute(10000, 50000, 100, SetupGrade.B, "SOL", [], Side.LONG)
        if result is not None:
            assert result.leverage <= 3.0

    def test_position_limit_reached(self):
        """Should return None when max_concurrent_positions reached."""
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams(max_concurrent_positions=2))
        pos1 = Position("BTC", Side.LONG, 0.1, 50000)
        pos2 = Position("ETH", Side.LONG, 1.0, 3000)
        result, reason = sizer.compute(10000, 50000, 500, SetupGrade.B, "SOL", [pos1, pos2], Side.LONG)
        assert result is None
        assert reason == "max_concurrent"

    def test_zero_equity_rejected(self):
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams())
        result, reason = sizer.compute(0, 50000, 500, SetupGrade.B, "BTC", [], Side.LONG)
        assert result is None
        assert reason == "invalid_inputs"

    def test_zero_stop_distance_rejected(self):
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams())
        result, reason = sizer.compute(10000, 50000, 0, SetupGrade.B, "BTC", [], Side.LONG)
        assert result is None
        assert reason == "invalid_inputs"

    def test_sizing_result_fields(self):
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams())
        result, _ = sizer.compute(10000, 50000, 500, SetupGrade.B, "BTC", [], Side.LONG)
        assert result is not None
        assert isinstance(result, SizingResult)
        assert result.notional == result.qty * 50000
        assert result.liquidation_price < 50000  # For long

    def test_short_liquidation_above_entry(self):
        sizer = PositionSizer(TrendRiskParams(), TrendLimitParams())
        result, _ = sizer.compute(10000, 50000, 500, SetupGrade.B, "BTC", [], Side.SHORT)
        assert result is not None
        assert result.liquidation_price > 50000

    def test_risk_scale_reduces_size(self):
        sizer = PositionSizer(TrendRiskParams(risk_pct_b=0.02), TrendLimitParams())
        full, _ = sizer.compute(10000, 50000, 500, SetupGrade.B, "BTC", [], Side.LONG, risk_scale=1.0)
        half, _ = sizer.compute(10000, 50000, 500, SetupGrade.B, "BTC", [], Side.LONG, risk_scale=0.5)
        assert full is not None and half is not None
        assert half.qty == pytest.approx(full.qty * 0.5)
        assert half.risk_pct_actual == pytest.approx(full.risk_pct_actual * 0.5)
