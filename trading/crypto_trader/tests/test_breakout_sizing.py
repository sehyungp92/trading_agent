"""Tests for breakout position sizing."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Position, SetupGrade, Side
from crypto_trader.strategy.breakout.config import BreakoutLimitParams, BreakoutRiskParams
from crypto_trader.strategy.breakout.sizing import PositionSizer, SizingResult

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sizer(risk_cfg=None, limit_cfg=None):
    r = risk_cfg or BreakoutRiskParams()
    l = limit_cfg or BreakoutLimitParams()
    return PositionSizer(r, l)


def _position(symbol="BTC"):
    return Position(
        symbol=symbol,
        direction=Side.LONG,
        qty=0.1,
        avg_entry=100.0,
    )


class TestPositionSizer:
    def test_a_plus_risk_tier(self):
        """is_a_plus uses risk_pct_a_plus (0.0225)."""
        sizer = _sizer()
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=100.0,
            stop_distance=5.0,
            grade=SetupGrade.A,
            is_a_plus=True,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        # risk_pct_a_plus = 0.0225, risk_dollars = 225, qty = 225/5 = 45
        assert result.risk_pct_actual == pytest.approx(0.0225, rel=0.01)

    def test_a_grade_risk_tier(self):
        """grade=A uses risk_pct_a (0.01875)."""
        sizer = _sizer()
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=100.0,
            stop_distance=5.0,
            grade=SetupGrade.A,
            is_a_plus=False,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        # risk_pct_a = 0.01875, risk_dollars = 187.5, qty = 187.5/5 = 37.5
        assert result.risk_pct_actual == pytest.approx(0.01875, rel=0.01)

    def test_b_grade_risk_tier(self):
        """grade=B uses risk_pct_b (0.015 from risk sweep)."""
        sizer = _sizer()
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=1000.0,
            stop_distance=5.0,
            grade=SetupGrade.B,
            is_a_plus=False,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        assert result.risk_pct_actual == pytest.approx(0.015, rel=0.01)

    def test_risk_scale_reduces_position_risk(self):
        """Supplemental setups can size below the base tier via risk_scale."""
        sizer = _sizer()
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=1000.0,
            stop_distance=5.0,
            grade=SetupGrade.B,
            is_a_plus=False,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
            risk_scale=0.5,
        )
        assert result is not None
        assert reason == ""
        assert result.risk_pct_actual == pytest.approx(0.0075, rel=0.01)

    def test_leverage_clamped(self):
        """High notional gets leverage clamped."""
        cfg = BreakoutRiskParams(
            risk_pct_a=0.05,  # Large risk -> high notional
            max_risk_pct=0.05,
            max_leverage_major=3.0,
        )
        sizer = _sizer(risk_cfg=cfg)
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=100.0,
            stop_distance=1.0,  # Tiny stop = huge qty
            grade=SetupGrade.A,
            is_a_plus=False,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is not None
        assert reason == ""
        assert result.was_reduced is True
        assert result.leverage == pytest.approx(3.0)

    def test_max_concurrent_reject(self):
        """Returns None when max_concurrent reached."""
        limit_cfg = BreakoutLimitParams(max_concurrent_positions=1)
        sizer = _sizer(limit_cfg=limit_cfg)
        result, reason = sizer.compute(
            equity=10000.0,
            entry_price=100.0,
            stop_distance=5.0,
            grade=SetupGrade.A,
            is_a_plus=False,
            symbol="ETH",
            open_positions=[_position("BTC")],
            direction=Side.LONG,
        )
        assert result is None
        assert reason == "max_concurrent"

    def test_zero_equity_returns_none(self):
        """equity <= 0 returns None."""
        sizer = _sizer()
        result, reason = sizer.compute(
            equity=0.0,
            entry_price=100.0,
            stop_distance=5.0,
            grade=SetupGrade.A,
            is_a_plus=False,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )
        assert result is None
        assert reason == "invalid_inputs"
