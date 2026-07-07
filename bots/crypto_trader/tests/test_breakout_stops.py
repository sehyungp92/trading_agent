"""Tests for breakout stop placement."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutStopParams
from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
from crypto_trader.strategy.breakout.stops import StopPlacer

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _zone(center=100.0, upper=105.0, lower=95.0):
    return BalanceZone(
        center=center,
        upper=upper,
        lower=lower,
        bars_in_zone=10,
        touches=3,
        formation_bar_idx=0,
        volume_contracting=False,
        width_atr=1.0,
    )


def _setup(direction=Side.LONG, zone=None):
    from crypto_trader.core.models import SetupGrade

    z = zone or _zone()
    return BreakoutSetupResult(
        grade=SetupGrade.A,
        is_a_plus=False,
        direction=direction,
        balance_zone=z,
        breakout_price=106.0,
        lvn_runway_atr=2.0,
        confluences=("h4_alignment",),
        room_r=2.5,
        volume_mult=1.5,
        body_ratio=0.7,
    )


def _bar(low=98.0, high=108.0, close=106.0, symbol="BTC"):
    return Bar(
        timestamp=TS,
        symbol=symbol,
        open=104.0,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.M30,
    )


class TestStopPlacer:
    def test_balance_edge_long(self):
        """Long stop at zone.lower."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=False,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        assert stop == 95.0

    def test_balance_edge_short(self):
        """Short stop at zone.upper."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=False,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        stop = sp.compute(
            setup=_setup(Side.SHORT),
            retest_bar=None,
            entry_price=94.0,
            atr=5.0,
            direction=Side.SHORT,
        )
        assert stop == 105.0

    def test_retest_extreme_long(self):
        """Model 2 long uses retest_bar.low."""
        cfg = BreakoutStopParams(
            use_balance_edge=False,
            use_retest_extreme=True,
            use_atr_stop=False,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        retest = _bar(low=97.0)
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=retest,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        assert stop == 97.0

    def test_atr_stop(self):
        """ATR-based stop computed correctly."""
        cfg = BreakoutStopParams(
            use_balance_edge=False,
            use_retest_extreme=False,
            use_atr_stop=True,
            atr_mult=1.5,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=100.0,
            atr=10.0,
            direction=Side.LONG,
        )
        # entry - atr_mult * atr = 100 - 1.5 * 10 = 85
        assert stop == pytest.approx(85.0)

    def test_use_farther_selects_most_generous(self):
        """use_farther=True selects the farthest stop for long (min)."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=True,
            atr_mult=2.0,
            use_farther=True,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        # balance edge long = 95.0; ATR stop = 106 - 2*5 = 96.0
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        assert stop == 95.0  # min(95, 96) = 95

    def test_use_tighter(self):
        """use_farther=False selects tightest stop."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=True,
            atr_mult=2.0,
            use_farther=False,
            buffer_pct=0.0,
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        # balance edge long = 95.0; ATR stop = 106 - 2*5 = 96.0
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        assert stop == 96.0  # max(95, 96) = 96

    def test_min_stop_distance(self):
        """Distance expanded to min_stop_atr * atr."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=False,
            buffer_pct=0.0,
            min_stop_atr=3.0,  # require 3 * atr = 15
        )
        sp = StopPlacer(cfg)
        # balance edge long = 95.0, dist = 106-95 = 11 < 15
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        # Should expand to entry - min_stop_atr*atr = 106 - 15 = 91
        assert stop == pytest.approx(91.0)

    def test_buffer_applied(self):
        """buffer_pct moves stop slightly beyond."""
        cfg = BreakoutStopParams(
            use_balance_edge=True,
            use_retest_extreme=False,
            use_atr_stop=False,
            buffer_pct=0.01,  # 1% buffer
            min_stop_atr=0.0,
        )
        sp = StopPlacer(cfg)
        # balance edge long = 95.0; buffer: 95 * (1 - 0.01) = 94.05
        stop = sp.compute(
            setup=_setup(Side.LONG),
            retest_bar=None,
            entry_price=106.0,
            atr=5.0,
            direction=Side.LONG,
        )
        assert stop == pytest.approx(95.0 * 0.99)
