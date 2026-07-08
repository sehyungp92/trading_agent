"""Tests for breakout confirmation (Model 1 close + Model 2 retest)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Bar, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutConfirmParams
from crypto_trader.strategy.breakout.confirmation import ConfirmationDetector, BreakoutConfirmation
from crypto_trader.strategy.breakout.setup import BreakoutSetupResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)


def _make_zone(upper: float = 105.0, lower: float = 95.0) -> BalanceZone:
    return BalanceZone(
        center=(upper + lower) / 2.0,
        upper=upper,
        lower=lower,
        bars_in_zone=12,
        touches=4,
        formation_bar_idx=0,
        volume_contracting=False,
        width_atr=1.0,
    )


def make_setup(
    direction: Side = Side.LONG,
    zone_upper: float = 105.0,
    zone_lower: float = 95.0,
) -> BreakoutSetupResult:
    """Build a BreakoutSetupResult with sensible defaults."""
    return BreakoutSetupResult(
        grade=SetupGrade.A,
        is_a_plus=False,
        direction=direction,
        balance_zone=_make_zone(upper=zone_upper, lower=zone_lower),
        breakout_price=zone_upper + 3.0 if direction == Side.LONG else zone_lower - 3.0,
        lvn_runway_atr=3.0,
        confluences=("h4_alignment", "volume_surge"),
        room_r=2.5,
        volume_mult=1.5,
        body_ratio=0.65,
    )


def _bar(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
) -> Bar:
    """Quick bar constructor."""
    if open_ is None:
        open_ = close - 1.0
    if high is None:
        high = max(close, open_) + 2.0
    if low is None:
        low = min(close, open_) - 2.0
    return Bar(
        timestamp=_TS,
        symbol="BTC",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        timeframe=TimeFrame.M30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfirmationDetector:
    """ConfirmationDetector tests for Model 1 and Model 2."""

    # -- Model 1 (breakout close) -----------------------------------------

    def test_model1_returns_confirmation(self):
        """check_breakout_close returns BreakoutConfirmation when gates pass."""
        cfg = BreakoutConfirmParams(model1_require_volume=False, model1_require_direction_close=False)
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        bar = _bar(close=110.0)
        conf = det.check_breakout_close(bar, setup, m30_ind=None)
        assert conf is not None
        assert isinstance(conf, BreakoutConfirmation)

    def test_model1_model_name(self):
        """confirmation.model == 'model1_close'."""
        cfg = BreakoutConfirmParams(model1_require_volume=False, model1_require_direction_close=False)
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        bar = _bar(close=110.0)
        conf = det.check_breakout_close(bar, setup, m30_ind=None)
        assert conf.model == "model1_close"

    # -- register / pending state -----------------------------------------

    def test_register_breakout(self):
        """register_breakout makes has_pending True."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        setup = make_setup(Side.LONG)
        assert det.has_pending("BTC") is False
        det.register_breakout("BTC", setup, bar_idx=10)
        assert det.has_pending("BTC") is True

    def test_clear_pending(self):
        """clear_pending removes the pending state."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        det.register_breakout("BTC", make_setup(), bar_idx=10)
        assert det.has_pending("BTC") is True
        det.clear_pending("BTC")
        assert det.has_pending("BTC") is False

    def test_replace_pending(self):
        """A new register_breakout replaces the old pending."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        setup1 = make_setup(direction=Side.LONG)
        setup2 = make_setup(direction=Side.SHORT, zone_upper=110, zone_lower=100)
        det.register_breakout("BTC", setup1, bar_idx=5)
        det.register_breakout("BTC", setup2, bar_idx=10)
        pending = det.get_pending_setup("BTC")
        assert pending is not None
        assert pending.direction == Side.SHORT

    def test_get_pending_setup(self):
        """get_pending_setup returns the stored setup."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        setup = make_setup(Side.LONG)
        det.register_breakout("BTC", setup, bar_idx=10)
        pending = det.get_pending_setup("BTC")
        assert pending is setup

    def test_get_pending_setup_none(self):
        """get_pending_setup returns None when nothing is pending."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        assert det.get_pending_setup("BTC") is None

    # -- Model 2 (retest) ------------------------------------------------

    def test_retest_long_near_upper(self):
        """LONG: price pulls back to zone.upper -> confirmed with bullish close."""
        cfg = BreakoutConfirmParams(retest_max_bars=6, retest_zone_atr=0.5)
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        # Bar pulls back: low near zone.upper (105), bullish close
        bar = _bar(close=106.0, open_=104.5, high=107.0, low=104.5)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is not None
        assert conf.model == "model2_retest"

    def test_retest_short_near_lower(self):
        """SHORT: price pulls back to zone.lower -> confirmed with bearish close."""
        cfg = BreakoutConfirmParams(retest_max_bars=6, retest_zone_atr=0.5)
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.SHORT, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        # Bar pulls back: high near zone.lower (95), bearish close
        bar = _bar(close=93.0, open_=95.5, high=95.5, low=92.0)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is not None
        assert conf.model == "model2_retest"

    def test_retest_expired(self):
        """bars_since > retest_max_bars -> returns None and clears pending."""
        cfg = BreakoutConfirmParams(retest_max_bars=3)
        det = ConfirmationDetector(cfg)
        det.register_breakout("BTC", make_setup(), bar_idx=10)

        bar = _bar(close=106.0, open_=104.5)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=20)
        assert conf is None
        assert det.has_pending("BTC") is False

    def test_retest_no_pending(self):
        """No pending -> check_retest returns None."""
        det = ConfirmationDetector(BreakoutConfirmParams())
        bar = _bar(close=106.0)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=15)
        assert conf is None

    def test_retest_requires_closing_direction(self):
        """LONG retest requires bullish close (close > open)."""
        cfg = BreakoutConfirmParams(retest_max_bars=6, retest_zone_atr=0.5)
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        # Bearish bar pulling back near upper -- close < open => not bullish
        bar = _bar(close=104.5, open_=106.0, high=107.0, low=104.0)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is None

    def test_retest_deletes_pending(self):
        """After a successful retest confirmation, pending is removed."""
        cfg = BreakoutConfirmParams(retest_max_bars=6, retest_zone_atr=0.5)
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        bar = _bar(close=106.0, open_=104.5, high=107.0, low=104.5)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is not None
        assert det.has_pending("BTC") is False

    def test_retest_rejection_gate_blocks_close_back_inside_value(self):
        """When enabled, retest confirmation requires the close to reject back outside the edge."""
        cfg = BreakoutConfirmParams(
            retest_max_bars=6,
            retest_zone_atr=0.5,
            retest_require_rejection=True,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        # Bullish close but still back inside the broken edge.
        bar = _bar(close=104.8, open_=104.0, high=106.2, low=104.4)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is None
        assert det.has_pending("BTC") is True

    def test_retest_rejection_gate_allows_close_back_outside_edge(self):
        """When enabled, retest confirmation passes if price rejects and closes back outside the edge."""
        cfg = BreakoutConfirmParams(
            retest_max_bars=6,
            retest_zone_atr=0.5,
            retest_require_rejection=True,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        bar = _bar(close=105.6, open_=104.8, high=106.5, low=104.6)
        conf = det.check_retest("BTC", bar, bars=[bar], atr=10.0, bar_index=12)
        assert conf is not None
        assert conf.model == "model2_retest"

    def test_retest_volume_decline_gate_blocks_heavier_retest_volume(self):
        """When enabled, retest confirmation rejects bars that do not show the required volume decline."""
        cfg = BreakoutConfirmParams(
            retest_max_bars=6,
            retest_zone_atr=0.5,
            retest_require_volume_decline=True,
            volume_decline_threshold=0.8,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        prev_bar = _bar(close=107.0, open_=106.0, high=108.0, low=105.5, volume=100.0)
        retest_bar = _bar(close=106.0, open_=104.5, high=107.0, low=104.5, volume=90.0)
        conf = det.check_retest(
            "BTC",
            retest_bar,
            bars=[prev_bar, retest_bar],
            atr=10.0,
            bar_index=12,
        )
        assert conf is None
        assert det.has_pending("BTC") is True

    def test_retest_volume_decline_gate_allows_lighter_retest_volume(self):
        """When enabled, retest confirmation passes when retest volume declines enough."""
        cfg = BreakoutConfirmParams(
            retest_max_bars=6,
            retest_zone_atr=0.5,
            retest_require_volume_decline=True,
            volume_decline_threshold=0.8,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(direction=Side.LONG, zone_upper=105, zone_lower=95)
        det.register_breakout("BTC", setup, bar_idx=10)

        prev_bar = _bar(close=107.0, open_=106.0, high=108.0, low=105.5, volume=100.0)
        retest_bar = _bar(close=106.0, open_=104.5, high=107.0, low=104.5, volume=70.0)
        conf = det.check_retest(
            "BTC",
            retest_bar,
            bars=[prev_bar, retest_bar],
            atr=10.0,
            bar_index=12,
        )
        assert conf is not None
        assert conf.model == "model2_retest"

    # -- Model 1 quality gates -----------------------------------------------

    def test_model1_rejects_low_volume(self):
        """Model 1 rejects bar when volume < min_volume_mult * volume_ma."""
        from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
        import numpy as np
        cfg = BreakoutConfirmParams(
            model1_require_volume=True, model1_min_volume_mult=1.3,
            model1_require_direction_close=False,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        # volume=100, volume_ma=100 => mult=1.0 < 1.3
        bar = _bar(close=110.0, open_=108.0, volume=100.0)
        empty = np.array([0.0])
        ind = IndicatorSnapshot(
            ema_fast=0, ema_mid=0, ema_slow=0,
            ema_fast_arr=empty, ema_mid_arr=empty, ema_slow_arr=empty,
            adx=0, di_plus=0, di_minus=0, adx_rising=False,
            atr=1.0, atr_avg=1.0, rsi=50.0, volume_ma=100.0,
        )
        conf = det.check_breakout_close(bar, setup, m30_ind=ind)
        assert conf is None

    def test_model1_accepts_sufficient_volume(self):
        """Model 1 accepts bar when volume >= min_volume_mult * volume_ma."""
        from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
        import numpy as np
        cfg = BreakoutConfirmParams(
            model1_require_volume=True, model1_min_volume_mult=1.3,
            model1_require_direction_close=False,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        # volume=150, volume_ma=100 => mult=1.5 >= 1.3
        bar = _bar(close=110.0, open_=108.0, volume=150.0)
        empty = np.array([0.0])
        ind = IndicatorSnapshot(
            ema_fast=0, ema_mid=0, ema_slow=0,
            ema_fast_arr=empty, ema_mid_arr=empty, ema_slow_arr=empty,
            adx=0, di_plus=0, di_minus=0, adx_rising=False,
            atr=1.0, atr_avg=1.0, rsi=50.0, volume_ma=100.0,
        )
        conf = det.check_breakout_close(bar, setup, m30_ind=ind)
        assert conf is not None
        assert conf.model == "model1_close"

    def test_model1_rejects_wrong_direction_close(self):
        """Long breakout with bearish close (close < open) returns None."""
        cfg = BreakoutConfirmParams(
            model1_require_volume=False,
            model1_require_direction_close=True,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        # Bearish bar: close=108 < open=110
        bar = _bar(close=108.0, open_=110.0, volume=100.0)
        conf = det.check_breakout_close(bar, setup, m30_ind=None)
        assert conf is None

    def test_model1_accepts_correct_direction_close(self):
        """Long breakout with bullish close (close > open) returns confirmation."""
        cfg = BreakoutConfirmParams(
            model1_require_volume=False,
            model1_require_direction_close=True,
        )
        det = ConfirmationDetector(cfg)
        setup = make_setup(Side.LONG)
        # Bullish bar: close=112 > open=108
        bar = _bar(close=112.0, open_=108.0, volume=100.0)
        conf = det.check_breakout_close(bar, setup, m30_ind=None)
        assert conf is not None
        assert conf.model == "model1_close"
