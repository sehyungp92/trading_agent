"""Tests for trend exit management."""

import pytest
from datetime import datetime, timezone

from crypto_trader.core.models import Bar, Position, Side, TimeFrame
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.trend.config import TrendExitParams
from crypto_trader.strategy.trend.exits import ExitManager, TrendExitState


def _make_bar(close, high=None, low=None, idx=0):
    return Bar(
        timestamp=datetime(2026, 3, 15, idx, 0, tzinfo=timezone.utc),
        symbol="BTC",
        open=close,
        high=high or close + 50,
        low=low or close - 50,
        close=close,
        volume=100.0,
        timeframe=TimeFrame.H1,
    )


def _make_ind(ema_fast=49000):
    return IndicatorSnapshot(
        ema_fast=ema_fast,
        ema_fast_arr=None,
        ema_mid=48000,
        ema_mid_arr=None,
        ema_slow=0,
        ema_slow_arr=None,
        atr=200.0,
        atr_avg=200.0,
        rsi=50.0,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        adx_rising=False,
        volume_ma=100.0,
    )


class TestExitManager:
    def _setup_manager(self, **kwargs):
        cfg = TrendExitParams(**kwargs)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 50000, 500, 1.0, Side.LONG)
        return mgr

    def test_tp1_triggered(self):
        """TP1 should trigger when peak_r >= tp1_r."""
        mgr = self._setup_manager(tp1_r=1.0, tp1_frac=0.25)
        pos = Position("BTC", Side.LONG, 1.0, 50000)
        # Bar with high at 50600 → peak_r = 600/500 = 1.2
        bar = _make_bar(50400, high=50600, low=50300)
        orders = mgr.manage(pos, bar, [bar], _make_ind(), None)
        assert len(orders) >= 1
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1
        assert tp1_orders[0].qty == 0.25  # tp1_frac * original_qty

    def test_tp1_not_retriggered(self):
        """TP1 should only trigger once."""
        mgr = self._setup_manager(tp1_r=1.0, tp1_frac=0.25)
        pos = Position("BTC", Side.LONG, 1.0, 50000)
        bar = _make_bar(50600, high=50700)
        mgr.manage(pos, bar, [bar], _make_ind(), None)  # TP1 triggers

        pos2 = Position("BTC", Side.LONG, 0.75, 50000)
        orders2 = mgr.manage(pos2, bar, [bar], _make_ind(), None)
        tp1_orders = [o for o in orders2 if o.tag == "tp1"]
        assert len(tp1_orders) == 0  # Not retriggered

    def test_tp2_after_tp1(self):
        """TP2 should only trigger after TP1."""
        mgr = self._setup_manager(tp1_r=1.0, tp1_frac=0.25, tp2_r=2.0, tp2_frac=0.50)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # First bar triggers TP1
        bar1 = _make_bar(50600, high=50600)
        mgr.manage(pos, bar1, [bar1], _make_ind(), None)

        # Second bar triggers TP2
        pos2 = Position("BTC", Side.LONG, 0.75, 50000)
        bar2 = _make_bar(51000, high=51100)  # peak_r = 1100/500 = 2.2
        orders2 = mgr.manage(pos2, bar2, [bar2], _make_ind(), None)
        tp2_orders = [o for o in orders2 if o.tag == "tp2"]
        assert len(tp2_orders) == 1

    def test_intra_bar_peak_r(self):
        """Peak R should use bar.high for long, not close."""
        mgr = self._setup_manager(tp1_r=1.0, tp1_frac=0.25)
        pos = Position("BTC", Side.LONG, 1.0, 50000)
        # Close at 50300 (0.6R) but high at 50600 (1.2R)
        bar = _make_bar(50300, high=50600, low=50200)
        orders = mgr.manage(pos, bar, [bar], _make_ind(), None)
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1  # Triggered by peak, not close

    def test_smart_be_after_sustained_1r(self):
        """BE should move after bars_above_1r >= be_min_bars_above."""
        mgr = self._setup_manager(tp1_r=1.0, tp1_frac=0.25, be_min_bars_above=2)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # Trigger TP1 first
        bar1 = _make_bar(50600, high=50600)
        mgr.manage(pos, bar1, [bar1], _make_ind(), None)

        pos2 = Position("BTC", Side.LONG, 0.75, 50000)
        # Two bars above 1R
        bar2 = _make_bar(50600, high=50700)
        mgr.manage(pos2, bar2, [bar2], _make_ind(), None)
        bar3 = _make_bar(50600, high=50700)
        mgr.manage(pos2, bar3, [bar3], _make_ind(), None)

        state = mgr.get_state("BTC")
        assert state.be_moved is True

    def test_be_price_calculation(self):
        """BE price should be entry + buffer_r * stop_distance."""
        mgr = self._setup_manager(be_buffer_r=0.2)
        mgr._states["BTC"].be_moved = True
        be = mgr.get_be_price("BTC")
        assert be == pytest.approx(50000 + 0.2 * 500)

    def test_time_stop_triggered(self):
        """Time stop fires when bars exceed threshold with low progress."""
        mgr = self._setup_manager(time_stop_bars=3, time_stop_min_progress_r=0.5)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # 4 bars with no progress
        for i in range(4):
            bar = _make_bar(50050, high=50100, low=49950)
            orders = mgr.manage(pos, bar, [bar], _make_ind(), None)

        # Should have time stop order in one of the iterations
        assert any(o.tag == "time_stop" for o in orders)

    def test_ema_failsafe_exit(self):
        """EMA fail-safe exits runner when price drops below EMA after expansion."""
        mgr = self._setup_manager(
            tp1_r=1.0, tp1_frac=0.25,
            ema_failsafe_enabled=True,
            ema_failsafe_min_expansion_r=1.0,
        )
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # Trigger TP1 and build MFE
        bar1 = _make_bar(50600, high=50700)
        mgr.manage(pos, bar1, [bar1], _make_ind(), None)

        # Now price drops below EMA
        pos2 = Position("BTC", Side.LONG, 0.75, 50000)
        bar2 = _make_bar(48800, high=49000, low=48700)  # Below EMA 49000
        orders = mgr.manage(pos2, bar2, [bar2], _make_ind(ema_fast=49000), None)
        ema_orders = [o for o in orders if o.tag == "ema_failsafe"]
        assert len(ema_orders) == 1

    def test_short_direction_r_calculation(self):
        """R-multiples for short positions calculated correctly."""
        cfg = TrendExitParams(tp1_r=1.0, tp1_frac=0.25)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 50000, 500, 1.0, Side.SHORT)
        pos = Position("BTC", Side.SHORT, 1.0, 50000)
        # For short: peak_r uses bar.low, current_r uses (entry - close) / stop
        bar = _make_bar(49400, high=49500, low=49300)  # peak_r = (50000-49300)/500 = 1.4
        orders = mgr.manage(pos, bar, [bar], _make_ind(), None)
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1

    def test_returns_list_of_orders(self):
        """manage() always returns list[Order]."""
        mgr = self._setup_manager()
        pos = Position("BTC", Side.LONG, 1.0, 50000)
        bar = _make_bar(50100)
        orders = mgr.manage(pos, bar, [bar], _make_ind(), None)
        assert isinstance(orders, list)

    def test_remove_position(self):
        """remove_position clears state."""
        mgr = self._setup_manager()
        state = mgr.remove_position("BTC")
        assert state is not None
        assert mgr.get_state("BTC") is None

    def test_remaining_qty_tracking(self):
        """State tracks remaining_qty from position."""
        mgr = self._setup_manager()
        pos = Position("BTC", Side.LONG, 0.8, 50000)
        bar = _make_bar(50100)
        mgr.manage(pos, bar, [bar], _make_ind(), None)
        state = mgr.get_state("BTC")
        assert state.remaining_qty == 0.8

    def test_mfe_mae_tracking(self):
        """MFE and MAE track peak/trough R-multiples."""
        mgr = self._setup_manager()
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # Bar with high R
        bar1 = _make_bar(50300, high=50400, low=50100)
        mgr.manage(pos, bar1, [bar1], _make_ind(), None)
        state = mgr.get_state("BTC")
        assert state.mfe_r > 0
        assert state.mae_r <= state.current_r


class TestQuickExit:
    """Tests for quick exit of stagnant trades."""

    def _setup_manager(self, **kwargs):
        cfg = TrendExitParams(**kwargs)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 50000, 500, 1.0, Side.LONG)
        return mgr

    def test_quick_exit_triggers(self):
        """Quick exit fires when bars_held >= threshold, mfe < max, r <= max."""
        mgr = self._setup_manager(
            quick_exit_enabled=True, quick_exit_bars=4,
            quick_exit_max_mfe_r=0.2, quick_exit_max_r=-0.2,
            time_stop_bars=999,  # Prevent time stop interference
        )
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # Simulate stagnant bars: close near entry, low MFE, negative R
        orders = []
        for i in range(5):
            bar = _make_bar(49850, high=49900, low=49800, idx=i)  # current_r = -0.3, peak_r ~= -0.2
            orders = mgr.manage(pos, bar, [bar], _make_ind(), None)

        qe_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(qe_orders) == 1
        assert qe_orders[0].qty == 1.0  # Full position exit

    def test_quick_exit_not_enough_bars(self):
        """Quick exit should not fire before threshold bars reached."""
        mgr = self._setup_manager(
            quick_exit_enabled=True, quick_exit_bars=8,
            quick_exit_max_mfe_r=0.2, quick_exit_max_r=-0.2,
        )
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # Only 3 bars — below threshold
        for i in range(3):
            bar = _make_bar(49850, high=49900, low=49800, idx=i)
            orders = mgr.manage(pos, bar, [bar], _make_ind(), None)

        qe_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(qe_orders) == 0

    def test_quick_exit_blocked_by_high_mfe(self):
        """Quick exit should not fire if MFE exceeds threshold."""
        mgr = self._setup_manager(
            quick_exit_enabled=True, quick_exit_bars=3,
            quick_exit_max_mfe_r=0.2, quick_exit_max_r=-0.2,
        )
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        # First bar: high MFE (trade was good, then came back)
        bar1 = _make_bar(50200, high=50300, low=50100, idx=0)  # peak_r=0.6
        mgr.manage(pos, bar1, [bar1], _make_ind(), None)

        # Subsequent bars: negative R
        for i in range(4):
            bar = _make_bar(49850, high=49900, low=49800, idx=1 + i)
            orders = mgr.manage(pos, bar, [bar], _make_ind(), None)

        # MFE is 0.6 > 0.2 threshold → quick exit should NOT fire
        qe_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(qe_orders) == 0

    def test_quick_exit_disabled(self):
        """Quick exit should not fire when disabled (default)."""
        mgr = self._setup_manager(quick_exit_enabled=False)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        for i in range(10):
            bar = _make_bar(49850, high=49900, low=49800, idx=i)
            orders = mgr.manage(pos, bar, [bar], _make_ind(), None)

        qe_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(qe_orders) == 0


class TestScratchExit:
    def test_scratch_exit_triggers_after_failed_follow_through(self):
        mgr = ExitManager(TrendExitParams(
            scratch_exit_enabled=True,
            scratch_peak_r=0.25,
            scratch_floor_r=0.0,
            scratch_min_bars=3,
            time_stop_bars=50,
            quick_exit_enabled=False,
        ))
        mgr.init_position("BTC", 50000, 500, 1.0, Side.LONG)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        mgr.manage(pos, _make_bar(50080, high=50150, low=50020, idx=0), [], _make_ind(), None)
        mgr.manage(pos, _make_bar(50040, high=50090, low=50000, idx=1), [], _make_ind(), None)
        orders = mgr.manage(pos, _make_bar(49990, high=50040, low=49970, idx=2), [], _make_ind(), None)

        scratch_orders = [o for o in orders if o.tag == "scratch_exit"]
        assert len(scratch_orders) == 1
        assert scratch_orders[0].qty == 1.0


class TestMfeLockExit:
    def test_mfe_lock_exits_after_peak_giveback(self):
        mgr = ExitManager(TrendExitParams(
            mfe_lock_exit_enabled=True,
            mfe_lock_trigger_r=1.0,
            mfe_lock_floor_r=0.2,
            mfe_lock_min_bars=2,
            time_stop_bars=50,
            quick_exit_enabled=False,
        ))
        mgr.init_position("BTC", 50000, 500, 1.0, Side.LONG)
        pos = Position("BTC", Side.LONG, 1.0, 50000)

        mgr.manage(pos, _make_bar(50550, high=50650, low=50450, idx=0), [], _make_ind(), None)
        orders = mgr.manage(pos, _make_bar(50050, high=50100, low=50000, idx=1), [], _make_ind(), None)

        lock_orders = [o for o in orders if o.tag == "mfe_lock_exit"]
        assert len(lock_orders) == 1
        assert lock_orders[0].qty == 1.0
