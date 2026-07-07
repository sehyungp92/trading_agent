"""Tests for breakout exit management."""

from datetime import datetime, timezone

import pytest

from crypto_trader.core.models import Bar, OrderType, Side, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutExitParams
from crypto_trader.strategy.breakout.exits import BreakoutExitState, ExitManager

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


def _bar(close=110.0, high=112.0, low=108.0, symbol="BTC"):
    return Bar(
        timestamp=TS,
        symbol=symbol,
        open=109.0,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
        timeframe=TimeFrame.M30,
    )


class TestExitManager:
    def test_init_position(self):
        """Creates exit state with correct fields."""
        mgr = ExitManager(BreakoutExitParams())
        zone = _zone()
        mgr.init_position("BTC", 106.0, 11.0, 1.0, Side.LONG, zone)
        state = mgr.get_state("BTC")
        assert state is not None
        assert state.entry_price == 106.0
        assert state.stop_distance == 11.0
        assert state.original_qty == 1.0
        assert state.remaining_qty == 1.0
        assert state.direction == Side.LONG
        assert state.balance_upper == 105.0
        assert state.balance_lower == 95.0

    def test_r_multiple_updates(self):
        """peak_r/mfe_r update correctly on bar."""
        mgr = ExitManager(BreakoutExitParams())
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # Bar with high=115 -> peak_r = (115-100)/10 = 1.5
        bar = _bar(close=112.0, high=115.0, low=109.0)
        mgr.process_bar(bar, "BTC")
        state = mgr.get_state("BTC")
        assert state.peak_r == pytest.approx(1.5)
        assert state.mfe_r == pytest.approx(1.5)
        assert state.current_r == pytest.approx(1.2)  # (112-100)/10

    def test_tp1_triggered(self):
        """peak_r >= tp1_r triggers TP1 order."""
        cfg = BreakoutExitParams(tp1_r=1.0, tp1_frac=0.3, invalidation_exit=False)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # peak_r = (115-100)/10 = 1.5 >= 1.0
        bar = _bar(close=112.0, high=115.0, low=109.0)
        orders = mgr.process_bar(bar, "BTC")
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1
        state = mgr.get_state("BTC")
        assert state.tp1_hit is True

    def test_early_lock_flag_applies_before_tp1(self):
        """A pre-TP1 MFE threshold marks the trade for stop tightening."""
        cfg = BreakoutExitParams(
            early_lock_enabled=True,
            early_lock_mfe_r=0.35,
            tp1_r=5.0,
            invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        bar = _bar(close=101.0, high=104.0, low=99.5)  # peak_r = 0.4
        orders = mgr.process_bar(bar, "BTC")
        state = mgr.get_state("BTC")
        assert orders == []
        assert state is not None
        assert state.early_lock_applied is True
        assert state.tp1_hit is False

    def test_tp1_qty_fraction(self):
        """TP1 order qty = original_qty * tp1_frac."""
        cfg = BreakoutExitParams(tp1_r=1.0, tp1_frac=0.30, invalidation_exit=False)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 2.0, Side.LONG, _zone())
        bar = _bar(close=112.0, high=115.0, low=109.0)
        orders = mgr.process_bar(bar, "BTC")
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1
        assert tp1_orders[0].qty == pytest.approx(0.6)  # 2.0 * 0.3

    def test_tp2_after_tp1(self):
        """TP2 triggers only after TP1 hit."""
        cfg = BreakoutExitParams(
            tp1_r=1.0, tp1_frac=0.3, tp2_r=2.0, tp2_frac=0.5, invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # First bar: trigger TP1 (peak=1.5R)
        bar1 = _bar(close=112.0, high=115.0, low=109.0)
        mgr.process_bar(bar1, "BTC")
        state = mgr.get_state("BTC")
        assert state.tp1_hit is True
        assert state.tp2_hit is False
        # Second bar: trigger TP2 (peak=2.5R)
        bar2 = _bar(close=122.0, high=125.0, low=120.0)
        orders = mgr.process_bar(bar2, "BTC")
        tp2_orders = [o for o in orders if o.tag == "tp2"]
        assert len(tp2_orders) == 1
        state = mgr.get_state("BTC")
        assert state.tp2_hit is True

    def test_be_moved_after_tp1(self):
        """be_moved set after TP1 + min bars."""
        cfg = BreakoutExitParams(
            tp1_r=1.0, tp1_frac=0.3, be_after_tp1=True,
            be_min_bars_above=2, invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # Bar 1: trigger TP1
        bar1 = _bar(close=112.0, high=115.0, low=109.0)
        mgr.process_bar(bar1, "BTC")
        state = mgr.get_state("BTC")
        assert state.tp1_hit is True
        # bars_since_entry=1, be_min_bars_above=2 -> not yet
        assert state.be_moved is False
        # Bar 2: bars_since_entry=2 >= 2
        bar2 = _bar(close=113.0, high=115.0, low=111.0)
        mgr.process_bar(bar2, "BTC")
        state = mgr.get_state("BTC")
        assert state.be_moved is True

    def test_time_stop(self):
        """bars >= time_stop_bars + low progress triggers close."""
        cfg = BreakoutExitParams(
            tp1_r=5.0,  # Won't trigger
            time_stop_bars=3,
            time_stop_min_progress_r=1.0,
            time_stop_action="close",
            invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # Simulate 3 bars with low progress (close ~100)
        flat_bar = _bar(close=101.0, high=102.0, low=99.0)
        for _ in range(3):
            orders = mgr.process_bar(flat_bar, "BTC")
        time_orders = [o for o in orders if o.tag == "time_stop"]
        assert len(time_orders) == 1

    def test_invalidation_exit_long(self):
        """Price re-enters balance zone triggers close for long."""
        cfg = BreakoutExitParams(invalidation_exit=True, invalidation_min_bars=0, tp1_r=5.0)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 106.0, 11.0, 1.0, Side.LONG, _zone())
        # Close below zone.upper (105) = invalidation
        bar = _bar(close=104.0, high=106.0, low=103.0)
        orders = mgr.process_bar(bar, "BTC")
        inv_orders = [o for o in orders if o.tag == "invalidation"]
        assert len(inv_orders) == 1
        assert inv_orders[0].side == Side.SHORT

    def test_invalidation_exit_short(self):
        """Short position invalidation."""
        cfg = BreakoutExitParams(invalidation_exit=True, invalidation_min_bars=0, tp1_r=5.0)
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 94.0, 11.0, 1.0, Side.SHORT, _zone())
        # Close above zone.lower (95) = invalidation
        bar = _bar(close=96.0, high=97.0, low=93.0)
        orders = mgr.process_bar(bar, "BTC")
        inv_orders = [o for o in orders if o.tag == "invalidation"]
        assert len(inv_orders) == 1
        assert inv_orders[0].side == Side.LONG

    def test_quick_exit(self):
        """Stagnant trade exits after bars threshold."""
        cfg = BreakoutExitParams(
            quick_exit_enabled=True,
            quick_exit_bars=2,
            quick_exit_max_mfe_r=0.5,
            quick_exit_max_r=0.0,
            tp1_r=5.0,  # Won't trigger
            invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        flat_bar = _bar(close=100.0, high=101.0, low=99.0)
        for _ in range(2):
            orders = mgr.process_bar(flat_bar, "BTC")
        qe_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(qe_orders) == 1

    def test_remove_position(self):
        """remove clears state."""
        mgr = ExitManager(BreakoutExitParams())
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        assert mgr.get_state("BTC") is not None
        removed = mgr.remove("BTC")
        assert removed is not None
        assert mgr.get_state("BTC") is None

    def test_tp2_cascade_guard_skips_when_tp2_leq_tp1(self):
        """TP2 is skipped when tp2_r <= tp1_r (prevents same-bar cascade)."""
        cfg = BreakoutExitParams(
            tp1_r=1.0, tp1_frac=0.3,
            tp2_r=0.8, tp2_frac=0.5,  # tp2 < tp1 -> cascade
            invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # peak_r = (115-100)/10 = 1.5 >= tp1=1.0 AND >= tp2=0.8
        bar = _bar(close=112.0, high=115.0, low=109.0)
        orders = mgr.process_bar(bar, "BTC")

        tp1_orders = [o for o in orders if o.tag == "tp1"]
        tp2_orders = [o for o in orders if o.tag == "tp2"]
        assert len(tp1_orders) == 1  # TP1 fires
        assert len(tp2_orders) == 0  # TP2 guarded — not fired
        state = mgr.get_state("BTC")
        assert state.tp1_hit is True
        assert state.tp2_hit is False

    def test_tp2_fires_normally_when_above_tp1(self):
        """TP2 fires on separate bar when tp2_r > tp1_r."""
        cfg = BreakoutExitParams(
            tp1_r=1.0, tp1_frac=0.3,
            tp2_r=2.0, tp2_frac=0.5,
            invalidation_exit=False,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 100.0, 10.0, 1.0, Side.LONG, _zone())
        # Bar 1: trigger TP1 only (peak=1.5R < tp2=2.0R)
        bar1 = _bar(close=112.0, high=115.0, low=109.0)
        orders1 = mgr.process_bar(bar1, "BTC")
        assert any(o.tag == "tp1" for o in orders1)
        assert not any(o.tag == "tp2" for o in orders1)
        # Bar 2: trigger TP2 (peak=2.5R >= tp2=2.0R)
        bar2 = _bar(close=122.0, high=125.0, low=120.0)
        orders2 = mgr.process_bar(bar2, "BTC")
        tp2_orders = [o for o in orders2 if o.tag == "tp2"]
        assert len(tp2_orders) == 1
        state = mgr.get_state("BTC")
        assert state.tp2_hit is True

    def test_no_orders_no_position(self):
        """process_bar with unknown sym returns empty list."""
        mgr = ExitManager(BreakoutExitParams())
        bar = _bar()
        orders = mgr.process_bar(bar, "UNKNOWN")
        assert orders == []

    def test_invalidation_blocked_before_min_bars(self):
        """Invalidation does NOT fire when bars_since_entry < invalidation_min_bars."""
        cfg = BreakoutExitParams(
            invalidation_exit=True, invalidation_min_bars=3,
            invalidation_depth_atr=0.0, tp1_r=5.0,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 106.0, 11.0, 1.0, Side.LONG, _zone())
        # Bar 1 & 2: close below zone.upper (105) — would normally invalidate
        inv_bar = _bar(close=104.0, high=106.0, low=103.0)
        orders1 = mgr.process_bar(inv_bar, "BTC")
        orders2 = mgr.process_bar(inv_bar, "BTC")
        inv1 = [o for o in orders1 if o.tag == "invalidation"]
        inv2 = [o for o in orders2 if o.tag == "invalidation"]
        assert len(inv1) == 0  # bars_since_entry=1 < 3
        assert len(inv2) == 0  # bars_since_entry=2 < 3

    def test_invalidation_fires_after_min_bars(self):
        """Invalidation fires after bars_since_entry >= invalidation_min_bars."""
        cfg = BreakoutExitParams(
            invalidation_exit=True, invalidation_min_bars=3,
            invalidation_depth_atr=0.0, tp1_r=5.0,
        )
        mgr = ExitManager(cfg)
        mgr.init_position("BTC", 106.0, 11.0, 1.0, Side.LONG, _zone())
        inv_bar = _bar(close=104.0, high=106.0, low=103.0)
        # Bars 1-2: no invalidation
        mgr.process_bar(inv_bar, "BTC")
        mgr.process_bar(inv_bar, "BTC")
        # Bar 3: bars_since_entry=3 >= 3 — invalidation fires
        orders3 = mgr.process_bar(inv_bar, "BTC")
        inv3 = [o for o in orders3 if o.tag == "invalidation"]
        assert len(inv3) == 1
        assert inv3[0].side == Side.SHORT
