"""Tests for exit efficiency fixes: intra-bar TP detection, MFE-floor trail, quick exit."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np

from crypto_trader.core.models import Bar, Position, Side, TimeFrame
from crypto_trader.strategy.momentum.config import (
    ExitParams,
    MomentumConfig,
    TrailParams,
)
from crypto_trader.strategy.momentum.exits import ExitManager, PositionExitState
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.momentum.trail import TrailManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indicators(atr: float = 2.0, volume_ma: float = 100.0) -> IndicatorSnapshot:
    dummy_arr = np.array([100.0])
    return IndicatorSnapshot(
        ema_fast=100.0, ema_mid=99.0, ema_slow=98.0,
        ema_fast_arr=dummy_arr, ema_mid_arr=dummy_arr, ema_slow_arr=dummy_arr,
        adx=25.0, di_plus=20.0, di_minus=15.0, adx_rising=True,
        atr=atr, atr_avg=atr, rsi=55.0, volume_ma=volume_ma,
    )


def _make_bar(
    close: float, high: float, low: float,
    open_: float | None = None, volume: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="BTC", open=open_ or close, high=high, low=low,
        close=close, volume=volume, timeframe=TimeFrame.M15,
    )


def _make_position(direction: Side = Side.LONG) -> Position:
    return Position(
        symbol="BTC", direction=direction, qty=1.0,
        avg_entry=100.0, unrealized_pnl=0.0,
    )


def _make_trail_bars(n: int = 25, base: float = 100.0) -> list[Bar]:
    bars: list[Bar] = []
    for i in range(n):
        ts = datetime(2026, 1, 1, i // 4, (i % 4) * 15, tzinfo=timezone.utc)
        low_adj = -2.0 if i % 3 == 1 else 0.0
        o = base + i * 0.5
        c = base + i * 0.5 + 0.3
        h = c + 1.0
        lo = o + low_adj - 0.5
        bars.append(Bar(
            timestamp=ts, symbol="BTC", open=o, high=h, low=lo,
            close=c, volume=100.0, timeframe=TimeFrame.M15,
        ))
    return bars


# ===========================================================================
# Fix 1: Intra-Bar TP Detection & MFE Tracking
# ===========================================================================


class TestIntraBarMFETracking:
    """MFE should use bar.high (long) / bar.low (short), not bar.close."""

    def test_mfe_uses_bar_high_for_long(self):
        params = ExitParams(tp1_r=99.0)  # TP unreachable
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # close=102 (0.4R), high=108 (1.6R)
        bar = _make_bar(close=102.0, high=108.0, low=99.0)
        mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state is not None
        # MFE should use high: (108-100)/5 = 1.6
        assert state.mfe_r == (108.0 - 100.0) / 5.0

    def test_mfe_uses_bar_low_for_short(self):
        params = ExitParams(tp1_r=99.0)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.SHORT)
        # close=98 (0.4R), low=92 (1.6R)
        bar = _make_bar(close=98.0, high=101.0, low=92.0)
        mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state is not None
        # MFE should use low: (100-92)/5 = 1.6
        assert state.mfe_r == (100.0 - 92.0) / 5.0

    def test_peak_r_field_set_correctly_long(self):
        params = ExitParams(tp1_r=99.0)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        bar = _make_bar(close=102.0, high=107.0, low=99.0)
        mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        # peak_r = (107-100)/5 = 1.4
        assert state.peak_r == (107.0 - 100.0) / 5.0

    def test_peak_r_field_set_correctly_short(self):
        params = ExitParams(tp1_r=99.0)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.SHORT)
        bar = _make_bar(close=98.0, high=101.0, low=93.0)
        mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        # peak_r = (100-93)/5 = 1.4
        assert state.peak_r == (100.0 - 93.0) / 5.0

    def test_mfe_accumulates_across_bars(self):
        params = ExitParams(tp1_r=99.0)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # Bar 1: high=106 → peak_r=1.2
        bar1 = _make_bar(close=103.0, high=106.0, low=99.0)
        mgr.manage(pos, bar1, [], None, MagicMock())
        assert mgr.get_state("BTC").mfe_r == 1.2

        # Bar 2: high=104 → peak_r=0.8. MFE stays 1.2 (max)
        bar2 = _make_bar(close=102.0, high=104.0, low=100.0)
        mgr.manage(pos, bar2, [], None, MagicMock())
        assert mgr.get_state("BTC").mfe_r == 1.2

        # Bar 3: high=110 → peak_r=2.0. MFE updates to 2.0
        bar3 = _make_bar(close=105.0, high=110.0, low=102.0)
        mgr.manage(pos, bar3, [], None, MagicMock())
        assert mgr.get_state("BTC").mfe_r == 2.0


class TestIntraBarTPDetection:
    """TP1/TP2 should trigger on intra-bar peak, not close."""

    def test_tp1_triggers_on_bar_high_long(self):
        """Bar close below TP1 but high above TP1 should trigger TP1."""
        params = ExitParams(tp1_r=1.2, tp1_frac=0.3)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # close=104 (0.8R), high=107 (1.4R > tp1_r=1.2)
        bar = _make_bar(close=104.0, high=107.0, low=99.0)
        orders = mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state.tp1_hit is True
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1

    def test_tp1_does_not_trigger_when_peak_below_threshold(self):
        """Neither close nor high above TP1 → no trigger."""
        params = ExitParams(tp1_r=1.2, tp1_frac=0.3)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # close=104 (0.8R), high=105 (1.0R < 1.2)
        bar = _make_bar(close=104.0, high=105.0, low=99.0)
        orders = mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state.tp1_hit is False
        assert all(o.tag != "tp1" for o in orders)

    def test_tp1_triggers_on_bar_low_short(self):
        """Short: bar close above TP1 but low below TP1 should trigger TP1."""
        params = ExitParams(tp1_r=1.2, tp1_frac=0.3)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.SHORT)
        # close=96 (0.8R), low=93 (1.4R > tp1_r=1.2)
        bar = _make_bar(close=96.0, high=101.0, low=93.0)
        orders = mgr.manage(pos, bar, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state.tp1_hit is True
        tp1_orders = [o for o in orders if o.tag == "tp1"]
        assert len(tp1_orders) == 1

    def test_tp2_triggers_on_intra_bar_peak(self):
        """TP2 should also use intra-bar peak."""
        params = ExitParams(tp1_r=1.0, tp1_frac=0.3, tp2_r=2.0, tp2_frac=0.2)
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # Bar 1: trigger TP1 (high=106, 1.2R > 1.0)
        bar1 = _make_bar(close=106.0, high=106.0, low=99.0)
        mgr.manage(pos, bar1, [], None, MagicMock())
        assert mgr.get_state("BTC").tp1_hit is True

        # Bar 2: close=107 (1.4R), high=111 (2.2R > tp2_r=2.0)
        bar2 = _make_bar(close=107.0, high=111.0, low=105.0)
        orders = mgr.manage(pos, bar2, [], None, MagicMock())

        state = mgr.get_state("BTC")
        assert state.tp2_hit is True
        tp2_orders = [o for o in orders if o.tag == "tp2"]
        assert len(tp2_orders) == 1


# ===========================================================================
# Fix 2: MFE-Floor Trail Buffer
# ===========================================================================


class TestMFEFloorTrailBuffer:
    """Trail buffer should not go below floor when MFE exceeds threshold."""

    def test_mfe_floor_enforces_minimum_buffer_long(self):
        """High MFE should prevent trail from getting too tight."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_mfe_floor_enabled=True,
            trail_mfe_floor_threshold=0.8,
            trail_mfe_floor_buffer=0.5,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_trail_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position(Side.LONG)

        # At current_r=2.0 (ceiling), R-adaptive gives buffer_mult=0.3
        # But with mfe_r=1.5 (>0.8 threshold), floor enforces min 0.5*ATR=1.0
        # Without floor: buffer = 2.0 * 0.3 = 0.6
        # With floor: buffer = max(0.6, 2.0 * 0.5) = max(0.6, 1.0) = 1.0
        result_with_floor = mgr.update(
            pos, bars, ind, current_stop=80.0,
            bars_since_entry=10, current_r=2.0, mfe_r=1.5,
        )

        # Now without floor
        params_no_floor = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_mfe_floor_enabled=False,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr2 = TrailManager(params_no_floor)
        result_without_floor = mgr2.update(
            pos, bars, ind, current_stop=80.0,
            bars_since_entry=10, current_r=2.0, mfe_r=1.5,
        )

        # Floor gives wider buffer → lower trail stop for long
        assert result_with_floor is not None
        assert result_without_floor is not None
        assert result_with_floor <= result_without_floor

    def test_mfe_floor_inactive_below_threshold(self):
        """MFE below threshold should not trigger floor."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_mfe_floor_enabled=True,
            trail_mfe_floor_threshold=0.8,
            trail_mfe_floor_buffer=0.5,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr1 = TrailManager(params)

        params_off = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_mfe_floor_enabled=False,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr2 = TrailManager(params_off)

        bars = _make_trail_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position(Side.LONG)

        # mfe_r=0.5 < threshold=0.8 → floor inactive → same result
        r1 = mgr1.update(pos, bars, ind, current_stop=80.0,
                         bars_since_entry=5, current_r=0.5, mfe_r=0.5)
        r2 = mgr2.update(pos, bars, ind, current_stop=80.0,
                         bars_since_entry=5, current_r=0.5, mfe_r=0.5)
        assert r1 == r2

    def test_mfe_floor_disabled_has_no_effect(self):
        """trail_mfe_floor_enabled=False should not apply floor."""
        params = TrailParams(
            trail_mfe_floor_enabled=False,
            trail_mfe_floor_threshold=0.8,
            trail_mfe_floor_buffer=10.0,  # huge value, would be obvious if applied
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_trail_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position(Side.LONG)

        result = mgr.update(pos, bars, ind, current_stop=80.0,
                            bars_since_entry=5, current_r=1.0, mfe_r=2.0)
        # Should return a value (trail active) without applying the huge floor
        assert result is not None

    def test_mfe_floor_works_with_legacy_trail(self):
        """MFE floor should also work when trail_r_adaptive=False."""
        params = TrailParams(
            trail_r_adaptive=False,
            trail_atr_buffer=0.2,  # small buffer
            trail_warmup_bars=0,
            trail_mfe_floor_enabled=True,
            trail_mfe_floor_threshold=0.8,
            trail_mfe_floor_buffer=1.0,  # floor buffer larger than atr_buffer
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_trail_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position(Side.LONG)

        # Legacy buffer: 2.0 * 0.2 = 0.4
        # Floor: 2.0 * 1.0 = 2.0
        # Result should use floor (wider buffer → lower stop for long)
        result = mgr.update(pos, bars, ind, current_stop=80.0,
                            bars_since_entry=5, current_r=1.0, mfe_r=1.5)
        assert result is not None


# ===========================================================================
# Fix 3: Quick Exit for Stagnant Trades
# ===========================================================================


class TestQuickExit:
    """Stagnant trades with low MFE should be exited early."""

    def test_quick_exit_triggers_stagnant_trade(self):
        """Trade with low MFE after N bars should be quick-exited."""
        params = ExitParams(
            tp1_r=99.0,  # unreachable
            quick_exit_enabled=True,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # Simulate 3 bars with no momentum (close at/below entry, small highs)
        # current_r = (99.8-100)/5 = -0.04 <= 0.0
        # peak_r = (100.5-100)/5 = 0.1 < 0.3 threshold
        for _ in range(3):
            bar = _make_bar(close=99.8, high=100.5, low=99.5)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        # After 3 bars with MFE < 0.3R and current_r <= 0.0, should quick exit
        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 1

    def test_quick_exit_does_not_trigger_before_bars_threshold(self):
        """Quick exit should not trigger before quick_exit_bars."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=True,
            quick_exit_bars=5,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # Only 3 bars (< 5 threshold)
        for _ in range(3):
            bar = _make_bar(close=99.5, high=100.2, low=99.0)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 0

    def test_quick_exit_does_not_trigger_with_high_mfe(self):
        """Trade that showed momentum (high MFE) should not be quick-exited."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=True,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # Bar 1: big spike (high=103 → 0.6R MFE > 0.3 threshold)
        bar1 = _make_bar(close=101.0, high=103.0, low=99.0)
        mgr.manage(pos, bar1, [], None, MagicMock())

        # Bars 2-3: price drops back
        for _ in range(2):
            bar = _make_bar(close=99.5, high=100.0, low=99.0)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 0

    def test_quick_exit_does_not_trigger_when_in_profit(self):
        """Trade currently in profit (current_r > max_r) should not be quick-exited."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=True,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # 3 bars with close above entry (current_r > 0)
        # but high is modest (MFE stays below 0.3R)
        for _ in range(3):
            bar = _make_bar(close=100.5, high=101.0, low=99.5)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        # current_r = (100.5-100)/5 = 0.1 > 0.0 = max_r → should NOT trigger
        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 0

    def test_quick_exit_disabled(self):
        """quick_exit_enabled=False should not trigger quick exit."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=False,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        for _ in range(5):
            bar = _make_bar(close=99.5, high=100.0, low=99.0)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 0

    def test_quick_exit_negative_max_r(self):
        """With max_r=-0.3, only trades at -0.3R or worse get quick-exited."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=True,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=-0.3,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.LONG)
        # 3 bars with current_r = -0.1 (close=99.5) > -0.3 → no trigger
        for _ in range(3):
            bar = _make_bar(close=99.5, high=100.0, low=99.0)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 0

    def test_quick_exit_short_position(self):
        """Quick exit works for short positions."""
        params = ExitParams(
            tp1_r=99.0,
            quick_exit_enabled=True,
            quick_exit_bars=3,
            quick_exit_max_mfe_r=0.3,
            quick_exit_max_r=0.0,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        pos = _make_position(Side.SHORT)
        # 3 bars with price near entry, no downward momentum
        for _ in range(3):
            bar = _make_bar(close=100.2, high=101.0, low=99.8)
            orders = mgr.manage(pos, bar, [], None, MagicMock())

        # current_r for short = (100-100.2)/5 = -0.04 <= 0.0
        # mfe_r from low: (100-99.8)/5 = 0.04 < 0.3
        quick_orders = [o for o in orders if o.tag == "quick_exit"]
        assert len(quick_orders) == 1


# ===========================================================================
# Config Defaults & Serialization
# ===========================================================================


class TestConfigDefaults:
    """New config fields have correct defaults and serialize properly."""

    def test_exit_params_quick_exit_defaults(self):
        params = ExitParams()
        assert params.quick_exit_enabled is True
        assert params.quick_exit_bars == 6
        assert params.quick_exit_max_mfe_r == 0.15
        assert params.quick_exit_max_r == -0.3

    def test_trail_params_mfe_floor_defaults(self):
        params = TrailParams()
        assert params.trail_mfe_floor_enabled is False
        assert params.trail_mfe_floor_threshold == 0.8
        assert params.trail_mfe_floor_buffer == 0.5

    def test_momentum_config_round_trip(self):
        """New fields survive to_dict() / from_dict() round-trip."""
        cfg = MomentumConfig()
        d = cfg.to_dict()
        cfg2 = MomentumConfig.from_dict(d)

        assert cfg2.exits.quick_exit_enabled == cfg.exits.quick_exit_enabled
        assert cfg2.exits.quick_exit_bars == cfg.exits.quick_exit_bars
        assert cfg2.exits.quick_exit_max_mfe_r == cfg.exits.quick_exit_max_mfe_r
        assert cfg2.exits.quick_exit_max_r == cfg.exits.quick_exit_max_r
        assert cfg2.trail.trail_mfe_floor_enabled == cfg.trail.trail_mfe_floor_enabled
        assert cfg2.trail.trail_mfe_floor_threshold == cfg.trail.trail_mfe_floor_threshold
        assert cfg2.trail.trail_mfe_floor_buffer == cfg.trail.trail_mfe_floor_buffer

    def test_position_exit_state_has_peak_r(self):
        state = PositionExitState()
        assert hasattr(state, "peak_r")
        assert state.peak_r == 0.0


# ===========================================================================
# Optimizer Experiments
# ===========================================================================


class TestOptimizerExperiments:
    """New experiments are present in the plugin phase generators."""

    def test_phase1_has_trail_and_mfe_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase1_candidates
        experiments = _phase1_candidates()
        names = [e.name for e in experiments]
        assert "TRAIL_CEILING_1_5" in names
        assert "TRAIL_CEILING_1_0" in names
        assert "TRAIL_MFE_FLOOR_HIGH" in names
        assert "TRAIL_MFE_FLOOR_MID" in names
        assert "STOP_ATR_1_5" in names
        assert "STOP_ATR_2_5" in names

    def test_phase2_has_tp_and_quick_exit_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase2_candidates
        experiments = _phase2_candidates()
        names = [e.name for e in experiments]
        assert "TP1_R_0_8" in names
        assert "TP1_FRAC_0_25" in names
        assert "TP2_R_1_5" in names
        assert "QUICK_BARS_4" in names
        assert "QUICK_BARS_3" in names
        assert "QUICK_MFE_0_3" in names
        assert "QUICK_R_0" in names
        assert "TIME_SOFT_8" in names
