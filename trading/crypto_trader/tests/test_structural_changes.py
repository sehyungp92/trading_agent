"""Tests for structural strategy changes: symbol direction filter, confluence gate, trail warmup."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from crypto_trader.core.models import Bar, Position, Side, TimeFrame
from crypto_trader.strategy.momentum.config import (
    ConfirmationParams,
    MomentumConfig,
    SymbolFilterParams,
    TrailParams,
)
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot
from crypto_trader.strategy.momentum.trail import TrailManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int = 25, base: float = 100.0) -> list[Bar]:
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


def _make_indicators(atr: float = 2.0) -> IndicatorSnapshot:
    dummy_arr = np.array([100.0])
    return IndicatorSnapshot(
        ema_fast=100.0, ema_mid=99.0, ema_slow=98.0,
        ema_fast_arr=dummy_arr, ema_mid_arr=dummy_arr, ema_slow_arr=dummy_arr,
        adx=25.0, di_plus=20.0, di_minus=15.0, adx_rising=True,
        atr=atr, atr_avg=atr, rsi=55.0, volume_ma=100.0,
    )


def _make_position(symbol: str = "BTC", direction: Side = Side.LONG) -> Position:
    return Position(
        symbol=symbol, direction=direction, qty=1.0,
        avg_entry=100.0, unrealized_pnl=0.0,
    )


# ===========================================================================
# SymbolFilterParams
# ===========================================================================


class TestSymbolFilterParams:
    """Test the SymbolFilterParams dataclass."""

    def test_defaults(self):
        sf = SymbolFilterParams()
        assert sf.btc_direction == "both"
        assert sf.eth_direction == "both"
        assert sf.sol_direction == "both"

    def test_custom_values(self):
        sf = SymbolFilterParams(
            btc_direction="long_only",
            eth_direction="disabled",
            sol_direction="short_only",
        )
        assert sf.btc_direction == "long_only"
        assert sf.eth_direction == "disabled"
        assert sf.sol_direction == "short_only"


class TestSymbolFilterInConfig:
    """Test symbol_filter is properly wired into MomentumConfig."""

    def test_config_has_symbol_filter(self):
        cfg = MomentumConfig()
        assert hasattr(cfg, "symbol_filter")
        assert isinstance(cfg.symbol_filter, SymbolFilterParams)

    def test_config_round_trip(self):
        cfg = MomentumConfig()
        cfg_dict = cfg.to_dict()
        assert "symbol_filter" in cfg_dict
        assert cfg_dict["symbol_filter"]["eth_direction"] == "both"

        restored = MomentumConfig.from_dict(cfg_dict)
        assert restored.symbol_filter.eth_direction == "both"
        assert restored.symbol_filter.btc_direction == "both"

    def test_config_from_dict_custom(self):
        d = MomentumConfig().to_dict()
        d["symbol_filter"]["btc_direction"] = "short_only"
        d["symbol_filter"]["eth_direction"] = "disabled"
        restored = MomentumConfig.from_dict(d)
        assert restored.symbol_filter.btc_direction == "short_only"
        assert restored.symbol_filter.eth_direction == "disabled"


class TestDirectionFilterLogic:
    """Test the direction filter logic as implemented in strategy."""

    @staticmethod
    def _check_filter(sym: str, direction: Side, sf: SymbolFilterParams) -> bool:
        """Replicate strategy step 5.5 logic — returns True if trade allowed."""
        rule = getattr(sf, f"{sym.lower().replace('usdt', '')}_direction", "both")
        if rule == "disabled":
            return False
        if rule == "long_only" and direction == Side.SHORT:
            return False
        if rule == "short_only" and direction == Side.LONG:
            return False
        return True

    def test_eth_long_only_blocks_short(self):
        sf = SymbolFilterParams(eth_direction="long_only")
        assert not self._check_filter("ETH", Side.SHORT, sf)

    def test_eth_long_only_allows_long(self):
        sf = SymbolFilterParams(eth_direction="long_only")
        assert self._check_filter("ETH", Side.LONG, sf)

    def test_disabled_blocks_both(self):
        sf = SymbolFilterParams(eth_direction="disabled")
        assert not self._check_filter("ETH", Side.LONG, sf)
        assert not self._check_filter("ETH", Side.SHORT, sf)

    def test_both_allows_all(self):
        sf = SymbolFilterParams(btc_direction="both")
        assert self._check_filter("BTC", Side.LONG, sf)
        assert self._check_filter("BTC", Side.SHORT, sf)

    def test_short_only_blocks_long(self):
        sf = SymbolFilterParams(sol_direction="short_only")
        assert not self._check_filter("SOL", Side.LONG, sf)
        assert self._check_filter("SOL", Side.SHORT, sf)

    def test_usdt_suffix_stripped(self):
        sf = SymbolFilterParams(btc_direction="long_only")
        assert self._check_filter("BTCUSDT", Side.LONG, sf)
        assert not self._check_filter("BTCUSDT", Side.SHORT, sf)

    def test_unknown_symbol_defaults_both(self):
        sf = SymbolFilterParams()
        # Unknown symbol has no matching attribute, getattr returns "both"
        assert self._check_filter("DOGE", Side.LONG, sf)
        assert self._check_filter("DOGE", Side.SHORT, sf)


# ===========================================================================
# Confluence Gate for Weak Confirmations
# ===========================================================================


class TestConfluenceGateConfig:
    """Test the new ConfirmationParams fields."""

    def test_defaults(self):
        cp = ConfirmationParams()
        assert cp.min_confluences_for_weak == 2
        assert "micro_structure_shift" in cp.weak_confirmations
        assert "shooting_star" in cp.weak_confirmations

    def test_round_trip(self):
        cfg = MomentumConfig()
        d = cfg.to_dict()
        assert d["confirmation"]["min_confluences_for_weak"] == 2
        restored = MomentumConfig.from_dict(d)
        assert restored.confirmation.min_confluences_for_weak == 2
        # tuples become lists in JSON but should work
        assert "micro_structure_shift" in restored.confirmation.weak_confirmations


class TestConfluenceGateLogic:
    """Test the confluence gate logic as implemented in strategy step 8.5."""

    @staticmethod
    def _check_gate(
        pattern_type: str,
        n_confluences: int,
        cfg: ConfirmationParams,
    ) -> bool:
        """Replicate strategy step 8.5 logic — returns True if trade allowed."""
        if pattern_type in cfg.weak_confirmations:
            if n_confluences < cfg.min_confluences_for_weak:
                return False
        return True

    def test_weak_confirmation_blocked_low_confluences(self):
        cp = ConfirmationParams(min_confluences_for_weak=2)
        assert not self._check_gate("micro_structure_shift", 1, cp)
        assert not self._check_gate("micro_structure_shift", 0, cp)

    def test_weak_confirmation_allowed_high_confluences(self):
        cp = ConfirmationParams(min_confluences_for_weak=2)
        assert self._check_gate("micro_structure_shift", 2, cp)
        assert self._check_gate("micro_structure_shift", 3, cp)

    def test_strong_confirmation_always_allowed(self):
        cp = ConfirmationParams(min_confluences_for_weak=2)
        assert self._check_gate("bullish_engulfing", 0, cp)
        assert self._check_gate("hammer", 0, cp)
        assert self._check_gate("inside_bar_break", 1, cp)

    def test_shooting_star_gated(self):
        cp = ConfirmationParams(min_confluences_for_weak=2)
        assert not self._check_gate("shooting_star", 1, cp)
        assert self._check_gate("shooting_star", 2, cp)

    def test_gate_disabled_with_zero(self):
        cp = ConfirmationParams(min_confluences_for_weak=0)
        assert self._check_gate("micro_structure_shift", 0, cp)

    def test_gate_with_threshold_three(self):
        cp = ConfirmationParams(min_confluences_for_weak=3)
        assert not self._check_gate("micro_structure_shift", 2, cp)
        assert self._check_gate("micro_structure_shift", 3, cp)


# ===========================================================================
# Trail Warmup Buffer
# ===========================================================================


class TestTrailWarmupConfig:
    """Test the new TrailParams warmup fields."""

    def test_defaults(self):
        tp = TrailParams()
        assert tp.trail_warmup_bars == 5
        assert tp.trail_warmup_buffer_mult == 1.0

    def test_round_trip(self):
        cfg = MomentumConfig()
        d = cfg.to_dict()
        assert d["trail"]["trail_warmup_bars"] == 5
        assert d["trail"]["trail_warmup_buffer_mult"] == 1.0
        restored = MomentumConfig.from_dict(d)
        assert restored.trail.trail_warmup_bars == 5
        assert restored.trail.trail_warmup_buffer_mult == 1.0


class TestTrailWarmupBuffer:
    """Test the decaying warmup buffer in TrailManager."""

    def test_warmup_produces_wider_buffer_at_activation(self):
        """Just after activation, the buffer should be wider than normal."""
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        # With warmup: activation_bars=3, warmup_bars=5, warmup_buffer_mult=1.0
        params_warmup = TrailParams(
            trail_activation_bars=3,
            trail_warmup_bars=5,
            trail_warmup_buffer_mult=1.0,
            trail_behind_structure=False,
            trail_behind_ema=True,
            trail_r_adaptive=False,
        )
        mgr_warmup = TrailManager(params_warmup)
        result_warmup = mgr_warmup.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=3, current_r=0.0,
        )

        # Without warmup: same params but warmup_bars=0
        params_no_warmup = TrailParams(
            trail_activation_bars=3,
            trail_warmup_bars=0,
            trail_warmup_buffer_mult=1.0,
            trail_behind_structure=False,
            trail_behind_ema=True,
            trail_r_adaptive=False,
        )
        mgr_no_warmup = TrailManager(params_no_warmup)
        result_no_warmup = mgr_no_warmup.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=3, current_r=0.0,
        )

        # Both should produce results
        assert result_warmup is not None
        assert result_no_warmup is not None
        # Warmup should be lower (wider buffer for long)
        assert result_warmup < result_no_warmup

    def test_warmup_decays_to_zero(self):
        """After warmup_bars elapse, extra buffer is zero."""
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        params_warmup = TrailParams(
            trail_activation_bars=3,
            trail_warmup_bars=5,
            trail_warmup_buffer_mult=1.0,
            trail_behind_structure=False,
            trail_behind_ema=True,
            trail_r_adaptive=False,
        )
        mgr_warmup = TrailManager(params_warmup)
        # bars_active = 8 - 3 = 5 => warmup_remaining = 1 - 5/5 = 0
        result_warmup = mgr_warmup.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=8, current_r=0.0,
        )

        params_no_warmup = TrailParams(
            trail_activation_bars=3,
            trail_warmup_bars=0,
            trail_warmup_buffer_mult=1.0,
            trail_behind_structure=False,
            trail_behind_ema=True,
            trail_r_adaptive=False,
        )
        mgr_no_warmup = TrailManager(params_no_warmup)
        result_no_warmup = mgr_no_warmup.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=8, current_r=0.0,
        )

        assert result_warmup is not None
        assert result_no_warmup is not None
        # After warmup fully decayed, both should be equal
        assert abs(result_warmup - result_no_warmup) < 1e-10

    def test_warmup_halfway_has_partial_buffer(self):
        """Halfway through warmup, buffer should be half the initial extra."""
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        params = TrailParams(
            trail_activation_bars=0,
            trail_warmup_bars=10,
            trail_warmup_buffer_mult=2.0,
            trail_behind_structure=False,
            trail_behind_ema=True,
            trail_r_adaptive=False,
        )

        # At bar 0: warmup_remaining = 1.0, extra = atr * 2.0 * 1.0 = 4.0
        mgr0 = TrailManager(params)
        result_start = mgr0.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=0, current_r=0.0,
        )

        # At bar 5: warmup_remaining = 0.5, extra = atr * 2.0 * 0.5 = 2.0
        mgr5 = TrailManager(params)
        result_mid = mgr5.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=5, current_r=0.0,
        )

        # At bar 10: warmup_remaining = 0.0, extra = 0
        mgr10 = TrailManager(params)
        result_end = mgr10.update(
            pos, bars, ind, current_stop=50.0,
            bars_since_entry=10, current_r=0.0,
        )

        assert result_start is not None
        assert result_mid is not None
        assert result_end is not None

        # For longs: lower trail = wider buffer
        # start < mid < end (as buffer shrinks, trail gets tighter/higher)
        assert result_start < result_mid < result_end

    def test_warmup_disabled_when_zero_bars(self):
        """trail_warmup_bars=0 should not add any extra buffer."""
        params = TrailParams(
            trail_warmup_bars=0,
            trail_warmup_buffer_mult=5.0,  # Large mult that would be obvious
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_r_adaptive=False,
        )
        mgr = TrailManager(params)
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        # warmup_bars=0 means the warmup block is skipped entirely
        result = mgr.update(pos, bars, ind, current_stop=50.0,
                           bars_since_entry=0, current_r=0.0)

        # Same without warmup mult
        params2 = TrailParams(
            trail_warmup_bars=0,
            trail_warmup_buffer_mult=0.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_r_adaptive=False,
        )
        mgr2 = TrailManager(params2)
        result2 = mgr2.update(pos, bars, ind, current_stop=50.0,
                             bars_since_entry=0, current_r=0.0)

        assert result is not None
        assert result2 is not None
        assert abs(result - result2) < 1e-10


# ===========================================================================
# Updated Default Verification
# ===========================================================================


class TestUpdatedDefaults:
    """Verify all default changes from the plan are applied."""

    def test_risk_defaults(self):
        cfg = MomentumConfig()
        assert cfg.risk.risk_pct_a == 0.02
        assert cfg.risk.risk_pct_b == 0.0125
        assert cfg.risk.max_leverage_major == 10.0
        assert cfg.risk.max_leverage_alt == 8.0
        assert cfg.risk.max_gross_risk == 0.05

    def test_setup_defaults(self):
        cfg = MomentumConfig()
        assert cfg.setup.min_confluences_b == 0

    def test_trail_defaults(self):
        cfg = MomentumConfig()
        assert cfg.trail.trail_activation_bars == 3
        assert cfg.trail.trail_activation_r == 0.5
        assert cfg.trail.trail_warmup_bars == 5
        assert cfg.trail.trail_warmup_buffer_mult == 1.0
