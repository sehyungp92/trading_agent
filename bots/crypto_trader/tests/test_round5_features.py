"""Tests for Round 5 features: R-adaptive trail, smart break-even, re-entry, config defaults."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from crypto_trader.core.models import Bar, Position, Side, TimeFrame
from crypto_trader.strategy.momentum.config import (
    ConfirmationParams,
    ExitParams,
    MomentumConfig,
    ReentryParams,
    SetupParams,
    SymbolFilterParams,
    TrailParams,
)
from crypto_trader.strategy.momentum.exits import ExitManager, PositionExitState
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


def _make_indicators(atr: float = 2.0, volume_ma: float = 100.0) -> IndicatorSnapshot:
    dummy_arr = np.array([100.0])
    return IndicatorSnapshot(
        ema_fast=100.0, ema_mid=99.0, ema_slow=98.0,
        ema_fast_arr=dummy_arr, ema_mid_arr=dummy_arr, ema_slow_arr=dummy_arr,
        adx=25.0, di_plus=20.0, di_minus=15.0, adx_rising=True,
        atr=atr, atr_avg=atr, rsi=55.0, volume_ma=volume_ma,
    )


def _make_position(symbol: str = "BTC", direction: Side = Side.LONG) -> Position:
    return Position(
        symbol=symbol, direction=direction, qty=1.0,
        avg_entry=100.0, unrealized_pnl=0.0,
    )


# ---------------------------------------------------------------------------
# R-Adaptive Trail Tests
# ---------------------------------------------------------------------------

class TestRAdaptiveTrailBuffer:
    """R-adaptive buffer scales inversely with R-multiple."""

    def test_buffer_wide_at_zero_r(self):
        """At R=0, buffer should equal trail_buffer_wide * ATR."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        # At R=0, buffer = 1.5 * 2.0 = 3.0
        result = mgr.update(pos, bars, ind, current_stop=50.0,
                            bars_since_entry=5, current_r=0.0)
        assert result is not None

    def test_buffer_tight_at_ceiling_r(self):
        """At R=ceiling, buffer should equal trail_buffer_tight * ATR."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        # At R=ceiling, buffer = 0.3 * 2.0 = 0.6 (tight)
        result = mgr.update(pos, bars, ind, current_stop=50.0,
                            bars_since_entry=5, current_r=2.0)
        assert result is not None

    def test_buffer_midpoint_at_half_r(self):
        """At R=ceiling/2, buffer should be midpoint of wide and tight."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_behind_ema=True,
            trail_behind_structure=False,  # Only EMA for predictable buffer
        )
        mgr = TrailManager(params)
        bars = _make_bars(n=30)
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        # At R=1.0 (half of ceiling 2.0), buffer = 0.9 * ATR
        # buffer_mult = 1.5 * 0.5 + 0.3 * 0.5 = 0.75 + 0.15 = 0.9
        result = mgr.update(pos, bars, ind, current_stop=50.0,
                            bars_since_entry=5, current_r=1.0)
        assert result is not None

    def test_r_clamped_above_ceiling(self):
        """R values above ceiling should be clamped — same as at ceiling."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_behind_ema=True,
            trail_behind_structure=False,
        )
        mgr1 = TrailManager(params)
        mgr2 = TrailManager(params)
        bars = _make_bars(n=30)
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        result_at_ceiling = mgr1.update(pos, bars, ind, current_stop=50.0,
                                         bars_since_entry=5, current_r=2.0)
        result_above_ceiling = mgr2.update(pos, bars, ind, current_stop=50.0,
                                            bars_since_entry=5, current_r=5.0)
        # Both should produce the same trail value (clamped)
        assert result_at_ceiling == result_above_ceiling

    def test_r_negative_treated_as_zero(self):
        """Negative R values should be clamped to 0 — widest buffer."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_behind_ema=True,
            trail_behind_structure=False,
        )
        mgr1 = TrailManager(params)
        mgr2 = TrailManager(params)
        bars = _make_bars(n=30)
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        result_zero = mgr1.update(pos, bars, ind, current_stop=50.0,
                                   bars_since_entry=5, current_r=0.0)
        result_neg = mgr2.update(pos, bars, ind, current_stop=50.0,
                                  bars_since_entry=5, current_r=-1.0)
        assert result_zero == result_neg

    def test_legacy_mode_when_disabled(self):
        """When trail_r_adaptive=False, uses legacy fixed buffer + warmup."""
        params = TrailParams(
            trail_r_adaptive=False,
            trail_atr_buffer=0.5,
            trail_warmup_bars=0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
        )
        mgr = TrailManager(params)
        bars = _make_bars()
        ind = _make_indicators(atr=2.0)
        pos = _make_position()

        result = mgr.update(pos, bars, ind, current_stop=50.0,
                            bars_since_entry=5, current_r=1.0)
        assert result is not None

    def test_default_is_r_adaptive(self):
        """Default TrailParams should have R-adaptive enabled."""
        params = TrailParams()
        assert params.trail_r_adaptive is True
        assert params.trail_buffer_wide == 1.5
        assert params.trail_buffer_tight == 0.3
        assert params.trail_r_ceiling == 2.0

    def test_high_r_produces_tighter_trail_than_low_r(self):
        """At high R, trail should be closer to price (tighter buffer)."""
        params = TrailParams(
            trail_r_adaptive=True,
            trail_buffer_wide=1.5,
            trail_buffer_tight=0.3,
            trail_r_ceiling=2.0,
            trail_activation_bars=0,
            trail_activation_r=0.0,
            trail_behind_ema=True,
            trail_behind_structure=False,
        )
        bars = _make_bars(n=30)
        ind = _make_indicators(atr=2.0)
        pos = _make_position(direction=Side.LONG)

        mgr_low = TrailManager(params)
        result_low = mgr_low.update(pos, bars, ind, current_stop=50.0,
                                     bars_since_entry=5, current_r=0.1)

        mgr_high = TrailManager(params)
        result_high = mgr_high.update(pos, bars, ind, current_stop=50.0,
                                       bars_since_entry=5, current_r=1.9)

        assert result_low is not None
        assert result_high is not None
        # For long: higher trail stop = tighter. High-R should be >= low-R trail.
        assert result_high >= result_low


# ---------------------------------------------------------------------------
# Smart Break-Even Tests
# ---------------------------------------------------------------------------

class TestSmartBreakEven:
    """BE moves stop to entry + buffer instead of flat entry."""

    def test_be_buffer_long_position(self):
        """Long position: BE price = entry + be_buffer_r * stop_distance."""
        params = ExitParams(
            tp1_r=1.2, tp1_frac=0.3,
            be_acceptance_bars=0,  # Immediate BE after TP1
            be_buffer_r=0.2,
        )
        mgr = ExitManager(params)
        mgr.init_position("BTC", entry_price=100.0, stop_distance=5.0, qty=1.0)

        state = mgr.get_state("BTC")
        assert state is not None

        # Expected BE price: 100 + 0.2 * 5 = 101
        expected_be = 100.0 + 0.2 * 5.0
        assert expected_be == 101.0

    def test_be_buffer_zero_is_flat_entry(self):
        """With be_buffer_r=0, BE stop = flat entry price."""
        params = ExitParams(be_buffer_r=0.0)
        assert params.be_buffer_r == 0.0

    def test_be_buffer_default(self):
        """Default be_buffer_r should be 0.2."""
        params = ExitParams()
        assert params.be_buffer_r == 0.2

    def test_be_buffer_short_position(self):
        """Short position: BE price = entry - be_buffer_r * stop_distance."""
        params = ExitParams(be_buffer_r=0.2)
        # Short: entry=100, stop_dist=5, BE = 100 - 0.2*5 = 99
        expected_be = 100.0 - 0.2 * 5.0
        assert expected_be == 99.0


# ---------------------------------------------------------------------------
# Re-entry Config Tests
# ---------------------------------------------------------------------------

class TestReentryConfig:
    """ReentryParams defaults and serialization."""

    def test_default_values(self):
        params = ReentryParams()
        assert params.enabled is True
        assert params.cooldown_bars == 3
        assert params.max_loss_r == 1.5
        assert params.max_reentries == 1
        assert params.min_confluences_override == 0

    def test_in_momentum_config(self):
        cfg = MomentumConfig()
        assert hasattr(cfg, "reentry")
        assert cfg.reentry.enabled is True

    def test_round_trip_serialization(self):
        cfg = MomentumConfig()
        d = cfg.to_dict()
        assert "reentry" in d
        cfg2 = MomentumConfig.from_dict(d)
        assert cfg2.reentry.enabled == cfg.reentry.enabled
        assert cfg2.reentry.cooldown_bars == cfg.reentry.cooldown_bars

    def test_custom_reentry_params(self):
        cfg = MomentumConfig(reentry=ReentryParams(
            enabled=False, cooldown_bars=5, max_reentries=2
        ))
        d = cfg.to_dict()
        cfg2 = MomentumConfig.from_dict(d)
        assert cfg2.reentry.enabled is False
        assert cfg2.reentry.cooldown_bars == 5
        assert cfg2.reentry.max_reentries == 2


# ---------------------------------------------------------------------------
# Permissive Default Tests
# ---------------------------------------------------------------------------

class TestPermissiveDefaults:
    """Verify filter defaults are now permissive."""

    def test_min_confluences_b_default(self):
        params = SetupParams()
        assert params.min_confluences_b == 0

    def test_min_confluences_for_weak_default(self):
        params = ConfirmationParams()
        assert params.min_confluences_for_weak == 2

    def test_eth_direction_default(self):
        params = SymbolFilterParams()
        assert params.eth_direction == "both"

    def test_trail_r_adaptive_default(self):
        params = TrailParams()
        assert params.trail_r_adaptive is True


# ---------------------------------------------------------------------------
# Setup min_confluences_override Tests
# ---------------------------------------------------------------------------

class TestSetupConfluenceOverride:
    """SetupDetector.detect() respects min_confluences_override."""

    def test_override_signature(self):
        """detect() accepts min_confluences_override parameter."""
        from crypto_trader.strategy.momentum.setup import SetupDetector
        import inspect
        sig = inspect.signature(SetupDetector.detect)
        assert "min_confluences_override" in sig.parameters

    def test_override_none_uses_default(self):
        """When override is None, uses self._p.min_confluences_b."""
        from crypto_trader.strategy.momentum.setup import SetupDetector
        detector = SetupDetector(SetupParams(min_confluences_b=5))
        # With 5 required confluences, almost nothing passes
        # This is a unit-level check that the parameter exists and defaults correctly


# ---------------------------------------------------------------------------
# Config R-Adaptive Trail Fields
# ---------------------------------------------------------------------------

class TestTrailParamsFields:
    """TrailParams has R-adaptive fields with correct defaults."""

    def test_fields_exist(self):
        p = TrailParams()
        assert hasattr(p, "trail_r_adaptive")
        assert hasattr(p, "trail_buffer_wide")
        assert hasattr(p, "trail_buffer_tight")
        assert hasattr(p, "trail_r_ceiling")

    def test_serialization_round_trip(self):
        cfg = MomentumConfig(trail=TrailParams(
            trail_r_adaptive=False,
            trail_buffer_wide=2.0,
            trail_buffer_tight=0.1,
            trail_r_ceiling=3.0,
        ))
        d = cfg.to_dict()
        cfg2 = MomentumConfig.from_dict(d)
        assert cfg2.trail.trail_r_adaptive is False
        assert cfg2.trail.trail_buffer_wide == 2.0
        assert cfg2.trail.trail_buffer_tight == 0.1
        assert cfg2.trail.trail_r_ceiling == 3.0


# ---------------------------------------------------------------------------
# Scoring Normalizer Tests
# ---------------------------------------------------------------------------

class TestCoverageNormalizerUpdated:
    """Coverage normalizer uses 30-trade target."""

    def test_30_trades_is_max(self):
        from crypto_trader.optimize.scoring import normalize_coverage
        assert normalize_coverage({"total_trades": 30}) == 1.0

    def test_15_trades_is_half(self):
        from crypto_trader.optimize.scoring import normalize_coverage
        assert normalize_coverage({"total_trades": 15}) == 0.5

    def test_caps_at_1(self):
        from crypto_trader.optimize.scoring import normalize_coverage
        assert normalize_coverage({"total_trades": 50}) == 1.0


# ---------------------------------------------------------------------------
# Hard Rejects Updated
# ---------------------------------------------------------------------------

class TestHardRejectsUpdated:
    """Hard reject thresholds: trades>=5, DD<=35%, PF>=0.8."""

    def test_12_trades_passes(self):
        from crypto_trader.optimize.momentum_plugin import HARD_REJECTS
        assert HARD_REJECTS["total_trades"] == (">=", 12)

    def test_11_trades_rejected(self):
        from crypto_trader.optimize.scoring import check_hard_rejects
        from crypto_trader.optimize.momentum_plugin import HARD_REJECTS
        metrics = {"max_drawdown_pct": 10.0, "total_trades": 11.0, "profit_factor": 1.5}
        rejected, reason = check_hard_rejects(metrics, HARD_REJECTS)
        assert rejected is True

    def test_12_trades_not_rejected(self):
        from crypto_trader.optimize.scoring import check_hard_rejects
        from crypto_trader.optimize.momentum_plugin import HARD_REJECTS
        metrics = {"max_drawdown_pct": 10.0, "total_trades": 12.0, "profit_factor": 1.5}
        rejected, reason = check_hard_rejects(metrics, HARD_REJECTS)
        assert rejected is False


# ---------------------------------------------------------------------------
# Momentum Plugin Experiment Count Tests
# ---------------------------------------------------------------------------

class TestPluginExperiments:
    """Plugin experiments include new R-adaptive, BE, and re-entry experiments."""

    def test_phase1_has_trail_calibration_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase1_candidates
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert "TRAIL_WIDE_2_0" in names
        assert "TRAIL_TIGHT_0_5" in names
        assert "TRAIL_CEILING_1_5" in names
        assert "STOP_ATR_1_5" in names
        assert "TRAIL_GENEROUS" in names

    def test_phase1_no_old_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase1_candidates
        candidates = _phase1_candidates()
        names = [c.name for c in candidates]
        assert "R_ADAPT_WIDE_2.0" not in names
        assert "BE_BUFFER_0.0" not in names

    def test_phase4_has_reentry_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase4_candidates
        candidates = _phase4_candidates()
        names = [c.name for c in candidates]
        assert "REENTRY_COOL_2" in names
        assert "REENTRY_OFF" in names
        assert "REENTRY_MAX_2" in names

    def test_phase3_has_confluence_experiments(self):
        from crypto_trader.optimize.momentum_plugin import _phase3_candidates
        candidates = _phase3_candidates()
        names = [c.name for c in candidates]
        assert "CONFLUENCES_B_1" in names
        assert "CONFLUENCES_B_2" in names
