"""Tests for breakout setup detection and grading."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from crypto_trader.core.models import Bar, SetupGrade, Side, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceZone
from crypto_trader.strategy.breakout.config import BreakoutSetupParams, BreakoutSymbolFilterParams
from crypto_trader.strategy.breakout.context import ContextBias
from crypto_trader.strategy.breakout.profile import VolumeProfiler, VolumeProfileResult
from crypto_trader.strategy.breakout.setup import BreakoutDetector, BreakoutSetupResult
from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)


def make_zone(
    center: float = 100.0,
    upper: float = 105.0,
    lower: float = 95.0,
    bars: int = 12,
    touches: int = 4,
    formation_bar_idx: int = 0,
    volume_contracting: bool = False,
    width_atr: float = 1.0,
) -> BalanceZone:
    """Build a BalanceZone with sensible defaults."""
    return BalanceZone(
        center=center,
        upper=upper,
        lower=lower,
        bars_in_zone=bars,
        touches=touches,
        formation_bar_idx=formation_bar_idx,
        volume_contracting=volume_contracting,
        width_atr=width_atr,
    )


def make_bar_at(
    price: float,
    volume: float = 100.0,
    high_offset: float = 5.0,
    low_offset: float = 5.0,
) -> Bar:
    """Build a Bar with close=price and open=price-1 (bullish body)."""
    return Bar(
        timestamp=_TS,
        symbol="BTC",
        open=price - 1.0,
        high=price + high_offset,
        low=price - low_offset,
        close=price,
        volume=volume,
        timeframe=TimeFrame.M30,
    )


def make_context(
    direction: Side | None = None,
    strength: str = "none",
) -> ContextBias:
    """Build a ContextBias."""
    return ContextBias(direction=direction, strength=strength, reasons=())


def make_snapshot(**overrides) -> IndicatorSnapshot:
    """Return an IndicatorSnapshot with reasonable defaults."""
    defaults = dict(
        ema_fast=105.0,
        ema_mid=100.0,
        ema_slow=95.0,
        ema_fast_arr=None,
        ema_mid_arr=None,
        ema_slow_arr=None,
        rsi=50.0,
        adx=25.0,
        di_plus=20.0,
        di_minus=15.0,
        atr=2.0,
        volume_ma=100.0,
        adx_rising=True,
        atr_avg=2.0,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def _mock_profiler(lvn_runway: float = 5.0) -> VolumeProfiler:
    """Return a VolumeProfiler mock with a configurable find_lvn_runway."""
    profiler = MagicMock(spec=VolumeProfiler)
    profiler.find_lvn_runway.return_value = lvn_runway
    return profiler


def _mock_profile(
    poc: float = 100.0,
    hvn_levels: tuple[float, ...] = (100.0,),
) -> VolumeProfileResult:
    """Return a VolumeProfileResult mock."""
    return VolumeProfileResult(
        poc=poc,
        vah=105.0,
        val=95.0,
        hvn_levels=hvn_levels,
        lvn_levels=(110.0, 85.0),
        bin_edges=(90.0, 95.0, 100.0, 105.0, 110.0),
        bin_volumes=(50.0, 100.0, 200.0, 80.0),
        total_volume=430.0,
        price_low=90.0,
        price_high=110.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBreakoutDetector:
    """BreakoutDetector.detect() tests."""

    def test_no_zones_returns_none(self):
        """Empty zones list returns None."""
        det = BreakoutDetector(BreakoutSetupParams())
        result = det.detect(
            bar=make_bar_at(110),
            zones=[],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_no_breakout_inside_zone(self):
        """Bar close inside zone returns None."""
        det = BreakoutDetector(BreakoutSetupParams())
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(100)  # Inside zone
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_long_breakout_above_upper(self):
        """Bar close above zone.upper detects LONG breakout."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_b=0.0,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(108, high_offset=3, low_offset=3)  # close=108 > upper=105
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is not None
        assert result.direction == Side.LONG

    def test_short_breakout_below_lower(self):
        """Bar close below zone.lower detects SHORT breakout."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_b=0.0,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        # Bearish bar: open > close, close below lower
        bar = Bar(
            timestamp=_TS,
            symbol="BTC",
            open=93.0,
            high=94.0,
            low=89.0,
            close=92.0,
            volume=100.0,
            timeframe=TimeFrame.M30,
        )
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is not None
        assert result.direction == Side.SHORT

    def test_breakout_too_close(self):
        """Breakout distance < min_breakout_atr returns None."""
        cfg = BreakoutSetupParams(min_breakout_atr=1.0, max_breakout_atr=5.0)
        det = BreakoutDetector(cfg)
        zone = make_zone(upper=105, lower=95)
        # close=105.5, dist=0.5, atr=10 => dist_atr=0.05 < 1.0
        bar = make_bar_at(105.5, high_offset=3, low_offset=3)
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_breakout_too_far(self):
        """Breakout distance > max_breakout_atr returns None."""
        cfg = BreakoutSetupParams(min_breakout_atr=0.1, max_breakout_atr=0.5)
        det = BreakoutDetector(cfg)
        zone = make_zone(upper=105, lower=95)
        # close=120, dist=15, atr=10 => dist_atr=1.5 > 0.5
        bar = make_bar_at(120, high_offset=3, low_offset=3)
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_body_ratio_filter(self):
        """Small body (doji) filtered out by body_ratio_min."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.01,
            max_breakout_atr=10.0,
            body_ratio_min=0.50,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(upper=105, lower=95)
        # Doji: open~close, wide range => body_ratio ~ 0.02
        bar = Bar(
            timestamp=_TS,
            symbol="BTC",
            open=107.99,
            high=118.0,
            low=96.0,
            close=108.0,
            volume=100.0,
            timeframe=TimeFrame.M30,
        )
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_grading_a_plus(self):
        """Enough confluences for A+ grade."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_a_plus=4,
            min_confluences_a=2,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            volume_surge_mult=1.0,
            min_lvn_runway_atr=0.1,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(
            center=100, upper=105, lower=95,
            bars=20,  # >= 16 for balance_duration confluence
            volume_contracting=True,
        )
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        # profile with poc inside zone + 2 HVN in zone
        profile = _mock_profile(poc=100.0, hvn_levels=(98.0, 102.0))
        context = make_context(direction=Side.LONG, strength="strong")
        ind = make_snapshot(ema_fast=110.0, volume_ma=50.0)  # bar.vol(100) >= 50*1.0

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=profile,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=context,
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None
        assert result.grade == SetupGrade.A
        assert result.is_a_plus is True
        # Should have many confluences
        assert len(result.confluences) >= 4

    def test_grading_a(self):
        """Enough confluences for A but not A+."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_a_plus=5,
            min_confluences_a=2,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            volume_surge_mult=1.0,
            min_lvn_runway_atr=0.1,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95, bars=10)
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        context = make_context(direction=Side.LONG, strength="strong")
        ind = make_snapshot(ema_fast=110.0, volume_ma=50.0)

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=context,
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None
        assert result.grade == SetupGrade.A
        assert result.is_a_plus is False
        assert len(result.confluences) >= 2

    def test_grading_b(self):
        """Minimal confluences for B grade."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_a_plus=6,
            min_confluences_a=4,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95, bars=5)
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        context = make_context()  # Neutral -- no h4_alignment confluence
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=0.0),
            context=context,
            m30_ind=None,
            atr=10.0,
        )
        assert result is not None
        assert result.grade == SetupGrade.B

    def test_countertrend_forces_b(self):
        """Countertrend (context opposite to breakout direction) caps grade at B."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_a_plus=4,
            min_confluences_a=2,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            volume_surge_mult=1.0,
            min_lvn_runway_atr=0.1,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(
            center=100, upper=105, lower=95,
            bars=20,
            volume_contracting=True,
        )
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        profile = _mock_profile(poc=100.0, hvn_levels=(98.0, 102.0))
        # Context is SHORT but breakout is LONG -- countertrend
        context = make_context(direction=Side.SHORT, strength="strong")
        ind = make_snapshot(ema_fast=110.0, volume_ma=50.0)

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=profile,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=context,
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None
        assert result.grade == SetupGrade.B
        assert result.is_a_plus is False

    def test_best_setup_selected(self):
        """Multiple zones -- best confluences win."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            volume_surge_mult=1.0,
            min_lvn_runway_atr=0.1,
        )
        det = BreakoutDetector(cfg)
        zone1 = make_zone(center=100, upper=105, lower=95, bars=5)
        zone2 = make_zone(
            center=100, upper=105, lower=95,
            bars=20, volume_contracting=True,
        )
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        context = make_context(direction=Side.LONG, strength="strong")
        ind = make_snapshot(ema_fast=110.0, volume_ma=50.0)

        result = det.detect(
            bar=bar,
            zones=[zone1, zone2],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=context,
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None
        # zone2 should win because it has more confluences (balance_duration + volume_contraction)
        assert result.balance_zone is zone2

    def test_volume_surge_confluence(self):
        """volume >= surge_mult x volume_ma adds 'volume_surge' confluence."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            volume_surge_mult=1.3,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(108, volume=200.0, high_offset=3, low_offset=3)
        ind = make_snapshot(volume_ma=100.0)  # 200/100 = 2.0 >= 1.3

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=0.0),
            context=make_context(),
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None
        assert "volume_surge" in result.confluences

    def test_require_volume_surge_blocks_low_volume_breakout(self):
        """When enabled, require_volume_surge blocks setups below the surge threshold."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            require_volume_surge=True,
            volume_surge_mult=1.3,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(108, volume=110.0, high_offset=3, low_offset=3)
        ind = make_snapshot(volume_ma=100.0)  # 1.1x < 1.3x threshold

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(direction=Side.LONG, strength="strong"),
            m30_ind=ind,
            atr=10.0,
        )
        assert result is None

    def test_require_volume_surge_allows_high_volume_breakout(self):
        """When enabled, require_volume_surge still allows strong-volume setups."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_a=0.0,
            min_room_r_b=0.0,
            require_volume_surge=True,
            volume_surge_mult=1.3,
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(108, volume=140.0, high_offset=3, low_offset=3)
        ind = make_snapshot(volume_ma=100.0)  # 1.4x >= 1.3x threshold

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(direction=Side.LONG, strength="strong"),
            m30_ind=ind,
            atr=10.0,
        )
        assert result is not None

    def test_relaxed_body_branch_allows_configured_symbol_direction(self):
        """A lower-body breakout can pass via the supplemental branch with tighter gates."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.55,
            relaxed_body_enabled=True,
            relaxed_body_min=0.30,
            relaxed_body_min_confluences=0,
            relaxed_body_min_room_r=0.0,
            relaxed_body_require_volume_surge=False,
            relaxed_body_risk_scale=0.4,
        )
        det = BreakoutDetector(
            cfg,
            BreakoutSymbolFilterParams(btc_relaxed_body_direction="both"),
        )
        zone = make_zone(center=100, upper=105, lower=95)
        bar = Bar(
            timestamp=_TS,
            symbol="BTC",
            open=107.0,
            high=109.0,
            low=106.0,
            close=108.0,
            volume=100.0,
            timeframe=TimeFrame.M30,
        )

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(),
            m30_ind=make_snapshot(volume_ma=100.0),
            atr=10.0,
        )
        assert result is not None
        assert result.grade == SetupGrade.B
        assert result.signal_variant == "relaxed_body"
        assert result.risk_scale == pytest.approx(0.4)

    def test_relaxed_body_branch_blocks_disallowed_direction(self):
        """The supplemental branch only applies to explicitly allowed symbol-directions."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.55,
            relaxed_body_enabled=True,
            relaxed_body_min=0.30,
            relaxed_body_min_confluences=0,
            relaxed_body_min_room_r=0.0,
            relaxed_body_require_volume_surge=False,
        )
        det = BreakoutDetector(
            cfg,
            BreakoutSymbolFilterParams(eth_relaxed_body_direction="long_only"),
        )
        zone = make_zone(center=100, upper=105, lower=95)
        bar = Bar(
            timestamp=_TS,
            symbol="ETH",
            open=93.0,
            high=94.0,
            low=91.0,
            close=92.0,
            volume=100.0,
            timeframe=TimeFrame.M30,
        )

        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=5.0),
            context=make_context(),
            m30_ind=make_snapshot(volume_ma=100.0),
            atr=10.0,
        )
        assert result is None

    def test_room_r_filter(self):
        """Insufficient room_r filters out the setup."""
        cfg = BreakoutSetupParams(
            min_breakout_atr=0.1,
            max_breakout_atr=5.0,
            body_ratio_min=0.05,
            min_confluences_b=0,
            min_room_r_b=5.0,  # Very high requirement
        )
        det = BreakoutDetector(cfg)
        zone = make_zone(center=100, upper=105, lower=95)
        bar = make_bar_at(108, high_offset=3, low_offset=3)
        # lvn_runway=0.5 => room_r = 0.5*10 / (10+3) ~ 0.38 < 5.0
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(lvn_runway=0.5),
            context=make_context(),
            m30_ind=None,
            atr=10.0,
        )
        assert result is None

    def test_zero_atr_returns_none(self):
        """atr <= 0 returns None immediately."""
        det = BreakoutDetector(BreakoutSetupParams())
        zone = make_zone()
        bar = make_bar_at(110)
        result = det.detect(
            bar=bar,
            zones=[zone],
            profile=None,
            profiler=_mock_profiler(),
            context=make_context(),
            m30_ind=None,
            atr=0.0,
        )
        assert result is None
