"""Tests for VolumeProfiler and VolumeProfileResult."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.models import Bar, Side, TimeFrame
from crypto_trader.strategy.breakout.config import ProfileParams
from crypto_trader.strategy.breakout.profile import VolumeProfiler, VolumeProfileResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(i: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * i)


def _bar(
    i: int,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=_ts(i),
        symbol="BTC",
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        timeframe=TimeFrame.M30,
    )


def _range_bars(n: int, base: float = 100.0, spread: float = 10.0, volume: float = 100.0) -> list[Bar]:
    """Create *n* bars oscillating around *base* within +-spread."""
    bars: list[Bar] = []
    for i in range(n):
        offset = spread * (0.5 if i % 2 == 0 else -0.5)
        o = base + offset
        c = base - offset
        h = max(o, c) + spread * 0.1
        l = min(o, c) - spread * 0.1
        bars.append(_bar(i, o, h, l, c, volume))
    return bars


def _clustered_bars(
    n: int,
    cluster_center: float,
    cluster_spread: float,
    outer_low: float,
    outer_high: float,
    cluster_volume: float = 500.0,
    outer_volume: float = 20.0,
) -> list[Bar]:
    """Half bars tightly clustered (high volume), half spread widely (low vol)."""
    bars: list[Bar] = []
    half = n // 2
    for i in range(half):
        o = cluster_center - cluster_spread * 0.3
        c = cluster_center + cluster_spread * 0.3
        h = cluster_center + cluster_spread * 0.5
        l = cluster_center - cluster_spread * 0.5
        bars.append(_bar(i, o, h, l, c, cluster_volume))
    for i in range(half, n):
        o = outer_low + 1
        c = outer_high - 1
        h = outer_high
        l = outer_low
        bars.append(_bar(i, o, h, l, c, outer_volume))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVolumeProfiler:
    """Tests for VolumeProfiler.build()."""

    def test_build_returns_none_insufficient_bars(self):
        cfg = ProfileParams(min_bars=20)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(10)  # fewer than 20
        assert profiler.build(bars) is None

    def test_build_basic_profile(self):
        cfg = ProfileParams(min_bars=5, num_bins=50)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25, base=100.0, spread=10.0)
        result = profiler.build(bars)

        assert result is not None
        assert result.price_low <= result.val <= result.poc <= result.vah <= result.price_high

    def test_poc_is_highest_volume_bin(self):
        cfg = ProfileParams(min_bars=5, num_bins=50)
        profiler = VolumeProfiler(cfg)
        # Cluster most volume around 100
        bars = _clustered_bars(30, cluster_center=100.0, cluster_spread=2.0,
                               outer_low=80.0, outer_high=120.0)
        result = profiler.build(bars)
        assert result is not None

        # POC should be the midpoint of the highest-volume bin
        max_vol_idx = max(range(len(result.bin_volumes)), key=lambda i: result.bin_volumes[i])
        expected_poc = (result.bin_edges[max_vol_idx] + result.bin_edges[max_vol_idx + 1]) / 2.0
        assert result.poc == pytest.approx(expected_poc)

    def test_value_area_contains_70pct(self):
        cfg = ProfileParams(min_bars=5, num_bins=50, value_area_pct=0.70)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(30, base=100.0, spread=10.0)
        result = profiler.build(bars)
        assert result is not None

        # Sum volume of bins fully inside [VAL, VAH]
        inside_vol = 0.0
        for i in range(len(result.bin_volumes)):
            bin_low = result.bin_edges[i]
            bin_high = result.bin_edges[i + 1]
            if bin_low >= result.val and bin_high <= result.vah:
                inside_vol += result.bin_volumes[i]

        # Should be at least 70% (may exceed due to discrete bin expansion)
        assert inside_vol / result.total_volume >= 0.70 - 0.05  # small tolerance

    def test_hvn_above_threshold(self):
        cfg = ProfileParams(min_bars=5, num_bins=50, hvn_threshold_pct=1.5)
        profiler = VolumeProfiler(cfg)
        bars = _clustered_bars(30, cluster_center=100.0, cluster_spread=2.0,
                               outer_low=80.0, outer_high=120.0)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.hvn_levels) > 0

        # Mean of non-zero bins
        non_zero = [v for v in result.bin_volumes if v > 0]
        mean_vol = sum(non_zero) / len(non_zero)
        # Every HVN bin must have volume >= threshold * mean
        for hvn in result.hvn_levels:
            for i in range(len(result.bin_volumes)):
                mid = (result.bin_edges[i] + result.bin_edges[i + 1]) / 2.0
                if abs(mid - hvn) < 1e-9:
                    assert result.bin_volumes[i] >= cfg.hvn_threshold_pct * mean_vol

    def test_lvn_below_threshold(self):
        cfg = ProfileParams(min_bars=5, num_bins=50, lvn_threshold_pct=0.5)
        profiler = VolumeProfiler(cfg)
        bars = _clustered_bars(30, cluster_center=100.0, cluster_spread=2.0,
                               outer_low=80.0, outer_high=120.0)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.lvn_levels) > 0

        non_zero = [v for v in result.bin_volumes if v > 0]
        mean_vol = sum(non_zero) / len(non_zero)
        for lvn in result.lvn_levels:
            for i in range(len(result.bin_volumes)):
                mid = (result.bin_edges[i] + result.bin_edges[i + 1]) / 2.0
                if abs(mid - lvn) < 1e-9:
                    assert 0 < result.bin_volumes[i] <= cfg.lvn_threshold_pct * mean_vol

    def test_doji_bar_handling(self):
        """Bar with high == low assigns volume to close bin."""
        cfg = ProfileParams(min_bars=2, num_bins=10)
        profiler = VolumeProfiler(cfg)
        # One normal bar to set price range, one doji
        bars = [
            _bar(0, 90.0, 110.0, 90.0, 100.0, 50.0),
            _bar(1, 100.0, 100.0, 100.0, 100.0, 200.0),  # doji
        ]
        result = profiler.build(bars)
        assert result is not None
        assert result.total_volume == pytest.approx(250.0, abs=1.0)

    def test_single_price_guard(self):
        """All bars at same price -- price_high adjusted to avoid zero range."""
        cfg = ProfileParams(min_bars=2, num_bins=10)
        profiler = VolumeProfiler(cfg)
        bars = [_bar(i, 100.0, 100.0, 100.0, 100.0, 50.0) for i in range(5)]
        result = profiler.build(bars)
        assert result is not None
        # price_high should be adjusted to price_low + 1.0
        assert result.price_high == pytest.approx(result.price_low + 1.0)

    def test_bin_edges_correct_count(self):
        num_bins = 40
        cfg = ProfileParams(min_bars=5, num_bins=num_bins)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(20)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.bin_edges) == num_bins + 1

    def test_total_volume_matches_sum(self):
        cfg = ProfileParams(min_bars=5, num_bins=50)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25)
        result = profiler.build(bars)
        assert result is not None
        assert result.total_volume == pytest.approx(sum(result.bin_volumes), rel=1e-6)

    def test_profile_different_num_bins_25(self):
        cfg = ProfileParams(min_bars=5, num_bins=25)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.bin_volumes) == 25

    def test_profile_different_num_bins_100(self):
        cfg = ProfileParams(min_bars=5, num_bins=100)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.bin_volumes) == 100


class TestFindLvnRunway:
    """Tests for VolumeProfiler.find_lvn_runway()."""

    def _build_profile_with_lvn(self) -> tuple[VolumeProfiler, VolumeProfileResult]:
        """Build a profile with known LVN region above cluster."""
        cfg = ProfileParams(min_bars=5, num_bins=50, lvn_threshold_pct=0.5)
        profiler = VolumeProfiler(cfg)
        # Heavy cluster at 100, sparse at 110-120
        bars = _clustered_bars(30, cluster_center=100.0, cluster_spread=2.0,
                               outer_low=80.0, outer_high=120.0,
                               cluster_volume=500.0, outer_volume=5.0)
        result = profiler.build(bars)
        assert result is not None
        return profiler, result

    def test_find_lvn_runway_long(self):
        profiler, profile = self._build_profile_with_lvn()
        atr = 5.0
        runway = profiler.find_lvn_runway(profile, 105.0, Side.LONG, atr)
        assert runway > 0.0

    def test_find_lvn_runway_short(self):
        profiler, profile = self._build_profile_with_lvn()
        atr = 5.0
        runway = profiler.find_lvn_runway(profile, 95.0, Side.SHORT, atr)
        assert runway > 0.0

    def test_find_lvn_runway_zero_atr(self):
        profiler, profile = self._build_profile_with_lvn()
        assert profiler.find_lvn_runway(profile, 100.0, Side.LONG, 0.0) == 0.0
        assert profiler.find_lvn_runway(profile, 100.0, Side.LONG, -1.0) == 0.0

    def test_value_area_expands_from_poc(self):
        cfg = ProfileParams(min_bars=5, num_bins=50)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25)
        result = profiler.build(bars)
        assert result is not None
        assert result.val <= result.poc <= result.vah

    def test_multiple_hvn_levels(self):
        """Clustered volume should create multiple adjacent HVN bins."""
        cfg = ProfileParams(min_bars=5, num_bins=50, hvn_threshold_pct=1.5)
        profiler = VolumeProfiler(cfg)
        bars = _clustered_bars(30, cluster_center=100.0, cluster_spread=4.0,
                               outer_low=80.0, outer_high=120.0,
                               cluster_volume=1000.0, outer_volume=5.0)
        result = profiler.build(bars)
        assert result is not None
        assert len(result.hvn_levels) >= 2

    def test_uniform_volume_no_hvn(self):
        """All bars with same range and volume -- no extreme HVNs expected."""
        cfg = ProfileParams(min_bars=5, num_bins=50, hvn_threshold_pct=1.5)
        profiler = VolumeProfiler(cfg)
        # Every bar identical range and volume
        bars = [_bar(i, 95.0, 105.0, 95.0, 100.0, 100.0) for i in range(30)]
        result = profiler.build(bars)
        assert result is not None
        # With perfectly uniform distribution, no bin should exceed 1.5x mean
        # (all bins in range should have equal volume)
        # There may be 0 HVNs or very few due to edge effects
        assert len(result.hvn_levels) <= 2

    def test_profile_result_is_frozen(self):
        cfg = ProfileParams(min_bars=5, num_bins=50)
        profiler = VolumeProfiler(cfg)
        bars = _range_bars(25)
        result = profiler.build(bars)
        assert result is not None
        with pytest.raises(AttributeError):
            result.poc = 999.0  # type: ignore[misc]

    def test_find_lvn_runway_no_lvn(self):
        """When all bins are above LVN threshold, runway should be 0."""
        cfg = ProfileParams(min_bars=5, num_bins=10, lvn_threshold_pct=0.5)
        profiler = VolumeProfiler(cfg)
        # Uniform high-volume bars -- every bin gets equal volume, all above threshold
        bars = [_bar(i, 95.0, 105.0, 95.0, 100.0, 100.0) for i in range(30)]
        result = profiler.build(bars)
        assert result is not None
        # All bins within the bar range should have similar (above-threshold) volume
        # Starting from inside the range, runway should be 0 (first bin is not LVN)
        runway = profiler.find_lvn_runway(result, 100.0, Side.LONG, 5.0)
        assert runway == pytest.approx(0.0)
