"""Tests for BalanceDetector and BalanceZone."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.strategy.breakout.balance import BalanceDetector, BalanceZone
from crypto_trader.strategy.breakout.config import BalanceParams
from crypto_trader.strategy.breakout.profile import VolumeProfileResult


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


def make_profile(
    poc: float = 100.0,
    vah: float = 105.0,
    val: float = 95.0,
    hvn_levels: tuple[float, ...] = (),
    lvn_levels: tuple[float, ...] = (),
) -> VolumeProfileResult:
    """Create a VolumeProfileResult with reasonable defaults."""
    return VolumeProfileResult(
        poc=poc,
        vah=vah,
        val=val,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels,
        bin_edges=tuple(float(x) for x in range(80, 122, 2)),  # 21 edges -> 20 bins
        bin_volumes=tuple(50.0 for _ in range(20)),
        total_volume=1000.0,
        price_low=80.0,
        price_high=120.0,
    )


def make_balance_bars(
    center: float,
    width: float,
    n_bars: int,
    volume: float = 100.0,
) -> list[Bar]:
    """Create bars oscillating around *center* within *width*.

    Bars alternate between slightly above and slightly below center,
    with occasional excursions outside the zone to produce touches.
    """
    bars: list[Bar] = []
    half_w = width / 2.0
    for i in range(n_bars):
        if i % 5 == 0:
            # Excursion outside to create a "touch" on re-entry
            o = center + half_w * 1.5
            c = center + half_w * 0.3
            h = center + half_w * 1.6
            l = center - half_w * 0.1
        elif i % 5 == 1:
            # Re-enter zone from outside
            o = center + half_w * 0.3
            c = center - half_w * 0.2
            h = center + half_w * 0.4
            l = center - half_w * 0.3
        else:
            offset = half_w * 0.3 * (1 if i % 2 == 0 else -1)
            o = center + offset
            c = center - offset
            h = center + half_w * 0.5
            l = center - half_w * 0.5
        bars.append(_bar(i, o, h, l, c, volume))
    return bars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBalanceDetector:
    """Tests for BalanceDetector."""

    def test_no_zones_without_hvn(self):
        """Empty hvn_levels creates no zones."""
        det = BalanceDetector(BalanceParams())
        profile = make_profile(hvn_levels=())
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        assert det.get_active_zones("BTC") == []

    def test_zone_detected_basic(self):
        """Bars oscillating around an HVN level creates a zone."""
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) >= 1

    def test_zone_center_matches_hvn(self):
        """Zone center equals the HVN level."""
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) >= 1
        assert zones[0].center == 100.0

    def test_zone_width_matches_config(self):
        """Zone width = zone_width_atr * atr."""
        atr = 5.0
        zone_width_atr = 2.0
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=zone_width_atr)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, zone_width_atr * atr, 20)
        det.update("BTC", bars, profile, atr=atr, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) >= 1
        expected_width = zone_width_atr * atr
        actual_width = zones[0].upper - zones[0].lower
        assert actual_width == pytest.approx(expected_width)

    def test_min_bars_filter(self):
        """Zone not created if bars_in_zone < min_bars_in_zone."""
        cfg = BalanceParams(min_bars_in_zone=100, min_touches=1, zone_width_atr=1.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 5.0, 10)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        assert det.get_active_zones("BTC") == []

    def test_min_touches_filter(self):
        """Zone not created if touches < min_touches."""
        cfg = BalanceParams(min_bars_in_zone=1, min_touches=100, zone_width_atr=2.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        assert det.get_active_zones("BTC") == []

    def test_zone_expiry(self):
        """Zone removed after max_zone_age_bars."""
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0,
                            max_zone_age_bars=10, dedup_atr_frac=0.3)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)

        # Create zone at bar_index=0
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        assert len(det.get_active_zones("BTC")) >= 1

        # Update at bar_index=11 with a far-away HVN so update() doesn't
        # early-return (hvn_levels must be non-empty), but the old zone
        # should be expired because 11 - 0 > max_zone_age_bars(10).
        # The new HVN at 200.0 won't produce a zone because no bars are near it.
        profile_far = make_profile(hvn_levels=(200.0,))
        det.update("BTC", bars, profile_far, atr=5.0, bar_index=11)
        assert len(det.get_active_zones("BTC")) == 0

    def test_dedup_nearby_zones(self):
        """Two HVNs within dedup distance create only one zone."""
        atr = 10.0
        dedup_frac = 0.3
        # Two HVNs 2.0 apart -- within 0.3 * 10 = 3.0
        cfg = BalanceParams(
            min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0,
            dedup_atr_frac=dedup_frac,
        )
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0, 102.0))
        bars = make_balance_bars(101.0, 20.0, 20)
        det.update("BTC", bars, profile, atr=atr, bar_index=0)
        zones = det.get_active_zones("BTC")
        # Second HVN should be deduped because |102 - 100| < 3.0
        assert len(zones) == 1

    def test_consume_zone(self):
        """consume_zone removes the specific zone."""
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) >= 1
        det.consume_zone("BTC", zones[0])
        assert len(det.get_active_zones("BTC")) == 0

    def test_clear_zones(self):
        """clear removes all zones for symbol."""
        cfg = BalanceParams(min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0)
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        bars = make_balance_bars(100.0, 10.0, 20)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        assert len(det.get_active_zones("BTC")) >= 1
        det.clear("BTC")
        assert det.get_active_zones("BTC") == []

    def test_volume_contraction_check(self):
        """volume_contracting set correctly when volume declines."""
        cfg = BalanceParams(
            min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0,
            require_volume_contraction=False, contraction_threshold=0.8,
        )
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        # First half high volume, second half low -- contraction
        bars: list[Bar] = []
        for i in range(20):
            vol = 200.0 if i < 10 else 50.0
            offset = 2.0 * (1 if i % 2 == 0 else -1)
            # Keep bars oscillating in zone and create touches
            if i % 5 == 0:
                o, c = 100.0 + 6.0, 100.0 + 1.0
                h, l = 100.0 + 7.0, 100.0
            else:
                o = 100.0 + offset
                c = 100.0 - offset
                h = 100.0 + 3.0
                l = 100.0 - 3.0
            bars.append(_bar(i, o, h, l, c, vol))
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) >= 1
        assert zones[0].volume_contracting is True

    def test_volume_contraction_required(self):
        """Zone rejected when contraction required but not present."""
        cfg = BalanceParams(
            min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0,
            require_volume_contraction=True, contraction_threshold=0.8,
        )
        det = BalanceDetector(cfg)
        profile = make_profile(hvn_levels=(100.0,))
        # Constant volume -- no contraction
        bars = make_balance_bars(100.0, 10.0, 20, volume=100.0)
        det.update("BTC", bars, profile, atr=5.0, bar_index=0)
        # Volume is constant, late_vol / early_vol ~ 1.0 which is NOT < 0.8
        # so contraction check fails when required
        assert det.get_active_zones("BTC") == []

    def test_get_active_zones_empty(self):
        """Returns empty list for unknown symbol."""
        det = BalanceDetector(BalanceParams())
        assert det.get_active_zones("UNKNOWN") == []

    def test_multiple_zones(self):
        """Multiple HVNs far apart create multiple zones."""
        atr = 5.0
        cfg = BalanceParams(
            min_bars_in_zone=3, min_touches=1, zone_width_atr=2.0,
            dedup_atr_frac=0.3,
        )
        det = BalanceDetector(cfg)
        # Two HVNs 20 apart -- well beyond dedup threshold of 0.3*5 = 1.5
        profile = make_profile(hvn_levels=(90.0, 110.0))
        # Bars that span the full range so both zones have enough bars
        bars: list[Bar] = []
        for i in range(20):
            # Alternate between oscillating around 90 and 110
            if i % 2 == 0:
                center = 90.0
            else:
                center = 110.0
            offset = 3.0 * (1 if i % 3 == 0 else -1)
            o = center + offset
            c = center - offset
            h = center + 4.0
            l = center - 4.0
            bars.append(_bar(i, o, h, l, c, 100.0))
        det.update("BTC", bars, profile, atr=atr, bar_index=0)
        zones = det.get_active_zones("BTC")
        assert len(zones) == 2

    def test_zone_is_frozen(self):
        """BalanceZone is frozen dataclass."""
        zone = BalanceZone(
            center=100.0, upper=105.0, lower=95.0,
            bars_in_zone=10, touches=3,
            formation_bar_idx=0, volume_contracting=False, width_atr=1.2,
        )
        with pytest.raises(AttributeError):
            zone.center = 200.0  # type: ignore[misc]
