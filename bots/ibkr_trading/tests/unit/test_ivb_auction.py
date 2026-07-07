"""IVB auction footprint and signal scoring tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp.ivb_auction.config import TradeDirection
from strategies.scalp.ivb_auction.footprint import FootprintBuilder
from strategies.scalp.ivb_auction.models import ScalpTick
from strategies.scalp.ivb_auction.signals import score_signal
from strategies.scalp.ivb_auction.target_model import reclaim_targets


def test_footprint_classifies_bid_ask_aggression() -> None:
    builder = FootprintBuilder(bar_seconds=30)
    ts = datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc)
    builder.on_tick(ScalpTick(ts=ts, price=100.25, size=3, bid=100.0, ask=100.25))
    builder.on_tick(ScalpTick(ts=ts + timedelta(seconds=1), price=100.0, size=2, bid=100.0, ask=100.25))
    bar = builder.flush()

    assert bar is not None
    assert bar.ask_volume == 3
    assert bar.bid_volume == 2
    assert bar.delta == 1


def test_ivb_signal_renormalizes_when_footprint_missing() -> None:
    score = score_signal(
        regime_quality=1.0,
        retest_quality=1.0,
        target_quality=1.0,
        volatility_quality=1.0,
        time_quality=1.0,
        absorption_quality=None,
        delta_confirmation=None,
    )

    assert score.total == 100.0
    assert not score.footprint_available
    assert "absorption" not in score.available_components


def test_ivb_reclaim_targets_rotate_back_through_auction_value() -> None:
    ivb = IVBLevels.from_bounds(120.0, 80.0, poc=100.0, vah=110.0, val=90.0)

    failed_upside = reclaim_targets(entry_price=110.0, direction=TradeDirection.SHORT, ivb=ivb)
    failed_downside = reclaim_targets(entry_price=90.0, direction=TradeDirection.LONG, ivb=ivb)

    assert failed_upside.tp1 == 100.0
    assert failed_upside.tp2 == 80.0
    assert failed_downside.tp1 == 100.0
    assert failed_downside.tp2 == 120.0
