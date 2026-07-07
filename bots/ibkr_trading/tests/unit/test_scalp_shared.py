"""Tests for strategies.scalp._shared (session, volume profile, fractal levels)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from strategies.scalp._shared.levels import detect_fractal_pivots
from strategies.scalp._shared.session import (
    ScalpSessionBlock,
    entries_allowed,
    get_session_block,
    must_flatten,
)
from strategies.scalp._shared.volume_profile import compute_volume_profile


ET = ZoneInfo("America/New_York")


def test_scalp_session_blocks_and_entry_permissions() -> None:
    assert get_session_block(datetime(2026, 4, 29, 9, 0, tzinfo=ET)) is ScalpSessionBlock.PRE_MARKET
    assert get_session_block(datetime(2026, 4, 29, 9, 45, tzinfo=ET)) is ScalpSessionBlock.IVB_FORMING
    assert get_session_block(datetime(2026, 4, 29, 10, 15, tzinfo=ET)) is ScalpSessionBlock.RTH_PRIME
    assert entries_allowed(ScalpSessionBlock.IVB_FORMING, "po3_reversal")
    assert not entries_allowed(ScalpSessionBlock.IVB_FORMING, "ivb_auction")
    assert must_flatten(datetime(2026, 4, 29, 15, 55, tzinfo=ET))


def test_volume_profile_expands_value_area_from_poc() -> None:
    result = compute_volume_profile(
        [100.0, 100.0, 100.25, 100.5, 100.75],
        [10, 10, 5, 4, 1],
        tick_size=0.25,
        value_area_pct=0.70,
    )

    assert result.poc == 100.0
    assert result.val == 100.0
    assert result.vah == 100.25
    assert result.total_volume == 30


def test_fractal_pivots_expose_confirmation_index() -> None:
    highs, lows = detect_fractal_pivots(
        highs=[1, 2, 5, 2, 1, 3, 2],
        lows=[5, 4, 3, 4, 1, 4, 5],
        left_n=2,
        right_n=2,
    )

    assert highs[0].index == 2
    assert highs[0].confirmed_at_index == 4
    assert lows[0].index == 4
    assert lows[0].confirmed_at_index == 6
