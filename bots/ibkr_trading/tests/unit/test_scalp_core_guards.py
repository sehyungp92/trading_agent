from __future__ import annotations

from datetime import datetime, timezone

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.session import ScalpSessionBlock
from strategies.scalp.ivb_auction.config import EntryTrigger, IvbModule, TradeDirection as IvbDirection
from strategies.scalp.ivb_auction.core.logic import on_bar as ivb_on_bar
from strategies.scalp.ivb_auction.core.state import IvbAuctionCoreState, IvbBarInput
from strategies.scalp.po3_reversal.config import SetupTier, TradeDirection as Po3Direction
from strategies.scalp.po3_reversal.core.logic import on_bar as po3_on_bar
from strategies.scalp.po3_reversal.core.state import Po3BarInput, Po3ReversalCoreState
from strategies.scalp.po3_reversal.models import Po3Context


def test_ivb_core_does_not_stack_entry_orders() -> None:
    ts = datetime(2026, 4, 29, 14, 5, tzinfo=timezone.utc)
    payload = IvbBarInput(
        symbol="NQ",
        bar_ts=ts,
        bar_ohlcv=(100.0, 106.0, 99.0, 105.0, 1000.0),
        session_block=ScalpSessionBlock.RTH_PRIME,
        ivb_levels=IVBLevels.from_bounds(104.0, 80.0, poc=95.0, vah=101.0, val=85.0),
        breakout_direction=IvbDirection.LONG,
        breakout_accepted=True,
        module=IvbModule.A1_CONTINUATION,
        trigger=EntryTrigger.PROFILE_RELOAD,
        entry_price=101.0,
        stop_price=84.75,
        tp1_price=130.0,
        qty=1,
        rr_to_tp1=1.8,
        signal_score=80.0,
        size_multiplier=0.75,
    )

    state, actions, _ = ivb_on_bar(IvbAuctionCoreState(), payload)
    state, duplicate_actions, _ = ivb_on_bar(state, payload)

    assert len(actions) == 1
    assert duplicate_actions == []


def test_po3_core_does_not_replace_working_setup_with_duplicate_signal() -> None:
    ts = datetime(2026, 4, 29, 14, 5, tzinfo=timezone.utc)
    payload = Po3BarInput(
        symbol="NQ",
        bar_ts=ts,
        bar_ohlcv=(100.0, 102.0, 99.0, 101.0, 1000.0),
        context=Po3Context(),
        tier=SetupTier.A,
        direction=Po3Direction.LONG,
        signal_score=5.5,
        signal_threshold=5.0,
        entry_price=102.25,
        stop_price=98.75,
        target_price=108.0,
        qty=1,
        rr=1.6,
        risk_approved=True,
    )

    state, actions, _ = po3_on_bar(Po3ReversalCoreState(), payload)
    original_setup_id = state.active_setup.setup_id if state.active_setup else ""
    state, duplicate_actions, _ = po3_on_bar(state, payload)

    assert len(actions) == 1
    assert duplicate_actions == []
    assert state.active_setup is not None
    assert state.active_setup.setup_id == original_setup_id
