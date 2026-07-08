from __future__ import annotations

from datetime import datetime, timedelta, timezone

from strategies.scalp.po3_reversal.config import TradeDirection
from strategies.scalp.po3_reversal.fvg import FvgState, FvgStateMachine
from strategies.scalp.po3_reversal.models import PriceBar


def _bar(i: int, o: float, h: float, l: float, c: float) -> PriceBar:
    return PriceBar(datetime(2026, 4, 29, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=i), o, h, l, c)


def test_bearish_fvg_inverts_to_bullish_ifvg_on_close_through() -> None:
    machine = FvgStateMachine()
    machine.update(_bar(0, 10, 11, 9, 10))
    machine.update(_bar(1, 10, 10.5, 9.5, 10))
    machine.update(_bar(2, 12, 13, 12, 12.5))
    gap = machine.gaps[-1]
    assert gap.direction is TradeDirection.LONG

    machine.update(_bar(3, 10, 10.5, 9.5, 9.75))

    assert gap.direction is TradeDirection.SHORT
    assert gap.state is FvgState.INVERTED

