from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from strategies.scalp.ivb_auction.footprint import FootprintBuilder
from strategies.scalp.ivb_auction.models import FootprintBarData, ScalpTick

from .preprocessing import NumpyTicks


class TickReplayMode(Enum):
    TICK_LEVEL = "tick_level"
    BAR_WITH_FOOTPRINT = "bar_with_footprint"
    BAR_ONLY = "bar_only"


@dataclass
class FootprintReplayResult:
    bars: list[FootprintBarData]


def build_footprint_bars(ticks: NumpyTicks | None, *, bar_seconds: int = 30) -> FootprintReplayResult:
    if ticks is None:
        return FootprintReplayResult([])
    builder = FootprintBuilder(bar_seconds=bar_seconds)
    bars: list[FootprintBarData] = []
    for idx, raw_ts in enumerate(ticks.timestamps):
        tick = ScalpTick(
            ts=_np_to_datetime(raw_ts),
            price=float(ticks.prices[idx]),
            size=float(ticks.sizes[idx]),
            bid=float(ticks.bid_prices[idx]) if len(ticks.bid_prices) else None,
            ask=float(ticks.ask_prices[idx]) if len(ticks.ask_prices) else None,
            side=int(ticks.sides[idx]) if len(ticks.sides) else 0,
        )
        completed = builder.on_tick(tick)
        if completed is not None:
            bars.append(completed)
    tail = builder.flush()
    if tail is not None:
        bars.append(tail)
    return FootprintReplayResult(bars)


def _np_to_datetime(value):
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()

