from __future__ import annotations

from datetime import timedelta

from .models import FootprintBarData, ScalpTick


class FootprintBuilder:
    def __init__(self, *, bar_seconds: int = 30) -> None:
        self.bar_seconds = bar_seconds
        self._ticks: list[ScalpTick] = []

    def on_tick(self, tick: ScalpTick) -> FootprintBarData | None:
        if self._ticks and tick.ts >= self._ticks[0].ts + timedelta(seconds=self.bar_seconds):
            completed = self.flush()
            self._ticks.append(tick)
            return completed
        self._ticks.append(tick)
        return None

    def flush(self) -> FootprintBarData | None:
        if not self._ticks:
            return None
        ticks = self._ticks
        self._ticks = []
        bid_volume = 0.0
        ask_volume = 0.0
        for tick in ticks:
            if tick.ask is not None and tick.price >= tick.ask:
                ask_volume += tick.size
            elif tick.bid is not None and tick.price <= tick.bid:
                bid_volume += tick.size
            elif tick.side > 0:
                ask_volume += tick.size
            elif tick.side < 0:
                bid_volume += tick.size
        prices = [tick.price for tick in ticks]
        return FootprintBarData(
            start_ts=ticks[0].ts,
            end_ts=ticks[-1].ts,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(tick.size for tick in ticks),
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        )
