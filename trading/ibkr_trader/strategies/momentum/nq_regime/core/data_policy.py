from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from strategies.momentum.nq_regime.core.state import BarData, BarEvent


@dataclass(frozen=True, slots=True)
class CompletedBarPolicy:
    """Completed-bar gate for the NQ regime event clock.

    The core consumes completed 5m bars. A 15m bar is attached only at the
    15-minute boundary, with its timestamp equal to the close/availability time.
    """

    base_minutes: int = 5
    context_minutes: int = 15

    def build_event(
        self,
        *,
        bar_5m: BarData,
        recent_5m: list[BarData],
        daily_context=None,
        live_context: dict | None = None,
    ) -> BarEvent:
        if not self.is_completed_bar_timestamp(bar_5m.ts, self.base_minutes):
            raise ValueError("5m bar timestamp must be the completed bar close time")
        bar_15m = None
        is_new = False
        if self.is_completed_bar_timestamp(bar_5m.ts, self.context_minutes):
            sample = [bar for bar in recent_5m[-3:] if bar.ts <= bar_5m.ts]
            if len(sample) == 3 and sample[-1].ts == bar_5m.ts:
                bar_15m = aggregate_bars(sample, ts=bar_5m.ts)
                is_new = True
        return BarEvent(
            ts=bar_5m.ts,
            bar_5m=bar_5m,
            bar_15m_closed=bar_15m,
            is_new_15m=is_new,
            daily_context=daily_context,
            live_context=dict(live_context or {}),
        )

    @staticmethod
    def is_completed_bar_timestamp(ts: datetime, minutes: int) -> bool:
        return ts.second == 0 and ts.microsecond == 0 and (ts.minute % minutes) == 0


def aggregate_bars(bars: list[BarData], *, ts: datetime) -> BarData:
    if not bars:
        raise ValueError("cannot aggregate empty bars")
    total_volume = sum(bar.volume for bar in bars)
    vwap = None
    if total_volume > 0:
        vwap = sum(((bar.high + bar.low + bar.close) / 3.0) * bar.volume for bar in bars) / total_volume
    return BarData(
        ts=ts,
        open=bars[0].open,
        high=max(bar.high for bar in bars),
        low=min(bar.low for bar in bars),
        close=bars[-1].close,
        volume=total_volume,
        vwap=vwap,
    )

