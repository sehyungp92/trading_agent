"""MissedOpportunityBackfiller — fills in outcome fields after the fact."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from crypto_trader.instrumentation.types import MissedOpportunityEvent

if TYPE_CHECKING:
    from crypto_trader.core.models import Bar


class MissedOpportunityBackfiller:
    """Fills outcome_1h/4h/24h fields for missed opportunity events.

    Backtest mode: post-backtest scan of price data.
    Live mode: periodic check of elapsed time since missed signal.
    """

    @staticmethod
    def backfill_from_bars(
        events: list[MissedOpportunityEvent],
        bars_by_symbol: dict[str, list[Bar]],
    ) -> None:
        """Backfill outcomes using historical bar data (backtest mode).

        Args:
            events: Missed opportunity events to backfill.
            bars_by_symbol: dict of symbol -> sorted list of bars (ascending time).
        """
        for event in events:
            if event.backfill_status == "complete":
                continue

            sym_bars = bars_by_symbol.get(event.pair, [])
            if not sym_bars:
                continue

            entry_price = event.hypothetical_entry
            if entry_price <= 0:
                continue

            signal_time = event.metadata.exchange_timestamp

            # Find bars at 1h, 4h, 24h after signal
            t_1h = signal_time + timedelta(hours=1)
            t_4h = signal_time + timedelta(hours=4)
            t_24h = signal_time + timedelta(hours=24)

            bar_1h = _find_nearest_bar(sym_bars, t_1h)
            bar_4h = _find_nearest_bar(sym_bars, t_4h)
            bar_24h = _find_nearest_bar(sym_bars, t_24h)

            if bar_1h is not None:
                event.outcome_1h = (bar_1h.close - entry_price) / entry_price * 100
            if bar_4h is not None:
                event.outcome_4h = (bar_4h.close - entry_price) / entry_price * 100
            if bar_24h is not None:
                event.outcome_24h = (bar_24h.close - entry_price) / entry_price * 100

            # Determine backfill completeness
            if event.outcome_24h is not None:
                event.backfill_status = "complete"
            elif event.outcome_1h is not None or event.outcome_4h is not None:
                event.backfill_status = "partial"


def _find_nearest_bar(bars: list[Bar], target_time) -> Bar | None:
    """Find the bar closest to target_time (binary search approach)."""
    if not bars:
        return None

    best = None
    best_delta = None

    lo, hi = 0, len(bars) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        bar_ts = bars[mid].timestamp
        delta = abs((bar_ts - target_time).total_seconds())

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = bars[mid]

        if bar_ts < target_time:
            lo = mid + 1
        elif bar_ts > target_time:
            hi = mid - 1
        else:
            return bars[mid]

    # Only return if within 2 hours tolerance
    if best_delta is not None and best_delta <= 7200:
        return best
    return None
