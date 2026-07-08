"""Live data feed — polls Hyperliquid candles endpoint for bar assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.core.market_time import candle_open_from_ms
from crypto_trader.core.runtime_types import MarketEvent, TimestampPolicy

log = structlog.get_logger()

# Hyperliquid candle interval strings
_TF_TO_INTERVAL = {
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.M30: "30m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
}

# Emission priority (lower = first)
_TF_PRIORITY = {
    TimeFrame.D1: 0,
    TimeFrame.H4: 1,
    TimeFrame.H1: 2,
    TimeFrame.M30: 3,
    TimeFrame.M15: 4,
    TimeFrame.M5: 5,
}


class BarAssembler:
    """Polls Hyperliquid candles and detects new bar completions.

    Each (symbol, timeframe) pair is polled independently. The last 3 candles
    are requested; the second-to-last is the most recently completed bar.
    A bar is emitted when its timestamp exceeds the last emitted timestamp
    for that (symbol, tf) pair.
    """

    def __init__(self, info: Any, symbols: list[str], timeframes: list[TimeFrame]) -> None:
        self._info = info
        self._symbols = symbols
        self._timeframes = sorted(timeframes, key=lambda tf: _TF_PRIORITY.get(tf, 99))

        # Track last emitted timestamp per (symbol, tf)
        self._last_emitted: dict[tuple[str, TimeFrame], datetime] = {}

    def poll_all(self) -> list[Bar]:
        """Poll all (symbol, tf) pairs and return newly completed bars.

        Returns bars sorted by emission priority (D1 → H4 → H1 → M30 → M15).
        """
        new_bars = []

        for tf in self._timeframes:
            for symbol in self._symbols:
                bar = self._poll_one(symbol, tf)
                if bar is not None:
                    new_bars.append(bar)

        return new_bars

    def _poll_one(self, symbol: str, tf: TimeFrame) -> Bar | None:
        """Poll one (symbol, tf) pair. Returns a new bar or None."""
        interval = _TF_TO_INTERVAL.get(tf)
        if interval is None:
            return None

        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            # Request last 3 candles
            lookback_ms = tf.minutes * 60 * 1000 * 3
            start_ms = now_ms - lookback_ms

            candles = self._info.candles_snapshot(symbol, interval, start_ms, now_ms)

            if not candles or len(candles) < 2:
                return None

            # Second-to-last candle is most recently completed
            completed = candles[-2]
            candle_ts = _candle_open_time(completed, tf)

            key = (symbol, tf)
            last = self._last_emitted.get(key)

            if last is None or candle_ts > last:
                self._last_emitted[key] = candle_ts

                bar = Bar(
                    timestamp=candle_ts,
                    symbol=symbol,
                    open=float(completed["o"]),
                    high=float(completed["h"]),
                    low=float(completed["l"]),
                    close=float(completed["c"]),
                    volume=float(completed["v"]),
                    timeframe=tf,
                )

                log.debug(
                    "feed.new_bar",
                    symbol=symbol,
                    tf=tf.value,
                    ts=str(candle_ts),
                )
                return bar

        except Exception:
            log.exception("feed.poll_failed", symbol=symbol, tf=tf.value)

        return None

    def set_last_emitted(self, symbol: str, tf: TimeFrame, ts: datetime) -> None:
        """Set the last emitted timestamp (used during warmup)."""
        self._last_emitted[(symbol, tf)] = ts


class LiveFeed:
    """Wraps BarAssembler for the full set of strategies.

    Computes the union of all required timeframes from strategy subscriptions.
    Loads historical warmup bars on startup.
    """

    def __init__(
        self,
        info: Any,
        symbols: list[str],
        strategy_timeframes: dict[str, list[TimeFrame]],
    ) -> None:
        # Compute union of all timeframes
        all_tfs: set[TimeFrame] = set()
        for tfs in strategy_timeframes.values():
            all_tfs.update(tfs)

        self._assembler = BarAssembler(info, symbols, sorted(all_tfs, key=lambda t: t.minutes))
        self._symbols = symbols
        self._all_tfs = all_tfs
        self._strategy_tfs = strategy_timeframes

    @property
    def assembler(self) -> BarAssembler:
        return self._assembler

    def poll(self) -> list[Bar]:
        """Poll for new bars."""
        return self._assembler.poll_all()

    def poll_market_events(self) -> list[MarketEvent]:
        """Poll and return normalized completed-bar events."""
        return [
            MarketEvent.from_bar(
                bar,
                source="live",
                timestamp_policy=TimestampPolicy.OPEN_TIME,
            )
            for bar in self.poll()
        ]

    def load_warmup_bars(self, info: Any, warmup_counts: dict[TimeFrame, int]) -> list[Bar]:
        """Load historical bars for strategy warmup.

        Args:
            info: Hyperliquid Info instance
            warmup_counts: {TimeFrame: n_bars} — how many bars to load per TF

        Returns:
            List of bars sorted chronologically, in TF priority order at boundaries
        """
        warmup_bars = []

        for tf, count in warmup_counts.items():
            if tf not in self._all_tfs:
                continue

            interval = _TF_TO_INTERVAL.get(tf)
            if interval is None:
                continue

            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            lookback_ms = tf.minutes * 60 * 1000 * (count + 5)  # extra margin

            for symbol in self._symbols:
                try:
                    candles = info.candles_snapshot(
                        symbol, interval, now_ms - lookback_ms, now_ms
                    )
                    if not candles:
                        continue

                    # Skip the last candle (currently forming)
                    completed = candles[:-1]
                    # Take only the last `count` bars
                    completed = completed[-count:]

                    for c in completed:
                        bar = Bar(
                            timestamp=_candle_open_time(c, tf),
                            symbol=symbol,
                            open=float(c["o"]),
                            high=float(c["h"]),
                            low=float(c["l"]),
                            close=float(c["c"]),
                            volume=float(c["v"]),
                            timeframe=tf,
                        )
                        warmup_bars.append(bar)

                    # Set last emitted to most recent completed bar
                    if completed:
                        last_ts = _candle_open_time(completed[-1], tf)
                        self._assembler.set_last_emitted(symbol, tf, last_ts)

                except Exception:
                    log.exception("feed.warmup_failed", symbol=symbol, tf=tf.value)

        # Sort by timestamp, then by TF priority for same timestamp
        warmup_bars.sort(key=lambda b: (b.timestamp, _TF_PRIORITY.get(b.timeframe, 99)))

        log.info("feed.warmup_loaded", bars=len(warmup_bars))
        return warmup_bars


def _candle_open_time(candle: dict, tf: TimeFrame) -> datetime:
    """Return canonical candle open time from Hyperliquid data."""
    return candle_open_from_ms(
        tf=tf,
        open_ms=candle.get("t"),
        close_ms=candle.get("T"),
    )
