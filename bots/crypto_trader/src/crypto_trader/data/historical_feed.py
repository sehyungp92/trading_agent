"""Historical data feed for backtesting with multi-timeframe alignment.

Loads Parquet data from ParquetStore, converts to Bar objects, and iterates
in chronological order with higher-TF bars emitted before corresponding
primary-TF bars (D1 -> H4 -> H1 -> M15).
"""

from __future__ import annotations

import heapq
from datetime import datetime, time as dt_time, timezone
from typing import Iterator

import structlog

from crypto_trader.core.models import Bar, TimeFrame
from crypto_trader.core.market_time import completes_higher_timeframe, higher_timeframe_open
from crypto_trader.core.runtime_types import MarketEvent, TimestampPolicy
from crypto_trader.data.store import ParquetStore

log = structlog.get_logger()

# Timeframe emission priority (lower = emitted first at boundary)
_TF_PRIORITY: dict[TimeFrame, int] = {
    TimeFrame.D1: 0,
    TimeFrame.H4: 1,
    TimeFrame.H1: 2,
    TimeFrame.M30: 3,
    TimeFrame.M15: 4,
    TimeFrame.M5: 5,
}


class HistoricalFeed:
    """Multi-timeframe historical data feed.

    Loading: For each symbol x timeframe, loads Parquet via ParquetStore and
    converts rows to Bar objects.

    Iteration: Primary TF (M15) drives the loop. At each primary bar,
    higher-TF bars that complete at this boundary are emitted BEFORE the
    primary bar, in order D1 -> H4 -> H1 -> M15.

    The feed does NOT advance the clock ??the StrategyEngine does that.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[TimeFrame],
        store: ParquetStore,
        start_date: datetime,
        end_date: datetime,
        primary_timeframe: TimeFrame = TimeFrame.M15,
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes
        self.store = store
        self.start_date = start_date
        self.end_date = end_date
        self.primary_timeframe = primary_timeframe

        # Loaded bars per (symbol, timeframe)
        self._bars: dict[tuple[str, TimeFrame], list[Bar]] = {}
        # Higher-TF bars indexed by open timestamp for O(1) lookup
        self._tf_bars: dict[tuple[str, TimeFrame], dict[datetime, Bar]] = {}
        # Higher TFs that need boundary detection
        self._higher_tfs: list[TimeFrame] = [
            tf for tf in timeframes if tf != primary_timeframe and tf.minutes > primary_timeframe.minutes
        ]
        # Current iteration position per (symbol, tf) for get_history
        self._iteration_pos: dict[tuple[str, TimeFrame], int] = {}

        self._load_all()

    def _load_all(self) -> None:
        """Load all requested data from ParquetStore."""
        start_dt = datetime.combine(self.start_date, datetime.min.time(), tzinfo=timezone.utc) if not isinstance(self.start_date, datetime) else self.start_date
        # End-of-day: include all bars on end_date (Finding 8)
        end_dt = datetime.combine(self.end_date, dt_time(23, 59, 59, 999999), tzinfo=timezone.utc) if not isinstance(self.end_date, datetime) else self.end_date
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.store.load_candles(symbol, tf.value)
                if df is None:
                    log.warning("feed.no_data", symbol=symbol, timeframe=tf.value)
                    self._bars[(symbol, tf)] = []
                    continue

                # Filter to date range
                df = df[(df["ts"] >= start_ms) & (df["ts"] <= end_ms)]

                bars = []
                for ts_ms, open_, high, low, close, volume in df.itertuples(index=False, name=None):
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    bars.append(Bar(
                        timestamp=ts,
                        symbol=symbol,
                        open=float(open_),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume),
                        timeframe=tf,
                    ))

                self._bars[(symbol, tf)] = bars
                log.info("feed.loaded", symbol=symbol, timeframe=tf.value, bars=len(bars))

                # Index higher-TF bars by open timestamp
                if tf != self.primary_timeframe:
                    self._tf_bars[(symbol, tf)] = {b.timestamp: b for b in bars}

    def subscribe(self, symbol: str, timeframes: list[TimeFrame]) -> None:
        """Validate that requested data is loaded."""
        for tf in timeframes:
            key = (symbol, tf)
            if key not in self._bars:
                raise ValueError(f"No data loaded for {symbol} {tf.value}")

    def __iter__(self) -> Iterator[Bar]:
        """Iterate bars with multi-TF alignment.

        For single-symbol: iterate primary TF bars. At each boundary, emit
        higher-TF bars first (D1 -> H4 -> H1 -> primary).

        For multi-symbol: merge-sort primary bars across symbols by timestamp.
        """
        if len(self.symbols) == 1:
            yield from self._iterate_single_symbol(self.symbols[0])
        else:
            yield from self._iterate_multi_symbol()

    def iter_market_events(self) -> Iterator[MarketEvent]:
        """Iterate normalized completed-bar events with explicit availability."""
        for bar in self:
            yield MarketEvent.from_bar(
                bar,
                source="historical",
                timestamp_policy=TimestampPolicy.OPEN_TIME,
            )

    def _iterate_single_symbol(self, symbol: str) -> Iterator[Bar]:
        """Iterate with multi-TF alignment for a single symbol."""
        primary_bars = self._bars.get((symbol, self.primary_timeframe), [])

        for i, pbar in enumerate(primary_bars):
            self._iteration_pos[(symbol, self.primary_timeframe)] = i

            # Emit higher-TF bars at boundaries, in priority order
            for htf in sorted(self._higher_tfs, key=lambda tf: _TF_PRIORITY.get(tf, 99)):
                htf_bar = self._check_boundary(symbol, htf, pbar)
                if htf_bar is not None:
                    yield htf_bar

            yield pbar

    def _iterate_multi_symbol(self) -> Iterator[Bar]:
        """Merge-sort primary bars across symbols by timestamp."""
        # Tuple ties are resolved by the configured symbol order so a suffix
        # backtest emits same-timestamp bars in the same order as a full run.
        symbol_order = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        heap: list[tuple[datetime, int, str, int]] = []
        for symbol in self.symbols:
            primary_bars = self._bars.get((symbol, self.primary_timeframe), [])
            if primary_bars:
                heapq.heappush(heap, (primary_bars[0].timestamp, symbol_order[symbol], symbol, 0))

        while heap:
            ts, _, symbol, idx = heapq.heappop(heap)
            primary_bars = self._bars[(symbol, self.primary_timeframe)]
            pbar = primary_bars[idx]

            self._iteration_pos[(symbol, self.primary_timeframe)] = idx

            # Emit higher-TF bars at boundaries
            for htf in sorted(self._higher_tfs, key=lambda tf: _TF_PRIORITY.get(tf, 99)):
                htf_bar = self._check_boundary(symbol, htf, pbar)
                if htf_bar is not None:
                    yield htf_bar

            yield pbar

            # Push next bar for this symbol
            next_idx = idx + 1
            if next_idx < len(primary_bars):
                heapq.heappush(heap, (primary_bars[next_idx].timestamp, symbol_order[symbol], symbol, next_idx))

    def _check_boundary(self, symbol: str, htf: TimeFrame, primary_bar: Bar) -> Bar | None:
        """Check if this primary bar completes a higher-TF bar.

        Bars use canonical open-time timestamps and become visible only after
        the primary bar that closes the higher-timeframe window.
        """
        ts = primary_bar.timestamp
        if not completes_higher_timeframe(ts, self.primary_timeframe, htf):
            return None

        htf_open = self._compute_htf_open(ts, htf)
        tf_bars = self._tf_bars.get((symbol, htf), {})
        return tf_bars.get(htf_open)

    @staticmethod
    def _compute_htf_open(primary_ts: datetime, htf: TimeFrame) -> datetime:
        """Compute the open timestamp of the higher-TF bar that contains this primary bar."""
        return higher_timeframe_open(primary_ts, htf)

    def get_history(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int,
    ) -> list[Bar]:
        """Get the last `count` bars up to the current iteration position."""
        bars = self._bars.get((symbol, timeframe), [])
        if not bars:
            return []

        if timeframe == self.primary_timeframe:
            pos = self._iteration_pos.get((symbol, timeframe), len(bars) - 1)
            end = pos + 1
        else:
            # For higher TFs, only return bars whose timestamp <= current primary bar time
            primary_pos = self._iteration_pos.get((symbol, self.primary_timeframe), 0)
            primary_bars = self._bars.get((symbol, self.primary_timeframe), [])
            if primary_bars and primary_pos < len(primary_bars):
                current_time = primary_bars[primary_pos].timestamp
                # Binary search: find last higher-TF bar with timestamp <= current_time
                end = 0
                for i, b in enumerate(bars):
                    if b.timestamp <= current_time:
                        end = i + 1
                    else:
                        break
            else:
                end = len(bars)

        start = max(0, end - count)
        return bars[start:end]
