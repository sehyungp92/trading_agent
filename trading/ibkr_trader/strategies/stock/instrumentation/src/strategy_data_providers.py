"""Read-only instrumentation data providers over live strategy state."""
from __future__ import annotations

from collections import deque
from datetime import datetime, time, timezone
from typing import Iterable, Sequence


def _to_ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def _filter_since(rows: list[list[float]], since: int | None) -> list[list[float]]:
    if since is None:
        return rows
    return [row for row in rows if row[0] >= since]


def _limit_rows(rows: list[list[float]], limit: int | None) -> list[list[float]]:
    if limit is None or limit <= 0:
        return rows
    return rows[-limit:]


def _aggregate_bars(
    bars: Sequence,
    *,
    timeframe_minutes: int,
) -> list[list[float]]:
    aggregated: list[list[float]] = []
    chunk: list = []
    for bar in bars:
        chunk.append(bar)
        if len(chunk) < timeframe_minutes:
            continue
        start_ts = getattr(chunk[0], "start_time", getattr(chunk[0], "ts", None))
        if start_ts is None:
            chunk = []
            continue
        aggregated.append(
            [
                _to_ms(start_ts),
                float(chunk[0].open),
                float(max(b.high for b in chunk)),
                float(min(b.low for b in chunk)),
                float(chunk[-1].close),
                float(sum(getattr(b, "volume", 0.0) for b in chunk)),
            ]
        )
        chunk = []
    return aggregated


def _bars_to_ohlcv(bars: Iterable, *, use_ts: bool = False) -> list[list[float]]:
    rows: list[list[float]] = []
    for bar in bars:
        ts = getattr(bar, "ts", None) if use_ts else getattr(bar, "start_time", None)
        if ts is None:
            continue
        rows.append(
            [
                _to_ms(ts),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(getattr(bar, "volume", 0.0)),
            ]
        )
    return rows


def _daily_bars_to_ohlcv(bars: Iterable) -> list[list[float]]:
    rows: list[list[float]] = []
    for bar in bars:
        trade_date = getattr(bar, "trade_date", None)
        if trade_date is None:
            continue
        ts = datetime.combine(trade_date, time(9, 30), tzinfo=timezone.utc)
        rows.append(
            [
                _to_ms(ts),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(getattr(bar, "volume", 0.0)),
            ]
        )
    return rows


def _atr_from_ohlcv(rows: Sequence[list[float]], period: int = 14) -> float | None:
    if len(rows) < period + 1:
        return None

    true_ranges: list[float] = []
    previous_close = rows[0][4]
    for row in rows[1:]:
        high = row[2]
        low = row[3]
        close = row[4]
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        previous_close = close

    sample = true_ranges[-period:]
    if not sample:
        return None
    return sum(sample) / len(sample)


class IARICInstrumentationDataProvider:
    """Expose IARIC live state through the instrumentation data-provider contract."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None:
            return 0.0, 0.0
        return float(market.bid or 0.0), float(market.ask or 0.0)

    def get_last_price(self, symbol: str) -> float:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None or market.last_price is None:
            return 0.0
        return float(market.last_price)

    def get_atr(self, symbol: str) -> float | None:
        rows = self.get_ohlcv(symbol, timeframe="5m", limit=30)
        atr = _atr_from_ohlcv(rows)
        if atr is not None:
            return atr
        item = self._engine._items.get(symbol)  # noqa: SLF001
        if item is None:
            return None
        return float(getattr(item, "daily_atr_estimate", 0.0) or 0.0)

    def get_ohlcv(
        self,
        symbol: str,
        *,
        timeframe: str = "1h",
        limit: int = 120,
        since: int | None = None,
    ) -> list[list[float]]:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None:
            return []

        if timeframe == "1m":
            rows = _bars_to_ohlcv(market.minute_bars)
        elif timeframe == "5m":
            rows = _bars_to_ohlcv(market.bars_5m)
        elif timeframe == "30m":
            rows = _bars_to_ohlcv(market.bars_30m)
        elif timeframe == "1h":
            rows = _aggregate_bars(list(market.bars_30m), timeframe_minutes=2)
        else:
            rows = []

        return _limit_rows(_filter_since(rows, since), limit)


class ALCBInstrumentationDataProvider:
    """Expose ALCB live state through the instrumentation data-provider contract."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None:
            return 0.0, 0.0
        return float(market.bid or 0.0), float(market.ask or 0.0)

    def get_last_price(self, symbol: str) -> float:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None or market.last_price is None:
            return 0.0
        return float(market.last_price)

    def get_atr(self, symbol: str) -> float | None:
        daily_rows = self.get_ohlcv(symbol, timeframe="1d", limit=30)
        atr = _atr_from_ohlcv(daily_rows)
        if atr is not None:
            return atr
        item = self._engine._items.get(symbol)  # noqa: SLF001
        if item is None:
            return None
        return float(getattr(item, "intraday_atr_seed", 0.0) or 0.0)

    def get_ohlcv(
        self,
        symbol: str,
        *,
        timeframe: str = "4h",
        limit: int = 120,
        since: int | None = None,
    ) -> list[list[float]]:
        market = self._engine._markets.get(symbol)  # noqa: SLF001
        if market is None:
            return []

        if timeframe == "1m":
            rows = _bars_to_ohlcv(market.minute_bars)
        elif timeframe == "5m":
            rows = _aggregate_bars(list(market.minute_bars), timeframe_minutes=5)
        elif timeframe == "30m":
            rows = _bars_to_ohlcv(market.bars_30m)
        elif timeframe == "1h":
            rows = _aggregate_bars(list(market.bars_30m), timeframe_minutes=2)
        elif timeframe == "4h":
            rows = _bars_to_ohlcv(market.bars_4h)
        elif timeframe == "1d":
            rows = _daily_bars_to_ohlcv(market.daily_bars)
        else:
            rows = []

        return _limit_rows(_filter_since(rows, since), limit)


class USORBInstrumentationDataProvider:
    """Expose US ORB live state through the instrumentation data-provider contract."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def get_bid_ask(self, symbol: str) -> tuple[float, float]:
        ctx = self._resolve_context(symbol)
        if ctx is None or ctx.quote is None:
            return 0.0, 0.0
        return float(ctx.quote.bid or 0.0), float(ctx.quote.ask or 0.0)

    def get_last_price(self, symbol: str) -> float:
        ctx = self._resolve_context(symbol)
        if ctx is None or ctx.last_price is None:
            return 0.0
        return float(ctx.last_price)

    def get_atr(self, symbol: str) -> float | None:
        ctx = self._resolve_context(symbol)
        if ctx is None:
            return None
        atr = _atr_from_ohlcv(self.get_ohlcv(symbol, timeframe="1m", limit=20))
        if atr is not None:
            return atr
        return float(getattr(ctx.cached, "atr1m14", 0.0) or 0.0)

    def get_ohlcv(
        self,
        symbol: str,
        *,
        timeframe: str = "1h",
        limit: int = 120,
        since: int | None = None,
    ) -> list[list[float]]:
        ctx = self._resolve_context(symbol)
        if ctx is None:
            return []

        minute_rows = _bars_to_ohlcv(ctx.bars, use_ts=True)
        if timeframe == "1m":
            rows = minute_rows
        elif timeframe == "5m":
            rows = _aggregate_bars(list(ctx.bars), timeframe_minutes=5)
        elif timeframe == "1h":
            rows = _aggregate_bars(list(ctx.bars), timeframe_minutes=60)
        else:
            rows = []

        return _limit_rows(_filter_since(rows, since), limit)

    def _resolve_context(self, symbol: str):
        normalized = symbol.upper()
        if normalized in getattr(self._engine, "_symbols", {}):  # noqa: SLF001
            return self._engine._symbols.get(normalized)  # noqa: SLF001

        if normalized in getattr(self._engine, "_proxies", {}):  # noqa: SLF001
            proxy_quote = self._engine._proxies.get(normalized)  # noqa: SLF001
            proxy_bars: deque = self._engine._proxy_bars.get(normalized, deque())  # noqa: SLF001

            class _ProxyContext:
                def __init__(self, quote, bars):
                    self.quote = quote
                    self.bars = bars
                    self.cached = type("Cached", (), {"atr1m14": 0.0})()

                @property
                def last_price(self):
                    if self.quote and self.quote.last > 0:
                        return self.quote.last
                    if self.bars:
                        return self.bars[-1].close
                    return None

            return _ProxyContext(proxy_quote, proxy_bars)

        return None
