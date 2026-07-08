"""Market data helpers for ALCB."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable

from ib_async import IB

from libs.broker_ibkr.mapping.contract_factory import ContractFactory
from libs.oms.models.instrument import Instrument

from .config import ET
from .models import Bar, CandidateItem, MarketSnapshot, QuoteSnapshot, ResearchDailyBar

logger = logging.getLogger(__name__)


@dataclass
class RateBudget:
    rate_per_second: float
    burst: float
    _tokens: float = field(init=False)
    _updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.burst
        self._updated_at = time.monotonic()

    def consume(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_second)
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True

    async def wait_for(self, cost: float = 1.0) -> None:
        while not self.consume(cost=cost):
            await asyncio.sleep(max(0.05, cost / max(self.rate_per_second, 1e-9)))


@dataclass
class SnapshotEntry:
    value: Any
    updated_at: datetime


class SnapshotCache:
    def __init__(self) -> None:
        self._data: dict[str, SnapshotEntry] = {}

    def put(self, key: str, value: Any, now: datetime | None = None) -> None:
        self._data[key] = SnapshotEntry(value=value, updated_at=now or datetime.now(timezone.utc))

    def get(self, key: str, max_age_s: float | None = None, now: datetime | None = None) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if max_age_s is None:
            return entry.value
        current = now or datetime.now(timezone.utc)
        if (current - entry.updated_at).total_seconds() > max_age_s:
            return None
        return entry.value

    def is_stale(self, key: str, max_age_s: float, now: datetime | None = None) -> bool:
        return self.get(key, max_age_s=max_age_s, now=now) is None


class _MinuteAccumulator:
    def __init__(self) -> None:
        self.current_minute: datetime | None = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.last_cumulative_volume: float | None = None

    def update(self, symbol: str, ts: datetime, price: float, cumulative_volume: float) -> Bar | None:
        minute = ts.replace(second=0, microsecond=0)
        current_cumulative_volume = max(0.0, cumulative_volume)
        if self.last_cumulative_volume is None:
            self.last_cumulative_volume = current_cumulative_volume
            self._reset(minute, price, 0.0)
            return None
        if current_cumulative_volume < self.last_cumulative_volume:
            volume_delta = current_cumulative_volume
        else:
            volume_delta = max(0.0, current_cumulative_volume - self.last_cumulative_volume)
        self.last_cumulative_volume = current_cumulative_volume
        if self.current_minute is None:
            self._reset(minute, price, volume_delta)
            return None
        if minute == self.current_minute:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price
            self.volume += volume_delta
            return None
        closed = Bar(
            symbol=symbol,
            start_time=self.current_minute,
            end_time=self.current_minute + timedelta(minutes=1),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )
        self._reset(minute, price, volume_delta)
        return closed

    def _reset(self, minute: datetime, price: float, volume_delta: float) -> None:
        self.current_minute = minute
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume_delta


class CanonicalBarBuilder:
    def __init__(self) -> None:
        self.completed_1m: dict[str, list[Bar]] = {}
        self._last_scanned_index: dict[tuple[str, int], int] = {}
        self._last_emitted_end: dict[tuple[str, int], datetime] = {}

    def ingest_bar(self, bar: Bar) -> None:
        bars = self.completed_1m.setdefault(bar.symbol, [])
        if bars and bars[-1].start_time >= bar.start_time:
            return
        bars.append(bar)

    def aggregate_new_bars(self, symbol: str, timeframe_minutes: int) -> list[Bar]:
        bars = self.completed_1m.get(symbol, [])
        key = (symbol, timeframe_minutes)
        scan_index = self._last_scanned_index.get(key, 0)
        last_emitted_end = self._last_emitted_end.get(key)
        produced: list[Bar] = []
        for index in range(scan_index, len(bars)):
            end_time = bars[index].end_time
            if end_time.minute % timeframe_minutes != 0:
                continue
            start_index = index - timeframe_minutes + 1
            if start_index < 0:
                continue
            chunk = bars[start_index : index + 1]
            if len(chunk) < timeframe_minutes:
                continue
            if chunk[0].start_time != end_time - timedelta(minutes=timeframe_minutes):
                continue
            if any(chunk[pos].end_time != chunk[pos + 1].start_time for pos in range(len(chunk) - 1)):
                continue
            if last_emitted_end is not None and end_time <= last_emitted_end:
                continue
            produced.append(
                Bar(
                    symbol=symbol,
                    start_time=chunk[0].start_time,
                    end_time=end_time,
                    open=chunk[0].open,
                    high=max(item.high for item in chunk),
                    low=min(item.low for item in chunk),
                    close=chunk[-1].close,
                    volume=sum(item.volume for item in chunk),
                )
            )
            last_emitted_end = end_time
        self._last_scanned_index[key] = len(bars)
        if last_emitted_end is not None:
            self._last_emitted_end[key] = last_emitted_end
        return produced


class StrategyDataStore:
    """Thin data facade over nightly artifacts + live market snapshots."""

    def __init__(self, items: dict[str, CandidateItem], markets: dict[str, MarketSnapshot]) -> None:
        self._items = items
        self._markets = markets

    @staticmethod
    def _merge_bars(history: list[Bar], live: list[Bar]) -> list[Bar]:
        rows: dict[datetime, Bar] = {}
        for bar in history + live:
            rows[bar.end_time] = bar
        return [rows[key] for key in sorted(rows)]

    def item(self, symbol: str) -> CandidateItem | None:
        return self._items.get(symbol.upper())

    def market(self, symbol: str) -> MarketSnapshot | None:
        return self._markets.get(symbol.upper())

    def daily_bars(self, symbol: str) -> list[ResearchDailyBar]:
        item = self.item(symbol)
        market = self.market(symbol)
        if market and market.daily_bars:
            return market.daily_bars
        return item.daily_bars[:] if item else []

    def bars_30m(self, symbol: str) -> list[Bar]:
        item = self.item(symbol)
        market = self.market(symbol)
        history = item.bars_30m[:] if item else []
        live = market.bars_30m[:] if market else []
        return self._merge_bars(history, live)

    def bars_4h(self, symbol: str) -> list[Bar]:
        market = self.market(symbol)
        if market and market.bars_4h:
            return market.bars_4h[:]
        return aggregate_bars(self.bars_30m(symbol), 8)

    def latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        market = self.market(symbol)
        return None if market is None else market.last_quote

    def current_week_30m_bars(self, symbol: str) -> list[Bar]:
        bars = self.bars_30m(symbol)
        if not bars:
            return []
        latest = bars[-1].end_time
        week_start = latest.date().toordinal() - latest.weekday()
        return [bar for bar in bars if bar.end_time.date().toordinal() >= week_start]


def aggregate_bars(bars: list[Bar], bars_per_chunk: int) -> list[Bar]:
    if bars_per_chunk <= 1:
        return bars[:]
    produced: list[Bar] = []

    def emit_session(session_bars: list[Bar]) -> None:
        for index in range(0, len(session_bars), bars_per_chunk):
            chunk = session_bars[index : index + bars_per_chunk]
            complete = len(chunk) == bars_per_chunk
            session_close = chunk[-1].end_time.astimezone(ET).time() >= dt_time(16, 0)
            if not complete and not session_close:
                continue
            produced.append(
                Bar(
                    symbol=chunk[0].symbol,
                    start_time=chunk[0].start_time,
                    end_time=chunk[-1].end_time,
                    open=chunk[0].open,
                    high=max(bar.high for bar in chunk),
                    low=min(bar.low for bar in chunk),
                    close=chunk[-1].close,
                    volume=sum(bar.volume for bar in chunk),
                )
            )

    current_key: tuple[str, Any] | None = None
    session_bars: list[Bar] = []
    for bar in bars:
        et_bar = bar.start_time.astimezone(ET)
        key = (bar.symbol, et_bar.date())
        if current_key is not None and key != current_key:
            emit_session(session_bars)
            session_bars = []
        current_key = key
        session_bars.append(bar)
    if session_bars:
        emit_session(session_bars)
    return produced


class IBMarketDataSource:
    """Lightweight streaming bridge with periodic historical refresh support."""

    _BLACKLIST_ERRORS = frozenset({10089})
    _TICK_BY_TICK_ERRORS = frozenset({10189, 10190})
    _BLACKLIST_DURATION = timedelta(hours=1)

    def __init__(
        self,
        ib: IB,
        contract_factory: ContractFactory,
        on_quote: Callable[[str, QuoteSnapshot], Any] | Callable[[str, QuoteSnapshot], Awaitable[Any]],
        on_bar: Callable[[str, Bar], Any] | Callable[[str, Bar], Awaitable[Any]],
        historical_requester: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._ib = ib
        self._factory = contract_factory
        self._on_quote = on_quote
        self._on_bar = on_bar
        self._historical_requester = historical_requester
        self._contracts: dict[str, Any] = {}
        self._builders: dict[str, _MinuteAccumulator] = {}
        self._logical_symbol_by_conid: dict[int, str] = {}
        self._logical_symbol_by_broker_symbol: dict[str, str] = {}
        self._tick_by_tick_disabled: set[str] = set()
        self._poll_budget = RateBudget(rate_per_second=2.0, burst=4.0)
        self._snapshot_cache = SnapshotCache()
        self._last_history_end: dict[str, datetime] = {}
        self._blacklisted: dict[str, datetime] = {}
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._ib.pendingTickersEvent += self._handle_pending_tickers
        self._ib.errorEvent += self._on_ib_error
        self._running = True

    async def stop(self) -> None:
        if not self._running:
            return
        self._ib.pendingTickersEvent -= self._handle_pending_tickers
        self._ib.errorEvent -= self._on_ib_error
        for symbol in list(self._contracts):
            self._remove_symbol(symbol)
        self._running = False

    def invalidate_subscriptions(self) -> None:
        self._contracts.clear()
        self._builders.clear()
        self._last_history_end.clear()
        self._blacklisted.clear()
        self._logical_symbol_by_conid.clear()
        self._logical_symbol_by_broker_symbol.clear()
        self._tick_by_tick_disabled.clear()

    def _remove_symbol(self, symbol: str) -> None:
        contract = self._contracts.pop(symbol, None)
        if contract is not None:
            self._ib.cancelTickByTickData(contract, "Last")
            self._ib.cancelTickByTickData(contract, "BidAsk")
            self._ib.cancelMktData(contract)
        self._builders.pop(symbol, None)
        for con_id, logical_symbol in list(self._logical_symbol_by_conid.items()):
            if logical_symbol == symbol:
                self._logical_symbol_by_conid.pop(con_id, None)
        for broker_symbol, logical_symbol in list(self._logical_symbol_by_broker_symbol.items()):
            if logical_symbol == symbol:
                self._logical_symbol_by_broker_symbol.pop(broker_symbol, None)

    def _register_contract_symbol(self, logical_symbol: str, contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            self._logical_symbol_by_conid[con_id] = logical_symbol
        broker_symbol = str(getattr(contract, "symbol", "") or "").upper()
        if broker_symbol:
            self._logical_symbol_by_broker_symbol[broker_symbol] = logical_symbol

    def _resolve_symbol(self, contract) -> str:
        logical_symbol = self._factory.logical_symbol_for_contract(contract)
        if logical_symbol:
            return logical_symbol.upper()
        con_id = int(getattr(contract, "conId", 0) or 0) if contract is not None else 0
        if con_id and con_id in self._logical_symbol_by_conid:
            return self._logical_symbol_by_conid[con_id]
        broker_symbol = str(getattr(contract, "symbol", "") or "").upper() if contract else ""
        if broker_symbol in self._logical_symbol_by_broker_symbol:
            return self._logical_symbol_by_broker_symbol[broker_symbol]
        return broker_symbol

    def _disable_tick_by_tick(self, symbol: str, *, error_code: int, error_string: str, contract) -> None:
        tracked_contract = self._contracts.get(symbol) or contract
        self._tick_by_tick_disabled.add(symbol)
        if tracked_contract is not None:
            try:
                self._ib.cancelTickByTickData(tracked_contract, "Last")
            except Exception:
                pass
            try:
                self._ib.cancelTickByTickData(tracked_contract, "BidAsk")
            except Exception:
                pass
        logger.warning(
            "Tick-by-tick unavailable for %s (code %d), continuing with reqMktData only: %s",
            symbol,
            error_code,
            error_string,
        )

    def _on_ib_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        symbol = self._resolve_symbol(contract)
        if errorCode in self._TICK_BY_TICK_ERRORS:
            if symbol and symbol in self._contracts and symbol not in self._tick_by_tick_disabled:
                self._disable_tick_by_tick(
                    symbol,
                    error_code=errorCode,
                    error_string=errorString,
                    contract=contract,
                )
            return
        if errorCode not in self._BLACKLIST_ERRORS or not symbol:
            return
        self._remove_symbol(symbol)
        self._blacklisted[symbol] = datetime.now(timezone.utc) + self._BLACKLIST_DURATION
        logger.warning("Blacklisted %s after market-data denial %s: %s", symbol, errorCode, errorString)

    async def ensure_hot_symbols(self, instruments: Iterable[Instrument]) -> None:
        wanted = {instrument.symbol: instrument for instrument in instruments}
        for symbol in list(self._contracts):
            if symbol not in wanted:
                self._remove_symbol(symbol)
        now = datetime.now(timezone.utc)
        for symbol, instrument in wanted.items():
            if symbol in self._contracts:
                continue
            if symbol in self._blacklisted and now < self._blacklisted[symbol]:
                continue
            if symbol in self._blacklisted:
                del self._blacklisted[symbol]
            contract, _ = await self._factory.resolve(symbol=instrument.root or instrument.symbol, instrument=instrument)
            self._contracts[symbol] = contract
            self._register_contract_symbol(symbol, contract)
            self._builders[symbol] = _MinuteAccumulator()
            self._ib.reqMktData(contract)
            if symbol not in self._tick_by_tick_disabled:
                self._ib.reqTickByTickData(contract, "Last")
                self._ib.reqTickByTickData(contract, "BidAsk")

    async def request_recent_bars(self, instrument: Instrument, duration: str = "1 D") -> list[Bar]:
        contract, _ = await self._factory.resolve(symbol=instrument.root or instrument.symbol, instrument=instrument)
        if self._historical_requester is not None:
            rows = await self._historical_requester(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                request_kind="recurring",
            )
        else:
            # Raw IB fallback for standalone tools/tests without UnifiedIBSession.
            rows = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                timeout=20,
            )
        result: list[Bar] = []
        for row in rows:
            start = row.date if isinstance(row.date, datetime) else datetime.now(timezone.utc)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            else:
                start = start.astimezone(timezone.utc)
            result.append(
                Bar(
                    symbol=instrument.symbol,
                    start_time=start,
                    end_time=start + timedelta(minutes=1),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                )
            )
        return result

    async def poll_due_bars(self, requests: Iterable[tuple[Instrument, int]], now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        for instrument, interval_s in requests:
            symbol = instrument.symbol.upper()
            cache_key = f"hist:{symbol}"
            if not self._snapshot_cache.is_stale(cache_key, max_age_s=max(1.0, float(interval_s)), now=current):
                continue
            self._snapshot_cache.put(cache_key, "pending", now=current)
            await self._poll_budget.wait_for()
            duration = "1 D" if symbol not in self._last_history_end else "1800 S"
            try:
                bars = await self.request_recent_bars(instrument, duration=duration)
            except Exception:
                continue
            last_end = self._last_history_end.get(symbol)
            new_bars = [bar for bar in bars if last_end is None or bar.end_time > last_end]
            if new_bars:
                self._last_history_end[symbol] = new_bars[-1].end_time
            for bar in new_bars:
                self._dispatch(self._on_bar(symbol, bar))

    def _handle_pending_tickers(self, tickers) -> None:
        now = datetime.now(timezone.utc)
        for ticker in tickers:
            contract = getattr(ticker, "contract", None)
            symbol = self._resolve_symbol(contract)
            if symbol not in self._contracts:
                continue
            bid = float(getattr(ticker, "bid", 0.0) or 0.0)
            ask = float(getattr(ticker, "ask", 0.0) or 0.0)
            last = float(getattr(ticker, "last", 0.0) or 0.0)
            midpoint = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(last, 0.0)
            spread_pct = ((ask - bid) / midpoint) if midpoint > 0 and bid > 0 and ask > 0 else 0.0
            quote = QuoteSnapshot(
                ts=now,
                bid=bid,
                ask=ask,
                last=last or midpoint,
                bid_size=float(getattr(ticker, "bidSize", 0.0) or 0.0),
                ask_size=float(getattr(ticker, "askSize", 0.0) or 0.0),
                cumulative_volume=float(getattr(ticker, "volume", 0.0) or 0.0),
                cumulative_value=float(getattr(ticker, "vwap", 0.0) or 0.0) * float(getattr(ticker, "volume", 0.0) or 0.0),
                vwap=float(getattr(ticker, "vwap", 0.0) or 0.0) or None,
                is_halted=bool(getattr(ticker, "halted", 0) or getattr(ticker, "delayedHalted", 0)),
                spread_pct=spread_pct,
            )
            self._dispatch(self._on_quote(symbol, quote))
            accumulator = self._builders.get(symbol)
            if accumulator is None or quote.last <= 0:
                continue
            bar = accumulator.update(symbol, now, quote.last, quote.cumulative_volume)
            if bar is not None:
                self._dispatch(self._on_bar(symbol, bar))

    @staticmethod
    def _dispatch(result: Any) -> None:
        if inspect.isawaitable(result):
            asyncio.create_task(result)
