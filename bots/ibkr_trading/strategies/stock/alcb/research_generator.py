"""IB-backed nightly research generator for ALCB."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import fmean
from typing import Any

from ib_async import IB, ScannerSubscription, Stock

from .artifact_store import persist_research_snapshot
from .config import ScannerSettings, StrategySettings
from .models import Bar, MarketResearch, ResearchDailyBar, ResearchSnapshot, ResearchSymbol, SectorResearch
from .signals import ema
from .universe_constituents import KNOWN_ETFS
from strategies.stock.live_universe import (
    BACKTESTED_INTRADAY_STOCK_SYMBOLS,
    LIVE_STOCK_UNIVERSE,
    LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS,
)

logger = logging.getLogger(__name__)

IB_REQUEST_RATE_PER_SECOND = 1.0
IB_REQUEST_RATE_BURST = 2.0
IB_REQUEST_CONCURRENCY = 6
HISTORICAL_REQUEST_TIMEOUT = 180.0
HISTORICAL_TIMEOUT_RETRY_DELAYS = (5.0, 15.0)


@dataclass
class _RateBudget:
    rate_per_second: float = 2.0
    burst: float = 4.0
    _tokens: float = 4.0
    _updated_at: float = 0.0

    def __post_init__(self) -> None:
        import time

        self._updated_at = time.monotonic()

    async def wait_for(self, cost: float = 1.0) -> None:
        import time

        while True:
            now = time.monotonic()
            elapsed = now - self._updated_at
            self._updated_at = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_second)
            if self._tokens >= cost:
                self._tokens -= cost
                return
            await asyncio.sleep(0.1)


def _bars_to_daily(symbol: str, rows) -> list[ResearchDailyBar]:
    result: list[ResearchDailyBar] = []
    for row in rows:
        dt = row.date if isinstance(row.date, datetime) else datetime.now(timezone.utc)
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                trade_day = dt.date()
            else:
                trade_day = dt.astimezone(timezone.utc).date()
        else:
            trade_day = date.today()
        result.append(
            ResearchDailyBar(
                trade_date=trade_day,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
        )
    return result


def _bars_to_30m(symbol: str, rows) -> list[Bar]:
    result: list[Bar] = []
    for row in rows:
        start = row.date if isinstance(row.date, datetime) else datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)
        result.append(
            Bar(
                symbol=symbol,
                start_time=start,
                end_time=start + timedelta(minutes=30),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
        )
    return result


def _flow_proxy_history(bars: list[ResearchDailyBar]) -> list[float]:
    history: list[float] = []
    for bar in bars[-40:]:
        signed = bar.typical_price * bar.volume
        if bar.close < bar.open:
            signed *= -1.0
        history.append(signed)
    return history


def _average_30m_volume(bars: list[Bar]) -> float:
    if not bars:
        return 0.0
    return fmean(bar.volume for bar in bars[-40:])


def _median_spread_pct(contract_details) -> float:
    if contract_details is None:
        return 0.002
    min_tick = float(getattr(contract_details, "minTick", 0.01) or 0.01)
    price = float(getattr(getattr(contract_details, "contract", None), "strike", 0.0) or 0.0)
    if price <= 0:
        return min_tick / 100.0
    return min_tick / price


async def _fetch_scanner_symbols(
    ib: IB,
    rate: _RateBudget,
    scanner: ScannerSettings,
) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for scan_code in scanner.scan_codes:
        try:
            await rate.wait_for()
            sub = ScannerSubscription(
                numberOfRows=scanner.rows_per_scan,
                instrument=scanner.instrument,
                locationCode=scanner.location_code,
                scanCode=scan_code,
                abovePrice=scanner.above_price,
                aboveVolume=scanner.above_volume,
                stockTypeFilter=scanner.stock_type_filter,
            )
            results = await ib.reqScannerDataAsync(sub)
        except Exception:
            logger.warning("Scanner supplement failed for %s", scan_code, exc_info=True)
            continue
        added = 0
        for row in results or []:
            details = getattr(row, "contractDetails", None)
            contract = getattr(details, "contract", None)
            symbol = str(getattr(contract, "symbol", "") or "").upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
            added += 1
        logger.info("Scanner %s returned %d unique symbols", scan_code, added)
    return symbols


async def _fetch_bars(ib: IB, contract, duration: str, bar_size: str):
    # ib_async defaults to 60s and cancels the request on timeout; broad
    # research sweeps need a wider window plus a retry for farm hiccups.
    total_attempts = len(HISTORICAL_TIMEOUT_RETRY_DELAYS) + 1
    label = getattr(contract, "localSymbol", None) or getattr(contract, "symbol", contract)
    for attempt in range(1, total_attempts + 1):
        retry_delay = (
            HISTORICAL_TIMEOUT_RETRY_DELAYS[attempt - 1]
            if attempt <= len(HISTORICAL_TIMEOUT_RETRY_DELAYS)
            else None
        )
        started_at = time.monotonic()
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                keepUpToDate=False,
                timeout=HISTORICAL_REQUEST_TIMEOUT,
            )
        except Exception:
            if retry_delay is None:
                raise
            logger.warning(
                "ALCB historical bars failed for %s %s %s (attempt %d/%d); retrying in %.0fs",
                label,
                duration,
                bar_size,
                attempt,
                total_attempts,
                retry_delay,
                exc_info=True,
            )
            await asyncio.sleep(retry_delay)
            continue

        elapsed = time.monotonic() - started_at
        timed_out = (
            not bars
            and HISTORICAL_REQUEST_TIMEOUT > 0
            and elapsed >= max(HISTORICAL_REQUEST_TIMEOUT * 0.9, HISTORICAL_REQUEST_TIMEOUT - 1.0)
        )
        if bars or not timed_out or retry_delay is None:
            if timed_out:
                logger.warning(
                    "ALCB historical bars timed out for %s %s %s after %.1fs; no retries left",
                    label,
                    duration,
                    bar_size,
                    elapsed,
                )
            return bars

        logger.warning(
            "ALCB historical bars timed out for %s %s %s after %.1fs (attempt %d/%d); retrying in %.0fs",
            label,
            duration,
            bar_size,
            elapsed,
            attempt,
            total_attempts,
            retry_delay,
        )
        await asyncio.sleep(retry_delay)

    return []


async def _resolve_stock(ib: IB, symbol: str, primary_exchange: str) -> tuple[Any | None, Any | None]:
    contract = Stock(symbol, "SMART", "USD", primaryExchange=primary_exchange or "")
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            return None, None
        contract = qualified[0]
        details = await ib.reqContractDetailsAsync(contract)
        return contract, details[0] if details else None
    except Exception:
        return None, None


def _market_metrics(spy_bars: list[ResearchDailyBar], universe: dict[str, ResearchSymbol]) -> MarketResearch:
    closes = [bar.close for bar in spy_bars]
    ema50 = ema(closes, min(50, len(closes))) if closes else []
    price_ok = bool(ema50 and closes[-1] >= ema50[-1])
    above_20 = 0
    total = 0
    for item in universe.values():
        item_closes = [bar.close for bar in item.daily_bars]
        if len(item_closes) < 20:
            continue
        total += 1
        if item_closes[-1] >= fmean(item_closes[-20:]):
            above_20 += 1
    breadth = (100.0 * above_20 / total) if total else 0.0
    return MarketResearch(
        price_ok=price_ok,
        breadth_pct_above_20dma=breadth,
        vix_percentile_1y=45.0,
        hy_spread_5d_bps_change=5.0,
        market_wide_institutional_selling=(not price_ok and breadth < 45.0),
    )


def _sector_metrics(universe: dict[str, ResearchSymbol]) -> dict[str, SectorResearch]:
    grouped: dict[str, list[ResearchSymbol]] = {}
    for item in universe.values():
        grouped.setdefault(item.sector or "Unknown", []).append(item)
    metrics: dict[str, SectorResearch] = {}
    for sector, items in grouped.items():
        flow = fmean(sum(symbol.flow_proxy_history[-5:]) for symbol in items if symbol.flow_proxy_history)
        breadth = fmean(
            1.0 if symbol.daily_bars and symbol.daily_bars[-1].close >= fmean(bar.close for bar in symbol.daily_bars[-20:]) else 0.0
            for symbol in items
            if len(symbol.daily_bars) >= 20
        ) if items else 0.0
        participation = fmean(min(symbol.adv20_usd / 100_000_000.0, 2.0) for symbol in items) if items else 0.0
        metrics[sector] = SectorResearch(name=sector, flow_trend_20d=flow / 1_000_000_000.0, breadth_20d=breadth, participation=participation)
    return metrics


async def generate_research_snapshot(
    trade_date: date,
    *,
    ib: IB,
    settings: StrategySettings | None = None,
) -> ResearchSnapshot:
    cfg = settings or StrategySettings()
    rate = _RateBudget(rate_per_second=IB_REQUEST_RATE_PER_SECOND, burst=IB_REQUEST_RATE_BURST)
    universe: dict[str, tuple[str, str]] = {
        symbol: (sector, primary)
        for symbol, sector, primary in LIVE_STOCK_UNIVERSE
    }
    logger.info(
        "ALCB focused live universe: %d symbols (%d backtested, %d Nasdaq/Dow additions)",
        len(universe),
        len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        len(LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS),
    )
    all_symbols = list(universe.keys())[: cfg.universe_cap]
    sem = asyncio.Semaphore(IB_REQUEST_CONCURRENCY)
    _progress = {"done": 0, "total": len(all_symbols)}

    async def _load_symbol(symbol: str) -> tuple[str, ResearchSymbol | None]:
        sector, primary = universe.get(symbol, ("Unknown", ""))
        async with sem:
            contract, details = await _resolve_stock(ib, symbol, primary)
            if contract is None:
                return symbol, None
            try:
                await rate.wait_for()
                daily_rows = await _fetch_bars(ib, contract, "2 Y", "1 day")
                await rate.wait_for()
                bars_30m_rows = await _fetch_bars(ib, contract, "90 D", "30 mins")
            except Exception:
                _progress["done"] += 1
                return symbol, None
        _progress["done"] += 1
        n = _progress["done"]
        if n % 50 == 0 or n == _progress["total"]:
            logger.info("ALCB fetch progress: %d / %d symbols", n, _progress["total"])
        daily_bars = _bars_to_daily(symbol, daily_rows)
        bars_30m = _bars_to_30m(symbol, bars_30m_rows)
        if len(daily_bars) < 70:
            return symbol, None
        avg_30m = _average_30m_volume(bars_30m)
        return symbol, ResearchSymbol(
            symbol=symbol,
            exchange="SMART",
            primary_exchange=primary,
            currency="USD",
            tick_size=float(getattr(details, "minTick", 0.01) or 0.01),
            point_value=1.0,
            sector=sector,
            price=float(daily_bars[-1].close),
            adv20_usd=fmean(bar.close * bar.volume for bar in daily_bars[-20:]),
            median_spread_pct=_median_spread_pct(details),
            earnings_within_sessions=10,
            blacklist_flag=False,
            halted_flag=False,
            severe_news_flag=False,
            etf_flag=symbol in KNOWN_ETFS,
            flow_proxy_history=_flow_proxy_history(daily_bars),
            daily_bars=daily_bars,
            bars_30m=bars_30m,
            sector_return_20d=0.0,
            sector_return_60d=0.0,
            intraday_atr_seed=0.0,
            average_30m_volume=avg_30m,
            median_30m_volume=avg_30m,
        )

    loaded = await asyncio.gather(*[_load_symbol(symbol) for symbol in all_symbols])
    symbols = {symbol: item for symbol, item in loaded if item is not None}
    spy = symbols.get("SPY")
    market = _market_metrics(spy.daily_bars if spy else [], symbols)
    sectors = _sector_metrics(symbols)
    snapshot = ResearchSnapshot(
        trade_date=trade_date,
        market=market,
        sectors=sectors,
        symbols=symbols,
        held_positions=[],
    )
    persist_research_snapshot(snapshot, settings=cfg)
    return snapshot
