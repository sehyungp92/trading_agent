"""Self-contained research snapshot generator using IB data.

Runs in the premarket window, fetches all data from Interactive Brokers,
computes derived fields, and writes a ResearchSnapshot JSON file that
``run_daily_selection`` consumes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, median
from zoneinfo import ZoneInfo

from .config import StrategySettings

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

IB_REQUEST_RATE_PER_SECOND = 1.0
IB_REQUEST_RATE_BURST = 2.0
IB_REQUEST_CONCURRENCY = 5
HISTORICAL_REQUEST_TIMEOUT = 180.0
HISTORICAL_TIMEOUT_RETRY_DELAYS = (5.0, 15.0)


# ---------------------------------------------------------------------------
# Rate limiter (mirrors strategy_iaric/data.py RateBudget)
# ---------------------------------------------------------------------------


@dataclass
class _RateBudget:
    rate_per_second: float = 1.0
    burst: float = 2.0
    _tokens: float = 2.0
    _updated_at: float = 0.0

    def __post_init__(self) -> None:
        self._updated_at = _time.monotonic()

    def _refill(self) -> None:
        now = _time.monotonic()
        elapsed = now - self._updated_at
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate_per_second)
        self._updated_at = now

    async def wait_for(self, cost: float = 1.0) -> None:
        while True:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return
            await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# IB helpers
# ---------------------------------------------------------------------------


def _stock_contract(symbol: str, exchange: str = "SMART", primary_exchange: str = ""):
    """Build an IB Stock contract."""
    from ib_async import Stock
    c = Stock(symbol, exchange, "USD")
    if primary_exchange:
        c.primaryExchange = primary_exchange
    return c


def _index_contract(symbol: str, exchange: str = "CBOE"):
    from ib_async import Index
    return Index(symbol, exchange, "USD")


async def _request_historical_bars(
    ib,
    contract,
    *,
    duration: str,
    bar_size: str,
    what: str = "TRADES",
):
    """Request historical bars with timeout-aware retries for nightly sweeps."""
    total_attempts = len(HISTORICAL_TIMEOUT_RETRY_DELAYS) + 1
    label = getattr(contract, "localSymbol", None) or getattr(contract, "symbol", contract)
    for attempt in range(1, total_attempts + 1):
        retry_delay = (
            HISTORICAL_TIMEOUT_RETRY_DELAYS[attempt - 1]
            if attempt <= len(HISTORICAL_TIMEOUT_RETRY_DELAYS)
            else None
        )
        started_at = _time.monotonic()
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what,
                useRTH=True,
                keepUpToDate=False,
                timeout=HISTORICAL_REQUEST_TIMEOUT,
            )
        except Exception:
            if retry_delay is None:
                raise
            logger.warning(
                "IARIC historical bars failed for %s %s %s %s (attempt %d/%d); retrying in %.0fs",
                label,
                duration,
                bar_size,
                what,
                attempt,
                total_attempts,
                retry_delay,
                exc_info=True,
            )
            await asyncio.sleep(retry_delay)
            continue

        elapsed = _time.monotonic() - started_at
        timed_out = (
            not bars
            and HISTORICAL_REQUEST_TIMEOUT > 0
            and elapsed >= max(HISTORICAL_REQUEST_TIMEOUT * 0.9, HISTORICAL_REQUEST_TIMEOUT - 1.0)
        )
        if bars or not timed_out or retry_delay is None:
            if timed_out:
                logger.warning(
                    "IARIC historical bars timed out for %s %s %s %s after %.1fs; no retries left",
                    label,
                    duration,
                    bar_size,
                    what,
                    elapsed,
                )
            return bars

        logger.warning(
            "IARIC historical bars timed out for %s %s %s %s after %.1fs (attempt %d/%d); retrying in %.0fs",
            label,
            duration,
            bar_size,
            what,
            elapsed,
            attempt,
            total_attempts,
            retry_delay,
        )
        await asyncio.sleep(retry_delay)

    return []


# ---------------------------------------------------------------------------
# Daily bar cache
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "daily_bars" / f"{symbol}.json"


def _load_cached_bars(cache_dir: Path, symbol: str) -> tuple[list[dict], str | None]:
    path = _cache_path(cache_dir, symbol)
    if not path.exists():
        return [], None
    with open(path) as f:
        data = json.load(f)
    return data.get("bars", []), data.get("last_updated")


def _save_cached_bars(cache_dir: Path, symbol: str, bars: list[dict], last_updated: str) -> None:
    path = _cache_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"symbol": symbol, "last_updated": last_updated, "bars": bars}, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Contract details cache
# ---------------------------------------------------------------------------


def _contract_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / "contract_details" / f"{symbol}.json"


def _load_cached_contract(cache_dir: Path, symbol: str) -> dict | None:
    path = _contract_cache_path(cache_dir, symbol)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_cached_contract(cache_dir: Path, symbol: str, data: dict) -> None:
    path = _contract_cache_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Flow proxy computation
# ---------------------------------------------------------------------------


def _compute_flow_proxy(bars: list[dict]) -> list[float]:
    """Dollar-weighted close position ratio (Chaikin-style)."""
    result: list[float] = []
    for bar in bars:
        high, low, close, volume = bar["high"], bar["low"], bar["close"], bar["volume"]
        width = max(high - low, 1e-9)
        cpr = (close - low) / width
        result.append(volume * (2.0 * cpr - 1.0))
    return result


def _compute_atr(bars: list[dict], period: int = 15) -> float:
    sample = bars[-period:]
    if len(sample) < 2:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(sample)):
        h, l, pc = sample[i]["high"], sample[i]["low"], sample[i - 1]["close"]
        true_ranges.append(max(h - l, abs(h - pc), abs(l - pc)))
    return fmean(true_ranges) if true_ranges else 0.0


# ---------------------------------------------------------------------------
# Event tag detection
# ---------------------------------------------------------------------------


def _apply_event_tags(bars: list[dict], earnings_bar_index: int | None = None) -> None:
    """Mutate bars in-place to add event_tag field."""
    for i, bar in enumerate(bars):
        bar.setdefault("event_tag", "")

    if len(bars) < 20:
        return

    for i in range(20, len(bars)):
        bar = bars[i]
        window = bars[i - 20:i]
        max_high = max(b["high"] for b in window)
        avg_vol = fmean(b["volume"] for b in window)
        width = max(bar["high"] - bar["low"], 1e-9)
        cpr = (bar["close"] - bar["low"]) / width

        if bar["close"] > max_high and avg_vol > 0 and bar["volume"] > 2.0 * avg_vol and cpr >= 0.7:
            bar["event_tag"] = "BREAKOUT"

    if earnings_bar_index is not None and 0 <= earnings_bar_index < len(bars):
        eb = bars[earnings_bar_index]
        avg_window = bars[max(0, earnings_bar_index - 20):earnings_bar_index]
        if avg_window:
            avg_vol = fmean(b["volume"] for b in avg_window)
            width = max(eb["high"] - eb["low"], 1e-9)
            cpr = (eb["close"] - eb["low"]) / width
            if avg_vol > 0 and eb["volume"] > 2.0 * avg_vol and cpr >= 0.6:
                eb["event_tag"] = "EARNINGS_CONTINUATION"


# ---------------------------------------------------------------------------
# Spread heuristic from ADV tier
# ---------------------------------------------------------------------------


def _spread_heuristic(adv_usd: float) -> float:
    if adv_usd > 100_000_000:
        return 0.0005
    if adv_usd > 50_000_000:
        return 0.0010
    if adv_usd > 20_000_000:
        return 0.0020
    return 0.0035


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


async def generate_research_snapshot(
    trade_date: date,
    ib,
    settings: StrategySettings | None = None,
) -> Path:
    """Generate a ``ResearchSnapshot`` JSON file from IB data.

    Parameters
    ----------
    trade_date:
        The trading date to generate data for.
    ib:
        A connected ``ib_insync.IB`` instance.
    settings:
        Strategy configuration (uses defaults if *None*).

    Returns
    -------
    Path to the written JSON file.
    """
    cfg = settings or StrategySettings()
    cache_dir = cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    rate = _RateBudget(rate_per_second=IB_REQUEST_RATE_PER_SECOND, burst=IB_REQUEST_RATE_BURST)
    sem = asyncio.Semaphore(IB_REQUEST_CONCURRENCY)

    # -- Load universe -------------------------------------------------------
    from .universe_constituents import KNOWN_ETFS
    from strategies.stock.live_universe import (
        BACKTESTED_INTRADAY_STOCK_SYMBOLS,
        LIVE_STOCK_UNIVERSE,
        LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS,
    )

    universe: dict[str, tuple[str, str]] = {}  # symbol -> (sector, primary_exchange)
    for sym, sector, pex in LIVE_STOCK_UNIVERSE:
        universe[sym] = (sector, pex)

    logger.info(
        "IARIC focused live universe: %d symbols (%d backtested, %d Nasdaq/Dow additions)",
        len(universe),
        len(BACKTESTED_INTRADAY_STOCK_SYMBOLS),
        len(LIVE_STOCK_UNIVERSE_ADDED_SYMBOLS),
    )

    all_symbols = list(universe.keys())[:600]
    logger.info("Research universe: %d symbols", len(all_symbols))

    # -- Load blacklist -------------------------------------------------------
    blacklist: set[str] = set()
    if cfg.blacklist_path.exists():
        blacklist = {
            line.strip().upper()
            for line in cfg.blacklist_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    # -- Resolve contracts (cached) -------------------------------------------
    contract_details: dict[str, dict] = {}
    symbols_needing_resolve: list[str] = []
    for sym in all_symbols:
        cached = _load_cached_contract(cache_dir, sym)
        if cached is not None:
            contract_details[sym] = cached
        else:
            symbols_needing_resolve.append(sym)

    if symbols_needing_resolve:
        logger.info("Resolving %d contracts via IB", len(symbols_needing_resolve))
        resolve_tasks = [
            _resolve_contract(ib, sym, universe.get(sym, ("", ""))[1], cache_dir, rate, sem)
            for sym in symbols_needing_resolve
        ]
        results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
        for sym, result in zip(symbols_needing_resolve, results):
            if isinstance(result, dict):
                contract_details[sym] = result
            else:
                logger.debug("Contract resolve failed for %s: %s", sym, result)

    resolved_symbols = [s for s in all_symbols if s in contract_details]
    logger.info("Resolved %d / %d contracts", len(resolved_symbols), len(all_symbols))

    # -- Fetch reference data (SPY, VIX, HYG, sector ETFs) -------------------
    ref_data = await _fetch_reference_data(ib, rate, sem)

    # -- Fetch daily bars for universe (incremental cache) --------------------
    today_str = trade_date.isoformat()
    bar_tasks = [
        _fetch_daily_bars_cached(ib, sym, contract_details[sym], cache_dir, today_str, rate, sem)
        for sym in resolved_symbols
    ]
    bar_results = await asyncio.gather(*bar_tasks, return_exceptions=True)
    daily_bars_by_symbol: dict[str, list[dict]] = {}
    for sym, result in zip(resolved_symbols, bar_results):
        if isinstance(result, list) and len(result) >= 5:
            daily_bars_by_symbol[sym] = result
        else:
            logger.debug("Skipping %s: insufficient bars (%s)", sym, type(result).__name__ if isinstance(result, Exception) else len(result) if isinstance(result, list) else "?")

    logger.info("Daily bars fetched for %d symbols", len(daily_bars_by_symbol))

    # -- Compute market regime data -------------------------------------------
    market_data = _compute_market_data(ref_data, daily_bars_by_symbol, universe)

    # -- Compute sector metrics -----------------------------------------------
    sector_metrics = _compute_sector_metrics(ref_data, daily_bars_by_symbol, universe)

    # -- Liquidity filter pass (for 30m bar fetch) ----------------------------
    liquid_symbols: list[str] = []
    for sym, bars in daily_bars_by_symbol.items():
        if len(bars) < 20:
            continue
        last_close = bars[-1]["close"]
        adv20 = fmean(b["close"] * b["volume"] for b in bars[-20:])
        if last_close >= cfg.min_price and adv20 >= cfg.min_adv_usd:
            liquid_symbols.append(sym)

    # -- Fetch intraday 30m bars for liquid symbols ---------------------------
    avg_30m_volume: dict[str, float] = {}
    if liquid_symbols:
        logger.info("Fetching 30m intraday bars for %d liquid symbols", len(liquid_symbols))
        intraday_tasks = [
            _fetch_intraday_30m(ib, sym, contract_details[sym], rate, sem)
            for sym in liquid_symbols
        ]
        intraday_results = await asyncio.gather(*intraday_tasks, return_exceptions=True)
        for sym, result in zip(liquid_symbols, intraday_results):
            if isinstance(result, (float, int)) and result > 0:
                avg_30m_volume[sym] = float(result)

    # -- Build per-symbol research entries ------------------------------------
    symbols_payload: dict[str, dict] = {}
    for sym in daily_bars_by_symbol:
        bars = daily_bars_by_symbol[sym]
        sector, pex = universe.get(sym, ("Unknown", ""))
        cd = contract_details.get(sym, {})

        last_close = bars[-1]["close"]
        adv20 = fmean(b["close"] * b["volume"] for b in bars[-20:]) if len(bars) >= 20 else 0.0
        spread_pct = _spread_heuristic(adv20)

        # Flow proxy from last 40 bars
        flow_bars = bars[-40:]
        flow_proxy_history = _compute_flow_proxy(flow_bars)

        # Event tags
        earnings_bar_idx = _find_earnings_bar_index(bars, cd.get("next_earnings_date"))
        _apply_event_tags(bars, earnings_bar_idx)

        # ATR seed
        atr_val = _compute_atr(bars)
        intraday_atr_seed = atr_val / max(last_close, 1e-9)

        # Sector returns from ETF data
        sector_etf = SECTOR_ETFS.get(sector)
        sector_ret_20d = 0.0
        sector_ret_60d = 0.0
        if sector_etf and sector_etf in ref_data.get("sector_etf_bars", {}):
            etf_bars = ref_data["sector_etf_bars"][sector_etf]
            if len(etf_bars) >= 21 and etf_bars[-21]["close"] > 0:
                sector_ret_20d = (etf_bars[-1]["close"] - etf_bars[-21]["close"]) / etf_bars[-21]["close"]
            if len(etf_bars) >= 61 and etf_bars[-61]["close"] > 0:
                sector_ret_60d = (etf_bars[-1]["close"] - etf_bars[-61]["close"]) / etf_bars[-61]["close"]

        # Earnings proximity
        earnings_within = _earnings_within_sessions(cd.get("next_earnings_date"), trade_date)

        avg_30m = avg_30m_volume.get(sym, 0.0)

        # Build bar dicts for JSON (with event_tag already applied)
        bar_dicts = [
            {
                "trade_date": b["trade_date"],
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b["volume"],
                "event_tag": b.get("event_tag", ""),
            }
            for b in bars
        ]

        symbols_payload[sym] = {
            "exchange": cd.get("exchange", "SMART"),
            "primary_exchange": cd.get("primary_exchange", pex),
            "currency": "USD",
            "tick_size": cd.get("min_tick", 0.01),
            "point_value": 1.0,
            "sector": sector,
            "price": last_close,
            "adv20_usd": round(adv20, 2),
            "median_spread_pct": spread_pct,
            "earnings_within_sessions": earnings_within,
            "blacklist_flag": sym.upper() in blacklist,
            "halted_flag": False,
            "severe_news_flag": False,
            "etf_flag": sym in KNOWN_ETFS,
            "adr_flag": cd.get("adr_flag", False),
            "preferred_flag": "PR" in sym or cd.get("preferred_flag", False),
            "otc_flag": cd.get("primary_exchange", pex) in ("PINK", "OTC"),
            "hard_to_borrow_flag": False,
            "flow_proxy_history": [round(v, 2) for v in flow_proxy_history],
            "daily_bars": bar_dicts,
            "sector_return_20d": round(sector_ret_20d, 6),
            "sector_return_60d": round(sector_ret_60d, 6),
            "intraday_atr_seed": round(intraday_atr_seed, 6),
            "average_30m_volume": round(avg_30m, 2),
            "expected_5m_volume": round(avg_30m / 6.0, 2) if avg_30m > 0 else 0.0,
        }

    # -- Held positions from previous day's state ----------------------------
    held_positions = _extract_held_positions(trade_date, cfg)

    # -- Assemble final snapshot JSON ----------------------------------------
    snapshot = {
        "trade_date": trade_date.isoformat(),
        "market": market_data,
        "sectors": sector_metrics,
        "symbols": symbols_payload,
        "held_positions": held_positions,
    }

    output_dir = cfg.research_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{trade_date.isoformat()}.json"
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(snapshot, f, default=str)
    tmp_path.replace(output_path)

    logger.info("Research snapshot written: %s (%d symbols)", output_path, len(symbols_payload))
    return output_path


# ---------------------------------------------------------------------------
# IB data fetchers
# ---------------------------------------------------------------------------


async def _resolve_contract(
    ib, symbol: str, primary_exchange: str, cache_dir: Path,
    rate: _RateBudget, sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        await rate.wait_for()
        contract = _stock_contract(symbol, primary_exchange=primary_exchange)
        details_list = await ib.reqContractDetailsAsync(contract)
        if not details_list:
            raise ValueError(f"No contract details for {symbol}")
        cd = details_list[0]
        # IB may expose next earnings via contract notes or custom fields
        next_earnings = None
        for attr in ("nextEarningsDate", "contractMonth"):
            val = getattr(cd, attr, None) or getattr(cd.contract, attr, None)
            if val and isinstance(val, str) and len(val) >= 8:
                try:
                    next_earnings = date.fromisoformat(val[:10]).isoformat()
                    break
                except (ValueError, TypeError):
                    pass

        result = {
            "con_id": cd.contract.conId,
            "symbol": cd.contract.symbol,
            "exchange": cd.contract.exchange or "SMART",
            "primary_exchange": cd.contract.primaryExchange or primary_exchange,
            "min_tick": cd.minTick or 0.01,
            "category": cd.category or "",
            "subcategory": cd.subcategory or "",
            "long_name": cd.longName or "",
            "adr_flag": "ADR" in (cd.category or "") or "ADR" in (cd.subcategory or ""),
            "preferred_flag": "Preferred" in (cd.longName or ""),
            "next_earnings_date": next_earnings,
        }
        _save_cached_contract(cache_dir, symbol, result)
        return result


async def _fetch_scanner_symbols(ib, rate: _RateBudget) -> list[str]:
    """Fetch momentum scanner results to supplement universe."""
    from ib_async import ScannerSubscription

    await rate.wait_for()
    sub = ScannerSubscription(
        numberOfRows=100,
        instrument="STK",
        locationCode="STK.US.MAJOR",
        scanCode="TOP_PERC_GAIN",
        abovePrice=10.0,
        aboveVolume=500_000,
    )
    results = await ib.reqScannerDataAsync(sub)
    symbols = [r.contractDetails.contract.symbol for r in (results or []) if r.contractDetails]
    logger.info("Scanner returned %d momentum symbols", len(symbols))
    return symbols


async def _fetch_reference_data(ib, rate: _RateBudget, sem: asyncio.Semaphore) -> dict:
    """Fetch SPY, VIX, HYG, and sector ETF bars."""
    result: dict = {}

    async def _fetch_bars(contract, duration: str, bar_size: str, what: str, key: str):
        async with sem:
            await rate.wait_for()
            bars = await _request_historical_bars(
                ib,
                contract,
                duration=duration,
                bar_size=bar_size,
                what=what,
            )
            result[key] = [
                {
                    "trade_date": str(b.date) if hasattr(b, "date") else "",
                    "open": float(b.open), "high": float(b.high),
                    "low": float(b.low), "close": float(b.close),
                    "volume": float(getattr(b, "volume", 0)),
                }
                for b in (bars or [])
            ]

    tasks = [
        _fetch_bars(_stock_contract("SPY", primary_exchange="ARCA"), "1 Y", "1 day", "TRADES", "spy_bars"),
        _fetch_bars(_index_contract("VIX", "CBOE"), "1 Y", "1 day", "MIDPOINT", "vix_bars"),
        _fetch_bars(_stock_contract("HYG", primary_exchange="ARCA"), "10 D", "1 day", "TRADES", "hyg_bars"),
    ]
    # Sector ETF bars — run concurrently with reference bars
    sector_etf_tasks = []
    for sector_name, etf_sym in SECTOR_ETFS.items():
        async def _sector_fetch(sym=etf_sym):
            async with sem:
                await rate.wait_for()
                contract = _stock_contract(sym, primary_exchange="ARCA")
                bars = await _request_historical_bars(
                    ib,
                    contract,
                    duration="120 D",
                    bar_size="1 day",
                    what="TRADES",
                )
                return sym, [
                    {
                        "trade_date": str(b.date), "open": float(b.open), "high": float(b.high),
                        "low": float(b.low), "close": float(b.close), "volume": float(getattr(b, "volume", 0)),
                    }
                    for b in (bars or [])
                ]
        sector_etf_tasks.append(_sector_fetch())

    # Run all reference + sector ETF fetches concurrently
    all_ref = tasks + sector_etf_tasks
    all_results = await asyncio.gather(*all_ref, return_exceptions=True)
    # Log failures for reference data (SPY/VIX/HYG)
    ref_labels = ["SPY", "VIX", "HYG"]
    for i, res in enumerate(all_results[:len(tasks)]):
        if isinstance(res, Exception):
            logger.warning("Reference data fetch failed for %s: %s", ref_labels[i], res)
    result["sector_etf_bars"] = {}
    for res in all_results[len(tasks):]:
        if isinstance(res, tuple):
            sym, bars = res
            result["sector_etf_bars"][sym] = bars
        elif isinstance(res, Exception):
            logger.debug("Sector ETF fetch failed: %s", res)

    return result


async def _fetch_daily_bars_cached(
    ib, symbol: str, cd: dict, cache_dir: Path,
    today_str: str, rate: _RateBudget, sem: asyncio.Semaphore,
) -> list[dict]:
    """Fetch daily bars with incremental caching."""
    cached_bars, last_updated = _load_cached_bars(cache_dir, symbol)

    if last_updated == today_str and len(cached_bars) >= 200:
        return cached_bars

    async with sem:
        await rate.wait_for()
        con_id = cd.get("con_id")
        pex = cd.get("primary_exchange", "")
        contract = _stock_contract(symbol, primary_exchange=pex)
        if con_id:
            contract.conId = con_id

        if cached_bars and last_updated:
            duration = "5 D"
        else:
            duration = "1 Y"  # 252 trading days for SMA200 + warmup

        bars = await _request_historical_bars(
            ib,
            contract,
            duration=duration,
            bar_size="1 day",
            what="TRADES",
        )

        new_bars = [
            {
                "trade_date": str(b.date),
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(getattr(b, "volume", 0)),
            }
            for b in (bars or [])
        ]

        if cached_bars and duration == "5 D":
            existing_dates = {b["trade_date"] for b in cached_bars}
            for nb in new_bars:
                if nb["trade_date"] not in existing_dates:
                    cached_bars.append(nb)
            merged = cached_bars[-120:]
        else:
            merged = new_bars[-120:]

        _save_cached_bars(cache_dir, symbol, merged, today_str)
        return merged


async def _fetch_intraday_30m(
    ib, symbol: str, cd: dict, rate: _RateBudget, sem: asyncio.Semaphore,
) -> float:
    """Fetch 5 days of 30-min bars, return average 30m volume."""
    async with sem:
        await rate.wait_for()
        pex = cd.get("primary_exchange", "")
        contract = _stock_contract(symbol, primary_exchange=pex)
        con_id = cd.get("con_id")
        if con_id:
            contract.conId = con_id

        bars = await _request_historical_bars(
            ib,
            contract,
            duration="5 D",
            bar_size="30 mins",
            what="TRADES",
        )
        if not bars:
            return 0.0
        volumes = [float(getattr(b, "volume", 0)) for b in bars]
        return fmean(volumes) if volumes else 0.0


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _compute_market_data(
    ref_data: dict,
    daily_bars: dict[str, list[dict]],
    universe: dict[str, tuple[str, str]],
) -> dict:
    """Compute market regime fields from reference data."""
    spy_bars = ref_data.get("spy_bars", [])
    vix_bars = ref_data.get("vix_bars", [])
    hyg_bars = ref_data.get("hyg_bars", [])

    # price_ok: SPY above its 200-day SMA
    price_ok = False
    if len(spy_bars) >= 200:
        spy_close = spy_bars[-1]["close"]
        sma200 = fmean(b["close"] for b in spy_bars[-200:])
        price_ok = spy_close > sma200

    # breadth: % of universe above their 20-day SMA
    above_20dma = 0
    total_checked = 0
    for sym, bars in daily_bars.items():
        if len(bars) >= 20:
            sma20 = fmean(b["close"] for b in bars[-20:])
            if bars[-1]["close"] > sma20:
                above_20dma += 1
            total_checked += 1
    breadth_pct = (above_20dma / max(total_checked, 1)) * 100.0

    # VIX percentile
    vix_percentile = 50.0
    if vix_bars:
        vix_closes = [b["close"] for b in vix_bars]
        current_vix = vix_closes[-1]
        below_count = sum(1 for v in vix_closes if v <= current_vix)
        vix_percentile = 100.0 * below_count / len(vix_closes)

    # HY spread proxy (inverse HYG change)
    hy_spread_change = 0.0
    if len(hyg_bars) >= 6:
        hyg_now = hyg_bars[-1]["close"]
        hyg_5d_ago = hyg_bars[-6]["close"]
        if hyg_5d_ago > 0:
            hy_spread_change = -(hyg_now - hyg_5d_ago) / hyg_5d_ago * 10000

    # Institutional selling detection
    market_wide_selling = False
    if len(spy_bars) >= 21:
        spy_last = spy_bars[-1]
        avg_vol_20 = fmean(b["volume"] for b in spy_bars[-21:-1])
        if (
            avg_vol_20 > 0
            and spy_last["volume"] > 2.0 * avg_vol_20
            and spy_last["close"] < spy_last["open"]
            and breadth_pct < 40.0
        ):
            market_wide_selling = True

    return {
        "price_ok": price_ok,
        "breadth_pct_above_20dma": round(breadth_pct, 2),
        "vix_percentile_1y": round(vix_percentile, 2),
        "hy_spread_5d_bps_change": round(hy_spread_change, 2),
        "market_wide_institutional_selling": market_wide_selling,
    }


def _compute_sector_metrics(
    ref_data: dict,
    daily_bars: dict[str, list[dict]],
    universe: dict[str, tuple[str, str]],
) -> dict[str, dict]:
    """Compute per-sector flow, breadth, and participation metrics."""
    sector_etf_bars = ref_data.get("sector_etf_bars", {})
    metrics: dict[str, dict] = {}

    # Group universe symbols by sector
    by_sector: dict[str, list[str]] = {}
    for sym, (sector, _) in universe.items():
        by_sector.setdefault(sector, []).append(sym)

    for sector_name, etf_sym in SECTOR_ETFS.items():
        etf_bars = sector_etf_bars.get(etf_sym, [])

        # flow_trend_20d from sector ETF (only compute for last 20 bars)
        flow_trend = 0.0
        if len(etf_bars) >= 20:
            flow_values = _compute_flow_proxy(etf_bars[-20:])
            flow_trend = fmean(flow_values) if flow_values else 0.0

        # breadth_20d: fraction of sector constituents above 20-day SMA
        sector_syms = by_sector.get(sector_name, [])
        above = 0
        total = 0
        for sym in sector_syms:
            bars = daily_bars.get(sym, [])
            if len(bars) >= 20:
                sma20 = fmean(b["close"] for b in bars[-20:])
                if bars[-1]["close"] > sma20:
                    above += 1
                total += 1
        breadth = above / max(total, 1)

        # participation: fraction with volume above their 20-day average
        active = 0
        total_p = 0
        for sym in sector_syms:
            bars = daily_bars.get(sym, [])
            if len(bars) >= 20:
                avg_vol = fmean(b["volume"] for b in bars[-20:])
                if avg_vol > 0 and bars[-1]["volume"] > avg_vol:
                    active += 1
                total_p += 1
        participation = active / max(total_p, 1)

        metrics[sector_name] = {
            "name": sector_name,
            "flow_trend_20d": round(flow_trend, 4),
            "breadth_20d": round(breadth, 4),
            "participation": round(participation, 4),
        }

    return metrics


def _find_earnings_bar_index(bars: list[dict], next_earnings_str: str | None) -> int | None:
    """Find the bar index near the earnings date."""
    if not next_earnings_str or not bars:
        return None
    try:
        ed = date.fromisoformat(next_earnings_str[:10])
    except (ValueError, TypeError):
        return None
    # Look backwards for a bar within 5 sessions of earnings
    for i in range(len(bars) - 1, max(-1, len(bars) - 6), -1):
        try:
            bd = date.fromisoformat(bars[i]["trade_date"][:10])
        except (ValueError, TypeError):
            continue
        if bd == ed or (0 < (bd - ed).days <= 1):
            return i
    return None


def _earnings_within_sessions(next_earnings_str: str | None, trade_date: date) -> int | None:
    if not next_earnings_str:
        return None
    try:
        ed = date.fromisoformat(next_earnings_str[:10])
    except (ValueError, TypeError):
        return None
    diff = (ed - trade_date).days
    if diff < 0:
        return None
    # Rough business day conversion
    sessions = int(diff * 5 / 7)
    return sessions


def _extract_held_positions(trade_date: date, cfg: StrategySettings) -> list[dict]:
    """Extract held positions from previous day's intraday state."""
    from .artifact_store import load_intraday_state

    # Try up to 4 days back (weekends/holidays)
    for offset in range(4):
        check_date = trade_date - timedelta(days=1 + offset)
        try:
            snapshot = load_intraday_state(check_date, settings=cfg)
            positions: list[dict] = []
            for sym_state in snapshot.symbols:
                pos = sym_state.position
                if pos is not None and pos.qty_open > 0:
                    positions.append({
                        "symbol": sym_state.symbol,
                        "entry_time": pos.entry_time.isoformat(),
                        "entry_price": pos.entry_price,
                        "size": pos.qty_open,
                        "stop": pos.current_stop,
                        "initial_r": pos.initial_risk_per_share,
                        "setup_tag": pos.setup_tag,
                        "carry_eligible_flag": False,
                    })
            return positions
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("Failed to load state for %s", check_date, exc_info=True)
            continue
    return []
