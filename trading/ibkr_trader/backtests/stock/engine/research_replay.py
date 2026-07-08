"""Research replay engine — builds historical ResearchSnapshot objects from cached bars.

This is the core of the stock backtest: it replaces the live research_generator.py
by computing all ResearchSnapshot fields from cached daily/intraday bar data.
The resulting snapshots are fed into the strategy's pure
``daily_selection_from_snapshot()`` functions.

Performance note: All date-based slicing uses pre-computed date→iloc maps
built at load time (O(1) lookup) rather than pd.index.normalize() per query.
"""
from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

from backtests.shared.auto.cache_keys import fingerprint_tree, stable_signature
from backtests.stock.config import UniverseConfig
from backtests.stock.data.cache import bar_path, load_bars
from backtests.stock.data.downloader import REFERENCE_SYMBOLS, SECTOR_ETFS

from strategies.stock.alcb.universe_constituents import KNOWN_ETFS, SP500_CONSTITUENTS
from strategies.stock.live_universe import BACKTESTED_INTRADAY_STOCK_SYMBOLS

# Strategy model imports — both ALCB and IARIC define identical
# MarketResearch / SectorResearch / ResearchDailyBar shapes.
import strategies.stock.alcb.models as alcb_models
import strategies.stock.alcb.config as alcb_config
import strategies.stock.alcb.research as alcb_research
import strategies.stock.iaric.models as iaric_models
import strategies.stock.iaric.config as iaric_config
import strategies.stock.iaric.research as iaric_research

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast date-index helpers
# ---------------------------------------------------------------------------

def _build_date_index(df: pd.DataFrame) -> tuple[list[date], list[int]]:
    """Build sorted parallel lists of (dates, last_iloc) for O(1) date lookup.

    Returns (sorted_dates, last_row_ilocs) where last_row_ilocs[i] is the
    iloc of the last row on sorted_dates[i].  Use with bisect for <= lookups.

    Uses numpy datetime64[D] for vectorized boundary detection instead of
    Python-level .date() iteration.  ~10x faster for large DataFrames.
    """
    if df.empty:
        return [], []
    # Truncate to day precision in numpy (vectorized C, no Python loop)
    days_ns = df.index.values.astype("datetime64[D]")
    n = len(days_ns)
    if n == 0:
        return [], []
    # Find boundaries where date changes (data is time-sorted)
    changes = np.where(days_ns[1:] != days_ns[:-1])[0]
    # Last iloc per date = boundary positions + final index
    last_ilocs_arr = np.append(changes, n - 1)
    # Convert only unique dates (~500) to Python date objects, not all rows (~78K)
    unique_days = days_ns[last_ilocs_arr]
    sorted_dates = [pd.Timestamp(d).date() for d in unique_days]
    return sorted_dates, last_ilocs_arr.tolist()


def _iloc_upto(sorted_dates: list[date], last_ilocs: list[int], trade_date: date) -> int:
    """Return the iloc of the last row with date <= trade_date, or -1 if none."""
    if not sorted_dates:
        return -1
    idx = bisect.bisect_right(sorted_dates, trade_date) - 1
    if idx < 0:
        return -1
    return last_ilocs[idx]


def _iloc_on(sorted_dates: list[date], last_ilocs: list[int], trade_date: date) -> tuple[int, int] | None:
    """Return (first_iloc, last_iloc) for rows exactly on trade_date, or None."""
    idx = bisect.bisect_left(sorted_dates, trade_date)
    if idx >= len(sorted_dates) or sorted_dates[idx] != trade_date:
        return None
    last = last_ilocs[idx]
    first = (last_ilocs[idx - 1] + 1) if idx > 0 else 0
    return first, last


def _iloc_after(sorted_dates: list[date], last_ilocs: list[int], trade_date: date) -> int:
    """Return iloc of first row with date > trade_date, or -1."""
    idx = bisect.bisect_right(sorted_dates, trade_date)
    if idx >= len(sorted_dates):
        return -1
    return (last_ilocs[idx - 1] + 1) if idx > 0 else 0


def _json_signature(value) -> str:
    """Build a stable JSON signature for settings-aware cache keys."""
    return stable_signature(value)


# ---------------------------------------------------------------------------
# Numpy pre-computation helpers
# ---------------------------------------------------------------------------

def _precompute_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract numpy arrays from DataFrame once at load time."""
    return {
        "open": df["open"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "close": df["close"].values.astype(np.float64),
        "volume": df["volume"].values.astype(np.float64),
    }


def _flow_proxy_array(arrs: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized Chaikin flow proxy: volume * (2*CPR - 1)."""
    h, l, c, v = arrs["high"], arrs["low"], arrs["close"], arrs["volume"]
    width = np.maximum(h - l, 1e-9)
    cpr = (c - l) / width
    return v * (2.0 * cpr - 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precompute_above_sma(closes: np.ndarray, period: int) -> np.ndarray:
    """Vectorized: returns bool array where closes[i] > SMA(period) at each i.

    Uses cumsum trick for O(N) rolling SMA instead of per-element numpy dispatch.
    Positions with < period bars are set to False.
    """
    n = len(closes)
    result = np.zeros(n, dtype=np.bool_)
    if n < period:
        return result
    cumsum = np.cumsum(closes)
    # sma[i] = (cumsum[i] - cumsum[i - period]) / period  for i >= period
    sma = np.empty(n, dtype=np.float64)
    sma[:period - 1] = 0.0
    sma[period - 1] = cumsum[period - 1] / period
    sma[period:] = (cumsum[period:] - cumsum[:n - period]) / period
    result[period - 1:] = closes[period - 1:] > sma[period - 1:]
    return result


def _precompute_vol_above_avg(volumes: np.ndarray, period: int) -> np.ndarray:
    """Vectorized: returns bool array where volumes[i] > mean(volumes[i-period:i]).

    Note: compares current volume against the PRIOR period average (excluding current),
    matching the original _compute_sector_research logic: np.mean(vols[-21:-1]).
    Positions with < period+1 bars are set to False.
    """
    n = len(volumes)
    result = np.zeros(n, dtype=np.bool_)
    if n < period + 1:
        return result
    cumsum = np.cumsum(volumes)
    # avg of prior period: (cumsum[i-1] - cumsum[i-1-period]) / period  for i >= period+1
    # At i=period: avg = cumsum[period-1] / period
    avg_prior = np.empty(n, dtype=np.float64)
    avg_prior[:period] = 0.0
    avg_prior[period] = cumsum[period - 1] / period
    avg_prior[period + 1:] = (cumsum[period:n - 1] - cumsum[:n - period - 1]) / period
    result[period:] = (avg_prior[period:] > 0) & (volumes[period:] > avg_prior[period:])
    return result


def _sma(closes: np.ndarray, period: int) -> float:
    """Simple moving average of the last *period* closes."""
    if len(closes) < period:
        return float(np.mean(closes)) if len(closes) > 0 else 0.0
    return float(np.mean(closes[-period:]))


def _percentile(value: float, window: np.ndarray) -> float:
    """Percentile of *value* within *window* (0-100)."""
    if len(window) == 0:
        return 50.0
    return float(np.sum(window <= value) / len(window) * 100.0)


def _daily_bars_from_arrays(
    arrs: dict[str, np.ndarray],
    sorted_dates: list[date],
    last_ilocs: list[int],
    trade_date: date,
    lookback: int = 250,
) -> list[alcb_models.ResearchDailyBar]:
    """Convert pre-computed arrays to ResearchDailyBar list, sliced to trade_date."""
    end_iloc = _iloc_upto(sorted_dates, last_ilocs, trade_date)
    if end_iloc < 0:
        return []
    start = max(0, end_iloc - lookback + 1)
    # Find dates for each row in range
    idx = bisect.bisect_right(sorted_dates, trade_date) - 1
    # Walk back to find which dates correspond to our range
    bars: list[alcb_models.ResearchDailyBar] = []
    di = idx  # pointer into sorted_dates
    # Build reverse mapping: for each row in [start, end_iloc], find its date
    row_dates: list[date] = []
    di = 0
    for i in range(start, end_iloc + 1):
        # Advance di until last_ilocs[di] >= i
        while di < len(last_ilocs) - 1 and last_ilocs[di] < i:
            di += 1
        row_dates.append(sorted_dates[di])

    for j, i in enumerate(range(start, end_iloc + 1)):
        bars.append(alcb_models.ResearchDailyBar(
            trade_date=row_dates[j],
            open=float(arrs["open"][i]),
            high=float(arrs["high"][i]),
            low=float(arrs["low"][i]),
            close=float(arrs["close"][i]),
            volume=float(arrs["volume"][i]),
        ))
    return bars


def _iaric_daily_bars_from_arrays(
    arrs: dict[str, np.ndarray],
    sorted_dates: list[date],
    last_ilocs: list[int],
    trade_date: date,
    lookback: int = 250,
) -> list[iaric_models.ResearchDailyBar]:
    """Same as above for IARIC ResearchDailyBar."""
    end_iloc = _iloc_upto(sorted_dates, last_ilocs, trade_date)
    if end_iloc < 0:
        return []
    start = max(0, end_iloc - lookback + 1)
    row_dates: list[date] = []
    di = 0
    for i in range(start, end_iloc + 1):
        while di < len(last_ilocs) - 1 and last_ilocs[di] < i:
            di += 1
        row_dates.append(sorted_dates[di])

    bars: list[iaric_models.ResearchDailyBar] = []
    for j, i in enumerate(range(start, end_iloc + 1)):
        bars.append(iaric_models.ResearchDailyBar(
            trade_date=row_dates[j],
            open=float(arrs["open"][i]),
            high=float(arrs["high"][i]),
            low=float(arrs["low"][i]),
            close=float(arrs["close"][i]),
            volume=float(arrs["volume"][i]),
        ))
    return bars


def _30m_bars_from_df(
    df: pd.DataFrame, end_date: date, lookback_days: int = 90,
) -> list[alcb_models.Bar]:
    """Convert 30m DataFrame to a list of ALCB ``Bar`` objects."""
    if df is None or df.empty:
        return []
    cutoff = pd.Timestamp(end_date, tz=df.index.tz)
    start = cutoff - pd.Timedelta(days=lookback_days)
    sliced = df.loc[(df.index >= start) & (df.index <= cutoff)]
    bars: list[alcb_models.Bar] = []
    for row in sliced.itertuples():
        ts = row.Index.to_pydatetime()
        bars.append(alcb_models.Bar(
            symbol="",  # filled by caller
            start_time=ts,
            end_time=ts,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(getattr(row, 'volume', 0)),
        ))
    return bars


# ---------------------------------------------------------------------------
# ResearchReplayEngine
# ---------------------------------------------------------------------------


class ResearchReplayEngine:
    """Build historical ResearchSnapshot objects from cached bar data.

    Usage::

        engine = ResearchReplayEngine(data_dir="backtests/stock/data/raw")
        engine.load_all_data()

        # For each trading day:
        alcb_snap = engine.build_alcb_snapshot(trading_date)
    """

    def __init__(
        self,
        data_dir: str | Path = "backtests/stock/data/raw",
        universe_config: UniverseConfig | None = None,
    ):
        self._data_dir = Path(data_dir)
        self._universe_config = universe_config or UniverseConfig()

        # (symbol, sector, exchange) tuples.  Stock backtests intentionally
        # load the focused intraday cohort by default; the broad S&P list is
        # still available by setting UniverseConfig(use_backtested_intraday_universe=False).
        if self._universe_config.use_backtested_intraday_universe:
            focused = set(BACKTESTED_INTRADAY_STOCK_SYMBOLS)
            self._universe = [
                row
                for row in SP500_CONSTITUENTS
                if row[0] in focused
            ]
        else:
            self._universe = list(SP500_CONSTITUENTS)
        self._sector_map: dict[str, str] = {sym: sector for sym, sector, _ in self._universe}
        self._exchange_map: dict[str, str] = {sym: exch for sym, _, exch in self._universe}

        # Sector → list of symbols (pre-computed for breadth/participation)
        self._sector_symbols: dict[str, list[str]] = defaultdict(list)
        for sym, sector, _ in self._universe:
            self._sector_symbols[sector].append(sym)

        # Cached DataFrames
        self._daily_cache: dict[str, pd.DataFrame] = {}
        self._intraday_30m_cache: dict[str, pd.DataFrame] = {}
        self._intraday_5m_cache: dict[str, pd.DataFrame] = {}
        self._ref_cache: dict[str, pd.DataFrame] = {}  # SPY, VIX, HYG, sector ETFs

        # Two-tier ALCB cache for auto-optimization:
        # Tier 1 (snapshot): keyed on (date, as_of, min_price, min_adv) — the only
        #   params that affect build_alcb_snapshot.  Eliminates redundant per-symbol
        #   research across candidates that share the same universe filter.
        # Tier 2 (selection): keyed on (date, as_of, full_settings_sig) — applies
        #   scoring/containment/quality gates on top of the cached snapshot.
        self._alcb_snapshot_cache: dict[tuple[date, date, float, float], alcb_models.ResearchSnapshot] = {}
        self._alcb_selection_cache: dict[tuple[date, date, str], alcb_models.CandidateArtifact] = {}
        # Two-tier IARIC cache (mirrors ALCB pattern above):
        # Tier 1 (snapshot): keyed on (date, min_price, min_adv) — only params
        #   that affect build_iaric_snapshot.
        # Tier 2 (selection): keyed on (date, full_settings_sig) — applies
        #   scoring/filtering gates on top of the cached snapshot.
        self._iaric_snapshot_cache: dict[tuple[date, float, float], iaric_models.ResearchSnapshot] = {}
        self._iaric_selection_cache: dict[tuple[date, str], iaric_models.WatchlistArtifact] = {}

        # Sector return cache: (sector, date, lookback) → float
        self._sector_return_cache: dict[tuple[str, date, int], float] = {}

        # Market/sector research caches — eliminates redundant recomputation
        # when ALCB+IARIC snapshots are built for the same date
        self._market_research_cache: dict[date, alcb_models.MarketResearch] = {}
        self._sector_research_cache: dict[date, dict[str, alcb_models.SectorResearch]] = {}

        # Pre-computed date indices (symbol → (sorted_dates, last_ilocs))
        self._daily_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._ref_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._30m_didx: dict[str, tuple[list[date], list[int]]] = {}
        self._5m_didx: dict[str, tuple[list[date], list[int]]] = {}

        # Pre-computed numpy arrays per symbol
        self._daily_arrs: dict[str, dict[str, np.ndarray]] = {}
        self._ref_arrs: dict[str, dict[str, np.ndarray]] = {}

        # Pre-computed flow proxy arrays
        self._daily_flow: dict[str, np.ndarray] = {}
        self._ref_flow: dict[str, np.ndarray] = {}

        # Pre-computed breadth/participation boolean arrays (O1 optimization)
        self._above_sma20: dict[str, np.ndarray] = {}
        self._vol_above_avg20: dict[str, np.ndarray] = {}

        # Pre-built bar object lists (built once in load_all_data, sliced per query)
        self._alcb_daily_bars: dict[str, list[alcb_models.ResearchDailyBar]] = {}
        self._iaric_daily_bars: dict[str, list[iaric_models.ResearchDailyBar]] = {}
        self._30m_bars: dict[str, list[alcb_models.Bar]] = {}
        self._30m_bar_dates: dict[str, list[date]] = {}  # per-bar dates for bisect
        self._5m_arrays: dict[str, dict[str, np.ndarray]] = {}  # per-symbol numpy arrays for on-demand bar creation
        self._5m_paths: dict[str, Path] = {}  # deferred 5m parquet paths for lazy loading

        self._trading_dates: list[date] = []
        self._data_fingerprint: str | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_all_data(self) -> None:
        """Load all parquet files into memory and pre-compute indices."""
        self._data_fingerprint = None
        all_symbols = [sym for sym, _, _ in self._universe]
        ref_symbols = list(REFERENCE_SYMBOLS)

        loaded = 0
        missing = 0

        # Daily bars for universe
        for sym in all_symbols:
            path = bar_path(self._data_dir, sym, "1d")
            if path.exists():
                df = load_bars(path)
                self._daily_cache[sym] = df
                self._daily_didx[sym] = _build_date_index(df)
                arrs = _precompute_arrays(df)
                self._daily_arrs[sym] = arrs
                self._daily_flow[sym] = _flow_proxy_array(arrs)
                loaded += 1
            else:
                missing += 1

        # Reference symbols (SPY, VIX, HYG, sector ETFs)
        for sym in ref_symbols:
            path = bar_path(self._data_dir, sym, "1d")
            if path.exists():
                df = load_bars(path)
                self._ref_cache[sym] = df
                self._ref_didx[sym] = _build_date_index(df)
                arrs = _precompute_arrays(df)
                self._ref_arrs[sym] = arrs
                self._ref_flow[sym] = _flow_proxy_array(arrs)
            # Also check if it's in the universe cache
            if sym in self._daily_cache and sym not in self._ref_cache:
                self._ref_cache[sym] = self._daily_cache[sym]
                self._ref_didx[sym] = self._daily_didx[sym]
                self._ref_arrs[sym] = self._daily_arrs[sym]
                self._ref_flow[sym] = self._daily_flow[sym]

        # Intraday bars (optional — only needed for Tier 2)
        for sym in all_symbols:
            path_30m = bar_path(self._data_dir, sym, "30m")
            if path_30m.exists():
                df = load_bars(path_30m)
                self._intraday_30m_cache[sym] = df
                self._30m_didx[sym] = _build_date_index(df)
            path_5m = bar_path(self._data_dir, sym, "5m")
            if path_5m.exists():
                self._5m_paths[sym] = path_5m  # defer loading to first access

        # Build trading date calendar from SPY
        spy_didx = self._ref_didx.get("SPY")
        if spy_didx:
            self._trading_dates = list(spy_didx[0])

        logger.info(
            "ResearchReplayEngine loaded: %d daily, %d ref, %d 30m, %d 5m (lazy), %d trading dates",
            loaded, len(self._ref_cache), len(self._intraday_30m_cache),
            len(self._5m_paths), len(self._trading_dates),
        )
        if missing > 0:
            logger.warning("Missing daily data for %d symbols", missing)

        # Pre-build bar object lists (built once, sliced per query — major perf win)
        self._prebuild_bar_caches()

        # Pre-compute breadth/participation boolean arrays (O1 optimization)
        self._precompute_breadth_arrays()

    def _prebuild_bar_caches(self) -> None:
        """Convert raw arrays/DataFrames to bar object lists once at load time.

        This eliminates per-date object creation overhead in _build_*_research_symbol.
        With 400 symbols × 500 dates, this reduces ~50M object creations to ~800K.
        """
        # Daily bars — ALCB and IARIC types
        for sym, arrs in self._daily_arrs.items():
            didx = self._daily_didx[sym]
            sorted_dates, last_ilocs = didx
            n = len(arrs["close"])
            opens, highs, lows, closes, volumes = (
                arrs["open"], arrs["high"], arrs["low"], arrs["close"], arrs["volume"],
            )
            bars_a: list[alcb_models.ResearchDailyBar] = [None] * n  # type: ignore[list-item]
            bars_i: list[iaric_models.ResearchDailyBar] = [None] * n  # type: ignore[list-item]
            di = 0
            for i in range(n):
                while di < len(last_ilocs) - 1 and last_ilocs[di] < i:
                    di += 1
                d = sorted_dates[di]
                o, h, l, c, v = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i]), float(volumes[i])
                bars_a[i] = alcb_models.ResearchDailyBar(trade_date=d, open=o, high=h, low=l, close=c, volume=v)
                bars_i[i] = iaric_models.ResearchDailyBar(trade_date=d, open=o, high=h, low=l, close=c, volume=v)
            self._alcb_daily_bars[sym] = bars_a
            self._iaric_daily_bars[sym] = bars_i

        # 30m bars -- pre-convert from DataFrames (eliminates itertuples overhead)
        for sym, df in self._intraday_30m_cache.items():
            if df.empty:
                continue
            timestamps = [t.to_pydatetime() for t in df.index]
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            closes = df["close"].values
            vols = df["volume"].values if "volume" in df.columns else np.zeros(len(df))
            n = len(df)
            bars: list[alcb_models.Bar] = [None] * n  # type: ignore[list-item]
            bar_dates: list[date] = [None] * n  # type: ignore[list-item]
            for i in range(n):
                bars[i] = alcb_models.Bar(
                    symbol=sym, start_time=timestamps[i], end_time=timestamps[i],
                    open=float(opens[i]), high=float(highs[i]), low=float(lows[i]),
                    close=float(closes[i]), volume=float(vols[i]),
                )
                bar_dates[i] = timestamps[i].date()
            self._30m_bars[sym] = bars
            self._30m_bar_dates[sym] = bar_dates

        # 5m bars -- on-demand per-date creation from cached numpy arrays
        # See _ensure_5m_arrays() and get_5m_bar_objects_for_date()

    def _precompute_breadth_arrays(self) -> None:
        """Pre-compute above-SMA20 and volume-above-avg20 boolean arrays for all symbols.

        These replace per-date numpy dispatch in _compute_market_research and
        _compute_sector_research with O(1) boolean lookups.
        """
        for sym, arrs in self._daily_arrs.items():
            self._above_sma20[sym] = _precompute_above_sma(arrs["close"], 20)
            self._vol_above_avg20[sym] = _precompute_vol_above_avg(arrs["volume"], 20)

    @property
    def trading_dates(self) -> list[date]:
        return self._trading_dates

    def data_fingerprint(self) -> str:
        if self._data_fingerprint is None:
            self._data_fingerprint = fingerprint_tree(self._data_dir, patterns=("*.parquet",))
        return self._data_fingerprint

    def _alcb_settings_signature(
        self,
        settings: alcb_config.StrategySettings | None,
    ) -> str:
        cfg = settings or alcb_config.StrategySettings()
        return _json_signature(asdict(cfg))

    def _iaric_settings_signature(
        self,
        settings: iaric_config.StrategySettings | None,
    ) -> str:
        cfg = settings or iaric_config.StrategySettings()
        return _json_signature(asdict(cfg))

    def clear_selection_cache(self) -> None:
        """Clear cached selection and derived research results."""
        self._alcb_snapshot_cache.clear()
        self._alcb_selection_cache.clear()
        self._iaric_snapshot_cache.clear()
        self._iaric_selection_cache.clear()
        self._sector_return_cache.clear()
        self._market_research_cache.clear()
        self._sector_research_cache.clear()

    def _slice_30m_bars(self, sym: str, trade_date: date, lookback_days: int = 90) -> list[alcb_models.Bar]:
        """Fast slice of pre-built 30m bar list using binary search."""
        bars = self._30m_bars.get(sym)
        if not bars:
            return []
        bar_dates = self._30m_bar_dates[sym]
        end_idx = bisect.bisect_right(bar_dates, trade_date)
        if end_idx == 0:
            return []
        start_date = trade_date - timedelta(days=lookback_days)
        start_idx = bisect.bisect_left(bar_dates, start_date)
        return bars[start_idx:end_idx]

    # ------------------------------------------------------------------
    # Fast slice helpers (internal)
    # ------------------------------------------------------------------

    def _get_closes_upto(self, sym: str, trade_date: date, cache: str = "daily") -> np.ndarray | None:
        """Get close prices up to trade_date as numpy array."""
        if cache == "daily":
            didx = self._daily_didx.get(sym)
            arrs = self._daily_arrs.get(sym)
        else:
            didx = self._ref_didx.get(sym)
            arrs = self._ref_arrs.get(sym)
        if didx is None or arrs is None:
            return None
        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < 0:
            return None
        return arrs["close"][:end + 1]

    def _get_arrays_upto(self, sym: str, trade_date: date, cache: str = "daily") -> tuple[dict[str, np.ndarray], int] | None:
        """Get pre-computed arrays and end iloc for symbol up to trade_date."""
        if cache == "daily":
            didx = self._daily_didx.get(sym)
            arrs = self._daily_arrs.get(sym)
        else:
            didx = self._ref_didx.get(sym)
            arrs = self._ref_arrs.get(sym)
        if didx is None or arrs is None:
            return None
        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < 0:
            return None
        return arrs, end

    # ------------------------------------------------------------------
    # Market-level fields
    # ------------------------------------------------------------------

    def _compute_market_research(self, trade_date: date) -> alcb_models.MarketResearch:
        """Compute market-level research fields from SPY, VIX, HYG daily bars."""
        cached = self._market_research_cache.get(trade_date)
        if cached is not None:
            return cached

        # SPY: price_ok = close > SMA200
        price_ok = False
        spy_result = self._get_arrays_upto("SPY", trade_date, "ref")
        spy_closes: np.ndarray | None = None
        spy_vols: np.ndarray | None = None
        if spy_result is not None:
            spy_arrs, spy_end = spy_result
            spy_closes = spy_arrs["close"][:spy_end + 1]
            spy_vols = spy_arrs["volume"][:spy_end + 1]
            if len(spy_closes) >= 200:
                price_ok = float(spy_closes[-1]) > _sma(spy_closes, 200)

        # Breadth: % of universe with close > SMA20 (precomputed boolean lookup)
        above_20 = 0
        total = 0
        for sym in self._daily_arrs:
            didx = self._daily_didx[sym]
            end = _iloc_upto(didx[0], didx[1], trade_date)
            if end < 19:  # need at least 20 bars
                continue
            if self._above_sma20[sym][end]:
                above_20 += 1
            total += 1
        breadth = (above_20 / total * 100.0) if total > 0 else 50.0

        # VIX percentile (1 year)
        vix_percentile = 50.0
        vix_closes = self._get_closes_upto("VIX", trade_date, "ref")
        if vix_closes is not None and len(vix_closes) >= 20:
            window = vix_closes[-252:] if len(vix_closes) >= 252 else vix_closes
            vix_percentile = _percentile(float(vix_closes[-1]), window)

        # HY spread proxy: -1 * HYG 5-day change * 10000
        hy_spread_change = 0.0
        hyg_closes = self._get_closes_upto("HYG", trade_date, "ref")
        if hyg_closes is not None and len(hyg_closes) >= 6:
            change_5d = (float(hyg_closes[-1]) - float(hyg_closes[-6])) / float(hyg_closes[-6])
            hy_spread_change = -1.0 * change_5d * 10000.0

        # Market-wide institutional selling
        mwis = False
        if spy_closes is not None and spy_vols is not None and len(spy_closes) >= 21:
            avg_vol_20 = float(np.mean(spy_vols[-21:-1]))
            if avg_vol_20 > 0:
                mwis = (
                    float(spy_vols[-1]) > 2.0 * avg_vol_20
                    and float(spy_closes[-1]) < float(spy_closes[-2])
                    and breadth < 40.0
                )

        result = alcb_models.MarketResearch(
            price_ok=price_ok,
            breadth_pct_above_20dma=breadth,
            vix_percentile_1y=vix_percentile,
            hy_spread_5d_bps_change=hy_spread_change,
            market_wide_institutional_selling=mwis,
        )
        self._market_research_cache[trade_date] = result
        return result

    # ------------------------------------------------------------------
    # Sector-level fields
    # ------------------------------------------------------------------

    def _compute_sector_research(
        self, trade_date: date,
    ) -> dict[str, alcb_models.SectorResearch]:
        """Compute per-sector research fields."""
        cached = self._sector_research_cache.get(trade_date)
        if cached is not None:
            return cached

        sectors: dict[str, alcb_models.SectorResearch] = {}

        for sector_name, etf_symbol in SECTOR_ETFS.items():
            # Flow trend from sector ETF
            flow_trend_20d = 0.0
            etf_didx = self._ref_didx.get(etf_symbol)
            etf_flow = self._ref_flow.get(etf_symbol)
            if etf_didx is not None and etf_flow is not None:
                end = _iloc_upto(etf_didx[0], etf_didx[1], trade_date)
                if end >= 19:
                    flow_trend_20d = float(np.mean(etf_flow[end - 19:end + 1]))

            # Breadth + Participation in one pass (precomputed boolean lookups)
            sector_syms = self._sector_symbols.get(sector_name, [])
            above = 0
            counted = 0
            participating = 0
            part_counted = 0
            for sym in sector_syms:
                didx = self._daily_didx.get(sym)
                if didx is None:
                    continue
                end = _iloc_upto(didx[0], didx[1], trade_date)
                # Breadth: close > SMA20 (needs >= 20 bars)
                if end >= 19:
                    if self._above_sma20[sym][end]:
                        above += 1
                    counted += 1
                # Participation: volume > avg20 volume (needs >= 21 bars)
                if end >= 20:
                    if self._vol_above_avg20[sym][end]:
                        participating += 1
                    part_counted += 1
            breadth_20d = (above / counted) if counted > 0 else 0.5
            participation = (participating / part_counted) if part_counted > 0 else 0.5

            sectors[sector_name] = alcb_models.SectorResearch(
                name=sector_name,
                flow_trend_20d=flow_trend_20d,
                breadth_20d=breadth_20d,
                participation=participation,
            )

        self._sector_research_cache[trade_date] = sectors
        return sectors

    def _compute_sector_return(
        self, sector: str, trade_date: date, lookback: int,
    ) -> float:
        """Compute sector ETF return over lookback days (cached)."""
        key = (sector, trade_date, lookback)
        cached = self._sector_return_cache.get(key)
        if cached is not None:
            return cached
        etf = SECTOR_ETFS.get(sector)
        if etf is None:
            self._sector_return_cache[key] = 0.0
            return 0.0
        closes = self._get_closes_upto(etf, trade_date, "ref")
        if closes is None or len(closes) < lookback + 1:
            self._sector_return_cache[key] = 0.0
            return 0.0
        old = float(closes[-(lookback + 1)])
        if old <= 0:
            self._sector_return_cache[key] = 0.0
            return 0.0
        val = (float(closes[-1]) - old) / old
        self._sector_return_cache[key] = val
        return val

    # ------------------------------------------------------------------
    # Per-symbol fields
    # ------------------------------------------------------------------

    def _build_alcb_research_symbol(
        self, sym: str, as_of_date: date,
    ) -> alcb_models.ResearchSymbol | None:
        """Build an ALCB ResearchSymbol for one stock as of a causal date."""
        didx = self._daily_didx.get(sym)
        arrs = self._daily_arrs.get(sym)
        if didx is None or arrs is None:
            return None

        end = _iloc_upto(didx[0], didx[1], as_of_date)
        if end < 0:
            return None
        start = max(0, end - 249)
        daily_bars = self._alcb_daily_bars[sym][start:end + 1]
        if len(daily_bars) < 20:
            return None

        closes = arrs["close"][start:end + 1]
        volumes = arrs["volume"][start:end + 1]
        price = float(closes[-1])

        # ADV20 (in USD)
        if len(closes) >= 20:
            adv20_usd = float(np.mean(closes[-20:] * volumes[-20:]))
        else:
            adv20_usd = float(np.mean(closes * volumes))

        # Flow proxy history (last 40 bars) — vectorized
        flow_arr = self._daily_flow[sym]
        flow_start = max(0, end - 39)
        flow_proxy_history = flow_arr[flow_start:end + 1].tolist()

        # Sector returns
        sector = self._sector_map.get(sym, "")
        sector_return_20d = self._compute_sector_return(sector, as_of_date, 20)
        sector_return_60d = self._compute_sector_return(sector, as_of_date, 60)

        # 30m bar data (for Tier 2 / qualify_breakout) — pre-built, fast slice
        bars_30m = self._slice_30m_bars(sym, as_of_date)

        # Average 30m volume
        avg_30m_vol = 0.0
        med_30m_vol = 0.0
        if bars_30m:
            vols = [b.volume for b in bars_30m if b.volume > 0]
            if vols:
                avg_30m_vol = float(np.mean(vols))
                med_30m_vol = float(np.median(vols))

        # Intraday ATR seed from daily ATR
        intraday_atr_seed = 0.0
        if len(daily_bars) >= 15:
            trs = []
            for i in range(1, min(15, len(daily_bars))):
                b = daily_bars[-i]
                pb = daily_bars[-(i + 1)]
                trs.append(max(b.high - b.low, abs(b.high - pb.close), abs(b.low - pb.close)))
            if trs:
                intraday_atr_seed = fmean(trs) / max(price, 1e-9)

        cfg = self._universe_config
        return alcb_models.ResearchSymbol(
            symbol=sym,
            exchange="SMART",
            primary_exchange=self._exchange_map.get(sym, "NASDAQ"),
            currency="USD",
            tick_size=0.01,
            point_value=1.0,
            sector=sector,
            price=price,
            adv20_usd=adv20_usd,
            median_spread_pct=cfg.default_spread_pct / 100.0,
            earnings_within_sessions=cfg.default_earnings_sessions,
            blacklist_flag=False,
            halted_flag=False,
            severe_news_flag=False,
            etf_flag=sym in KNOWN_ETFS,
            adr_flag=False,
            preferred_flag=False,
            otc_flag=False,
            hard_to_borrow_flag=False,
            biotech_flag=False,
            flow_proxy_history=flow_proxy_history,
            daily_bars=daily_bars,
            bars_30m=bars_30m,
            sector_return_20d=sector_return_20d,
            sector_return_60d=sector_return_60d,
            intraday_atr_seed=intraday_atr_seed,
            average_30m_volume=avg_30m_vol,
            median_30m_volume=med_30m_vol,
            expected_5m_volume=avg_30m_vol / 6.0 if avg_30m_vol > 0 else 0.0,
        )

    def _build_iaric_research_symbol(
        self, sym: str, trade_date: date,
    ) -> iaric_models.ResearchSymbol | None:
        """Build an IARIC ResearchSymbol for one stock on one date."""
        didx = self._daily_didx.get(sym)
        arrs = self._daily_arrs.get(sym)
        if didx is None or arrs is None:
            return None

        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < 0:
            return None
        start = max(0, end - 249)
        daily_bars = self._iaric_daily_bars[sym][start:end + 1]
        if len(daily_bars) < 20:
            return None

        closes = arrs["close"][start:end + 1]
        volumes = arrs["volume"][start:end + 1]
        price = float(closes[-1])

        if len(closes) >= 20:
            adv20_usd = float(np.mean(closes[-20:] * volumes[-20:]))
        else:
            adv20_usd = float(np.mean(closes * volumes))

        # Flow proxy history (vectorized)
        flow_arr = self._daily_flow[sym]
        flow_start = max(0, end - 39)
        flow_proxy_history = flow_arr[flow_start:end + 1].tolist()

        sector = self._sector_map.get(sym, "")
        sector_return_20d = self._compute_sector_return(sector, trade_date, 20)
        sector_return_60d = self._compute_sector_return(sector, trade_date, 60)

        # Intraday ATR seed
        intraday_atr_seed = 0.0
        if len(daily_bars) >= 15:
            trs = []
            for i in range(1, min(15, len(daily_bars))):
                b = daily_bars[-i]
                pb = daily_bars[-(i + 1)]
                trs.append(max(b.high - b.low, abs(b.high - pb.close), abs(b.low - pb.close)))
            if trs:
                intraday_atr_seed = fmean(trs) / max(price, 1e-9)

        # Average 30m volume (from 30m bars if available)
        avg_30m_vol = 0.0
        didx_30m = self._30m_didx.get(sym)
        if didx_30m is not None:
            arrs_30m = self._intraday_30m_cache[sym]
            end_30m = _iloc_upto(didx_30m[0], didx_30m[1], trade_date)
            if end_30m >= 0:
                avg_30m_vol = float(arrs_30m["volume"].values[:end_30m + 1].mean())

        cfg = self._universe_config
        return iaric_models.ResearchSymbol(
            symbol=sym,
            exchange="SMART",
            primary_exchange=self._exchange_map.get(sym, "NASDAQ"),
            currency="USD",
            tick_size=0.01,
            point_value=1.0,
            sector=sector,
            price=price,
            adv20_usd=adv20_usd,
            median_spread_pct=cfg.default_spread_pct / 100.0,
            earnings_within_sessions=cfg.default_earnings_sessions,
            blacklist_flag=False,
            halted_flag=False,
            severe_news_flag=False,
            etf_flag=sym in KNOWN_ETFS,
            adr_flag=False,
            preferred_flag=False,
            otc_flag=False,
            hard_to_borrow_flag=False,
            flow_proxy_history=flow_proxy_history,
            daily_bars=daily_bars,
            sector_return_20d=sector_return_20d,
            sector_return_60d=sector_return_60d,
            intraday_atr_seed=intraday_atr_seed,
            average_30m_volume=avg_30m_vol,
            expected_5m_volume=avg_30m_vol / 6.0 if avg_30m_vol > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Snapshot builders
    # ------------------------------------------------------------------

    def _skip_symbol_prefilter(
        self, sym: str, trade_date: date,
        min_price: float | None, min_adv_usd: float | None,
    ) -> bool:
        """Cheap pre-filter: return True if symbol should be skipped.

        Avoids expensive _build_*_research_symbol for symbols that will
        be rejected anyway (missing data, ETFs, low price, low ADV).
        """
        didx = self._daily_didx.get(sym)
        arrs = self._daily_arrs.get(sym)
        if didx is None or arrs is None:
            return True
        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < 19:
            return True
        if sym in KNOWN_ETFS:
            return True
        if min_price is not None and float(arrs["close"][end]) < min_price:
            return True
        if min_adv_usd is not None:
            c = arrs["close"][end - 19:end + 1]
            v = arrs["volume"][end - 19:end + 1]
            if float(np.dot(c, v)) / 20 < min_adv_usd:
                return True
        return False

    def build_alcb_snapshot(
        self,
        trade_date: date,
        *,
        min_price: float | None = None,
        min_adv_usd: float | None = None,
        as_of_date: date | None = None,
    ) -> alcb_models.ResearchSnapshot:
        """Build a complete ALCB ResearchSnapshot for *trade_date*.

        Optional pre-filters skip expensive _build_alcb_research_symbol for
        symbols that will be rejected anyway (ETFs, low price, low ADV).
        Without optional params, builds all symbols as before.
        """
        effective_date = as_of_date or trade_date
        market = self._compute_market_research(effective_date)
        sectors = self._compute_sector_research(effective_date)
        use_filter = min_price is not None or min_adv_usd is not None

        symbols: dict[str, alcb_models.ResearchSymbol] = {}
        for sym, _, _ in self._universe:
            if use_filter and self._skip_symbol_prefilter(sym, effective_date, min_price, min_adv_usd):
                continue
            rs = self._build_alcb_research_symbol(sym, effective_date)
            if rs is not None:
                symbols[sym] = rs

        return alcb_models.ResearchSnapshot(
            trade_date=trade_date,
            market=market,
            sectors=sectors,
            symbols=symbols,
            held_positions=[],  # Populated by daily engine during backtest
        )

    def build_iaric_snapshot(
        self, trade_date: date, *, min_price: float | None = None, min_adv_usd: float | None = None,
    ) -> iaric_models.ResearchSnapshot:
        """Build a complete IARIC ResearchSnapshot for *trade_date*.

        Optional pre-filters skip expensive _build_iaric_research_symbol for
        symbols that will be rejected anyway (ETFs, low price, low ADV).
        Without optional params, builds all symbols as before.
        """
        # IARIC uses identical MarketResearch / SectorResearch shapes
        alcb_market = self._compute_market_research(trade_date)
        market = iaric_models.MarketResearch(
            price_ok=alcb_market.price_ok,
            breadth_pct_above_20dma=alcb_market.breadth_pct_above_20dma,
            vix_percentile_1y=alcb_market.vix_percentile_1y,
            hy_spread_5d_bps_change=alcb_market.hy_spread_5d_bps_change,
            market_wide_institutional_selling=alcb_market.market_wide_institutional_selling,
        )

        alcb_sectors = self._compute_sector_research(trade_date)
        sectors: dict[str, iaric_models.SectorResearch] = {}
        for name, sr in alcb_sectors.items():
            sectors[name] = iaric_models.SectorResearch(
                name=sr.name,
                flow_trend_20d=sr.flow_trend_20d,
                breadth_20d=sr.breadth_20d,
                participation=sr.participation,
            )

        use_filter = min_price is not None or min_adv_usd is not None
        symbols: dict[str, iaric_models.ResearchSymbol] = {}
        for sym, _, _ in self._universe:
            if use_filter and self._skip_symbol_prefilter(sym, trade_date, min_price, min_adv_usd):
                continue
            rs = self._build_iaric_research_symbol(sym, trade_date)
            if rs is not None:
                symbols[sym] = rs

        snapshot = iaric_models.ResearchSnapshot(
            trade_date=trade_date,
            market=market,
            sectors=sectors,
            symbols=symbols,
            held_positions=[],
        )
        return snapshot

    # ------------------------------------------------------------------
    # Selection wrappers
    # ------------------------------------------------------------------

    def run_alcb_selection(
        self,
        snapshot: alcb_models.ResearchSnapshot,
        settings: alcb_config.StrategySettings | None = None,
    ) -> alcb_models.CandidateArtifact:
        """Run ALCB nightly selection on a replay snapshot."""
        return alcb_research.daily_selection_from_snapshot(
            snapshot=snapshot,
            settings=settings,
            diagnostics=None,
        )

    def run_iaric_selection(
        self,
        snapshot: iaric_models.ResearchSnapshot,
        settings: iaric_config.StrategySettings | None = None,
    ) -> iaric_models.WatchlistArtifact:
        """Run IARIC nightly selection on a replay snapshot."""
        return iaric_research.daily_selection_from_snapshot(
            snapshot=snapshot,
            settings=settings,
            diagnostics=None,
        )

    # ------------------------------------------------------------------
    # Convenience: run selection for a date
    # ------------------------------------------------------------------

    def alcb_selection_for_date(
        self,
        trade_date: date,
        settings: alcb_config.StrategySettings | None = None,
        *,
        as_of_date: date | None = None,
    ) -> alcb_models.CandidateArtifact:
        """Build snapshot + run ALCB selection in one call.

        By default, a trade on ``trade_date`` uses the previous trading day
        close as its research as-of date so the selection is fully causal.

        Uses a two-tier cache for auto-optimization efficiency:
          Tier 1 -- snapshot cache keyed on (date, as_of, min_price, min_adv).
            build_alcb_snapshot only depends on these four values, so candidates
            that differ only in downstream params (stops, sizing, exits) reuse
            the same snapshot without rebuilding per-symbol research.
          Tier 2 -- selection cache keyed on (date, as_of, full_settings_sig).
            run_alcb_selection applies containment/scoring/quality gates that
            depend on the full StrategySettings.
        """
        s = settings or alcb_config.StrategySettings()
        effective_as_of = as_of_date or self.get_prev_trading_date(trade_date)
        if effective_as_of is None:
            raise ValueError(f"No previous trading date available for {trade_date}")

        # Tier 2: full selection cache (cheapest check first)
        settings_sig = self._alcb_settings_signature(s)
        selection_key = (trade_date, effective_as_of, settings_sig)
        cached = self._alcb_selection_cache.get(selection_key)
        if cached is not None:
            return cached

        # Tier 1: snapshot cache (expensive build_alcb_snapshot)
        snapshot_key = (trade_date, effective_as_of, s.min_price, s.min_adv_usd)
        snapshot = self._alcb_snapshot_cache.get(snapshot_key)
        if snapshot is None:
            snapshot = self.build_alcb_snapshot(
                trade_date,
                min_price=s.min_price,
                min_adv_usd=s.min_adv_usd,
                as_of_date=effective_as_of,
            )
            self._alcb_snapshot_cache[snapshot_key] = snapshot

        result = self.run_alcb_selection(snapshot, settings)
        self._alcb_selection_cache[selection_key] = result
        return result

    def iaric_selection_for_date(
        self,
        trade_date: date,
        settings: iaric_config.StrategySettings | None = None,
    ) -> iaric_models.WatchlistArtifact:
        """Build snapshot + run IARIC selection in one call.

        Uses a two-tier cache for auto-optimization efficiency:
          Tier 1 -- snapshot cache keyed on (date, min_price, min_adv).
            build_iaric_snapshot only depends on these values, so candidates
            that differ only in downstream params (scoring, exits) reuse
            the same snapshot without rebuilding per-symbol research.
          Tier 2 -- selection cache keyed on (date, full_settings_sig).
            run_iaric_selection applies filtering/scoring gates that
            depend on the full StrategySettings.
        """
        s = settings or iaric_config.StrategySettings()

        # Tier 2: full selection cache (cheapest check first)
        settings_sig = self._iaric_settings_signature(s)
        selection_key = (trade_date, settings_sig)
        cached = self._iaric_selection_cache.get(selection_key)
        if cached is not None:
            return cached

        # Tier 1: snapshot cache (expensive build_iaric_snapshot)
        snapshot_key = (trade_date, s.min_price, s.min_adv_usd)
        snapshot = self._iaric_snapshot_cache.get(snapshot_key)
        if snapshot is None:
            snapshot = self.build_iaric_snapshot(
                trade_date, min_price=s.min_price, min_adv_usd=s.min_adv_usd,
            )
            self._iaric_snapshot_cache[snapshot_key] = snapshot

        result = self.run_iaric_selection(snapshot, settings)
        self._iaric_selection_cache[selection_key] = result
        return result

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def get_warmup_end_date(self, warmup_days: int = 250) -> date | None:
        """Return the first date with enough warmup history."""
        if len(self._trading_dates) <= warmup_days:
            return None
        return self._trading_dates[warmup_days]

    def tradable_dates(self, start: date, end: date) -> list[date]:
        """Return trading dates in [start, end]."""
        lo = bisect.bisect_left(self._trading_dates, start)
        hi = bisect.bisect_right(self._trading_dates, end)
        return self._trading_dates[lo:hi]

    def get_daily_close(self, symbol: str, trade_date: date) -> float | None:
        """Get a symbol's closing price on a specific date."""
        didx = self._daily_didx.get(symbol)
        arrs = self._daily_arrs.get(symbol)
        if didx is None or arrs is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        return float(arrs["close"][bounds[1]])

    def get_daily_ohlc(
        self, symbol: str, trade_date: date,
    ) -> tuple[float, float, float, float] | None:
        """Get (open, high, low, close) for a symbol on a date."""
        didx = self._daily_didx.get(symbol)
        arrs = self._daily_arrs.get(symbol)
        if didx is None or arrs is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        i = bounds[1]
        return (float(arrs["open"][i]), float(arrs["high"][i]),
                float(arrs["low"][i]), float(arrs["close"][i]))

    def get_flow_proxy_last_n(
        self, symbol: str, trade_date: date, n: int = 2,
    ) -> list[float] | None:
        """Get last *n* flow proxy values up to *trade_date* via direct array lookup.

        O(1) per call — avoids building a full 415-symbol snapshot just
        to check flow reversal on 1-3 carry positions.
        """
        flow = self._daily_flow.get(symbol)
        didx = self._daily_didx.get(symbol)
        if flow is None or didx is None:
            return None
        end = _iloc_upto(didx[0], didx[1], trade_date)
        if end < n - 1:
            return None
        return flow[end - n + 1:end + 1].tolist()

    def get_next_open(self, symbol: str, trade_date: date) -> float | None:
        """Get the next trading day's open price after *trade_date*."""
        didx = self._daily_didx.get(symbol)
        arrs = self._daily_arrs.get(symbol)
        if didx is None or arrs is None:
            return None
        i = _iloc_after(didx[0], didx[1], trade_date)
        if i < 0 or i >= len(arrs["open"]):
            return None
        return float(arrs["open"][i])

    def get_next_trading_date(self, trade_date: date) -> date | None:
        """Return the trading date after *trade_date*."""
        idx = bisect.bisect_right(self._trading_dates, trade_date)
        if idx >= len(self._trading_dates):
            return None
        return self._trading_dates[idx]

    def get_prev_trading_date(self, trade_date: date) -> date | None:
        """Return the trading date before *trade_date*."""
        idx = bisect.bisect_left(self._trading_dates, trade_date)
        if idx <= 0:
            return None
        return self._trading_dates[idx - 1]

    def get_30m_bars_for_date(
        self, symbol: str, trade_date: date,
    ) -> pd.DataFrame | None:
        """Get 30m bars for a single trading day."""
        didx = self._30m_didx.get(symbol)
        df = self._intraday_30m_cache.get(symbol)
        if didx is None or df is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        return df.iloc[bounds[0]:bounds[1] + 1]

    def get_5m_bars_for_date(
        self, symbol: str, trade_date: date,
    ) -> pd.DataFrame | None:
        """Get 5m bars for a single trading day (raw DataFrame)."""
        didx = self._5m_didx.get(symbol)
        df = self._intraday_5m_cache.get(symbol)
        if didx is None or df is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        return df.iloc[bounds[0]:bounds[1] + 1]

    def get_5m_arrays_for_date(
        self, symbol: str, trade_date: date,
    ) -> dict[str, np.ndarray] | None:
        """Get 5m OHLCV as numpy arrays for a single trading day.

        ~100x faster than DataFrame .iloc access for per-bar iteration
        in diagnostic functions.
        """
        self._ensure_5m_arrays(symbol)
        didx = self._5m_didx.get(symbol)
        arrs = self._5m_arrays.get(symbol)
        if didx is None or arrs is None:
            return None
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return None
        first, last = bounds
        s = slice(first, last + 1)
        return {
            "open": arrs["opens"][s],
            "high": arrs["highs"][s],
            "low": arrs["lows"][s],
            "close": arrs["closes"][s],
        }

    def get_30m_bar_objects_for_date(
        self, symbol: str, trade_date: date,
    ) -> list[alcb_models.Bar]:
        """Get pre-built ALCB 30m Bar objects for a single trading day.

        Returns a list slice from the pre-built cache -- zero object creation.
        """
        didx = self._30m_didx.get(symbol)
        bars = self._30m_bars.get(symbol)
        if didx is None or bars is None:
            return []
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return []
        return bars[bounds[0]:bounds[1] + 1]

    def _ensure_5m_data_loaded(self, symbol: str) -> None:
        """Lazily load 5m parquet + build date index on first access."""
        if symbol in self._intraday_5m_cache or symbol not in self._5m_paths:
            return
        path = self._5m_paths[symbol]
        df = load_bars(path)
        self._intraday_5m_cache[symbol] = df
        self._5m_didx[symbol] = _build_date_index(df)

    def _ensure_5m_arrays(self, symbol: str) -> None:
        """Lazily load parquet + extract numpy arrays for on-demand Bar creation.

        Extracts OHLCV arrays and timestamps once per symbol.  Bar objects are
        created on-demand per date in get_5m_bar_objects_for_date(), reducing
        object creation from ~7.7M (all bars for all symbols) to ~42K (only
        bars for actually-traded symbol-days).  ~90s → ~1s.
        """
        if symbol in self._5m_arrays:
            return
        self._ensure_5m_data_loaded(symbol)
        df = self._intraday_5m_cache.get(symbol)
        if df is None or df.empty:
            return
        self._5m_arrays[symbol] = {
            "opens": df["open"].values,
            "highs": df["high"].values,
            "lows": df["low"].values,
            "closes": df["close"].values,
            "vols": df["volume"].values if "volume" in df.columns else np.zeros(len(df)),
            "index": df.index,  # keep as DatetimeIndex -- convert only per-day slice
        }

    def get_5m_bar_objects_for_date(
        self, symbol: str, trade_date: date,
    ) -> list[iaric_models.Bar]:
        """Get IARIC 5m Bar objects for a single trading day.

        Creates Bar objects on-demand from cached numpy arrays -- only ~78 bars
        per call instead of pre-building ~78K bars per symbol.
        """
        self._ensure_5m_arrays(symbol)
        didx = self._5m_didx.get(symbol)
        arrs = self._5m_arrays.get(symbol)
        if didx is None or arrs is None:
            return []
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            return []
        first, last = bounds
        n = last - first + 1
        opens = arrs["opens"]
        highs = arrs["highs"]
        lows = arrs["lows"]
        closes = arrs["closes"]
        vols = arrs["vols"]
        # Convert only this day's ~78 timestamps (not all 78K per symbol)
        ts_slice = arrs["index"][first : last + 1].to_pydatetime()
        td5 = timedelta(minutes=5)
        bars: list[iaric_models.Bar] = [None] * n  # type: ignore[list-item]
        for i in range(n):
            idx = first + i
            bars[i] = iaric_models.Bar(
                symbol=symbol, start_time=ts_slice[i],
                end_time=ts_slice[i] + td5,
                open=float(opens[idx]), high=float(highs[idx]),
                low=float(lows[idx]), close=float(closes[idx]),
                volume=float(vols[idx]),
            )
        return bars

    def get_4h_bars_up_to(
        self, symbol: str, trade_date: date, max_bars: int = 100,
    ) -> list[alcb_models.Bar]:
        """Aggregate 30m bars into 4h candles up to (and including) trade_date.

        Each 4h bar is built from 8 consecutive 30m bars.  Returns at most
        ``max_bars`` 4h candles (most recent last).

        Uses pre-built 30m Bar objects from cache -- zero per-bar object creation.
        """
        didx = self._30m_didx.get(symbol)
        bars_all = self._30m_bars.get(symbol)
        if didx is None or bars_all is None:
            return []

        # Find iloc upper bound for trade_date
        bounds = _iloc_on(didx[0], didx[1], trade_date)
        if bounds is None:
            # trade_date not found; use all data before it
            idx = bisect.bisect_right(didx[0], trade_date)
            if idx == 0:
                return []
            end_iloc = didx[1][idx - 1]
        else:
            end_iloc = bounds[1]

        # Grab enough 30m bars to produce max_bars 4h bars (8 bars each)
        need_30m = max_bars * 8
        start_iloc = max(0, end_iloc + 1 - need_30m)
        raw_bars = bars_all[start_iloc : end_iloc + 1]
        if not raw_bars:
            return []

        # Aggregate into 4h (groups of 8)
        bars_4h: list[alcb_models.Bar] = []
        n_complete = (len(raw_bars) // 8) * 8
        for i in range(0, n_complete, 8):
            group = raw_bars[i : i + 8]
            bars_4h.append(alcb_models.Bar(
                symbol=symbol,
                start_time=group[0].start_time,
                end_time=group[-1].end_time,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            ))

        return bars_4h[-max_bars:]
