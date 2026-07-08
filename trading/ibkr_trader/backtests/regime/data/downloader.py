"""Data pipeline: FRED + yfinance download, cache as parquet."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs (market-priced or non-revised — safe for standard API)
_FRED_SERIES = {
    "VIX": "VIXCLS",
    "SPREAD": "BAMLH0A0HYM2",
    "SLOPE_10Y2Y": "T10Y2Y",
    "INFLATION": "T10YIE",
    "REAL_RATE_10Y": "DFII10",  # 10-Year Real Interest Rate (TIPS yield)
}

# ICSA requires ALFRED vintage discipline (point-in-time first-release values)
_ICSA_SERIES_ID = "ICSA"

# ETF tickers for yfinance
_ETF_TICKERS = ["SPY", "EFA", "TLT", "GLD", "BIL"]
_FEATURE_ETF_TICKERS = ["DBC"]  # Invesco DB Commodity Index (inception 2006-02)
_ET = ZoneInfo("America/New_York")
_DAILY_BAR_COMPLETE_HOUR_ET = 17
_DAILY_BAR_COMPLETE_MINUTE_ET = 5

def _require_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Set FRED_API_KEY env var (free: https://fred.stlouisfed.org/docs/api/api_key.html)"
        )
    return key


def _download_fred(start: str = "2002-01-01") -> pd.DataFrame:
    """Download all FRED series and return a combined daily DataFrame."""
    from fredapi import Fred

    key = _require_fred_key()
    fred = Fred(api_key=key)
    frames = {}
    for col, series_id in _FRED_SERIES.items():
        logger.info("Downloading FRED %s (%s)", col, series_id)
        s = _fred_call_with_retries(
            lambda: fred.get_series(series_id, observation_start=start),
            label=f"FRED {col} ({series_id})",
        )
        s.name = col
        frames[col] = s

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    return combined


def _download_icsa_vintage(start: str = "2002-01-01") -> pd.Series:
    """Download ICSA using ALFRED vintage API for point-in-time first-release values.

    Standard FRED get_series() returns latest-revised data, which introduces
    look-ahead bias. ALFRED provides every vintage, so we take the earliest
    release for each observation date — exactly what was known at the time.
    """
    from fredapi import Fred

    key = _require_fred_key()
    fred = Fred(api_key=key)

    logger.info("Downloading ICSA via ALFRED (point-in-time vintages)")
    releases = _fred_call_with_retries(
        lambda: fred.get_series_all_releases(_ICSA_SERIES_ID),
        label=f"ALFRED {_ICSA_SERIES_ID}",
    )
    # releases is a DataFrame with columns: realtime_start, date, value
    # or a MultiIndex Series — normalize to DataFrame
    if isinstance(releases, pd.Series):
        releases = releases.reset_index()
        releases.columns = ["realtime_start", "date", "value"]
    elif "realtime_start" not in releases.columns:
        releases = releases.reset_index()

    releases["date"] = pd.to_datetime(releases["date"])
    releases["realtime_start"] = pd.to_datetime(releases["realtime_start"])
    releases["value"] = pd.to_numeric(releases["value"], errors="coerce")

    # Filter to start date
    releases = releases[releases["date"] >= start]

    # For each observation date, take the first release (earliest realtime_start)
    releases = releases.sort_values(["date", "realtime_start"])
    first_release = releases.groupby("date").first()["value"]
    first_release.index.name = "date"
    first_release.name = "GROWTH_RAW"

    logger.info(
        "ICSA vintage: %d observation dates, range %s to %s",
        len(first_release),
        first_release.index.min().strftime("%Y-%m-%d"),
        first_release.index.max().strftime("%Y-%m-%d"),
    )
    return first_release


def _fred_call_with_retries(call, *, label: str, max_attempts: int = 4):
    """Run a FRED/ALFRED request with bounded backoff for transient outages."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            wait = min(2 ** (attempt - 1), 8)
            logger.warning(
                "%s request failed on attempt %d/%d: %s; retrying in %ss",
                label, attempt, max_attempts, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"{label} request failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def _download_etf_prices(start: str = "2002-01-01") -> pd.DataFrame:
    """Download adjusted close prices for ETFs via yfinance."""
    import yfinance as yf

    logger.info("Downloading ETF prices: %s", _ETF_TICKERS)
    data = yf.download(_ETF_TICKERS, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    # Remove timezone info if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return _filter_completed_daily_prices(prices)


def _download_feature_etfs(start: str = "2002-01-01") -> pd.DataFrame:
    """Download feature-only ETF prices (DBC etc.) via yfinance."""
    import yfinance as yf

    if not _FEATURE_ETF_TICKERS:
        return pd.DataFrame()

    logger.info("Downloading feature ETF prices: %s", _FEATURE_ETF_TICKERS)
    data = yf.download(_FEATURE_ETF_TICKERS, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=_FEATURE_ETF_TICKERS[0])
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return _filter_completed_daily_prices(prices)


def _filter_completed_daily_prices(
    prices: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Drop any current-session row before the completed daily-bar cutoff."""
    if prices.empty:
        return prices
    completed = _latest_completed_us_equity_session(now or datetime.now(timezone.utc))
    return prices.loc[prices.index <= completed].copy()


def _latest_completed_us_equity_session(now: datetime) -> pd.Timestamp:
    """Return the latest US weekday whose daily bar should be complete."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_et = now.astimezone(_ET)
    candidate = now_et.date()
    cutoff = now_et.replace(
        hour=_DAILY_BAR_COMPLETE_HOUR_ET,
        minute=_DAILY_BAR_COMPLETE_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    if now_et < cutoff:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return pd.Timestamp(candidate)


def build_all_data(
    data_dir: Path | None = None,
    *,
    write_manifest: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download all data, build the three DataFrames, cache as parquet.

    Returns:
        macro_df: columns [GROWTH, INFLATION]
        market_df: columns [VIX, SPREAD, SLOPE_10Y2Y, REAL_RATE_10Y, DBC]
        strat_ret_df: columns [SPY, EFA, TLT, GLD, CASH]
    """
    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download raw data
    fred_df = _download_fred()
    icsa_vintage = _download_icsa_vintage()
    etf_prices = _download_etf_prices()
    feature_etf_prices = _download_feature_etfs()

    # -- Build market_df --
    market_df = fred_df[["VIX", "SPREAD", "SLOPE_10Y2Y"]].copy()
    # Add REAL_RATE_10Y from FRED (DFII10)
    if "REAL_RATE_10Y" in fred_df.columns:
        market_df["REAL_RATE_10Y"] = fred_df["REAL_RATE_10Y"]
    # Add DBC log returns as a feature column (not an allocation target)
    # DBC inception is 2006-02 — pre-inception dates get 0.0 (no return data)
    if "DBC" in feature_etf_prices.columns:
        dbc_log_ret = np.log(feature_etf_prices["DBC"] / feature_etf_prices["DBC"].shift(1))
        market_df["DBC"] = dbc_log_ret.fillna(0.0)
    market_df = market_df.ffill()

    # -- Build macro_df --
    # Growth: negate ICSA (lower claims = stronger growth)
    # Uses ALFRED first-release values — point-in-time, no look-ahead bias
    # ICSA dates are Saturdays (week-ending) — reindex to daily and forward-fill
    growth_raw = -icsa_vintage
    growth_raw.name = "GROWTH"

    inflation = fred_df["INFLATION"].ffill()
    inflation.name = "INFLATION"

    # Build on INFLATION's daily index, then merge GROWTH with ffill
    macro_df = pd.DataFrame({"INFLATION": inflation})
    macro_df["GROWTH"] = growth_raw.reindex(macro_df.index, method="ffill")
    macro_df = macro_df[["GROWTH", "INFLATION"]]

    # -- Build strat_ret_df --
    # Compute daily log returns from ETF prices
    log_returns = np.log(etf_prices / etf_prices.shift(1))

    strat_ret_df = pd.DataFrame(index=log_returns.index)
    for col in ["SPY", "EFA", "TLT", "GLD"]:
        if col in log_returns.columns:
            strat_ret_df[col] = log_returns[col]

    # CASH = BIL returns
    if "BIL" in log_returns.columns:
        strat_ret_df["CASH"] = log_returns["BIL"]
    else:
        strat_ret_df["CASH"] = 0.0

    # Drop rows before both macro features are available
    # T10YIE starts Jan 2003; ICSA first-release may have a few leading NaN
    common_start = macro_df.dropna(how="any").index.min()
    if common_start is not None:
        macro_df = macro_df.loc[common_start:]
        market_df = market_df.loc[common_start:]
        strat_ret_df = strat_ret_df.loc[common_start:]

    # Drop any all-NaN leading rows
    strat_ret_df = strat_ret_df.dropna(how="all")
    macro_df = macro_df.reindex(strat_ret_df.index).ffill()
    market_df = market_df.reindex(strat_ret_df.index).ffill()

    # Cache
    macro_df.to_parquet(data_dir / "macro_df.parquet")
    market_df.to_parquet(data_dir / "market_df.parquet")
    strat_ret_df.to_parquet(data_dir / "strat_ret_df.parquet")
    if write_manifest:
        from regime.seed_manifest import write_seed_manifest

        write_seed_manifest(
            data_dir,
            generated_by="backtests.regime.data.downloader.build_all_data",
            source_versions=_source_versions(),
        )

    logger.info("Saved 3 parquet files to %s", data_dir)
    return macro_df, market_df, strat_ret_df


def load_cached_data(
    data_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached parquet files. Raises FileNotFoundError if not cached."""
    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")

    macro_df = pd.read_parquet(data_dir / "macro_df.parquet")
    market_df = pd.read_parquet(data_dir / "market_df.parquet")
    strat_ret_df = pd.read_parquet(data_dir / "strat_ret_df.parquet")
    return macro_df, market_df, strat_ret_df


def write_manifest_for_cached_data(data_dir: Path | None = None) -> dict[str, Any]:
    """Write a manifest for already-cached regime seed parquet files."""
    if data_dir is None:
        data_dir = Path("backtests/regime/data/raw")
    from regime.seed_manifest import write_seed_manifest

    return write_seed_manifest(
        Path(data_dir),
        generated_by="backtests.regime.data.downloader.write_manifest_for_cached_data",
        source_versions=_source_versions(),
    )


def _source_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("fredapi", "yfinance", "pandas", "pyarrow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "providers": {
            "fred": _FRED_SERIES,
            "alfred": {"GROWTH": _ICSA_SERIES_ID},
            "price_provider": "yfinance",
            "etf_tickers": list(_ETF_TICKERS),
            "feature_etf_tickers": list(_FEATURE_ETF_TICKERS),
        },
        "packages": packages,
        "fred_api_key_present": bool(os.environ.get("FRED_API_KEY")),
    }
