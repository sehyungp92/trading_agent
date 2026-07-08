"""LiveDataProvider: assembles macro_df, market_df, strat_ret_df from IBKR + FRED + cached parquets."""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FRED series and ETF symbols for live data overlay
_FRED_SERIES = {
    "VIX": "VIXCLS",
    "SPREAD": "BAMLH0A0HYM2",
    "SLOPE_10Y2Y": "T10Y2Y",
    "INFLATION": "T10YIE",
    "REAL_RATE_10Y": "DFII10",
}

_ETF_SYMBOLS = ["SPY", "EFA", "TLT", "GLD", "BIL", "DBC"]

_MARKET_FRED_COLS = ["VIX", "SPREAD", "SLOPE_10Y2Y", "REAL_RATE_10Y"]
_MACRO_GROWTH_SERIES = "ICSA"
_RETURN_AS_OF_COLS = ["SPY", "EFA", "TLT", "GLD"]
_REQUIRED_MARKET_COLS = ["VIX", "SPREAD", "SLOPE_10Y2Y"]
_REQUIRED_MACRO_COLS = ["GROWTH", "INFLATION"]
_MIN_HMM_LIVE_OBS = 252
_MAX_RETURN_DATA_LAG_DAYS = 10


class LiveDataProvider:
    """Assembles macro_df, market_df, strat_ret_df from IBKR + FRED + cached parquets."""

    def __init__(
        self,
        ib_session: Any,
        data_dir: Path,
        *,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._session = ib_session
        self._data_dir = data_dir
        self._now_provider = now_provider
        self._contracts: dict[str, Any] = {}
        self.last_data_as_of = ""
        self.last_data_status = ""

    async def qualify_contracts(self) -> None:
        """Qualify ETF contracts once at startup."""
        from ib_async import Stock

        for sym in _ETF_SYMBOLS:
            try:
                contract = Stock(sym, "SMART", "USD")
                qualified = await self._session.ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[sym] = qualified[0]
            except Exception as e:
                logger.warning("Regime: could not qualify %s: %s", sym, e)

        logger.info("Regime: qualified %d/%d ETF contracts", len(self._contracts), len(_ETF_SYMBOLS))

    async def build_live_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached parquets, overlay fresh IBKR + FRED data, return 3 DataFrames."""
        # 1. Bootstrap and load cached baseline
        bootstrap_status = _bootstrap_seed_status(self._data_dir)
        macro_df, market_df, strat_ret_df = self._load_cached()
        manifest_ok, manifest_status, _ = _seed_manifest_status(self._data_dir)
        if not manifest_ok:
            raise RuntimeError(
                "Regime seed manifest validation failed before live overlay: "
                f"{manifest_status}"
            )
        seed_return_date = _latest_common_return_as_of(strat_ret_df)
        ibkr_duration = _overlay_duration_str(seed_return_date, self._now())
        fred_start = _overlay_start_date(seed_return_date, self._now())
        status_parts: list[str] = [
            "cache=loaded",
            bootstrap_status,
            manifest_status,
            f"ibkr_window={ibkr_duration.replace(' ', '')}",
            f"fred_start={fred_start}",
        ]

        # 2. Fetch IBKR daily bars and overlay
        ibkr_prices = await _call_async_with_optional_kwargs(
            self._fetch_ibkr_bars,
            duration_str=ibkr_duration,
        )
        if ibkr_prices is not None and not ibkr_prices.empty:
            strat_ret_df, market_df = self._overlay_ibkr(
                ibkr_prices, strat_ret_df, market_df,
            )
            status_parts.append("ibkr=fresh_completed")
        else:
            status_parts.append("ibkr=cached")

        # 3. Fetch FRED macro data and overlay
        loop = asyncio.get_running_loop()
        try:
            fred_data = await loop.run_in_executor(
                None,
                lambda: _call_sync_with_optional_kwargs(
                    self._fetch_fred,
                    start=fred_start,
                ),
            )
            if fred_data is not None:
                fred_df, icsa_raw = fred_data
                macro_df, market_df = self._overlay_fred(
                    fred_df, icsa_raw, macro_df, market_df,
                )
                status_parts.append("fred=fresh")
            else:
                status_parts.append("fred=cached")
        except Exception:
            logger.warning("Regime: FRED fetch failed, using cached macro data", exc_info=True)
            status_parts.append("fred=cached")

        latest_return_date = _latest_common_return_as_of(strat_ret_df)
        if latest_return_date is None:
            raise RuntimeError(
                "Regime live data has no common completed return date for "
                f"{','.join(_RETURN_AS_OF_COLS)}"
            )

        # 4. Align to the latest completed common ETF return date. FRED often
        # updates on a date where IBKR daily bars are not yet available; do not
        # publish a weekly HMM signal on synthetic zero ETF returns.
        all_dates = macro_df.index.union(market_df.index).union(strat_ret_df.index)
        all_dates = all_dates[all_dates <= latest_return_date]
        macro_df = macro_df.reindex(all_dates).ffill()
        market_df = market_df.reindex(all_dates).ffill()
        strat_ret_df = strat_ret_df.reindex(all_dates)
        if "CASH" in strat_ret_df.columns:
            strat_ret_df["CASH"] = strat_ret_df["CASH"].fillna(0.0)

        # Drop leading rows where macro features are NaN
        first_valid = macro_df.dropna(how="any").index.min()
        if first_valid is not None:
            macro_df = macro_df.loc[first_valid:]
            market_df = market_df.loc[first_valid:]
            strat_ret_df = strat_ret_df.loc[first_valid:]

        lag_days = max(0, (self._now().date() - latest_return_date.date()).days)
        self.last_data_as_of = latest_return_date.date().isoformat()
        if lag_days > 0:
            status_parts.append(f"return_lag_days={lag_days}")
        backlog_status, backlog_ok = _hmm_backlog_status(
            macro_df,
            market_df,
            strat_ret_df,
            latest_return_date,
        )
        status_parts.append(backlog_status)
        self.last_data_status = ";".join(status_parts)
        if not backlog_ok:
            raise RuntimeError(
                "Regime live backlog is insufficient after cache + IBKR/FRED backfill: "
                f"data_as_of={self.last_data_as_of}, status={self.last_data_status}"
            )
        if lag_days > _MAX_RETURN_DATA_LAG_DAYS:
            raise RuntimeError(
                "Regime completed return data is stale: "
                f"data_as_of={self.last_data_as_of}, lag_days={lag_days}, "
                f"status={self.last_data_status}"
            )

        # 5. Save updated parquets back to data_dir
        try:
            macro_df.to_parquet(self._data_dir / "macro_df.parquet")
            market_df.to_parquet(self._data_dir / "market_df.parquet")
            strat_ret_df.to_parquet(self._data_dir / "strat_ret_df.parquet")
            from regime.seed_manifest import write_seed_manifest

            write_seed_manifest(
                self._data_dir,
                generated_by="regime.live.provider.LiveDataProvider",
                source_versions={
                    "runtime_overlay": {
                        "ibkr": "completed_daily_bars",
                        "fred": "recent_observations",
                    },
                    "status": ";".join(status_parts),
                },
            )
            logger.info("Regime: updated cached parquets in %s", self._data_dir)
        except Exception:
            logger.warning("Regime: failed to update parquet cache", exc_info=True)

        logger.info(
            "Regime: data assembled -- %d rows, range %s to %s",
            len(strat_ret_df),
            strat_ret_df.index.min().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
            strat_ret_df.index.max().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
        )
        return macro_df.copy(), market_df.copy(), strat_ret_df.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cached(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached parquets from data_dir. Raises FileNotFoundError if missing."""
        names = ("macro_df.parquet", "market_df.parquet", "strat_ret_df.parquet")
        for name in names:
            if not (self._data_dir / name).exists():
                raise FileNotFoundError(
                    f"Regime cached parquet '{name}' not found in {self._data_dir}. "
                    "Seed data via Dockerfile COPY or run "
                    "`FRED_API_KEY=<key> python -m backtests.regime.cli download --data-dir data/regime/raw`."
                )
        macro_df = pd.read_parquet(self._data_dir / "macro_df.parquet").copy()
        market_df = pd.read_parquet(self._data_dir / "market_df.parquet").copy()
        strat_ret_df = pd.read_parquet(self._data_dir / "strat_ret_df.parquet").copy()
        keep_cols = [col for col in [*_RETURN_AS_OF_COLS, "CASH"] if col in strat_ret_df.columns]
        stale_cols = [col for col in strat_ret_df.columns if col not in keep_cols]
        if stale_cols:
            logger.info("Regime: dropping stale cached return columns: %s", ",".join(stale_cols))
        return macro_df, market_df, strat_ret_df.loc[:, keep_cols].copy()

    async def _fetch_ibkr_bars(self, duration_str: str = "1 Y") -> pd.DataFrame | None:
        """Fetch 1Y daily bars for all qualified ETF contracts."""
        if not self._contracts:
            logger.warning("Regime: no IBKR contracts qualified, skipping IBKR fetch")
            return None

        all_prices: dict[str, pd.Series] = {}
        for sym, contract in self._contracts.items():
            try:
                bars = await self._request_daily_bars(
                    contract,
                    duration_str=duration_str,
                )
                if bars:
                    dates = pd.to_datetime([b.date for b in bars])
                    closes = pd.Series([b.close for b in bars], index=dates, name=sym, dtype=float)
                    all_prices[sym] = closes
                    logger.debug("Regime: fetched %d bars for %s", len(bars), sym)
                else:
                    logger.warning("Regime: empty bars for %s", sym)
            except Exception:
                logger.warning("Regime: failed to fetch IBKR bars for %s", sym, exc_info=True)

        if not all_prices:
            return None

        prices_df = pd.DataFrame(all_prices)
        prices_df.index.name = "date"
        # Remove timezone info if present
        if prices_df.index.tz is not None:
            prices_df.index = prices_df.index.tz_localize(None)
        return prices_df

    def _overlay_ibkr(
        self,
        ibkr_prices: pd.DataFrame,
        strat_ret_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Overlay IBKR-derived returns onto cached DataFrames."""
        ibkr_log_ret = np.log(ibkr_prices / ibkr_prices.shift(1))

        col_map = {"SPY": "SPY", "EFA": "EFA", "TLT": "TLT", "GLD": "GLD", "BIL": "CASH"}
        fresh_returns = pd.DataFrame(index=ibkr_log_ret.index)
        for ibkr_col, ret_col in col_map.items():
            if ibkr_col in ibkr_log_ret.columns and ret_col in strat_ret_df.columns:
                fresh_returns[ret_col] = ibkr_log_ret[ibkr_col]
        if not fresh_returns.empty:
            strat_ret_df = _overlay_frame(strat_ret_df, fresh_returns)

        # Overlay DBC log returns into market_df (overlay + extend)
        if "DBC" in ibkr_log_ret.columns and "DBC" in market_df.columns:
            fresh_dbc = ibkr_log_ret["DBC"].dropna()
            overlap = fresh_dbc.index.intersection(market_df.index)
            if len(overlap) > 0:
                market_df.loc[overlap, "DBC"] = fresh_dbc.loc[overlap]
            new_dbc_dates = fresh_dbc.index.difference(market_df.index)
            if len(new_dbc_dates) > 0:
                extension = pd.DataFrame(index=new_dbc_dates, columns=market_df.columns, dtype=float)
                extension["DBC"] = fresh_dbc.loc[new_dbc_dates]
                market_df = pd.concat([market_df, extension]).sort_index()

        logger.info(
            "Regime: IBKR overlay applied -- %d symbols, strat_ret range to %s",
            len(ibkr_log_ret.columns),
            strat_ret_df.index.max().strftime("%Y-%m-%d") if len(strat_ret_df) else "N/A",
        )
        return strat_ret_df, market_df

    async def _request_daily_bars(
        self,
        contract: Any,
        *,
        duration_str: str = "1 Y",
    ) -> Any:
        """Request only completed daily bars, using the runtime session wrapper when available."""
        if hasattr(self._session, "req_historical_data"):
            return await self._session.req_historical_data(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                request_kind="recurring",
                completed_only=True,
                as_of=self._now(),
            )

        bars = await self._session.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        filter_completed = getattr(self._session, "filter_completed_historical_bars", None)
        if bars and callable(filter_completed):
            bars = filter_completed(
                bars,
                bar_size_setting="1 day",
                useRTH=True,
                endDateTime="",
                as_of=self._now(),
            )
        return bars

    def _fetch_fred(self, start: str | None = None) -> tuple[pd.DataFrame, pd.Series] | None:
        """Fetch recent FRED data (blocking -- run in executor)."""
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            logger.warning("Regime: FRED_API_KEY not set, skipping FRED fetch")
            return None

        from fredapi import Fred

        fred = Fred(api_key=key)

        # Only need recent data to overlay on cached baseline
        from datetime import datetime, timedelta
        if start is None:
            start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        # Fetch 5 market series
        frames: dict[str, pd.Series] = {}
        for col, series_id in _FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start)
                s.name = col
                frames[col] = s
            except Exception as exc:
                logger.warning("Regime: FRED fetch failed for %s (%s): %s", col, series_id, exc)

        fred_df = pd.DataFrame(frames)
        fred_df.index = pd.to_datetime(fred_df.index)
        fred_df.index.name = "date"

        # Fetch ICSA (growth proxy) -- standard API, not ALFRED vintage
        icsa_raw = pd.Series(dtype=float)
        try:
            icsa = fred.get_series(_MACRO_GROWTH_SERIES, observation_start=start)
            icsa.index = pd.to_datetime(icsa.index)
            icsa_raw = icsa
        except Exception as exc:
            logger.warning("Regime: FRED fetch failed for ICSA: %s", exc)

        return fred_df, icsa_raw

    def _overlay_fred(
        self,
        fred_df: pd.DataFrame,
        icsa_raw: pd.Series,
        macro_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Overlay fresh FRED data onto cached DataFrames (overlay + extend)."""
        # Overlay + extend market_df columns (VIX, SPREAD, SLOPE_10Y2Y, REAL_RATE_10Y)
        for col in _MARKET_FRED_COLS:
            if col in fred_df.columns and col in market_df.columns:
                fresh = fred_df[col].dropna()
                overlap = fresh.index.intersection(market_df.index)
                if len(overlap) > 0:
                    market_df.loc[overlap, col] = fresh.loc[overlap]
                new_dates = fresh.index.difference(market_df.index)
                if len(new_dates) > 0:
                    extension = pd.DataFrame(index=new_dates, columns=market_df.columns, dtype=float)
                    extension[col] = fresh.loc[new_dates]
                    market_df = pd.concat([market_df, extension]).sort_index()

        # Overlay + extend INFLATION in macro_df (before GROWTH so macro_df index is extended)
        if "INFLATION" in fred_df.columns and "INFLATION" in macro_df.columns:
            fresh_infl = fred_df["INFLATION"].dropna()
            overlap = fresh_infl.index.intersection(macro_df.index)
            if len(overlap) > 0:
                macro_df.loc[overlap, "INFLATION"] = fresh_infl.loc[overlap]
            new_dates = fresh_infl.index.difference(macro_df.index)
            if len(new_dates) > 0:
                extension = pd.DataFrame(index=new_dates, columns=macro_df.columns, dtype=float)
                extension["INFLATION"] = fresh_infl.loc[new_dates]
                macro_df = pd.concat([macro_df, extension]).sort_index()

        # Overlay GROWTH in macro_df (negated ICSA, daily-reindexed with ffill onto extended index)
        if len(icsa_raw) > 0 and "GROWTH" in macro_df.columns:
            growth_fresh = -icsa_raw
            growth_daily = growth_fresh.reindex(macro_df.index, method="ffill").copy()
            valid = growth_daily.dropna()
            if len(valid) > 0:
                macro_df.loc[valid.index, "GROWTH"] = valid

        # Forward-fill gaps
        macro_df = macro_df.ffill()
        market_df = market_df.ffill()

        logger.info("Regime: FRED overlay applied")
        return macro_df, market_df

    def _now(self) -> datetime:
        now = self._now_provider() if self._now_provider is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now


def _latest_common_return_as_of(strat_ret_df: pd.DataFrame) -> pd.Timestamp | None:
    if strat_ret_df.empty:
        return None
    missing = [col for col in _RETURN_AS_OF_COLS if col not in strat_ret_df.columns]
    if missing:
        logger.warning("Regime: return data missing required columns for as-of date: %s", missing)
        return None
    valid = strat_ret_df[_RETURN_AS_OF_COLS].dropna(how="any")
    if valid.empty:
        return None
    return pd.Timestamp(valid.index.max())


def _overlay_frame(base_df: pd.DataFrame, fresh_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.reindex(base_df.index.union(fresh_df.index)).sort_index()
    for col in fresh_df.columns:
        if col not in out.columns:
            out[col] = float("nan")
        fresh = fresh_df[col].dropna()
        if not fresh.empty:
            out.loc[fresh.index, col] = fresh
    return out.sort_index()


def _hmm_backlog_status(
    macro_df: pd.DataFrame,
    market_df: pd.DataFrame,
    strat_ret_df: pd.DataFrame,
    latest_return_date: pd.Timestamp,
) -> tuple[str, bool]:
    problems: list[str] = []
    for name, df, cols in (
        ("macro", macro_df, _REQUIRED_MACRO_COLS),
        ("market", market_df, _REQUIRED_MARKET_COLS),
        ("returns", strat_ret_df, _RETURN_AS_OF_COLS),
    ):
        missing = [col for col in cols if col not in df.columns]
        if missing:
            problems.append(f"{name}_missing={','.join(missing)}")
            continue
        obs = len(df.loc[:latest_return_date, cols].dropna(how="any"))
        if obs < _MIN_HMM_LIVE_OBS:
            problems.append(f"{name}_obs={obs}/{_MIN_HMM_LIVE_OBS}")

    if problems:
        return "backlog=insufficient:" + ",".join(problems), False
    return "backlog=ok", True


def _bootstrap_seed_status(data_dir: Path) -> str:
    seed_dir_raw = os.environ.get("REGIME_SEED_DIR", "").strip()
    if not seed_dir_raw:
        return "seed_bootstrap=disabled"
    from regime.seed_manifest import bootstrap_seed_data_dir

    return bootstrap_seed_data_dir(data_dir, Path(seed_dir_raw))


def _seed_manifest_status(data_dir: Path) -> tuple[bool, str, dict[str, Any] | None]:
    require_manifest = os.environ.get("REGIME_REQUIRE_SEED_MANIFEST", "0").strip().lower()
    require = require_manifest in {"1", "true", "yes", "on"}
    from regime.seed_manifest import validate_seed_data_dir

    return validate_seed_data_dir(
        data_dir,
        require_manifest=require,
        validate_hashes=True,
    )


def _overlay_duration_str(
    seed_return_date: pd.Timestamp | None,
    now: datetime,
    *,
    min_days: int = 14,
    max_days: int = 365,
) -> str:
    if seed_return_date is None:
        return "1 Y"
    lag_days = max(1, (now.date() - seed_return_date.date()).days)
    days = min(max(lag_days + 7, min_days), max_days)
    if days >= 365:
        return "1 Y"
    return f"{days} D"


def _overlay_start_date(
    seed_return_date: pd.Timestamp | None,
    now: datetime,
    *,
    min_days: int = 30,
    default_days: int = 90,
) -> str:
    now_ts = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
    if seed_return_date is None:
        return (now_ts - pd.Timedelta(days=default_days)).date().isoformat()
    seed_ts = pd.Timestamp(seed_return_date).tz_localize(None) if pd.Timestamp(seed_return_date).tzinfo else pd.Timestamp(seed_return_date)
    start = min(
        seed_ts - pd.Timedelta(days=min_days),
        now_ts - pd.Timedelta(days=min_days),
    )
    return start.date().isoformat()


async def _call_async_with_optional_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    if not _accepts_any_kwargs(func):
        return await func()
    return await func(**kwargs)


def _call_sync_with_optional_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    if not _accepts_any_kwargs(func):
        return func()
    return func(**kwargs)


def _accepts_any_kwargs(func: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return any(
        param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD, param.VAR_KEYWORD)
        for param in signature.parameters.values()
    )
