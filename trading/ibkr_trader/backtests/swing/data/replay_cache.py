from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_paths
from backtests.shared.auto.replay_bundle import ReplayBundle

_REPLAY_CACHE: dict[str, ReplayBundle[Any]] = {}
_TPC_CONTEXT_SYMBOLS = {
    "QQQ": "NQ",
    "GLD": "GC",
}


def load_atrss_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.preprocessing import (
        align_daily_to_hourly,
        build_numpy_arrays,
        filter_rth,
        normalize_timezone,
    )
    from backtests.swing.engine.portfolio_engine import PortfolioData

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in symbols
    ]

    start_ts = _coerce_utc_timestamp(start_date)
    end_ts = _coerce_utc_timestamp(end_date, end_of_day=True)

    def _load() -> Any:
        data = PortfolioData()
        for symbol in symbols:
            hourly_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1h.parquet"))
            hourly_df = filter_rth(hourly_df)
            daily_df = normalize_timezone(load_bars(base_dir / f"{symbol}_1d.parquet"))
            hourly_df = _slice_timestamp_index(hourly_df, start_ts, end_ts)
            daily_df = _slice_timestamp_index(daily_df, start_ts, end_ts)
            data.hourly[symbol] = build_numpy_arrays(hourly_df)
            data.daily[symbol] = build_numpy_arrays(daily_df)
            data.daily_idx_maps[symbol] = align_daily_to_hourly(hourly_df, daily_df)
        return data

    return _build_bundle(
        "swing.atrss.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": symbols,
            "start_date": start_ts.isoformat() if start_ts is not None else None,
            "end_date": end_ts.isoformat() if end_ts is not None else None,
        },
        loader=_load,
    )


def _coerce_utc_timestamp(
    value: str | pd.Timestamp | None,
    *,
    end_of_day: bool = False,
) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _slice_timestamp_index(
    df: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    return df


def load_helix_replay_bundle(
    symbols: list[str],
    data_dir: Path,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    from backtests.swing.engine.helix_portfolio_engine import load_helix_data

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_1h.parquet"
        for symbol in symbols
    ] + [
        base_dir / f"{symbol}_1d.parquet"
        for symbol in symbols
    ]
    return _build_bundle(
        "swing.helix.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": tuple(symbols),
            "start_date": _coerce_utc_timestamp(start_date).isoformat() if start_date is not None else None,
            "end_date": _coerce_utc_timestamp(end_date, end_of_day=True).isoformat() if end_date is not None else None,
        },
        loader=lambda: load_helix_data(symbols, base_dir, start_date=start_date, end_date=end_date),
    )


def load_tpc_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    return _load_etf_15m_bundle(
        "swing.tpc.replay_bundle",
        data_dir,
        symbols,
        start_date,
        end_date,
        pullback_timeframe="1h",
    )


def load_tpc_pb30_replay_bundle(
    data_dir: Path,
    *,
    symbols: tuple[str, ...] = ("QQQ", "GLD"),
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> ReplayBundle[Any]:
    """Load TPC data with the pullback window backed by completed 30m bars.

    The TPC core still receives the compatibility key ``bars_1h`` because the
    pullback detector is shared. The bundle cache key includes
    ``pullback_timeframe`` so indicator arrays from the canonical 1h view
    cannot leak into the 30m research view.
    """

    return _load_etf_15m_bundle(
        "swing.tpc.pb30_replay_bundle",
        data_dir,
        symbols,
        start_date,
        end_date,
        pullback_timeframe="30m",
    )


def _load_etf_15m_bundle(
    namespace: str,
    data_dir: Path,
    symbols: tuple[str, ...],
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
    *,
    pullback_timeframe: str = "1h",
) -> ReplayBundle[Any]:
    from backtests.swing.data.cache import load_bars
    from backtests.swing.data.multitimeframe import (
        align_15m_to_30m,
        align_15m_to_1h,
        align_15m_to_4h,
        align_daily_to_15m,
        resample_15m_to_30m,
        resample_1h_to_4h,
    )
    from backtests.swing.data.preprocessing import build_numpy_arrays, normalize_timezone

    base_dir = Path(data_dir)
    source_paths = [
        base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in symbols
        for timeframe in ("15m", "1h", "1d")
    ]
    if namespace.startswith("swing.tpc."):
        source_paths += [
            base_dir / f"{context_symbol}_{timeframe}.parquet"
            for context_symbol in sorted({_TPC_CONTEXT_SYMBOLS.get(symbol, "") for symbol in symbols} - {""})
            for timeframe in ("1h", "1d")
        ]
    start_ts = _coerce_utc_timestamp(start_date)
    end_ts = _coerce_utc_timestamp(end_date, end_of_day=True)

    def _load() -> dict[str, dict[str, Any]]:
        data: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            df15 = normalize_timezone(load_bars(base_dir / f"{symbol}_15m.parquet"))
            df1h = normalize_timezone(load_bars(base_dir / f"{symbol}_1h.parquet"))
            dfd = normalize_timezone(load_bars(base_dir / f"{symbol}_1d.parquet"))
            df15 = _slice_timestamp_index(df15, start_ts, end_ts)
            df1h = _slice_timestamp_index(df1h, start_ts, end_ts)
            dfd = _slice_timestamp_index(dfd, start_ts, end_ts)
            df30 = resample_15m_to_30m(df15)
            df4h = resample_1h_to_4h(df1h)
            if pullback_timeframe == "1h":
                pullback_df = df1h
                idx_pullback = align_15m_to_1h(df15, df1h)
            elif pullback_timeframe == "30m":
                pullback_df = df30
                idx_pullback = align_15m_to_30m(df15, df30)
            else:
                raise ValueError(f"Unsupported TPC pullback_timeframe={pullback_timeframe!r}")
            context_symbol = _TPC_CONTEXT_SYMBOLS.get(symbol, "")
            context_indicators = {}
            if context_symbol:
                path_1h = base_dir / f"{context_symbol}_1h.parquet"
                path_1d = base_dir / f"{context_symbol}_1d.parquet"
                if path_1h.exists() and path_1d.exists():
                    context_1h = normalize_timezone(load_bars(path_1h))
                    context_daily = normalize_timezone(load_bars(path_1d))
                    context_1h = _slice_timestamp_index(context_1h, start_ts, end_ts)
                    context_daily = _slice_timestamp_index(context_daily, start_ts, end_ts)
                    context_indicators = _build_context_indicator_arrays(
                        df15,
                        context_1h,
                        context_daily,
                        align_15m_to_1h=align_15m_to_1h,
                        align_daily_to_15m=align_daily_to_15m,
                    )
            data[symbol] = {
                "bars_15m": build_numpy_arrays(df15),
                "bars_30m": build_numpy_arrays(df30),
                "bars_1h": build_numpy_arrays(pullback_df),
                "bars_4h": build_numpy_arrays(df4h),
                "bars_daily": build_numpy_arrays(dfd),
                "idx_30m": align_15m_to_30m(df15, df30),
                "idx_1h": idx_pullback,
                "idx_4h": align_15m_to_4h(df15, df4h),
                "idx_daily": align_daily_to_15m(df15, dfd),
                "context_symbol": context_symbol,
                "context_indicators": context_indicators,
            }
        return data

    # Phase 4: Keep the shared ETF namespace stable for TPC replay data.
    return _build_bundle(
        "swing.etf_15m_data",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "symbols": tuple(symbols),
            "start_date": start_ts.isoformat() if start_ts is not None else None,
            "end_date": end_ts.isoformat() if end_ts is not None else None,
            "pullback_timeframe": pullback_timeframe,
        },
        loader=_load,
    )


def _build_context_indicator_arrays(
    df15: pd.DataFrame,
    context_1h: pd.DataFrame,
    context_daily: pd.DataFrame,
    *,
    align_15m_to_1h,
    align_daily_to_15m,
) -> dict[str, np.ndarray]:
    if context_1h.empty or context_daily.empty:
        return {}
    close_1h = context_1h["close"].astype(float)
    close_daily = context_daily["close"].astype(float)
    hourly = pd.DataFrame(
        {
            "context_close_1h": close_1h,
            "context_sma20_1h": close_1h.rolling(20, min_periods=20).mean(),
            "context_sma50_1h": close_1h.rolling(50, min_periods=50).mean(),
            "context_ret12_1h": close_1h.pct_change(12),
            "context_ret24_1h": close_1h.pct_change(24),
        },
        index=context_1h.index,
    )
    daily = pd.DataFrame(
        {
            "context_close_daily": close_daily,
            "context_sma20_daily": close_daily.rolling(20, min_periods=20).mean(),
            "context_sma50_daily": close_daily.rolling(50, min_periods=50).mean(),
            "context_ret20_daily": close_daily.pct_change(20),
        },
        index=context_daily.index,
    )
    idx_1h = align_15m_to_1h(df15, context_1h)
    idx_daily = align_daily_to_15m(df15, context_daily)
    out: dict[str, np.ndarray] = {}
    for key in hourly.columns:
        out[key] = _take_aligned(hourly[key].to_numpy(dtype=float), idx_1h)
    for key in daily.columns:
        out[key] = _take_aligned(daily[key].to_numpy(dtype=float), idx_daily)
    return out


def _take_aligned(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan, dtype=float)
    if values.size == 0 or idx.size == 0:
        return out
    mask = (idx >= 0) & (idx < len(values))
    out[mask] = values[idx[mask]]
    return out


def load_unified_portfolio_replay_bundle(config) -> ReplayBundle[Any]:
    """Load the all-swing portfolio replay data behind a source-fingerprinted bundle."""

    from backtests.swing.engine.unified_portfolio_engine import load_unified_data

    base_dir = Path(config.data_dir)
    overlay_symbols = tuple(config.overlay_symbols if config.overlay_enabled else ())
    all_symbols = tuple(
        sorted(
            set(config.atrss_symbols)
            | set(config.helix_symbols)
            | set(getattr(config, "tpc_symbols", ()))
            | set(overlay_symbols)
        )
    )
    source_paths = [
        base_dir / f"{symbol}_{timeframe}.parquet"
        for symbol in all_symbols
        for timeframe in ("1h", "1d")
    ] + [
        base_dir / f"{symbol}_15m.parquet"
        for symbol in sorted(
            set(getattr(config, "tpc_symbols", ()))
        )
    ]
    return _build_bundle(
        "swing.unified.replay_bundle",
        source_paths=source_paths,
        root=base_dir,
        extra={
            "atrss_symbols": tuple(config.atrss_symbols),
            "helix_symbols": tuple(config.helix_symbols),
            "overlay_symbols": overlay_symbols,
            "overlay_enabled": bool(config.overlay_enabled),
        },
        loader=lambda: load_unified_data(config),
    )


def _build_bundle(
    namespace: str,
    *,
    source_paths: list[Path],
    root: Path,
    extra: dict[str, Any],
    loader,
) -> ReplayBundle[Any]:
    source_fingerprint = fingerprint_paths(source_paths, root=root)
    cache_key = build_cache_key(
        namespace,
        source_fingerprint=source_fingerprint,
        extra={
            "data_dir": str(root.resolve()),
            **extra,
        },
    )
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bundle = ReplayBundle(
        data=loader(),
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _REPLAY_CACHE[cache_key] = bundle
    return bundle
