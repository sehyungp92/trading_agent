from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.shared.auto.cache_keys import build_cache_key, fingerprint_paths
from backtests.shared.auto.replay_bundle import ReplayBundle

_REPLAY_CACHE: dict[str, ReplayBundle[dict[str, Any]]] = {}
_VDUB_REPLAY_CACHE: dict[str, ReplayBundle[dict[str, Any]]] = {}


def load_replay_bundle(
    symbol: str,
    data_dir: Path,
    *,
    include_fifteen_min: bool = False,
    include_thirty_min: bool = True,
    include_hourly: bool = True,
    include_four_hour: bool = True,
    include_daily: bool = True,
    include_daily_es: bool = True,
) -> ReplayBundle[dict[str, Any]]:
    from backtests.momentum.data.cache import load_bars
    from backtests.momentum.data.preprocessing import (
        align_daily_to_5m,
        align_higher_tf_to_5m,
        build_numpy_arrays,
        filter_eth,
        normalize_timezone,
        resample_5m_to_15m,
        resample_5m_to_1h,
        resample_5m_to_30m,
        resample_5m_to_4h,
        resample_5m_to_daily,
    )

    base_dir = Path(data_dir)
    five_min_path = base_dir / f"{symbol}_5m.parquet"
    daily_path = base_dir / f"{symbol}_1d.parquet"
    es_path = base_dir / "ES_1d.parquet"

    source_paths = [five_min_path]
    if include_daily and daily_path.exists():
        source_paths.append(daily_path)
    if include_daily_es and es_path.exists():
        source_paths.append(es_path)

    source_fingerprint = fingerprint_paths(source_paths, root=base_dir)
    cache_key = build_cache_key(
        "momentum.replay_bundle",
        source_fingerprint=source_fingerprint,
        extra={
            "data_dir": str(base_dir.resolve()),
            "symbol": symbol,
            "include_fifteen_min": include_fifteen_min,
            "include_thirty_min": include_thirty_min,
            "include_hourly": include_hourly,
            "include_four_hour": include_four_hour,
            "include_daily": include_daily,
            "include_daily_es": include_daily_es,
        },
    )
    cached = _REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    five_min_df = normalize_timezone(load_bars(five_min_path))
    five_min_df = filter_eth(five_min_df)
    data: dict[str, Any] = {"five_min": build_numpy_arrays(five_min_df)}

    if include_fifteen_min:
        fifteen_min_df = resample_5m_to_15m(five_min_df)
        data["fifteen_min"] = build_numpy_arrays(fifteen_min_df)
        data["fifteen_min_idx_map"] = align_higher_tf_to_5m(five_min_df, fifteen_min_df)

    if include_thirty_min:
        thirty_min_df = resample_5m_to_30m(five_min_df)
        data["thirty_min"] = build_numpy_arrays(thirty_min_df)
        data["thirty_min_idx_map"] = align_higher_tf_to_5m(five_min_df, thirty_min_df)

    if include_hourly:
        hourly_df = resample_5m_to_1h(five_min_df)
        data["hourly"] = build_numpy_arrays(hourly_df)
        data["hourly_idx_map"] = align_higher_tf_to_5m(five_min_df, hourly_df)

    if include_four_hour:
        four_hour_df = resample_5m_to_4h(five_min_df)
        data["four_hour"] = build_numpy_arrays(four_hour_df)
        data["four_hour_idx_map"] = align_higher_tf_to_5m(five_min_df, four_hour_df)

    if include_daily:
        if daily_path.exists():
            daily_df = normalize_timezone(load_bars(daily_path))
        else:
            daily_df = resample_5m_to_daily(five_min_df)
        data["daily"] = build_numpy_arrays(daily_df)
        data["daily_idx_map"] = align_daily_to_5m(five_min_df, daily_df)

    if include_daily_es:
        if es_path.exists():
            es_df = normalize_timezone(load_bars(es_path))
            data["daily_es"] = build_numpy_arrays(es_df)
            data["daily_es_idx_map"] = align_daily_to_5m(five_min_df, es_df)
        else:
            data["daily_es"] = None
            data["daily_es_idx_map"] = None

    bundle = ReplayBundle(
        data=data,
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _REPLAY_CACHE[cache_key] = bundle
    return bundle


def replay_engine_kwargs(bundle: ReplayBundle[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """Return only the kwargs accepted by replay engines.

    Replay bundle metadata is useful for optimizer caching, but the engines
    themselves should only receive market data arrays and index maps.
    """
    if isinstance(bundle, ReplayBundle):
        return dict(bundle.data)
    return {
        key: value
        for key, value in bundle.items()
        if not key.startswith("cache_")
    }


def load_vdub_replay_bundle(
    symbol: str,
    data_dir: Path,
    *,
    include_5m: bool = False,
) -> ReplayBundle[dict[str, Any]]:
    from backtests.momentum.cli import _load_vdubus_data

    base_dir = Path(data_dir)
    fifteen_min_path = base_dir / f"{symbol}_15m.parquet"
    es_daily_path = base_dir / "ES_1d.parquet"
    source_paths = [fifteen_min_path, es_daily_path]

    if include_5m:
        source_paths.append(base_dir / f"{symbol}_5m.parquet")

    source_fingerprint = fingerprint_paths(source_paths, root=base_dir)
    cache_key = build_cache_key(
        "momentum.vdub.replay_bundle",
        source_fingerprint=source_fingerprint,
        extra={
            "data_dir": str(base_dir.resolve()),
            "symbol": symbol,
            "include_5m": include_5m,
        },
    )
    cached = _VDUB_REPLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bundle = ReplayBundle(
        data=_load_vdubus_data(symbol, base_dir, include_5m=include_5m),
        cache_key=cache_key,
        cache_source_fingerprint=source_fingerprint,
    )
    _VDUB_REPLAY_CACHE[cache_key] = bundle
    return bundle
