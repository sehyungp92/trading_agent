"""Cross-timeframe OHLCV alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from trading_assistant_data.sources.ibkr.store import ensure_utc_index, resample_ohlcv


TIMEFRAME_RULES = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
}
DAILY_TIMEFRAMES = {"1d", "daily"}


@dataclass(frozen=True)
class AlignmentResult:
    symbol: str
    base_timeframe: str
    target_timeframe: str
    base_rows: int
    target_rows: int
    compared_rows: int
    missing_target_rows: int
    extra_target_rows: int
    mismatched_rows: int
    max_price_diff: float
    max_volume_diff: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def compare_timeframe_alignment(
    *,
    symbol: str,
    base: pd.DataFrame,
    target: pd.DataFrame,
    base_timeframe: str,
    target_timeframe: str,
    price_tolerance: float = 1e-9,
    volume_tolerance: float = 0.0,
) -> AlignmentResult:
    base = ensure_utc_index(base)
    target = ensure_utc_index(target)
    if base.empty or target.empty:
        status = "MISSING_BASE" if base.empty else "MISSING_TARGET"
        return AlignmentResult(symbol.upper(), base_timeframe, target_timeframe, len(base), len(target), 0, 0, 0, 0, 0.0, 0.0, status)
    derived = _derive_target_frame(base, target_timeframe)
    if derived is None:
        return AlignmentResult(symbol.upper(), base_timeframe, target_timeframe, len(base), len(target), 0, 0, 0, 0, 0.0, 0.0, "UNSUPPORTED")
    return compare_derived_frame_alignment(
        symbol=symbol,
        derived=derived,
        target=target,
        base_timeframe=base_timeframe,
        target_timeframe=target_timeframe,
        base_rows=len(base),
        price_tolerance=price_tolerance,
        volume_tolerance=volume_tolerance,
    )


def compare_derived_frame_alignment(
    *,
    symbol: str,
    derived: pd.DataFrame,
    target: pd.DataFrame,
    base_timeframe: str,
    target_timeframe: str,
    base_rows: int | None = None,
    price_tolerance: float = 1e-9,
    volume_tolerance: float = 0.0,
) -> AlignmentResult:
    symbol = symbol.upper()
    derived = ensure_utc_index(derived)
    target = ensure_utc_index(target)
    if derived.empty or target.empty:
        status = "NO_DERIVED" if derived.empty else "MISSING_TARGET"
        return AlignmentResult(symbol, base_timeframe, target_timeframe, base_rows or len(derived), len(target), 0, 0, 0, 0, 0.0, 0.0, status)

    overlap_start = max(derived.index[0], target.index[0])
    overlap_end = min(derived.index[-1], target.index[-1])
    if overlap_end < overlap_start:
        return AlignmentResult(symbol, base_timeframe, target_timeframe, base_rows or len(derived), len(target), 0, 0, 0, 0, 0.0, 0.0, "NO_OVERLAP")

    derived_overlap = derived[(derived.index >= overlap_start) & (derived.index <= overlap_end)]
    target_overlap = target[(target.index >= overlap_start) & (target.index <= overlap_end)]
    common_index = derived_overlap.index.intersection(target_overlap.index)
    missing_target_rows = len(derived_overlap.index.difference(target_overlap.index))
    extra_target_rows = len(target_overlap.index.difference(derived_overlap.index))
    if common_index.empty:
        return AlignmentResult(symbol, base_timeframe, target_timeframe, base_rows or len(derived), len(target), 0, missing_target_rows, extra_target_rows, 0, 0.0, 0.0, "NO_COMMON_TIMESTAMPS")

    price_columns = [column for column in ("open", "high", "low", "close") if column in derived.columns and column in target.columns]
    volume_columns = [column for column in ("volume",) if column in derived.columns and column in target.columns]
    derived_common = derived.loc[common_index]
    target_common = target.loc[common_index]

    max_price_diff = 0.0
    price_mismatch = pd.Series(False, index=common_index)
    if price_columns:
        price_diffs = (derived_common[price_columns] - target_common[price_columns]).abs()
        max_price_diff = float(price_diffs.max().max())
        price_mismatch = price_diffs.gt(price_tolerance).any(axis=1)

    max_volume_diff = 0.0
    volume_mismatch = pd.Series(False, index=common_index)
    if volume_columns:
        volume_diffs = (derived_common[volume_columns] - target_common[volume_columns]).abs()
        max_volume_diff = float(volume_diffs.max().max())
        volume_mismatch = volume_diffs.gt(volume_tolerance).any(axis=1)

    mismatched_rows = int((price_mismatch | volume_mismatch).sum())
    status = "OK" if missing_target_rows == 0 and extra_target_rows == 0 and mismatched_rows == 0 else "MISMATCH"
    return AlignmentResult(
        symbol=symbol,
        base_timeframe=base_timeframe,
        target_timeframe=target_timeframe,
        base_rows=base_rows or len(derived),
        target_rows=len(target),
        compared_rows=len(common_index),
        missing_target_rows=missing_target_rows,
        extra_target_rows=extra_target_rows,
        mismatched_rows=mismatched_rows,
        max_price_diff=max_price_diff,
        max_volume_diff=max_volume_diff,
        status=status,
    )


def _derive_target_frame(base: pd.DataFrame, target_timeframe: str) -> pd.DataFrame | None:
    target_key = target_timeframe.lower()
    if target_key in DAILY_TIMEFRAMES:
        return _resample_intraday_to_rth_daily(base)
    rule = TIMEFRAME_RULES.get(target_key)
    if rule is None:
        return None
    return resample_ohlcv(base, rule)


def _resample_intraday_to_rth_daily(df: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_utc_index(df)
    if frame.empty:
        return frame
    idx_et = frame.index.tz_convert(ZoneInfo("America/New_York"))
    minutes = idx_et.hour * 60 + idx_et.minute
    rth = frame.loc[(minutes >= 570) & (minutes < 960) & (idx_et.weekday < 5)]
    if rth.empty:
        return rth
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    available = {column: func for column, func in agg.items() if column in rth.columns}
    resampled = rth.resample("1D").agg(available)
    return resampled.dropna(
        subset=[column for column in ("open", "high", "low", "close") if column in resampled.columns]
    )

