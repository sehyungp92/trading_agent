"""Backward Panama stitching for physical futures contracts."""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

logger = logging.getLogger(__name__)


class StitchQualityError(ValueError):
    """Raised when a continuous futures stitch would use implausible data."""


def round_to_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return value
    return round(round(value / tick_size) * tick_size, 10)


def ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def stitch_panama(
    contract_data: dict[str, pd.DataFrame],
    rolls: list[tuple[date, str, str]],
    *,
    tick_size: float = 0.25,
    max_gap_range_mult: float | None = 12.0,
    min_gap_points: float = 250.0,
    fail_closed: bool = True,
) -> pd.DataFrame:
    """Stitch physical contracts with backward Panama adjustment.

    The gap guard rejects implausible roll offsets so a bad seam bar cannot
    shift all older history and create false momentum signals.
    """
    normalized = {
        key: ensure_utc_index(df)
        for key, df in contract_data.items()
        if df is not None and not df.empty
    }
    if not normalized:
        return pd.DataFrame()

    all_rolls = sorted(rolls, key=lambda item: item[0])
    if not all_rolls:
        return merge_frames(*normalized.values())

    ordered_months: list[str] = []
    for _roll_date, old, new in all_rolls:
        if old in normalized and old not in ordered_months:
            ordered_months.append(old)
        if new in normalized and new not in ordered_months:
            ordered_months.append(new)
    for month in sorted(normalized):
        if month not in ordered_months:
            ordered_months.append(month)

    segments: list[tuple[str, pd.DataFrame]] = []
    for month in ordered_months:
        frame = normalized.get(month)
        if frame is None or frame.empty:
            continue
        lower = _lower_roll(all_rolls, month)
        upper = _upper_roll(all_rolls, month)
        segment = frame
        if lower is not None:
            segment = segment[segment.index >= lower]
        if upper is not None:
            segment = segment[segment.index < upper]
        if not segment.empty:
            segments.append((month, segment))

    adjustments: dict[str, float] = {}
    cumulative = 0.0
    if ordered_months:
        adjustments[ordered_months[-1]] = 0.0

    for roll_date, old_month, new_month in reversed(all_rolls):
        if old_month not in normalized:
            continue
        if new_month not in normalized:
            adjustments[old_month] = cumulative
            continue
        roll_ts = pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
        old_frame = normalized[old_month]
        new_frame = normalized[new_month]
        old_before = old_frame[old_frame.index < roll_ts]
        new_after = new_frame[new_frame.index >= roll_ts]
        if old_before.empty or new_after.empty:
            adjustments[old_month] = cumulative
            continue
        gap = round_to_tick(float(new_after.iloc[0]["open"]) - float(old_before.iloc[-1]["close"]), tick_size)
        if max_gap_range_mult is not None:
            max_gap = _max_reasonable_gap(
                old_before,
                new_after,
                range_mult=max_gap_range_mult,
                min_gap_points=min_gap_points,
            )
            if abs(gap) > max_gap:
                message = (
                    f"Implausible Panama roll gap {gap:.2f} for {old_month}->{new_month} "
                    f"on {roll_date.isoformat()} exceeds {max_gap:.2f}"
                )
                logger.warning(message)
                if fail_closed:
                    return pd.DataFrame()
                raise StitchQualityError(message)
        cumulative += gap
        adjustments[old_month] = cumulative

    adjusted: list[pd.DataFrame] = []
    for month, segment in segments:
        adjustment = adjustments.get(month, 0.0)
        if adjustment:
            segment = segment.copy()
            for column in ("open", "high", "low", "close"):
                if column in segment.columns:
                    segment[column] = segment[column] - adjustment
        adjusted.append(segment)

    return merge_frames(*adjusted)


def _max_reasonable_gap(
    old_before: pd.DataFrame,
    new_after: pd.DataFrame,
    *,
    range_mult: float,
    min_gap_points: float,
) -> float:
    old_ranges = _true_ranges(old_before.tail(48))
    new_ranges = _true_ranges(new_after.head(48))
    ranges = pd.concat([old_ranges, new_ranges])
    ranges = ranges[ranges > 0]
    median_range = float(ranges.median()) if not ranges.empty else 0.0
    return max(float(min_gap_points), median_range * float(range_mult))


def _true_ranges(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or not {"high", "low", "close"}.issubset(frame.columns):
        return pd.Series(dtype=float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


def _lower_roll(rolls: list[tuple[date, str, str]], month: str) -> pd.Timestamp | None:
    for roll_date, _old, new in rolls:
        if new == month:
            return pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
    return None


def _upper_roll(rolls: list[tuple[date, str, str]], month: str) -> pd.Timestamp | None:
    for roll_date, old, _new in rolls:
        if old == month:
            return pd.Timestamp(datetime.combine(roll_date, datetime.min.time()), tz="UTC")
    return None
