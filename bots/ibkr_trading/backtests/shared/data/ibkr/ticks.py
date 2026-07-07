"""Shared IBKR historical tick downloader primitives."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .bars import build_future_contract
from .contracts import FuturesContractSpec
from .models import DownloadResult
from .pacing import RequestPacer, request_weight
from .store import merge_frames, read_parquet_if_exists, rich_tick_path, write_parquet_atomic

logger = logging.getLogger(__name__)

MAX_TICKS_PER_REQUEST = 1000
MAX_RETRIES = 4
PACING_VIOLATION_SLEEP_SECONDS = 65


def session_windows(
    *,
    end: datetime,
    days: int,
    session_start: time = time(9, 15),
    session_end: time = time(11, 30),
    timezone_name: str = "America/New_York",
) -> list[tuple[datetime, datetime]]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone_name)
    end_et = end.astimezone(tz)
    windows: list[tuple[datetime, datetime]] = []
    day = end_et.date()
    while len(windows) < days:
        if day.weekday() < 5:
            start_dt = datetime.combine(day, session_start, tzinfo=tz).astimezone(timezone.utc)
            end_dt = datetime.combine(day, session_end, tzinfo=tz).astimezone(timezone.utc)
            if end_dt <= end.astimezone(timezone.utc):
                windows.append((start_dt, end_dt))
        day = day - timedelta(days=1)
    windows.reverse()
    return windows


async def download_tick_windows(
    ib: Any,
    contract_spec: FuturesContractSpec,
    windows: list[tuple[datetime, datetime]],
    *,
    output_root: Path,
    tick_type: str = "TRADES",
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    merge_output_path: Path | None = None,
) -> DownloadResult:
    paths = [
        rich_tick_path(output_root, contract_spec.symbol, contract_spec.yyyymm, start.date().isoformat(), tick_type)
        for start, _end in windows
    ]
    if dry_run:
        return DownloadResult(
            symbol=contract_spec.symbol,
            what_to_show=tick_type,
            paths=paths,
            dry_run=True,
            messages=[f"{contract_spec.local_symbol} {tick_type}: {len(windows)} tick windows"],
        )

    pacer = pacer or RequestPacer()
    contract = await build_future_contract(ib, contract_spec)
    all_frames: list[pd.DataFrame] = []
    for (start, end), path in zip(windows, paths):
        existing = read_parquet_if_exists(path)
        start_for_request = start
        if not existing.empty:
            start_for_request = max(start, existing.index[-1].to_pydatetime() + timedelta(milliseconds=1))
        if start_for_request >= end:
            all_frames.append(existing)
            continue
        downloaded = await request_ticks_with_retry(
            ib,
            contract,
            start=start_for_request,
            end=end,
            tick_type=tick_type,
            pacer=pacer,
        )
        combined = merge_frames(existing, downloaded)
        if not combined.empty:
            write_parquet_atomic(combined, path)
            all_frames.append(combined)

    merged = merge_frames(*all_frames)
    if merge_output_path is not None and not merged.empty:
        write_parquet_atomic(merged, merge_output_path)
        paths.append(merge_output_path)

    return DownloadResult(
        symbol=contract_spec.symbol,
        what_to_show=tick_type,
        rows=len(merged),
        start=merged.index[0].to_pydatetime() if not merged.empty else None,
        end=merged.index[-1].to_pydatetime() if not merged.empty else None,
        paths=paths,
        messages=[f"{contract_spec.local_symbol} {tick_type}: {len(merged)} ticks"],
    )


async def request_ticks_with_retry(
    ib: Any,
    contract: Any,
    *,
    start: datetime,
    end: datetime,
    tick_type: str,
    pacer: RequestPacer | None = None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cursor = _ensure_utc(start)
    end = _ensure_utc(end)
    previous_cursor: datetime | None = None

    while cursor < end:
        if previous_cursor is not None and cursor <= previous_cursor:
            break
        previous_cursor = cursor
        response = await _request_tick_page(ib, contract, cursor, end, tick_type, pacer)
        frame = ticks_to_frame(response, tick_type)
        if frame.empty:
            break
        chunks.append(frame)
        last_ts = frame.index[-1].to_pydatetime()
        cursor = last_ts + timedelta(milliseconds=1)
        if len(frame) < MAX_TICKS_PER_REQUEST:
            break
    merged = merge_frames(*chunks)
    if merged.empty:
        return merged
    return merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]


async def _request_tick_page(
    ib: Any,
    contract: Any,
    start: datetime,
    end: datetime,
    tick_type: str,
    pacer: RequestPacer | None,
) -> list[Any]:
    start_str = _format_tick_time(start)
    end_str = _format_tick_time(end)
    signature = (
        getattr(contract, "conId", None),
        getattr(contract, "symbol", None),
        start_str,
        end_str,
        tick_type,
        MAX_TICKS_PER_REQUEST,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if pacer is not None:
                await pacer.wait(signature, weight=request_weight(tick_type))
            return await asyncio.wait_for(
                ib.reqHistoricalTicksAsync(
                    contract,
                    startDateTime=start_str,
                    endDateTime=end_str,
                    numberOfTicks=MAX_TICKS_PER_REQUEST,
                    whatToShow=tick_type,
                    useRth=False,
                    ignoreSize=False,
                    miscOptions=[],
                ),
                timeout=120,
            ) or []
        except Exception as exc:
            message = str(exc).lower()
            if "pacing" in message or "162" in message:
                await asyncio.sleep(PACING_VIOLATION_SLEEP_SECONDS)
            elif "no data" in message:
                return []
            elif attempt >= MAX_RETRIES:
                logger.warning("IBKR tick request failed after %d attempts: %s", MAX_RETRIES, exc)
                return []
            else:
                await asyncio.sleep(5 * attempt)
    return []


def ticks_to_frame(ticks: list[Any], tick_type: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for tick in ticks:
        timestamp = _tick_timestamp(tick)
        row: dict[str, Any] = {"timestamp": timestamp}
        if tick_type.upper() == "BID_ASK":
            row.update(
                {
                    "bid_price": _attr(tick, "priceBid", "bidPrice", "bid", default=0.0),
                    "ask_price": _attr(tick, "priceAsk", "askPrice", "ask", default=0.0),
                    "bid_size": _attr(tick, "sizeBid", "bidSize", default=0.0),
                    "ask_size": _attr(tick, "sizeAsk", "askSize", default=0.0),
                }
            )
        else:
            row.update(
                {
                    "price": _attr(tick, "price", default=0.0),
                    "size": _attr(tick, "size", default=0.0),
                    "side": 0,
                }
            )
        records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def _tick_timestamp(tick: Any) -> datetime:
    value = _attr(tick, "time", "date", default=None)
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _format_tick_time(value: datetime) -> str:
    value = _ensure_utc(value)
    return value.strftime("%Y%m%d %H:%M:%S UTC")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
