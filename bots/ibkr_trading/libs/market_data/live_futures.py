"""Live historical-data helpers for quarterly futures analysis streams."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Any

import pandas as pd

from .futures_roll import (
    active_contract_spec,
    generate_quarterly_contracts,
    is_supported_quarterly_future,
    normalize_root,
    roll_schedule,
    root_spec,
)
from .panama import ensure_utc_index, stitch_panama

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL_S = 45.0
_HISTORICAL_CACHE_MAX_ENTRIES = 256
_HISTORICAL_CACHE: dict[tuple[Any, ...], tuple[float, list[Any]]] = {}


@dataclass(frozen=True)
class AdjustedBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


async def req_panama_adjusted_historical_data(
    session: Any,
    contract: Any,
    *,
    symbol: str | None = None,
    endDateTime: Any = "",
    durationStr: str = "",
    barSizeSetting: str = "",
    whatToShow: str = "TRADES",
    useRTH: bool = False,
    formatDate: int = 1,
    keepUpToDate: bool = False,
    chartOptions: list[Any] | None = None,
    timeout: float | None = None,
    request_kind: str = "recurring",
    completed_only: bool = True,
    as_of: datetime | None = None,
    cache_ttl_s: float = _DEFAULT_CACHE_TTL_S,
) -> list[Any]:
    """Fetch physical futures contracts and return a Panama-adjusted series.

    If the requested root is not a supported quarterly index future, the call is
    passed through untouched. For supported roots, a missing old/new side of a
    roll inside the requested window returns an empty list so signal generation
    fails closed instead of mixing unadjusted contract gaps into indicators.
    """
    root = normalize_root(symbol or getattr(contract, "symbol", ""))
    if not is_supported_quarterly_future(root):
        return await _request_bars(
            session,
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate,
            chartOptions=chartOptions,
            timeout=timeout,
            request_kind=request_kind,
            completed_only=completed_only,
            as_of=as_of,
        )

    end = _resolve_end_datetime(endDateTime, as_of=as_of)
    start = end - _duration_to_timedelta(durationStr)
    cache_key = (
        _session_cache_identity(session),
        root,
        _cache_end_bucket(endDateTime, end),
        durationStr,
        barSizeSetting,
        whatToShow,
        bool(useRTH),
        int(formatDate),
        bool(completed_only),
        request_kind,
    )
    cached = _HISTORICAL_CACHE.get(cache_key)
    if cached and cache_ttl_s > 0 and (time.monotonic() - cached[0]) < cache_ttl_s:
        return list(cached[1])

    contracts = generate_quarterly_contracts(
        root,
        start=start,
        end=end,
        include_buffer_contracts=True,
    )
    if not contracts:
        active = active_contract_spec(root, as_of=end)
        contracts = [active] if active is not None else []
    if not contracts:
        logger.warning("No futures contracts generated for %s", root)
        return []

    contract_frames: dict[str, pd.DataFrame] = {}
    for spec in contracts:
        req_start, req_end = _contract_request_window(spec, contracts, start, end)
        if req_start >= req_end:
            continue
        physical = await _qualified_future(session, spec)
        if physical is None:
            logger.warning("Could not build physical futures contract for %s %s", root, spec.yyyymm)
            continue
        bars = await _request_bars(
            session,
            physical,
            endDateTime=_format_ib_end(req_end),
            durationStr=_duration_days(req_start, req_end),
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate,
            chartOptions=chartOptions,
            timeout=timeout,
            request_kind=request_kind,
            completed_only=completed_only,
            as_of=req_end,
        )
        frame = _bars_to_frame(bars)
        if not frame.empty:
            contract_frames[spec.yyyymm] = frame

    rolls = roll_schedule(contracts)
    if _missing_critical_roll_data(contract_frames, rolls, start.date(), end.date()):
        return []
    stitched = stitch_panama(contract_frames, rolls, tick_size=root_spec(root).tick_size)
    if stitched.empty:
        return []

    sliced = stitched[(stitched.index >= pd.Timestamp(start)) & (stitched.index <= pd.Timestamp(end))]
    result = _frame_to_bars(sliced)
    if cache_ttl_s > 0:
        _HISTORICAL_CACHE[cache_key] = (time.monotonic(), list(result))
        while len(_HISTORICAL_CACHE) > _HISTORICAL_CACHE_MAX_ENTRIES:
            _HISTORICAL_CACHE.pop(next(iter(_HISTORICAL_CACHE)))
    return result


def clear_historical_cache() -> None:
    _HISTORICAL_CACHE.clear()


def _session_cache_identity(session: Any) -> tuple[Any, ...]:
    groups = getattr(session, "groups", None)
    if isinstance(groups, dict) and groups:
        identity: list[tuple[Any, ...]] = []
        for group_id, group in sorted(groups.items()):
            config = getattr(group, "config", None)
            identity.append(
                (
                    group_id,
                    getattr(config, "host", ""),
                    getattr(config, "port", ""),
                    getattr(config, "client_id", ""),
                    id(getattr(getattr(group, "conn", None), "ib", None)),
                )
            )
        return tuple(identity)
    ib_obj = getattr(session, "ib", None)
    return (
        session.__class__.__module__,
        session.__class__.__qualname__,
        getattr(ib_obj, "clientId", None),
        id(ib_obj) if ib_obj is not None else id(session),
    )


async def _request_bars(session: Any, contract: Any, **kwargs: Any) -> list[Any]:
    if kwargs.get("timeout") is None:
        kwargs.pop("timeout", None)
    if kwargs.get("chartOptions") is None:
        kwargs.pop("chartOptions", None)
    return await session.req_historical_data(contract, **kwargs)


async def _qualified_future(session: Any, spec: Any) -> Any | None:
    try:
        from ib_async import Future
    except Exception:
        logger.exception("ib_async Future unavailable for physical contract build")
        return None
    contract = Future(
        symbol=spec.symbol,
        exchange=spec.exchange,
        currency=spec.currency,
        lastTradeDateOrContractMonth=spec.yyyymm,
        includeExpired=True,
    )
    if spec.trading_class:
        contract.tradingClass = spec.trading_class
    ib_obj = getattr(session, "ib", None)
    if ib_obj is None:
        return contract
    try:
        qualified = await ib_obj.qualifyContractsAsync(contract)
    except Exception:
        logger.warning("Physical contract qualification failed for %s %s", spec.symbol, spec.yyyymm, exc_info=True)
        return contract
    return qualified[0] if qualified else contract


def _contract_request_window(
    spec: Any,
    contracts: list[Any],
    start: datetime,
    end: datetime,
) -> tuple[datetime, datetime]:
    ordered = sorted(contracts, key=lambda contract: contract.expiry)
    idx = ordered.index(spec)
    prev_roll = ordered[idx - 1].roll_date if idx > 0 else date(1900, 1, 1)
    active_start = _date_start(prev_roll)
    active_end = _date_start(spec.roll_date)
    buffer = _bar_window_buffer(start, end)
    req_start = max(start - buffer, active_start - buffer)
    req_end = min(end + buffer, active_end + buffer, _date_start(spec.expiry) + timedelta(days=1))
    return req_start, req_end


def _bar_window_buffer(start: datetime, end: datetime) -> timedelta:
    span = end - start
    if span <= timedelta(days=5):
        return timedelta(days=3)
    if span <= timedelta(days=90):
        return timedelta(days=5)
    return timedelta(days=10)


def _missing_critical_roll_data(
    contract_frames: dict[str, pd.DataFrame],
    rolls: list[tuple[date, str, str]],
    start: date,
    end: date,
) -> bool:
    missing: list[tuple[date, str, str]] = []
    for roll_date, old_month, new_month in rolls:
        if start <= roll_date <= end:
            if old_month not in contract_frames or new_month not in contract_frames:
                missing.append((roll_date, old_month, new_month))
                continue
            roll_ts = pd.Timestamp(datetime.combine(roll_date, datetime_time.min), tz="UTC")
            old_before = contract_frames[old_month][contract_frames[old_month].index < roll_ts]
            new_after = contract_frames[new_month][contract_frames[new_month].index >= roll_ts]
            if old_before.empty or new_after.empty:
                missing.append((roll_date, old_month, new_month))
    if missing:
        logger.warning("Missing critical futures roll data for Panama stitching: %s", missing)
        return True
    return False


def _bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bar in bars or []:
        ts = _bar_value(bar, "date", None) or _bar_value(bar, "time", None) or _bar_value(bar, "timestamp", None)
        if ts is None:
            continue
        rows.append(
            {
                "date": ts,
                "open": float(_bar_value(bar, "open", 0.0) or 0.0),
                "high": float(_bar_value(bar, "high", 0.0) or 0.0),
                "low": float(_bar_value(bar, "low", 0.0) or 0.0),
                "close": float(_bar_value(bar, "close", 0.0) or 0.0),
                "volume": float(_bar_value(bar, "volume", 0.0) or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("date")
    return ensure_utc_index(frame)


def _frame_to_bars(frame: pd.DataFrame) -> list[AdjustedBar]:
    if frame.empty:
        return []
    output: list[AdjustedBar] = []
    for ts, row in frame.iterrows():
        output.append(
            AdjustedBar(
                date=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
            )
        )
    return output


def _bar_value(bar: Any, key: str, default: Any = None) -> Any:
    if isinstance(bar, dict):
        return bar.get(key, default)
    return getattr(bar, key, default)


def _duration_to_timedelta(duration: str) -> timedelta:
    text = (duration or "").strip().upper()
    if not text:
        return timedelta(days=2)
    parts = text.split()
    if len(parts) < 2:
        return timedelta(days=2)
    try:
        amount = int(float(parts[0]))
    except ValueError:
        return timedelta(days=2)
    unit = parts[1][0]
    if unit == "D":
        return timedelta(days=amount)
    if unit == "W":
        return timedelta(days=amount * 7)
    if unit == "M":
        return timedelta(days=amount * 31)
    if unit == "Y":
        return timedelta(days=amount * 365)
    return timedelta(days=amount)


def _duration_days(start: datetime, end: datetime) -> str:
    days = max(1, math.ceil((end - start).total_seconds() / 86400.0) + 1)
    return f"{days} D"


def _resolve_end_datetime(endDateTime: Any, *, as_of: datetime | None = None) -> datetime:
    if as_of is not None:
        return _as_utc(as_of)
    if isinstance(endDateTime, datetime):
        return _as_utc(endDateTime)
    if isinstance(endDateTime, date):
        return datetime.combine(endDateTime, datetime_time.min, tzinfo=timezone.utc)
    if isinstance(endDateTime, str) and endDateTime.strip():
        for fmt in ("%Y%m%d %H:%M:%S %Z", "%Y%m%d %H:%M:%S", "%Y%m%d"):
            try:
                parsed = datetime.strptime(endDateTime.strip(), fmt)
                return _as_utc(parsed)
            except ValueError:
                continue
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _date_start(value: date) -> datetime:
    return datetime.combine(value, datetime_time.min, tzinfo=timezone.utc)


def _format_ib_end(value: datetime) -> str:
    now = datetime.now(timezone.utc)
    if value >= now - timedelta(minutes=2):
        return ""
    return value.astimezone(timezone.utc).strftime("%Y%m%d %H:%M:%S UTC")


def _cache_end_bucket(endDateTime: Any, end: datetime) -> Any:
    if endDateTime:
        return str(endDateTime)
    return int(end.timestamp() // 60)
