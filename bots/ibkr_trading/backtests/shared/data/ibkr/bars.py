"""Shared IBKR historical bar downloader primitives."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FuturesContractSpec, generate_quarterly_contracts, roll_schedule, root_spec
from .models import BarDownloadRequest, ConnectionSettings, DownloadResult, DownloadWindow
from .pacing import RequestPacer, request_weight
from .stitch import stitch_panama
from .store import (
    detect_large_gaps,
    ensure_utc_index,
    merge_frames,
    read_parquet_if_exists,
    write_manifest,
    write_parquet_atomic,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
PACING_VIOLATION_SLEEP_SECONDS = 65
SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY = "ibkr_contfuture_legacy"
SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA = "ibkr_physical_futures_panama"

IB_BAR_SIZES: dict[str, str] = {
    "1s": "1 secs",
    "5s": "5 secs",
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "daily": "1 day",
}

CHUNK_DURATIONS: dict[str, str] = {
    "1s": "1800 S",
    "5s": "2 H",
    "1m": "1 D",
    "5m": "1 W",
    "15m": "3 W",
    "30m": "1 M",
    "1h": "2 M",
    "4h": "6 M",
    "1d": "1 Y",
    "daily": "1 Y",
}


def timeframe_to_ibkr(timeframe: str) -> str:
    return IB_BAR_SIZES.get(timeframe, timeframe)


def duration_to_timedelta(duration: str) -> timedelta:
    number_text, unit = duration.strip().split(maxsplit=1)
    number = int(number_text)
    unit = unit.upper()
    if unit == "S":
        return timedelta(seconds=number)
    if unit == "H":
        return timedelta(hours=number)
    if unit == "D":
        return timedelta(days=number)
    if unit == "W":
        return timedelta(weeks=number)
    if unit == "M":
        return timedelta(days=30 * number)
    if unit == "Y":
        return timedelta(days=365 * number)
    raise ValueError(f"Unsupported IBKR duration unit: {duration!r}")


def duration_to_days(duration: str) -> int:
    return max(1, duration_to_timedelta(duration).days)


def plan_bar_windows(start: datetime, end: datetime, timeframe: str) -> list[DownloadWindow]:
    start = _ensure_utc_dt(start)
    end = _ensure_utc_dt(end)
    if end <= start:
        return []
    duration = CHUNK_DURATIONS.get(timeframe, "1 W")
    step = duration_to_timedelta(duration)
    windows: list[DownloadWindow] = []
    cursor = end
    while cursor > start:
        window_start = max(start, cursor - step)
        windows.append(DownloadWindow(start=window_start, end=cursor, duration=duration))
        cursor = window_start - timedelta(seconds=1)
    return windows


async def connect_ib(settings: ConnectionSettings):
    from ib_async import IB

    ib = IB()
    await ib.connectAsync(settings.host, settings.port, clientId=settings.client_id, timeout=settings.timeout)
    return ib


async def download_contract_bars(
    ib: Any,
    contract_spec: FuturesContractSpec,
    *,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    what_to_show: str = "TRADES",
    use_rth: bool = False,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult:
    start = _ensure_utc_dt(start)
    end = _ensure_utc_dt(end)
    existing = read_parquet_if_exists(output_path)
    effective_start = start
    if latest_only and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)

    windows = _plan_windows_with_existing_gaps(
        existing,
        start=effective_start,
        end=end,
        timeframe=timeframe,
        include_existing_gaps=latest_only,
        hard_start=start,
    )
    if dry_run:
        return DownloadResult(
            symbol=contract_spec.symbol,
            timeframe=timeframe,
            what_to_show=what_to_show,
            dry_run=True,
            paths=[output_path],
            messages=[
                f"{contract_spec.local_symbol} {timeframe} {what_to_show}: {len(windows)} IBKR bar requests"
            ],
        )
    if not windows:
        return DownloadResult(
            symbol=contract_spec.symbol,
            timeframe=timeframe,
            what_to_show=what_to_show,
            rows=len(existing),
            start=_frame_start(existing),
            end=_frame_end(existing),
            paths=[output_path],
            messages=[f"{contract_spec.local_symbol} {timeframe}: up to date"],
        )

    pacer = pacer or RequestPacer()
    contract = await build_future_contract(ib, contract_spec)
    chunks: list[pd.DataFrame] = []
    empty_streak = 0
    previous_earliest: datetime | None = None
    for window in windows:
        bars = await request_bars_with_retry(
            ib,
            contract,
            end_dt=window.end,
            duration=window.duration,
            timeframe=timeframe,
            what_to_show=what_to_show,
            use_rth=use_rth,
            pacer=pacer,
        )
        if bars:
            frame = bars_to_frame(bars)
            if not frame.empty:
                earliest = frame.index[0].to_pydatetime()
                if previous_earliest is not None and earliest >= previous_earliest:
                    logger.info("%s %s stale progress at %s", contract_spec.local_symbol, timeframe, earliest)
                    break
                previous_earliest = earliest
                chunks.append(frame)
                empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 3 and not latest_only:
                break

    merged = merge_frames(existing, *chunks)
    if not merged.empty:
        merged = merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]
        write_parquet_atomic(merged, output_path)

    return DownloadResult(
        symbol=contract_spec.symbol,
        timeframe=timeframe,
        what_to_show=what_to_show,
        rows=len(merged),
        start=_frame_start(merged),
        end=_frame_end(merged),
        paths=[output_path],
        messages=[f"{contract_spec.local_symbol} {timeframe} {what_to_show}: {len(merged)} rows"],
    )


async def download_historical_bars(
    ib: Any,
    request: BarDownloadRequest,
    *,
    output_path: Path | None = None,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult | pd.DataFrame:
    """Generic stock/index downloader.

    Futures require either the physical contract/Panama helper or an explicit
    legacy ContFuture diagnostic opt-in.
    """
    end = request.end or datetime.now(timezone.utc)
    start = request.start or (end - duration_to_timedelta(request.duration))
    target_path = output_path or request.output_dir / f"{request.symbol}_{request.timeframe}.parquet"
    existing = read_parquet_if_exists(target_path) if latest_only else pd.DataFrame()
    effective_start = start
    if latest_only and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(request.timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)
    windows = _plan_windows_with_existing_gaps(
        existing,
        start=effective_start,
        end=end,
        timeframe=request.timeframe,
        include_existing_gaps=latest_only,
        hard_start=start,
    )
    if request.sec_type.upper() == "FUT" and not request.allow_contfuture_legacy:
        raise ValueError(
            "FUT historical bars require download_physical_futures_panama_bars; "
            "set allow_contfuture_legacy=True only for diagnostic IBKR ContFuture downloads"
        )
    if dry_run:
        metadata = {}
        if request.sec_type.upper() == "FUT":
            metadata = _contfuture_legacy_metadata(
                request,
                output_path=target_path,
                rows=0,
                start=start,
                end=end,
            )
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            dry_run=True,
            paths=[target_path],
            messages=[_dry_run_bar_message(request, len(windows))],
            metadata=metadata,
        )

    pacer = pacer or RequestPacer()
    logger.debug("download %s %s: %d windows planned", request.symbol, request.timeframe, len(windows))
    if request.sec_type.upper() == "FUT":
        downloaded = await download_contfuture_diagnostic(
            ib,
            request,
            start=effective_start,
            end=end,
            pacer=pacer,
        )
    else:
        contract = await build_generic_contract(ib, request)
        chunks: list[pd.DataFrame] = []
        for window in windows:
            bars = await request_bars_with_retry(
                ib,
                contract,
                end_dt=window.end,
                duration=window.duration,
                timeframe=request.timeframe,
                what_to_show=request.what_to_show,
                use_rth=request.use_rth,
                pacer=pacer,
            )
            if bars:
                chunks.append(bars_to_frame(bars))
        downloaded = merge_frames(*chunks)
    merged = merge_frames(existing, downloaded) if latest_only else downloaded
    if output_path is not None:
        if not merged.empty:
            write_parquet_atomic(merged, output_path)
        metadata = {}
        if request.sec_type.upper() == "FUT":
            metadata = _contfuture_legacy_metadata(
                request,
                output_path=output_path,
                rows=len(merged),
                start=_frame_start(merged),
                end=_frame_end(merged),
            )
            write_manifest(_manifest_path_for(output_path), metadata)
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            rows=len(merged),
            start=_frame_start(merged),
            end=_frame_end(merged),
            paths=[output_path],
            metadata=metadata,
        )
    if request.sec_type.upper() == "FUT":
        metadata = _contfuture_legacy_metadata(
            request,
            output_path=target_path,
            rows=len(merged),
            start=_frame_start(merged),
            end=_frame_end(merged),
        )
        write_manifest(_manifest_path_for(target_path), metadata)
    return merged


async def download_physical_futures_panama_bars(
    ib: Any,
    request: BarDownloadRequest,
    *,
    output_path: Path,
    pacer: RequestPacer | None = None,
    dry_run: bool = False,
    latest_only: bool = False,
) -> DownloadResult:
    """Download physical futures contracts and write a Panama-stitched series."""
    if request.sec_type.upper() != "FUT":
        raise ValueError(f"Physical futures downloader only supports FUT requests, got {request.sec_type!r}")

    end = _ensure_utc_dt(request.end or datetime.now(timezone.utc))
    start = _ensure_utc_dt(request.start or (end - duration_to_timedelta(request.duration)))
    existing = read_parquet_if_exists(output_path) if latest_only else pd.DataFrame()
    effective_start = start
    if latest_only and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(request.timeframe, "1 D"))
        effective_start = max(start, existing.index[-1].to_pydatetime() - overlap)

    contracts = generate_quarterly_contracts(
        request.symbol,
        start=effective_start,
        end=end,
        include_buffer_contracts=True,
    )
    if not contracts:
        raise ValueError(f"No quarterly futures contracts generated for {request.symbol}")

    metadata = _physical_futures_metadata(
        request,
        output_path=output_path,
        contracts=contracts,
        rows=0,
        start=start,
        end=end,
    )
    if dry_run:
        windows = [
            _contract_request_window(contract, contracts, effective_start, end)
            for contract in contracts
        ]
        planned = sum(1 for req_start, req_end in windows if req_start < req_end)
        return DownloadResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            dry_run=True,
            paths=[output_path],
            messages=[
                f"{request.symbol} {request.timeframe}: physical futures/Panama over {planned} contract windows"
            ],
            metadata=metadata,
        )

    pacer = pacer or RequestPacer()
    contract_frames: dict[str, pd.DataFrame] = {}
    contract_paths: dict[str, str] = {}
    for contract in contracts:
        req_start, req_end = _contract_request_window(contract, contracts, effective_start, end)
        if req_start >= req_end:
            continue
        contract_path = _physical_contract_bar_path(output_path, contract, request.timeframe, request.what_to_show)
        contract_paths[contract.yyyymm] = str(contract_path)
        await download_contract_bars(
            ib,
            contract,
            timeframe=request.timeframe,
            start=req_start,
            end=req_end,
            output_path=contract_path,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
            dry_run=False,
            latest_only=latest_only,
        )
        frame = read_parquet_if_exists(contract_path)
        if not frame.empty:
            contract_frames[contract.yyyymm] = frame

    missing_rolls = _missing_critical_roll_data(
        contract_frames,
        roll_schedule(contracts),
        effective_start.date(),
        end.date(),
    )
    if missing_rolls:
        raise ValueError(
            "Missing critical physical futures roll data for "
            f"{request.symbol}: {missing_rolls}. "
            "Supply archived physical-contract evidence or use a retention-covered lane."
        )

    stitched = stitch_panama(
        contract_frames,
        roll_schedule(contracts),
        tick_size=root_spec(request.symbol).tick_size,
    )
    if not stitched.empty:
        stitched = stitched[
            (stitched.index >= pd.Timestamp(start)) & (stitched.index <= pd.Timestamp(end))
        ]
    merged = merge_frames(existing, stitched) if latest_only else stitched
    if not merged.empty:
        write_parquet_atomic(merged, output_path)

    metadata = _physical_futures_metadata(
        request,
        output_path=output_path,
        contracts=contracts,
        rows=len(merged),
        start=_frame_start(merged),
        end=_frame_end(merged),
        contract_paths=contract_paths,
    )
    write_manifest(_manifest_path_for(output_path), metadata)
    return DownloadResult(
        symbol=request.symbol,
        timeframe=request.timeframe,
        what_to_show=request.what_to_show,
        rows=len(merged),
        start=_frame_start(merged),
        end=_frame_end(merged),
        paths=[output_path],
        messages=[f"{request.symbol} {request.timeframe} physical futures/Panama: {len(merged)} rows"],
        metadata=metadata,
    )


async def download_contfuture_diagnostic(
    ib: Any,
    request: BarDownloadRequest,
    *,
    start: datetime,
    end: datetime,
    pacer: RequestPacer | None,
) -> pd.DataFrame:
    """Diagnostic-only IBKR ContFuture downloader.

    ContFuture is opaque and must not be used for authoritative validation.
    """
    if not request.allow_contfuture_legacy:
        raise ValueError("ContFuture diagnostics require allow_contfuture_legacy=True")
    cont_contract = await build_generic_contract(ib, request)
    chunks: list[pd.DataFrame] = []

    if request.timeframe in {"1d", "daily"}:
        bars = await request_bars_with_retry(
            ib,
            cont_contract,
            end_dt="",
            duration=request.duration,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
        )
        if bars:
            chunks.append(bars_to_frame(bars))
        return merge_frames(*chunks)

    latest_duration = {
        "1m": "1 W",
        "5m": "1 M",
        "15m": "1 M",
        "30m": "2 M",
        "1h": "1 Y",
        "4h": "2 Y",
    }.get(request.timeframe, CHUNK_DURATIONS.get(request.timeframe, "1 W"))
    bars = await request_bars_with_retry(
        ib,
        cont_contract,
        end_dt="",
        duration=latest_duration,
        timeframe=request.timeframe,
        what_to_show=request.what_to_show,
        use_rth=request.use_rth,
        pacer=pacer,
    )
    earliest = end
    if bars:
        latest_frame = bars_to_frame(bars)
        chunks.append(latest_frame)
        earliest = latest_frame.index[0].to_pydatetime()

    if earliest <= start:
        return merge_frames(*chunks)

    chunk_contract = await build_legacy_chunked_contfuture_contract(ib, request)
    cursor = earliest - timedelta(seconds=1)
    empty_streak = 0
    previous_earliest: datetime | None = None
    duration = CHUNK_DURATIONS.get(request.timeframe, "1 W")
    while cursor > start and empty_streak < 5:
        bars = await request_bars_with_retry(
            ib,
            chunk_contract,
            end_dt=cursor,
            duration=duration,
            timeframe=request.timeframe,
            what_to_show=request.what_to_show,
            use_rth=request.use_rth,
            pacer=pacer,
        )
        if bars:
            frame = bars_to_frame(bars)
            chunks.append(frame)
            empty_streak = 0
            frame_earliest = frame.index[0].to_pydatetime()
            if previous_earliest is not None and frame_earliest >= previous_earliest:
                break
            previous_earliest = frame_earliest
            cursor = frame_earliest - timedelta(seconds=1)
        else:
            empty_streak += 1
            cursor = cursor - duration_to_timedelta(duration)
    merged = merge_frames(*chunks)
    if merged.empty:
        return merged
    return merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]


async def request_bars_with_retry(
    ib: Any,
    contract: Any,
    *,
    end_dt: datetime | str,
    duration: str,
    timeframe: str,
    what_to_show: str,
    use_rth: bool,
    pacer: RequestPacer | None = None,
    timeout: int = 120,
) -> list[Any]:
    bar_size = timeframe_to_ibkr(timeframe)
    end_str = _format_ib_end(end_dt)
    signature = (
        getattr(contract, "conId", None),
        getattr(contract, "symbol", None),
        getattr(contract, "lastTradeDateOrContractMonth", None),
        end_str,
        duration,
        bar_size,
        what_to_show,
        use_rth,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if pacer is not None:
                await pacer.wait(signature, weight=request_weight(what_to_show))
            return await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_str,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=2,
                    timeout=0,
                ),
                timeout=timeout,
            ) or []
        except Exception as exc:
            message = str(exc).lower()
            if "pacing" in message or "162" in message:
                logger.warning("IBKR pacing violation on %s %s; sleeping %ss", getattr(contract, "symbol", "?"), timeframe, PACING_VIOLATION_SLEEP_SECONDS)
                await asyncio.sleep(PACING_VIOLATION_SLEEP_SECONDS)
            elif "no data" in message or "HMDS query returned no data".lower() in message:
                return []
            elif attempt >= MAX_RETRIES:
                logger.warning("IBKR request failed after %d attempts: %s", MAX_RETRIES, exc)
                return []
            else:
                await asyncio.sleep(5 * attempt)
    return []


async def build_future_contract(ib: Any, spec: FuturesContractSpec):
    from ib_async import Future

    contract = Future(
        symbol=spec.symbol,
        exchange=spec.exchange,
        tradingClass=spec.ib_trading_class,
        lastTradeDateOrContractMonth=spec.yyyymm,
        includeExpired=True,
    )
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not qualify {spec.local_symbol}")
    contract = qualified[0]
    contract.includeExpired = True
    return contract


async def build_generic_contract(ib: Any, request: BarDownloadRequest):
    if request.sec_type.upper() == "STK":
        from ib_async import Stock

        contract = Stock(symbol=request.symbol, exchange=request.exchange, currency=request.currency)
        if request.primary_exchange:
            contract.primaryExchange = request.primary_exchange
    elif request.sec_type.upper() in {"IND", "INDEX"}:
        from ib_async import Index

        contract = Index(symbol=request.symbol, exchange=request.exchange, currency=request.currency)
    elif request.sec_type.upper() == "FUT":
        if not request.allow_contfuture_legacy:
            raise ValueError("ContFuture qualification requires allow_contfuture_legacy=True")
        from ib_async import ContFuture

        contract = ContFuture(
            symbol=request.symbol,
            exchange=request.exchange,
            tradingClass=request.ib_trading_class,
        )
    else:
        raise ValueError(f"Unsupported sec_type: {request.sec_type}")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not qualify {request.sec_type} {request.symbol}")
    return qualified[0]


async def build_legacy_chunked_contfuture_contract(ib: Any, request: BarDownloadRequest):
    if not request.allow_contfuture_legacy:
        raise ValueError("Chunked ContFuture diagnostics require allow_contfuture_legacy=True")
    from ib_async import Future

    cont = await build_generic_contract(ib, request)
    contract = Future(conId=cont.conId, exchange=request.exchange)
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not resolve chunked continuous future for {request.symbol}")
    return qualified[0]


async def build_chunked_continuous_future(ib: Any, request: BarDownloadRequest):
    """Backward-compatible alias for legacy diagnostic ContFuture callers."""
    return await build_legacy_chunked_contfuture_contract(ib, request)


def bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for bar in bars:
        row = {
            "time": bar.date if isinstance(bar.date, datetime) else pd.Timestamp(bar.date),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(getattr(bar, "volume", 0) or 0),
        }
        if hasattr(bar, "barCount"):
            row["bar_count"] = int(getattr(bar, "barCount") or 0)
        if hasattr(bar, "average"):
            row["wap"] = float(getattr(bar, "average") or 0.0)
        records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return ensure_utc_index(frame.set_index("time"))


def _plan_windows_with_existing_gaps(
    existing: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    timeframe: str,
    include_existing_gaps: bool,
    hard_start: datetime,
) -> list[DownloadWindow]:
    windows = plan_bar_windows(start, end, timeframe)
    if include_existing_gaps and not existing.empty:
        overlap = duration_to_timedelta(CHUNK_DURATIONS.get(timeframe, "1 D"))
        gap_windows: list[DownloadWindow] = []
        for gap in detect_large_gaps(existing, timeframe):
            gap_start = max(hard_start, gap.start - overlap)
            gap_end = min(end, gap.end + overlap)
            gap_windows.extend(plan_bar_windows(gap_start, gap_end, timeframe))
        windows = gap_windows + windows
    return _dedupe_windows(windows)


def _dedupe_windows(windows: list[DownloadWindow]) -> list[DownloadWindow]:
    seen: set[tuple[datetime, datetime, str]] = set()
    unique: list[DownloadWindow] = []
    for window in sorted(windows, key=lambda item: item.end, reverse=True):
        key = (window.start, window.end, window.duration)
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    return unique


def _dry_run_bar_message(request: BarDownloadRequest, window_count: int) -> str:
    if request.sec_type.upper() == "FUT":
        return (
            f"{request.symbol} {request.timeframe}: "
            f"{window_count} legacy ContFuture diagnostic IBKR bar requests"
        )
    return f"{request.symbol} {request.timeframe}: {window_count} IBKR bar requests"


def _physical_contract_bar_path(
    output_path: Path,
    contract: FuturesContractSpec,
    timeframe: str,
    what_to_show: str,
) -> Path:
    return (
        output_path.parent
        / "_physical_contracts"
        / contract.symbol.upper()
        / contract.yyyymm
        / f"{timeframe}_{what_to_show.lower()}.parquet"
    )


def _contract_request_window(
    spec: FuturesContractSpec,
    contracts: list[FuturesContractSpec],
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
) -> list[tuple[date, str, str]]:
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
    return missing


def _manifest_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def _contfuture_legacy_metadata(
    request: BarDownloadRequest,
    *,
    output_path: Path,
    rows: int,
    start: datetime | None,
    end: datetime | None,
) -> dict[str, object]:
    return {
        "source_kind": SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY,
        "usable_for_authoritative_validation": False,
        "family": request.family,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "sec_type": request.sec_type,
        "exchange": request.exchange,
        "trading_class": request.ib_trading_class,
        "what_to_show": request.what_to_show,
        "use_rth": request.use_rth,
        "duration": request.duration,
        "rows": rows,
        "start": _isoformat(start),
        "end": _isoformat(end),
        "output_path": str(output_path),
        "policy_note": (
            "IBKR ContFuture is opaque and diagnostic-only; use physical futures "
            "contract downloads plus Panama stitching for authority."
        ),
    }


def _physical_futures_metadata(
    request: BarDownloadRequest,
    *,
    output_path: Path,
    contracts: list[FuturesContractSpec],
    rows: int,
    start: datetime | None,
    end: datetime | None,
    contract_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "source_kind": SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA,
        "usable_for_authoritative_validation": True,
        "family": request.family,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "sec_type": request.sec_type,
        "exchange": request.exchange,
        "trading_class": request.ib_trading_class,
        "what_to_show": request.what_to_show,
        "use_rth": request.use_rth,
        "duration": request.duration,
        "rows": rows,
        "start": _isoformat(start),
        "end": _isoformat(end),
        "output_path": str(output_path),
        "calendar_session_policy": "request.use_rth controls IBKR RTH/ETH; sparse ETH bars are valid trades-only bars",
        "adjustment_roll_policy": "quarterly CME roll policy plus deterministic backward Panama stitching",
        "contracts": [
            {
                "yyyymm": contract.yyyymm,
                "local_symbol": contract.local_symbol,
                "expiry": contract.expiry.isoformat(),
                "roll_date": contract.roll_date.isoformat(),
                "exchange": contract.exchange,
                "trading_class": contract.ib_trading_class,
                "raw_path": (contract_paths or {}).get(contract.yyyymm, ""),
            }
            for contract in contracts
        ],
        "rolls": [
            {
                "roll_date": roll_date.isoformat(),
                "old_month": old_month,
                "new_month": new_month,
            }
            for roll_date, old_month, new_month in roll_schedule(contracts)
        ],
    }


def _date_start(value: date) -> datetime:
    return datetime.combine(value, datetime_time.min, tzinfo=timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _ensure_utc_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_ib_end(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    value = _ensure_utc_dt(value)
    return value.strftime("%Y%m%d %H:%M:%S UTC")


def _frame_start(df: pd.DataFrame) -> datetime | None:
    if df.empty:
        return None
    return df.index[0].to_pydatetime()


def _frame_end(df: pd.DataFrame) -> datetime | None:
    if df.empty:
        return None
    return df.index[-1].to_pydatetime()
