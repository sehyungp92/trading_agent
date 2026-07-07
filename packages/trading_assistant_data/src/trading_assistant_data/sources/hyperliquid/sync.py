"""Incremental Hyperliquid refresh into data-owned canonical partitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from trading_assistant_data.calendars.crypto import calendar_definition as crypto_calendar
from trading_assistant_data.io import write_json
from trading_assistant_data.normalization import (
    CRYPTO_ADJUSTMENT_POLICY,
    _manifest_for_partition,
    _rel,
)
from trading_assistant_data.repo import git_commit_sha
from trading_assistant_data.slices.writer import update_slice_index
from trading_assistant_data.sources.hyperliquid.downloader import INTERVALS, HyperliquidDownloader
from trading_assistant_data.sources.hyperliquid.store import canonicalize_candles, canonicalize_funding


class HyperliquidClient(Protocol):
    def candles(self, symbol: str, interval: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        ...

    def funding(self, symbol: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        ...


MAX_POINTS_PER_SOURCE_REQUEST = 1_000
MAX_FUNDING_POINTS_PER_SOURCE_REQUEST = 500


def sync_hyperliquid(
    *,
    repo_root: Path,
    symbols: list[str] | tuple[str, ...] | None = None,
    intervals: list[str] | tuple[str, ...] | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    years: int = 0,
    lookback_days: int = 30,
    latest: bool = False,
    funding: bool = False,
    overlap_bars: int = 200,
    dry_run: bool = False,
    downloader: HyperliquidClient | None = None,
) -> dict[str, Any]:
    """Refresh Hyperliquid data with bot-compatible merging and manifest/index updates.

    Dry runs never touch the network or filesystem; they report the windows that a real run
    would request from the source.
    """

    repo = Path(repo_root)
    symbol_list = [item.upper() for item in (symbols or ["BTC"])]
    interval_list = list(intervals or ["1m"])
    unsupported = sorted(set(interval_list).difference(INTERVALS))
    if unsupported:
        raise ValueError(f"unsupported Hyperliquid intervals: {', '.join(unsupported)}")
    end_dt = _coerce_datetime(end) or datetime.now(timezone.utc)
    client = downloader or HyperliquidDownloader()
    source_version = git_commit_sha(repo) or "working_tree_uncommitted"
    writes = []
    operations: list[dict[str, Any]] = []

    for symbol in symbol_list:
        for interval in interval_list:
            latest_existing = _latest_timestamp(_canonical_bars_root(repo, symbol, interval))
            start_dt = _resolve_start(
                explicit=start,
                end=end_dt,
                latest_existing=latest_existing,
                interval=interval,
                years=years,
                lookback_days=lookback_days,
                latest=latest,
                overlap_bars=overlap_bars,
            )
            operation = _operation(
                kind="candles",
                symbol=symbol,
                interval=interval,
                start=start_dt,
                end=end_dt,
                latest_existing=latest_existing,
                dry_run=dry_run,
            )
            if dry_run:
                operations.append(operation | {"status": "planned"})
                continue
            rows = _download_candles_paginated(client, symbol, interval, start=start_dt, end=end_dt)
            operation["rows_downloaded"] = len(rows)
            raw_path = _raw_path(repo, "candles", symbol, interval, start_dt, end_dt)
            write_json(raw_path, rows)
            operation["raw_path"] = _rel(raw_path, repo)
            if not rows:
                operations.append(operation | {"status": "no_data"})
                continue
            canonical = canonicalize_candles(
                pd.DataFrame(rows),
                symbol=symbol,
                interval=interval,
                source_file=_rel(raw_path, repo),
            )
            slice_writes = _merge_monthly_partitioned_slice(
                repo_root=repo,
                canonical=canonical,
                root=_canonical_bars_root(repo, symbol, interval),
                source="hyperliquid",
                market="crypto_perp",
                symbol=symbol,
                timeframe=interval,
                calendar=True,
                source_version=source_version,
                kind="trades",
                dry_run=False,
            )
            writes.extend(slice_writes)
            operation["status"] = "written"
            operation["rows_written"] = int(len(canonical))
            operation["manifest_paths"] = [_rel(item.manifest_path, repo) for item in slice_writes]
            operation["canonical_paths"] = [
                _rel(path, repo) for item in slice_writes for path in item.canonical_paths
            ]
            operations.append(operation)

        if funding:
            latest_existing = _latest_timestamp(_canonical_funding_root(repo, symbol))
            start_dt = _resolve_start(
                explicit=start,
                end=end_dt,
                latest_existing=latest_existing,
                interval="funding_1h",
                years=years,
                lookback_days=lookback_days,
                latest=latest,
                overlap_bars=max(1, overlap_bars // 480),
            )
            operation = _operation(
                kind="funding",
                symbol=symbol,
                interval="funding_1h",
                start=start_dt,
                end=end_dt,
                latest_existing=latest_existing,
                dry_run=dry_run,
            )
            if dry_run:
                operations.append(operation | {"status": "planned"})
                continue
            rows = _download_funding_paginated(client, symbol, start=start_dt, end=end_dt)
            operation["rows_downloaded"] = len(rows)
            raw_path = _raw_path(repo, "funding", symbol, "funding_1h", start_dt, end_dt)
            write_json(raw_path, rows)
            operation["raw_path"] = _rel(raw_path, repo)
            if not rows:
                operations.append(operation | {"status": "no_data"})
                continue
            canonical = canonicalize_funding(pd.DataFrame(rows), symbol=symbol, source_file=_rel(raw_path, repo))
            slice_writes = _merge_monthly_partitioned_slice(
                repo_root=repo,
                canonical=canonical,
                root=_canonical_funding_root(repo, symbol),
                source="hyperliquid",
                market="crypto_perp",
                symbol=symbol,
                timeframe="funding_1h",
                calendar=True,
                source_version=source_version,
                kind="funding",
                dry_run=False,
            )
            writes.extend(slice_writes)
            operation["status"] = "written"
            operation["rows_written"] = int(len(canonical))
            operation["manifest_paths"] = [_rel(item.manifest_path, repo) for item in slice_writes]
            operation["canonical_paths"] = [
                _rel(path, repo) for item in slice_writes for path in item.canonical_paths
            ]
            operations.append(operation)

    if writes:
        update_slice_index(repo, writes)
    return {
        "source": "hyperliquid",
        "dry_run": dry_run,
        "status": "planned" if dry_run else "complete",
        "source_version": source_version,
        "operation_count": len(operations),
        "slice_manifest_count": len(writes),
        "operations": operations,
    }


def _download_candles_paginated(
    client: HyperliquidClient,
    symbol: str,
    interval: str,
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    return _dedupe_time_rows(
        row
        for page_start, page_end in _request_pages(start, end, _interval_delta(interval))
        for row in client.candles(symbol, interval, start=page_start, end=page_end)
    )


def _download_funding_paginated(
    client: HyperliquidClient,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    return _dedupe_time_rows(
        row
        for page_start, page_end in _request_pages(
            start,
            end,
            timedelta(hours=1),
            max_points=MAX_FUNDING_POINTS_PER_SOURCE_REQUEST,
        )
        for row in client.funding(symbol, start=page_start, end=page_end)
    )


def _request_pages(
    start: datetime,
    end: datetime,
    step: timedelta,
    *,
    max_points: int = MAX_POINTS_PER_SOURCE_REQUEST,
) -> list[tuple[datetime, datetime]]:
    if end < start:
        return []
    page_span = step * max_points
    pages: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        page_end = min(end, cursor + page_span - step)
        pages.append((cursor, page_end))
        cursor = page_end + step
    return pages


def _dedupe_time_rows(rows: Any) -> list[dict[str, Any]]:
    by_time: dict[int, dict[str, Any]] = {}
    untimed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_time = item.get("t", item.get("time"))
        if raw_time is None:
            untimed.append(item)
            continue
        by_time.setdefault(int(raw_time), item)
    return [by_time[key] for key in sorted(by_time)] + untimed


def _merge_monthly_partitioned_slice(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    root: Path,
    source: str,
    market: str,
    symbol: str,
    timeframe: str,
    calendar: bool,
    source_version: str,
    kind: str,
    dry_run: bool,
):
    frame = canonical.assign(
        year=canonical["timestamp_utc"].dt.year.astype(str),
        month=canonical["timestamp_utc"].dt.month.map(lambda value: f"{int(value):02d}"),
    )
    writes = []
    for (year, month), group in frame.groupby(["year", "month"], sort=True):
        path = root / f"year={year}" / f"month={month}" / "part.parquet"
        output = group.drop(columns=["year", "month"])
        if path.exists():
            existing = pd.read_parquet(path)
            if timeframe.startswith("funding_") and "timestamp_utc" in existing.columns:
                existing = existing.copy()
                existing["timestamp_utc"] = pd.to_datetime(
                    existing["timestamp_utc"],
                    utc=True,
                ).dt.floor("h")
                if "timestamp_exchange" in existing.columns:
                    existing["timestamp_exchange"] = existing["timestamp_utc"].astype(str)
            output = pd.concat([existing, output], ignore_index=True)
            output = output.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, engine="pyarrow", index=False)
        writes.append(
            _manifest_for_partition(
                repo_root=repo_root,
                canonical_paths=[path],
                frame=output,
                source=source,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                calendar=crypto_calendar() if calendar else None,
                adjustment_policy=CRYPTO_ADJUSTMENT_POLICY,
                authoritative_allowed=True,
                source_version=source_version,
                dry_run=dry_run,
                extra_blocking_reasons=None,
            )
        )
    return writes


def _resolve_start(
    *,
    explicit: datetime | str | None,
    end: datetime,
    latest_existing: pd.Timestamp | None,
    interval: str,
    years: int,
    lookback_days: int,
    latest: bool,
    overlap_bars: int,
) -> datetime:
    if explicit is not None:
        return _coerce_datetime(explicit) or end
    if latest and latest_existing is not None:
        delta = _interval_delta(interval) * max(0, overlap_bars)
        return (latest_existing.to_pydatetime() - delta).astimezone(timezone.utc)
    if years > 0:
        return end - timedelta(days=365 * years)
    return end - timedelta(days=max(1, lookback_days))


def _latest_timestamp(root: Path) -> pd.Timestamp | None:
    latest: pd.Timestamp | None = None
    for path in sorted(root.rglob("part.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["timestamp_utc"])
        except Exception:
            continue
        if frame.empty:
            continue
        current = pd.Timestamp(pd.to_datetime(frame["timestamp_utc"], utc=True).max())
        latest = current if latest is None or current > latest else latest
    return latest


def _canonical_bars_root(repo_root: Path, symbol: str, interval: str) -> Path:
    return (
        repo_root
        / "data"
        / "canonical"
        / "bars"
        / "market=crypto_perp"
        / "source=hyperliquid"
        / "kind=trades"
        / f"symbol={symbol.upper()}"
        / f"timeframe={interval}"
    )


def _canonical_funding_root(repo_root: Path, symbol: str) -> Path:
    return (
        repo_root
        / "data"
        / "canonical"
        / "funding"
        / "market=crypto_perp"
        / "source=hyperliquid"
        / f"symbol={symbol.upper()}"
    )


def _raw_path(repo_root: Path, family: str, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    name = f"{_stamp(start)}_{_stamp(end)}.json"
    return (
        repo_root
        / "data"
        / "raw"
        / "hyperliquid"
        / family
        / f"symbol={symbol.upper()}"
        / f"interval={interval}"
        / name
    )


def _operation(
    *,
    kind: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    latest_existing: pd.Timestamp | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "symbol": symbol,
        "interval": interval,
        "download_start": start.isoformat(),
        "download_end": end.isoformat(),
        "latest_existing_ts": latest_existing.isoformat() if latest_existing is not None else "",
        "dry_run": dry_run,
    }


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _interval_delta(interval: str) -> timedelta:
    if interval in {"funding_1h", "funding"}:
        return timedelta(hours=1)
    if interval == "funding_8h":
        return timedelta(hours=8)
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    raise ValueError(f"unsupported interval: {interval}")


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
