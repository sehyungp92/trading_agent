"""Read-only KIS/KRX historical market-data adapter.

KIS remains fail-closed in the production sync CLI. This adapter is a provider-injected
authority slice for mocked/controlled refreshes: it writes raw data, canonical parquet,
lineage, checksums, and deterministic manifests without exposing any order methods.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from trading_assistant_data.calendars.core import (
    expected_trading_dates,
)
from trading_assistant_data.calendars.krx import (
    KIS_INTRADAY_CALENDAR_ID,
    calendar_definition,
    kis_intraday_expected_bar_opens,
    kis_intraday_calendar_definition,
)
from trading_assistant_data.checksums import (
    canonical_json_sha256,
    parquet_content_checksum,
    sha256_file,
    stable_row_hashes,
)
from trading_assistant_data.manifests import MarketDataManifest, MissingRange, write_model
from trading_assistant_data.slices import SliceWrite
from trading_assistant_data.slices.writer import update_slice_index
from trading_assistant_data.repo import git_commit_sha
from trading_assistant_data.source_authority import (
    SourceAuthorityContract,
    kis_krx_authority_contract,
)
from trading_assistant_data.sources.kis.auth import (
    KisCredentialSettings,
    issue_access_token,
)
from trading_assistant_data.sources.kis.read_only_client import KisReadOnlyClient


SUPPORTED_TIMEFRAMES = frozenset({"1m", "5m", "15m", "30m", "1h", "1d"})
SYMBOL_RE = re.compile(r"^\d{6}$")
DEFAULT_PULL_TIME = "2026-05-31T00:00:00Z"
DEFAULT_ADJUSTMENT_POLICY = "krx_split_adjusted_policy_v1"
KIS_INTRADAY_PAGE_ROWS = 120
KIS_MARKET_CLOSE_TIME = "15:30:00"


class KRXBarProvider(Protocol):
    def historical_bars(self, request: "KrxRefreshRequest") -> pd.DataFrame: ...


@dataclass(frozen=True)
class KrxRefreshRequest:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    exchange: str = "KRX"
    currency: str = "KRW"
    market_code: str = "J"
    corporate_action_policy: str = DEFAULT_ADJUSTMENT_POLICY
    calendar_holidays_path: str = ""
    pulled_at_utc: str = DEFAULT_PULL_TIME
    source_version: str = ""
    strategy_data_family: str = ""
    source_request_id: str = ""

    def normalized(self) -> "KrxRefreshRequest":
        return KrxRefreshRequest(
            symbol=self.symbol.strip().zfill(6),
            timeframe=self.timeframe.strip(),
            start=_as_utc_datetime(self.start),
            end=_as_utc_datetime(self.end),
            exchange=self.exchange.upper().strip(),
            currency=self.currency.upper().strip(),
            market_code=self.market_code.upper().strip(),
            corporate_action_policy=self.corporate_action_policy.strip(),
            calendar_holidays_path=self.calendar_holidays_path.strip(),
            pulled_at_utc=self.pulled_at_utc.strip(),
            source_version=self.source_version.strip(),
            strategy_data_family=self.strategy_data_family.strip(),
            source_request_id=self.source_request_id.strip(),
        )

    @property
    def config_hash(self) -> str:
        request = self.normalized()
        return canonical_json_sha256(
            {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "exchange": request.exchange,
                "currency": request.currency,
                "market_code": request.market_code,
                "corporate_action_policy": request.corporate_action_policy,
                "calendar_holidays_checksum": request.calendar_holidays_checksum,
                "strategy_data_family": request.strategy_data_family,
                "source_request_id": request.source_request_id,
            }
        )

    @property
    def calendar_holidays_checksum(self) -> str:
        path = Path(self.normalized().calendar_holidays_path)
        return sha256_file(path) if path.is_file() else ""

    @property
    def export_id(self) -> str:
        request = self.normalized()
        raw = "|".join(
            [
                "kis-krx",
                request.symbol,
                request.timeframe,
                request.start.isoformat(),
                request.end.isoformat(),
                request.config_hash,
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"kis-krx-{request.symbol}-{request.timeframe}-{digest}"


@dataclass(frozen=True)
class KrxRefreshResult:
    status: str
    request: KrxRefreshRequest
    raw_path: Path
    canonical_path: Path
    manifest_path: Path
    manifest: MarketDataManifest

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "symbol": self.request.symbol,
            "timeframe": self.request.timeframe,
            "start": self.request.start.isoformat(),
            "end": self.request.end.isoformat(),
            "raw_path": str(self.raw_path),
            "canonical_path": str(self.canonical_path),
            "manifest_path": str(self.manifest_path),
            "manifest_id": self.manifest.manifest_id,
            "checksum": self.manifest.checksum,
            "usable_for_authoritative_validation": (
                self.manifest.usable_for_authoritative_validation
            ),
            "blocking_reasons": self.manifest.blocking_reasons,
        }


class KISKrxReadOnlyAdapter:
    def __init__(
        self,
        provider: KRXBarProvider,
        *,
        contract: SourceAuthorityContract | None = None,
    ) -> None:
        self.provider = provider
        self.contract = contract or kis_krx_authority_contract()

    def refresh_historical_bars(
        self,
        *,
        repo_root: Path,
        request: KrxRefreshRequest,
        dry_run: bool = False,
    ) -> KrxRefreshResult:
        repo_root = Path(repo_root)
        request = request.normalized()
        _validate_request(request)
        contract_errors = self.contract.validation_errors()
        if contract_errors:
            raise ValueError("; ".join(contract_errors))

        source_version = request.source_version or git_commit_sha(repo_root) or ("0" * 40)
        raw_frame = _normalize_raw_provider_frame(self.provider.historical_bars(request), request)
        expected_bars, source_missing_ranges = _source_coverage(raw_frame, request)

        raw_path = _raw_path(repo_root, request)
        canonical_path = _canonical_path(repo_root, request)
        canonical = _canonical_frame(raw_frame, request, raw_path, repo_root)
        if not dry_run:
            _write_parquet(raw_frame, raw_path)
            _write_parquet(canonical, canonical_path)
        raw_checksum = parquet_content_checksum(raw_path) if raw_path.exists() else ""
        canonical_checksum = (
            parquet_content_checksum(canonical_path) if canonical_path.exists() else ""
        )
        manifest = _manifest(
            request=request,
            source_version=source_version,
            raw_path=raw_path,
            canonical_path=canonical_path,
            raw_checksum=raw_checksum,
            canonical_checksum=canonical_checksum,
            actual_bars=len(canonical),
            expected_bars=expected_bars,
            missing_ranges=source_missing_ranges,
            repo_root=repo_root,
            contract=self.contract,
        )
        manifest_path = _manifest_path(repo_root, manifest)
        if not dry_run:
            write_model(manifest_path, manifest)
            update_slice_index(
                repo_root,
                [SliceWrite(manifest_path, [canonical_path], manifest)],
            )
        return KrxRefreshResult(
            status="planned" if dry_run else "complete",
            request=request,
            raw_path=raw_path,
            canonical_path=canonical_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )


class DeterministicKrxProvider:
    """Deterministic KRX provider for authority and replay fixtures."""

    def historical_bars(self, request: KrxRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        index = _expected_index(request)
        symbol_bias = (sum(ord(char) for char in request.symbol) % 43) * 18.0
        base = 60_000.0 + symbol_bias + request.start.month * 650.0
        rows = []
        for ordinal, ts in enumerate(index):
            drift = ordinal * 4.5
            cycle = ((ordinal % 11) - 5) * 9.0
            open_ = base + drift + cycle
            close = open_ + (18.0 if ordinal % 6 else -6.0)
            high = max(open_, close) + 24.0
            low = min(open_, close) - 24.0
            volume = 15_000 + (ordinal % 31) * 73
            rows.append(
                {
                    "timestamp_utc": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(volume),
                }
            )
        return pd.DataFrame(rows)


class KisApiKrxProvider:
    """Live KIS quotation-history provider for read-only KRX bars."""

    def __init__(
        self,
        client: KisReadOnlyClient | None = None,
        *,
        settings: KisCredentialSettings | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            return
        credentials = issue_access_token(settings or KisCredentialSettings.from_env())
        self.client = KisReadOnlyClient(
            base_url=credentials.base_url,
            app_key=credentials.app_key,
            app_secret=credentials.app_secret,
            access_token=credentials.access_token,
        )

    def historical_bars(self, request: KrxRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        if request.timeframe.lower() in {"1d", "daily"}:
            payload = self.client.get_daily_chart(
                request.symbol,
                request.start.strftime("%Y%m%d"),
                request.end.strftime("%Y%m%d"),
            )
            return _kis_payload_to_frame(payload, request)
        return self._intraday_bars(request)

    def _intraday_bars(self, request: KrxRefreshRequest) -> pd.DataFrame:
        rows: dict[pd.Timestamp, dict] = {}
        start_kst, end_kst = _intraday_fetch_window(request)
        cursor = end_kst.to_pydatetime()
        max_pages = _optional_int_env("KIS_INTRADAY_MAX_PAGES")
        sleep_seconds = _kis_request_sleep_seconds()
        seen_cursors: set[str] = set()
        pages = 0
        one_minute_request = KrxRefreshRequest(
            symbol=request.symbol,
            timeframe="1m",
            start=start_kst.tz_convert("UTC").to_pydatetime(),
            end=end_kst.tz_convert("UTC").to_pydatetime(),
            exchange=request.exchange,
            currency=request.currency,
            market_code=request.market_code,
            corporate_action_policy=request.corporate_action_policy,
            calendar_holidays_path=request.calendar_holidays_path,
            pulled_at_utc=request.pulled_at_utc,
            source_version=request.source_version,
            strategy_data_family=request.strategy_data_family,
            source_request_id=request.source_request_id,
        ).normalized()
        while pd.Timestamp(cursor) >= start_kst:
            cursor_key = cursor.isoformat()
            if cursor_key in seen_cursors:
                raise RuntimeError(
                    f"KIS pagination did not advance for {request.symbol} at {cursor_key}"
                )
            seen_cursors.add(cursor_key)
            if max_pages is not None and pages >= max_pages:
                break
            payload = self.client.get_historical_minute_page(
                request.symbol,
                date_yyyymmdd=cursor.strftime("%Y%m%d"),
                hour_hhmmss=cursor.strftime("%H%M%S"),
                market_code=request.market_code,
                include_previous=True,
            )
            pages += 1
            frame = _kis_payload_to_frame(payload, one_minute_request)
            if frame.empty:
                cursor = _previous_krx_day_close(cursor)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
            for row in frame.to_dict("records"):
                timestamp = pd.Timestamp(row["timestamp_utc"])
                rows[timestamp] = row
            oldest_utc = pd.Timestamp(frame["timestamp_utc"].min())
            oldest_kst = oldest_utc.tz_convert("Asia/Seoul")
            if oldest_kst <= start_kst:
                break
            next_cursor = oldest_kst - pd.Timedelta(minutes=1)
            cursor = (
                _previous_krx_day_close(cursor)
                if next_cursor >= pd.Timestamp(cursor)
                else next_cursor.to_pydatetime()
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if not rows:
            return pd.DataFrame()
        minute_frame = pd.DataFrame(rows.values()).sort_values("timestamp_utc").reset_index(drop=True)
        return _aggregate_intraday_frame(minute_frame, request)


def _validate_request(request: KrxRefreshRequest) -> None:
    if not SYMBOL_RE.match(request.symbol):
        raise ValueError(f"unsupported KRX symbol: {request.symbol}")
    if request.timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported KRX timeframe: {request.timeframe}")
    if request.end < request.start:
        raise ValueError("request end must be >= start")
    if not request.pulled_at_utc:
        raise ValueError("pulled_at_utc is required")
    if not request.corporate_action_policy:
        raise ValueError("corporate_action_policy is required")


def _normalize_raw_provider_frame(frame: pd.DataFrame, request: KrxRefreshRequest) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("provider returned no bars")
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("provider frame missing columns: " + ", ".join(missing))
    result = frame.loc[:, ["timestamp_utc", "open", "high", "low", "close", "volume"]].copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = result.sort_values("timestamp_utc").drop_duplicates(
        subset=["timestamp_utc"], keep="last"
    )
    result["symbol"] = request.symbol
    result["exchange"] = request.exchange
    result["currency"] = request.currency
    result["market_code"] = request.market_code
    return result.reset_index(drop=True)


def _canonical_frame(
    raw_frame: pd.DataFrame,
    request: KrxRefreshRequest,
    raw_path: Path,
    repo_root: Path,
) -> pd.DataFrame:
    frame = raw_frame.loc[:, ["timestamp_utc", "open", "high", "low", "close", "volume"]].copy()
    frame["timestamp_exchange"] = frame["timestamp_utc"].dt.tz_convert("Asia/Seoul").map(
        lambda value: value.isoformat()
    )
    frame["symbol"] = request.symbol
    frame["market"] = "krx_equity"
    frame["source"] = "kis"
    frame["timeframe"] = request.timeframe
    frame["kind"] = "trades"
    frame["source_file"] = _rel(raw_path, repo_root)
    frame["source_row_hash"] = stable_row_hashes(
        raw_frame.loc[:, ["timestamp_utc", "open", "high", "low", "close", "volume"]]
    ).tolist()
    columns = [
        "timestamp_utc",
        "timestamp_exchange",
        "symbol",
        "market",
        "source",
        "timeframe",
        "kind",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_file",
        "source_row_hash",
    ]
    return frame.loc[:, columns]


def _manifest(
    *,
    request: KrxRefreshRequest,
    source_version: str,
    raw_path: Path,
    canonical_path: Path,
    raw_checksum: str,
    canonical_checksum: str,
    actual_bars: int,
    expected_bars: int,
    missing_ranges: list[MissingRange],
    repo_root: Path,
    contract: SourceAuthorityContract,
) -> MarketDataManifest:
    lineage = {
        "source_endpoint": (
            f"kis://uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice/"
            f"{request.market_code}/{request.symbol}"
        ),
        "export_id": request.export_id,
        "pulled_at_utc": request.pulled_at_utc,
        "config_hash": request.config_hash,
        "session_policy": _calendar_id(request),
        "strategy_data_family": request.strategy_data_family,
        "source_request_id": request.source_request_id,
        "corporate_action_policy": request.corporate_action_policy,
        "calendar_holidays_path": request.calendar_holidays_path,
        "calendar_holidays_checksum": request.calendar_holidays_checksum,
        "credential_contract_id": contract.credential_contract_id,
        "adapter_id": contract.adapter_id,
        "authority_contract_id": contract.contract_id,
        "pacing_policy": contract.pacing_policy,
        "timestamp_policy": "source_exchange_timestamp_converted_to_utc_v1",
        "timestamp_audit": "passed",
        "raw_write_checksum": raw_checksum,
        "canonical_write_checksum": canonical_checksum,
        "idempotency_key": canonical_json_sha256(
            {
                "source": "kis",
                "market": "krx_equity",
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "export_id": request.export_id,
            }
        ),
        "raw_path": _rel(raw_path, repo_root),
        "canonical_path": _rel(canonical_path, repo_root),
        "read_only": "true",
        "historical_intraday_page_rows": str(KIS_INTRADAY_PAGE_ROWS),
    }
    lineage_errors = contract.lineage_errors(lineage)
    if missing_ranges:
        lineage_errors.append("source-aware missing ranges present")
    return MarketDataManifest(
        source="kis",
        market="krx_equity",
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_ts=request.start,
        end_ts=request.end,
        expected_bars=expected_bars,
        actual_bars=actual_bars,
        coverage_ratio=actual_bars / expected_bars if expected_bars else 0.0,
        missing_ranges=missing_ranges,
        session_calendar=_calendar_id(request),
        timezone="Asia/Seoul",
        checksum=canonical_checksum,
        source_version=source_version,
        adjustment_policy=request.corporate_action_policy,
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        lineage=lineage,
        usable_for_authoritative_validation=not lineage_errors,
        blocking_reasons=lineage_errors,
    )


def _calendar(request: KrxRefreshRequest):
    holidays_path = Path(request.calendar_holidays_path) if request.calendar_holidays_path else None
    if request.timeframe.lower() in {"1d", "daily"}:
        return calendar_definition(holidays_path if holidays_path and holidays_path.exists() else None)
    return kis_intraday_calendar_definition(
        holidays_path if holidays_path and holidays_path.exists() else None
    )


def _calendar_id(request: KrxRefreshRequest) -> str:
    return "krx_equities_v1" if request.timeframe.lower() in {"1d", "daily"} else KIS_INTRADAY_CALENDAR_ID


def _expected_index(request: KrxRefreshRequest) -> pd.DatetimeIndex:
    calendar = _calendar(request)
    if request.timeframe.lower() in {"1d", "daily"}:
        rows = []
        for trading_date in expected_trading_dates(calendar, request.start, request.end):
            rows.append(
                pd.Timestamp(
                    f"{trading_date.isoformat()} {calendar.session_close}",
                    tz=calendar.timezone,
                ).tz_convert("UTC")
            )
        return pd.DatetimeIndex(rows).sort_values()
    return kis_intraday_expected_bar_opens(calendar, request.timeframe, request.start, request.end)


def _source_coverage(raw_frame: pd.DataFrame, request: KrxRefreshRequest) -> tuple[int, list[MissingRange]]:
    if request.timeframe.lower() in {"1d", "daily"}:
        expected_index = _expected_index(request)
        return len(expected_index), _source_missing_ranges(
            expected_index,
            pd.DatetimeIndex(raw_frame["timestamp_utc"]),
            request.timeframe,
        )
    actual_index = pd.DatetimeIndex(pd.to_datetime(raw_frame["timestamp_utc"], utc=True))
    return len(actual_index), _kis_intraday_missing_trading_dates(actual_index, request)


def _kis_intraday_missing_trading_dates(
    actual_index: pd.DatetimeIndex,
    request: KrxRefreshRequest,
) -> list[MissingRange]:
    calendar = _calendar(request)
    local = actual_index.tz_convert(calendar.timezone)
    actual_dates = {item.date() for item in local}
    expected_dates = [
        trading_date
        for trading_date in expected_trading_dates(calendar, request.start, request.end)
        if pd.Timestamp(request.start).tz_convert(calendar.timezone).date()
        <= trading_date
        <= pd.Timestamp(request.end).tz_convert(calendar.timezone).date()
    ]
    missing_dates = [value for value in expected_dates if value not in actual_dates]
    ranges = []
    for value in missing_dates:
        start = pd.Timestamp(f"{value.isoformat()} 09:00:00", tz=calendar.timezone)
        end = pd.Timestamp(f"{value.isoformat()} {KIS_MARKET_CLOSE_TIME}", tz=calendar.timezone)
        ranges.append(
            MissingRange(
                start_ts=start.tz_convert("UTC").to_pydatetime(),
                end_ts=end.tz_convert("UTC").to_pydatetime(),
                reason="missing KIS intraday trading date",
            )
        )
    return ranges


def _source_missing_ranges(
    expected_index: pd.DatetimeIndex,
    actual_index: pd.DatetimeIndex,
    timeframe: str,
) -> list[MissingRange]:
    actual = pd.DatetimeIndex(pd.to_datetime(actual_index, utc=True)).sort_values().unique()
    missing = expected_index.difference(actual)
    if missing.empty:
        return []
    return _ranges_from_missing(missing, timeframe)


def _ranges_from_missing(missing: pd.DatetimeIndex, timeframe: str) -> list[MissingRange]:
    if missing.empty:
        return []
    if timeframe.lower() in {"1d", "daily"}:
        step = pd.Timedelta(days=1)
    else:
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[timeframe]
        step = pd.Timedelta(minutes=minutes)
    ranges: list[MissingRange] = []
    start = missing[0]
    previous = missing[0]
    for current in missing[1:]:
        if current - previous > step:
            ranges.append(
                MissingRange(
                    start_ts=start.to_pydatetime(),
                    end_ts=previous.to_pydatetime(),
                    reason="missing expected source bar",
                )
            )
            start = current
        previous = current
    ranges.append(
        MissingRange(
            start_ts=start.to_pydatetime(),
            end_ts=previous.to_pydatetime(),
            reason="missing expected source bar",
        )
    )
    return ranges


def _raw_path(repo_root: Path, request: KrxRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "raw"
        / "kis"
        / "read_only_krx"
        / f"symbol={request.symbol}"
        / f"timeframe={request.timeframe}"
        / f"export_id={request.export_id}"
        / "bars.parquet"
    )


def _canonical_path(repo_root: Path, request: KrxRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "canonical"
        / "bars"
        / "market=krx_equity"
        / "source=kis"
        / "kind=read_only_authority"
        / f"symbol={request.symbol}"
        / f"timeframe={request.timeframe}"
        / f"year={request.start.year:04d}"
        / f"month={request.start.month:02d}"
        / "part.parquet"
    )


def _manifest_path(repo_root: Path, manifest: MarketDataManifest) -> Path:
    start = manifest.start_ts.strftime("%Y%m%dT%H%M%SZ")
    end = manifest.end_ts.strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(repo_root)
        / "data"
        / "manifests"
        / "slices"
        / manifest.source
        / manifest.market
        / manifest.symbol
        / manifest.timeframe
        / f"{start}_{end}.market_data_manifest.json"
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _as_utc_datetime(value: datetime) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def _kis_payload_to_frame(payload: dict, request: KrxRefreshRequest) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("output2") or payload.get("output") or []
    if isinstance(rows, dict):
        rows = [rows]
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _kis_row_timestamp(row, request)
        if timestamp is None:
            continue
        close = _first_number(row, "stck_prpr", "stck_clpr", "close", "prpr")
        open_ = _first_number(row, "stck_oprc", "open", "oprc", default=close)
        high = _first_number(row, "stck_hgpr", "high", "hgpr", default=close)
        low = _first_number(row, "stck_lwpr", "low", "lwpr", default=close)
        if min(open_, high, low, close) <= 0:
            continue
        parsed.append(
            {
                "timestamp_utc": timestamp,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": _kis_volume(row, request),
            }
        )
    frame = pd.DataFrame(parsed)
    if frame.empty:
        return frame
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if request.timeframe.lower() not in {"1d", "daily"}:
        frame = _normalize_intraday_volume(frame)
    frame = frame[(frame["timestamp_utc"] >= pd.Timestamp(request.start)) & (frame["timestamp_utc"] <= pd.Timestamp(request.end))]
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def _kis_row_timestamp(row: dict, request: KrxRefreshRequest) -> pd.Timestamp | None:
    date_value = str(
        row.get("stck_bsop_date") or row.get("bsop_date") or row.get("xymd") or row.get("date") or ""
    ).strip()
    time_value = str(
        row.get("stck_cntg_hour") or row.get("cntg_hour") or row.get("xhms") or row.get("time") or ""
    ).strip()
    if not date_value:
        return None
    try:
        if request.timeframe.lower() in {"1d", "daily"}:
            local = pd.Timestamp(f"{date_value} 15:30:00", tz="Asia/Seoul")
        else:
            if not time_value:
                return None
            local = pd.Timestamp(f"{date_value} {time_value.zfill(6)[:6]}", tz="Asia/Seoul")
        return local.tz_convert("UTC")
    except ValueError:
        return None


def _first_number(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                text = str(value).replace(",", "").strip()
                if not text or text == "-":
                    return default
                return float(text)
            except (TypeError, ValueError):
                return default
    return default


def _kis_volume(row: dict, request: KrxRefreshRequest) -> float:
    if request.timeframe.lower() in {"1d", "daily"}:
        return _first_number(row, "acml_vol", "cntg_vol", "volume")
    return _first_number(row, "cntg_vol", "acml_vol", "volume")


def _normalize_intraday_volume(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("timestamp_utc").copy()
    if len(frame) <= 1:
        return frame
    diffs = frame["volume"].diff().fillna(frame["volume"])
    non_negative_ratio = float((diffs >= 0).sum()) / float(len(diffs))
    if non_negative_ratio > 0.95 and float(frame["volume"].iloc[-1]) > float(diffs.sum()) * 3.0:
        frame["volume"] = diffs.clip(lower=0)
    return frame


def _intraday_fetch_window(request: KrxRefreshRequest) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_kst = pd.Timestamp(request.start).tz_convert("Asia/Seoul")
    end_kst = pd.Timestamp(request.end).tz_convert("Asia/Seoul")
    timeframe = request.timeframe.lower()
    if timeframe != "1m":
        end_kst += pd.Timedelta(minutes=_timeframe_minutes(timeframe) - 1)
        session_close = pd.Timestamp(
            f"{end_kst.date().isoformat()} {KIS_MARKET_CLOSE_TIME}", tz="Asia/Seoul"
        )
        end_kst = min(end_kst, session_close)
    return start_kst, end_kst


def _timeframe_minutes(timeframe: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[timeframe]


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def _kis_request_sleep_seconds() -> float:
    explicit = os.getenv("KIS_REQUEST_SLEEP_SECONDS", "").strip()
    if explicit:
        return float(explicit)
    min_interval = float(os.getenv("KIS_REQUEST_MIN_INTERVAL_SECONDS", "2.0"))
    extra_sleep = float(os.getenv("KIS_EXTRA_SLEEP_SECONDS", "0.55"))
    return max(0.0, min_interval + extra_sleep)


def _previous_krx_day_close(cursor: datetime) -> datetime:
    day = pd.Timestamp(cursor).tz_convert("Asia/Seoul").date() - timedelta(days=1)
    return pd.Timestamp(
        f"{day.isoformat()} {KIS_MARKET_CLOSE_TIME}", tz="Asia/Seoul"
    ).to_pydatetime()


def _aggregate_intraday_frame(frame: pd.DataFrame, request: KrxRefreshRequest) -> pd.DataFrame:
    timeframe = request.timeframe.lower()
    if timeframe == "1m":
        return frame
    minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}[timeframe]
    local = frame.copy()
    local["timestamp_exchange"] = pd.to_datetime(local["timestamp_utc"], utc=True).dt.tz_convert("Asia/Seoul")
    local["bucket"] = local["timestamp_exchange"].dt.floor(f"{minutes}min")
    grouped = (
        local.sort_values("timestamp_exchange")
        .groupby("bucket", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    grouped["timestamp_utc"] = pd.to_datetime(grouped["bucket"]).dt.tz_convert("UTC")
    grouped = grouped.loc[
        (grouped["timestamp_utc"] >= pd.Timestamp(request.start))
        & (grouped["timestamp_utc"] <= pd.Timestamp(request.end)),
        ["timestamp_utc", "open", "high", "low", "close", "volume"],
    ]
    return grouped.reset_index(drop=True)
