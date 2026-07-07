"""Read-only IBKR US-equity historical-data adapter.

This adapter is intentionally provider-injected and has no order/account surface. It
exists to prove authority contracts, raw retention, canonical writes, lineage, and
deterministic bundle reproduction before the production sync CLI is allowed to run
non-dry-run IBKR refreshes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from trading_assistant_data.calendars.core import TIMEFRAME_MINUTES
from trading_assistant_data.calendars.us_equities import (
    CALENDAR_ID,
    calendar_definition,
    extended_session_close_for_date,
    extended_session_open,
    session_close_for_date,
)
from trading_assistant_data.checksums import (
    canonical_json_sha256,
    parquet_content_checksum,
    stable_row_hashes,
)
from trading_assistant_data.manifests import MarketDataManifest, write_model
from trading_assistant_data.slices import SliceWrite
from trading_assistant_data.slices.writer import update_slice_index
from trading_assistant_data.repo import git_commit_sha
from trading_assistant_data.source_authority import (
    SourceAuthorityContract,
    ibkr_us_equity_authority_contract,
)


SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .-]{0,15}$")
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "30m", "1h", "1d"})
DEFAULT_PULL_TIME = "2026-05-31T00:00:00Z"
DEFAULT_CORPORATE_ACTION_POLICY = "us_equity_split_dividend_adjusted_policy_v1"
DEFAULT_RAW_ADJUSTMENT_POLICY = "ibkr_trades_unadjusted_raw_v1"


class HistoricalBarProvider(Protocol):
    def historical_bars(self, request: "UsEquityRefreshRequest") -> pd.DataFrame: ...


@dataclass(frozen=True)
class UsEquityRefreshRequest:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    exchange: str = "SMART"
    primary_exchange: str = ""
    sec_type: str = "STK"
    currency: str = "USD"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    corporate_action_policy: str = DEFAULT_CORPORATE_ACTION_POLICY
    raw_adjustment_policy: str = DEFAULT_RAW_ADJUSTMENT_POLICY
    pulled_at_utc: str = DEFAULT_PULL_TIME
    source_version: str = ""
    strategy_data_family: str = ""
    source_request_id: str = ""

    def normalized(self) -> "UsEquityRefreshRequest":
        return UsEquityRefreshRequest(
            symbol=self.symbol.upper().strip(),
            timeframe=self.timeframe.strip(),
            start=_as_utc_datetime(self.start),
            end=_as_utc_datetime(self.end),
            exchange=self.exchange.upper().strip(),
            primary_exchange=self.primary_exchange.upper().strip(),
            sec_type=self.sec_type.upper().strip(),
            currency=self.currency.upper().strip(),
            what_to_show=self.what_to_show.upper().strip(),
            use_rth=self.use_rth,
            corporate_action_policy=self.corporate_action_policy.strip(),
            raw_adjustment_policy=self.raw_adjustment_policy.strip(),
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
                "primary_exchange": request.primary_exchange,
                "sec_type": request.sec_type,
                "currency": request.currency,
                "what_to_show": request.what_to_show,
                "use_rth": request.use_rth,
                "corporate_action_policy": request.corporate_action_policy,
                "raw_adjustment_policy": request.raw_adjustment_policy,
                "strategy_data_family": request.strategy_data_family,
                "source_request_id": request.source_request_id,
            }
        )

    @property
    def export_id(self) -> str:
        request = self.normalized()
        raw = "|".join(
            [
                "ibkr-us-equity",
                request.symbol,
                request.timeframe,
                request.start.isoformat(),
                request.end.isoformat(),
                request.config_hash,
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"ibkr-us-{request.symbol.lower()}-{request.timeframe}-{digest}"


@dataclass(frozen=True)
class UsEquityRefreshResult:
    status: str
    request: UsEquityRefreshRequest
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


class IBKRUsEquityReadOnlyAdapter:
    def __init__(
        self,
        provider: HistoricalBarProvider,
        *,
        contract: SourceAuthorityContract | None = None,
    ) -> None:
        self.provider = provider
        self.contract = contract or ibkr_us_equity_authority_contract()

    def refresh_historical_bars(
        self,
        *,
        repo_root: Path,
        request: UsEquityRefreshRequest,
        dry_run: bool = False,
    ) -> UsEquityRefreshResult:
        repo_root = Path(repo_root)
        request = request.normalized()
        _validate_request(request)
        contract_errors = self.contract.validation_errors()
        if contract_errors:
            raise ValueError("; ".join(contract_errors))

        source_version = request.source_version or git_commit_sha(repo_root) or ("0" * 40)
        raw_frame = _normalize_raw_provider_frame(self.provider.historical_bars(request), request)
        expected_index = _expected_index(request)
        if _requires_dense_calendar_match(request):
            if len(raw_frame) != len(expected_index):
                raise ValueError(
                    f"provider returned {len(raw_frame)} bars but US equity calendar expected "
                    f"{len(expected_index)}"
                )
            if list(pd.DatetimeIndex(raw_frame["timestamp_utc"])) != list(expected_index):
                raise ValueError("provider timestamps do not match the US equity session calendar")
        else:
            unexpected = pd.DatetimeIndex(raw_frame["timestamp_utc"]).difference(expected_index)
            if len(unexpected):
                preview = ", ".join(value.isoformat() for value in unexpected[:5])
                raise ValueError(
                    "provider returned extended-hours bars outside the modeled US equity "
                    f"session calendar: {preview}"
                )
            coverage_holes = _sparse_full_trading_day_holes(raw_frame, request)
            if coverage_holes:
                raise ValueError(
                    "provider returned no extended-hours/RTH trade bars for consecutive "
                    "US equity trading dates: "
                    + ", ".join(coverage_holes[:5])
                )

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
            canonical=canonical,
            actual_bars=len(canonical),
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
        return UsEquityRefreshResult(
            status="planned" if dry_run else "complete",
            request=request,
            raw_path=raw_path,
            canonical_path=canonical_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )


class DeterministicUsEquityProvider:
    """Deterministic historical-bar provider for US equity authority fixtures."""

    def historical_bars(self, request: UsEquityRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        index = _expected_index(request)
        symbol_bias = (sum(ord(char) for char in request.symbol) % 37) * 0.8
        base = 100.0 + symbol_bias + request.start.month * 4.0
        rows = []
        for ordinal, ts in enumerate(index):
            drift = ordinal * 0.025
            cycle = ((ordinal % 13) - 6) * 0.04
            open_ = base + drift + cycle
            close = open_ + (0.12 if ordinal % 7 else -0.04)
            high = max(open_, close) + 0.18
            low = min(open_, close) - 0.18
            volume = 20_000 + (ordinal % 29) * 137
            rows.append(
                {
                    "timestamp_utc": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(volume),
                    "source_contract": request.symbol,
                    "source_conid": f"det-{request.symbol}",
                    "source_local_symbol": request.symbol,
                    "source_primary_exchange": request.primary_exchange or "DETERMINISTIC",
                    "contract_resolution_method": "deterministic_fixture",
                }
            )
        return pd.DataFrame(rows)


def _validate_request(request: UsEquityRefreshRequest) -> None:
    if not SYMBOL_RE.match(request.symbol):
        raise ValueError(f"unsupported US equity symbol: {request.symbol}")
    if request.timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported US equity timeframe: {request.timeframe}")
    if request.end < request.start:
        raise ValueError("request end must be >= start")
    if not request.pulled_at_utc:
        raise ValueError("pulled_at_utc is required")
    if not request.corporate_action_policy:
        raise ValueError("corporate_action_policy is required")
    if not request.raw_adjustment_policy:
        raise ValueError("raw_adjustment_policy is required")


def _normalize_raw_provider_frame(
    frame: pd.DataFrame, request: UsEquityRefreshRequest
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("provider returned no bars")
    required = {"timestamp_utc", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("provider frame missing columns: " + ", ".join(missing))
    columns = ["timestamp_utc", "open", "high", "low", "close", "volume"]
    optional_columns = [
        "source_contract",
        "source_conid",
        "source_local_symbol",
        "source_primary_exchange",
        "contract_resolution_method",
    ]
    columns.extend(column for column in optional_columns if column in frame.columns)
    result = frame.loc[:, columns].copy()
    if request.timeframe.lower() in {"1d", "daily"}:
        result["timestamp_utc"] = _daily_timestamps_at_session_close(result["timestamp_utc"])
    else:
        result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = result.sort_values("timestamp_utc").drop_duplicates(
        subset=["timestamp_utc"], keep="last"
    )
    result["symbol"] = request.symbol
    result["exchange"] = request.exchange
    result["primary_exchange"] = request.primary_exchange
    result["sec_type"] = request.sec_type
    result["currency"] = request.currency
    result["what_to_show"] = request.what_to_show
    result["use_rth"] = request.use_rth
    if "source_contract" not in result.columns:
        result["source_contract"] = request.symbol
    if "source_local_symbol" not in result.columns:
        result["source_local_symbol"] = request.symbol
    if "source_primary_exchange" not in result.columns:
        result["source_primary_exchange"] = request.primary_exchange
    for column in ("source_conid", "contract_resolution_method"):
        if column not in result.columns:
            result[column] = ""
    return result.reset_index(drop=True)


def _canonical_frame(
    raw_frame: pd.DataFrame,
    request: UsEquityRefreshRequest,
    raw_path: Path,
    repo_root: Path,
) -> pd.DataFrame:
    frame = raw_frame.loc[
        :,
        [
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_contract",
            "source_conid",
            "source_local_symbol",
            "source_primary_exchange",
            "contract_resolution_method",
        ],
    ].copy()
    frame["timestamp_exchange"] = frame["timestamp_utc"].dt.tz_convert("America/New_York").map(
        lambda value: value.isoformat()
    )
    frame["symbol"] = request.symbol
    frame["market"] = "us_equity"
    frame["source"] = "ibkr"
    frame["timeframe"] = request.timeframe
    frame["kind"] = "trades"
    frame["source_file"] = _rel(raw_path, repo_root)
    frame["is_rth"] = request.use_rth
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
        "source_contract",
        "source_conid",
        "source_local_symbol",
        "source_primary_exchange",
        "contract_resolution_method",
        "source_file",
        "is_rth",
        "source_row_hash",
    ]
    return frame.loc[:, columns]


def _manifest(
    *,
    request: UsEquityRefreshRequest,
    source_version: str,
    raw_path: Path,
    canonical_path: Path,
    raw_checksum: str,
    canonical_checksum: str,
    canonical: pd.DataFrame,
    actual_bars: int,
    repo_root: Path,
    contract: SourceAuthorityContract,
) -> MarketDataManifest:
    lineage = {
        "source_endpoint": (
            f"ibkr://historical-data/{request.sec_type}/{request.exchange}/"
            f"{request.primary_exchange}/{request.symbol}/{request.what_to_show}/"
            f"{'RTH' if request.use_rth else 'ETH'}"
        ),
        "export_id": request.export_id,
        "pulled_at_utc": request.pulled_at_utc,
        "config_hash": request.config_hash,
        "corporate_action_policy": request.corporate_action_policy,
        "raw_adjustment_policy": request.raw_adjustment_policy,
        "session_policy": _session_policy(request),
        "session_calendar": CALENDAR_ID,
        "strategy_data_family": request.strategy_data_family,
        "source_request_id": request.source_request_id,
        "expected_bar_policy": (
            "dense_rth_or_daily"
            if _requires_dense_calendar_match(request)
            else "sparse_extended_hours_trade_bars"
        ),
        "max_session_slots": str(len(_expected_index(request))),
        "use_rth": str(request.use_rth).lower(),
        "source_conid_coverage": _source_conid_coverage(canonical),
        "contract_resolution_cache": _contract_resolution_cache(canonical),
        "source_request_params_json": _source_request_params_json(request),
        "source_request_params_hash": canonical_json_sha256(_source_request_params(request)),
        "returned_row_count": str(actual_bars),
        "missing_full_trading_dates": ",".join(
            _missing_sparse_trading_dates(canonical, request)
        ),
        "multi_day_coverage_holes": ",".join(
            _sparse_full_trading_day_holes(canonical, request)
        ),
        "credential_contract_id": contract.credential_contract_id,
        "adapter_id": contract.adapter_id,
        "authority_contract_id": contract.contract_id,
        "pacing_policy": contract.pacing_policy,
        "raw_write_checksum": raw_checksum,
        "canonical_write_checksum": canonical_checksum,
        "idempotency_key": canonical_json_sha256(
            {
                "source": "ibkr",
                "market": "us_equity",
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
    }
    lineage_errors = contract.lineage_errors(lineage)
    if lineage["source_conid_coverage"] != "all_rows":
        lineage_errors.append("source_conid coverage incomplete")
    expected = _manifest_expected_bars(request, actual_bars)
    return MarketDataManifest(
        source="ibkr",
        market="us_equity",
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_ts=request.start,
        end_ts=request.end,
        expected_bars=expected,
        actual_bars=actual_bars,
        coverage_ratio=actual_bars / expected if expected else 0.0,
        missing_ranges=[],
        session_calendar=CALENDAR_ID,
        timezone="America/New_York",
        checksum=canonical_checksum,
        source_version=source_version,
        adjustment_policy=request.corporate_action_policy,
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        lineage=lineage,
        usable_for_authoritative_validation=not lineage_errors,
        blocking_reasons=lineage_errors,
    )


def _requires_dense_calendar_match(request: UsEquityRefreshRequest) -> bool:
    return request.use_rth or request.timeframe.lower() in {"1d", "daily"}


def _manifest_expected_bars(request: UsEquityRefreshRequest, actual_bars: int) -> int:
    if _requires_dense_calendar_match(request):
        return len(_expected_index(request))
    return actual_bars


def _missing_sparse_trading_dates(
    frame: pd.DataFrame,
    request: UsEquityRefreshRequest,
) -> list[str]:
    if _requires_dense_calendar_match(request):
        return []
    expected_dates = _expected_sparse_trading_dates(request)
    observed_dates = _observed_exchange_dates(frame, request)
    return [
        value.isoformat()
        for value in expected_dates
        if value not in observed_dates
    ]


def _sparse_full_trading_day_holes(
    frame: pd.DataFrame,
    request: UsEquityRefreshRequest,
) -> list[str]:
    missing = set(_missing_sparse_trading_dates(frame, request))
    if not missing:
        return []
    dates = [value.isoformat() for value in _expected_sparse_trading_dates(request)]
    holes: list[str] = []
    run: list[str] = []
    for value in dates:
        if value in missing:
            run.append(value)
            continue
        if len(run) >= 2:
            holes.append(f"{run[0]}..{run[-1]}")
        run = []
    if len(run) >= 2:
        holes.append(f"{run[0]}..{run[-1]}")
    return holes


def _expected_sparse_trading_dates(request: UsEquityRefreshRequest) -> list:
    expected = _expected_index(request)
    if expected.empty:
        return []
    tz = ZoneInfo(calendar_definition().timezone)
    return sorted({value.date() for value in expected.tz_convert(tz)})


def _observed_exchange_dates(
    frame: pd.DataFrame,
    request: UsEquityRefreshRequest,
) -> set:
    if frame.empty:
        return set()
    tz = ZoneInfo(calendar_definition().timezone)
    return set(pd.DatetimeIndex(frame["timestamp_utc"]).tz_convert(tz).date)


def _source_request_params_json(request: UsEquityRefreshRequest) -> str:
    return json.dumps(_source_request_params(request), sort_keys=True, separators=(",", ":"))


def _source_request_params(request: UsEquityRefreshRequest) -> dict[str, object]:
    return {
        "provider": "ibkr",
        "request_type": "historical_bars",
        "sec_type": request.sec_type,
        "symbol": request.symbol,
        "exchange": request.exchange,
        "primary_exchange": request.primary_exchange,
        "currency": request.currency,
        "timeframe": request.timeframe,
        "what_to_show": request.what_to_show,
        "use_rth": request.use_rth,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "corporate_action_policy": request.corporate_action_policy,
        "raw_adjustment_policy": request.raw_adjustment_policy,
        "strategy_data_family": request.strategy_data_family,
        "source_request_id": request.source_request_id,
    }


def _session_policy(request: UsEquityRefreshRequest) -> str:
    return (
        "us_equity_rth_session_0930_1600_new_york_v1"
        if _requires_dense_calendar_match(request)
        else "us_equity_extended_session_0400_2000_new_york_v1"
    )


def _expected_index(request: UsEquityRefreshRequest) -> pd.DatetimeIndex:
    calendar = calendar_definition()
    if request.timeframe.lower() in {"1d", "daily"}:
        rows = []
        current = request.start.date()
        final = request.end.date()
        while current <= final:
            if not calendar.is_trading_day(current):
                current += timedelta(days=1)
                continue
            rows.append(
                pd.Timestamp(
                    f"{current.isoformat()} {calendar.session_close}",
                    tz=calendar.timezone,
                ).tz_convert("UTC")
            )
            current += timedelta(days=1)
        return pd.DatetimeIndex(rows).sort_values()
    if not request.use_rth:
        return _expected_ibkr_extended_intraday_index(request)
    return _expected_ibkr_rth_intraday_index(request)


def _expected_ibkr_rth_intraday_index(request: UsEquityRefreshRequest) -> pd.DatetimeIndex:
    calendar = calendar_definition()
    minutes = TIMEFRAME_MINUTES[request.timeframe.lower()]
    start_utc = pd.Timestamp(request.start)
    end_utc = pd.Timestamp(request.end)
    tz = ZoneInfo(calendar.timezone)
    start_local = start_utc.tz_convert(tz).to_pydatetime()
    end_local = end_utc.tz_convert(tz).to_pydatetime()
    current = start_local.date()
    final = end_local.date()
    expected: list[pd.Timestamp] = []
    while current <= final:
        if calendar.is_trading_day(current):
            session_open = pd.Timestamp(f"{current.isoformat()} {calendar.session_open}", tz=tz)
            session_close = pd.Timestamp(
                datetime.combine(current, session_close_for_date(current), tz)
            )
            if minutes == 60:
                opens = _ibkr_hourly_rth_opens(session_open, session_close)
            else:
                opens = pd.date_range(
                    start=session_open,
                    end=session_close - timedelta(minutes=minutes),
                    freq=f"{minutes}min",
                )
            opens_utc = opens.tz_convert("UTC") if opens.size else opens
            expected.extend(opens_utc[(opens_utc >= start_utc) & (opens_utc <= end_utc)])
        current += timedelta(days=1)
    return pd.DatetimeIndex(expected).sort_values()


def _expected_ibkr_extended_intraday_index(request: UsEquityRefreshRequest) -> pd.DatetimeIndex:
    calendar = calendar_definition()
    minutes = TIMEFRAME_MINUTES[request.timeframe.lower()]
    start_utc = pd.Timestamp(request.start)
    end_utc = pd.Timestamp(request.end)
    tz = ZoneInfo(calendar.timezone)
    start_local = start_utc.tz_convert(tz).to_pydatetime()
    end_local = end_utc.tz_convert(tz).to_pydatetime()
    current = start_local.date()
    final = end_local.date()
    expected: list[pd.Timestamp] = []
    while current <= final:
        if calendar.is_trading_day(current):
            session_open = pd.Timestamp(datetime.combine(current, extended_session_open(), tz))
            session_close = pd.Timestamp(
                datetime.combine(current, extended_session_close_for_date(current), tz)
            )
            opens = pd.date_range(
                start=session_open,
                end=session_close - timedelta(minutes=minutes),
                freq=f"{minutes}min",
            )
            opens_utc = opens.tz_convert("UTC") if opens.size else opens
            expected.extend(opens_utc[(opens_utc >= start_utc) & (opens_utc <= end_utc)])
        current += timedelta(days=1)
    return pd.DatetimeIndex(expected).sort_values()


def _ibkr_hourly_rth_opens(
    session_open: pd.Timestamp,
    session_close: pd.Timestamp,
) -> pd.DatetimeIndex:
    """IBKR labels RTH 1h bars as 09:30, 10:00, 11:00 ... 15:00 local."""

    if session_close <= session_open:
        return pd.DatetimeIndex([], tz=session_open.tz)
    first_whole_hour = (session_open + pd.Timedelta(hours=1)).floor("h")
    whole_hour_end = session_close - pd.Timedelta(hours=1)
    opens = [session_open]
    if first_whole_hour <= whole_hour_end:
        opens.extend(pd.date_range(start=first_whole_hour, end=whole_hour_end, freq="60min"))
    return pd.DatetimeIndex(opens)


def _daily_timestamps_at_session_close(values: pd.Series) -> pd.Series:
    calendar = calendar_definition()
    close_time = calendar.session_close
    tz = ZoneInfo(calendar.timezone)
    aligned = []
    for value in values:
        trading_date = pd.Timestamp(value).date()
        aligned.append(
            pd.Timestamp(f"{trading_date.isoformat()} {close_time}", tz=tz).tz_convert("UTC")
        )
    return pd.Series(pd.DatetimeIndex(aligned), index=values.index)


def _source_conid_coverage(canonical: pd.DataFrame) -> str:
    if "source_conid" not in canonical.columns:
        return ""
    values = canonical["source_conid"].astype(str).str.strip()
    return "all_rows" if not values.empty and values.ne("").all() else ""


def _contract_resolution_cache(canonical: pd.DataFrame) -> str:
    required = {
        "source_contract",
        "source_conid",
        "source_local_symbol",
        "source_primary_exchange",
        "contract_resolution_method",
    }
    if not required.issubset(canonical.columns):
        return ""
    rows = (
        canonical.loc[
            :,
            [
                "source_contract",
                "source_conid",
                "source_local_symbol",
                "source_primary_exchange",
                "contract_resolution_method",
            ],
        ]
        .drop_duplicates()
        .sort_values(["source_contract", "source_conid"])
        .to_dict("records")
    )
    if not rows:
        return ""
    for row in rows:
        if not str(row.get("source_conid", "")).strip():
            return ""
    return canonical_json_sha256(rows)


def _raw_path(repo_root: Path, request: UsEquityRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "raw"
        / "ibkr"
        / "read_only_us_equity"
        / f"symbol={request.symbol}"
        / f"timeframe={request.timeframe}"
        / f"export_id={request.export_id}"
        / "bars.parquet"
    )


def _canonical_path(repo_root: Path, request: UsEquityRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "canonical"
        / "bars"
        / "market=us_equity"
        / "source=ibkr"
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
