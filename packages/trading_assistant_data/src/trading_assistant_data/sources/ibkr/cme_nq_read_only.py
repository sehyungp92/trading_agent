"""Read-only IBKR CME/NQ historical-data adapter.

The adapter has no live-order surface and requires an injected provider. This keeps the
production CLI fail-closed while allowing tests and controlled refresh jobs to exercise
raw writes, canonical writes, lineage, checksums, and idempotency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from trading_assistant_data.calendars.cme import (
    CALENDAR_ID,
    calendar_definition,
    rule_authority_checksum,
    rule_authority_path,
)
from trading_assistant_data.calendars.core import expected_bar_opens
from trading_assistant_data.checksums import (
    canonical_json_sha256,
    parquet_content_checksum,
    stable_row_hashes,
)
from trading_assistant_data.manifests import MarketDataManifest, MissingRange, write_model
from trading_assistant_data.slices import SliceWrite
from trading_assistant_data.slices.writer import update_slice_index
from trading_assistant_data.repo import git_commit_sha
from trading_assistant_data.source_authority import (
    SourceAuthorityContract,
    ibkr_cme_nq_authority_contract,
)


SUPPORTED_SYMBOLS = frozenset({"NQ", "MNQ", "ES", "MES"})
SUPPORTED_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1h", "4h", "1d"})
DEFAULT_ROLL_POLICY = "cme_quarterly_four_calendar_days_before_third_friday_v1"
DEFAULT_PULL_TIME = "2026-05-31T00:00:00Z"
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
CODE_MONTHS = {value: key for key, value in MONTH_CODES.items()}


class HistoricalBarProvider(Protocol):
    def historical_bars(self, request: "CmeNqRefreshRequest") -> pd.DataFrame: ...


@dataclass(frozen=True)
class CmeNqRefreshRequest:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    exchange: str = "CME"
    sec_type: str = "FUT"
    currency: str = "USD"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    roll_policy: str = DEFAULT_ROLL_POLICY
    contract_chain: tuple[str, ...] = ()
    pulled_at_utc: str = DEFAULT_PULL_TIME
    source_version: str = ""
    strategy_data_family: str = ""
    source_request_id: str = ""

    def normalized(self) -> "CmeNqRefreshRequest":
        return CmeNqRefreshRequest(
            symbol=self.symbol.upper().strip(),
            timeframe=self.timeframe.strip(),
            start=_as_utc_datetime(self.start),
            end=_as_utc_datetime(self.end),
            exchange=self.exchange.upper().strip(),
            sec_type=self.sec_type.upper().strip(),
            currency=self.currency.upper().strip(),
            what_to_show=self.what_to_show.upper().strip(),
            use_rth=self.use_rth,
            roll_policy=self.roll_policy.strip(),
            contract_chain=_normalized_contract_chain(
                self.symbol,
                self.start,
                self.end,
                self.contract_chain,
            ),
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
                "sec_type": request.sec_type,
                "currency": request.currency,
                "what_to_show": request.what_to_show,
                "use_rth": request.use_rth,
                "roll_policy": request.roll_policy,
                "contract_chain": list(request.contract_chain),
                "strategy_data_family": request.strategy_data_family,
                "source_request_id": request.source_request_id,
            }
        )

    @property
    def contract_chain_checksum(self) -> str:
        return canonical_json_sha256(
            {
                "roll_policy": self.normalized().roll_policy,
                "contract_chain": list(self.normalized().contract_chain),
            }
        )

    @property
    def continuous_construction_checksum(self) -> str:
        request = self.normalized()
        return canonical_json_sha256(
            {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "roll_policy": request.roll_policy,
                "contract_chain": list(request.contract_chain),
                "construction": "ibkr_physical_contract_chain_panama_v1",
            }
        )

    @property
    def export_id(self) -> str:
        request = self.normalized()
        raw = "|".join(
            [
                "ibkr",
                request.symbol,
                request.timeframe,
                request.start.isoformat(),
                request.end.isoformat(),
                request.config_hash,
            ]
        )
        return f"ibkr-{request.symbol.lower()}-{request.timeframe}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class CmeNqRefreshResult:
    status: str
    request: CmeNqRefreshRequest
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


class IBKRCmeNqReadOnlyAdapter:
    def __init__(
        self,
        provider: HistoricalBarProvider,
        *,
        contract: SourceAuthorityContract | None = None,
    ) -> None:
        self.provider = provider
        self.contract = contract or ibkr_cme_nq_authority_contract()

    def refresh_historical_bars(
        self,
        *,
        repo_root: Path,
        request: CmeNqRefreshRequest,
        dry_run: bool = False,
    ) -> CmeNqRefreshResult:
        repo_root = Path(repo_root)
        request = request.normalized()
        _validate_request(request)
        contract_errors = self.contract.validation_errors()
        if contract_errors:
            raise ValueError("; ".join(contract_errors))

        source_version = request.source_version or git_commit_sha(repo_root)
        if not source_version:
            source_version = "0" * 40
        raw_frame = _normalize_raw_provider_frame(self.provider.historical_bars(request), request)
        expected_index = _expected_index(request)
        source_missing_ranges = _source_missing_ranges(
            expected_index,
            pd.DatetimeIndex(raw_frame["timestamp_utc"]),
            request.timeframe,
        )

        raw_path = _raw_path(repo_root, request)
        canonical_path = _canonical_path(repo_root, request)
        canonical = _canonical_frame(raw_frame, request, raw_path)
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
            expected_bars=len(expected_index),
            missing_ranges=source_missing_ranges,
            repo_root=repo_root,
            contract=self.contract,
            source_contract_coverage=_source_contract_coverage(canonical),
            source_conid_coverage=_source_conid_coverage(canonical),
            contract_resolution_cache=_contract_resolution_cache(canonical),
        )
        manifest_path = _manifest_path(repo_root, manifest)
        if not dry_run:
            write_model(manifest_path, manifest)
            update_slice_index(
                repo_root,
                [SliceWrite(manifest_path, [canonical_path], manifest)],
            )
        return CmeNqRefreshResult(
            status="planned" if dry_run else "complete",
            request=request,
            raw_path=raw_path,
            canonical_path=canonical_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )


class DeterministicCmeNqProvider:
    """Deterministic historical-bar provider for contract and replay fixtures."""

    def historical_bars(self, request: CmeNqRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        index = _expected_index(request)
        rows = []
        base = 17_000.0 + request.start.month * 180.0
        for ordinal, ts in enumerate(index):
            drift = ordinal * 0.75
            cycle = ((ordinal % 17) - 8) * 1.25
            open_ = base + drift + cycle
            close = open_ + (2.5 if ordinal % 5 else -1.5)
            high = max(open_, close) + 4.0
            low = min(open_, close) - 4.0
            volume = 900 + (ordinal % 23) * 11
            rows.append(
                {
                    "timestamp_utc": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(volume),
                    "source_contract": _source_contract_for_timestamp(request, ts),
                    "source_conid": f"det-{_source_contract_for_timestamp(request, ts)}",
                    "contract_resolution_method": "deterministic_fixture",
                    "contract_last_trade_date": _contract_expiry_text(
                        request,
                        _source_contract_for_timestamp(request, ts),
                    ),
                }
            )
        return pd.DataFrame(rows)


def _validate_request(request: CmeNqRefreshRequest) -> None:
    if request.symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"unsupported CME equity-index futures symbol: {request.symbol}")
    if request.timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported CME futures timeframe: {request.timeframe}")
    if request.end < request.start:
        raise ValueError("request end must be >= start")
    if not request.contract_chain:
        raise ValueError("contract_chain is required")
    if not request.pulled_at_utc:
        raise ValueError("pulled_at_utc is required")


def _normalize_raw_provider_frame(frame: pd.DataFrame, request: CmeNqRefreshRequest) -> pd.DataFrame:
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
        "contract_resolution_method",
        "contract_last_trade_date",
    ]
    columns.extend(column for column in optional_columns if column in frame.columns)
    result = frame.loc[:, columns].copy()
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = result.sort_values("timestamp_utc").drop_duplicates(
        subset=["timestamp_utc"], keep="last"
    )
    result["symbol"] = request.symbol
    result["exchange"] = request.exchange
    result["sec_type"] = request.sec_type
    result["currency"] = request.currency
    result["what_to_show"] = request.what_to_show
    result["use_rth"] = request.use_rth
    if "source_contract" not in result.columns:
        result["source_contract"] = [
            _source_contract_for_timestamp(request, ts) for ts in result["timestamp_utc"]
        ]
    for column in ("source_conid", "contract_resolution_method", "contract_last_trade_date"):
        if column not in result.columns:
            result[column] = ""
    return result.reset_index(drop=True)


def _canonical_frame(
    raw_frame: pd.DataFrame,
    request: CmeNqRefreshRequest,
    raw_path: Path,
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
            "contract_resolution_method",
            "contract_last_trade_date",
        ],
    ].copy()
    frame["timestamp_exchange"] = frame["timestamp_utc"].dt.tz_convert("America/Chicago").map(
        lambda value: value.isoformat()
    )
    frame["symbol"] = request.symbol
    frame["market"] = "cme_futures"
    frame["source"] = "ibkr"
    frame["timeframe"] = request.timeframe
    frame["kind"] = "trades"
    frame["source_file"] = _rel(raw_path, raw_path.parents[5])
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
        "contract_resolution_method",
        "contract_last_trade_date",
        "source_file",
        "is_rth",
        "source_row_hash",
    ]
    return frame.loc[:, columns]


def _manifest(
    *,
    request: CmeNqRefreshRequest,
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
    source_contract_coverage: str,
    source_conid_coverage: str,
    contract_resolution_cache: str,
) -> MarketDataManifest:
    market_rule_checksum = rule_authority_checksum(repo_root)
    lineage = {
        "source_endpoint": (
            f"ibkr://historical-data/{request.sec_type}/{request.exchange}/"
            f"{request.symbol}/{request.what_to_show}"
        ),
        "export_id": request.export_id,
        "pulled_at_utc": request.pulled_at_utc,
        "config_hash": request.config_hash,
        "session_policy": CALENDAR_ID,
        "strategy_data_family": request.strategy_data_family,
        "source_request_id": request.source_request_id,
        "market_rule_authority_path": _rel(rule_authority_path(repo_root), repo_root),
        "market_rule_authority_checksum": market_rule_checksum,
        "roll_policy": request.roll_policy,
        "contract_chain_checksum": request.contract_chain_checksum,
        "continuous_construction_checksum": request.continuous_construction_checksum,
        "panama_quality_guard": "equity_index_min_gap_points_500_v1",
        "source_contract_coverage": source_contract_coverage,
        "source_conid_coverage": source_conid_coverage,
        "contract_resolution_cache": contract_resolution_cache,
        "credential_contract_id": contract.credential_contract_id,
        "adapter_id": contract.adapter_id,
        "authority_contract_id": contract.contract_id,
        "pacing_policy": contract.pacing_policy,
        "raw_write_checksum": raw_checksum,
        "canonical_write_checksum": canonical_checksum,
        "idempotency_key": canonical_json_sha256(
            {
                "source": "ibkr",
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
    if source_conid_coverage != "all_rows":
        lineage_errors.append("source_conid coverage incomplete")
    if missing_ranges:
        lineage_errors.append("source-aware missing ranges present")
    return MarketDataManifest(
        source="ibkr",
        market="cme_futures",
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_ts=request.start,
        end_ts=request.end,
        expected_bars=expected_bars,
        actual_bars=actual_bars,
        coverage_ratio=actual_bars / expected_bars if expected_bars else 0.0,
        missing_ranges=missing_ranges,
        session_calendar=CALENDAR_ID,
        timezone="America/Chicago",
        checksum=canonical_checksum,
        source_version=source_version,
        adjustment_policy="cme_futures_panama_v1",
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        lineage=lineage,
        usable_for_authoritative_validation=not lineage_errors,
        blocking_reasons=lineage_errors,
    )


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
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}[timeframe]
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


def _expected_index(request: CmeNqRefreshRequest) -> pd.DatetimeIndex:
    calendar = calendar_definition()
    if request.timeframe.lower() in {"1d", "daily"}:
        start_date = pd.Timestamp(request.start).date()
        end_date = pd.Timestamp(request.end).date()
        current = start_date
        rows = []
        while current <= end_date:
            if current.weekday() < 5 and current not in calendar.holidays:
                rows.append(pd.Timestamp(current, tz="UTC"))
            current += pd.Timedelta(days=1).to_pytimedelta()
        return pd.DatetimeIndex(rows).sort_values()
    return expected_bar_opens(calendar, request.timeframe, request.start, request.end)


def _normalized_contract_chain(
    symbol: str,
    start: datetime,
    end: datetime,
    contract_chain: tuple[str, ...],
) -> tuple[str, ...]:
    cleaned = tuple(item.strip().upper() for item in contract_chain if item.strip())
    if cleaned:
        return cleaned
    request_start = _as_utc_datetime(start)
    request_end = _as_utc_datetime(end)
    return tuple(_quarter_contracts(symbol.upper().strip(), request_start.date(), request_end.date()))


def _quarter_contracts(symbol: str, start_date: date, end_date: date) -> list[str]:
    contracts: list[str] = []
    for year in range(start_date.year - 1, end_date.year + 2):
        for month, code in MONTH_CODES.items():
            expiry = _third_friday(year, month)
            roll_date = expiry - timedelta(days=4)
            if roll_date < start_date or date(year, month, 1) > end_date + timedelta(days=120):
                continue
            contracts.append(f"{symbol}{code}{year % 10}")
    return contracts


def _source_contract_for_timestamp(request: CmeNqRefreshRequest, timestamp: pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp).tz_convert("UTC") if pd.Timestamp(timestamp).tzinfo else pd.Timestamp(timestamp).tz_localize("UTC")
    contracts = list(request.normalized().contract_chain)
    if not contracts:
        return request.symbol
    base_decade = (request.start.year // 10) * 10
    for contract in contracts:
        expiry = _expiry_for_contract(contract, base_decade)
        if expiry is None:
            continue
        roll_date = expiry - timedelta(days=4)
        if ts.date() <= roll_date:
            return contract
    return contracts[-1]


def _contract_expiry_text(request: CmeNqRefreshRequest, contract: str) -> str:
    expiry = _expiry_for_contract(contract, (request.start.year // 10) * 10)
    return expiry.strftime("%Y%m%d") if expiry is not None else ""


def _expiry_for_contract(contract: str, base_decade: int) -> date | None:
    if len(contract) < 3:
        return None
    code = contract[-2]
    if code not in CODE_MONTHS:
        return None
    year_digit = int(contract[-1])
    year = base_decade + year_digit
    if year < base_decade:
        year += 10
    return _third_friday(year, CODE_MONTHS[code])


def _third_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    first_friday_offset = (4 - current.weekday()) % 7
    return current + timedelta(days=first_friday_offset + 14)


def _source_contract_coverage(canonical: pd.DataFrame) -> str:
    if "source_contract" not in canonical.columns:
        return ""
    values = canonical["source_contract"].astype(str).str.strip()
    return "all_rows" if bool((values != "").all()) else "partial"


def _source_conid_coverage(canonical: pd.DataFrame) -> str:
    if "source_conid" not in canonical.columns:
        return ""
    values = canonical["source_conid"].astype(str).str.strip()
    return "all_rows" if bool((values != "").all()) else "partial"


def _contract_resolution_cache(canonical: pd.DataFrame) -> str:
    required = {
        "source_contract",
        "source_conid",
        "contract_resolution_method",
        "contract_last_trade_date",
    }
    if not required.issubset(canonical.columns):
        return ""
    entries = (
        canonical.loc[
            :,
            [
                "source_contract",
                "source_conid",
                "contract_resolution_method",
                "contract_last_trade_date",
            ],
        ]
        .drop_duplicates()
        .sort_values(["source_contract", "source_conid"])
    )
    payload = []
    for row in entries.to_dict(orient="records"):
        if not str(row.get("source_contract", "")).strip():
            continue
        if not str(row.get("source_conid", "")).strip():
            continue
        payload.append({key: str(value) for key, value in row.items()})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) if payload else ""


def _raw_path(repo_root: Path, request: CmeNqRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "raw"
        / "ibkr"
        / "read_only_cme_futures"
        / f"symbol={request.symbol}"
        / f"timeframe={request.timeframe}"
        / f"export_id={request.export_id}"
        / "bars.parquet"
    )


def _canonical_path(repo_root: Path, request: CmeNqRefreshRequest) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "canonical"
        / "bars"
        / "market=cme_futures"
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
