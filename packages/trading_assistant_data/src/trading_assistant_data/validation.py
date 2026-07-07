"""Manifest validation and coverage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .calendars import CalendarDefinition, expected_bars
from .calendars.core import TIMEFRAME_MINUTES, expected_bar_opens
from .calendars.krx import KIS_INTRADAY_CALENDAR_ID
from .manifests import DataBundleManifest, DataBundleStatus, MarketDataManifest, MissingRange
from .repo import is_git_commit_sha


REQUIRED_MARKET_FIELDS = (
    "checksum",
    "session_calendar",
    "source_version",
    "adjustment_policy",
    "fee_model_version",
    "slippage_model_version",
)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    status: str
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def market_manifest_errors(
    manifest: MarketDataManifest,
    *,
    min_coverage_ratio: float = 0.95,
) -> list[str]:
    errors = [
        f"{field} missing"
        for field in REQUIRED_MARKET_FIELDS
        if not str(getattr(manifest, field, "") or "").strip()
    ]
    if manifest.expected_bars <= 0:
        errors.append("expected_bars missing")
    if manifest.actual_bars <= 0:
        errors.append("actual_bars missing")
    if manifest.coverage_ratio < min_coverage_ratio:
        errors.append("coverage_ratio below threshold")
    if manifest.missing_ranges:
        errors.append("missing_ranges present")
    if manifest.source_version and not is_git_commit_sha(manifest.source_version):
        errors.append("source_version is not a git commit SHA")
    if manifest.usable_for_authoritative_validation:
        errors.extend(_authority_lineage_errors(manifest))
    errors.extend(manifest.blocking_reasons)
    return errors


def validate_market_manifest(
    manifest: MarketDataManifest,
    *,
    min_coverage_ratio: float = 0.95,
) -> ValidationReport:
    errors = market_manifest_errors(manifest, min_coverage_ratio=min_coverage_ratio)
    if manifest.usable_for_authoritative_validation and errors:
        errors.append("manifest marked authoritative despite contract errors")
    return ValidationReport(
        valid=not errors,
        status=manifest.usability.value,
        errors=errors,
        warnings=[],
    )


def bundle_errors(bundle: DataBundleManifest) -> list[str]:
    errors = list(bundle.authoritative_contract_errors())
    if bundle.status == DataBundleStatus.AUTHORITATIVE and errors:
        errors.append("bundle marked authoritative despite contract errors")
    if bundle.status != DataBundleStatus.AUTHORITATIVE and not bundle.diagnostics_only_reason:
        errors.append("diagnostics_only_reason missing")
    return errors


def validate_bundle(bundle: DataBundleManifest) -> ValidationReport:
    errors = bundle_errors(bundle)
    return ValidationReport(
        valid=not errors,
        status=bundle.status.value,
        errors=errors,
        warnings=[],
    )


def detect_missing_ranges(
    timestamps: Iterable[datetime | pd.Timestamp],
    timeframe: str,
    *,
    calendar: CalendarDefinition | None = None,
    exchange_timestamps: Iterable[datetime | pd.Timestamp | str] | None = None,
    tolerance: int = 1,
) -> list[MissingRange]:
    ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True)).sort_values().unique()
    if len(ts) < 2:
        return []
    if calendar is not None and _is_daily_timeframe(timeframe):
        return _daily_missing_ranges(ts, calendar, exchange_timestamps)
    if calendar is not None:
        return _calendar_missing_ranges(ts, timeframe, calendar)
    minutes = TIMEFRAME_MINUTES[timeframe.lower()]
    expected_delta = pd.Timedelta(minutes=minutes * tolerance)
    diffs = ts.to_series().diff().dropna()
    missing: list[MissingRange] = []
    for current, _delta in diffs[diffs > expected_delta].items():
        previous = ts[ts.get_loc(current) - 1]
        missing.append(
            MissingRange(
                start_ts=(previous + pd.Timedelta(minutes=minutes)).to_pydatetime(),
                end_ts=(current - pd.Timedelta(minutes=minutes)).to_pydatetime(),
                reason=f"gap>{minutes}min",
            )
        )
    return missing


def coverage_counts(
    timestamps: Iterable[datetime | pd.Timestamp],
    timeframe: str,
    calendar: CalendarDefinition,
    *,
    exchange_timestamps: Iterable[datetime | pd.Timestamp | str] | None = None,
) -> tuple[int, int, list[MissingRange]]:
    idx = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True)).sort_values().unique()
    if idx.empty:
        return 0, 0, []
    if _is_daily_timeframe(timeframe):
        actual_dates = _actual_daily_dates(exchange_timestamps if exchange_timestamps is not None else idx)
        expected = len(_expected_trading_dates_between(calendar, actual_dates))
        actual = len(actual_dates)
        missing = _daily_missing_ranges(idx, calendar, exchange_timestamps)
        return expected, actual, missing
    if calendar.calendar_id == KIS_INTRADAY_CALENDAR_ID:
        actual = len(idx)
        missing = _kis_intraday_missing_trading_dates(idx, calendar)
        return actual, actual, missing
    actual = len(idx)
    expected = expected_bars(calendar, timeframe, idx[0].to_pydatetime(), idx[-1].to_pydatetime())
    missing = detect_missing_ranges(idx, timeframe, calendar=calendar)
    return expected, actual, missing


def report_path(repo_root: Path, command_name: str) -> Path:
    return Path(repo_root) / "data" / "validation_reports" / f"{command_name}.json"


def _authority_lineage_errors(manifest: MarketDataManifest) -> list[str]:
    required = _required_lineage_fields(manifest)
    lineage = manifest.lineage or {}
    return [f"lineage.{field} missing" for field in required if not str(lineage.get(field, "")).strip()]


def _required_lineage_fields(manifest: MarketDataManifest) -> tuple[str, ...]:
    if manifest.source == "lrs" and manifest.market.startswith("krx"):
        fields = ("source_endpoint", "export_id", "pulled_at_utc", "config_hash", "corporate_action_policy")
        if "flow" in manifest.timeframe:
            return (*fields, "flow_schema_version")
        return fields
    if manifest.source == "kis" and manifest.market == "krx_equity":
        return ("source_endpoint", "export_id", "pulled_at_utc", "config_hash", "session_policy")
    if manifest.source == "ibkr" and manifest.market == "us_equity":
        return (
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "corporate_action_policy",
            "raw_adjustment_policy",
            "session_policy",
            "source_conid_coverage",
            "contract_resolution_cache",
            "source_request_params_hash",
            "returned_row_count",
        )
    if manifest.market == "cme_futures":
        fields = (
            "source_endpoint",
            "export_id",
            "pulled_at_utc",
            "config_hash",
            "session_policy",
            "market_rule_authority_checksum",
            "roll_policy",
            "contract_chain_checksum",
            "continuous_construction_checksum",
            "source_contract_coverage",
        )
        if manifest.timeframe.endswith("_bid_ask"):
            return (*fields, "quote_schema_version")
        return fields
    return ()


def _calendar_missing_ranges(
    timestamps: pd.DatetimeIndex,
    timeframe: str,
    calendar: CalendarDefinition,
) -> list[MissingRange]:
    if calendar.calendar_id == KIS_INTRADAY_CALENDAR_ID:
        return _kis_intraday_missing_trading_dates(timestamps, calendar)
    expected = expected_bar_opens(
            calendar,
            timeframe,
            timestamps[0].to_pydatetime(),
            timestamps[-1].to_pydatetime(),
    )
    if expected.empty:
        return []
    actual = pd.DatetimeIndex(timestamps).sort_values().unique()
    missing = expected.difference(actual)
    return _ranges_from_missing_timestamps(missing, timeframe)


def _kis_intraday_missing_trading_dates(
    timestamps: pd.DatetimeIndex,
    calendar: CalendarDefinition,
) -> list[MissingRange]:
    local = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_convert(calendar.timezone)
    actual_dates = {item.date() for item in local}
    expected_dates = _expected_trading_dates_between(calendar, actual_dates)
    missing_dates = [value for value in expected_dates if value not in actual_dates]
    ranges = []
    for value in missing_dates:
        start = pd.Timestamp(f"{value.isoformat()} {calendar.session_open}", tz=calendar.timezone)
        end = pd.Timestamp(f"{value.isoformat()} {calendar.session_close}", tz=calendar.timezone)
        ranges.append(
            MissingRange(
                start_ts=start.tz_convert("UTC").to_pydatetime(),
                end_ts=end.tz_convert("UTC").to_pydatetime(),
                reason="missing KIS intraday trading date",
            )
        )
    return ranges


def _daily_missing_ranges(
    timestamps: pd.DatetimeIndex,
    calendar: CalendarDefinition,
    exchange_timestamps: Iterable[datetime | pd.Timestamp | str] | None,
) -> list[MissingRange]:
    actual_dates = _actual_daily_dates(exchange_timestamps if exchange_timestamps is not None else timestamps)
    expected_dates = _expected_trading_dates_between(calendar, actual_dates)
    missing_dates = [value for value in expected_dates if value not in actual_dates]
    return [
        MissingRange(
            start_ts=pd.Timestamp(value).to_pydatetime(),
            end_ts=pd.Timestamp(value).to_pydatetime(),
            reason="missing trading date",
        )
        for value in missing_dates
    ]


def _actual_daily_dates(values: Iterable[datetime | pd.Timestamp | str]) -> set:
    dates = set()
    for value in values:
        if pd.isna(value):
            continue
        dates.add(pd.Timestamp(value).date())
    return dates


def _expected_trading_dates_between(calendar: CalendarDefinition, actual_dates: set) -> list:
    if not actual_dates:
        return []
    current = min(actual_dates)
    final = max(actual_dates)
    dates = []
    while current <= final:
        if calendar.is_trading_day(current):
            dates.append(current)
        current += pd.Timedelta(days=1).to_pytimedelta()
    return dates


def _ranges_from_missing_timestamps(missing: pd.DatetimeIndex, timeframe: str) -> list[MissingRange]:
    if missing.empty:
        return []
    minutes = TIMEFRAME_MINUTES[timeframe.lower()]
    expected_delta = pd.Timedelta(minutes=minutes)
    ranges: list[MissingRange] = []
    start = missing[0]
    previous = missing[0]
    for current in missing[1:]:
        if current - previous > expected_delta:
            ranges.append(
                MissingRange(
                    start_ts=start.to_pydatetime(),
                    end_ts=previous.to_pydatetime(),
                    reason="missing expected bar",
                )
            )
            start = current
        previous = current
    ranges.append(
        MissingRange(
            start_ts=start.to_pydatetime(),
            end_ts=previous.to_pydatetime(),
            reason="missing expected bar",
        )
    )
    return ranges


def _is_daily_timeframe(timeframe: str) -> bool:
    value = timeframe.lower()
    return value in {"1d", "daily"} or value.startswith("1d_") or value.endswith("_panama")
