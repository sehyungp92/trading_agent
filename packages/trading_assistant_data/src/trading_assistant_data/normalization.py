"""Reference snapshot to canonical parquet normalization."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from .calendars import CalendarDefinition
from .checksums import parquet_content_checksum, stable_row_hashes
from .io import read_json, write_json
from .manifests import MarketDataManifest, load_market_manifest
from .repo import git_commit_sha, is_git_commit_sha
from .slices import (
    CanonicalSlice,
    DataSliceProduct,
    SliceRequest,
    SliceWrite,
    timestamps_sorted_unique_utc,
)
from .slices.writer import update_slice_index
from .slices.market_rules import (
    cme_calendar,
    crypto_calendar,
    krx_calendar,
    krx_kis_intraday_calendar,
    us_equities_calendar,
)
from .sources.hyperliquid.store import canonicalize_candles, canonicalize_funding


DEFAULT_FEE_MODEL = "fees_v1"
DEFAULT_SLIPPAGE_MODEL = "slippage_v1"
CRYPTO_ADJUSTMENT_POLICY = "crypto_raw_perp_policy_v1"
CME_ADJUSTMENT_POLICY = "cme_futures_panama_v1"
KRX_ADJUSTMENT_POLICY = "krx_split_adjusted_policy_v1"
US_EQUITY_ADJUSTMENT_POLICY = "us_equity_raw_adjustment_policy_v1"
KRX_FLOW_ADJUSTMENT_POLICY = "krx_flow_panel_policy_v1"
REGIME_SEED_ADJUSTMENT_POLICY = "global_macro_regime_seed_policy_v1"


def normalize_all(repo_root: Path, *, snapshot: str = "2026-05-30", dry_run: bool = False) -> dict:
    reports = []
    reports.append(normalize_crypto(repo_root, snapshot=snapshot, dry_run=dry_run))
    reports.append(normalize_reference_trading_bars(repo_root, snapshot=snapshot, dry_run=dry_run))
    reports.append(normalize_krx_intraday(repo_root, snapshot=snapshot, dry_run=dry_run))
    reports.append(normalize_krx_daily(repo_root, snapshot=snapshot, dry_run=dry_run))
    reports.append(normalize_us_equity_stock_raw(repo_root, snapshot=snapshot, dry_run=dry_run))
    reports.append(normalize_trading_seed_data(repo_root, snapshot=snapshot, dry_run=dry_run))
    return {
        "snapshot": snapshot,
        "dry_run": dry_run,
        "reports": reports,
        "slice_manifest_count": sum(report.get("slice_manifest_count", 0) for report in reports),
    }


def normalize_crypto(repo_root: Path, *, snapshot: str, dry_run: bool = False) -> dict:
    imported = Path(repo_root) / "data" / "imported" / f"reference_snapshot_{snapshot}" / "crypto_trader"
    candles_root = imported / "data" / "candles"
    funding_root = imported / "data" / "funding"
    writes: list[SliceWrite] = []
    errors: list[str] = []
    calendar = crypto_calendar()
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"

    for source_path in sorted(candles_root.glob("*/*.parquet")):
        symbol = source_path.parent.name.upper()
        interval = source_path.stem
        try:
            source = pd.read_parquet(source_path)
            canonical = canonicalize_candles(
                source,
                symbol=symbol,
                interval=interval,
                source_file=_rel(source_path, repo_root),
            )
            writes.extend(
                _write_bar_partitions_and_manifests(
                    repo_root=repo_root,
                    canonical=canonical,
                    market="crypto_perp",
                    source="hyperliquid",
                    symbol=symbol,
                    timeframe=interval,
                    kind="trades",
                    calendar=calendar,
                    adjustment_policy=CRYPTO_ADJUSTMENT_POLICY,
                    authoritative_allowed=True,
                    source_version=source_version,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")

    for source_path in sorted(funding_root.glob("*.parquet")):
        symbol = source_path.stem.upper()
        try:
            source = pd.read_parquet(source_path)
            canonical = canonicalize_funding(
                source,
                symbol=symbol,
                source_file=_rel(source_path, repo_root),
            )
            writes.extend(
                _write_funding_partitions_and_manifests(
                    repo_root=repo_root,
                    canonical=canonical,
                    symbol=symbol,
                    calendar=calendar,
                    source_version=source_version,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")

    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "crypto_hyperliquid",
        "dry_run": dry_run,
        "slice_manifest_count": len(writes),
        "errors": errors,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def normalize_reference_trading_bars(repo_root: Path, *, snapshot: str, dry_run: bool = False) -> dict:
    raw_root = Path(repo_root) / "data" / "imported" / f"reference_snapshot_{snapshot}" / "trading" / "data" / "raw"
    writes: list[SliceWrite] = []
    errors: list[str] = []
    skipped: list[str] = []
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"
    source_paths = sorted(raw_root.glob("*.parquet"))
    available_stems = {path.stem for path in source_paths}
    calendar = cme_calendar()
    for source_path in source_paths:
        if _is_duplicate_daily_alias(source_path, available_stems):
            skipped.append(f"{source_path}: duplicate daily alias; using matching 1d file")
            continue
        try:
            symbol, timeframe, kind = _parse_trading_raw_name(source_path.stem)
            source = pd.read_parquet(source_path)
            canonical = canonicalize_ohlcv_frame(
                source,
                symbol=symbol,
                market="cme_futures",
                source="ibkr",
                timeframe=timeframe,
                kind=kind,
                source_file=_rel(source_path, repo_root),
            )
            writes.extend(
                _write_bar_partitions_and_manifests(
                    repo_root=repo_root,
                    canonical=canonical,
                    market="cme_futures",
                    source="ibkr",
                    symbol=symbol,
                    timeframe=timeframe,
                    kind=kind,
                    calendar=calendar,
                    adjustment_policy=CME_ADJUSTMENT_POLICY,
                    authoritative_allowed=False,
                    source_version=source_version,
                    dry_run=dry_run,
                    extra_blocking_reasons=["CME session/roll authority requires final exchange calendar and roll checksum"],
                )
            )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")
    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "trading_ibkr_raw",
        "dry_run": dry_run,
        "slice_manifest_count": len(writes),
        "errors": errors,
        "skipped": skipped,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def normalize_krx_intraday(
    repo_root: Path,
    *,
    snapshot: str,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> dict:
    source_root = (
        Path(repo_root)
        / "data"
        / "imported"
        / f"reference_snapshot_{snapshot}"
        / "k_stock_trader"
        / "data"
        / "kis_intraday_parquet"
    )
    writes: list[SliceWrite] = []
    errors: list[str] = []
    selected_symbols = {symbol.zfill(6) for symbol in symbols or []}
    selected_timeframes = {timeframe.strip() for timeframe in timeframes or [] if timeframe.strip()}
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(source_root.rglob("*.parquet")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0].zfill(6)
        timeframe = parts[1]
        if selected_symbols and symbol not in selected_symbols:
            continue
        if selected_timeframes and timeframe not in selected_timeframes:
            continue
        grouped.setdefault((symbol, timeframe), []).append(path)

    holiday_path = Path(repo_root) / "data" / "calendars" / "krx_holidays.yaml"
    calendar = krx_kis_intraday_calendar(holiday_path if holiday_path.exists() else None)
    for (symbol, timeframe), paths in grouped.items():
        try:
            frames = [pd.read_parquet(path) for path in paths]
            canonical = normalize_krx_intraday_frames(
                frames,
                symbol=symbol,
                timeframe=timeframe,
                source_files=[_rel(path, repo_root) for path in paths],
            )
            authoritative = _is_k_stock_kis_intraday_authority(
                repo_root,
                symbol=symbol,
                timeframe=timeframe,
            )
            writes.extend(
                _write_bar_partitions_and_manifests(
                    repo_root=repo_root,
                    canonical=canonical,
                    market="krx_equity",
                    source="kis",
                    symbol=symbol,
                    timeframe=timeframe,
                    kind="trades",
                    calendar=calendar,
                    adjustment_policy=KRX_ADJUSTMENT_POLICY,
                    authoritative_allowed=authoritative,
                    source_version=source_version,
                    dry_run=dry_run,
                    extra_blocking_reasons=None
                    if authoritative
                    else ["KIS intraday slice is outside the approved k_stock 5m universe"],
                )
            )
        except Exception as exc:
            errors.append(f"{symbol} {timeframe}: {exc}")
    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "krx_kis_intraday",
        "dry_run": dry_run,
        "symbol_filter_count": len(selected_symbols),
        "timeframe_filter": sorted(selected_timeframes),
        "slice_manifest_count": len(writes),
        "errors": errors,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def normalize_krx_daily(repo_root: Path, *, snapshot: str, dry_run: bool = False) -> dict:
    source_root = (
        Path(repo_root)
        / "data"
        / "imported"
        / f"reference_snapshot_{snapshot}"
        / "k_stock_trader"
        / "data"
        / "krx_daily_parquet"
        / "tables"
    )
    writes: list[SliceWrite] = []
    errors: list[str] = []
    reference_paths: list[str] = []
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"
    holiday_path = Path(repo_root) / "data" / "calendars" / "krx_holidays.yaml"
    calendar = krx_calendar(holiday_path if holiday_path.exists() else None)
    for file_name, market, symbol_column in (
        ("daily_ohlcv.parquet", "krx_equity", "ticker"),
        ("index_ohlcv.parquet", "krx_index", "index_code"),
    ):
        source_path = source_root / file_name
        if not source_path.exists():
            continue
        try:
            source = pd.read_parquet(source_path)
            for symbol, group in source.groupby(symbol_column, sort=True):
                canonical = canonicalize_krx_daily_ohlcv(
                    group,
                    symbol=_normalize_krx_symbol(symbol),
                    market=market,
                    kind="trades",
                    source_file=_rel(source_path, repo_root),
                )
                writes.extend(
                    _write_compact_bar_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        market=market,
                        source="lrs",
                        symbol=_normalize_krx_symbol(symbol),
                        timeframe="1d",
                        kind="trades",
                        calendar=calendar,
                        adjustment_policy=KRX_ADJUSTMENT_POLICY,
                        authoritative_allowed=True,
                        source_version=source_version,
                        dry_run=dry_run,
                        extra_blocking_reasons=None,
                    )
                )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")

    for file_name, kind, value_columns in (
        ("daily_flow.parquet", "daily_flow", ("foreign_net", "inst_net")),
        ("daily_foreign_flow.parquet", "daily_foreign_flow", ("foreign_net",)),
        ("daily_institutional_flow.parquet", "daily_institutional_flow", ("institutional_net",)),
    ):
        source_path = source_root / file_name
        if not source_path.exists():
            continue
        try:
            source = pd.read_parquet(source_path)
            for symbol, group in source.groupby("ticker", sort=True):
                canonical = canonicalize_daily_panel_frame(
                    group,
                    symbol=_normalize_krx_symbol(symbol),
                    market="krx_equity",
                    source="lrs",
                    timeframe=f"1d_{kind}",
                    kind=kind,
                    value_columns=list(value_columns),
                    exchange_timezone="Asia/Seoul",
                    source_file=_rel(source_path, repo_root),
                )
                writes.extend(
                    _write_compact_panel_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        market="krx_equity",
                        source="lrs",
                        symbol=_normalize_krx_symbol(symbol),
                        timeframe=f"1d_{kind}",
                        kind=kind,
                        calendar=calendar,
                        adjustment_policy=KRX_FLOW_ADJUSTMENT_POLICY,
                        authoritative_allowed=True,
                        source_version=source_version,
                        dry_run=dry_run,
                        extra_blocking_reasons=None,
                    )
                )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")

    sector_path = source_root / "sector_map.parquet"
    if sector_path.exists():
        try:
            sector_frame = pd.read_parquet(sector_path)
            reference_paths.extend(
                _write_static_reference_frame(
                    repo_root=repo_root,
                    frame=sector_frame,
                    market="krx_equity",
                    source="lrs",
                    kind="sector_map",
                    dry_run=dry_run,
                )
            )
            writes.append(
                _write_static_reference_slice_and_manifest(
                    repo_root=repo_root,
                    frame=sector_frame,
                    source_path=sector_path,
                    market="krx_equity",
                    source="lrs",
                    symbol="ALL",
                    timeframe="static_or_daily",
                    kind="sector_map",
                    calendar=calendar,
                    adjustment_policy="krx_sector_map_policy_v1",
                    source_version=source_version,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            errors.append(f"{sector_path}: {exc}")

    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "krx_lrs_daily_and_flow",
        "dry_run": dry_run,
        "slice_manifest_count": len(writes),
        "errors": errors,
        "reference_paths": reference_paths,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def normalize_us_equity_stock_raw(repo_root: Path, *, snapshot: str, dry_run: bool = False) -> dict:
    raw_root = (
        Path(repo_root)
        / "data"
        / "imported"
        / f"reference_snapshot_{snapshot}"
        / "trading"
        / "backtests"
        / "stock"
        / "data"
        / "raw"
    )
    writes: list[SliceWrite] = []
    errors: list[str] = []
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"
    declared_requests = _trading_stock_source_request_index(repo_root, snapshot=snapshot)
    declared_requirements = _trading_stock_requirement_keys(repo_root)

    for source_path in sorted(raw_root.glob("*.parquet")):
        try:
            symbol, timeframe = _parse_raw_symbol_timeframe(source_path.stem)
            request = declared_requests.get((symbol.upper(), timeframe))
            exact_requirement = (symbol.upper(), timeframe) in declared_requirements
            if request and exact_requirement:
                existing = _existing_archived_trading_stock_write(
                    repo_root=repo_root,
                    symbol=symbol,
                    timeframe=timeframe,
                    source_request=request,
                )
                if existing is not None:
                    writes.append(existing)
                    continue
            source = pd.read_parquet(source_path)
            canonical = canonicalize_ohlcv_frame(
                source,
                symbol=symbol,
                market="us_equity",
                source="ibkr",
                timeframe=timeframe,
                kind="trades",
                source_file=_rel(source_path, repo_root),
            )
            if request and exact_requirement:
                writes.extend(
                    _write_archived_us_equity_stock_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        source_path=source_path,
                        source_request=request,
                        symbol=symbol,
                        timeframe=timeframe,
                        source_version=source_version,
                        dry_run=dry_run,
                    )
                )
            else:
                writes.extend(
                    _write_compact_bar_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        market="us_equity",
                        source="ibkr",
                        symbol=symbol,
                        timeframe=timeframe,
                        kind="trades",
                        calendar=us_equities_calendar(),
                        adjustment_policy=US_EQUITY_ADJUSTMENT_POLICY,
                        authoritative_allowed=False,
                        source_version=source_version,
                        dry_run=dry_run,
                        extra_blocking_reasons=[
                            "US equity archive is outside the explicit trading_stock requirement allowlist"
                        ],
                    )
                )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")
    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "trading_stock_ibkr_raw",
        "dry_run": dry_run,
        "declared_request_count": len(declared_requests),
        "declared_requirement_count": len(declared_requirements),
        "slice_manifest_count": len(writes),
        "errors": errors,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def _write_archived_us_equity_stock_slice_and_manifest(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    source_path: Path,
    source_request: dict,
    symbol: str,
    timeframe: str,
    source_version: str,
    dry_run: bool,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    canonical_path = _compact_canonical_path(
        repo_root=repo_root,
        root="bars",
        market="us_equity",
        source="ibkr",
        kind="trades",
        symbol=symbol,
        timeframe=timeframe,
    )
    output = canonical.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    if not dry_run:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(canonical_path, engine="pyarrow", index=False)

    download = source_request.get("download_request", {})
    expectations = source_request.get("canonical_expectations", {})
    use_rth = bool(download.get("use_rth", timeframe in {"1d", "daily"}))
    calendar = us_equities_calendar()
    if timeframe in {"1d", "daily"} or use_rth:
        coverage = _slice_coverage_report(
            repo_root=repo_root,
            frame=output,
            source="ibkr",
            market="us_equity",
            symbol=symbol.upper(),
            timeframe=timeframe,
            calendar=calendar,
        )
        expected, actual, missing = (
            coverage.expected,
            coverage.actual,
            coverage.missing_ranges,
        )
    else:
        expected = len(output)
        actual = len(output)
        missing = []
    calendar_expected = expected

    checksum = "" if dry_run else parquet_content_checksum(canonical_path)
    raw_checksum = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else ""
    legacy_row_count = int(source_request.get("legacy_row_count") or -1)
    legacy_sha256 = str(source_request.get("legacy_sha256") or "")
    legacy_row_count_matches = legacy_row_count == len(output)
    legacy_sha256_matches = bool(raw_checksum and legacy_sha256 and raw_checksum == legacy_sha256)
    archived_cache_gap_ranges = list(missing)
    if missing and legacy_row_count_matches and legacy_sha256_matches:
        expected = actual
        missing = []

    blocking_reasons: list[str] = []
    if missing:
        blocking_reasons.append("missing ranges present")
    if not legacy_row_count_matches:
        blocking_reasons.append("legacy row count does not match declared source request")
    if not legacy_sha256_matches:
        blocking_reasons.append("legacy raw checksum does not match declared source request")
    if not is_git_commit_sha(source_version):
        blocking_reasons.append("source_version is not a data repo commit SHA")
    if not _timestamps_sorted_unique_utc(output):
        blocking_reasons.append("timestamps are not sorted unique UTC values")
    if timeframe not in {"1d", "daily"} and _illegal_us_equity_intraday_timestamp_count(output, use_rth) > 0:
        blocking_reasons.append("timestamps outside declared US equity session window")

    if not checksum:
        blocking_reasons.append("checksum missing")
    if not raw_checksum:
        blocking_reasons.append("raw checksum missing")

    lineage = _archived_us_equity_stock_lineage(
        repo_root=repo_root,
        source_path=source_path,
        source_request=source_request,
        frame=output,
        canonical_checksum=checksum,
        raw_checksum=raw_checksum,
        archived_cache_gap_ranges=archived_cache_gap_ranges,
        calendar_expected_bars=calendar_expected,
        legacy_row_count_matches=legacy_row_count_matches,
        legacy_sha256_matches=legacy_sha256_matches,
    )
    manifest = MarketDataManifest(
        source="ibkr",
        market="us_equity",
        symbol=symbol.upper(),
        timeframe=timeframe,
        start_ts=pd.Timestamp(output["timestamp_utc"].min()).to_pydatetime(),
        end_ts=pd.Timestamp(output["timestamp_utc"].max()).to_pydatetime(),
        expected_bars=expected,
        actual_bars=actual,
        coverage_ratio=(actual / expected if expected else 0.0),
        missing_ranges=missing,
        session_calendar=calendar.calendar_id,
        timezone=calendar.timezone,
        checksum=checksum,
        source_version=source_version,
        adjustment_policy=expectations.get(
            "canonical_adjustment_policy",
            US_EQUITY_ADJUSTMENT_POLICY,
        ),
        fee_model_version=DEFAULT_FEE_MODEL,
        slippage_model_version=DEFAULT_SLIPPAGE_MODEL,
        lineage=lineage,
        usable_for_authoritative_validation=not blocking_reasons and expected > 0 and actual > 0,
        blocking_reasons=blocking_reasons,
    )
    return [_write_slice_product(repo_root, manifest, [canonical_path], dry_run=dry_run)]


def _existing_archived_trading_stock_write(
    *,
    repo_root: Path,
    symbol: str,
    timeframe: str,
    source_request: dict,
) -> SliceWrite | None:
    existing = _existing_compact_write(
        repo_root=repo_root,
        root="bars",
        market="us_equity",
        source="ibkr",
        kind="trades",
        symbol=symbol,
        timeframe=timeframe,
    )
    if existing is None:
        return None
    manifest = existing.manifest
    lineage = manifest.lineage or {}
    if not manifest.usable_for_authoritative_validation:
        return None
    if lineage.get("strategy_data_family") != "trading_stock":
        return None
    if lineage.get("authority_status") != "archived_ibkr_stock_updater_parquet_exact_declared_request":
        return None
    if lineage.get("source_request_id") != str(source_request.get("request_id", "")):
        return None
    if not existing.canonical_paths or not existing.canonical_paths[0].exists():
        return None
    return existing


def _archived_us_equity_stock_lineage(
    *,
    repo_root: Path,
    source_path: Path,
    source_request: dict,
    frame: pd.DataFrame,
    canonical_checksum: str,
    raw_checksum: str,
    archived_cache_gap_ranges: list,
    calendar_expected_bars: int,
    legacy_row_count_matches: bool,
    legacy_sha256_matches: bool,
) -> dict[str, str]:
    rel_source = _rel(source_path, repo_root)
    download = source_request.get("download_request", {})
    expectations = source_request.get("canonical_expectations", {})
    identity_payload = {
        "sec_type": download.get("sec_type", "STK"),
        "symbol": download.get("symbol", ""),
        "exchange": download.get("exchange", "SMART"),
        "primary_exchange": download.get("primary_exchange", ""),
        "currency": download.get("currency", "USD"),
        "contract_resolution_policy": download.get("contract_resolution_policy", ""),
        "source_identity_authority": download.get("source_identity_authority", ""),
    }
    source_stats = _source_file_stats(repo_root, [rel_source])
    lineage_payload = {
        "source_request": source_request,
        "source_stats": source_stats,
        "row_count": len(frame),
        "raw_checksum": raw_checksum,
        "canonical_checksum": canonical_checksum,
    }
    source_request_params_hash = _stable_payload_hash(download)
    config_hash = _stable_payload_hash(lineage_payload)
    export_id = f"archived-ibkr-stock-parquet-v1:{config_hash[:16]}"
    lineage = {
        "authority_status": "archived_ibkr_stock_updater_parquet_exact_declared_request",
        "source_endpoint": str(source_request.get("source_endpoint", "")),
        "source_files": rel_source,
        "source_file_count": "1",
        "export_id": export_id,
        "pulled_at_utc": _latest_source_mtime_utc(repo_root, [rel_source]),
        "config_hash": config_hash,
        "corporate_action_policy": expectations.get(
            "corporate_action_policy",
            "split_dividend_policy_declared_per_bundle_v1",
        ),
        "raw_adjustment_policy": expectations.get(
            "raw_adjustment_policy",
            "ibkr_trades_unadjusted_raw_v1",
        ),
        "session_policy": expectations.get("session_policy", ""),
        "session_calendar": "us_equities_xnys_xnas_v1",
        "strategy_data_family": "trading_stock",
        "source_request_id": str(source_request.get("request_id", "")),
        "use_rth": str(bool(download.get("use_rth", False))).lower(),
        "primary_exchange": str(download.get("primary_exchange", "") or "").upper(),
        "source_conid_coverage": "archived_request_identity_only_no_row_conids",
        "source_contract_identity_level": "request_level_without_row_conid_columns",
        "contract_resolution_cache": _stable_payload_hash(identity_payload),
        "source_request_params_json": json.dumps(download, sort_keys=True, separators=(",", ":")),
        "source_request_params_hash": source_request_params_hash,
        "returned_row_count": str(len(frame)),
        "credential_contract_id": "ibkr_read_only_market_data_credentials_v1",
        "adapter_id": "archived_trading_stock_ibkr_updater_import_v1",
        "authority_contract_id": "archived_ibkr_us_equity_stock_import_contract_v1",
        "pacing_policy": "archived_import_no_live_pacing_v1",
        "raw_write_checksum": raw_checksum,
        "canonical_write_checksum": canonical_checksum,
        "idempotency_key": _stable_payload_hash(
            {
                "source": "ibkr",
                "market": "us_equity",
                "symbol": download.get("symbol", ""),
                "timeframe": download.get("timeframe", ""),
                "start": download.get("start", ""),
                "end": download.get("end", ""),
                "export_id": export_id,
            }
        ),
        "read_only": "true",
        "archive_import_policy": "source_owned_trading_stock_ibkr_updater_parquet_v1",
        "legacy_row_count_matched": str(legacy_row_count_matches).lower(),
        "legacy_sha256_matched": str(legacy_sha256_matches).lower(),
    }
    if archived_cache_gap_ranges:
        lineage.update(
            {
                "legacy_calendar_gap_policy": "preserve_exact_archived_stock_updater_cache_v1",
                "archived_cache_missing_range_count": str(len(archived_cache_gap_ranges)),
                "archived_cache_calendar_expected_bars": str(calendar_expected_bars),
                "archived_cache_actual_bars": str(len(frame)),
                "archived_cache_missing_ranges_json": json.dumps(
                    _missing_ranges_payload(archived_cache_gap_ranges),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return lineage


def _missing_ranges_payload(missing_ranges: list) -> list[dict[str, str]]:
    payload = []
    for item in missing_ranges:
        payload.append(
            {
                "start_ts": pd.Timestamp(getattr(item, "start_ts", "")).isoformat(),
                "end_ts": pd.Timestamp(getattr(item, "end_ts", "")).isoformat(),
                "reason": str(getattr(item, "reason", "")),
            }
        )
    return payload


def _trading_stock_source_request_index(repo_root: Path, *, snapshot: str) -> dict[tuple[str, str], dict]:
    path = (
        Path(repo_root)
        / "data"
        / "source_requests"
        / f"reference_snapshot_{snapshot}"
        / "requests"
        / "ibkr_us_equity_historical_bars.json"
    )
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    result: dict[tuple[str, str], dict] = {}
    for item in payload.get("requests", []):
        if item.get("legacy_family") != "trading_stock":
            continue
        if item.get("source_kind") != "ibkr_us_equity_historical_bars":
            continue
        key = (str(item.get("symbol", "")).upper(), str(item.get("timeframe", "")))
        result[key] = item
    return result


def _trading_stock_requirement_keys(repo_root: Path) -> set[tuple[str, str]]:
    path = Path(repo_root) / "data" / "requirements" / "strategies" / "trading_stock" / "portfolio.json"
    if not path.exists():
        return set()
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return set()
    return {
        (str(item.get("symbol", "")).upper(), str(item.get("timeframe", "")))
        for item in payload.get("requirements", [])
        if item.get("source") == "ibkr"
        and item.get("market") == "us_equity"
        and item.get("strategy_data_family") == "trading_stock"
    }


def _timestamps_sorted_unique_utc(frame: pd.DataFrame) -> bool:
    if "timestamp_utc" not in frame:
        return False
    return timestamps_sorted_unique_utc(frame["timestamp_utc"])


def _illegal_us_equity_intraday_timestamp_count(frame: pd.DataFrame, use_rth: bool) -> int:
    if frame.empty:
        return 0
    calendar = us_equities_calendar()
    local = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True)).tz_convert(calendar.timezone)
    from .calendars.us_equities import extended_session_close_for_date, extended_session_open, session_close_for_date

    local_series = pd.Series(local)
    local_dates = local_series.dt.date
    trading_day = local_dates.map(calendar.is_trading_day)
    minute_of_day = local_series.dt.hour * 60 + local_series.dt.minute
    if use_rth:
        open_minutes = _minutes_from_time(calendar.session_open)
        close_by_date = {
            value: session_close_for_date(value).hour * 60 + session_close_for_date(value).minute
            for value in set(local_dates)
        }
    else:
        open_value = extended_session_open()
        open_minutes = open_value.hour * 60 + open_value.minute
        close_by_date = {
            value: extended_session_close_for_date(value).hour * 60
            + extended_session_close_for_date(value).minute
            for value in set(local_dates)
        }
    close_minutes = local_dates.map(close_by_date)
    illegal = (~trading_day) | (minute_of_day < open_minutes) | (minute_of_day > close_minutes)
    return int(illegal.sum())


def _minutes_from_time(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def normalize_trading_seed_data(repo_root: Path, *, snapshot: str, dry_run: bool = False) -> dict:
    imported = Path(repo_root) / "data" / "imported" / f"reference_snapshot_{snapshot}" / "trading"
    writes: list[SliceWrite] = []
    errors: list[str] = []
    reference_paths: list[str] = []
    source_version = git_commit_sha(repo_root) or f"reference_snapshot_{snapshot}"
    cme = cme_calendar()
    us = us_equities_calendar()

    for family in ("swing", "momentum"):
        raw_root = imported / "backtests" / family / "data" / "raw"
        seed_source = f"ibkr_{family}_seed"
        for source_path in sorted(raw_root.glob("*.parquet")):
            try:
                symbol, timeframe = _parse_raw_symbol_timeframe(source_path.stem)
                market = _market_for_seed_symbol(symbol)
                calendar = cme if market == "cme_futures" else us
                canonical = canonicalize_ohlcv_frame(
                    pd.read_parquet(source_path),
                    symbol=symbol,
                    market=market,
                    source=seed_source,
                    timeframe=timeframe,
                    kind="trades",
                    source_file=_rel(source_path, repo_root),
                )
                writes.extend(
                    _write_compact_bar_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        market=market,
                        source=seed_source,
                        symbol=symbol,
                        timeframe=timeframe,
                        kind="trades",
                        calendar=calendar,
                        adjustment_policy=CME_ADJUSTMENT_POLICY
                        if market == "cme_futures"
                        else US_EQUITY_ADJUSTMENT_POLICY,
                        authoritative_allowed=False,
                        source_version=source_version,
                        dry_run=dry_run,
                        extra_blocking_reasons=[
                            f"{family} seed data authority requires production refresh lineage and calendar audit"
                        ],
                    )
                )
            except Exception as exc:
                errors.append(f"{source_path}: {exc}")
        for source_path in sorted(raw_root.glob("*.csv")):
            try:
                symbol, timeframe = _parse_raw_symbol_timeframe(source_path.stem)
                canonical = canonicalize_ohlcv_frame(
                    pd.read_csv(source_path),
                    symbol=symbol,
                    market="cme_futures",
                    source=seed_source,
                    timeframe=timeframe,
                    kind="panama_trades",
                    source_file=_rel(source_path, repo_root),
                )
                writes.extend(
                    _write_compact_bar_slice_and_manifest(
                        repo_root=repo_root,
                        canonical=canonical,
                        market="cme_futures",
                        source=seed_source,
                        symbol=symbol,
                        timeframe=timeframe,
                        kind="panama_trades",
                        calendar=cme,
                        adjustment_policy=CME_ADJUSTMENT_POLICY,
                        authoritative_allowed=False,
                        source_version=source_version,
                        dry_run=dry_run,
                        extra_blocking_reasons=[
                            f"{family} panama seed data requires roll checksum and production refresh lineage"
                        ],
                    )
                )
            except Exception as exc:
                errors.append(f"{source_path}: {exc}")

    regime_root = imported / "backtests" / "regime" / "data" / "raw"
    for source_path in sorted(regime_root.glob("*.parquet")):
        try:
            stem = source_path.stem
            canonical = canonicalize_daily_panel_frame(
                pd.read_parquet(source_path),
                symbol=stem,
                market="global_macro",
                source="seed",
                timeframe="1d",
                kind=stem,
                value_columns=None,
                source_file=_rel(source_path, repo_root),
            )
            writes.extend(
                _write_compact_panel_slice_and_manifest(
                    repo_root=repo_root,
                    canonical=canonical,
                    market="global_macro",
                    source="seed",
                    symbol=stem,
                    timeframe="1d",
                    kind=stem,
                    adjustment_policy=REGIME_SEED_ADJUSTMENT_POLICY,
                    source_version=source_version,
                    dry_run=dry_run,
                    extra_blocking_reasons=["regime seed authority requires source feed lineage"],
                )
            )
        except Exception as exc:
            errors.append(f"{source_path}: {exc}")

    seed_manifest = regime_root / "regime_seed_manifest.json"
    if seed_manifest.exists():
        target = (
            Path(repo_root)
            / "data"
            / "canonical"
            / "reference"
            / "market=global_macro"
            / "source=seed"
            / "kind=regime_seed_manifest"
            / "regime_seed_manifest.json"
        )
        if not dry_run:
            write_json(target, read_json(seed_manifest))
        reference_paths.append(_rel(target, repo_root))

    if not dry_run:
        update_slice_index(repo_root, writes)
    return {
        "name": "trading_swing_momentum_regime_seeds",
        "dry_run": dry_run,
        "slice_manifest_count": len(writes),
        "errors": errors,
        "reference_paths": reference_paths,
        "manifest_paths": [str(item.manifest_path) for item in writes],
    }


def canonicalize_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    market: str,
    source: str,
    timeframe: str,
    kind: str,
    source_file: str,
) -> pd.DataFrame:
    source_frame = frame.copy()
    timestamp_column = _source_column(source_frame, "timestamp_utc", "timestamp", "time", "Date", "date")
    if timestamp_column:
        timestamp = pd.to_datetime(source_frame[timestamp_column], utc=True)
    else:
        timestamp = pd.to_datetime(source_frame.index, utc=True)
    open_column = _source_column(source_frame, "open", "Open")
    high_column = _source_column(source_frame, "high", "High")
    low_column = _source_column(source_frame, "low", "Low")
    close_column = _source_column(source_frame, "close", "Close", "Adjusted_Close", "adjusted_close")
    volume_column = _source_column(source_frame, "volume", "Volume", "vol")
    if not all((open_column, high_column, low_column, close_column, volume_column)):
        raise ValueError("OHLCV frame requires open/high/low/close/volume columns")
    out = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "timestamp_exchange": timestamp.astype(str),
            "symbol": symbol.upper(),
            "market": market,
            "source": source,
            "timeframe": timeframe,
            "kind": kind,
            "open": source_frame[open_column].astype("float64"),
            "high": source_frame[high_column].astype("float64"),
            "low": source_frame[low_column].astype("float64"),
            "close": source_frame[close_column].astype("float64"),
            "volume": source_frame[volume_column].astype("float64"),
            "source_file": source_file,
        }
    )
    for optional in ("bar_count", "wap", "is_RTH", "is_rth"):
        if optional in source_frame.columns:
            target = "is_rth" if optional == "is_RTH" else optional
            out[target] = source_frame[optional]
    out["source_row_hash"] = stable_row_hashes(source_frame.reset_index(drop=False))
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def canonicalize_krx_daily_ohlcv(
    frame: pd.DataFrame,
    *,
    symbol: str,
    market: str,
    kind: str,
    source_file: str,
) -> pd.DataFrame:
    source_frame = frame.copy()
    timestamp_utc, timestamp_exchange = _localized_timestamps(source_frame["date"], "Asia/Seoul")
    out = pd.DataFrame(
        {
            "timestamp_utc": timestamp_utc,
            "timestamp_exchange": timestamp_exchange.astype(str),
            "symbol": symbol,
            "market": market,
            "source": "lrs",
            "timeframe": "1d",
            "kind": kind,
            "open": source_frame["open"].astype("float64"),
            "high": source_frame["high"].astype("float64"),
            "low": source_frame["low"].astype("float64"),
            "close": source_frame["close"].astype("float64"),
            "volume": source_frame["volume"].astype("float64"),
            "source_file": source_file,
        }
    )
    out["source_row_hash"] = stable_row_hashes(source_frame.reset_index(drop=False))
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def canonicalize_daily_panel_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    market: str,
    source: str,
    timeframe: str,
    kind: str,
    value_columns: list[str] | None,
    source_file: str,
    exchange_timezone: str = "UTC",
) -> pd.DataFrame:
    source_frame = frame.copy()
    if "date" in source_frame.columns:
        timestamp, timestamp_exchange = _localized_timestamps(source_frame["date"], exchange_timezone)
    elif "timestamp_utc" in source_frame.columns:
        timestamp = pd.to_datetime(source_frame["timestamp_utc"], utc=True)
        timestamp_exchange = timestamp
    elif "timestamp" in source_frame.columns:
        timestamp = pd.to_datetime(source_frame["timestamp"], utc=True)
        timestamp_exchange = timestamp
    else:
        timestamp = pd.to_datetime(source_frame.index, utc=True)
        timestamp_exchange = timestamp
    columns = value_columns or [
        str(column)
        for column in source_frame.columns
        if str(column) not in {"date", "timestamp", "timestamp_utc", "ticker", "symbol"}
    ]
    out = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "timestamp_exchange": timestamp_exchange.astype(str),
            "symbol": symbol.upper(),
            "market": market,
            "source": source,
            "timeframe": timeframe,
            "kind": kind,
            "source_file": source_file,
        }
    )
    for column in columns:
        out[str(column)] = source_frame[column]
    out["source_row_hash"] = stable_row_hashes(source_frame.reset_index(drop=False))
    return out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def normalize_krx_intraday_frames(
    frames: list[pd.DataFrame],
    *,
    symbol: str,
    timeframe: str,
    source_files: list[str] | None = None,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    source_files = source_files or [""] * len(frames)
    canonical_frames: list[pd.DataFrame] = []
    tz = ZoneInfo("Asia/Seoul")
    for frame, source_file in zip(frames, source_files, strict=False):
        if "timestamp" not in frame.columns:
            raise ValueError("KRX intraday frame requires timestamp column")
        timestamp = pd.to_datetime(frame["timestamp"])
        if timestamp.dt.tz is None:
            raise ValueError("KRX intraday timestamp must include Asia/Seoul timezone")
        exchange = timestamp.dt.tz_convert(tz)
        utc = exchange.dt.tz_convert("UTC")
        out = pd.DataFrame(
            {
                "timestamp_utc": utc,
                "timestamp_exchange": exchange.astype(str),
                "symbol": symbol.zfill(6),
                "market": "krx_equity",
                "source": "kis",
                "timeframe": timeframe,
                "kind": "trades",
                "open": frame["open"].astype("float64"),
                "high": frame["high"].astype("float64"),
                "low": frame["low"].astype("float64"),
                "close": frame["close"].astype("float64"),
                "volume": frame["volume"].astype("float64"),
                "source_file": source_file,
            }
        )
        out["source_row_hash"] = stable_row_hashes(frame.reset_index(drop=False))
        canonical_frames.append(out)
    merged = pd.concat(canonical_frames, ignore_index=True)
    return merged.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")


def _write_bar_partitions_and_manifests(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    market: str,
    source: str,
    symbol: str,
    timeframe: str,
    kind: str,
    calendar: CalendarDefinition | None,
    adjustment_policy: str,
    authoritative_allowed: bool,
    source_version: str,
    dry_run: bool,
    extra_blocking_reasons: list[str] | None = None,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    paths: list[Path] = []
    frame = canonical.assign(
        year=canonical["timestamp_utc"].dt.year.astype(str),
        month=canonical["timestamp_utc"].dt.month.map(lambda value: f"{int(value):02d}"),
    )
    writes: list[SliceWrite] = []
    for (year, month), group in frame.groupby(["year", "month"], sort=True):
        path = (
            Path(repo_root)
            / "data"
            / "canonical"
            / "bars"
            / f"market={market}"
            / f"source={source}"
            / f"kind={kind}"
            / f"symbol={symbol.upper()}"
            / f"timeframe={timeframe}"
            / f"year={year}"
            / f"month={month}"
            / "part.parquet"
        )
        paths = [path]
        output = group.drop(columns=["year", "month"])
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, engine="pyarrow", index=False)
        writes.append(
            _manifest_for_partition(
                repo_root=repo_root,
                canonical_paths=paths,
                frame=output,
                source=source,
                market=market,
                symbol=symbol.upper(),
                timeframe=timeframe,
                calendar=calendar,
                adjustment_policy=adjustment_policy,
                authoritative_allowed=authoritative_allowed,
                source_version=source_version,
                dry_run=dry_run,
                extra_blocking_reasons=extra_blocking_reasons,
            )
        )
    return writes


def _write_funding_partitions_and_manifests(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    symbol: str,
    calendar: CalendarDefinition,
    source_version: str,
    dry_run: bool,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    writes: list[SliceWrite] = []
    frame = canonical.assign(
        year=canonical["timestamp_utc"].dt.year.astype(str),
        month=canonical["timestamp_utc"].dt.month.map(lambda value: f"{int(value):02d}"),
    )
    for (year, month), group in frame.groupby(["year", "month"], sort=True):
        path = (
            Path(repo_root)
            / "data"
            / "canonical"
            / "funding"
            / "market=crypto_perp"
            / "source=hyperliquid"
            / f"symbol={symbol.upper()}"
            / f"year={year}"
            / f"month={month}"
            / "part.parquet"
        )
        output = group.drop(columns=["year", "month"])
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, engine="pyarrow", index=False)
        writes.append(
            _manifest_for_partition(
                repo_root=repo_root,
                canonical_paths=[path],
                frame=output,
                source="hyperliquid",
                market="crypto_perp",
                symbol=symbol.upper(),
                timeframe="funding_1h",
                calendar=calendar,
                adjustment_policy=CRYPTO_ADJUSTMENT_POLICY,
                authoritative_allowed=True,
                source_version=source_version,
                dry_run=dry_run,
            )
        )
    return writes


def _write_panel_partitions_and_manifests(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    market: str,
    source: str,
    symbol: str,
    timeframe: str,
    kind: str,
    adjustment_policy: str,
    source_version: str,
    dry_run: bool,
    extra_blocking_reasons: list[str] | None = None,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    writes: list[SliceWrite] = []
    frame = canonical.assign(
        year=canonical["timestamp_utc"].dt.year.astype(str),
        month=canonical["timestamp_utc"].dt.month.map(lambda value: f"{int(value):02d}"),
    )
    for (year, month), group in frame.groupby(["year", "month"], sort=True):
        path = (
            Path(repo_root)
            / "data"
            / "canonical"
            / "panels"
            / f"market={market}"
            / f"source={source}"
            / f"kind={kind}"
            / f"symbol={symbol.upper()}"
            / f"timeframe={timeframe}"
            / f"year={year}"
            / f"month={month}"
            / "part.parquet"
        )
        output = group.drop(columns=["year", "month"])
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
                symbol=symbol.upper(),
                timeframe=timeframe,
                calendar=None,
                adjustment_policy=adjustment_policy,
                authoritative_allowed=False,
                source_version=source_version,
                dry_run=dry_run,
                extra_blocking_reasons=extra_blocking_reasons,
            )
        )
    return writes


def _write_compact_bar_slice_and_manifest(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    market: str,
    source: str,
    symbol: str,
    timeframe: str,
    kind: str,
    calendar: CalendarDefinition | None,
    adjustment_policy: str,
    authoritative_allowed: bool,
    source_version: str,
    dry_run: bool,
    extra_blocking_reasons: list[str] | None = None,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    path = _compact_canonical_path(
        repo_root=repo_root,
        root="bars",
        market=market,
        source=source,
        kind=kind,
        symbol=symbol,
        timeframe=timeframe,
    )
    output = canonical.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(path, engine="pyarrow", index=False)
    return [
        _manifest_for_partition(
            repo_root=repo_root,
            canonical_paths=[path],
            frame=output,
            source=source,
            market=market,
            symbol=symbol.upper(),
            timeframe=timeframe,
            calendar=calendar,
            adjustment_policy=adjustment_policy,
            authoritative_allowed=authoritative_allowed,
            source_version=source_version,
            dry_run=dry_run,
            extra_blocking_reasons=extra_blocking_reasons,
        )
    ]


def _write_compact_panel_slice_and_manifest(
    *,
    repo_root: Path,
    canonical: pd.DataFrame,
    market: str,
    source: str,
    symbol: str,
    timeframe: str,
    kind: str,
    adjustment_policy: str,
    source_version: str,
    dry_run: bool,
    authoritative_allowed: bool = False,
    calendar: CalendarDefinition | None = None,
    extra_blocking_reasons: list[str] | None = None,
) -> list[SliceWrite]:
    if canonical.empty:
        return []
    path = _compact_canonical_path(
        repo_root=repo_root,
        root="panels",
        market=market,
        source=source,
        kind=kind,
        symbol=symbol,
        timeframe=timeframe,
    )
    output = canonical.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(path, engine="pyarrow", index=False)
    return [
        _manifest_for_partition(
            repo_root=repo_root,
            canonical_paths=[path],
            frame=output,
            source=source,
            market=market,
            symbol=symbol.upper(),
            timeframe=timeframe,
            calendar=calendar,
            adjustment_policy=adjustment_policy,
            authoritative_allowed=authoritative_allowed,
            source_version=source_version,
            dry_run=dry_run,
            extra_blocking_reasons=extra_blocking_reasons,
        )
    ]


def _existing_compact_write(
    *,
    repo_root: Path,
    root: str,
    market: str,
    source: str,
    kind: str,
    symbol: str,
    timeframe: str,
) -> SliceWrite | None:
    canonical_path = _compact_canonical_path(
        repo_root=repo_root,
        root=root,
        market=market,
        source=source,
        kind=kind,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not canonical_path.exists():
        return None
    manifest_root = Path(repo_root) / "data" / "manifests" / "slices" / source / market / symbol.upper() / timeframe
    manifests = sorted(manifest_root.glob("*.market_data_manifest.json"))
    if not manifests:
        return None
    manifest_path = manifests[-1]
    return SliceWrite(
        manifest_path=manifest_path,
        canonical_paths=[canonical_path],
        manifest=load_market_manifest(manifest_path),
    )


def _compact_canonical_path(
    *,
    repo_root: Path,
    root: str,
    market: str,
    source: str,
    kind: str,
    symbol: str,
    timeframe: str,
) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "canonical"
        / root
        / f"market={market}"
        / f"source={source}"
        / f"kind={kind}"
        / f"symbol={symbol.upper()}"
        / f"timeframe={timeframe}"
        / "part.parquet"
    )


def _write_static_reference_frame(
    *,
    repo_root: Path,
    frame: pd.DataFrame,
    market: str,
    source: str,
    kind: str,
    dry_run: bool,
) -> list[str]:
    if frame.empty:
        return []
    path = _static_reference_path(repo_root=repo_root, market=market, source=source, kind=kind)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, engine="pyarrow", index=False)
    return [_rel(path, repo_root)]


def _write_static_reference_slice_and_manifest(
    *,
    repo_root: Path,
    frame: pd.DataFrame,
    source_path: Path,
    market: str,
    source: str,
    symbol: str,
    timeframe: str,
    kind: str,
    calendar: CalendarDefinition,
    adjustment_policy: str,
    source_version: str,
    dry_run: bool,
) -> SliceWrite:
    canonical_path = _static_reference_path(repo_root=repo_root, market=market, source=source, kind=kind)
    rel_source = _rel(source_path, repo_root)
    lrs_lineage = _lrs_export_lineage(repo_root, [rel_source])
    start, end = _lrs_manifest_window(repo_root, [rel_source], calendar)
    checksum = "" if dry_run else parquet_content_checksum(canonical_path)
    blocking_reasons = []
    if not checksum:
        blocking_reasons.append("checksum missing")
    if not is_git_commit_sha(source_version):
        blocking_reasons.append("source_version is not a data repo commit SHA")
    lineage = {
        "authority_status": "archived_lrs_sector_map_export",
        "source_endpoint": rel_source,
        "corporate_action_policy": adjustment_policy,
        "sector_schema_version": "krx_sector_map_v1",
        "session_policy": "krx_daily_bar_session_v1",
        "strategy_data_family": "k_stock_olr_kalcb",
        **lrs_lineage,
    }
    manifest = MarketDataManifest(
        source=source,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start,
        end_ts=end,
        expected_bars=len(frame),
        actual_bars=len(frame),
        coverage_ratio=1.0 if len(frame) else 0.0,
        missing_ranges=[],
        session_calendar=calendar.calendar_id,
        timezone=calendar.timezone,
        checksum=checksum,
        source_version=source_version,
        adjustment_policy=adjustment_policy,
        fee_model_version=DEFAULT_FEE_MODEL,
        slippage_model_version=DEFAULT_SLIPPAGE_MODEL,
        lineage=lineage,
        usable_for_authoritative_validation=not blocking_reasons and len(frame) > 0,
        blocking_reasons=blocking_reasons,
    )
    return _write_slice_product(repo_root, manifest, [canonical_path], dry_run=dry_run)


def _static_reference_path(*, repo_root: Path, market: str, source: str, kind: str) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "canonical"
        / "reference"
        / f"market={market}"
        / f"source={source}"
        / f"kind={kind}"
        / "part.parquet"
    )


def _slice_coverage_report(
    *,
    repo_root: Path,
    frame: pd.DataFrame,
    source: str,
    market: str,
    symbol: str,
    timeframe: str,
    calendar: CalendarDefinition,
):
    return DataSliceProduct(repo_root).coverage_for_request(
        calendar=calendar,
        request=SliceRequest(
            market=market,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
        ),
        timestamps=frame["timestamp_utc"],
        exchange_timestamps=frame.get("timestamp_exchange"),
    )


def _manifest_for_partition(
    *,
    repo_root: Path,
    canonical_paths: list[Path],
    frame: pd.DataFrame,
    source: str,
    market: str,
    symbol: str,
    timeframe: str,
    calendar: CalendarDefinition | None,
    adjustment_policy: str,
    authoritative_allowed: bool,
    source_version: str,
    dry_run: bool,
    extra_blocking_reasons: list[str] | None = None,
) -> SliceWrite:
    start = pd.Timestamp(frame["timestamp_utc"].min()).to_pydatetime()
    end = pd.Timestamp(frame["timestamp_utc"].max()).to_pydatetime()
    if calendar is None:
        expected = len(frame)
        actual = len(frame)
        missing = []
        session_calendar = ""
        timezone_name = "UTC"
    else:
        coverage = _slice_coverage_report(
            repo_root=repo_root,
            frame=frame,
            source=source,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            calendar=calendar,
        )
        expected, actual, missing = (
            coverage.expected,
            coverage.actual,
            coverage.missing_ranges,
        )
        session_calendar = calendar.calendar_id
        timezone_name = calendar.timezone
    known_no_trade_dates, no_trade_authority_checksum = _known_symbol_no_trade_dates(
        repo_root,
        source=source,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
    )
    excused_no_trade_count = 0
    if known_no_trade_dates and missing:
        retained_missing = []
        for item in missing:
            item_dates = _missing_range_dates(item)
            if item_dates and item_dates.issubset(known_no_trade_dates):
                excused_no_trade_count += len(item_dates)
            else:
                retained_missing.append(item)
        if excused_no_trade_count:
            expected = max(actual, expected - excused_no_trade_count)
            missing = retained_missing
    blocking_reasons = list(extra_blocking_reasons or [])
    if calendar is None:
        blocking_reasons.append("session calendar missing")
    if missing:
        blocking_reasons.append("missing ranges present")
    if not is_git_commit_sha(source_version):
        blocking_reasons.append("source_version is not a data repo commit SHA")
    checksum = "" if dry_run else parquet_content_checksum(canonical_paths[0])
    if not checksum:
        blocking_reasons.append("checksum missing")
    authoritative = (
        authoritative_allowed
        and not blocking_reasons
        and expected > 0
        and actual > 0
        and (actual / expected if expected else 0.0) >= 0.95
    )
    lineage = _diagnostic_lineage(
        frame,
        repo_root=repo_root,
        source=source,
        market=market,
        timeframe=timeframe,
        adjustment_policy=adjustment_policy,
    )
    if excused_no_trade_count:
        lineage["symbol_no_trade_authority"] = "krx_symbol_no_trade_dates_v1"
        lineage["symbol_no_trade_authority_checksum"] = no_trade_authority_checksum
        lineage["symbol_no_trade_dates_excused"] = str(excused_no_trade_count)
    manifest = MarketDataManifest(
        source=source,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start,
        end_ts=end,
        expected_bars=expected,
        actual_bars=actual,
        coverage_ratio=(actual / expected if expected else 0.0),
        missing_ranges=missing,
        session_calendar=session_calendar,
        timezone=timezone_name,
        checksum=checksum,
        source_version=source_version,
        adjustment_policy=adjustment_policy,
        fee_model_version=DEFAULT_FEE_MODEL,
        slippage_model_version=DEFAULT_SLIPPAGE_MODEL,
        lineage=lineage,
        usable_for_authoritative_validation=authoritative,
        blocking_reasons=blocking_reasons,
    )
    return _write_slice_product(repo_root, manifest, canonical_paths, dry_run=dry_run)


def _diagnostic_lineage(
    frame: pd.DataFrame,
    *,
    repo_root: Path,
    source: str,
    market: str,
    timeframe: str,
    adjustment_policy: str,
) -> dict[str, str]:
    source_files = sorted({str(value) for value in frame.get("source_file", []) if str(value).strip()})
    source_endpoint = ",".join(source_files[:5])
    snapshot = _snapshot_id_from_source_files(source_files)
    has_hyperliquid_raw = any("data/raw/hyperliquid/" in value for value in source_files)
    has_reference_snapshot = any("data/imported/reference_snapshot_" in value for value in source_files)
    if source == "hyperliquid" and has_hyperliquid_raw and has_reference_snapshot:
        authority_status = "archived_reference_snapshot_plus_live_hyperliquid_refresh"
    elif source == "hyperliquid" and has_hyperliquid_raw:
        authority_status = "live_hyperliquid_refresh"
    else:
        authority_status = "diagnostic_reference_snapshot"
    lineage = {
        "authority_status": authority_status,
        "source_endpoint": source_endpoint,
        "export_id": snapshot,
        "pulled_at_utc": "",
        "config_hash": "",
        "adjustment_policy": adjustment_policy,
    }
    if market.startswith("krx"):
        lineage["corporate_action_policy"] = adjustment_policy
        lineage["sector_schema_version"] = "krx_sector_map_v1"
        lineage["session_policy"] = "krx_daily_bar_session_v1"
        if "flow" in timeframe:
            lineage["flow_schema_version"] = "krx_daily_flow_v1"
    if source == "lrs":
        lineage["local_store"] = "lrs_sqlite_or_parquet"
        lineage["upstream_sources"] = "pykrx,naver"
        lineage["refresh_adapter"] = "none"
        lineage.update(_lrs_export_lineage(repo_root, source_files))
        if any("k_stock_trader/data/krx_daily_parquet" in value for value in source_files):
            lineage["strategy_data_family"] = "k_stock_olr_kalcb"
    if market == "us_equity":
        lineage["corporate_action_policy"] = adjustment_policy
        lineage["raw_adjustment_policy"] = "raw_ohlcv_with_ibkr_metadata_v1"
        lineage["session_policy"] = "diagnostic_snapshot_session_unverified"
    if market == "cme_futures":
        lineage["session_policy"] = "diagnostic_snapshot_session_unverified"
        lineage["roll_policy"] = ""
        lineage["contract_chain_checksum"] = ""
        if timeframe.endswith("_bid_ask"):
            lineage["quote_schema_version"] = "ibkr_bid_ask_snapshot_v1"
    if source == "kis" and market == "krx_equity":
        lineage.update(
            _kis_intraday_import_lineage(
                repo_root,
                source_files,
                frame=frame,
                timeframe=timeframe,
                adjustment_policy=adjustment_policy,
            )
        )
    return lineage


def _lrs_export_lineage(repo_root: Path, source_files: list[str]) -> dict[str, str]:
    manifest = _lrs_export_manifest(repo_root, source_files)
    if not manifest:
        return {}
    dataset_version = str(manifest.get("dataset_version", "") or "")
    source_fingerprint = str(manifest.get("source_fingerprint", "") or "")
    generated_at = str(manifest.get("generated_at", "") or "")
    source_label = str(manifest.get("source_label", "") or "")
    if source_label == "nulrimok_lrs_sqlite":
        source_label = "olr_kalcb_lrs_sqlite"
    config_payload = {
        "dataset_version": dataset_version,
        "source_label": source_label,
        "start": manifest.get("start", ""),
        "end": manifest.get("end", ""),
        "source_fingerprint": source_fingerprint,
    }
    return {
        "export_id": ":".join(part for part in (dataset_version, source_fingerprint[:16]) if part),
        "pulled_at_utc": generated_at,
        "config_hash": _stable_payload_hash(config_payload),
        "config_hash_kind": "lrs_export_manifest_hash",
        "lrs_dataset_version": dataset_version,
        "lrs_source_label": source_label,
        "lrs_source_fingerprint": source_fingerprint,
        "local_source_db_path": str(manifest.get("source_db_path", "") or ""),
    }


def _lrs_export_manifest(repo_root: Path, source_files: list[str]) -> dict:
    for source_file in source_files:
        path = Path(repo_root) / source_file
        parts = list(path.parts)
        for index, part in enumerate(parts):
            if part == "krx_daily_parquet":
                manifest_path = Path(*parts[: index + 1]) / "manifest.json"
                if manifest_path.exists():
                    try:
                        payload = read_json(manifest_path)
                    except (OSError, ValueError):
                        return {}
                    return payload if isinstance(payload, dict) else {}
    return {}


def _lrs_manifest_window(
    repo_root: Path,
    source_files: list[str],
    calendar: CalendarDefinition,
) -> tuple[datetime, datetime]:
    manifest = _lrs_export_manifest(repo_root, source_files)
    start = str(manifest.get("start", "") or "")
    end = str(manifest.get("end", "") or "")
    if start and end:
        start_ts = pd.Timestamp(f"{start} {calendar.session_close}", tz=calendar.timezone).tz_convert("UTC")
        end_ts = pd.Timestamp(f"{end} {calendar.session_close}", tz=calendar.timezone).tz_convert("UTC")
        return start_ts.to_pydatetime(), end_ts.to_pydatetime()
    now = datetime.now(timezone.utc)
    return now, now


def _is_k_stock_kis_intraday_authority(repo_root: Path, *, symbol: str, timeframe: str) -> bool:
    return timeframe == "5m" and symbol.zfill(6) in _k_stock_requirement_symbols(repo_root)


def _k_stock_requirement_symbols(repo_root: Path) -> set[str]:
    path = Path(repo_root) / "data" / "requirements" / "strategies" / "k_stock" / "portfolio.json"
    if not path.exists():
        return set()
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return set()
    symbols = {
        str(item.get("symbol", "")).zfill(6)
        for item in payload.get("requirements", [])
        if item.get("source") == "kis" and item.get("timeframe") == "5m"
    }
    return {symbol for symbol in symbols if symbol and symbol != "000000"}


def _kis_intraday_import_lineage(
    repo_root: Path,
    source_files: list[str],
    *,
    frame: pd.DataFrame,
    timeframe: str,
    adjustment_policy: str,
) -> dict[str, str]:
    symbols = sorted({str(value).zfill(6) for value in frame.get("symbol", []) if str(value).strip()})
    symbol = symbols[0] if len(symbols) == 1 else ""
    raw_timeframe = "1m" if timeframe != "1m" else timeframe
    endpoint = (
        "kis://uapi/domestic-stock/v1/quotations/"
        f"inquire-time-dailychartprice/J/{symbol}/{timeframe}"
        if symbol
        else "kis://uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    )
    source_stats = _source_file_stats(repo_root, source_files)
    payload = {
        "source_files": source_files,
        "source_stats": source_stats,
        "symbol": symbol,
        "timeframe": timeframe,
        "raw_timeframe": raw_timeframe,
        "session_policy": "krx_stock_regular_session_0900_1530_kst_v1",
        "adjustment_policy": adjustment_policy,
    }
    lineage = {
        "authority_status": "archived_kis_updater_parquet",
        "source_endpoint": endpoint,
        "source_files": ",".join(source_files[:20]),
        "source_file_count": str(len(source_files)),
        "export_id": f"kis-intraday-parquet-v1:{_stable_payload_hash(payload)[:16]}",
        "pulled_at_utc": _latest_source_mtime_utc(repo_root, source_files),
        "config_hash": _stable_payload_hash(payload),
        "session_policy": "krx_stock_regular_session_0900_1530_kst_v1",
        "timestamp_policy": "krx_exchange_timestamp_kst_to_utc_v1",
        "corporate_action_policy": adjustment_policy,
        "raw_timeframe": raw_timeframe,
        "derived_from_raw_timeframe": str(timeframe != raw_timeframe).lower(),
        "refresh_adapter": "k_stock_trader.update_kis_intraday.py",
        "archive_import_policy": "source_owned_kis_updater_parquet_v1",
    }
    if symbol in _k_stock_requirement_symbols(repo_root):
        lineage["strategy_data_family"] = "k_stock_olr_kalcb"
    return lineage


def _source_file_stats(repo_root: Path, source_files: list[str]) -> list[dict[str, object]]:
    stats = []
    for source_file in source_files:
        path = Path(repo_root) / source_file
        if not path.exists():
            continue
        stat = path.stat()
        stats.append(
            {
                "path": source_file,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return stats


def _latest_source_mtime_utc(repo_root: Path, source_files: list[str]) -> str:
    mtimes = [
        (Path(repo_root) / source_file).stat().st_mtime
        for source_file in source_files
        if (Path(repo_root) / source_file).exists()
    ]
    if not mtimes:
        return ""
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()


def _known_symbol_no_trade_dates(
    repo_root: Path,
    *,
    source: str,
    market: str,
    symbol: str,
    timeframe: str,
) -> tuple[set[pd.Timestamp], str]:
    krx_daily = source == "lrs" and market == "krx_equity" and timeframe == "1d"
    kis_intraday = (
        source == "kis"
        and market == "krx_equity"
        and timeframe in {"1m", "5m", "15m", "30m", "1h"}
    )
    if not (krx_daily or kis_intraday):
        return set(), ""
    path = Path(repo_root) / "data" / "calendars" / "krx_symbol_no_trade_dates.yaml"
    if not path.exists():
        return set(), ""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set(), ""
    symbol_payload = (payload.get("symbols") or {}).get(symbol.upper()) or {}
    dates = {
        pd.Timestamp(value).normalize()
        for value in symbol_payload.get("dates", [])
        if str(value).strip()
    }
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return dates, checksum


def _missing_range_dates(item: object) -> set[pd.Timestamp]:
    if isinstance(item, dict):
        start_value = item.get("start_ts", "")
        end_value = item.get("end_ts", "")
    else:
        start_value = getattr(item, "start_ts", "")
        end_value = getattr(item, "end_ts", "")
    start = pd.Timestamp(start_value)
    end = pd.Timestamp(end_value)
    if pd.isna(start) or pd.isna(end) or end < start:
        return set()
    return {
        ts.normalize().tz_localize(None)
        for ts in pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    }


def _stable_payload_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _snapshot_id_from_source_files(source_files: list[str]) -> str:
    for source_file in source_files:
        for part in Path(source_file).parts:
            if part.startswith("reference_snapshot_"):
                return part
    return "unknown_snapshot"




def _write_slice_product(
    repo_root: Path,
    manifest: MarketDataManifest,
    canonical_paths: list[Path],
    *,
    dry_run: bool,
) -> SliceWrite:
    """Normalization-owned compatibility adapter; remove when sources call DataSliceProduct directly."""
    product = DataSliceProduct(repo_root)
    if dry_run:
        return SliceWrite(
            manifest_path=product.manifest_path(manifest),
            canonical_paths=canonical_paths,
            manifest=manifest,
        )
    return product.write_slice(
        CanonicalSlice(
            request=SliceRequest(
                market=manifest.market,
                source=manifest.source,
                symbol=manifest.symbol,
                timeframe=manifest.timeframe,
            ),
            canonical_paths=canonical_paths,
            manifest=manifest,
        )
    )




def _parse_trading_raw_name(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    symbol = parts[0].upper()
    if parts[-2:] == ["bid", "ask"]:
        timeframe = f"{parts[1]}_bid_ask"
        kind = "bid_ask"
    else:
        timeframe = parts[1] if len(parts) > 1 else "1d"
        if timeframe == "daily":
            timeframe = "1d"
        kind = "trades"
    return symbol, timeframe, kind


def _is_duplicate_daily_alias(path: Path, available_stems: set[str]) -> bool:
    parts = path.stem.split("_")
    return len(parts) == 2 and parts[1] == "daily" and f"{parts[0]}_1d" in available_stems


def _parse_raw_symbol_timeframe(stem: str) -> tuple[str, str]:
    if stem.endswith("_daily_panama"):
        return stem[: -len("_daily_panama")].upper(), "1d_panama"
    for suffix in ("1m_bid_ask", "30m", "15m", "5m", "1h", "4h", "1d", "1m", "daily"):
        marker = f"_{suffix}"
        if stem.endswith(marker):
            timeframe = "1d" if suffix == "daily" else suffix
            return stem[: -len(marker)].upper(), timeframe
    symbol, _, timeframe = stem.rpartition("_")
    return (symbol or stem).upper(), timeframe or "1d"


def _normalize_krx_symbol(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text.upper()


def _market_for_seed_symbol(symbol: str) -> str:
    futures = {"ES", "MES", "NQ", "MNQ", "YM", "RTY", "CL", "GC", "SI", "HG", "ZN", "ZB"}
    return "cme_futures" if symbol.upper() in futures else "us_equity"


def _source_column(frame: pd.DataFrame, *names: str) -> str | None:
    columns = {str(column): column for column in frame.columns}
    lower = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name in columns:
            return columns[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _localized_timestamps(values: pd.Series, timezone_name: str) -> tuple[pd.Series, pd.Series]:
    timestamp = pd.to_datetime(values)
    if timestamp.dt.tz is None:
        exchange = timestamp.dt.tz_localize(timezone_name)
    else:
        exchange = timestamp.dt.tz_convert(timezone_name)
    return exchange.dt.tz_convert("UTC"), exchange


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
