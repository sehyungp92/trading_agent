"""Declare reproducible source requests for imported legacy parquet files."""

from __future__ import annotations

import hashlib
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .checksums import sha256_file
from .io import write_json

SOURCE_REQUEST_MANIFEST_SCHEMA_VERSION = "source_request_manifest_v1"
SOURCE_REQUEST_SCHEMA_VERSION = "source_request_v1"

FUTURES_SYMBOLS = {
    "ES",
    "MES",
    "NQ",
    "MNQ",
    "RTY",
    "M2K",
    "YM",
    "MYM",
    "GC",
    "MGC",
    "CL",
    "MCL",
}

FUTURES_EXCHANGE_BY_SYMBOL = {
    "ES": "CME",
    "MES": "CME",
    "NQ": "CME",
    "MNQ": "CME",
    "RTY": "CME",
    "M2K": "CME",
    "YM": "CBOT",
    "MYM": "CBOT",
    "GC": "COMEX",
    "MGC": "COMEX",
    "CL": "NYMEX",
    "MCL": "NYMEX",
}

CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL"}

US_EQUITY_PRIMARY_EXCHANGE_FALLBACKS = {
    "AAPL": "NASDAQ",
    "GLD": "ARCA",
    "HYG": "ARCA",
    "IWM": "ARCA",
    "MSFT": "NASDAQ",
    "QQQ": "NASDAQ",
    "SPY": "ARCA",
    "USO": "ARCA",
    "XLB": "ARCA",
    "XLC": "ARCA",
    "XLE": "ARCA",
    "XLF": "ARCA",
    "XLI": "ARCA",
    "XLK": "ARCA",
    "XLP": "ARCA",
    "XLRE": "ARCA",
    "XLU": "ARCA",
    "XLV": "ARCA",
    "XLY": "ARCA",
}

TIMEFRAME_ALIASES = {
    "daily": "1d",
    "day": "1d",
    "1day": "1d",
}

KNOWN_TIMEFRAMES = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
    "daily",
    "1m_bid_ask",
    "funding_1h",
    "funding_8h",
}

REQUIRED_ENV_BY_SOURCE_KIND = {
    "hyperliquid_candles": [],
    "hyperliquid_funding": [],
    "ibkr_cme_futures_historical_bars": [
        "IBKR_HOST",
        "IBKR_PORT",
        "IBKR_CLIENT_ID",
        "IBKR_READ_ONLY_ACK",
    ],
    "ibkr_us_equity_historical_bars": [
        "IBKR_HOST",
        "IBKR_PORT",
        "IBKR_CLIENT_ID",
        "IBKR_READ_ONLY_ACK",
    ],
    "kis_krx_intraday_bars": [
        "KIS_APP_KEY",
        "KIS_APP_SECRET",
        "KIS_ACCOUNT_MODE",
        "KIS_READ_ONLY_ACK",
    ],
    "lrs_krx_daily_export": ["LRS_DB_PATH", "LRS_EXPORT_ROOT"],
    "lrs_krx_flow_export": ["LRS_DB_PATH", "LRS_EXPORT_ROOT"],
    "lrs_krx_table_export": ["LRS_DB_PATH", "LRS_EXPORT_ROOT"],
    "legacy_derived_seed": [],
}

_RAW_NAME_RE = re.compile(r"^(?P<symbol>.+)_(?P<timeframe>.+)$")
_KIS_INTRADAY_RE = re.compile(
    r"^(?P<symbol>\d{6})_(?P<timeframe>1m|5m|15m|30m|1h)_"
    r"(?P<start>\d{8})_(?P<end>\d{8})$"
)
_KRX_DATED_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9_]+)_(?P<table>[a-z_]+)_"
    r"(?P<start>\d{8})_(?P<end>\d{8})$"
)


@dataclass(frozen=True)
class ParsedRawName:
    symbol: str
    timeframe: str
    bar_type: str
    variant: str | None = None


def declare_source_requests(
    repo_root: Path,
    *,
    snapshot: str = "2026-05-30",
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build source request declarations for every parquet in a legacy snapshot."""

    repo_root = repo_root.resolve()
    snapshot_root = repo_root / "data" / "imported" / f"reference_snapshot_{snapshot}"
    if not snapshot_root.exists():
        msg = f"legacy snapshot not found: {snapshot_root}"
        raise FileNotFoundError(msg)

    parquet_paths = sorted(snapshot_root.rglob("*.parquet"))
    requests: list[dict[str, Any]] = []
    unclassified: list[str] = []
    trading_primary_exchanges = _load_trading_primary_exchange_map(repo_root)

    for path in parquet_paths:
        rel_to_snapshot = path.relative_to(snapshot_root).as_posix()
        rel_to_repo = path.relative_to(repo_root).as_posix()
        request = _classify_legacy_parquet(
            path=path,
            rel_to_snapshot=rel_to_snapshot,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            trading_primary_exchanges=trading_primary_exchanges,
        )
        if request is None:
            unclassified.append(rel_to_repo)
            continue
        requests.append(request)

    source_kind_counts = Counter(request["source_kind"] for request in requests)
    source_counts = Counter(request["source"] for request in requests)
    family_counts = Counter(request["legacy_family"] for request in requests)
    market_counts = Counter(request["market"] for request in requests)

    manifest = {
        "schema_version": SOURCE_REQUEST_MANIFEST_SCHEMA_VERSION,
        "snapshot": snapshot,
        "snapshot_root": snapshot_root.relative_to(repo_root).as_posix(),
        "legacy_file_count": len(parquet_paths),
        "request_count": len(requests),
        "unclassified_count": len(unclassified),
        "unclassified": unclassified,
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "legacy_family_counts": dict(sorted(family_counts.items())),
        "market_counts": dict(sorted(market_counts.items())),
        "requests": requests,
    }

    output_base = output_root or (
        repo_root / "data" / "source_requests" / f"reference_snapshot_{snapshot}"
    )
    if not dry_run:
        _write_source_request_artifacts(output_base=output_base, manifest=manifest)

    return manifest


def _classify_legacy_parquet(
    *,
    path: Path,
    rel_to_snapshot: str,
    rel_to_repo: str,
    snapshot: str,
    trading_primary_exchanges: dict[str, str],
) -> dict[str, Any] | None:
    parts = rel_to_snapshot.split("/")
    if parts[:3] == ["crypto_trader", "data", "candles"] and len(parts) == 5:
        symbol = parts[3].upper()
        timeframe = _normalize_timeframe(Path(parts[4]).stem)
        return _build_crypto_candle_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            symbol=symbol,
            timeframe=timeframe,
        )

    if parts[:3] == ["crypto_trader", "data", "funding"] and len(parts) == 4:
        symbol = Path(parts[3]).stem.upper()
        return _build_crypto_funding_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            symbol=symbol,
        )

    if (
        parts[:3] == ["k_stock_trader", "data", "kis_intraday_parquet"]
        and len(parts) == 5
    ):
        return _build_kis_intraday_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            stem=Path(parts[4]).stem,
        )

    if (
        parts[:3] == ["k_stock_trader", "data", "krx_daily_parquet"]
        and len(parts) >= 4
    ):
        return _build_krx_lrs_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            krx_parts=parts[3:],
        )

    if parts[:4] == ["trading", "backtests", "momentum", "data"] and len(parts) == 6:
        return _build_trading_raw_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family="trading_momentum",
            stem=Path(parts[5]).stem,
            trading_primary_exchanges=trading_primary_exchanges,
        )

    if parts[:4] == ["trading", "backtests", "swing", "data"] and len(parts) == 6:
        return _build_trading_raw_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family="trading_swing",
            stem=Path(parts[5]).stem,
            trading_primary_exchanges=trading_primary_exchanges,
        )

    if parts[:4] == ["trading", "backtests", "stock", "data"] and len(parts) == 6:
        return _build_trading_raw_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family="trading_stock",
            stem=Path(parts[5]).stem,
            force_market="us_equity",
            trading_primary_exchanges=trading_primary_exchanges,
        )

    if parts[:4] == ["trading", "backtests", "regime", "data"] and len(parts) == 6:
        return _build_legacy_seed_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family="trading_regime",
            seed_name=Path(parts[5]).stem,
        )

    if parts[:3] == ["trading", "data", "raw"] and len(parts) == 4:
        return _build_trading_raw_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family="trading_raw",
            stem=Path(parts[3]).stem,
            trading_primary_exchanges=trading_primary_exchanges,
        )

    return None


def _build_crypto_candle_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family="crypto_trader",
        source_kind="hyperliquid_candles",
        source="hyperliquid",
        market="crypto_perp",
        symbol=symbol,
        timeframe=timeframe,
        data_kind="ohlcv",
    )
    request.update(
        {
            "source_endpoint": f"hyperliquid://info/candles/{symbol}/{timeframe}",
            "download_request": {
                "provider": "hyperliquid",
                "request_type": "candles",
                "coin": symbol,
                "interval": timeframe,
                "start": request["legacy_time_bounds"].get("start"),
                "end": request["legacy_time_bounds"].get("end"),
            },
            "canonical_expectations": {
                "calendar": "hyperliquid_24x7_v1",
                "fee_policy": "hyperliquid_fee_schedule_v1",
                "slippage_policy": "crypto_portfolio_default_slippage_v1",
                "adjustment_policy": "crypto_raw_adjustment_policy_v1",
            },
            "authority_status": "declared_request_ready",
        }
    )
    return _finalize_request(request)


def _build_crypto_funding_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    symbol: str,
) -> dict[str, Any]:
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family="crypto_trader",
        source_kind="hyperliquid_funding",
        source="hyperliquid",
        market="crypto_perp",
        symbol=symbol,
        timeframe="funding_1h",
        data_kind="funding",
    )
    request.update(
        {
            "source_endpoint": f"hyperliquid://info/funding/{symbol}",
            "download_request": {
                "provider": "hyperliquid",
                "request_type": "funding_history",
                "coin": symbol,
                "start": request["legacy_time_bounds"].get("start"),
                "end": request["legacy_time_bounds"].get("end"),
            },
            "canonical_expectations": {
                "calendar": "hyperliquid_funding_1h_v1",
                "fee_policy": "hyperliquid_fee_schedule_v1",
                "slippage_policy": "not_applicable_funding_series_v1",
                "adjustment_policy": "crypto_funding_raw_adjustment_policy_v1",
            },
            "authority_status": "declared_request_ready",
        }
    )
    return _finalize_request(request)


def _build_kis_intraday_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    stem: str,
) -> dict[str, Any] | None:
    match = _KIS_INTRADAY_RE.match(stem)
    if not match:
        return None
    symbol = match.group("symbol")
    timeframe = _normalize_timeframe(match.group("timeframe"))
    filename_start = _yyyymmdd(match.group("start"))
    filename_end = _yyyymmdd(match.group("end"))
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family="k_stock_kis_intraday",
        source_kind="kis_krx_intraday_bars",
        source="kis",
        market="krx_equity",
        symbol=symbol,
        timeframe=timeframe,
        data_kind="ohlcv",
    )
    request.update(
        {
            "source_endpoint": (
                "kis://uapi/domestic-stock/v1/quotations/"
                f"inquire-time-dailychartprice/J/{symbol}/{timeframe}"
            ),
            "download_request": {
                "provider": "kis",
                "request_type": "domestic_stock_intraday_bars",
                "tr_id": "FHKST03010230",
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": symbol,
                "timeframe": timeframe,
                "raw_timeframe": "1m",
                "page_rows": 120,
                "filename_start": filename_start,
                "filename_end": filename_end,
                "start": request["legacy_time_bounds"].get("start") or filename_start,
                "end": request["legacy_time_bounds"].get("end") or filename_end,
                "include_previous_data": True,
                "cursor_params": ("FID_INPUT_DATE_1", "FID_INPUT_HOUR_1"),
                "derived_from_raw_timeframe": timeframe != "1m",
            },
            "canonical_expectations": {
                "calendar": "krx_equities_regular_session_v1",
                "session_policy": "krx_stock_regular_session_0900_1530_kst_v1",
                "timestamp_policy": "krx_exchange_timestamp_kst_to_utc_v1",
                "fee_policy": "k_stock_trader_fee_schedule_v1",
                "slippage_policy": "k_stock_trader_default_slippage_v1",
                "adjustment_policy": "krx_equity_raw_adjustment_policy_v1",
            },
            "authority_status": "declared_request_ready",
        }
    )
    return _finalize_request(request)


def _build_krx_lrs_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    krx_parts: list[str],
) -> dict[str, Any] | None:
    section = krx_parts[0]
    if section == "tables":
        table_name = Path(krx_parts[-1]).stem
        source_kind = "lrs_krx_table_export"
        market = _table_market(table_name)
        data_kind = _table_data_kind(table_name)
        symbol = "ALL"
        timeframe = _table_timeframe(table_name)
        download_request = {
            "provider": "lrs",
            "request_type": "local_research_table_export",
            "source_table": table_name,
            "export_scope": "table",
            "export_id_required": True,
            "pulled_at_utc_required": True,
            "config_hash_required": True,
            "source_db_env": "LRS_DB_PATH",
            "export_root_env": "LRS_EXPORT_ROOT",
        }
    else:
        stem = Path(krx_parts[-1]).stem
        match = _KRX_DATED_RE.match(stem)
        if not match:
            return None
        symbol = match.group("symbol")
        table_name = section
        source_kind = (
            "lrs_krx_daily_export"
            if section in {"daily_ohlcv", "index_ohlcv"}
            else "lrs_krx_flow_export"
        )
        market = "krx_index" if section == "index_ohlcv" else "krx_equity"
        data_kind = section
        timeframe = _table_timeframe(section)
        download_request = {
            "provider": "lrs",
            "request_type": "local_research_partition_export",
            "source_table": table_name,
            "symbol": symbol,
            "filename_start": _yyyymmdd(match.group("start")),
            "filename_end": _yyyymmdd(match.group("end")),
            "start": None,
            "end": None,
            "export_id_required": True,
            "pulled_at_utc_required": True,
            "config_hash_required": True,
            "source_db_env": "LRS_DB_PATH",
            "export_root_env": "LRS_EXPORT_ROOT",
        }

    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family="k_stock_krx_lrs",
        source_kind=source_kind,
        source="lrs",
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        data_kind=data_kind,
    )
    if source_kind != "lrs_krx_table_export":
        download_request["start"] = (
            request["legacy_time_bounds"].get("start") or download_request["filename_start"]
        )
        download_request["end"] = (
            request["legacy_time_bounds"].get("end") or download_request["filename_end"]
        )

    request.update(
        {
            "source_endpoint": f"lrs://local-research-export/{table_name}/{symbol}",
            "download_request": download_request,
            "canonical_expectations": {
                "calendar": "krx_equities_daily_calendar_v1",
                "session_policy": "krx_daily_bar_session_v1",
                "timestamp_policy": "krx_daily_date_anchor_v1",
                "fee_policy": "not_applicable_research_series_v1",
                "slippage_policy": "not_applicable_research_series_v1",
                "adjustment_policy": "krx_flow_raw_adjustment_policy_v1"
                if "flow" in data_kind
                else "krx_equity_raw_adjustment_policy_v1",
                "corporate_action_policy": "lrs_export_declares_corporate_action_policy_v1",
                "sector_flow_schema_version": "lrs_sector_flow_schema_v1",
            },
            "authority_status": "declared_local_research_export",
        }
    )
    return _finalize_request(request)


def _build_trading_raw_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    legacy_family: str,
    stem: str,
    trading_primary_exchanges: dict[str, str],
    force_market: str | None = None,
) -> dict[str, Any] | None:
    parsed = _parse_trading_raw_stem(stem)
    if parsed is None:
        return None

    market = force_market or (
        "cme_futures" if parsed.symbol in FUTURES_SYMBOLS else "us_equity"
    )
    if market == "cme_futures":
        return _build_ibkr_futures_request(
            path=path,
            rel_to_repo=rel_to_repo,
            snapshot=snapshot,
            legacy_family=legacy_family,
            symbol=parsed.symbol,
            timeframe=parsed.timeframe,
            bar_type=parsed.bar_type,
            variant=parsed.variant,
        )
    return _build_ibkr_us_equity_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family=legacy_family,
        symbol=parsed.symbol,
        timeframe=parsed.timeframe,
        bar_type=parsed.bar_type,
        variant=parsed.variant,
        primary_exchange=trading_primary_exchanges.get(parsed.symbol, ""),
    )


def _build_ibkr_futures_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    legacy_family: str,
    symbol: str,
    timeframe: str,
    bar_type: str,
    variant: str | None,
) -> dict[str, Any]:
    exchange = FUTURES_EXCHANGE_BY_SYMBOL.get(symbol, "CME")
    what_to_show = "BID_ASK" if bar_type == "bid_ask" else "TRADES"
    use_rth = timeframe == "1d"
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family=legacy_family,
        source_kind="ibkr_cme_futures_historical_bars",
        source="ibkr",
        market="cme_futures",
        symbol=symbol,
        timeframe=timeframe,
        data_kind=bar_type,
    )
    request.update(
        {
            "source_endpoint": (
                f"ibkr://historical-data/FUT/{exchange}/{symbol}/"
                f"{what_to_show}/{timeframe}"
            ),
            "download_request": {
                "provider": "ibkr",
                "request_type": "historical_bars",
                "sec_type": "FUT",
                "symbol": symbol,
                "exchange": exchange,
                "currency": "USD",
                "timeframe": timeframe,
                "what_to_show": what_to_show,
                "use_rth": use_rth,
                "start": request["legacy_time_bounds"].get("start"),
                "end": request["legacy_time_bounds"].get("end"),
                "contract_resolution_policy": "ibkr_contract_details_con_id_required_v1",
                "continuous_contract_policy": "ibkr_physical_contract_chain_panama_v1",
                "roll_policy": _futures_roll_policy(symbol),
                "variant": variant,
                "source_contract_column_required": True,
                "source_conid_column_required": True,
            },
            "canonical_expectations": {
                "calendar": _futures_calendar(symbol),
                "market_rule_authority": "data/market_rules/cme/equity_index_futures_v1.json",
                "session_policy": _futures_session_policy(symbol),
                "halt_policy": "source_product_halts_and_early_closes_required_v1",
                "fee_policy": "trading_futures_fee_schedule_v1",
                "slippage_policy": "trading_momentum_default_slippage_v1",
                "adjustment_policy": "cme_futures_panama_v1",
                "roll_checksum_required": True,
                "bid_ask_schema": "ibkr_bid_ask_bar_schema_v1"
                if bar_type == "bid_ask"
                else None,
            },
            "authority_status": "declared_requires_contract_chain",
        }
    )
    return _finalize_request(request)


def _build_ibkr_us_equity_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    legacy_family: str,
    symbol: str,
    timeframe: str,
    bar_type: str,
    variant: str | None,
    primary_exchange: str,
) -> dict[str, Any]:
    use_rth = timeframe in {"1d", "daily"}
    session_policy = (
        "us_equity_rth_session_0930_1600_new_york_v1"
        if use_rth
        else "us_equity_extended_session_0400_2000_new_york_v1"
    )
    calendar_policy = (
        "us_equities_rth_calendar_v1"
        if use_rth
        else "us_equities_extended_hours_calendar_v1"
    )
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family=legacy_family,
        source_kind="ibkr_us_equity_historical_bars",
        source="ibkr",
        market="us_equity",
        symbol=symbol,
        timeframe=timeframe,
        data_kind=bar_type,
    )
    request.update(
        {
            "source_endpoint": (
                "ibkr://historical-data/STK/SMART/"
                f"{primary_exchange or 'primary-exchange-required'}/{symbol}/TRADES/"
                f"{timeframe}/{'RTH' if use_rth else 'ETH'}"
            ),
            "download_request": {
                "provider": "ibkr",
                "request_type": "historical_bars",
                "sec_type": "STK",
                "symbol": symbol,
                "exchange": "SMART",
                "primary_exchange": primary_exchange,
                "currency": "USD",
                "timeframe": timeframe,
                "what_to_show": "TRADES",
                "use_rth": use_rth,
                "start": request["legacy_time_bounds"].get("start"),
                "end": request["legacy_time_bounds"].get("end"),
                "contract_resolution_policy": "ibkr_contract_details_con_id_required_v1",
                "source_identity_authority": "trading_config_contracts_and_routing_v1",
                "provider_contract_fields_required": [
                    "con_id",
                    "local_symbol",
                    "primary_exchange",
                ],
                "source_conid_column_required": True,
                "contract_resolution_cache_required": True,
                "variant": variant,
            },
            "canonical_expectations": {
                "calendar": calendar_policy,
                "session_policy": session_policy,
                "timestamp_policy": "exchange_timestamp_to_utc_v1",
                "fee_policy": "trading_stock_fee_schedule_v1",
                "slippage_policy": "trading_stock_default_slippage_v1",
                "raw_adjustment_policy": "ibkr_trades_unadjusted_raw_v1",
                "canonical_adjustment_policy": "us_equity_raw_adjustment_policy_v1",
                "corporate_action_policy": "split_dividend_policy_declared_per_bundle_v1",
            },
            "authority_status": "declared_request_ready",
        }
    )
    return _finalize_request(request)


def _build_legacy_seed_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    legacy_family: str,
    seed_name: str,
) -> dict[str, Any]:
    symbol = seed_name.upper()
    request = _base_request(
        path=path,
        rel_to_repo=rel_to_repo,
        snapshot=snapshot,
        legacy_family=legacy_family,
        source_kind="legacy_derived_seed",
        source="legacy_seed",
        market="global_macro",
        symbol=symbol,
        timeframe="1d",
        data_kind="derived_panel",
    )
    request.update(
        {
            "source_endpoint": f"legacy-seed://trading/regime/{seed_name}",
            "download_request": {
                "provider": "legacy_seed",
                "request_type": "derived_panel_rebuild",
                "seed_name": seed_name,
                "start": request["legacy_time_bounds"].get("start"),
                "end": request["legacy_time_bounds"].get("end"),
                "source_request_expansion_required": True,
            },
            "canonical_expectations": {
                "calendar": "global_macro_daily_calendar_v1",
                "fee_policy": "not_applicable_regime_seed_v1",
                "slippage_policy": "not_applicable_regime_seed_v1",
                "adjustment_policy": "regime_seed_derived_panel_policy_v1",
            },
            "authority_status": "declared_derived_seed_rebuild_required",
        }
    )
    return _finalize_request(request)


def _base_request(
    *,
    path: Path,
    rel_to_repo: str,
    snapshot: str,
    legacy_family: str,
    source_kind: str,
    source: str,
    market: str,
    symbol: str,
    timeframe: str,
    data_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_REQUEST_SCHEMA_VERSION,
        "legacy_snapshot": snapshot,
        "legacy_path": rel_to_repo,
        "legacy_family": legacy_family,
        "legacy_sha256": sha256_file(path),
        "legacy_row_count": _legacy_row_count(path),
        "legacy_time_bounds": _legacy_time_bounds(path),
        "source_kind": source_kind,
        "source": source,
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_kind": data_kind,
        "execution_requirements": {
            "required_env": REQUIRED_ENV_BY_SOURCE_KIND[source_kind],
            "read_only": True,
            "credentials_are_not_serialized": True,
        },
    }


def _load_trading_primary_exchange_map(repo_root: Path) -> dict[str, str]:
    mapping = dict(US_EQUITY_PRIMARY_EXCHANGE_FALLBACKS)
    repo_root = Path(repo_root).resolve()
    monorepo_root = repo_root.parent.parent if repo_root.parent.name == "packages" else repo_root.parent
    trading_root = monorepo_root / "bots" / "ibkr_trading"
    for rel_path in ("config/contracts.yaml", "config/routing.yaml"):
        payload = _read_yaml_mapping(trading_root / rel_path)
        for symbol, config in payload.items():
            if not isinstance(config, dict):
                continue
            primary = str(config.get("primary_exchange") or "").upper().strip()
            if primary:
                mapping[str(symbol).upper().strip()] = primary
    for rel_path in (
        "strategies/stock/alcb/universe_constituents.py",
        "strategies/stock/iaric/universe_constituents.py",
    ):
        mapping.update(_read_universe_primary_exchanges(trading_root / rel_path))
    return mapping


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_universe_primary_exchanges(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SP500_CONSTITUENTS" for target in targets):
            continue
        try:
            rows = ast.literal_eval(value)
        except Exception:
            return {}
        result: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, tuple) or len(row) < 3:
                continue
            symbol = str(row[0]).upper().strip()
            primary = str(row[2]).upper().strip()
            if symbol and primary:
                result[symbol] = primary
        return result
    return {}


def _finalize_request(request: dict[str, Any]) -> dict[str, Any]:
    request["canonical_expectations"] = {
        key: value
        for key, value in request["canonical_expectations"].items()
        if value is not None
    }
    request["request_id"] = _request_id(request)
    return request


def _parse_trading_raw_stem(stem: str) -> ParsedRawName | None:
    raw_stem = stem
    variant = None
    bar_type = "ohlcv"

    if raw_stem.lower().endswith("_1m_bid_ask"):
        symbol = raw_stem[: -len("_1m_bid_ask")].upper()
        return ParsedRawName(
            symbol=symbol,
            timeframe="1m",
            bar_type="bid_ask",
            variant=None,
        )

    match = _RAW_NAME_RE.match(raw_stem)
    if not match:
        return None

    symbol = match.group("symbol").upper()
    tail = match.group("timeframe").lower()

    if tail.endswith("_panama"):
        tail = tail.removesuffix("_panama")
        variant = "panama"
    elif tail.endswith("_direct"):
        tail = tail.removesuffix("_direct")
        variant = "direct"

    timeframe = _normalize_timeframe(tail)
    if timeframe not in KNOWN_TIMEFRAMES:
        return None
    return ParsedRawName(
        symbol=symbol,
        timeframe=timeframe,
        bar_type=bar_type,
        variant=variant,
    )


def _legacy_time_bounds(path: Path) -> dict[str, str | None]:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception:
        return {"column": None, "start": None, "end": None}

    names = parquet_file.schema.names
    for candidate in (
        "timestamp_utc",
        "timestamp",
        "datetime",
        "date",
        "time",
        "ts",
        "open_time",
        "__index_level_0__",
    ):
        if candidate not in names:
            continue
        try:
            table = parquet_file.read(columns=[candidate])
            series = table.column(0).to_pandas()
        except Exception:
            continue
        if series.empty:
            continue
        values = _coerce_utc_datetime(series, candidate).dropna()
        if values.empty:
            continue
        return {
            "column": candidate,
            "start": values.min().isoformat(),
            "end": values.max().isoformat(),
        }
    return {"column": None, "start": None, "end": None}


def _legacy_row_count(path: Path) -> int | None:
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def _coerce_utc_datetime(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return pd.to_datetime(series, utc=True, errors="coerce")
        max_abs = numeric.abs().max()
        if column in {"ts", "open_time"} or max_abs > 10**11:
            return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
        if column == "date" and 10_000_000 <= max_abs <= 99_999_999:
            return pd.to_datetime(
                series.astype("Int64").astype(str),
                format="%Y%m%d",
                utc=True,
                errors="coerce",
            )
    return pd.to_datetime(series, utc=True, errors="coerce")


def _request_id(request: dict[str, Any]) -> str:
    payload = {
        "schema_version": request["schema_version"],
        "legacy_path": request["legacy_path"],
        "source_kind": request["source_kind"],
        "source_endpoint": request["source_endpoint"],
        "download_request": request["download_request"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "src_req_" + hashlib.sha256(encoded).hexdigest()[:24]


def _write_source_request_artifacts(
    *,
    output_base: Path,
    manifest: dict[str, Any],
) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    write_json(output_base / "source_request_manifest.json", manifest)

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in manifest["requests"]:
        by_kind[request["source_kind"]].append(request)

    requests_dir = output_base / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    for source_kind, requests in sorted(by_kind.items()):
        write_json(
            requests_dir / f"{source_kind}.json",
            {
                "schema_version": SOURCE_REQUEST_MANIFEST_SCHEMA_VERSION,
                "snapshot": manifest["snapshot"],
                "source_kind": source_kind,
                "request_count": len(requests),
                "requests": requests,
            },
        )


def _normalize_timeframe(value: str) -> str:
    value = value.lower()
    return TIMEFRAME_ALIASES.get(value, value)


def _yyyymmdd(value: str) -> str:
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def _table_market(table_name: str) -> str:
    return "krx_index" if table_name.startswith("index_") else "krx_equity"


def _table_data_kind(table_name: str) -> str:
    if "flow" in table_name:
        return table_name
    if table_name in {"sector_map", "stock_metadata", "market_cap"}:
        return "reference_table"
    return table_name


def _table_timeframe(table_name: str) -> str:
    if "flow" in table_name:
        return "1d_flow"
    if table_name in {"daily_ohlcv", "index_ohlcv"}:
        return "1d"
    if table_name.startswith("daily_"):
        return "1d"
    return "static_or_daily"


def _futures_calendar(symbol: str) -> str:
    if symbol in {"ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM", "MYM"}:
        return "cme_equity_index_futures_calendar_v1"
    if symbol in {"GC", "MGC"}:
        return "comex_metals_futures_calendar_required_v1"
    if symbol in {"CL", "MCL"}:
        return "nymex_energy_futures_calendar_required_v1"
    return "cme_product_calendar_required_v1"


def _futures_session_policy(symbol: str) -> str:
    if symbol in {"ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM", "MYM"}:
        return "cme_equity_index_eth_session_with_daily_maintenance_halt_v1"
    return "futures_product_session_policy_required_v1"


def _futures_roll_policy(symbol: str) -> str:
    if symbol in {"ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM", "MYM"}:
        return "quarterly_index_future_roll_four_calendar_days_before_third_friday_v1"
    if symbol in {"GC", "MGC"}:
        return "metals_futures_contract_chain_roll_policy_required_v1"
    if symbol in {"CL", "MCL"}:
        return "energy_futures_contract_chain_roll_policy_required_v1"
    return "futures_contract_chain_roll_policy_required_v1"
