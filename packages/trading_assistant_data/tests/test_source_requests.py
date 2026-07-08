from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import trading_assistant_data.source_requests as source_requests
from tests.paths import MONOREPO_ROOT, package_workspace


def test_declares_exact_source_requests_for_representative_legacy_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    snapshot_root = repo_root / "data" / "imported" / "reference_snapshot_2026-05-30"

    _write_parquet(
        snapshot_root / "trading" / "backtests" / "momentum" / "data" / "raw" / "NQ_5m.parquet"
    )
    _write_parquet(
        snapshot_root / "trading" / "backtests" / "momentum" / "data" / "raw" / "ES_1d.parquet"
    )
    _write_parquet(
        snapshot_root / "trading" / "backtests" / "swing" / "data" / "raw" / "QQQ_1h.parquet"
    )
    _write_parquet(
        snapshot_root / "trading" / "backtests" / "stock" / "data" / "raw" / "MSFT_30m.parquet"
    )
    _write_parquet(
        snapshot_root
        / "k_stock_trader"
        / "data"
        / "kis_intraday_parquet"
        / "005930"
        / "005930_5m_20250512_20260512.parquet"
    )
    _write_parquet(
        snapshot_root
        / "k_stock_trader"
        / "data"
        / "krx_daily_parquet"
        / "daily_ohlcv"
        / "005930"
        / "005930_daily_ohlcv_20240101_20240531.parquet"
    )
    _write_parquet(snapshot_root / "crypto_trader" / "data" / "candles" / "BTC" / "1m.parquet")
    _write_parquet(snapshot_root / "crypto_trader" / "data" / "funding" / "BTC.parquet")

    manifest = source_requests.declare_source_requests(repo_root=repo_root)

    assert manifest["legacy_file_count"] == 8
    assert manifest["request_count"] == 8
    assert manifest["unclassified_count"] == 0
    by_path = {request["legacy_path"]: request for request in manifest["requests"]}

    nq = by_path[
        "data/imported/reference_snapshot_2026-05-30/trading/backtests/momentum/data/raw/NQ_5m.parquet"
    ]
    assert nq["source_kind"] == "ibkr_cme_futures_historical_bars"
    assert nq["download_request"]["sec_type"] == "FUT"
    assert (
        nq["download_request"]["continuous_contract_policy"]
        == "ibkr_physical_contract_chain_panama_v1"
    )
    assert nq["download_request"]["source_conid_column_required"] is True
    assert nq["canonical_expectations"]["roll_checksum_required"] is True

    qqq = by_path[
        "data/imported/reference_snapshot_2026-05-30/trading/backtests/swing/data/raw/QQQ_1h.parquet"
    ]
    assert qqq["source_kind"] == "ibkr_us_equity_historical_bars"
    assert qqq["download_request"]["sec_type"] == "STK"
    assert qqq["download_request"]["primary_exchange"] == "NASDAQ"
    assert (
        qqq["download_request"]["source_identity_authority"]
        == "trading_config_contracts_and_routing_v1"
    )
    assert qqq["download_request"]["source_conid_column_required"] is True
    assert qqq["download_request"]["contract_resolution_cache_required"] is True
    assert qqq["download_request"]["use_rth"] is False
    assert qqq["canonical_expectations"]["calendar"] == "us_equities_extended_hours_calendar_v1"

    msft = by_path[
        "data/imported/reference_snapshot_2026-05-30/trading/backtests/stock/data/raw/MSFT_30m.parquet"
    ]
    assert msft["download_request"]["use_rth"] is False
    assert msft["canonical_expectations"]["session_policy"].startswith(
        "us_equity_extended_session_"
    )

    kis = by_path[
        "data/imported/reference_snapshot_2026-05-30/k_stock_trader/data/kis_intraday_parquet/005930/005930_5m_20250512_20260512.parquet"
    ]
    assert kis["source_kind"] == "kis_krx_intraday_bars"
    assert kis["source_endpoint"].endswith("inquire-time-dailychartprice/J/005930/5m")
    assert kis["download_request"]["tr_id"] == "FHKST03010230"
    assert kis["download_request"]["raw_timeframe"] == "1m"
    assert kis["download_request"]["derived_from_raw_timeframe"] is True
    assert kis["download_request"]["include_previous_data"] is True
    assert kis["canonical_expectations"]["session_policy"].endswith("1530_kst_v1")

    lrs = by_path[
        "data/imported/reference_snapshot_2026-05-30/k_stock_trader/data/krx_daily_parquet/daily_ohlcv/005930/005930_daily_ohlcv_20240101_20240531.parquet"
    ]
    assert lrs["source"] == "lrs"
    assert lrs["download_request"]["export_id_required"] is True
    assert lrs["download_request"]["config_hash_required"] is True

    crypto = by_path[
        "data/imported/reference_snapshot_2026-05-30/crypto_trader/data/candles/BTC/1m.parquet"
    ]
    assert crypto["source"] == "hyperliquid"
    assert crypto["market"] == "crypto_perp"

    funding = by_path[
        "data/imported/reference_snapshot_2026-05-30/crypto_trader/data/funding/BTC.parquet"
    ]
    assert funding["source_kind"] == "hyperliquid_funding"
    assert funding["timeframe"] == "funding_1h"
    assert funding["canonical_expectations"]["calendar"] == "hyperliquid_funding_1h_v1"

    written_manifest = (
        repo_root
        / "data"
        / "source_requests"
        / "reference_snapshot_2026-05-30"
        / "source_request_manifest.json"
    )
    assert written_manifest.exists()
    assert json.loads(written_manifest.read_text(encoding="utf-8"))["request_count"] == 8


def test_declares_all_committed_reference_snapshot_parquets(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_root = repo_root / "data" / "imported" / "reference_snapshot_2026-05-30"
    if not snapshot_root.exists():
        pytest.skip("reference snapshot is not present")

    parquet_count = sum(1 for _ in snapshot_root.rglob("*.parquet"))

    monkeypatch.setattr(source_requests, "sha256_file", lambda _: "0" * 64)
    monkeypatch.setattr(
        source_requests,
        "_legacy_time_bounds",
        lambda _: {"column": None, "start": None, "end": None},
    )
    monkeypatch.setattr(source_requests, "_legacy_row_count", lambda _: None)

    manifest = source_requests.declare_source_requests(repo_root=repo_root, dry_run=True)

    assert parquet_count > 0
    assert manifest["legacy_file_count"] == parquet_count
    assert manifest["request_count"] == parquet_count
    assert manifest["unclassified_count"] == 0
    assert manifest["source_kind_counts"]["ibkr_cme_futures_historical_bars"] >= 2
    assert manifest["source_kind_counts"]["ibkr_us_equity_historical_bars"] >= 6
    assert manifest["source_kind_counts"]["kis_krx_intraday_bars"] >= 1
    assert manifest["source_kind_counts"]["lrs_krx_daily_export"] >= 1


def test_trading_stock_requirements_are_explicit_stock_family_allowlist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = (
        repo_root / "data" / "requirements" / "strategies" / "trading_stock" / "portfolio.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    requirements = payload["requirements"]

    assert payload["ownership_policy"] == "explicit_stock_family_allowlist_v1"
    approval_scope = payload["approval_scope"]
    assert approval_scope["live_intraday_symbol_count"] == 98
    assert approval_scope["live_intraday_requirement_count"] == 294
    assert approval_scope["daily_reference_context_symbol_count"] == 317
    assert approval_scope["daily_reference_context_requirement_count"] == 317
    assert approval_scope["declared_requirement_count"] == 611
    assert payload["archive_evidence_scope"]["legacy_archive_source_comparison_count"] == 611
    assert len(requirements) == 611
    assert {item["symbol"] for item in requirements}.isdisjoint({"QQQ", "GLD"})
    assert all(item["symbol"] != "*" for item in requirements)
    assert all(item["strategy_data_family"] == "trading_stock" for item in requirements)
    assert all(item["session_policy"] for item in requirements)
    assert all(str(item["use_rth"]).lower() in {"true", "false"} for item in requirements)


def test_source_requests_read_annotated_stock_universe_primary_exchanges() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    trading_root = MONOREPO_ROOT / "trading" / "ibkr_trader"
    if not trading_root.exists():
        pytest.skip("IBKR bot workspace is local-only and not available in CI")
    mapping = source_requests._load_trading_primary_exchange_map(repo_root)

    assert mapping["BRK B"] == "NYSE"
    assert mapping["A"] == "NYSE"
    assert mapping["AAPL"] == "NASDAQ"


def test_k_stock_requirements_are_explicit_olr_kalcb_universe_contract() -> None:
    repo_root = package_workspace("trading_assistant_data")
    requirements_path = (
        repo_root / "data" / "requirements" / "strategies" / "k_stock" / "portfolio.json"
    )
    universe_path = (
        MONOREPO_ROOT / "trading" / "k_stock_trader" / "config" / "olr_kalcb"
        / "olr_deployment_universe_103.yaml"
    )
    if not universe_path.exists():
        pytest.skip("K-stock bot workspace is local-only and not available in CI")
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    universe = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
    symbols = {str(symbol).zfill(6) for symbol in universe["symbols"]}
    requirements = payload["requirements"]

    assert payload["ownership_policy"] == "explicit_k_stock_olr_kalcb_universe_103_allowlist_v1"
    assert payload["symbol_count"] == 103
    assert payload["symbols_sha256"] == universe["symbols_sha256"]
    assert len(symbols) == 103
    assert len(requirements) == 516
    assert all(item["strategy_data_family"] == "k_stock_olr_kalcb" for item in requirements)
    assert all(item["symbol"] != "*" for item in requirements)

    by_role = {}
    for item in requirements:
        by_role.setdefault(item["data_role"], []).append(item)

    assert set(by_role) == {
        "daily_ohlcv",
        "daily_flow",
        "daily_foreign_flow",
        "daily_institutional_flow",
        "intraday_5m",
        "sector_map",
    }
    for role in (
        "daily_ohlcv",
        "daily_flow",
        "daily_foreign_flow",
        "daily_institutional_flow",
        "intraday_5m",
    ):
        assert {item["symbol"] for item in by_role[role]} == symbols
        assert len(by_role[role]) == 103
    assert {item["timeframe"] for item in by_role["intraday_5m"]} == {"5m"}
    assert by_role["sector_map"] == [
        {
            "source": "lrs",
            "market": "krx_equity",
            "symbol": "ALL",
            "timeframe": "static_or_daily",
            "strategy_data_family": "k_stock_olr_kalcb",
            "data_role": "sector_map",
            "data_root": "data/krx_daily_parquet",
            "source_table": "sector_map",
        }
    ]


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-05-01T00:05:00Z"], utc=True
            ),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10, 20],
        }
    ).to_parquet(path, index=False)
