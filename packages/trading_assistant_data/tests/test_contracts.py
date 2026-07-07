from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_assistant_data.bundle_builder import (
    build_bundle,
    export_filesystem,
    export_single_slice_coverage,
)
from trading_assistant_data.calendars.core import CalendarDefinition, expected_bars
from trading_assistant_data.calendars.cme import calendar_definition as cme_calendar
from trading_assistant_data.calendars.krx import calendar_definition as krx_calendar
from trading_assistant_data.calendars.krx import kis_intraday_calendar_definition
from trading_assistant_data.calendars.krx import kis_intraday_expected_bar_opens
from trading_assistant_data.calendars.us_equities import calendar_definition as us_equities_calendar
from trading_assistant_data.checksums import parquet_content_checksum
from trading_assistant_data.hygiene import clean_stale_cme_bid_ask_aliases
from trading_assistant_data.io import write_json
from trading_assistant_data.legacy_compare import compare_legacy_source_requests
from trading_assistant_data.manifests import (
    DataBundleManifest,
    DataBundleSlice,
    DataBundleStatus,
    MarketDataManifest,
    write_model,
)
from trading_assistant_data.normalization import normalize_krx_intraday_frames
import trading_assistant_data.normalization as normalization
from trading_assistant_data.reproduction import reproduce_data_bundle
from trading_assistant_data.source_authority import (
    ibkr_cme_nq_authority_contract,
    ibkr_us_equity_authority_contract,
    kis_krx_authority_contract,
    order_surface_errors,
)
from trading_assistant_data.sources.ibkr.cme_nq_read_only import (
    CmeNqRefreshRequest,
    DeterministicCmeNqProvider,
    IBKRCmeNqReadOnlyAdapter,
)
from trading_assistant_data.sources.ibkr.us_equity_read_only import (
    DeterministicUsEquityProvider,
    IBKRUsEquityReadOnlyAdapter,
    UsEquityRefreshRequest,
)
from trading_assistant_data.sources.hyperliquid.store import candle_open_from_ms, canonicalize_candles
from trading_assistant_data.sources.hyperliquid.sync import sync_hyperliquid
from trading_assistant_data.sources.kis.krx_read_only import (
    DeterministicKrxProvider,
    KISKrxReadOnlyAdapter,
    KisApiKrxProvider,
    KrxRefreshRequest,
)
from trading_assistant_data.sources.kis.read_only_client import KisReadOnlyClient
from trading_assistant_data.transforms.alignment import compare_derived_frame_alignment
from trading_assistant_data.validation import coverage_counts, detect_missing_ranges, validate_market_manifest
from tests.paths import MONOREPO_ROOT

FIXTURE_DATA_SHA = "a" * 40


def test_market_data_manifest_authoritative_happy_path() -> None:
    manifest = _manifest()

    report = validate_market_manifest(manifest)

    assert report.valid is True
    assert manifest.usable_for_authoritative_validation is True

    from trading_assistant.schemas.market_data_manifest import (
        MarketDataManifest as ConsumerManifest,
    )

    assert ConsumerManifest.model_validate(manifest.model_dump()).manifest_id == manifest.manifest_id


def test_market_data_manifest_blocks_missing_checksum() -> None:
    manifest = _manifest(checksum="", authoritative=False)

    report = validate_market_manifest(manifest)

    assert report.valid is False
    assert "checksum missing" in report.errors


def test_market_data_manifest_blocks_missing_calendar() -> None:
    manifest = _manifest(session_calendar="", authoritative=False)

    report = validate_market_manifest(manifest)

    assert report.valid is False
    assert "session_calendar missing" in report.errors


def test_data_bundle_manifest_authoritative_happy_path() -> None:
    bundle = _bundle(_manifest())

    assert bundle.status == DataBundleStatus.AUTHORITATIVE
    assert bundle.usable_for_authoritative_validation is True

    from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest as ConsumerBundle

    assert ConsumerBundle.model_validate(bundle.model_dump()).bundle_checksum == bundle.bundle_checksum


def test_committed_crypto_phased_optimizer_bundle_is_authoritative_and_addressable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = (
        repo_root
        / "data"
        / "bundles"
        / "monthly"
        / "2026-05"
        / "crypto_portfolio"
        / "phased_optimizer"
        / "data_bundle_manifest.json"
    )
    if not bundle_path.exists():
        pytest.skip("Crypto phased optimizer committed bundle fixture is not present")

    bundle = DataBundleManifest.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))

    assert bundle.status == DataBundleStatus.AUTHORITATIVE
    assert bundle.usable_for_authoritative_validation is True
    assert bundle.data_repo_path == "."
    assert bundle.bundle_checksum == "baad97bd7659f4208322ed383656192f40ef5945120dc283ad0c02063d8dc436"
    assert len(bundle.slice_manifests) == 18
    assert {item.symbol for item in bundle.slice_manifests} == {"BTC", "ETH", "SOL"}
    assert {item.timeframe for item in bundle.slice_manifests} == {
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
        "funding_1h",
    }
    assert (bundle_path.parent / "slice_index.json").exists()
    for item in bundle.slice_manifests:
        assert (repo_root / item.manifest_path).exists()


def test_crypto_phased_optimizer_requirements_match_strategy_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    requirements_path = (
        repo_root
        / "data"
        / "requirements"
        / "strategies"
        / "crypto_portfolio"
        / "phased_optimizer.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    requirements = payload["requirements"]

    candles = {
        item["timeframe"]
        for item in requirements
        if item.get("data_role") == "optimizer_candles"
    }
    funding = [item for item in requirements if item.get("data_role") == "funding"]

    assert candles == {"15m", "30m", "1h", "4h", "1d"}
    assert "1m" not in candles
    assert "5m" not in candles
    assert len(requirements) == 18
    assert len(funding) == 3
    assert {item["timeframe"] for item in funding} == {"funding_1h"}


def test_committed_crypto_phased_optimizer_reproduction_report_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = (
        repo_root
        / "data"
        / "bundles"
        / "monthly"
        / "2026-05"
        / "crypto_portfolio"
        / "phased_optimizer"
        / "data_bundle_manifest.json"
    )
    if not bundle_path.exists():
        pytest.skip("Crypto phased optimizer committed bundle fixture is not present")
    missing_canonical_paths = _missing_bundle_canonical_paths(repo_root, bundle_path)
    if missing_canonical_paths:
        pytest.skip(
            "Crypto phased optimizer canonical parquet data is local-only; "
            f"first missing path: {missing_canonical_paths[0]}"
        )

    report = reproduce_data_bundle(
        repo_root=repo_root,
        bundle_manifest_path=bundle_path,
        artifact_root=tmp_path / "data_reproduction",
    )

    assert report["ok"] is True
    assert Path(report["report_path"]).exists()
    assert report["bundle_checksum"] == report["recomputed_bundle_checksum"]
    assert report["slice_count"] == 18
    assert {item["symbol"] for item in report["slices"]} == {"BTC", "ETH", "SOL"}
    assert {item["timeframe"] for item in report["slices"]} == {
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
        "funding_1h",
    }
    assert all(check["passed"] for check in report["checks"])


def test_data_bundle_manifest_diagnostics_only_when_any_slice_is_not_authoritative() -> None:
    manifest = _manifest(authoritative=False, blocking_reasons=["gap"])
    bundle = _bundle(manifest, status=DataBundleStatus.DIAGNOSTICS_ONLY)

    assert bundle.usable_for_authoritative_validation is False
    assert bundle.diagnostics_only_reason


def test_bundle_checksum_changes_when_any_slice_checksum_changes() -> None:
    first = _bundle(_manifest(checksum="sha-a"))
    second = _bundle(_manifest(checksum="sha-b"))

    assert first.bundle_checksum != second.bundle_checksum


@pytest.fixture
def bundle_git_sha(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(
        "trading_assistant_data.bundle_builder.git_commit_exists",
        lambda _repo_root, commit_sha: commit_sha == FIXTURE_DATA_SHA,
    )
    return FIXTURE_DATA_SHA


def test_compatibility_export_matches_filesystem_adapter_layout(tmp_path: Path, bundle_git_sha: str) -> None:
    manifest_path = _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        slice_manifest_paths=[manifest_path],
    )

    export = export_filesystem(
        repo_root=tmp_path,
        run_month="2026-05",
        bundle_manifest_path=result.bundle_path,
    )

    expected = tmp_path / "data/export/filesystem/crypto_perp/BTC/1m/2026-05.parquet"
    assert str(expected) in export["exported"]
    assert expected.exists()


def test_single_slice_coverage_manifest_matches_monthly_default_path(tmp_path: Path, bundle_git_sha: str) -> None:
    manifest_path = _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        slice_manifest_paths=[manifest_path],
    )

    export = export_single_slice_coverage(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        bundle_manifest_path=result.bundle_path,
    )

    expected = tmp_path / "data/export/manifests/crypto_portfolio/portfolio/2026-05.coverage_manifest.json"
    assert export["path"] == str(expected)
    assert expected.exists()


def test_single_slice_coverage_manifest_preserves_authority_metadata(tmp_path: Path, bundle_git_sha: str) -> None:
    manifest_path = _write_sample_slice(tmp_path, symbol="ETH", timeframe="5m")
    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="eth_5m",
        slice_manifest_paths=[manifest_path],
    )

    export_single_slice_coverage(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="eth_5m",
        bundle_manifest_path=result.bundle_path,
    )
    payload = json.loads(
        (tmp_path / "data/export/manifests/crypto_portfolio/eth_5m/2026-05.coverage_manifest.json").read_text()
    )

    assert payload["checksum"]
    assert payload["source_version"] == bundle_git_sha
    assert payload["fee_model_version"] == "fees_v1"
    assert payload["slippage_model_version"] == "slippage_v1"
    assert payload["adjustment_policy"] == "crypto_raw_perp_policy_v1"
    assert payload["usable_for_authoritative_validation"] is True


def test_multi_slice_bundle_does_not_emit_aggregate_market_data_manifest(tmp_path: Path, bundle_git_sha: str) -> None:
    first = _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    second = _write_sample_slice(tmp_path, symbol="ETH", timeframe="1m")
    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        slice_manifest_paths=[first, second],
    )

    export = export_single_slice_coverage(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        bundle_manifest_path=result.bundle_path,
    )

    assert export["status"] == "skipped"
    assert not (tmp_path / "data/export/manifests/crypto_portfolio/portfolio/2026-05.coverage_manifest.json").exists()


def test_build_bundle_diagnostics_only_without_git_commit(tmp_path: Path) -> None:
    manifest_path = _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        slice_manifest_paths=[manifest_path],
    )

    assert result.bundle.status == DataBundleStatus.DIAGNOSTICS_ONLY
    assert result.bundle.data_repo_commit_sha == FIXTURE_DATA_SHA
    assert "commit is not available" in result.bundle.diagnostics_only_reason


def test_build_bundle_requires_explicit_slices_or_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --slice-manifest or --requirements-file"):
        build_bundle(
            repo_root=tmp_path,
            run_month="2026-05",
            bot_id="crypto_portfolio",
            strategy_id="portfolio",
        )


def test_build_bundle_uses_strategy_requirements_file(tmp_path: Path, bundle_git_sha: str) -> None:
    _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    _write_sample_slice(tmp_path, symbol="ETH", timeframe="1m")
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "BTC",
                    "timeframe": "1m",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        requirements_path=requirements_path,
    )

    assert result.bundle.status == DataBundleStatus.AUTHORITATIVE
    assert [item.symbol for item in result.bundle.slice_manifests] == ["BTC"]
    assert result.bundle.data_repo_commit_sha == bundle_git_sha


def test_build_bundle_fails_closed_when_concrete_requirement_is_missing(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "BTC",
                    "timeframe": "1m",
                },
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "ETH",
                    "timeframe": "1m",
                },
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        requirements_path=requirements_path,
    )

    assert result.bundle.data_repo_commit_sha == bundle_git_sha
    assert [item.symbol for item in result.bundle.slice_manifests] == ["BTC"]
    assert result.bundle.status == DataBundleStatus.DIAGNOSTICS_ONLY
    assert "missing required slices: hyperliquid/crypto_perp/ETH/1m" in (
        result.bundle.diagnostics_only_reason
    )


def test_trading_momentum_requirements_match_minimal_loader_contract() -> None:
    requirements_path = (
        Path(__file__).resolve().parents[1]
        / "data/requirements/strategies/trading_momentum/portfolio.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))

    requirements = {
        (item["source"], item["market"], item["symbol"], item["timeframe"])
        for item in payload["requirements"]
    }

    assert requirements == {
        ("ibkr", "cme_futures", "NQ", "5m"),
        ("ibkr", "cme_futures", "ES", "1d"),
    }


def test_trading_swing_requirements_match_default_etf_loader_contract() -> None:
    requirements_path = (
        Path(__file__).resolve().parents[1]
        / "data/requirements/strategies/trading_swing/portfolio.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))

    requirements = {
        (item["source"], item["market"], item["symbol"], item["timeframe"])
        for item in payload["requirements"]
    }

    assert requirements == {
        ("ibkr", "us_equity", "GLD", "15m"),
        ("ibkr", "us_equity", "GLD", "1h"),
        ("ibkr", "us_equity", "GLD", "1d"),
        ("ibkr", "us_equity", "QQQ", "15m"),
        ("ibkr", "us_equity", "QQQ", "1h"),
        ("ibkr", "us_equity", "QQQ", "1d"),
    }


def test_trading_stock_requirements_match_current_stock_strategy_contract() -> None:
    requirements_path = (
        Path(__file__).resolve().parents[1]
        / "data/requirements/strategies/trading_stock/portfolio.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))

    requirements = payload["requirements"]
    identities = {
        (item["source"], item["market"], item["symbol"], item["timeframe"])
        for item in requirements
    }

    assert payload["ownership_policy"] == "explicit_stock_family_allowlist_v1"
    approval_scope = payload["approval_scope"]
    assert approval_scope["live_universe_authority"].endswith(
        "live_universe.py::LIVE_STOCK_UNIVERSE_SYMBOLS"
    )
    assert approval_scope["live_intraday_symbol_count"] == 98
    assert approval_scope["live_intraday_requirement_count"] == 294
    assert approval_scope["daily_reference_context_symbol_count"] == 317
    assert approval_scope["daily_reference_context_requirement_count"] == 317
    assert approval_scope["declared_requirement_count"] == 611
    assert payload["archive_evidence_scope"]["legacy_archive_source_comparison_count"] == 611
    assert len(requirements) == 611
    assert all(symbol != "*" for _source, _market, symbol, _timeframe in identities)
    assert all(source == "ibkr" and market == "us_equity" for source, market, _symbol, _timeframe in identities)
    assert {item["timeframe"] for item in requirements} == {"1d", "5m", "30m"}
    assert {item["symbol"] for item in requirements}.isdisjoint({"QQQ", "GLD"})
    assert all(item["strategy_data_family"] == "trading_stock" for item in requirements)


def test_trading_stock_live_approval_scope_uses_98_symbol_lane_not_archive_count() -> None:
    stock_universe_path = (
        MONOREPO_ROOT / "bots" / "ibkr_trading" / "strategies" / "stock" / "live_universe.py"
    )
    if not stock_universe_path.exists():
        pytest.skip("IBKR bot workspace is local-only and not available in CI")
    text = stock_universe_path.read_text(encoding="utf-8")
    assert '"BRK B"' in text

    requirements_path = (
        Path(__file__).resolve().parents[1]
        / "data/requirements/strategies/trading_stock/portfolio.json"
    )
    payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    approval_scope = payload["approval_scope"]
    by_symbol: dict[str, set[str]] = {}
    for item in payload["requirements"]:
        by_symbol.setdefault(item["symbol"], set()).add(item["timeframe"])
    live_symbols = {
        symbol for symbol, timeframes in by_symbol.items() if timeframes == {"1d", "5m", "30m"}
    }
    live_requirements = [
        item for item in payload["requirements"] if item["symbol"] in live_symbols
    ]

    assert "BRK B" in live_symbols
    assert len(live_symbols) == approval_scope["live_intraday_symbol_count"] == 98
    assert len(live_requirements) == approval_scope["live_intraday_requirement_count"] == 294
    assert approval_scope["declared_requirement_count"] == 611
    assert approval_scope["declared_requirement_count"] != approval_scope["live_intraday_requirement_count"]


def _missing_bundle_canonical_paths(repo_root: Path, bundle_path: Path) -> list[Path]:
    slice_index_path = bundle_path.with_name("slice_index.json")
    if not slice_index_path.exists():
        return []
    slice_index = json.loads(slice_index_path.read_text(encoding="utf-8"))
    missing: list[Path] = []
    for item in slice_index.get("slices", []):
        if not isinstance(item, dict):
            continue
        for raw_path in item.get("canonical_paths", []):
            path = repo_root / str(raw_path)
            if not path.exists():
                missing.append(path)
    return missing


def test_kis_intraday_no_trade_authority_reuses_krx_symbol_dates(tmp_path: Path) -> None:
    calendar_dir = tmp_path / "data" / "calendars"
    calendar_dir.mkdir(parents=True)
    (calendar_dir / "krx_symbol_no_trade_dates.yaml").write_text(
        "\n".join(
            [
                "schema_version: krx_symbol_no_trade_dates_v1",
                "symbols:",
                '  "010120":',
                "    reason: lrs_zero_volume_flat_ohlcv_zero_flow_no_kis_intraday",
                "    dates:",
                '      - "2026-04-08"',
                '      - "2026-04-09"',
                '      - "2026-04-10"',
            ]
        ),
        encoding="utf-8",
    )

    dates, checksum = normalization._known_symbol_no_trade_dates(
        tmp_path,
        source="kis",
        market="krx_equity",
        symbol="010120",
        timeframe="5m",
    )

    assert checksum
    assert dates == {
        pd.Timestamp("2026-04-08"),
        pd.Timestamp("2026-04-09"),
        pd.Timestamp("2026-04-10"),
    }
    assert (
        normalization._known_symbol_no_trade_dates(
            tmp_path,
            source="kis",
            market="krx_equity",
            symbol="010120",
            timeframe="tick",
        )[0]
        == set()
    )
    assert normalization._missing_range_dates(
        {"start_ts": "2026-04-08T00:00:00Z", "end_ts": "2026-04-10T06:30:00Z"}
    ) == {
        pd.Timestamp("2026-04-08"),
        pd.Timestamp("2026-04-09"),
        pd.Timestamp("2026-04-10"),
    }


def test_archived_stock_normalizer_stamps_declared_legacy_cache_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(normalization, "git_commit_sha", lambda _repo_root: FIXTURE_DATA_SHA)
    raw_path = (
        tmp_path
        / "data/imported/reference_snapshot_2026-05-30/trading/backtests/stock/data/raw/DKNG_1d.parquet"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.25, 11.25],
            "volume": [1000, 1200],
        },
        index=pd.DatetimeIndex(
            ["2022-05-04T00:00:00Z", "2022-05-06T00:00:00Z"],
            name="time",
        ),
    )
    raw.to_parquet(raw_path, engine="pyarrow")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    rel_raw = str(raw_path.relative_to(tmp_path)).replace("\\", "/")

    source_request = {
        "schema_version": "source_request_v1",
        "request_id": "src_req_dkng",
        "legacy_family": "trading_stock",
        "legacy_path": rel_raw,
        "legacy_row_count": 2,
        "legacy_sha256": raw_sha,
        "source": "ibkr",
        "market": "us_equity",
        "source_kind": "ibkr_us_equity_historical_bars",
        "symbol": "DKNG",
        "timeframe": "1d",
        "source_endpoint": "ibkr://historical-data/STK/SMART/NASDAQ/DKNG/TRADES/1d/RTH",
        "download_request": {
            "provider": "ibkr",
            "request_type": "historical_bars",
            "sec_type": "STK",
            "symbol": "DKNG",
            "exchange": "SMART",
            "primary_exchange": "NASDAQ",
            "currency": "USD",
            "timeframe": "1d",
            "what_to_show": "TRADES",
            "use_rth": True,
            "start": "2022-05-04T00:00:00+00:00",
            "end": "2022-05-06T00:00:00+00:00",
            "contract_resolution_policy": "ibkr_contract_details_con_id_required_v1",
            "source_identity_authority": "trading_config_contracts_and_routing_v1",
        },
        "canonical_expectations": {
            "canonical_adjustment_policy": "us_equity_raw_adjustment_policy_v1",
            "corporate_action_policy": "split_dividend_policy_declared_per_bundle_v1",
            "raw_adjustment_policy": "ibkr_trades_unadjusted_raw_v1",
            "session_policy": "us_equity_rth_session_0930_1600_new_york_v1",
        },
    }
    write_json(
        tmp_path
        / "data/source_requests/reference_snapshot_2026-05-30/requests/ibkr_us_equity_historical_bars.json",
        {"schema_version": "source_request_manifest_v1", "requests": [source_request]},
    )
    write_json(
        tmp_path / "data/requirements/strategies/trading_stock/portfolio.json",
        {
            "schema_version": "strategy_data_requirements_v1",
            "ownership_policy": "explicit_stock_family_allowlist_v1",
            "requirements": [
                {
                    "source": "ibkr",
                    "market": "us_equity",
                    "symbol": "DKNG",
                    "timeframe": "1d",
                    "strategy_data_family": "trading_stock",
                    "session_policy": "us_equity_rth_session_0930_1600_new_york_v1",
                    "use_rth": "true",
                    "primary_exchange": "NASDAQ",
                }
            ],
        },
    )

    report = normalization.normalize_us_equity_stock_raw(tmp_path, snapshot="2026-05-30")

    assert report["slice_manifest_count"] == 1
    manifest = next((tmp_path / "data/manifests/slices/ibkr/us_equity/DKNG/1d").glob("*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["usable_for_authoritative_validation"] is True
    assert payload["missing_ranges"] == []
    assert payload["expected_bars"] == payload["actual_bars"] == 2
    lineage = payload["lineage"]
    assert lineage["authority_status"] == "archived_ibkr_stock_updater_parquet_exact_declared_request"
    assert lineage["source_conid_coverage"] == "archived_request_identity_only_no_row_conids"
    assert lineage["legacy_calendar_gap_policy"] == "preserve_exact_archived_stock_updater_cache_v1"
    assert "2022-05-05" in lineage["archived_cache_missing_ranges_json"]


def test_build_bundle_selects_only_indexed_manifests_for_requirements(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    stale_manifest = _manifest(symbol="ETH", timeframe="1m")
    stale_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp/ETH/1m"
        / "20260501T000000Z_20260501T000100Z.market_data_manifest.json"
    )
    write_model(stale_path, stale_manifest)
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "*",
                    "timeframe": "1m",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        requirements_path=requirements_path,
    )

    assert [item.symbol for item in result.bundle.slice_manifests] == ["BTC"]
    with pytest.raises(ValueError, match="not present in slice_index"):
        build_bundle(
            repo_root=tmp_path,
            run_month="2026-05",
            bot_id="crypto_portfolio",
            strategy_id="stale",
            slice_manifest_paths=[stale_path],
        )


def test_build_bundle_requirements_respect_family_ownership(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    wrong_family = _ibkr_us_equity_manifest(
        symbol="QQQ",
        timeframe="1h",
        strategy_data_family="trading_swing",
        checksum="swing-owned",
    )
    stock_family = _ibkr_us_equity_manifest(
        symbol="QQQ",
        timeframe="1h",
        strategy_data_family="trading_stock",
        checksum="stock-owned",
    )
    wrong_path = (
        tmp_path
        / "data/manifests/slices/ibkr/us_equity/QQQ/1h/"
        / "20260501T000000Z_20260501T000100Z.swing.json"
    )
    stock_path = (
        tmp_path
        / "data/manifests/slices/ibkr/us_equity/QQQ/1h/"
        / "20260501T000000Z_20260501T000100Z.stock.json"
    )
    write_model(wrong_path, wrong_family)
    write_model(stock_path, stock_family)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                _slice_index_entry(tmp_path, wrong_path, wrong_family),
                _slice_index_entry(tmp_path, stock_path, stock_family),
            ],
        },
    )
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "ibkr",
                    "market": "us_equity",
                    "symbol": "QQQ",
                    "timeframe": "1h",
                    "strategy_data_family": "trading_stock",
                    "session_policy": "us_equity_extended_session_0400_2000_new_york_v1",
                    "use_rth": "false",
                    "primary_exchange": "NASDAQ",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="trading_stock_family",
        strategy_id="stock_owned_reference",
        requirements_path=requirements_path,
        dry_run=True,
    )

    assert result.bundle.data_repo_commit_sha == bundle_git_sha
    assert [item.manifest_id for item in result.bundle.slice_manifests] == [
        stock_family.manifest_id
    ]
    assert result.bundle.status == DataBundleStatus.AUTHORITATIVE


def test_build_bundle_allows_archived_stock_cache_requirement_without_month_overlap(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    manifest = _ibkr_us_equity_manifest(
        symbol="HOLX",
        timeframe="1d",
        strategy_data_family="trading_stock",
        checksum="holx-archived",
    ).model_copy(
        update={
            "start_ts": datetime(2024, 3, 21, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 3, 20, tzinfo=timezone.utc),
            "lineage": {
                **_ibkr_us_equity_manifest(
                    symbol="HOLX",
                    timeframe="1d",
                    strategy_data_family="trading_stock",
                    checksum="holx-archived",
                ).lineage,
                "authority_status": "archived_ibkr_stock_updater_parquet_exact_declared_request",
                "archive_import_policy": "source_owned_trading_stock_ibkr_updater_parquet_v1",
                "session_policy": "us_equity_rth_session_0930_1600_new_york_v1",
                "use_rth": "true",
                "source_conid_coverage": "archived_request_identity_only_no_row_conids",
            },
        }
    )
    path = (
        tmp_path
        / "data/manifests/slices/ibkr/us_equity/HOLX/1d/"
        / "20240321T000000Z_20260320T000000Z.market_data_manifest.json"
    )
    write_model(path, manifest)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [_slice_index_entry(tmp_path, path, manifest)],
        },
    )
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "ibkr",
                    "market": "us_equity",
                    "symbol": "HOLX",
                    "timeframe": "1d",
                    "strategy_data_family": "trading_stock",
                    "session_policy": "us_equity_rth_session_0930_1600_new_york_v1",
                    "use_rth": "true",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="trading_stock_family",
        strategy_id="portfolio",
        requirements_path=requirements_path,
        dry_run=True,
    )

    assert [item.symbol for item in result.bundle.slice_manifests] == ["HOLX"]
    assert result.bundle.status == DataBundleStatus.AUTHORITATIVE


def test_legacy_compare_prefers_exact_source_request_manifest(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy/MSFT_5m.parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
        },
        index=pd.DatetimeIndex(["2026-05-01T13:30:00Z"], name="time"),
    ).to_parquet(legacy_path, engine="pyarrow")

    exact_canonical = tmp_path / "canonical/exact.parquet"
    wrong_canonical = tmp_path / "canonical/wrong.parquet"
    exact_canonical.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-05-01T13:30:00Z"], utc=True),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10],
        }
    ).to_parquet(exact_canonical, engine="pyarrow", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-05-01T13:30:00Z"], utc=True),
            "open": [200.0],
            "high": [201.0],
            "low": [199.0],
            "close": [200.5],
            "volume": [20],
        }
    ).to_parquet(wrong_canonical, engine="pyarrow", index=False)

    exact = _ibkr_us_equity_manifest(
        symbol="MSFT",
        timeframe="5m",
        strategy_data_family="trading_stock",
        checksum="exact",
    )
    exact = exact.model_copy(
        update={
            "start_ts": datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc),
            "lineage": {
                **exact.lineage,
                "source_request_id": "src_req_msft_5m",
                "authority_status": "archived_ibkr_stock_updater_parquet_exact_declared_request",
                "archive_import_policy": "source_owned_trading_stock_ibkr_updater_parquet_v1",
            }
        }
    )
    wrong = _ibkr_us_equity_manifest(
        symbol="MSFT",
        timeframe="5m",
        strategy_data_family="trading_stock",
        checksum="wrong",
    ).model_copy(
        update={
            "start_ts": datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc),
        }
    )
    exact_path = (
        tmp_path
        / "data/manifests/slices/ibkr/us_equity/MSFT/5m/20260501T133000Z_exact.json"
    )
    wrong_path = (
        tmp_path
        / "data/manifests/slices/ibkr/us_equity/MSFT/5m/20260501T133000Z_wrong.json"
    )
    write_model(exact_path, exact)
    write_model(wrong_path, wrong)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    **_slice_index_entry(tmp_path, exact_path, exact),
                    "canonical_paths": [str(exact_canonical.relative_to(tmp_path)).replace("\\", "/")],
                },
                {
                    **_slice_index_entry(tmp_path, wrong_path, wrong),
                    "canonical_paths": [str(wrong_canonical.relative_to(tmp_path)).replace("\\", "/")],
                },
            ],
        },
    )
    source_requests = tmp_path / "source_requests.json"
    write_json(
        source_requests,
        {
            "schema_version": "source_request_manifest_v1",
            "requests": [
                {
                    "request_id": "src_req_msft_5m",
                    "legacy_family": "trading_stock",
                    "legacy_path": str(legacy_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": "ibkr",
                    "market": "us_equity",
                    "source_kind": "ibkr_us_equity_historical_bars",
                    "symbol": "MSFT",
                    "timeframe": "5m",
                    "download_request": {"use_rth": False},
                }
            ],
        },
    )

    report = compare_legacy_source_requests(
        repo_root=tmp_path,
        source_request_manifest=source_requests,
        families=["trading_stock"],
        artifact_root=tmp_path / "artifacts",
    )

    assert report["ok"] is True
    assert report["comparisons"][0]["matched_manifest_ids"] == [exact.manifest_id]


def test_legacy_compare_lrs_flow_uses_latest_exchange_date_and_family_slice(tmp_path: Path) -> None:
    old_legacy_path = tmp_path / "legacy/000100_flow_old.parquet"
    latest_legacy_path = tmp_path / "legacy/000100_flow_latest.parquet"
    old_legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ticker": ["000100"], "date": ["2026-04-30"], "foreign_net": [10.0]}
    ).to_parquet(old_legacy_path, engine="pyarrow", index=False)
    pd.DataFrame(
        {"ticker": ["000100"], "date": ["2026-05-01"], "foreign_net": [20.0]}
    ).to_parquet(latest_legacy_path, engine="pyarrow", index=False)

    right_canonical = tmp_path / "canonical/right_flow.parquet"
    wrong_canonical = tmp_path / "canonical/wrong_probe.parquet"
    right_canonical.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-04-30T15:00:00Z"], utc=True),
            "timestamp_exchange": ["2026-05-01 00:00:00+09:00"],
            "symbol": ["000100"],
            "market": ["krx_equity"],
            "source": ["lrs"],
            "timeframe": ["1d_daily_foreign_flow"],
            "kind": ["daily_foreign_flow"],
            "foreign_net": [20.0],
        }
    ).to_parquet(right_canonical, engine="pyarrow", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-04-30T15:00:00Z"], utc=True),
            "timestamp_exchange": ["2026-05-01 00:00:00+09:00"],
            "foreign_net": [999.0],
        }
    ).to_parquet(wrong_canonical, engine="pyarrow", index=False)

    right = _manifest(
        checksum=parquet_content_checksum(right_canonical),
        session_calendar="krx_equities_daily_calendar_v1",
        symbol="000100",
        timeframe="1d_daily_foreign_flow",
        adjustment_policy="krx_equity_raw_adjustment_policy_v1",
        lineage={"strategy_data_family": "k_stock_olr_kalcb"},
    ).model_copy(
        update={
            "source": "lrs",
            "market": "krx_equity",
            "start_ts": datetime(2026, 4, 30, 15, 0, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 4, 30, 15, 0, tzinfo=timezone.utc),
            "expected_bars": 1,
            "actual_bars": 1,
        }
    )
    wrong = right.model_copy(
        update={
            "checksum": parquet_content_checksum(wrong_canonical),
            "manifest_id": "wrong-probe-manifest",
        }
    )
    right_path = (
        tmp_path
        / "data/manifests/slices/lrs/krx_equity/000100/1d_daily_foreign_flow/right.json"
    )
    wrong_path = (
        tmp_path
        / "data/manifests/slices/lrs/krx_equity/000100/1d_daily_foreign_flow/wrong.json"
    )
    write_model(right_path, right)
    write_model(wrong_path, wrong)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    **_slice_index_entry(tmp_path, right_path, right),
                    "canonical_paths": [str(right_canonical.relative_to(tmp_path)).replace("\\", "/")],
                },
                {
                    **{
                        key: value
                        for key, value in _slice_index_entry(tmp_path, wrong_path, wrong).items()
                        if key != "strategy_data_family"
                    },
                    "canonical_paths": [str(wrong_canonical.relative_to(tmp_path)).replace("\\", "/")],
                },
            ],
        },
    )
    source_requests = tmp_path / "source_requests.json"
    write_json(
        source_requests,
        {
            "schema_version": "source_request_manifest_v1",
            "requests": [
                {
                    "request_id": "src_req_old",
                    "legacy_family": "k_stock_krx_lrs",
                    "legacy_path": str(old_legacy_path.relative_to(tmp_path)).replace("\\", "/"),
                    "legacy_time_bounds": {"end": "2026-04-30T00:00:00+00:00"},
                    "source": "lrs",
                    "market": "krx_equity",
                    "source_kind": "lrs_krx_flow_export",
                    "data_kind": "daily_foreign_flow",
                    "symbol": "000100",
                    "timeframe": "1d_flow",
                },
                {
                    "request_id": "src_req_latest",
                    "legacy_family": "k_stock_krx_lrs",
                    "legacy_path": str(latest_legacy_path.relative_to(tmp_path)).replace("\\", "/"),
                    "legacy_time_bounds": {"end": "2026-05-01T00:00:00+00:00"},
                    "source": "lrs",
                    "market": "krx_equity",
                    "source_kind": "lrs_krx_flow_export",
                    "data_kind": "daily_foreign_flow",
                    "symbol": "000100",
                    "timeframe": "1d_flow",
                },
            ],
        },
    )

    report = compare_legacy_source_requests(
        repo_root=tmp_path,
        source_request_manifest=source_requests,
        families=["k_stock_krx_lrs"],
        latest_only=True,
        artifact_root=tmp_path / "artifacts",
    )

    assert report["ok"] is True
    assert report["request_count"] == 1
    assert report["comparisons"][0]["request_id"] == "src_req_latest"
    assert report["comparisons"][0]["matched_manifest_ids"] == [right.manifest_id]


def test_legacy_compare_accepts_cme_physical_panama_retention_overlap(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy/NQ_5m.parquet"
    legacy_path.parent.mkdir(parents=True)
    legacy = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2023-12-29T23:55:00Z",
                    "2024-03-11T00:00:00Z",
                    "2024-03-11T00:05:00Z",
                ]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.25, 101.25, 102.25],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    legacy.to_parquet(legacy_path, engine="pyarrow", index=False)

    canonical_path = (
        tmp_path
        / "data/canonical/bars/market=cme_futures/source=ibkr/kind=panama_trades"
        / "symbol=NQ/timeframe=5m/part.parquet"
    )
    canonical_path.parent.mkdir(parents=True)
    canonical = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2024-03-11T00:00:00Z", "2024-03-11T00:05:00Z"]),
            "timestamp_exchange": ["2024-03-10 19:00:00-05:00", "2024-03-10 19:05:00-05:00"],
            "symbol": ["NQ", "NQ"],
            "market": ["cme_futures", "cme_futures"],
            "source": ["ibkr", "ibkr"],
            "timeframe": ["5m", "5m"],
            "kind": ["panama_trades", "panama_trades"],
            "open": [201.0, 202.0],
            "high": [201.5, 202.5],
            "low": [200.5, 201.5],
            "close": [201.25, 202.25],
            "volume": [20.0, 30.0],
        }
    )
    canonical.to_parquet(canonical_path, engine="pyarrow", index=False)
    manifest = _manifest(
        checksum=parquet_content_checksum(canonical_path),
        session_calendar="cme_equity_index_futures_eth_v1",
        symbol="NQ",
        timeframe="5m",
        adjustment_policy="cme_futures_panama_v1",
        lineage={
            "source_request_id": "src_req_nq_5m",
            "source_contract_coverage": "all_rows",
            "source_conid_coverage": "all_rows",
            "roll_policy": "quarterly_index_future_roll_four_calendar_days_before_third_friday_v1",
            "contract_chain_checksum": "chain-sha",
            "continuous_construction_checksum": "panama-sha",
        },
    ).model_copy(
        update={
            "source": "ibkr",
            "market": "cme_futures",
            "start_ts": datetime(2024, 3, 11, 0, 0, tzinfo=timezone.utc),
            "end_ts": datetime(2024, 3, 11, 0, 5, tzinfo=timezone.utc),
        }
    )
    manifest_path = (
        tmp_path
        / "data/manifests/slices/ibkr/cme_futures/NQ/5m"
        / "20240311T000000Z_20240311T000500Z.market_data_manifest.json"
    )
    write_model(manifest_path, manifest)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    **_slice_index_entry(tmp_path, manifest_path, manifest),
                    "canonical_paths": [str(canonical_path.relative_to(tmp_path)).replace("\\", "/")],
                }
            ],
        },
    )
    source_requests = tmp_path / "source_requests.json"
    write_json(
        source_requests,
        {
            "schema_version": "source_request_manifest_v1",
            "requests": [
                {
                    "request_id": "src_req_nq_5m",
                    "legacy_family": "trading_momentum",
                    "legacy_path": str(legacy_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": "ibkr",
                    "market": "cme_futures",
                    "source_kind": "ibkr_cme_futures_historical_bars",
                    "symbol": "NQ",
                    "timeframe": "5m",
                    "download_request": {
                        "continuous_contract_policy": "ibkr_physical_contract_chain_panama_v1"
                    },
                }
            ],
        },
    )

    report = compare_legacy_source_requests(
        repo_root=tmp_path,
        source_request_manifest=source_requests,
        families=["trading_momentum"],
        artifact_root=tmp_path / "artifacts",
    )

    assert report["ok"] is True
    accepted = report["comparisons"][0]["accepted_difference"]
    assert accepted["accepted"] is True
    assert accepted["reason"] == "cme_physical_chain_panama_retention_cutoff_accepted"
    assert accepted["retention_covered_subset"] is True


def test_build_bundle_ignores_contained_duplicate_slices_for_requirements(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    broad = _manifest(symbol="BTC", timeframe="1m").model_copy(
        update={
            "start_ts": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc),
            "expected_bars": 44_640,
            "actual_bars": 44_640,
        }
    )
    contained = _manifest(symbol="BTC", timeframe="1m", checksum="contained").model_copy(
        update={
            "start_ts": datetime(2026, 5, 7, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 12, 23, 59, tzinfo=timezone.utc),
            "expected_bars": 8_640,
            "actual_bars": 8_640,
        }
    )
    broad_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp/BTC/1m"
        / "20260501T000000Z_20260531T235900Z.market_data_manifest.json"
    )
    contained_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp/BTC/1m"
        / "20260507T000000Z_20260512T235900Z.market_data_manifest.json"
    )
    write_model(broad_path, broad)
    write_model(contained_path, contained)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    "manifest_id": broad.manifest_id,
                    "manifest_path": str(broad_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": broad.source,
                    "market": broad.market,
                    "symbol": broad.symbol,
                    "timeframe": broad.timeframe,
                    "checksum": broad.checksum,
                    "canonical_paths": [],
                },
                {
                    "manifest_id": contained.manifest_id,
                    "manifest_path": str(contained_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": contained.source,
                    "market": contained.market,
                    "symbol": contained.symbol,
                    "timeframe": contained.timeframe,
                    "checksum": contained.checksum,
                    "canonical_paths": [],
                },
            ],
        },
    )
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "BTC",
                    "timeframe": "1m",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        requirements_path=requirements_path,
        dry_run=True,
    )

    assert result.bundle.data_repo_commit_sha == bundle_git_sha
    assert [item.manifest_id for item in result.bundle.slice_manifests] == [broad.manifest_id]


def test_build_bundle_prefers_authoritative_slice_over_broader_diagnostic_duplicate(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    diagnostic = _manifest(
        symbol="BTC",
        timeframe="1m",
        checksum="diagnostic",
        authoritative=False,
        blocking_reasons=["source-aware missing ranges present"],
    ).model_copy(
        update={
            "start_ts": datetime(2026, 5, 1, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 31, 23, 59, tzinfo=timezone.utc),
            "expected_bars": 44_640,
            "actual_bars": 44_000,
        }
    )
    authoritative = _manifest(symbol="BTC", timeframe="1m", checksum="authoritative").model_copy(
        update={
            "start_ts": datetime(2026, 5, 7, tzinfo=timezone.utc),
            "end_ts": datetime(2026, 5, 12, 23, 59, tzinfo=timezone.utc),
            "expected_bars": 8_640,
            "actual_bars": 8_640,
        }
    )
    diagnostic_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp/BTC/1m"
        / "20260501T000000Z_20260531T235900Z.market_data_manifest.json"
    )
    authoritative_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp/BTC/1m"
        / "20260507T000000Z_20260512T235900Z.market_data_manifest.json"
    )
    write_model(diagnostic_path, diagnostic)
    write_model(authoritative_path, authoritative)
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    "manifest_id": diagnostic.manifest_id,
                    "manifest_path": str(diagnostic_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": diagnostic.source,
                    "market": diagnostic.market,
                    "symbol": diagnostic.symbol,
                    "timeframe": diagnostic.timeframe,
                    "checksum": diagnostic.checksum,
                    "canonical_paths": [],
                },
                {
                    "manifest_id": authoritative.manifest_id,
                    "manifest_path": str(authoritative_path.relative_to(tmp_path)).replace("\\", "/"),
                    "source": authoritative.source,
                    "market": authoritative.market,
                    "symbol": authoritative.symbol,
                    "timeframe": authoritative.timeframe,
                    "checksum": authoritative.checksum,
                    "canonical_paths": [],
                },
            ],
        },
    )
    requirements_path = tmp_path / "requirements.json"
    write_json(
        requirements_path,
        {
            "schema_version": "strategy_data_requirements_v1",
            "requirements": [
                {
                    "source": "hyperliquid",
                    "market": "crypto_perp",
                    "symbol": "BTC",
                    "timeframe": "1m",
                }
            ],
        },
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="btc_1m",
        requirements_path=requirements_path,
        dry_run=True,
    )

    assert result.bundle.data_repo_commit_sha == bundle_git_sha
    assert [item.manifest_id for item in result.bundle.slice_manifests] == [
        authoritative.manifest_id
    ]
    assert result.bundle.status == DataBundleStatus.AUTHORITATIVE


def test_build_bundle_diagnostics_only_when_policy_versions_mixed(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    first = _write_sample_slice(tmp_path, symbol="BTC", timeframe="1m")
    second = _write_sample_slice(
        tmp_path,
        symbol="ETH",
        timeframe="1m",
        adjustment_policy="crypto_raw_perp_policy_v2",
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="crypto_portfolio",
        strategy_id="portfolio",
        slice_manifest_paths=[first, second],
    )

    assert result.bundle.status == DataBundleStatus.DIAGNOSTICS_ONLY
    assert "adjustment_policy mismatch" in result.bundle.diagnostics_only_reason


def test_build_bundle_allows_mixed_price_and_flow_adjustment_policies(
    tmp_path: Path,
    bundle_git_sha: str,
) -> None:
    price = _write_sample_slice(
        tmp_path,
        symbol="000100",
        timeframe="1d",
        adjustment_policy="krx_split_adjusted_policy_v1",
    )
    flow = _write_sample_slice(
        tmp_path,
        symbol="000100",
        timeframe="1d_daily_flow",
        adjustment_policy="krx_flow_panel_policy_v1",
    )

    result = build_bundle(
        repo_root=tmp_path,
        run_month="2026-05",
        bot_id="k_stock_olr_kalcb",
        strategy_id="portfolio",
        slice_manifest_paths=[price, flow],
    )

    assert result.bundle.status == DataBundleStatus.AUTHORITATIVE
    assert result.bundle.adjustment_policy == (
        "mixed_adjustment_policy:krx_flow_panel_policy_v1,krx_split_adjusted_policy_v1"
    )


def test_data_bundle_manifest_can_be_passed_directly_to_monthly_run_manifest(tmp_path: Path) -> None:
    bundle_path = tmp_path / "data_bundle_manifest.json"
    bundle = _bundle(_manifest())
    write_model(bundle_path, bundle)

    from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest

    run = MonthlyRunManifest(
        run_id="monthly-crypto-portfolio-2026-05",
        run_month="2026-05",
        bot_id="crypto",
        strategy_id="portfolio",
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 31),
        market_data_manifest_path=str(bundle_path),
        data_bundle_manifest_path=str(bundle_path),
        data_bundle_checksum=bundle.bundle_checksum,
        telemetry_manifest_path=str(tmp_path / "telemetry.json"),
        artifact_root=str(tmp_path / "artifacts"),
    )

    assert run.data_bundle_manifest_path == str(bundle_path)
    assert run.data_bundle_checksum == bundle.bundle_checksum


def test_crypto_ts_ms_converts_to_utc_open_time() -> None:
    ts = candle_open_from_ms(1776206400000)

    assert ts.tzinfo is not None
    assert ts == pd.Timestamp("2026-04-14T22:40:00Z")


def test_hyperliquid_live_candle_columns_are_canonicalized() -> None:
    out = canonicalize_candles(
        pd.DataFrame(
            [
                {
                    "t": 1776206400000,
                    "o": "1.0",
                    "h": "2.0",
                    "l": "0.5",
                    "c": "1.5",
                    "v": "10.0",
                }
            ]
        ),
        symbol="btc",
        interval="1m",
        source_file="fixture.json",
    )

    assert out.iloc[0]["symbol"] == "BTC"
    assert out.iloc[0]["timestamp_utc"] == pd.Timestamp("2026-04-14T22:40:00Z")
    assert float(out.iloc[0]["close"]) == 1.5


def test_hyperliquid_sync_merges_overlap_and_updates_manifest_index(tmp_path: Path) -> None:
    class FakeHyperliquid:
        def candles(self, symbol: str, interval: str, *, start: datetime, end: datetime) -> list[dict]:
            assert symbol == "BTC"
            assert interval == "1m"
            assert start == datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
            return [
                {"t": 1777593660000, "o": "20", "h": "30", "l": "10", "c": "22", "v": "200"},
                {"t": 1777593720000, "o": "30", "h": "40", "l": "20", "c": "33", "v": "300"},
            ]

        def funding(self, symbol: str, *, start: datetime, end: datetime) -> list[dict]:
            raise AssertionError("funding should not be called")

    existing = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T00:01:00Z"]),
            "timestamp_exchange": ["2026-05-01 00:00:00+00:00", "2026-05-01 00:01:00+00:00"],
            "symbol": ["BTC", "BTC"],
            "market": ["crypto_perp", "crypto_perp"],
            "source": ["hyperliquid", "hyperliquid"],
            "timeframe": ["1m", "1m"],
            "kind": ["trades", "trades"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "source_file": ["seed", "seed"],
            "source_ts_ms": [1777593600000, 1777593660000],
            "source_row_hash": ["a", "b"],
        }
    )
    canonical_path = (
        tmp_path
        / "data/canonical/bars/market=crypto_perp/source=hyperliquid/kind=trades"
        / "symbol=BTC/timeframe=1m/year=2026/month=05/part.parquet"
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    existing.to_parquet(canonical_path, engine="pyarrow", index=False)

    result = sync_hyperliquid(
        repo_root=tmp_path,
        symbols=["BTC"],
        intervals=["1m"],
        end=datetime(2026, 5, 1, 0, 2, tzinfo=timezone.utc),
        latest=True,
        overlap_bars=1,
        downloader=FakeHyperliquid(),
    )

    merged = pd.read_parquet(canonical_path)
    assert result["status"] == "complete"
    assert result["slice_manifest_count"] == 1
    assert len(merged) == 3
    assert float(merged.loc[merged["timestamp_utc"] == pd.Timestamp("2026-05-01T00:01:00Z"), "close"].iloc[0]) == 2.0
    assert (tmp_path / "data/manifests/slices/slice_index.json").exists()
    assert list((tmp_path / "data/raw/hyperliquid/candles/symbol=BTC/interval=1m").glob("*.json"))


def test_hyperliquid_sync_pages_long_source_windows(tmp_path: Path) -> None:
    class FakeHyperliquid:
        def __init__(self) -> None:
            self.calls: list[tuple[datetime, datetime]] = []

        def candles(self, symbol: str, interval: str, *, start: datetime, end: datetime) -> list[dict]:
            self.calls.append((start, end))
            timestamps = pd.date_range(start=start, end=end, freq="1min", tz="UTC")
            return [
                {
                    "t": int(timestamp.timestamp() * 1000),
                    "o": "1",
                    "h": "2",
                    "l": "0.5",
                    "c": "1.5",
                    "v": "10",
                }
                for timestamp in timestamps
            ]

        def funding(self, symbol: str, *, start: datetime, end: datetime) -> list[dict]:
            raise AssertionError("funding should not be called")

    downloader = FakeHyperliquid()

    result = sync_hyperliquid(
        repo_root=tmp_path,
        symbols=["BTC"],
        intervals=["1m"],
        start=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 2, 15, 59, tzinfo=timezone.utc),
        downloader=downloader,
    )

    canonical_path = (
        tmp_path
        / "data/canonical/bars/market=crypto_perp/source=hyperliquid/kind=trades"
        / "symbol=BTC/timeframe=1m/year=2026/month=05/part.parquet"
    )
    frame = pd.read_parquet(canonical_path)
    assert result["status"] == "complete"
    assert len(downloader.calls) == 3
    assert len(frame) == 2_400
    assert frame["timestamp_utc"].is_unique


def test_hyperliquid_sync_pages_funding_source_windows_at_api_cap(tmp_path: Path) -> None:
    class FakeHyperliquid:
        def __init__(self) -> None:
            self.funding_calls: list[tuple[datetime, datetime]] = []

        def candles(self, symbol: str, interval: str, *, start: datetime, end: datetime) -> list[dict]:
            return []

        def funding(self, symbol: str, *, start: datetime, end: datetime) -> list[dict]:
            self.funding_calls.append((start, end))
            timestamps = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
            return [
                {
                    "time": int(timestamp.timestamp() * 1000) + 45,
                    "fundingRate": "0.00001",
                }
                for timestamp in timestamps
            ]

    downloader = FakeHyperliquid()

    result = sync_hyperliquid(
        repo_root=tmp_path,
        symbols=["BTC"],
        intervals=["1d"],
        start=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 31, 23, 0, tzinfo=timezone.utc),
        funding=True,
        downloader=downloader,
    )

    canonical_path = (
        tmp_path
        / "data/canonical/funding/market=crypto_perp/source=hyperliquid"
        / "symbol=BTC/year=2026/month=05/part.parquet"
    )
    frame = pd.read_parquet(canonical_path)
    manifest_path = next(
        (
            tmp_path
            / "data/manifests/slices/hyperliquid/crypto_perp/BTC/funding_1h"
        ).glob("*.json")
    )
    manifest = MarketDataManifest.model_validate(json.loads(manifest_path.read_text()))

    assert result["status"] == "complete"
    assert len(downloader.funding_calls) == 2
    assert len(frame) == 744
    assert frame["timestamp_utc"].is_unique
    assert str(frame["timestamp_utc"].iloc[0]).endswith("00:00")
    assert manifest.timeframe == "funding_1h"
    assert manifest.actual_bars == 744
    assert manifest.expected_bars == 744
    assert manifest.missing_ranges == []
    assert manifest.blocking_reasons == ["source_version is not a data repo commit SHA"]


def test_clean_stale_cme_bid_ask_aliases_removes_only_ambiguous_1m_paths(tmp_path: Path) -> None:
    stale_dir = (
        tmp_path
        / "data/canonical/bars/market=cme_futures/source=ibkr/kind=bid_ask"
        / "symbol=NQ/timeframe=1m/year=2026/month=05"
    )
    good_dir = (
        tmp_path
        / "data/canonical/bars/market=cme_futures/source=ibkr/kind=bid_ask"
        / "symbol=NQ/timeframe=1m_bid_ask/year=2026/month=05"
    )
    stale_dir.mkdir(parents=True)
    good_dir.mkdir(parents=True)
    (stale_dir / "part.parquet").write_text("stale", encoding="utf-8")
    (good_dir / "part.parquet").write_text("good", encoding="utf-8")
    stale_manifest = tmp_path / "data/manifests/slices/ibkr/cme_futures/NQ/1m/stale.json"
    good_manifest = tmp_path / "data/manifests/slices/ibkr/cme_futures/NQ/1m_bid_ask/good.json"
    stale_manifest.parent.mkdir(parents=True)
    good_manifest.parent.mkdir(parents=True)
    stale_manifest.write_text("{}", encoding="utf-8")
    good_manifest.write_text("{}", encoding="utf-8")
    write_json(
        tmp_path / "data/manifests/slices/slice_index.json",
        {
            "schema_version": "slice_index_v1",
            "slices": [
                {
                    "manifest_id": "stale",
                    "manifest_path": str(stale_manifest.relative_to(tmp_path)).replace("\\", "/"),
                    "source": "ibkr",
                    "market": "cme_futures",
                    "symbol": "NQ",
                    "timeframe": "1m",
                    "canonical_paths": [
                        str((stale_dir / "part.parquet").relative_to(tmp_path)).replace("\\", "/")
                    ],
                },
                {
                    "manifest_id": "good",
                    "manifest_path": str(good_manifest.relative_to(tmp_path)).replace("\\", "/"),
                    "source": "ibkr",
                    "market": "cme_futures",
                    "symbol": "NQ",
                    "timeframe": "1m_bid_ask",
                    "canonical_paths": [
                        str((good_dir / "part.parquet").relative_to(tmp_path)).replace("\\", "/")
                    ],
                },
            ],
        },
    )

    result = clean_stale_cme_bid_ask_aliases(repo_root=tmp_path)

    index = json.loads((tmp_path / "data/manifests/slices/slice_index.json").read_text(encoding="utf-8"))
    assert result["removed_index_entries"] == 1
    assert not stale_manifest.exists()
    assert not stale_dir.parents[1].exists()
    assert good_manifest.exists()
    assert (good_dir / "part.parquet").exists()
    assert [item["manifest_id"] for item in index["slices"]] == ["good"]


def test_krx_timestamp_policy_requires_exchange_timezone() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-12 09:00:00")],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
        }
    )

    with pytest.raises(ValueError, match="Asia/Seoul timezone"):
        normalize_krx_intraday_frames([frame], symbol="000100", timeframe="1m")


def test_kis_client_contains_no_order_methods() -> None:
    forbidden = {"buy", "sell", "cancel", "revise", "place_order", "submit_order"}
    method_names = {
        name
        for name in dir(KisReadOnlyClient)
        if callable(getattr(KisReadOnlyClient, name)) and not name.startswith("_")
    }

    assert method_names.isdisjoint(forbidden)


def test_source_authority_contracts_lock_read_only_lineage_requirements() -> None:
    ibkr = ibkr_cme_nq_authority_contract()
    ibkr_us = ibkr_us_equity_authority_contract()
    kis = kis_krx_authority_contract()

    assert ibkr.validation_errors() == []
    assert ibkr_us.validation_errors() == []
    assert kis.validation_errors() == []
    assert ibkr.read_only is True
    assert ibkr_us.read_only is True
    assert kis.read_only is True
    assert ibkr.supports_live_trading is False
    assert ibkr_us.supports_live_trading is False
    assert kis.supports_live_trading is False
    assert "contract_chain_checksum" in ibkr.required_lineage_fields
    assert "source_conid_coverage" in ibkr.required_lineage_fields
    assert "contract_resolution_cache" in ibkr.required_lineage_fields
    assert "corporate_action_policy" in ibkr_us.required_lineage_fields
    assert "raw_adjustment_policy" in ibkr_us.required_lineage_fields
    assert "source_conid_coverage" in ibkr_us.required_lineage_fields
    assert "contract_resolution_cache" in ibkr_us.required_lineage_fields
    assert "session_policy" in kis.required_lineage_fields


def test_ibkr_cme_nq_adapter_writes_authoritative_read_only_slice(tmp_path: Path) -> None:
    _write_cme_rule_authority(tmp_path)
    adapter = IBKRCmeNqReadOnlyAdapter(DeterministicCmeNqProvider())
    request = CmeNqRefreshRequest(
        symbol="NQ",
        timeframe="5m",
        start=datetime(2026, 5, 3, 22, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 21, 55, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert order_surface_errors(adapter) == []
    assert result.status == "complete"
    assert result.raw_path.exists()
    assert result.canonical_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.lineage["credential_contract_id"] == (
        "ibkr_read_only_market_data_credentials_v1"
    )
    assert result.manifest.lineage["market_rule_authority_checksum"]
    assert result.manifest.lineage["source_contract_coverage"] == "all_rows"
    assert result.manifest.lineage["source_conid_coverage"] == "all_rows"
    assert result.manifest.lineage["contract_resolution_cache"]
    assert result.manifest.lineage["raw_write_checksum"]
    assert result.manifest.lineage["canonical_write_checksum"] == parquet_content_checksum(
        result.canonical_path
    )
    assert validate_market_manifest(result.manifest).valid is True
    index = json.loads(
        (tmp_path / "data/manifests/slices/slice_index.json").read_text(encoding="utf-8")
    )
    assert index["slices"][0]["manifest_id"] == result.manifest.manifest_id


def test_ibkr_cme_adapter_supports_es_daily_source_contracts(tmp_path: Path) -> None:
    _write_cme_rule_authority(tmp_path)
    adapter = IBKRCmeNqReadOnlyAdapter(DeterministicCmeNqProvider())
    request = CmeNqRefreshRequest(
        symbol="ES",
        timeframe="1d",
        start=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 8, 0, 0, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.symbol == "ES"
    assert result.manifest.timeframe == "1d"
    frame = pd.read_parquet(result.canonical_path)
    assert "source_contract" in frame.columns
    assert frame["source_contract"].astype(str).str.startswith("ES").all()


def test_ibkr_cme_adapter_blocks_missing_required_contract_resolution(tmp_path: Path) -> None:
    _write_cme_rule_authority(tmp_path)
    adapter = IBKRCmeNqReadOnlyAdapter(_MissingConidCmeProvider())
    request = CmeNqRefreshRequest(
        symbol="NQ",
        timeframe="5m",
        start=datetime(2026, 5, 3, 22, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 21, 55, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert result.manifest.usable_for_authoritative_validation is False
    assert "source_conid coverage incomplete" in result.manifest.blocking_reasons
    assert "lineage.contract_resolution_cache missing" in result.manifest.blocking_reasons


def test_ibkr_us_equity_adapter_writes_authoritative_read_only_slice(tmp_path: Path) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(DeterministicUsEquityProvider())
    request = UsEquityRefreshRequest(
        symbol="MSFT",
        timeframe="5m",
        start=datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 20, 0, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert order_surface_errors(adapter) == []
    assert result.status == "complete"
    assert result.raw_path.exists()
    assert result.canonical_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.source == "ibkr"
    assert result.manifest.market == "us_equity"
    assert result.manifest.lineage["corporate_action_policy"]
    assert result.manifest.lineage["raw_adjustment_policy"]
    assert result.manifest.lineage["source_conid_coverage"] == "all_rows"
    assert result.manifest.lineage["contract_resolution_cache"]
    assert result.manifest.lineage["canonical_write_checksum"] == parquet_content_checksum(
        result.canonical_path
    )
    frame = pd.read_parquet(result.canonical_path)
    assert frame["source_conid"].astype(str).str.strip().ne("").all()
    assert validate_market_manifest(result.manifest).valid is True


def test_ibkr_us_equity_adapter_blocks_missing_contract_resolution(tmp_path: Path) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(_MissingConidUsEquityProvider())
    request = UsEquityRefreshRequest(
        symbol="MSFT",
        timeframe="5m",
        start=datetime(2026, 5, 4, 13, 30, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 20, 0, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert result.manifest.usable_for_authoritative_validation is False
    assert "source_conid coverage incomplete" in result.manifest.blocking_reasons
    assert "lineage.contract_resolution_cache missing" in result.manifest.blocking_reasons


def test_ibkr_us_equity_adapter_normalizes_daily_date_labels_to_session_close(
    tmp_path: Path,
) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(_DailyDateLabelUsEquityProvider())
    request = UsEquityRefreshRequest(
        symbol="QQQ",
        timeframe="1d",
        start=datetime(2021, 2, 12, tzinfo=timezone.utc),
        end=datetime(2021, 2, 16, tzinfo=timezone.utc),
        primary_exchange="NASDAQ",
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.expected_bars == 2
    assert result.manifest.actual_bars == 2
    frame = pd.read_parquet(result.canonical_path)
    assert frame["timestamp_exchange"].tolist() == [
        "2021-02-12T16:00:00-05:00",
        "2021-02-16T16:00:00-05:00",
    ]


def test_ibkr_us_equity_intraday_calendar_models_early_closes_and_hourly_labels(
    tmp_path: Path,
) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(DeterministicUsEquityProvider())
    early_close = datetime(2024, 11, 29, tzinfo=timezone.utc)

    qqq_15m = adapter.refresh_historical_bars(
        repo_root=tmp_path,
        request=UsEquityRefreshRequest(
            symbol="QQQ",
            timeframe="15m",
            start=early_close,
            end=datetime(2024, 11, 29, 23, 59, tzinfo=timezone.utc),
            primary_exchange="NASDAQ",
            source_version=FIXTURE_DATA_SHA,
        ),
    )
    qqq_1h = adapter.refresh_historical_bars(
        repo_root=tmp_path,
        request=UsEquityRefreshRequest(
            symbol="QQQ",
            timeframe="1h",
            start=early_close,
            end=datetime(2024, 11, 29, 23, 59, tzinfo=timezone.utc),
            primary_exchange="NASDAQ",
            source_version=FIXTURE_DATA_SHA,
        ),
    )

    assert qqq_15m.manifest.expected_bars == 14
    assert qqq_15m.manifest.actual_bars == 14
    assert qqq_1h.manifest.expected_bars == 4
    assert qqq_1h.manifest.actual_bars == 4
    frame = pd.read_parquet(qqq_1h.canonical_path)
    assert frame["timestamp_exchange"].tolist() == [
        "2024-11-29T09:30:00-05:00",
        "2024-11-29T10:00:00-05:00",
        "2024-11-29T11:00:00-05:00",
        "2024-11-29T12:00:00-05:00",
    ]


def test_ibkr_us_equity_extended_hours_calendar_matches_swing_legacy_policy(
    tmp_path: Path,
) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(DeterministicUsEquityProvider())
    full_day = adapter.refresh_historical_bars(
        repo_root=tmp_path,
        request=UsEquityRefreshRequest(
            symbol="QQQ",
            timeframe="15m",
            start=datetime(2024, 11, 27, 9, 0, tzinfo=timezone.utc),
            end=datetime(2024, 11, 28, 1, 0, tzinfo=timezone.utc),
            primary_exchange="NASDAQ",
            use_rth=False,
            source_version=FIXTURE_DATA_SHA,
        ),
    )
    early_close = adapter.refresh_historical_bars(
        repo_root=tmp_path,
        request=UsEquityRefreshRequest(
            symbol="QQQ",
            timeframe="1h",
            start=datetime(2024, 11, 29, 9, 0, tzinfo=timezone.utc),
            end=datetime(2024, 11, 30, 1, 0, tzinfo=timezone.utc),
            primary_exchange="NASDAQ",
            use_rth=False,
            source_version=FIXTURE_DATA_SHA,
        ),
    )

    assert full_day.manifest.expected_bars == 64
    assert full_day.manifest.actual_bars == 64
    assert early_close.manifest.expected_bars == 13
    assert early_close.manifest.actual_bars == 13


def test_ibkr_us_equity_extended_hours_allows_sparse_trade_bars(tmp_path: Path) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(_SparseExtendedUsEquityProvider())
    result = adapter.refresh_historical_bars(
        repo_root=tmp_path,
        request=UsEquityRefreshRequest(
            symbol="GLD",
            timeframe="15m",
            start=datetime(2024, 11, 27, 9, 0, tzinfo=timezone.utc),
            end=datetime(2024, 11, 28, 1, 0, tzinfo=timezone.utc),
            primary_exchange="ARCA",
            use_rth=False,
            source_version=FIXTURE_DATA_SHA,
        ),
    )

    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.expected_bars == result.manifest.actual_bars
    assert result.manifest.lineage["expected_bar_policy"] == "sparse_extended_hours_trade_bars"
    assert result.manifest.lineage["max_session_slots"] == "64"
    assert result.manifest.lineage["returned_row_count"] == str(result.manifest.actual_bars)
    assert result.manifest.lineage["source_request_params_hash"]
    assert '"use_rth":false' in result.manifest.lineage["source_request_params_json"]
    assert result.manifest.lineage["multi_day_coverage_holes"] == ""


def test_ibkr_us_equity_extended_hours_blocks_multi_day_coverage_holes(
    tmp_path: Path,
) -> None:
    adapter = IBKRUsEquityReadOnlyAdapter(_SparseExtendedCoverageHoleUsEquityProvider())

    with pytest.raises(ValueError, match="consecutive US equity trading dates"):
        adapter.refresh_historical_bars(
            repo_root=tmp_path,
            request=UsEquityRefreshRequest(
                symbol="GLD",
                timeframe="15m",
                start=datetime(2024, 11, 25, 9, 0, tzinfo=timezone.utc),
                end=datetime(2024, 11, 30, 1, 0, tzinfo=timezone.utc),
                primary_exchange="ARCA",
                use_rth=False,
                source_version=FIXTURE_DATA_SHA,
            ),
        )


def test_kis_krx_adapter_writes_authoritative_read_only_slice(tmp_path: Path) -> None:
    adapter = KISKrxReadOnlyAdapter(DeterministicKrxProvider())
    request = KrxRefreshRequest(
        symbol="005930",
        timeframe="5m",
        start=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 8, 0, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    result = adapter.refresh_historical_bars(repo_root=tmp_path, request=request)

    assert order_surface_errors(adapter) == []
    assert result.status == "complete"
    assert result.raw_path.exists()
    assert result.canonical_path.exists()
    assert result.manifest_path.exists()
    assert result.manifest.usable_for_authoritative_validation is True
    assert result.manifest.source == "kis"
    assert result.manifest.market == "krx_equity"
    assert result.manifest.lineage["timestamp_audit"] == "passed"
    assert result.manifest.lineage["canonical_write_checksum"] == parquet_content_checksum(
        result.canonical_path
    )
    assert validate_market_manifest(result.manifest).valid is True


def test_kis_api_provider_uses_reference_historical_intraday_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKisClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def get_minute_chart(self, *_args, **_kwargs) -> dict:
            raise AssertionError("historical sync must not use current-minute chart endpoint")

        def get_historical_minute_page(
            self,
            symbol: str,
            *,
            date_yyyymmdd: str,
            hour_hhmmss: str,
            market_code: str,
            include_previous: bool,
        ) -> dict:
            self.calls.append(
                {
                    "symbol": symbol,
                    "date": date_yyyymmdd,
                    "hour": hour_hhmmss,
                    "market": market_code,
                    "include_previous": str(include_previous),
                }
            )
            if len(self.calls) == 1:
                return {"output2": [_kis_row("090400", 104), _kis_row("090300", 103)]}
            return {
                "output2": [
                    _kis_row("090200", 102),
                    _kis_row("090100", 101),
                    _kis_row("090000", 100),
                ]
            }

    monkeypatch.setenv("KIS_REQUEST_SLEEP_SECONDS", "0")
    monkeypatch.delenv("KIS_INTRADAY_MAX_PAGES", raising=False)
    client = FakeKisClient()
    provider = KisApiKrxProvider(client=client)
    request = KrxRefreshRequest(
        symbol="005930",
        timeframe="5m",
        start=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        source_version=FIXTURE_DATA_SHA,
    )

    frame = provider.historical_bars(request)

    assert [call["hour"] for call in client.calls] == ["090400", "090200"]
    assert len(frame) == 1
    assert pd.Timestamp(frame.iloc[0]["timestamp_utc"]) == pd.Timestamp("2026-05-04T00:00:00Z")
    assert float(frame.iloc[0]["open"]) == 100.0
    assert float(frame.iloc[0]["close"]) == 104.0
    assert float(frame.iloc[0]["volume"]) == 50.0


def _kis_row(hour: str, price: int) -> dict[str, str]:
    return {
        "stck_bsop_date": "20260504",
        "stck_cntg_hour": hour,
        "stck_oprc": str(price),
        "stck_hgpr": str(price + 1),
        "stck_lwpr": str(price - 1),
        "stck_prpr": str(price),
        "cntg_vol": "10",
    }


def test_ibkr_alignment_detects_derived_timeframe_mismatch() -> None:
    index = pd.date_range("2026-05-01T00:00:00Z", periods=1, freq="5min")
    derived = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [1.0], "close": [2.0], "volume": [10]}, index=index)
    target = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [1.0], "close": [3.0], "volume": [10]}, index=index)

    result = compare_derived_frame_alignment(
        symbol="NQ",
        derived=derived,
        target=target,
        base_timeframe="1m",
        target_timeframe="5m",
    )

    assert result.status == "MISMATCH"
    assert result.mismatched_rows == 1


def test_duplicate_krx_intraday_files_are_merged_and_deduped() -> None:
    ts = pd.Timestamp("2026-05-12 09:00:00", tz="Asia/Seoul")
    first = pd.DataFrame({"timestamp": [ts], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})
    second = pd.DataFrame({"timestamp": [ts], "open": [2], "high": [2], "low": [2], "close": [2], "volume": [2]})

    out = normalize_krx_intraday_frames([first, second], symbol="000100", timeframe="1m")

    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 2.0
    assert out.iloc[0]["timestamp_exchange"].endswith("+09:00")


def test_missing_ranges_are_reported() -> None:
    timestamps = [
        datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 0, 4, tzinfo=timezone.utc),
    ]

    missing = detect_missing_ranges(timestamps, "1m")

    assert len(missing) == 1
    assert missing[0].start_ts.minute == 2
    assert missing[0].end_ts.minute == 3


def test_expected_bars_respect_calendar_holidays() -> None:
    calendar = CalendarDefinition(
        calendar_id="fixture",
        timezone="UTC",
        session_open="09:00",
        session_close="09:02",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset({date(2026, 1, 1)}),
        version="v1",
    )

    count = expected_bars(
        calendar,
        "1m",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 23, 59, tzinfo=timezone.utc),
    )

    assert count == 2


def test_daily_coverage_ignores_weekend_gaps() -> None:
    calendar = CalendarDefinition(
        calendar_id="fixture",
        timezone="America/New_York",
        session_open="09:30",
        session_close="16:00",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(),
        version="v1",
    )

    expected, actual, missing = coverage_counts(
        pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-04T00:00:00Z"]),
        "1d",
        calendar,
    )

    assert expected == 2
    assert actual == 2
    assert missing == []


def test_daily_coverage_uses_exchange_dates_for_krx_midnight_labels() -> None:
    calendar = CalendarDefinition(
        calendar_id="fixture",
        timezone="Asia/Seoul",
        session_open="09:00",
        session_close="15:30",
        weekdays=(0, 1, 2, 3, 4),
        holidays=frozenset(),
        version="v1",
    )

    expected, actual, missing = coverage_counts(
        pd.to_datetime(["2026-05-03T15:00:00Z", "2026-05-04T15:00:00Z"]),
        "1d",
        calendar,
        exchange_timestamps=["2026-05-04 00:00:00+09:00", "2026-05-05 00:00:00+09:00"],
    )

    assert expected == 2
    assert actual == 2
    assert missing == []


def test_krx_calendar_includes_temporary_closure_dates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calendar = krx_calendar(repo_root / "data/calendars/krx_holidays.yaml")

    assert calendar.session_close == "15:30"
    assert calendar.is_trading_day(date(2024, 10, 1)) is False
    assert calendar.is_trading_day(date(2025, 1, 27)) is False


def test_kis_intraday_calendar_models_reference_updater_session_close() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calendar = kis_intraday_calendar_definition(repo_root / "data/calendars/krx_holidays.yaml")

    assert calendar.session_close == "15:30"
    assert expected_bars(
        calendar,
        "1m",
        datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 4, 23, 59, tzinfo=timezone.utc),
    ) == 390
    expected_1m = kis_intraday_expected_bar_opens(
        calendar,
        "1m",
        datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 4, 6, 30, tzinfo=timezone.utc),
    )
    expected_5m = kis_intraday_expected_bar_opens(
        calendar,
        "5m",
        datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 4, 6, 30, tzinfo=timezone.utc),
    )

    assert len(expected_1m) == 381
    assert len(expected_5m) == 77
    assert expected_5m.tz_convert("Asia/Seoul").strftime("%H:%M").tolist()[-3:] == [
        "15:10",
        "15:15",
        "15:30",
    ]


def test_kis_intraday_coverage_allows_sparse_trade_slots() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calendar = kis_intraday_calendar_definition(repo_root / "data/calendars/krx_holidays.yaml")
    expected = kis_intraday_expected_bar_opens(
        calendar,
        "5m",
        datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 4, 6, 30, tzinfo=timezone.utc),
    )
    sparse = expected.delete([10, 20, 30])

    expected_count, actual_count, missing = coverage_counts(sparse, "5m", calendar)

    assert expected_count == actual_count == len(sparse)
    assert missing == []


def test_us_equity_calendar_covers_historical_holidays() -> None:
    calendar = us_equities_calendar()

    assert calendar.is_trading_day(date(2021, 2, 15)) is False
    assert calendar.is_trading_day(date(2023, 6, 19)) is False
    assert calendar.is_trading_day(date(2024, 3, 29)) is False
    assert calendar.is_trading_day(date(2025, 1, 9)) is False


def test_cme_equity_index_calendar_models_sunday_evening_sessions() -> None:
    calendar = cme_calendar()

    assert calendar.weekdays == (6, 0, 1, 2, 3)
    assert calendar.is_trading_day(date(2025, 1, 9)) is False
    assert expected_bars(
        calendar,
        "1m_bid_ask",
        datetime(2026, 5, 31, 22, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 31, 22, 4, tzinfo=timezone.utc),
    ) == 5
    assert expected_bars(
        calendar,
        "1h",
        datetime(2026, 5, 31, 22, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc),
    ) > 0


def test_cme_equity_index_calendar_removes_known_closure_ranges() -> None:
    calendar = cme_calendar()

    assert expected_bars(
        calendar,
        "5m",
        datetime(2025, 1, 8, 23, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 9, 21, 55, tzinfo=timezone.utc),
    ) == 186
    assert expected_bars(
        calendar,
        "5m",
        datetime(2025, 7, 2, 22, 0, tzinfo=timezone.utc),
        datetime(2025, 7, 3, 20, 55, tzinfo=timezone.utc),
    ) == 231


def test_authoritative_bundle_requires_fee_slippage_and_adjustment_versions() -> None:
    with pytest.raises(ValueError, match="authoritative data bundle missing required fields"):
        DataBundleManifest(
            data_repo_path="/tmp/data",
            data_repo_commit_sha="sha",
            slice_manifests=[
                DataBundleSlice(
                    manifest_path="slice.json",
                    manifest_id="slice-1",
                    source="fixture",
                    market="crypto_perp",
                    symbol="BTC",
                    timeframe="1m",
                    checksum="sha",
                    calendar="crypto_utc_24_7_v1",
                    authoritative=True,
                )
            ],
            calendars=["crypto_utc_24_7_v1"],
            status=DataBundleStatus.AUTHORITATIVE,
        )


def _manifest(
    *,
    checksum: str = "slice-sha",
    session_calendar: str = "crypto_utc_24_7_v1",
    symbol: str = "BTC",
    timeframe: str = "1m",
    source_version: str = FIXTURE_DATA_SHA,
    adjustment_policy: str = "crypto_raw_perp_policy_v1",
    fee_model_version: str = "fees_v1",
    slippage_model_version: str = "slippage_v1",
    authoritative: bool = True,
    blocking_reasons: list[str] | None = None,
    lineage: dict[str, str] | None = None,
) -> MarketDataManifest:
    return MarketDataManifest(
        source="hyperliquid",
        market="crypto_perp",
        symbol=symbol,
        timeframe=timeframe,
        start_ts=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        expected_bars=2,
        actual_bars=2,
        coverage_ratio=1.0,
        session_calendar=session_calendar,
        timezone="UTC",
        checksum=checksum,
        source_version=source_version,
        adjustment_policy=adjustment_policy,
        fee_model_version=fee_model_version,
        slippage_model_version=slippage_model_version,
        lineage=lineage or {},
        usable_for_authoritative_validation=authoritative,
        blocking_reasons=blocking_reasons or [],
    )


def _ibkr_us_equity_manifest(
    *,
    symbol: str,
    timeframe: str,
    strategy_data_family: str,
    checksum: str,
) -> MarketDataManifest:
    primary_exchange = "NASDAQ" if symbol == "QQQ" else "NYSE"
    return _manifest(
        checksum=checksum,
        session_calendar="us_equities_xnys_xnas_v1",
        symbol=symbol,
        timeframe=timeframe,
        adjustment_policy="us_equity_split_dividend_adjusted_policy_v1",
        lineage={
            "source_endpoint": f"ibkr://historical-data/STK/SMART/{primary_exchange}/{symbol}/TRADES/ETH",
            "export_id": f"fixture-{strategy_data_family}-{symbol}-{timeframe}",
            "pulled_at_utc": "2026-05-31T00:00:00Z",
            "config_hash": "fixture-config",
            "corporate_action_policy": "us_equity_split_dividend_adjusted_policy_v1",
            "raw_adjustment_policy": "ibkr_trades_unadjusted_raw_v1",
            "session_policy": "us_equity_extended_session_0400_2000_new_york_v1",
            "source_conid_coverage": "all_rows",
            "contract_resolution_cache": "fixture-cache",
            "source_request_params_hash": "fixture-params",
            "source_request_params_json": json.dumps(
                {"primary_exchange": primary_exchange, "use_rth": False},
                sort_keys=True,
            ),
            "returned_row_count": "2",
            "strategy_data_family": strategy_data_family,
            "source_request_id": f"src_req_{strategy_data_family}_{symbol}_{timeframe}",
            "use_rth": "false",
        },
    ).model_copy(update={"source": "ibkr", "market": "us_equity"})


def _slice_index_entry(
    repo_root: Path,
    manifest_path: Path,
    manifest: MarketDataManifest,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "manifest_id": manifest.manifest_id,
        "manifest_path": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
        "source": manifest.source,
        "market": manifest.market,
        "symbol": manifest.symbol,
        "timeframe": manifest.timeframe,
        "checksum": manifest.checksum,
        "canonical_paths": [],
    }
    family = str((manifest.lineage or {}).get("strategy_data_family") or "")
    if family:
        entry["strategy_data_family"] = family
    return entry


def _write_cme_rule_authority(repo_root: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "market_rules"
        / "cme"
        / "equity_index_futures_v1.json"
    )
    target = repo_root / "data" / "market_rules" / "cme" / "equity_index_futures_v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


class _MissingConidCmeProvider:
    def historical_bars(self, request: CmeNqRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        timestamps = pd.date_range(request.start, request.end, freq="5min", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp_utc": timestamps,
                "open": [18_000.0 + index for index in range(len(timestamps))],
                "high": [18_005.0 + index for index in range(len(timestamps))],
                "low": [17_995.0 + index for index in range(len(timestamps))],
                "close": [18_001.0 + index for index in range(len(timestamps))],
                "volume": [1000.0 for _ in timestamps],
                "source_contract": ["NQM6" for _ in timestamps],
            }
        )


class _MissingConidUsEquityProvider:
    def historical_bars(self, request: UsEquityRefreshRequest) -> pd.DataFrame:
        request = request.normalized()
        frame = DeterministicUsEquityProvider().historical_bars(request)
        frame["source_conid"] = ""
        return frame


class _DailyDateLabelUsEquityProvider:
    def historical_bars(self, request: UsEquityRefreshRequest) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_utc": ["2021-02-12", "2021-02-16"],
                "open": [310.0, 312.0],
                "high": [311.0, 313.0],
                "low": [309.0, 311.0],
                "close": [310.5, 312.5],
                "volume": [10_000.0, 11_000.0],
                "source_contract": [request.symbol, request.symbol],
                "source_conid": ["320227571", "320227571"],
                "source_local_symbol": [request.symbol, request.symbol],
                "source_primary_exchange": ["NASDAQ", "NASDAQ"],
                "contract_resolution_method": ["fixture", "fixture"],
            }
        )


class _SparseExtendedUsEquityProvider:
    def historical_bars(self, request: UsEquityRefreshRequest) -> pd.DataFrame:
        frame = DeterministicUsEquityProvider().historical_bars(request)
        return frame.iloc[[index for index in range(len(frame)) if index % 10 != 0]].reset_index(
            drop=True
        )


class _SparseExtendedCoverageHoleUsEquityProvider:
    def historical_bars(self, request: UsEquityRefreshRequest) -> pd.DataFrame:
        frame = DeterministicUsEquityProvider().historical_bars(request)
        exchange_dates = pd.DatetimeIndex(frame["timestamp_utc"]).tz_convert("America/New_York").date
        blocked_dates = {date(2024, 11, 26), date(2024, 11, 27)}
        return frame[[value not in blocked_dates for value in exchange_dates]].reset_index(
            drop=True
        )


def _bundle(
    manifest: MarketDataManifest,
    *,
    status: DataBundleStatus = DataBundleStatus.AUTHORITATIVE,
) -> DataBundleManifest:
    return DataBundleManifest(
        data_repo_path="/tmp/trading_assistant_data",
        data_repo_commit_sha=FIXTURE_DATA_SHA,
        slice_manifests=[
            DataBundleSlice(
                manifest_path="slice.json",
                manifest_id=manifest.manifest_id,
                source=manifest.source,
                market=manifest.market,
                symbol=manifest.symbol,
                timeframe=manifest.timeframe,
                start_ts=manifest.start_ts,
                end_ts=manifest.end_ts,
                checksum=manifest.checksum,
                calendar=manifest.session_calendar,
                authoritative=manifest.usable_for_authoritative_validation,
            )
        ],
        calendars=[manifest.session_calendar] if manifest.session_calendar else [],
        fee_model_version=manifest.fee_model_version,
        slippage_model_version=manifest.slippage_model_version,
        adjustment_policy=manifest.adjustment_policy,
        status=status,
        diagnostics_only_reason="" if status == DataBundleStatus.AUTHORITATIVE else "not all slices authoritative",
    )


def _write_sample_slice(
    tmp_path: Path,
    *,
    symbol: str,
    timeframe: str,
    adjustment_policy: str = "crypto_raw_perp_policy_v1",
) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T00:01:00Z"]),
            "timestamp_exchange": ["2026-05-01 00:00:00+00:00", "2026-05-01 00:01:00+00:00"],
            "symbol": [symbol, symbol],
            "market": ["crypto_perp", "crypto_perp"],
            "source": ["hyperliquid", "hyperliquid"],
            "timeframe": [timeframe, timeframe],
            "kind": ["trades", "trades"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "source_file": ["fixture", "fixture"],
            "source_row_hash": ["a", "b"],
        }
    )
    canonical_path = (
        tmp_path
        / "data/canonical/bars/market=crypto_perp/source=hyperliquid/kind=trades"
        / f"symbol={symbol}/timeframe={timeframe}/year=2026/month=05/part.parquet"
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(canonical_path, engine="pyarrow", index=False)
    manifest = _manifest(
        checksum=parquet_content_checksum(canonical_path),
        symbol=symbol,
        timeframe=timeframe,
        adjustment_policy=adjustment_policy,
    )
    manifest_path = (
        tmp_path
        / "data/manifests/slices/hyperliquid/crypto_perp"
        / symbol
        / timeframe
        / "20260501T000000Z_20260501T000100Z.market_data_manifest.json"
    )
    write_model(manifest_path, manifest)
    index_path = tmp_path / "data/manifests/slices/slice_index.json"
    payload = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": "slice_index_v1", "slices": []}
    entry = {
        "manifest_id": manifest.manifest_id,
        "manifest_path": str(manifest_path.relative_to(tmp_path)).replace("\\", "/"),
        "source": manifest.source,
        "market": manifest.market,
        "symbol": manifest.symbol,
        "timeframe": manifest.timeframe,
        "checksum": manifest.checksum,
        "canonical_paths": [str(canonical_path.relative_to(tmp_path)).replace("\\", "/")],
    }
    payload["slices"] = [item for item in payload["slices"] if item["manifest_id"] != manifest.manifest_id]
    payload["slices"].append(entry)
    write_json(index_path, payload)
    return manifest_path
