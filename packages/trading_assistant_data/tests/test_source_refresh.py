from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_assistant_data.source_refresh import (
    sync_ibkr_from_source_requests,
    sync_kis_from_source_requests,
)


def test_ibkr_sync_dry_run_selects_momentum_source_requests(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source_request_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requests": [
                    _request("trading_momentum", "NQ", "5m"),
                    _request("trading_momentum", "ES", "1d"),
                    _request("trading_swing", "QQQ", "1h", source="ibkr", market="us_equity"),
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = sync_ibkr_from_source_requests(
        repo_root=tmp_path,
        source_request_manifest=manifest_path,
        families=["trading_momentum"],
        dry_run=True,
    )

    assert payload["status"] == "planned"
    assert payload["request_count"] == 2
    assert {(item["symbol"], item["timeframe"]) for item in payload["requests"]} == {
        ("ES", "1d"),
        ("NQ", "5m"),
    }


def test_kis_sync_dry_run_selects_k_stock_strategy_family_requests(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source_request_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requests": [
                    _kis_request("k_stock_kis_intraday", "000100", "5m"),
                    _kis_request("k_stock_kis_intraday", "000100", "15m"),
                    _kis_request("other_kis_family", "005930", "5m"),
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = sync_kis_from_source_requests(
        repo_root=tmp_path,
        source_request_manifest=manifest_path,
        families=["k_stock_olr_kalcb"],
        intervals=["5m"],
        dry_run=True,
    )

    assert payload["status"] == "planned"
    assert payload["request_count"] == 1
    assert payload["requests"][0]["legacy_family"] == "k_stock_kis_intraday"
    assert payload["requests"][0]["strategy_data_family"] == "k_stock_olr_kalcb"
    assert payload["requests"][0]["symbol"] == "000100"
    assert payload["requests"][0]["timeframe"] == "5m"


def test_ibkr_sync_non_dry_run_requires_network_and_write_ack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "source_request_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requests": [
                    _request(
                        "trading_swing",
                        "QQQ",
                        "1h",
                        source="ibkr",
                        market="us_equity",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TA_SOURCE_REFRESH_ALLOW_NETWORK", raising=False)
    monkeypatch.delenv("TA_SOURCE_REFRESH_ALLOW_WRITE", raising=False)

    with pytest.raises(RuntimeError, match="TA_SOURCE_REFRESH_ALLOW_NETWORK"):
            sync_ibkr_from_source_requests(
                repo_root=tmp_path,
                source_request_manifest=manifest_path,
                families=["trading_swing"],
                dry_run=False,
            )


def test_ibkr_full_legacy_blocks_pre_retention_futures_before_network(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source_request_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requests": [
                    _request(
                        "trading_momentum",
                        "NQ",
                        "5m",
                        start="2023-07-05T14:15:00+00:00",
                        end="2026-05-01T20:55:00+00:00",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    probe_path = _write_contract_probe(tmp_path)

    payload = sync_ibkr_from_source_requests(
        repo_root=tmp_path,
        source_request_manifest=manifest_path,
        families=["trading_momentum"],
        dry_run=False,
        contract_probe_path=probe_path,
    )

    assert payload["status"] == "failed"
    assert payload["result_count"] == 0
    assert payload["failures"][0]["missing_contracts"][0] == "NQU3"
    assert "archived raw payloads" in payload["failures"][0]["error"]


def test_ibkr_retention_covered_trims_futures_window_from_probe(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source_request_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "requests": [
                    _request(
                        "trading_momentum",
                        "NQ",
                        "5m",
                        start="2023-07-05T14:15:00+00:00",
                        end="2026-05-01T20:55:00+00:00",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    probe_path = _write_contract_probe(tmp_path)

    payload = sync_ibkr_from_source_requests(
        repo_root=tmp_path,
        source_request_manifest=manifest_path,
        families=["trading_momentum"],
        coverage_mode="retention-covered",
        contract_probe_path=probe_path,
        dry_run=True,
    )

    assert payload["status"] == "planned"
    assert payload["request_count"] == 1
    request = payload["requests"][0]
    assert request["start"] == "2024-03-11T00:00:00+00:00"
    adjustment = payload["coverage_report"]["adjustments"][0]
    assert adjustment["original_start"] == "2023-07-05T14:15:00+00:00"
    assert adjustment["contract_chain"][0] == "NQM4"
    assert "retention-covered live TWS lane" in adjustment["coverage_note"]


def _request(
    family: str,
    symbol: str,
    timeframe: str,
    *,
    source: str = "ibkr",
    market: str = "cme_futures",
    start: str = "2026-05-01T00:00:00+00:00",
    end: str = "2026-05-02T00:00:00+00:00",
) -> dict:
    source_kind = (
        "ibkr_cme_futures_historical_bars"
        if market == "cme_futures"
        else "ibkr_us_equity_historical_bars"
    )
    return {
        "request_id": f"src_req_{family}_{symbol}_{timeframe}",
        "legacy_family": family,
        "legacy_path": f"legacy/{symbol}_{timeframe}.parquet",
        "source": source,
        "source_kind": source_kind,
        "source_endpoint": f"ibkr://{symbol}/{timeframe}",
        "market": market,
        "symbol": symbol,
        "timeframe": timeframe,
        "download_request": {
            "provider": "ibkr",
            "request_type": "historical_bars",
            "sec_type": "FUT" if market == "cme_futures" else "STK",
            "symbol": symbol,
            "exchange": "CME" if market == "cme_futures" else "SMART",
            "currency": "USD",
            "timeframe": timeframe,
            "what_to_show": "TRADES",
            "use_rth": False if market == "cme_futures" else True,
            "start": start,
            "end": end,
        },
    }


def _kis_request(
    family: str,
    symbol: str,
    timeframe: str,
    *,
    start: str = "2026-05-01T00:00:00+00:00",
    end: str = "2026-05-02T00:00:00+00:00",
) -> dict:
    return {
        "request_id": f"src_req_{family}_{symbol}_{timeframe}",
        "legacy_family": family,
        "legacy_path": f"legacy/{symbol}_{timeframe}.parquet",
        "source": "kis",
        "source_kind": "kis_krx_intraday_bars",
        "source_endpoint": f"kis://{symbol}/{timeframe}",
        "market": "krx_equity",
        "symbol": symbol,
        "timeframe": timeframe,
        "download_request": {
            "provider": "kis",
            "request_type": "historical_bars",
            "fid_input_iscd": symbol,
            "fid_cond_mrkt_div_code": "J",
            "timeframe": timeframe,
            "start": start,
            "end": end,
        },
    }


def _write_contract_probe(root: Path) -> Path:
    path = root / "ibkr-contract-resolution-probe.json"
    path.write_text(
        json.dumps(
            {
                "roots": {
                    "NQ": {
                        "contracts": [
                            {
                                "local_symbol": "NQM4",
                                "last_trade_date_or_contract_month": "20240621",
                                "con_id": "620730920",
                            },
                            {
                                "local_symbol": "NQU4",
                                "last_trade_date_or_contract_month": "20240920",
                                "con_id": "637533450",
                            },
                            {
                                "local_symbol": "NQZ4",
                                "last_trade_date_or_contract_month": "20241220",
                                "con_id": "563947733",
                            },
                            {
                                "local_symbol": "NQH5",
                                "last_trade_date_or_contract_month": "20250321",
                                "con_id": "666754605",
                            },
                            {
                                "local_symbol": "NQM5",
                                "last_trade_date_or_contract_month": "20250620",
                                "con_id": "672387474",
                            },
                            {
                                "local_symbol": "NQU5",
                                "last_trade_date_or_contract_month": "20250919",
                                "con_id": "691171690",
                            },
                            {
                                "local_symbol": "NQZ5",
                                "last_trade_date_or_contract_month": "20251219",
                                "con_id": "563947738",
                            },
                            {
                                "local_symbol": "NQH6",
                                "last_trade_date_or_contract_month": "20260320",
                                "con_id": "730283097",
                            },
                            {
                                "local_symbol": "NQM6",
                                "last_trade_date_or_contract_month": "20260618",
                                "con_id": "750150196",
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path
