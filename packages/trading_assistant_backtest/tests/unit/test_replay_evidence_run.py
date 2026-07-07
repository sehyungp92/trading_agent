from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_assistant_backtest.contract_models import DataBundleManifest, DataBundleSlice
from trading_assistant_backtest.validation.replay_evidence_run import (
    _preferred_replay_paths,
    _walk_forward_window,
    _walk_forward_leakage_checks,
    _walk_forward_leakage_pass,
)


def test_walk_forward_leakage_checks_fail_duplicate_bundle_checksums() -> None:
    runs = [
        {
            "run_month": "2026-03",
            "bundle_checksum": "same",
            "window": {"start": "2026-03-01", "end": "2026-03-31"},
        },
        {
            "run_month": "2026-04",
            "bundle_checksum": "same",
            "window": {"start": "2026-04-01", "end": "2026-04-30"},
        },
        {
            "run_month": "2026-05",
            "bundle_checksum": "unique",
            "window": {"start": "2026-05-01", "end": "2026-05-31"},
        },
    ]

    checks = _walk_forward_leakage_checks(runs)

    assert checks["window_order_strictly_increasing"] is True
    assert checks["bundle_checksums_unique"] is False
    assert _walk_forward_leakage_pass(checks) is False


def test_walk_forward_window_uses_monthly_bundle_path_not_widest_context_slice() -> None:
    bundle = DataBundleManifest(
        data_repo_path=".",
        data_repo_commit_sha="a" * 40,
        data_repo_branch="main",
        slice_manifests=[
            DataBundleSlice(
                manifest_path="slice.json",
                manifest_id="daily-context",
                source="krx",
                market="krx_equity",
                symbol="005930",
                timeframe="1d",
                start_ts="2024-02-18T00:00:00Z",
                end_ts="2026-05-28T00:00:00Z",
                checksum="b" * 64,
                calendar="krx_stock_regular",
                authoritative=True,
            )
        ],
        calendars=["krx_stock_regular"],
        fee_model_version="fee_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="raw",
        status="authoritative",
    )

    start, end = _walk_forward_window(
        Path("data/bundles/monthly/2026-03/k_stock_olr_kalcb/portfolio/data_bundle_manifest.json"),
        bundle,
    )

    assert start == date(2026, 3, 1)
    assert end == date(2026, 3, 31)


def test_preferred_replay_paths_need_three_windows_before_replacing_portfolio() -> None:
    preferred = [
        Path("2026-03/trading_stock_family/us_msft_5m/data_bundle_manifest.json"),
        Path("2026-04/trading_stock_family/us_msft_5m/data_bundle_manifest.json"),
    ]
    fallback = [
        Path("2026-03/trading_stock_family/portfolio/data_bundle_manifest.json"),
        Path("2026-04/trading_stock_family/portfolio/data_bundle_manifest.json"),
        Path("2026-05/trading_stock_family/portfolio/data_bundle_manifest.json"),
    ]

    assert _preferred_replay_paths(preferred, fallback=fallback) == sorted(fallback)
    assert _preferred_replay_paths([*preferred, Path("2026-05/x")], fallback=fallback) == [
        *preferred,
        Path("2026-05/x"),
    ]
