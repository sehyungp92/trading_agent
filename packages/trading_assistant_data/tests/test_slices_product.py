from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading_assistant_data.manifests import MarketDataManifest
from trading_assistant_data.slices import (
    CanonicalSlice,
    DataSliceProduct,
    SliceRequest,
    SliceWrite,
    timestamps_sorted_unique_utc,
)
from trading_assistant_data.slices import SliceWrite as ProductSliceWrite
from trading_assistant_data.slices.authority import SliceAuthorityStatus
from trading_assistant_data.slices.coverage import coverage_report
from trading_assistant_data.slices.market_rules import crypto_calendar
from trading_assistant_data.slices.writer import slice_manifest_path


def test_slice_write_is_owned_by_slice_product() -> None:
    assert SliceWrite is ProductSliceWrite


def test_slice_manifest_path_matches_legacy_layout(tmp_path: Path) -> None:
    manifest = MarketDataManifest(
        source="ibkr",
        market="cme_futures",
        symbol="NQ",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    assert slice_manifest_path(tmp_path, manifest) == (
        tmp_path
        / "data"
        / "manifests"
        / "slices"
        / "ibkr"
        / "cme_futures"
        / "NQ"
        / "1m"
        / "20260501T000000Z_20260502T000000Z.market_data_manifest.json"
    )


def test_data_slice_product_write_slice_uses_manifest_writer(tmp_path: Path) -> None:
    manifest = MarketDataManifest(
        source="ibkr",
        market="cme_futures",
        symbol="NQ",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    canonical = CanonicalSlice(
        request=SliceRequest(market="cme_futures", source="ibkr", symbol="NQ", timeframe="1m"),
        canonical_paths=[tmp_path / "NQ.parquet"],
        manifest=manifest,
    )

    write = DataSliceProduct(tmp_path).write_slice(canonical)

    assert write.manifest_path == slice_manifest_path(tmp_path, manifest)
    assert write.manifest_path.exists()
    assert write.canonical_paths == [tmp_path / "NQ.parquet"]
    assert write.manifest is manifest


def test_slice_coverage_delegates_to_validation_calendar() -> None:
    calendar = crypto_calendar()
    coverage = coverage_report(
        timestamps=[
            datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        ],
        timeframe="1m",
        calendar=calendar,
    )

    assert coverage.expected == 2
    assert coverage.actual == 2
    assert coverage.missing_ranges == []


def test_data_slice_product_coverage_report_uses_declared_canonical_inputs(tmp_path: Path) -> None:
    calendar = crypto_calendar()
    manifest = MarketDataManifest(
        source="hyperliquid",
        market="crypto",
        symbol="BTC",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
    )
    canonical = CanonicalSlice(
        request=SliceRequest(market="crypto", source="hyperliquid", symbol="BTC", timeframe="1m"),
        canonical_paths=[],
        manifest=manifest,
        timestamps=[
            datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        ],
    )

    coverage = DataSliceProduct(tmp_path).coverage_report(calendar=calendar, canonical=canonical)

    assert coverage.expected == 2
    assert coverage.actual == 2
    assert coverage.missing_ranges == []


def test_data_slice_product_coverage_for_request_builds_declared_canonical_slice(
    tmp_path: Path,
) -> None:
    coverage = DataSliceProduct(tmp_path).coverage_for_request(
        calendar=crypto_calendar(),
        request=SliceRequest(
            market="crypto",
            source="hyperliquid",
            symbol="BTC",
            timeframe="1m",
        ),
        timestamps=[
            datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        ],
    )

    assert coverage.expected == 2
    assert coverage.actual == 2
    assert coverage.missing_ranges == []


def test_data_slice_product_coverage_report_requires_canonical_timestamps(tmp_path: Path) -> None:
    manifest = MarketDataManifest(
        source="hyperliquid",
        market="crypto",
        symbol="BTC",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
    )
    canonical = CanonicalSlice(
        request=SliceRequest(market="crypto", source="hyperliquid", symbol="BTC", timeframe="1m"),
        canonical_paths=[],
        manifest=manifest,
    )

    try:
        DataSliceProduct(tmp_path).coverage_report(calendar=crypto_calendar(), canonical=canonical)
    except ValueError as exc:
        assert "CanonicalSlice.timestamps" in str(exc)
    else:
        raise AssertionError("coverage_report should require declared canonical timestamps")


def test_slice_authority_status_view() -> None:
    status = SliceAuthorityStatus(status="live_hyperliquid_refresh")

    assert status.usable


def test_data_slice_product_authority_status_uses_declared_manifest_fields(tmp_path: Path) -> None:
    manifest = MarketDataManifest(
        source="hyperliquid",
        market="crypto",
        symbol="BTC",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        lineage={
            "authority_status": "live_hyperliquid_refresh",
            "authority_contract_id": "hyperliquid-read-only",
        },
        blocking_reasons=["coverage gap"],
    )

    status = DataSliceProduct(tmp_path).authority_status(manifest)

    assert status.status == "live_hyperliquid_refresh"
    assert status.source_contract_id == "hyperliquid-read-only"
    assert status.blocking_reasons == ["coverage gap"]


def test_data_slice_product_authority_status_falls_back_to_manifest_usability(
    tmp_path: Path,
) -> None:
    manifest = MarketDataManifest(
        source="hyperliquid",
        market="crypto",
        symbol="BTC",
        timeframe="1m",
        start_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        expected_bars=2,
        actual_bars=2,
    )

    status = DataSliceProduct(tmp_path).authority_status(manifest)

    assert status.status == "diagnostics_only"
    assert status.source_contract_id == ""
    assert status.blocking_reasons == []


def test_slice_product_validates_sorted_unique_utc_timestamps() -> None:
    assert timestamps_sorted_unique_utc([
        datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
    ])
    assert not timestamps_sorted_unique_utc([
        datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
    ])
