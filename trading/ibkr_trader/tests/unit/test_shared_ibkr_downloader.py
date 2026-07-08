from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backtests.scalp.data.downloader import _failed_alignment_messages, download_scalp_data
from backtests.scalp.data.preprocessing import load_bar_data
from backtests.momentum.data.downloader import derive_aligned_momentum_timeframes, check_aligned_momentum_timeframes
from backtests.shared.data.ibkr.alignment import check_symbol_alignment, compare_timeframe_alignment
from backtests.shared.data.ibkr.bars import (
    SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY,
    SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA,
    download_contract_bars,
    download_historical_bars,
    download_physical_futures_panama_bars,
)
from backtests.shared.data.ibkr.contracts import generate_quarterly_contracts, roll_schedule
from backtests.shared.data.ibkr.models import BarDownloadRequest, DownloadResult
from backtests.shared.data.ibkr.pacing import RequestPacer, request_weight
from backtests.shared.data.ibkr.stitch import stitch_panama
from backtests.shared.data.ibkr.store import write_compatibility_bars, write_parquet_atomic
from backtests.shared.data.ibkr.sync import sync_families
from backtests.shared.data.ibkr.ticks import ticks_to_frame


def test_quarterly_contract_generation_covers_two_year_window() -> None:
    contracts = generate_quarterly_contracts(
        "ES",
        start=date(2024, 4, 29),
        end=date(2026, 4, 29),
        as_of=date(2026, 4, 29),
    )

    months = [contract.yyyymm for contract in contracts]
    assert "202406" in months
    assert "202603" in months
    assert "202606" in months
    assert all(contract.symbol == "ES" for contract in contracts)
    assert all(contract.local_symbol.startswith("ES") for contract in contracts)
    assert roll_schedule(contracts)


def test_panama_stitch_adjusts_older_contracts_to_newer_price_level() -> None:
    old = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [10],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-13T21:00:00Z")]),
    )
    new = pd.DataFrame(
        {
            "open": [110.0],
            "high": [111.0],
            "low": [109.0],
            "close": [110.5],
            "volume": [20],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-16T00:00:00Z")]),
    )

    stitched = stitch_panama(
        {"202603": old, "202606": new},
        [(date(2026, 3, 16), "202603", "202606")],
        tick_size=0.25,
    )

    assert list(stitched["close"]) == [90.0, 110.5]
    assert list(stitched["open"]) == [90.0, 110.0]


def test_panama_stitch_keeps_isolated_contract_when_middle_contract_missing() -> None:
    frame_a = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-13T21:00:00Z")]),
    )
    frame_b = pd.DataFrame(
        {"open": [110.0], "high": [111.0], "low": [109.0], "close": [110.0], "volume": [20]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-16T00:00:00Z")]),
    )
    isolated = pd.DataFrame(
        {"open": [140.0], "high": [141.0], "low": [139.0], "close": [140.0], "volume": [30]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-12-16T00:00:00Z")]),
    )

    stitched = stitch_panama(
        {"202603": frame_a, "202606": frame_b, "202612": isolated},
        [
            (date(2026, 3, 16), "202603", "202606"),
            (date(2026, 6, 15), "202606", "202609"),
            (date(2026, 9, 14), "202609", "202612"),
        ],
        tick_size=0.25,
    )

    assert len(stitched) == 3
    assert stitched.iloc[-1]["close"] == 140.0


@pytest.mark.asyncio
async def test_request_pacer_applies_identical_cooldown_and_bid_ask_weight() -> None:
    now = 0.0

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    pacer = RequestPacer(
        min_interval_seconds=12,
        identical_cooldown_seconds=15,
        max_weight_per_window=60,
        sleep_fn=sleep,
        now_fn=lambda: now,
    )

    await pacer.wait(("NQ", "1m"), weight=request_weight("BID_ASK"))
    await pacer.wait(("NQ", "1m"), weight=request_weight("BID_ASK"))

    assert now == 15
    assert request_weight("BID_ASK") == 2
    assert request_weight("TRADES") == 1


def test_compatibility_files_load_through_existing_scalp_loader(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.25, 2.25],
            "volume": [100, 200],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-04-28T13:30:00Z"), pd.Timestamp("2026-04-28T13:31:00Z")]
        ),
    )
    write_compatibility_bars(df, tmp_path, "NQ", "1m")

    loaded = load_bar_data(tmp_path, "NQ")

    assert len(loaded["1m"]) == 2
    assert loaded["1m"].closes.tolist() == [1.25, 2.25]


def test_tick_converter_supports_trade_and_bid_ask_shapes() -> None:
    trade_frame = ticks_to_frame(
        [SimpleNamespace(time=datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc), price=100.25, size=3)],
        "TRADES",
    )
    bid_ask_frame = ticks_to_frame(
        [
            SimpleNamespace(
                time=datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc),
                priceBid=100.0,
                priceAsk=100.25,
                sizeBid=10,
                sizeAsk=12,
            )
        ],
        "BID_ASK",
    )

    assert trade_frame.iloc[0]["price"] == 100.25
    assert trade_frame.iloc[0]["size"] == 3
    assert bid_ask_frame.iloc[0]["bid_price"] == 100.0
    assert bid_ask_frame.iloc[0]["ask_size"] == 12


@pytest.mark.asyncio
async def test_scalp_downloader_dry_run_plans_without_ibkr(tmp_path: Path) -> None:
    results = await download_scalp_data(
        output_dir=tmp_path,
        symbols=["NQ"],
        years=2,
        dry_run=True,
        tick_mode="none",
        include_bid_ask=False,
    )

    messages = [message for result in results for message in result.messages]
    assert any("NQ" in message and "1m" in message for message in messages)
    assert any("IBKR bar requests" in message for message in messages)
    assert any("NQ 5m: derived from 1m" in message for message in messages)
    assert any("NQ 4h: derived from 1m" in message for message in messages)
    assert all(result.dry_run for result in results)


@pytest.mark.asyncio
async def test_scalp_downloader_defaults_ticks_to_required_nq_only(tmp_path: Path) -> None:
    results = await download_scalp_data(
        output_dir=tmp_path,
        symbols=["NQ", "ES"],
        years=2,
        dry_run=True,
        tick_mode="recent-gaps",
    )

    tick_messages = [
        message
        for result in results
        if result.what_to_show in {"TRADES", "BID_ASK"}
        for message in result.messages
        if "tick windows" in message
    ]

    assert any(message.startswith("NQ") for message in tick_messages)
    assert not any(message.startswith("ES") for message in tick_messages)


def test_failed_alignment_messages_detects_non_ok_alignment() -> None:
    messages = _failed_alignment_messages(
        [
            DownloadResult(symbol="NQ", what_to_show="ALIGNMENT", messages=["OK NQ 1m->5m: compared=10"]),
            DownloadResult(symbol="ES", what_to_show="ALIGNMENT", messages=["MISMATCH ES 1m->5m: compared=10"]),
            DownloadResult(symbol="NQ", what_to_show="TRADES", messages=["MISMATCH should not count"]),
        ]
    )

    assert messages == ["MISMATCH ES 1m->5m: compared=10"]


@pytest.mark.asyncio
async def test_latest_dry_run_includes_existing_interior_gap(tmp_path: Path) -> None:
    contract = generate_quarterly_contracts(
        "NQ",
        start=date(2026, 1, 1),
        end=date(2026, 3, 20),
        as_of=date(2026, 3, 1),
    )[1]
    output_path = tmp_path / "NQ_202603_1m.parquet"
    existing = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1, 1],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-01-02T00:00:00Z"), pd.Timestamp("2026-01-10T00:00:00Z")]
        ),
    )
    write_parquet_atomic(existing, output_path)

    result = await download_contract_bars(
        None,
        contract,
        timeframe="1m",
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 11, tzinfo=timezone.utc),
        output_path=output_path,
        dry_run=True,
        latest_only=True,
    )

    assert result.dry_run
    assert "IBKR bar requests" in result.messages[0]
    assert int(result.messages[0].split(": ")[1].split()[0]) > 1


@pytest.mark.asyncio
async def test_generic_futures_download_requires_explicit_legacy_flag(tmp_path: Path) -> None:
    request = BarDownloadRequest(
        symbol="NQ",
        timeframe="5m",
        sec_type="FUT",
        exchange="CME",
        trading_class="NQ",
        end=datetime(2026, 4, 28, tzinfo=timezone.utc),
        duration="1 M",
    )

    with pytest.raises(ValueError, match="download_physical_futures_panama_bars"):
        await download_historical_bars(
            None,
            request,
            output_path=tmp_path / "NQ_5m.parquet",
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_legacy_contfuture_dry_run_is_marked_diagnostic(tmp_path: Path) -> None:
    request = BarDownloadRequest(
        symbol="NQ",
        timeframe="5m",
        sec_type="FUT",
        exchange="CME",
        trading_class="NQ",
        end=datetime(2026, 4, 28, tzinfo=timezone.utc),
        duration="1 M",
        allow_contfuture_legacy=True,
    )

    result = await download_historical_bars(
        None,
        request,
        output_path=tmp_path / "NQ_5m.parquet",
        dry_run=True,
    )

    assert result.metadata["source_kind"] == SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY
    assert result.metadata["usable_for_authoritative_validation"] is False
    assert "legacy ContFuture diagnostic" in result.messages[0]


@pytest.mark.asyncio
async def test_legacy_contfuture_output_writes_diagnostic_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backtests.shared.data.ibkr import bars as bars_module

    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.5],
            "low": [0.5],
            "close": [1.25],
            "volume": [100],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-28T13:30:00Z")]),
    )

    async def fake_download_contfuture_diagnostic(*args, **kwargs) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(
        bars_module,
        "download_contfuture_diagnostic",
        fake_download_contfuture_diagnostic,
    )
    request = BarDownloadRequest(
        symbol="NQ",
        timeframe="5m",
        sec_type="FUT",
        exchange="CME",
        trading_class="NQ",
        end=datetime(2026, 4, 28, tzinfo=timezone.utc),
        duration="1 M",
        allow_contfuture_legacy=True,
    )

    result = await download_historical_bars(
        None,
        request,
        output_path=tmp_path / "NQ_5m.parquet",
    )
    manifest = json.loads((tmp_path / "NQ_5m.manifest.json").read_text(encoding="utf-8"))

    assert result.metadata["source_kind"] == SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY
    assert manifest["source_kind"] == SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY
    assert manifest["usable_for_authoritative_validation"] is False


@pytest.mark.asyncio
async def test_physical_futures_panama_dry_run_is_authoritative(tmp_path: Path) -> None:
    request = BarDownloadRequest(
        symbol="NQ",
        timeframe="5m",
        sec_type="FUT",
        exchange="CME",
        trading_class="NQ",
        end=datetime(2026, 4, 28, tzinfo=timezone.utc),
        duration="1 M",
        family="momentum",
    )

    result = await download_physical_futures_panama_bars(
        None,
        request,
        output_path=tmp_path / "NQ_5m.parquet",
        dry_run=True,
    )

    assert result.metadata["source_kind"] == SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA
    assert result.metadata["usable_for_authoritative_validation"] is True
    assert "physical futures/Panama" in result.messages[0]


@pytest.mark.asyncio
async def test_momentum_sync_dry_run_uses_physical_futures_authority() -> None:
    results = await sync_families(families=["momentum"], years=1, dry_run=True)
    source_kinds = [result.metadata.get("source_kind") for result in results if result.metadata]

    assert SOURCE_KIND_IBKR_PHYSICAL_FUTURES_PANAMA in source_kinds
    assert SOURCE_KIND_IBKR_CONT_FUTURE_LEGACY not in source_kinds


def test_alignment_checker_passes_when_target_is_derived_from_base(tmp_path: Path) -> None:
    one_min = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.1, 2.1, 3.1, 4.1, 5.1],
            "low": [0.9, 1.9, 2.9, 3.9, 4.9],
            "close": [1.05, 2.05, 3.05, 4.05, 5.05],
            "volume": [10, 20, 30, 40, 50],
        },
        index=pd.date_range("2026-04-28T13:31:00Z", periods=5, freq="1min"),
    )
    five_min = pd.DataFrame(
        {
            "open": [1.0],
            "high": [5.1],
            "low": [0.9],
            "close": [5.05],
            "volume": [150],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-28T13:35:00Z")]),
    )
    write_parquet_atomic(one_min, tmp_path / "NQ_1m.parquet")
    write_parquet_atomic(five_min, tmp_path / "NQ_5m.parquet")

    result = check_symbol_alignment(tmp_path, "NQ", target_timeframes=("5m",))[0]

    assert result.ok
    assert result.compared_rows == 1


def test_alignment_checker_flags_mismatched_target() -> None:
    base = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=pd.date_range("2026-04-28T13:31:00Z", periods=5, freq="1min"),
    )
    target = pd.DataFrame(
        {"open": [1.0], "high": [5.0], "low": [1.0], "close": [6.0], "volume": [5]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-28T13:35:00Z")]),
    )

    result = compare_timeframe_alignment(
        symbol="NQ",
        base=base,
        target=target,
        base_timeframe="1m",
        target_timeframe="5m",
    )

    assert not result.ok
    assert result.mismatched_rows == 1


def test_momentum_derivation_writes_strategy_aligned_files(tmp_path: Path) -> None:
    five_min = pd.DataFrame(
        {
            "open": list(range(12)),
            "high": [value + 0.5 for value in range(12)],
            "low": [value - 0.5 for value in range(12)],
            "close": [value + 0.25 for value in range(12)],
            "volume": [10] * 12,
        },
        index=pd.date_range("2026-04-28T13:30:00Z", periods=12, freq="5min"),
    )
    write_parquet_atomic(five_min, tmp_path / "NQ_5m.parquet")

    paths = derive_aligned_momentum_timeframes("NQ", tmp_path, targets=("15m", "1h", "1d"))
    results = check_aligned_momentum_timeframes("NQ", tmp_path, targets=("15m", "1h", "1d"))

    assert set(paths) == {"15m", "1h", "1d"}
    assert all(result.ok for result in results)


def test_momentum_loader_ignores_mismatched_direct_daily(tmp_path: Path) -> None:
    from backtests.momentum.cli import _load_nqdtc_data

    five_min = pd.DataFrame(
        {
            "open": list(range(120)),
            "high": [value + 0.5 for value in range(120)],
            "low": [value - 0.5 for value in range(120)],
            "close": [value + 0.25 for value in range(120)],
            "volume": [10] * 120,
        },
        index=pd.date_range("2026-04-28T13:30:00Z", periods=120, freq="5min"),
    )
    mismatched_daily = pd.DataFrame(
        {"open": [999.0], "high": [999.0], "low": [999.0], "close": [999.0], "volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-28T00:00:00Z")]),
    )
    write_parquet_atomic(five_min, tmp_path / "NQ_5m.parquet")
    write_parquet_atomic(mismatched_daily, tmp_path / "NQ_1d.parquet")

    loaded = _load_nqdtc_data("NQ", tmp_path)

    assert loaded["daily"].closes.tolist() == [77.25]
