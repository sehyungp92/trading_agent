from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from backtests.regime.data.downloader import _filter_completed_daily_prices
from regime.seed_manifest import (
    SEED_MANIFEST_NAME,
    bootstrap_seed_data_dir,
    validate_seed_data_dir,
    write_seed_manifest,
)


def _write_seed(data_dir, start: str, periods: int) -> None:
    dates = pd.date_range(start, periods=periods, freq="D")
    macro_df = pd.DataFrame(
        {"GROWTH": [-200_000.0] * periods, "INFLATION": [2.0] * periods},
        index=dates,
    )
    market_df = pd.DataFrame(
        {
            "VIX": [18.0] * periods,
            "SPREAD": [2.5] * periods,
            "SLOPE_10Y2Y": [0.5] * periods,
            "REAL_RATE_10Y": [1.0] * periods,
            "DBC": [0.0] * periods,
        },
        index=dates,
    )
    strat_ret_df = pd.DataFrame(
        {
            "SPY": [0.001] * periods,
            "EFA": [0.001] * periods,
            "TLT": [0.0005] * periods,
            "GLD": [0.0002] * periods,
            "CASH": [0.0] * periods,
        },
        index=dates,
    )
    macro_df.to_parquet(data_dir / "macro_df.parquet")
    market_df.to_parquet(data_dir / "market_df.parquet")
    strat_ret_df.to_parquet(data_dir / "strat_ret_df.parquet")


def test_seed_manifest_validates_required_files_hashes_and_as_of(tmp_path) -> None:
    _write_seed(tmp_path, "2026-01-01", 10)

    manifest = write_seed_manifest(
        tmp_path,
        generated_by="unit-test",
        source_versions={"source": "unit"},
    )
    ok, status, loaded = validate_seed_data_dir(
        tmp_path,
        require_manifest=True,
        validate_hashes=True,
    )

    assert ok
    assert status == "seed_manifest=ok:data_as_of=2026-01-10"
    assert loaded is not None
    assert manifest["data_as_of"] == "2026-01-10"
    assert manifest["row_counts"]["strat_ret_df.parquet"] == 10


def test_seed_manifest_detects_stale_hash_after_file_change(tmp_path) -> None:
    _write_seed(tmp_path, "2026-01-01", 10)
    write_seed_manifest(tmp_path, generated_by="unit-test")

    strat = pd.read_parquet(tmp_path / "strat_ret_df.parquet")
    strat.loc[strat.index[-1], "SPY"] = 0.02
    strat.to_parquet(tmp_path / "strat_ret_df.parquet")

    ok, status, _ = validate_seed_data_dir(
        tmp_path,
        require_manifest=True,
        validate_hashes=True,
    )

    assert not ok
    assert "sha256_mismatch" in status


def test_bootstrap_copies_fresher_image_seed_into_runtime_cache(tmp_path) -> None:
    seed_dir = tmp_path / "seed"
    target_dir = tmp_path / "runtime"
    seed_dir.mkdir()
    target_dir.mkdir()
    _write_seed(seed_dir, "2026-01-01", 20)
    _write_seed(target_dir, "2025-12-01", 5)
    write_seed_manifest(seed_dir, generated_by="seed")
    write_seed_manifest(target_dir, generated_by="runtime")

    status = bootstrap_seed_data_dir(target_dir, seed_dir)
    copied_manifest = json.loads((target_dir / SEED_MANIFEST_NAME).read_text())

    assert status == "seed_bootstrap=copied:data_as_of=2026-01-20"
    assert copied_manifest["data_as_of"] == "2026-01-20"


def test_offline_seed_drops_current_session_before_daily_bar_complete() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2026-05-08", "2026-05-11"]),
    )

    filtered = _filter_completed_daily_prices(
        prices,
        now=datetime(2026, 5, 11, 11, 38, tzinfo=ZoneInfo("America/New_York")),
    )

    assert filtered.index.max() == pd.Timestamp("2026-05-08")
