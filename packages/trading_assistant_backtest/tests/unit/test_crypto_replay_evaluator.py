from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    DataBundleSlice,
    DataBundleStatus,
)
from trading_assistant_backtest.strategies.crypto.replay_evaluator import _load_bundle_frames


def test_crypto_replay_loader_skips_funding_context_slices(tmp_path: Path) -> None:
    data_root = tmp_path / "data_repo"
    bundle_root = tmp_path / "bundle"
    candle_path = data_root / "data" / "canonical" / "btc_15m.parquet"
    funding_path = data_root / "data" / "canonical" / "btc_funding_1h.parquet"
    candle_path.parent.mkdir(parents=True)
    bundle_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp_utc": datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 10.0,
            },
            {
                "timestamp_utc": datetime(2026, 5, 1, 0, 15, tzinfo=UTC),
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 11.0,
            },
        ]
    ).to_parquet(candle_path, engine="pyarrow")
    pd.DataFrame(
        [
            {
                "timestamp_utc": datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
                "rate": 0.0001,
            }
        ]
    ).to_parquet(funding_path, engine="pyarrow")
    (bundle_root / "slice_index.json").write_text(
        json.dumps(
            {
                "slices": [
                    {
                        "manifest_id": "btc-15m",
                        "canonical_paths": ["data/canonical/btc_15m.parquet"],
                    },
                    {
                        "manifest_id": "btc-funding",
                        "canonical_paths": ["data/canonical/btc_funding_1h.parquet"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = DataBundleManifest(
        data_repo_path=str(data_root),
        data_repo_commit_sha="data-sha",
        slice_manifests=[
            DataBundleSlice(
                manifest_path="btc-15m.json",
                manifest_id="btc-15m",
                source="hyperliquid",
                market="crypto_perp",
                symbol="BTC",
                timeframe="15m",
                checksum="candle-sha",
                calendar="crypto_utc_24_7_v1",
                authoritative=True,
            ),
            DataBundleSlice(
                manifest_path="btc-funding.json",
                manifest_id="btc-funding",
                source="hyperliquid",
                market="crypto_perp",
                symbol="BTC",
                timeframe="funding_1h",
                checksum="funding-sha",
                calendar="crypto_utc_24_7_v1",
                authoritative=True,
            ),
        ],
        calendars=["crypto_utc_24_7_v1"],
        fee_model_version="fees_v1",
        slippage_model_version="slippage_v1",
        adjustment_policy="crypto_raw_perp_policy_v1",
        status=DataBundleStatus.AUTHORITATIVE,
    )

    frames = _load_bundle_frames(manifest, bundle_root / "data_bundle_manifest.json")

    assert sorted(frames) == [("BTC", "15m")]
    assert frames[("BTC", "15m")]["close"].tolist() == [101.0, 102.0]
