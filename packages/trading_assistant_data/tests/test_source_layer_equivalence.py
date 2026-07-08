from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from trading_assistant_data.sources.hyperliquid.sync import sync_hyperliquid
from trading_assistant_data.sources.ibkr.live_read_only import _bar_frame, _bars_to_dataframe
from trading_assistant_data.sources.kis.krx_read_only import (
    KrxRefreshRequest,
    _aggregate_intraday_frame,
    _kis_payload_to_frame,
)
from trading_assistant_data.sources.lrs.export_daily import export_table as export_lrs_table


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_hyperliquid_bot_store_and_central_sync_write_same_decision_rows(tmp_path: Path) -> None:
    _prepend_path(REPO_ROOT / "trading" / "crypto_trader" / "src")
    from crypto_trader.data.downloader import HyperliquidDownloader as BotHyperliquidDownloader
    from crypto_trader.data.store import ParquetStore

    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc)
    candle_rows = [
        _hl_candle("2026-05-01T00:00:00Z", 100.0),
        _hl_candle("2026-05-01T00:01:00Z", 101.0),
        _hl_candle("2026-05-01T00:01:00Z", 101.5),
        _hl_candle("2026-05-01T00:02:00Z", 102.0),
    ]
    funding_rows = [
        _hl_funding("2026-05-01T00:00:00Z", "0.0001"),
        _hl_funding("2026-05-01T01:00:00Z", "0.0002"),
        _hl_funding("2026-05-01T01:00:00Z", "0.00025"),
    ]

    bot_store = ParquetStore(tmp_path / "bot_crypto")
    bot_downloader = BotHyperliquidDownloader(store=bot_store, rate_limit=0.0)
    bot_downloader._info = _BotHyperliquidInfo(candle_rows, funding_rows)
    bot_store.save_candles(
        "BTC",
        "1m",
        bot_downloader.download_candles("BTC", "1m", _ms(start), _ms(end)),
    )
    bot_store.save_funding(
        "BTC",
        bot_downloader.download_funding("BTC", _ms(start), _ms(end)),
    )

    sync_hyperliquid(
        repo_root=tmp_path / "central_crypto",
        symbols=["BTC"],
        intervals=["1m"],
        start=start,
        end=end,
        funding=True,
        downloader=_CentralHyperliquidClient(candle_rows, funding_rows),
    )

    central_root = tmp_path / "central_crypto"
    central_candles = pd.read_parquet(
        central_root
        / "data/canonical/bars/market=crypto_perp/source=hyperliquid/kind=trades"
        / "symbol=BTC/timeframe=1m/year=2026/month=05/part.parquet"
    )
    central_funding = pd.read_parquet(
        central_root
        / "data/canonical/funding/market=crypto_perp/source=hyperliquid"
        / "symbol=BTC/year=2026/month=05/part.parquet"
    )

    pd.testing.assert_frame_equal(
        bot_store.load_candles("BTC", "1m"),
        _central_crypto_candle_core(central_candles),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bot_store.load_funding("BTC"),
        _central_crypto_funding_core(central_funding),
        check_dtype=False,
    )


def test_hyperliquid_existing_overlap_precedence_matches_bot_store(tmp_path: Path) -> None:
    _prepend_path(REPO_ROOT / "trading" / "crypto_trader" / "src")
    from crypto_trader.data.downloader import HyperliquidDownloader as BotHyperliquidDownloader
    from crypto_trader.data.store import ParquetStore

    seed_candles = [
        _hl_candle("2026-05-01T00:00:00Z", 100.0),
        _hl_candle("2026-05-01T00:01:00Z", 101.0),
    ]
    update_candles = [
        _hl_candle("2026-05-01T00:01:00Z", 101.5),
        _hl_candle("2026-05-01T00:02:00Z", 102.0),
    ]

    bot_store = ParquetStore(tmp_path / "bot_overlap")
    bot_downloader = BotHyperliquidDownloader(store=bot_store, rate_limit=0.0)
    bot_store.save_candles("BTC", "1m", bot_downloader._candles_to_df(seed_candles))
    bot_store.save_candles("BTC", "1m", bot_downloader._candles_to_df(update_candles))

    central_root = tmp_path / "central_overlap"
    sync_hyperliquid(
        repo_root=central_root,
        symbols=["BTC"],
        intervals=["1m"],
        start=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        downloader=_CentralHyperliquidClient(seed_candles, []),
    )
    sync_hyperliquid(
        repo_root=central_root,
        symbols=["BTC"],
        intervals=["1m"],
        start=datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 0, 2, tzinfo=timezone.utc),
        downloader=_CentralHyperliquidClient(update_candles, []),
    )

    central_candles = pd.read_parquet(
        central_root
        / "data/canonical/bars/market=crypto_perp/source=hyperliquid/kind=trades"
        / "symbol=BTC/timeframe=1m/year=2026/month=05/part.parquet"
    )

    pd.testing.assert_frame_equal(
        bot_store.load_candles("BTC", "1m"),
        _central_crypto_candle_core(central_candles),
        check_dtype=False,
    )


def test_ibkr_bot_bar_normalizer_and_central_provider_normalizer_match() -> None:
    _prepend_path(REPO_ROOT / "trading" / "ibkr_trader")
    _prepend_path(REPO_ROOT)
    from trading.ibkr_trader.backtests.shared.data.ibkr.bars import bars_to_frame

    bars = [
        _IBBar(datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc), 100.0, 101.0, 99.5, 100.5, 1000),
        _IBBar(datetime(2026, 5, 1, 13, 35, tzinfo=timezone.utc), 100.5, 102.0, 100.0, 101.5, 1200),
    ]

    legacy = bars_to_frame(bars)
    central_raw = _bars_to_dataframe(SimpleNamespace(), bars).rename(columns={"date": "timestamp_utc"})
    central = _bar_frame(
        central_raw.assign(
            source_contract="QQQ",
            source_conid="det-QQQ",
            source_local_symbol="QQQ",
            source_primary_exchange="NASDAQ",
            contract_resolution_method="deterministic_fixture",
        )
    )

    pd.testing.assert_frame_equal(
        _legacy_ibkr_core(legacy),
        _central_ohlcv_core(central),
        check_dtype=False,
    )


def test_kis_intraday_bot_parser_and_central_read_only_parser_match() -> None:
    _prepend_path(REPO_ROOT / "trading" / "k_stock_trader")
    _prepend_path(REPO_ROOT)
    from trading.k_stock_trader.scripts import update_kis_intraday as legacy_kis

    rows = [
        _kis_row("090000", 70_000, 10),
        _kis_row("090100", 70_050, 11),
        _kis_row("090200", 70_100, 12),
        _kis_row("090300", 70_150, 13),
        _kis_row("090400", 70_200, 14),
    ]
    payload = {"output2": rows}
    request_1m = KrxRefreshRequest(
        symbol="005930",
        timeframe="1m",
        start=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 0, 4, tzinfo=timezone.utc),
    )
    request_5m = KrxRefreshRequest(
        symbol="005930",
        timeframe="5m",
        start=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
    )

    central_1m = _kis_payload_to_frame(payload, request_1m)
    legacy_1m = legacy_kis.normalize_bar_frame(
        pd.DataFrame(
            row
            for row in (legacy_kis._normalize_kis_minute_row(item) for item in rows)
            if row is not None
        )
    )
    central_5m = _aggregate_intraday_frame(central_1m, request_5m)
    legacy_5m = legacy_kis.aggregate_timeframe(legacy_1m, "5m")

    pd.testing.assert_frame_equal(
        _legacy_kis_core(legacy_1m),
        _central_ohlcv_core(central_1m),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        _legacy_kis_core(legacy_5m),
        _central_ohlcv_core(central_5m),
        check_dtype=False,
    )


def test_lrs_central_table_export_matches_k_stock_full_mirror_tables(tmp_path: Path) -> None:
    _prepend_path(REPO_ROOT / "trading" / "k_stock_trader")
    _prepend_path(REPO_ROOT)
    from trading.k_stock_trader.strategy_common.daily_lrs_parquet import export_lrs_sqlite_to_parquet

    db_path = tmp_path / "lrs.db"
    _write_lrs_fixture(db_path)
    bot_root = tmp_path / "bot_lrs"
    central_root = tmp_path / "central_lrs"

    export_lrs_sqlite_to_parquet(db_path, bot_root, universe=["005930"], source_label="fixture")
    for table in ("daily_ohlcv", "daily_flow"):
        export_lrs_table(db_path, table, central_root / f"{table}.parquet")
        pd.testing.assert_frame_equal(
            pd.read_parquet(bot_root / "tables" / f"{table}.parquet"),
            pd.read_parquet(central_root / f"{table}.parquet"),
            check_dtype=False,
        )


def _prepend_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ms(value: datetime | str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _hl_candle(timestamp: str, close: float) -> dict[str, Any]:
    open_ = close - 0.5
    return {
        "t": _ms(timestamp),
        "o": str(open_),
        "h": str(close + 1.0),
        "l": str(open_ - 1.0),
        "c": str(close),
        "v": str(close * 10.0),
    }


def _hl_funding(timestamp: str, rate: str) -> dict[str, Any]:
    return {"time": _ms(timestamp), "fundingRate": rate}


class _BotHyperliquidInfo:
    def __init__(self, candle_rows: list[dict[str, Any]], funding_rows: list[dict[str, Any]]) -> None:
        self.candle_rows = candle_rows
        self.funding_rows = funding_rows

    def candles_snapshot(self, _coin: str, _interval: str, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
        return [row for row in self.candle_rows if start_ts <= int(row["t"]) <= end_ts]

    def funding_history(self, _coin: str, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
        return [row for row in self.funding_rows if start_ts <= int(row["time"]) <= end_ts]


class _CentralHyperliquidClient:
    def __init__(self, candle_rows: list[dict[str, Any]], funding_rows: list[dict[str, Any]]) -> None:
        self.candle_rows = candle_rows
        self.funding_rows = funding_rows

    def candles(self, _symbol: str, _interval: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [row for row in self.candle_rows if _ms(start) <= int(row["t"]) <= _ms(end)]

    def funding(self, _symbol: str, *, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return [row for row in self.funding_rows if _ms(start) <= int(row["time"]) <= _ms(end)]


def _central_crypto_candle_core(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(ts=frame["source_ts_ms"])
        .loc[:, ["ts", "open", "high", "low", "close", "volume"]]
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _central_crypto_funding_core(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(ts=frame["source_ts_ms"])
        .rename(columns={"rate": "rate"})
        .loc[:, ["ts", "rate"]]
        .sort_values("ts")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class _IBBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def _legacy_ibkr_core(frame: pd.DataFrame) -> pd.DataFrame:
    core = frame.reset_index().rename(columns={"time": "timestamp_utc"})
    return _central_ohlcv_core(core)


def _central_ohlcv_core(frame: pd.DataFrame) -> pd.DataFrame:
    core = frame.loc[:, ["timestamp_utc", "open", "high", "low", "close", "volume"]].copy()
    core["timestamp_utc"] = pd.to_datetime(core["timestamp_utc"], utc=True)
    return core.sort_values("timestamp_utc").reset_index(drop=True)


def _kis_row(hour: str, price: int, volume: int) -> dict[str, str]:
    return {
        "stck_bsop_date": "20260504",
        "stck_cntg_hour": hour,
        "stck_oprc": str(price),
        "stck_hgpr": str(price + 100),
        "stck_lwpr": str(price - 100),
        "stck_prpr": str(price + 50),
        "cntg_vol": str(volume),
    }


def _legacy_kis_core(frame: pd.DataFrame) -> pd.DataFrame:
    core = frame.rename(columns={"timestamp": "timestamp_utc"})
    return _central_ohlcv_core(core)


def _write_lrs_fixture(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE daily_ohlcv (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"
        )
        conn.execute(
            "CREATE TABLE daily_flow (ticker TEXT, date TEXT, foreign_net REAL, inst_net REAL)"
        )
        conn.executemany(
            "INSERT INTO daily_ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("005930", "2026-05-04", 70000, 70500, 69800, 70200, 1000000),
                ("005930", "2026-05-05", 70200, 71000, 70100, 70800, 1100000),
            ],
        )
        conn.executemany(
            "INSERT INTO daily_flow VALUES (?, ?, ?, ?)",
            [
                ("005930", "2026-05-04", 1000, -500),
                ("005930", "2026-05-05", 1200, -200),
            ],
        )
