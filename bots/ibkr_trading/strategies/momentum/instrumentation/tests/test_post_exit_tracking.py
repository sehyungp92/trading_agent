import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from strategies.momentum.instrumentation.src.trade_logger import TradeEvent, TradeLogger
from strategies.momentum.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot


def _mock_snapshot_service():
    service = MagicMock(spec=MarketSnapshotService)
    service.capture_now.return_value = MarketSnapshot(
        snapshot_id="test_snap", symbol="NQ",
        timestamp="2026-03-01T10:00:00Z",
        bid=20500.0, ask=20500.50, mid=20500.25, spread_bps=0.24,
        last_trade_price=20500.25, atr_14=85.0,
    )
    return service


def test_trade_event_has_post_exit_fields():
    te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
    assert te.post_exit_1h_price is None
    assert te.post_exit_4h_price is None
    assert te.post_exit_1h_move_pct is None
    assert te.post_exit_4h_move_pct is None
    assert te.post_exit_backfill_status == "pending"


def test_log_exit_queues_backfill(tmp_path):
    config = {"bot_id": "test", "data_dir": str(tmp_path), "data_source_id": "test"}
    snapshot_svc = _mock_snapshot_service()
    tl = TradeLogger(config, snapshot_svc)

    # Log entry then exit
    tl.log_entry(
        trade_id="t1", pair="NQ", side="LONG", entry_price=21000.0,
        position_size=5, position_size_quote=2100000.0,
        entry_signal="test", entry_signal_id="s1", entry_signal_strength=0.5,
        active_filters=[], passed_filters=[], strategy_params={},
    )
    tl.log_exit(trade_id="t1", exit_price=21050.0, exit_reason="TRAILING_STOP")

    assert len(tl._pending_exit_backfills) == 1
    item = tl._pending_exit_backfills[0]
    assert item["trade_id"] == "t1"
    assert item["pair"] == "NQ"
    assert item["side"] == "LONG"
    assert item["exit_price"] == 21050.0


def test_run_post_exit_backfill_computes_move(tmp_path):
    config = {"bot_id": "test", "data_dir": str(tmp_path), "data_source_id": "test"}
    snapshot_svc = _mock_snapshot_service()
    tl = TradeLogger(config, snapshot_svc)

    exit_time = datetime.now(timezone.utc) - timedelta(hours=5)
    tl._pending_exit_backfills.append({
        "trade_id": "t1",
        "pair": "NQ",
        "side": "LONG",
        "exit_price": 21000.0,
        "exit_time": exit_time,
        "file_date": exit_time.strftime("%Y-%m-%d"),
    })

    # Write a fake exit event to the trades file
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    filepath = trades_dir / f"trades_{exit_time.strftime('%Y-%m-%d')}.jsonl"
    fake_exit = {"trade_id": "t1", "stage": "exit", "exit_price": 21000.0}
    filepath.write_text(json.dumps(fake_exit) + "\n")

    # Mock data provider with candles
    mock_dp = MagicMock()
    base_ms = int(exit_time.timestamp() * 1000)
    candles = []
    for i in range(60):
        ts_ms = base_ms + i * 300_000  # 5-min intervals
        candles.append([ts_ms, 21000 + i * 2, 21010 + i * 2, 20990 + i * 2, 21005 + i * 2, 100])
    mock_dp.get_ohlcv.return_value = candles

    tl.run_post_exit_backfill(mock_dp)

    assert len(tl._pending_exit_backfills) == 0  # completed

    # Verify the JSONL was updated
    updated = json.loads(filepath.read_text().strip())
    assert updated["post_exit_backfill_status"] == "complete"
    assert updated["post_exit_1h_price"] is not None
    assert updated["post_exit_4h_price"] is not None


def test_run_post_exit_backfill_skips_recent(tmp_path):
    """Items less than 4h old should be skipped."""
    config = {"bot_id": "test", "data_dir": str(tmp_path), "data_source_id": "test"}
    snapshot_svc = _mock_snapshot_service()
    tl = TradeLogger(config, snapshot_svc)

    exit_time = datetime.now(timezone.utc) - timedelta(hours=2)
    tl._pending_exit_backfills.append({
        "trade_id": "t1",
        "pair": "NQ",
        "side": "LONG",
        "exit_price": 21000.0,
        "exit_time": exit_time,
        "file_date": exit_time.strftime("%Y-%m-%d"),
    })

    mock_dp = MagicMock()
    tl.run_post_exit_backfill(mock_dp)

    # Should not have been processed
    assert len(tl._pending_exit_backfills) == 1
    mock_dp.get_ohlcv.assert_not_called()


def test_run_post_exit_backfill_short_move_pct(tmp_path):
    """Short side move_pct should be inverted (exit - post) / exit."""
    config = {"bot_id": "test", "data_dir": str(tmp_path), "data_source_id": "test"}
    snapshot_svc = _mock_snapshot_service()
    tl = TradeLogger(config, snapshot_svc)

    exit_time = datetime.now(timezone.utc) - timedelta(hours=5)
    tl._pending_exit_backfills.append({
        "trade_id": "t1",
        "pair": "NQ",
        "side": "SHORT",
        "exit_price": 21000.0,
        "exit_time": exit_time,
        "file_date": exit_time.strftime("%Y-%m-%d"),
    })

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    filepath = trades_dir / f"trades_{exit_time.strftime('%Y-%m-%d')}.jsonl"
    fake_exit = {"trade_id": "t1", "stage": "exit", "exit_price": 21000.0}
    filepath.write_text(json.dumps(fake_exit) + "\n")

    # Price goes DOWN after exit — good for short (positive move_pct)
    mock_dp = MagicMock()
    base_ms = int(exit_time.timestamp() * 1000)
    candles = []
    for i in range(60):
        ts_ms = base_ms + i * 300_000
        # Price drops from 21000 to ~20880 over 5h
        close_price = 21000.0 - i * 2
        candles.append([ts_ms, close_price + 5, close_price + 10, close_price - 5, close_price, 100])
    mock_dp.get_ohlcv.return_value = candles

    tl.run_post_exit_backfill(mock_dp)

    assert len(tl._pending_exit_backfills) == 0
    updated = json.loads(filepath.read_text().strip())
    # For SHORT, price going down means positive move_pct
    assert updated["post_exit_1h_move_pct"] > 0
    assert updated["post_exit_4h_move_pct"] > 0


def test_update_trade_event_only_updates_exit_stage(tmp_path):
    """_update_trade_event should only update exit events, not entry events."""
    config = {"bot_id": "test", "data_dir": str(tmp_path), "data_source_id": "test"}
    snapshot_svc = _mock_snapshot_service()
    tl = TradeLogger(config, snapshot_svc)

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    filepath = trades_dir / "trades_2026-03-01.jsonl"

    entry_event = {"trade_id": "t1", "stage": "entry", "entry_price": 21000.0}
    exit_event = {"trade_id": "t1", "stage": "exit", "exit_price": 21050.0}
    filepath.write_text(json.dumps(entry_event) + "\n" + json.dumps(exit_event) + "\n")

    tl._update_trade_event("t1", "2026-03-01", {"post_exit_backfill_status": "complete"})

    lines = filepath.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    exit_ = json.loads(lines[1])
    assert "post_exit_backfill_status" not in entry
    assert exit_["post_exit_backfill_status"] == "complete"
