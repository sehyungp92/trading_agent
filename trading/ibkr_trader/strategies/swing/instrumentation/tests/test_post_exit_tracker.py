"""Tests for post-exit price tracker."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock


def test_backfill_computes_post_exit_movement(tmp_path):
    """Backfill should compute 1h and 4h price movement after exit."""
    from strategies.swing.instrumentation.src.post_exit_tracker import PostExitTracker

    # Write a completed trade
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    exit_time = datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc)
    trade = {
        "trade_id": "t1",
        "pair": "QQQ",
        "side": "LONG",
        "exit_price": 500.0,
        "exit_time": exit_time.isoformat(),
        "stage": "exit",
        "post_exit_1h_pct": None,
        "post_exit_4h_pct": None,
    }
    trade_file = trades_dir / f"trades_{exit_time.strftime('%Y-%m-%d')}.jsonl"
    trade_file.write_text(json.dumps(trade) + "\n")

    # Mock data provider that returns prices at +1h and +4h
    data_provider = MagicMock()
    data_provider.get_price_at.side_effect = lambda sym, ts: {
        exit_time + timedelta(hours=1): 505.0,
        exit_time + timedelta(hours=4): 510.0,
    }.get(ts)

    tracker = PostExitTracker(data_dir=str(tmp_path), data_provider=data_provider)
    results = tracker.run_backfill()

    assert len(results) == 1
    assert results[0]["post_exit_1h_pct"] == 1.0  # (505-500)/500 * 100
    assert results[0]["post_exit_4h_pct"] == 2.0  # (510-500)/500 * 100


def test_backfill_skips_already_filled(tmp_path):
    """Trades with post_exit data already filled should be skipped."""
    from strategies.swing.instrumentation.src.post_exit_tracker import PostExitTracker

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    trade = {
        "trade_id": "t1",
        "pair": "QQQ",
        "side": "LONG",
        "exit_price": 500.0,
        "exit_time": "2026-03-01T14:00:00+00:00",
        "stage": "exit",
        "post_exit_1h_pct": 1.0,
        "post_exit_4h_pct": 2.0,
    }
    trade_file = trades_dir / "trades_2026-03-01.jsonl"
    trade_file.write_text(json.dumps(trade) + "\n")

    tracker = PostExitTracker(data_dir=str(tmp_path), data_provider=MagicMock())
    results = tracker.run_backfill()
    assert len(results) == 0


def test_backfill_skips_recent_trades(tmp_path):
    """Trades exited less than 4h ago should not be backfilled yet."""
    from strategies.swing.instrumentation.src.post_exit_tracker import PostExitTracker

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    recent_exit = datetime.now(timezone.utc) - timedelta(hours=1)
    trade = {
        "trade_id": "t_recent",
        "pair": "QQQ",
        "side": "LONG",
        "exit_price": 500.0,
        "exit_time": recent_exit.isoformat(),
        "stage": "exit",
    }
    trade_file = trades_dir / f"trades_{recent_exit.strftime('%Y-%m-%d')}.jsonl"
    trade_file.write_text(json.dumps(trade) + "\n")

    tracker = PostExitTracker(data_dir=str(tmp_path), data_provider=MagicMock())
    results = tracker.run_backfill()
    assert len(results) == 0
