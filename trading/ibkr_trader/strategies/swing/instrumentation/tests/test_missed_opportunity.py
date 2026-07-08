"""Tests for MissedOpportunityLogger."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from strategies.swing.instrumentation.src.missed_opportunity import MissedOpportunityLogger
from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot


class TestMissedOpportunity:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
            "market_snapshots": {"interval_seconds": 60, "symbols": []},
        }
        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = MarketSnapshot(
            snapshot_id="test", symbol="QQQ", timestamp="2026-03-01T10:00:00Z",
            bid=500, ask=500.10, mid=500.05, spread_bps=2.0, last_trade_price=500.05,
            atr_14=5.0, volume_24h=50000000,
        )
        self.logger = MissedOpportunityLogger(self.config, self.snap_service)

    def test_log_missed_creates_event(self):
        event = self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="EMA cross bullish", signal_id="ema_cross_bull",
            signal_strength=0.75, blocked_by="volume_filter",
            block_reason="Volume below threshold",
        )
        assert event.pair == "QQQ"
        assert event.blocked_by == "volume_filter"
        assert event.backfill_status == "pending"

    def test_event_written_to_file(self):
        self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="test_filter",
        )
        files = list(Path(self.tmpdir).joinpath("missed").glob("*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text().strip())
        assert data["blocked_by"] == "test_filter"

    def test_assumption_tags_present(self):
        event = self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="quality_gate",
        )
        assert len(event.assumption_tags) > 0
        assert any("fill" in tag for tag in event.assumption_tags)
        assert any("fees" in tag for tag in event.assumption_tags)

    def test_simulation_policy_included(self):
        event = self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="quality_gate",
        )
        assert event.simulation_policy is not None
        assert "entry_fill_model" in event.simulation_policy

    def test_hypothetical_entry_price_set(self):
        event = self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="quality_gate",
        )
        assert event.hypothetical_entry_price > 0

    def test_failure_does_not_crash(self):
        self.snap_service.capture_now.side_effect = Exception("broken")
        event = self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="test",
        )
        # Should return degraded event, not crash
        assert event is not None

    def test_pending_backfills_queued(self):
        self.logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="test",
        )
        assert len(self.logger._pending_backfills) == 1
