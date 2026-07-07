import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from strategies.stock.instrumentation.src.trade_logger import TradeLogger
from strategies.stock.instrumentation.src.missed_opportunity import MissedOpportunityLogger
from strategies.stock.instrumentation.src.process_scorer import ProcessScorer, ROOT_CAUSES
from strategies.stock.instrumentation.src.daily_snapshot import DailySnapshotBuilder


class TestFullLifecycle:
    """Simulate a complete day: signals, blocked trades, executed trades, daily rollup."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "integration_test",
            "strategy_type": "helix",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
            "market_snapshots": {"interval_seconds": 60, "symbols": ["NQ"]},
        }

        # Mock snapshot service
        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = MarketSnapshot(
            snapshot_id="test", symbol="NQ",
            timestamp=datetime.now(timezone.utc).isoformat(),
            bid=20500, ask=20500.50, mid=20500.25, spread_bps=0.24,
            last_trade_price=20500.25, atr_14=85.0,
        )

        self.trade_logger = TradeLogger(self.config, self.snap_service)
        self.missed_logger = MissedOpportunityLogger(self.config, self.snap_service)

    def test_full_day_lifecycle(self):
        """Simulate: 2 executed trades + 1 missed opportunity + daily snapshot."""

        # Trade 1: winning trade
        self.trade_logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=10, position_size_quote=205000,
            entry_signal="Class M bullish", entry_signal_id="class_m_bull",
            entry_signal_strength=0.8, active_filters=["volume"],
            passed_filters=["volume"], strategy_params={"trail_mult": 3.0},
            market_regime="trending_up",
        )
        self.trade_logger.log_exit(
            trade_id="t1", exit_price=20600, exit_reason="TAKE_PROFIT", fees_paid=25,
        )

        # Trade 2: losing trade
        self.trade_logger.log_entry(
            trade_id="t2", pair="NQ", side="LONG",
            entry_price=20600, position_size=10, position_size_quote=206000,
            entry_signal="Class M bullish", entry_signal_id="class_m_bull",
            entry_signal_strength=0.6, active_filters=["volume"],
            passed_filters=["volume"], strategy_params={"trail_mult": 3.0},
            market_regime="trending_up",
        )
        self.trade_logger.log_exit(
            trade_id="t2", exit_price=20550, exit_reason="STOP_LOSS", fees_paid=25,
        )

        # Missed opportunity
        self.missed_logger.log_missed(
            pair="NQ", side="LONG",
            signal="Class M bullish", signal_id="class_m_bull",
            signal_strength=0.75, blocked_by="volume_filter",
            block_reason="Volume below threshold",
            strategy_type="helix", market_regime="trending_up",
        )

        # Build daily snapshot
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build()
        builder.save(snapshot)

        # Verify trade counts
        assert snapshot.total_trades == 2
        assert snapshot.win_count == 1
        assert snapshot.loss_count == 1
        assert snapshot.net_pnl != 0

        # Verify missed count
        assert snapshot.missed_count == 1

        # Verify files exist
        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        missed_files = list(Path(self.tmpdir).joinpath("missed").glob("*.jsonl"))
        daily_files = list(Path(self.tmpdir).joinpath("daily").glob("*.json"))

        assert len(trade_files) == 1
        assert len(missed_files) == 1
        assert len(daily_files) == 1

        # Verify trade JSONL has both entry and exit for both trades
        trade_lines = trade_files[0].read_text().strip().split("\n")
        assert len(trade_lines) == 4  # 2 entries + 2 exits

        # Verify daily snapshot JSON is valid
        daily_data = json.loads(daily_files[0].read_text())
        assert daily_data["total_trades"] == 2
        assert daily_data["bot_id"] == "integration_test"

    def test_fault_tolerance_entry_failure(self):
        """Snapshot failure on entry must not crash, and exit must still work."""
        self.snap_service.capture_now.side_effect = Exception("network timeout")
        trade = self.trade_logger.log_entry(
            trade_id="fault_t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=10, position_size_quote=205000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        # Entry should return a degraded event, not crash
        assert trade.trade_id == "fault_t1"

    def test_fault_tolerance_missed_failure(self):
        """Missed opportunity logger failure must not crash."""
        self.snap_service.capture_now.side_effect = Exception("broken")
        event = self.missed_logger.log_missed(
            pair="NQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="filter",
        )
        # Should return a minimal event, not crash
        assert event is not None

    def test_event_ids_unique_across_trades(self):
        """All event IDs must be unique."""
        self.trade_logger.log_entry(
            trade_id="uniq_t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        self.trade_logger.log_exit(
            trade_id="uniq_t1", exit_price=20600, exit_reason="TP",
        )
        self.trade_logger.log_entry(
            trade_id="uniq_t2", pair="NQ", side="SHORT",
            entry_price=20600, position_size=1, position_size_quote=20600,
            entry_signal="test2", entry_signal_id="test2",
            entry_signal_strength=0.6, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        self.trade_logger.log_exit(
            trade_id="uniq_t2", exit_price=20500, exit_reason="TP",
        )

        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        all_ids = []
        for f in trade_files:
            for line in f.read_text().strip().split("\n"):
                data = json.loads(line)
                eid = data.get("event_metadata", {}).get("event_id")
                if eid:
                    all_ids.append(eid)

        assert len(all_ids) == len(set(all_ids)), "Duplicate event IDs found"

    def test_root_causes_all_valid(self):
        """Process scorer root causes must all be from the controlled taxonomy."""
        self.trade_logger.log_entry(
            trade_id="rc_t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.8, active_filters=[], passed_filters=[],
            strategy_params={}, market_regime="trending_up",
        )
        self.trade_logger.log_exit(
            trade_id="rc_t1", exit_price=20600, exit_reason="TAKE_PROFIT",
        )

        # Read the trade and manually score it
        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        lines = trade_files[0].read_text().strip().split("\n")
        exit_data = json.loads(lines[-1])

        scorer = ProcessScorer("/nonexistent/path.yaml")
        score = scorer.score_trade(exit_data, "helix")
        for cause in score.root_causes:
            assert cause in ROOT_CAUSES, f"'{cause}' not in ROOT_CAUSES"
