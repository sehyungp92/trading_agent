"""Integration test — simulates a full day of trading activity."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from strategies.swing.instrumentation.src.trade_logger import TradeLogger
from strategies.swing.instrumentation.src.missed_opportunity import MissedOpportunityLogger
from strategies.swing.instrumentation.src.daily_snapshot import DailySnapshotBuilder


class TestFullLifecycle:
    """Simulate: signals, blocked trades, executed trades, daily rollup."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "integration_test",
            "strategy_type": "ATRSS",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
            "market_snapshots": {"interval_seconds": 60, "symbols": ["QQQ"]},
        }

        self.snap_service = MagicMock(spec=MarketSnapshotService)
        self.snap_service.capture_now.return_value = MarketSnapshot(
            snapshot_id="test", symbol="QQQ",
            timestamp=datetime.now(timezone.utc).isoformat(),
            bid=500, ask=500.10, mid=500.05, spread_bps=2.0,
            last_trade_price=500.05, atr_14=5.0, volume_24h=50000000,
        )

        self.trade_logger = TradeLogger(self.config, self.snap_service)
        self.missed_logger = MissedOpportunityLogger(self.config, self.snap_service)

    def test_full_day_lifecycle(self):
        """Simulate: 2 executed trades + 1 missed opportunity + daily snapshot."""

        # Trade 1: winning trade
        self.trade_logger.log_entry(
            trade_id="t1", pair="QQQ", side="LONG",
            entry_price=500, position_size=10, position_size_quote=5000,
            entry_signal="Pullback to EMA", entry_signal_id="pullback_signal",
            entry_signal_strength=0.8, active_filters=["quality_gate", "time_filter"],
            passed_filters=["quality_gate", "time_filter"],
            strategy_params={"ema_fast": 20, "ema_slow": 55},
            strategy_id="ATRSS", market_regime="trending_up",
        )
        self.trade_logger.log_exit(
            trade_id="t1", exit_price=510, exit_reason="TAKE_PROFIT", fees_paid=5,
        )

        # Trade 2: losing trade
        self.trade_logger.log_entry(
            trade_id="t2", pair="QQQ", side="LONG",
            entry_price=510, position_size=10, position_size_quote=5100,
            entry_signal="Breakout pullback", entry_signal_id="breakout_pullback",
            entry_signal_strength=0.6, active_filters=["quality_gate"],
            passed_filters=["quality_gate"],
            strategy_params={"ema_fast": 20},
            strategy_id="ATRSS", market_regime="trending_up",
        )
        self.trade_logger.log_exit(
            trade_id="t2", exit_price=505, exit_reason="STOP_LOSS", fees_paid=5,
        )

        # Missed opportunity
        self.missed_logger.log_missed(
            pair="QQQ", side="LONG",
            signal="Pullback to EMA", signal_id="pullback_signal",
            signal_strength=0.75, blocked_by="quality_gate",
            block_reason="Quality score 3.5 below threshold 4.0",
            strategy_type="ATRSS", strategy_id="ATRSS",
            market_regime="trending_up",
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

        # Verify missed opportunity count
        assert snapshot.missed_count == 1

        # Verify files exist
        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        missed_files = list(Path(self.tmpdir).joinpath("missed").glob("*.jsonl"))
        daily_files = list(Path(self.tmpdir).joinpath("daily").glob("*.json"))

        assert len(trade_files) == 1
        assert len(missed_files) == 1
        assert len(daily_files) == 1

        # Verify trade JSONL has both entries and exits
        trade_lines = trade_files[0].read_text().strip().split("\n")
        assert len(trade_lines) == 4  # 2 entries + 2 exits

        # Verify daily snapshot JSON is valid
        daily_data = json.loads(daily_files[0].read_text())
        assert daily_data["total_trades"] == 2
        assert daily_data["bot_id"] == "integration_test"

    def test_instrumentation_failure_does_not_block(self):
        """Verify that broken instrumentation never prevents trade execution."""
        # Break the snapshot service
        self.snap_service.capture_now.side_effect = Exception("snapshot service down")

        # Entry should still return a trade object (degraded)
        trade = self.trade_logger.log_entry(
            trade_id="t_broken", pair="QQQ", side="LONG",
            entry_price=500, position_size=10, position_size_quote=5000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        assert trade is not None
        assert trade.trade_id == "t_broken"

        # Missed opportunity should still return an event (degraded)
        event = self.missed_logger.log_missed(
            pair="QQQ", side="LONG",
            signal="test", signal_id="test",
            signal_strength=0.5, blocked_by="test_filter",
        )
        assert event is not None

    def test_event_ids_are_unique(self):
        """Every event must have a unique event_id."""
        # Create multiple trades
        for i in range(5):
            self.trade_logger.log_entry(
                trade_id=f"t{i}", pair="QQQ", side="LONG",
                entry_price=500 + i, position_size=10, position_size_quote=5000,
                entry_signal="test", entry_signal_id="test",
                entry_signal_strength=0.5, active_filters=[], passed_filters=[],
                strategy_params={},
            )

        # Read all events and check IDs
        trade_files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        assert len(trade_files) == 1

        event_ids = set()
        for line in trade_files[0].read_text().strip().split("\n"):
            data = json.loads(line)
            eid = data.get("event_metadata", {}).get("event_id")
            if eid:
                assert eid not in event_ids, f"Duplicate event_id: {eid}"
                event_ids.add(eid)

    def test_enriched_entry_exit_cycle(self):
        """Verify enriched fields flow through entry-exit-snapshot cycle."""
        # Entry with enriched metadata
        trade = self.trade_logger.log_entry(
            trade_id="t_enriched", pair="QQQ", side="LONG",
            entry_price=500, position_size=10, position_size_quote=5000,
            entry_signal="Pullback", entry_signal_id="pull_001",
            entry_signal_strength=0.85, active_filters=["quality"],
            passed_filters=["quality"],
            strategy_params={"ema_fast": 20},
            strategy_id="ATRSS", market_regime="trending_up",
            # Enriched fields
            drawdown_pct_at_entry=3.5,
            drawdown_tier_at_entry="NORMAL",
            position_size_multiplier=1.0,
            market_session="RTH",
            minutes_into_session=45,
            overnight_gap_pct=0.5,
            experiment_id="exp_001",
            concurrent_positions_strategy=2,
        )
        assert trade is not None
        assert trade.drawdown_pct_at_entry == 3.5
        assert trade.drawdown_tier_at_entry == "NORMAL"
        assert trade.market_session == "RTH"
        assert trade.experiment_id == "exp_001"
        assert trade.concurrent_positions_strategy == 2

        # Exit with MFE/MAE
        exit_trade = self.trade_logger.log_exit(
            trade_id="t_enriched", exit_price=510,
            exit_reason="TAKE_PROFIT", fees_paid=5,
            mfe_price=515, mae_price=498,
            mfe_pct=0.03, mae_pct=0.004,
            mfe_r=2.5, mae_r=0.33,
            exit_efficiency=0.667,
        )
        assert exit_trade is not None
        assert exit_trade.mfe_price == 515
        assert exit_trade.mae_price == 498
        assert exit_trade.mfe_pct == 0.03
        assert exit_trade.exit_efficiency == 0.667

        # Verify daily snapshot includes enriched aggregates
        builder = DailySnapshotBuilder(self.config)
        snapshot = builder.build()
        assert snapshot.avg_mfe_pct is not None
        assert snapshot.avg_mae_pct is not None
        # Session breakdown should have RTH entry
        # (depends on trade event having market_session in exit record)
