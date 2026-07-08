"""Tests for TradeLogger."""
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

from strategies.swing.instrumentation.src.trade_logger import TradeLogger, TradeEvent
from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot


class TestTradeLogger:
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
            snapshot_id="test", symbol="BTC/USDT", timestamp="2026-03-01T10:00:00Z",
            bid=50000, ask=50010, mid=50005, spread_bps=2.0, last_trade_price=50005,
            atr_14=500, volume_24h=1000000,
        )
        self.logger = TradeLogger(self.config, self.snap_service)

    def test_log_entry_creates_event(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50005, position_size=0.1, position_size_quote=5000.5,
            entry_signal="EMA cross", entry_signal_id="ema_cross",
            entry_signal_strength=0.8, active_filters=["volume"],
            passed_filters=["volume"], strategy_params={"ema_fast": 12},
        )
        assert trade.trade_id == "t1"
        assert trade.side == "LONG"
        assert trade.stage == "entry"

    def test_log_exit_computes_pnl(self):
        self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        trade = self.logger.log_exit(
            trade_id="t1", exit_price=51000, exit_reason="TAKE_PROFIT", fees_paid=50,
        )
        assert trade is not None
        assert trade.pnl == 950.0
        assert trade.stage == "exit"

    def test_log_exit_short_pnl(self):
        self.logger.log_entry(
            trade_id="t2", pair="BTC/USDT", side="SHORT",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        trade = self.logger.log_exit(
            trade_id="t2", exit_price=49000, exit_reason="TAKE_PROFIT", fees_paid=50,
        )
        assert trade is not None
        assert trade.pnl == 950.0  # (50000-49000)*1.0 - 50

    def test_log_exit_missing_trade_returns_none(self):
        result = self.logger.log_exit(
            trade_id="nonexistent", exit_price=51000, exit_reason="SIGNAL",
        )
        assert result is None

    def test_entry_failure_does_not_crash(self):
        self.snap_service.capture_now.side_effect = Exception("broken")
        trade = self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        assert trade.trade_id == "t1"

    def test_events_written_to_jsonl(self):
        self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        assert len(files) == 1

    def test_slippage_computed(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50010, position_size=1.0, position_size_quote=50010,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
            expected_entry_price=50000,
        )
        assert trade.entry_slippage_bps is not None
        assert trade.entry_slippage_bps > 0

    def test_get_open_trades(self):
        self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        open_trades = self.logger.get_open_trades()
        assert "t1" in open_trades

    def test_strategy_id_captured(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="QQQ", side="LONG",
            entry_price=500, position_size=10, position_size_quote=5000,
            entry_signal="pullback", entry_signal_id="pullback_signal",
            entry_signal_strength=0.7, active_filters=[], passed_filters=[],
            strategy_params={}, strategy_id="ATRSS",
        )
        assert trade.strategy_id == "ATRSS"

    def test_trade_event_has_enriched_fields(self):
        """Verify new enriched fields can be set and serialized via to_dict()."""
        trade = self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )

        # Manually set the enriched fields
        trade.signal_factors = [{"factor": "EMA_cross", "strength": 0.85}]
        trade.filter_decisions = [{"filter": "volume", "passed": True, "distance_to_threshold": 0.2}]
        trade.sizing_inputs = {"kelly_fraction": 0.25, "max_loss": 500}
        trade.portfolio_state_at_entry = {"total_exposure": 0.5, "direction": "LONG"}

        # Verify we can serialize to dict
        trade_dict = trade.to_dict()

        assert trade_dict["signal_factors"] == [{"factor": "EMA_cross", "strength": 0.85}]
        assert trade_dict["filter_decisions"] == [{"filter": "volume", "passed": True, "distance_to_threshold": 0.2}]
        assert trade_dict["sizing_inputs"] == {"kelly_fraction": 0.25, "max_loss": 500}
        assert trade_dict["portfolio_state_at_entry"] == {"total_exposure": 0.5, "direction": "LONG"}

    def test_trade_event_enriched_fields_default_empty(self):
        """Verify defaults: empty list for signal_factors/filter_decisions, None for sizing_inputs/portfolio_state_at_entry."""
        trade = self.logger.log_entry(
            trade_id="t1", pair="BTC/USDT", side="LONG",
            entry_price=50000, position_size=1.0, position_size_quote=50000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )

        # Verify defaults
        assert trade.signal_factors == []
        assert trade.filter_decisions == []
        assert trade.sizing_inputs is None
        assert trade.portfolio_state_at_entry is None

    def test_log_entry_stores_enriched_fields(self):
        """log_entry must pass signal_factors, filter_decisions, sizing_inputs, portfolio_state to TradeEvent."""
        sf = [{"factor_name": "adx", "factor_value": 30, "threshold": 25, "contribution": "trend"}]
        fd = [{"filter_name": "gate", "threshold": 3, "actual_value": 5, "passed": True, "margin_pct": 66.7}]
        si = {"target_risk_pct": 0.02, "account_equity": 100000, "volatility_basis": 1.5, "sizing_model": "atr"}
        ps = {"total_exposure_pct": 0.3, "net_direction": "LONG", "num_positions": 2, "correlated_positions": []}

        event = self.logger.log_entry(
            trade_id="t_enriched",
            pair="QQQ", side="LONG",
            entry_price=500.0, position_size=10.0, position_size_quote=5000.0,
            entry_signal="PB", entry_signal_id="x", entry_signal_strength=0.7,
            active_filters=["gate"], passed_filters=["gate"],
            strategy_params={"atrh": 1.5},
            signal_factors=sf,
            filter_decisions=fd,
            sizing_inputs=si,
            portfolio_state_at_entry=ps,
        )

        assert event.signal_factors == sf
        assert event.filter_decisions == fd
        assert event.sizing_inputs == si
        assert event.portfolio_state_at_entry == ps


class TestTradeEventEnrichment:
    def test_excursion_fields_default_none(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.mfe_price is None
        assert evt.mae_price is None
        assert evt.mfe_pct is None
        assert evt.mae_pct is None
        assert evt.mfe_r is None
        assert evt.mae_r is None
        assert evt.exit_efficiency is None

    def test_drawdown_fields_default_none(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.drawdown_pct_at_entry is None
        assert evt.drawdown_tier_at_entry is None
        assert evt.position_size_multiplier is None

    def test_session_gap_metadata_fields_default_none(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.market_session is None
        assert evt.minutes_into_session is None
        assert evt.overnight_gap_pct is None
        assert evt.prev_close_price is None
        assert evt.experiment_id is None
        assert evt.concurrent_positions_strategy is None
        assert evt.correlated_pairs_detail is None

    def test_overlay_state_field_default_none(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.overlay_state is None

    def test_execution_timeline_field(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.execution_timeline is None
        evt.execution_timeline = {
            "signal_generated_at": "2026-03-01T10:00:00Z",
            "fill_confirmed_at": "2026-03-01T10:00:00.500Z",
        }
        d = evt.to_dict()
        assert d["execution_timeline"]["signal_generated_at"] == "2026-03-01T10:00:00Z"

    def test_experiment_variant_field(self):
        evt = TradeEvent(trade_id="test", event_metadata={}, entry_snapshot={})
        assert evt.experiment_variant is None
        evt.experiment_variant = "control"
        d = evt.to_dict()
        assert d["experiment_variant"] == "control"

    def test_overlay_state_kwarg_persisted(self):
        """overlay_state passed as kwarg should be persisted on the TradeEvent."""
        config = {
            "bot_id": "test_bot",
            "data_dir": __import__("tempfile").mkdtemp(),
            "data_source_id": "test",
            "market_snapshots": {"interval_seconds": 60, "symbols": []},
        }
        from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot
        from unittest.mock import MagicMock
        snap_service = MagicMock(spec=MarketSnapshotService)
        snap_service.capture_now.return_value = MarketSnapshot(
            snapshot_id="test", symbol="QQQ", timestamp="2026-03-01T10:00:00Z",
            bid=500, ask=501, mid=500.5, spread_bps=2.0, last_trade_price=500.5,
            atr_14=5.0, volume_24h=1000000,
        )
        from strategies.swing.instrumentation.src.trade_logger import TradeLogger
        tl = TradeLogger(config, snap_service)
        trade = tl.log_entry(
            trade_id="t_overlay", pair="QQQ", side="LONG",
            entry_price=500, position_size=10, position_size_quote=5000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
            overlay_state={"qqq_ema_bullish": True, "gld_ema_bullish": False},
        )
        assert trade.overlay_state == {"qqq_ema_bullish": True, "gld_ema_bullish": False}
