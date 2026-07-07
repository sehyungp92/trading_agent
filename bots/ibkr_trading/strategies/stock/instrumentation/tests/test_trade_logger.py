import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock
from strategies.stock.instrumentation.src.trade_logger import TradeLogger, TradeEvent
from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshotService, MarketSnapshot
from strategies.stock.instrumentation.src.process_scorer import ProcessScore


def _mock_snapshot_service():
    service = MagicMock(spec=MarketSnapshotService)
    service.capture_now.return_value = MarketSnapshot(
        snapshot_id="test_snap", symbol="NQ",
        timestamp="2026-03-01T10:00:00Z",
        bid=20500.0, ask=20500.50, mid=20500.25, spread_bps=0.24,
        last_trade_price=20500.25, atr_14=85.0,
    )
    return service


class TestTradeLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "data_source_id": "test",
            "market_snapshots": {"interval_seconds": 60, "symbols": []},
        }
        self.snap_service = _mock_snapshot_service()
        self.logger = TradeLogger(self.config, self.snap_service)

    def test_log_entry_creates_event(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="Class M bullish", entry_signal_id="class_m_bull",
            entry_signal_strength=0.8, active_filters=["volume"],
            passed_filters=["volume"], strategy_params={"trail_mult": 3.0},
        )
        assert trade.trade_id == "t1"
        assert trade.side == "LONG"
        assert trade.stage == "entry"
        assert trade.entry_price == 20500

    def test_log_exit_computes_pnl_long(self):
        self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        trade = self.logger.log_exit(
            trade_id="t1", exit_price=20100, exit_reason="TAKE_PROFIT", fees_paid=10,
        )
        assert trade is not None
        assert trade.pnl == 90.0  # (20100 - 20000) * 1.0 - 10
        assert trade.stage == "exit"

    def test_log_exit_computes_pnl_short(self):
        self.logger.log_entry(
            trade_id="t2", pair="NQ", side="SHORT",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        trade = self.logger.log_exit(
            trade_id="t2", exit_price=19900, exit_reason="TAKE_PROFIT", fees_paid=10,
        )
        assert trade is not None
        assert trade.pnl == 90.0  # (20000 - 19900) * 1.0 - 10
        assert trade.stage == "exit"

    def test_log_exit_missing_trade_returns_none(self):
        result = self.logger.log_exit(
            trade_id="nonexistent", exit_price=21000, exit_reason="SIGNAL",
        )
        assert result is None

    def test_entry_failure_does_not_crash(self):
        """Instrumentation failure must never block trading."""
        self.snap_service.capture_now.side_effect = Exception("broken")
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        # Should return a minimal trade, not crash
        assert trade.trade_id == "t1"

    def test_events_written_to_jsonl(self):
        self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        assert len(files) == 1

    def test_entry_and_exit_both_written(self):
        self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        self.logger.log_exit(
            trade_id="t1", exit_price=20100, exit_reason="TAKE_PROFIT",
        )
        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        entry_data = json.loads(lines[0])
        exit_data = json.loads(lines[1])
        assert entry_data["stage"] == "entry"
        assert exit_data["stage"] == "exit"

    def test_entry_captures_snapshot_fields(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        assert trade.atr_at_entry == 85.0
        assert trade.spread_at_entry_bps == 0.24

    def test_entry_slippage_computed(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20505, position_size=1.0, position_size_quote=20505,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={}, expected_entry_price=20500,
        )
        assert trade.entry_slippage_bps is not None
        assert trade.entry_slippage_bps > 0

    def test_get_open_trades(self):
        self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        open_trades = self.logger.get_open_trades()
        assert "t1" in open_trades
        self.logger.log_exit(trade_id="t1", exit_price=20100, exit_reason="TP")
        open_trades = self.logger.get_open_trades()
        assert "t1" not in open_trades

    def test_trade_event_has_enriched_fields(self):
        """TradeEvent must have signal_factors, filter_decisions, sizing_inputs, futures context, concurrent positions, drawdown state, and post-exit tracking."""
        te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})

        # Signal confluence (feedback highest-impact #1)
        assert te.signal_factors == []
        assert isinstance(te.signal_factors, list)

        # Filter threshold context (feedback highest-impact #2)
        assert te.filter_decisions == []
        assert isinstance(te.filter_decisions, list)

        # Position sizing inputs (feedback highest-impact #3)
        assert te.sizing_inputs is None

        # Futures-specific context (feedback critical gap #5)
        assert te.session_type == ""
        assert te.contract_month == ""
        assert te.margin_used_pct is None

        # Concurrent position tracking (feedback critical gap #4)
        assert te.concurrent_positions_at_entry is None

        # Drawdown state (feedback critical gap #3)
        assert te.drawdown_pct is None
        assert te.drawdown_tier == ""
        assert te.drawdown_size_mult is None

        # Post-exit price tracking (highest-impact #5)
        assert te.post_exit_1h_price is None
        assert te.post_exit_4h_price is None
        assert te.post_exit_1h_move_pct is None
        assert te.post_exit_4h_move_pct is None
        assert te.post_exit_backfill_status == "pending"

    def test_trade_event_enriched_fields_serialize(self):
        """Enriched fields must round-trip through to_dict() / asdict()."""
        te = TradeEvent(
            trade_id="t1", event_metadata={}, entry_snapshot={},
            signal_factors=[{"factor_name": "trend", "factor_value": 0.85, "threshold": 0.5, "contribution": 0.35}],
            filter_decisions=[{"filter_name": "high_vol", "threshold": 97, "actual_value": 95, "passed": True, "margin_pct": 2.1}],
            sizing_inputs={"target_risk_pct": 0.01, "account_equity": 100000, "sizing_model": "fixed_frac"},
            session_type="RTH",
            contract_month="2026-03",
            margin_used_pct=42.5,
            concurrent_positions_at_entry=2,
            drawdown_pct=3.5,
            drawdown_tier="full",
            drawdown_size_mult=1.0,
            post_exit_1h_price=20600.0,
            post_exit_4h_price=20650.0,
            post_exit_1h_move_pct=0.49,
            post_exit_4h_move_pct=0.73,
            post_exit_backfill_status="complete",
        )
        d = te.to_dict()
        assert d["signal_factors"][0]["factor_name"] == "trend"
        assert d["filter_decisions"][0]["filter_name"] == "high_vol"
        assert d["sizing_inputs"]["sizing_model"] == "fixed_frac"
        assert d["session_type"] == "RTH"
        assert d["contract_month"] == "2026-03"
        assert d["margin_used_pct"] == 42.5
        assert d["concurrent_positions_at_entry"] == 2
        assert d["drawdown_pct"] == 3.5
        assert d["drawdown_tier"] == "full"
        assert d["drawdown_size_mult"] == 1.0
        assert d["post_exit_1h_price"] == 20600.0
        assert d["post_exit_4h_price"] == 20650.0
        assert d["post_exit_1h_move_pct"] == 0.49
        assert d["post_exit_4h_move_pct"] == 0.73
        assert d["post_exit_backfill_status"] == "complete"

    def test_enriched_fields_do_not_affect_existing_entry_exit(self):
        """Adding enriched fields must not change behavior of existing log_entry/log_exit."""
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="Class M bullish", entry_signal_id="class_m_bull",
            entry_signal_strength=0.8, active_filters=["volume"],
            passed_filters=["volume"], strategy_params={"trail_mult": 3.0},
        )
        # Enriched fields should have defaults
        assert trade.signal_factors == []
        assert trade.filter_decisions == []
        assert trade.sizing_inputs is None
        assert trade.post_exit_backfill_status == "pending"

        # Exit should still work normally
        exit_trade = self.logger.log_exit(
            trade_id="t1", exit_price=20600, exit_reason="TRAIL",
        )
        assert exit_trade is not None
        assert exit_trade.pnl == 100.0
        assert exit_trade.signal_factors == []
        assert exit_trade.post_exit_backfill_status == "pending"

    def test_log_exit_populates_excursion_percentages(self):
        self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={"stop0": 19900},
        )
        trade = self.logger.log_exit(
            trade_id="t1",
            exit_price=20100,
            exit_reason="TAKE_PROFIT",
            mfe_r=2.0,
            mae_r=-0.5,
            mfe_price=20200,
            mae_price=19950,
        )
        assert trade is not None
        assert trade.mfe_pct == 1.0
        assert trade.mae_pct == -0.25
        assert trade.exit_efficiency is not None

    def test_log_exit_embeds_process_score_into_trade_event(self):
        scorer = MagicMock()
        scorer.score_trade.return_value = ProcessScore(
            trade_id="t1",
            process_quality_score=47,
            root_causes=["regime_mismatch", "weak_signal"],
            evidence_refs=["market_regime=ranging"],
            positive_factors=[],
            negative_factors=["weak_signal"],
            classification="bad_process",
        )
        logger = TradeLogger(self.config, self.snap_service, process_scorer=scorer)
        logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20000, position_size=1.0, position_size_quote=20000,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.5, active_filters=[], passed_filters=[],
            strategy_params={"stop0": 19900},
        )

        trade = logger.log_exit(
            trade_id="t1",
            exit_price=19900,
            exit_reason="STOP_LOSS",
        )

        assert trade is not None
        assert trade.process_quality_score == 47
        assert trade.root_causes == ["regime_mismatch", "weak_signal"]
        assert trade.evidence_refs == ["market_regime=ranging"]

        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        payload = json.loads(files[0].read_text(encoding="utf-8").strip().split("\n")[-1])
        assert payload["process_quality_score"] == 47
        assert payload["root_causes"] == ["regime_mismatch", "weak_signal"]
        assert payload["evidence_refs"] == ["market_regime=ranging"]

    def test_log_entry_serializes_enriched_fields_immediately(self):
        self.logger.log_entry(
            trade_id="t-entry",
            pair="AAPL",
            side="LONG",
            entry_price=200.0,
            position_size=10.0,
            position_size_quote=2000.0,
            entry_signal="reclaim",
            entry_signal_id="setup-1",
            entry_signal_strength=0.9,
            active_filters=["spread_gate", "confidence_gate"],
            passed_filters=["confidence_gate"],
            strategy_params={"stop0": 198.5},
            signal_factors=[{"factor_name": "confidence", "factor_value": 1.0, "threshold": 0.5, "contribution": 1.0}],
            filter_decisions=[{"filter_name": "spread_gate", "threshold": 0.02, "actual_value": 0.015, "passed": True}],
            sizing_inputs={"qty": 10, "risk_per_share": 1.5},
            session_type="RTH",
            contract_month="AAPL",
            margin_used_pct=12.5,
            concurrent_positions=1,
            drawdown_pct=0.4,
            drawdown_tier="normal",
            drawdown_size_mult=1.0,
            signal_evolution=[{"bars_ago": 0, "close": 200.0}],
            execution_timestamps={"order_submitted_at": "2026-03-12T14:30:00+00:00"},
            experiment_id="exp-1",
            experiment_variant="variant-a",
        )

        files = list(Path(self.tmpdir).joinpath("trades").glob("*.jsonl"))
        payload = json.loads(files[0].read_text(encoding="utf-8").strip().split("\n")[0])
        assert payload["signal_factors"][0]["factor_name"] == "confidence"
        assert payload["filter_decisions"][0]["filter_name"] == "spread_gate"
        assert payload["sizing_inputs"]["qty"] == 10
        assert payload["session_type"] == "RTH"
        assert payload["concurrent_positions_at_entry"] == 1
        assert payload["experiment_id"] == "exp-1"
        assert payload["experiment_variant"] == "variant-a"

    def test_log_entry_sets_strategy_type(self):
        logger = TradeLogger(
            {**self.config, "strategy_type": "helix"},
            self.snap_service,
        )
        trade = logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.8, active_filters=[], passed_filters=[],
            strategy_params={"trail_mult": 3.0},
        )
        assert trade.strategy_type == "helix"

    def test_log_entry_computes_param_set_id(self):
        import hashlib, json as _json
        params = {"trail_mult": 3.0, "stop_atr": 1.5}
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.8, active_filters=[], passed_filters=[],
            strategy_params=params,
        )
        expected = hashlib.sha256(
            _json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        assert trade.param_set_id == expected

    def test_log_entry_param_set_id_none_when_no_params(self):
        trade = self.logger.log_entry(
            trade_id="t1", pair="NQ", side="LONG",
            entry_price=20500, position_size=1.0, position_size_quote=20500,
            entry_signal="test", entry_signal_id="test",
            entry_signal_strength=0.8, active_filters=[], passed_filters=[],
            strategy_params={},
        )
        assert trade.param_set_id is None

    def test_signal_evolution_defaults_none(self):
        """signal_evolution field defaults to None for backward compatibility."""
        te = TradeEvent(trade_id="t1", event_metadata={}, entry_snapshot={})
        assert te.signal_evolution is None
        d = te.to_dict()
        assert d["signal_evolution"] is None

    def test_signal_evolution_serializes(self):
        """signal_evolution round-trips through to_dict()."""
        evolution = [
            {"bars_ago": 4, "close": 21050.0, "ema_fast": 21040.5, "atr": 45.2},
            {"bars_ago": 3, "close": 21065.0, "ema_fast": 21045.2, "atr": 44.8},
            {"bars_ago": 2, "close": 21080.0, "ema_fast": 21052.1, "atr": 44.5},
            {"bars_ago": 1, "close": 21095.0, "ema_fast": 21060.8, "atr": 45.0},
            {"bars_ago": 0, "close": 21110.0, "ema_fast": 21070.3, "atr": 45.3},
        ]
        te = TradeEvent(
            trade_id="t1", event_metadata={}, entry_snapshot={},
            signal_evolution=evolution,
        )
        d = te.to_dict()
        assert len(d["signal_evolution"]) == 5
        assert d["signal_evolution"][0]["bars_ago"] == 4
        assert d["signal_evolution"][4]["close"] == 21110.0

        # Verify JSON round-trip
        serialized = json.dumps(d, default=str)
        parsed = json.loads(serialized)
        assert parsed["signal_evolution"] == evolution
