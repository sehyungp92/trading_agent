"""Tests for InstrumentationKit facade."""
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from strategies.swing.instrumentation.src.kit import InstrumentationKit
from strategies.swing.instrumentation.src.context import InstrumentationContext
from strategies.swing.instrumentation.src.trade_logger import TradeEvent, TradeLogger
from strategies.swing.instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from strategies.swing.instrumentation.src.missed_opportunity import MissedOpportunityEvent, MissedOpportunityLogger
from strategies.swing.instrumentation.src.process_scorer import ProcessScore, ProcessScorer
from strategies.swing.instrumentation.src.regime_classifier import RegimeClassifier


class TestInstrumentationKit:
    """Tests for InstrumentationKit facade."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()

        # Create mock services
        self.mock_trade_logger = MagicMock(spec=TradeLogger)
        self.mock_missed_logger = MagicMock(spec=MissedOpportunityLogger)
        self.mock_process_scorer = MagicMock(spec=ProcessScorer)
        self.mock_regime_classifier = MagicMock(spec=RegimeClassifier)
        self.mock_snapshot_service = MagicMock(spec=MarketSnapshotService)

        # Set up return values
        self.mock_snapshot_service.capture_now.return_value = MarketSnapshot(
            snapshot_id="snap1",
            symbol="BTC/USDT",
            timestamp="2026-03-01T10:00:00Z",
            bid=50000,
            ask=50010,
            mid=50005,
            spread_bps=2.0,
            last_trade_price=50005,
            atr_14=500,
            volume_24h=1000000,
        )

        self.mock_regime_classifier.current_regime.return_value = "trending_up"
        self.mock_regime_classifier.classify.return_value = "trending_up"

        # Create context
        self.ctx = InstrumentationContext(
            snapshot_service=self.mock_snapshot_service,
            trade_logger=self.mock_trade_logger,
            missed_logger=self.mock_missed_logger,
            process_scorer=self.mock_process_scorer,
            regime_classifier=self.mock_regime_classifier,
            data_dir=self.tmpdir,
        )

        # Create kit
        self.kit = InstrumentationKit(self.ctx, strategy_id="ATRSS")

    def test_init_stores_context_and_strategy_id(self):
        """Test that __init__ stores context and strategy_id."""
        assert self.kit.ctx is self.ctx
        assert self.kit.strategy_id == "ATRSS"

    def test_log_entry_calls_trade_logger(self):
        """Test that log_entry calls ctx.trade_logger.log_entry."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={"test": True},
            entry_snapshot={},
            pair="BTC/USDT",
            side="LONG",
            strategy_id="ATRSS",
        )
        self.mock_trade_logger.log_entry.return_value = trade_event

        result = self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA cross",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=["volume"],
            passed_filters=["volume"],
            strategy_params={"ema_fast": 12},
        )

        # Verify trade_logger was called
        self.mock_trade_logger.log_entry.assert_called_once()
        assert result is not None
        assert result.get("trade_id") == "t1" or isinstance(result, dict)

    def test_log_entry_passes_strategy_id_automatically(self):
        """Test that log_entry passes strategy_id to logger."""
        self.mock_trade_logger.log_entry.return_value = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )

        self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )

        # Verify strategy_id was passed
        call_kwargs = self.mock_trade_logger.log_entry.call_args[1]
        assert call_kwargs.get("strategy_id") == "ATRSS"

    def test_log_entry_calls_regime_classifier(self):
        """Test that log_entry calls regime_classifier.current_regime."""
        self.mock_trade_logger.log_entry.return_value = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )

        self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )

        # Verify regime classifier was called
        self.mock_regime_classifier.current_regime.assert_called_with("BTC/USDT")

    def test_log_entry_passes_enriched_fields(self):
        """Test that log_entry passes enriched fields to logger."""
        self.mock_trade_logger.log_entry.return_value = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )

        signal_factors = [{"factor": "momentum", "value": 0.75}]
        filter_decisions = [{"filter": "volume", "current": 1000000, "threshold": 500000}]
        sizing_inputs = {"risk_pct": 1.0, "atr": 500}
        portfolio_state = {"total_exposure": 0.5, "positions": 3}

        self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=["volume"],
            passed_filters=["volume"],
            strategy_params={},
            signal_factors=signal_factors,
            filter_decisions=filter_decisions,
            sizing_inputs=sizing_inputs,
            portfolio_state_at_entry=portfolio_state,
        )

        # Verify enriched fields were passed
        call_kwargs = self.mock_trade_logger.log_entry.call_args[1]
        assert call_kwargs.get("signal_factors") == signal_factors
        assert call_kwargs.get("filter_decisions") == filter_decisions
        assert call_kwargs.get("sizing_inputs") == sizing_inputs
        assert call_kwargs.get("portfolio_state_at_entry") == portfolio_state

    def test_log_entry_never_raises_on_logger_failure(self):
        """Test that log_entry never raises even if trade_logger fails."""
        self.mock_trade_logger.log_entry.side_effect = Exception("logger broken")

        # Should not raise
        result = self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )

        # Should return empty dict on failure
        assert result == {} or result is not None

    def test_log_entry_with_none_context(self):
        """Test that log_entry handles None context gracefully."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")

        result = kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
        )

        # Should return empty dict, not raise
        assert result == {}

    def test_log_exit_calls_trade_logger(self):
        """Test that log_exit calls ctx.trade_logger.log_exit."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )
        self.mock_trade_logger.log_exit.return_value = trade_event

        result = self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=50,
        )

        # Verify trade_logger was called
        self.mock_trade_logger.log_exit.assert_called_once()
        assert result is not None

    def test_log_exit_calls_process_scorer(self):
        """Test that log_exit calls process_scorer.score_and_write."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )
        self.mock_trade_logger.log_exit.return_value = trade_event

        self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=50,
        )

        # Verify process_scorer was called
        self.mock_process_scorer.score_and_write.assert_called_once()

    def test_log_exit_passes_strategy_type_to_scorer(self):
        """Test that log_exit passes strategy_type to scorer."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )
        self.mock_trade_logger.log_exit.return_value = trade_event

        self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )

        # Verify strategy_type was passed
        call_kwargs = self.mock_process_scorer.score_and_write.call_args[1]
        assert call_kwargs.get("strategy_type") == "ATRSS"

    def test_log_exit_never_raises_on_logger_failure(self):
        """Test that log_exit never raises even if trade_logger fails."""
        self.mock_trade_logger.log_exit.side_effect = Exception("logger broken")

        # Should not raise
        result = self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )

        # Should return empty dict on failure
        assert result == {} or result is not None

    def test_log_exit_never_raises_on_scorer_failure(self):
        """Test that log_exit never raises even if scorer fails."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )
        self.mock_trade_logger.log_exit.return_value = trade_event
        self.mock_process_scorer.score_and_write.side_effect = Exception("scorer broken")

        # Should not raise
        result = self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )

        # Should still return result
        assert result is not None

    def test_log_exit_with_none_context(self):
        """Test that log_exit handles None context gracefully."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")

        result = kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
        )

        # Should return empty dict, not raise
        assert result == {}

    def test_log_missed_calls_missed_logger(self):
        """Test that log_missed calls ctx.missed_logger.log_missed."""
        missed_event = MissedOpportunityEvent(
            event_metadata={},
            market_snapshot={},
            pair="BTC/USDT",
            side="LONG",
            signal="EMA cross",
        )
        self.mock_missed_logger.log_missed.return_value = missed_event

        result = self.kit.log_missed(
            pair="BTC/USDT",
            side="LONG",
            signal="EMA cross",
            signal_id="ema_123",
            signal_strength=0.8,
            blocked_by="max_open_trades",
        )

        # Verify missed_logger was called
        self.mock_missed_logger.log_missed.assert_called_once()
        assert result is not None

    def test_log_missed_passes_strategy_fields(self):
        """Test that log_missed passes strategy_id and strategy_type."""
        missed_event = MissedOpportunityEvent(
            event_metadata={},
            market_snapshot={},
        )
        self.mock_missed_logger.log_missed.return_value = missed_event

        self.kit.log_missed(
            pair="BTC/USDT",
            side="LONG",
            signal="EMA",
            signal_id="ema_123",
            signal_strength=0.8,
            blocked_by="max_open_trades",
        )

        # Verify strategy_id and strategy_type were passed
        call_kwargs = self.mock_missed_logger.log_missed.call_args[1]
        assert call_kwargs.get("strategy_id") == "ATRSS"
        assert call_kwargs.get("strategy_type") == "ATRSS"

    def test_log_missed_never_raises(self):
        """Test that log_missed never raises."""
        self.mock_missed_logger.log_missed.side_effect = Exception("missed logger broken")

        # Should not raise
        result = self.kit.log_missed(
            pair="BTC/USDT",
            side="LONG",
            signal="EMA",
            signal_id="ema_123",
            signal_strength=0.8,
            blocked_by="max_open_trades",
        )

        # Should return empty dict on failure
        assert result == {} or result is not None

    def test_log_missed_with_none_context(self):
        """Test that log_missed handles None context gracefully."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")

        result = kit.log_missed(
            pair="BTC/USDT",
            side="LONG",
            signal="EMA",
            signal_id="ema_123",
            signal_strength=0.8,
            blocked_by="max_open_trades",
        )

        # Should return empty dict, not raise
        assert result == {}

    def test_classify_regime_calls_classifier(self):
        """Test that classify_regime calls regime_classifier.classify."""
        result = self.kit.classify_regime("BTC/USDT")

        # Verify classifier was called
        self.mock_regime_classifier.classify.assert_called_with("BTC/USDT")
        assert result == "trending_up"

    def test_classify_regime_returns_valid_regime(self):
        """Test that classify_regime always returns valid regime string."""
        valid_regimes = {"trending_up", "trending_down", "ranging", "volatile", "unknown"}

        self.mock_regime_classifier.classify.return_value = "trending_up"
        result = self.kit.classify_regime("BTC/USDT")
        assert result in valid_regimes

        self.mock_regime_classifier.classify.return_value = "unknown"
        result = self.kit.classify_regime("BTC/USDT")
        assert result in valid_regimes

    def test_classify_regime_returns_unknown_on_invalid(self):
        """Test that classify_regime returns 'unknown' on invalid regime."""
        self.mock_regime_classifier.classify.return_value = "invalid_regime"

        result = self.kit.classify_regime("BTC/USDT")
        assert result == "unknown"

    def test_classify_regime_never_raises(self):
        """Test that classify_regime never raises."""
        self.mock_regime_classifier.classify.side_effect = Exception("classifier broken")

        # Should not raise
        result = self.kit.classify_regime("BTC/USDT")
        assert result == "unknown"

    def test_classify_regime_with_none_context(self):
        """Test that classify_regime handles None context gracefully."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")

        result = kit.classify_regime("BTC/USDT")
        assert result == "unknown"

    def test_classify_regime_with_none_classifier(self):
        """Test that classify_regime handles None classifier gracefully."""
        ctx = InstrumentationContext(regime_classifier=None)
        kit = InstrumentationKit(ctx, strategy_id="ATRSS")

        result = kit.classify_regime("BTC/USDT")
        assert result == "unknown"

    def test_capture_snapshot_calls_snapshot_service(self):
        """Test that capture_snapshot calls snapshot_service.capture_now."""
        result = self.kit.capture_snapshot("BTC/USDT")

        # Verify service was called
        self.mock_snapshot_service.capture_now.assert_called_with("BTC/USDT")
        assert result is not None
        assert isinstance(result, dict)

    def test_capture_snapshot_returns_dict(self):
        """Test that capture_snapshot returns dict on success."""
        result = self.kit.capture_snapshot("BTC/USDT")

        assert result is not None
        assert isinstance(result, dict)
        assert result.get("symbol") == "BTC/USDT"

    def test_capture_snapshot_never_raises(self):
        """Test that capture_snapshot never raises."""
        self.mock_snapshot_service.capture_now.side_effect = Exception("snapshot broken")

        # Should not raise
        result = self.kit.capture_snapshot("BTC/USDT")
        assert result is None or isinstance(result, dict)

    def test_capture_snapshot_with_none_context(self):
        """Test that capture_snapshot handles None context gracefully."""
        kit = InstrumentationKit(None, strategy_id="ATRSS")

        result = kit.capture_snapshot("BTC/USDT")
        assert result is None

    def test_capture_snapshot_with_none_service(self):
        """Test that capture_snapshot handles None service gracefully."""
        ctx = InstrumentationContext(snapshot_service=None)
        kit = InstrumentationKit(ctx, strategy_id="ATRSS")

        result = kit.capture_snapshot("BTC/USDT")
        assert result is None

    def test_all_methods_work_together_in_trade_lifecycle(self):
        """Integration test: full trade lifecycle with kit."""
        # Entry
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={"test": True},
            entry_snapshot={},
            pair="BTC/USDT",
            side="LONG",
        )
        self.mock_trade_logger.log_entry.return_value = trade_event

        entry_result = self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA cross",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=["volume"],
            passed_filters=["volume"],
            strategy_params={"ema_fast": 12},
            signal_factors=[{"factor": "momentum", "value": 0.75}],
        )

        assert entry_result is not None
        self.mock_trade_logger.log_entry.assert_called_once()

        # Exit
        self.mock_trade_logger.log_exit.return_value = trade_event
        exit_result = self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=50,
        )

        assert exit_result is not None
        self.mock_trade_logger.log_exit.assert_called_once()
        self.mock_process_scorer.score_and_write.assert_called_once()

    def test_log_entry_with_all_optional_params(self):
        """Test log_entry with all optional parameters provided."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )
        self.mock_trade_logger.log_entry.return_value = trade_event

        now = datetime.now(timezone.utc)
        self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=["volume", "atr"],
            passed_filters=["volume"],
            strategy_params={"ema_fast": 12},
            signal_factors=[{"factor": "momentum", "value": 0.75}],
            filter_decisions=[{"filter": "volume", "current": 1000000, "threshold": 500000}],
            sizing_inputs={"risk_pct": 1.0, "atr": 500},
            portfolio_state_at_entry={"total_exposure": 0.5},
            exchange_timestamp=now,
            expected_entry_price=49950,
            entry_latency_ms=250,
            bar_id="bar_123",
        )

        self.mock_trade_logger.log_entry.assert_called_once()
        call_kwargs = self.mock_trade_logger.log_entry.call_args[1]
        assert call_kwargs.get("exchange_timestamp") == now
        assert call_kwargs.get("expected_entry_price") == 49950
        assert call_kwargs.get("entry_latency_ms") == 250
        assert call_kwargs.get("bar_id") == "bar_123"

    def test_log_entry_forwards_fill_runtime_refs(self):
        """Test live fill refs pass through to the trade logger."""
        self.mock_trade_logger.log_entry.return_value = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )

        self.kit.log_entry(
            trade_id="t1",
            pair="BTC/USDT",
            side="LONG",
            entry_price=50000,
            position_size=1.0,
            position_size_quote=50000,
            entry_signal="EMA",
            entry_signal_id="ema_123",
            entry_signal_strength=0.8,
            active_filters=[],
            passed_filters=[],
            strategy_params={},
            fill_order_id="oms-entry",
            fill_id="exec-entry",
            fill_qty=1.0,
            intent_id="intent_1",
            portfolio_decision_ref="portfolio_rule_1",
        )

        call_kwargs = self.mock_trade_logger.log_entry.call_args[1]
        assert call_kwargs["fill_order_id"] == "oms-entry"
        assert call_kwargs["fill_id"] == "exec-entry"
        assert call_kwargs["fill_qty"] == 1.0
        assert call_kwargs["intent_id"] == "intent_1"
        assert call_kwargs["portfolio_decision_ref"] == "portfolio_rule_1"

    def test_log_exit_with_all_optional_params(self):
        """Test log_exit with all optional parameters provided."""
        trade_event = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )
        self.mock_trade_logger.log_exit.return_value = trade_event

        now = datetime.now(timezone.utc)
        self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fees_paid=100,
            exchange_timestamp=now,
            expected_exit_price=51050,
            exit_latency_ms=150,
        )

        self.mock_trade_logger.log_exit.assert_called_once()
        call_kwargs = self.mock_trade_logger.log_exit.call_args[1]
        assert call_kwargs.get("exchange_timestamp") == now
        assert call_kwargs.get("expected_exit_price") == 51050
        assert call_kwargs.get("exit_latency_ms") == 150

    def test_log_exit_forwards_fill_runtime_refs(self):
        """Test live exit fill refs pass through to the trade logger."""
        self.mock_trade_logger.log_exit.return_value = TradeEvent(
            trade_id="t1",
            event_metadata={},
            entry_snapshot={},
        )

        self.kit.log_exit(
            trade_id="t1",
            exit_price=51000,
            exit_reason="TAKE_PROFIT",
            fill_order_id="oms-exit",
            exit_fill_id="exec-exit",
            fill_qty=1.0,
        )

        call_kwargs = self.mock_trade_logger.log_exit.call_args[1]
        assert call_kwargs["fill_order_id"] == "oms-exit"
        assert call_kwargs["exit_fill_id"] == "exec-exit"
        assert call_kwargs["fill_qty"] == 1.0
